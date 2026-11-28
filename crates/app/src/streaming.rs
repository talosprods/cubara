//! Node streaming: decides which nodes should be resident around the camera
//! and drives `cubara_world`'s mesh pool to keep them that way, handing
//! finished geometry to the renderer.
//!
//! This is the piece `cubara-render` itself can no longer do
//! (`ARCHITECTURE.md` §1, issue #38's tracking arc sub-issue #110): its
//! inputs are meshes, origins and a camera, nothing that knows what a
//! `World` or a `NodeKey` is. `cubara-app` is the one crate that depends on
//! both `cubara-render` and `cubara-world`, so the glue lives here.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use cubara_render::{MeshedNode, NodeId, Renderer};
use cubara_voxel::{BlockRegistry, ChunkCoord};
use cubara_world::mesh::{sort_batch, BuiltNode, MeshPool};
use cubara_world::node::{self, NodeKey};
use cubara_world::visibility;
use cubara_world::TerrainBlocks;
use cubara_world::World;

use crate::far_streaming::{hole_in_blocks, VoxelBox};
use crate::frame_work::{Count, Laps, Phase};

/// How many times sooner detail coarsens with height and depth than with
/// horizontal distance (`cubara_world::node::desired_nodes_3d`).
///
/// **There is no vertical band any more.** It used to stream only ±2
/// chunk-layers around the camera, chosen to pass the perf gate by drawing
/// less: build 40 blocks up and the ground stopped existing. The render
/// distance is now the same in every direction, paid for by leaving out
/// covered faces, and only the *detail* falls off faster vertically.
///
/// **Two, measured** (radius 64, `--bench 64 --eye … --squash k`, Windows /
/// RTX 4060; `BENCHMARKS.md`): at the surface 5,603 FPS with `2` against 6,293
/// with `4`, deep underground 5,096 against 6,985. `4` is faster, but caves
/// exist only at full detail and it keeps that only 40 blocks up and down, so
/// the bottom of a shaft you look into turns to solid rock. `2` kept 80.
///
/// **Only beyond the near field** (`cubara_world::node::squashed_gap`, 2026-11-27):
/// height counts once out to 96 blocks, so level 0 is a ball around the eye
/// there -- standing on a peak, the ground 75 blocks down is level 0 too -- and
/// every ring reaches 48 blocks further up and down than this squash alone
/// would put it: level 0 now 128 blocks (`docs/PHASE3_ARCHITECTURE.md` §9.6.2).
pub(crate) const VERTICAL_LOD_SQUASH: i32 = 2;

/// The distance morph's bands for the game's ring schedule and squash
/// (issue #380's study; `cubara_world::morph`): what the renderer moves
/// each level's vertices to their parent's height over.
pub(crate) fn morph_params() -> cubara_render::MorphParams {
    morph_params_for(node::DEFAULT_RING_SCHEDULE, VERTICAL_LOD_SQUASH)
}

/// [`morph_params`] for any ring schedule.
pub(crate) fn morph_params_for(
    schedule: &node::RingSchedule,
    squash: i32,
) -> cubara_render::MorphParams {
    cubara_render::MorphParams {
        bands: cubara_world::morph::morph_bands(schedule, squash)
            .map(|band| band.map(|b| (b.start, b.end))),
        squash: squash as f32,
        shift: cubara_world::morph::vertical_shift_blocks(squash),
    }
}

/// How far the voxel nodes reach, in chunks: the outer edge of
/// [`node::DEFAULT_RING_SCHEDULE`]'s last (coarsest) ring. Beyond it the far
/// terrain takes over (`crate::far_streaming`).
pub(crate) fn render_radius_chunks() -> i32 {
    node::DEFAULT_RING_SCHEDULE
        .last()
        .expect("the ring schedule is never empty")
        .1
}

/// The nodes in `visible` that cover the same space `node` does: either its
/// eight children one level finer (the camera came closer and the node split)
/// or its parent one level coarser (the camera left and eight merged).
///
/// Checking exactly one level each way is enough because the ring schedule
/// keeps neighbours within one level of each other
/// (`cubara_world::node::desired_nodes`), and a node is replaced *in place* --
/// by what the octree puts in its own footprint, which is its children or its
/// parent and nothing else.
///
/// Empty means nothing is taking this space over: the camera walked away from
/// it, or the visibility search stopped reaching it. Those unload at once;
/// only a *replacement* is worth waiting for.
fn replacements_of(node: NodeKey, visible: &HashSet<NodeKey>) -> Vec<NodeKey> {
    node.children()
        .into_iter()
        .chain([node.parent()])
        .filter(|n| visible.contains(n))
        .collect()
}

/// Whether what `node` joins is worth keeping while `desired` is the render
/// distance.
///
/// What a node joins does not depend on where the camera is, but a node
/// outside the render distance is not worth remembering -- except one level
/// either side of it: a node that has just split or merged is searched through
/// on what is known of the nodes that drew its space before, until its own
/// links arrive (`cubara_world::visibility`'s stand-in). Forgetting the parent
/// the moment it split made the new children walls again, one search later.
fn worth_remembering(node: NodeKey, desired: &HashSet<NodeKey>) -> bool {
    desired.contains(&node)
        || desired.contains(&node.parent())
        || node.children().iter().any(|c| desired.contains(c))
}

/// Record that `node` has just left the visible set, if it is worth
/// measuring as a swap at all (design decision 1, issue #378): something
/// still replaces it. A node nothing replaces is not a swap and is never
/// added -- it simply leaves. `or_insert` keeps the *first* frame `node` was
/// found stale: a later search re-confirming the same staleness must not
/// reset the clock, or a node held across several searches would only ever
/// show the latest one's short remaining wait.
fn mark_stale(
    pending: &mut HashMap<NodeKey, u64>,
    node: NodeKey,
    visible: &HashSet<NodeKey>,
    frame: u64,
) {
    if !replacements_of(node, visible).is_empty() {
        pending.entry(node).or_insert(frame);
    }
}

/// `node` has just unloaded -- its replacement landed, or (checked again,
/// because the visible set can have moved on since it was marked) nothing
/// replaces it any more. Returns the swap lag in frames, `now` minus the
/// frame it was first found stale, when it was a genuine swap; `None` when
/// it was not pending at all (never went stale, or was dropped by
/// [`forget_visible_again`]) or nothing replaces it any more.
fn resolve_swap(
    pending: &mut HashMap<NodeKey, u64>,
    node: NodeKey,
    visible: &HashSet<NodeKey>,
    now: u64,
) -> Option<u64> {
    let since = pending.remove(&node)?;
    if replacements_of(node, visible).is_empty() {
        None
    } else {
        Some(now - since)
    }
}

/// A node seen visible again before its replacement ever landed was not a
/// swap after all -- the camera came back (issue #378's done-when list).
/// Dropped silently rather than recorded, and rather than left to linger:
/// the next time this node goes stale it starts a fresh clock.
fn forget_visible_again(pending: &mut HashMap<NodeKey, u64>, visible: &HashSet<NodeKey>) {
    pending.retain(|node, _| !visible.contains(node));
}

/// Whether `node` can leave the arena this frame without leaving a hole.
///
/// **This is the whole fix for #262.** A node used to be unloaded the moment
/// it left the visible set, while what replaces it was still being meshed on
/// the worker pool -- tens of milliseconds during which nothing at all was
/// drawn there, so walking towards a mountain showed sky through it, over and
/// over as the search re-ran. Waiting costs a few frames of a slightly coarser
/// (or finer) mountain; not waiting costs a hole.
///
/// Drawing both meanwhile is not the alternative: a node and its replacement
/// approximate the same surface, so both in the arena at once is z-fighting
/// rather than a hole. Hence *swap*, in one `apply_node_updates` call.
///
/// **Nor does it leave while a bare node beside it is still loading**
/// (`loading`: the nodes seen and not drawn yet). The nodes replacing this
/// one can show openings it did not: a pit's floor a block lower, a cave it
/// drew shut (issue #423; the low flight's hole of 2026-11-21). The node seen
/// through such an opening may be one nothing has drawn so far -- under the
/// pit's floor, behind the cave's mouth. The search asks for it in the same
/// answer as the replacements (`cubara_world::visibility`'s stand-in), but it
/// replaces nothing, so nothing kept this node drawn until it had arrived:
/// sky in the opening for the frames between. A line of sight leaves this
/// node's space through one of its faces, so the nodes waited for are the
/// ones [`beside`] it, and of those only the [`bare`] ones: a neighbour that
/// is itself a replacement still has the node it replaces drawn in its place.
fn can_unload(
    node: NodeKey,
    visible: &HashSet<NodeKey>,
    resident: &HashSet<NodeKey>,
    loading: &[NodeKey],
) -> bool {
    let replacements = replacements_of(node, visible);
    replacements.is_empty()
        || (replacements.iter().all(|r| resident.contains(r))
            && !loading
                .iter()
                .any(|&n| beside(node, n) && bare(n, resident)))
}

/// Whether the boxes of `a` and `b` share a face: they touch along one axis
/// and overlap along the other two. Sharing only an edge or a corner is not
/// beside -- a line of sight crosses from node to node through faces.
fn beside(a: NodeKey, b: NodeKey) -> bool {
    let ((a0, a1), (b0, b1)) = (node_box(a), node_box(b));
    (0..3).any(|i| {
        (a1[i] == b0[i] || b1[i] == a0[i])
            && (0..3)
                .filter(|&j| j != i)
                .all(|j| a0[j] < b1[j] && b0[j] < a1[j])
    })
}

/// Whether nothing is drawn in `node`'s place while it loads: not the parent
/// it would split from, not a child it would merge from, and not an earlier
/// mesh of its own (a node meshed again after an edit stays drawn). Ground
/// the search has only now reached.
fn bare(node: NodeKey, resident: &HashSet<NodeKey>) -> bool {
    !resident.contains(&node)
        && !resident.contains(&node.parent())
        && !node.children().iter().any(|c| resident.contains(c))
}

/// The order to ask the mesh pool for `to_load` in: **what a swap waits for
/// first, one swap after another** -- each node that replaces one drawn now
/// (a child of a resident node, or the parent of resident children: not
/// [`bare`]) and each bare node beside a node `held` until it arrives
/// ([`can_unload`]), grouped by the swap it finishes, the nearest swap first
/// -- then the rest, nearest first. The pool takes jobs in the order asked,
/// and a search's answer asks for the swaps' replacements together with
/// ground it only now sees and the air and rock it must open to see further,
/// much of it nearer than the ring edges where the swaps are: by distance
/// alone a swap waited behind all of it, and its eight children were spread
/// over the batch, so it ended with the batch. A swap that lands late is a
/// moment the old detail stays on screen, and where a hole of a pixel or two
/// can flicker (#451's bench: the swap lag's p99 rose by half with the near
/// field's added nodes).
fn request_order(
    to_load: Vec<NodeKey>,
    resident: &HashSet<NodeKey>,
    held: &HashSet<NodeKey>,
    center: ChunkCoord,
) -> Vec<NodeKey> {
    let waited_for = beside_held(held, center);
    // The drawn node a swap replaces: the parent its children replace, the
    // children's parent itself, or the held node a bare one is beside.
    let swap_of = |n: NodeKey| {
        if resident.contains(&n.parent()) {
            Some(n.parent())
        } else if n.children().iter().any(|c| resident.contains(c)) {
            Some(n)
        } else {
            waited_for.get(&n).copied()
        }
    };
    let (mut urgent, mut rest) = (Vec::new(), Vec::new());
    for n in to_load {
        match swap_of(n) {
            Some(swap) => urgent.push((n, swap)),
            None => rest.push(n),
        }
    }
    urgent.sort_by_key(|&(n, swap)| {
        (
            node::node_dist_sq(swap, center),
            swap,
            node::node_dist_sq(n, center),
            n,
        )
    });
    node::sort_nearest_first(&mut rest, center);
    urgent.into_iter().map(|(n, _)| n).chain(rest).collect()
}

/// Every node [`beside`] one of the `held` nodes that a selection can hold
/// -- the same level across each face, or a level coarser or finer
/// (neighbours differ by at most one level, `cubara_world::node::desired_nodes_3d`)
/// -- each with the held node it is beside, the nearest one if several. Found
/// from each node's own neighbourhood rather than by testing every pair, so
/// a burst of swaps costs the main thread nothing like `held` times
/// `to_load`.
fn beside_held(held: &HashSet<NodeKey>, center: ChunkCoord) -> HashMap<NodeKey, NodeKey> {
    let mut nearest_first: Vec<NodeKey> = held.iter().copied().collect();
    node::sort_nearest_first(&mut nearest_first, center);
    let mut found = HashMap::new();
    for n in nearest_first {
        for axis in 0..3 {
            for step in [-1, 1] {
                let mut pos = n.pos;
                pos[axis] += step;
                let across = NodeKey::new(n.level, pos);
                for candidate in std::iter::once(across)
                    .chain(std::iter::once(across.parent()))
                    .chain(across.children())
                {
                    if beside(n, candidate) {
                        found.entry(candidate).or_insert(n);
                    }
                }
            }
        }
    }
    found
}

/// A box in blocks, `(min, max)` with `max` exclusive: what the streamer
/// reports a node's, or the far hole's, extent as to a measurement.
pub type BlockBox = ([f32; 3], [f32; 3]);

/// A node's box, in chunks, `max` exclusive.
fn node_box(node: NodeKey) -> ([i32; 3], [i32; 3]) {
    let o = node.chunk_origin();
    let e = node.extent_chunks();
    ([o.x, o.y, o.z], [o.x + e, o.y + e, o.z + e])
}

/// The far terrain's next hole: `prev` cut down to `want` at once, and grown
/// to all of `want` only once `drawn(kept)` says the voxels draw what lies
/// outside `kept` -- or at once when nothing of `prev` is kept (a first frame,
/// a jump). See [`NodeStreaming::voxel_box`].
pub(crate) fn next_far_hole(
    prev: ([i32; 3], [i32; 3]),
    want: ([i32; 3], [i32; 3]),
    drawn: impl FnOnce(([i32; 3], [i32; 3])) -> bool,
) -> ([i32; 3], [i32; 3]) {
    let kept = intersect(prev, want);
    if kept == EMPTY || kept == want || drawn(kept) {
        want
    } else {
        kept
    }
}

/// A box with nothing in it.
const EMPTY: ([i32; 3], [i32; 3]) = ([0; 3], [0; 3]);

/// The overlap of two boxes; [`EMPTY`] when there is none.
fn intersect(a: ([i32; 3], [i32; 3]), b: ([i32; 3], [i32; 3])) -> ([i32; 3], [i32; 3]) {
    let min = [0, 1, 2].map(|i| a.0[i].max(b.0[i]));
    let max = [0, 1, 2].map(|i| a.1[i].min(b.1[i]));
    if (0..3).all(|i| min[i] < max[i]) {
        (min, max)
    } else {
        EMPTY
    }
}

fn overlaps(a: ([i32; 3], [i32; 3]), b: ([i32; 3], [i32; 3])) -> bool {
    (0..3).all(|i| a.0[i] < b.1[i] && b.0[i] < a.1[i])
}

fn contains(outer: ([i32; 3], [i32; 3]), inner: ([i32; 3], [i32; 3])) -> bool {
    (0..3).all(|i| outer.0[i] <= inner.0[i] && inner.1[i] <= outer.1[i])
}

pub(crate) fn to_node_id(node: NodeKey) -> NodeId {
    NodeId {
        level: node.level,
        pos: node.pos,
    }
}

/// Which model (`docs/PROPOSAL_LOD_TRANSITIONS.md` §3 R2, issue #332
/// decision 3, #433) a node at `level` would draw its trees with: the real
/// oak at levels 1-2, its silhouette from level 3 up. Level 0 never matters -- it has no
/// trees to convert (`cubara_world::mesh::build_node` only populates
/// `BuiltNode::trees` at `level >= 1`) -- so this is total rather than an
/// `Option`, for the simplicity of always having an answer to plug into
/// [`MeshedNode::tree_model`] even where `trees` ends up empty.
pub(crate) fn tree_model_for(level: u32) -> cubara_render::TreeModel {
    if level <= 2 {
        cubara_render::TreeModel::Oak
    } else {
        cubara_render::TreeModel::Silhouette
    }
}

/// A node's trees as `cubara-render` draws them: where each trunk stands,
/// its height and its morph (`cubara_world::mesh::NodeTree`, #433). The one
/// conversion, for the live path, `--bench` and the transition study alike.
pub(crate) fn tree_placements(
    trees: &[cubara_world::mesh::NodeTree],
) -> Vec<cubara_render::TreePlacement> {
    trees
        .iter()
        .map(|t| cubara_render::TreePlacement {
            base: [t.site.x, t.site.base, t.site.z],
            height: t.site.height,
            morph: t.morph,
        })
        .collect()
}

/// A finished `cubara_world::mesh` job, converted to what `cubara-render`
/// understands -- `None` if the node was empty. Shared by the live
/// (worker-pool) path here and the one-shot `--bench`/`--screenshot` paths,
/// since both end up needing exactly this conversion.
pub(crate) fn to_meshed_node(built: BuiltNode) -> Option<MeshedNode> {
    let geometry = built.geometry?;
    let tree_model = tree_model_for(built.node.level);
    Some(MeshedNode {
        id: to_node_id(built.node),
        origin: geometry.origin,
        scale: geometry.scale,
        mesh: geometry.mesh,
        local_aabb: geometry.local_aabb,
        trees: tree_placements(&built.trees),
        tree_model,
    })
}

/// Where a node is in the **rendering** lifecycle
/// (`docs/PHASE2_ARCHITECTURE.md` §11.1).
///
/// This is the other half of block 2.6, and the half that deliberately did
/// **not** move into `cubara-world`. A node is a rendering unit: it exists at
/// the level it does because of its distance from a *camera*, and above level 0
/// one node covers up to 512 chunks. The simulation's lifecycle
/// ([`cubara_world::ChunkState`]) is per chunk and keyed off the *player*. As
/// one enum, a chunk would go dormant because it was far from the camera.
///
/// The states were always here -- as `HashSet` membership plus whatever
/// `MeshPool` was holding. This names them, and changes nothing: #47's bar for
/// this half is explicitly "re-expressed in terms of states with no behaviour
/// change", and restructuring the two containers into one would be a real
/// change with real risk and no behavioural benefit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeResidency {
    /// Not wanted, or not asked for yet.
    Absent,
    /// Requested; a worker is meshing it.
    InFlight,
    /// Meshed and handed to the renderer (or known to be empty).
    Resident,
}

/// Owns the mesh-worker pool and the resident node set; drives
/// [`Renderer::apply_node_updates`] each frame. One per live `Renderer` (they
/// share a lifecycle -- see `main.rs`).
pub struct NodeStreaming {
    /// Which ids the terrain is made of, including the oak.
    blocks: TerrainBlocks,
    registry: Arc<BlockRegistry>,
    layer_of: Arc<dyn Fn(&str) -> u32 + Send + Sync>,
    mesh_pool: MeshPool,
    /// Nodes meshed and uploaded (or known empty) -- a `NodeKey` already
    /// names both position and detail level, so residency is just set
    /// membership.
    resident: HashSet<NodeKey>,
    /// The chunk the camera was in as of the last [`update`](Self::update)
    /// call; `None` before the first one, so it always streams in the
    /// initial region rather than needing a separate priming call.
    center: Option<ChunkCoord>,
    /// What joins what inside each node generated so far -- kept after a node's
    /// mesh is dropped for being out of sight, so the search can still pass
    /// through it without generating it again. Behind an [`Arc`] so a search
    /// can take a snapshot without copying it; a write while a search holds
    /// one copies once.
    links: Arc<HashMap<NodeKey, visibility::NodeLinks>>,
    /// Nodes that have left the visible set but are still drawn, because what
    /// replaces them is not resident yet (#262). Dropping them on time is what
    /// made mountains flicker; they leave in the same `apply_node_updates` call
    /// that brings their replacement in.
    held: HashSet<NodeKey>,
    /// The centre of the last search applied: what `visible` was worked out
    /// around.
    searched: Option<ChunkCoord>,
    /// The box, in chunks, every node of which is drawn now, and the box the
    /// nodes draw once loaded ([`voxel_box`](Self::voxel_box)). Empty
    /// (`min == max`) before the first frame.
    far_hole: ([i32; 3], [i32; 3]),
    far_want: ([i32; 3], [i32; 3]),
    /// The nodes the camera could see ([`visibility::search`]); the only
    /// ones meshed and drawn.
    visible: HashSet<NodeKey>,
    /// Whether that search found a way out to the sky
    /// ([`visibility::Sight::open_to_sky`]): what the far terrain is drawn
    /// for ([`far_in_sight`](Self::far_in_sight)).
    open_to_sky: bool,
    /// New links arrived since the last search was started.
    visibility_stale: bool,
    /// Frames since the last search was started, to spread searches over a
    /// burst of arriving meshes.
    frames_since_visibility: u32,
    /// Searches run here, off the frame: over a full render distance one takes
    /// tens of milliseconds, which on the main thread would be a hitch every
    /// time the camera crossed a chunk.
    searcher: Searcher,
    /// The newest search started, so an older answer arriving late is ignored.
    generation: u64,
    /// A search is running that has not answered yet.
    searching: bool,
    /// This streamer's own [`update`](Self::update) calls, counted rather
    /// than timed (rule 2: no clock inside `NodeStreaming`). The only frame
    /// index [`swap_lags`](Self::swap_lags) knows about -- the bench
    /// converts it to milliseconds with its own frame times, and to blocks
    /// with the flight speed (issue #378).
    frame: u64,
    /// Nodes something still replaces, not yet landed: the frame
    /// ([`frame`](Self::frame)) the search that first found each one stale
    /// was started. Cleared without being counted if the node becomes
    /// visible again ([`forget_visible_again`]) or the world is forgotten
    /// ([`forget_world`](Self::forget_world)) -- the whole cost of this
    /// measurement while it is not being read.
    swap_pending: HashMap<NodeKey, u64>,
    /// Every swap's lag so far, in frames: [`frame`](Self::frame) when it
    /// unloaded minus when it was first found stale. Only genuine swaps
    /// (issue #378 decision 1) -- a node simply left behind, or one that
    /// became visible again first, never adds one.
    swap_lags: Vec<u64>,
}

/// One search to run: the camera's chunk and a snapshot of what is known.
struct SearchJob {
    generation: u64,
    /// [`NodeStreaming::frame`] when this search was started -- carried
    /// through to the answer so a node this search first finds stale
    /// remembers the frame the *search* started on, not the (later) one its
    /// answer is applied on. That gap is the search's own latency, which
    /// issue #378 decision 1 counts as part of the swap lag.
    frame: u64,
    center: ChunkCoord,
    links: Arc<HashMap<NodeKey, visibility::NodeLinks>>,
}

/// A search's answer: the whole render distance around `center`, and what of it
/// can be seen.
struct SearchDone {
    generation: u64,
    /// The job's own [`SearchJob::frame`], echoed back.
    frame: u64,
    center: ChunkCoord,
    desired: HashSet<NodeKey>,
    visible: HashSet<NodeKey>,
    open_to_sky: bool,
}

/// A thread that works out the render distance and what of it is visible.
struct Searcher {
    jobs: std::sync::mpsc::Sender<SearchJob>,
    done: std::sync::mpsc::Receiver<SearchDone>,
    _thread: std::thread::JoinHandle<()>,
}

impl Searcher {
    fn new() -> Self {
        let (jobs, job_rx) = std::sync::mpsc::channel::<SearchJob>();
        let (done_tx, done) = std::sync::mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("cubara-visibility".into())
            .spawn(move || {
                while let Ok(mut job) = job_rx.recv() {
                    // Only the newest matters; skip any that queued behind it.
                    while let Ok(newer) = job_rx.try_recv() {
                        job = newer;
                    }
                    let desired: HashSet<NodeKey> = node::desired_nodes_3d(
                        job.center,
                        VERTICAL_LOD_SQUASH,
                        node::DEFAULT_RING_SCHEDULE,
                    )
                    .into_iter()
                    .collect();
                    let links = &job.links;
                    let sight =
                        visibility::search(job.center, &desired, |n| links.get(&n).copied());
                    let answer = SearchDone {
                        generation: job.generation,
                        frame: job.frame,
                        center: job.center,
                        desired,
                        visible: sight.nodes,
                        open_to_sky: sight.open_to_sky,
                    };
                    if done_tx.send(answer).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn visibility thread");
        Self {
            jobs,
            done,
            _thread: thread,
        }
    }
}

/// What is known of the last search, for [`far_in_sight`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchState {
    /// The chunk the last search applied was run from; `None` before the
    /// first.
    pub searched: Option<ChunkCoord>,
    /// Another search is running or due: the last answer may be out of date.
    pub pending: bool,
    /// The last search's [`visibility::Sight::open_to_sky`].
    pub open_to_sky: bool,
}

/// Whether any of the far terrain could be seen from `eye`'s chunk -- issue
/// #424: an eye shut in underground sees none of it, and should pay for none
/// of it.
///
/// **Not drawn only when the search says so about this very place**: its
/// answer is for the chunk the eye is in now, nothing has changed since, and
/// it found no way out to the sky. Whenever that is not certain -- the eye
/// has crossed into a chunk not searched from yet, a search is running, the
/// world was just replaced -- the far terrain is drawn. Drawing it unseen
/// costs a few frames of time; not drawing it where it can be seen is a hole
/// in the horizon, at the mouth of every cave.
pub fn far_in_sight(search: SearchState, eye: ChunkCoord) -> bool {
    let sure = search.searched == Some(eye) && !search.pending;
    !sure || search.open_to_sky
}

/// How many frames arriving meshes may wait before another search is started.
/// Searches run on their own thread, but each takes tens of milliseconds, so
/// starting one every frame during a load would only queue answers nobody
/// waits for.
const VISIBILITY_EVERY_FRAMES: u32 = 10;

impl NodeStreaming {
    /// Takes the registry already behind an [`Arc`] rather than wrapping one
    /// itself: `Game` needs the *same* registry to resolve a broken block's
    /// name to an item (block 2.1d, #143), and two loads would be two id
    /// spaces -- ids are assigned per registry by sorted name (§1.2), so the
    /// same number would mean different materials on each side.
    pub fn new(
        registry: Arc<BlockRegistry>,
        structures: &cubara_voxel::StructureRegistry,
        ores: &cubara_voxel::OreRegistry,
        layer_of: impl Fn(&str) -> u32 + Send + Sync + 'static,
    ) -> Self {
        Self {
            // Resolved once, here, and carried with every meshing job.
            // Meshing used to re-derive this per node from the registry, which
            // was both wasted work and blind to structures (block 2.3a).
            blocks: TerrainBlocks::from_registry(&registry)
                .with_oak(structures, &registry)
                .with_ores(ores, &registry),
            registry,
            layer_of: Arc::new(layer_of),
            mesh_pool: MeshPool::new(),
            resident: HashSet::new(),
            held: HashSet::new(),
            searched: None,
            far_hole: ([0; 3], [0; 3]),
            far_want: ([0; 3], [0; 3]),
            center: None,
            links: Arc::new(HashMap::new()),
            visible: HashSet::new(),
            open_to_sky: true,
            visibility_stale: false,
            frames_since_visibility: 0,
            searcher: Searcher::new(),
            generation: 0,
            searching: false,
            frame: 0,
            swap_pending: HashMap::new(),
            swap_lags: Vec::new(),
        }
    }

    /// Bring the streamed set in line with the camera's current chunk (if it
    /// moved), and apply whatever's finished meshing since the last call.
    /// Cheap when nothing changed: `stream_around` only runs on a
    /// chunk-boundary crossing.
    pub fn update(&mut self, renderer: &mut Renderer, world: &World, eye: [f32; 3]) {
        self.update_lapped(renderer, world, eye, &mut ());
    }

    /// [`update`](Self::update), telling `laps` where its work went as it
    /// goes (issue #454): the search answers applied, the meshes taken, the
    /// held nodes let go -- and how many of each. Still no clock in here:
    /// whether one is read is the caller's `laps`' business, and the game's
    /// reads none.
    pub fn update_lapped(
        &mut self,
        renderer: &mut Renderer,
        world: &World,
        eye: [f32; 3],
        laps: &mut impl Laps,
    ) {
        // The frame index `swap_lags` is measured in (decision 2, #378): no
        // clock in here, just this call's own count.
        self.frame += 1;
        let center = ChunkCoord::from_world_pos(eye);
        self.frames_since_visibility += 1;
        if self.center != Some(center) {
            self.center = Some(center);
            self.start_search(center);
        } else if self.visibility_stale
            && !self.searching
            && self.frames_since_visibility >= VISIBILITY_EVERY_FRAMES
        {
            self.start_search(center);
        }
        while let Ok(done) = self.searcher.done.try_recv() {
            if done.generation == self.generation {
                self.searching = false;
                laps.count(Count::Answers, 1);
                self.follow_visibility(renderer, world, done, laps);
            }
        }
        laps.lap(Phase::Search);
        self.drain_meshes(renderer, laps);
    }

    /// Force a re-mesh of the chunk `cc` (e.g. after an edit): the worker
    /// re-reads the edit overlay, and the next [`update`](Self::update)'s
    /// `drain_meshes` swaps the geometry in atomically, so there's no gap.
    ///
    /// Always re-meshes `cc`'s level-0 node, not whatever node is currently
    /// resident there: an edit only ever happens within player reach, which
    /// is well inside the always-full-resolution near field the ring
    /// schedule keeps around the player (`World::node_at`'s doc comment) --
    /// a coarser node can never be the one actually representing an
    /// editable chunk.
    ///
    /// The job gets a [`World::snapshot`] taken now (issue #387): the edit is
    /// in it, and the tick that follows does not copy the world for it.
    pub fn invalidate(&mut self, world: &World, cc: ChunkCoord) {
        let node = NodeKey::containing(cc, 0);
        self.mesh_pool.cancel(node);
        let snapshot = Arc::new(world.snapshot());
        self.mesh_pool
            .request(&snapshot, &self.registry, &self.layer_of, node, self.blocks);
    }

    /// Drop everything streamed from the world that was just replaced -- the
    /// pause menu's New World -- so the next [`update`](Self::update) streams
    /// the new one in from nothing.
    ///
    /// **Not** the #262 swap, which keeps a node drawn until what replaces it
    /// has arrived: that is right for a change of detail over the *same*
    /// terrain, and wrong here, where every resident mesh is of terrain that
    /// no longer exists. A frame of sky while the new world meshes is correct;
    /// the old world's hills standing in for it are not.
    pub fn reset(&mut self, renderer: &mut Renderer) {
        // Every node goes now, and so does whatever was queued for the
        // renderer: the new world reuses the old one's node ids.
        self.forget_world();
        renderer.clear_nodes();
    }

    /// [`reset`](Self::reset)'s bookkeeping, apart from the GPU: forget every
    /// node, every link and every search, and return what the renderer must
    /// drop.
    fn forget_world(&mut self) -> Vec<NodeId> {
        // `held` is a subset of `resident` (still drawn, already superseded),
        // so draining `resident` unloads both.
        let unload = self.resident.drain().map(to_node_id).collect();
        self.held.clear();
        // Pending swaps are for terrain that no longer exists either; none of
        // them landed, so none is counted.
        self.swap_pending.clear();
        let in_flight: Vec<NodeKey> = self.mesh_pool.in_flight().collect();
        for node in in_flight {
            self.mesh_pool.cancel(node);
        }
        // What the old world's air joined says nothing about the new one's.
        self.links = Arc::new(HashMap::new());
        self.visible.clear();
        self.open_to_sky = true;
        self.visibility_stale = false;
        // A search still running was for the old world's links; moving the
        // generation on is what makes its answer land ignored.
        self.generation += 1;
        self.searching = false;
        // No centre means the next update starts a search wherever the camera
        // is, the same way the very first frame does.
        self.center = None;
        unload
    }

    /// Hand the search thread the camera's chunk and what is known now.
    fn start_search(&mut self, center: ChunkCoord) {
        self.generation += 1;
        self.searching = true;
        self.visibility_stale = false;
        self.frames_since_visibility = 0;
        let job = SearchJob {
            generation: self.generation,
            frame: self.frame,
            center,
            links: Arc::clone(&self.links),
        };
        // A closed channel means the thread died with a panic, which it
        // reports itself; drawing what is already loaded is the best left.
        let _ = self.searcher.jobs.send(job);
    }

    /// Apply a search's answer: drop what cannot be seen, and ask for what can
    /// and is not here yet, nearest first.
    ///
    /// **Unseen nodes are not meshed, not uploaded and not drawn** -- and a
    /// node the search has not been able to reach is never even generated. A
    /// node not yet generated counts as seen but is not searched through until
    /// its links arrive, so the visible set grows outward along what can be
    /// seen as meshes come in.
    fn follow_visibility(
        &mut self,
        renderer: &mut Renderer,
        world: &World,
        done: SearchDone,
        laps: &mut impl Laps,
    ) {
        self.searched = Some(done.center);
        let desired = done.desired;
        Arc::make_mut(&mut self.links).retain(|&n, _| worth_remembering(n, &desired));
        self.visible = done.visible;
        self.open_to_sky = done.open_to_sky;

        let stale: Vec<NodeKey> = self
            .resident
            .iter()
            .copied()
            .chain(self.mesh_pool.in_flight())
            .filter(|n| !self.visible.contains(n))
            .collect();
        // Nothing in flight is worth finishing once it has left the visible
        // set -- that is work, not geometry, and cancelling it is what keeps
        // walking back and forth from queueing meshes nobody wants.
        for &node in &stale {
            self.mesh_pool.cancel(node);
        }
        // A node seen visible again before it ever swapped was not a swap
        // (issue #378's done-when list) -- drop any pending measurement of it
        // before the loop below can start a fresh one for something else.
        forget_visible_again(&mut self.swap_pending, &self.visible);
        // ...but what is already *drawn* only leaves once its replacement has
        // arrived (#262). `held` keeps drawing until then; it is re-checked
        // here and again as each mesh lands, so the swap happens on the frame
        // the replacement is ready rather than on the next search.
        let resident_stale: Vec<NodeKey> = stale
            .into_iter()
            .filter(|n| self.resident.contains(n))
            .collect();
        // This search's answer is what first makes each of these nodes stale
        // (if nothing already had); #378 decision 1 starts the clock here,
        // on `done.frame`, not on `self.frame` when this call runs -- the gap
        // between the two is the search's own latency, which is part of what
        // the player waits for.
        for &node in &resident_stale {
            mark_stale(&mut self.swap_pending, node, &self.visible, done.frame);
        }
        let (unload_now, held, loading) = self.leave_or_hold(resident_stale);
        for &node in &unload_now {
            self.resident.remove(&node);
            if let Some(lag) = resolve_swap(&mut self.swap_pending, node, &self.visible, self.frame)
            {
                self.swap_lags.push(lag);
            }
        }
        self.held = held.into_iter().collect();
        let to_unload: Vec<NodeId> = unload_now.into_iter().map(to_node_id).collect();
        renderer.apply_node_updates(to_unload, std::iter::empty());

        let to_load: Vec<NodeKey> = loading
            .into_iter()
            .filter(|n| !self.mesh_pool.is_in_flight(*n))
            .collect();
        let to_load = request_order(to_load, &self.resident, &self.held, done.center);
        laps.count(Count::Asked, to_load.len() as u64);
        // One snapshot for the whole batch (issue #387 decision 2): every job
        // this search starts meshes the same world, and shares the one copy.
        let snapshot = Arc::new(world.snapshot());
        for node in to_load {
            self.mesh_pool
                .request(&snapshot, &self.registry, &self.layer_of, node, self.blocks);
        }
    }

    /// The voxels' box this frame, for an eye at `eye`, as the far terrain
    /// must know it (`crate::far_streaming::VoxelBox`): the box the nodes
    /// draw once loaded, and the part of it they draw now -- the owner's
    /// holes while flying (2026-11-09: "laag detail niveau, dan is het weg,
    /// heel kort daarna laadt het op een hoger niveau").
    ///
    /// The nodes' own box ([`node::covered_box_3d`]) moves the moment the eye
    /// crosses a chunk, by a whole coarsest node on its far side -- 128 blocks
    /// of ground about a kilometre ahead. Handing that to the voxels at once
    /// sank the far terrain there while the nodes that draw it were still
    /// being meshed: sky, then the finer ground. So the part they draw
    /// **shrinks at once** (where the voxels stop drawing, the far terrain
    /// draws again) and **grows only when every node the search wants in the
    /// new part is drawn**; for those few frames the far terrain stands in
    /// for the strip (`FarPlan::step`), and there is never a frame of
    /// neither.
    ///
    /// With nothing kept -- the first frame, or a jump such as `/tp` that
    /// leaves the old box behind entirely -- the new box is taken at once, as
    /// it always was: that is a loading moment, not a flight.
    pub fn voxel_box(&mut self, renderer: &Renderer, eye: [f32; 3]) -> VoxelBox {
        self.voxel_box_after(renderer.uploads_pending(), eye)
    }

    /// [`voxel_box`](Self::voxel_box), with `uploads` meshes still on their
    /// way to the renderer.
    fn voxel_box_after(&mut self, uploads: usize, eye: [f32; 3]) -> VoxelBox {
        let center = ChunkCoord::from_world_pos(eye);
        let want = node::covered_box_3d(center, node::DEFAULT_RING_SCHEDULE);
        let hole = next_far_hole(self.far_hole, want, |kept| {
            self.drawn_within(uploads, center, want, kept)
        });
        self.far_hole = hole;
        self.far_want = want;
        VoxelBox {
            wanted: hole_in_blocks(want),
            drawn: hole_in_blocks(hole),
        }
    }

    /// Whether every node the search around `center` wants inside `want` and
    /// not wholly inside `have` is drawn: arrived from the mesh workers and
    /// through the renderer's upload queue (`uploads` still in it).
    fn drawn_within(
        &self,
        uploads: usize,
        center: ChunkCoord,
        want: ([i32; 3], [i32; 3]),
        have: ([i32; 3], [i32; 3]),
    ) -> bool {
        self.searched == Some(center)
            && uploads == 0
            && self
                .visible
                .iter()
                .filter(|n| {
                    let b = node_box(**n);
                    overlaps(b, want) && !contains(have, b)
                })
                .all(|n| self.resident.contains(n))
    }

    /// Whether any of the far terrain could be seen by an eye at `eye`
    /// ([`far_in_sight`]): what the renderer is told before it draws
    /// (`Renderer::set_far_shown`).
    pub fn far_in_sight(&self, eye: [f32; 3]) -> bool {
        far_in_sight(
            SearchState {
                searched: self.searched,
                // A search running, or due because links arrived or a block
                // changed, may answer differently from the last one.
                pending: self.searching || self.visibility_stale,
                open_to_sky: self.open_to_sky,
            },
            ChunkCoord::from_world_pos(eye),
        )
    }

    /// How many nodes are drawn, held for a replacement, being meshed, and
    /// seen -- for a measurement's log (`crate::flight`).
    pub fn counts(&self) -> (usize, usize, usize, usize) {
        (
            self.resident.len(),
            self.held.len(),
            self.mesh_pool.in_flight().count(),
            self.visible.len(),
        )
    }

    /// Nothing left to do for this camera: no search running or due, nothing
    /// being meshed, nothing drawn in place of what replaces it. What a
    /// finished frame waits for (`crate::flight`).
    pub fn settled(&self) -> bool {
        !self.searching
            && !self.visibility_stale
            && self.held.is_empty()
            && self.mesh_pool.in_flight().next().is_none()
            && self.far_hole == self.far_want
    }

    /// Where `node` is in the rendering lifecycle (§11.1).
    ///
    /// Derived from the existing containers rather than stored separately --
    /// two sources of truth for one fact is how they drift apart.
    ///
    /// Nothing calls this yet: naming the states is what #47 asked for, and the
    /// streaming loop reads the containers directly because that is the code
    /// that already worked. It is the accessor anything asking "what is this
    /// node doing" should use rather than reaching into `resident`.
    #[allow(dead_code)]
    pub fn residency(&self, node: NodeKey) -> NodeResidency {
        if self.resident.contains(&node) {
            NodeResidency::Resident
        } else if self.mesh_pool.is_in_flight(node) {
            NodeResidency::InFlight
        } else {
            NodeResidency::Absent
        }
    }

    /// Take finished meshes from the worker pool and hand them to the
    /// renderer, in a fixed order (ascending `NodeKey`, [`sort_batch`])
    /// rather than whatever order the workers happened to finish in --
    /// issue #83. Marks nodes resident immediately (so they aren't
    /// re-requested); [`Renderer::apply_node_updates`] paces the actual GPU
    /// upload itself.
    fn drain_meshes(&mut self, renderer: &mut Renderer, laps: &mut impl Laps) {
        let built = self.mesh_pool.poll();
        laps.count(Count::Meshes, built.len() as u64);
        let meshed: Vec<MeshedNode> = sort_batch(built)
            .into_iter()
            .filter_map(|built| {
                if self.links.get(&built.node) != Some(&built.links) {
                    Arc::make_mut(&mut self.links).insert(built.node, built.links);
                    self.visibility_stale = true;
                }
                // Went out of sight while it was being meshed: remember what it
                // joins, but do not upload it.
                if !self.visible.contains(&built.node) {
                    return None;
                }
                self.resident.insert(built.node);
                to_meshed_node(built)
            })
            .collect();
        laps.lap(Phase::Meshes);
        // The arriving meshes may be exactly what a held node was waiting for
        // (#262). Release those in the *same* call, so the coarse node and the
        // fine ones that replace it are never both in the arena -- and never
        // neither.
        let released = self.release_held(laps);
        laps.lap(Phase::Held);
        renderer.apply_node_updates(released, meshed);
        laps.lap(Phase::Meshes);
    }

    /// Let go of the held nodes that can leave now ([`can_unload`]), and
    /// return them for the renderer to drop:
    /// [`drain_meshes`](Self::drain_meshes)'s bookkeeping, apart from the
    /// GPU.
    fn release_held(&mut self, laps: &mut impl Laps) -> Vec<NodeId> {
        if self.held.is_empty() {
            return Vec::new();
        }
        // What is seen and not drawn yet is being meshed
        // (`follow_visibility` asks for all of it), so the pool's own short
        // list is read, every frame something is held, rather than
        // `visible`, which is a hundred times longer. A node being meshed
        // again after an edit is in it too, and is not bare.
        let loading: Vec<NodeKey> = self.mesh_pool.in_flight().collect();
        let ready: Vec<NodeKey> = self
            .held
            .iter()
            .filter(|&&n| can_unload(n, &self.visible, &self.resident, &loading))
            .copied()
            .collect();
        laps.count(Count::Held, self.held.len() as u64);
        laps.count(Count::InFlight, loading.len() as u64);
        laps.count(Count::Released, ready.len() as u64);
        for node in &ready {
            self.held.remove(node);
            self.resident.remove(node);
            if let Some(lag) =
                resolve_swap(&mut self.swap_pending, *node, &self.visible, self.frame)
            {
                self.swap_lags.push(lag);
            }
        }
        ready.into_iter().map(to_node_id).collect()
    }

    /// Of the drawn nodes a search's answer has made stale: the ones that
    /// leave at once, the ones held until they can ([`can_unload`]), and --
    /// since both need it -- what the answer sees that is not drawn yet.
    /// [`follow_visibility`](Self::follow_visibility)'s decision, apart from
    /// the GPU.
    fn leave_or_hold(&self, stale: Vec<NodeKey>) -> (Vec<NodeKey>, Vec<NodeKey>, Vec<NodeKey>) {
        // Asked for by `follow_visibility` next, or already being meshed. A
        // stale node waits for the bare ones beside it as well as for its
        // replacements.
        let loading = loading_nodes(&self.visible, &self.resident);
        let (leave, hold) = stale
            .into_iter()
            .partition(|&n| can_unload(n, &self.visible, &self.resident, &loading));
        (leave, hold, loading)
    }

    /// Every swap's lag so far, in frames (issue #378): a node with a
    /// replacement counted from the frame the search that first found it
    /// stale was started to the frame it actually unloaded, because its
    /// replacement had landed. A node nothing replaces, and one seen visible
    /// again before it swapped, never adds one (design decision 1 and the
    /// done-when list). No clock in here (rule 2) -- the caller converts
    /// frames to milliseconds with its own frame times, and to blocks with
    /// the flight speed.
    pub fn swap_lags(&self) -> &[u64] {
        &self.swap_lags
    }

    /// The boxes, in blocks (`max` exclusive), of the nodes whose swap lag
    /// is open right now: found stale by a search, still drawn because what
    /// replaces them has not landed. What `--bench flight` attributes a kept
    /// frame's stale terrain to (issue #384 decision 3). Sorted, so the
    /// `HashMap`'s order never reaches a result.
    pub fn open_swaps(&self) -> Vec<BlockBox> {
        let mut nodes: Vec<NodeKey> = self.swap_pending.keys().copied().collect();
        nodes.sort();
        boxes_in_blocks(nodes)
    }

    /// The boxes, in blocks (`max` exclusive), of the nodes the last search
    /// could see that are not drawn yet: being meshed, or awaiting their
    /// request -- the frontier the search marked seen but has not searched
    /// through, since its links have not arrived. Where a kept frame shows
    /// the far terrain, or nothing, in place of land that is still loading
    /// (issue #398 decision 1, share (b)). Sorted, like
    /// [`open_swaps`](Self::open_swaps).
    pub fn loading_nodes(&self) -> Vec<BlockBox> {
        let mut nodes: Vec<NodeKey> = loading_nodes(&self.visible, &self.resident);
        nodes.sort();
        boxes_in_blocks(nodes)
    }

    /// Whether the search for the camera's current chunk has not been applied
    /// yet: the visible set is still the previous chunk's, so a node the new
    /// chunk wants and the old did not is in nobody's list (issue #398, the
    /// "not yet searched" part of share (b) that
    /// [`loading_nodes`](Self::loading_nodes) cannot name). For the
    /// measurement's log.
    pub fn search_behind(&self) -> bool {
        self.center.is_some() && self.searched != self.center
    }

    /// The box, in blocks, every node of which is drawn now, and the box the
    /// nodes draw once loaded ([`voxel_box`](Self::voxel_box)); `None` while
    /// the two agree. Where they differ, the far terrain stands in for nodes
    /// not drawn yet -- requested, in flight, or not even searched for; as
    /// level 3 draws them once its stand-in is generated
    /// (`crate::far_streaming::far_stand`) -- so ground seen there in a kept
    /// frame is land still loading (issue #398 decision 1, share (b)),
    /// whichever list the node is or is not in yet.
    pub fn far_strip(&self) -> Option<(BlockBox, BlockBox)> {
        (self.far_hole != self.far_want).then(|| {
            let blocks = |(min, max): ([i32; 3], [i32; 3])| {
                (min.map(|c| (c * 16) as f32), max.map(|c| (c * 16) as f32))
            };
            (blocks(self.far_hole), blocks(self.far_want))
        })
    }
}

/// The nodes in `visible` that are not in `resident`: seen, wanted, not
/// drawn -- in flight on the mesh pool or about to be requested. Pure, so
/// the rule is tested on its own.
fn loading_nodes(visible: &HashSet<NodeKey>, resident: &HashSet<NodeKey>) -> Vec<NodeKey> {
    visible
        .iter()
        .filter(|n| !resident.contains(n))
        .copied()
        .collect()
}

/// `nodes`' boxes in blocks, `max` exclusive, in the order given.
fn boxes_in_blocks(nodes: Vec<NodeKey>) -> Vec<BlockBox> {
    nodes
        .into_iter()
        .map(|n| {
            let (min, max) = node_box(n);
            (min.map(|c| (c * 16) as f32), max.map(|c| (c * 16) as f32))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Issue #424: the far terrain is left out only on the search's own
    /// word, about the chunk the eye is in, with nothing pending.
    #[test]
    fn the_far_terrain_is_out_of_sight_only_when_the_search_is_sure() {
        let here = ChunkCoord::new(0, -3, 0);
        let shut_in = SearchState {
            searched: Some(here),
            pending: false,
            open_to_sky: false,
        };
        assert!(!far_in_sight(shut_in, here), "shut in underground");
        // A cave mouth, or the surface: the search found the sky.
        let open = SearchState {
            open_to_sky: true,
            ..shut_in
        };
        assert!(far_in_sight(open, here));
        // The eye has walked into the next chunk, not searched from yet: it
        // may be the one with the shaft in its roof.
        assert!(far_in_sight(shut_in, ChunkCoord::new(1, -3, 0)));
        // A block was dug out, or meshes arrived: the answer may change.
        let pending = SearchState {
            pending: true,
            ..shut_in
        };
        assert!(far_in_sight(pending, here));
        // Nothing searched yet: the first frame, or a new world.
        let nothing = SearchState {
            searched: None,
            ..shut_in
        };
        assert!(far_in_sight(nothing, here));
    }

    fn real_streaming() -> NodeStreaming {
        let registry = Arc::new(cubara_render::load_registry());
        NodeStreaming::new(
            registry,
            &cubara_server::assets::load_structure_registry(),
            &cubara_server::assets::load_ore_registry(),
            |_: &str| 0,
        )
    }

    /// **What the node streaming asks of the mesh pool, and how long a swap
    /// waits for it, along the bench's flights** (#451's quiet bench: the
    /// swap lag rose with the near field's added nodes). GPU-free: the
    /// flight's eye at 60 frames a second; the game's selection and
    /// visibility search as [`NodeStreaming`] runs them (a search when the
    /// eye crosses a chunk, and ten frames after new links arrived); every
    /// node a search asks for built on this thread by the pool's own job
    /// (`build_node_shared`, one snapshot and parent cache per batch) and
    /// timed. Each job is classed by what it turned out to be: sky (above all
    /// the generator places, free), air (generated, nothing solid), rock
    /// (solid, nothing to draw) or ground (a mesh). Each batch is then run
    /// through a modelled pool of `MODEL_WORKERS` workers at the measured
    /// costs, in the order the game asks ([`request_order`]) and in nearest-
    /// first order alone, and for every node held for a swap the time until
    /// the last job it waits for is done ([`can_unload`]) is added to the
    /// search's own: a model of the swap lag without the frames between a
    /// search and its answer, or between a mesh and its upload. The spawn is loaded first, as the bench does,
    /// and not counted. Prints per flight; asserts nothing.
    #[test]
    #[ignore = "builds every node three flights ask for: cargo test -p cubara --release streaming_work_report -- --ignored --nocapture"]
    fn streaming_work_report() {
        use crate::flight::{eye_at, toward_peak, Flight, FLY_SPEED};
        use cubara_world::mesh::{build_node_shared, ParentCache};
        use std::time::Instant;

        /// The pool the model runs: the M3's eight cores less the main
        /// thread's (`MeshPool::new`).
        const MODEL_WORKERS: usize = 7;
        const CLASSES: [&str; 4] = ["sky", "air", "rock", "ground"];

        let registry = cubara_render::load_registry();
        let blocks = TerrainBlocks::from_registry(&registry)
            .with_oak(&cubara_server::assets::load_structure_registry(), &registry)
            .with_ores(&cubara_server::assets::load_ore_registry(), &registry);
        let layer_of = |_: &str| 0u32;
        let world = World::new();
        let sky_from = cubara_world::WorldGen::highest_generated_y(blocks);
        let (peak_start, peak_look) = toward_peak(&cubara_world::WorldGen::new(world.seed()));
        let flights = [
            ("low", [8.0, 120.0, 8.0], [1.0, -0.15, 0.3], 20.0f32),
            ("toward the peak", peak_start, peak_look, 30.0),
            ("down", [8.0, 450.0, 8.0], [0.4, -0.8, 0.2], 15.0),
        ];
        // When each job of `order` is done if it goes through the modelled
        // pool, each taking `cost` -- in milliseconds from the batch's start.
        let finish = |order: &[NodeKey], cost: &HashMap<NodeKey, f64>| -> HashMap<NodeKey, f64> {
            let mut free = [0.0f64; MODEL_WORKERS];
            let mut done = HashMap::new();
            for n in order {
                let w = (0..MODEL_WORKERS)
                    .min_by(|&a, &b| free[a].total_cmp(&free[b]))
                    .unwrap_or(0);
                free[w] += cost[n];
                done.insert(*n, free[w]);
            }
            done
        };
        let quantiles = |mut v: Vec<f64>| -> String {
            v.sort_by(f64::total_cmp);
            let q = |p: f64| {
                v.get(((v.len() as f64 - 1.0) * p).round() as usize)
                    .copied()
                    .unwrap_or(0.0)
            };
            format!(
                "p50/p90/p99/max {:.0}/{:.0}/{:.0}/{:.0} ms",
                q(0.5),
                q(0.9),
                q(0.99),
                q(1.0)
            )
        };
        for (name, start, look, seconds) in flights {
            let flight = Flight {
                start,
                look,
                speed: FLY_SPEED,
                seconds,
                fps: 60,
                every: 1.0,
                size: (1, 1),
                quality: crate::far_streaming::FarQuality::Off,
                far_quad_scale: crate::far_streaming::FAR_QUAD_SCALE,
                morph: true,
            };
            let mut links: HashMap<NodeKey, visibility::NodeLinks> = HashMap::new();
            let mut resident: HashSet<NodeKey> = HashSet::new();
            let mut jobs = [0u64; 4];
            let mut time = [0.0f64; 4];
            let mut levels = [[0u64; 4]; 4];
            let (mut searches, mut search_ms, mut select_ms) = (0u64, Vec::new(), Vec::new());
            let (mut swaps_asked, mut swaps_nearest) = (Vec::new(), Vec::new());
            // One search and the batch it asks for, as `follow_visibility`
            // does it; `count` false for the spawn's.
            let mut round = |center: ChunkCoord, count: bool| -> usize {
                let began = Instant::now();
                let desired: HashSet<NodeKey> = node::desired_nodes_3d(
                    center,
                    VERTICAL_LOD_SQUASH,
                    node::DEFAULT_RING_SCHEDULE,
                )
                .into_iter()
                .collect();
                let selected = began.elapsed().as_secs_f64() * 1000.0;
                let sight = visibility::search(center, &desired, |n| links.get(&n).copied());
                let searched = began.elapsed().as_secs_f64() * 1000.0;
                links.retain(|&n, _| worth_remembering(n, &desired));
                let visible = sight.nodes;
                // Drawn nodes no longer visible: held while something
                // replaces them, gone at once otherwise.
                let stale: Vec<NodeKey> = resident
                    .iter()
                    .copied()
                    .filter(|n| !visible.contains(n))
                    .collect();
                let held: HashSet<NodeKey> = stale
                    .iter()
                    .copied()
                    .filter(|&n| {
                        replacements_of(n, &visible)
                            .iter()
                            .any(|r| !resident.contains(r))
                    })
                    .collect();
                resident.retain(|n| visible.contains(n) || held.contains(n));
                let to_load: Vec<NodeKey> = visible
                    .iter()
                    .copied()
                    .filter(|n| !resident.contains(n))
                    .collect();
                let asked = request_order(to_load.clone(), &resident, &held, center);
                // What each held node waits for: what replaces it, and the
                // bare nodes beside it (`can_unload`).
                let swaps: Vec<Vec<NodeKey>> = held
                    .iter()
                    .map(|&h| {
                        to_load
                            .iter()
                            .copied()
                            .filter(|&n| {
                                n.parent() == h
                                    || h.parent() == n
                                    || (bare(n, &resident) && beside(h, n))
                            })
                            .collect::<Vec<_>>()
                    })
                    .filter(|waits| !waits.is_empty())
                    .collect();
                let mut nearest = to_load;
                node::sort_nearest_first(&mut nearest, center);
                let snapshot = Arc::new(world.snapshot());
                let cache = ParentCache::default();
                let mut cost = HashMap::new();
                for &node in &asked {
                    let began = Instant::now();
                    let built =
                        build_node_shared(&snapshot, &registry, &layer_of, node, blocks, &cache);
                    let took = began.elapsed().as_secs_f64() * 1000.0;
                    cost.insert(node, took);
                    let class = if built.geometry.is_some() {
                        3
                    } else if built.links != visibility::OPEN {
                        2
                    } else if node.world_origin()[1] > sky_from {
                        0
                    } else {
                        1
                    };
                    if count {
                        jobs[class] += 1;
                        time[class] += took;
                        levels[class][node.level as usize] += 1;
                    }
                    links.insert(node, built.links);
                }
                if count {
                    searches += 1;
                    search_ms.push(searched);
                    select_ms.push(selected);
                    for (order, lags) in
                        [(&asked, &mut swaps_asked), (&nearest, &mut swaps_nearest)]
                    {
                        let done = finish(order, &cost);
                        for waits in &swaps {
                            let last = waits.iter().map(|n| done[n]).fold(0.0, f64::max);
                            lags.push(searched + last);
                        }
                    }
                }
                // Everything asked for has arrived; what it replaces leaves.
                resident.retain(|n| visible.contains(n));
                resident.extend(asked.iter().copied());
                asked.len()
            };
            let at = ChunkCoord::from_world_pos(eye_at(&flight, 0.0));
            while round(at, false) > 0 {}
            let (mut center, mut since, mut fresh) = (at, 0u32, false);
            for f in 1..=(seconds * 60.0) as u32 {
                let here = ChunkCoord::from_world_pos(eye_at(&flight, f as f32 / 60.0));
                since += 1;
                if here != center || (fresh && since >= VISIBILITY_EVERY_FRAMES) {
                    center = here;
                    since = 0;
                    fresh = round(here, true) > 0;
                }
            }
            let total: f64 = time.iter().sum();
            println!(
                "FLIGHT {name} ({seconds} s): jobs {} = {}; job time {:.0} ms = {} ({:.0} ms per second of flight)",
                jobs.iter().sum::<u64>(),
                (0..4).map(|k| format!("{} {}", CLASSES[k], jobs[k])).collect::<Vec<_>>().join(", "),
                total,
                (0..4).map(|k| format!("{} {:.0}", CLASSES[k], time[k])).collect::<Vec<_>>().join(", "),
                total / seconds as f64,
            );
            for k in 0..4 {
                println!(
                    "  {name} {}: jobs by level {:?}, {:.3} ms a job",
                    CLASSES[k],
                    levels[k],
                    time[k] / jobs[k].max(1) as f64
                );
            }
            println!(
                "  {name} searches: {searches}, {} (of which the selection {})",
                quantiles(search_ms),
                quantiles(select_ms)
            );
            println!(
                "  {name} modelled swap lag, {} swaps that wait for a job: asked as the game asks {}; nearest first {}",
                swaps_asked.len(),
                quantiles(swaps_asked),
                quantiles(swaps_nearest)
            );
        }
    }

    /// **A node's trees reach the renderer with their own height, foot and
    /// morph** (#433 item 2): the conversion the live path, `--bench` and
    /// the transition study share. Distinct values in every field, so that
    /// a field read from the wrong place (or a constant) is seen.
    #[test]
    fn a_trees_placement_carries_its_own_height_foot_and_morph() {
        use cubara_world::mesh::NodeTree;
        use cubara_world::TreeSite;
        let tree = |x, z, base, height, morph| NodeTree {
            site: TreeSite {
                x,
                z,
                base,
                height,
                canopy_radius: 2,
            },
            morph,
        };
        let placed = tree_placements(&[tree(10, -20, 31, 4, -2), tree(-7, 3, 55, 6, 3)]);
        assert_eq!(placed.len(), 2);
        assert_eq!(
            (placed[0].base, placed[0].height, placed[0].morph),
            ([10, 31, -20], 4, -2)
        );
        assert_eq!(
            (placed[1].base, placed[1].height, placed[1].morph),
            ([-7, 55, 3], 6, 3)
        );
    }

    /// Issue #332 decision 3 (#433: the silhouette): the real oak at levels
    /// 1-2, its silhouette from level 3 up -- the line `lod_probe.rs`'s M6 table's "instanced" column
    /// describes in prose rather than asserting against directly.
    #[test]
    fn tree_model_for_splits_at_level_three() {
        assert_eq!(tree_model_for(1), cubara_render::TreeModel::Oak);
        assert_eq!(tree_model_for(2), cubara_render::TreeModel::Oak);
        assert_eq!(tree_model_for(3), cubara_render::TreeModel::Silhouette);
        assert_eq!(tree_model_for(4), cubara_render::TreeModel::Silhouette);
    }

    /// New World replaces every block: nothing streamed from the old world may
    /// survive into the new one, or the old hills are drawn over the new
    /// terrain -- a resident node the new world also wants is never meshed
    /// again, because it looks already there.
    #[test]
    fn forgetting_the_world_leaves_nothing_of_it() {
        let mut s = real_streaming();
        let world = Arc::new(World::new().snapshot());
        let (drawn, superseded, meshing) = (
            NodeKey::new(0, [0, 0, 0]),
            NodeKey::new(1, [1, 0, 0]),
            NodeKey::new(0, [5, 0, 5]),
        );
        s.resident.insert(drawn);
        s.resident.insert(superseded);
        s.held.insert(superseded);
        s.visible.insert(drawn);
        s.mesh_pool
            .request(&world, &s.registry, &s.layer_of, meshing, s.blocks);
        let built =
            cubara_world::mesh::build_node(&world, &s.registry, &|_: &str| 0, drawn, s.blocks);
        Arc::make_mut(&mut s.links).insert(drawn, built.links);
        s.center = Some(ChunkCoord::new(0, 0, 0));
        s.searching = true;
        let generation = s.generation;

        let unloaded: HashSet<NodeId> = s.forget_world().into_iter().collect();

        assert_eq!(
            unloaded,
            HashSet::from([to_node_id(drawn), to_node_id(superseded)]),
            "the renderer keeps drawing the old world"
        );
        assert!(
            s.resident.is_empty(),
            "the new world's meshes would be skipped"
        );
        assert!(s.held.is_empty());
        assert!(s.visible.is_empty());
        assert!(s.links.is_empty(), "the search walks the old world's caves");
        assert!(
            !s.mesh_pool.is_in_flight(meshing),
            "an old-world mesh can land"
        );
        assert_eq!(s.center, None, "the next update would not search again");
        assert!(!s.searching, "waiting on a search that no longer counts");
        assert!(
            s.generation > generation,
            "an old-world search can still land"
        );
    }

    /// Chunks a node covers: `2^level` on each axis from its own grid position.
    fn covers(node: NodeKey) -> Vec<[i32; 3]> {
        let e = 1i32 << node.level;
        let [x, y, z] = node.pos;
        let mut out = Vec::new();
        for cx in x * e..x * e + e {
            for cy in y * e..y * e + e {
                for cz in z * e..z * e + e {
                    out.push([cx, cy, cz]);
                }
            }
        }
        out
    }

    fn children_of(node: NodeKey) -> Vec<NodeKey> {
        let finer = node.level - 1;
        let [x, y, z] = node.pos;
        let mut out = Vec::new();
        for dx in 0..2 {
            for dy in 0..2 {
                for dz in 0..2 {
                    out.push(NodeKey::new(finer, [2 * x + dx, 2 * y + dy, 2 * z + dz]));
                }
            }
        }
        out
    }

    /// The owner walked towards a mountain and saw it flicker (#262): the
    /// coarse node left the arena the instant it stopped being visible, while
    /// its eight replacements were still on the worker pool.
    #[test]
    fn a_node_stays_until_every_child_replacing_it_has_arrived() {
        let parent = NodeKey::new(2, [3, 0, -4]);
        let kids = children_of(parent);
        let visible: HashSet<NodeKey> = kids.iter().copied().collect();

        // Nothing has arrived: holding it is the only thing that is not a hole.
        let mut resident: HashSet<NodeKey> = HashSet::from([parent]);
        assert!(!can_unload(parent, &visible, &resident, &[]));

        // Seven of eight is still a hole -- an `any` here instead of `all`
        // would leave one eighth of the mountain missing, which is precisely
        // the shape of the bug being fixed.
        for kid in kids.iter().take(7) {
            resident.insert(*kid);
            assert!(
                !can_unload(parent, &visible, &resident, &[]),
                "released with {kid:?} the last one missing"
            );
        }
        resident.insert(kids[7]);
        assert!(can_unload(parent, &visible, &resident, &[]));
    }

    /// Walking away merges eight into one, and had the identical hole. A fix
    /// that only handled splitting would pass the test above and still flicker
    /// on the way back down the mountain.
    #[test]
    fn eight_children_stay_until_the_parent_replacing_them_has_arrived() {
        let parent = NodeKey::new(2, [3, 0, -4]);
        let kids = children_of(parent);
        let visible: HashSet<NodeKey> = HashSet::from([parent]);
        let resident: HashSet<NodeKey> = kids.iter().copied().collect();

        for kid in &kids {
            assert!(!can_unload(*kid, &visible, &resident, &[]));
        }
        let with_parent: HashSet<NodeKey> = resident.union(&visible).copied().collect();
        for kid in &kids {
            assert!(can_unload(*kid, &visible, &with_parent, &[]));
        }
    }

    /// **The low flight's hole, on the side that lets a node go** (issue
    /// #423). A level-1 node splits; its eight children have arrived. One of
    /// them shows a pit the parent drew shut, and the pit's floor is in the
    /// node below -- which nothing has drawn so far and which is still being
    /// meshed. Letting the parent go now shows sky at the bottom of the pit.
    #[test]
    fn a_node_stays_until_the_bare_node_beside_it_has_arrived() {
        let parent = NodeKey::new(1, [11, 1, 2]);
        let kids = children_of(parent);
        // Under the parent's box, beside it: the level-0 node below a child.
        let below = NodeKey::new(0, [22, 1, 5]);
        assert!(beside(parent, below) && !kids.contains(&below));
        let visible: HashSet<NodeKey> = kids.iter().copied().chain([below]).collect();
        let mut resident: HashSet<NodeKey> = kids.iter().copied().chain([parent]).collect();

        assert!(
            !can_unload(parent, &visible, &resident, &[below]),
            "released with the node under the pit's floor still loading"
        );
        // It arrived: nothing is loading, and the parent goes.
        resident.insert(below);
        assert!(can_unload(parent, &visible, &resident, &[]));
    }

    /// A split parent whose children are all drawn, held, with the bare
    /// node under one of them seen: the streamer's own state for the two
    /// tests below.
    fn held_parent_beside_a_bare_node(s: &mut NodeStreaming) -> (NodeKey, NodeKey) {
        let parent = NodeKey::new(1, [11, 1, 2]);
        let below = NodeKey::new(0, [22, 1, 5]);
        for kid in children_of(parent) {
            s.visible.insert(kid);
            s.resident.insert(kid);
        }
        s.visible.insert(below);
        s.resident.insert(parent);
        (parent, below)
    }

    /// The rule above reaches the frame: as meshes land, a held node is let
    /// go only once the bare node beside it is no longer **in the mesh
    /// pool** -- which is where the frame reads what is still loading from.
    #[test]
    fn a_held_node_is_not_released_while_the_pool_meshes_a_bare_node_beside_it() {
        let mut s = real_streaming();
        let world = Arc::new(World::new().snapshot());
        let (parent, below) = held_parent_beside_a_bare_node(&mut s);
        s.held.insert(parent);
        s.mesh_pool
            .request(&world, &s.registry, &s.layer_of, below, s.blocks);

        assert_eq!(
            s.release_held(&mut ()),
            Vec::new(),
            "released while the node beside it was being meshed"
        );
        assert!(s.held.contains(&parent) && s.resident.contains(&parent));
        // The pool no longer meshes it -- it landed, or was given up -- and
        // the parent goes. Read from the pool and not from what is seen and
        // not drawn: a node nobody is meshing would never land, and waiting
        // for it would hold the parent for ever.
        s.mesh_pool.cancel(below);
        assert_eq!(s.release_held(&mut ()), vec![to_node_id(parent)]);
        assert!(!s.held.contains(&parent) && !s.resident.contains(&parent));
    }

    /// And it reaches a search's answer: a stale node whose replacements are
    /// all drawn already is still held for the bare node that answer has
    /// just seen beside it, which is about to be asked for.
    #[test]
    fn a_search_answer_holds_a_node_for_the_bare_node_it_sees_beside_it() {
        let mut s = real_streaming();
        let (parent, below) = held_parent_beside_a_bare_node(&mut s);

        let (leave, hold, loading) = s.leave_or_hold(vec![parent]);
        assert_eq!(
            (leave, hold),
            (Vec::new(), vec![parent]),
            "left with the node beside it not asked for yet"
        );
        assert_eq!(loading, vec![below], "what is seen and not drawn");
        // With that node drawn nothing is loading, and the parent leaves.
        s.resident.insert(below);
        assert_eq!(
            s.leave_or_hold(vec![parent]),
            (vec![parent], Vec::new(), Vec::new())
        );
    }

    /// Only a **bare** node **beside** it holds a node back. Anything else
    /// that is loading has something drawn in its place, or cannot be seen
    /// into from this node's space -- and waiting for those would tie every
    /// swap to every other.
    #[test]
    fn a_node_does_not_wait_for_what_is_drawn_meanwhile_or_not_beside_it() {
        let parent = NodeKey::new(1, [11, 1, 2]);
        let kids = children_of(parent);
        let with = |more: &[NodeKey]| -> HashSet<NodeKey> {
            kids.iter().copied().chain(more.iter().copied()).collect()
        };
        let drawn = |more: &[NodeKey]| -> HashSet<NodeKey> {
            kids.iter()
                .copied()
                .chain([parent])
                .chain(more.iter().copied())
                .collect()
        };

        // A neighbour's child, with that neighbour still drawn: the
        // neighbour stands in its place until it arrives.
        let neighbour = NodeKey::new(1, [12, 1, 2]);
        let its_child = children_of(neighbour)[0];
        assert!(beside(parent, its_child));
        assert!(can_unload(
            parent,
            &with(&[its_child]),
            &drawn(&[neighbour]),
            &[its_child]
        ));
        // The same child with the neighbour gone is bare, and is waited for.
        assert!(!can_unload(
            parent,
            &with(&[its_child]),
            &drawn(&[]),
            &[its_child]
        ));

        // A coarser node merging beside it, its children still drawn.
        let merging = NodeKey::new(2, [6, 0, 1]);
        let one_of_its = children_of(merging)[0];
        assert!(beside(parent, merging));
        assert!(can_unload(
            parent,
            &with(&[merging]),
            &drawn(&[one_of_its]),
            &[merging]
        ));
        assert!(!can_unload(
            parent,
            &with(&[merging]),
            &drawn(&[]),
            &[merging]
        ));

        // A node beside it that is being meshed again after an edit: its
        // earlier mesh is drawn meanwhile.
        let edited = NodeKey::new(0, [22, 1, 5]);
        assert!(beside(parent, edited));
        assert!(can_unload(
            parent,
            &with(&[edited]),
            &drawn(&[edited]),
            &[edited]
        ));

        // Bare, and not beside: two nodes away, and touching only an edge.
        let far = NodeKey::new(0, [26, 2, 4]);
        let edge = NodeKey::new(0, [24, 1, 4]);
        assert!(!beside(parent, far) && !beside(parent, edge));
        assert!(can_unload(
            parent,
            &with(&[far, edge]),
            &drawn(&[]),
            &[far, edge]
        ));

        // And a node nothing replaces leaves at once whatever is loading
        // beside it: its place is not being looked into any more.
        let gone = NodeKey::new(1, [40, 0, 40]);
        let beside_it = NodeKey::new(1, [41, 0, 40]);
        assert!(can_unload(
            gone,
            &HashSet::from([beside_it]),
            &HashSet::from([gone]),
            &[beside_it]
        ));
    }

    /// Beside is a shared face, at any two levels; an edge or a corner in
    /// common is not, and neither is a node inside the other.
    #[test]
    fn beside_is_a_shared_face() {
        let node = NodeKey::new(1, [0, 0, 0]); // chunks 0..2 on every axis
        for (other, want) in [
            (NodeKey::new(1, [1, 0, 0]), true),
            (NodeKey::new(1, [-1, 0, 0]), true),
            (NodeKey::new(1, [0, 1, 0]), true),
            (NodeKey::new(1, [0, -1, 0]), true),
            (NodeKey::new(1, [0, 0, 1]), true),
            (NodeKey::new(1, [0, 0, -1]), true),
            // A finer node on a face, and a coarser one that holds a face.
            (NodeKey::new(0, [2, 1, 0]), true),
            (NodeKey::new(0, [1, -1, 1]), true),
            (NodeKey::new(2, [0, 0, -1]), true),
            // An edge, a corner.
            (NodeKey::new(1, [1, 1, 0]), false),
            (NodeKey::new(1, [1, 1, 1]), false),
            (NodeKey::new(0, [2, 2, 0]), false),
            // Apart.
            (NodeKey::new(1, [2, 0, 0]), false),
            (NodeKey::new(0, [3, 0, 0]), false),
            // Inside it, around it, itself.
            (NodeKey::new(0, [1, 1, 1]), false),
            (NodeKey::new(2, [0, 0, 0]), false),
            (node, false),
        ] {
            assert_eq!(beside(node, other), want, "{other:?}");
            assert_eq!(beside(other, node), want, "{other:?}, the other way round");
        }
    }

    /// Only *replacement* is worth waiting for. A node the camera has simply
    /// left behind must go at once, or the render distance stops bounding
    /// anything and the arena fills with the world behind you.
    #[test]
    fn what_split_or_merged_is_remembered_until_its_replacement_is_known() {
        let node = NodeKey::new(1, [2, -1, 5]);
        let desired: HashSet<NodeKey> = [node].into_iter().collect();
        assert!(worth_remembering(node, &desired));
        assert!(
            worth_remembering(node.parent(), &desired),
            "forgot the parent it split from"
        );
        assert!(
            worth_remembering(node.children()[3], &desired),
            "forgot a child it merged from"
        );
        assert!(!worth_remembering(node.parent().parent(), &desired));
        assert!(!worth_remembering(NodeKey::new(1, [9, 9, 9]), &desired));
    }

    #[test]
    fn a_node_nothing_is_replacing_leaves_immediately() {
        let gone = NodeKey::new(1, [40, 0, 40]);
        let elsewhere: HashSet<NodeKey> = HashSet::from([NodeKey::new(1, [-40, 0, -40])]);
        let resident: HashSet<NodeKey> = HashSet::from([gone]);
        assert!(can_unload(gone, &elsewhere, &resident, &[]));
        // Including when nothing at all is visible (walked into a cave).
        assert!(can_unload(gone, &HashSet::new(), &resident, &[]));
    }

    /// The property the old code broke, stated over chunks rather than nodes:
    /// while detail changes, every chunk that was drawn stays drawn.
    #[test]
    fn no_chunk_is_ever_uncovered_while_the_detail_changes() {
        let parent = NodeKey::new(2, [1, 0, 1]);
        let kids = children_of(parent);
        let before: HashSet<[i32; 3]> = covers(parent).into_iter().collect();
        let visible: HashSet<NodeKey> = kids.iter().copied().collect();

        // Feed the children in one at a time, unloading the parent as soon as
        // the policy allows, and check coverage after every single step.
        let mut resident: HashSet<NodeKey> = HashSet::from([parent]);
        let mut parent_gone = false;
        for kid in kids.iter().chain(std::iter::once(&kids[7])) {
            resident.insert(*kid);
            if !parent_gone && can_unload(parent, &visible, &resident, &[]) {
                resident.remove(&parent);
                parent_gone = true;
            }
            let drawn: HashSet<[i32; 3]> = resident.iter().flat_map(|&n| covers(n)).collect();
            for chunk in &before {
                assert!(
                    drawn.contains(chunk),
                    "chunk {chunk:?} was drawn and then was not"
                );
            }
        }
        assert!(
            parent_gone,
            "the coarse node must leave once it is replaced"
        );
    }

    /// The rendering lifecycle is a *different* lifecycle from the simulation's
    /// (§11.1), and this is where that is asserted rather than only written
    /// down: the two enums are not convertible, and nothing here mentions a
    /// `ChunkCoord`.
    /// The far terrain gives ground to the voxels only once they draw it, and
    /// takes it back at once: never neither (the owner's holes while flying).
    #[test]
    fn the_far_hole_grows_only_when_the_voxels_are_drawn_and_shrinks_at_once() {
        let old = ([0, 0, 0], [8, 8, 8]);
        // The eye moved +x: the box gains x 8..16 and loses x 0..8.
        let moved = ([8, 0, 0], [16, 8, 8]);
        let shifted = ([4, 0, 0], [12, 8, 8]);
        // Overlapping: keep the overlap until the new part is drawn...
        assert_eq!(
            next_far_hole(old, shifted, |_| false),
            ([4, 0, 0], [8, 8, 8])
        );
        // ...and take all of it once it is, asked about exactly the overlap.
        let mut asked = None;
        let grown = next_far_hole(old, shifted, |kept| {
            asked = Some(kept);
            true
        });
        assert_eq!(grown, shifted);
        assert_eq!(asked, Some(([4, 0, 0], [8, 8, 8])));
        // Shrinking is never waited for.
        assert_eq!(
            next_far_hole(old, ([0, 0, 0], [4, 8, 8]), |_| false),
            ([0, 0, 0], [4, 8, 8])
        );
        // Nothing kept (first frame, a jump): the new box at once.
        assert_eq!(next_far_hole(EMPTY, old, |_| false), old);
        assert_eq!(next_far_hole(old, moved, |_| false), moved);
    }

    #[test]
    fn node_residency_names_the_states_that_already_existed() {
        let states = [
            NodeResidency::Absent,
            NodeResidency::InFlight,
            NodeResidency::Resident,
        ];
        // Distinct, and exhaustive: absent, asked for, arrived.
        for (i, a) in states.iter().enumerate() {
            for (j, b) in states.iter().enumerate() {
                assert_eq!(i == j, a == b, "{a:?} vs {b:?}");
            }
        }
    }

    // Issue #378's swap-lag measurement: `mark_stale`/`resolve_swap`/
    // `forget_visible_again` are the pure bookkeeping `follow_visibility` and
    // `drain_meshes` call, kept separate so they can be tested without a GPU
    // or a `World` -- the same reason `can_unload` and `replacements_of`
    // above are free functions rather than inlined.

    /// Done-when #1: a node swapped on the frame its replacement lands has a
    /// lag equal to that frame minus the one the search that found it stale
    /// started on.
    #[test]
    fn swap_lag_is_the_frames_between_going_stale_and_landing() {
        let parent = NodeKey::new(2, [3, 0, -4]);
        let replacement: HashSet<NodeKey> = children_of(parent).into_iter().collect();
        let mut pending = HashMap::new();

        mark_stale(&mut pending, parent, &replacement, 10);
        assert_eq!(
            resolve_swap(&mut pending, parent, &replacement, 42),
            Some(32),
            "landed 32 frames after the search that first found it stale"
        );
        assert!(pending.is_empty(), "a resolved swap must not stay pending");
    }

    /// A later search re-confirming the same staleness must not push the
    /// start of the clock later -- `mark_stale` keeps the first frame.
    #[test]
    fn mark_stale_keeps_the_first_frame_not_the_latest() {
        let parent = NodeKey::new(2, [3, 0, -4]);
        let replacement: HashSet<NodeKey> = children_of(parent).into_iter().collect();
        let mut pending = HashMap::new();

        mark_stale(&mut pending, parent, &replacement, 5);
        mark_stale(&mut pending, parent, &replacement, 50);
        assert_eq!(pending[&parent], 5);
        assert_eq!(
            resolve_swap(&mut pending, parent, &replacement, 60),
            Some(55)
        );
    }

    /// Done-when #2 (half): a node nothing replaces is never even marked --
    /// it is not a swap, so there is nothing to time.
    #[test]
    fn mark_stale_ignores_a_node_nothing_replaces() {
        let gone = NodeKey::new(1, [40, 0, 40]);
        let elsewhere: HashSet<NodeKey> = HashSet::from([NodeKey::new(1, [-40, 0, -40])]);
        let mut pending = HashMap::new();

        mark_stale(&mut pending, gone, &elsewhere, 7);
        assert!(pending.is_empty());
    }

    /// Done-when #2 (other half): a node marked stale while something
    /// replaced it, which later stops being replaced (the camera moved on
    /// again before it swapped), must not be counted as a swap when it
    /// finally leaves.
    #[test]
    fn resolve_swap_forgets_a_node_nothing_replaces_any_more() {
        let parent = NodeKey::new(2, [3, 0, -4]);
        let replacement: HashSet<NodeKey> = children_of(parent).into_iter().collect();
        let mut pending = HashMap::new();
        mark_stale(&mut pending, parent, &replacement, 3);

        let nothing_replaces_it_now = HashSet::new();
        assert_eq!(
            resolve_swap(&mut pending, parent, &nothing_replaces_it_now, 9),
            None,
            "left because nothing replaces it, not because a replacement landed"
        );
        assert!(
            pending.is_empty(),
            "left either way -- resolved, not retried"
        );
    }

    /// Done-when #3: a node seen visible again before it ever swapped was
    /// not a swap after all, and records nothing when it is later dropped.
    #[test]
    fn a_node_visible_again_before_swapping_records_nothing() {
        let parent = NodeKey::new(2, [3, 0, -4]);
        let replacement: HashSet<NodeKey> = children_of(parent).into_iter().collect();
        let mut pending = HashMap::new();
        mark_stale(&mut pending, parent, &replacement, 1);

        let visible_again: HashSet<NodeKey> = HashSet::from([parent]);
        forget_visible_again(&mut pending, &visible_again);
        assert!(pending.is_empty());
        assert_eq!(
            resolve_swap(&mut pending, parent, &replacement, 100),
            None,
            "nothing was pending any more -- it must not silently restart"
        );
    }

    /// `forget_visible_again` only drops nodes that actually came back; a
    /// node still legitimately held keeps its stale-since frame.
    #[test]
    fn forget_visible_again_leaves_nodes_still_stale_alone() {
        let parent = NodeKey::new(2, [3, 0, -4]);
        let replacement: HashSet<NodeKey> = children_of(parent).into_iter().collect();
        let mut pending = HashMap::new();
        mark_stale(&mut pending, parent, &replacement, 4);

        // `parent` is still not in the visible set -- only its replacement is.
        forget_visible_again(&mut pending, &replacement);
        assert_eq!(
            pending[&parent], 4,
            "still genuinely stale, not visible again"
        );
    }

    /// Done-when #4: `reset` (`forget_world`) forgets pending swaps rather
    /// than counting them as having landed -- the terrain they were waiting
    /// on no longer exists either.
    #[test]
    fn forget_world_drops_pending_swaps_without_counting_them() {
        let mut s = real_streaming();
        s.swap_pending.insert(NodeKey::new(0, [9, 9, 9]), 1);
        s.frame = 500;

        s.forget_world();

        assert!(
            s.swap_pending.is_empty(),
            "a reset world has nothing pending"
        );
        assert!(
            s.swap_lags.is_empty(),
            "an unresolved swap must not be counted as a landed one"
        );
    }

    #[test]
    fn swap_lags_starts_empty() {
        let s = real_streaming();
        assert!(s.swap_lags().is_empty());
    }

    /// Issue #398 decision 1, share (b): a node the search can see and the
    /// renderer does not draw yet is loading -- whether it is in flight or
    /// still waiting to be requested -- and a node that is drawn, or one the
    /// search cannot see, is not.
    /// **What a swap waits for is asked for first, one swap after another**
    /// (#451): the children replacing a resident parent, the parent
    /// replacing resident children, and a bare node beside a node held until
    /// it arrives go ahead of ground and air only now seen -- even nearer
    /// ones -- the nodes of one swap together, the nearest swap first; the
    /// rest nearest first; nothing dropped or asked for twice. Goes red with
    /// the nearest-first order alone (the near frontier first), with the held
    /// node's bare neighbour left among the rest, or with a swap's nodes
    /// interleaved with another's.
    #[test]
    fn what_a_swap_waits_for_is_asked_for_first() {
        let center = ChunkCoord::new(0, 0, 0);
        // Two level-1 nodes splitting, as far from the eye as each other:
        // their children replace them, and by distance alone they interleave.
        let split_a = NodeKey::new(1, [5, 0, 0]);
        let split_b = NodeKey::new(1, [0, 0, 5]);
        // Children of a level-1 node merging into it: it replaces them.
        let merged = NodeKey::new(1, [-6, 0, 0]);
        let merging_child = merged.children()[0];
        // A level-0 node held until the bare node beside it arrives.
        let held_node = NodeKey::new(0, [0, 0, 8]);
        let beside_held = NodeKey::new(0, [-1, 0, 8]);
        // Ground and air only now seen, nearer than all of the above.
        let frontier = [NodeKey::new(0, [1, 1, 0]), NodeKey::new(0, [0, -2, 1])];
        let resident: HashSet<NodeKey> = [split_a, split_b, merging_child, held_node]
            .into_iter()
            .collect();
        let held: HashSet<NodeKey> = [held_node].into_iter().collect();
        let (children_a, children_b) = (split_a.children(), split_b.children());
        let mut to_load: Vec<NodeKey> = children_a.iter().chain(&children_b).copied().collect();
        to_load.extend([merged, beside_held]);
        to_load.extend(frontier);
        let order = request_order(to_load.clone(), &resident, &held, center);
        assert_eq!(order.len(), to_load.len());
        let set: HashSet<NodeKey> = order.iter().copied().collect();
        assert_eq!(set.len(), order.len(), "a node asked for twice");
        // One swap after another, the nearest swap first: the held node's
        // bare neighbour (its swap 8 chunks out), then each splitting node's
        // eight children together (11 chunks out; the tie goes by key), then
        // the merge (12 chunks out); then the frontier.
        let swap = |range: std::ops::Range<usize>| -> HashSet<NodeKey> {
            order[range].iter().copied().collect()
        };
        assert_eq!(order[0], beside_held, "{order:?}");
        assert_eq!(
            swap(1..9),
            children_b.iter().copied().collect(),
            "{order:?}"
        );
        assert_eq!(
            swap(9..17),
            children_a.iter().copied().collect(),
            "{order:?}"
        );
        assert_eq!(order[17], merged, "{order:?}");
        // The rest nearest first.
        let rest = &order[18..];
        assert!(frontier.iter().all(|n| rest.contains(n)), "{order:?}");
        let mut nearest = rest.to_vec();
        node::sort_nearest_first(&mut nearest, center);
        assert_eq!(rest, &nearest[..]);
    }

    /// [`beside_held`] finds exactly the nodes one level apart or less that
    /// share a face with one of its nodes: against every such node in a
    /// neighbourhood, tested pair by pair with [`beside`].
    #[test]
    fn beside_held_is_every_node_beside_one_of_them() {
        for n in [
            NodeKey::new(0, [3, -2, 5]),
            NodeKey::new(1, [-1, 0, 2]),
            NodeKey::new(2, [0, 1, -1]),
        ] {
            let found: HashSet<NodeKey> =
                beside_held(&[n].into_iter().collect(), ChunkCoord::new(0, 0, 0))
                    .into_keys()
                    .collect();
            let mut expected = HashSet::new();
            for level in n.level.saturating_sub(1)..=n.level + 1 {
                let scale = |p: i32| {
                    if level > n.level {
                        p.div_euclid(2)
                    } else if level < n.level {
                        p * 2
                    } else {
                        p
                    }
                };
                let c = n.pos.map(scale);
                for dx in -3..=4 {
                    for dy in -3..=4 {
                        for dz in -3..=4 {
                            let m = NodeKey::new(level, [c[0] + dx, c[1] + dy, c[2] + dz]);
                            if beside(n, m) {
                                expected.insert(m);
                            }
                        }
                    }
                }
            }
            assert!(!expected.is_empty());
            assert_eq!(found, expected, "{n:?}");
        }
    }

    #[test]
    fn a_loading_node_is_visible_and_not_resident() {
        let (drawn, meshing, waiting, unseen) = (
            NodeKey::new(1, [0, 0, 0]),
            NodeKey::new(1, [1, 0, 0]),
            NodeKey::new(2, [1, 0, 1]),
            NodeKey::new(1, [5, 0, 5]),
        );
        let visible = HashSet::from([drawn, meshing, waiting]);
        let resident = HashSet::from([drawn, unseen]);
        let mut loading = loading_nodes(&visible, &resident);
        loading.sort();
        assert_eq!(loading, vec![meshing, waiting]);
        // Through the streamer, in blocks and sorted: level 1 at grid x 1 is
        // blocks 32..64, level 2 at grid (1, 0, 1) is 64..128.
        let mut s = real_streaming();
        s.visible = visible;
        s.resident = resident;
        assert_eq!(
            s.loading_nodes(),
            vec![
                ([32.0, 0.0, 0.0], [64.0, 32.0, 32.0]),
                ([64.0, 0.0, 64.0], [128.0, 64.0, 128.0]),
            ]
        );
    }

    /// The far terrain's strip is the two boxes, in blocks, while the hole
    /// it leaves lags behind what the voxels want; nothing while they agree.
    /// **The voxels' box hands the far terrain both boxes**: after the eye
    /// crosses into a new box, the box the nodes want and -- until the
    /// search for the new chunk has answered and every node it sees in the
    /// strip is drawn -- the part of it they draw, which the far terrain
    /// stands in for the rest of (`crate::far_streaming::far_stand`). Then
    /// the two agree.
    #[test]
    fn the_voxel_box_hands_over_the_strip_only_once_its_nodes_are_drawn() {
        let mut s = real_streaming();
        let (before, after) = (ChunkCoord::new(7, 6, 2), ChunkCoord::new(8, 6, 2));
        let old = node::covered_box_3d(before, node::DEFAULT_RING_SCHEDULE);
        let new = node::covered_box_3d(after, node::DEFAULT_RING_SCHEDULE);
        (s.far_hole, s.far_want) = (old, old);
        let eye = [133.0, 101.0, 45.0];
        assert_eq!(ChunkCoord::from_world_pos(eye), after);
        let waiting = s.voxel_box_after(0, eye);
        assert_eq!(
            waiting.wanted,
            hole_in_blocks(new),
            "the box the nodes want"
        );
        assert_eq!(
            waiting.drawn,
            hole_in_blocks(next_far_hole(old, new, |_| false)),
            "the part they draw while the strip's nodes are not searched for"
        );
        assert_ne!(
            waiting.drawn, waiting.wanted,
            "no strip waits: the test tells nothing"
        );
        // The search has answered and sees nothing in the strip undrawn; one
        // mesh still on its way to the renderer, then none.
        s.searched = Some(after);
        assert_eq!(s.voxel_box_after(1, eye), waiting, "a mesh still to go up");
        let done = s.voxel_box_after(0, eye);
        assert_eq!(
            done,
            VoxelBox::settled(hole_in_blocks(new)),
            "all of it drawn"
        );
    }

    #[test]
    fn the_far_strip_is_the_hole_and_the_want_while_they_differ() {
        let mut s = real_streaming();
        assert_eq!(s.far_strip(), None, "nothing wanted yet");
        s.far_hole = ([0, 0, 0], [8, 8, 8]);
        s.far_want = ([4, 0, 0], [12, 8, 8]);
        assert_eq!(
            s.far_strip(),
            Some((
                ([0.0, 0.0, 0.0], [128.0, 128.0, 128.0]),
                ([64.0, 0.0, 0.0], [192.0, 128.0, 128.0]),
            ))
        );
        s.far_hole = s.far_want;
        assert_eq!(s.far_strip(), None, "the voxels draw all they want");
    }

    /// The search is behind the camera between the chunk crossing and the
    /// answer being applied -- not before the first frame, and not once the
    /// answer for the current chunk has landed.
    #[test]
    fn the_search_is_behind_between_a_crossing_and_its_answer() {
        let mut s = real_streaming();
        assert!(!s.search_behind(), "nothing asked for yet");
        s.center = Some(ChunkCoord::new(3, 0, 3));
        assert!(s.search_behind(), "crossed, no answer yet");
        s.searched = Some(ChunkCoord::new(2, 0, 3));
        assert!(s.search_behind(), "the answer is the old chunk's");
        s.searched = s.center;
        assert!(!s.search_behind());
    }

    /// The worst of one quantity over a replayed flight, and when.
    #[derive(Clone, Copy, Default)]
    struct Worst {
        value: f64,
        t: f32,
    }

    impl Worst {
        fn offer(&mut self, value: f64, t: f32) {
            if value > self.value {
                *self = Worst { value, t };
            }
        }
    }

    impl std::fmt::Display for Worst {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:.0} (t {:.2} s)", self.value, self.t)
        }
    }

    /// [`can_unload`], counting the [`beside`] checks it makes: the work
    /// that grows with held nodes times loading ones. Checked against the
    /// real one on every call by [`node_burst_report`].
    fn can_unload_counted(
        node: NodeKey,
        visible: &HashSet<NodeKey>,
        resident: &HashSet<NodeKey>,
        loading: &[NodeKey],
    ) -> (bool, u64) {
        let replacements = replacements_of(node, visible);
        if replacements.is_empty() {
            return (true, 0);
        }
        if !replacements.iter().all(|r| resident.contains(r)) {
            return (false, 0);
        }
        let mut checks = 0;
        for &n in loading {
            checks += 1;
            if beside(node, n) && bare(n, resident) {
                return (false, checks);
            }
        }
        (true, checks)
    }

    /// **What the voxel nodes ask of the main thread, frame by frame, along
    /// the gate's four flights** (issue #454: rare frames of 17-34 ms, and
    /// which main-thread work grows with a burst). GPU-free: the flight's
    /// eye at 60 frames a second; the game's selection and visibility search
    /// as [`NodeStreaming`] runs them (a search when the eye crosses a
    /// chunk, and ten frames after new links arrived), its answer applied as
    /// many frames later as the search took here; every node asked for built
    /// on this thread by the pool's own job (`build_node_shared`), timed,
    /// and run through a modelled pool of `MODEL_WORKERS` workers in the
    /// order asked; a finished mesh taken on the first frame after it is
    /// done. Per frame it counts what [`NodeStreaming::update`] does in
    /// proportion to a burst: the nodes an answer asks for and the stale
    /// ones it sorts, the meshes taken and what they upload, the held nodes
    /// checked against the nodes in flight ([`can_unload`]'s [`beside`]
    /// checks, in an answer and in `release_held`), and the maps -- and it
    /// times the real [`can_unload`] passes and the answer's own
    /// bookkeeping. The start is loaded first, as the bench does, and not
    /// counted. Prints the worst frame of each; asserts only that its
    /// counting `can_unload` agrees with the real one.
    #[test]
    #[ignore = "builds every node four flights ask for: cargo test -p cubara --release node_burst_report -- --ignored --nocapture"]
    fn node_burst_report() {
        use crate::flight::{eye_at, toward_peak, Flight, FLY_SPEED};
        use cubara_world::mesh::{build_node_shared, ParentCache};
        use std::time::Instant;

        /// The pool the model runs: the M3's eight cores less the main
        /// thread's (`MeshPool::new`).
        const MODEL_WORKERS: usize = 7;
        const FRAME_MS: f64 = 1000.0 / 60.0;

        /// What a node's job came to: its time on this thread, its links,
        /// and the bytes its mesh uploads (none when nothing is drawn).
        #[derive(Clone, Copy)]
        struct Job {
            ms: f64,
            links: visibility::NodeLinks,
            bytes: u64,
        }
        struct Answer {
            due: u32,
            center: ChunkCoord,
            desired: HashSet<NodeKey>,
            visible: HashSet<NodeKey>,
        }
        /// The worst frame of everything counted.
        #[derive(Default)]
        struct Report {
            answers: u64,
            asked: Worst,
            stale: Worst,
            loading: Worst,
            answer_checks: Worst,
            answer_ms: Worst,
            meshes: Worst,
            uploads: Worst,
            upload_mb: Worst,
            held: Worst,
            in_flight: Worst,
            release_checks: Worst,
            release_ms: Worst,
            links: Worst,
            visible: Worst,
            resident: Worst,
            checks: u64,
            check_ms: f64,
            /// Every frame's `release_held` pass: when, held, in flight,
            /// `beside` checks, ms.
            releases: Vec<(f32, usize, usize, u64, f64)>,
            /// Every answer: when, the links before `retain`, stale nodes,
            /// loading nodes, and the answer's own work in ms (`retain`
            /// alone, then the whole).
            answer_list: Vec<(f32, usize, usize, usize, f64, f64)>,
        }

        let registry = cubara_render::load_registry();
        let blocks = TerrainBlocks::from_registry(&registry)
            .with_oak(&cubara_server::assets::load_structure_registry(), &registry)
            .with_ores(&cubara_server::assets::load_ore_registry(), &registry);
        let layer_of = |_: &str| 0u32;
        let world = World::new();
        let cache = ParentCache::default();
        let mut jobs: HashMap<NodeKey, Job> = HashMap::new();
        let (peak_start, peak_look) = toward_peak(&cubara_world::WorldGen::new(world.seed()));
        let flights = [
            ("low", [8.0, 120.0, 8.0], [1.0, -0.15, 0.3], 20.0f32),
            ("high", [8.0, 400.0, 8.0], [1.0, -0.35, 0.3], 20.0),
            ("down", [8.0, 450.0, 8.0], [0.4, -0.8, 0.2], 15.0),
            ("toward the peak", peak_start, peak_look, 30.0),
        ];
        for (name, start, look, seconds) in flights {
            let flight = Flight {
                start,
                look,
                speed: FLY_SPEED,
                seconds,
                fps: 60,
                every: 1.0,
                size: (1, 1),
                quality: crate::far_streaming::FarQuality::Off,
                far_quad_scale: crate::far_streaming::FAR_QUAD_SCALE,
                morph: true,
            };
            let mut links: HashMap<NodeKey, visibility::NodeLinks> = HashMap::new();
            let mut resident: HashSet<NodeKey> = HashSet::new();
            let mut held: HashSet<NodeKey> = HashSet::new();
            let mut visible: HashSet<NodeKey> = HashSet::new();
            // Each node being meshed, and when its job is done (ms).
            let mut in_flight: HashMap<NodeKey, f64> = HashMap::new();
            let mut workers = [0.0f64; MODEL_WORKERS];
            let mut answer: Option<Answer> = None;
            let (mut center, mut stale_links, mut since) = (None, false, 0u32);
            let mut report = Report::default();
            // One frame, `at` the flight's time or `None` while the start
            // loads (not counted). Returns whether anything is still to do.
            let mut frame = |f: u32, eye: [f32; 3], at: Option<f32>, report: &mut Report| -> bool {
                let now = f64::from(f) * FRAME_MS;
                let count = at.is_some();
                let t = at.unwrap_or(0.0);
                let here = ChunkCoord::from_world_pos(eye);
                since += 1;
                let searching = answer.is_some();
                if center != Some(here) || (stale_links && !searching && since >= 10) {
                    center = Some(here);
                    stale_links = false;
                    since = 0;
                    let began = Instant::now();
                    let desired: HashSet<NodeKey> = node::desired_nodes_3d(
                        here,
                        VERTICAL_LOD_SQUASH,
                        node::DEFAULT_RING_SCHEDULE,
                    )
                    .into_iter()
                    .collect();
                    let sight = visibility::search(here, &desired, |n| links.get(&n).copied());
                    let took = began.elapsed().as_secs_f64() * 1000.0;
                    answer = Some(Answer {
                        due: f + ((took / FRAME_MS).ceil() as u32).max(1),
                        center: here,
                        desired,
                        visible: sight.nodes,
                    });
                }
                let mut asked = 0usize;
                if answer.as_ref().is_some_and(|a| a.due <= f) {
                    let a = answer.take().expect("due");
                    let began = Instant::now();
                    let links_before = links.len();
                    links.retain(|&n, _| worth_remembering(n, &a.desired));
                    let retain_ms = began.elapsed().as_secs_f64() * 1000.0;
                    visible = a.visible;
                    let stale: Vec<NodeKey> = resident
                        .iter()
                        .copied()
                        .chain(in_flight.keys().copied())
                        .filter(|n| !visible.contains(n))
                        .collect();
                    for n in &stale {
                        in_flight.remove(n);
                    }
                    let resident_stale: Vec<NodeKey> = stale
                        .iter()
                        .copied()
                        .filter(|n| resident.contains(n))
                        .collect();
                    let loading = loading_nodes(&visible, &resident);
                    let (leave, hold): (Vec<NodeKey>, Vec<NodeKey>) = resident_stale
                        .iter()
                        .partition(|&&n| can_unload(n, &visible, &resident, &loading));
                    let answer_ms = began.elapsed().as_secs_f64() * 1000.0;
                    let mut checks = 0;
                    for &n in &resident_stale {
                        let (ok, c) = can_unload_counted(n, &visible, &resident, &loading);
                        assert_eq!(ok, leave.contains(&n), "the counted can_unload disagrees");
                        checks += c;
                    }
                    for n in &leave {
                        resident.remove(n);
                    }
                    held = hold.into_iter().collect();
                    let to_load: Vec<NodeKey> = loading
                        .iter()
                        .copied()
                        .filter(|n| !in_flight.contains_key(n))
                        .collect();
                    // The order `follow_visibility` asks in (#451).
                    let to_load = request_order(to_load, &resident, &held, a.center);
                    asked = to_load.len();
                    let snapshot = Arc::new(world.snapshot());
                    for n in to_load {
                        let job = *jobs.entry(n).or_insert_with(|| {
                            let began = Instant::now();
                            let built = build_node_shared(
                                &snapshot, &registry, &layer_of, n, blocks, &cache,
                            );
                            let ms = began.elapsed().as_secs_f64() * 1000.0;
                            let bytes = built.geometry.as_ref().map_or(0, |g| {
                                (g.mesh.vertices.len()
                                    * std::mem::size_of::<cubara_voxel::Vertex>()
                                    + g.mesh.indices.len() * 4)
                                    as u64
                            });
                            Job {
                                ms,
                                links: built.links,
                                bytes,
                            }
                        });
                        let w = (0..MODEL_WORKERS)
                            .min_by(|&x, &y| workers[x].total_cmp(&workers[y]))
                            .unwrap_or(0);
                        workers[w] = workers[w].max(now) + job.ms;
                        in_flight.insert(n, workers[w]);
                    }
                    if count {
                        report.answers += 1;
                        report.stale.offer(stale.len() as f64, t);
                        report.loading.offer(loading.len() as f64, t);
                        report.answer_checks.offer(checks as f64, t);
                        report.answer_ms.offer(answer_ms, t);
                        report.answer_list.push((
                            t,
                            links_before,
                            stale.len(),
                            loading.len(),
                            retain_ms,
                            answer_ms,
                        ));
                        report.checks += checks;
                        report.check_ms += answer_ms;
                    }
                }
                // The meshes done by now, in the order `sort_batch` takes them.
                let mut done: Vec<NodeKey> = in_flight
                    .iter()
                    .filter(|&(_, &at)| at <= now)
                    .map(|(&n, _)| n)
                    .collect();
                done.sort();
                let (mut uploads, mut bytes) = (0u64, 0u64);
                for n in &done {
                    in_flight.remove(n);
                    let job = jobs[n];
                    if links.get(n) != Some(&job.links) {
                        links.insert(*n, job.links);
                        stale_links = true;
                    }
                    if !visible.contains(n) {
                        continue;
                    }
                    resident.insert(*n);
                    if job.bytes > 0 {
                        uploads += 1;
                        bytes += job.bytes;
                    }
                }
                let (mut release_checks, mut release_ms) = (0u64, 0.0f64);
                let (held_now, flying) = (held.len(), in_flight.len());
                if !held.is_empty() {
                    let began = Instant::now();
                    let loading: Vec<NodeKey> = in_flight.keys().copied().collect();
                    let ready: Vec<NodeKey> = held
                        .iter()
                        .filter(|&&n| can_unload(n, &visible, &resident, &loading))
                        .copied()
                        .collect();
                    release_ms = began.elapsed().as_secs_f64() * 1000.0;
                    for &n in &held {
                        let (ok, c) = can_unload_counted(n, &visible, &resident, &loading);
                        assert_eq!(ok, ready.contains(&n), "the counted can_unload disagrees");
                        release_checks += c;
                    }
                    for n in &ready {
                        held.remove(n);
                        resident.remove(n);
                    }
                }
                if count {
                    report.asked.offer(asked as f64, t);
                    report.meshes.offer(done.len() as f64, t);
                    report.uploads.offer(uploads as f64, t);
                    report.upload_mb.offer(bytes as f64 / (1024.0 * 1024.0), t);
                    report.held.offer(held_now as f64, t);
                    report.in_flight.offer(flying as f64, t);
                    report.release_checks.offer(release_checks as f64, t);
                    report.release_ms.offer(release_ms, t);
                    report.links.offer(links.len() as f64, t);
                    report.visible.offer(visible.len() as f64, t);
                    report.resident.offer(resident.len() as f64, t);
                    report.checks += release_checks;
                    report.check_ms += release_ms;
                    if held_now > 0 {
                        report
                            .releases
                            .push((t, held_now, flying, release_checks, release_ms));
                    }
                }
                answer.is_some() || stale_links || !in_flight.is_empty() || !held.is_empty()
            };
            // The start, loaded completely, as `--bench flight` does.
            let at = eye_at(&flight, 0.0);
            let mut f = 0u32;
            let mut ignored = Report::default();
            while frame(f, at, None, &mut ignored) {
                f += 1;
                assert!(f < 100_000, "{name}: the start never loaded");
            }
            let loaded = f;
            for k in 1..=(seconds * 60.0) as u32 {
                let eye = eye_at(&flight, k as f32 / 60.0);
                frame(loaded + k, eye, Some(k as f32 / 60.0), &mut report);
            }
            let r = &report;
            // Seconds of the flight replayed: `t` counts from the start.
            let mb = r.upload_mb;
            println!(
                "NODES {name} ({seconds} s, {} answers): per frame, worst: nodes asked {}, stale \
                 {}, loading {}; can_unload in an answer {} beside checks, its answer {:.2} ms \
                 (t {:.2} s); meshes taken {}, uploads {} nodes {:.1} MB (t {:.2} s); held {}, in \
                 flight {}, release_held {} beside checks, {:.2} ms (t {:.2} s); links {}, \
                 visible {}, resident {}; {:.1} ns a beside check over the flight",
                r.answers,
                r.asked,
                r.stale,
                r.loading,
                r.answer_checks,
                r.answer_ms.value,
                r.answer_ms.t,
                r.meshes,
                r.uploads,
                mb.value,
                mb.t,
                r.held,
                r.in_flight,
                r.release_checks,
                r.release_ms.value,
                r.release_ms.t,
                r.links,
                r.visible,
                r.resident,
                r.check_ms * 1e6 / r.checks.max(1) as f64,
            );
            let mut answers = r.answer_list.clone();
            answers.sort_by(|a, b| b.5.total_cmp(&a.5));
            println!(
                "  {name} answers, the longest: {}",
                answers
                    .iter()
                    .take(3)
                    .map(|(t, l, s, n, retain, ms)| format!(
                        "t {t:.2} s {ms:.2} ms ({l} links, retain {retain:.2} ms; {s} stale, {n} \
                         loading)"
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            let mut releases = r.releases.clone();
            releases.sort_by(|a, b| b.4.total_cmp(&a.4));
            let over = |ms: f64| releases.iter().filter(|r| r.4 > ms).count();
            println!(
                "  {name} release_held: {} frames ran it; over 1/2/4/8 ms: {}/{}/{}/{}; the \
                 longest: {}",
                releases.len(),
                over(1.0),
                over(2.0),
                over(4.0),
                over(8.0),
                releases
                    .iter()
                    .take(5)
                    .map(|(t, h, n, c, ms)| format!(
                        "t {t:.2} s {ms:.2} ms ({h} held x {n} in flight, {c} checks)"
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }
}

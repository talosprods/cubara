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
/// the bottom of a shaft you look into turns to solid rock. `2` keeps 80.
pub(crate) const VERTICAL_LOD_SQUASH: i32 = 2;

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
/// as what drew its space before, until its own links arrive
/// (`cubara_world::visibility`'s stand-in). Forgetting the parent the moment
/// it split made the new children walls again, one search later.
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
fn can_unload(node: NodeKey, visible: &HashSet<NodeKey>, resident: &HashSet<NodeKey>) -> bool {
    let replacements = replacements_of(node, visible);
    replacements.is_empty() || replacements.iter().all(|r| resident.contains(r))
}

/// A node's box, in chunks, `max` exclusive.
fn node_box(node: NodeKey) -> ([i32; 3], [i32; 3]) {
    let o = node.chunk_origin();
    let e = node.extent_chunks();
    ([o.x, o.y, o.z], [o.x + e, o.y + e, o.z + e])
}

/// The far terrain's next hole: `prev` cut down to `want` at once, and grown
/// to all of `want` only once `drawn(kept)` says the voxels draw what lies
/// outside `kept` -- or at once when nothing of `prev` is kept (a first frame,
/// a jump). See [`NodeStreaming::far_hole`].
fn next_far_hole(
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

/// Which static model (`docs/PROPOSAL_LOD_TRANSITIONS.md` §3 R2, issue #332
/// decision 3) a node at `level` would draw its trees with: the real oak at
/// levels 1-2, two boxes from level 3 up. Level 0 never matters -- it has no
/// trees to convert (`cubara_world::mesh::build_node` only populates
/// `BuiltNode::trees` at `level >= 1`) -- so this is total rather than an
/// `Option`, for the simplicity of always having an answer to plug into
/// [`MeshedNode::tree_model`] even where `trees` ends up empty.
pub(crate) fn tree_model_for(level: u32) -> cubara_render::TreeModel {
    if level <= 2 {
        cubara_render::TreeModel::Oak
    } else {
        cubara_render::TreeModel::Box
    }
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
        aabb: geometry.aabb,
        trees: built.trees.iter().map(|t| [t.x, t.base, t.z]).collect(),
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
    /// The box, in chunks, the far terrain currently leaves to these nodes
    /// ([`far_hole`](Self::far_hole)), and the box it should leave once they
    /// are drawn. Empty (`min == max`) before the first frame.
    far_hole: ([i32; 3], [i32; 3]),
    far_want: ([i32; 3], [i32; 3]),
    /// The nodes the camera could see ([`visibility::visible_nodes`]); the only
    /// ones meshed and drawn.
    visible: HashSet<NodeKey>,
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
                    let visible =
                        visibility::visible_nodes(job.center, &desired, |n| links.get(&n).copied());
                    let answer = SearchDone {
                        generation: job.generation,
                        frame: job.frame,
                        center: job.center,
                        desired,
                        visible,
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
    pub fn update(&mut self, renderer: &mut Renderer, world: &Arc<World>, eye: [f32; 3]) {
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
                self.follow_visibility(renderer, world, done);
            }
        }
        self.drain_meshes(renderer);
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
    pub fn invalidate(&mut self, world: &Arc<World>, cc: ChunkCoord) {
        let node = NodeKey::containing(cc, 0);
        self.mesh_pool.cancel(node);
        self.mesh_pool
            .request(world, &self.registry, &self.layer_of, node, self.blocks);
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
    fn follow_visibility(&mut self, renderer: &mut Renderer, world: &Arc<World>, done: SearchDone) {
        self.searched = Some(done.center);
        let desired = done.desired;
        Arc::make_mut(&mut self.links).retain(|&n, _| worth_remembering(n, &desired));
        self.visible = done.visible;

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
        let (unload_now, held): (Vec<NodeKey>, Vec<NodeKey>) = resident_stale
            .into_iter()
            .partition(|&n| can_unload(n, &self.visible, &self.resident));
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

        let mut to_load: Vec<NodeKey> = self
            .visible
            .iter()
            .filter(|n| !self.resident.contains(n) && !self.mesh_pool.is_in_flight(**n))
            .copied()
            .collect();
        node::sort_nearest_first(&mut to_load, done.center);
        for node in to_load {
            self.mesh_pool
                .request(world, &self.registry, &self.layer_of, node, self.blocks);
        }
    }

    /// The box the far terrain may leave to the voxel nodes this frame, for
    /// an eye at `eye` -- the owner's holes while flying (2026-11-09: "laag
    /// detail niveau, dan is het weg, heel kort daarna laadt het op een hoger
    /// niveau").
    ///
    /// The nodes' own box ([`node::covered_box_3d`]) moves the moment the eye
    /// crosses a chunk, by a whole coarsest node on its far side -- 128 blocks
    /// of ground about a kilometre ahead. Handing that to the voxels at once
    /// sank the far terrain there while the nodes that draw it were still
    /// being meshed: sky, then the finer ground. So the box the far terrain
    /// leaves **shrinks at once** (where the voxels stop drawing, it draws
    /// again) and **grows only when every node the search wants in the new
    /// part is drawn**, which costs a few frames of both drawing that strip
    /// and never a frame of neither.
    ///
    /// With nothing kept -- the first frame, or a jump such as `/tp` that
    /// leaves the old box behind entirely -- the new box is taken at once, as
    /// it always was: that is a loading moment, not a flight.
    pub fn far_hole(&mut self, renderer: &Renderer, eye: [f32; 3]) -> cubara_world::far::Hole {
        let center = ChunkCoord::from_world_pos(eye);
        let want = node::covered_box_3d(center, node::DEFAULT_RING_SCHEDULE);
        let hole = next_far_hole(self.far_hole, want, |kept| {
            self.drawn_within(renderer, center, want, kept)
        });
        self.far_hole = hole;
        self.far_want = want;
        crate::far_streaming::hole_in_blocks(hole)
    }

    /// Whether every node the search around `center` wants inside `want` and
    /// not wholly inside `have` is drawn: arrived from the mesh workers and
    /// through the renderer's upload queue.
    fn drawn_within(
        &self,
        renderer: &Renderer,
        center: ChunkCoord,
        want: ([i32; 3], [i32; 3]),
        have: ([i32; 3], [i32; 3]),
    ) -> bool {
        self.searched == Some(center)
            && renderer.uploads_pending() == 0
            && self
                .visible
                .iter()
                .filter(|n| {
                    let b = node_box(**n);
                    overlaps(b, want) && !contains(have, b)
                })
                .all(|n| self.resident.contains(n))
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
    fn drain_meshes(&mut self, renderer: &mut Renderer) {
        let meshed: Vec<MeshedNode> = sort_batch(self.mesh_pool.poll())
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
        // The arriving meshes may be exactly what a held node was waiting for
        // (#262). Release those in the *same* call, so the coarse node and the
        // fine ones that replace it are never both in the arena -- and never
        // neither.
        let released: Vec<NodeId> = if self.held.is_empty() {
            Vec::new()
        } else {
            let ready: Vec<NodeKey> = self
                .held
                .iter()
                .filter(|&&n| can_unload(n, &self.visible, &self.resident))
                .copied()
                .collect();
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
        };
        renderer.apply_node_updates(released, meshed);
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real_streaming() -> NodeStreaming {
        let registry = Arc::new(cubara_render::load_registry());
        NodeStreaming::new(
            registry,
            &cubara_server::assets::load_structure_registry(),
            &cubara_server::assets::load_ore_registry(),
            |_: &str| 0,
        )
    }

    /// Issue #332 decision 3: the real oak at levels 1-2, boxes from level 3
    /// up -- the line `lod_probe.rs`'s M6 table's "instanced" column
    /// describes in prose rather than asserting against directly.
    #[test]
    fn tree_model_for_splits_at_level_three() {
        assert_eq!(tree_model_for(1), cubara_render::TreeModel::Oak);
        assert_eq!(tree_model_for(2), cubara_render::TreeModel::Oak);
        assert_eq!(tree_model_for(3), cubara_render::TreeModel::Box);
        assert_eq!(tree_model_for(4), cubara_render::TreeModel::Box);
    }

    /// New World replaces every block: nothing streamed from the old world may
    /// survive into the new one, or the old hills are drawn over the new
    /// terrain -- a resident node the new world also wants is never meshed
    /// again, because it looks already there.
    #[test]
    fn forgetting_the_world_leaves_nothing_of_it() {
        let mut s = real_streaming();
        let world = Arc::new(World::new());
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
        assert!(!can_unload(parent, &visible, &resident));

        // Seven of eight is still a hole -- an `any` here instead of `all`
        // would leave one eighth of the mountain missing, which is precisely
        // the shape of the bug being fixed.
        for kid in kids.iter().take(7) {
            resident.insert(*kid);
            assert!(
                !can_unload(parent, &visible, &resident),
                "released with {kid:?} the last one missing"
            );
        }
        resident.insert(kids[7]);
        assert!(can_unload(parent, &visible, &resident));
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
            assert!(!can_unload(*kid, &visible, &resident));
        }
        let with_parent: HashSet<NodeKey> = resident.union(&visible).copied().collect();
        for kid in &kids {
            assert!(can_unload(*kid, &visible, &with_parent));
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
        assert!(can_unload(gone, &elsewhere, &resident));
        // Including when nothing at all is visible (walked into a cave).
        assert!(can_unload(gone, &HashSet::new(), &resident));
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
            if !parent_gone && can_unload(parent, &visible, &resident) {
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
}

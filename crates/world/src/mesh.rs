//! Node meshing — synchronous and backgrounded.
//!
//! Worldgen + greedy meshing is the CPU-heavy part of streaming a node in, and
//! doing it on the main thread means every streaming update stalls the frame (a
//! visible hitch). [`MeshPool`] moves that work onto a pool of worker threads:
//! the caller *requests* a node, the workers generate + mesh it, and finished
//! [`BuiltNode`]s are drained each frame and handed to a renderer (the only step
//! that needs the GPU, and therefore the only step that lives outside this
//! crate). [`mesh_region`] is the synchronous equivalent for callers that build
//! a whole scene in one shot and don't need a worker pool (headless bench,
//! screenshot, golden-image tests).
//!
//! This lives in `cubara-world`, not `cubara-render`, because it is pure CPU
//! work on chunk/node data (`docs/PHASE1_ARCHITECTURE.md` §1) — the renderer's
//! inputs are meshes, origins and a camera, nothing that knows what a `World`
//! or a `NodeKey` is. See issue #38's tracking arc, sub-issue #110.

use std::collections::HashMap;
use std::ops::RangeInclusive;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use cubara_voxel::{
    build_mesh_bounded_occluded_node_morph, Aabb, BlockRegistry, ChunkCoord, Mesh, MeshContext,
    MorphColumns, NodeChunk, ParentFaces, ParentView,
};

use crate::node::{desired_nodes, NodeKey, RingSchedule};
use crate::worldgen::TreeSite;
use crate::{TerrainBlocks, WorldSnapshot};

/// A meshed node's geometry, ready to hand to a renderer: the triangle mesh
/// (node-local, §5.2 -- placing it is a GPU-side per-node origin add), its
/// node-local bounds for frustum culling, and the origin/scale pair a vertex
/// shader needs to do that placement (§5.2/§5.3).
///
/// Nothing in here is an absolute `f32` position (issue #379): the origin is
/// the integer block it is, and the bounds are relative to it, so a renderer
/// re-expresses them against its render anchor with one exact integer
/// subtraction and no `f32` round trip through world space.
pub struct NodeGeometry {
    pub mesh: Mesh,
    /// The mesh's bounds **relative to `origin`**, in blocks.
    pub local_aabb: Aabb,
    /// The node's minimum-corner block ([`NodeKey::world_origin`]): kept as
    /// the integer it is, so the renderer's render anchor (issue #379) is
    /// subtracted from it exactly.
    pub origin: [i32; 3],
    pub scale: f32,
}

impl NodeGeometry {
    /// The bounds in world space -- `local_aabb` moved to `origin` -- for a
    /// CPU-side consumer that frames or culls in absolute coordinates
    /// (`--bench`'s orbit, the LOD probe). Exact below 2^24 blocks; a
    /// renderer should not use this but subtract its anchor from `origin`.
    pub fn world_aabb(&self) -> Aabb {
        self.local_aabb.at_block(self.origin)
    }
}

/// A finished meshing job: the node and its geometry — or `None` if it was empty
/// (still reported so the caller marks it resident and stops re-requesting).
pub struct BuiltNode {
    pub node: NodeKey,
    pub geometry: Option<NodeGeometry>,
    /// What air joins inside each of the node's sub-blocks, for visibility
    /// culling ([`crate::visibility`]). Worked out from the same blocks the mesh
    /// was, in the same job, so the two can never describe different contents.
    /// An empty node joins everything.
    pub links: crate::visibility::NodeLinks,
    /// Trees `node`'s footprint grows (issue #332, R2 decision 1), standing
    /// on their own drawn top -- empty at level 0 (real voxel trees already
    /// mesh into `geometry`) and whenever `geometry` is `None` (nothing is
    /// drawn here at all, so nothing stands on it either).
    pub trees: Vec<NodeTree>,
}

/// One tree a coarse node draws as an instance (issue #433): its site,
/// standing on the cell of the node that draws the ground under its trunk
/// ([`WorldGen::node_trees`](crate::WorldGen::node_trees): its own trunk
/// height and canopy, its base on that cell's top), and how far the
/// distance morph moves it at the end of the node's band, in blocks.
///
/// `morph` **is the move of the ground under its trunk**: the column of the
/// node's [`MorphColumns`] that holds the trunk, its target minus its top --
/// the number the mesher bakes into that column's top face. So the tree
/// rides the ground it stands on through the band, and stands where the
/// next level stands it when the ring edge swaps them; before #433 it stayed
/// put while the ground moved, and jumped at the swap. Zero where the column
/// does not morph, and at a level that does not ([`crate::morph::MORPH_LEVELS`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeTree {
    pub site: TreeSite,
    pub morph: i32,
}

/// The trees `node` draws as instances ([`NodeTree`]), their morph read off
/// `columns` -- the node's own [`MorphColumns`], the ones its mesh is built
/// with -- or worked out here where `columns` is not given.
pub fn node_trees(
    world: &WorldSnapshot,
    node: NodeKey,
    blocks: TerrainBlocks,
    columns: Option<&MorphColumns>,
) -> Vec<NodeTree> {
    let sites = world.node_trees(node, blocks);
    if sites.is_empty() {
        return Vec::new();
    }
    let own;
    let columns = match columns {
        Some(c) => Some(c),
        None => {
            own = crate::morph::node_columns(world, node, blocks);
            own.as_ref()
        }
    };
    let w = node.extent_chunks();
    let [ox, oy, oz] = node.world_origin();
    sites
        .into_iter()
        .map(|site| {
            let morph = columns.map_or(0, |c| {
                let (cx, cz) = ((site.x - ox).div_euclid(w), (site.z - oz).div_euclid(w));
                let [top, target, _, _] = c.at(cx, cz);
                // The column's rows are the node's block rows: its top row
                // is the tree's base, whatever node of the column asks.
                debug_assert_eq!(
                    top + oy,
                    site.base,
                    "the tree at ({}, {}) does not stand on its column's top",
                    site.x,
                    site.z
                );
                target - top
            });
            NodeTree { site, morph }
        })
        .collect()
}

/// Generate `node` once, and both mesh it and work out what it joins -- what a
/// meshing job produces.
pub fn build_node(
    world: &WorldSnapshot,
    registry: &BlockRegistry,
    layer_of: &dyn Fn(&str) -> u32,
    node: NodeKey,
    blocks: TerrainBlocks,
) -> BuiltNode {
    let parents = |parent| parent_faces(world, registry, layer_of, parent, blocks);
    build_node_with(world, registry, layer_of, node, blocks, &parents)
}

/// [`build_node`], with its parent's faces (the colour morph) from a
/// [`ParentCache`] its siblings share: what the [`MeshPool`]'s workers do.
pub fn build_node_shared(
    world: &Arc<WorldSnapshot>,
    registry: &BlockRegistry,
    layer_of: &dyn Fn(&str) -> u32,
    node: NodeKey,
    blocks: TerrainBlocks,
    cache: &ParentCache,
) -> BuiltNode {
    let parents = |parent| {
        cache.get(world, parent, || {
            parent_faces(world, registry, layer_of, parent, blocks)
        })
    };
    build_node_with(world, registry, layer_of, node, blocks, &parents)
}

/// The faces of a parent node, indexed, for the colour morph: made by
/// [`parent_faces`], or read from a [`ParentCache`].
type Parents<'a> = dyn Fn(NodeKey) -> Option<Arc<cubara_voxel::ParentFaces>> + 'a;

fn build_node_with(
    world: &WorldSnapshot,
    registry: &BlockRegistry,
    layer_of: &dyn Fn(&str) -> u32,
    node: NodeKey,
    blocks: TerrainBlocks,
    parents: &Parents,
) -> BuiltNode {
    let chunk = world.node_at(node, blocks);
    let links = crate::visibility::node_links(chunk.as_ref(), registry);
    // Worked out once: the mesh bakes them, and the trees ride them.
    let columns = chunk
        .as_ref()
        .and_then(|_| morph_columns(world, node, blocks, parents));
    let geometry = chunk.as_ref().and_then(|c| {
        mesh_chunk_with(world, registry, layer_of, node, blocks, c, columns.as_ref())
    });
    // Both halves of this guard are belt-and-braces, not load-bearing on
    // their own, which is why neither has a dedicated test here (found by
    // `check-tests-can-fail.sh`, recorded rather than chased): `node.level
    // >= 1` is redundant with `World::node_trees`'s own `level == 0` check,
    // and `geometry.is_some()` only skips wasted computation for a node
    // nothing will draw -- `to_meshed_node` (cubara-app) discards `trees`
    // right along with a `None` geometry anyway, so computing it here for a
    // sky node would be silently thrown away, not wrong.
    let trees = if node.level >= 1 && geometry.is_some() {
        node_trees(world, node, blocks, columns.as_ref())
    } else {
        Vec::new()
    };
    BuiltNode {
        node,
        links,
        geometry,
        trees,
    }
}

/// Put a batch of finished mesh jobs into a fixed order — ascending
/// `NodeKey` — before anything uploads them.
///
/// Worker threads finish jobs in whatever order the OS happens to schedule
/// them, which is not the same order every run, and a GPU arena's suballocator
/// is typically first-fit: whichever job is applied first claims the earliest
/// free slot. Left unsorted, the arena's slab layout — which node ends up at
/// which GPU offset — would depend on thread timing instead of world state
/// (issue #83). Sorting first makes it depend on world state alone, which is
/// `ARCHITECTURE.md` Rule 1's requirement that a parallel step's results are
/// merged in a fixed order. `NodeKey`'s `Ord` is total and fixed (level then
/// pos) for exactly this reason.
pub fn sort_batch(mut batch: Vec<BuiltNode>) -> Vec<BuiltNode> {
    batch.sort_by_key(|b| b.node);
    batch
}

/// Generate + mesh `node`'s content against `world`, resolving textures via
/// `registry`/`layer_of` — the synchronous building block both the worker pool
/// ([`MeshPool`]) and the whole-region helper ([`mesh_region`]) share.
pub fn mesh_node(
    world: &WorldSnapshot,
    registry: &BlockRegistry,
    layer_of: &dyn Fn(&str) -> u32,
    node: NodeKey,
    blocks: TerrainBlocks,
) -> Option<NodeGeometry> {
    mesh_node_with(world, registry, layer_of, node, blocks, true)
}

/// [`mesh_node`], with the distance morph baked in or not (issue #380's
/// study): `false` is the mesh as it was before the morph, which a census
/// counts the morph's cost against.
pub fn mesh_node_with(
    world: &WorldSnapshot,
    registry: &BlockRegistry,
    layer_of: &dyn Fn(&str) -> u32,
    node: NodeKey,
    blocks: TerrainBlocks,
    morph: bool,
) -> Option<NodeGeometry> {
    let chunk = world.node_at(node, blocks)?;
    let parents = |parent| parent_faces(world, registry, layer_of, parent, blocks);
    let columns = morph
        .then(|| morph_columns(world, node, blocks, &parents))
        .flatten();
    mesh_chunk_with(
        world,
        registry,
        layer_of,
        node,
        blocks,
        &chunk,
        columns.as_ref(),
    )
}

/// The distance morph's columns for `node` ([`crate::morph::node_columns`]),
/// with its parent's faces for the colour morph; `None` at a level that
/// does not morph.
fn morph_columns(
    world: &WorldSnapshot,
    node: NodeKey,
    blocks: TerrainBlocks,
    parents: &Parents,
) -> Option<MorphColumns> {
    crate::morph::node_columns(world, node, blocks).map(|mut c| {
        let parent = node.parent();
        let (p, o) = (parent.world_origin(), node.world_origin());
        c.parent = parents(parent)
            .map(|faces| ParentView::new(faces, [o[0] - p[0], o[1] - p[1], o[2] - p[2]]));
        c
    })
}

/// Mesh `node` from its already generated blocks.
///
/// `scale` is always 1.0 now (R1b, issue #327 decision 3): a level-0
/// `Chunk`'s positions were already node-local blocks (its own cell is 1
/// block), and a coarse node's mesh now bakes its own per-axis `[2^level, v,
/// 2^level]` scale straight into its vertex positions
/// ([`cubara_voxel::AnisoChunk::build_mesh_with_neighbours`]) instead of
/// leaving a single scalar for the GPU's `node_origins.w` to multiply by --
/// which a single scalar couldn't represent anyway, once the vertical cell
/// stopped matching the horizontal one.
///
/// The distance morph is baked in where `columns` gives it ([`morph_columns`]:
/// the node's level morphs, and the parent's faces are the colour morph's
/// source, `docs/PHASE3_ARCHITECTURE.md` §9.6); `None` meshes it as built.
#[allow(clippy::too_many_arguments)]
fn mesh_chunk_with(
    world: &WorldSnapshot,
    registry: &BlockRegistry,
    layer_of: &dyn Fn(&str) -> u32,
    node: NodeKey,
    blocks: TerrainBlocks,
    chunk: &NodeChunk,
    columns: Option<&MorphColumns>,
) -> Option<NodeGeometry> {
    let ctx = MeshContext { registry, layer_of };
    let origin = node.world_origin();
    let scale = 1.0f32;
    let ground = std::cell::RefCell::new(BorderGround::default());
    let covered =
        |gx, gy, gz| border_samples(world, node, blocks, &ground, [gx, gy, gz], Want::All);
    let may_be_solid =
        |gx, gy, gz| border_samples(world, node, blocks, &ground, [gx, gy, gz], Want::Any);
    let (mesh, local_aabb) =
        build_mesh_bounded_occluded_node_morph(chunk, &ctx, covered, may_be_solid, columns)?;
    Some(NodeGeometry {
        mesh,
        local_aabb,
        origin,
        scale,
    })
}

/// Every chunk whose mesh can change when the block at `pos` changes: its own,
/// and -- since a chunk leaves out border faces its neighbour covers -- the
/// neighbour across each border `pos` sits on.
///
/// Without the neighbours, digging out a block at the edge of a chunk leaves
/// the next chunk still hiding the face that now looks into the hole.
/// The faces of `parent`, indexed for the colour morph of its children
/// (`cubara_voxel::ParentFaces`, `docs/PHASE3_ARCHITECTURE.md` §9.6): the
/// parent node generated and meshed by the game's own generator and mesher.
///
/// **Meshed without a morph of its own.** The game draws the parent with its
/// own morph keyed in, which cuts its quads at its cells and lays latent
/// risers; neither changes what it shows at rest -- a quad is only ever cut
/// where every cell of it is alike, and a latent riser has no area -- and
/// the parent is at rest wherever its child is swapped for it (its own band
/// starts past its child's ring). Keyed in, it would ask for its own parent
/// in turn.
///
/// `None` where the parent draws nothing (a node of sky); its children then
/// keep their own colours.
fn parent_faces(
    world: &WorldSnapshot,
    registry: &BlockRegistry,
    layer_of: &dyn Fn(&str) -> u32,
    parent: NodeKey,
    blocks: TerrainBlocks,
) -> Option<Arc<ParentFaces>> {
    let chunk = world.node_at(parent, blocks)?;
    let geometry = mesh_chunk_with(world, registry, layer_of, parent, blocks, &chunk, None)?;
    let w = parent.extent_chunks();
    Some(Arc::new(ParentFaces::from_mesh(&geometry.mesh, [w, 1, w])))
}

/// How many parent nodes a [`ParentCache`] keeps. Chosen, not derived: a
/// node splits into eight children requested together, and a streaming
/// update splits a few parents at once; past this the oldest is made again
/// if asked for.
const PARENT_CACHE: usize = 32;

/// Parent nodes' faces already made ([`parent_faces`]), shared by the
/// workers meshing their children: a node splits into eight, and each child
/// of it needs the same parent generated and meshed. **An accelerator only**
/// (`docs/PHASE3_ARCHITECTURE.md` §3): what it holds is a pure function of
/// the snapshot and the node, so a job finds what it would have made. Keyed
/// by the snapshot itself (held, so no other can take its address) and the
/// node; a worker asking for a parent another is making waits for it rather
/// than making it twice.
#[derive(Default)]
pub struct ParentCache {
    entries: Mutex<std::collections::VecDeque<ParentEntry>>,
}

type ParentEntry = (
    Arc<WorldSnapshot>,
    NodeKey,
    Arc<std::sync::OnceLock<Option<Arc<ParentFaces>>>>,
);

impl ParentCache {
    /// `parent`'s faces in `world`, made by `make` unless already made.
    pub fn get(
        &self,
        world: &Arc<WorldSnapshot>,
        parent: NodeKey,
        make: impl FnOnce() -> Option<Arc<ParentFaces>>,
    ) -> Option<Arc<ParentFaces>> {
        let slot = {
            let mut entries = self.entries.lock().expect("parent cache lock");
            match entries
                .iter()
                .find(|(w, n, _)| Arc::ptr_eq(w, world) && *n == parent)
            {
                Some((_, _, slot)) => Arc::clone(slot),
                None => {
                    let slot = Arc::new(std::sync::OnceLock::new());
                    entries.push_back((Arc::clone(world), parent, Arc::clone(&slot)));
                    if entries.len() > PARENT_CACHE {
                        entries.pop_front();
                    }
                    slot
                }
            }
        };
        slot.get_or_init(make).clone()
    }
}

pub fn chunks_affected_by_edit(pos: [i32; 3]) -> Vec<ChunkCoord> {
    let size = cubara_voxel::Chunk::SIZE as i32;
    let own = ChunkCoord::new(
        pos[0].div_euclid(size),
        pos[1].div_euclid(size),
        pos[2].div_euclid(size),
    );
    let mut out = vec![own];
    for axis in 0..3 {
        let local = pos[axis].rem_euclid(size);
        let step = match local {
            0 => -1,
            l if l == size - 1 => 1,
            _ => continue,
        };
        let mut c = [own.x, own.y, own.z];
        c[axis] += step;
        out.push(ChunkCoord::new(c[0], c[1], c[2]));
    }
    out
}

/// The ground [`border_samples`] has asked about while one node is meshed:
/// level 0's columns by `(x, z)`, and the cells of coarser neighbours by
/// corner and width.
#[derive(Default)]
struct BorderGround {
    columns: HashMap<(i32, i32), i32>,
    cells: HashMap<(i32, i32, i32), crate::worldgen::CellGround>,
}

/// Which answer [`border_samples`] gives over the cells it looks at.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Want {
    /// Solid for every neighbour: the face may be left out.
    All,
    /// Solid for some neighbour: a skirt may be needed.
    Any,
}

/// What `node`'s neighbours draw at its outside cell `g` -- coordinates in
/// the lattice the node is drawn on (`NodeChunk::mesh_dims`: block rows at
/// every level), with exactly one axis at `-1` or that axis's cell count:
/// whether every neighbour draws it solid ([`Want::All`], so a border face
/// against it can be left out) or some neighbour might ([`Want::Any`], so a
/// skirt beside it may be needed).
///
/// **The neighbour may be drawn at a different level of detail**, and which one
/// depends on where the camera is, which a node's mesh cannot know: it is built
/// once and kept while the camera moves. So a face is dropped only when the
/// outside is solid for *every* neighbour this node could have. Rings of
/// detail are nested one level apart, so that is three:
///
/// - **the same level** -- one cell, sampled where that node samples it;
/// - **one coarser** -- the double-size cell that contains it;
/// - **one finer** -- the four half-size cells that touch the face.
///
/// Each is asked what it draws **at this block row**
/// (`World::certainly_solid_at_v`): a cell is solid at a block at or under
/// its column's exact top, in a row that is not cave. Level 3's rows are
/// four blocks tall and cut off at that top (issue #423), so a row is not
/// the question.
///
/// If any of those is air, some neighbour draws an opening there, and this
/// face is what a player looking through it would see -- so it stays. Drop it
/// and the result is a hole into the inside of the terrain.
///
/// **At the edge of the render distance the walls go too**: the outside is not
/// drawn, but it is solid ground. What used to be a grey cliff where the drawn
/// world stopped is now the terrain simply ending -- which is what it is.
fn border_samples(
    world: &WorldSnapshot,
    node: NodeKey,
    blocks: TerrainBlocks,
    ground: &std::cell::RefCell<BorderGround>,
    g: [i32; 3],
    want: Want,
) -> bool {
    let level = node.level;
    // The lattice this node is drawn on (`NodeChunk::mesh_dims`): cells
    // `2^level` blocks wide and deep, in rows **one block tall at every
    // level**. Level 3 stores rows of four and is drawn cut off at its exact
    // tops, on block rows (issue #423), so its border is asked about block
    // by block too. `g` is a coordinate in that lattice (exactly one axis at
    // `-1` or that axis's own cell count).
    let w = node.extent_chunks();
    let step = [w, node.drawn_row_blocks(), w];
    let dims = [16, 16 * w / step[1], 16];
    let origin = node.world_origin();
    let p = [
        origin[0] + g[0] * step[0],
        origin[1] + g[1] * step[1],
        origin[2] + g[2] * step[2],
    ];
    // Whether the cell of level `at` that holds block `q` is drawn solid
    // there: its top and its own rows (`World::certainly_solid_at_v`).
    // Surface heights are the expensive part of a sample, and one node's
    // border touches few columns, so each cell's are computed once per node.
    let solid = |q: [i32; 3], at: u32| {
        if at == 0 {
            if want == Want::Any {
                // Trees count here: leaving one out could drop a skirt a
                // trunk at the border needs, where for covering it can only
                // keep a face.
                return world.is_solid_at(q[0], q[1], q[2], blocks);
            }
            let surface = *ground
                .borrow_mut()
                .columns
                .entry((q[0], q[2]))
                .or_insert_with(|| world.surface_height(q[0], q[2]));
            return world.certainly_solid_at(q[0], q[1], q[2], surface);
        }
        let step_h = 1 << at;
        let (x, z) = (
            q[0].div_euclid(step_h) * step_h,
            q[2].div_euclid(step_h) * step_h,
        );
        let mut ground = ground.borrow_mut();
        let cell = ground
            .cells
            .entry((x, z, step_h))
            .or_insert_with(|| world.cell_ground(x, z, step_h));
        world.certainly_solid_at_v(q[1], at, cell)
    };
    // `All` stops at the first air, `Any` at the first solid.
    let decided = |is_solid: bool| match want {
        Want::All => !is_solid,
        Want::Any => is_solid,
    };
    let answer_if_decided = want == Want::Any;

    if decided(solid(p, level)) {
        return answer_if_decided;
    }
    // The coarser neighbour's cell that holds this block.
    if decided(solid(p, level + 1)) {
        return answer_if_decided;
    }
    if level > 0 {
        // The finer layer that touches this face: the near half of the
        // outside cell along the face's axis, and both halves across it --
        // horizontally, that is; a row is one block at every level.
        let half = [w / 2, step[1], w / 2];
        let axis = (0..3)
            .find(|&k| g[k] == -1 || g[k] == dims[k])
            .expect("a border cell");
        let (a, b) = ((axis + 1) % 3, (axis + 2) % 3);
        let mut f = p;
        if g[axis] == -1 {
            f[axis] += step[axis] - half[axis];
        }
        let steps = |k: usize| (0..step[k]).step_by(half[k] as usize);
        for da in steps(a) {
            for db in steps(b) {
                let mut r = f;
                r[a] += da;
                r[b] += db;
                if decided(solid(r, level - 1)) {
                    return answer_if_decided;
                }
            }
        }
    }
    !answer_if_decided
}

/// Mesh every node [`desired_nodes`] wants for `schedule` around `center`,
/// synchronously, in ascending [`NodeKey`] order (matching [`sort_batch`]'s
/// ordering guarantee) — for callers that build a whole scene in one shot and
/// don't need a worker pool: the headless bench, screenshot, and golden-image
/// test paths.
pub fn mesh_region(
    world: &WorldSnapshot,
    registry: &BlockRegistry,
    layer_of: &dyn Fn(&str) -> u32,
    center: ChunkCoord,
    y_range: RangeInclusive<i32>,
    schedule: &RingSchedule,
    blocks: TerrainBlocks,
) -> Vec<BuiltNode> {
    mesh_nodes(
        world,
        registry,
        layer_of,
        desired_nodes(center, y_range, schedule),
        blocks,
    )
}

/// Mesh exactly `nodes`, synchronously, in ascending [`NodeKey`] order -- what
/// [`mesh_region`] does for its band, for a caller that chose the nodes some
/// other way (a 3D selection).
pub fn mesh_nodes(
    world: &WorldSnapshot,
    registry: &BlockRegistry,
    layer_of: &dyn Fn(&str) -> u32,
    mut nodes: Vec<NodeKey>,
    blocks: TerrainBlocks,
) -> Vec<BuiltNode> {
    nodes.sort();
    nodes
        .into_iter()
        .map(|node| build_node(world, registry, layer_of, node, blocks))
        .collect()
}

/// One meshing job: what to mesh, the world snapshot to mesh it from, and the
/// registry + texture-layer resolver to resolve solidity and texturing
/// against.
///
/// The snapshot travels *with* the job rather than the workers reaching for
/// shared state (`ARCHITECTURE.md` Rule 2), so a job always meshes a
/// consistent view and readers never block a writer. It is a
/// [`WorldSnapshot`] -- the terrain and the edit overlay, by reference -- and
/// **not** an `Arc<World>` (issue #387 decision 2): holding the whole world
/// made the tick's `Arc::make_mut` copy every edit ever made while a job was
/// in flight. `layer_of` is a callback rather than a concrete type (mirroring
/// `MeshContext`'s own convention) so this crate never has to name whatever
/// GPU-side type actually owns the texture array — that stays the caller's
/// business.
type Job = (
    Arc<WorldSnapshot>,
    Arc<BlockRegistry>,
    Arc<dyn Fn(&str) -> u32 + Send + Sync>,
    NodeKey,
    // Resolved once by the caller and carried with the job, rather than each
    // worker re-deriving it from the registry per node -- which is what this
    // used to do, and which also had no way to know about structures.
    TerrainBlocks,
    // Which request this is -- see `MeshPool::in_flight`.
    u64,
);

/// A pool of worker threads that mesh nodes off the main thread.
///
/// Tracks which nodes are currently in flight, so a node is never requested
/// twice while its job is outstanding, and a result for a node no longer
/// wanted (unloaded while it was being meshed) is dropped by
/// [`poll`](Self::poll) instead of surfaced. A `NodeKey` already names both a
/// node's position *and* its detail level, so a change in desired detail is a
/// different key entirely, not a re-request of the same one.
pub struct MeshPool {
    job_tx: Sender<Job>,
    result_rx: Receiver<(u64, BuiltNode)>,
    /// Each wanted node, and the ticket of the one request whose result is
    /// wanted for it. **The ticket, not the node, is what a result is checked
    /// against:** a cancelled job still runs to the end and sends a result
    /// that names the same node, so after cancel-and-request-again the node
    /// alone cannot tell the stale mesh from its replacement.
    in_flight: HashMap<NodeKey, u64>,
    next_ticket: u64,
    _workers: Vec<JoinHandle<()>>,
}

impl MeshPool {
    /// Spawn a pool sized to leave the main thread a core to itself.
    pub fn new() -> Self {
        let workers = std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1))
            .unwrap_or(1)
            .max(1);
        Self::with_workers(workers)
    }

    fn with_workers(workers: usize) -> Self {
        let (job_tx, job_rx) = std::sync::mpsc::channel::<Job>();
        let (result_tx, result_rx) = std::sync::mpsc::channel::<(u64, BuiltNode)>();
        // One receiver shared by all workers: each grabs the next job under the lock,
        // then releases it and meshes in parallel with the others.
        let job_rx = Arc::new(Mutex::new(job_rx));

        let cache = Arc::new(ParentCache::default());
        let _workers = (0..workers)
            .map(|_| {
                let jobs = Arc::clone(&job_rx);
                let cache = Arc::clone(&cache);
                let results = result_tx.clone();
                std::thread::Builder::new()
                    .name("cubara-mesher".into())
                    .spawn(move || loop {
                        let (world, registry, layer_of, node, blocks, ticket) = {
                            let rx = jobs.lock().expect("mesher job lock");
                            match rx.recv() {
                                Ok(job) => job,
                                // All senders dropped (pool dropped) — exit.
                                Err(_) => break,
                            }
                        };
                        let built =
                            build_node_shared(&world, &registry, &*layer_of, node, blocks, &cache);
                        if results.send((ticket, built)).is_err() {
                            break; // caller gone
                        }
                    })
                    .expect("spawn mesher thread")
            })
            .collect();

        Self {
            job_tx,
            result_rx,
            in_flight: HashMap::new(),
            next_ticket: 0,
            _workers,
        }
    }

    /// Queue `node` for meshing against the `world` snapshot, `registry` and
    /// `layer_of`, unless it's already in flight.
    ///
    /// The caller passes the world it wants meshed, so a job can never observe
    /// an edit that lands after it was queued. **One snapshot per batch**
    /// (issue #387 decision 2): a caller requesting several nodes takes
    /// `Arc::new(world.snapshot())` once and passes it to each -- the jobs
    /// share it, and nothing in here copies it.
    pub fn request(
        &mut self,
        world: &Arc<WorldSnapshot>,
        registry: &Arc<BlockRegistry>,
        layer_of: &Arc<dyn Fn(&str) -> u32 + Send + Sync>,
        node: NodeKey,
        blocks: TerrainBlocks,
    ) {
        if let std::collections::hash_map::Entry::Vacant(slot) = self.in_flight.entry(node) {
            let ticket = self.next_ticket;
            self.next_ticket += 1;
            slot.insert(ticket);
            // Send can only fail if all workers died; nothing useful to do if so.
            let _ = self.job_tx.send((
                Arc::clone(world),
                Arc::clone(registry),
                Arc::clone(layer_of),
                node,
                blocks,
                ticket,
            ));
        }
    }

    /// Forget an in-flight node: the worker still finishes it, but its result
    /// will be dropped by [`poll`](Self::poll) instead of surfaced.
    pub fn cancel(&mut self, node: NodeKey) {
        self.in_flight.remove(&node);
    }

    /// Whether `node` is currently being meshed.
    pub fn is_in_flight(&self, node: NodeKey) -> bool {
        self.in_flight.contains_key(&node)
    }

    /// The nodes currently being meshed (so the caller can unload ones that
    /// fell out of range before their mesh was ready).
    pub fn in_flight(&self) -> impl ExactSizeIterator<Item = NodeKey> + '_ {
        self.in_flight.keys().copied()
    }

    /// Take all finished results that are still wanted, clearing them from the
    /// in-flight set. Non-blocking.
    pub fn poll(&mut self) -> Vec<BuiltNode> {
        let mut done = Vec::new();
        while let Ok((ticket, built)) = self.result_rx.try_recv() {
            if self.in_flight.get(&built.node) == Some(&ticket) {
                self.in_flight.remove(&built.node);
                done.push(built);
            }
        }
        done
    }
}

impl Default for MeshPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::World;
    use cubara_voxel::{DropRule, Faces, Interact, Material, Shape};
    use std::collections::HashMap;

    /// A registry with the three real material *names* -- `mesh_node` resolves
    /// `TerrainBlocks::from_registry` by name (block 1.4c), so a fixture missing
    /// any of them panics inside the worker thread and the test hangs waiting
    /// for a result that never arrives, rather than failing loudly. These tests
    /// only care about solid-vs-air and worker-pool mechanics (registry
    /// mechanics are tested in `cubara_voxel::registry`), so all three are plain
    /// `All` materials, and there's no real texture layer mapping since these
    /// tests don't care about texturing either.
    fn test_registry() -> Arc<BlockRegistry> {
        let material = |name: &str| {
            (
                std::path::PathBuf::from("test-fixture.ron"),
                Material {
                    name: name.to_string(),
                    solid: true,
                    faces: Faces::All(name.to_string()),
                    shapes: vec![Shape::Full],
                    drops: DropRule::SameName,
                    requires_tier: 0,
                    hardness: Some(1),
                    interact: Interact::None,
                },
            )
        };
        Arc::new(
            BlockRegistry::from_materials(vec![
                material("cubara:grass"),
                material("cubara:soil"),
                material("cubara:stone"),
            ])
            .expect("fixture registry is valid"),
        )
    }

    /// Whether `node`'s cell containing world point `p` is solid, as the node
    /// itself draws it. `chunks` caches generated nodes across calls.
    fn drawn_solid(
        world: &WorldSnapshot,
        blocks: TerrainBlocks,
        chunks: &mut HashMap<NodeKey, Option<NodeChunk>>,
        node: NodeKey,
        p: [i32; 3],
    ) -> bool {
        let chunk = chunks
            .entry(node)
            .or_insert_with(|| world.node_at(node, blocks));
        let o = node.world_origin();
        chunk.as_ref().is_some_and(|c| {
            // The lattice the node is drawn on: level 3's rows of four are
            // cut at exact tops and drawn on block rows.
            let scale = c.mesh_scale();
            let i = |k: usize| ((p[k] - o[k]) / scale[k]) as usize;
            c.drawn(i(0), i(1), i(2)) != cubara_voxel::BlockId::AIR
        })
    }

    /// Whether `node`'s mesh has a face on the plane `axis = at`, pointing
    /// `sign` along it, covering the unit square with min corner `p`.
    fn has_face(mesh: &Mesh, node: NodeKey, axis: usize, at: i32, sign: i8, p: [i32; 3]) -> bool {
        // R1b, decision 3: every node's mesh positions are already final
        // node-local *block* coordinates (a level-0 `Chunk`'s always were,
        // since its own cell is 1 block; a coarse node's now bakes its own
        // per-axis scale in) -- so placing a vertex in world space is just
        // adding the origin, never multiplying by `extent_chunks` again.
        let o = node.world_origin();
        let face = cubara_voxel::Face::from_axis_sign(axis, sign);
        let world = |v: cubara_voxel::Vertex| {
            [
                o[0] + v.x() as i32,
                o[1] + v.y() as i32,
                o[2] + v.z() as i32,
            ]
        };
        let (a, b) = ((axis + 1) % 3, (axis + 2) % 3);
        mesh.vertices.chunks(4).any(|q| {
            if q[0].face() != face || world(q[0])[axis] != at {
                return false;
            }
            let corners: Vec<[i32; 3]> = q.iter().map(|&v| world(v)).collect();
            let lo = |k: usize| corners.iter().map(|c| c[k]).min().unwrap();
            let hi = |k: usize| corners.iter().map(|c| c[k]).max().unwrap();
            (lo(a)..hi(a)).contains(&p[a]) && (lo(b)..hi(b)).contains(&p[b])
        })
    }

    /// **No holes where two nodes meet** -- the condition border covering must
    /// never break, at every level pairing the ring schedule produces.
    ///
    /// Walk the shared plane one block at a time. Wherever one side draws
    /// solid and the other air, a player in that air can look at the solid
    /// side, so the solid side must have a face there. If neither has one, the
    /// player sees through the terrain.
    #[test]
    fn where_two_nodes_meet_every_visible_solid_cell_has_a_face() {
        let registry = test_registry();
        let layer_of = |_: &str| 0;
        let blocks = TerrainBlocks::from_registry(&registry);
        let world = World::new();
        let mut meshes: HashMap<NodeKey, Option<Mesh>> = HashMap::new();
        let mut chunks = HashMap::new();
        let mut face_at = |n: NodeKey, axis: usize, at: i32, sign: i8, p: [i32; 3]| {
            meshes
                .entry(n)
                .or_insert_with(|| {
                    mesh_node(&world, &registry, &layer_of, n, blocks).map(|g| g.mesh)
                })
                .as_ref()
                .is_some_and(|m| has_face(m, n, axis, at, sign, p))
        };

        let mut checked = 0usize;
        let mut mismatched = 0usize;
        // Pairs (level of A, level of B), B beyond A along +axis. Level 0 has
        // caves and the coarser ones do not, so 0-1 is where the two sides
        // disagree the most.
        for (la, lb) in [
            (0u32, 0u32),
            (0, 1),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 2),
            (3, 3),
        ] {
            let big = 1i32 << la.max(lb);
            for axis in [0usize, 2] {
                for col in 0..6i32 {
                    for layer in [-1i32, 0, 1] {
                        // A plane both levels' node grids share.
                        let mut plane_chunk = [0i32; 3];
                        plane_chunk[axis] = (col * 7 + 3) * big;
                        let other = if axis == 0 { 2 } else { 0 };
                        plane_chunk[other] = (col * 5 - 9) * big;
                        plane_chunk[1] = layer * big;
                        let a_node = NodeKey::containing(
                            ChunkCoord::new(
                                plane_chunk[0] - (axis == 0) as i32,
                                plane_chunk[1],
                                plane_chunk[2] - (axis == 2) as i32,
                            ),
                            la,
                        );
                        let plane = plane_chunk[axis] * cubara_voxel::Chunk::SIZE as i32;
                        let (ao, asz) = (a_node.world_origin(), 16 * a_node.extent_chunks());
                        let (a, b) = ((axis + 1) % 3, (axis + 2) % 3);
                        for da in 0..asz {
                            for db in 0..asz {
                                let mut p = [0i32; 3];
                                p[a] = ao[a] + da;
                                p[b] = ao[b] + db;
                                // A's cell is just below the plane, B's at it.
                                p[axis] = plane - 1;
                                let a_solid = drawn_solid(&world, blocks, &mut chunks, a_node, p);
                                let mut pb = p;
                                pb[axis] = plane;
                                let b_node = NodeKey::containing(
                                    ChunkCoord::from_world_pos([
                                        pb[0] as f32,
                                        pb[1] as f32,
                                        pb[2] as f32,
                                    ]),
                                    lb,
                                );
                                let b_solid = drawn_solid(&world, blocks, &mut chunks, b_node, pb);
                                checked += 1;
                                if a_solid == b_solid {
                                    continue;
                                }
                                mismatched += 1;
                                let ok = if a_solid {
                                    face_at(a_node, axis, plane, 1, p)
                                } else {
                                    face_at(b_node, axis, plane, -1, pb)
                                };
                                assert!(
                                    ok,
                                    "a hole: levels {la}|{lb}, plane {axis}={plane}, block {p:?}, solid on the {} side has no face",
                                    if a_solid { "near" } else { "far" }
                                );
                            }
                        }
                    }
                }
            }
        }
        assert!(
            mismatched > 100,
            "only {mismatched} of {checked} cells differ: the test sees nothing"
        );
    }

    /// **Flat ground is drawn at its surface at every level** (issue #423;
    /// found by the lead's audit of the 512-block ring, 2026-11-20). A field
    /// made by hand, flat at height `t`, as each coarse level generates and
    /// meshes it: the top faces stand at `t + 1`, where level 0's do -- for
    /// every `t` mod 4.
    ///
    /// Level 3 used to round its top to rows of four blocks: -1, +2, +1, 0
    /// blocks off for `t` mod 4 = 0, 1, 2, 3. On a plain at 1 mod 4 it stood
    /// two blocks above level 2, a wall ringing the player 512 blocks out.
    #[test]
    fn a_flat_field_is_drawn_at_its_surface_at_every_level() {
        use crate::worldgen::NodeGround;
        let registry = test_registry();
        let layer_of = |_: &str| 0;
        let ctx = MeshContext {
            registry: &registry,
            layer_of: &layer_of,
        };
        let blocks = TerrainBlocks::from_registry(&registry);
        let gen = crate::WorldGen::new(World::new().seed());
        let mut failures = Vec::new();
        for t in 40..48 {
            for level in 1..=3u32 {
                let node = NodeKey::new(level, [0, 0, 0]);
                let (w, v) = (node.extent_chunks(), node.cell_height_blocks());
                let flat = NodeGround::over(w, |_, _| t);
                // The node that holds the surface.
                let oy = t.div_euclid(16 * w) * 16 * w;
                let chunk = gen.generate_aniso_over([0, oy, 0], w, v, blocks, &flat);
                let mesh = chunk.build_mesh(&ctx);
                // Each cell's highest top face, in blocks from the node's foot.
                let mut tops = std::collections::BTreeMap::<(u32, u32), u32>::new();
                for quad in mesh.vertices.chunks(4) {
                    if quad[0].face() != cubara_voxel::Face::PosY {
                        continue;
                    }
                    let lo =
                        |f: fn(&cubara_voxel::Vertex) -> u32| quad.iter().map(f).min().unwrap();
                    let hi =
                        |f: fn(&cubara_voxel::Vertex) -> u32| quad.iter().map(f).max().unwrap();
                    for z in (lo(|q| q.z())..hi(|q| q.z())).step_by(w as usize) {
                        for x in (lo(|q| q.x())..hi(|q| q.x())).step_by(w as usize) {
                            let top = tops.entry((x, z)).or_default();
                            *top = (*top).max(quad[0].y());
                        }
                    }
                }
                // The world's caves still carve this field (it is not where
                // the world's own ground is, so they are many) and open a
                // pit in some cells. Nothing stands above the surface, and
                // a good part of the field is at it: a quarter of the cells
                // at least (chosen, not derived; 105 to 185 of 256 are).
                let highest = oy + tops.values().copied().max().unwrap_or(0) as i32;
                let at_it = tops.values().filter(|&&y| oy + y as i32 == t + 1).count();
                if highest != t + 1 || at_it < 64 {
                    failures.push(format!(
                        "surface {t} (mod 4 = {}), level {level}: the highest top face is at {highest}, level 0's is at {}; {at_it} of 256 cells are there",
                        t % 4,
                        t + 1
                    ));
                }
            }
        }
        assert!(failures.is_empty(), "\n  {}", failures.join("\n  "));
    }

    /// `NodeKey::drawn_row_blocks` is what a generated node says of itself:
    /// the lattice its mesh is built on, and `border_samples` is asked
    /// about, has rows that tall and fills the node -- at level 3 too, whose
    /// stored rows are four blocks.
    #[test]
    fn a_node_is_drawn_in_block_rows() {
        let registry = test_registry();
        let blocks = TerrainBlocks::from_registry(&registry);
        let world = World::new();
        for level in 0..=3u32 {
            let node = (-2..3)
                .map(|ny| NodeKey::new(level, [0, ny, 0]))
                .find_map(|node| world.node_at(node, blocks).map(|chunk| (node, chunk)));
            let (node, chunk) = node.expect("a node with ground in it");
            let rows = node.drawn_row_blocks();
            assert_eq!(chunk.mesh_scale()[1], rows, "level {level}");
            assert_eq!(
                chunk.mesh_dims()[1] as i32 * rows,
                16 * node.extent_chunks(),
                "level {level}: the rows fill the node"
            );
            assert_eq!(
                chunk.dims()[1] as i32 * node.cell_height_blocks(),
                16 * node.extent_chunks(),
                "level {level}: the stored rows fill it too"
            );
        }
    }

    #[test]
    fn an_edit_dirties_its_chunk_and_the_neighbours_it_borders() {
        let c = ChunkCoord::new;
        assert_eq!(
            chunks_affected_by_edit([5, 5, 5]),
            vec![c(0, 0, 0)],
            "interior"
        );
        assert_eq!(
            chunks_affected_by_edit([0, 5, 15]),
            vec![c(0, 0, 0), c(-1, 0, 0), c(0, 0, 1)],
            "on two borders"
        );
        // Negative coordinates: block -1 is local 15 of chunk -1.
        assert_eq!(
            chunks_affected_by_edit([-1, -16, 7]),
            vec![c(-1, -1, 0), c(0, -1, 0), c(-1, -2, 0)]
        );
    }

    /// A block dug out right at a node's border opens a hole the neighbour
    /// must show its side of -- the edit is the only thing that says so.
    #[test]
    fn digging_at_a_border_shows_the_neighbours_side() {
        let registry = test_registry();
        let layer_of = |_: &str| 0;
        let blocks = TerrainBlocks::from_registry(&registry);
        let mut world = World::new();
        // Deep enough to be solid rock on both sides of the x = 0 plane.
        let (near, far) = ([-1, -40, 5], [0, -40, 5]);
        assert!(
            world.is_solid_at(near[0], near[1], near[2], blocks),
            "not rock"
        );
        assert!(
            world.is_solid_at(far[0], far[1], far[2], blocks),
            "not rock"
        );
        let far_node = NodeKey::containing(ChunkCoord::from_world_pos([0.0, -40.0, 5.0]), 0);
        let before = mesh_node(&world, &registry, &layer_of, far_node, blocks);
        assert!(
            !before
                .as_ref()
                .is_some_and(|g| has_face(&g.mesh, far_node, 0, 0, -1, far)),
            "a face against solid rock: covering is not on, so this proves nothing"
        );

        world.set_block(near[0], near[1], near[2], cubara_voxel::BlockId::AIR);
        let after = mesh_node(&world, &registry, &layer_of, far_node, blocks).expect("rock");
        assert!(
            has_face(&after.mesh, far_node, 0, 0, -1, far),
            "dug a hole beside the border and the neighbour shows no side there"
        );
    }

    fn zero_layer() -> Arc<dyn Fn(&str) -> u32 + Send + Sync> {
        Arc::new(|_: &str| 0)
    }

    #[test]
    fn pool_results_match_synchronous_meshing() {
        // Meshing on workers must produce exactly what the synchronous path does,
        // for every requested node (including empty ones, reported as None),
        // across a mix of levels.
        let world = Arc::new(World::new().snapshot());
        let registry = test_registry();
        let layer_of = zero_layer();
        let nodes = [
            NodeKey::new(0, [0, 0, 0]),
            NodeKey::new(0, [1, 0, 0]),
            NodeKey::new(0, [0, 1, 0]),
            NodeKey::new(1, [0, 0, 0]),
            NodeKey::new(2, [-1, 0, 0]),
            NodeKey::new(3, [0, 5, 0]), // high enough to plausibly be empty
        ];
        let mut pool = MeshPool::with_workers(3);
        for &n in &nodes {
            pool.request(
                &world,
                &registry,
                &layer_of,
                n,
                TerrainBlocks::from_registry(&registry),
            );
        }

        let mut got: HashMap<NodeKey, Option<usize>> = HashMap::new();
        while pool.in_flight().next().is_some() {
            for built in pool.poll() {
                got.insert(built.node, built.geometry.map(|g| g.mesh.triangle_count()));
            }
            std::thread::yield_now();
        }

        assert_eq!(got.len(), nodes.len(), "every requested node returns once");
        for &n in &nodes {
            let expect = mesh_node(
                &world,
                &registry,
                &*layer_of,
                n,
                TerrainBlocks::from_registry(&registry),
            )
            .map(|g| g.mesh.triangle_count());
            assert_eq!(got.get(&n).copied().flatten(), expect, "mismatch at {n:?}");
        }
    }

    #[test]
    fn cancelled_nodes_are_dropped_by_poll() {
        let world = Arc::new(World::new().snapshot());
        let registry = test_registry();
        let layer_of = zero_layer();
        let mut pool = MeshPool::with_workers(1);
        let n = NodeKey::new(0, [0, 0, 0]);
        pool.request(
            &world,
            &registry,
            &layer_of,
            n,
            TerrainBlocks::from_registry(&registry),
        );
        pool.cancel(n);
        // Give the worker time to finish and enqueue its (now unwanted) result.
        while pool.in_flight().next().is_some() {
            std::thread::yield_now();
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert!(pool.poll().is_empty(), "cancelled result must not surface");
    }

    /// Cancel a node, then ask for it again against a newer world: what comes
    /// back must be the newer mesh. The cancelled job still runs, and its result
    /// names the same node -- so "is this node wanted?" cannot tell the two
    /// apart, and taking the stale one also drops the right one when it lands.
    /// That is two blocks broken in one chunk before the first re-mesh returns,
    /// with the second hole never drawn, and it is every node of the old world
    /// when the whole world is replaced.
    #[test]
    fn a_cancelled_result_is_not_taken_for_its_replacement() {
        let registry = test_registry();
        let layer_of = zero_layer();
        let blocks = TerrainBlocks::from_registry(&registry);
        let rock = [5, -40, 5];
        let node = NodeKey::containing(ChunkCoord::from_block(rock[0], rock[1], rock[2]), 0);
        let old = Arc::new(World::new().snapshot());
        assert!(
            old.is_solid_at(rock[0], rock[1], rock[2], blocks),
            "not rock"
        );
        let mut dug = World::new();
        dug.set_block(rock[0], rock[1], rock[2], cubara_voxel::BlockId::AIR);
        let new = Arc::new(dug.snapshot());
        let tris = |w: &WorldSnapshot| {
            mesh_node(w, &registry, &*layer_of, node, blocks).map(|g| g.mesh.triangle_count())
        };
        assert_ne!(
            tris(&old),
            tris(&new),
            "the two worlds must mesh differently"
        );

        // One worker, so the jobs run in the order they were queued: the stale
        // one always finishes first, which is the case that goes wrong.
        let mut pool = MeshPool::with_workers(1);
        pool.request(&old, &registry, &layer_of, node, blocks);
        pool.cancel(node);
        pool.request(&new, &registry, &layer_of, node, blocks);
        let mut got = Vec::new();
        // A bounded wait, so a pool that never answers fails here rather than
        // hanging the test run (and `check-tests-can-fail.sh` with it). Counted
        // in sleeps, not read off the clock: Rule 1's check keeps
        // the wall clock out of this crate, tests included.
        let mut waited_ms = 0;
        while pool.is_in_flight(node) {
            assert!(waited_ms < 30_000, "no result in 30 s");
            got.extend(pool.poll());
            std::thread::sleep(std::time::Duration::from_millis(1));
            waited_ms += 1;
        }
        assert_eq!(got.len(), 1, "one result for one wanted node");
        assert_eq!(
            got[0].geometry.as_ref().map(|g| g.mesh.triangle_count()),
            tris(&new),
            "the cancelled job's mesh was taken for the one that replaced it"
        );
    }

    #[test]
    fn requesting_an_in_flight_node_again_surfaces_exactly_once() {
        // A `NodeKey` already names both position and detail level, so there is
        // no "different level supersedes" case -- a different level is simply a
        // different key. Re-requesting the same key while it's in flight must
        // stay a no-op.
        let world = Arc::new(World::new().snapshot());
        let registry = test_registry();
        let layer_of = zero_layer();
        let mut pool = MeshPool::with_workers(1);
        let n = NodeKey::new(0, [0, 0, 0]);
        pool.request(
            &world,
            &registry,
            &layer_of,
            n,
            TerrainBlocks::from_registry(&registry),
        );
        pool.request(
            &world,
            &registry,
            &layer_of,
            n,
            TerrainBlocks::from_registry(&registry),
        );
        let mut results = Vec::new();
        while pool.in_flight().next().is_some() {
            results.extend(pool.poll());
            std::thread::yield_now();
        }
        assert_eq!(results.len(), 1, "must surface exactly once");
    }

    #[test]
    fn mesh_region_returns_nodes_in_ascending_order() {
        let world = World::new();
        let registry = test_registry();
        let layer_of = zero_layer();
        let schedule = [(0u32, 2i32)];
        let built = mesh_region(
            &world,
            &registry,
            &*layer_of,
            ChunkCoord::new(0, 0, 0),
            0..=1,
            &schedule,
            TerrainBlocks::from_registry(&registry),
        );
        assert_eq!(built.len(), 5 * 5 * 2, "(2*2+1)^2 columns * 2 y layers");
        let mut sorted = built.iter().map(|b| b.node).collect::<Vec<_>>();
        sorted.sort();
        assert_eq!(
            built.iter().map(|b| b.node).collect::<Vec<_>>(),
            sorted,
            "already in ascending NodeKey order"
        );
    }

    /// **The parent cache is an accelerator only**: a node built with its
    /// siblings' shared parent is built exactly as alone, and a parent is
    /// made once for all the children that ask with the same snapshot --
    /// again for another snapshot, and again once it has aged out.
    #[test]
    fn a_parent_is_made_once_for_its_children_and_changes_nothing() {
        let registry = test_registry();
        let blocks = TerrainBlocks::from_registry(&registry);
        let world = Arc::new(World::new().snapshot());
        let layer = |_: &str| 0u32;
        let cache = ParentCache::default();
        // The level-0 nodes under a level-1 node holding the surface.
        let y = world.surface_height(8, 8);
        let parent = NodeKey::new(1, [0, y.div_euclid(32), 0]);
        let mut built = 0;
        for k in 0..8 {
            let child = NodeKey::new(
                0,
                [
                    parent.pos[0] * 2 + (k & 1),
                    parent.pos[1] * 2 + ((k >> 1) & 1),
                    parent.pos[2] * 2 + (k >> 2),
                ],
            );
            assert_eq!(child.parent(), parent);
            let alone = build_node(&world, &registry, &layer, child, blocks);
            let shared = build_node_shared(&world, &registry, &layer, child, blocks, &cache);
            let vertices = |b: &BuiltNode| b.geometry.as_ref().map(|g| g.mesh.vertices.clone());
            assert_eq!(vertices(&alone), vertices(&shared), "{child:?}");
            built += alone.geometry.is_some() as usize;
        }
        assert!(built > 0, "no child of {parent:?} draws anything");
        let made = std::cell::Cell::new(0);
        let make = || {
            made.set(made.get() + 1);
            None
        };
        // The cache holds the parent the children made: asking again makes
        // nothing.
        let _ = cache.get(&world, parent, make);
        assert_eq!(made.get(), 0, "made again for the same snapshot");
        let other = Arc::new(World::new().snapshot());
        let _ = cache.get(&other, parent, make);
        assert_eq!(made.get(), 1, "another snapshot is another world");
        for k in 0..PARENT_CACHE as i32 {
            let _ = cache.get(&other, NodeKey::new(1, [k, 100, 0]), || None);
        }
        let _ = cache.get(&world, parent, make);
        assert_eq!(made.get(), 2, "the oldest ages out");
    }
}

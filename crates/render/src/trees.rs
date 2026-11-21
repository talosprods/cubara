//! Trees beyond the first ring (`docs/PROPOSAL_LOD_TRANSITIONS.md` §3 R2,
//! issue #332; made to match level 0 by #433): the instance models, and the
//! arena that tracks which resident node contributed which instances.
//!
//! **Positions and meshes only** (`ARCHITECTURE.md` Rule 3) -- this module
//! never sees a `WorldGen` or a `NodeKey`; it reads
//! [`cubara_voxel::StructureRegistry`]/[`cubara_voxel::BlockRegistry`] (the
//! same lower layer `cubara-world` reads to resolve `TerrainBlocks`) only to
//! build the *models*, once, at startup -- not to decide where a tree
//! stands. `cubara-app`'s streaming glue (`crates/app/src/streaming.rs`)
//! is the only caller that knows about `cubara_world::mesh::node_trees`; it
//! hands this module plain [`TreePlacement`]s.
//!
//! **A tree looks the same at every distance** (#433, measured by
//! `tests/lod_consistency`'s tree harness). Every model is built from the
//! one tree shape ([`cubara_voxel::tree_part_at`]) by the voxel mesher a
//! level-0 tree is meshed with, one per trunk height the oak can have:
//!
//! - **its own height**: an instance draws the model of its tree's trunk
//!   height ([`TreePlacement::height`]), not one height for all;
//! - **its own shade**: the mesher's baked AO is kept, as each vertex's
//!   openness, and `trees.wgsl` lights it as `mesh.wgsl` lights a block;
//! - **its own outline at level 3**: the [`TreeModel::Silhouette`] is the
//!   oak's own surface merged into as few quads as it takes, in flat
//!   colours that are the oak's own per face direction -- the same outline
//!   from every side, in fewer triangles and no texture. It replaced two
//!   boxes, whose square canopy changed the outline at the 512 m edge.

use std::collections::BTreeMap;
use std::ops::{Range, RangeInclusive};

use cubara_voxel::{AnisoChunk, BlockId, BlockRegistry, MeshContext, StructureRegistry};

use crate::anchor::{Anchor, NodeBox};
use crate::arena::NodeId;

/// Which model a node's trees draw with (issue #332 decision 3, #433 item
/// 4): the real oak at levels 1-2, its silhouette from level 3. One per
/// resident node -- a node is wholly one level, so its trees are wholly
/// one model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TreeModel {
    Oak,
    Silhouette,
}

/// One tree a node draws (issue #433): where its trunk's base block stands,
/// in world blocks; how tall its trunk is (which model of that height it is
/// drawn with); and how far the distance morph moves it at the end of its
/// node's band, in blocks -- the move of the ground under its trunk
/// (`cubara_world::mesh::NodeTree`), so the two ride the band together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TreePlacement {
    pub base: [i32; 3],
    pub height: i32,
    pub morph: i32,
}

/// The shapes every tree model is built for, from `assets/structures`: the
/// oak's trunk heights, one model of each, and its canopy radius.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeShapes {
    pub heights: RangeInclusive<i32>,
    pub canopy_radius: i32,
}

impl TreeShapes {
    /// `oak.ron`'s own numbers, or no shape at all when the structure did
    /// not load -- a world with no tree data grows no trees
    /// (`cubara_world::TerrainBlocks::with_oak` has the same tolerance), so
    /// there is nothing to build a model for.
    pub fn of(structures: &StructureRegistry) -> Self {
        structures.get("cubara:oak").map_or(
            Self {
                // Empty on purpose: no oak, no heights, no models.
                #[allow(clippy::reversed_empty_ranges)]
                heights: 1..=0,
                canopy_radius: 0,
            },
            |def| Self {
                heights: def.trunk.height.0..=def.trunk.height.1,
                canopy_radius: def.canopy.radius,
            },
        )
    }

    /// How many models of each kind there are.
    pub fn count(&self) -> usize {
        self.heights.clone().count()
    }

    /// Which model draws a trunk `height` blocks tall: its index in
    /// [`heights`](Self::heights). Every placement comes from the same
    /// structure data, so a height outside the range is a bug; it is drawn
    /// as the nearest rather than not at all.
    pub fn index(&self, height: i32) -> usize {
        debug_assert!(
            self.heights.contains(&height),
            "a tree {height} blocks tall, outside the oak's {:?}",
            self.heights
        );
        (height.clamp(*self.heights.start(), *self.heights.end()) - self.heights.start()) as usize
    }

    /// The local bounding box the tallest tree occupies, relative to its
    /// own position (the trunk's base block) -- in continuous world-space
    /// extents, not `tree_bounds`' own block-*index* convention (its `max`
    /// names the highest solid block's index; that block's own far corner,
    /// in continuous space, is one past it, so the upper bound here is
    /// `tree_bounds`' `max` plus one on every axis; the lower bound needs no
    /// such adjustment, since a block's near corner *is* its index).
    ///
    /// What [`TreeArena::insert`] grows a node's culling AABB by: a canopy
    /// rising above the node's own terrain must not be culled while the
    /// ground under it sits just outside the frustum (issue #332 decision
    /// 2's follow-up -- a node's drawn geometry and its trees are not the
    /// same shape, and culling by the first alone can cut off the second).
    pub fn local_bounds(&self) -> (glam::Vec3, glam::Vec3) {
        if self.count() == 0 {
            return (glam::Vec3::ZERO, glam::Vec3::ZERO);
        }
        let (min, max) = cubara_voxel::tree_bounds(*self.heights.end(), self.canopy_radius);
        (
            glam::Vec3::new(min[0] as f32, min[1] as f32, min[2] as f32),
            glam::Vec3::new(
                max[0] as f32 + 1.0,
                max[1] as f32 + 1.0,
                max[2] as f32 + 1.0,
            ),
        )
    }
}

/// One tree-model vertex: a local position (the trunk's base block is
/// `(0, 0, 0)`), a face normal, this corner's own texel-tile coordinate (as
/// `mesh.wgsl`'s `u`/`v`), the texture-array layer to sample -- or a
/// negative value, meaning "untextured, use `color` instead" (the
/// silhouette's flat colours) -- that colour, and how open the corner is,
/// `0..=1`: what `trees.wgsl` lights it by, `mix(ao_floor, 1, open)`, as
/// `mesh.wgsl` lights a block (#433 item 3).
///
/// Unpacked floats rather than `cubara_voxel::Vertex`'s packed 16 bytes:
/// a tree model is a few hundred vertices built once at startup, not a
/// greedy-meshed arena of millions, so there is nothing to amortise a packed
/// format against, and a flat layout keeps this module independent of the
/// arena's bit-packing scheme.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TreeVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub layer: f32,
    pub color: [f32; 3],
    pub open: f32,
}

/// One instance: where the model's local origin (the trunk's base block)
/// stands, **relative to its node's origin**, and one packed word: that
/// node's slot in the `node_origins` storage buffer (`crate::arena`) and the
/// distance morph's level and lift (#433) -- `trees.wgsl` adds the origin
/// the way `mesh.wgsl` places a packed vertex, and moves the tree with the
/// ground under it.
///
/// Node-relative rather than absolute (issue #379 decision 2, after review
/// finding F16): the render anchor moves every 2,048 blocks of travel, and
/// when it does, every absolute position the GPU holds has to be rewritten.
/// With the instances relative to their node, a rebase rewrites one `vec4`
/// per resident node (`ChunkArena::rebase`) and nothing per instance; this
/// buffer is only ever rewritten when the set of *contributing nodes*
/// changes (`TreeArena::prepare`), exactly as before.
///
/// The word is laid out as `cubara_voxel::Vertex`'s word 2 lays out the
/// same three things: node index in bits 0-15, the morph's level in 20-21,
/// its lift at full morph in 22-31, signed, in steps of
/// [`cubara_voxel::MORPH_DELTA_STEP`] -- so `trees.wgsl` reads them with
/// `mesh.wgsl`'s own shifts.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TreeInstance {
    pub position: [f32; 3],
    pub packed: u32,
}

const NODE_INDEX_MASK: u32 = 0xFFFF;
const MORPH_LEVEL_SHIFT: u32 = 20;
const MORPH_DELTA_SHIFT: u32 = 22;
/// The lift's range, in steps: ten signed bits.
const MORPH_DELTA_STEPS: RangeInclusive<i32> = -512..=511;

impl TreeInstance {
    /// An instance at `position` (relative to its node) of the node in
    /// slot `node_index`, lifted `morph` blocks at the end of the band of
    /// `morph_level`.
    pub fn new(position: [f32; 3], node_index: u32, morph_level: u32, morph: i32) -> Self {
        debug_assert!(node_index <= NODE_INDEX_MASK, "node index out of range");
        let steps = (morph as f32 / cubara_voxel::MORPH_DELTA_STEP).round() as i32;
        debug_assert!(
            MORPH_DELTA_STEPS.contains(&steps),
            "a tree's morph of {morph} blocks does not fit its ten bits"
        );
        let steps = steps.clamp(*MORPH_DELTA_STEPS.start(), *MORPH_DELTA_STEPS.end());
        Self {
            position,
            packed: (node_index & NODE_INDEX_MASK)
                | ((morph_level & 0b11) << MORPH_LEVEL_SHIFT)
                | ((steps as u32) << MORPH_DELTA_SHIFT),
        }
    }

    pub fn node_index(self) -> u32 {
        self.packed & NODE_INDEX_MASK
    }

    pub fn morph_level(self) -> u32 {
        (self.packed >> MORPH_LEVEL_SHIFT) & 0b11
    }

    /// The lift at full morph, in blocks: the arithmetic shift carries the
    /// sign, as `trees.wgsl`'s does.
    pub fn morph(self) -> f32 {
        ((self.packed as i32) >> MORPH_DELTA_SHIFT) as f32 * cubara_voxel::MORPH_DELTA_STEP
    }
}

/// A model's CPU-side triangle data, before it goes to the GPU.
#[derive(Clone, Debug, Default)]
pub struct TreeModelMesh {
    pub vertices: Vec<TreeVertex>,
    pub indices: Vec<u32>,
}

/// One flat quad, its corners ordered so it winds counter-clockwise seen
/// from outside -- the tree pipeline's `front_face: Ccw` / `cull_mode:
/// Back` -- whatever order they were given in (checked by cross product in
/// this module's tests).
fn push_quad(
    mesh: &mut TreeModelMesh,
    mut corners: [[f32; 3]; 4],
    normal: [f32; 3],
    color: [f32; 3],
    open: f32,
) {
    let v = corners.map(glam::Vec3::from);
    if (v[1] - v[0])
        .cross(v[2] - v[0])
        .dot(glam::Vec3::from(normal))
        < 0.0
    {
        corners.reverse();
    }
    let base = mesh.vertices.len() as u32;
    for position in corners {
        mesh.vertices.push(TreeVertex {
            position,
            normal,
            uv: [0.0, 0.0],
            layer: -1.0,
            color,
            open,
        });
    }
    mesh.indices
        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

/// The real oak with a trunk `height` blocks tall, meshed from
/// `assets/structures/oak.ron` with the same voxel mesher a level-0 tree is
/// built with (decision 3): local positions are block coordinates relative
/// to the trunk's base block, so an instance at a tree site's
/// `(x, base, z)` lands exactly where the real voxel trunk would. `None` if
/// the structure or any of its blocks is missing -- same tolerance
/// [`cubara_world::TerrainBlocks::with_oak`] has, for a world with no tree
/// data yet.
///
/// **One model per height** (#433 item 2): the tree's own trunk height, as
/// `tree_part_at` grows it, rather than one representative height whose
/// canopy jumped a block up or down at the 160 m edge.
///
/// **With level 0's shade** (#433 item 3): the mesher bakes AO by the corner
/// rule a level-0 tree's faces get (`cubara_voxel::face_ao_of`), and every
/// vertex keeps it as its openness ([`cubara_voxel::Vertex::openness`]).
/// The tree is meshed standing on a layer of the block it grows on, so the
/// foot of its trunk is shaded as level 0 shades it on level ground; that
/// layer's own faces (everything at or under the tree's foot) are left out.
pub fn build_oak_mesh(
    registry: &BlockRegistry,
    structures: &StructureRegistry,
    layer_of: &dyn Fn(&str) -> u32,
    height: i32,
) -> Option<TreeModelMesh> {
    let def = structures.get("cubara:oak")?;
    let log = registry.id_of(&def.trunk.block)?;
    let leaves = registry.id_of(&def.canopy.block)?;
    let ground = registry.id_of(&def.grows_on)?;
    let r = def.canopy.radius;
    let (min, max) = cubara_voxel::tree_bounds(height, r);
    // One row under the tree's own box: the ground it stands on.
    let floor = min[1] - 1;
    // No test pins these three `+ 1`s individually (`check-tests-can-fail.sh`
    // found the middle one survive a `+1` -> `+2` mutation): `tree_part_at`
    // only ever returns `Some` inside `tree_bounds`' own box, so a lattice
    // sized *larger* than necessary samples extra all-air cells at the
    // excess rows/columns -- the mesher emits no faces for those, so the
    // model's actual triangles are unaffected either way. A lattice sized
    // *smaller* than `tree_bounds` would be a real bug (a clipped canopy),
    // and `every_model_is_its_tree_to_the_block` would see it.
    let dims = [
        (max[0] - min[0] + 1) as usize,
        (max[1] - floor + 1) as usize,
        (max[2] - min[2] + 1) as usize,
    ];
    let chunk = AnisoChunk::from_fn(dims, [1, 1, 1], |x, y, z| {
        let dx = x as i32 + min[0];
        let dy = y as i32 + floor;
        let dz = z as i32 + min[2];
        if dy < min[1] {
            return ground;
        }
        match cubara_voxel::tree_part_at(dx.abs(), dz.abs(), dy, height, r) {
            Some(cubara_voxel::TreePart::Trunk) => log,
            Some(cubara_voxel::TreePart::Canopy) => leaves,
            None => BlockId::AIR,
        }
    });
    let ctx = MeshContext { registry, layer_of };
    // Nothing outside the lattice: no border face is covered and no skirt
    // is laid; out of it is air, for the faces and for their AO alike.
    let mesh = chunk.build_mesh_with_neighbours(&ctx, |_, _, _| false, |_, _, _| false);
    let mut out = TreeModelMesh::default();
    for six in mesh.indices.chunks(6) {
        let base = *six.iter().min().expect("six indices") as usize;
        let corners = &mesh.vertices[base..base + 4];
        // The ground's faces: its top, under the tree's foot, and its sides.
        if corners.iter().all(|v| v.y() as i32 + floor <= min[1]) {
            continue;
        }
        let first = out.vertices.len() as u32;
        out.vertices.extend(corners.iter().map(|v| TreeVertex {
            position: [
                v.x() as f32 + min[0] as f32,
                v.y() as f32 + floor as f32,
                v.z() as f32 + min[2] as f32,
            ],
            normal: v.face().normal(),
            uv: [v.u() as f32, v.v() as f32],
            layer: v.tex_layer() as f32,
            color: [1.0, 1.0, 1.0],
            open: v.openness(),
        }));
        // The quad as the mesher triangulated it (its diagonal follows its
        // AO), moved to where its corners now are.
        out.indices
            .extend(six.iter().map(|&i| first + (i - base as u32)));
    }
    Some(out)
}

/// **The oak's own outline in flat colours** (#433 item 4): `oak`'s surface
/// -- the same faces, so the same silhouette from every side -- merged into
/// as few quads as it takes, each face direction and texture one flat
/// colour: the texture's mean (`colour_of(layer)`), shaded by the mean of
/// the oak's openness over those faces. `trees.wgsl` lights a vertex by
/// `mix(ao_floor, 1, open)`, which is linear in `open`, so each direction's
/// area-weighted colour is the oak's exactly: what level 2 shows of a tree
/// in the mean, per direction, level 3 shows too.
///
/// It is what level 3 draws (a tree there is a few pixels tall, under the
/// size a texture or an AO gradient shows at). The two boxes it replaced
/// had the same size but a square canopy, so the outline changed at the
/// 512 m edge, in flat colours that were not the oak's.
pub fn build_silhouette_mesh(
    oak: &TreeModelMesh,
    colour_of: &dyn Fn(u32) -> [f32; 3],
) -> TreeModelMesh {
    // Per face direction (+X -X +Y -Y +Z -Z, `Face` order), the plane, and
    // the texture: the unit cells the oak's quads cover there.
    type Key = (usize, i32, u32);
    let mut cells: BTreeMap<Key, std::collections::BTreeSet<(i32, i32)>> = BTreeMap::new();
    // Per face direction and texture: area, and openness times area.
    let mut shade: BTreeMap<(usize, u32), (f64, f64)> = BTreeMap::new();
    for six in oak.indices.chunks(6) {
        let base = *six.iter().min().expect("six indices") as usize;
        let vs = &oak.vertices[base..base + 4];
        let n = vs[0].normal;
        let axis = (0..3).find(|&a| n[a] != 0.0).expect("an axis-aligned face");
        let dir = 2 * axis + usize::from(n[axis] < 0.0);
        let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
        let plane = vs[0].position[axis].round() as i32;
        let layer = vs[0].layer as u32;
        let lo = |a: usize| {
            vs.iter()
                .map(|p| p.position[a])
                .fold(f32::MAX, f32::min)
                .round() as i32
        };
        let hi = |a: usize| {
            vs.iter()
                .map(|p| p.position[a])
                .fold(f32::MIN, f32::max)
                .round() as i32
        };
        let set = cells.entry((dir, plane, layer)).or_default();
        for a in lo(u)..hi(u) {
            for b in lo(v)..hi(v) {
                set.insert((a, b));
            }
        }
        // The quad's mean openness over the two triangles it is drawn as:
        // the corners on the shared diagonal count in both.
        let o: Vec<f64> = vs.iter().map(|p| p.open as f64).collect();
        let mean = if six[0] as usize == base {
            (2.0 * (o[0] + o[2]) + o[1] + o[3]) / 6.0
        } else {
            (2.0 * (o[1] + o[3]) + o[0] + o[2]) / 6.0
        };
        let area = ((hi(u) - lo(u)) * (hi(v) - lo(v))) as f64;
        let s = shade.entry((dir, layer)).or_default();
        s.0 += area;
        s.1 += mean * area;
    }
    let mut out = TreeModelMesh::default();
    for ((dir, plane, layer), mut set) in cells {
        let axis = dir / 2;
        let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
        let mut normal = [0.0f32; 3];
        normal[axis] = if dir % 2 == 0 { 1.0 } else { -1.0 };
        let (area, open) = shade[&(dir, layer)];
        let open = (open / area) as f32;
        let colour = colour_of(layer);
        // Greedy rectangles: the first cell left, as far along `u` as the
        // row runs, then as many rows along `v` as run as far.
        while let Some(&(a0, b0)) = set.iter().next() {
            let mut a1 = a0 + 1;
            while set.contains(&(a1, b0)) {
                a1 += 1;
            }
            let mut b1 = b0 + 1;
            while (a0..a1).all(|a| set.contains(&(a, b1))) {
                b1 += 1;
            }
            for b in b0..b1 {
                for a in a0..a1 {
                    set.remove(&(a, b));
                }
            }
            let corner = |a: i32, b: i32| {
                let mut p = [0.0f32; 3];
                p[axis] = plane as f32;
                p[u] = a as f32;
                p[v] = b as f32;
                p
            };
            push_quad(
                &mut out,
                [
                    corner(a0, b0),
                    corner(a1, b0),
                    corner(a1, b1),
                    corner(a0, b1),
                ],
                normal,
                colour,
                open,
            );
        }
    }
    out
}

/// Every model of both kinds, CPU-side, one per trunk height in
/// [`TreeShapes::heights`]: what [`build_tree_models`] uploads, and what
/// `tests/lod_consistency` measures. Empty when the oak's data is missing.
pub fn tree_model_meshes(
    registry: &BlockRegistry,
    structures: &StructureRegistry,
    layer_of: &dyn Fn(&str) -> u32,
) -> (Vec<TreeModelMesh>, Vec<TreeModelMesh>) {
    let names = registry.texture_names();
    let colour_of = |layer: u32| {
        names
            .iter()
            .find(|name| layer_of(name) == layer)
            .and_then(|name| crate::materials::mean_color(name))
            .unwrap_or([0.5, 0.5, 0.5])
    };
    let oaks: Option<Vec<TreeModelMesh>> = TreeShapes::of(structures)
        .heights
        .map(|h| build_oak_mesh(registry, structures, layer_of, h))
        .collect();
    let oaks = oaks.unwrap_or_default();
    let silhouettes = oaks
        .iter()
        .map(|oak| build_silhouette_mesh(oak, &colour_of))
        .collect();
    (oaks, silhouettes)
}

/// One model's GPU buffers, uploaded once and never rewritten.
pub struct TreeModelGpu {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

fn upload_model(device: &wgpu::Device, queue: &wgpu::Queue, mesh: &TreeModelMesh) -> TreeModelGpu {
    let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tree-model-vertices"),
        size: (mesh.vertices.len().max(1) * std::mem::size_of::<TreeVertex>()) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !mesh.vertices.is_empty() {
        queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&mesh.vertices));
    }
    let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tree-model-indices"),
        size: (mesh.indices.len().max(1) * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !mesh.indices.is_empty() {
        queue.write_buffer(&index_buffer, 0, bytemuck::cast_slice(&mesh.indices));
    }
    TreeModelGpu {
        vertex_buffer,
        index_buffer,
        index_count: mesh.indices.len() as u32,
    }
}

/// Every model's GPU buffers, built once when the renderer starts
/// ([`crate::SceneRenderer::new`]): per kind, one per trunk height, in
/// [`TreeShapes::heights`] order -- the order [`TreeArena`]'s height ranges
/// are in. Both empty only when `assets/structures/oak.ron`'s data is
/// missing, and then no tree is ever placed either.
pub struct TreeModels {
    pub oak: Vec<TreeModelGpu>,
    pub silhouette: Vec<TreeModelGpu>,
}

impl TreeModels {
    /// The models of `kind`, one per height.
    pub fn of(&self, kind: TreeModel) -> &[TreeModelGpu] {
        match kind {
            TreeModel::Oak => &self.oak,
            TreeModel::Silhouette => &self.silhouette,
        }
    }
}

/// Build [`TreeModels`] from the already-loaded registry/structures this
/// renderer's other startup code resolved -- no new file I/O beyond what
/// [`crate::render::load_mesh_assets`]'s caller already does for the block
/// registry; this loads `assets/structures` itself, the same file
/// `cubara_server::assets::load_structure_registry` loads for the live
/// streaming glue, since `cubara-render` cannot depend on either that crate
/// or `cubara-world` (Rule 3) to share the already-loaded one.
pub fn build_tree_models(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    registry: &BlockRegistry,
    structures: &StructureRegistry,
    layer_of: &dyn Fn(&str) -> u32,
) -> TreeModels {
    let (oaks, silhouettes) = tree_model_meshes(registry, structures, layer_of);
    TreeModels {
        oak: oaks
            .iter()
            .map(|m| upload_model(device, queue, m))
            .collect(),
        silhouette: silhouettes
            .iter()
            .map(|m| upload_model(device, queue, m))
            .collect(),
    }
}

/// Grow-on-demand instance buffer bookkeeping, one per model kind -- the
/// same "grow and keep the new capacity" shape `SceneRenderer`'s figure
/// vertex buffer already uses -- and which of its instances each height's
/// model draws.
struct InstanceBuffer {
    buffer: Option<wgpu::Buffer>,
    capacity: usize,
    /// Per trunk height (in [`TreeShapes::heights`] order), the instances
    /// that height's model draws: consecutive, in height order.
    ranges: Vec<Range<u32>>,
}

impl InstanceBuffer {
    const fn new() -> Self {
        Self {
            buffer: None,
            capacity: 0,
            ranges: Vec::new(),
        }
    }

    /// Upload `by_height` -- per height, its instances -- one height after
    /// the other.
    fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        by_height: &[Vec<TreeInstance>],
    ) {
        self.ranges.clear();
        let mut start = 0u32;
        for h in by_height {
            self.ranges.push(start..start + h.len() as u32);
            start += h.len() as u32;
        }
        let total = start as usize;
        if total == 0 {
            return;
        }
        let instances: Vec<TreeInstance> = by_height.concat();
        // Reuse the existing GPU buffer (just `queue.write_buffer` into it)
        // whenever it already has room, rather than allocating a new one on
        // every call -- `check-tests-can-fail.sh` found this condition
        // survive an "always reallocate" mutation, undocumented until now:
        // the difference is GPU buffer *identity* across calls, which
        // nothing in this module's test surface can observe without
        // instrumenting wgpu itself. It is also no longer this type's main
        // defence against per-frame cost -- `TreeArena::prepare` now skips
        // calling `upload` at all when the contributing node set hasn't
        // changed (see its own doc comment), which is the decision that
        // actually has a test (`tests/tree_culling.rs`'s "turning away"
        // case exercises both the changed and -- implicitly, by not
        // panicking -- the unchanged path). This check only matters for the
        // remaining case: contributors changed, but the count still fits.
        if self.buffer.is_none() || total > self.capacity {
            self.capacity = total.next_power_of_two();
            self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tree-instances"),
                size: (self.capacity * std::mem::size_of::<TreeInstance>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        queue.write_buffer(
            self.buffer.as_ref().expect("just ensured"),
            0,
            bytemuck::cast_slice(&instances),
        );
    }

    /// The buffer and each height's range of it, or `None` with nothing to
    /// draw.
    fn instances(&self) -> Option<(&wgpu::Buffer, &[Range<u32>])> {
        self.buffer
            .as_ref()
            .filter(|_| self.ranges.iter().any(|r| !r.is_empty()))
            .map(|b| (b, self.ranges.as_slice()))
    }
}

/// Per-node tree positions, by resident node: what each resident node
/// contributed, plus a culling AABB that encloses *both* the node's own
/// terrain and every one of its trees (not the terrain's AABB alone --
/// issue #332 decision 2's follow-up: a canopy rising above the node's
/// drawn ground must stay visible even on a frame where that ground itself
/// is just outside the frustum, or a tree whose base is off-screen but
/// whose crown is not would pop at the screen edge). Dropping or replacing
/// a node drops exactly its own trees and nothing else's.
///
/// **Only drawn (frustum-visible) nodes contribute** (decision 2) --
/// [`prepare`](Self::prepare) re-filters `by_node` by the *same* frustum
/// test [`crate::arena::ChunkArena::prepare`] culls the terrain with
/// (`Frustum::intersects_aabb`, called here rather than reimplemented,
/// `ARCHITECTURE.md` Rule 5), every frame a camera is rendered from. That
/// visibility test runs unconditionally every frame (it has to: the camera
/// can move), but re-flattening the surviving positions into a fresh `Vec`
/// and re-uploading them to the GPU does not -- `prepare` only does that
/// when the actual *set* of contributing nodes changed since the last call,
/// which on a still camera (most frames of most benches, and plenty of real
/// play) is never. See [`last_oak_contributors`](Self::last_oak_contributors)'s
/// doc comment.
pub struct TreeArena {
    by_node: BTreeMap<NodeId, TreeNode>,
    /// The trunk heights there are models of, and how far the tallest tree
    /// reaches from its own position -- `(min, max)` offsets added to a
    /// node's own AABB, over every tree it contributes, in
    /// [`insert`](Self::insert). Fixed for the arena's whole lifetime.
    shapes: TreeShapes,
    local_bounds: (glam::Vec3, glam::Vec3),
    /// The render anchor every node's culling box is relative to -- the
    /// same one the `ChunkArena` whose `node_origins` the instances index
    /// has, since the two are rebased together (`crate::anchor`).
    anchor: Anchor,
    oak: InstanceBuffer,
    silhouette: InstanceBuffer,
    /// The node ids [`prepare`](Self::prepare) found contributing oak
    /// instances last time it ran, in the same (fixed, `NodeId`-ordered)
    /// order they were visited in -- so two calls' lists compare equal
    /// exactly when the same nodes passed the frustum test both times, with
    /// nothing having to hash or sort anything to find out. When they do,
    /// `prepare` skips rebuilding the flattened position `Vec` and skips
    /// `queue.write_buffer` entirely for that model -- the actual per-frame
    /// cost this type was found to carry even with an unchanged camera
    /// (measured: comparing a few hundred `NodeId`s costs far less than
    /// copying their positions and uploading them, which is what used to
    /// happen unconditionally).
    last_oak_contributors: Vec<NodeId>,
    last_silhouette_contributors: Vec<NodeId>,
}

impl TreeArena {
    pub fn new(shapes: TreeShapes, anchor: Anchor) -> Self {
        Self {
            by_node: BTreeMap::new(),
            local_bounds: shapes.local_bounds(),
            shapes,
            anchor,
            oak: InstanceBuffer::new(),
            silhouette: InstanceBuffer::new(),
            last_oak_contributors: Vec::new(),
            last_silhouette_contributors: Vec::new(),
        }
    }

    /// Record (or replace) `id`'s trees: each [`TreePlacement`]'s base (the
    /// trunk's base block, in world blocks) is stored relative to the
    /// node's `origin` -- an exact integer subtraction -- alongside
    /// `node_index`, the node's slot in the arena's `node_origins` buffer
    /// that `trees.wgsl` adds back, and the morph of `id`'s level
    /// ([`TreeInstance`]); and filed under its trunk height's model.
    /// `node_local_aabb` is the node's own (terrain) bounds **relative to
    /// `origin`**, as the mesher gives them; the box actually stored is that
    /// grown by the tallest tree's reach around every tree, where it stands
    /// and where the morph takes it -- all of it node-local, so no absolute
    /// `f32` position is formed -- so a tall canopy standing near a node's
    /// edge is never culled while the node's own ground is still in view,
    /// or vice versa. An empty `trees` is the same as having none --
    /// removed outright, so a node that grows no trees never occupies a slot
    /// [`prepare`](Self::prepare) has to visit for nothing.
    pub fn insert(
        &mut self,
        id: NodeId,
        node_index: u32,
        origin: [i32; 3],
        model: TreeModel,
        trees: Vec<TreePlacement>,
        node_local_aabb: crate::culling::Aabb,
    ) {
        if trees.is_empty() {
            self.remove(id);
            return;
        }
        // The CPU invariant behind `trees.wgsl`'s `node_origins[node_index]`
        // (issue #412): every instance stored here indexes the `ChunkArena`'s
        // origins buffer, which is exactly `MAX_NODES` entries long, and a
        // real slot always comes from that arena's allocator over
        // `0..MAX_NODES` (`ChunkArena::node_index`, read right after a
        // successful insert -- `render.rs`'s `ArenaSink::try_insert`). A
        // standalone caller (a test, a future second arena) that makes up an
        // index past the buffer would be an out-of-bounds storage read on
        // the GPU -- zero on a real GPU, a fault on a software rasteriser
        // with no robust buffer access -- so it is refused here, on the CPU,
        // never clamped in the shader.
        debug_assert!(
            node_index < crate::arena::MAX_NODES,
            "tree node_index {node_index} is past the node-origins buffer \
             ({} slots) -- an instance may only reference a slot the ChunkArena \
             can have handed out",
            crate::arena::MAX_NODES
        );
        let (local_min, local_max) = self.local_bounds;
        let mut min = node_local_aabb.min;
        let mut max = node_local_aabb.max;
        let instances = trees
            .iter()
            .map(|t| {
                // Where the tree stands relative to its node: integers, so
                // exact, and small, so exact in `f32` too.
                let rel = glam::Vec3::new(
                    (t.base[0] as i64 - origin[0] as i64) as f32,
                    (t.base[1] as i64 - origin[1] as i64) as f32,
                    (t.base[2] as i64 - origin[2] as i64) as f32,
                );
                let lifted = rel + glam::Vec3::Y * t.morph as f32;
                min = min.min(rel + local_min).min(lifted + local_min);
                max = max.max(rel + local_max).max(lifted + local_max);
                (
                    self.shapes.index(t.height),
                    TreeInstance::new(rel.to_array(), node_index, id.level, t.morph),
                )
            })
            .collect();
        let place = NodeBox::new(origin, crate::culling::Aabb::new(min, max));
        self.by_node.insert(
            id,
            TreeNode {
                model,
                instances,
                aabb: place.relative_to(self.anchor),
                place,
            },
        );
    }

    pub fn remove(&mut self, id: NodeId) {
        self.by_node.remove(&id);
    }

    /// The render anchor every culling box here is relative to. Carried in
    /// [`crate::scene::TreeDraw`] so the one scene pass can check it against
    /// the `ChunkArena` whose `node_origins` the instances index.
    pub fn anchor(&self) -> Anchor {
        self.anchor
    }

    /// Re-express every node's culling box against `anchor` (issue #379
    /// decision 2). The instances are node-relative and are not touched --
    /// their node's origin moves in the `ChunkArena`'s buffer, which is
    /// rebased alongside this.
    pub fn rebase(&mut self, anchor: Anchor) {
        if anchor == self.anchor {
            return;
        }
        self.anchor = anchor;
        for node in self.by_node.values_mut() {
            node.aabb = node.place.relative_to(anchor);
        }
    }

    /// Forget every node at once -- the streamed world was just replaced
    /// ([`crate::Renderer::clear_nodes`]). Also forgets what contributed
    /// last frame: the next [`prepare`](Self::prepare) must not compare the
    /// new (empty) world against the old one's memory and conclude nothing
    /// changed.
    pub fn clear(&mut self) {
        self.by_node.clear();
        self.last_oak_contributors.clear();
        self.last_silhouette_contributors.clear();
    }

    /// Re-test every resident node's (tree-inclusive) AABB against
    /// `frustum` -- the same test, on the same per-node AABBs in spirit,
    /// [`crate::arena::ChunkArena::prepare`] culls the terrain with (issue
    /// #332 decision 2) -- every frame. Only when the resulting *set* of
    /// contributing nodes differs from last frame's does it rebuild the
    /// flattened instance list and upload it; see
    /// [`last_oak_contributors`](Self::last_oak_contributors). Call once per
    /// frame, alongside `ChunkArena::prepare`.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frustum: &crate::culling::Frustum,
    ) {
        puffin::profile_function!();
        // `BTreeMap` iteration order is `NodeId`'s `Ord` (level then pos) --
        // fixed, so two calls that found the same nodes in view produce
        // identically-ordered lists, which is what makes the `!=` checks
        // below a correct "did anything actually change" test rather than a
        // false negative from pure reordering.
        let mut oak_contributors = Vec::new();
        let mut silhouette_contributors = Vec::new();
        for (&id, node) in &self.by_node {
            if !frustum.intersects_aabb(&node.aabb) {
                continue;
            }
            match node.model {
                TreeModel::Oak => oak_contributors.push(id),
                TreeModel::Silhouette => silhouette_contributors.push(id),
            }
        }
        let heights = self.shapes.count();
        if oak_contributors != self.last_oak_contributors {
            let by_height = flatten(&self.by_node, &oak_contributors, heights);
            self.oak.upload(device, queue, &by_height);
            self.last_oak_contributors = oak_contributors;
        }
        if silhouette_contributors != self.last_silhouette_contributors {
            let by_height = flatten(&self.by_node, &silhouette_contributors, heights);
            self.silhouette.upload(device, queue, &by_height);
            self.last_silhouette_contributors = silhouette_contributors;
        }
    }

    /// The oak instances to draw: the buffer, and per trunk height (in
    /// [`TreeShapes::heights`] order) the range of it that height's model
    /// draws.
    pub fn oak_instances(&self) -> Option<(&wgpu::Buffer, &[Range<u32>])> {
        self.oak.instances()
    }

    /// [`oak_instances`](Self::oak_instances), for the silhouette.
    pub fn silhouette_instances(&self) -> Option<(&wgpu::Buffer, &[Range<u32>])> {
        self.silhouette.instances()
    }

    /// How many nodes are currently recorded, resident or not -- a white-box
    /// accessor for this crate's tests, which can check `insert`/`remove`'s
    /// bookkeeping (and `ArenaSink::try_insert`'s, `crate::render`) without
    /// a device to call [`prepare`](Self::prepare) through. The
    /// frustum-culling behaviour itself is proven in
    /// `tests/tree_culling.rs`, which does have one.
    #[cfg(test)]
    pub(crate) fn node_count(&self) -> usize {
        self.by_node.len()
    }

    /// The stored (tree-inclusive) culling AABB for `id`, if resident -- a
    /// white-box accessor for this module's own unit tests, which check
    /// [`insert`](Self::insert)'s AABB union directly rather than through a
    /// real `Frustum`/device (that proof lives in `tests/tree_culling.rs`).
    #[cfg(test)]
    fn aabb_of(&self, id: NodeId) -> Option<crate::culling::Aabb> {
        self.by_node.get(&id).map(|node| node.aabb)
    }

    /// The node-relative instances stored for `id`, each with its height's
    /// model index -- white-box, for the unit test that pins the
    /// subtraction [`insert`](Self::insert) does.
    #[cfg(test)]
    fn instances_of(&self, id: NodeId) -> Option<&[(usize, TreeInstance)]> {
        self.by_node.get(&id).map(|node| node.instances.as_slice())
    }
}

/// What one resident node contributed: its model, its instances (already
/// node-relative, [`TreeInstance`]), each with the index of its trunk
/// height's model, and the culling box enclosing both the node's terrain
/// and every one of its trees -- `place` in world terms, and `aabb`
/// relative to the arena's anchor, which is what the frustum tests.
struct TreeNode {
    model: TreeModel,
    instances: Vec<(usize, TreeInstance)>,
    place: NodeBox,
    aabb: crate::culling::Aabb,
}

/// The instances every node in `ids` contributed, per trunk height, each
/// height's in node order -- what [`TreeArena::prepare`] uploads for one
/// model once it has decided the set of contributors actually changed.
fn flatten(
    by_node: &BTreeMap<NodeId, TreeNode>,
    ids: &[NodeId],
    heights: usize,
) -> Vec<Vec<TreeInstance>> {
    let mut out = vec![Vec::new(); heights];
    for id in ids {
        for &(h, instance) in &by_node[id].instances {
            out[h].push(instance);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real_assets() -> (
        BlockRegistry,
        StructureRegistry,
        crate::materials::TextureLayers,
    ) {
        let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let registry = cubara_voxel::BlockRegistry::load(&repo_root.join("assets/blocks"))
            .expect("assets/blocks must load");
        let structures =
            cubara_voxel::StructureRegistry::load(&repo_root.join("assets/structures"))
                .expect("assets/structures must load");
        let layers = crate::materials::TextureLayers::from_registry(&registry);
        (registry, structures, layers)
    }

    /// Every face of every model must wind so its cross product points the
    /// same way as the normal it carries, or the tree pipeline's back-face
    /// culling (`front_face: Ccw`) would drop exactly the faces a viewer
    /// outside the tree needs to see. The oak's come from the mesher; the
    /// silhouette's from [`push_quad`], which orders them itself.
    #[test]
    fn every_model_face_winds_outward() {
        let (registry, structures, layers) = real_assets();
        let (oaks, silhouettes) =
            tree_model_meshes(&registry, &structures, &|n| layers.layer_of(n));
        assert!(!oaks.is_empty(), "oak.ron ships");
        for mesh in oaks.iter().chain(&silhouettes) {
            for tri in mesh.indices.chunks(3) {
                let p = |k: usize| glam::Vec3::from(mesh.vertices[tri[k] as usize].position);
                let normal = glam::Vec3::from(mesh.vertices[tri[0] as usize].normal);
                let cross = (p(1) - p(0)).cross(p(2) - p(0));
                assert!(
                    cross.normalize().dot(normal) > 0.99,
                    "a face at {:?} wound inward: cross {cross:?} vs normal {normal:?}",
                    p(0)
                );
            }
        }
    }

    /// The cells a model's faces enclose, as a set of blocks: a block is in
    /// the tree when the faces seen along +x from outside it cross an odd
    /// number of times before it (the model's faces bound a closed solid,
    /// the trunk's foot open onto the ground).
    fn solid_cells(mesh: &TreeModelMesh) -> std::collections::BTreeSet<[i32; 3]> {
        // Every x-facing face, as the unit squares it covers.
        let mut walls: BTreeMap<(i32, i32), Vec<i32>> = BTreeMap::new();
        for q in mesh.indices.chunks(6) {
            let base = *q.iter().min().unwrap() as usize;
            let vs = &mesh.vertices[base..base + 4];
            if vs[0].normal[0] == 0.0 {
                continue;
            }
            let x = vs[0].position[0].round() as i32;
            let lo = |a: usize| vs.iter().map(|v| v.position[a]).fold(f32::MAX, f32::min) as i32;
            let hi = |a: usize| vs.iter().map(|v| v.position[a]).fold(f32::MIN, f32::max) as i32;
            for y in lo(1)..hi(1) {
                for z in lo(2)..hi(2) {
                    walls.entry((y, z)).or_default().push(x);
                }
            }
        }
        let mut out = std::collections::BTreeSet::new();
        for ((y, z), mut xs) in walls {
            xs.sort();
            for pair in xs.chunks(2) {
                for x in pair[0]..pair[1] {
                    out.insert([x, y, z]);
                }
            }
        }
        out
    }

    /// **Every model is its tree, to the block** (#433 items 2 and 4): the
    /// oak and the silhouette of each trunk height enclose exactly the
    /// blocks `tree_part_at` grows for that height -- so an instance stands
    /// as tall as its own tree, not one representative height, and the
    /// silhouette has the oak's outline from every side. Goes red when a
    /// model is built for the wrong height, a lattice clips the canopy, or
    /// the silhouette's merge drops or adds a cell.
    #[test]
    fn every_model_is_its_tree_to_the_block() {
        let (registry, structures, layers) = real_assets();
        let shapes = TreeShapes::of(&structures);
        let (oaks, silhouettes) =
            tree_model_meshes(&registry, &structures, &|n| layers.layer_of(n));
        assert_eq!(shapes.heights, 4..=6, "oak.ron's trunk heights");
        assert_eq!((oaks.len(), silhouettes.len()), (3, 3));
        for (k, height) in shapes.heights.clone().enumerate() {
            let r = shapes.canopy_radius;
            let mut want = std::collections::BTreeSet::new();
            let (min, max) = cubara_voxel::tree_bounds(height, r);
            for y in min[1]..=max[1] {
                for z in min[2]..=max[2] {
                    for x in min[0]..=max[0] {
                        if cubara_voxel::tree_part_at(x.abs(), z.abs(), y, height, r).is_some() {
                            want.insert([x, y, z]);
                        }
                    }
                }
            }
            assert_eq!(solid_cells(&oaks[k]), want, "the oak {height} blocks tall");
            assert_eq!(
                solid_cells(&silhouettes[k]),
                want,
                "the silhouette {height} blocks tall"
            );
        }
    }

    /// **The silhouette is the oak's surface in fewer quads, each direction
    /// the oak's own colour** (#433 item 4): per face direction the same
    /// area, and the same area-weighted openness per texture -- which
    /// `trees.wgsl`'s `mix(ao_floor, 1, open)` turns into the same mean
    /// light -- in fewer triangles than the oak. Goes red when the merge
    /// loses area, a direction's shade is averaged with another's, or the
    /// openness is not area-weighted.
    #[test]
    fn the_silhouette_shows_the_oaks_area_and_shade_per_direction() {
        let (registry, structures, layers) = real_assets();
        let (oaks, silhouettes) =
            tree_model_meshes(&registry, &structures, &|n| layers.layer_of(n));
        let per_direction = |mesh: &TreeModelMesh| {
            let mut out: BTreeMap<(i32, i32, i32), (f64, f64)> = BTreeMap::new();
            for q in mesh.indices.chunks(6) {
                let base = *q.iter().min().unwrap() as usize;
                let vs = &mesh.vertices[base..base + 4];
                let p = |k: usize| glam::Vec3::from(vs[k].position);
                let area = (p(1) - p(0)).cross(p(3) - p(0)).length() as f64;
                let o: Vec<f64> = vs.iter().map(|v| v.open as f64).collect();
                let open = if q[0] as usize == base {
                    (2.0 * (o[0] + o[2]) + o[1] + o[3]) / 6.0
                } else {
                    (2.0 * (o[1] + o[3]) + o[0] + o[2]) / 6.0
                };
                let n = vs[0].normal.map(|c| c as i32);
                let e = out.entry((n[0], n[1], n[2])).or_default();
                e.0 += area;
                e.1 += open * area;
            }
            out
        };
        for (oak, silhouette) in oaks.iter().zip(&silhouettes) {
            let (a, b) = (per_direction(oak), per_direction(silhouette));
            assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
            for (dir, (area, open)) in &a {
                let (s_area, s_open) = b[dir];
                assert!(
                    (area - s_area).abs() < 1e-6,
                    "{dir:?}: area {area} against {s_area}"
                );
                assert!(
                    (open / area - s_open / s_area).abs() < 1e-5,
                    "{dir:?}: openness {} against {}",
                    open / area,
                    s_open / s_area
                );
            }
            let (t_oak, t_sil) = (oak.indices.len() / 3, silhouette.indices.len() / 3);
            eprintln!("TREE MODEL: oak {t_oak} triangles, silhouette {t_sil}");
            assert!(
                t_sil < t_oak,
                "silhouette {t_sil} against the oak's {t_oak}"
            );
            // A guard against the merge silently stopping (the oak's own
            // quads re-emitted one by one are as many as the oak's), not an
            // exact pin: the canopy's stepped edge is most of what is left.
            // Chosen, not derived.
            assert!(t_sil <= 120, "the silhouette grew to {t_sil} triangles");
            assert!(
                silhouette.vertices.iter().all(|v| v.layer < 0.0),
                "flat colours"
            );
        }
    }

    /// **The oak keeps level 0's shade** (#433 item 3): its vertices carry
    /// the mesher's baked openness, not a constant -- dark under the canopy
    /// and at the trunk's foot, where it stands on the ground it was meshed
    /// on -- and none of that ground's faces is in the model.
    #[test]
    fn the_oak_is_shaded_where_level_0_shades_it() {
        let (registry, structures, layers) = real_assets();
        let (oaks, _) = tree_model_meshes(&registry, &structures, &|n| layers.layer_of(n));
        for oak in &oaks {
            assert!(
                oak.vertices.iter().all(|v| v.position[1] >= 0.0),
                "the ground's faces were left in"
            );
            // The trunk's side faces: the corners at its foot sit on the
            // ground, the ones under the canopy are under the leaves.
            let trunk_foot: Vec<f32> = oak
                .vertices
                .iter()
                .filter(|v| v.normal[1] == 0.0 && v.position[1] == 0.0)
                .map(|v| v.open)
                .collect();
            assert!(!trunk_foot.is_empty(), "the trunk reaches the ground");
            assert!(
                trunk_foot.iter().all(|&o| o < 1.0),
                "the trunk's foot is not shaded: {trunk_foot:?}"
            );
            assert!(
                oak.vertices.iter().any(|v| v.open == 1.0),
                "nothing is open: the openness is not the mesher's"
            );
        }
    }

    /// The packed word round-trips: slot, level and a lift of either sign,
    /// as `trees.wgsl` reads them.
    #[test]
    fn an_instance_carries_its_slot_and_its_morph() {
        for (slot, level, morph) in [(0, 0, 0), (17, 1, -3), (16_383, 2, 23), (5, 3, -128)] {
            let t = TreeInstance::new([1.0, 2.0, 3.0], slot, level, morph);
            assert_eq!(
                (t.node_index(), t.morph_level(), t.morph()),
                (slot, level, morph as f32)
            );
        }
    }

    /// No reach beyond the node's own AABB -- for tests that only care about
    /// `insert`/`remove` bookkeeping, not the AABB union itself.
    fn no_local_bounds() -> TreeArena {
        let mut arena = TreeArena::new(
            TreeShapes {
                heights: 4..=6,
                canopy_radius: 2,
            },
            crate::anchor::ORIGIN,
        );
        arena.local_bounds = (glam::Vec3::ZERO, glam::Vec3::ZERO);
        arena
    }

    fn tree(base: [i32; 3]) -> TreePlacement {
        TreePlacement {
            base,
            height: 5,
            morph: 0,
        }
    }

    #[test]
    fn tree_arena_starts_with_no_instances() {
        let arena = no_local_bounds();
        assert!(arena.oak_instances().is_none());
        assert!(arena.silhouette_instances().is_none());
    }

    fn unit_aabb() -> crate::culling::Aabb {
        crate::culling::Aabb::new(glam::Vec3::ZERO, glam::Vec3::ONE)
    }

    #[test]
    fn inserting_an_empty_list_is_the_same_as_having_none() {
        let mut arena = no_local_bounds();
        let id = NodeId {
            level: 1,
            pos: [0, 0, 0],
        };
        arena.insert(id, 0, [0; 3], TreeModel::Oak, Vec::new(), unit_aabb());
        assert_eq!(arena.node_count(), 0, "an empty insert recorded a node");
    }

    /// Issue #412: `trees.wgsl` reads `node_origins[node_index]` with no
    /// bounds check (a clamp there would hide the bug, decision 2), so the
    /// invariant that every stored instance indexes a slot the origins
    /// buffer actually has (`0..MAX_NODES`) is enforced on the CPU, at
    /// `insert`. The live path cannot produce such an index (the slot comes
    /// from the `ChunkArena`'s own allocator); this pins what happens when a
    /// standalone caller makes one up. `debug_assert`, so a debug build --
    /// every test and every CI run -- is what catches it.
    #[test]
    #[should_panic(expected = "past the node-origins buffer")]
    #[cfg(debug_assertions)]
    fn an_instance_may_not_reference_a_slot_the_origins_buffer_does_not_have() {
        let mut arena = no_local_bounds();
        let id = NodeId {
            level: 1,
            pos: [0, 0, 0],
        };
        arena.insert(
            id,
            crate::arena::MAX_NODES,
            [0; 3],
            TreeModel::Oak,
            vec![tree([0, 0, 0])],
            unit_aabb(),
        );
    }

    /// The companion to the test above: the last slot the arena can hand out
    /// (`MAX_NODES - 1`) is a valid index, not an off-by-one rejection.
    #[test]
    fn the_last_slot_the_arena_can_hand_out_is_a_valid_instance_index() {
        let mut arena = no_local_bounds();
        let id = NodeId {
            level: 1,
            pos: [0, 0, 0],
        };
        arena.insert(
            id,
            crate::arena::MAX_NODES - 1,
            [0; 3],
            TreeModel::Oak,
            vec![tree([0, 0, 0])],
            unit_aabb(),
        );
        assert_eq!(arena.node_count(), 1);
    }

    #[test]
    fn insert_and_remove_track_exactly_what_is_recorded() {
        let mut arena = no_local_bounds();
        let id = NodeId {
            level: 1,
            pos: [0, 0, 0],
        };
        arena.remove(id);
        assert_eq!(arena.node_count(), 0, "nothing was there to remove");
        arena.insert(
            id,
            0,
            [0; 3],
            TreeModel::Silhouette,
            vec![tree([1, 2, 3])],
            unit_aabb(),
        );
        assert_eq!(arena.node_count(), 1);
        arena.remove(id);
        assert_eq!(
            arena.node_count(),
            0,
            "removing a present node must drop it"
        );
    }

    /// `insert`'s own AABB math: the stored culling box is the union of the
    /// node's AABB and every tree's position grown by the tallest tree's
    /// reach -- where it stands and where its morph takes it -- not the
    /// node's AABB alone (issue #332 decision 2's follow-up). A CPU-only
    /// complement to `tree_culling.rs`'s real-frustum proof of the same
    /// thing.
    #[test]
    fn insert_grows_the_aabb_to_cover_every_tree() {
        let shapes = TreeShapes {
            heights: 4..=6,
            canopy_radius: 2,
        };
        let local_bounds = shapes.local_bounds();
        assert_eq!(
            local_bounds,
            (glam::vec3(-2.0, 0.0, -2.0), glam::vec3(3.0, 7.0, 3.0)),
            "the tallest oak's box, one past its highest block"
        );
        let mut arena = TreeArena::new(shapes, crate::anchor::ORIGIN);
        let id = NodeId {
            level: 1,
            pos: [0, 0, 0],
        };
        let node_aabb =
            crate::culling::Aabb::new(glam::vec3(0.0, 0.0, 0.0), glam::vec3(16.0, 4.0, 16.0));
        // A tree standing right at the node's edge, tall enough that its
        // canopy reaches well above the node's own (short) AABB, and one
        // the morph lowers below it.
        let tall = TreePlacement {
            base: [15, 4, 2],
            height: 6,
            morph: 3,
        };
        let sunk = TreePlacement {
            base: [1, 1, 1],
            height: 4,
            morph: -5,
        };
        arena.insert(id, 0, [0; 3], TreeModel::Oak, vec![tall, sunk], node_aabb);
        let stored = arena.aabb_of(id).expect("just inserted");
        // The canopy's top where the morph lifts it: 4 + 3 + 7 = 14.
        assert_eq!(stored.max.y, 14.0);
        assert_eq!(stored.max.x, 18.0, "the canopy past the node's edge");
        // The sunk tree's foot at full morph: 1 - 5 = -4.
        assert_eq!(stored.min.y, -4.0);
    }

    /// `TreeShapes::local_bounds`' own `max + 1` conversion (block-index to
    /// continuous extent), checked against `tree_bounds` directly at the
    /// tallest height.
    #[test]
    fn local_bounds_converts_block_indices_to_continuous_extents() {
        let shapes = TreeShapes {
            heights: 3..=5,
            canopy_radius: 2,
        };
        let (min, max) = shapes.local_bounds();
        let (expect_min, expect_max) = cubara_voxel::tree_bounds(5, 2);
        assert_eq!(
            min,
            glam::Vec3::new(
                expect_min[0] as f32,
                expect_min[1] as f32,
                expect_min[2] as f32
            ),
            "the min corner needs no adjustment -- a block's near corner is its own index"
        );
        assert_eq!(
            max,
            glam::Vec3::new(
                expect_max[0] as f32 + 1.0,
                expect_max[1] as f32 + 1.0,
                expect_max[2] as f32 + 1.0,
            ),
            "the max corner is one past the highest solid block's index on every axis"
        );
    }

    /// Issue #379 decision 2: an instance is stored relative to its node's
    /// origin, with the node's slot in `node_origins`, so `trees.wgsl` adds
    /// the origin back from the same buffer `mesh.wgsl` reads -- and a
    /// rebase of that buffer moves the trees with their node, touching no
    /// instance. Each carries its node's level and its own morph, and is
    /// filed under its height's model. The culling box is built from the
    /// node-local bounds and the same node-relative offsets, then placed
    /// against the anchor (the origin here, so it comes out in world blocks).
    #[test]
    fn instances_are_relative_to_their_node_and_carry_its_slot() {
        let mut arena = no_local_bounds();
        let id = NodeId {
            level: 2,
            pos: [3, 0, -1],
        };
        let origin = [192, 0, -64];
        let node_local_aabb =
            crate::culling::Aabb::new(glam::vec3(0.0, 0.0, 0.0), glam::vec3(64.0, 20.0, 64.0));
        arena.insert(
            id,
            17,
            origin,
            TreeModel::Oak,
            vec![
                TreePlacement {
                    base: [200, 12, -60],
                    height: 6,
                    morph: -2,
                },
                TreePlacement {
                    base: [255, 9, -1],
                    height: 4,
                    morph: 0,
                },
            ],
            node_local_aabb,
        );
        assert_eq!(
            arena.instances_of(id),
            Some(
                &[
                    (2, TreeInstance::new([8.0, 12.0, 4.0], 17, 2, -2)),
                    (0, TreeInstance::new([63.0, 9.0, 63.0], 17, 2, 0)),
                ][..]
            )
        );
        let stored = arena.aabb_of(id).expect("just inserted");
        assert_eq!(
            (stored.min, stored.max),
            (glam::vec3(192.0, 0.0, -64.0), glam::vec3(256.0, 20.0, 0.0)),
            "with no local reach, trees inside the node leave its box as it is, \
             placed at the node's origin"
        );
    }
}

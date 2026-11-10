//! Trees beyond the first ring (`docs/PROPOSAL_LOD_TRANSITIONS.md` §3 R2,
//! issue #332): the two static instance models, and the arena that tracks
//! which resident node contributed which instances.
//!
//! **Positions and meshes only** (`ARCHITECTURE.md` Rule 3) -- this module
//! never sees a `WorldGen` or a `NodeKey`; it reads
//! [`cubara_voxel::StructureRegistry`]/[`cubara_voxel::BlockRegistry`] (the
//! same lower layer `cubara-world` reads to resolve `TerrainBlocks`) only to
//! build the two *models*, once, at startup -- not to decide where a tree
//! stands. `cubara-app`'s streaming glue (`crates/app/src/streaming.rs`)
//! is the only caller that knows about `WorldGen::node_trees`; it hands this
//! module plain block positions.

use std::collections::BTreeMap;

use cubara_voxel::{AnisoChunk, BlockId, BlockRegistry, MeshContext, StructureRegistry};

use crate::arena::NodeId;

/// Which static model a node's trees draw with (issue #332 decision 3):
/// the real oak at levels 1-2, two boxes at level 3. One per resident node
/// -- a node is wholly one level, so its trees are wholly one model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreeModel {
    Oak,
    Box,
}

/// One tree-model vertex: a local position (the trunk's base block is
/// `(0, 0, 0)`), a face normal, this corner's own texel-tile coordinate (as
/// `mesh.wgsl`'s `u`/`v`), the texture-array layer to sample -- or a
/// negative value, meaning "untextured, use `color` instead" (the box
/// model's mean colours, decision 3) -- and that colour.
///
/// Unpacked floats rather than `cubara_voxel::Vertex`'s packed 12 bytes:
/// a tree model is a few hundred vertices built once at startup, not a
/// greedy-meshed arena of millions, so there is nothing to amortise a packed
/// format against, and a flat layout keeps this module independent of the
/// arena's bit-packing scheme.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TreeVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub layer: f32,
    pub color: [f32; 3],
}

/// One instance: where the model's local origin (the trunk's base block)
/// stands, **relative to its node's origin**, plus that node's slot in the
/// `node_origins` storage buffer (`crate::arena`) -- `trees.wgsl` adds the
/// two the way `mesh.wgsl` places a packed vertex.
///
/// Node-relative rather than absolute (issue #379 decision 2, after review
/// finding F16): the render anchor moves every 2,048 blocks of travel, and
/// when it does, every absolute position the GPU holds has to be rewritten.
/// With the instances relative to their node, a rebase rewrites one `vec4`
/// per resident node (`ChunkArena::rebase`) and nothing per instance; this
/// buffer is only ever rewritten when the set of *contributing nodes*
/// changes (`TreeArena::prepare`), exactly as before.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TreeInstance {
    pub position: [f32; 3],
    pub node_index: u32,
}

/// A model's CPU-side triangle data, before it goes to the GPU.
#[derive(Default)]
pub struct TreeModelMesh {
    pub vertices: Vec<TreeVertex>,
    pub indices: Vec<u32>,
}

fn push_quad(mesh: &mut TreeModelMesh, corners: [[f32; 3]; 4], normal: [f32; 3], color: [f32; 3]) {
    let base = mesh.vertices.len() as u32;
    for position in corners {
        mesh.vertices.push(TreeVertex {
            position,
            normal,
            uv: [0.0, 0.0],
            layer: -1.0,
            color,
        });
    }
    mesh.indices
        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

/// An axis-aligned box from `min` to `max`, as 6 quads (one flat colour, no
/// texture) -- CCW from outside on every face, matching the mesh pipeline's
/// `front_face: Ccw` / `cull_mode: Back` (checked by cross product in this
/// module's tests).
fn push_box(mesh: &mut TreeModelMesh, min: [f32; 3], max: [f32; 3], color: [f32; 3]) {
    let ([x0, y0, z0], [x1, y1, z1]) = (min, max);
    push_quad(
        mesh,
        [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]],
        [1.0, 0.0, 0.0],
        color,
    );
    push_quad(
        mesh,
        [[x0, y0, z1], [x0, y1, z1], [x0, y1, z0], [x0, y0, z0]],
        [-1.0, 0.0, 0.0],
        color,
    );
    push_quad(
        mesh,
        [[x0, y1, z0], [x0, y1, z1], [x1, y1, z1], [x1, y1, z0]],
        [0.0, 1.0, 0.0],
        color,
    );
    push_quad(
        mesh,
        [[x0, y0, z1], [x0, y0, z0], [x1, y0, z0], [x1, y0, z1]],
        [0.0, -1.0, 0.0],
        color,
    );
    push_quad(
        mesh,
        [[x1, y0, z1], [x1, y1, z1], [x0, y1, z1], [x0, y0, z1]],
        [0.0, 0.0, 1.0],
        color,
    );
    push_quad(
        mesh,
        [[x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [x1, y0, z0]],
        [0.0, 0.0, -1.0],
        color,
    );
}

/// The real oak, meshed once from `assets/structures/oak.ron` with the same
/// voxel mesher a level-0 tree is built with (decision 3): local positions
/// are block coordinates relative to the trunk's base block, so an instance
/// at a tree site's `(x, base, z)` lands exactly where the real voxel trunk
/// would. `None` if the structure or either of its blocks is missing --
/// same tolerance [`cubara_world::TerrainBlocks::with_oak`] has, for a world
/// with no tree data yet.
///
/// A representative trunk height (the midpoint of `oak.ron`'s `(lo, hi)`
/// range) rather than the real per-tree height: decision 3 asks for the oak
/// "meshed once", one draw per model, so every instance shares one mesh
/// regardless of its real site's rolled height. The difference is at most
/// one block either way of the mean -- smaller than the canopy's own radius
/// -- and is the explicit cost of one draw instead of a draw per height.
pub fn build_oak_mesh(
    registry: &BlockRegistry,
    structures: &StructureRegistry,
    layer_of: &dyn Fn(&str) -> u32,
) -> Option<TreeModelMesh> {
    let def = structures.get("cubara:oak")?;
    let log = registry.id_of(&def.trunk.block)?;
    let leaves = registry.id_of(&def.canopy.block)?;
    let grows_on_exists = registry.id_of(&def.grows_on).is_some();
    if !grows_on_exists {
        return None;
    }
    let (lo, hi) = def.trunk.height;
    let height = (lo + hi) / 2;
    let r = def.canopy.radius;
    let (min, max) = cubara_voxel::tree_bounds(height, r);
    // No test pins these three `+ 1`s individually (`check-tests-can-fail.sh`
    // found the middle one survive a `+1` -> `+2` mutation): `tree_part_at`
    // only ever returns `Some` inside `tree_bounds`' own box, so a lattice
    // sized *larger* than necessary samples extra all-air cells at the
    // excess rows/columns -- `AnisoChunk::build_mesh` emits no faces for
    // those, so the model's actual triangles (which the budget test above
    // does check) are unaffected either way. A lattice sized *smaller* than
    // `tree_bounds` would be a real bug (a clipped canopy), and that shape
    // of mistake is not available to this particular mutation operator (it
    // only tried growing the dimension), which is why survival here is
    // informative about the mutator's own coverage, not about this line.
    let dims = [
        (max[0] - min[0] + 1) as usize,
        (max[1] - min[1] + 1) as usize,
        (max[2] - min[2] + 1) as usize,
    ];
    let chunk = AnisoChunk::from_fn(dims, [1, 1, 1], |x, y, z| {
        let dx = x as i32 + min[0];
        let dz = z as i32 + min[2];
        let dy = y as i32 + min[1];
        match cubara_voxel::tree_part_at(dx.abs(), dz.abs(), dy, height, r) {
            Some(cubara_voxel::TreePart::Trunk) => log,
            Some(cubara_voxel::TreePart::Canopy) => leaves,
            None => BlockId::AIR,
        }
    });
    let ctx = MeshContext { registry, layer_of };
    let mesh = chunk.build_mesh(&ctx);
    let vertices = mesh
        .vertices
        .iter()
        .map(|v| TreeVertex {
            position: [
                v.x() as f32 + min[0] as f32,
                v.y() as f32 + min[1] as f32,
                v.z() as f32 + min[2] as f32,
            ],
            normal: v.face().normal(),
            uv: [v.u() as f32, v.v() as f32],
            layer: v.tex_layer() as f32,
            color: [1.0, 1.0, 1.0],
        })
        .collect();
    Some(TreeModelMesh {
        vertices,
        indices: mesh.indices,
    })
}

/// Two boxes (trunk, canopy) sized to the oak's own bounds, wearing the log
/// and leaf textures' mean colours (decision 3) -- `height`/`canopy_radius`
/// the same representative numbers [`build_oak_mesh`] uses, so the box model
/// is the same size the real one would be at that edge.
pub fn build_box_mesh(
    height: i32,
    canopy_radius: i32,
    trunk_color: [f32; 3],
    leaf_color: [f32; 3],
) -> TreeModelMesh {
    let top = height - 1;
    let r = canopy_radius as f32;
    let mut mesh = TreeModelMesh::default();
    push_box(
        &mut mesh,
        [0.0, 0.0, 0.0],
        [1.0, height as f32, 1.0],
        trunk_color,
    );
    push_box(
        &mut mesh,
        [-r, (top - canopy_radius) as f32, -r],
        [r + 1.0, (top + 2) as f32, r + 1.0],
        leaf_color,
    );
    mesh
}

/// `(height, canopy_radius)` for the box model -- `oak.ron`'s own numbers
/// when the structure loaded, or a plain fallback (never reached once
/// `assets/structures/oak.ron` ships, same tolerance every other oak-shaped
/// default in this codebase has for missing data).
pub fn box_shape(structures: &StructureRegistry) -> (i32, i32) {
    structures.get("cubara:oak").map_or((5, 2), |def| {
        (
            (def.trunk.height.0 + def.trunk.height.1) / 2,
            def.canopy.radius,
        )
    })
}

/// The local bounding box one tree instance occupies, relative to its own
/// position (the trunk's base block) -- in continuous world-space extents,
/// not `tree_bounds`' own block-*index* convention (its `max` names the
/// highest solid block's index; that block's own far corner, in continuous
/// space, is one past it, so the upper bound here is `tree_bounds`' `max`
/// plus one on every axis; the lower bound needs no such adjustment, since a
/// block's near corner *is* its index).
///
/// What [`TreeArena::insert`] grows a node's culling AABB by: a canopy
/// rising above the node's own terrain must not be culled while the ground
/// under it sits just outside the frustum (issue #332 decision 2's
/// follow-up -- a node's drawn geometry and its trees are not the same
/// shape, and culling by the first alone can cut off the second).
pub fn tree_instance_bounds(structures: &StructureRegistry) -> (glam::Vec3, glam::Vec3) {
    let (height, r) = box_shape(structures);
    let (min, max) = cubara_voxel::tree_bounds(height, r);
    (
        glam::Vec3::new(min[0] as f32, min[1] as f32, min[2] as f32),
        glam::Vec3::new(
            max[0] as f32 + 1.0,
            max[1] as f32 + 1.0,
            max[2] as f32 + 1.0,
        ),
    )
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

/// Both models' GPU buffers, built once when the renderer starts
/// ([`crate::SceneRenderer::new`]) -- `oak` is `None` only when
/// `assets/structures/oak.ron`'s data is missing, same tolerance the rest of
/// the oak-placement path has.
pub struct TreeModels {
    pub oak: Option<TreeModelGpu>,
    pub boxed: TreeModelGpu,
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
    let oak =
        build_oak_mesh(registry, structures, layer_of).map(|m| upload_model(device, queue, &m));
    let (height, r) = box_shape(structures);
    let trunk_color = crate::materials::mean_color("oak_log_side").unwrap_or([0.45, 0.30, 0.15]);
    let leaf_color = crate::materials::mean_color("oak_leaves").unwrap_or([0.20, 0.45, 0.15]);
    let boxed = upload_model(
        device,
        queue,
        &build_box_mesh(height, r, trunk_color, leaf_color),
    );
    TreeModels { oak, boxed }
}

/// Grow-on-demand instance buffer bookkeeping, shared by the oak and box
/// instance lists -- the same "grow and keep the new capacity" shape
/// `SceneRenderer`'s figure vertex buffer already uses.
struct InstanceBuffer {
    buffer: Option<wgpu::Buffer>,
    capacity: usize,
    count: u32,
}

impl InstanceBuffer {
    const fn new() -> Self {
        Self {
            buffer: None,
            capacity: 0,
            count: 0,
        }
    }

    fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, instances: &[TreeInstance]) {
        self.count = instances.len() as u32;
        if instances.is_empty() {
            return;
        }
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
        if self.buffer.is_none() || instances.len() > self.capacity {
            self.capacity = instances.len().next_power_of_two();
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
            bytemuck::cast_slice(instances),
        );
    }

    fn instances(&self) -> Option<(&wgpu::Buffer, u32)> {
        self.buffer
            .as_ref()
            .filter(|_| self.count > 0)
            .map(|b| (b, self.count))
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
    /// How far a tree reaches from its own position, in each direction --
    /// `(min, max)` offsets added to a node's own AABB, over every tree it
    /// contributes, in [`insert`](Self::insert). Fixed for the arena's whole
    /// lifetime (both static models are sized to the same real oak, decision
    /// 3), so this is set once at construction rather than looked up per
    /// insert.
    local_bounds: (glam::Vec3, glam::Vec3),
    oak: InstanceBuffer,
    boxed: InstanceBuffer,
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
    last_box_contributors: Vec<NodeId>,
}

impl TreeArena {
    pub fn new(local_bounds: (glam::Vec3, glam::Vec3)) -> Self {
        Self {
            by_node: BTreeMap::new(),
            local_bounds,
            oak: InstanceBuffer::new(),
            boxed: InstanceBuffer::new(),
            last_oak_contributors: Vec::new(),
            last_box_contributors: Vec::new(),
        }
    }

    /// Record (or replace) `id`'s trees: `trees` are block positions (the
    /// trunk's base block, in world blocks), stored relative to the node's
    /// `origin` -- an exact integer subtraction -- alongside `node_index`,
    /// the node's slot in the arena's `node_origins` buffer that
    /// `trees.wgsl` adds back ([`TreeInstance`]). `node_aabb` is the node's
    /// own (terrain) bounds; the AABB actually stored is `node_aabb` grown
    /// by [`local_bounds`](Self::local_bounds) around every tree, so a tall
    /// canopy standing near a node's edge is never culled while the node's
    /// own ground is still in view, or vice versa. An empty `trees` is the
    /// same as having none -- removed outright, so a node that grows no
    /// trees never occupies a slot [`prepare`](Self::prepare) has to visit
    /// for nothing.
    pub fn insert(
        &mut self,
        id: NodeId,
        node_index: u32,
        origin: [i32; 3],
        model: TreeModel,
        trees: Vec<[i32; 3]>,
        node_aabb: crate::culling::Aabb,
    ) {
        if trees.is_empty() {
            self.remove(id);
            return;
        }
        let (local_min, local_max) = self.local_bounds;
        let mut min = node_aabb.min;
        let mut max = node_aabb.max;
        let instances = trees
            .iter()
            .map(|&t| {
                let p = glam::Vec3::new(t[0] as f32, t[1] as f32, t[2] as f32);
                min = min.min(p + local_min);
                max = max.max(p + local_max);
                TreeInstance {
                    position: [
                        (t[0] - origin[0]) as f32,
                        (t[1] - origin[1]) as f32,
                        (t[2] - origin[2]) as f32,
                    ],
                    node_index,
                }
            })
            .collect();
        self.by_node.insert(
            id,
            TreeNode {
                model,
                instances,
                aabb: crate::culling::Aabb::new(min, max),
            },
        );
    }

    pub fn remove(&mut self, id: NodeId) {
        self.by_node.remove(&id);
    }

    /// Forget every node at once -- the streamed world was just replaced
    /// ([`crate::Renderer::clear_nodes`]). Also forgets what contributed
    /// last frame: the next [`prepare`](Self::prepare) must not compare the
    /// new (empty) world against the old one's memory and conclude nothing
    /// changed.
    pub fn clear(&mut self) {
        self.by_node.clear();
        self.last_oak_contributors.clear();
        self.last_box_contributors.clear();
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
        let mut box_contributors = Vec::new();
        for (&id, node) in &self.by_node {
            if !frustum.intersects_aabb(&node.aabb) {
                continue;
            }
            match node.model {
                TreeModel::Oak => oak_contributors.push(id),
                TreeModel::Box => box_contributors.push(id),
            }
        }
        if oak_contributors != self.last_oak_contributors {
            let instances = flatten(&self.by_node, &oak_contributors);
            self.oak.upload(device, queue, &instances);
            self.last_oak_contributors = oak_contributors;
        }
        if box_contributors != self.last_box_contributors {
            let instances = flatten(&self.by_node, &box_contributors);
            self.boxed.upload(device, queue, &instances);
            self.last_box_contributors = box_contributors;
        }
    }

    pub fn oak_instances(&self) -> Option<(&wgpu::Buffer, u32)> {
        self.oak.instances()
    }

    pub fn box_instances(&self) -> Option<(&wgpu::Buffer, u32)> {
        self.boxed.instances()
    }

    /// How many nodes are currently recorded, resident or not -- a white-box
    /// accessor for this module's own (GPU-free) unit tests, which can check
    /// `insert`/`remove`'s bookkeeping without a device to call
    /// [`prepare`](Self::prepare) through. The frustum-culling behaviour
    /// itself is proven in `tests/tree_culling.rs`, which does have one.
    #[cfg(test)]
    fn node_count(&self) -> usize {
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

    /// The node-relative instances stored for `id` -- white-box, for the
    /// unit test that pins the subtraction [`insert`](Self::insert) does.
    #[cfg(test)]
    fn instances_of(&self, id: NodeId) -> Option<&[TreeInstance]> {
        self.by_node.get(&id).map(|node| node.instances.as_slice())
    }
}

/// What one resident node contributed: its model, its instances (already
/// node-relative, [`TreeInstance`]) and the culling box enclosing both the
/// node's terrain and every one of its trees.
struct TreeNode {
    model: TreeModel,
    instances: Vec<TreeInstance>,
    aabb: crate::culling::Aabb,
}

/// The instances every node in `ids` contributed, in order -- what
/// [`TreeArena::prepare`] uploads for one model once it has decided the set
/// of contributors actually changed.
fn flatten(by_node: &BTreeMap<NodeId, TreeNode>, ids: &[NodeId]) -> Vec<TreeInstance> {
    let mut out = Vec::new();
    for id in ids {
        out.extend_from_slice(&by_node[id].instances);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every face of [`push_box`] must wind so its cross product points the
    /// same way as the normal it was pushed with, or the mesh pipeline's
    /// back-face culling (`front_face: Ccw`) would drop exactly the faces a
    /// viewer outside the box needs to see.
    #[test]
    fn every_box_face_winds_outward() {
        let mut mesh = TreeModelMesh::default();
        push_box(&mut mesh, [0.0, 0.0, 0.0], [2.0, 3.0, 4.0], [1.0, 1.0, 1.0]);
        assert_eq!(mesh.vertices.len(), 24, "6 faces of 4 vertices");
        for quad in mesh.indices.chunks(6) {
            let p0 = glam::Vec3::from(mesh.vertices[quad[0] as usize].position);
            let p1 = glam::Vec3::from(mesh.vertices[quad[1] as usize].position);
            let p2 = glam::Vec3::from(mesh.vertices[quad[2] as usize].position);
            let normal = glam::Vec3::from(mesh.vertices[quad[0] as usize].normal);
            let cross = (p1 - p0).cross(p2 - p0);
            assert!(
                cross.normalize().dot(normal) > 0.99,
                "face at {p0:?} wound inward: cross {cross:?} vs normal {normal:?}"
            );
        }
    }

    #[test]
    fn a_box_model_is_two_boxes_24_triangles() {
        let mesh = build_box_mesh(5, 2, [0.4, 0.3, 0.1], [0.2, 0.5, 0.2]);
        assert_eq!(
            mesh.indices.len() / 3,
            24,
            "2 boxes * 6 faces * 2 triangles"
        );
    }

    /// `build_box_mesh`'s own bounding box, sized to the real oak's
    /// (decision 3: "sized to the oak's bounds"), caught by checking actual
    /// vertex positions -- `a_box_model_is_two_boxes_24_triangles` only
    /// counts triangles, which an off-by-one in `top` would not change at
    /// all (confirmed: `check-tests-can-fail.sh` found exactly that mutation
    /// -- `height - 1` to `height - 2` -- survived until this test existed).
    ///
    /// `tree_bounds` names block *indices* (the highest solid block is at
    /// `max`); a block's own far corner in continuous space is one past its
    /// index, so the mesh's outer corner is `max + 1` on every axis, not
    /// `max` itself -- the minimum corner needs no such adjustment, since a
    /// block's near corner *is* its index.
    #[test]
    fn the_box_model_matches_tree_bounds() {
        let (height, r) = (5, 2);
        let mesh = build_box_mesh(height, r, [0.4, 0.3, 0.1], [0.2, 0.5, 0.2]);
        let (expect_min, expect_max) = cubara_voxel::tree_bounds(height, r);
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for v in &mesh.vertices {
            for a in 0..3 {
                min[a] = min[a].min(v.position[a]);
                max[a] = max[a].max(v.position[a]);
            }
        }
        for a in 0..3 {
            assert_eq!(
                min[a], expect_min[a] as f32,
                "axis {a}: min is off (an off-by-one in `top` moves this)"
            );
            assert_eq!(
                max[a],
                expect_max[a] as f32 + 1.0,
                "axis {a}: max is off (an off-by-one in `top` moves this)"
            );
        }
    }

    /// No reach beyond the node's own AABB -- for tests that only care about
    /// `insert`/`remove` bookkeeping, not the AABB union itself.
    fn no_local_bounds() -> TreeArena {
        TreeArena::new((glam::Vec3::ZERO, glam::Vec3::ZERO))
    }

    #[test]
    fn tree_arena_starts_with_no_instances() {
        let arena = no_local_bounds();
        assert!(arena.oak_instances().is_none());
        assert!(arena.box_instances().is_none());
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

    #[test]
    fn insert_and_remove_track_exactly_what_is_recorded() {
        let mut arena = no_local_bounds();
        let id = NodeId {
            level: 1,
            pos: [0, 0, 0],
        };
        arena.remove(id);
        assert_eq!(arena.node_count(), 0, "nothing was there to remove");
        arena.insert(id, 0, [0; 3], TreeModel::Box, vec![[1, 2, 3]], unit_aabb());
        assert_eq!(arena.node_count(), 1);
        arena.remove(id);
        assert_eq!(
            arena.node_count(),
            0,
            "removing a present node must drop it"
        );
    }

    /// `insert`'s own AABB math: the stored culling box is the union of the
    /// node's AABB and every tree's position grown by `local_bounds`, not
    /// the node's AABB alone (issue #332 decision 2's follow-up). A CPU-only
    /// complement to `tree_culling.rs`'s real-frustum proof of the same
    /// thing.
    #[test]
    fn insert_grows_the_aabb_to_cover_every_tree() {
        let local_bounds = (glam::vec3(-2.0, 0.0, -2.0), glam::vec3(3.0, 10.0, 3.0));
        let mut arena = TreeArena::new(local_bounds);
        let id = NodeId {
            level: 1,
            pos: [0, 0, 0],
        };
        let node_aabb =
            crate::culling::Aabb::new(glam::vec3(0.0, 0.0, 0.0), glam::vec3(16.0, 4.0, 16.0));
        // A tree standing right at the node's edge, tall enough that its
        // canopy reaches well above the node's own (short) AABB.
        let tree = [15, 4, 2];
        arena.insert(id, 0, [0; 3], TreeModel::Oak, vec![tree], node_aabb);
        let stored = arena.aabb_of(id).expect("just inserted");
        let tree = glam::vec3(15.0, 4.0, 2.0);
        assert_eq!(
            stored.min,
            node_aabb.min.min(tree + local_bounds.0),
            "min must cover the tree's own low corner too"
        );
        assert_eq!(
            stored.max,
            node_aabb.max.max(tree + local_bounds.1),
            "max must cover the canopy, which reaches above the node's own AABB"
        );
        // Concretely: the canopy's top (tree.y + 10 = 14) is well above the
        // node's own top (4), so the union's top must be 14, not 4.
        assert_eq!(stored.max.y, 14.0);
    }

    /// Decision 3's own estimate ("about 200 triangles"): prints the real
    /// oak model's triangle count against the real `assets/` data, and
    /// fails loudly if it grows past a loose budget -- a guard against the
    /// model silently bloating (e.g. a representative height picked at the
    /// high end of the range, or a mesher regression that stops culling
    /// interior faces) rather than an exact pin.
    #[test]
    fn the_real_oak_model_stays_inside_its_triangle_budget() {
        let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let registry = cubara_voxel::BlockRegistry::load(&repo_root.join("assets/blocks"))
            .expect("assets/blocks must load");
        let structures =
            cubara_voxel::StructureRegistry::load(&repo_root.join("assets/structures"))
                .expect("assets/structures must load");
        let layers = crate::materials::TextureLayers::from_registry(&registry);
        let layer_of = |name: &str| layers.layer_of(name);
        let mesh = build_oak_mesh(&registry, &structures, &layer_of).expect("oak.ron ships");
        let triangles = mesh.indices.len() / 3;
        eprintln!(
            "OAK MODEL: {} vertices, {triangles} triangles",
            mesh.vertices.len()
        );
        assert!(
            triangles <= 260,
            "oak model grew to {triangles} triangles, over decision 3's ~200-triangle budget"
        );
    }

    /// `tree_instance_bounds`'s own `max + 1` conversion (block-index to
    /// continuous extent): caught `check-tests-can-fail.sh` finding `+ 1.0`
    /// -> `+ 2.0` survive, because `insert_grows_the_aabb_to_cover_every_tree`
    /// above supplies its own hand-picked `local_bounds` rather than calling
    /// this function at all. Checked against `tree_bounds` directly, with a
    /// fixed `(height, canopy_radius)` rather than the real asset data, so
    /// this doesn't drift if `oak.ron` is retuned.
    #[test]
    fn tree_instance_bounds_converts_block_indices_to_continuous_extents() {
        let structures = cubara_voxel::StructureRegistry::from_defs(vec![(
            std::path::PathBuf::from("test-fixture.ron"),
            cubara_voxel::StructureDef {
                name: "cubara:oak".to_string(),
                trunk: cubara_voxel::structure::TrunkDef {
                    block: "cubara:oak_log".to_string(),
                    height: (5, 5),
                },
                canopy: cubara_voxel::structure::CanopyDef {
                    block: "cubara:oak_leaves".to_string(),
                    radius: 2,
                },
                grows_on: "cubara:grass".to_string(),
                density: 12,
            },
        )])
        .expect("fixture structure is valid");
        let (min, max) = tree_instance_bounds(&structures);
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
    /// instance. The culling box, by contrast, stays in world blocks: it is
    /// what the frustum tests, and that box is in the same space as the
    /// node's own AABB.
    #[test]
    fn instances_are_relative_to_their_node_and_carry_its_slot() {
        let mut arena = no_local_bounds();
        let id = NodeId {
            level: 2,
            pos: [3, 0, -1],
        };
        let origin = [192, 0, -64];
        let node_aabb =
            crate::culling::Aabb::new(glam::vec3(192.0, 0.0, -64.0), glam::vec3(256.0, 20.0, 0.0));
        arena.insert(
            id,
            17,
            origin,
            TreeModel::Oak,
            vec![[200, 12, -60], [255, 9, -1]],
            node_aabb,
        );
        assert_eq!(
            arena.instances_of(id),
            Some(
                &[
                    TreeInstance {
                        position: [8.0, 12.0, 4.0],
                        node_index: 17,
                    },
                    TreeInstance {
                        position: [63.0, 9.0, 63.0],
                        node_index: 17,
                    },
                ][..]
            )
        );
        let stored = arena.aabb_of(id).expect("just inserted");
        assert_eq!(
            (stored.min, stored.max),
            (node_aabb.min, node_aabb.max),
            "with no local reach, trees inside the node leave its box as it is"
        );
    }
}

//! The anisotropic node lattice R1b introduces
//! (`docs/PROPOSAL_LOD_TRANSITIONS.md` §3 R1, decisions 1/2/3/6 of issue #327).
//!
//! A level-0 node is still a plain 16³ [`Chunk`] of 1-block cells. A coarse
//! (level > 0) node keeps its horizontal cells at `2^level` blocks wide, but
//! its *vertical* cells become `v` blocks tall (`v` = 1 at rings 1-2, `v` = 4
//! at ring 3) instead of `2^level` -- the core fix for the ring-edge riser
//! doubling the proposal diagnoses. The node's world footprint and lattice
//! *cell count* on x/z stay 16, exactly as today; only the y cell count and
//! each axis's own vertex-position scale change. [`AnisoChunk`] is that
//! lattice: a rectangular `[nx, ny, nz]` grid, palette-compressed exactly like
//! [`Chunk`] (same [`ChunkStorage`], which was already volume-agnostic), with
//! its own per-axis vertex scale baked in so a mesh built from it already
//! carries final node-local *block* positions (decision 3) rather than
//! lattice-cell ones -- the GPU-side `node_origins.w` scale this node is
//! placed with is then always 1.0.
//!
//! [`NodeChunk`] is the "one storage type" (decision 6) [`crate::World`]'s
//! `node_at` hands to every consumer that needs to read a node's contents --
//! meshing (`cubara_world::mesh`), border/visibility sampling
//! (`cubara_world::mesh::border_samples`), and cave-culling links
//! (`cubara_world::visibility::node_links`) -- so all three read the exact
//! same data for a given node, never a second "reference" copy of it
//! (`ARCHITECTURE.md` Rule 5). It is a thin sum of [`Chunk`] (level 0) and
//! [`AnisoChunk`] (level > 0), not a rewrite of either.

use crate::block::BlockId;
use crate::mesh::Mesh;
use crate::storage::ChunkStorage;
use crate::voxel::{greedy_mesh, Chunk, MeshContext};

/// A rectangular node lattice with independent per-axis cell counts and a
/// fixed per-axis vertex-position scale (R1b). Never used for level 0, which
/// stays a plain [`Chunk`] (decision 1/6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnisoChunk {
    /// Cell counts `[nx, ny, nz]`.
    dims: [usize; 3],
    /// Blocks per cell on each axis -- `[2^level, v, 2^level]` for a coarse
    /// node. Baked into every vertex position [`build_mesh_with_neighbours`]
    /// emits (decision 3), rather than left for the GPU's uniform
    /// `node_origins.w` multiply to apply, because that is a single scalar
    /// and cannot represent a different scale per axis.
    ///
    /// [`build_mesh_with_neighbours`]: Self::build_mesh_with_neighbours
    scale: [i32; 3],
    storage: ChunkStorage,
}

impl AnisoChunk {
    #[inline]
    fn volume(dims: [usize; 3]) -> usize {
        dims[0] * dims[1] * dims[2]
    }

    #[inline]
    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.dims[1] + y) * self.dims[0] + x
    }

    /// Build a lattice of `dims` cells, each `scale` blocks wide on its own
    /// axis, by asking `f` for each cell's id -- the anisotropic counterpart
    /// of [`Chunk::from_fn`], and built the same way (sample into a flat
    /// buffer, then choose `Uniform`/`Palette` in one pass) for the same
    /// reason: a per-cell `set` would visit the promote/repack state machine
    /// every cell, and `f` here is worldgen.
    pub fn from_fn(
        dims: [usize; 3],
        scale: [i32; 3],
        mut f: impl FnMut(usize, usize, usize) -> BlockId,
    ) -> Self {
        let mut ids = Vec::with_capacity(Self::volume(dims));
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    ids.push(f(x, y, z));
                }
            }
        }
        Self {
            dims,
            scale,
            storage: ChunkStorage::from_ids(&ids),
        }
    }

    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    pub fn scale(&self) -> [i32; 3] {
        self.scale
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> BlockId {
        self.storage.get(self.index(x, y, z))
    }

    /// Whether the lattice has no solid cells, same contract as
    /// [`Chunk::is_empty`].
    pub fn is_empty(&self) -> bool {
        match &self.storage {
            ChunkStorage::Uniform(id) => *id == BlockId::AIR,
            ChunkStorage::Palette { .. } => {
                (0..Self::volume(self.dims)).all(|idx| self.storage.get(idx) == BlockId::AIR)
            }
        }
    }

    /// [`Chunk::block_at`]'s bounds handling, over this lattice's own `dims`.
    fn block_at(&self, x: i32, y: i32, z: i32) -> BlockId {
        if x < 0 || y < 0 || z < 0 {
            return BlockId::AIR;
        }
        let (x, y, z) = (x as usize, y as usize, z as usize);
        if x >= self.dims[0] || y >= self.dims[1] || z >= self.dims[2] {
            return BlockId::AIR;
        }
        self.get(x, y, z)
    }

    /// [`Chunk::build_mesh_with_neighbours`], over this lattice's own `dims`
    /// and `scale` -- so the emitted vertex positions are already final
    /// node-local *block* coordinates (decision 3), not lattice-cell ones.
    pub fn build_mesh_with_neighbours(
        &self,
        ctx: &MeshContext,
        occluded: impl Fn(i32, i32, i32) -> bool,
        may_be_solid: impl Fn(i32, i32, i32) -> bool,
    ) -> Mesh {
        let dims = [
            self.dims[0] as i32,
            self.dims[1] as i32,
            self.dims[2] as i32,
        ];
        greedy_mesh(
            dims,
            self.scale,
            |x, y, z| self.block_at(x, y, z),
            occluded,
            may_be_solid,
            ctx,
        )
    }

    pub fn build_mesh_occluded(
        &self,
        ctx: &MeshContext,
        occluded: impl Fn(i32, i32, i32) -> bool,
    ) -> Mesh {
        self.build_mesh_with_neighbours(ctx, occluded, |_, _, _| true)
    }

    pub fn build_mesh(&self, ctx: &MeshContext) -> Mesh {
        self.build_mesh_occluded(ctx, |_, _, _| false)
    }
}

/// The one storage type `World::node_at` hands to meshing, border sampling
/// and visibility links alike (decision 6) -- a level-0 [`Chunk`] unchanged,
/// or a coarse node's [`AnisoChunk`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeChunk {
    Uniform(Chunk),
    Aniso(AnisoChunk),
}

impl NodeChunk {
    /// Cell counts `[nx, ny, nz]` -- `[Chunk::SIZE; 3]` for `Uniform`.
    pub fn dims(&self) -> [usize; 3] {
        match self {
            NodeChunk::Uniform(_) => [Chunk::SIZE; 3],
            NodeChunk::Aniso(a) => a.dims(),
        }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> BlockId {
        match self {
            NodeChunk::Uniform(c) => c.get(x, y, z),
            NodeChunk::Aniso(a) => a.get(x, y, z),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            NodeChunk::Uniform(c) => c.is_empty(),
            NodeChunk::Aniso(a) => a.is_empty(),
        }
    }

    /// The mesh for this node, with border faces/skirts resolved the same
    /// way [`Chunk::build_mesh_with_neighbours`] does -- already in final
    /// node-local block positions for the `Aniso` case (its own baked
    /// `scale`), and in 0..16 lattice-local positions (equivalently, block
    /// positions at a node-local scale of 1) for `Uniform`.
    pub fn build_mesh_with_neighbours(
        &self,
        ctx: &MeshContext,
        occluded: impl Fn(i32, i32, i32) -> bool,
        may_be_solid: impl Fn(i32, i32, i32) -> bool,
    ) -> Mesh {
        match self {
            NodeChunk::Uniform(c) => c.build_mesh_with_neighbours(ctx, occluded, may_be_solid),
            NodeChunk::Aniso(a) => a.build_mesh_with_neighbours(ctx, occluded, may_be_solid),
        }
    }

    /// [`build_mesh_with_neighbours`](Self::build_mesh_with_neighbours), on
    /// its own with no neighbour (every border face drawn, no skirts) --
    /// [`Chunk::build_mesh`]'s contract, for a caller that has a `NodeChunk`
    /// rather than a concrete `Chunk` (e.g. a debug census tool that means to
    /// mesh a node without border covering).
    pub fn build_mesh(&self, ctx: &MeshContext) -> Mesh {
        self.build_mesh_with_neighbours(ctx, |_, _, _| false, |_, _, _| true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{DropRule, Faces, Interact, Material, Shape};

    fn registry() -> crate::registry::BlockRegistry {
        crate::registry::BlockRegistry::from_materials(vec![(
            std::path::PathBuf::from("test-fixture.ron"),
            Material {
                name: "cubara:stone".to_string(),
                solid: true,
                faces: Faces::All("stone".to_string()),
                shapes: vec![Shape::Full],
                drops: DropRule::SameName,
                requires_tier: 0,
                hardness: Some(1),
                interact: Interact::None,
            },
        )])
        .expect("fixture registry is valid")
    }

    fn zero_layer(_: &str) -> u32 {
        0
    }

    #[test]
    fn an_aniso_chunk_round_trips_get_at_every_corner() {
        let dims = [16usize, 32, 16];
        let chunk = AnisoChunk::from_fn(dims, [2, 1, 2], |x, y, z| {
            if (x, y, z) == (0, 0, 0) || (x, y, z) == (15, 31, 15) {
                BlockId::STONE
            } else {
                BlockId::AIR
            }
        });
        assert_eq!(chunk.get(0, 0, 0), BlockId::STONE);
        assert_eq!(chunk.get(15, 31, 15), BlockId::STONE);
        assert_eq!(chunk.get(1, 1, 1), BlockId::AIR);
        assert!(!chunk.is_empty());
    }

    #[test]
    fn an_empty_aniso_chunk_is_empty() {
        let chunk = AnisoChunk::from_fn([16, 32, 16], [2, 1, 2], |_, _, _| BlockId::AIR);
        assert!(chunk.is_empty());
    }

    /// A single solid cell meshes into exactly six faces, each at the extent
    /// its own axis's baked scale implies -- the same proof R1a's
    /// `a_non_cubic_lattice_scales_each_axis_independently` gives `greedy_mesh`
    /// directly, now through `AnisoChunk`'s own wrapper.
    #[test]
    fn aniso_chunk_bakes_its_own_per_axis_scale_into_positions() {
        let registry = registry();
        let ctx = MeshContext {
            registry: &registry,
            layer_of: &zero_layer,
        };
        let dims = [4usize, 8, 4];
        let chunk = AnisoChunk::from_fn(dims, [2, 1, 2], |x, y, z| {
            if (x, y, z) == (1, 3, 1) {
                BlockId::STONE
            } else {
                BlockId::AIR
            }
        });
        let mesh = chunk.build_mesh(&ctx);
        assert_eq!(mesh.triangle_count(), 12, "6 unmerged faces, no skirts");
        let (mut min, mut max) = ([u32::MAX; 3], [0u32; 3]);
        for v in &mesh.vertices {
            for (k, c) in [v.x(), v.y(), v.z()].into_iter().enumerate() {
                min[k] = min[k].min(c);
                max[k] = max[k].max(c);
            }
        }
        assert_eq!((min, max), ([2, 3, 2], [4, 4, 4]));
    }

    #[test]
    fn node_chunk_dispatches_to_the_right_variant() {
        let registry = registry();
        let ctx = MeshContext {
            registry: &registry,
            layer_of: &zero_layer,
        };
        let mut level0 = Chunk::from_fn(|_, _, _| BlockId::AIR);
        level0.set(5, 5, 5, BlockId::STONE);
        let uniform = NodeChunk::Uniform(level0);
        assert_eq!(uniform.dims(), [16, 16, 16]);
        assert_eq!(uniform.get(5, 5, 5), BlockId::STONE);
        assert!(!uniform.is_empty());
        assert_eq!(
            uniform
                .build_mesh_with_neighbours(&ctx, |_, _, _| false, |_, _, _| true)
                .triangle_count(),
            12
        );

        let aniso = NodeChunk::Aniso(AnisoChunk::from_fn([16, 32, 16], [2, 1, 2], |x, y, z| {
            if (x, y, z) == (5, 5, 5) {
                BlockId::STONE
            } else {
                BlockId::AIR
            }
        }));
        assert_eq!(aniso.dims(), [16, 32, 16]);
        assert_eq!(aniso.get(5, 5, 5), BlockId::STONE);
        assert!(!aniso.is_empty());
        assert_eq!(
            aniso
                .build_mesh_with_neighbours(&ctx, |_, _, _| false, |_, _, _| true)
                .triangle_count(),
            12
        );
    }
}

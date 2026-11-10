//! Voxel data structures and CPU-side meshing.
//!
//! This crate is the lowest engine layer: a chunk of blocks ([`Chunk`]) and the
//! greedy mesher that turns it into an indexed triangle [`Mesh`] of [`Vertex`]es.
//! It knows nothing about the GPU beyond the vertex-buffer layout, so worldgen and
//! the renderer can share these types without depending on each other.

pub mod angle;
pub mod aniso;
pub mod block;
pub mod bounds;
pub mod connectivity;
pub mod coord;
pub mod fingerprint;
pub mod fixed;
pub mod item;
pub mod mesh;
pub mod ore;
pub mod recipe;
pub mod registry;
pub mod smelt;
pub mod storage;
pub mod structure;
pub mod tree_shape;
pub mod voxel;

pub use angle::Angle;
pub use aniso::{AnisoChunk, NodeChunk};
pub use block::BlockId;
pub use bounds::{
    build_mesh_bounded, build_mesh_bounded_occluded, build_mesh_bounded_occluded_node, Aabb,
};
pub use connectivity::FaceLinks;
pub use coord::ChunkCoord;
pub use fixed::{Fixed, FixedVec3, FRAC_BITS};
pub use item::{
    ItemDef, ItemId, ItemRegistry, ItemRegistryError, ItemStack, ItemState, Rarity, StackError,
};
pub use mesh::{Face, Mesh, Vertex};
pub use ore::{OreDef, OreError, OreRegistry};
pub use recipe::{Recipe, RecipeBook, RecipeDef, RecipeError, RecipeOutputDef, MAX_GRID};
pub use registry::{
    BlockRegistry, DropRule, Faces, Interact, ItemDrop, Material, RegistryError, Shape,
};
pub use smelt::{SmeltBook, SmeltError, SmeltRecipe, SmeltRecipeDef};
pub use storage::{ChunkPayloadError, ChunkStorage};
pub use structure::{StructureDef, StructureError, StructureRegistry};
pub use tree_shape::{tree_bounds, tree_part_at, TreePart};
pub use voxel::{face_ao, Chunk, MeshContext};

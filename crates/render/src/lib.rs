//! GPU renderer.
//!
//! Owns the wgpu device/surface, the render pipeline, and per-frame draw submission,
//! plus view-frustum [`culling`] and optional CPU [`profiling`]. Consumes meshes from
//! [`cubara_voxel`] -- nothing else; this crate never depends on `cubara_world`
//! (`ARCHITECTURE.md` §1). What to stream in is entirely the caller's decision
//! ([`ChunkArena::from_meshed`] for a one-shot scene, [`Renderer::apply_node_updates`]
//! for the live, incremental case); the shared building blocks (pipeline, depth
//! view, camera uniform, the [`ChunkArena`]) are public so headless paths
//! (benchmark, screenshot) render exactly what the window does. Receives a
//! [`CameraPose`] to render from -- it owns no camera state of its own, no input, no
//! movement (`ARCHITECTURE.md` Rule 3; camera movement is `cubara-sim`'s job as of
//! block 1.6).

pub mod aggregate;
pub mod anchor;
mod arena;
pub mod crack;
pub mod culling;
pub mod far;
pub mod figure;
mod gpu_timer;
pub mod haze;
pub mod headless;
pub mod instance;
pub mod materials;
pub mod panel;
pub mod profiling;
mod render;
mod scene;
mod text;
mod trees;
mod uploads;

pub use anchor::Anchor;
pub use arena::{faces_facing, ArenaUsage, ChunkArena, MeshedNode, NodeId};
pub use culling::{Aabb, Frustum};
pub use far::{
    FarCulling, FarDir, FarFace, FarParams, FarPatch, FarSlot, FarTerrain, FarVertex,
    FAR_MAX_RISE, FAR_QUADS, FAR_TOP_RANGE,
};
pub use figure::{figure_vertices, FigureVertex, PlayerView};
pub use gpu_timer::{
    classify_sample, timestamp_indices, timing_proven, GpuSample, GpuTimer, TimestampRing,
};
pub use headless::{Frame, PanelLayout, Shot};
pub use instance::new_instance;
pub use materials::{load_icon, swatch_color, MeshAssets};
pub use panel::{InventoryPanel, PanelSlot, PanelSlotKind};
pub use profiling::Profiler;
pub use render::{
    camera_bind_group_layout, create_depth_view, gpu_driven_features, grab_cursor,
    load_mesh_assets, load_registry, load_structures, load_tree_models, CameraPose, CameraUniform,
    FrameUniform, Lighting, MorphParams, Presented, RebaseReport, Renderer,
};
pub use scene::HealthView;
pub use scene::{
    FurnaceGauges, GpuTimestamps, HotbarSlot, HotbarView, Hud, PanelView, SceneFrame,
    SceneRenderer, TreeDraw,
};
pub use text::TextRenderer;
pub use trees::{
    build_oak_mesh, build_silhouette_mesh, tree_model_meshes, TreeArena, TreeInstance, TreeModel,
    TreeModelMesh, TreePlacement, TreeShapes, TreeVertex,
};

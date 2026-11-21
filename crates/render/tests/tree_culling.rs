//! Issue #332 decision 2: "only drawn (frustum-visible) nodes contribute".
//!
//! `TreeArena::prepare` must drop a resident node's trees exactly the way
//! `ChunkArena::prepare` drops its geometry -- the same `Frustum::
//! intersects_aabb` test, against the same per-node AABB
//! (`ARCHITECTURE.md` Rule 5: one culling path, not two). An integration
//! test rather than a `trees.rs` unit test because it needs a real device:
//! the thing being proven is what ends up in the uploaded instance buffer,
//! which `trees.rs`'s own (GPU-free) unit tests cannot read back.

use cubara_render::{
    Aabb, CameraUniform, Frustum, NodeId, TreeArena, TreeModel, TreePlacement, TreeShapes,
};

/// Serialises instance/device creation and teardown across this binary's
/// test threads (issues #265 and #412; see the module's own doc comment).
#[path = "common/gpu_lifecycle.rs"]
mod gpu_lifecycle;

/// A headless device, or `None` on a CI runner with no GPU adapter -- the
/// same convention `mesh_arena_integration.rs`'s `test_device` uses.
fn test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = cubara_render::new_instance();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .ok()?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("cubara-test-tree-culling-device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .ok()
}

/// A camera at the origin looking down `+x`, far enough out to frame nodes a
/// few hundred blocks ahead.
fn forward_frustum() -> Frustum {
    let vp = CameraUniform::look_view_proj(16.0 / 9.0, glam::Vec3::ZERO, glam::Vec3::X);
    Frustum::from_view_proj(vp)
}

/// The oak's shapes -- for tests that only care about the frustum re-test
/// itself, not the AABB union (issue #332 decision 2's follow-up, its own
/// test below): a tree's few blocks of reach are nothing against the
/// hundred-block margins these nodes stand in or out of the frustum by.
fn oak_shapes() -> TreeShapes {
    TreeShapes {
        heights: 4..=6,
        canopy_radius: 2,
    }
}

/// A tree of the oak's middle height standing at `base`, unmorphed.
fn tree(base: [i32; 3]) -> TreePlacement {
    TreePlacement {
        base,
        height: 5,
        morph: 0,
    }
}

/// How many instances a kind draws, over every height; `None` with none.
fn drawn(instances: Option<(&wgpu::Buffer, &[std::ops::Range<u32>])>) -> Option<u32> {
    instances.map(|(_, ranges)| ranges.iter().map(|r| r.len() as u32).sum())
}

#[test]
fn only_nodes_in_the_frustum_contribute_instances() {
    // First, so it drops last -- after the device and queue it protects.
    let _gpu = gpu_lifecycle::lock();
    let Some((device, queue)) = test_device() else {
        cubara_render::headless::skip_or_require_gpu(
            "only_nodes_in_the_frustum_contribute_instances",
        );
        return;
    };
    let frustum = forward_frustum();

    let mut arena = TreeArena::new(oak_shapes(), cubara_render::anchor::ORIGIN);
    let ahead = NodeId {
        level: 1,
        pos: [0, 0, 0],
    };
    let behind = NodeId {
        level: 1,
        pos: [1, 0, 0],
    };
    // Ahead of the camera, inside the frustum.
    arena.insert(
        ahead,
        0,
        [0; 3],
        TreeModel::Oak,
        vec![tree([100, 0, 0])],
        Aabb::new(glam::vec3(90.0, -5.0, -5.0), glam::vec3(110.0, 5.0, 5.0)),
    );
    // Directly behind the camera -- resident, but nothing the camera can see.
    arena.insert(
        behind,
        1,
        [0; 3],
        TreeModel::Oak,
        vec![tree([-100, 0, 0])],
        Aabb::new(glam::vec3(-110.0, -5.0, -5.0), glam::vec3(-90.0, 5.0, 5.0)),
    );

    arena.prepare(&device, &queue, &frustum);
    let count = drawn(arena.oak_instances())
        .expect("the node ahead of the camera must contribute an instance");
    assert_eq!(
        count, 1,
        "a resident node behind the camera must not contribute an instance -- \
         only drawn (frustum-visible) nodes may (issue #332 decision 2)"
    );
}

#[test]
fn a_node_leaving_the_frustum_stops_contributing_without_being_removed() {
    // The same resident node, never touched by `insert`/`remove` again,
    // simply stops drawing once the camera turns away from it -- proof this
    // is a per-frame re-test against the frustum, not a one-time filter
    // baked in at `insert` time.
    // First, so it drops last -- after the device and queue it protects.
    let _gpu = gpu_lifecycle::lock();
    let Some((device, queue)) = test_device() else {
        cubara_render::headless::skip_or_require_gpu(
            "a_node_leaving_the_frustum_stops_contributing_without_being_removed",
        );
        return;
    };
    let mut arena = TreeArena::new(oak_shapes(), cubara_render::anchor::ORIGIN);
    let node = NodeId {
        level: 1,
        pos: [0, 0, 0],
    };
    arena.insert(
        node,
        0,
        [0; 3],
        TreeModel::Silhouette,
        vec![tree([100, 0, 0])],
        Aabb::new(glam::vec3(90.0, -5.0, -5.0), glam::vec3(110.0, 5.0, 5.0)),
    );

    arena.prepare(&device, &queue, &forward_frustum());
    assert_eq!(drawn(arena.silhouette_instances()), Some(1));

    // Look the other way: the same node, same AABB, same `by_node` entry --
    // only the frustum changed.
    let away = Frustum::from_view_proj(CameraUniform::look_view_proj(
        16.0 / 9.0,
        glam::Vec3::ZERO,
        -glam::Vec3::X,
    ));
    arena.prepare(&device, &queue, &away);
    assert_eq!(
        drawn(arena.silhouette_instances()),
        None,
        "turning away must stop drawing the node's trees, with no insert/remove call"
    );
}

/// Issue #332 decision 2's follow-up: the culling box must enclose the
/// trees, not just the node's own (terrain) AABB. A node whose ground sits
/// well below the frustum, but whose one tree's canopy reaches far enough
/// up to be inside it, must still contribute -- `TreeArena::insert` grows
/// the stored AABB by `local_bounds` around every tree for exactly this
/// case (a canopy rising above a node's own drawn top near the screen's top
/// edge, where the node's terrain AABB alone can already have left the
/// frustum while the leaves have not).
#[test]
fn a_canopy_above_the_frustum_line_still_contributes_even_though_the_ground_does_not() {
    // First, so it drops last -- after the device and queue it protects.
    let _gpu = gpu_lifecycle::lock();
    let Some((device, queue)) = test_device() else {
        cubara_render::headless::skip_or_require_gpu(
            "a_canopy_above_the_frustum_line_still_contributes_even_though_the_ground_does_not",
        );
        return;
    };
    // Level camera at the origin looking down +x: at depth x the visible
    // vertical half-extent is x * tan(30 deg) (half of the fixed 60 deg
    // vertical FOV) -- about 58 at x = 100.
    let frustum = forward_frustum();

    // The node's own ground: a thin slab far below that line (top y = -100,
    // against a ~58 boundary at this depth -- a wide, deliberate margin, not
    // a near-miss that floating point could flip).
    let node_aabb = Aabb::new(
        glam::vec3(95.0, -110.0, -5.0),
        glam::vec3(105.0, -100.0, 5.0),
    );
    // One tree standing right at the top of that ground, tall enough (a
    // trunk of 199 blocks, whose canopy's apex is the block at 199 and its
    // top face at 200: `TreeShapes::local_bounds`' upper bound) that its canopy's top
    // lands at y = -100 + 200 = 100 -- comfortably inside the frustum
    // vertically, against the same ~58 boundary.
    let shapes = TreeShapes {
        heights: 199..=199,
        canopy_radius: 3,
    };
    assert_eq!(shapes.local_bounds().1.y, 200.0);
    let tree_position = TreePlacement {
        base: [100, -100, 0],
        height: 199,
        morph: 0,
    };

    let mut arena = TreeArena::new(shapes, cubara_render::anchor::ORIGIN);
    let node = NodeId {
        level: 1,
        pos: [0, 0, 0],
    };
    arena.insert(
        node,
        0,
        [0; 3],
        TreeModel::Oak,
        vec![tree_position],
        node_aabb,
    );

    arena.prepare(&device, &queue, &frustum);
    assert_eq!(
        drawn(arena.oak_instances()),
        Some(1),
        "the canopy reaches into the frustum even though the node's own \
         ground is well below it -- the culling box must cover both"
    );
}

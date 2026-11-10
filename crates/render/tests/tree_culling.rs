//! Issue #332 decision 2: "only drawn (frustum-visible) nodes contribute".
//!
//! `TreeArena::prepare` must drop a resident node's trees exactly the way
//! `ChunkArena::prepare` drops its geometry -- the same `Frustum::
//! intersects_aabb` test, against the same per-node AABB
//! (`ARCHITECTURE.md` Rule 5: one culling path, not two). An integration
//! test rather than a `trees.rs` unit test because it needs a real device:
//! the thing being proven is what ends up in the uploaded instance buffer,
//! which `trees.rs`'s own (GPU-free) unit tests cannot read back.

use cubara_render::{Aabb, CameraUniform, Frustum, NodeId, TreeArena, TreeModel};

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

/// No reach beyond a node's own AABB -- for tests that only care about the
/// frustum re-test itself, not the AABB union (issue #332 decision 2's
/// follow-up, its own test below).
fn no_reach() -> (glam::Vec3, glam::Vec3) {
    (glam::Vec3::ZERO, glam::Vec3::ZERO)
}

#[test]
fn only_nodes_in_the_frustum_contribute_instances() {
    let Some((device, queue)) = test_device() else {
        eprintln!("SKIP only_nodes_in_the_frustum_contribute_instances: no GPU adapter");
        return;
    };
    let frustum = forward_frustum();

    let mut arena = TreeArena::new(no_reach());
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
        TreeModel::Oak,
        vec![[100.0, 0.0, 0.0]],
        Aabb::new(glam::vec3(90.0, -5.0, -5.0), glam::vec3(110.0, 5.0, 5.0)),
    );
    // Directly behind the camera -- resident, but nothing the camera can see.
    arena.insert(
        behind,
        TreeModel::Oak,
        vec![[-100.0, 0.0, 0.0]],
        Aabb::new(glam::vec3(-110.0, -5.0, -5.0), glam::vec3(-90.0, 5.0, 5.0)),
    );

    arena.prepare(&device, &queue, &frustum);
    let (_, count) = arena
        .oak_instances()
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
    let Some((device, queue)) = test_device() else {
        eprintln!(
            "SKIP a_node_leaving_the_frustum_stops_contributing_without_being_removed: no GPU adapter"
        );
        return;
    };
    let mut arena = TreeArena::new(no_reach());
    let node = NodeId {
        level: 1,
        pos: [0, 0, 0],
    };
    arena.insert(
        node,
        TreeModel::Box,
        vec![[100.0, 0.0, 0.0]],
        Aabb::new(glam::vec3(90.0, -5.0, -5.0), glam::vec3(110.0, 5.0, 5.0)),
    );

    arena.prepare(&device, &queue, &forward_frustum());
    assert_eq!(arena.box_instances().map(|(_, n)| n), Some(1));

    // Look the other way: the same node, same AABB, same `by_node` entry --
    // only the frustum changed.
    let away = Frustum::from_view_proj(CameraUniform::look_view_proj(
        16.0 / 9.0,
        glam::Vec3::ZERO,
        -glam::Vec3::X,
    ));
    arena.prepare(&device, &queue, &away);
    assert_eq!(
        arena.box_instances(),
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
    let Some((device, queue)) = test_device() else {
        eprintln!(
            "SKIP a_canopy_above_the_frustum_line_still_contributes_even_though_the_ground_does_not: \
             no GPU adapter"
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
    // One tree standing right at the top of that ground, with a reach tall
    // enough (local_bounds' upper bound) that its canopy's top lands at
    // y = -100 + 200 = 100 -- comfortably inside the frustum vertically,
    // against the same ~58 boundary.
    let local_bounds = (glam::vec3(-3.0, 0.0, -3.0), glam::vec3(3.0, 200.0, 3.0));
    let tree_position = [100.0, -100.0, 0.0];

    let mut arena = TreeArena::new(local_bounds);
    let node = NodeId {
        level: 1,
        pos: [0, 0, 0],
    };
    arena.insert(node, TreeModel::Oak, vec![tree_position], node_aabb);

    arena.prepare(&device, &queue, &frustum);
    assert_eq!(
        arena.oak_instances().map(|(_, n)| n),
        Some(1),
        "the canopy reaches into the frustum even though the node's own \
         ground is well below it -- the culling box must cover both"
    );
}

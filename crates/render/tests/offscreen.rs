//! The window's renderer with no window ([`Renderer::offscreen`]): what
//! `--bench flight` measures through, so it has to draw what it is handed and
//! stop drawing what it is told to drop, the way the window does.

use cubara_render::{CameraPose, Hud, MeshedNode, NodeId, Renderer};
use cubara_voxel::ChunkCoord;
use cubara_world::mesh::{mesh_node, mesh_region};
use cubara_world::node::{schedule_for_radius, NodeKey};
use cubara_world::{TerrainBlocks, World};

fn draw(renderer: &mut Renderer, camera: CameraPose) -> Vec<u8> {
    renderer.render(
        camera,
        None,
        None,
        &[],
        Hud {
            hotbar: None,
            panel: None,
            health: None,
            crosshair: false,
            menu: None,
        },
        0.0,
    );
    renderer
        .read_frame()
        .expect("an offscreen renderer reads its frame back")
        .pixels
}

fn id(node: NodeKey) -> NodeId {
    NodeId {
        level: node.level,
        pos: node.pos,
    }
}

#[test]
fn an_offscreen_renderer_draws_what_it_is_handed_and_nothing_it_dropped() {
    let world = World::new();
    let surface = world.surface_height(8, 8) as f32;
    let camera = CameraPose {
        eye: glam::DVec3::new(8.0, surface as f64 + 24.0, 8.0),
        look_dir: glam::Vec3::new(0.05, -1.0, 0.03),
    };
    let Some((mut renderer, assets)) = Renderer::offscreen(64, 48, camera) else {
        cubara_render::headless::skip_or_require_gpu(
            "an_offscreen_renderer_draws_what_it_is_handed_and_nothing_it_dropped",
        );
        return;
    };
    assert_eq!(renderer.size(), (64, 48));
    let layer_of = |name: &str| assets.layers.layer_of(name);
    let blocks = TerrainBlocks::from_registry(&assets.registry);
    let node = NodeKey::containing(ChunkCoord::from_world_pos([8.0, surface, 8.0]), 1);
    let geometry = mesh_node(&world, &assets.registry, &layer_of, node, blocks)
        .expect("the ground under spawn has something to draw");
    let meshed = MeshedNode {
        id: id(node),
        origin: geometry.origin,
        scale: geometry.scale,
        mesh: geometry.mesh,
        local_aabb: geometry.local_aabb,
        trees: Vec::new(),
        tree_model: cubara_render::TreeModel::Oak,
    };

    let sky = draw(&mut renderer, camera);
    renderer.apply_node_updates([], [meshed]);
    let ground = draw(&mut renderer, camera);
    assert_eq!(
        renderer.uploads_pending(),
        0,
        "one node waited past a frame"
    );
    let pixels = sky.len() / 4;
    let differ = sky
        .as_chunks::<4>()
        .0
        .iter()
        .zip(ground.as_chunks::<4>().0)
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        differ > pixels / 2,
        "the node handed over is not what the frame shows ({differ} of {pixels} pixels differ)"
    );

    renderer.apply_node_updates([id(node)], []);
    assert_eq!(
        draw(&mut renderer, camera),
        sky,
        "a dropped node is still drawn"
    );
}

/// Issue #379 through the window's own renderer: the camera jumps 2^20
/// blocks (a `/tp`), the world's one node arrives at the matching place,
/// and the frame is the same bytes as before the jump. That exercises what a
/// headless shot cannot: `Renderer::update` moving its anchor
/// (`anchor::rebase_if_needed`) *before* the frame's uploads, so the node
/// that lands this frame is placed against the anchor the frame draws with.
#[test]
fn a_teleport_across_the_anchor_grid_leaves_the_frame_unchanged() {
    const SHIFT: i32 = 1_048_576;
    let world = World::new();
    let surface = world.surface_height(8, 8);
    let camera = CameraPose {
        eye: glam::DVec3::new(8.3, surface as f64 + 24.0, 8.3),
        look_dir: glam::Vec3::new(0.05, -1.0, 0.03),
    };
    let Some((mut renderer, assets)) = Renderer::offscreen(64, 48, camera) else {
        cubara_render::headless::skip_or_require_gpu(
            "a_teleport_across_the_anchor_grid_leaves_the_frame_unchanged",
        );
        return;
    };
    let layer_of = |name: &str| assets.layers.layer_of(name);
    let blocks = TerrainBlocks::from_registry(&assets.registry);
    let node = NodeKey::containing(ChunkCoord::from_block(8, surface, 8), 1);
    let meshed = |shift: i32| {
        let geometry = mesh_node(&world, &assets.registry, &layer_of, node, blocks)
            .expect("the ground under spawn has something to draw");
        let mut id = id(node);
        // A level-1 node is 32 blocks a side.
        id.pos[0] += shift / 32;
        id.pos[2] += shift / 32;
        MeshedNode {
            id,
            origin: [
                geometry.origin[0] + shift,
                geometry.origin[1],
                geometry.origin[2] + shift,
            ],
            scale: geometry.scale,
            mesh: geometry.mesh,
            // Local to the node, so the shift does not touch it.
            local_aabb: geometry.local_aabb,
            trees: Vec::new(),
            tree_model: cubara_render::TreeModel::Oak,
        }
    };

    renderer.apply_node_updates([], [meshed(0)]);
    let here = draw(&mut renderer, camera);
    assert_eq!(renderer.anchor(), [0; 3]);

    // The jump: the node goes with it, in the same batch.
    let there_id = meshed(SHIFT).id;
    renderer.apply_node_updates([id(node)], [meshed(SHIFT)]);
    let jumped = CameraPose {
        eye: camera.eye + glam::DVec3::new(SHIFT as f64, 0.0, SHIFT as f64),
        ..camera
    };
    let there = draw(&mut renderer, jumped);
    assert_eq!(
        renderer.anchor(),
        [SHIFT, 0, SHIFT],
        "the renderer did not rebase on the jump"
    );
    assert_eq!(renderer.uploads_pending(), 0);
    let pixels = here.len() / 4;
    let differ = here
        .as_chunks::<4>()
        .0
        .iter()
        .zip(there.as_chunks::<4>().0)
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        differ <= 2,
        "the frame 2^20 blocks out is not the frame at spawn: {differ} of {pixels} pixels differ"
    );
    // And it really drew the moved node, not the sky: dropping it changes
    // the frame.
    renderer.apply_node_updates([there_id], []);
    let sky = draw(&mut renderer, jumped);
    assert_ne!(sky, there, "the moved node was not drawn at all");
}

/// **Issue #379 decision 2's done-when, as a measurement:** "the rebase
/// frame's own work is <= 1 ms". Not a pass/fail test -- a machine's clock
/// is not CI's -- but the reproducible way to get the number the design doc
/// quotes (`docs/PHASE3_ARCHITECTURE.md` §7):
///
/// ```text
/// cargo test -p cubara-render --release --test offscreen -- --ignored --nocapture rebase_cost_report
/// ```
///
/// The scene is gate-sized on purpose: every node of a radius-10 region
/// (level 0 through 2, each coarse node given a tree so the tree arena has
/// something to rebase too) and the far terrain filled to the window's
/// capacity, through the window's own renderer. The eye then jumps 2^20
/// blocks, the renderer rebases on that frame, and `Renderer::last_rebase`
/// says what the rebase itself cost -- the CPU side plus the `write_buffer`
/// calls, which is the work decision 2 bounds. Printed twice (there and back)
/// so a cold first call is visible as such.
#[test]
#[ignore = "a measurement, not a check: prints what a rebase costs with a gate-sized scene"]
fn rebase_cost_report() {
    use cubara_render::far::{FarParams, FarPatch};
    const SHIFT: i32 = 1_048_576;
    let world = World::new();
    let surface = world.surface_height(8, 8);
    let camera = CameraPose {
        eye: glam::DVec3::new(8.5, surface as f64 + 24.0, 8.5),
        look_dir: glam::Vec3::new(0.3, -0.4, -0.7),
    };
    let Some((mut renderer, assets)) = Renderer::offscreen(320, 180, camera) else {
        cubara_render::headless::skip_or_require_gpu("rebase_cost_report");
        return;
    };
    let layer_of = |name: &str| assets.layers.layer_of(name);
    let blocks = TerrainBlocks::from_registry(&assets.registry);
    // Radius 16: about as many nodes as the gate's ground eye has resident.
    let radius = 16;
    let nodes: Vec<MeshedNode> = mesh_region(
        &world,
        &assets.registry,
        &layer_of,
        ChunkCoord::new(0, 0, 0),
        0..=2,
        &schedule_for_radius(radius),
        blocks,
    )
    .into_iter()
    .filter_map(|built| {
        let geometry = built.geometry?;
        let origin = geometry.origin;
        // One tree per coarse node, on its own ground, so the tree arena's
        // boxes are part of what the rebase rewrites.
        let trees = if built.node.level >= 1 {
            vec![[origin[0] + 1, origin[1] + 1, origin[2] + 1]]
        } else {
            Vec::new()
        };
        Some(MeshedNode {
            id: id(built.node),
            origin,
            scale: geometry.scale,
            mesh: geometry.mesh,
            local_aabb: geometry.local_aabb,
            trees,
            tree_model: cubara_render::TreeModel::Oak,
        })
    })
    .collect();
    let node_count = nodes.len();
    let with_trees = nodes.iter().filter(|n| !n.trees.is_empty()).count();
    renderer.apply_node_updates([], nodes);
    // The far terrain to capacity: flat patches tiled outward.
    let flat = vec![surface as f32 - 4.0; cubara_render::FAR_VERTS * cubara_render::FAR_VERTS];
    let mut far_patches = 0usize;
    'far: for ring in 1..200i64 {
        for i in -ring..=ring {
            for j in -ring..=ring {
                if i.abs() != ring && j.abs() != ring {
                    continue;
                }
                let patch = FarPatch {
                    origin: [i * 128, j * 128],
                    quad: 4.0,
                    heights: &flat,
                    skirt: 8.0,
                };
                if renderer.far_insert(patch).is_none() {
                    break 'far;
                }
                far_patches += 1;
            }
        }
    }
    renderer.set_far_params(FarParams {
        hole: Some(([-160, 0, -160], [176, 48, 176])),
        materials: cubara_render::far::ground_materials(cubara_world::SOIL_DEPTH as f32),
    });
    // Let the paced upload queue drain, as the window would over frames.
    let mut frames = 0;
    while renderer.uploads_pending() > 0 {
        draw(&mut renderer, camera);
        frames += 1;
        assert!(frames < 100_000, "the upload queue never drained");
    }
    assert_eq!(renderer.anchor(), [0; 3]);

    // There and back: two rebases, each on its own frame.
    let jumped = CameraPose {
        eye: camera.eye + glam::DVec3::new(SHIFT as f64, 0.0, SHIFT as f64),
        ..camera
    };
    let mut reports = Vec::new();
    for pose in [jumped, camera] {
        let frame = std::time::Instant::now();
        draw(&mut renderer, pose);
        let whole_frame = frame.elapsed();
        let report = renderer
            .last_rebase()
            .expect("a jump of 2^20 blocks rebases on its frame");
        reports.push((report, whole_frame));
    }
    assert_eq!(reports[0].0.anchor, [SHIFT, 0, SHIFT]);
    assert_eq!(reports[1].0.anchor, [0; 3]);
    for (report, whole_frame) in &reports {
        println!(
            "REBASE on {}: anchor -> {:?}: {} nodes ({} meshed, {} with trees) + {} far \
             patches rewritten in {:.3} ms ({:.2} us per node); the whole offscreen frame \
             took {:.3} ms",
            renderer.adapter_name(),
            report.anchor,
            report.nodes,
            node_count,
            with_trees,
            report.far_patches,
            report.took.as_secs_f64() * 1000.0,
            report.took.as_secs_f64() * 1e6 / report.nodes.max(1) as f64,
            whole_frame.as_secs_f64() * 1000.0,
        );
    }
    let _ = far_patches;
}

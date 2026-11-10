//! The window's renderer with no window ([`Renderer::offscreen`]): what
//! `--bench flight` measures through, so it has to draw what it is handed and
//! stop drawing what it is told to drop, the way the window does.

use cubara_render::{CameraPose, Hud, MeshedNode, NodeId, Renderer};
use cubara_voxel::ChunkCoord;
use cubara_world::mesh::mesh_node;
use cubara_world::node::NodeKey;
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
        aabb: geometry.aabb,
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
        let s = shift as f32;
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
            aabb: cubara_render::Aabb::new(
                geometry.aabb.min + glam::vec3(s, 0.0, s),
                geometry.aabb.max + glam::vec3(s, 0.0, s),
            ),
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

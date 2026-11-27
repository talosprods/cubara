//! **No sky through the ground at the owner's spot**, rendered
//! (`docs/PHASE3_ARCHITECTURE.md` §9.12).
//!
//! The player-view tour (2026-11-27) stood at column (5376, 3408) -- the
//! owner's spot -- looked north and found a stone-topped block floating over
//! a gap of sky 346 blocks away, at pixels (1643..1666, 347..372) of a 2,560
//! x 1,664 frame; every one of those sky pixels hits solid ground at level
//! 0 (`World::raycast`). A level-2 cave, open through ground level 0 draws
//! (`crates/world/tests/coarse_cave_openings.rs` holds the rule, GPU-free).
//!
//! This renders what the game draws there -- its own node selection around
//! the standing eye, the game's mesher and renderer -- at the owner's two
//! frame sizes (the tour's 2,560 x 1,664 and his fullscreen 2,940 x 1,846),
//! and looks at every sky pixel around the block that has ground above it
//! in its screen column (sky *under* terrain): where level 0's ray through
//! it hits ground, the frame shows sky through the ground. None may be wider
//! than a sliver.
//!
//! **A sliver is not this class**: a vertical edge of a coarse cell leans a
//! pixel or two in perspective, and where the cell in front of it stands at
//! its mean, under level 0's columns, the sky shows under the lean -- one or
//! two pixels wide, as the tour found under many silhouettes. A cave a
//! level opens is a cell wide: at least ten pixels at the distances each
//! level is drawn at on these screens (a level-1 cell of 2 blocks at 288,
//! a level-2 cell of 4 at 512, a level-3 cell of 8 at 1,024). So a cluster
//! three pixels wide or more fails (chosen, not derived).
//!
//! Only the nodes of the game's selection that reach the ground around the
//! block are meshed (the full radius of 64 chunks is thousands of nodes),
//! with no distance morph (the block is outside every morph band, and the
//! tour's frames with and without it agree there) and no far terrain. A sky
//! pixel is only counted where level 0's ray hits ground inside that box,
//! so the land left out cannot be mistaken for a gap.

use std::path::Path;

use cubara_render::headless::{self, Shot};
use cubara_render::materials::TextureLayers;
use cubara_render::{MeshedNode, NodeId, TreeModel};
use cubara_voxel::{BlockRegistry, ChunkCoord};
use cubara_world::mesh::mesh_nodes;
use cubara_world::node::{desired_nodes_3d, schedule_for_radius};
use cubara_world::{TerrainBlocks, World};

#[path = "common/gpu_lifecycle.rs"]
mod gpu_lifecycle;

/// The game's render distance, in chunks (`render_radius_chunks`).
const RADIUS: i32 = 64;
/// The game's vertical squash of the rings -- mirrors
/// `cubara_app::streaming::VERTICAL_LOD_SQUASH`, which this test cannot
/// import (`ARCHITECTURE.md` Rule 3).
const SQUASH: i32 = 2;
/// The owner's spot: the column he stands on, and where he looks.
const STAND: [i32; 2] = [5376, 3408];
const LOOK: [f32; 3] = [0.0, -0.1, -1.0];
/// The block the tour found floating, and the box of ground around it whose
/// nodes are meshed: `[x, z]` from, to (exclusive).
const BLOCK: [f32; 3] = [5461.5, 138.5, 3072.5];
const BOX: ([i32; 2], [i32; 2]) = ([5376, 2944], [5568, 3136]);
/// The widest a cluster of sky through the ground may be, in pixels: a
/// sliver of perspective lean, never a cell (see the module comment).
const SLIVER: usize = 2;

fn real_registry() -> BlockRegistry {
    let assets_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/blocks");
    BlockRegistry::load(&assets_dir).expect("assets/blocks must be valid")
}

/// `cubara_app::flight::pixel_ray`, the renderer's projection (a 60-degree
/// vertical field of view, `Y` up), mirrored for the same reason as
/// [`SQUASH`].
fn pixel_ray(look: [f32; 3], (w, h): (u32, u32), x: u32, y: u32) -> glam::Vec3 {
    let forward = glam::Vec3::from(look).normalize();
    let right = forward.cross(glam::Vec3::Y).normalize();
    let up = right.cross(forward);
    let t = 30f32.to_radians().tan();
    let ndc_x = (x as f32 + 0.5) / w as f32 * 2.0 - 1.0;
    let ndc_y = 1.0 - (y as f32 + 0.5) / h as f32 * 2.0;
    (forward + right * (ndc_x * t * w as f32 / h as f32) + up * (ndc_y * t)).normalize()
}

/// The pixel a world point falls on: [`pixel_ray`] turned round.
fn project(eye: glam::Vec3, look: [f32; 3], (w, h): (u32, u32), p: glam::Vec3) -> (f32, f32) {
    let forward = glam::Vec3::from(look).normalize();
    let right = forward.cross(glam::Vec3::Y).normalize();
    let up = right.cross(forward);
    let t = 30f32.to_radians().tan();
    let d = p - eye;
    let depth = d.dot(forward);
    let ndc_x = d.dot(right) / (depth * t * w as f32 / h as f32);
    let ndc_y = d.dot(up) / (depth * t);
    (
        (ndc_x + 1.0) / 2.0 * w as f32 - 0.5,
        (1.0 - ndc_y) / 2.0 * h as f32 - 0.5,
    )
}

/// `cubara_app::streaming::to_meshed_node`, mirrored as `golden.rs` does --
/// from a borrowed node, since the scene is drawn at two sizes.
fn to_meshed_node(built: &cubara_world::mesh::BuiltNode) -> Option<MeshedNode> {
    let geometry = built.geometry.as_ref()?;
    Some(MeshedNode {
        id: NodeId {
            level: built.node.level,
            pos: built.node.pos,
        },
        origin: geometry.origin,
        scale: geometry.scale,
        mesh: cubara_voxel::Mesh {
            vertices: geometry.mesh.vertices.clone(),
            indices: geometry.mesh.indices.clone(),
            face_indices: geometry.mesh.face_indices,
        },
        local_aabb: geometry.local_aabb,
        trees: Vec::new(),
        tree_model: TreeModel::Oak,
    })
}

#[test]
fn no_sky_shows_through_the_ground_at_the_owners_spot() {
    let world = World::new();
    let registry = real_registry();
    let layers = TextureLayers::from_registry(&registry);
    let layer_of = |name: &str| layers.layer_of(name);
    // The terrain alone: a tree is not ground, and the box has none to hide.
    let blocks = TerrainBlocks::from_registry(&registry);
    // The standing eye: on the column's top block, at eye height.
    let top = world.column(STAND[0], STAND[1]).top;
    let eye = glam::Vec3::new(
        STAND[0] as f32 + 0.5,
        top as f32 + 1.0 + 1.62,
        STAND[1] as f32 + 0.5,
    );
    let centre = ChunkCoord::from_world_pos(eye.to_array());
    let ([x0, z0], [x1, z1]) = BOX;
    let nodes: Vec<_> = desired_nodes_3d(centre, SQUASH, &schedule_for_radius(RADIUS))
        .into_iter()
        .filter(|n| {
            let (o, size) = (n.world_origin(), 16 << n.level);
            o[0] < x1 && o[0] + size > x0 && o[2] < z1 && o[2] + size > z0
                // The hill's ground: from under the eye to over the summit.
                && o[1] < 192 && o[1] + size > 32
        })
        .collect();
    let levels: Vec<u32> = nodes.iter().map(|n| n.level).collect();
    assert!(
        levels.contains(&2),
        "the block's ground is drawn at level 2 here: {levels:?}"
    );
    let built = mesh_nodes(&world, &registry, &layer_of, nodes, blocks);
    let mut report = Vec::new();
    let mut failures = Vec::new();
    for size in [(2560, 1664), (2940, 1846)] {
        let shot = Shot {
            width: size.0,
            height: size.1,
            region_radius: RADIUS,
            camera: Some((eye.as_dvec3(), glam::Vec3::from(LOOK))),
            ..Shot::default()
        };
        let frame = {
            let _gpu = gpu_lifecycle::lock();
            headless::render(built.iter().filter_map(to_meshed_node), shot)
        };
        let Some(frame) = frame else {
            headless::skip_or_require_gpu("no_sky_shows_through_the_ground_at_the_owners_spot");
            return;
        };
        let (w, h) = (frame.width, frame.height);
        let px = |x: u32, y: u32| -> [u8; 3] {
            let i = ((y * w + x) * 4) as usize;
            [frame.pixels[i], frame.pixels[i + 1], frame.pixels[i + 2]]
        };
        // The clear colour: the top-left corner looks far over the land.
        let sky = px(0, 0);
        let is_sky = |x: u32, y: u32| px(x, y) == sky;
        // Around the block: its pixel, 120 a side at 1,664 rows.
        let (bx, by) = project(eye, LOOK, size, glam::Vec3::from(BLOCK));
        let half = (60.0 * h as f32 / 1664.0) as i32;
        let (cx, cy) = (bx as i32, by as i32);
        assert!(
            (half..w as i32 - half).contains(&cx) && (half..h as i32 - half).contains(&cy),
            "the block is at pixel ({bx:.0}, {by:.0}) of {w} x {h}"
        );
        // The first row with ground drawn, per column of the frame.
        let first_ground: Vec<u32> = (0..w)
            .map(|x| (0..h).find(|&y| !is_sky(x, y)).unwrap_or(h))
            .collect();
        let mut through = vec![false; (2 * half * 2 * half) as usize];
        let (mut under, mut ground_drawn) = (0usize, 0usize);
        for y in cy - half..cy + half {
            for x in cx - half..cx + half {
                let (ux, uy) = (x as u32, y as u32);
                if !is_sky(ux, uy) {
                    ground_drawn += 1;
                    continue;
                }
                // Sky under ground: something drawn above it in its column.
                if first_ground[ux as usize] >= uy {
                    continue;
                }
                under += 1;
                let dir = pixel_ray(LOOK, (w, h), ux, uy);
                let Some(hit) = world.raycast(eye.to_array(), dir.to_array(), 2_000.0, blocks)
                else {
                    continue;
                };
                let [hx, _, hz] = hit.block;
                if (x0..x1).contains(&hx) && (z0..z1).contains(&hz) {
                    through[((y - cy + half) * 2 * half + x - cx + half) as usize] = true;
                }
            }
        }
        // Clusters of sky through the ground, by their width.
        let side = (2 * half) as usize;
        let mut seen = vec![false; through.len()];
        let mut widest = 0usize;
        let mut count = 0usize;
        for start in 0..through.len() {
            if !through[start] || seen[start] {
                continue;
            }
            let (mut lo, mut hi, mut n) = (usize::MAX, 0usize, 0usize);
            let mut stack = vec![start];
            seen[start] = true;
            while let Some(k) = stack.pop() {
                let (x, y) = (k % side, k / side);
                (lo, hi, n) = (lo.min(x), hi.max(x), n + 1);
                for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if (0..side as i32).contains(&nx) && (0..side as i32).contains(&ny) {
                        let q = ny as usize * side + nx as usize;
                        if through[q] && !seen[q] {
                            seen[q] = true;
                            stack.push(q);
                        }
                    }
                }
            }
            count += n;
            let wide = hi - lo + 1;
            widest = widest.max(wide);
            if wide > SLIVER {
                failures.push(format!(
                    "{w} x {h}: {n} pixels of sky through level 0's ground, {wide} wide, at x {}..{} around ({cx}, {cy})",
                    cx - half + lo as i32,
                    cx - half + hi as i32
                ));
            }
        }
        report.push(format!(
            "{w} x {h}: block at ({bx:.0}, {by:.0}); {ground_drawn} pixels of ground, {under} of sky under ground, {count} through level 0's ground, widest {widest}"
        ));
        assert!(
            ground_drawn > (half * half) as usize,
            "{w} x {h}: only {ground_drawn} pixels of ground around the block"
        );
    }
    println!("{}", report.join("\n"));
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

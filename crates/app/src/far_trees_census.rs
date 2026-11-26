//! What trees on the far terrain would cost (issue #433 item 1): at the
//! gate's three eyes, along the bench's camera path, how many far trees are
//! in view and how tall each is on screen -- so the stand-in and the
//! distance its trees fade at are chosen from a count, not a guess.
//!
//! Measurement, not a check. Everything is the game's own: the bench's eyes
//! and camera path (`far_census`), the far selection (`far_view_scaled`,
//! `far::build`), the world's tree placement (`far_trees::patch_trees`) and
//! the renderer's `Frustum`.
//!
//! ```bash
//! cargo test -p cubara --bin cubara far_trees_census -- --ignored --nocapture
//! ```

use cubara_render::{Aabb, CameraUniform, Frustum};
use cubara_voxel::ChunkCoord;
use cubara_world::far::{self, PatchHeights};
use cubara_world::far_trees::{patch_trees, FarTree};
use cubara_world::{node, TerrainBlocks, World, WorldGen};
use glam::Vec3;

use crate::bench::{GateEye, GATE_EYES};
use crate::far_streaming::{far_view_scaled, FarQuality, FAR_QUAD_SCALE};

const SIZE: (u32, u32) = (1920, 1080);
const YAWS: usize = 40;

/// Pixels a block is tall at a distance of one block, at [`SIZE`]'s height
/// and the game's 60-degree vertical field of view.
fn px_per_block_at_one() -> f64 {
    SIZE.1 as f64 / 2.0 / (30f64.to_radians()).tan()
}

fn blocks() -> TerrainBlocks {
    let registry = cubara_render::load_registry();
    TerrainBlocks::from_registry(&registry)
        .with_oak(&crate::game::load_structure_registry(), &registry)
        .with_ores(&crate::game::load_ore_registry(), &registry)
}

fn select(gen: &WorldGen, eye: &GateEye, quality: FarQuality) -> (far::Hole, Vec<PatchHeights>) {
    let schedule = node::schedule_for_radius(crate::streaming::render_radius_chunks());
    let (min, max) = node::covered_box_3d(ChunkCoord::from_world_pos(eye.eye), &schedule);
    let hole = far::Hole {
        min: min.map(|c| (c * 16) as f64),
        max: max.map(|c| (c * 16) as f64),
    };
    let view = far_view_scaled(
        eye.eye.map(f64::from),
        SIZE.1,
        Some(hole),
        quality,
        FAR_QUAD_SCALE,
    )
    .expect("a quality that draws");
    (hole, far::build(gen, &view))
}

fn in_hole(hole: &far::Hole, t: &FarTree) -> bool {
    let p = [t.x as f64 + 0.5, t.base as f64, t.z as f64 + 0.5];
    (0..3).all(|a| hole.min[a] < p[a] && p[a] < hole.max[a])
}

/// Pixel-height buckets, from the top.
const BUCKETS: [f64; 9] = [8.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0, 0.0];

fn bucket(px: f64) -> usize {
    BUCKETS.iter().position(|&b| px >= b).unwrap_or(BUCKETS.len() - 1)
}

#[test]
#[ignore]
fn far_trees_census() {
    let gen = WorldGen::new(World::new().seed());
    let blocks = blocks();
    let k = px_per_block_at_one();
    let aspect = SIZE.0 as f32 / SIZE.1 as f32;
    println!("{}x{}, 60 deg; {YAWS} directions along the bench's path; Medium at x{FAR_QUAD_SCALE}", SIZE.0, SIZE.1);
    for quality in [FarQuality::Medium, FarQuality::High] {
        for eye in GATE_EYES.iter() {
            let (hole, patches) = select(&gen, eye, quality);
            let started = std::time::Instant::now();
            let mut by_level: std::collections::BTreeMap<u32, (usize, usize)> = Default::default();
            let mut trees: Vec<FarTree> = Vec::new();
            for p in &patches {
                let e = by_level.entry(p.key.level).or_default();
                e.0 += 1;
                if p.key.level > 8 {
                    continue;
                }
                let t = patch_trees(&gen, p.key, &p.tops, blocks);
                e.1 += t.len();
                trees.extend(t.into_iter().filter(|t| !in_hole(&hole, t)));
            }
            let gen_ms = started.elapsed().as_secs_f64() * 1000.0;
            let eye_v = Vec3::from(eye.eye);
            let mut seen = [0f64; BUCKETS.len()];
            for s in 0..YAWS {
                let frame = 200.0 + (s as f32 + 0.5) * 2000.0 / YAWS as f32;
                let yaw = frame / 240.0 * 0.3;
                let dir = glam::vec3(yaw.cos(), -0.25, yaw.sin());
                let frustum = Frustum::from_view_proj(CameraUniform::look_view_proj(aspect, eye_v, dir));
                for t in &trees {
                    let b = Aabb {
                        min: Vec3::new(t.x as f32 - 2.0, t.base as f32, t.z as f32 - 2.0),
                        max: Vec3::new(t.x as f32 + 3.0, (t.base + t.height + 1) as f32, t.z as f32 + 3.0),
                    };
                    if !frustum.intersects_aabb(&b) {
                        continue;
                    }
                    let mid = Vec3::new(t.x as f32 + 0.5, t.base as f32 + (t.height + 1) as f32 / 2.0, t.z as f32 + 0.5);
                    let d = mid.distance(eye_v) as f64;
                    let px = (t.height + 1) as f64 * k / d;
                    seen[bucket(px)] += 1.0;
                }
            }
            println!();
            println!("{} at {:?}, {}: {} patches; trees in patches of level <= 8 outside the hole: {} ({gen_ms:.0} ms to place)", eye.name, eye.eye, quality.name(), patches.len(), trees.len());
            for (l, (n, t)) in &by_level {
                if *l <= 8 {
                    println!("  level {l}: {n} patches, {t} trees ({:.0} a patch)", *t as f64 / (*n).max(1) as f64);
                } else {
                    println!("  level {l}: {n} patches");
                }
            }
            let mut cum = 0.0;
            println!("  | tree px tall | in view a frame | cumulative from the top |");
            for (i, n) in seen.iter().enumerate() {
                cum += n / YAWS as f64;
                let lo = BUCKETS[i];
                let hi = if i == 0 { f64::INFINITY } else { BUCKETS[i - 1] };
                println!("  | {lo}..{hi} | {:.0} | {cum:.0} |", n / YAWS as f64);
            }
        }
    }
}

//! What the far terrain costs: what `--bench gate` draws at its three eyes, at
//! every far quality and quad size, counted by what the GPU is bound by --
//! the vertices it shades (issue #422, `docs/PHASE3_ARCHITECTURE.md` §9.6).
//!
//! Measurement, not a check. Every patch in view is the same
//! [`FAR_PATCH_VERTICES`] vertices whatever the ground, so the census is the
//! number of patches the bench's own camera path sees: `far_view` (and so
//! the quality and quad scale), the bench's eyes, camera path and hole,
//! `CameraUniform`'s projection and the renderer's `Frustum`. It can be
//! checked against the bench's `far terrain drawn` line.
//!
//! ```bash
//! cargo test --release -p cubara --bin cubara far_census -- --ignored --nocapture
//! ```

use cubara_render::far::{issued, FAR_PATCH_VERTICES};
use cubara_render::{Aabb, CameraUniform, FarFace, Frustum};
use cubara_voxel::ChunkCoord;
use cubara_world::far::{self, PatchHeights, PATCH_QUADS};
use cubara_world::{node, World, WorldGen};
use glam::Vec3;

use crate::bench::{GateEye, GATE_EYES};
use crate::far_streaming::{far_view_scaled, to_far_faces, FarQuality};

const SIZE: (u32, u32) = (1920, 1080);

/// Camera directions sampled along the path the bench measures: frame
/// 200 to 2,200 at 1/240 virtual s a frame, a yaw of 0.3 rad per virtual s.
const YAWS: usize = 40;

/// The triangles one patch was drawn with while the far terrain was a smooth
/// surface: a 35 x 35 grid of vertices -- its 33 x 33 heights and the skirt
/// ring -- two triangles a cell, every patch alike.
const SMOOTH_TRIANGLES: f64 = 34.0 * 34.0 * 2.0;

/// What a gate eye's camera path sees, averaged over its frames.
#[derive(Default)]
struct Census {
    frames: f64,
    patches: f64,
    draws: f64,
    vertices: f64,
}

/// The far terrain's selection for `gate_eye` at `quality`, with every quad,
/// error bound and floor `coarser` times as wide on screen as the quality
/// sets them (1: the quality as the game draws it).
fn select_at(
    gen: &WorldGen,
    gate_eye: &GateEye,
    quality: FarQuality,
    coarser: f64,
) -> (far::Hole, Vec<PatchHeights>) {
    let eye = gate_eye.eye;
    let schedule = node::schedule_for_radius(crate::streaming::render_radius_chunks());
    let (min, max) = node::covered_box_3d(ChunkCoord::from_world_pos(eye), &schedule);
    let hole = far::Hole {
        min: min.map(|c| (c * 16) as f64),
        max: max.map(|c| (c * 16) as f64),
    };
    let view = far_view_scaled(eye.map(f64::from), SIZE.1, Some(hole), quality, coarser)
        .expect("a quality that draws the far terrain");
    (hole, far::build(gen, &view))
}

/// What the far terrain costs from `gate_eye`, over the bench's camera path.
fn census_of(gate_eye: &GateEye, hole: far::Hole, patches: &[PatchHeights]) -> Census {
    let eye = gate_eye.eye;
    let placement = |p: &PatchHeights| {
        let [ox, oz] = p.key.origin();
        (
            [ox as f32, oz as f32],
            p.key.quad() as f32,
            p.skirt_blocks() as f32,
        )
    };
    // The renderer's boxes: each patch's ground from the bottom of its
    // lowest wall to its highest top, against the origin, in f32.
    let boxes: Vec<Aabb> = patches
        .iter()
        .map(|p| {
            let (origin, quad, skirt) = placement(p);
            let faces: Vec<FarFace> = to_far_faces(p);
            let ys = faces.iter().flat_map(|f| f.corners()).map(|v| v.y(skirt));
            let (lo, hi) = ys.fold((f32::MAX, f32::MIN), |(l, h), y| (l.min(y), h.max(y)));
            let size = quad * PATCH_QUADS as f32;
            Aabb {
                min: Vec3::new(origin[0], lo, origin[1]),
                max: Vec3::new(origin[0] + size, hi, origin[1] + size),
            }
        })
        .collect();
    let hole_f32 = (
        Vec3::from(hole.min.map(|v| v as f32)),
        Vec3::from(hole.max.map(|v| v as f32)),
    );
    let in_hole = |b: &Aabb| hole_f32.0.cmple(b.min).all() && b.max.cmple(hole_f32.1).all();
    let eye_v = Vec3::from(eye);
    let aspect = SIZE.0 as f32 / SIZE.1 as f32;
    let mut census = Census::default();
    for s in 0..YAWS {
        let frame = 200.0 + (s as f32 + 0.5) * 2000.0 / YAWS as f32;
        let yaw = frame / 240.0 * 0.3;
        let dir = glam::vec3(yaw.cos(), -0.25, yaw.sin());
        let frustum = Frustum::from_view_proj(CameraUniform::look_view_proj(aspect, eye_v, dir));
        let in_view = boxes
            .iter()
            .filter(|b| frustum.intersects_aabb(b) && !in_hole(b))
            .count() as u32;
        let (draws, vertices) = issued(in_view);
        census.frames += 1.0;
        census.patches += in_view as f64;
        census.draws += draws as f64;
        census.vertices += vertices as f64;
    }
    census
}

/// `v` a frame, with thousands separated.
fn per_frame(v: f64, frames: f64) -> String {
    let s = ((v / frames).round() as u64).to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// What the far terrain costs at every gate eye and every quality: the
/// vertices issued a frame (what the pass is bound by), against what the
/// smooth surface it replaced drew.
#[test]
#[ignore]
fn far_census() {
    let gen = WorldGen::new(World::new().seed());
    println!(
        "{}x{}, 60 deg vertical; {YAWS} directions along the bench's path; {FAR_PATCH_VERTICES} \
         vertices a patch.",
        SIZE.0, SIZE.1
    );
    println!();
    println!("| eye | quality | patches | draws | vertices issued | smooth, vertices |");
    println!("|---|---|--:|--:|--:|--:|");
    for gate_eye in &GATE_EYES {
        for quality in [FarQuality::Low, FarQuality::Medium, FarQuality::High] {
            let (hole, patches) = select_at(&gen, gate_eye, quality, 1.0);
            let c = census_of(gate_eye, hole, &patches);
            println!(
                "| {} | {} | {} | {:.1} | {} | {} |",
                gate_eye.name,
                quality.name(),
                per_frame(c.patches, c.frames),
                c.draws / c.frames,
                per_frame(c.vertices, c.frames),
                per_frame(c.patches * SMOOTH_TRIANGLES / 2.0, c.frames),
            );
        }
    }
}

/// The owner's question as numbers (issue #422, decision 3's budget): how
/// much wider must the quads of each quality be drawn for the far terrain to
/// cost what it should? Every pixel size of the quality -- quad, error bound,
/// floor -- times the same factor (`--far-quad-scale`).
#[test]
#[ignore]
fn far_census_equal_cost() {
    let gen = WorldGen::new(World::new().seed());
    println!("| eye | quality | quads drawn | patches | vertices issued |");
    println!("|---|---|---|--:|--:|");
    for gate_eye in &GATE_EYES {
        for quality in [FarQuality::Low, FarQuality::Medium, FarQuality::High] {
            for coarser in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5] {
                let (hole, patches) = select_at(&gen, gate_eye, quality, coarser);
                let c = census_of(gate_eye, hole, &patches);
                println!(
                    "| {} | {} | x{coarser} | {} | {} |",
                    gate_eye.name,
                    quality.name(),
                    per_frame(c.patches, c.frames),
                    per_frame(c.vertices, c.frames),
                );
            }
        }
    }
}

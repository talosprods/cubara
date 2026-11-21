//! Where the far terrain's triangles are: what `--bench gate` draws at its
//! three eyes, at every far quality, counted by what a triangle is (a top, a
//! wall, a wall on a patch's edge), by distance, by how tall its face stands
//! on screen, and by what becomes of it -- and what the same view cost while
//! the far terrain was a smooth surface (issue #422).
//!
//! Measurement, not a check. The M3 is GPU-bound on these triangles at about a
//! nanosecond each, and the first lever is knowing which of them earn their
//! keep. Everything here is the game's own: `far_view`, the bench's eyes,
//! camera path and hole, `CameraUniform`'s projection, the renderer's
//! `Frustum`, and the renderer's own functions for which faces are drawn
//! (`groups_facing`, `facing_runs`), in which draw (`class_of`), and where a
//! corner ends up (`place`) -- so the patch count it prints is the bench's
//! `far terrain drawn` line, and can be checked against it.
//!
//! ```bash
//! cargo test --release -p cubara --bin cubara far_census -- --ignored --nocapture
//! ```

use std::collections::BTreeMap;

use cubara_render::far::{
    class_of, facing_runs, group_counts, groups_facing, hidden_height, issued, lowest_over, place,
    stored_order, CLASS_FACES, QUAD_INDICES,
};
use cubara_render::{Aabb, CameraUniform, FarDir, FarFace, Frustum};
use cubara_voxel::ChunkCoord;
use cubara_world::far::{self, PatchHeights, PATCH_QUADS};
use cubara_world::{node, World, WorldGen};
use glam::{Vec3, Vec4};

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

/// What a triangle is part of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Part {
    Top,
    Wall,
    /// A wall on a patch's edge: the step to the neighbouring patch, and
    /// the skirt under it.
    EdgeWall,
}

/// Triangles in one class, split by what becomes of them.
#[derive(Clone, Copy, Default)]
struct Row {
    total: f64,
    front: f64,
    back: f64,
    /// Seen exactly edge-on ([`Fate::EdgeOn`]).
    edge_on: f64,
    /// Wholly outside one plane of the view: clipped away.
    off_screen: f64,
    /// No area: a face the join folded flat or cut off at the box's floor.
    zero_area: f64,
    /// A corner moved by the join to the voxels' box
    /// (`cubara_render::far::place`).
    in_hole: f64,
}

impl Row {
    fn add(&mut self, t: &Triangle) {
        self.total += 1.0;
        match t.fate {
            Fate::Front => self.front += 1.0,
            Fate::Back => self.back += 1.0,
            Fate::EdgeOn => self.edge_on += 1.0,
            Fate::OffScreen => self.off_screen += 1.0,
            Fate::ZeroArea => self.zero_area += 1.0,
        }
        if t.in_hole {
            self.in_hole += 1.0;
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Fate {
    Front,
    Back,
    /// Its plane passes through the eye: neither side is shown, and on
    /// screen it is a line with no area. The gate eyes stand on whole blocks
    /// and every face plane is on a whole block, so this is not rare here: a
    /// top at the eye's own height, a wall on the eye's own x or z.
    EdgeOn,
    OffScreen,
    ZeroArea,
}

struct Triangle {
    fate: Fate,
    in_hole: bool,
    /// From the eye to its centroid, in blocks.
    distance: f32,
    /// Its face's height on screen, in pixels; `None` when a corner is
    /// behind the eye.
    face_px: Option<f32>,
    part: Part,
}

#[derive(Default)]
struct Census {
    frames: f64,
    patches: f64,
    /// Faces of the patches in view, whichever way they look: what would be
    /// drawn without the direction culling.
    faces_in_view: f64,
    /// Runs of faces drawn, by size class (`cubara_render::far::CLASS_FACES`).
    runs: Vec<f64>,
    /// Draws issued: a class with a run in it, a frame.
    draws: f64,
    /// Vertices issued, the padding of each run up to its class included.
    vertices: f64,
    all: Row,
    by_octave: BTreeMap<i32, Row>,
    by_height: BTreeMap<i32, Row>,
    by_part: BTreeMap<Part, Row>,
    /// Triangles in front of the eye whose facing by their normal and by
    /// their winding on screen disagree -- the check on the facing columns.
    facing_disagrees: f64,
    facing_checked: f64,
}

/// Height buckets: `[2^(k-1), 2^k)` pixels, `k` from -2 (under 1/4 px) to 6
/// (32 px and up); `i32::MAX` for a face with a corner behind the eye.
fn height_bucket(px: Option<f32>) -> i32 {
    match px {
        None => i32::MAX,
        Some(px) => (px.max(1e-6).log2().floor() as i32 + 1).clamp(-2, 6),
    }
}

fn height_label(k: i32) -> String {
    let px = |e: i32| {
        let v = 2f32.powi(e);
        if v < 1.0 {
            format!("{v}")
        } else {
            format!("{v:.0}")
        }
    };
    match k {
        i32::MAX => "a corner behind the eye".to_string(),
        -2 => format!("< {} px", px(-2)),
        6 => format!(">= {} px", px(5)),
        k => format!("{}-{} px", px(k - 1), px(k)),
    }
}

/// `[2^k, 2^(k+1))` blocks, in thousands of 1,024.
fn octave_label(k: i32) -> String {
    let kb = |e: i32| format!("{}", 2f64.powi(e - 10));
    format!("{}-{}k", kb(k), kb(k + 1))
}

/// The two triangles of `face` as `far.wgsl` draws them.
fn triangles(
    face: &FarFace,
    patch: ([f32; 2], f32, f32),
    hole: (Vec3, Vec3),
    view_proj: glam::Mat4,
    (eye, hidden): (Vec3, f32),
    census: &mut Census,
    mut emit: impl FnMut(Triangle),
) {
    let corners = face.corners();
    let world = corners.map(|v| place(&v, patch, Some(hole), hidden));
    let moved = [0, 1, 2, 3].map(|c| world[c] != place(&corners[c], patch, None, 0.0));
    let clip = world.map(|p| view_proj * Vec4::new(p.x, p.y, p.z, 1.0));
    let outside_one_plane = |c: [Vec4; 3]| {
        let all = |f: &dyn Fn(Vec4) -> bool| c.iter().all(|&v| f(v));
        all(&|v| v.x > v.w)
            || all(&|v| v.x < -v.w)
            || all(&|v| v.y > v.w)
            || all(&|v| v.y < -v.w)
            || all(&|v| v.z > v.w)
            || all(&|v| v.z < 0.0)
    };
    let face_px = clip.iter().all(|c| c.w > 0.0).then(|| {
        let ys = clip.map(|c| c.y / c.w * SIZE.1 as f32 / 2.0);
        let (lo, hi) = ys
            .iter()
            .fold((f32::MAX, f32::MIN), |(l, h), &y| (l.min(y), h.max(y)));
        hi - lo
    });
    let part = match (face.dir, face.border) {
        (FarDir::Top, _) => Part::Top,
        (_, false) => Part::Wall,
        (_, true) => Part::EdgeWall,
    };
    for t in QUAD_INDICES.chunks(3) {
        let [i0, i1, i2] = [t[0] as usize, t[1] as usize, t[2] as usize];
        let (p0, p1, p2) = (world[i0], world[i1], world[i2]);
        // In f64, where it is exact: the corners are whole blocks
        // (`place`), and so are the gate eyes.
        let [d0, d1, d2] = [p0, p1, p2].map(|p| p.as_dvec3());
        let normal = (d1 - d0).cross(d2 - d0);
        let facing = normal.dot(eye.as_dvec3() - d0);
        let fate = if normal.length_squared() == 0.0 {
            Fate::ZeroArea
        } else if outside_one_plane([clip[i0], clip[i1], clip[i2]]) {
            Fate::OffScreen
        } else if facing == 0.0 {
            Fate::EdgeOn
        } else if facing > 0.0 {
            Fate::Front
        } else {
            Fate::Back
        };
        // The same facing from the winding on screen: counter-clockwise is
        // front (wgpu's default `FrontFace::Ccw`, which the far pipeline
        // keeps). In f64 as well, so that what is checked is the rule and
        // not `f32`'s rounding of a sliver's area. An edge-on triangle has
        // no winding to check: its area on screen is rounding noise of
        // either sign (measured at the ground eye: 40% of them positive).
        if matches!(fate, Fate::Front | Fate::Back) && [i0, i1, i2].iter().all(|&v| clip[v].w > 0.0)
        {
            let m = view_proj.as_dmat4();
            let s = [d0, d1, d2].map(|p| {
                let c = m * p.extend(1.0);
                glam::DVec2::new(c.x, c.y) / c.w
            });
            let area = (s[1] - s[0]).perp_dot(s[2] - s[0]);
            census.facing_checked += 1.0;
            if (area > 0.0) != (fate == Fate::Front) {
                census.facing_disagrees += 1.0;
            }
        }
        emit(Triangle {
            fate,
            in_hole: moved[i0] || moved[i1] || moved[i2],
            distance: ((p0 + p1 + p2) / 3.0 - eye).length(),
            face_px,
            part,
        });
    }
}

/// The far terrain's selection for `gate_eye` at `quality`, with every quad,
/// error bound and floor `coarser` times as wide on screen as the quality
/// names them (1: the sizes the smooth surface was drawn at; the game draws
/// at `far_streaming::FAR_QUAD_SCALE`).
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
/// `detail`: also every triangle's fate, which is most of the time.
fn census_of(
    gate_eye: &GateEye,
    hole: far::Hole,
    patches: &[PatchHeights],
    detail: bool,
) -> Census {
    let eye = gate_eye.eye;
    // The renderer's view of them: its faces, in the order it stores them,
    // its boxes and its hole (`to_far_params`), in f32.
    let faces: Vec<Vec<FarFace>> = patches
        .iter()
        .map(|p| stored_order(&to_far_faces(p)))
        .collect();
    let placement = |p: &PatchHeights| {
        let [ox, oz] = p.key.origin();
        (
            [ox as f32, oz as f32],
            p.key.quad() as f32,
            p.skirt_blocks() as f32,
        )
    };
    let boxes: Vec<Aabb> = patches
        .iter()
        .zip(&faces)
        .map(|(p, faces)| {
            let (origin, quad, skirt) = placement(p);
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
    // Where the hidden vertices go, as `FarTerrain::prepare` works it out.
    let hole_box = Aabb {
        min: hole_f32.0,
        max: hole_f32.1,
    };
    let held: Vec<Option<Aabb>> = boxes.iter().copied().map(Some).collect();
    let eye_v = Vec3::from(eye);
    let hidden = hidden_height(hole_f32.0, hole_f32.1, eye_v, lowest_over(&held, &hole_box));

    let aspect = SIZE.0 as f32 / SIZE.1 as f32;
    let mut census = Census {
        runs: vec![0.0; CLASS_FACES.len()],
        ..Census::default()
    };
    for s in 0..YAWS {
        let frame = 200.0 + (s as f32 + 0.5) * 2000.0 / YAWS as f32;
        let yaw = frame / 240.0 * 0.3;
        let dir = glam::vec3(yaw.cos(), -0.25, yaw.sin());
        let view_proj = CameraUniform::look_view_proj(aspect, eye_v, dir);
        let frustum = Frustum::from_view_proj(view_proj);
        census.frames += 1.0;
        let mut frame_runs = vec![0usize; CLASS_FACES.len()];
        for (i, p) in patches.iter().enumerate() {
            if !frustum.intersects_aabb(&boxes[i]) || in_hole(&boxes[i]) {
                continue;
            }
            census.patches += 1.0;
            census.faces_in_view += faces[i].len() as f64;
            let facing = groups_facing(&boxes[i], eye_v);
            let mut runs = Vec::new();
            facing_runs(group_counts(&faces[i]), facing, |first, count| {
                runs.push((first as usize, count as usize));
            });
            for (first, count) in runs {
                frame_runs[class_of(count as u32)] += 1;
                if !detail {
                    census.all.total += count as f64 * 2.0;
                    continue;
                }
                for face in &faces[i][first..first + count] {
                    let mut tris = Vec::with_capacity(2);
                    triangles(
                        face,
                        placement(p),
                        hole_f32,
                        view_proj,
                        (eye_v, hidden),
                        &mut census,
                        |t| tris.push(t),
                    );
                    for t in &tris {
                        census.all.add(t);
                        let octave = t.distance.max(1.0).log2().floor() as i32;
                        census.by_octave.entry(octave).or_default().add(t);
                        census
                            .by_height
                            .entry(height_bucket(t.face_px))
                            .or_default()
                            .add(t);
                        census.by_part.entry(t.part).or_default().add(t);
                    }
                }
            }
        }
        let (draws, vertices) = issued(&frame_runs);
        census.draws += draws as f64;
        census.vertices += vertices as f64;
        for (class, n) in frame_runs.into_iter().enumerate() {
            census.runs[class] += n as f64;
        }
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

fn print_table(title: &str, rows: &[(String, Row)], frames: f64, all: f64) {
    let pct = |v: f64, of: f64| {
        if of > 0.0 {
            format!("{:.0}%", 100.0 * v / of)
        } else {
            "-".to_string()
        }
    };
    println!();
    println!("| {title} | triangles | share | front | back | edge-on | off screen | zero area | moved by the join |");
    println!("|---|--:|--:|--:|--:|--:|--:|--:|--:|");
    for (label, r) in rows {
        println!(
            "| {label} | {} | {} | {} | {} | {} | {} | {} | {} |",
            per_frame(r.total, frames),
            pct(r.total, all),
            pct(r.front, r.total),
            pct(r.back, r.total),
            pct(r.edge_on, r.total),
            pct(r.off_screen, r.total),
            pct(r.zero_area, r.total),
            pct(r.in_hole, r.total),
        );
    }
}

/// The table issue #422 is judged by: at every gate eye and every quality,
/// what the far terrain cost as a smooth surface and what it costs as
/// blocks -- with every face, and with only those that can face the eye.
#[test]
#[ignore]
fn far_census() {
    let gen = WorldGen::new(World::new().seed());
    println!(
        "{}x{}, 60 deg vertical; {YAWS} directions along the bench's path.",
        SIZE.0, SIZE.1
    );
    println!();
    println!(
        "| eye | quality | patches | smooth triangles | blocks, every face | blocks, facing the \
         eye | against smooth | draws | vertices issued | padding |"
    );
    println!("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|");
    for gate_eye in &GATE_EYES {
        for quality in [FarQuality::Low, FarQuality::Medium, FarQuality::High] {
            let (hole, patches) = select_at(&gen, gate_eye, quality, 1.0);
            let c = census_of(gate_eye, hole, &patches, false);
            let smooth = c.patches * SMOOTH_TRIANGLES;
            println!(
                "| {} | {} | {} | {} | {} | {} | {:+.0}% | {:.1} | {} | {:+.0}% |",
                gate_eye.name,
                quality.name(),
                per_frame(c.patches, c.frames),
                per_frame(smooth, c.frames),
                per_frame(c.faces_in_view * 2.0, c.frames),
                per_frame(c.all.total, c.frames),
                100.0 * (c.all.total / smooth - 1.0),
                c.draws / c.frames,
                per_frame(c.vertices, c.frames),
                100.0 * (c.vertices / (c.all.total * 2.0) - 1.0),
            );
        }
    }
}

/// The owner's question as numbers (issue #422, decision 3's budget): how
/// much wider must the quads of each quality be drawn for the blocks to cost
/// what the smooth surface costs at that quality today? Every pixel size of
/// the quality -- quad, error bound, floor -- times the same factor.
#[test]
#[ignore]
fn far_census_equal_cost() {
    let gen = WorldGen::new(World::new().seed());
    println!("| eye | quality | quads drawn | patches | triangles, facing the eye | against today's smooth |");
    println!("|---|---|---|--:|--:|--:|");
    for gate_eye in &GATE_EYES {
        for quality in [FarQuality::Low, FarQuality::Medium, FarQuality::High] {
            let (hole, patches) = select_at(&gen, gate_eye, quality, 1.0);
            let today = census_of(gate_eye, hole, &patches, false);
            let smooth = today.patches / today.frames * SMOOTH_TRIANGLES;
            for coarser in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5] {
                let (hole, patches) = select_at(&gen, gate_eye, quality, coarser);
                let c = census_of(gate_eye, hole, &patches, false);
                let triangles = c.all.total / c.frames;
                println!(
                    "| {} | {} | x{coarser} | {} | {} | {:+.0}% |",
                    gate_eye.name,
                    quality.name(),
                    per_frame(c.patches, c.frames),
                    per_frame(c.all.total, c.frames),
                    100.0 * (triangles / smooth - 1.0),
                );
            }
        }
    }
}

/// Where High's triangles are at each gate eye: by what they are, by
/// distance and by size on screen, and what becomes of each.
#[test]
#[ignore]
fn far_census_detail() {
    let gen = WorldGen::new(World::new().seed());
    for gate_eye in &GATE_EYES {
        let (hole, patches) = select_at(&gen, gate_eye, FarQuality::High, 1.0);
        let c = census_of(gate_eye, hole, &patches, true);
        let frames = c.frames;
        println!();
        println!(
            "### Gate eye {} {:?}: {} patches, {} triangles a frame",
            gate_eye.name,
            gate_eye.eye,
            per_frame(c.patches, frames),
            per_frame(c.all.total, frames),
        );
        println!(
            "High at {}x{}, 60 deg vertical; {YAWS} directions along the bench's path; \
             hole {:?}..{:?}; facing checked on screen for {:.0} triangles a frame, \
             {:.0} disagree; {:.0} a frame are seen exactly edge-on",
            SIZE.0,
            SIZE.1,
            hole.min,
            hole.max,
            c.facing_checked / frames,
            c.facing_disagrees / frames,
            c.all.edge_on / frames,
        );
        // The facing columns rest on this, and it is exact: a corner is on
        // whole blocks and so is a gate eye, so a triangle is in front of
        // the eye, behind it, or exactly edge-on -- which is its own column,
        // because an edge-on triangle has no winding. Until that column
        // existed this check allowed one disagreement in a thousand and the
        // ground eye had 3.7: every one of them a top at the eye's own
        // height or a wall on the eye's own x or z, counted as facing away
        // while its area on screen was rounding noise of either sign.
        assert_eq!(
            c.facing_disagrees, 0.0,
            "normal and winding disagree on {} of {} triangles",
            c.facing_disagrees, c.facing_checked
        );
        let all = c.all.total;
        let total = [("all".to_string(), c.all)];
        print_table("", &total, frames, all);
        let rows: Vec<_> = c
            .by_part
            .iter()
            .map(|(&k, &r)| (format!("{k:?}"), r))
            .collect();
        print_table("part", &rows, frames, all);
        let rows: Vec<_> = c
            .by_octave
            .iter()
            .map(|(&k, &r)| (octave_label(k), r))
            .collect();
        print_table("distance, blocks", &rows, frames, all);
        let rows: Vec<_> = c
            .by_height
            .iter()
            .map(|(&k, &r)| (height_label(k), r))
            .collect();
        print_table("face height on screen", &rows, frames, all);
        println!();
        println!("| runs of up to (faces) | runs a frame |");
        println!("|--:|--:|");
        for (class, runs) in c.runs.iter().enumerate() {
            if *runs > 0.0 {
                println!("| {} | {:.1} |", CLASS_FACES[class], runs / frames);
            }
        }
    }
}

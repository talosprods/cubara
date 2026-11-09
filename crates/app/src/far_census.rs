//! Where the far terrain's triangles are: what `--bench gate` draws at its
//! three eyes, counted by distance, by how tall a quad stands on screen, by
//! which way it faces, by why its patch stopped splitting, and by whether the
//! voxels' hole sank it.
//!
//! Measurement, not a check. The M3 is GPU-bound on these triangles at about a
//! nanosecond each, and the first lever is knowing which of them earn their
//! keep. Everything here is the game's own: `far_view` at High, the bench's
//! eyes, camera path and hole, `CameraUniform`'s projection and the renderer's
//! `Frustum` -- so the patch count it prints is the bench's `far terrain
//! drawn` line, and can be checked against it.
//!
//! ```bash
//! cargo test --release -p cubara --bin cubara far_census -- --ignored --nocapture
//! ```

use std::collections::{BTreeMap, HashMap};

use cubara_render::{Aabb, CameraUniform, Frustum};
use cubara_voxel::ChunkCoord;
use cubara_world::far::{self, FarView, PatchHeights, PatchKey, PATCH_QUADS, PATCH_VERTS};
use cubara_world::{node, World, WorldGen};
use glam::{Vec3, Vec4};

use crate::bench::{GateEye, GATE_EYES};
use crate::far_streaming::{far_view, FarQuality};

const SIZE: (u32, u32) = (1920, 1080);

/// Camera directions sampled along the path the bench measures: frame
/// 200 to 2,200 at 1/240 virtual s a frame, a yaw of 0.3 rad per virtual s.
const YAWS: usize = 40;

/// The vertex grid `far.wgsl` draws a patch with: its own vertices and a ring
/// dropped into the skirt.
const GRID: usize = PATCH_VERTS + 2;

/// Why `far::select` stopped splitting a patch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Stop {
    /// Its parent was split because the eye was near; this one is far enough,
    /// and smooth enough.
    Distance,
    /// Its parent was split only because its error would show; this one's
    /// does not.
    Error,
    /// Its error would show, but its children would be narrower than the
    /// floor (or it is level 0).
    Floor,
}

/// `far::select`'s decision for one patch, from its error. A mirror of
/// `walk`, checked against it: every patch `select` keeps must come out
/// `Kept` or `KeptAtFloor` here, and every parent of one `Split*`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Decision {
    SplitNear,
    SplitRough,
    Kept,
    KeptAtFloor,
}

fn decide(view: &FarView, key: PatchKey, error: f32) -> Decision {
    let (lo, hi) = far::surface_bounds();
    let [ox, oz] = key.origin();
    let size = key.size() as f64;
    let min = [ox as f64, lo, oz as f64];
    let max = [ox as f64 + size, hi, oz as f64 + size];
    let d = (0..3)
        .map(|a| {
            (min[a] - view.eye[a])
                .max(view.eye[a] - max[a])
                .max(0.0)
                .powi(2)
        })
        .sum::<f64>()
        .sqrt();
    let near = d < view.split_ratio * size;
    let children_wide_enough = (key.quad() as f64 / 2.0) >= view.min_quad * d;
    let shows = error as f64 > view.max_error * d;
    if key.level > 0 && near {
        Decision::SplitNear
    } else if key.level > 0 && children_wide_enough && shows {
        Decision::SplitRough
    } else if shows || near {
        // Rough, or near at level 0: either way, nothing finer to split into.
        Decision::KeptAtFloor
    } else {
        Decision::Kept
    }
}

/// `far::select`'s root level for `radius`: the coarsest patches at least
/// that wide.
fn root_level(radius: f64) -> u32 {
    (0..)
        .find(|&l| ((PATCH_QUADS as i64) << l) >= radius.ceil() as i64)
        .unwrap()
}

/// Triangles in one class, split by what becomes of them.
#[derive(Clone, Copy, Default)]
struct Row {
    total: f64,
    front: f64,
    back: f64,
    /// Wholly outside one plane of the view: clipped away.
    off_screen: f64,
    /// Two of its corners are the same point (the skirt's corners).
    zero_area: f64,
    /// All three corners strictly inside the voxels' hole, so sunk below it.
    in_hole: f64,
}

impl Row {
    fn add(&mut self, t: &Triangle) {
        self.total += 1.0;
        match t.fate {
            Fate::Front => self.front += 1.0,
            Fate::Back => self.back += 1.0,
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
    OffScreen,
    ZeroArea,
}

struct Triangle {
    fate: Fate,
    in_hole: bool,
    /// From the eye to its centroid, in blocks.
    distance: f32,
    /// Its quad's height on screen, in pixels; `None` when a corner is
    /// behind the eye.
    quad_px: Option<f32>,
    skirt: bool,
}

#[derive(Default)]
struct Census {
    frames: f64,
    patches: f64,
    all: Row,
    by_octave: BTreeMap<i32, Row>,
    by_height: BTreeMap<i32, Row>,
    by_stop: BTreeMap<Stop, Row>,
    surface: Row,
    skirt: Row,
    /// Triangles in front of the eye whose facing by their normal and by
    /// their winding on screen disagree -- the check on the facing columns.
    facing_disagrees: f64,
    facing_checked: f64,
}

/// Height buckets: `[2^(k-1), 2^k)` pixels, `k` from -2 (under 1/4 px) to 6
/// (32 px and up); `i32::MAX` for a quad with a corner behind the eye.
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

/// Every triangle of `patch` as `far.wgsl` draws it.
fn triangles(
    patch: &PatchHeights,
    hole: (Vec3, Vec3),
    view_proj: glam::Mat4,
    eye: Vec3,
    out: &mut Vec<Triangle>,
    census: &mut Census,
) {
    let [ox, oz] = patch.key.origin();
    let quad = patch.key.quad() as f32;
    let skirt_depth = patch.skirt();
    let mut world = Vec::with_capacity(GRID * GRID);
    let mut sunk = Vec::with_capacity(GRID * GRID);
    let mut clip = Vec::with_capacity(GRID * GRID);
    for gz in 0..GRID {
        for gx in 0..GRID {
            let i = (gx as i32 - 1).clamp(0, PATCH_QUADS as i32) as usize;
            let j = (gz as i32 - 1).clamp(0, PATCH_QUADS as i32) as usize;
            let ring = gx == 0 || gx == GRID - 1 || gz == 0 || gz == GRID - 1;
            let mut y = patch.heights[j * PATCH_VERTS + i];
            if ring {
                y -= skirt_depth;
            }
            let mut p = Vec3::new(ox as f32 + i as f32 * quad, y, oz as f32 + j as f32 * quad);
            let inside = p.cmpgt(hole.0).all() && p.cmplt(hole.1).all();
            if inside {
                p.y = hole.0.y - 1.0;
            }
            world.push(p);
            sunk.push(inside);
            clip.push(view_proj * Vec4::new(p.x, p.y, p.z, 1.0));
        }
    }
    let outside_one_plane = |c: [Vec4; 3]| {
        let all = |f: &dyn Fn(Vec4) -> bool| c.iter().all(|&v| f(v));
        all(&|v| v.x > v.w)
            || all(&|v| v.x < -v.w)
            || all(&|v| v.y > v.w)
            || all(&|v| v.y < -v.w)
            || all(&|v| v.z > v.w)
            || all(&|v| v.z < 0.0)
    };
    for z in 0..GRID - 1 {
        for x in 0..GRID - 1 {
            let a = z * GRID + x;
            let (b, c, d) = (a + 1, a + GRID, a + GRID + 1);
            let corners = [a, b, c, d];
            let quad_px = corners.iter().all(|&v| clip[v].w > 0.0).then(|| {
                let ys = corners.map(|v| clip[v].y / clip[v].w * SIZE.1 as f32 / 2.0);
                let (lo, hi) = ys
                    .iter()
                    .fold((f32::MAX, f32::MIN), |(l, h), &y| (l.min(y), h.max(y)));
                hi - lo
            });
            let skirt = x == 0 || z == 0 || x == GRID - 2 || z == GRID - 2;
            // `far.rs`'s `grid_indices`: [a, c, b] and [b, c, d].
            for [i0, i1, i2] in [[a, c, b], [b, c, d]] {
                let (p0, p1, p2) = (world[i0], world[i1], world[i2]);
                let normal = (p1 - p0).cross(p2 - p0);
                let fate = if normal.length_squared() == 0.0 {
                    Fate::ZeroArea
                } else if outside_one_plane([clip[i0], clip[i1], clip[i2]]) {
                    Fate::OffScreen
                } else if normal.dot(eye - p0) > 0.0 {
                    Fate::Front
                } else {
                    Fate::Back
                };
                // The same facing from the winding on screen: counter-clockwise
                // is front (wgpu's default `FrontFace::Ccw`, which the far
                // pipeline keeps).
                if matches!(fate, Fate::Front | Fate::Back)
                    && [i0, i1, i2].iter().all(|&v| clip[v].w > 0.0)
                {
                    let s = [i0, i1, i2].map(|v| clip[v].truncate().truncate() / clip[v].w);
                    let area = (s[1] - s[0]).perp_dot(s[2] - s[0]);
                    census.facing_checked += 1.0;
                    if (area > 0.0) != (fate == Fate::Front) {
                        census.facing_disagrees += 1.0;
                    }
                }
                out.push(Triangle {
                    fate,
                    in_hole: sunk[i0] && sunk[i1] && sunk[i2],
                    distance: ((p0 + p1 + p2) / 3.0 - eye).length(),
                    quad_px,
                    skirt,
                });
            }
        }
    }
}

fn census_at(gen: &WorldGen, gate_eye: &GateEye) -> (FarView, Census) {
    let eye = gate_eye.eye;
    let schedule = node::schedule_for_radius(crate::streaming::render_radius_chunks());
    let (min, max) = node::covered_box_3d(ChunkCoord::from_world_pos(eye), &schedule);
    let hole = far::Hole {
        min: min.map(|c| (c * 16) as f64),
        max: max.map(|c| (c * 16) as f64),
    };
    let view = far_view(eye.map(f64::from), SIZE.1, Some(hole), FarQuality::High)
        .expect("High draws the far terrain");
    let patches = far::build(gen, &view);

    // Why each one stopped, from its own decision and its parent's.
    let root = root_level(view.radius);
    let mut parents: HashMap<PatchKey, f32> = HashMap::new();
    let stops: Vec<Stop> = patches
        .iter()
        .map(|p| match decide(&view, p.key, p.error) {
            Decision::KeptAtFloor => Stop::Floor,
            Decision::Kept if p.key.level == root => Stop::Distance,
            Decision::Kept => {
                let k = p.key;
                let parent = PatchKey::new(k.level + 1, k.x >> 1, k.z >> 1);
                let error = *parents
                    .entry(parent)
                    .or_insert_with(|| far::generate(gen, parent).error);
                match decide(&view, parent, error) {
                    Decision::SplitNear => Stop::Distance,
                    Decision::SplitRough => Stop::Error,
                    other => panic!("{k:?} was kept, but its parent would be {other:?}"),
                }
            }
            other => panic!("{:?} was kept, but the mirror would {other:?}", p.key),
        })
        .collect();

    // The renderer's view of them: its boxes (`far.rs`'s `patch_box`) and
    // its hole (`to_far_params`), in f32.
    let boxes: Vec<Aabb> = patches
        .iter()
        .map(|p| {
            let [ox, oz] = p.key.origin();
            let size = p.key.size() as f32;
            let lo = p.heights.iter().copied().fold(f32::INFINITY, f32::min) - p.skirt();
            let hi = p.heights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            Aabb {
                min: Vec3::new(ox as f32, lo, oz as f32),
                max: Vec3::new(ox as f32 + size, hi, oz as f32 + size),
            }
        })
        .collect();
    let hole_f32 = (
        Vec3::from(hole.min.map(|v| v as f32)),
        Vec3::from(hole.max.map(|v| v as f32)),
    );
    let in_hole = |b: &Aabb| hole_f32.0.cmple(b.min).all() && b.max.cmple(hole_f32.1).all();

    let aspect = SIZE.0 as f32 / SIZE.1 as f32;
    let eye_v = Vec3::from(eye);
    let mut census = Census::default();
    let mut tris = Vec::new();
    for s in 0..YAWS {
        let frame = 200.0 + (s as f32 + 0.5) * 2000.0 / YAWS as f32;
        let yaw = frame / 240.0 * 0.3;
        let dir = glam::vec3(yaw.cos(), -0.25, yaw.sin());
        let view_proj = CameraUniform::look_view_proj(aspect, eye_v, dir);
        let frustum = Frustum::from_view_proj(view_proj);
        census.frames += 1.0;
        for (i, p) in patches.iter().enumerate() {
            if !frustum.intersects_aabb(&boxes[i]) || in_hole(&boxes[i]) {
                continue;
            }
            census.patches += 1.0;
            tris.clear();
            triangles(p, hole_f32, view_proj, eye_v, &mut tris, &mut census);
            for t in &tris {
                census.all.add(t);
                let octave = t.distance.max(1.0).log2().floor() as i32;
                census.by_octave.entry(octave).or_default().add(t);
                census
                    .by_height
                    .entry(height_bucket(t.quad_px))
                    .or_default()
                    .add(t);
                census.by_stop.entry(stops[i]).or_default().add(t);
                if t.skirt {
                    census.skirt.add(t);
                } else {
                    census.surface.add(t);
                }
            }
        }
    }
    (view, census)
}

fn print_table(title: &str, rows: &[(String, Row)], frames: f64, all: f64) {
    let n = |v: f64| {
        let v = (v / frames).round() as u64;
        let s = v.to_string();
        let mut out = String::new();
        for (i, ch) in s.chars().enumerate() {
            if i > 0 && (s.len() - i).is_multiple_of(3) {
                out.push(',');
            }
            out.push(ch);
        }
        out
    };
    let pct = |v: f64, of: f64| {
        if of > 0.0 {
            format!("{:.0}%", 100.0 * v / of)
        } else {
            "-".to_string()
        }
    };
    println!();
    println!("| {title} | triangles | share | front | back | off screen | zero area | all 3 in the hole |");
    println!("|---|--:|--:|--:|--:|--:|--:|--:|");
    for (label, r) in rows {
        println!(
            "| {label} | {} | {} | {} | {} | {} | {} | {} |",
            n(r.total),
            pct(r.total, all),
            pct(r.front, r.total),
            pct(r.back, r.total),
            pct(r.off_screen, r.total),
            pct(r.zero_area, r.total),
            pct(r.in_hole, r.total),
        );
    }
}

#[test]
#[ignore]
fn far_census() {
    let gen = WorldGen::new(World::new().seed());
    for gate_eye in &GATE_EYES {
        let (view, c) = census_at(&gen, gate_eye);
        let frames = c.frames;
        println!();
        println!(
            "### Gate eye {} {:?}: {:.0} patches, {:.0} triangles a frame",
            gate_eye.name,
            gate_eye.eye,
            c.patches / frames,
            c.all.total / frames,
        );
        println!(
            "High at {}x{}, 60 deg vertical; {YAWS} directions along the bench's path; \
             hole {:?}..{:?}; facing checked on screen for {:.0} triangles a frame, \
             {:.0} disagree",
            SIZE.0,
            SIZE.1,
            view.hole.map(|h| h.min),
            view.hole.map(|h| h.max),
            c.facing_checked / frames,
            c.facing_disagrees / frames,
        );
        // The facing columns rest on this.
        assert!(
            c.facing_disagrees <= c.facing_checked * 1e-3,
            "normal and winding disagree on {} of {} triangles",
            c.facing_disagrees,
            c.facing_checked
        );
        let all = c.all.total;
        let total = [("all".to_string(), c.all)];
        print_table("", &total, frames, all);
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
        print_table("quad height on screen", &rows, frames, all);
        let rows: Vec<_> = c
            .by_stop
            .iter()
            .map(|(&k, &r)| (format!("{k:?}").to_lowercase(), r))
            .collect();
        print_table("patch stopped splitting at", &rows, frames, all);
        let rows = [
            ("surface".to_string(), c.surface),
            ("skirt".to_string(), c.skirt),
        ];
        print_table("part", &rows, frames, all);
    }
}

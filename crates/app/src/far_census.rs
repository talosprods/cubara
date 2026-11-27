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
    class_of, frame_runs, hidden_height, issued, lowest_over, patch_box, place, runs_room,
    segment_runs, stored_order, tile_masks, CullPatch, PatchCull, CLASS_FACES, PATCH_TILES,
    QUAD_INDICES, TILES, TILE_QUADS,
};
use cubara_render::{Aabb, CameraUniform, FarCulling, FarDir, FarFace, FarPatch, Frustum};
use cubara_voxel::ChunkCoord;
use cubara_world::far::{self, PatchHeights, PATCH_QUADS};
use cubara_world::{node, World, WorldGen};
use glam::{Vec3, Vec4};

use crate::bench::{GateEye, GATE_EYES};
use crate::far_census_horizon::{Horizon, HorizonTile};
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
    /// Its face's direction (`FarDir as u8`) and whether it is a wall on
    /// the patch's edge: the census by kind.
    kind: (u8, bool),
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
    /// The most runs one frame drew.
    most_runs: usize,
    /// Milliseconds the cull took (`frame_runs`), all frames.
    cull_ms: f64,
    all: Row,
    by_octave: BTreeMap<i32, Row>,
    by_height: BTreeMap<i32, Row>,
    by_part: BTreeMap<Part, Row>,
    /// By direction (`FarDir as u8`) and whether a wall is on the patch's
    /// edge.
    by_kind: BTreeMap<(u8, bool), Row>,
    /// Faces of the patches in view under other meshing rules
    /// ([`merged_counts`]), `[tops, walls]` per rule.
    merged: [[f64; 2]; MERGE_RULES.len()],
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
            kind: (face.dir as u8, face.border),
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

/// The far terrain of one selection as the renderer holds it: each patch's
/// faces in the order it stores them, its placement, box and tiles, and the
/// hole -- in world blocks, which is the renderer's frame with its anchor at
/// the origin.
struct Scene {
    faces: Vec<Vec<FarFace>>,
    placement: Vec<([f32; 2], f32, f32)>,
    boxes: Vec<Aabb>,
    culls: Vec<PatchCull>,
    hole: (Vec3, Vec3),
    /// Where the hidden vertices go, as `FarTerrain::prepare` works it out.
    hidden: f32,
}

impl Scene {
    fn of(eye: Vec3, hole: far::Hole, patches: &[PatchHeights]) -> Self {
        let faces: Vec<Vec<FarFace>> = patches
            .iter()
            .map(|p| stored_order(&to_far_faces(p)))
            .collect();
        let placement: Vec<_> = patches
            .iter()
            .map(|p| {
                let [ox, oz] = p.key.origin();
                (
                    [ox as f32, oz as f32],
                    p.key.quad() as f32,
                    p.skirt_blocks() as f32,
                )
            })
            .collect();
        // The renderer's own box of each patch.
        let boxes: Vec<Aabb> = patches
            .iter()
            .zip(&faces)
            .zip(&placement)
            .map(|((p, faces), &(_, quad, skirt))| {
                patch_box(&FarPatch {
                    origin: p.key.origin(),
                    quad,
                    faces,
                    skirt,
                })
            })
            .collect();
        let culls = faces
            .iter()
            .zip(&placement)
            .map(|(f, &(_, _, skirt))| PatchCull::of(f, skirt))
            .collect();
        let hole = (
            Vec3::from(hole.min.map(|v| v as f32)),
            Vec3::from(hole.max.map(|v| v as f32)),
        );
        let held: Vec<Option<Aabb>> = boxes.iter().copied().map(Some).collect();
        let hidden = hidden_height(hole.0, hole.1, eye, lowest_over(&held, &hole_box(hole)));
        Self {
            faces,
            placement,
            boxes,
            culls,
            hole,
            hidden,
        }
    }

    /// The patches a frame draws from: those whose box the frustum passes
    /// and the hole does not hold.
    fn in_view(&self, frustum: &Frustum) -> Vec<usize> {
        let (lo, hi) = self.hole;
        (0..self.boxes.len())
            .filter(|&i| {
                let b = &self.boxes[i];
                frustum.intersects_aabb(b) && !(lo.cmple(b.min).all() && b.max.cmple(hi).all())
            })
            .collect()
    }

    /// The patches of `in_view` as the cull takes them.
    fn inputs(&self, in_view: &[usize]) -> Vec<CullPatch<'_>> {
        in_view
            .iter()
            .map(|&i| CullPatch {
                role: cubara_render::far::FarRole::Terrain,
                aabb: self.boxes[i],
                origin: self.placement[i].0,
                quad: self.placement[i].1,
                dy: 0.0,
                cull: &self.culls[i],
            })
            .collect()
    }

    /// The runs `FarTerrain::prepare` draws of `in_view`: `(patch, first
    /// face, faces)`, `patch` an index into the scene.
    fn runs(
        &self,
        in_view: &[usize],
        (frustum, eye): (&Frustum, Vec3),
        culling: FarCulling,
    ) -> Vec<(usize, u32, u32)> {
        let mut out = Vec::new();
        frame_runs(
            &self.inputs(in_view),
            (frustum, eye),
            (Some(&hole_box(self.hole)), culling),
            runs_room(GAME_CAPACITY),
            &mut out,
        );
        out.into_iter()
            .map(|(n, first, count)| (in_view[n], first, count))
            .collect()
    }

    /// The runs the renderer's tiles would leave with the horizon
    /// (`far_census_horizon`) on top: a candidate, measured.
    fn horizon_runs(
        &self,
        in_view: &[usize],
        (frustum, eye): (&Frustum, Vec3),
        horizon: &mut Horizon,
    ) -> Vec<(usize, u32, u32)> {
        let hole = hole_box(self.hole);
        let inputs = self.inputs(in_view);
        let mut masks: Vec<[u8; TILES]> = inputs
            .iter()
            .map(|p| tile_masks(p, (frustum, eye), Some(&hole)))
            .collect();
        let mut tiles = Vec::new();
        let mut owners = Vec::new();
        for (n, p) in inputs.iter().enumerate() {
            let floors = solid_floors(&self.faces[in_view[n]]);
            for t in 0..TILES {
                let Some(bounds) = p.tile_box(t).filter(|_| masks[n][t] != 0) else {
                    continue;
                };
                let side = (TILE_QUADS as f32) * p.quad;
                let (tx, tz) = (
                    (t % PATCH_TILES) as f32 * side,
                    (t / PATCH_TILES) as f32 * side,
                );
                tiles.push(HorizonTile {
                    bounds,
                    square: [
                        p.origin[0] + tx,
                        p.origin[1] + tz,
                        p.origin[0] + tx + side,
                        p.origin[1] + tz + side,
                    ],
                    floor: floors[t],
                    exempt: cubara_render::far::hole_reaches(Some(&hole), &bounds),
                });
                owners.push((n, t));
            }
        }
        let mut shown = Vec::new();
        horizon.cull(eye, &tiles, &mut shown);
        for (&(n, t), &seen) in owners.iter().zip(&shown) {
            if !seen {
                masks[n][t] = 0;
            }
        }
        let mut out = Vec::new();
        for (n, p) in inputs.iter().enumerate() {
            segment_runs(&p.cull.starts, &masks[n], |first, count| {
                out.push((in_view[n], first, count))
            });
        }
        out
    }
}

/// For the horizon: the lowest top over each tile's quads, where every quad
/// of the tile is under a top -- the ground solid under it -- and `-inf`
/// where one is not (then the tile hides nothing).
fn solid_floors(faces: &[FarFace]) -> [f32; TILES] {
    let mut floors = [f32::INFINITY; TILES];
    let mut covered = [0usize; TILES];
    for f in faces.iter().filter(|f| f.dir == FarDir::Top) {
        for j in f.j0 as usize..f.j1 as usize {
            for i in f.i0 as usize..f.i1 as usize {
                let t = (j / TILE_QUADS) * PATCH_TILES + i / TILE_QUADS;
                floors[t] = floors[t].min(f.top as f32);
                covered[t] += 1;
            }
        }
    }
    for (floor, n) in floors.iter_mut().zip(covered) {
        if n < TILE_QUADS * TILE_QUADS {
            *floor = f32::NEG_INFINITY;
        }
    }
    floors
}

/// `face.wgsl`'s `CRACK_SEAL`: how far a corner moves out of its face, per
/// block of distance.
const CRACK_SEAL: f32 = 0.00006;

/// The window's room for far patches (`render.rs`'s `FAR_PATCH_CAPACITY`),
/// which sets how many runs a frame has room for.
const GAME_CAPACITY: u32 = 4096;

fn hole_box((min, max): (Vec3, Vec3)) -> Aabb {
    Aabb { min, max }
}

/// How finely the census culls: the game's, or `FAR_CENSUS_CULLING=patches`
/// / `tiles` to see what the finer levels take.
fn culling() -> FarCulling {
    match std::env::var("FAR_CENSUS_CULLING").as_deref() {
        Ok("patches") => FarCulling::Patches,
        Ok("tiles") => FarCulling::Tiles,
        _ => FarCulling::default(),
    }
}

/// What the far terrain costs from `gate_eye`, over the bench's camera path.
/// `detail`: also every triangle's fate, which is most of the time.
fn census_of(
    gate_eye: &GateEye,
    hole: far::Hole,
    patches: &[PatchHeights],
    detail: bool,
) -> Census {
    let eye_v = Vec3::from(gate_eye.eye);
    let scene = Scene::of(eye_v, hole, patches);
    let merged: Vec<[[usize; 2]; MERGE_RULES.len()]> = if detail {
        patches.iter().map(merged_counts).collect()
    } else {
        Vec::new()
    };
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
        let in_view = scene.in_view(&frustum);
        for &i in &in_view {
            census.patches += 1.0;
            census.faces_in_view += scene.faces[i].len() as f64;
            if detail {
                for (rule, counts) in merged[i].iter().enumerate() {
                    census.merged[rule][0] += counts[0] as f64;
                    census.merged[rule][1] += counts[1] as f64;
                }
            }
        }
        let started = std::time::Instant::now();
        let runs = scene.runs(&in_view, (&frustum, eye_v), culling());
        census.cull_ms += started.elapsed().as_secs_f64() * 1000.0;
        census.most_runs = census.most_runs.max(runs.len());
        let mut frame_runs = vec![0usize; CLASS_FACES.len()];
        for (i, first, count) in runs {
            frame_runs[class_of(count)] += 1;
            if !detail {
                census.all.total += count as f64 * 2.0;
                continue;
            }
            for face in &scene.faces[i][first as usize..(first + count) as usize] {
                let mut tris = Vec::with_capacity(2);
                triangles(
                    face,
                    scene.placement[i],
                    scene.hole,
                    view_proj,
                    (eye_v, scene.hidden),
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
                    census.by_kind.entry(t.kind).or_default().add(t);
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

/// Other ways to mesh the same tops, counted by [`merged_counts`]: whether
/// two faces must wear the same to merge, and whether a face may reach
/// into the ground (a top on under higher quads, a wall down past the
/// lower top) -- which hides it there and draws nothing new. None of them
/// keeps to `far::JOIN_GRID`.
const MERGE_RULES: [(&str, bool, bool); 4] = [
    ("same wear, no join grid", true, false),
    ("same wear, no join grid, into the ground", true, true),
    ("any wear, no join grid", false, false),
    ("any wear, no join grid, into the ground", false, true),
];

/// The faces `p` would have under each of [`MERGE_RULES`], `[tops, walls]`:
/// the mesher's greedy order (`far::mesh`), with its limits swapped.
fn merged_counts(p: &PatchHeights) -> [[usize; 2]; MERGE_RULES.len()] {
    const N: usize = PATCH_QUADS;
    let top = |i: i32, j: i32| p.top(i, j);
    MERGE_RULES.map(|(_, wear, into)| {
        // Tops.
        let looks = |q: usize| (p.ground[q].top, p.ground[q].top_open);
        let mut done = vec![false; N * N];
        let mut tops = 0;
        for j in 0..N {
            for i in 0..N {
                let first = j * N + i;
                if done[first] {
                    continue;
                }
                let h = top(i as i32, j as i32);
                let fits = |q: usize, done: &[bool]| {
                    let t = top((q % N) as i32, (q / N) as i32);
                    if t == h && (!wear || looks(q) == looks(first)) {
                        into || !done[q]
                    } else {
                        into && t > h
                    }
                };
                let mut w = 1;
                while i + w < N && fits(first + w, &done) {
                    w += 1;
                }
                let mut d = 1;
                while j + d < N && (0..w).all(|k| fits((j + d) * N + i + k, &done)) {
                    d += 1;
                }
                for dj in 0..d {
                    for di in 0..w {
                        let q = (j + dj) * N + i + di;
                        if top((q % N) as i32, (q / N) as i32) == h
                            && (!wear || looks(q) == looks(first))
                        {
                            done[q] = true;
                        }
                    }
                }
                tops += 1;
            }
        }
        // Walls.
        let mut walls = 0;
        for along_x in [true, false] {
            for line in 0..=N as i32 {
                // (direction, top, rise, border, what it shows)
                let wall = |k: usize| {
                    let (below, above) = if along_x {
                        (top(line - 1, k as i32), top(line, k as i32))
                    } else {
                        (top(k as i32, line - 1), top(k as i32, line))
                    };
                    let shown = |high: bool, d: usize| {
                        let across = if high { line } else { line - 1 } as usize;
                        let g = if along_x {
                            p.ground[k * N + across]
                        } else {
                            p.ground[across * N + k]
                        };
                        (g.sides[d], g.side_open[d])
                    };
                    let (pos, neg) = if along_x { (0, 1) } else { (2, 3) };
                    if line == 0 {
                        Some((neg, above, (above - below).max(0), true, shown(true, neg)))
                    } else if line == N as i32 {
                        Some((pos, below, (below - above).max(0), true, shown(false, pos)))
                    } else if below > above {
                        Some((pos, below, below - above, false, shown(false, pos)))
                    } else if above > below {
                        Some((neg, above, above - below, false, shown(true, neg)))
                    } else {
                        None
                    }
                };
                let same = |a: (usize, i32, i32, bool, ([u8; 2], u8)),
                            b: (usize, i32, i32, bool, ([u8; 2], u8))| {
                    a.0 == b.0
                        && a.1 == b.1
                        && a.3 == b.3
                        && (into || a.2 == b.2)
                        && (!wear || a.4 == b.4)
                };
                let mut k = 0;
                while k < N {
                    let Some(this) = wall(k) else {
                        k += 1;
                        continue;
                    };
                    let mut end = k + 1;
                    while end < N && wall(end).is_some_and(|w| same(this, w)) {
                        end += 1;
                    }
                    walls += 1;
                    k = end;
                }
            }
        }
        [tops, walls]
    })
}

/// **The census the far terrain's triangle budget is judged by**: at each
/// gate eye -- the hill first, the one the gate turns on -- at Medium and
/// at several quad scales, the triangles drawn by kind (tops; walls by the
/// way they face; walls on a patch's edge) and what becomes of each, and
/// how many faces other meshing rules would leave of the same patches.
#[test]
#[ignore]
fn far_census_by_kind() {
    let gen = WorldGen::new(World::new().seed());
    let scales: Vec<f64> = std::env::var("FAR_CENSUS_SCALES")
        .ok()
        .map(|s| s.split(',').filter_map(|v| v.parse().ok()).collect())
        .unwrap_or_else(|| vec![1.0, 1.5, 2.0, 2.5, 3.0]);
    let quality = match std::env::var("FAR_CENSUS_QUALITY").as_deref() {
        Ok("High") => FarQuality::High,
        Ok("Low") => FarQuality::Low,
        _ => FarQuality::Medium,
    };
    println!(
        "{}x{}, {}, {YAWS} directions along the bench's path; per frame.",
        SIZE.0,
        SIZE.1,
        quality.name()
    );
    let order = [&GATE_EYES[1], &GATE_EYES[0], &GATE_EYES[2]];
    for gate_eye in order {
        println!();
        println!("### Gate eye {} {:?}", gate_eye.name, gate_eye.eye);
        println!();
        println!(
            "| scale | patches | triangles | vertices issued | tops | +x walls | -x walls | +z \
             walls | -z walls | edge walls | front | back | edge-on | off screen | zero area | \
             moved by the join |"
        );
        println!("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|");
        let mut merges = Vec::new();
        for &scale in &scales {
            let (hole, patches) = select_at(&gen, gate_eye, quality, scale);
            let c = census_of(gate_eye, hole, &patches, true);
            let f = c.frames;
            let kind = |dir: u8, border: Option<bool>| -> f64 {
                c.by_kind
                    .iter()
                    .filter(|((d, b), _)| *d == dir && border.is_none_or(|x| x == *b))
                    .map(|(_, r)| r.total)
                    .sum()
            };
            let edge: f64 = c
                .by_kind
                .iter()
                .filter(|((_, b), _)| *b)
                .map(|(_, r)| r.total)
                .sum();
            let pct = |v: f64| format!("{:.1}%", 100.0 * v / c.all.total);
            println!(
                "| x{scale} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} \
                 | {} |",
                per_frame(c.patches, f),
                per_frame(c.all.total, f),
                per_frame(c.vertices, f),
                per_frame(kind(FarDir::Top as u8, None), f),
                per_frame(kind(FarDir::PosX as u8, Some(false)), f),
                per_frame(kind(FarDir::NegX as u8, Some(false)), f),
                per_frame(kind(FarDir::PosZ as u8, Some(false)), f),
                per_frame(kind(FarDir::NegZ as u8, Some(false)), f),
                per_frame(edge, f),
                pct(c.all.front),
                pct(c.all.back),
                pct(c.all.edge_on),
                pct(c.all.off_screen),
                pct(c.all.zero_area),
                pct(c.all.in_hole),
            );
            merges.push((scale, c.faces_in_view, c.merged, f, c.by_kind.clone()));
        }
        println!();
        println!("Back-facing and off-screen triangles by kind, at each scale:");
        println!();
        println!("| scale | kind | triangles | front | back | edge-on | off screen | zero area |");
        println!("|---|---|--:|--:|--:|--:|--:|--:|");
        for (scale, _, _, f, by_kind) in &merges {
            for ((dir, border), r) in by_kind {
                let pct = |v: f64| format!("{:.0}%", 100.0 * v / r.total);
                println!(
                    "| x{scale} | {:?}{} | {} | {} | {} | {} | {} | {} |",
                    FarDir::ALL[*dir as usize],
                    if *border { " edge" } else { "" },
                    per_frame(r.total, *f),
                    pct(r.front),
                    pct(r.back),
                    pct(r.edge_on),
                    pct(r.off_screen),
                    pct(r.zero_area),
                );
            }
        }
        println!();
        println!("Faces of the patches in view, every direction, under other meshing rules:");
        println!();
        print!("| scale | as built |");
        for (name, _, _) in MERGE_RULES {
            print!(" {name}: tops + walls |");
        }
        println!();
        println!("|---|--:|{}", "--:|".repeat(MERGE_RULES.len()));
        for (scale, faces, merged, f, _) in &merges {
            print!("| x{scale} | {} |", per_frame(*faces, *f));
            for [t, w] in merged {
                print!(
                    " {} + {} = {} ({:+.0}%) |",
                    per_frame(*t, *f),
                    per_frame(*w, *f),
                    per_frame(t + w, *f),
                    100.0 * ((t + w) / faces - 1.0)
                );
            }
            println!();
        }
    }
}

/// A software depth buffer, for [`far_census_visible`]: which triangles own
/// a pixel once everything in front of them is drawn. Depth is `1 / w`,
/// interpolated across the screen as the rasteriser does; larger is nearer.
struct Raster {
    w: usize,
    h: usize,
    depth: Vec<f64>,
    id: Vec<u32>,
}

impl Raster {
    fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            depth: vec![0.0; w * h],
            id: vec![u32::MAX; w * h],
        }
    }

    fn clear(&mut self) {
        self.depth.fill(0.0);
        self.id.fill(u32::MAX);
    }

    /// One triangle in clip space, cut at the near plane.
    fn draw(&mut self, clip: [glam::DVec4; 3], id: u32) {
        const NEAR_W: f64 = 0.05;
        let mut poly: Vec<glam::DVec4> = Vec::with_capacity(4);
        for k in 0..3 {
            let (a, b) = (clip[k], clip[(k + 1) % 3]);
            if a.w >= NEAR_W {
                poly.push(a);
            }
            if (a.w >= NEAR_W) != (b.w >= NEAR_W) {
                let t = (NEAR_W - a.w) / (b.w - a.w);
                poly.push(a + (b - a) * t);
            }
        }
        if poly.len() < 3 {
            return;
        }
        let (w, h) = (self.w as f64, self.h as f64);
        let s: Vec<[f64; 3]> = poly
            .iter()
            .map(|c| {
                let iw = 1.0 / c.w;
                [(c.x * iw * 0.5 + 0.5) * w, (0.5 - c.y * iw * 0.5) * h, iw]
            })
            .collect();
        for k in 1..s.len() - 1 {
            self.tri(s[0], s[k], s[k + 1], id);
        }
    }

    fn tri(&mut self, a: [f64; 3], b: [f64; 3], c: [f64; 3], id: u32) {
        let edge = |p: [f64; 3], q: [f64; 3], x: f64, y: f64| {
            (q[0] - p[0]) * (y - p[1]) - (q[1] - p[1]) * (x - p[0])
        };
        let area = edge(a, b, c[0], c[1]);
        if area == 0.0 || !area.is_finite() {
            return;
        }
        let lo_x = a[0].min(b[0]).min(c[0]).floor().max(0.0);
        let hi_x = a[0].max(b[0]).max(c[0]).ceil().min(self.w as f64 - 1.0);
        let lo_y = a[1].min(b[1]).min(c[1]).floor().max(0.0);
        let hi_y = a[1].max(b[1]).max(c[1]).ceil().min(self.h as f64 - 1.0);
        if lo_x > hi_x || lo_y > hi_y {
            return;
        }
        for py in lo_y as usize..=hi_y as usize {
            let y = py as f64 + 0.5;
            for px in lo_x as usize..=hi_x as usize {
                let x = px as f64 + 0.5;
                let (w0, w1, w2) = (
                    edge(b, c, x, y) / area,
                    edge(c, a, x, y) / area,
                    edge(a, b, x, y) / area,
                );
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                let iw = w0 * a[2] + w1 * b[2] + w2 * c[2];
                let at = py * self.w + px;
                if iw > self.depth[at] {
                    self.depth[at] = iw;
                    self.id[at] = id;
                }
            }
        }
    }
}

/// The ground the voxels draw inside the hole, for the software depth
/// buffer: the far terrain's own patches over the hole (selected as if
/// there were none, so level 3's quads of 8 there -- the coarsest voxel
/// cells, top for top), their faces whose every corner is inside the hole's
/// square of ground, where they are, unjoined. What hides the far terrain
/// behind a line of sight that leaves the hole under the far terrain's
/// skirt on its side: in the game the voxels' ground does, and without a
/// stand-in for it such a line would see what no player sees.
fn voxel_stand_in(
    gen: &WorldGen,
    gate_eye: &GateEye,
    scale: f64,
    hole: far::Hole,
) -> Vec<[Vec3; 4]> {
    let view = far_view_scaled(
        gate_eye.eye.map(f64::from),
        SIZE.1,
        None,
        FarQuality::Medium,
        scale,
    )
    .expect("a quality that draws the far terrain");
    let patches = far::build(gen, &view);
    let (lo, hi) = (hole.min, hole.max);
    let mut out = Vec::new();
    for p in &patches {
        let [ox, oz] = p.key.origin();
        let placement = (
            [ox as f32, oz as f32],
            p.key.quad() as f32,
            p.skirt_blocks() as f32,
        );
        for face in to_far_faces(p) {
            let corners = face.corners().map(|v| place(&v, placement, None, 0.0));
            let inside = corners.iter().all(|c| {
                (c.x as f64) >= lo[0]
                    && (c.x as f64) <= hi[0]
                    && (c.z as f64) >= lo[2]
                    && (c.z as f64) <= hi[2]
            });
            if inside {
                out.push(corners);
            }
        }
    }
    out
}

/// What the cull leaves of the far terrain, and whether it ever leaves out
/// a triangle that shows: at each gate eye and scale, Medium, every fourth
/// of the census's directions -- the triangles the patch-wide cull drew
/// before tiles; those [`Scene::runs`] draws without the horizon and with
/// it, and what the horizon takes per frame; and, for the record, the
/// triangles that own a pixel of a software depth buffer of everything the
/// patch-wide cull drew (an ideal occlusion cull).
///
/// **The check**: every triangle that owns a pixel there is in a run the
/// renderer draws. A cull that leaves one out has hidden ground the eye
/// sees. No voxels are drawn, so nothing of the far terrain is hidden by
/// them.
#[test]
#[ignore]
fn far_census_visible() {
    use cubara_render::far::{facing_runs, group_counts, groups_facing};
    let gen = WorldGen::new(World::new().seed());
    let scales: Vec<f64> = std::env::var("FAR_CENSUS_SCALES")
        .ok()
        .map(|s| s.split(',').filter_map(|v| v.parse().ok()).collect())
        .unwrap_or_else(|| vec![1.5, 2.0, 2.5, 3.0]);
    let eyes: Vec<&GateEye> = match std::env::var("FAR_CENSUS_EYE").as_deref() {
        Ok("hill") => vec![&GATE_EYES[1]],
        Ok("ground") => vec![&GATE_EYES[0]],
        Ok("flight") => vec![&GATE_EYES[2]],
        _ => vec![&GATE_EYES[1], &GATE_EYES[0], &GATE_EYES[2]],
    };
    println!(
        "{}x{}, Medium, tiles of {} quads, every 4th of {YAWS} directions; per frame.",
        SIZE.0,
        SIZE.1,
        cubara_render::far::TILE_QUADS
    );
    let mut raster = Raster::new(SIZE.0 as usize, SIZE.1 as usize);
    let mut horizon = Horizon::default();
    let mut wrongly_left_out = 0.0;
    for gate_eye in eyes {
        println!();
        println!("### Gate eye {} {:?}", gate_eye.name, gate_eye.eye);
        println!();
        println!(
            "| scale | patches | whole patches | tiles | tiles + horizon | horizon, ms | runs | \
             vertices issued | own a pixel |"
        );
        println!("|---|--:|--:|--:|--:|--:|--:|--:|--:|");
        for &scale in &scales {
            let (hole, patches) = select_at(&gen, gate_eye, FarQuality::Medium, scale);
            let eye = Vec3::from(gate_eye.eye);
            let scene = Scene::of(eye, hole, &patches);
            let stand_in = voxel_stand_in(&gen, gate_eye, scale, hole);
            let aspect = SIZE.0 as f32 / SIZE.1 as f32;
            let mut frames = 0.0;
            // patches, whole, tiles, + horizon, ms, runs, vertices, own a pixel
            let mut sum = [0f64; 8];
            for s in (0..YAWS).step_by(4) {
                let frame = 200.0 + (s as f32 + 0.5) * 2000.0 / YAWS as f32;
                let yaw = frame / 240.0 * 0.3;
                let dir = glam::vec3(yaw.cos(), -0.25, yaw.sin());
                let view_proj = CameraUniform::look_view_proj(aspect, eye, dir);
                let m = view_proj.as_dmat4();
                let frustum = Frustum::from_view_proj(view_proj);
                frames += 1.0;
                let in_view = scene.in_view(&frustum);
                sum[0] += in_view.len() as f64;
                // The patch-wide cull, as before tiles: every face of every
                // group that faces the patch's box. Rasterised, front faces
                // only, each triangle under its own id.
                raster.clear();
                // The voxels' ground first, under an id that is nobody's.
                for corners in &stand_in {
                    let clip = corners
                        .map(|p| m * glam::DVec4::new(p.x as f64, p.y as f64, p.z as f64, 1.0));
                    for idx in QUAD_INDICES.chunks(3) {
                        let [a, b, c] = [idx[0], idx[1], idx[2]].map(|v| clip[v as usize]);
                        raster.draw([a, b, c], u32::MAX - 1);
                    }
                }
                let mut ids: Vec<(usize, usize)> = Vec::new();
                for &i in &in_view {
                    let mut runs = Vec::new();
                    facing_runs(
                        group_counts(&scene.faces[i]),
                        groups_facing(&scene.boxes[i], eye),
                        |first, count| runs.push((first as usize, count as usize)),
                    );
                    for (first, count) in runs {
                        for k in first..first + count {
                            let face = &scene.faces[i][k];
                            let mut tris = Vec::with_capacity(2);
                            let mut scratch_census = Census::default();
                            triangles(
                                face,
                                scene.placement[i],
                                scene.hole,
                                view_proj,
                                (eye, scene.hidden),
                                &mut scratch_census,
                                |t| tris.push(t),
                            );
                            // Where `far.wgsl` puts each corner: the join's
                            // place, then the crack seal (`face.wgsl`'s
                            // `CRACK_SEAL`) -- without it a T-junction is a
                            // pinhole here that the GPU never shows.
                            let world = face.corners().map(|v| {
                                let p =
                                    place(&v, scene.placement[i], Some(scene.hole), scene.hidden);
                                p + v.outward() * (CRACK_SEAL * p.distance(eye))
                            });
                            for (t, idx) in tris.iter().zip(QUAD_INDICES.chunks(3)) {
                                let id = ids.len() as u32;
                                ids.push((i, k));
                                if t.fate == Fate::Front {
                                    let clip = [idx[0], idx[1], idx[2]].map(|v| {
                                        let p = world[v as usize];
                                        m * glam::DVec4::new(
                                            p.x as f64, p.y as f64, p.z as f64, 1.0,
                                        )
                                    });
                                    raster.draw(clip, id);
                                }
                            }
                        }
                    }
                }
                sum[1] += ids.len() as f64;
                let mut seen = std::collections::HashSet::new();
                for &id in &raster.id {
                    if id < u32::MAX - 1 {
                        seen.insert(id);
                    }
                }
                sum[7] += seen.len() as f64;
                let tiles = scene.runs(&in_view, (&frustum, eye), FarCulling::Tiles);
                sum[2] += tiles.iter().map(|r| r.2 as f64 * 2.0).sum::<f64>();
                let started = std::time::Instant::now();
                let runs = scene.horizon_runs(&in_view, (&frustum, eye), &mut horizon);
                sum[4] += started.elapsed().as_secs_f64() * 1000.0;
                sum[3] += runs.iter().map(|r| r.2 as f64 * 2.0).sum::<f64>();
                sum[5] += runs.len() as f64;
                let mut per_class = vec![0usize; CLASS_FACES.len()];
                for r in &runs {
                    per_class[class_of(r.2)] += 1;
                }
                sum[6] += issued(&per_class).1 as f64;
                // Every face that owns a pixel is drawn, with the horizon
                // and without.
                for (what, runs) in [("tiles", &tiles), ("tiles + horizon", &runs)] {
                    let drawn: std::collections::HashSet<(usize, usize)> = runs
                        .iter()
                        .flat_map(|&(i, first, count)| {
                            (first as usize..(first + count) as usize).map(move |k| (i, k))
                        })
                        .collect();
                    for &id in &seen {
                        let (i, k) = ids[id as usize];
                        if !drawn.contains(&(i, k)) {
                            wrongly_left_out += 1.0;
                            if wrongly_left_out <= 8.0 {
                                println!(
                                    "LEFT OUT by {what}: eye {}, scale {scale}, direction {s}, \
                                     patch {:?}, face {:?}",
                                    gate_eye.name, patches[i].key, scene.faces[i][k]
                                );
                            }
                        }
                    }
                }
            }
            println!(
                "| x{scale} | {} | {} | {} | {} | {:.3} | {} | {} | {} |",
                per_frame(sum[0], frames),
                per_frame(sum[1], frames),
                per_frame(sum[2], frames),
                per_frame(sum[3], frames),
                sum[4] / frames,
                per_frame(sum[5], frames),
                per_frame(sum[6], frames),
                per_frame(sum[7], frames),
            );
        }
    }
    println!();
    println!("Triangles that own a pixel and are not drawn: {wrongly_left_out}");
    assert_eq!(
        wrongly_left_out, 0.0,
        "the cull left out ground the eye sees"
    );
}

/// **The quad-scale scan `FAR_QUAD_SCALE` is chosen by**: at each gate eye,
/// Medium, the patches in view, triangles and vertices a frame the
/// renderer's cull draws, per scale (`FAR_CENSUS_SCALES`, by default 2 to
/// 3) -- against what whole patches drew at the scale before tiles, 3
/// (`FAR_CENSUS_CULLING=patches` prints that cull at every scale).
#[test]
#[ignore]
fn far_census_scale_scan() {
    let gen = WorldGen::new(World::new().seed());
    let scales: Vec<f64> = std::env::var("FAR_CENSUS_SCALES")
        .ok()
        .map(|s| s.split(',').filter_map(|v| v.parse().ok()).collect())
        .unwrap_or_else(|| (0..=20).map(|k| 2.0 + k as f64 * 0.05).collect());
    println!(
        "{}x{}, Medium, {:?}, {YAWS} directions along the bench's path; per frame.",
        SIZE.0,
        SIZE.1,
        culling()
    );
    println!();
    println!(
        "| scale | hill: patches / triangles / vertices / draws / runs, most / cull ms | ground | \
         flight |"
    );
    println!("|---|--:|--:|--:|");
    for &scale in &scales {
        print!("| x{scale} |");
        for gate_eye in [&GATE_EYES[1], &GATE_EYES[0], &GATE_EYES[2]] {
            let (hole, patches) = select_at(&gen, gate_eye, FarQuality::Medium, scale);
            let c = census_of(gate_eye, hole, &patches, false);
            print!(
                " {} / {} / {} / {:.1} / {} / {:.3} |",
                per_frame(c.patches, c.frames),
                per_frame(c.all.total, c.frames),
                per_frame(c.vertices, c.frames),
                c.draws / c.frames,
                c.most_runs,
                c.cull_ms / c.frames
            );
        }
        println!();
    }
}

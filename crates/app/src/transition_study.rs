//! Issue #380's study (the amendment of 2026-11-20): what a ring edge
//! changes on screen over a straight approach toward a steep mountain,
//! frame by frame -- today's swap against the distance morph
//! (`cubara_world::morph`, CDLOD geomorphing) -- and R6's slide and a
//! dithered fade worked out from the same swaps.
//!
//! **Deterministic.** Every node is meshed on the spot (`build_node`, the
//! game's job) and every frame is drawn with everything uploaded, so a run
//! is a function of the seed, the approach and the schedule alone: no
//! streaming workers, no frame budget, no wall clock.
//!
//! **What one frame's change is.** The camera moves on, so two frames differ
//! everywhere. What one ring edge does is isolated by drawing each frame
//! twice from the same camera: as it is (`after`), and with that edge's part
//! held at the frame before (`before`) -- its swaps undone, and the morph of
//! its finer level measured from the eye of the frame before. Whatever
//! differs is that edge's doing and nothing else's; so the "swapped area"
//! is the whole frame, exactly.
//!
//! The approach is the gate's toward-peak flight (`flight::toward_peak`):
//! from 900 blocks out, level at the summit's height, 24 blocks a second at
//! 60 frames a second for 30 s, at 1920 x 1080. The far terrain is off (the
//! voxels end at the schedule's outer ring): nothing here is about it.
//!
//! Run (GPU; minutes; writes the tables and the PNGs to
//! `$CUBARA_STUDY_OUT`, or to `target/transition-study` without it):
//!
//! ```text
//! cargo test -p cubara --bin cubara transition_study -- --ignored --nocapture
//! ```

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cubara_render::{Aabb, CameraPose, CameraUniform, Frame, Frustum, MeshedNode, MorphParams};
use cubara_render::{Renderer, TreeModel};
use cubara_voxel::{BlockRegistry, ChunkCoord, Mesh, Vertex};
use cubara_world::mesh::build_node;
use cubara_world::morph::morph_bands;
use cubara_world::node::{desired_nodes_3d, NodeKey, RingSchedule, DEFAULT_RING_SCHEDULE};
use cubara_world::{TerrainBlocks, World, WorldGen, WorldSnapshot};

use crate::far_streaming::FAR_VIEW_RADIUS;
use crate::flight::{toward_peak, FLY_SPEED};
use crate::streaming::{morph_params_for, to_node_id, tree_model_for, VERTICAL_LOD_SQUASH};

/// The gate's resolution.
const SIZE: (u32, u32) = (1920, 1080);
/// The flights' frame rate.
const FPS: f32 = 60.0;
/// The toward-peak flight's length (`check-phase-gate.sh`).
const SECONDS: f32 = 30.0;
/// The goldens' per-pixel tolerance (`flight::POP_TOLERANCE`).
const TOLERANCE: i32 = 12;
/// The ring edges, by the coarser level of the pair: 0<->1, 1<->2, 2<->3.
const EDGES: [u32; 3] = [1, 2, 3];
/// A dithered fade's length in frames: R6's 250 ms at 60 Hz (#380 decision
/// 1), the same window a dither would be given.
const FADE_FRAMES: f32 = 15.0;
/// R6's half-fade in frames (#380 amendment E): what its slide spreads a
/// covered height over.
const HALF_FADE_FRAMES: f32 = 7.5;

/// Whether an RGBA pixel shows the clear colour (`flight::NOTHING`).
fn nothing(p: &[u8]) -> bool {
    p[0] >= 250 && p[1] <= 5 && p[2] >= 250
}

/// Rec. 601 luminance, 0..=255 (`flight::luminance`).
fn luminance(p: &[u8]) -> i32 {
    ((299 * p[0] as u32 + 587 * p[1] as u32 + 114 * p[2] as u32) / 1000) as i32
}

/// What changed between two frames from one camera.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Change {
    /// Pixels whose luminance moved by more than [`TOLERANCE`], or that went
    /// between land and nothing (`flight::changed_pixels`).
    pub changed: u32,
    /// Of those, land <-> nothing: a silhouette moving, or a hole.
    pub flips: u32,
    /// The summed luminance change over pixels that are land in both, /255.
    pub luminance: f64,
    /// The largest single pixel's luminance change, land in both.
    pub max_step: i32,
    /// The largest move of the topmost land pixel of any column, in pixels
    /// (`flight::silhouette_shift`).
    pub silhouette: u32,
}

/// [`Change`] from `before` to `after`.
pub fn change(before: &Frame, after: &Frame) -> Change {
    let (w, h) = (after.width as usize, after.height as usize);
    let mut c = Change::default();
    let mut top_b = vec![usize::MAX; w];
    let mut top_a = vec![usize::MAX; w];
    let mut sum = 0u64;
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            let (pb, pa) = (&before.pixels[i..i + 4], &after.pixels[i..i + 4]);
            let (nb, na) = (nothing(pb), nothing(pa));
            if !nb && top_b[x] == usize::MAX {
                top_b[x] = y;
            }
            if !na && top_a[x] == usize::MAX {
                top_a[x] = y;
            }
            if nb != na {
                c.flips += 1;
                c.changed += 1;
                continue;
            }
            if na {
                continue;
            }
            let step = (luminance(pa) - luminance(pb)).abs();
            sum += step as u64;
            c.max_step = c.max_step.max(step);
            if step > TOLERANCE {
                c.changed += 1;
            }
        }
    }
    c.luminance = sum as f64 / 255.0;
    c.silhouette = top_b
        .iter()
        .zip(&top_a)
        .filter(|(b, a)| **b != usize::MAX && **a != usize::MAX)
        .map(|(b, a)| b.abs_diff(*a) as u32)
        .max()
        .unwrap_or(0);
    c
}

/// Pixels that show nothing in `morphed` where `fixed` (the same nodes, the
/// same camera, no morph) shows land, with land in `morphed` within two
/// pixels above **and** below: a gap the morph opened in the ground, as
/// against a silhouette it lowered (sky above).
pub fn holes(morphed: &Frame, fixed: &Frame) -> u32 {
    let (w, h) = (morphed.width as usize, morphed.height as usize);
    let land = |f: &Frame, x: usize, y: usize| !nothing(&f.pixels[(y * w + x) * 4..][..4]);
    let mut count = 0;
    for y in 2..h.saturating_sub(2) {
        for x in 0..w {
            if land(morphed, x, y) || !land(fixed, x, y) {
                continue;
            }
            let above = (1..=2).any(|k| land(morphed, x, y - k));
            let below = (1..=2).any(|k| land(morphed, x, y + k));
            if above && below {
                count += 1;
            }
        }
    }
    count
}

/// Pixels of nothing in `frame` with land within two pixels above **and**
/// below: a gap through the ground, whatever drew it -- the measure of a
/// frame's own sparkles, which [`holes`] is compared against.
pub fn isolated_nothing(frame: &Frame) -> u32 {
    let (w, h) = (frame.width as usize, frame.height as usize);
    let land = |x: usize, y: usize| !nothing(&frame.pixels[(y * w + x) * 4..][..4]);
    let mut count = 0;
    for y in 2..h.saturating_sub(2) {
        for x in 0..w {
            if land(x, y) {
                continue;
            }
            if (1..=2).any(|k| land(x, y - k)) && (1..=2).any(|k| land(x, y + k)) {
                count += 1;
            }
        }
    }
    count
}

/// The swaps between two selections, per ring edge ([`EDGES`] order): the
/// nodes that leave and the nodes that arrive. A node is part of a swap at
/// edge `L` when it is a level-`L` node replaced by its children or the
/// reverse; a node that only enters or leaves the render distance is not.
pub fn edge_swaps(
    before: &HashSet<NodeKey>,
    after: &HashSet<NodeKey>,
) -> [(BTreeSet<NodeKey>, BTreeSet<NodeKey>); 3] {
    let mut out: [(BTreeSet<NodeKey>, BTreeSet<NodeKey>); 3] = Default::default();
    let slot = |level: u32| EDGES.iter().position(|&l| l == level);
    for &n in before.difference(after) {
        let Some(e) = slot(n.level) else { continue };
        let children: Vec<NodeKey> = n
            .children()
            .into_iter()
            .filter(|c| after.contains(c))
            .collect();
        if !children.is_empty() {
            out[e].0.insert(n);
            out[e].1.extend(children);
        }
    }
    for &n in after.difference(before) {
        let Some(e) = slot(n.level) else { continue };
        let children: Vec<NodeKey> = n
            .children()
            .into_iter()
            .filter(|c| before.contains(c))
            .collect();
        if !children.is_empty() {
            out[e].1.insert(n);
            out[e].0.extend(children);
        }
    }
    out
}

/// One node's mesh, kept to be uploaded again whenever a frame wants it.
struct Built {
    origin: [i32; 3],
    scale: f32,
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    face_indices: [u32; 6],
    local_aabb: Aabb,
    trees: Vec<[i32; 3]>,
    model: TreeModel,
}

/// The world and every node meshed so far.
struct Scene {
    snapshot: WorldSnapshot,
    registry: Arc<BlockRegistry>,
    layer_of: Box<dyn Fn(&str) -> u32>,
    blocks: TerrainBlocks,
    cache: HashMap<NodeKey, Option<Arc<Built>>>,
}

impl Scene {
    fn built(&mut self, node: NodeKey) -> Option<Arc<Built>> {
        if let Some(b) = self.cache.get(&node) {
            return b.clone();
        }
        let b = build_node(
            &self.snapshot,
            &self.registry,
            &*self.layer_of,
            node,
            self.blocks,
        );
        let built = b.geometry.map(|g| {
            Arc::new(Built {
                origin: g.origin,
                scale: g.scale,
                vertices: g.mesh.vertices,
                indices: g.mesh.indices,
                face_indices: g.mesh.face_indices,
                local_aabb: g.local_aabb,
                trees: b.trees.iter().map(|t| [t.x, t.base, t.z]).collect(),
                model: tree_model_for(node.level),
            })
        });
        self.cache.insert(node, built.clone());
        built
    }

    fn meshed(&mut self, node: NodeKey) -> Option<MeshedNode> {
        let b = self.built(node)?;
        Some(MeshedNode {
            id: to_node_id(node),
            origin: b.origin,
            scale: b.scale,
            mesh: Mesh {
                vertices: b.vertices.clone(),
                indices: b.indices.clone(),
                face_indices: b.face_indices,
            },
            local_aabb: b.local_aabb,
            trees: b.trees.clone(),
            tree_model: b.model,
        })
    }
}

/// The renderer and what it holds.
struct Viewer {
    renderer: Renderer,
    resident: BTreeSet<NodeKey>,
    look: glam::Vec3,
}

impl Viewer {
    /// Make `want` exactly what is resident.
    fn show(&mut self, scene: &mut Scene, want: &BTreeSet<NodeKey>) {
        let remove: Vec<_> = self
            .resident
            .difference(want)
            .map(|&n| to_node_id(n))
            .collect();
        let add: Vec<MeshedNode> = want
            .difference(&self.resident)
            .filter_map(|&n| scene.meshed(n))
            .collect();
        self.renderer.apply_node_updates(remove, add);
        self.resident = want.clone();
    }

    /// Draw from `eye`, each level's morph measured from `morph_eyes`, with
    /// everything uploaded.
    fn draw(&mut self, eye: [f32; 3], morph_eyes: [[f32; 3]; 4]) -> Frame {
        let d = |e: [f32; 3]| glam::DVec3::new(e[0] as f64, e[1] as f64, e[2] as f64);
        let pose = CameraPose {
            eye: d(eye),
            look_dir: self.look,
        };
        self.renderer.set_morph_eyes(Some(morph_eyes.map(d)));
        loop {
            self.renderer.render(
                pose,
                None,
                None,
                &[],
                cubara_render::Hud {
                    hotbar: None,
                    panel: None,
                    health: None,
                    crosshair: false,
                    menu: None,
                },
                FAR_VIEW_RADIUS as f32,
            );
            if self.renderer.uploads_pending() == 0 {
                break;
            }
        }
        self.renderer
            .read_frame()
            .expect("an offscreen renderer has a frame")
    }
}

/// The approach: every frame's eye, and the one direction looked in.
struct Approach {
    eyes: Vec<[f32; 3]>,
    look: [f32; 3],
    peak: [f32; 3],
}

/// How high above the ground the walking eye is: a player's eye over the
/// top face of the block they stand on (`cubara_sim`'s 1.62, rounded).
const EYE_HEIGHT: f32 = 1.6;

/// Which approach: the gate's toward-peak flight, level with the summit, or
/// the same line walked on the ground (`CUBARA_STUDY_PATH=walk`) -- the eye
/// a player's height over the ground under it, looking at the summit from
/// where the walk starts. Both at the flight's 24 blocks a second (walking
/// is about a sixth of that, which only the morph's per-frame step depends
/// on: it scales with the speed).
fn approach(gen: &WorldGen) -> Approach {
    let (start, look) = toward_peak(gen);
    // `CUBARA_STUDY_SECONDS` shortens the approach for a quick look.
    let seconds = std::env::var("CUBARA_STUDY_SECONDS")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(SECONDS);
    let walk = std::env::var("CUBARA_STUDY_PATH").is_ok_and(|p| p == "walk");
    let frames = (seconds * FPS).round() as usize;
    let step = FLY_SPEED / FPS;
    // `toward_peak` starts 900 blocks out, level with the summit (+3).
    let peak = [
        start[0] + look[0] * 900.0,
        start[1] - 3.0,
        start[2] + look[2] * 900.0,
    ];
    let eyes: Vec<[f32; 3]> = (0..=frames)
        .map(|i| {
            let t = i as f32 * step;
            let (x, z) = (start[0] + look[0] * t, start[2] + look[2] * t);
            let y = if walk {
                gen.surface_height(x.floor() as i32, z.floor() as i32) as f32 + 1.0 + EYE_HEIGHT
            } else {
                start[1]
            };
            [x, y, z]
        })
        .collect();
    let look = if walk {
        let e = eyes[0];
        glam::Vec3::new(peak[0] - e[0], peak[1] - e[1], peak[2] - e[2])
            .normalize()
            .to_array()
    } else {
        look
    };
    Approach { eyes, look, peak }
}

/// What one variant measured at one edge, over the approach.
#[derive(Clone, Debug, Default)]
struct EdgeTally {
    /// Per frame (index into the approach), the change.
    frames: Vec<(usize, Change)>,
    /// How many swaps (coarse nodes) the edge made in view.
    swaps: usize,
}

impl EdgeTally {
    fn worst(&self) -> Option<(usize, Change)> {
        self.frames
            .iter()
            .copied()
            .max_by_key(|(i, c)| (c.changed, std::cmp::Reverse(*i)))
    }
    fn total_changed(&self) -> u64 {
        self.frames.iter().map(|(_, c)| c.changed as u64).sum()
    }
    fn total_luminance(&self) -> f64 {
        self.frames.iter().map(|(_, c)| c.luminance).sum()
    }
    fn max_step(&self) -> i32 {
        self.frames
            .iter()
            .map(|(_, c)| c.max_step)
            .max()
            .unwrap_or(0)
    }
    fn max_silhouette(&self) -> u32 {
        self.frames
            .iter()
            .map(|(_, c)| c.silhouette)
            .max()
            .unwrap_or(0)
    }
    fn max_luminance(&self) -> f64 {
        self.frames
            .iter()
            .map(|(_, c)| c.luminance)
            .fold(0.0, f64::max)
    }
    fn frames_over_tolerance(&self) -> usize {
        self.frames.iter().filter(|(_, c)| c.changed > 0).count()
    }
}

/// One run's results.
struct Run {
    tallies: [EdgeTally; 3],
    /// The worst frame's pair per edge: (frame, pixels changed, before, after).
    kept: [Option<(usize, u32, Frame, Frame)>; 3],
    /// Holes the morph opened: (frames checked, frames with any, most in
    /// one, isolated nothing in the morphed frames, in the same frames as
    /// meshed -- summed over the frames checked).
    holes: (usize, usize, u32, u64, u64),
    /// The approach strip: (frame, morphed, as meshed) at evenly spaced frames.
    strip: Vec<(usize, Frame, Frame)>,
    /// Swaps per edge with their eye, for the height-only numbers.
    swap_events: Vec<(u32, NodeKey, [f32; 3])>,
    /// The pair at each `focus` frame asked for: (edge, frame, before, after).
    focused: Vec<(usize, usize, Frame, Frame)>,
    /// What the flight's own motion changes from one frame to the next
    /// (pixels over the tolerance), every thirtieth frame: the scale the
    /// level-of-detail changes are seen against.
    motion: Vec<u32>,
}

/// Run the approach under `schedule`, morphing (`morph`) or not, keeping
/// each edge's pair at its `focus` frame as well as its worst.
fn run(
    scene: &mut Scene,
    viewer: &mut Viewer,
    schedule: &RingSchedule,
    morph: bool,
    focus: [Option<usize>; 3],
) -> Run {
    let gen = WorldGen::new(World::new().seed());
    let path = approach(&gen);
    let params = if morph {
        morph_params_for(schedule, VERTICAL_LOD_SQUASH)
    } else {
        MorphParams::OFF
    };
    viewer.renderer.set_morph(params);
    let aspect = SIZE.0 as f32 / SIZE.1 as f32;
    let mut selections: HashMap<ChunkCoord, HashSet<NodeKey>> = HashMap::new();
    let mut select = |eye: [f32; 3]| {
        let c = ChunkCoord::from_world_pos(eye);
        selections
            .entry(c)
            .or_insert_with(|| {
                desired_nodes_3d(c, VERTICAL_LOD_SQUASH, schedule)
                    .into_iter()
                    .collect()
            })
            .clone()
    };
    // What a frame draws of a selection: what its view can show, with room
    // above and below for what the morph moves.
    let view = |eye: [f32; 3], nodes: &HashSet<NodeKey>| -> BTreeSet<NodeKey> {
        let vp = CameraUniform::look_view_proj(aspect, glam::Vec3::from(eye), viewer_look(&path));
        let frustum = Frustum::from_view_proj(vp);
        nodes
            .iter()
            .copied()
            .filter(|n| {
                let o = n.world_origin();
                let size = 16.0 * n.extent_chunks() as f32;
                let min = glam::vec3(o[0] as f32, o[1] as f32 - 64.0, o[2] as f32);
                let max = min + glam::vec3(size, size + 128.0, size);
                frustum.intersects_aabb(&Aabb::new(min, max))
            })
            .collect()
    };

    let mut tallies: [EdgeTally; 3] = Default::default();
    let mut kept: [Option<(usize, u32, Frame, Frame)>; 3] = [None, None, None];
    let mut holes_seen = (0usize, 0usize, 0u32, 0u64, 0u64);
    let mut strip = Vec::new();
    let mut swap_events = Vec::new();
    let mut motion = Vec::new();
    let mut focused = Vec::new();
    let mut previous: Option<Frame> = None;
    let strip_at: Vec<usize> = (0..6).map(|k| k * (path.eyes.len() - 1) / 5).collect();

    let mut prev_sel = select(path.eyes[0]);
    let first = view(path.eyes[0], &prev_sel);
    viewer.show(scene, &first);
    for i in 1..path.eyes.len() {
        let (e0, e) = (path.eyes[i - 1], path.eyes[i]);
        let sel = select(e);
        let swaps = edge_swaps(&prev_sel, &sel);
        let now = view(e, &sel);
        let mut before: [Option<Frame>; 3] = [None, None, None];
        for (k, (out, inn)) in swaps.iter().enumerate() {
            if out.is_empty() && inn.is_empty() {
                continue;
            }
            let in_view: Vec<NodeKey> = out
                .iter()
                .filter(|n| n.level == EDGES[k])
                .chain(inn.iter().filter(|n| n.level == EDGES[k]))
                .copied()
                .filter(|n| now.contains(n) || view(e, &HashSet::from([*n])).contains(n))
                .collect();
            for n in &in_view {
                swap_events.push((EDGES[k], *n, e));
            }
            tallies[k].swaps += in_view.len();
            let held: BTreeSet<NodeKey> = now
                .difference(inn)
                .copied()
                .chain(view(e, &out.iter().copied().collect()))
                .collect();
            viewer.show(scene, &held);
            let mut eyes = [e; 4];
            eyes[(EDGES[k] - 1) as usize] = e0;
            before[k] = Some(viewer.draw(e, eyes));
        }
        viewer.show(scene, &now);
        let after = viewer.draw(e, [e; 4]);
        for k in 0..3 {
            if morph && before[k].is_none() {
                let mut eyes = [e; 4];
                eyes[(EDGES[k] - 1) as usize] = e0;
                before[k] = Some(viewer.draw(e, eyes));
            }
            let Some(b) = before[k].take() else { continue };
            let c = change(&b, &after);
            if focus[k] == Some(i) {
                focused.push((k, i, b.clone(), after.clone()));
            }
            if c.changed > 0 || c.luminance > 0.0 {
                tallies[k].frames.push((i, c));
            }
            let worse = kept[k].as_ref().is_none_or(|(_, n, _, _)| c.changed > *n);
            if worse && c.changed > 0 {
                kept[k] = Some((i, c.changed, b, after.clone()));
            }
        }
        // Holes, every fourth frame, and the strip: the same nodes drawn
        // as meshed.
        let check_holes = morph && i % 4 == 0;
        let in_strip = strip_at.contains(&i);
        if check_holes || in_strip {
            viewer.renderer.set_morph(MorphParams::OFF);
            let fixed = viewer.draw(e, [e; 4]);
            viewer.renderer.set_morph(params);
            if check_holes {
                let h = holes(&after, &fixed);
                holes_seen.0 += 1;
                if h > 0 {
                    holes_seen.1 += 1;
                }
                holes_seen.2 = holes_seen.2.max(h);
                holes_seen.3 += isolated_nothing(&after) as u64;
                holes_seen.4 += isolated_nothing(&fixed) as u64;
            }
            if in_strip {
                strip.push((i, after.clone(), fixed));
            }
        }
        if let Some(p) = previous.take() {
            motion.push(change(&p, &after).changed);
        }
        if i % 30 == 29 {
            previous = Some(after);
        }
        prev_sel = sel;
        if i % 300 == 0 {
            eprintln!(
                "transition study: frame {i} of {}, {} nodes meshed",
                path.eyes.len() - 1,
                scene.cache.len()
            );
        }
    }
    Run {
        tallies,
        kept,
        holes: holes_seen,
        strip,
        swap_events,
        focused,
        motion,
    }
}

fn viewer_look(path: &Approach) -> glam::Vec3 {
    glam::Vec3::from(path.look).normalize()
}

/// Where the study writes its files.
fn out_dir() -> PathBuf {
    std::env::var_os("CUBARA_STUDY_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/transition-study")
        })
}

fn save(path: &Path, frame: &Frame) {
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    image::save_buffer(
        path,
        &frame.pixels,
        frame.width,
        frame.height,
        image::ExtendedColorType::Rgba8,
    )
    .unwrap_or_else(|e| panic!("could not write {}: {e}", path.display()));
}

/// `after` with every pixel that changed from `before` painted: red where the
/// luminance moved by more than the tolerance, yellow where land and nothing
#[allow(clippy::chunks_exact_to_as_chunks)] // the index also addresses `before`/`after`
/// traded places, the rest dimmed.
fn diff_image(before: &Frame, after: &Frame) -> Frame {
    let mut out = after.clone();
    for (i, px) in out.pixels.chunks_exact_mut(4).enumerate() {
        let (pb, pa) = (
            &before.pixels[i * 4..i * 4 + 4],
            &after.pixels[i * 4..i * 4 + 4],
        );
        let flip = nothing(pb) != nothing(pa);
        let step = (luminance(pa) - luminance(pb)).abs();
        if flip {
            px.copy_from_slice(&[255, 230, 0, 255]);
        } else if !nothing(pa) && step > TOLERANCE {
            px.copy_from_slice(&[255, 0, 0, 255]);
        } else {
            for c in &mut px[..3] {
                *c /= 3;
            }
        }
    }
    out
}

/// Height-only numbers for one swap (`lod_probe::r6`'s arithmetic, on the
/// rule of #432): per finer cell of the coarse node's footprint in view,
/// `d` = finer top - coarser top, and in pixels at that cell's distance:
/// the swap's largest step, R6's largest single-frame step (what its slide
/// of one row leaves to pop, plus a half-fade's frame of what it covers),
/// and the morph's per-frame step at the flight's speed across its band.
fn height_steps(
    snapshot: &WorldSnapshot,
    coarse: NodeKey,
    eye: [f32; 3],
    band: Option<f32>,
) -> Option<(f32, f32, f32, i32)> {
    let fine = coarse.level.checked_sub(1)?;
    let (wf, wc) = (1i32 << fine, 1i32 << coarse.level);
    let o = coarse.world_origin();
    let extent = 16 * coarse.extent_chunks();
    // R6's slide: one stored row of the coarser level (#380 decision 2).
    let delta = coarse.cell_height_blocks();
    let pixel = std::f32::consts::FRAC_PI_3 / SIZE.1 as f32;
    let (mut swap, mut r6, mut morph, mut max_d) = (0f32, 0f32, 0f32, 0i32);
    let mut any = false;
    for z in (0..extent).step_by(wf as usize) {
        for x in (0..extent).step_by(wf as usize) {
            let (cx, cz) = (o[0] + x, o[2] + z);
            let f = snapshot.cell_surface_height(cx, cz, wf);
            if f < o[1] || f >= o[1] + extent {
                continue;
            }
            let c =
                snapshot.cell_surface_height(cx.div_euclid(wc) * wc, cz.div_euclid(wc) * wc, wc);
            let d = f - c;
            let centre = [
                cx as f32 + wf as f32 * 0.5,
                f as f32 + 1.0,
                cz as f32 + wf as f32 * 0.5,
            ];
            let dist = ((centre[0] - eye[0]).powi(2)
                + (centre[1] - eye[1]).powi(2)
                + (centre[2] - eye[2]).powi(2))
            .sqrt()
            .max(1.0);
            let px = |blocks: f32| blocks / (dist * pixel);
            let a = d.abs();
            any = true;
            max_d = max_d.max(a);
            swap = swap.max(px(a as f32));
            r6 = r6.max(px(
                (a - delta).max(0) as f32 + a.min(delta) as f32 / HALF_FADE_FRAMES
            ));
            if let Some(b) = band {
                // The morph moves `d` over `b` blocks of the metric, at the
                // flight's 0.4 blocks a frame; at the band, about as far out
                // as the swap.
                morph = morph.max(px(a as f32 * (FLY_SPEED / FPS) / b));
            }
        }
    }
    any.then_some((swap, r6, morph, max_d))
}

fn edge_name(level: u32, schedule: &RingSchedule) -> String {
    let radius = schedule
        .iter()
        .find(|&&(l, _)| l == level - 1)
        .map_or(0, |&(_, r)| r * 16);
    format!("{}<->{} ({radius} m)", level - 1, level)
}

/// The study's report for one schedule, and its pictures.
fn study(name: &str, schedule: &RingSchedule) -> String {
    let world = World::new();
    let camera = CameraPose {
        eye: glam::DVec3::ZERO,
        look_dir: glam::Vec3::X,
    };
    let Some((mut renderer, assets)) = Renderer::offscreen(SIZE.0, SIZE.1, camera) else {
        return "no GPU adapter: nothing measured".to_string();
    };
    renderer.set_clear_color(wgpu::Color {
        r: 1.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    });
    let layers = assets.layers;
    let registry = Arc::new(assets.registry);
    // `CUBARA_STUDY_NO_TREES` grows no trees at any level: what is left of
    // a swap's change without them.
    let trees = std::env::var_os("CUBARA_STUDY_NO_TREES").is_none();
    let blocks = TerrainBlocks::from_registry(&registry);
    let blocks = if trees {
        blocks.with_oak(&crate::game::load_structure_registry(), &registry)
    } else {
        blocks
    }
    .with_ores(&crate::game::load_ore_registry(), &registry);
    let mut scene = Scene {
        snapshot: world.snapshot(),
        registry,
        layer_of: Box::new(move |n: &str| layers.layer_of(n)),
        blocks,
        cache: HashMap::new(),
    };
    let gen = WorldGen::new(world.seed());
    let path = approach(&gen);
    let mut viewer = Viewer {
        renderer,
        resident: BTreeSet::new(),
        look: viewer_look(&path),
    };
    let dir = out_dir().join(name);

    let swap = run(&mut scene, &mut viewer, schedule, false, [None; 3]);
    viewer.show(&mut scene, &BTreeSet::new());
    let focus = [0, 1, 2].map(|k| swap.kept[k].as_ref().map(|(i, _, _, _)| *i));
    let morph = run(&mut scene, &mut viewer, schedule, true, focus);

    let bands = morph_bands(schedule, VERTICAL_LOD_SQUASH);
    let mut s = String::new();
    let _ = writeln!(
        s,
        "## {name}: schedule {schedule:?}, bands {:?}",
        bands.map(|b| b.map(|b| (b.start, b.end)))
    );
    let _ = writeln!(
        s,
        "approach: {}, peak at {:?}, from {:?} to {:?}, look {:?}, {} frames at {FPS} fps, {} blocks a frame",
        if std::env::var("CUBARA_STUDY_PATH").is_ok_and(|p| p == "walk") {
            "walked on the ground toward the summit"
        } else {
            "the toward-peak flight, level with the summit"
        },
        path.peak,
        path.eyes[0],
        path.eyes[path.eyes.len() - 1],
        path.look,
        path.eyes.len() - 1,
        FLY_SPEED / FPS
    );
    let _ = writeln!(
        s,
        "\n| edge | variant | swaps in view | frames that change | largest single-frame change (px over tol.) | its luminance (px-equiv.) | largest pixel step /255 | largest silhouette move px | total over the approach (px over tol.) | total luminance (px-equiv.) |"
    );
    let _ = writeln!(s, "|---|---|---|---|---|---|---|---|---|---|");
    for (k, &level) in EDGES.iter().enumerate() {
        for (label, r) in [("today's swap", &swap), ("morph", &morph)] {
            let t = &r.tallies[k];
            let worst = t.worst();
            let _ = writeln!(
                s,
                "| {} | {label} | {} | {} | {} (frame {}) | {:.0} | {} | {} | {} | {:.0} |",
                edge_name(level, schedule),
                t.swaps,
                t.frames_over_tolerance(),
                worst.map_or(0, |w| w.1.changed),
                worst.map_or(0, |w| w.0),
                t.max_luminance(),
                t.max_step(),
                t.max_silhouette(),
                t.total_changed(),
                t.total_luminance()
            );
        }
        // A dithered fade, worked out from the swap: every changed pixel
        // switches once, at its own frame of FADE_FRAMES -- the same pixels
        // and the same steps, a fifteenth of them a frame.
        let t = &swap.tallies[k];
        let worst = t.worst().map_or(0, |w| w.1.changed);
        let _ = writeln!(
            s,
            "| {} | dither (computed) | {} | x{FADE_FRAMES} | {:.0} | {:.0} | {} | -- | {} | {:.0} |",
            edge_name(level, schedule),
            t.swaps,
            worst as f32 / FADE_FRAMES,
            t.max_luminance() / FADE_FRAMES as f64,
            t.max_step(),
            t.total_changed(),
            t.total_luminance()
        );
    }
    let _ = writeln!(
        s,
        "\nholes the morph opened (a pixel of nothing with land above and below, where the same nodes unmorphed show land): {} frames checked, {} with any, at most {} pixels; isolated nothing over those frames: {} morphed, {} as meshed (the mesher's own sparkles)",
        morph.holes.0, morph.holes.1, morph.holes.2, morph.holes.3, morph.holes.4
    );

    if !swap.motion.is_empty() {
        let mut m = swap.motion.clone();
        m.sort_unstable();
        let _ = writeln!(
            s,
            "\nthe flight's own motion, frame to frame (no level of detail changed, pixels over the tolerance): median {}, min {}, max {} of {} frames sampled",
            m[m.len() / 2],
            m[0],
            m[m.len() - 1],
            m.len()
        );
    }

    // Height-only, per swap in view (both runs saw the same swaps).
    let _ = writeln!(
        s,
        "\nheight only, per swap in view: d = finer top - coarser top over the coarse node's footprint, in pixels at each cell's distance (1080 rows, 60 degrees)"
    );
    let _ = writeln!(
        s,
        "| edge | swaps | max abs d blocks p50/max | swap: largest step px p50/p95/max | R6 (Delta one row, slide over 7.5 frames): largest single-frame step px p50/p95/max | morph: largest per-frame step px p50/p95/max |"
    );
    let _ = writeln!(s, "|---|---|---|---|---|---|");
    for &level in &EDGES {
        let band = bands[(level - 1) as usize].map(|b| b.end - b.start);
        let mut rows: Vec<(f32, f32, f32, i32)> = swap
            .swap_events
            .iter()
            .filter(|(l, n, _)| *l == level && n.level == level)
            .filter_map(|(_, n, eye)| height_steps(&scene.snapshot, *n, *eye, band))
            .collect();
        if rows.is_empty() {
            let _ = writeln!(
                s,
                "| {} | 0 | -- | -- | -- | -- |",
                edge_name(level, schedule)
            );
            continue;
        }
        let pct = |v: &mut Vec<f32>, q: f32| -> f32 {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[((v.len() - 1) as f32 * q).round() as usize]
        };
        let mut sw: Vec<f32> = rows.iter().map(|r| r.0).collect();
        let mut r6: Vec<f32> = rows.iter().map(|r| r.1).collect();
        let mut mo: Vec<f32> = rows.iter().map(|r| r.2).collect();
        let mut md: Vec<f32> = rows.iter().map(|r| r.3 as f32).collect();
        rows.clear();
        let _ = writeln!(
            s,
            "| {} | {} | {:.0}/{:.0} | {:.1}/{:.1}/{:.1} | {:.1}/{:.1}/{:.1} | {} |",
            edge_name(level, schedule),
            sw.len(),
            pct(&mut md, 0.5),
            pct(&mut md, 1.0),
            pct(&mut sw, 0.5),
            pct(&mut sw, 0.95),
            pct(&mut sw, 1.0),
            pct(&mut r6, 0.5),
            pct(&mut r6, 0.95),
            pct(&mut r6, 1.0),
            if band.is_some() {
                format!(
                    "{:.2}/{:.2}/{:.2}",
                    pct(&mut mo, 0.5),
                    pct(&mut mo, 0.95),
                    pct(&mut mo, 1.0)
                )
            } else {
                "no band: swaps".to_string()
            },
        );
    }

    // Pictures.
    for (k, &level) in EDGES.iter().enumerate() {
        let tag = format!("edge{}{}", level - 1, level);
        if let Some((i, _, b, a)) = &swap.kept[k] {
            save(&dir.join(format!("{tag}_swap_f{i:04}_before.png")), b);
            save(&dir.join(format!("{tag}_swap_f{i:04}_after.png")), a);
            save(
                &dir.join(format!("{tag}_swap_f{i:04}_diff.png")),
                &diff_image(b, a),
            );
        }
        if let Some((i, _, b, a)) = &morph.kept[k] {
            save(&dir.join(format!("{tag}_morph_f{i:04}_before.png")), b);
            save(&dir.join(format!("{tag}_morph_f{i:04}_after.png")), a);
            save(
                &dir.join(format!("{tag}_morph_f{i:04}_diff.png")),
                &diff_image(b, a),
            );
        }
    }
    for (k, i, b, a) in &morph.focused {
        let tag = format!("edge{}{}", EDGES[*k] - 1, EDGES[*k]);
        let c = change(b, a);
        save(
            &dir.join(format!("{tag}_morph_at_swap_f{i:04}_before.png")),
            b,
        );
        save(
            &dir.join(format!("{tag}_morph_at_swap_f{i:04}_after.png")),
            a,
        );
        save(
            &dir.join(format!("{tag}_morph_at_swap_f{i:04}_diff.png")),
            &diff_image(b, a),
        );
        let _ = writeln!(
            s,
            "\nmorph at the swap's worst frame, edge {}: frame {i}, {} pixels over the tolerance ({} land<->nothing), largest step {}",
            edge_name(EDGES[*k], schedule),
            c.changed,
            c.flips,
            c.max_step
        );
    }
    for (i, morphed, fixed) in &morph.strip {
        save(&dir.join(format!("strip_f{i:04}_morph.png")), morphed);
        save(&dir.join(format!("strip_f{i:04}_static.png")), fixed);
    }
    // Per frame, for plotting: the change each edge made, both variants.
    let mut csv = String::from("frame,edge,variant,changed,flips,luminance,max_step,silhouette\n");
    for (label, r) in [("swap", &swap), ("morph", &morph)] {
        for (k, &level) in EDGES.iter().enumerate() {
            for (i, c) in &r.tallies[k].frames {
                let _ = writeln!(
                    csv,
                    "{i},{},{label},{},{},{:.2},{},{}",
                    level, c.changed, c.flips, c.luminance, c.max_step, c.silhouette
                );
            }
        }
    }
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(dir.join("frames.csv"), csv);
    let _ = std::fs::write(dir.join("report.md"), &s);
    s
}

/// The rings brought nearer, as a weak PC's quality setting might: every
/// radius halved. **A stand-in, chosen, not derived**: no quality setting
/// moves the voxel rings today (they are `DEFAULT_RING_SCHEDULE` at every
/// quality), and which schedule a weak PC gets is the owner's to decide.
const HALF_SCHEDULE: &[(u32, i32)] = &[(0, 5), (1, 9), (2, 16), (3, 32)];

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(rows: &[&str]) -> Frame {
        let (w, h) = (rows[0].len() as u32, rows.len() as u32);
        let mut pixels = Vec::new();
        for r in rows {
            for ch in r.chars() {
                pixels.extend_from_slice(match ch {
                    '.' => &[255, 0, 255, 255], // nothing
                    'g' => &[60, 140, 50, 255],
                    'G' => &[90, 200, 80, 255],
                    _ => &[100, 100, 100, 255],
                });
            }
        }
        Frame {
            width: w,
            height: h,
            pixels,
        }
    }

    #[test]
    fn a_change_counts_steps_over_the_tolerance_and_flips_and_the_silhouette() {
        let before = frame(&["....", "gg..", "gggg", "gggg"]);
        let after = frame(&["....", "....", "ggGg", "gggg"]);
        let c = change(&before, &after);
        assert_eq!(c.flips, 2, "the two pixels of the ridge that dropped");
        assert_eq!(c.changed, 3, "the flips and the one brightened pixel");
        assert_eq!(c.silhouette, 1);
        assert!(c.max_step > TOLERANCE);
    }

    #[test]
    fn a_hole_is_nothing_between_land_and_a_lowered_ridge_is_not() {
        let fixed = frame(&["....", "gggg", "gggg", "gggg", "gggg", "gggg"]);
        // Column 0: the top row lowered (sky above): not a hole. Column 2:
        // a pixel of nothing with land above and below: a hole.
        let morphed = frame(&["....", ".ggg", "gg.g", "gggg", "gggg", "gggg"]);
        assert_eq!(holes(&morphed, &fixed), 1);
    }

    #[test]
    fn swaps_are_split_by_edge_and_a_plain_departure_is_none() {
        let parent = NodeKey::new(2, [0, 0, 0]);
        let before: HashSet<NodeKey> = [parent, NodeKey::new(1, [9, 0, 0])].into();
        let mut after: HashSet<NodeKey> = parent.children().into_iter().collect();
        after.insert(NodeKey::new(0, [40, 0, 0]));
        let swaps = edge_swaps(&before, &after);
        assert!(swaps[0].0.is_empty() && swaps[0].1.is_empty());
        assert_eq!(swaps[1].0, BTreeSet::from([parent]));
        assert_eq!(swaps[1].1.len(), 8);
        assert!(swaps[2].0.is_empty());
        // The reverse is a merge at the same edge.
        let back = edge_swaps(&after, &before);
        assert_eq!(back[1].1, BTreeSet::from([parent]));
        assert_eq!(back[1].0.len(), 8);
    }

    /// Where a pixel of the toward-peak flight looks, and what the morphed
    /// mesh draws around it: a debugging aid for a crack found on screen
    /// (`CRACK_AT=frame,x,y`, default the one of the first flights).
    #[test]
    #[ignore = "a debugging aid: CRACK_AT=frame,x,y cargo test -p cubara --bin cubara locate_crack -- --ignored --nocapture"]
    fn locate_crack() {
        use cubara_world::morph::{morph_factor, morph_metric};
        let at: Vec<f32> = std::env::var("CRACK_AT")
            .unwrap_or_else(|_| "1,1720,848".into())
            .split(',')
            .filter_map(|p| p.trim().parse().ok())
            .collect();
        let (frame, px, py) = (at[0], at[1], at[2]);
        let world = World::new();
        let snapshot = world.snapshot();
        let gen = WorldGen::new(world.seed());
        let (start, look) = toward_peak(&gen);
        let t = frame * FLY_SPEED / FPS;
        let eye = glam::Vec3::from(start) + glam::Vec3::from(look) * t;
        let aspect = SIZE.0 as f32 / SIZE.1 as f32;
        let vp = CameraUniform::look_view_proj(aspect, eye, glam::Vec3::from(look).normalize());
        let inv = vp.inverse();
        let ndc = glam::vec2(
            (px + 0.5) / SIZE.0 as f32 * 2.0 - 1.0,
            1.0 - (py + 0.5) / SIZE.1 as f32 * 2.0,
        );
        let far = inv * glam::vec4(ndc.x, ndc.y, 0.5, 1.0);
        let dir = (far.truncate() / far.w - eye).normalize();
        let mut hit = None;
        let mut d = 0.0f32;
        while d < 3000.0 {
            let p = eye + dir * d;
            if p.y <= snapshot.surface_height(p.x.floor() as i32, p.z.floor() as i32) as f32 + 1.0 {
                hit = Some(p);
                break;
            }
            d += 0.02;
        }
        let hit = hit.expect("the ray meets the ground");
        println!("eye {eye:?} dir {dir:?}: level 0's ground at {hit:?}, {d:.1} blocks out");
        let schedule = DEFAULT_RING_SCHEDULE;
        let bands = morph_bands(schedule, VERTICAL_LOD_SQUASH);
        let nodes = desired_nodes_3d(
            ChunkCoord::from_world_pos(eye.to_array()),
            VERTICAL_LOD_SQUASH,
            schedule,
        );
        let registry = cubara_render::load_registry();
        let blocks = TerrainBlocks::from_registry(&registry)
            .with_oak(&crate::game::load_structure_registry(), &registry)
            .with_ores(&crate::game::load_ore_registry(), &registry);
        let layer = |_: &str| 0u32;
        let near = |n: &NodeKey| {
            let o = n.world_origin();
            let s = 16.0 * n.extent_chunks() as f32;
            (0..3).all(|k| hit[k] >= o[k] as f32 - 8.0 && hit[k] <= o[k] as f32 + s + 8.0)
        };
        for n in nodes.iter().filter(|n| near(n)) {
            let band = bands[n.level as usize];
            println!("node {n:?} origin {:?} band {band:?}", n.world_origin());
            if let Some(c) = cubara_world::morph::node_columns(&snapshot, *n, blocks) {
                let w = 1 << n.level;
                let o = n.world_origin();
                let (cx, cz) = (
                    ((hit.x as i32 - o[0]).div_euclid(w)),
                    ((hit.z as i32 - o[2]).div_euclid(w)),
                );
                for dz in -1..=1 {
                    let row: Vec<String> = (-1..=1)
                        .map(|dx| {
                            let (x, z) = (cx + dx, cz + dz);
                            if (-1..=16).contains(&x) && (-1..=16).contains(&z) {
                                let [t, g, f, tf] = c.at(x, z);
                                format!(
                                    "({x},{z}) top {} target {} floor {} tfloor {}",
                                    t + o[1],
                                    g + o[1],
                                    f + o[1],
                                    tf + o[1]
                                )
                            } else {
                                "--".to_string()
                            }
                        })
                        .collect();
                    println!("  columns: {}", row.join(" | "));
                }
            }
            let Some(g) = cubara_world::mesh::mesh_node(&snapshot, &registry, &layer, *n, blocks)
            else {
                println!("  (nothing drawn)");
                continue;
            };
            let o = g.origin;
            for q in g.mesh.vertices.chunks(4) {
                let w: Vec<[f32; 3]> = q
                    .iter()
                    .map(|v| {
                        [
                            o[0] as f32 + v.x() as f32,
                            o[1] as f32 + v.y() as f32,
                            o[2] as f32 + v.z() as f32,
                        ]
                    })
                    .collect();
                // The quad's extent comes within 1.5 blocks of the hit.
                let lo = |k: usize| w.iter().map(|p| p[k]).fold(f32::MAX, f32::min);
                let hi = |k: usize| w.iter().map(|p| p[k]).fold(f32::MIN, f32::max);
                let close = (0..3).all(|k| {
                    let r = if k == 1 { 6.0 } else { 1.5 };
                    hit[k] >= lo(k) - r && hit[k] <= hi(k) + r
                });
                if !close {
                    continue;
                }
                let moved: Vec<String> = q
                    .iter()
                    .zip(&w)
                    .map(|(v, p)| {
                        let f = band.map_or(0.0, |b| {
                            morph_factor(
                                morph_metric(
                                    [p[0] - eye.x, p[1] - eye.y, p[2] - eye.z],
                                    VERTICAL_LOD_SQUASH,
                                ),
                                b,
                            )
                        });
                        format!(
                            "({},{},{}){}{:+}*{f:.3}={:.3}",
                            p[0],
                            p[1],
                            p[2],
                            if v.is_latent() { "L" } else { "" },
                            v.morph_delta(),
                            p[1] + v.morph_delta() * f
                        )
                    })
                    .collect();
                println!("  {:?} {}", q[0].face(), moved.join(" "));
            }
        }
    }

    /// [`locate_crack`]'s exact form: the pixel's ray against the real
    /// triangles of every node of the flight's selection that it passes
    /// near, with and without the morph (each vertex moved by its own
    /// factor, as `mesh.wgsl` does, crack seal left out), and the faces
    /// around where the unmorphed ray lands.
    #[test]
    #[ignore = "a debugging aid: CRACK_AT=frame,x,y cargo test -p cubara --bin cubara trace_crack -- --ignored --nocapture"]
    fn trace_crack() {
        use cubara_world::morph::{morph_factor, morph_metric};
        let at: Vec<f32> = std::env::var("CRACK_AT")
            .unwrap_or_else(|_| "1,1720,848".into())
            .split(',')
            .filter_map(|p| p.trim().parse().ok())
            .collect();
        let (frame, px, py) = (at[0], at[1], at[2]);
        let world = World::new();
        let snapshot = world.snapshot();
        let gen = WorldGen::new(world.seed());
        let (start, look) = toward_peak(&gen);
        let eye = glam::Vec3::from(start) + glam::Vec3::from(look) * (frame * FLY_SPEED / FPS);
        let aspect = SIZE.0 as f32 / SIZE.1 as f32;
        let vp = CameraUniform::look_view_proj(aspect, eye, glam::Vec3::from(look).normalize());
        let inv = vp.inverse();
        let registry = cubara_render::load_registry();
        let blocks = TerrainBlocks::from_registry(&registry)
            .with_oak(&crate::game::load_structure_registry(), &registry)
            .with_ores(&crate::game::load_ore_registry(), &registry);
        let layer = |_: &str| 0u32;
        let schedule = DEFAULT_RING_SCHEDULE;
        let bands = morph_bands(schedule, VERTICAL_LOD_SQUASH);
        let nodes = desired_nodes_3d(
            ChunkCoord::from_world_pos(eye.to_array()),
            VERTICAL_LOD_SQUASH,
            schedule,
        );
        // Triangles of one node, world space, morphed by `morph` (0 or 1 as
        // a switch), each tagged with its quad's description.
        let mut cache: HashMap<NodeKey, Option<cubara_world::mesh::NodeGeometry>> = HashMap::new();
        let ray_hits =
            |dir: glam::Vec3,
             morph: bool,
             cache: &mut HashMap<NodeKey, Option<cubara_world::mesh::NodeGeometry>>| {
                let mut best: Option<(f32, NodeKey, usize)> = None;
                for n in &nodes {
                    let o = n.world_origin();
                    let size = 16.0 * n.extent_chunks() as f32;
                    let (lo, hi) = (
                        glam::vec3(o[0] as f32, o[1] as f32 - 40.0, o[2] as f32),
                        glam::vec3(
                            o[0] as f32 + size,
                            o[1] as f32 + size + 40.0,
                            o[2] as f32 + size,
                        ),
                    );
                    // Slab test.
                    let (mut t0, mut t1) = (0.0f32, 3000.0f32);
                    for k in 0..3 {
                        let inv_d = 1.0 / dir[k];
                        let (a, b) = ((lo[k] - eye[k]) * inv_d, (hi[k] - eye[k]) * inv_d);
                        t0 = t0.max(a.min(b));
                        t1 = t1.min(a.max(b));
                    }
                    if t0 > t1 {
                        continue;
                    }
                    let g = cache.entry(*n).or_insert_with(|| {
                        cubara_world::mesh::mesh_node(&snapshot, &registry, &layer, *n, blocks)
                    });
                    let Some(g) = g else { continue };
                    let band = bands[n.level as usize];
                    let pos = |i: u32| {
                        let v = g.mesh.vertices[i as usize];
                        let mut p = glam::vec3(
                            g.origin[0] as f32 + v.x() as f32,
                            g.origin[1] as f32 + v.y() as f32,
                            g.origin[2] as f32 + v.z() as f32,
                        );
                        if morph {
                            let f = band.map_or(0.0, |b| {
                                morph_factor(
                                    morph_metric((p - eye).to_array(), VERTICAL_LOD_SQUASH),
                                    b,
                                )
                            });
                            p.y += v.morph_delta() * f;
                        }
                        p
                    };
                    for (k, tri) in g.mesh.indices.chunks(3).enumerate() {
                        let (a, b, c) = (pos(tri[0]), pos(tri[1]), pos(tri[2]));
                        // Moller-Trumbore, both sides.
                        let (e1, e2) = (b - a, c - a);
                        let h = dir.cross(e2);
                        let det = e1.dot(h);
                        if det.abs() < 1e-9 {
                            continue;
                        }
                        let inv_det = 1.0 / det;
                        let s0 = eye - a;
                        let u = s0.dot(h) * inv_det;
                        if !(0.0..=1.0).contains(&u) {
                            continue;
                        }
                        let q = s0.cross(e1);
                        let v = dir.dot(q) * inv_det;
                        if v < 0.0 || u + v > 1.0 {
                            continue;
                        }
                        let t = e2.dot(q) * inv_det;
                        if t > 0.1 && best.is_none_or(|(bt, _, _)| t < bt) {
                            best = Some((t, *n, k));
                        }
                    }
                }
                best
            };
        for (sx, sy) in [
            (0.5f32, 0.5f32),
            (0.25, 0.25),
            (0.75, 0.25),
            (0.25, 0.75),
            (0.75, 0.75),
        ] {
            let ndc = glam::vec2(
                (px + sx) / SIZE.0 as f32 * 2.0 - 1.0,
                1.0 - (py + sy) / SIZE.1 as f32 * 2.0,
            );
            let far = inv * glam::vec4(ndc.x, ndc.y, 0.5, 1.0);
            let dir = (far.truncate() / far.w - eye).normalize();
            let fixed = ray_hits(dir, false, &mut cache);
            let morphed = ray_hits(dir, true, &mut cache);
            println!(
                "sample ({sx},{sy}): as meshed {:?} at {:?}; morphed {:?} at {:?}",
                fixed.map(|h| (h.0, h.1)),
                fixed.map(|h| eye + dir * h.0),
                morphed.map(|h| (h.0, h.1)),
                morphed.map(|h| eye + dir * h.0)
            );
            if sx != 0.5 {
                continue;
            }
            let Some((t, _, _)) = fixed else { continue };
            let hit = eye + dir * t;
            for n in &nodes {
                let Some(Some(g)) = cache.get(n) else {
                    continue;
                };
                let o = g.origin;
                let band = bands[n.level as usize];
                let mut lines = Vec::new();
                for q in g.mesh.vertices.chunks(4) {
                    let w: Vec<glam::Vec3> = q
                        .iter()
                        .map(|v| {
                            glam::vec3(
                                o[0] as f32 + v.x() as f32,
                                o[1] as f32 + v.y() as f32,
                                o[2] as f32 + v.z() as f32,
                            )
                        })
                        .collect();
                    let lo = w.iter().fold(glam::Vec3::splat(f32::MAX), |a, b| a.min(*b));
                    let hi = w.iter().fold(glam::Vec3::splat(f32::MIN), |a, b| a.max(*b));
                    let r = glam::vec3(1.5, 4.0, 1.5);
                    if (hit.cmplt(lo - r)).any() || (hit.cmpgt(hi + r)).any() {
                        continue;
                    }
                    let desc: Vec<String> = q
                        .iter()
                        .zip(&w)
                        .map(|(v, p)| {
                            let f = band.map_or(0.0, |b| {
                                morph_factor(
                                    morph_metric((*p - eye).to_array(), VERTICAL_LOD_SQUASH),
                                    b,
                                )
                            });
                            format!(
                                "({},{},{}){}{:+}->{:.2}",
                                p.x,
                                p.y,
                                p.z,
                                if v.is_latent() { "L" } else { "" },
                                v.morph_delta(),
                                p.y + v.morph_delta() * f
                            )
                        })
                        .collect();
                    lines.push(format!("    {:?} {}", q[0].face(), desc.join(" ")));
                }
                if lines.is_empty() {
                    continue;
                }
                println!("  node {n:?} origin {:?}", n.world_origin());
                if let Some(c) = cubara_world::morph::node_columns(&snapshot, *n, blocks) {
                    let w = 1 << n.level;
                    let (cx, cz) = (
                        (hit.x as i32 - o[0]).div_euclid(w),
                        (hit.z.round() as i32 - o[2]).div_euclid(w),
                    );
                    for dz in -1..=0 {
                        for dx in -1..=1 {
                            let (x, z) = (cx + dx, cz + dz);
                            if (-1..=16).contains(&x) && (-1..=16).contains(&z) {
                                let [tp, tg, fl, tf] = c.at(x, z);
                                println!(
                                    "    column ({x},{z}) top {} target {} floor {} tfloor {}",
                                    tp + o[1],
                                    tg + o[1],
                                    fl + o[1],
                                    tf + o[1]
                                );
                            }
                        }
                    }
                }
                for l in lines {
                    println!("{l}");
                }
            }
        }
    }

    /// The study at the game's schedule and at every radius halved.
    #[test]
    #[ignore = "GPU, minutes: cargo test -p cubara --bin cubara transition_study -- --ignored --nocapture"]
    fn transition_study_report() {
        // `CUBARA_STUDY_SCHEDULES=default` (or `half`) runs one of the two.
        let which = std::env::var("CUBARA_STUDY_SCHEDULES").unwrap_or_default();
        let mut report = String::new();
        for (name, schedule) in [("default", DEFAULT_RING_SCHEDULE), ("half", HALF_SCHEDULE)] {
            if which.is_empty() || which.split(',').any(|w| w == name) {
                report.push_str(&study(name, schedule));
                report.push('\n');
            }
        }
        println!("{report}");
        let _ = std::fs::write(out_dir().join("report.md"), &report);
    }
}

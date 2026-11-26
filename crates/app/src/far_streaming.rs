//! Streams the far terrain around the camera (`docs/PROPOSAL_FAR_VIEW.md`,
//! block F3c): which patches the view wants ([`cubara_world::far::select`]),
//! generating the new ones on worker threads, and swapping them into the
//! renderer's [`cubara_render::FarTerrain`] without leaving holes.
//!
//! The same rule the voxel nodes learned in #262 holds here: nothing is taken
//! away before what replaces it exists. A patch the view no longer wants stays
//! drawn while any patch covering the same ground is still being generated.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Condvar, Mutex};

use cubara_render::aggregate::Materials;
use cubara_render::{FarDir, FarFace, FarParams, FarPatch, FarRole, FarSlot, Renderer};
#[cfg(test)]
use cubara_voxel::ChunkCoord;
use cubara_world::far::{self, FarView, Hole, PatchHeights, PatchKey};
use cubara_world::node;
use cubara_world::WorldGen;

/// How far the far terrain reaches, in blocks: the owner's "262 km first"
/// (2026-09-26, `PROPOSAL_FAR_VIEW.md`).
pub const FAR_VIEW_RADIUS: f64 = 262_144.0;

/// How finely the far terrain is drawn -- the per-PC choice between looks and
/// frame rate the owner asked for (2026-09-26): picked by a benchmark to stay
/// above the monitor's refresh rate, or by the player in the options.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FarQuality {
    /// No far terrain: the world ends with the voxels, at 1 km.
    Off,
    Low,
    Medium,
    /// The settings `PROPOSAL_FAR_VIEW.md` §3.2 measured, and the default.
    #[default]
    High,
}

/// What a quality level means, in pixels on screen.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Detail {
    /// Quads no wider than this, by distance alone.
    quad_px: f64,
    /// Split where a patch's 99th-percentile gap to its finer self would
    /// show more than this (`PROPOSAL_FAR_VIEW.md` §3.2).
    error_px: f64,
    /// ...but never into quads narrower than this.
    floor_px: f64,
}

impl FarQuality {
    /// From the best-looking down: the order a benchmark tries them in.
    pub const BEST_FIRST: [FarQuality; 4] = [Self::High, Self::Medium, Self::Low, Self::Off];

    fn detail(self) -> Option<Detail> {
        let (quad_px, error_px, floor_px) = match self {
            Self::Off => return None,
            Self::Low => (32.0, 6.0, 8.0),
            Self::Medium => (16.0, 4.0, 4.0),
            Self::High => (16.0, 2.0, 4.0),
        };
        Some(Detail {
            quad_px,
            error_px,
            floor_px,
        })
    }

    /// One step cheaper, or `Off` already.
    pub fn lower(self) -> Self {
        match self {
            Self::High => Self::Medium,
            Self::Medium => Self::Low,
            Self::Low | Self::Off => Self::Off,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
        }
    }
}

/// The camera's vertical field of view (`render.rs`'s projection).
const FOV_Y: f64 = std::f64::consts::FRAC_PI_3;

/// **The selection is worked out again once the eye has moved this share of
/// its slack** ([`far::Selection::slack`]: how far the eye can move before
/// any decision of the walk could change), which makes the drawn selection
/// the one the eye itself selects on every frame (#398's plan A). It was a
/// fixed distance -- 32 blocks, then one -- and for up to that long a far
/// mountain stood a level coarser than the eye asks for: with blocks
/// (#422) a level coarser is quads twice as wide, and at the foot of a far
/// mountain a pixel of land turned to sky until the finer patch came (the
/// low flight's holes at t 3.0 s, and at t 11.29 s at a quad scale of 2.7).
/// **Chosen, not derived**: 99%, against rounding in the distances.
const RESELECT_SAFETY: f64 = 0.99;

/// **How much nearer than the eye the prefetch selects, as a share of each
/// distance** (#398's plan B): the far terrain's patches are generated for
/// a view in which everything is this much nearer, so the children a split
/// will want exist before the eye gets there. At 1,024 blocks it is 60
/// blocks ahead (2.5 s of flight); a patch takes 6 to 9 ms to generate,
/// 18 to 20 ms at the 99th percentile while flying (#398 step 1).
/// **Chosen, not derived**: the issue's starting value, which covers that
/// latency many times over.
const PREFETCH_MARGIN: f64 = 1.0 / 16.0;

/// **How far the eye moves before the prefetch is worked out again**, in
/// blocks -- a small part of the margin above at the distances the far
/// terrain starts at, and a walk saved on most frames. Also worked out
/// again when a patch it asked for arrives (its error may split more),
/// and when the voxels' box changes. **Chosen, not derived** (#398's plan).
const PREFETCH_RESELECT_BLOCKS: f64 = 8.0;

/// One coarsest voxel node across, in blocks: how far the voxels' box moves
/// at once (`cubara_world::node::covered_box_3d`), and so how much ground
/// round its edge the far terrain draws again when it moves on.
fn coarsest_node_blocks() -> f64 {
    let (level, _) = *node::DEFAULT_RING_SCHEDULE
        .last()
        .expect("the ring schedule has a coarsest ring");
    (16i64 << level) as f64
}

/// The view the prefetch walks ([`PREFETCH_MARGIN`]): `view` with every
/// distance that much nearer -- so it splits wherever the eye's own view
/// would split a little further on, and reaches a little further out --
/// and its hole one coarsest node smaller on every side: the ground the
/// voxels' box hands back when it moves on, which the far terrain must draw
/// the frame it does (`NodeStreaming::voxel_box` shrinks the hole at once).
///
/// Every patch the eye's own view keeps or splits, this one keeps or
/// splits too: each of its tests to split is the eye's or looser, and its
/// hole is inside the eye's. So what it walks holds every patch a nearby
/// eye selects.
fn ahead_view(view: &FarView) -> FarView {
    let grow = 1.0 + PREFETCH_MARGIN;
    FarView {
        split_ratio: view.split_ratio * grow,
        max_error: view.max_error / grow,
        min_quad: view.min_quad / grow,
        radius: view.radius * grow,
        hole: view
            .hole
            .and_then(|h| h.grown_across(-coarsest_node_blocks())),
        ..*view
    }
}

/// **How large the far terrain's quads are on screen**, as a factor on the
/// pixel sizes a quality names ([`FarQuality::detail`]): its quads, its
/// error bound and its floor are all this many times as wide as the sizes
/// `PROPOSAL_FAR_VIEW.md` §3.2 measured for the smooth surface (at Medium,
/// quads of at most 16 pixels). The one number for it: the window, the
/// benchmarks, the flights and the screenshot all draw at this unless
/// `--far-quad-scale F` says otherwise ([`far_view`]).
///
/// Above 1 because the far terrain is blocks now (issue #422): a block
/// surface costs about nine vertices a quad where the smooth one shared one
/// among four, and the far pass's cost follows the triangles it draws
/// (`docs/PHASE3_ARCHITECTURE.md` §9.8). At 1 the gate is not met at any
/// quality. **Chosen, not derived**: 3 was the smallest size a quiet M3 held
/// 1,000 FPS at Medium with, with room to spare (`BENCHMARKS.md` footnote
/// 96). Culled by tiles (`cubara_render::far::cull_runs`), the far terrain
/// sends fewer triangles -- but the ones tiles leave out are those the GPU
/// threw away early (facing away, off screen), about 1 ns each on the quiet
/// M3, and a finer scale adds ones that face the eye on screen, about 2 ns
/// each. 2.7 is the smallest size whose step in the gate's hill eye's
/// selection (47 patches, from 2.688 to 2.86) is expected to cost no more
/// there than 3 did before tiles, by more than the bench's noise: 146,586
/// triangles, 133,833 of them facing the eye on screen, against 193,641 and
/// 122,962 (`far_census_scale_scan`, `far_census_by_kind`): 3.6 to 4.3%
/// faster than that, by the two costs above. The steps below it -- 50
/// patches from 2.676, 52 from 2.62 -- come out 1.0 to 2.4% faster and 0.7%
/// slower to 1.1% faster, inside the bench's noise; 2.09 measured 1.6 and
/// 4.2% slower. The owner left the size to the engine ("as long as it is
/// good", 2026-11-21); the numbers are in §9.8.
///
/// **2.7 again since #398 closed the far selection's lag.** It was held at
/// 3 for a day (the lead, 2026-11-26): at 2.7 the quiet low flight showed a
/// hole in every run (2 pixels at (542, 388), t 11.29 to 11.36 s), the
/// selection splitting a level-4 patch into level 3 right after the
/// voxels' strip hand-over opened that pixel, and the live frame keeping
/// the parent for the 2 to 5 frames the children took to generate while the
/// finished frame had them at once. The plan now works its selection out
/// again within its slack and generates the patches a little ahead of the
/// eye ([`PREFETCH_MARGIN`]): on the low flight replayed without a GPU it
/// draws the eye's own selection on every frame at 2.7 and at 3
/// (`the_live_selection_is_the_exact_one_on_every_frame`).
pub const FAR_QUAD_SCALE: f64 = 2.7;

/// `--far-quad-scale F`: a factor for [`far_view_scaled`] in place of
/// [`FAR_QUAD_SCALE`], for measuring another size. Only a finite number from
/// 0.5 to 8 -- a typo is refused, not measured as the scale that was asked
/// for.
pub fn parse_quad_scale(text: &str) -> Option<f64> {
    text.parse::<f64>()
        .ok()
        .filter(|f| f.is_finite() && (0.5..=8.0).contains(f))
}

/// The view the far terrain is selected for at `quality`, scaled to a screen
/// `screen_height` pixels tall, leaving `hole` to the voxels: the game's, at
/// [`FAR_QUAD_SCALE`]. `None` when the quality is `Off`.
///
/// Only tests ask this way: the window's plan, the benchmarks, the flights
/// and the screenshot each carry the scale they were started with (the
/// constant, or `--far-quad-scale`) to [`far_view_scaled`].
#[cfg(test)]
pub fn far_view(
    eye: [f64; 3],
    screen_height: u32,
    hole: Option<Hole>,
    quality: FarQuality,
) -> Option<FarView> {
    far_view_scaled(eye, screen_height, hole, quality, FAR_QUAD_SCALE)
}

/// [`far_view`] with every pixel size of the quality -- its quads, its error
/// bound and its floor -- `scale` times as large as the quality names them.
/// The game draws at [`FAR_QUAD_SCALE`]; `--far-quad-scale` and `far_census`
/// ask what the others would cost.
pub fn far_view_scaled(
    eye: [f64; 3],
    screen_height: u32,
    hole: Option<Hole>,
    quality: FarQuality,
    scale: f64,
) -> Option<FarView> {
    let d = quality.detail()?;
    let pixel = FOV_Y / screen_height.max(1) as f64 * scale;
    Some(FarView {
        eye,
        split_ratio: far::split_ratio_for(d.quad_px * scale, screen_height.max(1) as f64, FOV_Y),
        max_error: d.error_px * pixel,
        min_quad: d.floor_px * pixel,
        radius: FAR_VIEW_RADIUS,
        hole,
    })
}

/// A box of chunks (`max` exclusive) as the far terrain's hole, in blocks.
pub fn hole_in_blocks((min, max): ([i32; 3], [i32; 3])) -> Hole {
    let blocks = |c: [i32; 3]| c.map(|v| (v * 16) as f64);
    Hole {
        min: blocks(min),
        max: blocks(max),
    }
}

/// The box the voxel nodes around `eye` draw once loaded, in blocks
/// ([`node::covered_box_3d`] for the game's own schedule). What the hole
/// becomes; the game hands it over as the nodes arrive
/// (`NodeStreaming::voxel_box`).
#[cfg(test)]
pub fn voxel_hole(eye: [f32; 3]) -> Hole {
    hole_in_blocks(node::covered_box_3d(
        ChunkCoord::from_world_pos(eye),
        node::DEFAULT_RING_SCHEDULE,
    ))
}

/// The renderer's form of a patch's faces: the same rectangles, field for
/// field (the render crate does not know the world crate, Rule 3).
pub fn to_far_faces(p: &PatchHeights) -> Vec<FarFace> {
    p.faces
        .iter()
        .map(|f| FarFace {
            dir: match f.dir {
                far::FaceDir::Top => FarDir::Top,
                far::FaceDir::PosX => FarDir::PosX,
                far::FaceDir::NegX => FarDir::NegX,
                far::FaceDir::PosZ => FarDir::PosZ,
                far::FaceDir::NegZ => FarDir::NegZ,
            },
            i0: f.i0,
            j0: f.j0,
            i1: f.i1,
            j1: f.j1,
            top: f.top,
            rise: f.rise,
            border: f.border,
            mix: f.mix,
            open: f.open,
        })
        .collect()
}

/// The renderer's form of a patch, with its `faces` ([`to_far_faces`]).
pub fn to_far_patch<'a>(p: &PatchHeights, faces: &'a [FarFace]) -> FarPatch<'a> {
    FarPatch {
        // The integer it is: the renderer subtracts its anchor from it
        // exactly (issue #379).
        origin: p.key.origin(),
        quad: p.key.quad() as f32,
        faces,
        skirt: p.skirt_blocks() as f32,
    }
}

/// The renderer's form of a hole. A hole's edges are chunk edges
/// (`hole_in_blocks`), so the `f64`s are integers and the conversion is
/// exact; the renderer wants them integer to subtract its anchor exactly
/// (issue #379).
pub fn to_far_params(hole: Hole, materials: Materials) -> FarParams {
    FarParams {
        hole: Some((hole.min.map(|v| v as i32), hole.max.map(|v| v as i32))),
        materials,
    }
}

/// What the far terrain is made of: each block face's texture averaged, and
/// this world's soil depth, so a far riser shows what a near one would.
pub fn far_materials() -> Materials {
    cubara_render::far::ground_materials(cubara_world::SOIL_DEPTH as f32)
}

/// **The voxels' box, as the far terrain is told it each frame**
/// (`NodeStreaming::voxel_box`): the box the voxel nodes round the eye draw
/// once all of them are loaded, and the part of it they draw now. The two
/// differ for the few frames after the box has moved on -- by a whole
/// coarsest node on its far side, a kilometre ahead -- while the nodes of
/// the strip it gained are searched for, meshed and drawn. The far terrain
/// stands in for that strip ([`standin_keys`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoxelBox {
    /// The box the voxel nodes draw once loaded: what a finished frame
    /// leaves to them.
    pub wanted: Hole,
    /// The part of `wanted` every node of which is drawn now (inside it).
    pub drawn: Hole,
}

impl VoxelBox {
    /// A box whose nodes are all drawn.
    #[cfg(test)]
    pub fn settled(hole: Hole) -> Self {
        Self {
            wanted: hole,
            drawn: hole,
        }
    }
}

/// Whether the square of ground from `(x, z)` to `(x + side, z + side)` is
/// inside `hole` across the ground (edges included).
fn within(hole: Hole, [x, z]: [f64; 2], side: f64) -> bool {
    hole.min[0] <= x && x + side <= hole.max[0] && hole.min[2] <= z && z + side <= hole.max[2]
}

/// The patches of the finest far level ([`far::FINEST_LEVEL`]) with ground
/// in `outer` and outside `inner`, across the ground -- in key order.
fn finest_patches_between(outer: Hole, inner: Hole) -> Vec<PatchKey> {
    let level = far::FINEST_LEVEL;
    let size = (far::PATCH_QUADS as i64) << level;
    let span = |a: usize| {
        (outer.min[a].floor() as i64).div_euclid(size)..=(outer.max[a].ceil() as i64 - 1).div_euclid(size)
    };
    let mut keys = Vec::new();
    for z in span(2) {
        for x in span(0) {
            let key = PatchKey::new(level, x as i32, z as i32);
            let [ox, oz] = key.origin().map(|o| o as f64);
            // The patch's ground in `outer` (some: the span is of patches
            // that reach into it), and whether all of that is in `inner`.
            let lo = [ox.max(outer.min[0]), oz.max(outer.min[2])];
            let hi = [
                (ox + size as f64).min(outer.max[0]),
                (oz + size as f64).min(outer.max[2]),
            ];
            let all_inner = inner.min[0] <= lo[0]
                && hi[0] <= inner.max[0]
                && inner.min[2] <= lo[1]
                && hi[1] <= inner.max[2];
            if !all_inner {
                keys.push(key);
            }
        }
    }
    keys
}

/// **Whether the far terrain can stand in for the strip as the voxels will
/// draw it**: every height of ground any level draws ([`far::ground_range`])
/// is inside both boxes -- so they differ across the ground only, where
/// ground can be, and every top over the strip is one the strip's nodes
/// will draw. Otherwise (an eye some 900 blocks above the ground or 770
/// under it, the box's floor or ceiling through the hills) a move of the
/// box up or down hands over a slab of ground under all of it, and the far
/// terrain stands in for that as it always did: selected and holed at
/// `drawn`.
fn standin_possible(voxels: VoxelBox) -> bool {
    let (lo, hi) = far::ground_range();
    [voxels.wanted, voxels.drawn]
        .iter()
        .all(|b| b.min[1] < lo && hi <= b.max[1])
}

/// **The far terrain's stand-in for the strip** (`docs/PHASE3_ARCHITECTURE.md`
/// §9.11): the patches of the finest far level -- level 3's cells, top for
/// top and wall for wall (§9.8) -- over the ground inside `wanted` and
/// outside `drawn`, while the voxel nodes that will draw it load. `None`
/// when it cannot stand in ([`standin_possible`]); empty when nothing
/// waits. In key order.
///
/// Until it was there the far terrain stood in for the strip as it draws
/// it -- quads of 16 to 32 -- and a sliver of sky between two of its ridges
/// that level 3 covers slid over a pixel: land, sky, land, the owner's
/// flicker (the low flight's t 5.49 s at (211, 358), #443).
pub fn standin_keys(voxels: VoxelBox) -> Option<Vec<PatchKey>> {
    if voxels.wanted == voxels.drawn {
        return Some(Vec::new());
    }
    standin_possible(voxels).then(|| finest_patches_between(voxels.wanted, voxels.drawn))
}

/// **What the far terrain draws for the voxels' box `voxels`**, with the
/// patches `generated` says are: the box it is selected and holed at, and
/// the stand-in's patches. Once every patch of the stand-in is generated,
/// the finished frame's far terrain -- holed at the box the voxels want --
/// and level 3 over the strip ([`standin_keys`]); until then the far
/// terrain holed at the box they draw, standing in for the strip itself at
/// its own quads, as it always did. The one rule for it: [`FarPlan::step`]
/// draws by it, and the GPU test of the strip (`strip_standin`) draws what
/// it says.
pub fn far_stand(voxels: VoxelBox, generated: impl Fn(PatchKey) -> bool) -> (Hole, Vec<PatchKey>) {
    match standin_keys(voxels) {
        Some(keys) if keys.iter().all(|&k| generated(k)) => (voxels.wanted, keys),
        _ => (voxels.drawn, Vec::new()),
    }
}

/// The finest far level's patches round `wanted`, one coarsest voxel node
/// deep: the ground the voxels' box gains when it moves on
/// (`node::covered_box_3d` moves it a node at a time), whichever way --
/// generated before it moves, so the far terrain stands in for the strip
/// the frame it does ([`standin_keys`]).
pub(crate) fn standin_ring(wanted: Hole) -> Vec<PatchKey> {
    wanted
        .grown_across(coarsest_node_blocks())
        .map_or_else(Vec::new, |grown| finest_patches_between(grown, wanted))
}

/// **The faces of `patch` the voxel nodes of the strip will draw**: those of
/// a quad in it -- a top's own, a wall's the higher of the two it stands
/// between ([`FarFace::owner`]), as a voxel mesher gives a riser to the
/// column it is the side of. A wall on the strip's side towards the drawn
/// voxels is the strip's where the strip is the higher, and theirs
/// otherwise; one on its far side the strip's where it is the higher, and
/// the far terrain's beyond. So every face is drawn once: by the stand-in,
/// the voxels or the far terrain, as the finished frame draws it.
///
/// No face crosses a [`far::JOIN_GRID`] line and the boxes' sides are on
/// them, so a face's first quad says where all of it is.
pub fn standin_faces(patch: &PatchHeights, voxels: VoxelBox) -> Vec<FarFace> {
    let [ox, oz] = patch.key.origin().map(|o| o as f64);
    let quad = patch.key.quad() as f64;
    to_far_faces(patch)
        .into_iter()
        .filter(|face| {
            let [i, j] = face.owner();
            let at = [ox + i as f64 * quad, oz + j as f64 * quad];
            within(voxels.wanted, at, quad) && !within(voxels.drawn, at, quad)
        })
        .collect()
}

/// Every coarser patch holding `key`, up to `top_level`.
fn ancestors(key: PatchKey, top_level: u32) -> impl Iterator<Item = PatchKey> {
    (key.level + 1..=top_level).map(move |level| {
        let shift = level - key.level;
        PatchKey::new(level, key.x >> shift, key.z >> shift)
    })
}

/// The coarsest level a selection reaches: the roots [`far::select`] starts from.
fn top_level(keys: &[PatchKey]) -> u32 {
    keys.iter().map(|k| k.level).max().unwrap_or(0)
}

/// Which patches to draw: every wanted one that is ready, and -- while any
/// wanted one is still being generated -- whatever is drawn now over the same
/// ground, so the ground never goes missing in between (#262's rule).
fn draw_set(
    wanted: &[PatchKey],
    ready: &impl Fn(PatchKey) -> bool,
    drawn: &HashSet<PatchKey>,
) -> HashSet<PatchKey> {
    let top = top_level(wanted).max(drawn.iter().map(|k| k.level).max().unwrap_or(0));
    let mut out: HashSet<PatchKey> = wanted.iter().copied().filter(|k| ready(*k)).collect();
    let missing: HashSet<PatchKey> = wanted.iter().copied().filter(|k| !ready(*k)).collect();
    if missing.is_empty() {
        return out;
    }
    // A drawn patch covers missing ground if it is a missing patch's ancestor,
    // or has a missing patch among its own ancestors.
    let above_missing: HashSet<PatchKey> =
        missing.iter().flat_map(|&m| ancestors(m, top)).collect();
    for &old in drawn {
        if out.contains(&old) {
            continue;
        }
        if above_missing.contains(&old) || ancestors(old, top).any(|a| missing.contains(&a)) {
            out.insert(old);
        }
    }
    out
}

/// What changes on the renderer to go from drawing `drawn` to drawing `draw`:
/// the patches that leave, and the ones that arrive. A patch in both stays
/// where it is -- uploading it again would be work, and dropping it would be a
/// hole.
fn swap(drawn: &HashSet<PatchKey>, draw: &HashSet<PatchKey>) -> (Vec<PatchKey>, Vec<PatchKey>) {
    let mut leave: Vec<PatchKey> = drawn.difference(draw).copied().collect();
    let mut arrive: Vec<PatchKey> = draw.difference(drawn).copied().collect();
    leave.sort();
    arrive.sort();
    (leave, arrive)
}

/// Which generated patches are still of use: what is drawn, what is wanted,
/// and what decides a split -- every ancestor of a wanted patch, since its
/// error is what split it. Drop an ancestor and the next selection no longer
/// knows why its ground was split, merges it back, and then splits it again
/// once the ancestor is regenerated: a selection that never settles (review
/// of #284).
///
/// Every ancestor up to `root_level`, the level the selection walks down
/// from ([`far::root_level_for`]) -- not only up to the coarsest patch
/// wanted. A root split for its roughness alone, too far away to be split
/// by distance, is no wanted patch's level once all of them are split, and
/// was dropped: the plan then asked for it, split it and dropped it again,
/// for ever. Quads of 16 pixels split every root by distance, so it took
/// the wider quads of [`FAR_QUAD_SCALE`] to show it (at the tests' radius;
/// the game's roots are too large to be rough at their distance).
fn to_keep(wanted: &[PatchKey], draw: HashSet<PatchKey>, root_level: u32) -> HashSet<PatchKey> {
    let top = top_level(wanted).max(root_level);
    let mut keep = draw;
    for &k in wanted {
        keep.insert(k);
        keep.extend(ancestors(k, top));
    }
    keep
}

/// How soon a patch asked for is wanted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Urgency {
    /// The selection at the eye wants it: drawn ground waits for it.
    Now,
    /// The prefetch walked it: wanted a little further on, if at all.
    Ahead,
}

/// One patch to generate, and the generation (of [`FarStreaming`]) it was
/// asked for in.
type Job = (u64, WorldGen, PatchKey);

/// The pool's two queues: a worker takes from `now` while it has anything,
/// so a patch the eye wants never waits behind the prefetch.
#[derive(Default)]
struct Queues {
    now: VecDeque<Job>,
    ahead: VecDeque<Job>,
    closed: bool,
}

impl Queues {
    /// Generate `job`'s patch, after everything asked for as urgently.
    fn push(&mut self, job: Job, urgency: Urgency) {
        match urgency {
            Urgency::Now => self.now.push_back(job),
            Urgency::Ahead => self.ahead.push_back(job),
        }
    }

    /// A patch the prefetch asked for is wanted now: to the back of the
    /// urgent queue, if no worker has taken it yet.
    fn promote(&mut self, key: PatchKey) {
        if let Some(at) = self.ahead.iter().position(|j| j.2 == key) {
            let job = self.ahead.remove(at).expect("found it");
            self.now.push_back(job);
        }
    }

    /// The oldest urgent job, else the oldest prefetch.
    fn take(&mut self) -> Option<Job> {
        self.now.pop_front().or_else(|| self.ahead.pop_front())
    }

    fn clear(&mut self) {
        self.now.clear();
        self.ahead.clear();
    }
}

/// A pool of threads generating patches. Each job carries the generation it
/// was asked for in, so a result from before a reset (a new world) is dropped.
struct Pool {
    queues: Arc<(Mutex<Queues>, Condvar)>,
    done: Receiver<(u64, PatchHeights)>,
    _workers: Vec<std::thread::JoinHandle<()>>,
}

impl Pool {
    fn new() -> Self {
        // Half the cores: the voxel mesher has the rest, and it is the one the
        // player is standing in.
        let workers = std::thread::available_parallelism()
            .map(|n| n.get() / 2)
            .unwrap_or(1)
            .max(1);
        let queues = Arc::new((Mutex::new(Queues::default()), Condvar::new()));
        let (done_tx, done) = std::sync::mpsc::channel();
        let _workers = (0..workers)
            .map(|_| {
                let queues = Arc::clone(&queues);
                let done = done_tx.clone();
                std::thread::Builder::new()
                    .name("cubara-far".into())
                    .spawn(move || {
                        while let Some((generation, gen, key)) = next_job(&queues) {
                            if done.send((generation, far::generate(&gen, key))).is_err() {
                                break;
                            }
                        }
                    })
                    .expect("spawn far-terrain worker")
            })
            .collect();
        Self {
            queues,
            done,
            _workers,
        }
    }

    fn with_queues(&self, change: impl FnOnce(&mut Queues)) {
        let (lock, wake) = &*self.queues;
        change(&mut lock.lock().expect("far queues"));
        wake.notify_all();
    }

    fn ask(&self, job: Job, urgency: Urgency) {
        self.with_queues(|q| q.push(job, urgency));
    }

    fn promote(&self, key: PatchKey) {
        self.with_queues(|q| q.promote(key));
    }

    /// Forget everything not taken yet (a reset: a new world).
    fn clear(&self) {
        self.with_queues(Queues::clear);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.with_queues(|q| q.closed = true);
    }
}

/// A worker's next job: the oldest urgent one, else the oldest prefetch,
/// waiting for one; `None` once the pool is closed.
fn next_job(queues: &(Mutex<Queues>, Condvar)) -> Option<Job> {
    let (lock, wake) = queues;
    let mut q = lock.lock().expect("far queues");
    loop {
        if q.closed {
            return None;
        }
        if let Some(job) = q.take() {
            return Some(job);
        }
        q = wake.wait(q).expect("far queues");
    }
}

/// What one frame changes: patches to generate, slots to free, patches to
/// upload. Empty when nothing changed.
#[derive(Debug, Default, PartialEq)]
struct Step {
    /// What the selection at the eye wants and nobody is generating:
    /// generated before anything else.
    request: Vec<PatchKey>,
    /// What the prefetch walked and nobody is generating: generated when
    /// nothing is wanted now.
    prefetch: Vec<PatchKey>,
    /// Asked for by the prefetch, wanted now ([`Pool::promote`]).
    promote: Vec<PatchKey>,
    leave: Vec<PatchKey>,
    arrive: Vec<PatchKey>,
    /// The stand-in for the strip to take away, and the one to draw, cut
    /// for `standin_box` ([`standin_faces`]).
    standin_leave: Vec<PatchKey>,
    standin_arrive: Vec<PatchKey>,
    standin_box: Option<VoxelBox>,
}

/// The far terrain's streaming decisions, with no GPU and no threads
/// (`ARCHITECTURE.md` rule 6: the timing of streaming can only be pinned by a
/// test if the decisions can run in one).
///
/// The plan does not hold what is drawn: [`FarStreaming`]'s `slots` is the one
/// source of truth for that, and is handed to [`step`](Self::step) each time.
///
/// **What is drawn is the selection the eye itself makes, on every frame**
/// (#398): the selection is worked out again before the eye can have moved
/// far enough to change it ([`RESELECT_SAFETY`]), and what it will want is
/// generated before it wants it ([`PREFETCH_MARGIN`]). A patch asked for
/// late still takes its frames to generate, and #262's rule holds while it
/// does ([`draw_set`]) -- the prefetch is what keeps that from happening.
struct FarPlan {
    quality: FarQuality,
    /// [`FAR_VIEW_RADIUS`] in the game; tests use less.
    radius: f64,
    /// The quality's pixel sizes times this ([`far_view_scaled`]):
    /// [`FAR_QUAD_SCALE`] unless `--far-quad-scale` says otherwise.
    quad_scale: f64,
    /// Every patch generated and still of use: the ones drawn, the ones
    /// whose error decides whether their area is split, and the ones the
    /// prefetch walked.
    heights: HashMap<PatchKey, PatchHeights>,
    /// Asked for and not arrived yet, and how urgently.
    in_flight: HashMap<PatchKey, Urgency>,
    /// Where the eye was when the selection was last worked out, and how far
    /// it could move from there before the selection could change
    /// ([`far::Selection::slack`]).
    selected_at: Option<[f64; 3]>,
    slack: f64,
    /// The last selection.
    wanted: HashSet<PatchKey>,
    /// Where the eye was when the prefetch was last worked out, and every
    /// patch it walked: the ones it keeps and the ones it splits
    /// ([`ahead_view`]).
    ahead_at: Option<[f64; 3]>,
    ahead: HashSet<PatchKey>,
    /// The hole the last selection left to the voxels: the box they want
    /// while the stand-in can stand in for the strip, else the box they
    /// draw ([`FarPlan::step`]).
    hole: Option<Hole>,
    /// The finest level's patches round the voxels' box ([`standin_ring`]),
    /// generated before the box moves on, and the box they were worked out
    /// for.
    ring: HashSet<PatchKey>,
    ring_for: Option<Hole>,
    /// The patches of the strip the voxels' box has now ([`standin_keys`]),
    /// generated or not: kept while it waits.
    strip: Vec<PatchKey>,
    /// What stands in for the strip now: the box it was cut for and its
    /// patches; `None` while nothing does.
    standin: Option<(VoxelBox, Vec<PatchKey>)>,
    /// The screen height the last selection was made for.
    screen_height: u32,
    /// A patch the selection wants arrived since it was last worked out.
    arrived: bool,
    /// A patch the prefetch walked arrived since it was last worked out.
    ahead_arrived: bool,
    /// How many times the selection and the prefetch have been worked out,
    /// for tests of what a frame costs.
    #[cfg(test)]
    selects: u64,
    #[cfg(test)]
    prefetches: u64,
}

/// How far apart two eyes are, in blocks.
fn eye_distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>().sqrt()
}

impl FarPlan {
    fn new(quality: FarQuality, radius: f64) -> Self {
        Self {
            quality,
            radius,
            quad_scale: FAR_QUAD_SCALE,
            heights: HashMap::new(),
            in_flight: HashMap::new(),
            selected_at: None,
            slack: 0.0,
            wanted: HashSet::new(),
            ahead_at: None,
            ahead: HashSet::new(),
            hole: None,
            ring: HashSet::new(),
            ring_for: None,
            strip: Vec::new(),
            standin: None,
            screen_height: 0,
            arrived: false,
            ahead_arrived: false,
            #[cfg(test)]
            selects: 0,
            #[cfg(test)]
            prefetches: 0,
        }
    }

    /// Forget every patch, generated or generating, and start the selection
    /// again (a new world, a new quality).
    fn clear(&mut self) {
        self.heights.clear();
        self.in_flight.clear();
        self.selected_at = None;
        self.wanted.clear();
        self.ahead_at = None;
        self.ahead.clear();
        self.ring.clear();
        self.ring_for = None;
        self.strip.clear();
        self.standin = None;
        self.arrived = false;
        self.ahead_arrived = false;
    }

    /// A patch finished generating. The selection is worked out again only
    /// if it wants this patch: a patch it does not want changes none of its
    /// decisions (each reads the error of a patch it keeps, and those are
    /// the patches it wants), so nothing else can move it (#398 §6.2's
    /// lever). The prefetch is worked out again for one it walked.
    fn take(&mut self, patch: PatchHeights) {
        let key = patch.key;
        let urgency = self.in_flight.remove(&key);
        self.heights.insert(key, patch);
        self.arrived |= self.wanted.contains(&key);
        // A patch of the ring decides nothing the prefetch walks: it waits
        // for the box to move.
        self.ahead_arrived |= self.ahead.contains(&key)
            || (urgency == Some(Urgency::Ahead) && !self.ring.contains(&key));
    }

    /// Nothing generating, and the selection and the prefetch up to date
    /// with what has arrived.
    fn settled(&self) -> bool {
        self.in_flight.is_empty() && !self.arrived && !self.ahead_arrived
    }

    /// Work the selection out again on the next [`step`](Self::step), however
    /// little the eye has moved.
    fn reselect(&mut self) {
        self.selected_at = None;
    }

    /// The heights of a generated patch still held.
    fn patch(&self, key: PatchKey) -> Option<&PatchHeights> {
        self.heights.get(&key)
    }

    /// The hole the far terrain is drawn with now ([`step`](Self::step)).
    fn hole(&self) -> Option<Hole> {
        self.hole
    }

    /// Ask at once for every patch of the stand-in for the strip `voxels`
    /// has ([`standin_keys`]) that is not generated: the strip is in sight
    /// now. (The ring has them generated before the box moves; a box that
    /// moves twice before its first strip is drawn can outrun it.)
    fn ask_for_standin(&mut self, voxels: VoxelBox, step: &mut Step) {
        self.strip = standin_keys(voxels).unwrap_or_default();
        for &k in &self.strip {
            if self.heights.contains_key(&k) {
                continue;
            }
            match self.in_flight.get(&k) {
                None => step.request.push(k),
                Some(Urgency::Ahead) => step.promote.push(k),
                Some(Urgency::Now) => continue,
            }
            self.in_flight.insert(k, Urgency::Now);
        }
    }

    /// Draw `now` as the stand-in from this frame on, if it is not what is
    /// drawn already: the old one leaves and the new one arrives in the
    /// same frame.
    fn show_standin(&mut self, now: Option<(VoxelBox, Vec<PatchKey>)>, step: &mut Step) {
        if now == self.standin {
            return;
        }
        if let Some((_, old)) = self.standin.take() {
            step.standin_leave = old;
        }
        if let Some((voxels, keys)) = &now {
            step.standin_arrive.clone_from(keys);
            step.standin_box = Some(*voxels);
        }
        self.standin = now;
    }

    /// Ask for the ring round `wanted` ([`standin_ring`]) when the box is a
    /// new one, behind what the eye wants.
    fn ring_round(&mut self, wanted: Hole, step: &mut Step) {
        if self.ring_for == Some(wanted) {
            return;
        }
        self.ring_for = Some(wanted);
        self.ring = standin_ring(wanted).into_iter().collect();
        let ask: Vec<PatchKey> = self
            .ring
            .iter()
            .copied()
            .filter(|k| !self.heights.contains_key(k) && !self.in_flight.contains_key(k))
            .collect();
        for &k in &ask {
            self.in_flight.insert(k, Urgency::Ahead);
        }
        step.prefetch.extend(ask);
        // Finest first, then in key order, with the prefetch's (rule 1).
        step.prefetch.sort();
    }

    /// Decide what this frame changes for an eye at `eye`: work the selection
    /// out again if the eye has moved as far as its slack, the hole or the
    /// screen changed, or a patch it wants arrived -- the prefetch likewise,
    /// on its own cadence -- and say what to ask for, free and upload.
    /// `drawn` is what the renderer holds now; it is called only if either
    /// is worked out.
    ///
    /// **While a strip of the voxels' box waits for its nodes** (`voxels`'
    /// two boxes differ) the far terrain is what the finished frame draws --
    /// selected and holed at the box the voxels want -- and the stand-in
    /// draws the strip as level 3 will ([`standin_keys`]), as soon as all of
    /// it is generated; until then the far terrain is selected and holed at
    /// the box they draw, and stands in for the strip itself.
    fn step(
        &mut self,
        eye: [f64; 3],
        screen_height: u32,
        voxels: VoxelBox,
        drawn: impl FnOnce() -> HashSet<PatchKey>,
    ) -> Step {
        let mut step = Step::default();
        let on = self.quality.detail().is_some();
        let (hole, standin) = if on {
            self.ask_for_standin(voxels, &mut step);
            far_stand(voxels, |k| self.heights.contains_key(&k))
        } else {
            (voxels.drawn, Vec::new())
        };
        self.show_standin((!standin.is_empty()).then_some((voxels, standin)), &mut step);
        let mut step = self.select(eye, screen_height, hole, drawn, step);
        // Last, so nothing the eye wants now is asked for behind it.
        if on {
            self.ring_round(voxels.wanted, &mut step);
        }
        step
    }

    /// [`step`](Self::step)'s far terrain: the selection and the prefetch
    /// for the voxels' box `hole`, into `step`.
    fn select(
        &mut self,
        eye: [f64; 3],
        screen_height: u32,
        hole: Hole,
        drawn: impl FnOnce() -> HashSet<PatchKey>,
        mut step: Step,
    ) -> Step {
        // The hole is the voxels' to say (`NodeStreaming::voxel_box`): where
        // it shrinks, patches it used to swallow are wanted at once.
        let hole_moved = self.hole != Some(hole);
        self.hole = Some(hole);
        let screen_moved = self.screen_height != screen_height;
        self.screen_height = screen_height;
        let changed = hole_moved || screen_moved;
        let select_due = changed
            || self.arrived
            || self.selected_at.is_none_or(|at| {
                let moved = eye_distance(at, eye);
                moved > 0.0 && moved >= self.slack * RESELECT_SAFETY
            });
        let prefetch_due = changed
            || self.ahead_arrived
            || self
                .ahead_at
                .is_none_or(|at| eye_distance(at, eye) > PREFETCH_RESELECT_BLOCKS);
        if !(select_due || prefetch_due) {
            return step;
        }
        let drawn = drawn();

        let Some(view) = far_view_scaled(
            eye,
            screen_height,
            Some(hole),
            self.quality,
            self.quad_scale,
        ) else {
            // Off: nothing beyond the voxels.
            self.selected_at = Some(eye);
            self.ahead_at = Some(eye);
            self.wanted.clear();
            self.ahead.clear();
            step.leave = drawn.into_iter().collect();
            step.leave.sort();
            return step;
        };
        let view = FarView {
            radius: self.radius,
            ..view
        };
        let error_of = |heights: &HashMap<PatchKey, PatchHeights>, k: PatchKey| {
            heights.get(&k).map(|p| p.error)
        };

        // The selection first, so what it wants is asked for urgently.
        let wanted = select_due.then(|| {
            self.selected_at = Some(eye);
            self.arrived = false;
            #[cfg(test)]
            {
                self.selects += 1;
            }
            let walk = far::select_walk(&view, |k| error_of(&self.heights, k));
            self.slack = walk.slack;
            // Ask for what is missing, nearest first: the ground under the
            // eye matters more than the horizon. What the prefetch asked
            // for and no worker has started moves to the urgent queue.
            for &k in &walk.patches {
                if self.heights.contains_key(&k) {
                    continue;
                }
                match self.in_flight.get(&k) {
                    None => step.request.push(k),
                    Some(Urgency::Ahead) => step.promote.push(k),
                    Some(Urgency::Now) => {}
                }
            }
            step.request.sort();
            step.promote.sort();
            for &k in step.request.iter().chain(&step.promote) {
                self.in_flight.insert(k, Urgency::Now);
            }
            walk.patches
        });

        if prefetch_due {
            self.ahead_at = Some(eye);
            self.ahead_arrived = false;
            #[cfg(test)]
            {
                self.prefetches += 1;
            }
            let walk = far::select_walk(&ahead_view(&view), |k| error_of(&self.heights, k));
            self.ahead = walk.patches.into_iter().chain(walk.split).collect();
            let mut ask: Vec<PatchKey> = self
                .ahead
                .iter()
                .copied()
                .filter(|k| !self.heights.contains_key(k) && !self.in_flight.contains_key(k))
                .collect();
            // Finest first, then in key order, so the order is a function of
            // the plan's state and not of the set's (rule 1).
            ask.sort();
            for &k in &ask {
                self.in_flight.insert(k, Urgency::Ahead);
            }
            step.prefetch.extend(ask);
            step.prefetch.sort();
        }

        if let Some(wanted) = wanted {
            // Swap the drawn set, never leaving ground uncovered.
            let draw = draw_set(&wanted, &|k| self.heights.contains_key(&k), &drawn);
            (step.leave, step.arrive) = swap(&drawn, &draw);
            let mut keep = to_keep(&wanted, draw, far::root_level_for(view.radius));
            keep.extend(self.ahead.iter().copied());
            keep.extend(self.ring.iter().copied());
            keep.extend(self.strip.iter().copied());
            self.heights.retain(|k, _| keep.contains(k));
            self.wanted = wanted.into_iter().collect();
        }
        step
    }
}

/// The far terrain's streaming state. One per window, beside the voxel
/// [`crate::streaming::NodeStreaming`].
pub struct FarStreaming {
    gen: WorldGen,
    seed: u64,
    generation: u64,
    pool: Pool,
    plan: FarPlan,
    slots: HashMap<PatchKey, FarSlot>,
    /// The stand-in for the strip the voxels' box is loading
    /// ([`standin_keys`]): the same patches as the far terrain's when a key
    /// is both, in slots of their own, cut to the strip.
    standin_slots: HashMap<PatchKey, FarSlot>,
    materials: Materials,
    /// Said once when the renderer runs out of slots, not every frame.
    warned_full: bool,
}

impl FarStreaming {
    pub fn new(seed: u64, quality: FarQuality) -> Self {
        Self {
            gen: WorldGen::new(seed),
            seed,
            generation: 0,
            pool: Pool::new(),
            plan: FarPlan::new(quality, FAR_VIEW_RADIUS),
            slots: HashMap::new(),
            standin_slots: HashMap::new(),
            materials: far_materials(),
            warned_full: false,
        }
    }

    /// Draw with the quality's quads, error bound and floor `scale` times as
    /// wide on screen as it names them, in place of [`FAR_QUAD_SCALE`]
    /// (`--far-quad-scale`). Before the first frame: patches already drawn
    /// are not selected again.
    pub fn with_quad_scale(mut self, scale: f64) -> Self {
        self.plan.quad_scale = scale;
        self
    }

    /// The seed the far terrain is generated from.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Draw at a different quality from now on: every patch is dropped and
    /// the selection starts again, since a patch's heights do not depend on
    /// it but which patches are wanted does.
    pub fn set_quality(&mut self, quality: FarQuality, renderer: &mut Renderer) {
        if quality != self.plan.quality {
            let seed = self.seed;
            self.plan.quality = quality;
            self.reset(seed, renderer);
        }
    }

    /// A different world -- New World, a load, joining a server: drop every
    /// patch, drawn or generating, and start again from `seed`.
    pub fn reset(&mut self, seed: u64, renderer: &mut Renderer) {
        for (_, slot) in self.slots.drain().chain(self.standin_slots.drain()) {
            renderer.far_remove(slot);
        }
        self.plan.clear();
        self.pool.clear();
        self.generation += 1;
        self.gen = WorldGen::new(seed);
        self.seed = seed;
    }

    /// Nothing generating, and the selection up to date with what has
    /// arrived (`crate::flight`).
    pub fn settled(&self) -> bool {
        self.plan.settled()
    }

    /// Work the selection out again on the next [`update`](Self::update),
    /// however little the eye has moved. The slack already makes it the
    /// eye's own; a finished frame asks anyway, so that what it compares
    /// against does not rest on that.
    pub fn reselect(&mut self) {
        self.plan.reselect();
    }

    /// Bring the far terrain in line with the camera at `eye` and the
    /// voxels' box: take in what has been generated, work the selection out
    /// again if the eye has moved or new patches arrived, ask for what is
    /// missing -- and, a little ahead, for what will be -- and stand in for
    /// a strip of the box still loading ([`FarPlan::step`]).
    pub fn update(&mut self, renderer: &mut Renderer, eye: [f32; 3], voxels: VoxelBox) {
        while let Ok((generation, patch)) = self.pool.done.try_recv() {
            if generation != self.generation {
                continue;
            }
            self.plan.take(patch);
        }
        let eye64 = eye.map(|v| v as f64);
        let slots = &self.slots;
        let step = self.plan.step(eye64, renderer.size().1, voxels, || {
            slots.keys().copied().collect()
        });
        // The hole this frame's far terrain is drawn with: the box the
        // voxels want while the stand-in draws the strip, else the box they
        // draw.
        let hole = self.plan.hole().unwrap_or(voxels.drawn);
        renderer.set_far_params(to_far_params(hole, self.materials));
        for key in step.standin_leave {
            if let Some(slot) = self.standin_slots.remove(&key) {
                renderer.far_remove(slot);
            }
        }
        if let Some(strip) = step.standin_box {
            for key in step.standin_arrive {
                let Some(patch) = self.plan.patch(key) else {
                    continue;
                };
                let faces = standin_faces(patch, strip);
                match renderer.far_insert_as(to_far_patch(patch, &faces), FarRole::StandIn) {
                    Some(slot) => {
                        self.standin_slots.insert(key, slot);
                    }
                    None if !self.warned_full => {
                        log::warn!(
                            "far terrain: out of slots or vertices -- a strip of ground \
                             still loading is not drawn"
                        );
                        self.warned_full = true;
                    }
                    None => {}
                }
            }
        }

        for key in step.promote {
            self.pool.promote(key);
        }
        for key in step.request {
            self.pool
                .ask((self.generation, self.gen, key), Urgency::Now);
        }
        for key in step.prefetch {
            self.pool
                .ask((self.generation, self.gen, key), Urgency::Ahead);
        }
        for key in step.leave {
            if let Some(slot) = self.slots.remove(&key) {
                renderer.far_remove(slot);
            }
        }
        for key in step.arrive {
            let Some(patch) = self.plan.patch(key) else {
                continue;
            };
            match renderer.far_insert(to_far_patch(patch, &to_far_faces(patch))) {
                Some(slot) => {
                    self.slots.insert(key, slot);
                }
                None if !self.warned_full => {
                    log::warn!(
                        "far terrain: out of slots or vertices -- some distant ground is not drawn"
                    );
                    self.warned_full = true;
                }
                None => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each step down is cheaper and never better: a coarser error bound, a
    /// coarser floor, quads no finer -- so stepping down when frames are
    /// missed can only help.
    #[test]
    fn every_step_down_is_cheaper() {
        let mut q = FarQuality::High;
        while q != FarQuality::Off {
            let lower = q.lower();
            let (a, b) = (q.detail().unwrap(), lower.detail());
            if let Some(b) = b {
                assert!(
                    b.error_px > a.error_px || b.floor_px > a.floor_px,
                    "{q:?} -> {lower:?}"
                );
                assert!(
                    b.error_px >= a.error_px && b.floor_px >= a.floor_px && b.quad_px >= a.quad_px
                );
            }
            q = lower;
        }
        assert_eq!(FarQuality::Off.lower(), FarQuality::Off);
        assert_eq!(FarQuality::BEST_FIRST[0], FarQuality::default());
    }

    /// Found by check-tests-can-fail.sh: skipping every insert survived, since
    /// nothing short of a window reached the swap. Now it is a function.
    #[test]
    fn the_swap_adds_what_is_new_drops_what_is_gone_and_keeps_the_rest() {
        let (a, b, c) = (
            PatchKey::new(3, 0, 0),
            PatchKey::new(3, 1, 0),
            PatchKey::new(3, 2, 0),
        );
        let drawn: HashSet<_> = [a, b].into();
        let draw: HashSet<_> = [b, c].into();
        let (leave, arrive) = swap(&drawn, &draw);
        assert_eq!(leave, vec![a]);
        assert_eq!(arrive, vec![c], "a new patch was not uploaded");
        let (leave, arrive) = swap(&draw, &draw);
        assert!(
            leave.is_empty() && arrive.is_empty(),
            "an unchanged set moved"
        );
        let (_, arrive) = swap(&HashSet::new(), &draw);
        assert_eq!(arrive, vec![b, c], "a fresh start uploaded nothing");
    }

    #[test]
    fn ancestors_climb_one_level_at_a_time() {
        let a: Vec<_> = ancestors(PatchKey::new(2, -5, 9), 5).collect();
        assert_eq!(
            a,
            vec![
                PatchKey::new(3, -3, 4),
                PatchKey::new(4, -2, 2),
                PatchKey::new(5, -1, 1),
            ]
        );
    }

    /// The camera came closer: a coarse patch is split into four children, and
    /// only two are generated yet. The parent must stay drawn until all four
    /// are -- dropping it would leave half its ground missing.
    #[test]
    fn a_coarse_patch_stays_until_every_child_has_arrived() {
        let parent = PatchKey::new(4, 2, 2);
        let kids = parent.children();
        let drawn: HashSet<_> = [parent].into();
        let ready = |k: PatchKey| k == kids[0] || k == kids[1];
        let draw = draw_set(&kids, &ready, &drawn);
        assert!(
            draw.contains(&parent),
            "the parent left before its children arrived"
        );
        assert!(draw.contains(&kids[0]) && draw.contains(&kids[1]));
        assert!(
            !draw.contains(&kids[2]),
            "a patch with no heights was drawn"
        );

        let all = |_: PatchKey| true;
        let draw = draw_set(&kids, &all, &drawn);
        assert!(
            !draw.contains(&parent),
            "the parent stayed after it was replaced"
        );
    }

    /// The camera moved away: four children merge into their parent, which is
    /// not generated yet. The children stay until it is.
    #[test]
    fn fine_patches_stay_until_the_coarse_one_replacing_them_has_arrived() {
        let parent = PatchKey::new(4, 2, 2);
        let drawn: HashSet<_> = parent.children().into_iter().collect();
        let draw = draw_set(&[parent], &|_| false, &drawn);
        assert_eq!(
            draw, drawn,
            "ground went missing while its replacement generated"
        );
        let draw = draw_set(&[parent], &|_| true, &drawn);
        assert_eq!(draw, [parent].into());
    }

    #[test]
    fn a_patch_nothing_replaces_leaves_at_once() {
        let far_away = PatchKey::new(4, 40, 40);
        let wanted = [PatchKey::new(4, 2, 2)];
        let draw = draw_set(&wanted, &|_| false, &[far_away].into());
        assert!(!draw.contains(&far_away));
    }

    /// The hole is the voxel nodes' own box, in blocks -- the far terrain must
    /// neither draw over them nor leave a gap at their edge.
    #[test]
    fn the_hole_is_the_voxel_region_in_blocks() {
        let hole = voxel_hole([8.0, 300.0, 8.0]);
        let (min, max) =
            node::covered_box_3d(ChunkCoord::new(0, 18, 0), node::DEFAULT_RING_SCHEDULE);
        assert_eq!(hole.min, min.map(|c| (c * 16) as f64));
        assert_eq!(hole.max, max.map(|c| (c * 16) as f64));
        assert!(hole.min[0] < -1000.0 && hole.max[0] > 1000.0, "{hole:?}");
    }

    /// The same numbers the far terrain module measured (`far.rs`, §3.2),
    /// every one of them [`FAR_QUAD_SCALE`] times as wide -- and the scale
    /// asked for where one is: the quads, the error bound and the floor all
    /// follow it.
    #[test]
    fn the_view_uses_the_measured_settings() {
        let pixel = FOV_Y / 1080.0;
        for q in FarQuality::BEST_FIRST {
            let Some(game) = far_view([0.0; 3], 1080, None, q) else {
                assert_eq!(q, FarQuality::Off, "only Off draws nothing");
                continue;
            };
            let d = q.detail().unwrap();
            for (v, scale) in [
                (game, FAR_QUAD_SCALE),
                (far_view_scaled([0.0; 3], 1080, None, q, 1.0).unwrap(), 1.0),
                (far_view_scaled([0.0; 3], 1080, None, q, 3.5).unwrap(), 3.5),
            ] {
                assert!((v.max_error / pixel - d.error_px * scale).abs() < 1e-9);
                assert!((v.min_quad / pixel - d.floor_px * scale).abs() < 1e-9);
                let quad_angle = 1.0 / (v.split_ratio * far::PATCH_QUADS as f64);
                assert!((quad_angle / pixel - d.quad_px * scale).abs() < 1e-6);
                assert_eq!(v.radius, FAR_VIEW_RADIUS);
            }
        }
        // High is what `PROPOSAL_FAR_VIEW.md` §3.2 measured.
        let high = FarQuality::High.detail().unwrap();
        assert_eq!(
            (high.quad_px, high.error_px, high.floor_px),
            (16.0, 2.0, 4.0)
        );
    }

    /// What is kept between selections is enough to make the same selection
    /// again -- and keeping less is not, which is what makes this a test.
    ///
    /// The cache also holds everything generated around a second eye far
    /// away, as it would after the player moved: that is what the rule has
    /// to drop, and what it must not drop with it.
    ///
    /// At the quality's own sizes ([`SIM_QUAD_SCALE`]): at the game's wider
    /// quads (3) no ancestor's split reaches this eye's selection, so keeping
    /// less moves nothing and the test would test nothing (its last assert).
    #[test]
    fn keeping_the_ancestors_keeps_the_selection_still() {
        let gen = WorldGen::new(0x005E_ED00_00C0_FFEE);
        let view_at = |eye: [f64; 3]| FarView {
            radius: 20_000.0,
            ..far_view_scaled(
                eye,
                720,
                Some(voxel_hole(eye.map(|v| v as f32))),
                FarQuality::High,
                SIM_QUAD_SCALE,
            )
            .unwrap()
        };
        let mut heights: HashMap<PatchKey, PatchHeights> = HashMap::new();
        let converge = |view: &FarView, heights: &mut HashMap<PatchKey, PatchHeights>| loop {
            let wanted = far::select(view, |k| heights.get(&k).map(|p| p.error));
            let new: Vec<_> = wanted
                .iter()
                .filter(|k| !heights.contains_key(k))
                .copied()
                .collect();
            if new.is_empty() {
                return wanted;
            }
            for k in new {
                heights.insert(k, far::generate(&gen, k));
            }
        };
        converge(&view_at([60_000.0, 300.0, 60_000.0]), &mut heights);
        let here = view_at([8.0, 300.0, 8.0]);
        let wanted = converge(&here, &mut heights);
        let select = |h: &HashMap<PatchKey, PatchHeights>| {
            far::select(&here, |k| h.get(&k).map(|p| p.error))
        };

        let keep = to_keep(
            &wanted,
            wanted.iter().copied().collect(),
            far::root_level_for(here.radius),
        );
        let mut kept = heights.clone();
        kept.retain(|k, _| keep.contains(k));
        assert!(
            kept.len() < heights.len(),
            "nothing was dropped, so nothing was tested"
        );
        assert_eq!(
            select(&kept),
            wanted,
            "the selection moved with only the kept patches"
        );

        let mut bare = heights;
        bare.retain(|k, _| wanted.contains(k));
        assert_ne!(
            select(&bare),
            wanted,
            "without the ancestors it should have moved"
        );
    }

    /// **A root split for its roughness alone keeps its heights.** Its
    /// children are all that is wanted, so no wanted patch is of its level;
    /// kept only up to the coarsest wanted patch, it was dropped, selected
    /// again unsplit (its error unknown), generated, split and dropped
    /// again: `standing_still_selects_nothing` and six more never settled
    /// at [`FAR_QUAD_SCALE`], where a root can be too far to be split by
    /// distance.
    #[test]
    fn a_root_all_of_whose_ground_is_split_is_kept() {
        let root = PatchKey::new(7, 0, -1);
        let wanted: Vec<PatchKey> = root.children().to_vec();
        let keep = to_keep(&wanted, wanted.iter().copied().collect(), root.level);
        assert!(keep.contains(&root), "the root was dropped: {keep:?}");
        // Two levels down as well: every ancestor on the way up.
        let deeper: Vec<PatchKey> = wanted[0].children().to_vec();
        let keep = to_keep(&deeper, HashSet::new(), root.level);
        assert!(
            keep.contains(&root) && keep.contains(&wanted[0]),
            "{keep:?}"
        );
    }

    /// Fog ends at the far terrain's edge, not past it: past it, the edge of
    /// the world would be as hard as it was before fog existed.
    #[test]
    fn fog_finishes_inside_the_far_terrain() {
        let (_, fog_end) = cubara_render::Lighting::fog_range(FAR_VIEW_RADIUS as f32);
        assert!(fog_end < FAR_VIEW_RADIUS as f32 && fog_end > 0.5 * FAR_VIEW_RADIUS as f32);
    }

    #[test]
    fn the_renderer_and_the_world_agree_on_the_patch_size() {
        assert_eq!(cubara_render::FAR_QUADS, far::PATCH_QUADS);
        // A face travels in three words (`FarFace::pack`): every height the
        // terrain can have, and the tallest step between two of them, must
        // fit their fields, or a mountain is drawn clamped.
        let (lo, hi) = far::surface_bounds();
        assert!(cubara_render::FAR_TOP_RANGE.contains(&(lo.floor() as i32 - 1)));
        assert!(cubara_render::FAR_TOP_RANGE.contains(&(hi.ceil() as i32 + 1)));
        assert!((hi - lo).ceil() as u32 <= cubara_render::FAR_MAX_RISE);
        // And the renderer's face is the world's, field for field.
        let patch = far::generate(&WorldGen::new(7), PatchKey::new(5, -27, 151));
        let faces = to_far_faces(&patch);
        assert_eq!(faces.len(), patch.faces.len());
        for (ours, theirs) in faces.iter().zip(&patch.faces) {
            assert_eq!(ours.dir as usize, theirs.dir as usize);
            assert_eq!(
                (
                    ours.i0,
                    ours.j0,
                    ours.i1,
                    ours.j1,
                    ours.top,
                    ours.rise,
                    ours.border
                ),
                (
                    theirs.i0,
                    theirs.j0,
                    theirs.i1,
                    theirs.j1,
                    theirs.top,
                    theirs.rise,
                    theirs.border
                )
            );
        }
        assert!(faces.len() <= cubara_render::far::FAR_MAX_FACES);
        for (ours, theirs) in faces.iter().zip(&patch.faces) {
            assert_eq!((ours.mix, ours.open), (theirs.mix, theirs.open));
        }
    }

    // --- The stand-in for the voxels' strip (§9.11) ----------------------

    /// The voxels' box on the low flight's move at t 5.27 s (x crosses 128):
    /// the box it wants, and what of it it draws while the strip loads --
    /// `NodeStreaming`'s own rule.
    fn low_flight_move() -> VoxelBox {
        let box_at = |c: [i32; 3]| {
            node::covered_box_3d(ChunkCoord::new(c[0], c[1], c[2]), node::DEFAULT_RING_SCHEDULE)
        };
        let want = box_at([8, 6, 2]);
        let drawn = crate::streaming::next_far_hole(box_at([7, 6, 2]), want, |_| false);
        VoxelBox {
            wanted: hole_in_blocks(want),
            drawn: hole_in_blocks(drawn),
        }
    }

    /// **The stand-in is the finest far level over the strip and nothing
    /// else**: on the low flight's move, the strip x 1,152 to 1,280 the
    /// length of the box takes the column of nine patches of 256 blocks at
    /// x 1,024 -- none of the box's other edges, none beyond -- and nothing
    /// stands in while nothing waits.
    #[test]
    fn the_stand_in_is_the_finest_patches_over_the_strip() {
        let voxels = low_flight_move();
        assert_eq!(
            (voxels.wanted.max[0], voxels.drawn.max[0]),
            (1280.0, 1152.0),
            "the move's strip"
        );
        let keys = standin_keys(voxels).expect("the boxes differ across the ground only");
        let column: Vec<PatchKey> = (-4..=4).map(|z| PatchKey::new(3, 4, z)).collect();
        assert_eq!(keys, column);
        assert_eq!(far::FINEST_LEVEL, 3, "level 3's cells are quads of 8");
        let settled = VoxelBox::settled(voxels.wanted);
        assert_eq!(standin_keys(settled), Some(Vec::new()), "nothing waits");
    }

    /// **The strip is stood in for only while every top it can hold is inside
    /// both boxes**: a box whose floor or ceiling runs through the ground
    /// (an eye some 900 blocks above it or 770 under it) leaves a slab of
    /// ground under all of it to hand over, and the far terrain stands in
    /// for that as it always did ([`far_stand`] falls back).
    #[test]
    fn a_box_whose_floor_cuts_the_ground_has_no_stand_in() {
        let voxels = low_flight_move();
        let (lo, hi) = far::ground_range();
        let with_floor = |b: Hole, y: f64| Hole {
            min: [b.min[0], y, b.min[2]],
            ..b
        };
        let with_ceiling = |b: Hole, y: f64| Hole {
            max: [b.max[0], y, b.max[2]],
            ..b
        };
        for (what, cut) in [
            ("a floor at the lowest top", VoxelBox {
                wanted: with_floor(voxels.wanted, lo),
                drawn: with_floor(voxels.drawn, lo),
            }),
            ("a ceiling under the highest top", VoxelBox {
                wanted: with_ceiling(voxels.wanted, hi - 1.0),
                drawn: with_ceiling(voxels.drawn, hi - 1.0),
            }),
            ("the drawn box's floor at the lowest top", VoxelBox {
                drawn: with_floor(voxels.drawn, lo),
                ..voxels
            }),
        ] {
            assert_eq!(standin_keys(cut), None, "{what}");
            assert_eq!(far_stand(cut, |_| true), (cut.drawn, Vec::new()), "{what}");
        }
        // Just clear of the ground on both sides, it stands in.
        let clear = VoxelBox {
            wanted: with_ceiling(with_floor(voxels.wanted, lo - 1.0), hi),
            drawn: with_ceiling(with_floor(voxels.drawn, lo - 1.0), hi),
        };
        assert!(standin_keys(clear).is_some_and(|k| !k.is_empty()), "clear of the ground");
    }

    /// **The far terrain is the finished frame's only with every patch of
    /// the stand-in generated** ([`far_stand`]): holed at the box the voxels
    /// want and the stand-in drawn; with one patch missing, holed at the box
    /// they draw and nothing standing in; with nothing waiting, the box
    /// they want.
    #[test]
    fn the_far_terrain_is_the_finished_frames_once_the_stand_in_is_generated() {
        let voxels = low_flight_move();
        let keys = standin_keys(voxels).unwrap();
        assert_eq!(far_stand(voxels, |_| true), (voxels.wanted, keys.clone()));
        let missing = keys[4];
        assert_eq!(
            far_stand(voxels, |k| k != missing),
            (voxels.drawn, Vec::new()),
            "one patch of the stand-in still to come"
        );
        let settled = VoxelBox::settled(voxels.wanted);
        assert_eq!(far_stand(settled, |_| false), (voxels.wanted, Vec::new()));
    }

    /// **The ring holds every strip one move of the box can gain**: whichever
    /// way the eye crosses a chunk -- along x, along z, or both in one
    /// frame -- the box moves a coarsest node on, and every patch the
    /// stand-in then needs is in the ring round the box it left, so it was
    /// generated before the move.
    #[test]
    fn the_ring_holds_every_strip_a_move_can_gain() {
        let voxels = low_flight_move();
        let ring: HashSet<PatchKey> = standin_ring(voxels.wanted).into_iter().collect();
        let node = coarsest_node_blocks();
        let mut needed = 0;
        for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)] {
            let moved = Hole {
                min: [
                    voxels.wanted.min[0] + dx as f64 * node,
                    voxels.wanted.min[1],
                    voxels.wanted.min[2] + dz as f64 * node,
                ],
                max: [
                    voxels.wanted.max[0] + dx as f64 * node,
                    voxels.wanted.max[1],
                    voxels.wanted.max[2] + dz as f64 * node,
                ],
            };
            let drawn = overlap(voxels.wanted, moved).expect("a move keeps most of the box");
            let strip = standin_keys(VoxelBox {
                wanted: moved,
                drawn,
            })
            .expect("a move across the ground");
            for k in &strip {
                assert!(ring.contains(k), "({dx}, {dz}): {k:?} is not in the ring");
            }
            needed += strip.len();
        }
        assert!(needed >= 8 * 9, "{needed} patches over the eight moves' strips");
        // The ring is the box's edge, not its ground: none of it wholly
        // inside the box.
        let inside = |k: &PatchKey| {
            let [x, z] = k.origin().map(|o| o as f64);
            let size = k.size() as f64;
            voxels.wanted.min[0] <= x
                && x + size <= voxels.wanted.max[0]
                && voxels.wanted.min[2] <= z
                && z + size <= voxels.wanted.max[2]
        };
        assert!(!ring.iter().any(inside), "a patch inside the box is in the ring");
    }

    /// **Each face of the strip is the stand-in's exactly when the strip's
    /// nodes will draw it** ([`standin_faces`]): every top of the strip's
    /// quads, once; a wall on the strip's side towards the drawn voxels
    /// only where the strip is the higher (facing them), and one on its far
    /// side only where the strip is the higher (facing out) -- the rest are
    /// the voxels' and the far terrain's. On the patch of the strip the
    /// sliver looked through.
    #[test]
    fn the_stand_in_draws_each_face_the_strips_nodes_will_draw() {
        let voxels = low_flight_move();
        let gen = WorldGen::new(cubara_world::World::new().seed());
        let patch = far::generate(&gen, PatchKey::new(3, 4, -2));
        let faces = standin_faces(&patch, voxels);
        let tops: u32 = faces
            .iter()
            .filter(|f| f.dir == FarDir::Top)
            .map(|f| (f.i1 - f.i0) as u32 * (f.j1 - f.j0) as u32)
            .sum();
        // The strip is the patch's high half in x (i 16 to 32), all of z.
        assert_eq!(tops, 16 * 32, "the strip's quads, each under one top");
        for f in &faces {
            let [i, _] = f.owner();
            assert!(i >= 16, "{f:?} belongs to a drawn voxel's quad");
            match (f.dir, f.i0) {
                (FarDir::PosX, 16) => panic!("{f:?}: the drawn voxel's riser"),
                (FarDir::NegX, 32) => panic!("{f:?}: the far terrain's wall beyond the box"),
                _ => {}
            }
        }
        // Not vacuous: both of the strip's sides have walls of its own.
        let count = |dir, i0| faces.iter().filter(|f| f.dir == dir && f.i0 == i0).count();
        assert!(count(FarDir::NegX, 16) > 0, "no wall faces the drawn voxels");
        assert!(count(FarDir::PosX, 32) > 0, "no wall faces out of the box");
        let all = to_far_faces(&patch);
        assert!(
            all.iter().any(|f| f.dir == FarDir::PosX && f.i0 == 16),
            "the drawn voxels never stand higher here: the test tells nothing"
        );
        // The strip's end, where it meets the box's side z = 1,152 halfway
        // across its last patch: nothing beyond it, and the walls on it the
        // strip's only where it is the higher (facing out).
        let end = far::generate(&gen, PatchKey::new(3, 4, 4));
        let faces = standin_faces(&end, voxels);
        let tops: u32 = faces
            .iter()
            .filter(|f| f.dir == FarDir::Top)
            .map(|f| (f.i1 - f.i0) as u32 * (f.j1 - f.j0) as u32)
            .sum();
        assert_eq!(tops, 16 * 16, "the end patch's quads in the strip");
        for f in &faces {
            let [i, j] = f.owner();
            assert!(i >= 16 && j < 16, "{f:?} is not the strip's");
            assert!(
                !(f.dir == FarDir::NegZ && f.j0 == 16),
                "{f:?}: the far terrain's wall beyond the box"
            );
        }
        // Not vacuous: the ground beyond the box stands higher somewhere
        // along it, and the patch has the walls the stand-in leaves out.
        assert!(
            to_far_faces(&end)
                .iter()
                .any(|f| f.dir == FarDir::NegZ && f.j0 == 16),
            "nothing beyond the box stands higher: the test tells nothing"
        );
    }

    /// **A strip the ring does not hold is asked for at once**, and stood in
    /// for as soon as it is generated: the voxels' box moved twice before
    /// its first strip was drawn, so the second strip is a node beyond the
    /// ring round the box the plan last saw. Until it arrives the far
    /// terrain stands in itself, holed at the box the voxels draw.
    #[test]
    fn a_strip_the_ring_does_not_hold_is_asked_for_at_once() {
        let voxels = low_flight_move();
        let mut sim = Sim::at_scale(
            TEST_RADIUS,
            voxels.drawn,
            PoolModel::new(4, 2),
            FAR_QUAD_SCALE,
        );
        let eye = [133.0, 101.0, 45.0];
        sim.settle(eye);
        // Two nodes on in +x at once: a strip two nodes wide, the outer one
        // beyond the ring round `drawn`.
        let node = coarsest_node_blocks();
        let wanted = Hole {
            min: [voxels.drawn.min[0] + 2.0 * node, voxels.drawn.min[1], voxels.drawn.min[2]],
            max: [voxels.drawn.max[0] + 2.0 * node, voxels.drawn.max[1], voxels.drawn.max[2]],
        };
        sim.voxels = VoxelBox {
            wanted,
            drawn: overlap(voxels.drawn, wanted).expect("a move keeps most of the box"),
        };
        let strip = standin_keys(sim.voxels).expect("a strip across the ground");
        let ring = standin_ring(voxels.drawn);
        let beyond: Vec<PatchKey> = strip.iter().copied().filter(|k| !ring.contains(k)).collect();
        assert!(!beyond.is_empty(), "the ring holds the strip: the test tells nothing");
        let step = sim.frame(eye, Vec::new());
        for k in &beyond {
            assert!(
                step.request.contains(k) || step.promote.contains(k),
                "{k:?} of the strip was not asked for at once"
            );
        }
        assert!(sim.standin.is_empty(), "stood in before it was generated");
        assert_eq!(sim.plan.hole(), Some(sim.voxels.drawn), "holed at the box they want too soon");
        // The pool brings it; then the stand-in stands.
        for _ in 0..200 {
            let done = sim.pool.tick();
            sim.frame(eye, done);
            if !sim.standin.is_empty() {
                break;
            }
        }
        assert_eq!(sim.standin, strip.iter().copied().collect(), "the strip never stood in");
        assert_eq!(sim.plan.hole(), Some(sim.voxels.wanted));
    }

    // --- The plan without a GPU (#398, rule 6) --------------------------

    use std::cell::RefCell;

    /// Smaller than the game's radius so a flight runs in seconds; the cost
    /// report uses the game's. Far enough for what changes along the low
    /// flight -- the splits within 1 to 3 km -- to be the game's: a patch is
    /// split by its own distance and error, whatever the radius.
    const TEST_RADIUS: f64 = 4096.0;

    /// The gate's low flight: from (8, 120, 8) along (1, -0.15, 0.3) at the
    /// speed `flight::FLY_SPEED`, one eye per frame at 60 Hz.
    fn low_flight(frames: usize) -> Vec<[f64; 3]> {
        let (dx, dy, dz) = (1.0f64, -0.15f64, 0.3f64);
        let n = (dx * dx + dy * dy + dz * dz).sqrt();
        (0..frames)
            .map(|f| {
                let s = crate::flight::FLY_SPEED as f64 * f as f64 / 60.0 / n;
                [8.0 + dx * s, 120.0 + dy * s, 8.0 + dz * s]
            })
            .collect()
    }

    /// What the worker pool does, in frames: `workers` workers over the
    /// pool's two queues ([`Queues`]) in request order, the urgent one
    /// first, each patch taking `service_frames` frames.
    struct PoolModel {
        workers: usize,
        service_frames: u32,
        now: VecDeque<PatchKey>,
        ahead: VecDeque<PatchKey>,
        busy: Vec<(PatchKey, u32)>,
    }

    impl PoolModel {
        fn new(workers: usize, service_frames: u32) -> Self {
            Self {
                workers,
                service_frames,
                now: VecDeque::new(),
                ahead: VecDeque::new(),
                busy: Vec::new(),
            }
        }

        /// Every worker is as fast as a frame and there are plenty: what
        /// arrives is what was asked for in the frame before.
        fn instant() -> Self {
            Self::new(100_000, 1)
        }

        fn fill(&mut self) {
            while self.busy.len() < self.workers {
                let Some(key) = self.now.pop_front().or_else(|| self.ahead.pop_front()) else {
                    break;
                };
                self.busy.push((key, self.service_frames));
            }
        }

        /// What a [`Step`] asks of the pool, as [`FarStreaming::update`]
        /// asks it of [`Pool`].
        fn apply(&mut self, step: &Step) {
            for key in &step.promote {
                if let Some(at) = self.ahead.iter().position(|k| k == key) {
                    self.ahead.remove(at);
                    self.now.push_back(*key);
                }
            }
            self.now.extend(step.request.iter().copied());
            self.ahead.extend(step.prefetch.iter().copied());
            self.fill();
        }

        fn queued(&self) -> usize {
            self.now.len() + self.ahead.len()
        }

        /// One frame passes: what finished.
        fn tick(&mut self) -> Vec<PatchKey> {
            for (_, left) in &mut self.busy {
                *left -= 1;
            }
            let done = self
                .busy
                .iter()
                .filter(|(_, left)| *left == 0)
                .map(|(k, _)| *k)
                .collect();
            self.busy.retain(|(_, left)| *left > 0);
            self.fill();
            done
        }

        /// Everything asked for, finished now.
        fn flush(&mut self) -> Vec<PatchKey> {
            let mut all: Vec<PatchKey> = self.busy.drain(..).map(|(k, _)| k).collect();
            all.extend(self.now.drain(..));
            all.extend(self.ahead.drain(..));
            all
        }
    }

    /// A plan, a pool, and what a renderer would hold. Tests compare `shown`
    /// and `standin`, never anything the plan believes.
    struct Sim {
        plan: FarPlan,
        pool: PoolModel,
        shown: HashSet<PatchKey>,
        /// What the renderer holds as the stand-in for a strip, and the box
        /// it was cut for.
        standin: HashSet<PatchKey>,
        standin_box: Option<VoxelBox>,
        /// Generated heights, shared by the pool and the reference (a
        /// per-test cache, not a static: rule 2).
        memo: RefCell<HashMap<PatchKey, PatchHeights>>,
        gen: WorldGen,
        /// The voxels' box, as `NodeStreaming::voxel_box` hands it over.
        voxels: VoxelBox,
        radius: f64,
        /// The quad scale the plan and the reference ([`Sim::exact`]) both
        /// select at.
        scale: f64,
        /// `fly` asserts that drawn, still-wanted ground stays covered.
        check_coverage: bool,
    }

    /// The quad scale the plan's mechanics are tried at: quads of 16 pixels,
    /// the scenario these tests were written on. What they pin -- what is
    /// asked for, kept, swapped and dropped, and when -- does not depend on
    /// the scale; whether anything happens in three seconds of flight inside
    /// [`TEST_RADIUS`] does, and at [`FAR_QUAD_SCALE`] little would (the
    /// tests' own guards against proving nothing say so). The tests of what
    /// is drawn while flying fly the game's.
    const SIM_QUAD_SCALE: f64 = 1.0;

    /// Is the ground of `key` covered by `shown`: by `key` itself, by a
    /// coarser patch holding it, or by finer patches that together fill it?
    fn covered(shown: &HashSet<PatchKey>, key: PatchKey) -> bool {
        shown.contains(&key)
            || ancestors(key, 20).any(|a| shown.contains(&a))
            || (key.level > 0 && key.children().iter().all(|c| covered(shown, *c)))
    }

    impl Sim {
        fn new(radius: f64, hole: Hole, pool: PoolModel) -> Self {
            Self::at_scale(radius, hole, pool, SIM_QUAD_SCALE)
        }

        fn at_scale(radius: f64, hole: Hole, pool: PoolModel, scale: f64) -> Self {
            let seed = cubara_world::World::new().seed();
            let mut plan = FarPlan::new(FarQuality::High, radius);
            plan.quad_scale = scale;
            Self {
                plan,
                pool,
                shown: HashSet::new(),
                standin: HashSet::new(),
                standin_box: None,
                memo: RefCell::new(HashMap::new()),
                gen: WorldGen::new(seed),
                voxels: VoxelBox::settled(hole),
                radius,
                scale,
                check_coverage: false,
            }
        }

        fn heights(&self, key: PatchKey) -> PatchHeights {
            self.memo
                .borrow_mut()
                .entry(key)
                .or_insert_with(|| far::generate(&self.gen, key))
                .clone()
        }

        /// The converged selection at `eye`, holed at the box the voxels
        /// want -- the finished frame's: select against what is known,
        /// generate what is missing, repeat (the loop of `far::build`, and of
        /// `keeping_the_ancestors_keeps_the_selection_still`).
        fn exact(&self, eye: [f64; 3]) -> HashSet<PatchKey> {
            self.exact_holed(eye, self.voxels.wanted)
        }

        /// [`exact`](Self::exact) with the voxels' box `hole`.
        fn exact_holed(&self, eye: [f64; 3], hole: Hole) -> HashSet<PatchKey> {
            let view = FarView {
                radius: self.radius,
                ..far_view_scaled(eye, 1080, Some(hole), FarQuality::High, self.scale).unwrap()
            };
            loop {
                let wanted = far::select(&view, |k| self.memo.borrow().get(&k).map(|p| p.error));
                let missing: Vec<PatchKey> = wanted
                    .iter()
                    .copied()
                    .filter(|k| !self.memo.borrow().contains_key(k))
                    .collect();
                if missing.is_empty() {
                    return wanted.into_iter().collect();
                }
                for k in missing {
                    self.heights(k);
                }
            }
        }

        /// One frame: what the pool finished arrives, the plan decides, the
        /// renderer applies it (asserting it is consistent), and the new
        /// requests go to the pool.
        fn frame(&mut self, eye: [f64; 3], finished: Vec<PatchKey>) -> Step {
            for key in finished {
                let patch = self.heights(key);
                self.plan.take(patch);
            }
            let shown = &self.shown;
            let step = self.plan.step(eye, 1080, self.voxels, || shown.clone());
            for key in &step.leave {
                assert!(self.shown.remove(key), "{key:?} left but was not shown");
            }
            for key in &step.arrive {
                assert!(
                    self.plan.patch(*key).is_some(),
                    "{key:?} arrived without heights"
                );
                assert!(self.shown.insert(*key), "{key:?} arrived twice");
            }
            for key in &step.standin_leave {
                assert!(
                    self.standin.remove(key),
                    "{key:?} left the stand-in but was not in it"
                );
            }
            if !step.standin_leave.is_empty() && step.standin_arrive.is_empty() {
                self.standin_box = None;
            }
            if let Some(strip) = step.standin_box {
                assert!(self.standin.is_empty(), "a stand-in arrived over another");
                for key in &step.standin_arrive {
                    assert!(
                        self.plan.patch(*key).is_some(),
                        "{key:?} stood in without heights"
                    );
                    self.standin.insert(*key);
                }
                self.standin_box = Some(strip);
            }
            self.pool.apply(&step);
            step
        }

        /// Fly `eyes` through the modelled pool; `shown` after each frame.
        ///
        /// With `check_coverage`, every frame also asserts #262's rule at the
        /// renderer: ground that was covered by something shown last frame
        /// and is still wanted (by the exact selection at this eye) is still
        /// covered -- by the same patch, a coarser one, or finer ones.
        fn fly(&mut self, eyes: &[[f64; 3]]) -> Vec<HashSet<PatchKey>> {
            let check = self.check_coverage;
            eyes.iter()
                .enumerate()
                .map(|(frame, &eye)| {
                    let before = self.shown.clone();
                    let done = self.pool.tick();
                    self.frame(eye, done);
                    if check {
                        for want in self.exact(eye) {
                            assert!(
                                !covered(&before, want) || covered(&self.shown, want),
                                "frame {frame}: ground of {want:?} was drawn and is wanted, \
                                 and the renderer no longer covers it"
                            );
                        }
                    }
                    self.shown.clone()
                })
                .collect()
        }

        /// What a finished frame at `eye` is (`FarStreaming::settled`): ask
        /// again, and wait for everything with an instant pool.
        fn settle(&mut self, eye: [f64; 3]) {
            self.plan.reselect();
            for _ in 0..500 {
                let done = self.pool.flush();
                let step = self.frame(eye, done);
                if step == Step::default() && self.plan.settled() && self.pool.queued() == 0 {
                    return;
                }
            }
            panic!("the plan never settled");
        }
    }

    fn sim_at_start(pool: PoolModel) -> (Sim, [f64; 3]) {
        let start = low_flight(1)[0];
        let hole = voxel_hole(start.map(|v| v as f32));
        (Sim::new(TEST_RADIUS, hole, pool), start)
    }

    /// The overlap of two boxes, `None` when they do not overlap.
    fn overlap(a: Hole, b: Hole) -> Option<Hole> {
        let min = [0, 1, 2].map(|i| a.min[i].max(b.min[i]));
        let max = [0, 1, 2].map(|i| a.max[i].min(b.max[i]));
        (0..3).all(|i| min[i] < max[i]).then_some(Hole { min, max })
    }

    /// The voxels' box as `NodeStreaming::voxel_box` hands it over while the
    /// eye flies: the voxels want the box round the eye's chunk; when it
    /// moves, the part they draw is cut down to what the old and new boxes
    /// share at once, and grows to the new box `lag` frames later -- the
    /// frames its nodes take to be searched for, meshed and drawn.
    struct HoleLag {
        hole: Hole,
        wanted: Hole,
        since: usize,
        lag: usize,
        moves: usize,
    }

    impl HoleLag {
        fn new(eye: [f64; 3], lag: usize) -> Self {
            let hole = voxel_hole(eye.map(|v| v as f32));
            Self {
                hole,
                wanted: hole,
                since: 0,
                lag,
                moves: 0,
            }
        }

        /// The two boxes now.
        fn voxels(&self) -> VoxelBox {
            VoxelBox {
                wanted: self.wanted,
                drawn: self.hole,
            }
        }

        fn at(&mut self, eye: [f64; 3]) -> VoxelBox {
            let want = voxel_hole(eye.map(|v| v as f32));
            if want != self.wanted {
                self.moves += 1;
                self.since = 0;
                self.hole = overlap(self.hole, want).unwrap_or(want);
                self.wanted = want;
            } else {
                self.since += 1;
                if self.since >= self.lag {
                    self.hole = want;
                }
            }
            self.voxels()
        }
    }

    /// The low flight's frames whose drawn far terrain is not the exact
    /// selection at their own eye, flown at `scale` through a pool of 4
    /// workers taking 2 frames a patch (the M3's `cores / 2`, slower than
    /// the 6 to 9 ms a patch #398 step 1 measured), the voxels' box handed
    /// over 12 frames after each move. Also how often what was drawn
    /// changed, and how many patches were asked for, to show it was tried.
    fn frames_behind(scale: f64, frames: usize) -> Behind {
        let eyes = low_flight(frames);
        let mut holes = HoleLag::new(eyes[0], 12);
        let mut sim = Sim::at_scale(TEST_RADIUS, holes.wanted, PoolModel::new(4, 2), scale);
        sim.settle(eyes[0]);
        let mut out = Behind::default();
        for (f, &eye) in eyes.iter().enumerate() {
            sim.voxels = holes.at(eye);
            let before = sim.shown.clone();
            let done = sim.pool.tick();
            let step = sim.frame(eye, done);
            out.asked += step.request.len() + step.prefetch.len();
            out.changes += usize::from(sim.shown != before);
            let at = format!("frame {f} (t {:.2} s)", f as f64 / 60.0);
            // The finished frame's far terrain: the eye's own selection,
            // holed at the box the voxels want.
            let exact = sim.exact(eye);
            if sim.shown != exact {
                let mut drawn: Vec<_> = sim.shown.difference(&exact).copied().collect();
                let mut wanted: Vec<_> = exact.difference(&sim.shown).copied().collect();
                drawn.sort();
                wanted.sort();
                out.behind.push(format!(
                    "{at}: drawn {drawn:?}, the eye's own {wanted:?}"
                ));
            }
            // ...and over the strip the box still loads, level 3.
            let strip: HashSet<PatchKey> = standin_keys(sim.voxels)
                .expect("the low flight's boxes differ across the ground only")
                .into_iter()
                .collect();
            out.waiting += usize::from(!strip.is_empty());
            let cut_for = if strip.is_empty() {
                None
            } else {
                Some(sim.voxels)
            };
            if sim.standin != strip || sim.standin_box != cut_for {
                out.behind.push(format!(
                    "{at}: the stand-in is {:?} cut for {:?}, the strip's {strip:?}",
                    sim.standin, sim.standin_box
                ));
            }
        }
        out.moves = holes.moves;
        out
    }

    /// What [`frames_behind`] found.
    #[derive(Default)]
    struct Behind {
        /// The frames whose far terrain is not the finished frame's, and why.
        behind: Vec<String>,
        /// Frames in which the drawn set changed.
        changes: usize,
        /// Patches asked for in flight.
        asked: usize,
        /// Frames with a strip of the voxels' box waiting for its nodes.
        waiting: usize,
        /// Moves of the voxels' box.
        moves: usize,
    }

    /// **What is drawn is the far terrain the eye itself selects, on every
    /// frame of the low flight** (#398), at the game's quad scale and at
    /// 3. The low flight's holes were frames where it was not: at 3 a
    /// level-5 patch split at t 2.98 s and was drawn whole for two more
    /// frames while its children were generated (the quiet M3's hole at
    /// (1804, 352) and 600 stale pixels at t 3.0); at 2.7 a level-4 patch at
    /// t 11.30 s ((542, 388)); and at every move of the voxels' box the
    /// patches the hole had swallowed behind it were drawn a frame late.
    /// The plan as it was showed 21 such frames at 3 and 10 at 2.7 (the
    /// game's radius, four workers at a frame a patch). The selection is
    /// now worked out again within its slack and the patches a little
    /// ahead of the eye are generated before it wants them.
    ///
    /// **And while a strip of the voxels' box waits for its nodes, the far
    /// terrain is still the finished frame's**, holed at the box the voxels
    /// want, with level 3's stand-in over the strip (§9.11) -- from the
    /// frame the box moves, because the ring round it was generated before.
    /// Until then the far terrain was selected and holed at the box the
    /// voxels draw, and stood in for the strip at its own quads.
    #[test]
    fn the_live_selection_is_the_exact_one_on_every_frame() {
        for scale in [FAR_QUAD_SCALE, 3.0] {
            let found = frames_behind(scale, 1200);
            println!(
                "x{scale}: {} of 1200 frames behind; drawn set changed in {}; {} patches \
                 asked for; the voxels' box moved {} times, a strip waited in {} frames",
                found.behind.len(),
                found.changes,
                found.asked,
                found.moves,
                found.waiting,
            );
            assert!(
                found.behind.is_empty(),
                "x{scale}: {} frames draw other far terrain than the finished frame:\n{}",
                found.behind.len(),
                found.behind.join("\n")
            );
            // Not vacuous: the far terrain changed under the eye, patches
            // had to be generated on the way (most were generated ahead,
            // while the plan settled at the start), the voxels' box moved,
            // and its strips waited (12 frames a move).
            assert!(
                found.changes >= 8,
                "x{scale}: the drawn set changed {} times",
                found.changes
            );
            assert!(found.asked > 0, "x{scale}: nothing was asked for in flight");
            assert!(found.moves >= 3, "x{scale}: the box moved {} times", found.moves);
            assert!(
                found.waiting >= 12 * found.moves,
                "x{scale}: a strip waited in {} frames",
                found.waiting
            );
        }
    }

    /// With everything arrived and the eye where it was, what is shown is the
    /// converged selection -- the reference the flight's settled pass uses.
    #[test]
    fn a_settled_plan_shows_the_exact_selection() {
        let (mut sim, start) = sim_at_start(PoolModel::new(4, 2));
        sim.settle(start);
        assert_eq!(sim.shown, sim.exact(start));
        assert!(sim.shown.len() > 100, "{} patches", sim.shown.len());
    }

    /// **At the game's quad scale the plan settles, on the exact
    /// selection.** With quads this wide a root of the test's radius is too
    /// far to be split by distance and is split for its roughness alone;
    /// its heights were then dropped (`to_keep` kept ancestors only up to
    /// the coarsest wanted patch), and the plan asked for it, split it and
    /// dropped it again without end -- "the plan never settled", which this
    /// was red with. Not vacuous: some root here is split, all of it.
    #[test]
    fn a_plan_at_the_games_quad_scale_settles_on_the_exact_selection() {
        let start = low_flight(1)[0];
        let hole = voxel_hole(start.map(|v| v as f32));
        let mut sim = Sim::at_scale(TEST_RADIUS, hole, PoolModel::new(4, 2), FAR_QUAD_SCALE);
        sim.settle(start);
        assert_eq!(sim.shown, sim.exact(start));
        let root_level = far::root_level_for(TEST_RADIUS);
        assert!(
            sim.shown.iter().all(|k| k.level < root_level),
            "a root is drawn whole, so no root's heights were needed"
        );
        // And it stays settled: nothing more is selected, asked for or moved.
        let (selects, prefetches) = (sim.plan.selects, sim.plan.prefetches);
        for _ in 0..20 {
            let done = sim.pool.tick();
            assert_eq!(sim.frame(start, done), Step::default());
        }
        assert_eq!(
            (sim.plan.selects, sim.plan.prefetches),
            (selects, prefetches)
        );
    }

    /// A settled plan at a standing eye costs nothing: no step, no select,
    /// no prefetch.
    #[test]
    fn standing_still_selects_nothing() {
        let (mut sim, start) = sim_at_start(PoolModel::instant());
        sim.settle(start);
        let (selects, prefetches) = (sim.plan.selects, sim.plan.prefetches);
        for _ in 0..100 {
            let done = sim.pool.tick();
            assert_eq!(sim.frame(start, done), Step::default());
        }
        assert_eq!(sim.plan.selects, selects, "a standing eye selected again");
        assert_eq!(
            sim.plan.prefetches, prefetches,
            "a standing eye prefetched again"
        );
    }

    /// The things that make the plan work again besides the eye: a new hole,
    /// a new screen height, and `reselect`.
    #[test]
    fn a_new_hole_screen_height_or_reselect_works_the_selection_out_again() {
        let (mut sim, start) = sim_at_start(PoolModel::instant());
        sim.settle(start);
        let n = sim.plan.selects;
        let (hole, shown) = (sim.voxels.wanted, sim.shown.clone());
        sim.plan.reselect();
        let settled = VoxelBox::settled;
        sim.plan.step(start, 1080, settled(hole), || shown.clone());
        assert_eq!(sim.plan.selects, n + 1, "reselect did nothing");
        sim.plan.step(start, 720, settled(hole), || shown.clone());
        assert_eq!(sim.plan.selects, n + 2, "a new screen height did nothing");
        let hole = Hole {
            min: [hole.min[0] + 16.0, hole.min[1], hole.min[2]],
            ..hole
        };
        sim.plan.step(start, 720, settled(hole), || shown.clone());
        assert_eq!(sim.plan.selects, n + 3, "a new hole did nothing");
        sim.plan.step(start, 720, settled(hole), || shown.clone());
        assert_eq!(sim.plan.selects, n + 3, "nothing changed but it selected");
    }

    /// **The selection is worked out again only once the eye has moved as
    /// far as its slack**, and then at once: half the slack away nothing is
    /// selected, a little past it the plan selects. A standing eye whose
    /// slack is nothing at all selects nothing either.
    #[test]
    fn the_selection_is_worked_out_again_once_the_eye_has_moved_its_slack() {
        let (mut sim, start) = sim_at_start(PoolModel::instant());
        sim.settle(start);
        let slack = sim.plan.slack;
        assert!(
            slack > 0.5 && slack.is_finite(),
            "a slack of {slack} blocks tests nothing"
        );
        let n = sim.plan.selects;
        let (hole, shown) = (sim.voxels, sim.shown.clone());
        let at = |d: f64| [start[0] + d, start[1], start[2]];
        sim.plan.step(at(slack * 0.5), 1080, hole, || shown.clone());
        assert_eq!(sim.plan.selects, n, "half the slack away it selected");
        sim.plan
            .step(at(slack * 1.01), 1080, hole, || shown.clone());
        assert_eq!(sim.plan.selects, n + 1, "past the slack it did not select");
        sim.plan.slack = 0.0;
        let here = at(slack * 1.01);
        sim.plan.step(here, 1080, hole, || shown.clone());
        assert_eq!(sim.plan.selects, n + 1, "a standing eye selected");
    }

    /// **A patch the selection does not want selects nothing when it
    /// arrives** (#398 §6.2's lever): none of the selection's decisions read
    /// its error. One it wants does, and one the prefetch walked works the
    /// prefetch out again.
    #[test]
    fn only_an_arrival_the_selection_wants_works_it_out_again() {
        let (mut sim, start) = sim_at_start(PoolModel::instant());
        sim.settle(start);
        let (n, p) = (sim.plan.selects, sim.plan.prefetches);
        let (hole, shown) = (sim.voxels, sim.shown.clone());
        // A patch nobody wants, far off.
        let stray = PatchKey::new(5, 40, 40);
        sim.plan.take(sim.heights(stray));
        sim.plan.step(start, 1080, hole, || shown.clone());
        assert_eq!(
            (sim.plan.selects, sim.plan.prefetches),
            (n, p),
            "a stray arrival"
        );
        // One the selection wants, generated again. (The prefetch walks
        // every patch the selection does, so it is worked out again too.)
        let wanted = *sim.plan.wanted.iter().min().expect("something is wanted");
        sim.plan.take(sim.heights(wanted));
        sim.plan.step(start, 1080, hole, || shown.clone());
        assert_eq!(sim.plan.selects, n + 1, "a wanted arrival selected nothing");
        let p = sim.plan.prefetches;
        // One the prefetch walked and the selection does not want.
        let ahead = *sim
            .plan
            .ahead
            .iter()
            .filter(|k| !sim.plan.wanted.contains(k))
            .min()
            .expect("the prefetch walks more than is wanted");
        sim.plan.take(sim.heights(ahead));
        sim.plan.step(start, 1080, hole, || shown.clone());
        assert_eq!(sim.plan.selects, n + 1, "a prefetch arrival selected");
        assert_eq!(
            sim.plan.prefetches,
            p + 1,
            "a prefetch arrival did not prefetch"
        );
    }

    /// **The prefetch walks every patch a nearby eye selects** -- the
    /// promise [`ahead_view`] makes. By distance alone (no errors known, so
    /// every split is the split distance's or the width floor's), at the low
    /// flight's start and halfway, the patches the eye selects from 24
    /// blocks away in each of eight directions are all among the patches the
    /// prefetch walks from here, kept or split; and from a point a whole
    /// margin further on they are not, which is what makes the margin the
    /// thing tested.
    #[test]
    fn the_prefetch_walks_every_patch_a_nearby_eye_selects() {
        let view_at = |eye: [f64; 3]| FarView {
            radius: TEST_RADIUS,
            ..far_view_scaled(
                eye,
                1080,
                Some(voxel_hole([8.0, 120.0, 8.0])),
                FarQuality::High,
                SIM_QUAD_SCALE,
            )
            .unwrap()
        };
        let mut missed_far_off = 0;
        for eye in [[8.0, 120.0, 8.0], [240.0, 85.0, 75.0]] {
            let ahead = far::select_walk(&ahead_view(&view_at(eye)), |_| None);
            let walked: HashSet<PatchKey> = ahead.patches.into_iter().chain(ahead.split).collect();
            for (dx, dz) in [
                (1.0, 0.0),
                (-1.0, 0.0),
                (0.0, 1.0),
                (0.0, -1.0),
                (0.7, 0.7),
                (-0.7, 0.7),
                (0.7, -0.7),
                (-0.7, -0.7),
            ] {
                let near = [eye[0] + 24.0 * dx, eye[1], eye[2] + 24.0 * dz];
                for k in far::select(&view_at(near), |_| None) {
                    assert!(
                        walked.contains(&k),
                        "{k:?}, selected 24 blocks from {eye:?}, was not prefetched there"
                    );
                }
                let far_off = [eye[0] + 400.0 * dx, eye[1], eye[2] + 400.0 * dz];
                missed_far_off += far::select(&view_at(far_off), |_| None)
                    .into_iter()
                    .filter(|k| !walked.contains(k))
                    .count();
            }
        }
        assert!(
            missed_far_off > 0,
            "the prefetch held every patch even 400 blocks off: the margin tests nothing"
        );
    }

    /// **The prefetch waits behind what the eye wants**: the pool's workers
    /// take the urgent queue first, and a patch the prefetch asked for that
    /// the eye now wants moves to it.
    #[test]
    fn the_prefetch_waits_behind_what_the_eye_wants() {
        let (a, b, c) = (
            PatchKey::new(4, 0, 0),
            PatchKey::new(4, 1, 0),
            PatchKey::new(4, 2, 0),
        );
        let mut pool = PoolModel::new(1, 1);
        // The worker is busy with `a`; `b` and `c` wait for the prefetch.
        pool.apply(&Step {
            request: vec![a],
            prefetch: vec![b, c],
            ..Step::default()
        });
        // `c` is wanted now.
        pool.apply(&Step {
            promote: vec![c],
            ..Step::default()
        });
        assert_eq!(pool.tick(), vec![a]);
        assert_eq!(pool.tick(), vec![c], "the promoted patch did not go first");
        assert_eq!(pool.tick(), vec![b]);
        // And the plan asks so: a wanted patch already asked for by the
        // prefetch is promoted, not asked for twice.
        let (mut sim, start) = sim_at_start(PoolModel::new(1, 50));
        let first = sim.frame(start, Vec::new());
        assert!(!first.request.is_empty() && !first.prefetch.is_empty());
        assert!(
            first.request.iter().all(|k| !first.prefetch.contains(k)),
            "a patch was asked for urgently and for the prefetch at once"
        );
    }

    /// The real pool's queues: the urgent one first, a promoted prefetch
    /// at its back, each queue in the order asked; and a closed pool hands
    /// its workers nothing.
    #[test]
    fn the_pool_takes_the_urgent_queue_first() {
        let gen = WorldGen::new(1);
        let job = |x: i32| (0u64, gen, PatchKey::new(4, x, 0));
        let mut q = Queues::default();
        q.push(job(1), Urgency::Ahead);
        q.push(job(2), Urgency::Ahead);
        q.push(job(3), Urgency::Now);
        q.push(job(4), Urgency::Ahead);
        q.promote(PatchKey::new(4, 2, 0));
        q.promote(PatchKey::new(4, 9, 0));
        let order: Vec<i32> = std::iter::from_fn(|| q.take()).map(|j| j.2.x).collect();
        assert_eq!(order, vec![3, 2, 1, 4]);
        q.push(job(5), Urgency::Now);
        q.clear();
        assert!(q.take().is_none(), "clear left a job");
        let closed = (
            Mutex::new(Queues {
                closed: true,
                ..Queues::default()
            }),
            Condvar::new(),
        );
        assert!(
            next_job(&closed).is_none(),
            "a closed pool handed out a job"
        );
        let open = (Mutex::new(Queues::default()), Condvar::new());
        open.0.lock().unwrap().push(job(6), Urgency::Ahead);
        assert_eq!(next_job(&open).map(|j| j.2.x), Some(6));
    }

    /// **A patch the prefetch asked for and the eye now wants is promoted**,
    /// not asked for a second time; the plan then counts it as urgent.
    #[test]
    fn a_wanted_patch_the_prefetch_asked_for_is_promoted() {
        let (mut sim, start) = sim_at_start(PoolModel::instant());
        sim.settle(start);
        let k = *sim.plan.wanted.iter().min().expect("something is wanted");
        sim.plan.heights.remove(&k);
        sim.plan.in_flight.insert(k, Urgency::Ahead);
        sim.plan.reselect();
        let shown = sim.shown.clone();
        let step = sim.plan.step(start, 1080, sim.voxels, || shown.clone());
        assert_eq!(step.promote, vec![k], "not promoted");
        assert!(!step.request.contains(&k), "asked for twice");
        assert_eq!(sim.plan.in_flight.get(&k), Some(&Urgency::Now));
    }

    /// **When the voxels' box moves, the far terrain it hands back is
    /// already generated**: the patches its hole swallowed on the side it
    /// leaves are drawn on the frame of the move, and nothing is asked for
    /// urgently. The plan as it was drew that side a frame or more late,
    /// with nothing in its place (behind the eye on the low flight; in sight
    /// of anyone looking back).
    #[test]
    fn a_box_move_finds_what_it_hands_back_generated() {
        // x crosses 128 between these two eyes: the box moves +x by a
        // coarsest node.
        let (before, after) = ([127.0, 102.0, 44.0], [129.0, 102.0, 44.0]);
        let mut holes = HoleLag::new(before, 1000);
        let mut sim = Sim::at_scale(
            TEST_RADIUS,
            holes.wanted,
            PoolModel::new(4, 2),
            FAR_QUAD_SCALE,
        );
        sim.settle(before);
        let old = sim.voxels.wanted;
        sim.voxels = holes.at(after);
        assert_eq!(holes.moves, 1, "the box did not move");
        let step = sim.frame(after, Vec::new());
        assert!(
            step.request.is_empty() && step.promote.is_empty(),
            "the move needed {:?} and {:?}",
            step.request,
            step.promote
        );
        assert_eq!(sim.shown, sim.exact(after), "the move drew late");
        // ...and the strip the box gained stands in as level 3 from that
        // frame (§9.11): the ring round the old box had it generated.
        let strip = standin_keys(sim.voxels).expect("a strip across the ground only");
        assert!(!strip.is_empty(), "the move gained no strip");
        assert_eq!(
            sim.standin,
            strip.iter().copied().collect(),
            "the strip did not stand in on the frame of the move"
        );
        // Not vacuous: ground the old hole swallowed arrived.
        assert!(
            step.arrive
                .iter()
                .any(|k| (k.origin()[0] as f64) < old.min[0] + coarsest_node_blocks()),
            "nothing arrived on the side the box left: {:?}",
            step.arrive
        );
    }

    /// **The plan selects at the quad scale it is given** -- the game's
    /// ([`FAR_QUAD_SCALE`]) unless told another (`--far-quad-scale`, through
    /// [`FarStreaming::with_quad_scale`]): what it first asks for is what
    /// [`far_view_scaled`] at that scale selects, and a finer scale asks for
    /// more.
    #[test]
    fn the_plan_selects_at_its_quad_scale() {
        let eye = [8.0, 120.0, 8.0];
        let hole = voxel_hole(eye.map(|v| v as f32));
        let asked = |scale: Option<f64>| -> HashSet<PatchKey> {
            let mut plan = FarPlan::new(FarQuality::Medium, TEST_RADIUS);
            if let Some(scale) = scale {
                plan.quad_scale = scale;
            }
            let step = plan.step(eye, 1080, VoxelBox::settled(hole), HashSet::new);
            step.request.into_iter().collect()
        };
        let selected = |scale: f64| -> HashSet<PatchKey> {
            let view = FarView {
                radius: TEST_RADIUS,
                ..far_view_scaled(eye, 1080, Some(hole), FarQuality::Medium, scale).unwrap()
            };
            far::select(&view, |_| None).into_iter().collect()
        };
        assert_eq!(asked(None), selected(FAR_QUAD_SCALE), "the game's scale");
        assert_eq!(asked(Some(1.0)), selected(1.0), "a scale asked for");
        assert!(
            selected(1.0).len() > selected(FAR_QUAD_SCALE).len(),
            "the two scales select alike, and the test tells nothing"
        );
    }

    /// `--far-quad-scale` takes a number from 0.5 to 8 and refuses the rest:
    /// a typo must not be measured as some other scale.
    #[test]
    fn a_quad_scale_is_a_number_in_range_or_refused() {
        assert_eq!(parse_quad_scale("2.5"), Some(2.5));
        assert_eq!(parse_quad_scale("3"), Some(3.0));
        assert_eq!(parse_quad_scale("0.5"), Some(0.5));
        assert_eq!(parse_quad_scale("8"), Some(8.0));
        for wrong in ["", "x3", "3x", "0", "0.4", "8.5", "-2", "nan", "inf"] {
            assert_eq!(parse_quad_scale(wrong), None, "{wrong:?}");
        }
    }

    /// `Off` draws nothing beyond the voxels: everything shown leaves, and
    /// nothing is asked for -- nor stood in for a strip still loading.
    #[test]
    fn off_removes_everything_and_asks_for_nothing() {
        let mut plan = FarPlan::new(FarQuality::Off, TEST_RADIUS);
        let drawn: HashSet<PatchKey> = [
            PatchKey::new(3, 2, 2),
            PatchKey::new(3, 1, 5),
            PatchKey::new(4, 0, 0),
        ]
        .into();
        let step = plan.step([133.0, 101.0, 45.0], 1080, low_flight_move(), || {
            drawn.clone()
        });
        let mut sorted: Vec<_> = drawn.into_iter().collect();
        sorted.sort();
        assert_eq!(step.leave, sorted);
        assert!(step.request.is_empty() && step.prefetch.is_empty() && step.arrive.is_empty());
        assert!(step.standin_arrive.is_empty() && step.promote.is_empty(), "a stand-in");
    }

    /// A patch asked for is not asked for again while it is generating, and
    /// the ones nearest the eye (finest levels) go first.
    #[test]
    fn requests_are_made_once_and_finest_first() {
        let (mut sim, start) = sim_at_start(PoolModel::new(4, 2));
        let step = sim.frame(start, Vec::new());
        assert!(!step.request.is_empty());
        for asked in [&step.request, &step.prefetch] {
            assert!(
                asked.windows(2).all(|w| w[0].level <= w[1].level),
                "not finest first"
            );
        }
        sim.plan.reselect();
        let again = sim.frame(start, Vec::new());
        let before: HashSet<_> = step.request.iter().chain(&step.prefetch).collect();
        assert!(
            again
                .request
                .iter()
                .chain(&again.prefetch)
                .all(|k| !before.contains(k)),
            "a patch still generating was asked for again"
        );
    }

    /// `far_plan_cost_report`'s flight, timed: what one `step` costs on the
    /// main thread (#398 §6.2). Not a check -- a machine's clock is not CI's:
    ///
    /// ```text
    /// cargo test --release -p cubara --bin cubara -- --ignored --nocapture far_plan_cost_report
    /// ```
    ///
    /// The game's radius, the low flight at 60 Hz for 600 frames, the pool
    /// modelled as 4 workers (the M3's `cores / 2`) at 2 frames a patch. The
    /// memo is warmed by a first, untimed flight so generation is not in the
    /// numbers.
    #[test]
    #[ignore = "a measurement, not a check: prints what the plan's step costs"]
    fn far_plan_cost_report() {
        let eyes = low_flight(600);
        let start = eyes[0];
        let hole = voxel_hole(start.map(|v| v as f32));
        let scale = std::env::var("FAR_PLAN_SCALE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(FAR_QUAD_SCALE);
        let game = |pool| Sim::at_scale(FAR_VIEW_RADIUS, hole, pool, scale);
        let mut sim = game(PoolModel::new(4, 2));
        sim.settle(start);
        sim.fly(&eyes);
        // The timed flight, from a fresh plan over the warm memo.
        let memo = std::mem::take(&mut sim.memo);
        let mut sim = game(PoolModel::new(4, 2));
        sim.memo = memo;
        sim.settle(start);
        let (selects_before, prefetches_before) = (sim.plan.selects, sim.plan.prefetches);
        let mut holes = HoleLag::new(start, 12);
        let mut times = Vec::new();
        for &eye in &eyes {
            sim.voxels = holes.at(eye);
            let finished = sim.pool.tick();
            for key in finished {
                let patch = sim.heights(key);
                sim.plan.take(patch);
            }
            let shown = &sim.shown;
            let t = std::time::Instant::now();
            let step = sim.plan.step(eye, 1080, sim.voxels, || shown.clone());
            times.push(t.elapsed());
            for key in &step.leave {
                sim.shown.remove(key);
            }
            sim.shown.extend(step.arrive.iter().copied());
            sim.pool.apply(&step);
        }
        let ms = |d: &std::time::Duration| d.as_secs_f64() * 1e3;
        let sum: f64 = times.iter().map(ms).sum();
        let mut sorted: Vec<f64> = times.iter().map(ms).collect();
        sorted.sort_by(f64::total_cmp);
        let p99 = sorted[(sorted.len() * 99 / 100).min(sorted.len() - 1)];
        println!(
            "FAR_PLAN_COST x{scale} frames {} sum {sum:.2} ms mean {:.4} ms p99 {p99:.3} ms \
             max {:.3} ms; selects {} prefetches {}; shown {}; held {}",
            times.len(),
            sum / times.len() as f64,
            sorted.last().unwrap(),
            sim.plan.selects - selects_before,
            sim.plan.prefetches - prefetches_before,
            sim.shown.len(),
            sim.plan.heights.len(),
        );
    }

    /// A patch taken in is not "settled" until a selection has seen it; and
    /// clearing forgets what was being generated.
    #[test]
    fn settled_means_nothing_generating_and_nothing_unseen() {
        let (mut sim, start) = sim_at_start(PoolModel::new(4, 2));
        let step = sim.frame(start, Vec::new());
        assert!(!sim.plan.settled(), "patches are generating");
        sim.plan.clear();
        assert!(sim.plan.settled(), "clear kept what was generating");
        // A wanted patch arriving, which no selection has seen yet.
        sim.plan.wanted.insert(step.request[0]);
        let patch = sim.heights(step.request[0]);
        sim.plan.take(patch);
        assert!(!sim.plan.settled(), "an arrival no selection has seen");
        // A patch the prefetch walked, arriving: its error may split more
        // of what the prefetch walks, so not settled until it has seen it.
        sim.plan.clear();
        let ahead = step.prefetch[0];
        sim.plan.ahead.insert(ahead);
        let patch = sim.heights(ahead);
        sim.plan.take(patch);
        assert!(
            !sim.plan.settled(),
            "a prefetch arrival no prefetch has seen"
        );
    }

    /// Patches nothing wants any more are dropped: after moving far away and
    /// settling, the plan holds the exact selection's patches, their
    /// ancestors and what the prefetch walked -- not everything it ever
    /// generated.
    #[test]
    fn what_is_no_longer_wanted_is_dropped() {
        let (mut sim, start) = sim_at_start(PoolModel::instant());
        sim.settle(start);
        let there = [start[0] + 3_000.0, start[1], start[2] + 900.0];
        sim.voxels = VoxelBox::settled(voxel_hole(there.map(|v| v as f32)));
        sim.settle(there);
        assert_eq!(sim.shown, sim.exact(there));
        let ever = sim.memo.borrow().len();
        assert!(
            sim.plan.heights.len() < ever,
            "{} held of {ever} generated: nothing was dropped",
            sim.plan.heights.len()
        );
        assert!(sim.shown.iter().all(|k| sim.plan.heights.contains_key(k)));
    }

    /// #262's rule, seen from the renderer: nothing is taken away before what
    /// replaces it exists. Flown through the modelled pool (4 workers, 2
    /// frames a patch), so patches do arrive late and `draw_set` has to hold
    /// the old ones; `fly` asserts the coverage on every frame.
    #[test]
    fn drawn_ground_stays_covered_while_flying() {
        let (mut sim, start) = sim_at_start(PoolModel::new(4, 2));
        sim.settle(start);
        sim.check_coverage = true;
        let eyes = low_flight(180);
        let shown = sim.fly(&eyes);
        // Not vacuous: the drawn set really changed while flying.
        let changes = shown.windows(2).filter(|w| w[0] != w[1]).count();
        assert!(changes >= 3, "only {changes} frames changed what is shown");
    }
}

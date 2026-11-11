//! Streams the far terrain around the camera (`docs/PROPOSAL_FAR_VIEW.md`,
//! block F3c): which patches the view wants ([`cubara_world::far::select`]),
//! generating the new ones on worker threads, and swapping them into the
//! renderer's [`cubara_render::FarTerrain`] without leaving holes.
//!
//! The same rule the voxel nodes learned in #262 holds here: nothing is taken
//! away before what replaces it exists. A patch the view no longer wants stays
//! drawn while any patch covering the same ground is still being generated.

use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{Receiver, Sender};

use cubara_render::aggregate::Materials;
use cubara_render::{FarParams, FarPatch, FarSlot, Renderer};
#[cfg(test)]
use cubara_voxel::ChunkCoord;
use cubara_world::far::{self, FarView, Hole, PatchHeights, PatchKey};
#[cfg(test)]
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

/// How far the eye moves before the selection is worked out again, in blocks.
/// The finest far patches are a few hundred blocks wide, so this is well
/// inside one of them.
const RESELECT_BLOCKS: f64 = 32.0;

/// The view the far terrain is selected for at `quality`, scaled to a screen
/// `screen_height` pixels tall, leaving `hole` to the voxels. `None` when the
/// quality is `Off`.
pub fn far_view(
    eye: [f64; 3],
    screen_height: u32,
    hole: Option<Hole>,
    quality: FarQuality,
) -> Option<FarView> {
    let d = quality.detail()?;
    let pixel = FOV_Y / screen_height.max(1) as f64;
    Some(FarView {
        eye,
        split_ratio: far::split_ratio_for(d.quad_px, screen_height.max(1) as f64, FOV_Y),
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
/// (`NodeStreaming::far_hole`).
#[cfg(test)]
pub fn voxel_hole(eye: [f32; 3]) -> Hole {
    hole_in_blocks(node::covered_box_3d(
        ChunkCoord::from_world_pos(eye),
        node::DEFAULT_RING_SCHEDULE,
    ))
}

/// The renderer's form of a patch.
pub fn to_far_patch(p: &PatchHeights) -> FarPatch<'_> {
    FarPatch {
        // The integer it is: the renderer subtracts its anchor from it
        // exactly (issue #379).
        origin: p.key.origin(),
        quad: p.key.quad() as f32,
        heights: &p.heights,
        skirt: p.skirt(),
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
fn to_keep(wanted: &[PatchKey], draw: HashSet<PatchKey>) -> HashSet<PatchKey> {
    let top = top_level(wanted);
    let mut keep = draw;
    for &k in wanted {
        keep.insert(k);
        keep.extend(ancestors(k, top));
    }
    keep
}

/// A pool of threads generating patches. Each job carries the generation it
/// was asked for in, so a result from before a reset (a new world) is dropped.
struct Pool {
    jobs: Sender<(u64, WorldGen, PatchKey)>,
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
        let (jobs, job_rx) = std::sync::mpsc::channel::<(u64, WorldGen, PatchKey)>();
        let (done_tx, done) = std::sync::mpsc::channel();
        let job_rx = std::sync::Arc::new(std::sync::Mutex::new(job_rx));
        let _workers = (0..workers)
            .map(|_| {
                let jobs = std::sync::Arc::clone(&job_rx);
                let done = done_tx.clone();
                std::thread::Builder::new()
                    .name("cubara-far".into())
                    .spawn(move || loop {
                        let job = jobs.lock().expect("far job lock").recv();
                        let Ok((generation, gen, key)) = job else {
                            break;
                        };
                        if done.send((generation, far::generate(&gen, key))).is_err() {
                            break;
                        }
                    })
                    .expect("spawn far-terrain worker")
            })
            .collect();
        Self {
            jobs,
            done,
            _workers,
        }
    }
}

/// What one frame changes: patches to generate, slots to free, patches to
/// upload. Empty when nothing changed.
#[derive(Debug, Default, PartialEq)]
struct Step {
    request: Vec<PatchKey>,
    leave: Vec<PatchKey>,
    arrive: Vec<PatchKey>,
}

/// The far terrain's streaming decisions, with no GPU and no threads
/// (`ARCHITECTURE.md` rule 6: the timing of streaming can only be pinned by a
/// test if the decisions can run in one).
///
/// The plan does not hold what is drawn: [`FarStreaming`]'s `slots` is the one
/// source of truth for that, and is handed to [`step`](Self::step) each time.
struct FarPlan {
    quality: FarQuality,
    /// [`FAR_VIEW_RADIUS`] in the game; tests use less.
    radius: f64,
    /// Every patch generated and still of use: the ones drawn, and the ones
    /// whose error decides whether their area is split.
    heights: HashMap<PatchKey, PatchHeights>,
    in_flight: HashSet<PatchKey>,
    /// Where the eye was when the selection was last worked out.
    selected_at: Option<[f64; 3]>,
    /// The hole the last selection left to the voxels.
    hole: Option<Hole>,
    /// The screen height the last selection was made for.
    screen_height: u32,
    /// New patches arrived since the last selection.
    arrived: bool,
    /// How many times [`far::select`] has run, for tests of what a frame costs.
    #[cfg(test)]
    selects: u64,
}

impl FarPlan {
    fn new(quality: FarQuality, radius: f64) -> Self {
        Self {
            quality,
            radius,
            heights: HashMap::new(),
            in_flight: HashSet::new(),
            selected_at: None,
            hole: None,
            screen_height: 0,
            arrived: false,
            #[cfg(test)]
            selects: 0,
        }
    }

    /// Forget every patch, generated or generating, and start the selection
    /// again (a new world, a new quality).
    fn clear(&mut self) {
        self.heights.clear();
        self.in_flight.clear();
        self.selected_at = None;
        self.arrived = false;
    }

    /// A patch finished generating.
    fn take(&mut self, patch: PatchHeights) {
        self.in_flight.remove(&patch.key);
        self.heights.insert(patch.key, patch);
        self.arrived = true;
    }

    /// Nothing generating, and the selection up to date with what has
    /// arrived.
    fn settled(&self) -> bool {
        self.in_flight.is_empty() && !self.arrived
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

    /// Decide what this frame changes for an eye at `eye`: work the selection
    /// out again if the eye has moved or new patches arrived, and say what to
    /// ask for, free and upload. `drawn` is what the renderer holds now; it is
    /// called only if the selection is worked out.
    fn step(
        &mut self,
        eye: [f64; 3],
        screen_height: u32,
        hole: Hole,
        drawn: impl FnOnce() -> HashSet<PatchKey>,
    ) -> Step {
        let moved = self.selected_at.is_none_or(|at| {
            let d = (0..3).map(|a| (at[a] - eye[a]).powi(2)).sum::<f64>().sqrt();
            d > RESELECT_BLOCKS
        });
        // The hole is the voxels' to say (`NodeStreaming::far_hole`): where it
        // shrinks, patches it used to swallow are wanted at once.
        let hole_moved = self.hole != Some(hole);
        self.hole = Some(hole);
        let screen_moved = self.screen_height != screen_height;
        self.screen_height = screen_height;
        if !(moved || hole_moved || screen_moved || self.arrived) {
            return Step::default();
        }
        self.selected_at = Some(eye);
        self.arrived = false;
        let drawn = drawn();

        let Some(view) = far_view(eye, screen_height, Some(hole), self.quality) else {
            // Off: nothing beyond the voxels.
            let mut leave: Vec<PatchKey> = drawn.into_iter().collect();
            leave.sort();
            return Step {
                leave,
                ..Step::default()
            };
        };
        let view = FarView {
            radius: self.radius,
            ..view
        };
        #[cfg(test)]
        {
            self.selects += 1;
        }
        let wanted = far::select(&view, |k| self.heights.get(&k).map(|p| p.error));

        // Ask for what is missing, nearest first: the ground under the eye
        // matters more than the horizon.
        let mut request: Vec<PatchKey> = wanted
            .iter()
            .copied()
            .filter(|k| !self.heights.contains_key(k) && !self.in_flight.contains(k))
            .collect();
        request.sort_by_key(|k| k.level);
        self.in_flight.extend(request.iter().copied());

        // Swap the drawn set, never leaving ground uncovered.
        let draw = draw_set(&wanted, &|k| self.heights.contains_key(&k), &drawn);
        let (leave, arrive) = swap(&drawn, &draw);
        let keep = to_keep(&wanted, draw);
        self.heights.retain(|k, _| keep.contains(k));
        Step {
            request,
            leave,
            arrive,
        }
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
            materials: far_materials(),
            warned_full: false,
        }
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
        for (_, slot) in self.slots.drain() {
            renderer.far_remove(slot);
        }
        self.plan.clear();
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
    /// however little the eye has moved: what a finished frame at a new eye
    /// needs, which [`RESELECT_BLOCKS`] would otherwise leave to the last
    /// eye's.
    pub fn reselect(&mut self) {
        self.plan.reselect();
    }

    /// Bring the far terrain in line with the camera at `eye`: take in what
    /// has been generated, work the selection out again if the eye has moved
    /// or new patches arrived, and ask for what is missing.
    pub fn update(&mut self, renderer: &mut Renderer, eye: [f32; 3], hole: Hole) {
        while let Ok((generation, patch)) = self.pool.done.try_recv() {
            if generation != self.generation {
                continue;
            }
            self.plan.take(patch);
        }
        renderer.set_far_params(to_far_params(hole, self.materials));
        let eye64 = eye.map(|v| v as f64);
        let slots = &self.slots;
        let step = self.plan.step(eye64, renderer.size().1, hole, || {
            slots.keys().copied().collect()
        });

        for key in step.request {
            let _ = self.pool.jobs.send((self.generation, self.gen, key));
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
            match renderer.far_insert(to_far_patch(patch)) {
                Some(slot) => {
                    self.slots.insert(key, slot);
                }
                None if !self.warned_full => {
                    log::warn!(
                        "far terrain: every slot is taken -- some distant ground is not drawn"
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

    /// The same numbers the far terrain module measured (`far.rs`, §3.2).
    #[test]
    fn the_view_uses_the_measured_settings() {
        let pixel = FOV_Y / 1080.0;
        for q in FarQuality::BEST_FIRST {
            let Some(v) = far_view([0.0; 3], 1080, None, q) else {
                assert_eq!(q, FarQuality::Off, "only Off draws nothing");
                continue;
            };
            let d = q.detail().unwrap();
            assert!((v.max_error / pixel - d.error_px).abs() < 1e-9);
            assert!((v.min_quad / pixel - d.floor_px).abs() < 1e-9);
            let quad_angle = 1.0 / (v.split_ratio * far::PATCH_QUADS as f64);
            assert!((quad_angle / pixel - d.quad_px).abs() < 1e-6);
            assert_eq!(v.radius, FAR_VIEW_RADIUS);
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
    #[test]
    fn keeping_the_ancestors_keeps_the_selection_still() {
        let gen = WorldGen::new(0x005E_ED00_00C0_FFEE);
        let view_at = |eye: [f64; 3]| FarView {
            radius: 20_000.0,
            ..far_view(
                eye,
                720,
                Some(voxel_hole(eye.map(|v| v as f32))),
                FarQuality::High,
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

        let keep = to_keep(&wanted, wanted.iter().copied().collect());
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
        assert_eq!(cubara_render::FAR_VERTS, far::PATCH_VERTS);
    }

    // --- The plan without a GPU (#398, rule 6) --------------------------

    use std::cell::RefCell;
    use std::collections::VecDeque;

    /// Smaller than the game's radius so a flight of 600 frames runs in
    /// seconds; the cost report uses the game's.
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

    /// What the worker pool does, in frames: `workers` workers over one FIFO
    /// in request order, each patch taking `service_frames` frames.
    struct PoolModel {
        workers: usize,
        service_frames: u32,
        queue: VecDeque<PatchKey>,
        busy: Vec<(PatchKey, u32)>,
    }

    impl PoolModel {
        fn new(workers: usize, service_frames: u32) -> Self {
            Self {
                workers,
                service_frames,
                queue: VecDeque::new(),
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
                let Some(key) = self.queue.pop_front() else {
                    break;
                };
                self.busy.push((key, self.service_frames));
            }
        }

        fn request(&mut self, keys: &[PatchKey]) {
            self.queue.extend(keys.iter().copied());
            self.fill();
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
            all.extend(self.queue.drain(..));
            all
        }
    }

    /// A plan, a pool, and what a renderer would hold. Tests compare `shown`,
    /// never anything the plan believes.
    struct Sim {
        plan: FarPlan,
        pool: PoolModel,
        shown: HashSet<PatchKey>,
        /// Generated heights, shared by the pool and the reference (a
        /// per-test cache, not a static: rule 2).
        memo: RefCell<HashMap<PatchKey, PatchHeights>>,
        gen: WorldGen,
        hole: Hole,
        radius: f64,
        /// `fly` asserts that drawn, still-wanted ground stays covered.
        check_coverage: bool,
    }

    /// Is the ground of `key` covered by `shown`: by `key` itself, by a
    /// coarser patch holding it, or by finer patches that together fill it?
    fn covered(shown: &HashSet<PatchKey>, key: PatchKey) -> bool {
        shown.contains(&key)
            || ancestors(key, 20).any(|a| shown.contains(&a))
            || (key.level > 0 && key.children().iter().all(|c| covered(shown, *c)))
    }

    impl Sim {
        fn new(radius: f64, hole: Hole, pool: PoolModel) -> Self {
            let seed = cubara_world::World::new().seed();
            Self {
                plan: FarPlan::new(FarQuality::High, radius),
                pool,
                shown: HashSet::new(),
                memo: RefCell::new(HashMap::new()),
                gen: WorldGen::new(seed),
                hole,
                radius,
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

        /// The converged selection at `eye`: select against what is known,
        /// generate what is missing, repeat (the loop of `far::build`, and of
        /// `keeping_the_ancestors_keeps_the_selection_still`).
        fn exact(&self, eye: [f64; 3]) -> HashSet<PatchKey> {
            let view = FarView {
                radius: self.radius,
                ..far_view(eye, 1080, Some(self.hole), FarQuality::High).unwrap()
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
            let step = self.plan.step(eye, 1080, self.hole, || shown.clone());
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
            self.pool.request(&step.request);
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
                if step == Step::default() && self.plan.settled() && self.pool.queue.is_empty() {
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

    /// Behaviour of today's `RESELECT_BLOCKS`, pinned so that the plan that
    /// replaces it (#398 PR B) has something to differ from. Replaced by
    /// `the_live_selection_is_the_exact_one_on_every_frame` there.
    #[test]
    fn today_the_live_selection_trails_the_exact_one() {
        let (mut sim, start) = sim_at_start(PoolModel::instant());
        sim.settle(start);
        let eyes = low_flight(180);
        let shown = sim.fly(&eyes);
        let behind = eyes
            .iter()
            .zip(&shown)
            .filter(|(eye, shown)| **shown != sim.exact(**eye))
            .count();
        println!(
            "{behind} of {} frames show something other than the exact selection",
            eyes.len()
        );
        assert!(
            behind > 0,
            "the live selection never differed from the exact one in 3 s"
        );
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

    /// A settled plan at a standing eye costs nothing: no step, no select.
    #[test]
    fn standing_still_selects_nothing() {
        let (mut sim, start) = sim_at_start(PoolModel::instant());
        sim.settle(start);
        let selects = sim.plan.selects;
        for _ in 0..100 {
            let done = sim.pool.tick();
            assert_eq!(sim.frame(start, done), Step::default());
        }
        assert_eq!(sim.plan.selects, selects, "a standing eye selected again");
    }

    /// The things that make the plan work again besides the eye: a new hole,
    /// a new screen height, and `reselect`.
    #[test]
    fn a_new_hole_screen_height_or_reselect_works_the_selection_out_again() {
        let (mut sim, start) = sim_at_start(PoolModel::instant());
        sim.settle(start);
        let n = sim.plan.selects;
        let (hole, shown) = (sim.hole, sim.shown.clone());
        sim.plan.reselect();
        sim.plan.step(start, 1080, hole, || shown.clone());
        assert_eq!(sim.plan.selects, n + 1, "reselect did nothing");
        sim.plan.step(start, 720, hole, || shown.clone());
        assert_eq!(sim.plan.selects, n + 2, "a new screen height did nothing");
        let hole = Hole {
            min: [hole.min[0] + 16.0, hole.min[1], hole.min[2]],
            ..hole
        };
        sim.plan.step(start, 720, hole, || shown.clone());
        assert_eq!(sim.plan.selects, n + 3, "a new hole did nothing");
        sim.plan.step(start, 720, hole, || shown.clone());
        assert_eq!(sim.plan.selects, n + 3, "nothing changed but it selected");
    }

    /// `Off` draws nothing beyond the voxels: everything shown leaves, and
    /// nothing is asked for.
    #[test]
    fn off_removes_everything_and_asks_for_nothing() {
        let mut plan = FarPlan::new(FarQuality::Off, TEST_RADIUS);
        let drawn: HashSet<PatchKey> = [
            PatchKey::new(3, 2, 2),
            PatchKey::new(3, 1, 5),
            PatchKey::new(4, 0, 0),
        ]
        .into();
        let hole = voxel_hole([8.0, 120.0, 8.0]);
        let step = plan.step([8.0, 120.0, 8.0], 1080, hole, || drawn.clone());
        let mut sorted: Vec<_> = drawn.into_iter().collect();
        sorted.sort();
        assert_eq!(step.leave, sorted);
        assert!(step.request.is_empty() && step.arrive.is_empty());
    }

    /// A patch asked for is not asked for again while it is generating, and
    /// the ones nearest the eye (finest levels) go first.
    #[test]
    fn requests_are_made_once_and_finest_first() {
        let (mut sim, start) = sim_at_start(PoolModel::new(4, 2));
        let step = sim.frame(start, Vec::new());
        assert!(!step.request.is_empty());
        assert!(
            step.request.windows(2).all(|w| w[0].level <= w[1].level),
            "not finest first"
        );
        sim.plan.reselect();
        let again = sim.frame(start, Vec::new());
        assert!(
            again.request.iter().all(|k| !step.request.contains(k)),
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
    /// modelled as 4 workers (the M3's `cores / 2`) at 2 frames a patch (a
    /// placeholder until #398 step 1's measured service time is applied:
    /// 6-9 ms, i.e. one frame). The memo is warmed by a first, untimed
    /// flight so generation is not in the numbers.
    #[test]
    #[ignore = "a measurement, not a check: prints what the plan's step costs"]
    fn far_plan_cost_report() {
        let eyes = low_flight(600);
        let start = eyes[0];
        let hole = voxel_hole(start.map(|v| v as f32));
        let mut sim = Sim::new(FAR_VIEW_RADIUS, hole, PoolModel::new(4, 2));
        sim.settle(start);
        sim.fly(&eyes);
        // The timed flight, from a fresh plan over the warm memo.
        let memo = std::mem::take(&mut sim.memo);
        let mut sim = Sim::new(FAR_VIEW_RADIUS, hole, PoolModel::new(4, 2));
        sim.memo = memo;
        sim.settle(start);
        let selects_before = sim.plan.selects;
        let mut times = Vec::new();
        for &eye in &eyes {
            let finished = sim.pool.tick();
            for key in finished {
                let patch = sim.heights(key);
                sim.plan.take(patch);
            }
            let shown = &sim.shown;
            let t = std::time::Instant::now();
            let step = sim.plan.step(eye, 1080, hole, || shown.clone());
            times.push(t.elapsed());
            for key in &step.leave {
                sim.shown.remove(key);
            }
            sim.shown.extend(step.arrive.iter().copied());
            sim.pool.request(&step.request);
        }
        let ms = |d: &std::time::Duration| d.as_secs_f64() * 1e3;
        let sum: f64 = times.iter().map(ms).sum();
        let mut sorted: Vec<f64> = times.iter().map(ms).collect();
        sorted.sort_by(f64::total_cmp);
        let p99 = sorted[(sorted.len() * 99 / 100).min(sorted.len() - 1)];
        println!(
            "FAR_PLAN_COST frames {} sum {sum:.2} ms mean {:.4} ms p99 {p99:.3} ms max {:.3} ms; \
             selects {} (exact; no prefetch in this version); shown {}",
            times.len(),
            sum / times.len() as f64,
            sorted.last().unwrap(),
            sim.plan.selects - selects_before,
            sim.shown.len(),
        );
    }

    /// Past `RESELECT_BLOCKS` the selection is worked out again; within it,
    /// with nothing arrived, it is not.
    #[test]
    fn the_eye_moving_past_the_reselect_distance_selects_again() {
        let (mut sim, start) = sim_at_start(PoolModel::instant());
        sim.settle(start);
        let n = sim.plan.selects;
        let (hole, shown) = (sim.hole, sim.shown.clone());
        let near = [start[0] + RESELECT_BLOCKS * 0.5, start[1], start[2]];
        sim.plan.step(near, 1080, hole, || shown.clone());
        assert_eq!(sim.plan.selects, n, "a small move selected");
        // Exactly the distance is not past it (a move of 32 is exact in f64).
        let edge = [start[0] + RESELECT_BLOCKS, start[1], start[2]];
        let from = [start[0], start[1], start[2]];
        sim.plan.selected_at = Some(from);
        sim.plan.step(edge, 1080, hole, || shown.clone());
        assert_eq!(
            sim.plan.selects, n,
            "a move of exactly the distance selected"
        );
        let far = [start[0] + RESELECT_BLOCKS * 2.0, start[1], start[2]];
        sim.plan.step(far, 1080, hole, || shown.clone());
        assert_eq!(sim.plan.selects, n + 1, "a move past the distance did not");
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
        let patch = sim.heights(step.request[0]);
        sim.plan.take(patch);
        assert!(!sim.plan.settled(), "an arrival no selection has seen");
    }

    /// Patches nothing wants any more are dropped: after moving far away and
    /// settling, the plan holds the exact selection's patches and their
    /// ancestors, not everything it ever generated.
    #[test]
    fn what_is_no_longer_wanted_is_dropped() {
        let (mut sim, start) = sim_at_start(PoolModel::instant());
        sim.settle(start);
        let there = [start[0] + 3_000.0, start[1], start[2] + 900.0];
        sim.hole = voxel_hole(there.map(|v| v as f32));
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

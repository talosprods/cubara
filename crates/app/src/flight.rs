//! `--bench flight`: how much of what a player sees while flying is still
//! loading.
//!
//! The owner, 2026-09-27, after flying around: "veel inladen ... zou fijn zijn
//! als dat consistenter goed is". A frame rate says nothing about that, and
//! neither does a golden image of a world loaded up front. So this flies the
//! game's own streaming ([`NodeStreaming`], [`FarStreaming`]) and the game's
//! own renderer, pointed at a texture instead of a window
//! ([`Renderer::offscreen`]), along a straight line at flying speed in real
//! time -- and then flies the same line again, stopping at every sampled
//! point until nothing is left to load. The fraction of pixels that differ
//! between the two at the same eye is what was still loading.
//!
//! Two passes rather than one: settling a second world at each point while
//! the first flies would hand the first's mesh workers free time, and it
//! would look better than a player's.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cubara_render::{CameraPose, Frame, Renderer};
use cubara_world::World;

use crate::far_streaming::{FarQuality, FarStreaming, FAR_VIEW_RADIUS};
use crate::streaming::NodeStreaming;

/// Blocks per second: `cubara_sim::player`'s free-fly speed.
pub const FLY_SPEED: f32 = 24.0;

/// What frames clear to while measuring: magenta, a colour the land never
/// takes (the fog fades it into the sky's own colour), so a pixel showing it
/// is a pixel nothing was drawn on.
const NOTHING: wgpu::Color = wgpu::Color {
    r: 1.0,
    g: 0.0,
    b: 1.0,
    a: 1.0,
};

/// Whether an RGBA pixel shows [`NOTHING`].
fn shows_nothing(p: &[u8]) -> bool {
    p[0] >= 250 && p[1] <= 5 && p[2] >= 250
}

/// The longest run of frames, at the flight's 60 Hz, that a pixel may show
/// nothing between two frames of land and still count as a hole rather than
/// sky it really looks at: half a second. Longer gaps are caught by the kept
/// frames' comparison with the finished frame.
const HOLE_MAX_FRAMES: u32 = 30;

/// Finds holes in a sequence of frames: pixels **inside** land -- land on
/// all four sides -- that show nothing for at most [`HOLE_MAX_FRAMES`], then
/// land again. The owner's "laag detail niveau, dan is het weg, heel kort
/// daarna laadt het op een hoger niveau".
///
/// Inside, because a silhouette is not a hole: a stepped ridge against the sky
/// slides across pixel boundaries as the eye moves, and a pixel on its edge
/// goes land, sky, land within a few frames without anything missing (seen in
/// the first measurement: every such pixel was on a skyline). Sky that is
/// really there is not counted either -- it does not go back to land within
/// half a second on a straight flight -- nor is a run still open when the
/// flight ends. Pure, so the rule is tested without a GPU.
pub struct HoleTracker {
    width: usize,
    /// Per pixel: whether it showed land in the last frame.
    land: Vec<bool>,
    /// Per pixel: the frame its current run of nothing began after land, or
    /// `u32::MAX`.
    run_start: Vec<u32>,
    frame: u32,
    /// Per frame: the pixels that were holes in it.
    pub per_frame: Vec<u32>,
}

impl HoleTracker {
    /// For frames `width` pixels wide and `pixels` in all.
    pub fn new(width: usize, pixels: usize) -> Self {
        Self {
            width: width.max(1),
            land: vec![false; pixels],
            run_start: vec![u32::MAX; pixels],
            frame: 0,
            per_frame: Vec::new(),
        }
    }

    /// Take the next frame: per pixel, whether it shows nothing.
    pub fn push(&mut self, nothing: impl IntoIterator<Item = bool>) {
        self.per_frame.push(0);
        let before = self.land.clone();
        let (w, n) = (self.width, before.len());
        // Land last frame on all four sides: inside land, not on its edge.
        // The screen's border has a side off screen, so it never counts.
        let inside = |i: usize| {
            let x = i % w;
            x > 0
                && x + 1 < w
                && i >= w
                && i + w < n
                && before[i - 1]
                && before[i + 1]
                && before[i - w]
                && before[i + w]
        };
        for (i, empty) in nothing.into_iter().enumerate() {
            if empty {
                if before[i] && inside(i) {
                    self.run_start[i] = self.frame;
                }
                self.land[i] = false;
            } else {
                let start = self.run_start[i];
                if start != u32::MAX && self.frame - start <= HOLE_MAX_FRAMES {
                    for f in start..self.frame {
                        self.per_frame[f as usize] += 1;
                    }
                }
                self.run_start[i] = u32::MAX;
                self.land[i] = true;
            }
        }
        self.frame += 1;
    }
}

/// A pixel counts as still loading when a channel is off by more than this
/// from the finished frame. Far above what a far patch one level coarser or
/// a node one level off moves a pixel by on open ground, far below ground
/// against sky.
const TOLERANCE: u8 = 40;

/// What to fly.
#[derive(Clone, Debug)]
pub struct Flight {
    pub start: [f32; 3],
    /// The direction flown, which is also the one looked in.
    pub look: [f32; 3],
    pub speed: f32,
    pub seconds: f32,
    /// Frames per second the flight is paced at: a monitor's refresh rate,
    /// since the renderer uploads a fixed number of meshes per frame.
    pub fps: u32,
    /// How often a frame is kept to compare, in seconds of flight.
    pub every: f32,
    pub size: (u32, u32),
    pub quality: FarQuality,
}

/// Where the eye is `t` seconds into `flight`.
pub fn eye_at(flight: &Flight, t: f32) -> [f32; 3] {
    let dir = glam::Vec3::from(flight.look).normalize_or_zero();
    (glam::Vec3::from(flight.start) + dir * flight.speed * t).to_array()
}

/// How far from spawn [`toward_peak`] looks for a mountain, in blocks.
const PEAK_SEARCH: i32 = 4096;

/// How far before the peak the flight toward it starts, in blocks: inside the
/// far terrain, so the flight crosses the voxel rings' edges on the way in.
const PEAK_APPROACH: f32 = 900.0;

/// The owner's flight toward a mountain: the tallest surface within
/// [`PEAK_SEARCH`] of spawn, approached level at its summit's height from
/// [`PEAK_APPROACH`] away on the side facing spawn. Returns the start and the
/// direction. Deterministic for a seed: a 32-block grid, ties to the first.
pub fn toward_peak(gen: &cubara_world::WorldGen) -> ([f32; 3], [f32; 3]) {
    let mut best = (i32::MIN, 0, 0);
    for x in (-PEAK_SEARCH..=PEAK_SEARCH).step_by(32) {
        for z in (-PEAK_SEARCH..=PEAK_SEARCH).step_by(32) {
            let h = gen.surface_height(x, z);
            if h > best.0 {
                best = (h, x, z);
            }
        }
    }
    let (h, px, pz) = best;
    let peak = glam::vec2(px as f32, pz as f32);
    let away = (-peak).normalize_or(glam::Vec2::X);
    let start = peak + away * PEAK_APPROACH;
    let y = h as f32 + 3.0;
    let look = glam::vec3(peak.x - start.x, 0.0, peak.y - start.y).normalize();
    log::info!("flight: toward the peak at ({px}, {h}, {pz}), from {start:.0}");
    ([start.x, y, start.y], look.to_array())
}

/// The fraction of `live`'s pixels more than [`TOLERANCE`] off `settled`'s.
pub fn loading_fraction(live: &Frame, settled: &Frame) -> f64 {
    cubara_render::headless::compare(&live.pixels, &settled.pixels, TOLERANCE).differing_fraction
}

/// One kept frame: when, where, and how much of it was still loading.
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub t: f32,
    pub eye: [f32; 3],
    pub loading: f64,
    /// The share of its pixels that showed nothing where the finished frame
    /// shows land.
    pub hole: f64,
}

/// The share of `live`'s pixels that show nothing where `settled` shows land
/// on the pixel and on all four sides of it: inside land, so a skyline drawn a
/// pixel differently (the live pass's far terrain was selected a little behind
/// the eye) is not a hole. See [`HoleTracker`].
pub fn hole_fraction(live: &Frame, settled: &Frame) -> f64 {
    let w = settled.width as usize;
    let land: Vec<bool> = settled
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| !shows_nothing(&p[..]))
        .collect();
    let n = land.len();
    let inside = |i: usize| {
        !i.is_multiple_of(w)
            && i % w + 1 < w
            && i >= w
            && i + w < n
            && land[i]
            && land[i - 1]
            && land[i + 1]
            && land[i - w]
            && land[i + w]
    };
    let holes = live
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .enumerate()
        .filter(|(i, l)| shows_nothing(&l[..]) && inside(*i))
        .count();
    holes as f64 / n.max(1) as f64
}

/// The swap-lag distribution for one flight (issue #378): how long a node
/// that should have been swapped for a different level of detail kept being
/// drawn. `NodeStreaming` only counts frames of its own `update` calls (rule
/// 2: no clock inside it); this is where those become milliseconds, with
/// the flight's own pacing, and blocks travelled, with its speed -- the two
/// units decision 3 asks the report to carry.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SwapLagStats {
    pub count: usize,
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
    pub p50_blocks: f64,
    pub p90_blocks: f64,
    pub p99_blocks: f64,
    pub max_blocks: f64,
}

impl SwapLagStats {
    /// `lags` is [`crate::streaming::NodeStreaming::swap_lags`]'s frame
    /// counts; `frames_per_second` and `speed` (blocks/s) convert a frame
    /// count to time and distance the way the flight itself paced and moved.
    /// The rate is the one the flight *achieved*, frames drawn over seconds
    /// flown -- not the one it asked for: a frame that runs late is not made
    /// up (`fly`'s pacing), so under load a frame is longer than 1/`fps` and
    /// the nominal rate would understate the lag.
    pub fn of(lags: &[u64], frames_per_second: f64, speed: f32) -> Self {
        if lags.is_empty() {
            return Self::default();
        }
        let mut sorted = lags.to_vec();
        sorted.sort_unstable();
        let at = |per_mille: usize| sorted[(sorted.len() - 1) * per_mille / 1000] as f64;
        let rate = frames_per_second.max(1e-3);
        let ms_per_frame = 1000.0 / rate;
        let blocks_per_frame = speed as f64 / rate;
        let (p50, p90, p99, max) = (at(500), at(900), at(990), *sorted.last().unwrap() as f64);
        Self {
            count: sorted.len(),
            p50_ms: p50 * ms_per_frame,
            p90_ms: p90 * ms_per_frame,
            p99_ms: p99 * ms_per_frame,
            max_ms: max * ms_per_frame,
            p50_blocks: p50 * blocks_per_frame,
            p90_blocks: p90 * blocks_per_frame,
            p99_blocks: p99 * blocks_per_frame,
            max_blocks: max * blocks_per_frame,
        }
    }
}

/// What a flight found.
#[derive(Debug)]
pub struct FlightReport {
    pub samples: Vec<Sample>,
    /// The part of each frame of the real-time pass the game spent streaming
    /// and drawing, in seconds. Not the whole frame: the rest is this
    /// measurement's sleep standing in for a display, and a sleep wakes when
    /// it likes -- about 4 ms late at p99 on an idle M3.
    pub work_times: Vec<f32>,
    /// Per frame of the real-time pass, the pixels that were holes
    /// ([`HoleTracker`]).
    pub holes: Vec<u32>,
    /// Pixels in a frame.
    pub pixels: usize,
    /// How long a node kept being drawn after it should have swapped for a
    /// different level of detail (issue #378) -- the live pass only; the
    /// settled pass jumps around and is not what a player experiences.
    pub swap_lags: SwapLagStats,
}

impl FlightReport {
    /// The share of kept frames with more than `fraction` of the screen still
    /// loading.
    pub fn frames_over(&self, fraction: f64) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let over = self.samples.iter().filter(|s| s.loading > fraction).count();
        over as f64 / self.samples.len() as f64
    }

    /// The mean fraction of the screen still loading.
    pub fn mean(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.iter().map(|s| s.loading).sum::<f64>() / self.samples.len() as f64
    }

    /// The worst kept frame.
    pub fn worst(&self) -> Option<Sample> {
        self.samples
            .iter()
            .copied()
            .max_by(|a, b| a.loading.total_cmp(&b.loading))
    }

    pub fn line(&self) -> String {
        let worst = self.worst().unwrap_or(Sample {
            t: 0.0,
            eye: [0.0; 3],
            loading: 0.0,
            hole: 0.0,
        });
        let tail = WorkTail::of(&self.work_times);
        let (any, visible, worst_px) = self.hole_frames();
        let kept_holes = self.samples.iter().filter(|s| s.hole > 0.0).count();
        let kept_worst = self.samples.iter().map(|s| s.hole).fold(0.0, f64::max);
        let s = &self.swap_lags;
        format!(
            "FLIGHT: loading mean {:.2}% worst {:.2}% (at {:.2} s, {:.0?}), \
             frames over 1%: {:.0}%, work p99 {:.1} ms p99.9 {:.1} ms max {:.1} ms, \
             over 16.7 ms {} over 33.3 ms {} ({} kept frames) | \
             HOLES: {any} of {} frames show a hole, {visible} over 0.1% of the screen, \
             worst {:.3}%; kept frames with nothing where the finished frame has land: \
             {kept_holes}, worst {:.3}% | \
             SWAP LAG: n {}, p50/p90/p99/max {:.0}/{:.0}/{:.0}/{:.0} ms \
             ({:.2}/{:.2}/{:.2}/{:.2} blocks)",
            self.mean() * 100.0,
            worst.loading * 100.0,
            worst.t,
            worst.eye,
            self.frames_over(0.01) * 100.0,
            tail.p99 * 1000.0,
            tail.p999 * 1000.0,
            tail.max * 1000.0,
            tail.over_one,
            tail.over_two,
            self.samples.len(),
            self.holes.len(),
            worst_px as f64 * 100.0 / self.pixels.max(1) as f64,
            kept_worst * 100.0,
            s.count,
            s.p50_ms,
            s.p90_ms,
            s.p99_ms,
            s.max_ms,
            s.p50_blocks,
            s.p90_blocks,
            s.p99_blocks,
            s.max_blocks,
        )
    }

    /// Frames of the real-time pass with any hole pixel, with holes over 0.1%
    /// of the screen, and the most hole pixels in one frame.
    pub fn hole_frames(&self) -> (usize, usize, u32) {
        let visible = (self.pixels / 1000) as u32;
        (
            self.holes.iter().filter(|&&n| n > 0).count(),
            self.holes.iter().filter(|&&n| n > visible).count(),
            self.holes.iter().copied().max().unwrap_or(0),
        )
    }
}

/// The slow end of the work per frame, in seconds -- what a player feels as a
/// stutter (block 3.0a; the exit gate's item 0 asks for p99.9 under a 60 Hz
/// interval and no frame over two).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WorkTail {
    pub p99: f32,
    pub p999: f32,
    pub max: f32,
    /// Frames whose work alone took longer than one 60 Hz interval, and two.
    pub over_one: usize,
    pub over_two: usize,
}

/// One frame at 60 Hz, in seconds.
const INTERVAL_60HZ: f32 = 1.0 / 60.0;

impl WorkTail {
    pub fn of(times: &[f32]) -> Self {
        let mut sorted = times.to_vec();
        sorted.sort_by(f32::total_cmp);
        let at = |per_mille: usize| {
            sorted
                .get(sorted.len().saturating_sub(1) * per_mille / 1000)
                .copied()
                .unwrap_or(0.0)
        };
        Self {
            p99: at(990),
            p999: at(999),
            max: sorted.last().copied().unwrap_or(0.0),
            over_one: times.iter().filter(|&&t| t > INTERVAL_60HZ).count(),
            over_two: times.iter().filter(|&&t| t > 2.0 * INTERVAL_60HZ).count(),
        }
    }
}

/// What `--bench walk` found: every frame's whole time, start to GPU done.
pub struct WalkReport {
    pub frame_times: Vec<f32>,
    pub ticks: u64,
    pub distance: f32,
}

impl WalkReport {
    pub fn line(&self) -> String {
        let t = WorkTail::of(&self.frame_times);
        format!(
            "WALK: {} frames, {} ticks, walked {:.0} blocks | frame p99 {:.1} ms p99.9 {:.1} ms \
             max {:.1} ms, over 16.7 ms {} over 33.3 ms {}",
            self.frame_times.len(),
            self.ticks,
            self.distance,
            t.p99 * 1000.0,
            t.p999 * 1000.0,
            t.max * 1000.0,
            t.over_one,
            t.over_two,
        )
    }
}

/// `--bench walk`: the game's own loop, offscreen -- [`Game::advance`] (the
/// simulation and the server, in process), the voxel and far streaming, the
/// renderer -- while a scripted player walks forward for `seconds`, jumping
/// every second to take steps. Each frame is timed from its start until the
/// GPU has finished it, paced at `fps`: the whole frame the exit gate's item 0
/// asks about (block 3.0a), where `--bench flight` times only streaming and
/// drawing. Starts once the spawn has loaded. `None` without a GPU.
///
/// With `fly`, F4 first: the same loop with the player spectating, flying
/// forward through everything at free-fly speed -- the streaming under load,
/// with the simulation still ticking.
pub fn walk(seconds: f32, size: (u32, u32), fps: u32, fly: bool) -> Option<WalkReport> {
    use crate::game::{load_item_registry, load_recipe_book, Game};
    use winit::keyboard::KeyCode;

    let mut game = Game::new();
    let (mut renderer, assets) = Renderer::offscreen(size.0, size.1, game.camera_pose())?;
    let layers = assets.layers;
    let registry = Arc::new(assets.registry);
    let items = load_item_registry();
    let recipes = load_recipe_book(&items);
    game.set_assets(Arc::clone(&registry), items, recipes);
    let mut nodes = NodeStreaming::new(
        registry,
        &crate::game::load_structure_registry(),
        &crate::game::load_ore_registry(),
        move |name: &str| layers.layer_of(name),
    );
    let mut far = FarStreaming::new(game.world().seed(), FarQuality::High);
    let budget = Duration::from_secs_f64(1.0 / fps.max(1) as f64);
    let mut frame = |game: &mut Game, dt: f32, renderer: &mut Renderer| {
        for cc in game.advance(dt) {
            nodes.invalidate(game.world(), cc);
        }
        let camera = game.camera_pose();
        let eye = camera.eye.as_vec3().to_array();
        nodes.update(renderer, game.world(), eye);
        let hole = nodes.far_hole(renderer, eye);
        far.update(renderer, eye, hole);
        renderer.render(
            camera,
            game.selected_block(),
            game.cracking(),
            &game.other_players(),
            cubara_render::Hud {
                hotbar: None,
                panel: None,
                health: Some(game.health_view()),
                crosshair: true,
                menu: None,
            },
            FAR_VIEW_RADIUS as f32,
        );
        let _ = renderer.device().poll(wgpu::PollType::wait_indefinitely());
        nodes.settled() && far.settled() && renderer.uploads_pending() == 0
    };
    // Load the spawn first, standing still: what the walk measures is
    // walking through a loaded world, not loading one.
    let waiting = Instant::now();
    while !frame(&mut game, budget.as_secs_f32(), &mut renderer) {
        if waiting.elapsed() > Duration::from_secs(120) {
            log::warn!("walk: the spawn did not settle in two minutes");
            break;
        }
    }
    let start_pos = game.camera_pose().eye;
    let ticks_before = game.sim_tick();
    if fly {
        game.key_input(KeyCode::F4, true);
        frame(&mut game, budget.as_secs_f32(), &mut renderer);
        game.key_input(KeyCode::F4, false);
    }
    game.key_input(KeyCode::KeyW, true);
    let mut frame_times = Vec::new();
    let start = Instant::now();
    let mut last = start;
    let mut next_jump = 1.0;
    let mut next_frame = start;
    while start.elapsed().as_secs_f32() < seconds {
        let t = start.elapsed().as_secs_f32();
        let jump = !fly && t >= next_jump;
        if jump {
            game.key_input(KeyCode::Space, true);
            next_jump += 1.0;
        }
        let began = Instant::now();
        let dt = (began - last).as_secs_f32();
        last = began;
        frame(&mut game, dt, &mut renderer);
        frame_times.push(began.elapsed().as_secs_f32());
        if jump {
            game.key_input(KeyCode::Space, false);
        }
        next_frame += budget;
        let now = Instant::now();
        match next_frame.checked_duration_since(now) {
            Some(rest) => std::thread::sleep(rest),
            None => next_frame = now,
        }
    }
    Some(WalkReport {
        frame_times,
        ticks: game.sim_tick() - ticks_before,
        distance: (game.camera_pose().eye - start_pos).length() as f32,
    })
}

/// Everything a flight streams: the world, the game's two streamers, and the
/// game's renderer.
struct Streamed {
    world: Arc<World>,
    nodes: NodeStreaming,
    far: FarStreaming,
    renderer: Renderer,
}

impl Streamed {
    fn new(flight: &Flight) -> Option<Self> {
        let camera = pose(flight, flight.start);
        let (mut renderer, assets) = Renderer::offscreen(flight.size.0, flight.size.1, camera)?;
        renderer.set_clear_color(NOTHING);
        let layers = assets.layers;
        let registry = Arc::new(assets.registry);
        let nodes = NodeStreaming::new(
            registry,
            &crate::game::load_structure_registry(),
            &crate::game::load_ore_registry(),
            move |name: &str| layers.layer_of(name),
        );
        let world = Arc::new(World::new());
        let far = FarStreaming::new(world.seed(), flight.quality);
        Some(Self {
            world,
            nodes,
            far,
            renderer,
        })
    }

    /// Stream for `eye` and draw one frame, the way the game's frame does.
    fn frame(&mut self, flight: &Flight, eye: [f32; 3]) {
        self.nodes.update(&mut self.renderer, &self.world, eye);
        let hole = self.nodes.far_hole(&self.renderer, eye);
        self.far.update(&mut self.renderer, eye, hole);
        self.renderer.render(
            pose(flight, eye),
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
    }

    /// Stream and draw at `eye` until nothing is left to load, then once more
    /// so what arrived last is drawn.
    fn settle(&mut self, flight: &Flight, eye: [f32; 3]) {
        let waiting = Instant::now();
        loop {
            self.frame(flight, eye);
            if self.settled() {
                break;
            }
            if waiting.elapsed() > Duration::from_secs(60) {
                log::warn!("flight: nothing settled at {eye:?} in a minute");
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        self.frame(flight, eye);
    }

    fn settled(&self) -> bool {
        self.nodes.settled() && self.far.settled() && self.renderer.uploads_pending() == 0
    }
}

fn pose(flight: &Flight, eye: [f32; 3]) -> CameraPose {
    CameraPose {
        eye: glam::Vec3::from(eye).as_dvec3(),
        look_dir: glam::Vec3::from(flight.look).normalize_or_zero(),
    }
}

/// Fly `flight` and compare what was on screen with what should have been.
/// With `frames`, the worst kept frame and its finished version are written
/// there as `worst-live.png` and `worst-settled.png`. `None` without a GPU.
pub fn fly(flight: &Flight, frames: Option<&Path>) -> Option<FlightReport> {
    // The real-time pass: what a player sees -- once the start is loaded, as
    // it is for a player who has been standing there. Loading a world from
    // nothing is a different question, with its own answer.
    let mut live = Streamed::new(flight)?;
    live.settle(flight, flight.start);
    let budget = Duration::from_secs_f64(1.0 / flight.fps.max(1) as f64);
    let mut kept: Vec<(f32, [f32; 3], Frame)> = Vec::new();
    let mut work_times = Vec::new();
    let mut holes: Option<HoleTracker> = None;
    let start = Instant::now();
    let mut next_frame = start;
    let mut next_keep = 0.0;
    loop {
        let t = start.elapsed().as_secs_f32();
        if t > flight.seconds {
            break;
        }
        let eye = eye_at(flight, t);
        let working = Instant::now();
        live.frame(flight, eye);
        work_times.push(working.elapsed().as_secs_f32());
        // Every frame, for the holes: one lasts a few frames, and the kept
        // ones are a quarter of a second apart. Read after the work is timed,
        // so the readback is not counted as the game's.
        let shown = live.renderer.read_frame()?;
        let tracker = holes
            .get_or_insert_with(|| HoleTracker::new(shown.width as usize, shown.pixels.len() / 4));
        tracker.push(
            shown
                .pixels
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| shows_nothing(p)),
        );
        if t >= next_keep {
            let (resident, held, meshing, visible) = live.nodes.counts();
            log::debug!(
                "flight: t {t:5.2} s: {resident} resident, {held} held, {meshing} meshing, \
                 {visible} visible, {} uploads queued",
                live.renderer.uploads_pending()
            );
            kept.push((t, eye, shown));
            next_keep += flight.every;
        }
        // Paced like a display showing `fps` frames a second: a frame that
        // ran late starts the next at once rather than catching up.
        next_frame += budget;
        let now = Instant::now();
        match next_frame.checked_duration_since(now) {
            Some(rest) => std::thread::sleep(rest),
            None => next_frame = now,
        }
    }
    // The live pass only (#378): the settled pass below jumps the eye around
    // to reconstruct each kept frame, which is not a swap a player ever
    // waits on.
    let achieved = work_times.len() as f64 / flight.seconds.max(1e-3) as f64;
    let swap_lags = SwapLagStats::of(live.nodes.swap_lags(), achieved, flight.speed);
    drop(live);

    // The settled pass: the same eyes, each loaded completely.
    let mut settled = Streamed::new(flight)?;
    let mut samples = Vec::with_capacity(kept.len());
    let mut worst: Option<(f64, (Frame, Frame))> = None;
    for (t, eye, frame) in kept {
        settled.far.reselect();
        settled.settle(flight, eye);
        let reference = settled.renderer.read_frame()?;
        let loading = loading_fraction(&frame, &reference);
        let hole = hole_fraction(&frame, &reference);
        log::info!(
            "flight: t {t:5.2} s at {eye:.0?}: {:.2}% loading, {:.3}% nothing where land is",
            loading * 100.0,
            hole * 100.0
        );
        if let Some(dir) = frames.filter(|_| loading > 0.01) {
            for (kind, f) in [("live", &frame), ("settled", &reference)] {
                save(&dir.join(format!("t{:05.2}-{kind}.png", t)), f);
            }
        }
        keep_worst(&mut worst, loading, (frame, reference));
        samples.push(Sample {
            t,
            eye,
            loading,
            hole,
        });
    }
    // The measurement's own zero. The first kept frame is at the start, which
    // the live pass loaded completely before it set off, so it must match its
    // finished frame exactly; a difference is the two passes disagreeing about
    // something other than loading -- the camera, the far terrain's selection,
    // the lighting -- and every number here would carry it. (Checked by
    // flying at speed 0: every kept frame 0.00%.)
    if let Some(first) = samples.first().filter(|s| s.loading > 0.0) {
        log::warn!(
            "flight: the start, loaded completely, differs from itself by {:.2}% -- \
             every number below includes that",
            first.loading * 100.0
        );
    }
    if let (Some(dir), Some((_, (live, settled)))) = (frames, worst) {
        for (name, frame) in [("worst-live.png", live), ("worst-settled.png", settled)] {
            save(&dir.join(name), &frame);
        }
    }
    let pixels = (flight.size.0 * flight.size.1) as usize;
    Some(FlightReport {
        samples,
        work_times,
        holes: holes.map_or_else(Vec::new, |h| h.per_frame),
        pixels,
        swap_lags,
    })
}

/// Keep `candidate` as the worst so far if it is: more of it was loading.
fn keep_worst<T>(worst: &mut Option<(f64, T)>, loading: f64, candidate: T) {
    if worst.as_ref().is_none_or(|(l, _)| loading > *l) {
        *worst = Some((loading, candidate));
    }
}

fn save(path: &Path, frame: &Frame) {
    if let Err(e) = image::save_buffer(
        path,
        &frame.pixels,
        frame.width,
        frame.height,
        image::ExtendedColorType::Rgba8,
    ) {
        log::error!("flight: could not write {}: {e}", path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flight() -> Flight {
        Flight {
            start: [0.0, 100.0, 0.0],
            look: [3.0, 0.0, 4.0],
            speed: 10.0,
            seconds: 5.0,
            fps: 60,
            every: 0.5,
            size: (64, 32),
            quality: FarQuality::High,
        }
    }

    #[test]
    fn the_eye_moves_along_the_look_at_the_speed() {
        let f = flight();
        assert_eq!(eye_at(&f, 0.0), [0.0, 100.0, 0.0]);
        let e = eye_at(&f, 2.0);
        assert!(
            (e[0] - 12.0).abs() < 1e-4 && (e[2] - 16.0).abs() < 1e-4,
            "{e:?}"
        );
        assert_eq!(e[1], 100.0);
    }

    fn frame(pixels: Vec<u8>) -> Frame {
        Frame {
            width: (pixels.len() / 4) as u32,
            height: 1,
            pixels,
        }
    }

    #[test]
    fn a_pixel_counts_as_loading_only_when_far_off() {
        let settled = frame(vec![100, 100, 100, 255, 100, 100, 100, 255]);
        let near = frame(vec![120, 100, 100, 255, 100, 100, 100, 255]);
        let hole = frame(vec![200, 100, 100, 255, 100, 100, 100, 255]);
        assert_eq!(loading_fraction(&near, &settled), 0.0);
        assert_eq!(loading_fraction(&hole, &settled), 0.5);
    }

    #[test]
    fn the_report_counts_the_frames_over_a_share() {
        let sample = |loading| Sample {
            t: 0.0,
            eye: [0.0; 3],
            loading,
            hole: 0.0,
        };
        let r = FlightReport {
            samples: vec![sample(0.0), sample(0.02), sample(0.005), sample(0.1)],
            work_times: vec![0.002; 10],
            holes: Vec::new(),
            pixels: 4,
            swap_lags: SwapLagStats::default(),
        };
        assert_eq!(r.frames_over(0.01), 0.5);
        assert!((r.mean() - 0.03125).abs() < 1e-12);
        assert_eq!(r.worst().unwrap().loading, 0.1);
    }

    /// Land all round, nothing for a few frames, land again: a hole, counted
    /// in every frame it showed. A pixel on a skyline, one on the screen's
    /// border, and a gap longer than half a second are not.
    #[test]
    fn a_hole_is_land_all_round_then_briefly_nothing_then_land() {
        // 5 x 3 pixels. Index 6 is inside land: a 3-frame hole (frames 1..=3).
        // Index 7 sits under 2, which is sky throughout: a skyline pixel doing
        // the same is not a hole. Index 3, on the top border, neither.
        // Index 8 shows nothing for 32 frames: too long to be a hole.
        let (n, l) = (true, false);
        let base = |f: usize| {
            let mut px = [l; 15];
            px[2] = n;
            if (1..=3).contains(&f) {
                px[6] = n;
                px[7] = n;
                px[3] = n;
            }
            if (1..=32).contains(&f) {
                px[8] = n;
            }
            px
        };
        let frames: Vec<[bool; 15]> = (0..=33).map(base).collect();
        let mut t = HoleTracker::new(5, 15);
        for f in &frames {
            t.push(f.iter().copied());
        }
        let mut want = vec![0u32; frames.len()];
        for w in &mut want[1..=3] {
            *w = 1;
        }
        assert_eq!(t.per_frame, want);
    }

    /// A kept hole is nothing in the live frame where the finished frame has
    /// land on the pixel and on all four sides; a skyline pixel drawn
    /// differently is not one, and neither is sky in both.
    #[test]
    fn a_kept_hole_is_nothing_inside_the_finished_frames_land() {
        let (nothing, land) = ([255u8, 0, 255, 255], [90u8, 120, 60, 255]);
        let image = |px: [[u8; 4]; 9]| Frame {
            width: 3,
            height: 3,
            pixels: px.concat(),
        };
        let done = image([land; 9]);
        let mut live = [land; 9];
        live[4] = nothing;
        assert_eq!(hole_fraction(&image(live), &done), 1.0 / 9.0);
        // The same pixel, but the finished frame has sky above it.
        let mut skyline = [land; 9];
        skyline[1] = nothing;
        assert_eq!(hole_fraction(&image(live), &image(skyline)), 0.0);
        // Sky in both.
        assert_eq!(
            hole_fraction(&image([nothing; 9]), &image([nothing; 9])),
            0.0
        );
    }

    /// The tail reads the slow end: p99.9 and the worst, and counts frames
    /// over one and two 60 Hz intervals -- not frames at exactly one.
    #[test]
    fn the_work_tail_is_the_slow_end() {
        let mut times = vec![0.002f32; 1000];
        times[10] = 0.020; // over one interval
        times[20] = 0.040; // over two
        times[30] = 1.0 / 60.0; // exactly one: not over
        let t = WorkTail::of(&times);
        assert_eq!((t.over_one, t.over_two), (2, 1));
        assert_eq!(t.max, 0.040);
        assert_eq!(t.p999, 0.020, "p99.9 of 1,000 frames is the second worst");
        assert_eq!(t.p99, 0.002);
    }

    #[test]
    fn the_frames_kept_are_the_worst_ones() {
        let mut worst = None;
        for (loading, name) in [(0.02, "a"), (0.3, "b"), (0.1, "c"), (0.3, "d")] {
            keep_worst(&mut worst, loading, name);
        }
        assert_eq!(worst, Some((0.3, "b")), "kept a later or a lesser frame");
    }

    /// Issue #378: frames become milliseconds at the flight's own pace, and
    /// blocks at its speed -- at 60 FPS and 24 blocks/s (`FLY_SPEED`), one
    /// frame is 1000/60 ms and 24/60 of a block.
    #[test]
    fn swap_lag_stats_convert_frames_with_the_flights_own_pace_and_speed() {
        let lags = [1u64, 2, 10, 100];
        let s = SwapLagStats::of(&lags, 60.0, 24.0);
        assert_eq!(s.count, 4);
        assert!((s.max_ms - 100.0 * 1000.0 / 60.0).abs() < 1e-9);
        assert!((s.max_blocks - 100.0 * 24.0 / 60.0).abs() < 1e-9);
        // p99 of these 4 sorted values is index (4-1)*990/1000 = 2: `10`.
        assert!((s.p99_ms - 10.0 * 1000.0 / 60.0).abs() < 1e-9);
    }

    /// A flight that only managed 30 frames a second under load: the same
    /// 30-frame lag is a whole second and 24 blocks, not the half second the
    /// 60 FPS it asked for would claim.
    #[test]
    fn a_slow_flight_converts_at_the_rate_it_achieved() {
        let s = SwapLagStats::of(&[30], 30.0, 24.0);
        assert!((s.max_ms - 1000.0).abs() < 1e-9);
        assert!((s.max_blocks - 24.0).abs() < 1e-9);
    }

    #[test]
    fn swap_lag_stats_of_nothing_is_the_default() {
        assert_eq!(SwapLagStats::of(&[], 60.0, 24.0), SwapLagStats::default());
    }
}

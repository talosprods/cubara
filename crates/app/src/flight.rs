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
    /// The world's seed; `None` is the default world.
    pub seed: Option<u64>,
}

/// Where the eye is `t` seconds into `flight`.
pub fn eye_at(flight: &Flight, t: f32) -> [f32; 3] {
    let dir = glam::Vec3::from(flight.look).normalize_or_zero();
    (glam::Vec3::from(flight.start) + dir * flight.speed * t).to_array()
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
        });
        let (work_p99, work_max) = p99_and_max(&self.work_times);
        format!(
            "FLIGHT: loading mean {:.2}% worst {:.2}% (at {:.2} s, {:.0?}), \
             frames over 1%: {:.0}%, work p99 {:.1} ms max {:.1} ms ({} kept frames)",
            self.mean() * 100.0,
            worst.loading * 100.0,
            worst.t,
            worst.eye,
            self.frames_over(0.01) * 100.0,
            work_p99 * 1000.0,
            work_max * 1000.0,
            self.samples.len()
        )
    }
}

fn p99_and_max(times: &[f32]) -> (f32, f32) {
    let mut sorted = times.to_vec();
    sorted.sort_by(f32::total_cmp);
    let p99 = sorted
        .get(sorted.len().saturating_sub(1) * 99 / 100)
        .copied()
        .unwrap_or(0.0);
    (p99, sorted.last().copied().unwrap_or(0.0))
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
        let (renderer, assets) = Renderer::offscreen(flight.size.0, flight.size.1, camera)?;
        let layers = assets.layers;
        let registry = Arc::new(assets.registry);
        let nodes = NodeStreaming::new(
            registry,
            &crate::game::load_structure_registry(),
            &crate::game::load_ore_registry(),
            move |name: &str| layers.layer_of(name),
        );
        let world = Arc::new(match flight.seed {
            Some(seed) => World::with_seed(seed),
            None => World::new(),
        });
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
        self.far.update(&mut self.renderer, eye);
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
        eye: glam::Vec3::from(eye),
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
    let mut pacer = Pacer::new(flight.fps);
    let mut kept: Vec<(f32, [f32; 3], Frame)> = Vec::new();
    let mut work_times = Vec::new();
    let start = Instant::now();
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
        if t >= next_keep {
            let (resident, held, meshing, visible) = live.nodes.counts();
            log::debug!(
                "flight: t {t:5.2} s: {resident} resident, {held} held, {meshing} meshing, \
                 {visible} visible, {} uploads queued",
                live.renderer.uploads_pending()
            );
            kept.push((t, eye, live.renderer.read_frame()?));
            next_keep += flight.every;
        }
        pacer.wait();
    }
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
        log::info!(
            "flight: t {t:5.2} s at {eye:.0?}: {:.2}% loading",
            loading * 100.0
        );
        if let Some(dir) = frames.filter(|_| loading > 0.01) {
            for (kind, f) in [("live", &frame), ("settled", &reference)] {
                save(&dir.join(format!("t{:05.2}-{kind}.png", t)), f);
            }
        }
        keep_worst(&mut worst, loading, (frame, reference));
        samples.push(Sample { t, eye, loading });
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
    Some(FlightReport {
        samples,
        work_times,
    })
}

/// Paces a loop like a display showing `fps` frames a second: a frame that
/// ran late starts the next at once rather than catching up.
struct Pacer {
    budget: Duration,
    next: Instant,
}

impl Pacer {
    fn new(fps: u32) -> Self {
        Self {
            budget: Duration::from_secs_f64(1.0 / fps.max(1) as f64),
            next: Instant::now(),
        }
    }

    /// Call at the end of a frame.
    fn wait(&mut self) {
        self.next += self.budget;
        let now = Instant::now();
        match self.next.checked_duration_since(now) {
            Some(rest) => std::thread::sleep(rest),
            None => self.next = now,
        }
    }
}

/// A frame counts as matching the finished one, or as differing again, at
/// this share of pixels (0.1%: a small triangle on a 3136x860 screen is
/// 0.15% at a thousand pixels a side).
const STILL_SHARE: f64 = 0.001;

/// Per-channel tolerance when comparing two frames of a camera that did not
/// move: the same scene draws the same, so this only absorbs rounding.
const STILL_TOLERANCE: u8 = 4;

/// Most memory the distinct frames of a still run may take.
const STILL_FRAME_BYTES: usize = 3 << 30;

/// What a still run found, per frame, in the order drawn.
#[derive(Debug, Default, PartialEq)]
pub struct StillReport {
    pub frames: usize,
    /// The first frame that matched the finished one.
    pub first_match: Option<usize>,
    /// Frames after `first_match` that differ again, with the share of pixels.
    pub flicker: Vec<(usize, f64)>,
    /// Frames before `first_match` with sky where the finished frame has none.
    pub holes: Vec<(usize, f64)>,
}

impl StillReport {
    pub fn line(&self) -> String {
        let worst = |v: &[(usize, f64)]| v.iter().map(|&(_, x)| x).fold(0.0, f64::max) * 100.0;
        format!(
            "STILL: frames {}, after-settle flicker frames {} (worst {:.2}%), \
             loading-hole frames {} (worst {:.2}%)",
            self.frames,
            self.flicker.len(),
            worst(&self.flicker),
            self.holes.len(),
            worst(&self.holes)
        )
    }
}

/// Judge a run from, per frame, its share of pixels that differ from the
/// finished frame and its share that are sky where the finished frame is not.
/// Pure so that the rule -- what counts as flicker, what as a hole -- is
/// tested without a GPU.
pub fn judge_still(differs: &[f64], sky_holes: &[f64]) -> StillReport {
    let first_match = differs.iter().position(|&d| d < STILL_SHARE);
    let after = first_match.map_or(differs.len(), |m| m + 1);
    let flicker = (after..differs.len())
        .filter(|&i| differs[i] > STILL_SHARE)
        .map(|i| (i, differs[i]))
        .collect();
    let holes = (0..after.min(differs.len()))
        .filter(|&i| sky_holes[i] > STILL_SHARE)
        .map(|i| (i, sky_holes[i]))
        .collect();
    StillReport {
        frames: differs.len(),
        first_match,
        flicker,
        holes,
    }
}

/// The share of `live`'s pixels that are `sky` where `settled` is not.
fn sky_hole_share(live: &Frame, settled: &Frame, sky: [u8; 3]) -> f64 {
    let is_sky = |p: &[u8]| (0..3).all(|c| p[c].abs_diff(sky[c]) <= 3);
    let n = live
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .zip(settled.pixels.as_chunks::<4>().0.iter())
        .filter(|(l, s)| is_sky(&l[..]) && !is_sky(&s[..]))
        .count();
    n as f64 / (live.pixels.len() / 4) as f64
}

/// `--bench still`: stand at `flight.start` looking along `flight.look` in a
/// world that starts empty, as New World does, drawing every frame for
/// `flight.seconds`; then load everything and take that as the reference. See
/// [`judge_still`] for what is reported. With `frames`, the offending frames
/// and `reference.png` are written there. `None` without a GPU.
pub fn stand(flight: &Flight, frames: Option<&Path>) -> Option<StillReport> {
    let mut game = Streamed::new(flight)?;
    let eye = flight.start;
    let mut pacer = Pacer::new(flight.fps);
    // Frames that repeat the one before are stored once: loaded, a still
    // camera draws the same frame for as long as it likes.
    let mut distinct: Vec<Frame> = Vec::new();
    let mut order: Vec<Option<usize>> = Vec::new();
    let mut bytes = 0usize;
    let start = Instant::now();
    while start.elapsed().as_secs_f32() <= flight.seconds {
        game.frame(flight, eye);
        let frame = game.renderer.read_frame()?;
        if distinct.last().is_some_and(|l| l.pixels == frame.pixels) {
            order.push(Some(distinct.len() - 1));
        } else if bytes + frame.pixels.len() <= STILL_FRAME_BYTES {
            bytes += frame.pixels.len();
            distinct.push(frame);
            order.push(Some(distinct.len() - 1));
        } else {
            order.push(None);
        }
        pacer.wait();
    }
    if order.contains(&None) {
        log::warn!("still: out of frame memory; later changing frames are not judged");
    }
    let loaded_at = start.elapsed().as_secs_f32();
    game.settle(flight, eye);
    let reference = game.renderer.read_frame()?;
    log::info!(
        "still: {} frames in {loaded_at:.1} s ({} distinct), settled after {:.1} s more",
        order.len(),
        distinct.len(),
        start.elapsed().as_secs_f32() - loaded_at
    );

    // The sky is what the very first frame, of nothing, is.
    let first = distinct.first()?;
    let sky = [first.pixels[0], first.pixels[1], first.pixels[2]];
    if first
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .any(|p| p[..3] != sky)
    {
        log::warn!("still: the first frame is not all sky; holes are measured against its corner");
    }
    let measure = |f: &Frame| {
        (
            cubara_render::headless::compare(&f.pixels, &reference.pixels, STILL_TOLERANCE)
                .differing_fraction,
            sky_hole_share(f, &reference, sky),
        )
    };
    let per_distinct: Vec<(f64, f64)> = distinct.iter().map(measure).collect();
    let (differs, holes): (Vec<f64>, Vec<f64>) = order
        .iter()
        .map(|i| i.map_or((0.0, 0.0), |i| per_distinct[i]))
        .unzip();
    let report = judge_still(&differs, &holes);
    if let Some(dir) = frames {
        let _ = std::fs::create_dir_all(dir);
        save(&dir.join("reference.png"), &reference);
        let write = |kind: &str, picks: Vec<(usize, f64)>| {
            for (i, share) in picks {
                if let Some(Some(d)) = order.get(i) {
                    let name = format!("f{i:05}-{kind}-{:.2}pct.png", share * 100.0);
                    save(&dir.join(name), &distinct[*d]);
                }
            }
        };
        write("flicker", report.flicker.iter().take(60).copied().collect());
        // The worst holes, and the last ones: the ones that outlive the rest.
        let mut worst = report.holes.clone();
        worst.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut picks: Vec<_> = worst.into_iter().take(5).collect();
        picks.extend(report.holes.iter().rev().take(5).copied());
        picks.sort_by_key(|p| p.0);
        picks.dedup_by_key(|p| p.0);
        write("hole", picks);
    }
    Some(report)
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
            seed: None,
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
        };
        let r = FlightReport {
            samples: vec![sample(0.0), sample(0.02), sample(0.005), sample(0.1)],
            work_times: vec![0.002; 10],
        };
        assert_eq!(r.frames_over(0.01), 0.5);
        assert!((r.mean() - 0.03125).abs() < 1e-12);
        assert_eq!(r.worst().unwrap().loading, 0.1);
    }

    #[test]
    fn flicker_is_a_difference_after_the_first_match_and_a_hole_before_it() {
        // Loading, loading, matches, flickers, matches, flickers a little.
        let differs = [1.0, 0.5, 0.0, 0.01, 0.0, 0.0005];
        let holes = [1.0, 0.2, 0.0, 0.01, 0.0, 0.0];
        let r = judge_still(&differs, &holes);
        assert_eq!(r.first_match, Some(2));
        assert_eq!(r.flicker, vec![(3, 0.01)]);
        assert_eq!(r.holes, vec![(0, 1.0), (1, 0.2)]);
        // Never matching: everything is loading, nothing is flicker.
        let r = judge_still(&[1.0, 0.9], &[1.0, 0.0]);
        assert_eq!(
            (r.first_match, r.flicker.len(), r.holes.len()),
            (None, 0, 1)
        );
    }

    #[test]
    fn a_sky_hole_is_sky_where_the_finished_frame_is_not() {
        let sky = [10, 20, 30];
        let px = |c: [u8; 3]| [c[0], c[1], c[2], 255];
        let live = frame([px(sky), px(sky), px([1, 1, 1])].concat());
        let done = frame([px(sky), px([200, 0, 0]), px(sky)].concat());
        assert!((sky_hole_share(&live, &done, sky) - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn the_frames_kept_are_the_worst_ones() {
        let mut worst = None;
        for (loading, name) in [(0.02, "a"), (0.3, "b"), (0.1, "c"), (0.3, "d")] {
            keep_worst(&mut worst, loading, name);
        }
        assert_eq!(worst, Some((0.3, "b")), "kept a later or a lesser frame");
    }
}

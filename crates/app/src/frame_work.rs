//! Where a frame's work went (issue #454): the frame's phases timed one after
//! another, the size of each phase's job counted, and the longest frames of
//! a run broken down by phase -- the `WORST FRAMES` line `--bench flight`
//! prints beside its `FLIGHT` line.
//!
//! `--bench flight` showed rare frames of 17-34 ms in its work, main
//! included, and nothing said which part of the frame took the time: a burst
//! of uploads, of far patches, a search answer -- or none of them, which is
//! the machine's. This says.
//!
//! **The game's frame and the flight's are the same calls.** The streamers
//! report their phases through a [`Laps`] as they go
//! (`NodeStreaming::update_lapped`, `FarStreaming::update_lapped`); the game
//! passes `()`, which reads no clock and keeps nothing, so nothing in the
//! streamers ever reads a clock itself. The renderer times its own phases
//! (`cubara_render::RenderWork`) and the flight's [`Stopwatch`] splits its
//! lap by them.
//!
//! **The parts sum to the frame.** Every lap is the time since the one before
//! it, from one `Instant` each, so a frame's phases add up to the time from
//! its first read to its last exactly -- and that sum is the work
//! `--bench flight` reports for the frame ([`FrameWork::total`]). What the
//! renderer did not time itself is [`Phase::RenderRest`], so the split of its
//! lap adds up as well.
//!
//! **What it costs:** one `Instant::now()` per lap, 13 a flight frame, and
//! the renderer's own nine -- 14-26 ns each on an M3; 0.4-0.5 us a frame
//! with the record kept, against frames of milliseconds (`lap_cost_report`).
//! The game's frame pays only the renderer's nine. Nothing here changes what
//! a frame does.

use std::time::{Duration, Instant};

use cubara_render::RenderWork;

/// A part of a frame's work, in the order a frame does them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Starting a visibility search, and applying the answers that arrived:
    /// what is dropped, held and asked for (`NodeStreaming::follow_visibility`).
    Search,
    /// Taking finished meshes from the mesh pool and queueing them for the
    /// renderer.
    Meshes,
    /// Letting go of held nodes whose replacements have arrived
    /// (`NodeStreaming::release_held`).
    Held,
    /// The voxels' box the far terrain is holed at, and whether the far
    /// terrain is in sight at all.
    VoxelBox,
    /// Taking the far patches the far pool generated.
    FarArrivals,
    /// The far terrain's plan: its selection and its prefetch (`FarPlan::step`).
    FarPlan,
    /// Asking the far pool for patches.
    FarAsks,
    /// Uploading far patches, the strip's stand-in included, and freeing the
    /// slots of those that left.
    FarInserts,
    /// The renderer moving its anchor (once per 4,096 blocks of travel).
    Rebase,
    /// The renderer applying queued node uploads and drops.
    Uploads,
    /// The renderer waiting for a lighting pipeline rebuild (offscreen only).
    LightingWait,
    /// The renderer culling the voxel nodes and writing their draw list.
    Cull,
    /// The renderer culling the far terrain.
    FarCull,
    /// The renderer's tree instance lists.
    Trees,
    /// The renderer recording the frame's commands.
    Encode,
    /// The renderer handing them to the GPU, with every buffer written this
    /// frame.
    Submit,
    /// What the renderer's frame spent outside the phases above.
    RenderRest,
}

/// How many [`Phase`]s there are.
pub const PHASES: usize = 17;

impl Phase {
    /// Every phase, in order.
    pub const ALL: [Phase; PHASES] = [
        Phase::Search,
        Phase::Meshes,
        Phase::Held,
        Phase::VoxelBox,
        Phase::FarArrivals,
        Phase::FarPlan,
        Phase::FarAsks,
        Phase::FarInserts,
        Phase::Rebase,
        Phase::Uploads,
        Phase::LightingWait,
        Phase::Cull,
        Phase::FarCull,
        Phase::Trees,
        Phase::Encode,
        Phase::Submit,
        Phase::RenderRest,
    ];

    /// The phase's name in a line.
    pub fn name(self) -> &'static str {
        match self {
            Phase::Search => "search",
            Phase::Meshes => "meshes",
            Phase::Held => "held",
            Phase::VoxelBox => "voxel box",
            Phase::FarArrivals => "far arrivals",
            Phase::FarPlan => "far plan",
            Phase::FarAsks => "far asks",
            Phase::FarInserts => "far inserts",
            Phase::Rebase => "rebase",
            Phase::Uploads => "uploads",
            Phase::LightingWait => "lighting wait",
            Phase::Cull => "cull",
            Phase::FarCull => "far cull",
            Phase::Trees => "trees",
            Phase::Encode => "encode",
            Phase::Submit => "submit",
            Phase::RenderRest => "render rest",
        }
    }

    /// What the phase counts, in the order a line names them.
    fn counts(self) -> &'static [Count] {
        match self {
            Phase::Search => &[Count::Answers, Count::Asked],
            Phase::Meshes => &[Count::Meshes],
            Phase::Held => &[Count::Held, Count::InFlight, Count::Released],
            Phase::FarArrivals => &[Count::FarArrivals],
            Phase::FarAsks => &[Count::FarAsks],
            Phase::FarInserts => &[Count::FarInserts, Count::FarBytes, Count::FarFreed],
            Phase::Uploads => &[Count::Uploaded, Count::UploadBytes, Count::Dropped],
            _ => &[],
        }
    }
}

/// The size of a phase's job: what grows with a burst.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Count {
    /// Search answers applied.
    Answers,
    /// Nodes those answers asked the mesh pool for.
    Asked,
    /// Finished meshes taken from the pool.
    Meshes,
    /// Nodes held for a replacement when the held were checked.
    Held,
    /// Nodes being meshed then: what each held node is checked against.
    InFlight,
    /// Held nodes let go.
    Released,
    /// Far patches taken from the far pool.
    FarArrivals,
    /// Far patches asked for (urgent, ahead, or moved to urgent).
    FarAsks,
    /// Far patches uploaded.
    FarInserts,
    /// Their faces, in bytes.
    FarBytes,
    /// Far slots freed.
    FarFreed,
    /// Nodes the renderer placed in its arena.
    Uploaded,
    /// Their vertices and indices, in bytes.
    UploadBytes,
    /// Nodes the renderer dropped.
    Dropped,
}

/// How many [`Count`]s there are.
pub const COUNTS: usize = 14;

impl Count {
    /// How a line names `n` of it.
    fn show(self, n: u64) -> String {
        let unit = match self {
            Count::FarBytes | Count::UploadBytes => return bytes(n),
            Count::Answers => "answers",
            Count::Asked => "nodes asked",
            Count::Meshes => "meshes",
            Count::Held => "held",
            Count::InFlight => "in flight",
            Count::Released => "released",
            Count::FarArrivals | Count::FarInserts => "patches",
            Count::FarAsks => "patches asked",
            Count::FarFreed => "freed",
            Count::Uploaded => "nodes",
            Count::Dropped => "dropped",
        };
        format!("{n} {unit}")
    }
}

/// `n` bytes, in a line: kilobytes, or megabytes from one.
fn bytes(n: u64) -> String {
    const MB: f64 = 1024.0 * 1024.0;
    if n as f64 >= MB {
        format!("{:.1} MB", n as f64 / MB)
    } else {
        format!("{:.0} KB", n as f64 / 1024.0)
    }
}

/// What a frame reports about its own work as it goes. The game passes `()`:
/// no clock read, nothing kept. `--bench flight` passes a [`Stopwatch`].
pub trait Laps {
    /// The work since the last lap was `phase`'s.
    fn lap(&mut self, phase: Phase);
    /// The frame did `n` more of `what`.
    fn count(&mut self, what: Count, n: u64);
}

impl Laps for () {
    fn lap(&mut self, _: Phase) {}
    fn count(&mut self, _: Count, _: u64) {}
}

/// One frame's work: each phase's wall time and each count.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameWork {
    /// When the frame was drawn, in seconds of flight.
    pub t: f32,
    phases: [Duration; PHASES],
    counts: [u64; COUNTS],
}

impl FrameWork {
    /// `phase` took `d` more.
    pub fn add(&mut self, phase: Phase, d: Duration) {
        self.phases[phase as usize] += d;
    }

    /// `n` more of `what`.
    pub fn add_count(&mut self, what: Count, n: u64) {
        self.counts[what as usize] += n;
    }

    pub fn phase(&self, phase: Phase) -> Duration {
        self.phases[phase as usize]
    }

    pub fn count(&self, what: Count) -> u64 {
        self.counts[what as usize]
    }

    /// The frame's work: every phase's, summed -- what `--bench flight`
    /// reports as the frame's work.
    pub fn total(&self) -> Duration {
        self.phases.iter().sum()
    }

    /// The phase that took longest; of equal ones, the earlier in the frame.
    pub fn longest(&self) -> Phase {
        Phase::ALL
            .into_iter()
            .rev()
            .max_by_key(|&p| self.phase(p))
            .unwrap_or(Phase::Search)
    }

    /// The renderer's frame, `span` long, split by what the renderer timed
    /// itself; what it did not is [`Phase::RenderRest`], so the parts sum to
    /// `span` (the renderer's phases lie inside its frame).
    pub fn add_render(&mut self, span: Duration, work: &RenderWork) {
        let parts = [
            (Phase::Rebase, work.rebase),
            (Phase::Uploads, work.uploads),
            (Phase::LightingWait, work.lighting_wait),
            (Phase::Cull, work.cull),
            (Phase::FarCull, work.far_cull),
            (Phase::Trees, work.trees),
            (Phase::Encode, work.encode),
            (Phase::Submit, work.submit),
        ];
        let timed: Duration = parts.iter().map(|&(_, d)| d).sum();
        for (phase, d) in parts {
            self.add(phase, d);
        }
        self.add(Phase::RenderRest, span.saturating_sub(timed));
        self.add_count(Count::Uploaded, u64::from(work.uploaded_nodes));
        self.add_count(Count::UploadBytes, work.uploaded_bytes);
        self.add_count(Count::Dropped, u64::from(work.dropped_nodes));
    }
}

/// A [`Laps`] that times: each lap is the time since the one before.
pub struct Stopwatch {
    last: Instant,
    work: FrameWork,
}

impl Stopwatch {
    pub fn start() -> Self {
        Self::start_at(Instant::now())
    }

    /// Started at `now`.
    pub fn start_at(now: Instant) -> Self {
        Self {
            last: now,
            work: FrameWork::default(),
        }
    }

    /// A lap of `phase` ending at `now`.
    pub fn lap_at(&mut self, phase: Phase, now: Instant) {
        self.work
            .add(phase, now.saturating_duration_since(self.last));
        self.last = now;
    }

    /// A lap that was the renderer's frame, split by what it timed itself
    /// ([`FrameWork::add_render`]).
    pub fn lap_render(&mut self, work: &RenderWork) {
        self.lap_render_at(work, Instant::now());
    }

    /// [`lap_render`](Self::lap_render), ending at `now`.
    pub fn lap_render_at(&mut self, work: &RenderWork, now: Instant) {
        self.work
            .add_render(now.saturating_duration_since(self.last), work);
        self.last = now;
    }

    /// The frame's work, laps and counts.
    pub fn finish(self) -> FrameWork {
        self.work
    }
}

impl Laps for Stopwatch {
    fn lap(&mut self, phase: Phase) {
        self.lap_at(phase, Instant::now());
    }

    fn count(&mut self, what: Count, n: u64) {
        self.work.add_count(what, n);
    }
}

/// The `n` frames with the most work, the longest first; of equal ones, the
/// earlier.
pub fn worst(frames: &[FrameWork], n: usize) -> Vec<FrameWork> {
    let mut sorted = frames.to_vec();
    // Stable: equal frames keep their order, which is the flight's.
    sorted.sort_by_key(|f| std::cmp::Reverse(f.total()));
    sorted.truncate(n);
    sorted
}

/// Each phase's mean over `frames`, and its counts' means rounded down: the
/// frame the worst ones are read against. Its parts sum to the mean work.
pub fn mean(frames: &[FrameWork]) -> FrameWork {
    let mut sum = FrameWork::default();
    let n = frames.len().max(1);
    for f in frames {
        for p in Phase::ALL {
            sum.add(p, f.phase(p));
        }
        for c in 0..COUNTS {
            sum.counts[c] += f.counts[c];
        }
    }
    let mut mean = FrameWork::default();
    for p in Phase::ALL {
        mean.add(p, sum.phase(p) / n as u32);
    }
    for c in 0..COUNTS {
        mean.counts[c] = sum.counts[c] / n as u64;
    }
    mean
}

/// Of the frames whose work is over `limit`, how many had each phase as
/// their longest -- every such frame given to exactly one phase, the most
/// frames first (then in frame order).
pub fn by_longest(frames: &[FrameWork], limit: Duration) -> Vec<(Phase, usize)> {
    let mut tally = [0usize; PHASES];
    for f in frames.iter().filter(|f| f.total() > limit) {
        tally[f.longest() as usize] += 1;
    }
    let mut out: Vec<(Phase, usize)> = Phase::ALL
        .into_iter()
        .map(|p| (p, tally[p as usize]))
        .filter(|&(_, n)| n > 0)
        .collect();
    out.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
    out
}

/// A phase is named in a line from this long; shorter ones are summed into
/// one term. Chosen, not derived: a tenth of a millisecond is under a
/// twentieth of the 2-3 ms a flight frame takes.
const NAMED_FROM: Duration = Duration::from_micros(100);

/// How many of the longest frames the line breaks down. Chosen, not
/// derived: the flights showed at most four frames over 16.7 ms each.
pub const WORST_SHOWN: usize = 5;

/// One frame broken down: its work, then each phase of at least
/// [`NAMED_FROM`], longest first, with what it counted, then the rest summed.
/// The longest phase is always named.
fn breakdown(f: &FrameWork) -> String {
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    let mut phases: Vec<Phase> = Phase::ALL.to_vec();
    phases.sort_by_key(|&p| std::cmp::Reverse(f.phase(p)));
    let longest = f.longest();
    let (named, small): (Vec<Phase>, Vec<Phase>) = phases
        .into_iter()
        .partition(|&p| p == longest || f.phase(p) >= NAMED_FROM);
    let mut terms: Vec<String> = named
        .iter()
        .map(|&p| {
            let counted: Vec<String> = p
                .counts()
                .iter()
                .filter(|&&c| f.count(c) > 0)
                .map(|&c| c.show(f.count(c)))
                .collect();
            if counted.is_empty() {
                format!("{} {:.2}", p.name(), ms(f.phase(p)))
            } else {
                format!(
                    "{} {:.2} ({})",
                    p.name(),
                    ms(f.phase(p)),
                    counted.join(", ")
                )
            }
        })
        .collect();
    if !small.is_empty() {
        let rest: Duration = small.iter().map(|&p| f.phase(p)).sum();
        terms.push(format!("{} more {:.2}", small.len(), ms(rest)));
    }
    format!("{:.2} ms = {}", ms(f.total()), terms.join(" + "))
}

/// The `WORST FRAMES` line: the mean frame broken down by phase, every frame
/// over a 60 Hz interval given to its longest phase, and the
/// [`WORST_SHOWN`] longest frames broken down -- when, how long, and each
/// phase with what it counted. Every breakdown sums to its frame's work.
pub fn worst_line(frames: &[FrameWork]) -> String {
    let interval = Duration::from_secs_f64(1.0 / 60.0);
    let over = by_longest(frames, interval);
    let over_n: usize = over.iter().map(|&(_, n)| n).sum();
    let over_text = if over.is_empty() {
        String::new()
    } else {
        let by: Vec<String> = over
            .iter()
            .map(|&(p, n)| format!("{} {n}", p.name()))
            .collect();
        format!(", longest phase: {}", by.join(", "))
    };
    let worst: Vec<String> = worst(frames, WORST_SHOWN)
        .iter()
        .map(|f| format!("t {:.2} s {}", f.t, breakdown(f)))
        .collect();
    format!(
        "WORST FRAMES: {} frames, mean {} | frames over 16.7 ms: {over_n}{over_text} | \
         the {} longest: {}",
        frames.len(),
        breakdown(&mean(frames)),
        worst.len(),
        worst.join(" | "),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: f64) -> Duration {
        Duration::from_secs_f64(n / 1000.0)
    }

    /// A frame whose phases took `phases` (in milliseconds), at `t`.
    fn frame(t: f32, phases: &[(Phase, f64)]) -> FrameWork {
        let mut f = FrameWork {
            t,
            ..FrameWork::default()
        };
        for &(p, d) in phases {
            f.add(p, ms(d));
        }
        f
    }

    #[test]
    fn the_phase_list_and_its_names_cover_every_phase_once() {
        for (i, p) in Phase::ALL.into_iter().enumerate() {
            assert_eq!(p as usize, i, "{p:?} out of order");
        }
        let mut names: Vec<&str> = Phase::ALL.iter().map(|p| p.name()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), PHASES, "two phases share a name");
    }

    /// **The parts sum to the frame's work**: each lap is the time since the
    /// one before, so the laps add up to the time from the stopwatch's start
    /// to its last lap, to the nanosecond -- and the renderer's lap is split
    /// without losing any of it.
    #[test]
    fn the_laps_sum_to_the_time_from_the_start_to_the_last_lap() {
        let start = Instant::now();
        let at = |us: u64| start + Duration::from_micros(us);
        let mut watch = Stopwatch::start_at(start);
        watch.lap_at(Phase::Search, at(130));
        watch.lap_at(Phase::Meshes, at(450));
        watch.lap_at(Phase::Held, at(451));
        // A phase lapped twice adds up.
        watch.lap_at(Phase::Meshes, at(700));
        watch.lap_at(Phase::FarPlan, at(2_100));
        let renderer = RenderWork {
            uploads: Duration::from_micros(900),
            cull: Duration::from_micros(400),
            submit: Duration::from_micros(250),
            uploaded_nodes: 40,
            uploaded_bytes: 3 << 20,
            dropped_nodes: 2,
            ..RenderWork::default()
        };
        watch.lap_render_at(&renderer, at(3_900));
        let f = watch.finish();
        assert_eq!(f.total(), Duration::from_micros(3_900));
        assert_eq!(f.phase(Phase::Meshes), Duration::from_micros(320 + 249));
        assert_eq!(f.phase(Phase::Uploads), Duration::from_micros(900));
        // The renderer's lap was 1,800 us, 1,550 of it timed by itself.
        assert_eq!(f.phase(Phase::RenderRest), Duration::from_micros(250));
        assert_eq!(f.count(Count::Uploaded), 40);
        assert_eq!(f.count(Count::UploadBytes), 3 << 20);
        assert_eq!(f.count(Count::Dropped), 2);
    }

    #[test]
    fn counts_add_up_through_the_laps() {
        let mut watch = Stopwatch::start();
        watch.count(Count::FarInserts, 12);
        watch.count(Count::FarInserts, 3);
        watch.count(Count::Asked, 200);
        let f = watch.finish();
        assert_eq!(f.count(Count::FarInserts), 15);
        assert_eq!(f.count(Count::Asked), 200);
        assert_eq!(f.count(Count::Meshes), 0);
    }

    #[test]
    fn the_worst_frames_are_the_longest_longest_first() {
        let frames: Vec<FrameWork> = [1.0, 5.0, 3.0, 9.0, 2.0, 5.0]
            .iter()
            .enumerate()
            .map(|(i, &d)| frame(i as f32, &[(Phase::Cull, d)]))
            .collect();
        let worst = worst(&frames, 3);
        let ts: Vec<f32> = worst.iter().map(|f| f.t).collect();
        // 9, then the two 5s in flight order.
        assert_eq!(ts, vec![3.0, 1.0, 5.0]);
        assert_eq!(super::worst(&frames, 10).len(), 6);
    }

    /// **A long phase is attributed to that phase**: a frame of normal
    /// phases and one long far insert names the far inserts, first, with
    /// what they counted.
    #[test]
    fn a_long_phase_is_named_first_with_its_counts() {
        let mut f = frame(
            12.34,
            &[
                (Phase::Search, 0.05),
                (Phase::FarInserts, 20.0),
                (Phase::Cull, 0.8),
                (Phase::Submit, 0.5),
                (Phase::Encode, 0.02),
            ],
        );
        f.add_count(Count::FarInserts, 130);
        f.add_count(Count::FarBytes, 3 << 20);
        assert_eq!(f.longest(), Phase::FarInserts);
        assert_eq!(
            breakdown(&f),
            "21.37 ms = far inserts 20.00 (130 patches, 3.0 MB) + cull 0.80 + submit 0.50 \
             + 14 more 0.07"
        );
    }

    /// A frame whose every phase is under the line's threshold still names
    /// its longest: a breakdown always says where the time went.
    #[test]
    fn a_frame_of_small_phases_still_names_its_longest() {
        let f = frame(0.0, &[(Phase::Cull, 0.06), (Phase::Submit, 0.03)]);
        assert_eq!(breakdown(&f), "0.09 ms = cull 0.06 + 16 more 0.03");
    }

    #[test]
    fn of_equal_phases_the_earlier_is_the_longest() {
        let f = frame(0.0, &[(Phase::Uploads, 2.0), (Phase::Meshes, 2.0)]);
        assert_eq!(f.longest(), Phase::Meshes);
        // ...and a frame that did nothing still names one.
        assert_eq!(FrameWork::default().longest(), Phase::Search);
    }

    #[test]
    fn every_frame_over_the_limit_goes_to_its_longest_phase() {
        let frames = [
            frame(0.0, &[(Phase::Uploads, 20.0), (Phase::Cull, 1.0)]),
            frame(1.0, &[(Phase::Cull, 2.0)]),
            frame(2.0, &[(Phase::FarInserts, 12.0), (Phase::Uploads, 8.0)]),
            frame(3.0, &[(Phase::Uploads, 17.0)]),
            // On the limit is not over it.
            frame(4.0, &[(Phase::Search, 10.0), (Phase::Held, 6.7)]),
        ];
        assert_eq!(
            by_longest(&frames, ms(16.7)),
            vec![(Phase::Uploads, 2), (Phase::FarInserts, 1)]
        );
    }

    #[test]
    fn the_mean_frame_is_each_phases_mean_and_sums_to_the_mean_work() {
        let mut a = frame(0.0, &[(Phase::Cull, 1.0), (Phase::Uploads, 3.0)]);
        a.add_count(Count::Uploaded, 10);
        let mut b = frame(1.0, &[(Phase::Cull, 3.0), (Phase::Submit, 2.0)]);
        b.add_count(Count::Uploaded, 5);
        let m = mean(&[a, b]);
        assert_eq!(m.phase(Phase::Cull), ms(2.0));
        assert_eq!(m.phase(Phase::Uploads), ms(1.5));
        assert_eq!(m.phase(Phase::Submit), ms(1.0));
        assert_eq!(m.total(), (a.total() + b.total()) / 2);
        assert_eq!(m.count(Count::Uploaded), 7);
        assert_eq!(mean(&[]), FrameWork::default());
    }

    #[test]
    fn the_line_names_the_mean_the_frames_over_and_the_worst() {
        let mut long = frame(
            7.5,
            &[
                (Phase::Uploads, 18.0),
                (Phase::Cull, 1.0),
                (Phase::Held, 0.02),
            ],
        );
        long.add_count(Count::Uploaded, 212);
        long.add_count(Count::UploadBytes, 38 << 20);
        let frames = [frame(0.0, &[(Phase::Cull, 1.0)]), long];
        assert_eq!(
            worst_line(&frames),
            "WORST FRAMES: 2 frames, mean 10.01 ms = uploads 9.00 (106 nodes, 19.0 MB) + \
             cull 1.00 + 15 more 0.01 | frames over 16.7 ms: 1, longest phase: uploads 1 | \
             the 2 longest: t 7.50 s 19.02 ms = uploads 18.00 (212 nodes, 38.0 MB) + \
             cull 1.00 + 15 more 0.02 | t 0.00 s 1.00 ms = cull 1.00 + 16 more 0.00"
        );
    }

    /// **What the laps cost a frame** (issue #454: the instrument must not
    /// change the times it measures): a clock read, and a frame's worth of
    /// laps as `--bench flight` takes them -- the start, the streamers' nine
    /// laps, the voxel box's two, the renderer's split and the record --
    /// timed over a million frames. The renderer's own nine reads are nine
    /// clock reads. Prints; asserts nothing.
    #[test]
    #[ignore = "a timing: cargo test -p cubara --release lap_cost_report -- --ignored --nocapture"]
    fn lap_cost_report() {
        const N: u32 = 1_000_000;
        let began = Instant::now();
        let mut last = began;
        for _ in 0..N {
            last = std::hint::black_box(Instant::now());
        }
        let read = (last - began) / N;
        let laps = [
            Phase::Search,
            Phase::Meshes,
            Phase::Held,
            Phase::Meshes,
            Phase::VoxelBox,
            Phase::FarArrivals,
            Phase::FarPlan,
            Phase::FarInserts,
            Phase::FarAsks,
            Phase::FarInserts,
            Phase::VoxelBox,
        ];
        let mut kept = Vec::with_capacity(N as usize);
        let began = Instant::now();
        for _ in 0..N {
            let mut watch = Stopwatch::start();
            for p in laps {
                watch.lap(p);
            }
            watch.count(Count::Asked, 3);
            watch.lap_render(&RenderWork::default());
            kept.push(watch.finish());
        }
        let frame = began.elapsed() / N;
        std::hint::black_box(&kept);
        println!(
            "LAP COST: {} ns a clock read; {} ns a frame's laps ({} reads, the renderer's \
             split, the record kept); with the renderer's 9 reads, {} ns a frame",
            read.as_nanos(),
            frame.as_nanos(),
            laps.len() + 2,
            (frame + read * 9).as_nanos(),
        );
    }

    #[test]
    fn bytes_read_as_kilobytes_then_megabytes() {
        assert_eq!(bytes(0), "0 KB");
        assert_eq!(bytes(1536), "2 KB");
        assert_eq!(bytes(1 << 20), "1.0 MB");
        assert_eq!(bytes(38 << 20), "38.0 MB");
    }
}

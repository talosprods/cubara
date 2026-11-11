//! `--bench walk --window`: what a present-to-present trace says, without a
//! window (issue #388).
//!
//! The window loop in `main.rs` records when each `present()` returned; this
//! module turns those instants into gaps, splits start-up from play, lists the
//! worst gaps and prints the line. Pure, so it is tested with no display
//! (decision 5). The statistics themselves are [`WorkTail`] -- one percentile
//! implementation (Rule 5).
//!
//! What a gap shows depends on the present mode. Under Fifo the loop cannot get
//! a drawable until the display released one, so gaps sit at the refresh
//! interval and a gap over `MISSED_REFRESH` is a refresh that showed the
//! previous frame: the closest this gets to the glass. Under Immediate (the
//! only other mode Metal offers) the gap is the loop's own cadence: it shows the
//! app thread did not hitch, not which frames reached a refresh. wgpu 30 has no
//! presented-time API, so on-glass time is out of reach here.

use crate::flight::WorkTail;
use crate::paths::{same_path, SavePaths};

/// The part of the run that counts as play, in seconds from process start.
/// Chosen, not derived: equal to `options::SETTLE_SECONDS`, the game's own
/// "loading is over" allowance.
pub const STEADY_AFTER: f64 = 10.0;
/// A gap longer than this means a refresh went by with no new frame, in
/// seconds. Chosen, not derived: 1.5 x 1/60, the factor of `FrameWatch`'s
/// `MISSED`.
pub const MISSED_REFRESH: f32 = 0.025;
/// How many of the worst gaps the line lists. Chosen, not derived.
pub const WORST_KEPT: usize = 5;
/// How long after a rebuild is requested its frame is looked for, in seconds.
/// Chosen, not derived.
pub const REBUILD_WINDOW: f64 = 1.0;

/// `--present fifo|mailbox|immediate` (case-insensitive); `None` for anything
/// else.
pub fn parse_present(text: &str) -> Option<wgpu::PresentMode> {
    match text.to_ascii_lowercase().as_str() {
        "fifo" => Some(wgpu::PresentMode::Fifo),
        "mailbox" => Some(wgpu::PresentMode::Mailbox),
        "immediate" => Some(wgpu::PresentMode::Immediate),
        _ => None,
    }
}

/// One gap: when the later present returned (seconds since process start) and
/// how long after the one before it (seconds).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gap {
    pub at: f64,
    pub took: f32,
}

/// Present-to-present gaps of `presents` (seconds since process start, in
/// order).
pub fn gaps(presents: &[f64]) -> Vec<Gap> {
    presents
        .windows(2)
        .map(|w| Gap {
            at: w[1],
            took: (w[1] - w[0]) as f32,
        })
        .collect()
}

/// How many gaps took longer than `seconds`.
pub fn over(gaps: &[Gap], seconds: f32) -> usize {
    gaps.iter().filter(|g| g.took > seconds).count()
}

/// The `n` longest gaps, longest first (ties: earliest first).
pub fn worst(gaps: &[Gap], n: usize) -> Vec<Gap> {
    let mut sorted = gaps.to_vec();
    sorted.sort_by(|a, b| b.took.total_cmp(&a.took).then(a.at.total_cmp(&b.at)));
    sorted.truncate(n);
    sorted
}

/// The longest gap that ended in `[from, from + REBUILD_WINDOW]`, if any.
pub fn longest_after(gaps: &[Gap], from: f64) -> Option<Gap> {
    let inside: Vec<Gap> = gaps
        .iter()
        .filter(|g| g.at >= from && g.at <= from + REBUILD_WINDOW)
        .copied()
        .collect();
    worst(&inside, 1).first().copied()
}

/// `Err` with a message when `paths` are the owner's own.
pub fn refuse_owners_world(paths: &SavePaths) -> Result<(), String> {
    let standard = SavePaths::standard();
    if same_path(&paths.world, &standard.world) || same_path(&paths.settings, &standard.settings) {
        Err(
            "--bench walk --window needs --world <dir>: it will not run in the owner's saves"
                .to_string(),
        )
    } else {
        Ok(())
    }
}

/// Whether the arguments ask for the windowed walk: `Ok(true)`, a plain start
/// or another bench `Ok(false)`, and `Err` (the message `main` prints before
/// exiting with 2) for `--window` without `--bench walk`, or in the owner's
/// own world.
pub fn check_window_args(args: &[String], paths: &SavePaths) -> Result<bool, String> {
    let window = args.iter().any(|a| a == "--window");
    let walk = args
        .iter()
        .position(|a| a == "--bench")
        .and_then(|i| args.get(i + 1))
        .is_some_and(|w| w == "walk");
    match (window, walk) {
        (false, _) => Ok(false),
        (true, false) => Err("--window goes with --bench walk".to_string()),
        (true, true) => refuse_owners_world(paths).map(|()| true),
    }
}

/// How long past `seconds` a windowed run may last before the watchdog gives
/// up: an occluded window can stop redrawing, and under Fifo `nextDrawable`
/// blocks inside `render`, so no loop hook could end the run.
pub const WATCHDOG_SLACK: f64 = 30.0;

/// Presents and skipped frames as the window loop saw them.
#[derive(Default)]
pub struct Trace {
    pub presents: Vec<f64>,
    pub acquires: Vec<f32>,
    pub skipped: u32,
}

impl Trace {
    /// One frame's outcome: `Some((present time in seconds from process
    /// start, acquire wait in seconds))` when it was presented, `None` when
    /// the surface had no texture and the frame was skipped.
    pub fn record(&mut self, presented: Option<(f64, f32)>) {
        match presented {
            Some((at, acquire)) => {
                self.presents.push(at);
                self.acquires.push(acquire);
            }
            None => self.skipped += 1,
        }
    }
}

/// What became of the start-up `--bench tune` child.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tune {
    /// `--no-tune`.
    Off,
    /// Finished this many seconds after process start.
    Finished(f64),
    StillRunning,
    /// The settings already matched this display, so no child was started.
    NotNeeded,
}

/// Everything the windowed walk recorded.
pub struct PresentReport {
    pub presents: Vec<f64>,
    /// Seconds inside `get_current_texture`, one per present.
    pub acquires: Vec<f32>,
    pub size: (u32, u32),
    pub mode: String,
    pub refresh_hz: u32,
    pub fullscreen: bool,
    /// Frames `render` returned `None` for.
    pub skipped: u32,
    /// `Focused(false)` plus `Occluded(true)` events.
    pub focus_lost: u32,
    pub tune: Tune,
    pub far: String,
    /// When the resize was requested, seconds from process start.
    pub rebuild: Option<f64>,
    pub fly: bool,
    pub moved: f32,
    pub ticks: u64,
}

fn ms(seconds: f32) -> f32 {
    seconds * 1000.0
}

impl PresentReport {
    pub fn line(&self) -> String {
        let all = gaps(&self.presents);
        let play: Vec<Gap> = all
            .iter()
            .filter(|g| g.at >= STEADY_AFTER)
            .copied()
            .collect();
        let took = |g: &[Gap]| g.iter().map(|g| g.took).collect::<Vec<f32>>();
        let t = WorkTail::of(&took(&all));
        let p = WorkTail::of(&took(&play));
        let a = WorkTail::of(&self.acquires);
        let first = match self.presents.first() {
            Some(f) => format!("{f:.2} s"),
            None => "n/a".to_string(),
        };
        let worst_gaps = worst(&all, WORST_KEPT)
            .iter()
            .map(|g| format!("{:.1} ms @ {:.1} s", ms(g.took), g.at))
            .collect::<Vec<_>>()
            .join(", ");
        let tune = match self.tune {
            Tune::Off => "off".to_string(),
            Tune::NotNeeded => "not needed".to_string(),
            Tune::Finished(at) => format!("done at {at:.1} s"),
            Tune::StillRunning => "still running".to_string(),
        };
        let rebuild = match self.rebuild {
            None => "none".to_string(),
            Some(at) => match longest_after(&all, at) {
                Some(g) => format!("at {at:.1} s: {:.1} ms @ {:.1} s", ms(g.took), g.at),
                None => format!("at {at:.1} s: no present followed"),
            },
        };
        format!(
            "PRESENT: p99 {:.1} / p99.9 {:.1} / max {:.1} ms, over 16.7 ms {} over 33.3 ms {} \
             ({} frames) | {}x{} {} {} {} Hz | over 25.0 ms {} | after {:.0} s: p99.9 {:.1} \
             max {:.1} ms, over 25.0 ms {} over 33.3 ms {} ({} frames) | acquire wait p99.9 \
             {:.1} max {:.1} ms | first present {first} | worst {worst_gaps} | tune {tune} | \
             far {} | rebuild {rebuild} | skipped {} focus lost {} | {} {:.0} blocks, {} ticks",
            ms(t.p99),
            ms(t.p999),
            ms(t.max),
            t.over_one,
            t.over_two,
            all.len(),
            self.size.0,
            self.size.1,
            if self.fullscreen {
                "fullscreen"
            } else {
                "windowed"
            },
            self.mode,
            self.refresh_hz,
            over(&all, MISSED_REFRESH),
            STEADY_AFTER,
            ms(p.p999),
            ms(p.max),
            over(&play, MISSED_REFRESH),
            p.over_two,
            play.len(),
            ms(a.p999),
            ms(a.max),
            self.far,
            self.skipped,
            self.focus_lost,
            if self.fly { "fly" } else { "walk" },
            self.moved,
            self.ticks,
        )
    }
}

/// `--trace`: every gap, for a plot.
pub fn trace_csv(gaps: &[Gap]) -> String {
    let mut out = String::from("t_s,gap_ms\n");
    for g in gaps {
        out.push_str(&format!("{:.4},{:.3}\n", g.at, ms(g.took)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(presents: Vec<f64>) -> PresentReport {
        let n = presents.len();
        PresentReport {
            presents,
            acquires: vec![0.0001; n],
            size: (1280, 720),
            mode: "Immediate".to_string(),
            refresh_hz: 60,
            fullscreen: false,
            skipped: 0,
            focus_lost: 0,
            tune: Tune::NotNeeded,
            far: "High".to_string(),
            rebuild: None,
            fly: false,
            moved: 0.0,
            ticks: 0,
        }
    }

    /// `n` presents 1 ms apart, starting at `from`.
    fn steady(from: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| from + i as f64 * 0.001).collect()
    }

    #[test]
    fn one_50_ms_gap_is_the_max_and_the_only_frame_over_two_intervals() {
        let mut p = steady(0.5, 19_500); // 0.5 s .. 20.0 s
        let last = *p.last().unwrap();
        p.extend((0..500).map(|i| last + 0.05 + i as f64 * 0.001));
        let r = report(p);
        let line = r.line();
        // The first clause is the whole run: p99 and p99.9 stay at the 1 ms
        // cadence, only the max sees the 50 ms gap.
        assert!(
            line.starts_with("PRESENT: p99 1.0 / p99.9 1.0 / max 50.0 ms, "),
            "{line}"
        );
        assert!(line.contains("over 33.3 ms 1 "), "{line}");
        assert_eq!(over(&gaps(&r.presents), 0.0333), 1);
    }

    #[test]
    fn a_gap_while_loading_is_not_counted_as_play() {
        // 50 ms gaps ending at 5 s and at 20 s, 1 ms apart otherwise.
        let mut p = steady(0.0, 4_950); // last at 4.949
        let mut t = 4.949 + 0.05;
        while t < 20.0 {
            p.push(t);
            t += 0.001;
        }
        p.push(t + 0.05 - 0.001);
        let line = report(p).line();
        assert!(line.contains("over 33.3 ms 2 "), "{line}");
        let after = line.split("after 10 s:").nth(1).unwrap();
        assert!(after.contains("over 33.3 ms 1 "), "{line}");
    }

    #[test]
    fn a_gap_ending_exactly_at_ten_seconds_is_play() {
        let line = report(vec![0.0, 9.9, 10.0, 10.5]).line();
        // Gaps end at 9.9, 10.0 and 10.5: two are play.
        let after = line.split("after 10 s:").nth(1).unwrap();
        assert!(after.contains("(2 frames)"), "{line}");
    }

    #[test]
    fn the_worst_gaps_are_longest_first_with_their_times() {
        let mut g: Vec<Gap> = (0..20)
            .map(|i| Gap {
                at: i as f64,
                took: 0.001,
            })
            .collect();
        for (k, at) in [3usize, 5, 7, 9, 11, 13, 15].iter().enumerate() {
            g[*at].took = 0.040 + 0.005 * k as f32;
        }
        let w = worst(&g, WORST_KEPT);
        assert_eq!(w.len(), 5);
        let ats: Vec<f64> = w.iter().map(|g| g.at).collect();
        assert_eq!(ats, vec![15.0, 13.0, 11.0, 9.0, 7.0]);
    }

    #[test]
    fn a_missed_refresh_is_over_25_ms_not_over_16_7() {
        let line = report(vec![0.0, 0.017, 0.043, 0.077]).line();
        assert!(line.contains("over 16.7 ms 3 over 33.3 ms 1 "), "{line}");
        assert!(line.contains("| over 25.0 ms 2 |"), "{line}");
    }

    #[test]
    fn the_rebuilds_frame_is_the_longest_in_the_second_after_it() {
        let g = [
            Gap {
                at: 29.9,
                took: 0.060,
            },
            Gap {
                at: 30.2,
                took: 0.045,
            },
            Gap {
                at: 31.5,
                took: 0.080,
            },
        ];
        assert_eq!(longest_after(&g, 30.0), Some(g[1]));
        assert_eq!(longest_after(&g, 40.0), None);
    }

    #[test]
    fn present_modes_parse_and_anything_else_does_not() {
        assert_eq!(parse_present("fifo"), Some(wgpu::PresentMode::Fifo));
        assert_eq!(parse_present("Mailbox"), Some(wgpu::PresentMode::Mailbox));
        assert_eq!(
            parse_present("IMMEDIATE"),
            Some(wgpu::PresentMode::Immediate)
        );
        assert_eq!(parse_present("vsync"), None);
        assert_eq!(parse_present(""), None);
    }

    #[test]
    fn the_owners_world_is_refused_and_a_test_world_is_not() {
        assert!(refuse_owners_world(&SavePaths::standard()).is_err());
        let test = SavePaths::for_world(std::env::temp_dir().join("x/world"));
        assert!(refuse_owners_world(&test).is_ok());
    }

    #[test]
    fn the_line_names_size_mode_and_tune() {
        let mut r = report(steady(0.0, 10));
        r.size = (2560, 1664);
        r.tune = Tune::Finished(12.3);
        let line = r.line();
        assert!(line.contains("2560x1664"), "{line}");
        assert!(line.contains("Immediate"), "{line}");
        assert!(line.contains("tune done at 12.3 s"), "{line}");
    }

    #[test]
    fn the_acquire_wait_is_reported_apart_from_the_gap() {
        let mut r = report(steady(0.0, 1000));
        r.acquires[500] = 0.030;
        let line = r.line();
        let wait = line.split("acquire wait").nth(1).unwrap();
        assert!(wait.contains("max 30.0 ms"), "{line}");
        assert!(line.contains("max 1.0 ms"), "{line}");
    }

    /// A gap exactly at the threshold is not over it.
    #[test]
    fn over_means_strictly_longer() {
        let g = [Gap {
            at: 1.0,
            took: 0.05,
        }];
        assert_eq!(over(&g, 0.05), 0);
        assert_eq!(over(&g, 0.049), 1);
    }

    #[test]
    fn the_line_says_whether_the_window_was_fullscreen() {
        let mut r = report(steady(0.0, 10));
        assert!(r.line().contains("windowed"), "{}", r.line());
        r.fullscreen = true;
        let line = r.line();
        assert!(
            line.contains("fullscreen") && !line.contains("windowed"),
            "{line}"
        );
    }

    #[test]
    fn a_presented_frame_counts_and_a_skipped_one_does_not() {
        let mut t = Trace::default();
        t.record(Some((1.0, 0.001)));
        t.record(None);
        t.record(Some((1.5, 0.002)));
        assert_eq!(t.presents, vec![1.0, 1.5]);
        assert_eq!(t.acquires, vec![0.001, 0.002]);
        assert_eq!(t.skipped, 1);
    }

    fn argv(words: &[&str]) -> Vec<String> {
        words.iter().map(|w| w.to_string()).collect()
    }

    #[test]
    fn window_without_bench_walk_or_in_the_owners_world_is_refused() {
        let test = SavePaths::for_world(std::env::temp_dir().join("x/world"));
        let own = SavePaths::standard();
        assert_eq!(check_window_args(&argv(&["c"]), &own), Ok(false));
        assert_eq!(
            check_window_args(&argv(&["c", "--bench", "walk"]), &own),
            Ok(false)
        );
        assert!(check_window_args(&argv(&["c", "--window"]), &test).is_err());
        assert!(check_window_args(&argv(&["c", "--bench", "gate", "--window"]), &test).is_err());
        assert!(check_window_args(&argv(&["c", "--bench", "walk", "--window"]), &own).is_err());
        assert_eq!(
            check_window_args(&argv(&["c", "--bench", "walk", "--window"]), &test),
            Ok(true)
        );
    }

    #[test]
    fn the_trace_has_a_header_and_one_row_per_gap() {
        let csv = trace_csv(&gaps(&[0.0, 0.5, 1.0]));
        assert_eq!(csv, "t_s,gap_ms\n0.5000,500.000\n1.0000,500.000\n");
    }
}

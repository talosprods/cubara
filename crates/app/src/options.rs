//! The options screen and the frame-rate guard: the owner's "per PC, choose
//! looks or performance, and never below the monitor's refresh rate"
//! (2026-09-26), apart from the window that shows it.
//!
//! What is drawn and what a key does are pure functions here; `main.rs` only
//! routes keys and spawns the benchmark.

use crate::far_streaming::FarQuality;
use crate::settings::{FrameCap, Settings};

/// What the options screen says: the far terrain's quality and who chose it,
/// the frame cap, the keys, and whether a benchmark is running.
pub fn options_text(settings: &Settings, target_fps: u32, benchmarking: bool) -> String {
    let who = if settings.by_hand {
        "chosen by you".to_string()
    } else {
        match &settings.tuned {
            Some(t) => format!("chosen by the benchmark for {} FPS", t.target_fps),
            None => "not benchmarked yet".to_string(),
        }
    };
    let bench = if benchmarking {
        format!("benchmarking this PC for {target_fps} FPS...")
    } else {
        format!("[B] benchmark this PC for {target_fps} FPS")
    };
    let cap = match settings.frame_cap {
        FrameCap::Vsync => format!("the monitor's {target_fps} FPS (vsync)"),
        FrameCap::Unlimited => "unlimited".to_string(),
    };
    format!(
        "-- OPTIONS --\n\
         [Esc] back\n\
         Far terrain: {} ({who})\n\
         [1] Off  [2] Low  [3] Medium  [4] High  -- looks or frame rate\n\
         [A] let the benchmark choose\n\
         Frame rate: {cap}\n\
         [V] vsync or unlimited\n\
         {bench}",
        settings.far.name()
    )
}

/// What a key on the options screen asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptionsKey {
    /// The player picks a quality; theirs from now on.
    Quality(FarQuality),
    /// Hand the choice back to the benchmark.
    Auto,
    /// Benchmark now.
    Benchmark,
    /// Cap the window at the monitor's refresh rate, or lift the cap.
    FrameCap,
}

/// The options screen's keys, by the character they type.
pub fn options_key(c: char) -> Option<OptionsKey> {
    Some(match c.to_ascii_lowercase() {
        '1' => OptionsKey::Quality(FarQuality::Off),
        '2' => OptionsKey::Quality(FarQuality::Low),
        '3' => OptionsKey::Quality(FarQuality::Medium),
        '4' => OptionsKey::Quality(FarQuality::High),
        'a' => OptionsKey::Auto,
        'b' => OptionsKey::Benchmark,
        'v' => OptionsKey::FrameCap,
        _ => return None,
    })
}

/// Apply `key` to `settings`: a picked quality is the player's, and stays;
/// `Auto` clears both that and the last benchmark, so one runs again;
/// `FrameCap` switches between vsync and unlimited.
pub fn apply(settings: &mut Settings, key: OptionsKey) {
    match key {
        OptionsKey::Quality(q) => {
            settings.far = q;
            settings.by_hand = true;
        }
        OptionsKey::Auto => {
            settings.by_hand = false;
            settings.tuned = None;
        }
        OptionsKey::Benchmark => {}
        OptionsKey::FrameCap => settings.frame_cap = settings.frame_cap.toggled(),
    }
}

/// The frame rate to hold: the monitor's refresh rate, rounded to the nearest
/// whole frame (a "59.94 Hz" monitor reports 59,940 mHz). 60 when the
/// monitor does not say.
pub fn target_fps(refresh_millihertz: Option<u32>) -> u32 {
    refresh_millihertz
        .map(|mhz| (mhz + 500) / 1000)
        .filter(|&fps| fps > 0)
        .unwrap_or(60)
}

/// Seconds after a start, a new world or a quality change during which missed
/// frames are not counted: meshing and streaming a world in costs frames that
/// say nothing about what playing it costs.
const SETTLE_SECONDS: f32 = 10.0;

/// How long a stretch of frames is judged over.
const WINDOW_SECONDS: f32 = 3.0;

/// A frame that took this many times its budget missed its refresh.
const MISSED: f32 = 1.5;

/// The share of a stretch's frames that may miss before the quality steps down.
const MISS_SHARE: f32 = 0.2;

/// Watches frame times for the game falling below the monitor's refresh rate
/// while playing -- the owner's "never below your refresh rate" -- and says
/// when to step the far terrain down one quality.
///
/// With the display in sync (vsync, the window's default since issue #425) a
/// frame takes a whole number of refreshes, so a frame that misses one takes
/// about twice its budget -- the tests feed it frame times of that shape
/// (`fifo_frames`). A stretch where a fifth
/// of frames miss is a machine that cannot hold the rate at this quality; an
/// occasional hitch is not, and is left alone. It only ever steps *down* --
/// finding headroom to step up is the benchmark's job, since a frame waiting
/// on the display says nothing about how much faster it could have been.
#[derive(Clone, Debug)]
pub struct FrameWatch {
    budget: f32,
    settling: f32,
    elapsed: f32,
    frames: u32,
    missed: u32,
    /// Whether the window can be seen. Hidden, its frames say nothing: under
    /// vsync an occluded or minimised window may wait far longer than a
    /// refresh for a drawable, or not draw at all.
    visible: bool,
}

/// The time one refresh lasts at `target_fps`, in seconds.
fn budget_for(target_fps: u32) -> f32 {
    1.0 / target_fps.max(1) as f32
}

impl FrameWatch {
    pub fn new(target_fps: u32) -> Self {
        Self {
            budget: budget_for(target_fps),
            settling: SETTLE_SECONDS,
            elapsed: 0.0,
            frames: 0,
            missed: 0,
            visible: true,
        }
    }

    /// The window is on a monitor refreshing at `target_fps` now. A change is
    /// a new budget and a new settling period: frames paced for the old
    /// monitor are not judged against this one. Without it a window started
    /// at 144 Hz and moved to a 60 Hz screen would see every vsynced frame as
    /// a miss, and step the quality down to Off (review of #426).
    pub fn set_target(&mut self, target_fps: u32) {
        let budget = budget_for(target_fps);
        if budget != self.budget {
            self.budget = budget;
            self.settle();
        }
    }

    /// The window became hidden (occluded, minimised) or visible again. While
    /// hidden no frame counts; either change discards the stretch being
    /// judged, so nothing from before is held against what comes after.
    pub fn set_visible(&mut self, visible: bool) {
        if visible != self.visible {
            self.visible = visible;
            self.discard();
        }
    }

    /// Forget the stretch being judged.
    fn discard(&mut self) {
        self.elapsed = 0.0;
        self.frames = 0;
        self.missed = 0;
    }

    /// Start counting again after a settling period: a new world, a new
    /// quality, a benchmark finishing.
    pub fn settle(&mut self) {
        self.settling = SETTLE_SECONDS;
        self.discard();
    }

    /// One frame took `dt` seconds. `true`: step the quality down now.
    pub fn frame(&mut self, dt: f32) -> bool {
        if !self.visible {
            return false;
        }
        if self.settling > 0.0 {
            self.settling -= dt;
            return false;
        }
        self.elapsed += dt;
        self.frames += 1;
        if dt > MISSED * self.budget {
            self.missed += 1;
        }
        if self.elapsed < WINDOW_SECONDS {
            return false;
        }
        let too_many = self.missed as f32 > MISS_SHARE * self.frames as f32;
        if too_many {
            self.settle();
        } else {
            self.discard();
        }
        too_many
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `frames` frames of `dt` each; whether any of them asked to step down.
    fn run(watch: &mut FrameWatch, frames: u32, dt: f32) -> bool {
        (0..frames).any(|_| watch.frame(dt))
    }

    #[test]
    fn a_steady_refresh_rate_never_steps_down() {
        let mut w = FrameWatch::new(60);
        assert!(!run(&mut w, 60 * 60, 1.0 / 60.0));
    }

    #[test]
    fn missing_every_other_refresh_steps_down_once_settled() {
        let mut w = FrameWatch::new(60);
        // During the first ten seconds nothing counts, however bad: 290
        // frames at 30 FPS is under ten seconds.
        assert!(!run(&mut w, 290, 2.0 / 60.0), "a loading world counted");
        // Then a stretch at 30 FPS on a 60 Hz monitor steps down...
        assert!(run(&mut w, 150, 2.0 / 60.0));
        // ...and settles again before judging the new quality.
        assert!(
            !run(&mut w, 290, 2.0 / 60.0),
            "stepped twice without settling"
        );
    }

    #[test]
    fn an_occasional_hitch_is_left_alone() {
        let mut w = FrameWatch::new(144);
        let budget = 1.0 / 144.0;
        let mut stepped = false;
        for i in 0..144 * 30 {
            let dt = if i % 20 == 0 { 3.0 * budget } else { budget };
            stepped |= w.frame(dt);
        }
        assert!(
            !stepped,
            "one frame in twenty missing is a hitch, not a machine too slow"
        );
    }

    /// Frame times as Fifo produces them: a frame is shown at the first
    /// refresh after its work is done, so it lasts a whole number of refresh
    /// intervals, never less than one. `jitter` is how far the display's own
    /// timing strays, alternating early and late, as a share of an interval.
    fn fifo_frames(refresh_hz: u32, work: impl Iterator<Item = f32>, jitter: f32) -> Vec<f32> {
        let interval = 1.0 / refresh_hz as f32;
        work.enumerate()
            .map(|(i, w)| {
                let refreshes = (w / interval).ceil().max(1.0);
                let stray = if i % 2 == 0 { jitter } else { -jitter };
                (refreshes + stray) * interval
            })
            .collect()
    }

    /// Whether `frames` ever ask to step down.
    fn steps_down(refresh_hz: u32, frames: &[f32]) -> bool {
        let mut w = FrameWatch::new(refresh_hz);
        frames.iter().any(|&dt| w.frame(dt))
    }

    /// Issue #425 decision 4. The model itself first, so the tests below
    /// cannot pass on frame times Fifo would never produce.
    #[test]
    fn fifo_frames_last_whole_refreshes() {
        let ms =
            |hz, work_ms: f32| fifo_frames(hz, std::iter::once(work_ms / 1000.0), 0.0)[0] * 1000.0;
        assert!((ms(60, 2.0) - 16.667).abs() < 0.01, "fast work waits");
        assert!((ms(60, 16.0) - 16.667).abs() < 0.01);
        assert!((ms(60, 18.0) - 33.333).abs() < 0.01, "a miss is two");
        assert!((ms(60, 40.0) - 50.0).abs() < 0.01);
        assert!((ms(144, 8.0) - 13.889).abs() < 0.01);
    }

    /// Work that fits the refresh, paced by Fifo, is never a miss -- however
    /// little or much of the budget it uses, and with the display's jitter.
    #[test]
    fn fifo_paced_frames_never_step_down() {
        for hz in [60, 120, 144] {
            let interval = 1.0 / hz as f32;
            for share in [0.1, 0.5, 0.95] {
                let frames = fifo_frames(hz, (0..hz * 60).map(|_| share * interval), 0.1);
                assert!(!steps_down(hz, &frames), "{hz} Hz, {share} of the budget");
            }
        }
    }

    /// Work a little over the budget misses every refresh under Fifo: each
    /// frame lasts two intervals, and that steps down once settled.
    #[test]
    fn fifo_frames_that_miss_the_refresh_step_down() {
        for hz in [60, 120, 144] {
            let interval = 1.0 / hz as f32;
            let frames = fifo_frames(hz, (0..hz * 30).map(|_| 1.1 * interval), 0.1);
            assert!(steps_down(hz, &frames), "{hz} Hz");
            // Not before the world has had its time to load.
            let loading = (SETTLE_SECONDS / (2.0 * interval)) as usize - 5;
            assert!(!steps_down(hz, &frames[..loading]), "{hz} Hz, loading");
        }
    }

    /// One Fifo miss in three is a machine too slow; one in twenty a hitch.
    #[test]
    fn fifo_misses_step_down_by_their_share() {
        let interval = 1.0 / 60.0;
        let every = |n: u32| {
            let work = (0..60 * 60).map(move |i| if i % n == 0 { 1.2 } else { 0.4 } * interval);
            fifo_frames(60, work, 0.1)
        };
        assert!(steps_down(60, &every(3)));
        assert!(!steps_down(60, &every(20)));
    }

    /// Review of #426: started on a 144 Hz monitor, moved to a 60 Hz one.
    /// Fifo now paces every frame at 16.7 ms, 2.4 times the old budget; told
    /// about the new monitor, the watch must not call any of them a miss.
    #[test]
    fn frames_paced_for_the_new_monitor_do_not_step_down() {
        let mut w = FrameWatch::new(144);
        let at_144 = fifo_frames(144, (0..144 * 20).map(|_| 0.002), 0.1);
        assert!(!at_144.iter().any(|&dt| w.frame(dt)), "settled at 144 Hz");
        w.set_target(60);
        let at_60 = fifo_frames(60, (0..60 * 60).map(|_| 0.002), 0.1);
        assert!(
            !at_60.iter().any(|&dt| w.frame(dt)),
            "stepped down at 60 Hz"
        );
        // And it still judges, against the new budget: missing 60 Hz counts.
        let missing = fifo_frames(60, (0..60 * 30).map(|_| 0.018), 0.1);
        assert!(
            missing.iter().any(|&dt| w.frame(dt)),
            "blind after the move"
        );
    }

    /// The other direction: 60 Hz frames are fine until the monitor is a
    /// 144 Hz one, and then they are misses -- after a settling period.
    #[test]
    fn a_new_budget_starts_a_new_settling_period() {
        let mut w = FrameWatch::new(60);
        let slow = 1.0 / 60.0;
        assert!(!run(&mut w, 60 * 20, slow));
        w.set_target(144);
        let settling = (SETTLE_SECONDS / slow) as u32 - 5;
        assert!(!run(&mut w, settling, slow), "judged while settling");
        assert!(run(&mut w, 60 * 5, slow), "60 FPS on a 144 Hz monitor");
        // The same target again is no change: no new settling period.
        let mut w = FrameWatch::new(60);
        assert!(!run(&mut w, 290, 2.0 / 60.0));
        w.set_target(60);
        assert!(run(&mut w, 150, 2.0 / 60.0), "settled again for no change");
    }

    /// A hidden window's frames -- a long wait for a drawable under vsync --
    /// are not misses, and are not held against the frames after it.
    #[test]
    fn frames_of_a_hidden_window_do_not_count() {
        let mut w = FrameWatch::new(60);
        assert!(!run(&mut w, 60 * 12, 1.0 / 60.0), "settled");
        w.set_visible(false);
        assert!(!run(&mut w, 200, 1.0), "a hidden window stepped down");
        w.set_visible(true);
        assert!(!run(&mut w, 60 * 30, 1.0 / 60.0));
        // Half a stretch of misses, hidden, then visible: the half is gone.
        let mut w = FrameWatch::new(60);
        assert!(!run(&mut w, 60 * 10 + 5, 1.0 / 60.0), "settled");
        assert!(!run(&mut w, 40, 2.0 / 60.0), "under three seconds");
        w.set_visible(false);
        w.set_visible(true);
        assert!(!run(&mut w, 60 * 4, 1.0 / 60.0), "the old misses counted");
        // Visible, it still judges.
        assert!(run(&mut w, 150, 2.0 / 60.0));
    }

    #[test]
    fn the_target_is_the_monitors_refresh_rate() {
        assert_eq!(target_fps(Some(59_940)), 60);
        assert_eq!(target_fps(Some(143_998)), 144);
        assert_eq!(target_fps(Some(240_000)), 240);
        assert_eq!(target_fps(None), 60);
        assert_eq!(target_fps(Some(0)), 60, "a monitor reporting nothing");
    }

    #[test]
    fn picking_a_quality_makes_it_the_players_and_auto_gives_it_back() {
        let mut s = Settings::default();
        apply(&mut s, OptionsKey::Quality(FarQuality::Low));
        assert_eq!(s.far, FarQuality::Low);
        assert!(
            s.by_hand && !s.needs_tuning(60, "gpu"),
            "the benchmark would overrule it"
        );
        apply(&mut s, OptionsKey::Auto);
        assert!(
            !s.by_hand && s.needs_tuning(60, "gpu"),
            "auto did not ask for a benchmark"
        );
    }

    #[test]
    fn the_keys_are_one_to_four_a_and_b() {
        assert_eq!(options_key('1'), Some(OptionsKey::Quality(FarQuality::Off)));
        assert_eq!(
            options_key('4'),
            Some(OptionsKey::Quality(FarQuality::High))
        );
        assert_eq!(options_key('A'), Some(OptionsKey::Auto));
        assert_eq!(options_key('b'), Some(OptionsKey::Benchmark));
        assert_eq!(options_key('5'), None);
    }

    /// Issue #425: one toggle, and it changes nothing else.
    #[test]
    fn v_switches_the_frame_cap_and_the_screen_says_which() {
        assert_eq!(options_key('v'), Some(OptionsKey::FrameCap));
        assert_eq!(options_key('V'), Some(OptionsKey::FrameCap));
        let mut s = Settings::default();
        let t = options_text(&s, 144, false);
        assert!(
            t.contains("Frame rate: the monitor's 144 FPS (vsync)"),
            "{t}"
        );
        apply(&mut s, OptionsKey::FrameCap);
        assert_eq!(s.frame_cap, FrameCap::Unlimited);
        assert!(!s.by_hand, "the far quality became the player's");
        let t = options_text(&s, 144, false);
        assert!(t.contains("Frame rate: unlimited"), "{t}");
        apply(&mut s, OptionsKey::FrameCap);
        assert_eq!(s.frame_cap, FrameCap::Vsync);
    }

    #[test]
    fn the_screen_says_who_chose_and_whether_a_benchmark_runs() {
        let mut s = Settings::default();
        assert!(options_text(&s, 144, false).contains("not benchmarked yet"));
        assert!(options_text(&s, 144, true).contains("benchmarking this PC for 144 FPS"));
        apply(&mut s, OptionsKey::Quality(FarQuality::Medium));
        let t = options_text(&s, 144, false);
        assert!(t.contains("Medium (chosen by you)"), "{t}");
    }
}

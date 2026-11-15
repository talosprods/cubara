//! Settings that belong to this PC rather than to a world: how finely the far
//! terrain is drawn, what that choice was tuned for, and whether the window
//! is capped at the monitor's refresh rate.
//!
//! The owner's call (2026-09-26): a benchmark decides, per PC, what keeps the
//! game at or above the monitor's refresh rate, and the player can choose
//! looks over frame rate or the other way round in the options. So the choice
//! lives here, next to the saves but not in any one of them.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::far_streaming::FarQuality;

/// How fast the game window draws (issue #425; the owner, 2026-11-14: no
/// 2,000 frames a second for a screen that shows 60).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrameCap {
    /// One frame per refresh of the monitor.
    #[default]
    Vsync,
    /// As fast as the machine goes.
    Unlimited,
}

impl FrameCap {
    /// The other one: the options screen's toggle.
    pub fn toggled(self) -> Self {
        match self {
            FrameCap::Vsync => FrameCap::Unlimited,
            FrameCap::Unlimited => FrameCap::Vsync,
        }
    }
}

/// Everything that is this machine's.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Settings {
    pub far: FarQuality,
    /// What `far` was chosen for by the benchmark, if it was.
    #[serde(default)]
    pub tuned: Option<Tuned>,
    /// The player picked `far` in the options. Their choice stands: no
    /// benchmark overrules it, and missed frames do not step it down.
    #[serde(default)]
    pub by_hand: bool,
    /// Whether the window is capped at the monitor's refresh rate. Absent in
    /// a file written before the setting existed: capped.
    #[serde(default)]
    pub frame_cap: FrameCap,
}

/// The conditions a benchmark chose the settings under. A different monitor
/// or a different GPU is a reason to benchmark again.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tuned {
    /// The frame rate the choice had to hold: the monitor's refresh rate.
    pub target_fps: u32,
    /// The GPU it was measured on.
    pub gpu: String,
}

impl Settings {
    /// Whether a benchmark should run before these settings can be trusted
    /// for a monitor at `target_fps` and a GPU called `gpu`: never
    /// benchmarked, or benchmarked for something else. A choice the player
    /// made by hand is theirs, and is not overruled.
    pub fn needs_tuning(&self, target_fps: u32, gpu: &str) -> bool {
        if self.by_hand {
            return false;
        }
        match &self.tuned {
            Some(t) => t.target_fps != target_fps || t.gpu != gpu,
            None => true,
        }
    }
}

/// What to do with a benchmark's answer, given the settings as they stand now:
/// `Some(answer)` to take it, `None` to ignore it because the player picked the
/// quality by hand ("keys 1-4 pick a quality, which is then the player's own,
/// so no benchmark or step-down overrules it", `PROPOSAL_FAR_VIEW.md` 6c).
/// Used by the game when the child ends and by the child before it writes.
///
/// A benchmark decides the far quality and nothing else: the frame cap stays
/// what it is in `current`, whatever the answer carries.
pub fn accept_tuned(current: &Settings, tuned: Settings) -> Option<Settings> {
    if current.by_hand {
        None
    } else {
        Some(Settings {
            frame_cap: current.frame_cap,
            ..tuned
        })
    }
}

/// The settings at `path`, or `None` when there are none yet or they do not
/// read -- a broken settings file is a reason to benchmark again, not to
/// refuse to start.
pub fn load(path: &Path) -> Option<Settings> {
    let text = std::fs::read_to_string(path).ok()?;
    match ron::from_str(&text) {
        Ok(s) => Some(s),
        Err(e) => {
            log::warn!("{} does not read ({e}); using defaults", path.display());
            None
        }
    }
}

pub fn save(path: &Path, settings: &Settings) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let text = ron::ser::to_string_pretty(settings, ron::ser::PrettyConfig::default())
        .map_err(std::io::Error::other)?;
    std::fs::write(path, text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tuned(fps: u32, gpu: &str) -> Settings {
        Settings {
            far: FarQuality::Medium,
            tuned: Some(Tuned {
                target_fps: fps,
                gpu: gpu.to_string(),
            }),
            by_hand: false,
            frame_cap: FrameCap::default(),
        }
    }

    #[test]
    fn a_fresh_pc_is_benchmarked() {
        assert!(Settings::default().needs_tuning(60, "Apple M3"));
    }

    #[test]
    fn a_benchmark_holds_until_the_monitor_or_gpu_changes() {
        let s = tuned(144, "Apple M3");
        assert!(!s.needs_tuning(144, "Apple M3"));
        assert!(s.needs_tuning(60, "Apple M3"), "a different monitor");
        assert!(s.needs_tuning(144, "RTX 4060"), "a different GPU");
    }

    #[test]
    fn a_choice_made_by_hand_is_not_overruled() {
        for far in FarQuality::BEST_FIRST {
            let s = Settings {
                far,
                tuned: None,
                by_hand: true,
                ..Settings::default()
            };
            assert!(!s.needs_tuning(60, "Apple M3"), "{far:?}");
        }
    }

    /// A benchmark that lands after the player picked by hand is ignored, and
    /// one that lands on an unpicked setting is taken.
    #[test]
    fn a_benchmark_result_never_overrules_a_hand_picked_quality() {
        let by_hand = Settings {
            far: FarQuality::High,
            tuned: None,
            by_hand: true,
            ..Settings::default()
        };
        assert_eq!(accept_tuned(&by_hand, tuned(60, "gpu")), None);
        let auto = Settings::default();
        assert_eq!(
            accept_tuned(&auto, tuned(60, "gpu")),
            Some(tuned(60, "gpu"))
        );
    }

    #[test]
    fn settings_survive_a_round_trip_and_a_broken_file_is_none() {
        let dir = std::env::temp_dir().join(format!(
            "cubara-settings-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = dir.join("settings.ron");
        assert_eq!(load(&path), None, "no file yet");
        let s = tuned(240, "RTX 4060");
        save(&path, &s).unwrap();
        assert_eq!(load(&path), Some(s));
        std::fs::write(&path, "not ron at all {").unwrap();
        assert_eq!(load(&path), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A fresh directory under the system's temp one, never `saves/`.
    fn temp_settings(name: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "cubara-settings-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = dir.join("settings.ron");
        (dir, path)
    }

    /// Issue #425: the window is capped unless the player says otherwise.
    #[test]
    fn the_frame_cap_defaults_to_vsync() {
        assert_eq!(Settings::default().frame_cap, FrameCap::Vsync);
        assert_eq!(FrameCap::Vsync.toggled(), FrameCap::Unlimited);
        assert_eq!(FrameCap::Unlimited.toggled(), FrameCap::Vsync);
    }

    /// Both values, because a cap that is not written reads back as the
    /// default and only the other one shows it.
    #[test]
    fn the_frame_cap_survives_a_round_trip() {
        let (dir, path) = temp_settings("cap");
        for frame_cap in [FrameCap::Unlimited, FrameCap::Vsync] {
            let s = Settings {
                frame_cap,
                ..tuned(60, "gpu")
            };
            save(&path, &s).unwrap();
            assert_eq!(load(&path), Some(s), "{frame_cap:?}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A settings file from before the cap existed still reads, as capped.
    #[test]
    fn settings_written_before_the_frame_cap_read_as_vsync() {
        let (dir, path) = temp_settings("old");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&path, "(far: Medium, tuned: None, by_hand: true)").unwrap();
        let s = load(&path).expect("an old file reads");
        assert_eq!(s.frame_cap, FrameCap::Vsync);
        assert!(s.by_hand);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The benchmark's answer is a far quality; a cap the player lifted
    /// while it ran is still lifted afterwards.
    #[test]
    fn a_benchmark_result_leaves_the_frame_cap_alone() {
        let current = Settings {
            frame_cap: FrameCap::Unlimited,
            ..Settings::default()
        };
        let taken = accept_tuned(&current, tuned(60, "gpu")).expect("not by hand");
        assert_eq!(taken.frame_cap, FrameCap::Unlimited);
        assert_eq!(taken.far, FarQuality::Medium);
    }
}

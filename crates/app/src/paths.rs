//! Where the game window keeps its world and its settings.
//!
//! One value, built once in `main` from the command line and passed to whatever
//! reads or writes either (`ARCHITECTURE.md` Rule 2: no ambient state). The
//! point is `--world <dir>`: the same flag, with the same meaning, as the
//! dedicated server's, so the real game can be started for a test against a
//! world of its own instead of the owner's `saves/world`.

use std::path::{Path, PathBuf};

/// The world directory and the settings file of one running game.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SavePaths {
    /// The world save, which `Game::save`/`load` use and New World moves aside
    /// from (the old one is kept as `world-<label>-<n>` *beside* this).
    pub world: PathBuf,
    /// This PC's settings (`settings.rs`).
    pub settings: PathBuf,
}

impl SavePaths {
    /// The player's own: `saves/world` and `saves/settings.ron` in the repo.
    pub fn standard() -> Self {
        Self::beside_root(&cubara_server::assets::repo_root())
    }

    fn beside_root(root: &Path) -> Self {
        Self {
            world: root.join("saves/world"),
            settings: root.join("saves/settings.ron"),
        }
    }

    /// A game on `world`, with its settings "beside the world, in the saves
    /// folder": `<world>/../settings.ron`.
    pub fn for_world(world: PathBuf) -> Self {
        let settings = world.join("..").join("settings.ron");
        Self { world, settings }
    }

    /// From the program's arguments: `--world <dir>` picks the world, nothing
    /// picks the standard paths. A `--world` without a value is an error, not a
    /// silent fall back to the owner's world.
    pub fn from_args(args: &[String]) -> Result<Self, String> {
        match args.iter().position(|a| a == "--world") {
            None => Ok(Self::standard()),
            Some(i) => match args.get(i + 1) {
                Some(dir) if !dir.starts_with("--") => Ok(Self::for_world(PathBuf::from(dir))),
                _ => Err("--world needs a directory, e.g. --world /tmp/cubara-test/world".into()),
            },
        }
    }

    /// What `cargo test` uses: a directory of its own. A test that called
    /// `save()` once wrote a fresh world over the owner's real save, player and
    /// inventory included; a test can no longer reach it.
    pub fn for_tests() -> Self {
        Self::for_world(std::env::temp_dir().join("cubara-test-saves-world"))
    }

    /// The arguments that make a child process (`--bench tune`, which writes
    /// the settings) look where this game looks. Empty for the standard paths,
    /// so a normal game's child is spawned exactly as before.
    pub fn child_args(&self) -> Vec<String> {
        if *self == Self::standard() {
            Vec::new()
        } else {
            vec!["--world".into(), self.world.to_string_lossy().into_owned()]
        }
    }
}

impl Default for SavePaths {
    fn default() -> Self {
        if cfg!(test) {
            Self::for_tests()
        } else {
            Self::standard()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    /// Without `--world` the paths are the ones the game always used.
    #[test]
    fn default_paths_are_the_saves_folder_in_the_repo() {
        let p = SavePaths::from_args(&args(&["cubara"])).unwrap();
        let root = cubara_server::assets::repo_root();
        assert_eq!(p.world, root.join("saves/world"));
        assert_eq!(p.settings, root.join("saves/settings.ron"));
    }

    #[test]
    fn world_flag_moves_the_world_and_the_settings_beside_it() {
        let p = SavePaths::from_args(&args(&["cubara", "--world", "X"])).unwrap();
        assert_eq!(p.world, PathBuf::from("X"));
        assert_eq!(
            p.settings,
            PathBuf::from("X").join("..").join("settings.ron")
        );
    }

    #[test]
    fn world_flag_without_a_value_is_refused() {
        assert!(SavePaths::from_args(&args(&["cubara", "--world"])).is_err());
        assert!(SavePaths::from_args(&args(&["cubara", "--world", "--bench"])).is_err());
    }

    /// The tuning child writes the settings, so it has to be told where.
    #[test]
    fn the_benchmark_child_is_told_the_world_dir() {
        let p = SavePaths::for_world(PathBuf::from("X"));
        assert_eq!(p.child_args(), args(&["--world", "X"]));
        assert!(SavePaths::standard().child_args().is_empty());
    }
}

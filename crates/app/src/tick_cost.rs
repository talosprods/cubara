//! `--bench edits`: what the tick costs against how much of the world has
//! ever been edited, with a mesh job's snapshot held across every tick
//! (issue #387, critical review F4).
//!
//! The question it answers: does the tick's CPU time depend on the number of
//! blocks that have ever been edited? It should not. Issue #387's design
//! decision 5 says the 10,000-edit fixture's tick must come within 5% of the
//! same fixture with no edits, both with a mesh job held across every tick.
//! Decision 1 says to measure before changing anything, so this bench prints
//! the count of whole-`World` clones the run paid for (`World::clone_generation`)
//! and what they weighed, alongside the tick time.
//!
//! No GPU and no window: a [`Server`] is ticked directly, the way the
//! dedicated server ticks it. Ticks are counted, not paced -- the `clones/s`
//! printed is per second *at the game's tick rate* (60 Hz), which is the rate
//! the game would pay it at, not how fast this machine ran the loop.
//!
//! **What "a mesh job in flight" means here.** A job holds a snapshot taken
//! when it was requested. `Arc::make_mut` copies only while that `Arc` is
//! shared, and the tick's copy is a *new* `Arc` nobody else holds -- so a
//! single job held forever costs exactly one clone, on the first tick. The
//! case that costs is the game's: a new job requested from the current world
//! every frame, which is what streaming does while the player moves and what
//! an edit does for its re-mesh. So this fixture requests one per tick.
//!
//! Before #387's redesign that snapshot was the `Arc<World>` itself, and this
//! fixture read 60 clones/s and 8.4 MB/s at 10,000 edits, the tick 7-8x the
//! empty world's. It is a `WorldSnapshot` now (`World::snapshot`), which
//! shares the edits by reference, so the tick has nothing to copy.

use std::sync::Arc;
use std::time::{Duration, Instant};

use cubara_server::Server;
use cubara_sim::{InputFrame, PlayerId, PlayerInputs};

/// Ticks per measured run: ten seconds of game time.
pub const DEFAULT_TICKS: usize = 600;

/// The edited fixture's edit count.
pub const EDITS: usize = 10_000;

/// The side, in blocks, of the square the edits fill.
const AREA: i32 = 64;

/// The game's tick rate, for turning clones per tick into clones per second.
const TICKS_PER_SECOND: f64 = 60.0;

/// Rounds each fixture is measured; the fastest tick time is the one reported,
/// since the slower rounds measure the machine's other work, not the tick's.
const ROUNDS: usize = 3;

/// One fixture's measurement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TickCost {
    pub edits: usize,
    pub ticks: usize,
    /// The sum of every `tick_sim_all` call's own time, best round.
    pub tick_time: Duration,
    /// Whole-`World` clones the run paid for.
    pub clones: u64,
    /// What one of those clones copies, in bytes (`World::clone_bytes`).
    pub bytes_each: usize,
}

impl TickCost {
    pub fn per_tick_us(&self) -> f64 {
        self.tick_time.as_secs_f64() * 1e6 / self.ticks.max(1) as f64
    }

    /// Clones per second at the game's tick rate.
    pub fn clones_per_second(&self) -> f64 {
        self.clones as f64 / self.ticks.max(1) as f64 * TICKS_PER_SECOND
    }

    pub fn bytes_per_second(&self) -> f64 {
        self.clones_per_second() * self.bytes_each as f64
    }

    fn term(&self) -> String {
        format!(
            "{} edits: {:.1} us/tick, {} clones in {} ticks = {:.0} clones/s, {:.0} bytes/s ({} bytes each)",
            self.edits,
            self.per_tick_us(),
            self.clones,
            self.ticks,
            self.clones_per_second(),
            self.bytes_per_second(),
            self.bytes_each,
        )
    }
}

/// The two fixtures decision 5 compares: no edits, and [`EDITS`] of them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Comparison {
    pub empty: TickCost,
    pub edited: TickCost,
}

impl Comparison {
    /// The edited fixture's tick time over the empty one's: decision 5 asks
    /// for at most 1.05.
    pub fn ratio(&self) -> f64 {
        self.edited.per_tick_us() / self.empty.per_tick_us().max(f64::EPSILON)
    }

    pub fn line(&self) -> String {
        format!(
            "EDITS: {} | {} | edited/empty tick time {:.2}x (decision 5: at most 1.05x)",
            self.empty.term(),
            self.edited.term(),
            self.ratio(),
        )
    }
}

/// Positions the edited fixture fills, in the order it fills them: a square
/// of [`AREA`] blocks a side, layer by layer, deep enough underground that
/// the player standing at the spawn never touches them.
fn edit_positions(count: usize) -> impl Iterator<Item = [i32; 3]> {
    (0..)
        .flat_map(move |layer| {
            (0..AREA).flat_map(move |z| (0..AREA).map(move |x| [x, -40 - layer, z]))
        })
        .take(count)
}

/// Measure one fixture: a server with `edits` blocks edited, a mesh job's
/// snapshot held across each of `ticks` ticks of `tick_sim_all`, timed.
pub fn measure(edits: usize, ticks: usize) -> TickCost {
    let mut best: Option<TickCost> = None;
    for _ in 0..ROUNDS {
        let mut server = Server::new();
        // Real assets, so the tick is the game's tick (the player stands on
        // real terrain); no save directory exists under this name, so nothing
        // is loaded and nothing is written.
        server.open(std::path::Path::new("cubara-nonexistent-tick-cost-fixture"));
        let stone = server.terrain().stone;
        {
            let world = Arc::make_mut(&mut server.world);
            for [x, y, z] in edit_positions(edits) {
                world.set_block(x, y, z, stone);
            }
        }
        let inputs = PlayerInputs::one(PlayerId::LOCAL, InputFrame::default());
        let clones_before = server.world.clone_generation();
        let mut tick_time = Duration::ZERO;
        for _ in 0..ticks {
            // One mesh job in flight across this tick: what `MeshPool::request`
            // hands a worker (`cubara_world::mesh`'s `Job`), taken from the
            // world as it is now. A new one every tick, because that is what
            // streaming does while the player moves -- each frame requests the
            // nodes that came into view -- and what an edit does for its
            // re-mesh. Held across the tick, dropped after it, as a job that
            // finishes within a frame is. Before #387 this was
            // `Arc::clone(&server.world)`, and the tick copied every edit.
            let held = Arc::new(server.world.snapshot());
            let started = Instant::now();
            server.tick_sim_all(&inputs);
            tick_time += started.elapsed();
            drop(held);
        }
        let cost = TickCost {
            edits,
            ticks,
            tick_time,
            clones: server.world.clone_generation() - clones_before,
            bytes_each: server.world.clone_bytes(),
        };
        if best.is_none_or(|b| cost.tick_time < b.tick_time) {
            best = Some(cost);
        }
    }
    best.expect("at least one round")
}

/// Both fixtures, `ticks` ticks each.
pub fn compare(ticks: usize) -> Comparison {
    Comparison {
        empty: measure(0, ticks),
        edited: measure(EDITS, ticks),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_edits_fill_the_area_and_no_more() {
        let positions: Vec<[i32; 3]> = edit_positions(EDITS).collect();
        assert_eq!(positions.len(), EDITS);
        assert!(positions.iter().all(|p| (0..AREA).contains(&p[0])));
        assert!(positions.iter().all(|p| (0..AREA).contains(&p[2])));
        let mut unique = positions.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), EDITS, "an edit position repeats");
    }

    #[test]
    fn the_line_prints_both_fixtures_and_their_ratio() {
        let cost = |edits, us: u64, clones| TickCost {
            edits,
            ticks: 600,
            tick_time: Duration::from_micros(us * 600),
            clones,
            bytes_each: edits * 14,
        };
        let c = Comparison {
            empty: cost(0, 100, 0),
            edited: cost(EDITS, 150, 600),
        };
        let line = c.line();
        assert!(
            line.contains("0 edits: 100.0 us/tick, 0 clones in 600 ticks = 0 clones/s, 0 bytes/s (0 bytes each)"),
            "{line}"
        );
        assert!(
            line.contains("10000 edits: 150.0 us/tick, 600 clones in 600 ticks = 60 clones/s, 8400000 bytes/s (140000 bytes each)"),
            "{line}"
        );
        assert!(line.contains("tick time 1.50x"), "{line}");
    }

    /// The fixture measures what it says: a held snapshot across a tick with
    /// edits in the world is a case the instrumentation must see. One tick,
    /// so the test costs nothing.
    #[test]
    fn a_measurement_counts_the_edits_it_made() {
        let cost = measure(50, 1);
        assert_eq!(cost.edits, 50);
        assert_eq!(cost.ticks, 1);
        assert_eq!(cost.bytes_each, 50 * 14);
    }
}

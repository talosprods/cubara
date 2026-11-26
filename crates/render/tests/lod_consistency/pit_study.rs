//! **How is a pit drawn from afar?** The study behind
//! `docs/PHASE3_ARCHITECTURE.md` §9.9, as a measurement.
//!
//! A pit is a cave open to the sky: a column whose surface block a cave has
//! carved away, its floor (`WorldGen::pit_floor`) the first ground under it.
//! Level 0 draws every one. Until §9.9 a coarse cell was cave only where all
//! of its ground is (§9.4's nesting rule), so the share of the ground a
//! level drew open shrank with the cell and a pit appeared out of flat grass
//! on the way in -- the lead's zoom pairs of 2026-11-25 rated it the worst
//! thing left at the 160, 288 and 512-block edges.
//!
//! Every candidate rule is worked out here from the world's own columns
//! (`surface_height`, `pit_floor`, `solid_on`), over real ground: around the
//! spawn, along the line the forest zoom pairs are shot on, and around the
//! owner's mountain. What a rule is judged by, per cell width:
//!
//! - **the depression it draws from above** against level 0's: the columns
//!   drawn at least one block (and at least five: deeper than the soil)
//!   under what the same level draws of the ground with no caves, their
//!   mean depth, and the volume;
//! - **presence**: the share of level 0's pits the level shows anything of
//!   -- what is left is a pit that appears out of nothing at a finer level;
//! - **the swap**: the coarser level's drawn top minus the finer one's on
//!   the columns a pit touches, in blocks and in pixels at the edge;
//! - **what an eye sees of it**: parallel rays from 15, 30 and 60 degrees
//!   above the horizon, four ways round -- how much further they travel
//!   than over the same level's ground with no caves, and how many of them
//!   do by a block or more;
//! - **the trees** that stand on a cell the rule sinks.
//!
//! ```text
//! cargo test -p cubara-render --test lod_consistency -- --ignored --nocapture pit_rule_study_report
//! ```

use super::far_study::{far_top, FarRule};
use super::harness::Setup;
use cubara_world::WorldGen;

/// Ground either side of a region that the window takes in, in blocks: the
/// cells a summit is judged among, at the widest cell studied.
const PAD: i32 = 64;

/// `cubara_world`'s own (private there): how far under a cell's highest
/// column a coarse node still carves a cave.
const DISTANT_CAVE_DEPTH: i32 = 24;

/// The cell widths studied: the voxel levels 1 to 3, and far quads of 16
/// and 32 blocks (sampled on a 4 x 4 grid, as the far terrain samples them).
const WIDTHS: [i32; 5] = [2, 4, 8, 16, 32];

/// The distance each width is first drawn at, in blocks: the ring edges of
/// `DEFAULT_RING_SCHEDULE`, then where far quads of 16 pixels at 1080 rows
/// are 16 and 32 blocks wide (`far_study.rs`'s edges).
const EDGE_OF: [i32; 5] = [160, 288, 512, 1032, 2064];

/// Pixels a one-block step covers at `distance` blocks on 1080 rows at the
/// renderer's 60 degrees of vertical field of view.
fn px_per_block(distance: i32) -> f64 {
    let pixel_angle = 60f64.to_radians() / 1080.0;
    1.0 / distance as f64 / pixel_angle
}

/// Where a cell's top stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Top {
    /// `TopRule` over the surfaces: a pit does not move a top (main).
    Surface,
    /// `TopRule` over **what the sky sees of each column**: its surface
    /// block, or the floor of the pit a cave has opened there.
    Sky,
    /// `TopRule` over the surfaces, lowered by the cell's mean pit depth
    /// (its share of pit columns times their depth), rounded apart.
    SurfaceSunk,
    /// A cell at least half of whose columns are pit stands at the mean of
    /// those columns' floors; any other at `TopRule` over the surfaces.
    MajorityFloor,
    /// [`Top::Sky`], and a cell that holds a pit's bottom -- the first
    /// lowest column of the nine cells around is its own -- at the lowest
    /// mean of its four half-width cells: the summit rule, mirrored.
    SkyBottoms,
}

/// Where a cell's row is cave.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Carve {
    /// Where every block of ground in it is cave at level 0 (§9.4).
    All,
    /// Where at least this share of its ground is.
    Share(i64, i64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PitRule {
    pub name: &'static str,
    pub top: Top,
    pub carve: Carve,
}

/// The rule built: `WorldGen::cell_top` over `WorldGen::column_top`.
pub const CHOSEN: PitRule = PitRule {
    name: "the mean of what the sky sees (chosen)",
    top: Top::Sky,
    carve: Carve::All,
};

pub const RULES: [PitRule; 7] = [
    PitRule {
        name: "the mean of the surfaces; cave where all the ground is (main)",
        top: Top::Surface,
        carve: Carve::All,
    },
    CHOSEN,
    PitRule {
        name: "the surfaces' top, sunk by share x depth",
        top: Top::SurfaceSunk,
        carve: Carve::All,
    },
    PitRule {
        name: "cave where at least half the ground is",
        top: Top::Surface,
        carve: Carve::Share(1, 2),
    },
    PitRule {
        name: "cave where at least a quarter of the ground is",
        top: Top::Surface,
        carve: Carve::Share(1, 4),
    },
    PitRule {
        name: "a cell at least half pit stands at its floors' mean",
        top: Top::MajorityFloor,
        carve: Carve::All,
    },
    PitRule {
        name: "what the sky sees, a pit's bottom at its lowest half",
        top: Top::SkyBottoms,
        carve: Carve::All,
    },
];

fn div_round_half_even(a: i64, b: i64) -> i64 {
    let (q, r) = (a.div_euclid(b), a.rem_euclid(b));
    match (2 * r).cmp(&b) {
        std::cmp::Ordering::Less => q,
        std::cmp::Ordering::Greater => q + 1,
        std::cmp::Ordering::Equal => q + (q & 1),
    }
}

/// One level-0 column: where it stands, its surface block and the highest
/// block of ground the sky sees in it.
#[derive(Clone, Copy)]
struct Column {
    x: i32,
    z: i32,
    surface: i32,
    sky: i32,
}

/// The ground of a square region and [`PAD`] blocks around it: every
/// column's surface and what the sky sees of it.
pub struct Ground {
    x0: i32,
    z0: i32,
    /// The region's side, in blocks: a multiple of the widest cell.
    pub side: i32,
    surface: Vec<i32>,
    sky: Vec<i32>,
}

impl Ground {
    pub fn over(gen: &WorldGen, x0: i32, z0: i32, side: i32) -> Self {
        let full = side + 2 * PAD;
        let (mut surface, mut sky) = (Vec::new(), Vec::new());
        for j in 0..full {
            for i in 0..full {
                let (x, z) = (x0 - PAD + i, z0 - PAD + j);
                let s = gen.surface_height(x, z);
                surface.push(s);
                sky.push(gen.pit_floor(x, z, s).unwrap_or(s));
            }
        }
        Self {
            x0,
            z0,
            side,
            surface,
            sky,
        }
    }

    fn index(&self, x: i32, z: i32) -> usize {
        let full = self.side + 2 * PAD;
        let (i, j) = (x - self.x0 + PAD, z - self.z0 + PAD);
        assert!(
            (0..full).contains(&i) && (0..full).contains(&j),
            "a column outside the window"
        );
        (j * full + i) as usize
    }

    fn column(&self, x: i32, z: i32) -> Column {
        let k = self.index(x, z);
        Column {
            x,
            z,
            surface: self.surface[k],
            sky: self.sky[k],
        }
    }

    /// The region's columns of one field, row by row (`z` outer).
    fn region(&self, field: &[i32]) -> Vec<i32> {
        (0..self.side * self.side)
            .map(|k| field[self.index(self.x0 + k % self.side, self.z0 + k / self.side)])
            .collect()
    }

    /// What level 0 draws from above: the sky's view of every column.
    pub fn level_0(&self) -> Vec<i32> {
        self.region(&self.sky)
    }

    /// The ground with no caves: every column's surface.
    pub fn level_0_whole(&self) -> Vec<i32> {
        self.region(&self.surface)
    }

    /// The columns a cell's top is taken over: all of them up to 8 blocks,
    /// the far terrain's 4 x 4 grid, centred, above.
    fn cell_columns(&self, x0: i32, z0: i32, w: i32) -> Vec<Column> {
        let count = if w <= 8 { w } else { 4 };
        let step = w / count;
        let lead = (step - 1) / 2;
        (0..count * count)
            .map(|k| self.column(x0 + lead + k % count * step, z0 + lead + k / count * step))
            .collect()
    }

    /// `TopRule` (the game's, `far_study::far_top`'s copy of it) over the
    /// surfaces or over what the sky sees.
    fn top_rule(&self, sky: bool, x0: i32, z0: i32, w: i32) -> i32 {
        let field = if sky { &self.sky } else { &self.surface };
        far_top(&|x, z| field[self.index(x, z)], FarRule::TopRule, x0, z0, w)
    }

    /// The top block of the cell at `(x0, z0)` under `top`.
    fn cell_top(&self, top: Top, x0: i32, z0: i32, w: i32) -> i32 {
        let columns = self.cell_columns(x0, z0, w);
        let n = columns.len() as i64;
        match top {
            Top::Surface => self.top_rule(false, x0, z0, w),
            Top::Sky => self.top_rule(true, x0, z0, w),
            Top::SurfaceSunk => {
                let depth: i64 = columns.iter().map(|c| (c.surface - c.sky) as i64).sum();
                self.top_rule(false, x0, z0, w) - div_round_half_even(depth, n) as i32
            }
            Top::MajorityFloor => {
                let pits: Vec<i64> = columns
                    .iter()
                    .filter(|c| c.sky < c.surface)
                    .map(|c| c.sky as i64)
                    .collect();
                if 2 * pits.len() as i64 >= n {
                    div_round_half_even(pits.iter().sum(), pits.len() as i64) as i32
                } else {
                    self.top_rule(false, x0, z0, w)
                }
            }
            Top::SkyBottoms => {
                let sky: i64 = columns.iter().map(|c| c.sky as i64).sum();
                let top = self.top_rule(true, x0, z0, w);
                if top != div_round_half_even(sky, n) as i32 {
                    // A summit already.
                    return top;
                }
                // The summit rule on the ground turned over.
                -far_top(
                    &|x, z| -self.sky[self.index(x, z)],
                    FarRule::TopRule,
                    x0,
                    z0,
                    w,
                )
            }
        }
    }

    /// What a cell whose top block is `top` shows from above once its rows
    /// of `v` blocks are carved by `carve`: the highest block of the first
    /// row down that is not cave.
    fn carved_top(gen: &WorldGen, columns: &[Column], top: i32, v: i32, carve: Carve) -> i32 {
        let peak = columns
            .iter()
            .map(|c| c.surface)
            .max()
            .expect("a cell covers a column");
        let mut y = top.div_euclid(v) * v;
        loop {
            let last = (y + v - 1).min(top);
            if y <= peak - DISTANT_CAVE_DEPTH {
                return last;
            }
            let (mut ground, mut cave) = (0i64, 0i64);
            'count: for c in columns {
                for row in y..=last.min(c.surface) {
                    ground += 1;
                    if !gen.solid_on(c.x, row, c.z, c.surface) {
                        cave += 1;
                    } else if carve == Carve::All {
                        break 'count;
                    }
                }
            }
            let is_cave = ground > 0
                && match carve {
                    Carve::All => cave == ground,
                    Carve::Share(num, den) => cave * den >= ground * num,
                };
            if !is_cave {
                return last;
            }
            y -= v;
        }
    }

    /// Every region column's top as `rule` draws it from above at cell
    /// width `w`, row by row. Level 3 (`w = 8`) is carved in rows of four
    /// blocks and cut off at its top, as the game stores and draws it. The
    /// far terrain (`w > 8`) draws no caves: only a rule that carves by
    /// share is taken to sink a far quad that far.
    pub fn drawn(&self, gen: &WorldGen, rule: PitRule, w: i32) -> Vec<i32> {
        let mut out = vec![0; (self.side * self.side) as usize];
        let v = if w == 8 { 4 } else { 1 };
        for cj in 0..self.side / w {
            for ci in 0..self.side / w {
                let (x0, z0) = (self.x0 + ci * w, self.z0 + cj * w);
                let mut top = self.cell_top(rule.top, x0, z0, w);
                if w <= 8 || rule.carve != Carve::All {
                    top = Self::carved_top(gen, &self.cell_columns(x0, z0, w), top, v, rule.carve);
                }
                for b in 0..w {
                    for a in 0..w {
                        out[((cj * w + b) * self.side + ci * w + a) as usize] = top;
                    }
                }
            }
        }
        out
    }

    /// The ground with no caves as a level of cell width `w` draws it:
    /// `TopRule` over the surfaces, nothing carved.
    pub fn whole(&self, w: i32) -> Vec<i32> {
        self.drawn_tops(Top::Surface, w)
    }

    /// Every region column's cell top under `top` at cell width `w`, with
    /// nothing carved, row by row.
    pub fn drawn_tops(&self, top: Top, w: i32) -> Vec<i32> {
        let mut out = vec![0; (self.side * self.side) as usize];
        for cj in 0..self.side / w {
            for ci in 0..self.side / w {
                let top = self.cell_top(top, self.x0 + ci * w, self.z0 + cj * w, w);
                for b in 0..w {
                    for a in 0..w {
                        out[((cj * w + b) * self.side + ci * w + a) as usize] = top;
                    }
                }
            }
        }
        out
    }

    /// Level 0's pits: the groups of pit columns that touch along an edge,
    /// each a list of region columns.
    pub fn pits(&self) -> Vec<Vec<usize>> {
        let (whole, sky) = (self.level_0_whole(), self.level_0());
        let n = self.side;
        let mut seen = vec![false; whole.len()];
        let mut pits = Vec::new();
        for start in 0..whole.len() {
            if seen[start] || sky[start] >= whole[start] {
                continue;
            }
            let mut pit = vec![start];
            seen[start] = true;
            let mut next = 0;
            while next < pit.len() {
                let (i, j) = (pit[next] as i32 % n, pit[next] as i32 / n);
                next += 1;
                for (di, dj) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    let (a, b) = (i + di, j + dj);
                    if !(0..n).contains(&a) || !(0..n).contains(&b) {
                        continue;
                    }
                    let k = (b * n + a) as usize;
                    if !seen[k] && sky[k] < whole[k] {
                        seen[k] = true;
                        pit.push(k);
                    }
                }
            }
            pits.push(pit);
        }
        pits
    }
}

/// What a level draws of the pits, from above: against the same level's
/// ground with no caves.
#[derive(Clone, Copy, Default, Debug)]
pub struct Depression {
    /// Columns drawn at least one block under the whole ground.
    pub area: f64,
    /// Columns drawn at least five under it: deeper than the soil.
    pub deep: f64,
    /// Blocks of air between the two.
    pub volume: f64,
    /// Columns drawn *above* the whole ground (a rule that lifts).
    pub raised: f64,
}

impl Depression {
    pub fn of(whole: &[i32], drawn: &[i32]) -> Self {
        let mut out = Self::default();
        for (&w, &d) in whole.iter().zip(drawn) {
            let depth = w - d;
            out.area += (depth >= 1) as u8 as f64;
            out.deep += (depth >= 5) as u8 as f64;
            out.volume += depth.max(0) as f64;
            out.raised += (depth < 0) as u8 as f64;
        }
        out
    }

    fn add(&mut self, other: &Self) {
        self.area += other.area;
        self.deep += other.deep;
        self.volume += other.volume;
        self.raised += other.raised;
    }
}

/// How much further parallel rays travel into `with` than into `without`
/// -- two fields of top blocks over the same square, `side` columns a side
/// -- coming down at `tan` (the tangent of their angle above the horizon)
/// from each of the four axis directions, half a block apart: the sum of
/// the extra length in block-widths of ray (a volume seen), and the rays
/// that travel at least a block further (an area seen, in square blocks of
/// the picture).
pub fn seen(with: &[i32], without: &[i32], side: i32, tan: f64) -> (f64, f64) {
    const SPACING: f64 = 0.5;
    let per_ray = (1.0 + tan * tan).sqrt();
    let dc = SPACING * per_ray;
    let n = side as usize;
    let (mut volume, mut area) = (0.0, 0.0);
    // The prefix horizon of a profile of top faces: a ray of intercept `c`
    // (its height where it enters the row) passes column `k` when
    // `c > h[k] + (k + 1) * tan`.
    let horizon = |h: &[f64]| -> Vec<f64> {
        let mut high = f64::NEG_INFINITY;
        h.iter()
            .enumerate()
            .map(|(k, &top)| {
                high = high.max(top + (k + 1) as f64 * tan);
                high
            })
            .collect()
    };
    let hit = |h: &[f64], k: usize, c: f64| (k as f64).max((c - h[k]) / tan);
    let (mut a, mut b) = (vec![0.0; n], vec![0.0; n]);
    for way in 0..4 {
        for row in 0..n {
            for k in 0..n {
                let along = if way % 2 == 0 { k } else { n - 1 - k };
                let at = if way < 2 {
                    row * n + along
                } else {
                    along * n + row
                };
                // The top face is one above the top block.
                a[k] = with[at] as f64 + 1.0;
                b[k] = without[at] as f64 + 1.0;
            }
            let (ha, hb) = (horizon(&a), horizon(&b));
            let lowest = b.iter().copied().fold(f64::INFINITY, f64::min);
            let (mut ia, mut ib) = (0, 0);
            let mut c = lowest;
            loop {
                while ia < n && ha[ia] < c {
                    ia += 1;
                }
                while ib < n && hb[ib] < c {
                    ib += 1;
                }
                if ia == n || ib == n {
                    break;
                }
                let further = (hit(&a, ia, c) - hit(&b, ib, c)) * per_ray;
                volume += further * SPACING;
                area += (further >= 1.0) as u8 as f64 * SPACING;
                c += dc;
            }
        }
    }
    (volume, area)
}

/// Every region column's top block as **the game** draws it at cell width
/// `w`: `WorldGen::cell_top`, and above 8 blocks the far terrain's
/// `cell_top_sampled` over its 4 x 4 grid. Row by row.
fn game_tops(gen: &WorldGen, ground: &Ground, w: i32) -> Vec<i32> {
    let side = ground.side;
    let mut out = vec![0; (side * side) as usize];
    for cj in 0..side / w {
        for ci in 0..side / w {
            let (x0, z0) = (ground.x0 + ci * w, ground.z0 + cj * w);
            let top = if w <= 8 {
                gen.cell_top(x0, z0, w)
            } else {
                gen.cell_top_sampled(x0, z0, w, 4)
            };
            for b in 0..w {
                for a in 0..w {
                    out[((cj * w + b) * side + ci * w + a) as usize] = top;
                }
            }
        }
    }
    out
}

/// The study's copy of the chosen rule is the game's: every cell's top over
/// a pitted square near the spawn, at every width studied, and level 0's
/// own columns. Cells a pit sinks must be among them at every width, or the
/// two could agree on the surfaces alone.
#[test]
fn the_chosen_pit_rule_is_the_games() {
    let gen = WorldGen::new(cubara_world::World::new().seed());
    let ground = Ground::over(&gen, 0, 0, 256);
    for (k, &top) in ground.level_0().iter().enumerate() {
        let (x, z) = (k as i32 % 256, k as i32 / 256);
        assert_eq!(gen.column_top(x, z), top, "({x}, {z})");
        assert_eq!(gen.cell_top(x, z, 1), top, "({x}, {z})");
    }
    for w in WIDTHS {
        let (study, game) = (
            ground.drawn_tops(CHOSEN.top, w),
            game_tops(&gen, &ground, w),
        );
        assert_eq!(study, game, "width {w}");
        let sunk = study
            .iter()
            .zip(ground.whole(w))
            .filter(|&(&top, whole)| top < whole)
            .count();
        assert!(
            sunk >= 16 * (w * w) as usize,
            "width {w}: only {sunk} columns stand on a cell a pit sinks"
        );
    }
}

/// **Every level draws level 0's pits** (`PHASE3_ARCHITECTURE.md` §9.9),
/// by the game's own tops over the square around the spawn:
///
/// - the air a level draws over the pits is level 0's, within
///   [`VOLUME`] of it at the voxel levels' widths and within [`FAR_VOLUME`]
///   at far quads of 16 and 32, which are sampled;
/// - every pit of 64 columns or more shows at every voxel level, and nine
///   in ten at far quads of 16: none appears out of flat ground at the
///   160, 288 or 512-block edge;
/// - a swap has no direction: over every column the coarser top minus the
///   finer one comes to nothing.
///
/// The rule before -- the mean of the surfaces, a cell cave only where all
/// of its ground is -- is measured beside it and is red on the first at
/// every width, and on the second from cells of 8 on: it drew a fifth of
/// the air at cells of 8 and showed half the large pits.
#[test]
fn every_level_draws_level_0s_pits() {
    /// How far the air a voxel level draws over the pits may be from level
    /// 0's. **Chosen, not derived**: measured x0.994, x0.993 and x0.988 at
    /// cells of 2, 4 and 8 here (rounding, and the summits' lift), and up
    /// to x1.07 on the owner's mountain in the study.
    const VOLUME: f64 = 0.06;
    /// The same for a far quad of 16 or 32 blocks, sampled on 4 x 4
    /// columns. **Chosen, not derived**: measured x0.991 and x0.914.
    const FAR_VOLUME: f64 = 0.15;
    /// The most the mean of a swap's change over every column may be, in
    /// blocks, at the voxel widths and at the sampled far ones. **Chosen,
    /// not derived**: measured 0.02 at most, and 0.02 and 0.17 at far
    /// quads of 16 and 32 (a tenth of a pixel where those are drawn).
    const LEAN: (f64, f64) = (0.04, 0.25);
    let gen = WorldGen::new(cubara_world::World::new().seed());
    let ground = Ground::over(&gen, -256, -256, 512);
    let (sky, surface) = (ground.level_0(), ground.level_0_whole());
    let level0 = Depression::of(&surface, &sky);
    let deepest = surface.iter().zip(&sky).map(|(s, t)| s - t).max();
    println!("the deepest pit column: {deepest:?} blocks");
    let pits = ground.pits();
    let large: Vec<&Vec<usize>> = pits.iter().filter(|pit| pit.len() >= 64).collect();
    assert!(large.len() >= 40, "only {} large pits", large.len());
    let mut finer = sky;
    for w in WIDTHS {
        let whole = ground.whole(w);
        let drawn = game_tops(&gen, &ground, w);
        let shows = |tops: &[i32]| {
            large
                .iter()
                .filter(|pit| pit.iter().any(|&c| whole[c] - tops[c] >= 1))
                .count()
        };
        let (volume, shown) = (
            Depression::of(&whole, &drawn).volume / level0.volume,
            shows(&drawn),
        );
        let before = ground.drawn(&gen, RULES[0], w);
        let (volume_before, shown_before) = (
            Depression::of(&whole, &before).volume / level0.volume,
            shows(&before),
        );
        let lean = drawn
            .iter()
            .zip(&finer)
            .map(|(c, f)| (c - f) as f64)
            .sum::<f64>()
            / drawn.len() as f64;
        println!(
            "width {w}: the air over the pits x{volume:.3} of level 0's (before: x{volume_before:.3}), {shown} of {} large pits shown (before: {shown_before}), a swap leans {lean:+.4} blocks",
            large.len()
        );
        let (allowed, least) = if w <= 8 {
            (VOLUME, large.len())
        } else if w == 16 {
            (FAR_VOLUME, large.len() * 9 / 10)
        } else {
            (FAR_VOLUME, large.len() * 3 / 4)
        };
        assert!(
            (volume - 1.0).abs() <= allowed,
            "width {w}: the air over the pits is x{volume:.3} of level 0's, allowed {allowed}"
        );
        assert!(
            shown >= least,
            "width {w}: {shown} of {} large pits are drawn, {least} wanted",
            large.len()
        );
        assert!(
            lean.abs() <= if w <= 8 { LEAN.0 } else { LEAN.1 },
            "width {w}: a swap leans {lean:+.3} blocks over every column"
        );
        assert!(
            (volume_before - 1.0).abs() > allowed || shown_before < least,
            "width {w}: the rule before passes too (x{volume_before:.3}, {shown_before} shown): this test tells nothing apart"
        );
        finer = drawn;
    }
}

/// **The nodes the game generates draw the pits where the rule says**: at
/// every voxel level, over a pitted square near the spawn, each cell's
/// highest drawn block is its `cell_top` -- or lower, and then only by the
/// nesting rule of §9.4: every block of ground between the two is cave at
/// level 0. Level 0's own highest block is its column's top exactly. Cells
/// that a pit sinks must be among them, and so must cells the nesting rule
/// carves further.
///
/// Goes red when a node's ground is the surfaces' again, when a row is
/// called cave on fewer than all its blocks, or when a top is cut off at
/// the wrong block.
#[test]
fn the_nodes_draw_the_pits_where_the_rule_says() {
    use cubara_world::node::NodeKey;
    const SIDE: i32 = 128;
    let setup = Setup::new();
    let gen = &setup.gen;
    let ground = Ground::over(gen, 0, 0, SIDE);
    let (sky, surface) = (ground.level_0(), ground.level_0_whole());
    let lowest = *sky.iter().min().expect("columns");
    let highest = *surface.iter().max().expect("columns");
    for level in 0..=3u32 {
        let w = 1i32 << level;
        let size = 16 * w;
        let cells = SIDE / w;
        // Each cell's highest drawn block, from the nodes themselves.
        let mut drawn: Vec<Option<i32>> = vec![None; (cells * cells) as usize];
        for nz in 0..SIDE / size {
            for nx in 0..SIDE / size {
                for ny in (lowest - 1).div_euclid(size)..=highest.div_euclid(size) {
                    let node = NodeKey::new(level, [nx, ny, nz]);
                    let Some(chunk) = setup.world.node_at(node, setup.blocks) else {
                        continue;
                    };
                    let origin = node.world_origin();
                    let (dims, rows) = (chunk.mesh_dims(), chunk.mesh_scale()[1]);
                    assert_eq!(rows, 1, "level {level} is drawn in block rows");
                    for lz in 0..dims[2] {
                        for lx in 0..dims[0] {
                            let top = (0..dims[1])
                                .rev()
                                .find(|&ly| setup.registry.is_solid(chunk.drawn(lx, ly, lz)))
                                .map(|ly| origin[1] + ly as i32);
                            let c = ((nz * 16 + lz as i32) * cells + nx * 16 + lx as i32) as usize;
                            drawn[c] = drawn[c].max(top);
                        }
                    }
                }
            }
        }
        let (mut sunk, mut carved) = (0, 0);
        for cj in 0..cells {
            for ci in 0..cells {
                let (x0, z0) = (ci * w, cj * w);
                let top = gen.cell_top(x0, z0, w);
                let drew = drawn[(cj * cells + ci) as usize]
                    .unwrap_or_else(|| panic!("level {level}: nothing drawn at ({x0}, {z0})"));
                assert!(
                    drew <= top,
                    "level {level}: the cell at ({x0}, {z0}) is drawn to {drew}, above its top {top}"
                );
                sunk += (top < ground.top_rule(false, x0, z0, w)) as usize;
                if drew == top {
                    continue;
                }
                assert!(
                    level > 0,
                    "level 0 at ({x0}, {z0}): {drew}, its top is {top}"
                );
                carved += 1;
                for k in 0..w * w {
                    let column = gen.column(x0 + k % w, z0 + k / w);
                    for y in drew + 1..=top.min(column.top) {
                        assert!(
                            !gen.solid_on(x0 + k % w, y, z0 + k / w, column.surface),
                            "level {level}: the cell at ({x0}, {z0}) is open down to {drew} over rock at ({}, {y}, {})",
                            x0 + k % w,
                            z0 + k / w
                        );
                    }
                }
            }
        }
        println!("level {level}: {sunk} cells a pit sinks, {carved} carved further");
        assert!(
            sunk >= 16,
            "level {level}: only {sunk} cells a pit sinks: nothing was tested"
        );
    }
}

/// One class of ground studied: a name and its regions `(x0, z0, side)`.
struct Class {
    name: &'static str,
    regions: Vec<(i32, i32, i32)>,
}

/// The ground the study runs over. **Chosen, not derived**: where the
/// player starts, the line the forest zoom pairs are shot along (eyes at
/// `(-448, -1344) + 0.7071 * D` for `D` up to 3,000, looking back at the
/// wood), and the owner's mountain (`far_study.rs`'s `ZOOM_PEAK`).
fn classes() -> Vec<Class> {
    let square = |cx: i32, cz: i32, side: i32| {
        (
            (cx - side / 2).div_euclid(64) * 64,
            (cz - side / 2).div_euclid(64) * 64,
            side,
        )
    };
    vec![
        Class {
            name: "around the spawn (1,024 blocks square)",
            regions: vec![square(0, 0, 1024)],
        },
        Class {
            name: "along the forest line (seven squares of 512)",
            regions: (0..7)
                .map(|k| {
                    // 500 blocks along the line (0.7071, 0.7071), in whole
                    // blocks.
                    let d = 7071 * 500 * k / 10000;
                    square(-448 + d, -1344 + d, 512)
                })
                .collect(),
        },
        Class {
            name: "the owner's mountain (512 blocks square)",
            regions: vec![square(-3072, -3232, 512)],
        },
    ]
}

/// The mean, the 95th percentile and the largest of `values`' magnitudes,
/// and the signed mean.
fn step_stats(values: &mut [f64]) -> (f64, f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let signed = values.iter().sum::<f64>() / values.len() as f64;
    for v in values.iter_mut() {
        *v = v.abs();
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let p95 = values[(values.len() * 95).div_ceil(100).max(1) - 1];
    (mean, p95, values[values.len() - 1], signed)
}

/// Everything measured of one rule over one class of ground.
#[derive(Default, Clone)]
struct Measured {
    /// Per width.
    depression: [Depression; 5],
    /// Per width: pits shown, of small (under 16 columns), middling and
    /// large (64 and over) ones.
    shown: [[f64; 3]; 5],
    /// Per width: the coarser level's top minus the next finer one's, on
    /// the columns a pit touches.
    step: [Vec<f64>; 5],
    /// The same over every column: the systematic part.
    step_all: [f64; 5],
    /// Per width and angle: the volume and the area seen.
    seen: [[(f64, f64); 3]; 5],
    /// Per voxel width: trees on a cell the rule sinks, and by how much in
    /// all and at most.
    trees: [(f64, f64, f64); 3],
}

const ANGLES: [f64; 3] = [15.0, 30.0, 60.0];

/// The study: every rule over every class of ground.
///
/// ```text
/// cargo test -p cubara-render --test lod_consistency -- --ignored --nocapture pit_rule_study_report
/// ```
#[test]
#[ignore]
fn pit_rule_study_report() {
    let setup = Setup::new();
    let gen = &setup.gen;
    let tree_blocks = setup.tree_blocks();
    for class in classes() {
        let mut level0 = Depression::default();
        let mut level0_seen = [(0.0, 0.0); 3];
        let mut pit_count = [0.0; 3];
        let (mut sizes, mut depths) = (Vec::new(), Vec::new());
        let (mut columns, mut trees_in_all) = (0.0, 0.0);
        let mut measured = vec![Measured::default(); RULES.len()];
        for &(x0, z0, side) in &class.regions {
            let ground = Ground::over(gen, x0, z0, side);
            let (sky, surface) = (ground.level_0(), ground.level_0_whole());
            columns += sky.len() as f64;
            level0.add(&Depression::of(&surface, &sky));
            for (a, angle) in ANGLES.iter().enumerate() {
                let s = seen(&sky, &surface, side, angle.to_radians().tan());
                level0_seen[a].0 += s.0;
                level0_seen[a].1 += s.1;
            }
            let pits = ground.pits();
            let size_class = |pit: &Vec<usize>| match pit.len() {
                0..=15 => 0,
                16..=63 => 1,
                _ => 2,
            };
            for pit in &pits {
                pit_count[size_class(pit)] += 1.0;
                sizes.push(pit.len() as f64);
                depths.extend(pit.iter().map(|&k| (surface[k] - sky[k]) as f64));
            }
            // The trees of the region, by trunk column.
            let trees: Vec<usize> = (0..side / 16)
                .flat_map(|cj| (0..side / 16).map(move |ci| (ci, cj)))
                .filter_map(|(ci, cj)| {
                    gen.tree_site(x0.div_euclid(16) + ci, z0.div_euclid(16) + cj, tree_blocks)
                })
                .map(|t| ((t.z - z0) * side + t.x - x0) as usize)
                .collect();
            trees_in_all += trees.len() as f64;
            for (r, &rule) in RULES.iter().enumerate() {
                let m = &mut measured[r];
                let mut finer = sky.clone();
                // Where the finer level draws the ground sunk.
                let mut finer_sunk: Vec<bool> =
                    sky.iter().zip(&surface).map(|(s, w)| s < w).collect();
                for (k, &w) in WIDTHS.iter().enumerate() {
                    let whole = ground.whole(w);
                    let drawn = ground.drawn(gen, rule, w);
                    m.depression[k].add(&Depression::of(&whole, &drawn));
                    for pit in &pits {
                        let shown = pit.iter().any(|&c| whole[c] - drawn[c] >= 1);
                        m.shown[k][size_class(pit)] += shown as u8 as f64;
                    }
                    for c in 0..drawn.len() {
                        let d = (drawn[c] - finer[c]) as f64;
                        m.step_all[k] += d;
                        // A column a pit touches: level 0 has one there, or
                        // either level draws the ground sunk.
                        let sunk = drawn[c] < whole[c];
                        if sky[c] < surface[c] || sunk || finer_sunk[c] {
                            m.step[k].push(d);
                        }
                        finer_sunk[c] = sunk;
                    }
                    for (a, angle) in ANGLES.iter().enumerate() {
                        let s = seen(&drawn, &whole, side, angle.to_radians().tan());
                        m.seen[k][a].0 += s.0;
                        m.seen[k][a].1 += s.1;
                    }
                    if k < 3 {
                        for &t in &trees {
                            let sunk = (whole[t] - drawn[t]) as f64;
                            if sunk >= 1.0 {
                                m.trees[k].0 += 1.0;
                                m.trees[k].1 += sunk;
                                m.trees[k].2 = m.trees[k].2.max(sunk);
                            }
                        }
                    }
                    finer = drawn;
                }
            }
        }
        sizes.sort_by(|a, b| a.total_cmp(b));
        depths.sort_by(|a, b| a.total_cmp(b));
        let pct = |v: &[f64], p: usize| {
            if v.is_empty() {
                0.0
            } else {
                v[(v.len() * p).div_ceil(100).max(1) - 1]
            }
        };
        println!("\n## {}\n", class.name);
        println!(
            "level 0: a cave opens {:.2}% of the columns ({} of {}), in {} pits ({} under 16 columns, {} of 16 to 63, {} larger); a pit is {:.0} / {:.0} / {:.0} columns (median / 90th percentile / largest) and a pit column {:.0} / {:.0} / {:.0} blocks deep; {:.1}% of the pit columns are deeper than the soil; volume {} blocks; {} trees",
            100.0 * level0.area / columns,
            level0.area,
            columns,
            sizes.len(),
            pit_count[0],
            pit_count[1],
            pit_count[2],
            pct(&sizes, 50),
            pct(&sizes, 90),
            pct(&sizes, 100),
            pct(&depths, 50),
            pct(&depths, 90),
            pct(&depths, 100),
            100.0 * level0.deep / level0.area.max(1.0),
            level0.volume,
            trees_in_all,
        );
        let ratio = |a: f64, b: f64| if b > 0.0 { a / b } else { f64::NAN };
        let five = |f: &dyn Fn(usize) -> String| -> String {
            (0..5).map(f).collect::<Vec<_>>().join(" / ")
        };

        println!("\nThe depression from above, against level 0's (cell width 2 / 4 / 8 / far 16 / far 32):\n");
        println!("| rule | area (any depth) | area deeper than the soil | volume | mean depth, blocks (level 0: {:.1}) | columns lifted |", ratio(level0.volume, level0.area));
        println!("|---|---|---|---|---|---|");
        for (r, rule) in RULES.iter().enumerate() {
            let d = &measured[r].depression;
            println!(
                "| {} | {} | {} | {} | {} | {} |",
                rule.name,
                five(&|k| format!("x{:.2}", ratio(d[k].area, level0.area))),
                five(&|k| format!("x{:.2}", ratio(d[k].deep, level0.deep))),
                five(&|k| format!("x{:.2}", ratio(d[k].volume, level0.volume))),
                five(&|k| format!("{:.1}", ratio(d[k].volume, d[k].area))),
                five(&|k| format!("{:.0}", d[k].raised)),
            );
        }

        println!("\nPits a level shows anything of (small / middling / large, % of level 0's):\n");
        println!("| rule | w 2 | w 4 | w 8 | far 16 | far 32 |");
        println!("|---|---|---|---|---|---|");
        for (r, rule) in RULES.iter().enumerate() {
            let cell = |k: usize| {
                (0..3)
                    .map(|s| {
                        format!(
                            "{:.0}",
                            100.0 * ratio(measured[r].shown[k][s], pit_count[s])
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" / ")
            };
            println!(
                "| {} | {} | {} | {} | {} | {} |",
                rule.name,
                cell(0),
                cell(1),
                cell(2),
                cell(3),
                cell(4)
            );
        }

        println!("\nThe swap on the columns a pit touches: coarser minus finer top, mean / 95th percentile / largest in blocks, and the mean in pixels at the edge (1080 rows); then all of it, in blocks moved per 1,000 columns of ground, and the signed mean over every column, in pixels:\n");
        println!("| rule | 1 -> 2 at 160 | 2 -> 4 at 288 | 4 -> 8 at 512 | 8 -> 16 at 1,032 | 16 -> 32 at 2,064 | blocks moved per 1,000 columns | every column, px |");
        println!("|---|---|---|---|---|---|---|---|");
        for (r, rule) in RULES.iter().enumerate() {
            let m = &mut measured[r];
            let mut moved = [0.0; 5];
            let cells: Vec<String> = (0..5)
                .map(|k| {
                    let (mean, p95, max, _) = step_stats(&mut m.step[k]);
                    moved[k] = mean * m.step[k].len() as f64 / columns * 1000.0;
                    format!(
                        "{:.1} / {:.0} / {:.0}; {:.1} px",
                        mean,
                        p95,
                        max,
                        mean * px_per_block(EDGE_OF[k])
                    )
                })
                .collect();
            println!(
                "| {} | {} | {} | {} |",
                rule.name,
                cells.join(" | "),
                five(&|k| format!("{:.0}", moved[k])),
                five(&|k| format!("{:+.2}", m.step_all[k] / columns * px_per_block(EDGE_OF[k]))),
            );
        }

        println!("\nWhat an eye sees of the pits, against level 0 (volume seen; area seen), per cell width 2 / 4 / 8 / far 16 / far 32:\n");
        println!("| rule | from 15 degrees | from 30 degrees | from 60 degrees |");
        println!("|---|---|---|---|");
        for (r, rule) in RULES.iter().enumerate() {
            let cell = |a: usize| {
                format!(
                    "{}; {}",
                    five(&|k| format!("x{:.2}", ratio(measured[r].seen[k][a].0, level0_seen[a].0))),
                    five(&|k| format!("x{:.2}", ratio(measured[r].seen[k][a].1, level0_seen[a].1))),
                )
            };
            println!(
                "| {} | {} | {} | {} |",
                rule.name,
                cell(0),
                cell(1),
                cell(2)
            );
        }

        println!("\nTrees on a cell the rule sinks (of {trees_in_all}), w 2 / 4 / 8: how many, their mean and largest drop in blocks:\n");
        println!("| rule | trees | mean drop | largest |");
        println!("|---|---|---|---|");
        for (r, rule) in RULES.iter().enumerate() {
            let t = &measured[r].trees;
            let three = |f: &dyn Fn(usize) -> String| -> String {
                (0..3).map(f).collect::<Vec<_>>().join(" / ")
            };
            println!(
                "| {} | {} | {} | {} |",
                rule.name,
                three(&|k| format!("{:.0}", t[k].0)),
                three(&|k| format!("{:.1}", ratio(t[k].1, t[k].0))),
                three(&|k| format!("{:.0}", t[k].2)),
            );
        }
    }
}

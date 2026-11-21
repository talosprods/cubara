//! **Which rule gives a coarse cell its top?** The study behind issue #423's
//! height rule, as a measurement.
//!
//! The owner's requirement (2026-11-20): approaching a mountain, it must not
//! be seen to shift or change at all; only detail may appear. A level swap
//! replaces every coarse cell by the finer cells under it, so what the eye
//! sees change at a ring edge is, per level-0 column, **the coarser level's
//! drawn top minus the finer one's**. That difference is what a rule is
//! judged by here, in pixels at the distance of the edge it happens at:
//!
//! - **the surface** facing the eye: the difference over every column. Its
//!   *mean* is the systematic part -- the whole slope sinking or rising as
//!   the edge passes over it -- and its 95th percentile the size of what
//!   changes at all;
//! - **the outline** against the sky: the same difference for the skyline,
//!   the highest top along a line of sight (four directions, seen from far
//!   off and level with the ground), on lines where level 0 has a crest
//!   inside the window and not at its cut edge.
//!
//! Every rule is worked out from the height field alone, in a window of
//! [`SIDE`] blocks around each of the harness's seeded footprints, so the
//! candidates are compared on the same ground and no rule but the chosen one
//! has to exist in the game. `the_chosen_rule_is_the_games` holds the copy of
//! the chosen rule here to `WorldGen::cell_top`.
//!
//! ```text
//! cargo test -p cubara-render --test lod_consistency -- --ignored --nocapture height_rule_study_report
//! ```

use super::harness::{draw_footprints, Footprint, Terrain, FOOTPRINT, TERRAINS};
use super::{FOOTPRINT_SEED, PER_KIND};
use cubara_world::WorldGen;

/// Ground either side of a footprint that the window takes in, in blocks: a
/// skyline needs the crest that makes it, and a cell its neighbours.
const PAD: i32 = 64;
/// The window's side, in blocks.
const SIDE: i32 = FOOTPRINT + 2 * PAD;
/// A skyline counts where level 0's crest along the line stands at least
/// this far inside the window: at the cut edge the "crest" is only where the
/// window stops. **Chosen, not derived**: two of the widest cells.
const CREST_MARGIN: i32 = 16;

/// The ring edges of `DEFAULT_RING_SCHEDULE`, in blocks, and the level that
/// is drawn just outside each (the level inside is one finer).
const EDGES: [(i32, u32); 3] = [(160, 1), (288, 2), (512, 3)];

/// The displays the change is put in pixels for: rows of pixels, at the
/// renderer's 60 degrees of vertical field of view. 1080 is the gate's; the
/// owner plays at 5120 x 1440 and 2560 x 1664.
const DISPLAY_ROWS: [u32; 3] = [1080, 1440, 1664];

/// Pixels a one-block step covers at `distance` blocks on a display of
/// `rows` rows (`lod_probe::max_step_px`'s figure: the small-angle one).
fn px_per_block(rows: u32, distance: i32) -> f64 {
    let pixel_angle = 60f64.to_radians() / rows as f64;
    1.0 / distance as f64 / pixel_angle
}

/// A candidate rule for a cell's top, from the level-0 columns it covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    /// The highest column (#423 decision 1 as first built).
    Max,
    /// The mean of the columns, rounded down (main before #423 took the mean
    /// of four of them, rounded down).
    MeanFloor,
    /// The mean of the columns, rounded to the nearest block.
    MeanRound,
    /// The median column (the lower of the middle two).
    Median,
    /// The column three quarters of the way up.
    Quantile75,
    /// Half way between the lowest and the highest column.
    Midrange,
    /// The highest column where the cell holds a crest, the rounded mean
    /// elsewhere. A cell holds a crest when its highest column stands above
    /// the highest columns of both neighbouring cells along x, along z or
    /// along either diagonal.
    CrestOrMean,
    /// **The level top**: fit a plane through the cell's columns (least
    /// squares), take the slope out, and keep the highest column of what is
    /// left -- the plane, lifted until it rests on the cell's highest bump,
    /// read at the cell's centre. On a plane slope that is the mean; over a
    /// peak or across a ridge line, where the slopes either side cancel, it
    /// is the highest column.
    LevelMax,
    /// As [`Rule::LevelMax`], but the bump the lifted plane rests on is the
    /// one three quarters of the way up the cell's detrended columns.
    LevelQuantile75,
    /// [`Rule::MeanRound`] with a half rounded to the even block, so that
    /// rounding itself lifts nothing.
    MeanEven,
    /// [`Rule::LevelMax`] with a half rounded to the even block, and never
    /// above the cell's highest column.
    LevelMaxEven,
    /// [`Rule::LevelMaxEven`], and the highest column itself in a cell that
    /// holds a summit: the highest column of its own and its eight
    /// neighbouring cells (the first of them in `(z, x)` order where several
    /// are as high).
    LevelMaxSummits,
    /// [`Rule::MeanEven`], and the highest column in a cell that holds a
    /// summit: the rule #432 built, which grew towers on the crests.
    MeanSummits,
    /// [`Rule::MeanSummits`] with a summit's lift over the mean capped at
    /// half the cell's width.
    MeanSummitsCapHalf,
    /// [`Rule::MeanSummits`] with a summit at the column three quarters of
    /// the way up the cell (never under the mean) instead of the highest.
    MeanSummitsQ75,
    /// The same at nine tenths of the way up.
    MeanSummitsQ90,
    /// [`Rule::MeanSummits`] with a summit at the highest mean of the four
    /// half-width cells it splits into. **The rule chosen**:
    /// `WorldGen::cell_top`.
    MeanSummitsChildMean,
}

pub const RULES: [Rule; 17] = [
    Rule::Max,
    Rule::MeanFloor,
    Rule::MeanRound,
    Rule::Median,
    Rule::Quantile75,
    Rule::Midrange,
    Rule::CrestOrMean,
    Rule::LevelMax,
    Rule::LevelQuantile75,
    Rule::MeanEven,
    Rule::LevelMaxEven,
    Rule::LevelMaxSummits,
    Rule::MeanSummits,
    Rule::MeanSummitsCapHalf,
    Rule::MeanSummitsQ75,
    Rule::MeanSummitsQ90,
    Rule::MeanSummitsChildMean,
];

impl Rule {
    fn name(self) -> &'static str {
        match self {
            Rule::Max => "max",
            Rule::MeanFloor => "mean, floor",
            Rule::MeanRound => "mean, rounded",
            Rule::Median => "median",
            Rule::Quantile75 => "75th percentile",
            Rule::Midrange => "midrange",
            Rule::CrestOrMean => "crest: max, else mean",
            Rule::LevelMax => "level top (detrended max)",
            Rule::LevelQuantile75 => "level 75th percentile",
            Rule::MeanEven => "mean, half to even",
            Rule::LevelMaxEven => "level top, half to even",
            Rule::LevelMaxSummits => "level top + summits",
            Rule::MeanSummits => "mean + summits (#432)",
            Rule::MeanSummitsCapHalf => "mean + summits, lift <= w/2",
            Rule::MeanSummitsQ75 => "mean + summits at p75",
            Rule::MeanSummitsQ90 => "mean + summits at p90",
            Rule::MeanSummitsChildMean => "mean + summits at best half (chosen)",
        }
    }
}

/// `a / b`, rounded to the nearest whole, halves up.
fn div_round(a: i64, b: i64) -> i64 {
    (2 * a + b).div_euclid(2 * b)
}

/// `a / b` (`b > 0`), rounded to the nearest whole, a half to the even one.
fn div_round_even(a: i64, b: i64) -> i64 {
    let (q, r) = (a.div_euclid(b), a.rem_euclid(b));
    match (2 * r).cmp(&b) {
        std::cmp::Ordering::Less => q,
        std::cmp::Ordering::Greater => q + 1,
        std::cmp::Ordering::Equal => q + (q & 1),
    }
}

/// The heights of a cell with the least-squares plane's slope taken out,
/// each times `q` (returned): `h * q - (A * u + B * v)`, `u` and `v` a
/// column's doubled offset from the cell's centre. Whole numbers throughout.
fn detrended(h: &[i32], w: i32) -> (Vec<i64>, i64) {
    let (w64, n) = (w as i64, (w * w) as usize);
    assert_eq!(h.len(), n);
    // Sum of u^2 over every column: w * (w (w^2 - 1) / 3).
    let q = (w64 * w64 * (w64 * w64 - 1) / 3).max(1);
    let offset = |k: i32| (2 * k - (w - 1)) as i64;
    let (mut a, mut b) = (0i64, 0i64);
    for (k, &height) in h.iter().enumerate() {
        let (i, j) = (k as i32 % w, k as i32 / w);
        a += offset(i) * height as i64;
        b += offset(j) * height as i64;
    }
    let out = h
        .iter()
        .enumerate()
        .map(|(k, &height)| {
            let (i, j) = (k as i32 % w, k as i32 / w);
            height as i64 * q - a * offset(i) - b * offset(j)
        })
        .collect();
    (out, q)
}

/// The sorted value `num / den` of the way up `sorted` (the lower one where
/// it falls between two).
fn quantile<T: Copy>(sorted: &[T], num: usize, den: usize) -> T {
    sorted[(sorted.len() * num).div_ceil(den).max(1) - 1]
}

/// A cell's top under `rule`: `h` its `w * w` level-0 surfaces, row by row;
/// `crest` whether it holds a crest ([`Rule::CrestOrMean`]) and `summit`
/// whether it holds a summit ([`Rule::LevelMaxSummits`]).
pub fn top(rule: Rule, h: &[i32], w: i32, crest: bool, summit: bool) -> i32 {
    let n = h.len() as i64;
    let sum: i64 = h.iter().map(|&v| v as i64).sum();
    let (min, max) = (
        *h.iter().min().expect("a cell covers a column"),
        *h.iter().max().expect("a cell covers a column"),
    );
    let sorted = || {
        let mut s = h.to_vec();
        s.sort_unstable();
        s
    };
    match rule {
        Rule::Max => max,
        Rule::MeanFloor => sum.div_euclid(n) as i32,
        Rule::MeanRound => div_round(sum, n) as i32,
        Rule::Median => quantile(&sorted(), 1, 2),
        Rule::Quantile75 => quantile(&sorted(), 3, 4),
        Rule::Midrange => div_round(min as i64 + max as i64, 2) as i32,
        Rule::CrestOrMean => {
            if crest {
                max
            } else {
                div_round(sum, n) as i32
            }
        }
        Rule::LevelMax => {
            let (d, q) = detrended(h, w);
            div_round(*d.iter().max().expect("a column"), q) as i32
        }
        Rule::LevelQuantile75 => {
            // The plane at the centre is the mean; what is left over a
            // column is its residual. Lift the plane to the residual three
            // quarters of the way up.
            let (mut d, q) = detrended(h, w);
            d.sort_unstable();
            div_round(quantile(&d, 3, 4), q) as i32
        }
        Rule::MeanEven => div_round_even(sum, n) as i32,
        Rule::LevelMaxEven => {
            let (d, q) = detrended(h, w);
            (div_round_even(*d.iter().max().expect("a column"), q) as i32).min(max)
        }
        Rule::LevelMaxSummits => {
            if summit {
                max
            } else {
                top(Rule::LevelMaxEven, h, w, crest, false)
            }
        }
        Rule::MeanSummits => {
            if summit {
                max
            } else {
                div_round_even(sum, n) as i32
            }
        }
        Rule::MeanSummitsCapHalf => {
            let mean = div_round_even(sum, n) as i32;
            if summit {
                max.min(mean + w / 2)
            } else {
                mean
            }
        }
        Rule::MeanSummitsQ75 | Rule::MeanSummitsQ90 => {
            let mean = div_round_even(sum, n) as i32;
            let (num, den) = if rule == Rule::MeanSummitsQ75 {
                (3, 4)
            } else {
                (9, 10)
            };
            if summit {
                quantile(&sorted(), num, den).max(mean)
            } else {
                mean
            }
        }
        Rule::MeanSummitsChildMean => {
            let mean = div_round_even(sum, n) as i32;
            if !summit || w < 2 {
                return mean;
            }
            let half = w / 2;
            (0..4)
                .map(|q| {
                    let (qa, qb) = (q % 2 * half, q / 2 * half);
                    let s: i64 = (0..half * half)
                        .map(|k| h[((qb + k / half) * w + qa + k % half) as usize] as i64)
                        .sum();
                    div_round_even(s, (half * half) as i64) as i32
                })
                .max()
                .expect("four children")
                .max(mean)
        }
    }
}

/// The level-0 surface over a window of [`SIDE`] blocks around a footprint.
pub struct Window {
    x0: i32,
    z0: i32,
    h: Vec<i32>,
}

impl Window {
    pub fn around(gen: &WorldGen, fp: Footprint) -> Self {
        let (x0, z0) = (fp.x0 - PAD, fp.z0 - PAD);
        let h = (0..SIDE)
            .flat_map(|j| (0..SIDE).map(move |i| (i, j)))
            .map(|(i, j)| gen.surface_height(x0 + i, z0 + j))
            .collect();
        Self { x0, z0, h }
    }

    fn at(&self, i: i32, j: i32) -> i32 {
        self.h[(j * SIDE + i) as usize]
    }

    /// Each column's drawn top block under `rule` at `level`: the top of the
    /// cell of `2^level` blocks that holds it. Row by row, `z` outer.
    pub fn tops(&self, rule: Rule, level: u32) -> Vec<i32> {
        let w = 1i32 << level;
        if w == 1 {
            return self.h.clone();
        }
        let cells = SIDE / w;
        let cell = |ci: i32, cj: i32| -> Vec<i32> {
            (0..w)
                .flat_map(|b| (0..w).map(move |a| (a, b)))
                .map(|(a, b)| self.at(ci * w + a, cj * w + b))
                .collect()
        };
        // Every cell's highest column, for the crest rule.
        let highest: Vec<i32> = (0..cells * cells)
            .map(|k| {
                *cell(k % cells, k / cells)
                    .iter()
                    .max()
                    .expect("a cell covers a column")
            })
            .collect();
        let high = |ci: i32, cj: i32| -> Option<i32> {
            ((0..cells).contains(&ci) && (0..cells).contains(&cj))
                .then(|| highest[(cj * cells + ci) as usize])
        };
        let mut out = vec![0; (SIDE * SIDE) as usize];
        for cj in 0..cells {
            for ci in 0..cells {
                let own = highest[(cj * cells + ci) as usize];
                let crest = [(1, 0), (0, 1), (1, 1), (1, -1)]
                    .into_iter()
                    .any(|(di, dj)| {
                        // A neighbour outside the window is not known to be lower.
                        let lower = |n: Option<i32>| n.is_some_and(|n| n < own);
                        lower(high(ci + di, cj + dj)) && lower(high(ci - di, cj - dj))
                    });
                // A summit: the first highest column of the nine cells
                // around (`z`, then `x`) is in this one. A cell at the
                // window's edge is not known to hold one.
                let inside = (1..cells - 1).contains(&ci) && (1..cells - 1).contains(&cj);
                let summit = inside && {
                    let (mut first, mut at) = (i32::MIN, (0, 0));
                    for j in (cj - 1) * w..(cj + 2) * w {
                        for i in (ci - 1) * w..(ci + 2) * w {
                            if self.at(i, j) > first {
                                (first, at) = (self.at(i, j), (i, j));
                            }
                        }
                    }
                    at.0 / w == ci && at.1 / w == cj
                };
                let t = top(rule, &cell(ci, cj), w, crest, summit);
                for b in 0..w {
                    for a in 0..w {
                        out[((cj * w + b) * SIDE + ci * w + a) as usize] = t;
                    }
                }
            }
        }
        out
    }
}

/// The lines of sight the outline is taken along: each a list of window
/// columns `(i, j)`, in the order the eye looks through them. Along x, along
/// z and along both diagonals; only lines through the footprint.
fn sight_lines() -> Vec<Vec<(i32, i32)>> {
    let mut lines = Vec::new();
    let inner = PAD..PAD + FOOTPRINT;
    for k in inner.clone() {
        lines.push((0..SIDE).map(|n| (n, k)).collect());
        lines.push((0..SIDE).map(|n| (k, n)).collect());
    }
    // Diagonals through the footprint's own diagonal band.
    for k in -(FOOTPRINT - 1)..FOOTPRINT {
        // i - j = k
        lines.push(
            (0..SIDE)
                .map(|n| (n, n - k))
                .filter(|&(_, j)| (0..SIDE).contains(&j))
                .collect(),
        );
        // i + j = SIDE - 1 + k
        lines.push(
            (0..SIDE)
                .map(|n| (n, SIDE - 1 + k - n))
                .filter(|&(_, j)| (0..SIDE).contains(&j))
                .collect(),
        );
    }
    lines
}

/// The highest top along `line`, and where it first and last stands.
fn skyline(tops: &[i32], line: &[(i32, i32)]) -> (i32, usize, usize) {
    let at = |n: usize| tops[(line[n].1 * SIDE + line[n].0) as usize];
    let high = (0..line.len()).map(at).max().expect("a line of columns");
    let first = (0..line.len())
        .find(|&n| at(n) == high)
        .expect("the highest");
    let last = (0..line.len())
        .rev()
        .find(|&n| at(n) == high)
        .expect("the highest");
    (high, first, last)
}

/// What one edge changes, pooled: per-column and per-line differences, in
/// blocks, the coarser level minus the finer.
#[derive(Default, Clone)]
struct Change {
    surface: Vec<f64>,
    outline: Vec<f64>,
    /// Columns where the coarser level stands under the finer one: land
    /// that appears out of the sky on approach.
    under: usize,
}

/// The mean, and the 95th percentile and the largest of the magnitude.
fn stats(values: &[f64]) -> (f64, f64, f64, f64) {
    super::harness::spread(values)
}

/// A level's tops against level 0's: how far the drawn ground stands from
/// the real one, pooled.
#[derive(Default, Clone)]
struct Against0 {
    column: Vec<f64>,
    outline: Vec<f64>,
    /// Per footprint whose summit stands inside it: the level's highest top
    /// over the footprint minus level 0's.
    peak: Vec<f64>,
}

struct Studied {
    /// Per rule, per kind of terrain, per edge.
    change: Vec<Vec<[Change; 3]>>,
    /// Per rule, per kind, per coarse level (1 to 3).
    against0: Vec<Vec<[Against0; 3]>>,
    /// Lines of sight with a crest inside the window, per kind.
    crest_lines: Vec<usize>,
}

fn study(gen: &WorldGen, footprints: &[(Terrain, Footprint)]) -> Studied {
    let lines = sight_lines();
    let kinds = TERRAINS.len();
    let mut out = Studied {
        change: vec![vec![Default::default(); kinds]; RULES.len()],
        against0: vec![vec![Default::default(); kinds]; RULES.len()],
        crest_lines: vec![0; kinds],
    };
    for &(kind, fp) in footprints {
        let k = TERRAINS.iter().position(|&t| t == kind).expect("a kind");
        let window = Window::around(gen, fp);
        let level0 = window.tops(Rule::Max, 0);
        // The lines level 0 has a crest on, inside the window.
        let crests: Vec<&Vec<(i32, i32)>> = lines
            .iter()
            .filter(|line| {
                let (_, first, last) = skyline(&level0, line);
                let margin = CREST_MARGIN as usize;
                first >= margin && last + margin < line.len()
            })
            .collect();
        out.crest_lines[k] += crests.len();
        let inner = |i: i32| (PAD..PAD + FOOTPRINT).contains(&i);
        // Whether the footprint's highest column is the window's too: a summit.
        let peak_of = |tops: &[i32], inside: bool| {
            (0..SIDE * SIDE)
                .filter(|&c| !inside || (inner(c % SIDE) && inner(c / SIDE)))
                .map(|c| tops[c as usize])
                .max()
                .expect("columns")
        };
        let summit = peak_of(&level0, true) == peak_of(&level0, false);
        for (r, &rule) in RULES.iter().enumerate() {
            let tops: Vec<Vec<i32>> = (0..=3).map(|level| window.tops(rule, level)).collect();
            for (e, &(_, coarse)) in EDGES.iter().enumerate() {
                let (c, f) = (&tops[coarse as usize], &tops[coarse as usize - 1]);
                let change = &mut out.change[r][k][e];
                for j in PAD..PAD + FOOTPRINT {
                    for i in PAD..PAD + FOOTPRINT {
                        let d = c[(j * SIDE + i) as usize] - f[(j * SIDE + i) as usize];
                        change.surface.push(d as f64);
                        change.under += (d < 0) as usize;
                    }
                }
                for line in &crests {
                    change
                        .outline
                        .push((skyline(c, line).0 - skyline(f, line).0) as f64);
                }
            }
            // Levels 1 to 3, each against level 0.
            for (a, level_tops) in out.against0[r][k].iter_mut().zip(&tops[1..]) {
                for j in PAD..PAD + FOOTPRINT {
                    for i in PAD..PAD + FOOTPRINT {
                        let c = (j * SIDE + i) as usize;
                        a.column.push((level_tops[c] - level0[c]) as f64);
                    }
                }
                for line in &crests {
                    a.outline
                        .push((skyline(level_tops, line).0 - skyline(&level0, line).0) as f64);
                }
                if summit {
                    // Over the whole window: where two columns are as high
                    // as each other the summit is the first one's cell,
                    // which may be the cell beside the footprint.
                    a.peak
                        .push((peak_of(level_tops, false) - peak_of(&level0, false)) as f64);
                }
            }
        }
    }
    out
}

fn kind_name(kind: Terrain) -> &'static str {
    match kind {
        Terrain::Flat => "rolling hills",
        Terrain::Gentle => "gentle slope",
        Terrain::Steep => "steep mountain",
        Terrain::Cliff => "cliffs",
    }
}

/// **The rule the study chose is the one the game has**: this file's copy
/// of it (`Rule::MeanSummitsChildMean`, written out from the height field so that it
/// can stand beside the rules that were not chosen) gives every cell the top
/// `WorldGen::cell_top` gives it, at every coarse level, over one window of
/// each kind of terrain. Without this the table could describe a rule the
/// game does not run.
#[test]
fn the_chosen_rule_is_the_games() {
    let gen = WorldGen::new(cubara_world::World::new().seed());
    let footprints = draw_footprints(&gen, FOOTPRINT_SEED, 1);
    assert_eq!(footprints.len(), TERRAINS.len());
    let (mut cells, mut summits) = (0, 0);
    for (kind, fp) in footprints {
        let window = Window::around(&gen, fp);
        for level in 1..=3u32 {
            let w = 1i32 << level;
            let tops = window.tops(Rule::MeanSummitsChildMean, level);
            let means = window.tops(Rule::MeanEven, level);
            // Every cell with its eight neighbours inside the window.
            for cj in 1..SIDE / w - 1 {
                for ci in 1..SIDE / w - 1 {
                    let at = ((cj * w) * SIDE + ci * w) as usize;
                    let (x, z) = (window.x0 + ci * w, window.z0 + cj * w);
                    assert_eq!(
                        tops[at],
                        gen.cell_top(x, z, w),
                        "{kind:?}: the cell of {w} at ({x}, {z})"
                    );
                    cells += 1;
                    summits += (tops[at] != means[at]) as usize;
                }
            }
        }
    }
    assert!(
        cells > 10_000 && summits > 50,
        "{cells} cells, {summits} of them summits that stand above their mean"
    );
}

/// The study, printed: per kind of terrain, every rule's change at each ring
/// edge -- the coarser level minus the finer one -- as a table in pixels at
/// 1080 rows (the other displays are the same numbers times 1.33 and 1.54),
/// then the same in blocks with the spread, then every level against level 0.
#[test]
#[ignore]
fn height_rule_study_report() {
    let gen = WorldGen::new(cubara_world::World::new().seed());
    let footprints = draw_footprints(&gen, FOOTPRINT_SEED, PER_KIND);
    let studied = study(&gen, &footprints);
    println!(
        "footprint seed {FOOTPRINT_SEED:#x}; window {SIDE} blocks; crest margin {CREST_MARGIN}"
    );
    println!(
        "pixels per block at 160 / 288 / 512 blocks: {}",
        DISPLAY_ROWS
            .iter()
            .map(|&rows| format!(
                "{rows} rows {:.2} / {:.2} / {:.2}",
                px_per_block(rows, 160),
                px_per_block(rows, 288),
                px_per_block(rows, 512)
            ))
            .collect::<Vec<_>>()
            .join("; ")
    );
    let edges =
        |f: &dyn Fn(usize) -> String| (0..EDGES.len()).map(f).collect::<Vec<_>>().join(" / ");
    for (k, &kind) in TERRAINS.iter().enumerate() {
        println!(
            "\n### {} ({} lines of sight with a crest; pixels at 1080 rows, at the 160 / 288 / 512-block edges)\n",
            kind_name(kind),
            studied.crest_lines[k]
        );
        println!("| rule | surface: mean | surface: 95th pct | outline: mean | outline: 95th pct | columns the coarser level stands under, % | summit, blocks, L1 / L2 / L3 against level 0 |");
        println!("|---|---|---|---|---|---|---|");
        for (r, &rule) in RULES.iter().enumerate() {
            let c = &studied.change[r][k];
            let px = |e: usize, blocks: f64| blocks * px_per_block(DISPLAY_ROWS[0], EDGES[e].0);
            let summit: Vec<String> = (0..3)
                .map(|l| {
                    let a = &studied.against0[r][k][l];
                    if a.peak.is_empty() {
                        "-".into()
                    } else {
                        format!("{:+.1}", stats(&a.peak).0)
                    }
                })
                .collect();
            println!(
                "| {} | {} | {} | {} | {} | {} | {} ({}) |",
                rule.name(),
                edges(&|e| format!("{:+.1}", px(e, stats(&c[e].surface).0))),
                edges(&|e| format!("{:.1}", px(e, stats(&c[e].surface).2))),
                edges(&|e| format!("{:+.1}", px(e, stats(&c[e].outline).0))),
                edges(&|e| format!("{:.1}", px(e, stats(&c[e].outline).2))),
                edges(&|e| format!(
                    "{:.0}",
                    c[e].under as f64 * 100.0 / c[e].surface.len() as f64
                )),
                summit.join(" / "),
                studied.against0[r][k][0].peak.len(),
            );
        }
        println!("\nin blocks -- surface: mean / mean magnitude / 95th pct / largest; outline: mean / mean magnitude / 95th pct:\n");
        for (r, &rule) in RULES.iter().enumerate() {
            let c = &studied.change[r][k];
            println!(
                "{:<28} | {}",
                rule.name(),
                (0..EDGES.len())
                    .map(|e| {
                        let (mean, mag, p95, max) = stats(&c[e].surface);
                        let (omean, omag, op95, _) = stats(&c[e].outline);
                        format!(
                            "{}: {mean:+.2} / {mag:.2} / {p95:.0} / {max:.0}; {omean:+.2} / {omag:.2} / {op95:.0}",
                            EDGES[e].0
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" | ")
            );
        }
        println!("\neach level against level 0, blocks -- column: mean / 95th pct; outline: mean / 95th pct:\n");
        for (r, &rule) in RULES.iter().enumerate() {
            let row: Vec<String> = (0..3)
                .map(|l| {
                    let a = &studied.against0[r][k][l];
                    let (mean, _, p95, _) = stats(&a.column);
                    let (omean, _, op95, _) = stats(&a.outline);
                    format!("L{}: {mean:+.2} / {p95:.0}; {omean:+.2} / {op95:.0}", l + 1)
                })
                .collect();
            println!("{:<28} | {}", rule.name(), row.join(" | "));
        }
    }
}

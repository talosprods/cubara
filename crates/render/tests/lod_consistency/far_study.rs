//! **Which rule gives a far quad its top?** The study behind #422's far
//! height rule, as a measurement, in the terms of `study.rs`: what a swap
//! changes is, per level-0 column, the coarser drawn top minus the finer one,
//! in pixels at the distance the swap happens at.
//!
//! The far terrain keeps a quad no wider than 16 pixels (`FarQuality::High`),
//! so a quad of `q` blocks is drawn from `q * 64.5` blocks (at 1080 rows:
//! `1 / (16 * pixel)`) and swaps for one of `q / 2` at half that. The edges
//! studied, as (finer, coarser) quad widths and the distance in blocks:
//! the join of the voxel rings (level 3's cell of 8) with the far terrain's
//! first quads at 1,024, then 16 to 32 at 2,064, 32 to 64 at 4,128 and 64 to
//! 128 at 8,256.
//!
//! The candidates are every one a *rule for quads wider than 8 blocks*
//! (`WorldGen::cell_top_sampled`'s `n` x `n` sampling, `n` = 4, as the game
//! has it): up to 8 blocks the voxel levels' own rule holds, and no
//! candidate touches it. Each is worked out from the height field alone, so
//! the ones not chosen need not exist in the game;
//! `the_chosen_far_rule_is_the_games` holds the chosen one to
//! `WorldGen::cell_top_sampled`.
//!
//! The study was first run when a summit stood at its highest column (#432)
//! and chose the voxels' rule unchanged. #437 then moved the voxels' summit
//! to the highest mean of a cell's four half-width cells; the far terrain
//! follows, because it has no rule of its own, and the study was run again
//! with the old rule as a candidate beside it (`FarRule::Highest`).
//!
//! ```text
//! cargo test -p cubara-render --test lod_consistency -- --ignored --nocapture far_height_rule_study_report
//! ```

use super::harness::{draw_footprints, spread, Footprint, Terrain, FOOTPRINT, TERRAINS};
use super::{FOOTPRINT_SEED, PER_KIND};
use cubara_world::WorldGen;

/// The region measured, in blocks a side, centred on a footprint: wide
/// enough for two cells of the widest quad studied.
const REGION: i32 = 256;
/// Ground around the region the rules look at (a cell's neighbours, and the
/// crests of a skyline). Two of the widest cells.
const PAD: i32 = 256;
const SIDE: i32 = REGION + 2 * PAD;
/// A skyline counts where level 0's crest stands at least this far inside
/// the window. **Chosen, not derived**: a quarter of the widest cell, inside the region.
const CREST_MARGIN: i32 = 32;
/// Samples a side for quads wider than 8, as the game has them.
const SAMPLES: i32 = 4;
/// Every how-manieth line of sight is looked along. **Chosen, not derived**:
/// the lines are 256 apart per direction otherwise, far more than needed.
const LINE_EVERY: i32 = 8;

/// (finer width, coarser width, distance of the swap in blocks at 1080 rows).
const EDGES: [(i32, i32, i32); 4] = [
    (8, 16, 1032),
    (16, 32, 2064),
    (32, 64, 4128),
    (64, 128, 8256),
];

fn px_per_block(rows: u32, distance: i32) -> f64 {
    let pixel_angle = 60f64.to_radians() / rows as f64;
    1.0 / distance as f64 / pixel_angle
}

/// Where a quad that holds a summit stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Summit {
    /// At its mean, like any other quad.
    No,
    /// At its highest sample (#432's rule, the voxels' until #437).
    Highest,
    /// At the highest mean of the four quarters of its sample grid (#437).
    BestHalf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FarRule {
    /// What `WorldGen::cell_top_sampled` is today: the mean of the samples,
    /// and where the quad holds a summit among its eight neighbours the
    /// highest mean of the four quarters of its sample grid (#437).
    TopRule,
    /// The same with the summit at its highest sample: the rule this study
    /// first chose, when it was the voxels' too (#432).
    Highest,
    /// The mean of the samples, half to even: no summits.
    Mean,
    /// A summit at its highest sample, judged among the 24 cells within two
    /// cells.
    Summit5,
    /// A summit at its highest sample, its lift over the mean at most
    /// `w / 4` blocks.
    CapQuarter,
    /// The same at most `w / 8` blocks.
    CapEighth,
    /// Summit among 24, lift at most `w / 4`.
    Summit5CapQuarter,
    /// TopRule over 8 x 8 samples instead of 4 x 4 (four times the heights).
    TopRule8,
}

pub const FAR_RULES: [FarRule; 8] = [
    FarRule::TopRule,
    FarRule::Highest,
    FarRule::Mean,
    FarRule::Summit5,
    FarRule::CapQuarter,
    FarRule::CapEighth,
    FarRule::Summit5CapQuarter,
    FarRule::TopRule8,
];

impl FarRule {
    fn name(self) -> &'static str {
        match self {
            FarRule::TopRule => "TopRule (today, #437)",
            FarRule::Highest => "summit at its highest sample (#432)",
            FarRule::Mean => "plain mean",
            FarRule::Summit5 => "highest, summit among 24 cells",
            FarRule::CapQuarter => "highest, lift <= w/4",
            FarRule::CapEighth => "highest, lift <= w/8",
            FarRule::Summit5CapQuarter => "highest among 24, lift <= w/4",
            FarRule::TopRule8 => "TopRule, 8 x 8 samples",
        }
    }
    /// How many cells either side a summit is judged among, the cap on its
    /// lift as a fraction of the width (0: none), and where it stands.
    fn parts(self) -> (i32, i32, Summit) {
        match self {
            FarRule::TopRule => (1, 0, Summit::BestHalf),
            FarRule::Highest => (1, 0, Summit::Highest),
            FarRule::Mean => (1, 0, Summit::No),
            FarRule::Summit5 => (2, 0, Summit::Highest),
            FarRule::CapQuarter => (1, 4, Summit::Highest),
            FarRule::CapEighth => (1, 8, Summit::Highest),
            FarRule::Summit5CapQuarter => (2, 4, Summit::Highest),
            FarRule::TopRule8 => (1, 0, Summit::BestHalf),
        }
    }
}

fn div_round_half_even(a: i64, b: i64) -> i64 {
    let (q, r) = (a.div_euclid(b), a.rem_euclid(b));
    match (2 * r).cmp(&b) {
        std::cmp::Ordering::Less => q,
        std::cmp::Ordering::Greater => q + 1,
        std::cmp::Ordering::Equal => q + (q & 1),
    }
}

/// A quad's top under `rule`: the cell `(x0, z0)` of width `w`, `n` samples a
/// side, from the height field `height`. Up to 8 blocks the voxel levels'
/// rule (`n` is then `w`, and the rule is `TopRule`'s whatever `rule` is).
pub fn far_top(height: &impl Fn(i32, i32) -> i32, rule: FarRule, x0: i32, z0: i32, w: i32) -> i32 {
    let (reach, cap_div, summit) = if w <= 8 {
        FarRule::TopRule.parts()
    } else {
        rule.parts()
    };
    let n = if w <= 8 {
        w
    } else if rule == FarRule::TopRule8 {
        8
    } else {
        SAMPLES
    };
    let count = n.clamp(1, w);
    let step = w / count;
    let lead = (step - 1) / 2;
    let sample = |ci: i32, cj: i32, a: i32, b: i32| {
        height(x0 + ci * w + lead + a * step, z0 + cj * w + lead + b * step)
    };
    let (mut sum, mut high, mut first) = (0i64, i32::MIN, (0, 0));
    // The four quarters of the sample grid: each one's sum and count.
    let half = count / 2;
    let (mut quarter, mut quarter_count) = ([0i64; 4], [0i64; 4]);
    for b in 0..count {
        for a in 0..count {
            let h = sample(0, 0, a, b);
            sum += h as i64;
            let q = (a >= half) as usize + 2 * (b >= half) as usize;
            quarter[q] += h as i64;
            quarter_count[q] += 1;
            if h > high {
                (high, first) = (h, (a, b));
            }
        }
    }
    let mean = div_round_half_even(sum, (count * count) as i64) as i32;
    if mean == high || summit == Summit::No {
        return mean;
    }
    for cj in -reach..=reach {
        for ci in -reach..=reach {
            if (ci, cj) == (0, 0) {
                continue;
            }
            for b in 0..count {
                for a in 0..count {
                    let h = sample(ci, cj, a, b);
                    let before = cj < 0 || (cj == 0 && (b < first.1 || (b == first.1 && ci < 0)));
                    if h > high || (h == high && before) {
                        return mean;
                    }
                }
            }
        }
    }
    match summit {
        Summit::BestHalf => (0..4)
            .map(|q| div_round_half_even(quarter[q], quarter_count[q]) as i32)
            .max()
            .expect("four quarters"),
        _ if cap_div > 0 && w > 8 => mean + (high - mean).min(w / cap_div),
        _ => high,
    }
}

struct Window {
    x0: i32,
    z0: i32,
    h: Vec<i32>,
}

impl Window {
    /// The window around a footprint: region aligned to the widest cell.
    fn around(gen: &WorldGen, fp: Footprint) -> Self {
        let centre = (fp.x0 + FOOTPRINT / 2, fp.z0 + FOOTPRINT / 2);
        let align = |c: i32| (c - REGION / 2).div_euclid(128) * 128 - PAD;
        let (x0, z0) = (align(centre.0), align(centre.1));
        let h = (0..SIDE)
            .flat_map(|j| (0..SIDE).map(move |i| (i, j)))
            .map(|(i, j)| gen.surface_height(x0 + i, z0 + j))
            .collect();
        Self { x0, z0, h }
    }
    fn height(&self, x: i32, z: i32) -> i32 {
        let (i, j) = (x - self.x0, z - self.z0);
        if (0..SIDE).contains(&i) && (0..SIDE).contains(&j) {
            self.h[(j * SIDE + i) as usize]
        } else {
            i32::MIN / 2
        }
    }
    /// Every column's top under `rule` at quad width `w`, window row by row.
    /// Cells whose neighbours leave the window are not drawn (`NaN`).
    fn tops(&self, rule: FarRule, w: i32) -> Vec<f64> {
        let mut out = vec![f64::NAN; (SIDE * SIDE) as usize];
        if w == 1 {
            for (o, &h) in out.iter_mut().zip(&self.h) {
                *o = h as f64;
            }
            return out;
        }
        let height = |x: i32, z: i32| self.height(x, z);
        let inset = 2 * w; // room for the neighbours a summit looks at
        for cj in (inset..SIDE - inset).step_by(w as usize) {
            for ci in (inset..SIDE - inset).step_by(w as usize) {
                let t = far_top(&height, rule, self.x0 + ci, self.z0 + cj, w) as f64;
                for b in 0..w {
                    for a in 0..w {
                        out[((cj + b) * SIDE + ci + a) as usize] = t;
                    }
                }
            }
        }
        out
    }
}

/// Lines of sight through the region: along x, along z and along both
/// diagonals, every [`LINE_EVERY`]th.
fn sight_lines() -> Vec<Vec<(i32, i32)>> {
    let mut lines: Vec<Vec<(i32, i32)>> = Vec::new();
    for k in (PAD..PAD + REGION).step_by(LINE_EVERY as usize) {
        lines.push((0..SIDE).map(|n| (n, k)).collect());
        lines.push((0..SIDE).map(|n| (k, n)).collect());
    }
    for k in (-REGION..REGION).step_by(LINE_EVERY as usize * 2) {
        lines.push(
            (0..SIDE)
                .map(|n| (n, n - k))
                .filter(|&(_, j)| (0..SIDE).contains(&j))
                .collect(),
        );
        lines.push(
            (0..SIDE)
                .map(|n| (n, SIDE - 1 + k - n))
                .filter(|&(_, j)| (0..SIDE).contains(&j))
                .collect(),
        );
    }
    // Only what the region holds: the same columns for every width.
    let inside = |c: i32| (PAD..PAD + REGION).contains(&c);
    lines
        .into_iter()
        .map(|l| {
            l.into_iter()
                .filter(|&(i, j)| inside(i) && inside(j))
                .collect::<Vec<_>>()
        })
        .filter(|l| l.len() > 2 * CREST_MARGIN as usize)
        .collect()
}

fn skyline(tops: &[f64], line: &[(i32, i32)]) -> (f64, usize, usize) {
    let at = |n: usize| tops[(line[n].1 * SIDE + line[n].0) as usize];
    let vals: Vec<f64> = (0..line.len()).map(at).collect();
    let high = vals
        .iter()
        .copied()
        .filter(|v| !v.is_nan())
        .fold(f64::MIN, f64::max);
    let first = vals.iter().position(|&v| v == high).unwrap_or(0);
    let last = vals.iter().rposition(|&v| v == high).unwrap_or(0);
    (high, first, last)
}

#[derive(Default, Clone)]
struct Pool {
    surface: Vec<f64>,
    outline: Vec<f64>,
}

#[derive(Default, Clone)]
struct Peaks {
    column: Vec<f64>,
    peak: Vec<f64>,
}

struct Studied {
    /// [rule][kind][edge]
    change: Vec<Vec<[Pool; 4]>>,
    /// [rule][kind][width 8,16,32,64,128] against level 0
    against0: Vec<Vec<[Peaks; 5]>>,
}

const WIDTHS: [i32; 5] = [8, 16, 32, 64, 128];

fn study(gen: &WorldGen, footprints: &[(Terrain, Footprint)]) -> Studied {
    let lines = sight_lines();
    let kinds = TERRAINS.len();
    let mut out = Studied {
        change: vec![vec![Default::default(); kinds]; FAR_RULES.len()],
        against0: vec![vec![Default::default(); kinds]; FAR_RULES.len()],
    };
    let region = PAD as usize..(PAD + REGION) as usize;
    for &(kind, fp) in footprints {
        let k = TERRAINS.iter().position(|&t| t == kind).expect("a kind");
        let window = Window::around(gen, fp);
        let level0 = window.tops(FarRule::TopRule, 1);
        let crests: Vec<&Vec<(i32, i32)>> = lines
            .iter()
            .filter(|line| {
                let (_, first, last) = skyline(&level0, line);
                let m = CREST_MARGIN as usize;
                first >= m && last + m < line.len()
            })
            .collect();
        let peak_of = |tops: &[f64], inside: bool| {
            (0..SIDE * SIDE)
                .filter(|&c| {
                    let (i, j) = ((c % SIDE) as usize, (c / SIDE) as usize);
                    !inside || (region.contains(&i) && region.contains(&j))
                })
                .map(|c| tops[c as usize])
                .filter(|v| !v.is_nan())
                .fold(f64::MIN, f64::max)
        };
        let summit = peak_of(&level0, true) == peak_of(&level0, false);
        for (r, &rule) in FAR_RULES.iter().enumerate() {
            let tops: Vec<Vec<f64>> = WIDTHS.iter().map(|&w| window.tops(rule, w)).collect();
            let at =
                |w: i32| tops[WIDTHS.iter().position(|&x| x == w).expect("a width")].as_slice();
            for (e, &(fine, coarse, _)) in EDGES.iter().enumerate() {
                let (c, f) = (at(coarse), at(fine));
                let pool = &mut out.change[r][k][e];
                for j in region.clone() {
                    for i in region.clone() {
                        let d = c[j * SIDE as usize + i] - f[j * SIDE as usize + i];
                        if !d.is_nan() {
                            pool.surface.push(d);
                        }
                    }
                }
                for line in &crests {
                    let d = skyline(c, line).0 - skyline(f, line).0;
                    if !d.is_nan() {
                        pool.outline.push(d);
                    }
                }
            }
            for (a, level) in out.against0[r][k].iter_mut().zip(&tops) {
                for j in region.clone() {
                    for i in region.clone() {
                        let c = j * SIDE as usize + i;
                        let d = level[c] - level0[c];
                        if !d.is_nan() {
                            a.column.push(d);
                        }
                    }
                }
                if summit {
                    a.peak.push(peak_of(level, true) - peak_of(&level0, true));
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

/// **The far rule the study chose is the game's**: this file's copy gives
/// every quad the top `WorldGen::cell_top_sampled` gives it, at widths 16 to
/// 128, over a window of each kind of terrain.
#[test]
fn the_chosen_far_rule_is_the_games() {
    let gen = WorldGen::new(cubara_world::World::new().seed());
    let footprints = draw_footprints(&gen, FOOTPRINT_SEED, 1);
    let (mut cells, mut apart) = (0, 0);
    for (_, fp) in footprints {
        let window = Window::around(&gen, fp);
        let height = |x: i32, z: i32| window.height(x, z);
        for w in [8, 16, 32, 64, 128] {
            let inset = 2 * w;
            for cj in (inset..SIDE - inset).step_by(w as usize) {
                for ci in (inset..SIDE - inset).step_by(w as usize) {
                    let (x, z) = (window.x0 + ci, window.z0 + cj);
                    let n = if w <= 8 { w } else { SAMPLES };
                    assert_eq!(
                        far_top(&height, CHOSEN, x, z, w),
                        gen.cell_top_sampled(x, z, w, n),
                        "the cell of {w} at ({x}, {z})"
                    );
                    cells += 1;
                    apart += (far_top(&height, FarRule::Mean, x, z, w)
                        != far_top(&height, CHOSEN, x, z, w)) as usize;
                }
            }
        }
    }
    assert!(cells > 200 && apart > 5, "{cells} cells, {apart} summits");
}

/// The rule chosen: see `WorldGen::cell_top_sampled`.
pub const CHOSEN: FarRule = FarRule::TopRule;

/// The study, printed.
#[test]
#[ignore]
fn far_height_rule_study_report() {
    let gen = WorldGen::new(cubara_world::World::new().seed());
    let footprints = draw_footprints(&gen, FOOTPRINT_SEED, PER_KIND);
    let studied = study(&gen, &footprints);
    println!("footprint seed {FOOTPRINT_SEED:#x}; region {REGION}, pad {PAD}; samples {SAMPLES}");
    let px = |e: usize, blocks: f64| blocks * px_per_block(1080, EDGES[e].2);
    let edges =
        |f: &dyn Fn(usize) -> String| (0..EDGES.len()).map(f).collect::<Vec<_>>().join(" / ");
    let mut all = vec![vec![Pool::default(); EDGES.len()]; FAR_RULES.len()];
    for (k, &kind) in TERRAINS.iter().enumerate() {
        println!(
            "\n### {} (pixels at 1080 rows; edges 8>16 @1032, 16>32 @2064, 32>64 @4128, 64>128 @8256)\n",
            kind_name(kind)
        );
        println!("| rule | surface mean | surface p95 | outline mean | outline p95 | peak vs level 0, blocks: w8 / 16 / 32 / 64 / 128 |");
        println!("|---|---|---|---|---|---|");
        for (r, &rule) in FAR_RULES.iter().enumerate() {
            let c = &studied.change[r][k];
            for e in 0..EDGES.len() {
                all[r][e].surface.extend(&c[e].surface);
                all[r][e].outline.extend(&c[e].outline);
            }
            let peaks: Vec<String> = (0..WIDTHS.len())
                .map(|l| {
                    let a = &studied.against0[r][k][l];
                    if a.peak.is_empty() {
                        "-".into()
                    } else {
                        format!("{:+.1}", spread(&a.peak).0)
                    }
                })
                .collect();
            println!(
                "| {} | {} | {} | {} | {} | {} ({}) |",
                rule.name(),
                edges(&|e| format!("{:+.1}", px(e, spread(&c[e].surface).0))),
                edges(&|e| format!("{:.1}", px(e, spread(&c[e].surface).2))),
                edges(&|e| format!("{:+.1}", px(e, spread(&c[e].outline).0))),
                edges(&|e| format!("{:.1}", px(e, spread(&c[e].outline).2))),
                peaks.join(" / "),
                studied.against0[r][k][0].peak.len(),
            );
        }
    }
    println!("\n### all kinds pooled\n");
    println!("| rule | surface mean | surface p95 | outline mean | outline p95 |");
    println!("|---|---|---|---|---|");
    for (r, &rule) in FAR_RULES.iter().enumerate() {
        let c = &all[r];
        println!(
            "| {} | {} | {} | {} | {} |",
            rule.name(),
            edges(&|e| format!("{:+.1}", px(e, spread(&c[e].surface).0))),
            edges(&|e| format!("{:.1}", px(e, spread(&c[e].surface).2))),
            edges(&|e| format!("{:+.1}", px(e, spread(&c[e].outline).0))),
            edges(&|e| format!("{:.1}", px(e, spread(&c[e].outline).2))),
        );
    }
    println!("\nblocks, surface mean / p95 per edge, pooled:");
    for (r, &rule) in FAR_RULES.iter().enumerate() {
        println!(
            "{:<30} {}",
            rule.name(),
            edges(&|e| format!(
                "{:+.2} / {:.0}",
                spread(&all[r][e].surface).0,
                spread(&all[r][e].surface).2
            ))
        );
    }
}

/// The mountain the zoom test looks at (`--eye` on a line towards it, the
/// same ground from 3,000 blocks down to 100): this seed's peak, and the
/// low corner of the square of ground the two reports below measure.
const ZOOM_PEAK: [i32; 2] = [-3072, -3232];

/// The mean of the columns `WorldGen::cell_top_sampled(x0, z0, w, n)`
/// samples, half to even: the rule without its summits, at any width.
fn sampled_mean(height: &impl Fn(i32, i32) -> i32, x0: i32, z0: i32, w: i32, n: i32) -> i32 {
    let count = n.clamp(1, w);
    let step = w / count;
    let lead = (step - 1) / 2;
    let sum: i64 = (0..count)
        .flat_map(|b| (0..count).map(move |a| (a, b)))
        .map(|(a, b)| height(x0 + lead + a * step, z0 + lead + b * step) as i64)
        .sum();
    div_round_half_even(sum, (count * count) as i64) as i32
}

/// **Where the spires on a far ridge come from** (the zoom test of
/// 2026-11-21: the far terrain at 1,200 blocks, enlarged, against the same
/// ground at 400 drawn by level 2 -- "thin vertical spires on peak tops and
/// ridges"). Per cell width, over 1,792 blocks square around the peak: how
/// many cells stand at least their own width above all four neighbours, by
/// the rule as it is, by the mean alone, and with the summit's lift over the
/// mean capped -- and what each cap costs a summit in height.
///
/// A spire is the summit's lift, and it is the rule's at every width: a
/// cell that holds a summit stands above its mean while the cells around
/// it stand at theirs. At 8 blocks that is level 3's cell and the far
/// terrain's finest quad alike (one number, `far::quad_top`), so there was
/// no far rule to bring into line: when this was first run a summit stood
/// at its highest column (#432) and the rule left 87, 53, 24 and 9 such
/// cells at widths 4 to 32; #437 moved the voxels' summit to the mean of
/// its highest half, the far terrain with it, and this prints 33, 12, 8
/// and 5 (`far::block_tests::a_far_crest_grows_few_towers` holds the last
/// three). The caps are of the lift that is left.
///
/// ```text
/// cargo test --release -p cubara-render --test lod_consistency -- --ignored --nocapture summit_lift_report
/// ```
#[test]
#[ignore]
fn summit_lift_report() {
    use std::collections::HashMap;
    let gen = WorldGen::new(cubara_world::World::new().seed());
    const SPAN: i32 = 1792;
    let (x_lo, z_lo) = (ZOOM_PEAK[0] - 1024, ZOOM_PEAK[1] - 1120);
    let height = |x: i32, z: i32| gen.surface_height(x, z);
    println!("| cell width | summits lifted (mean, most) | cells a width above all four neighbours: rule / mean / lift capped at w/2 / at w/4 | a summit lowered by the cap w/2 (mean, most) | by w/4 |");
    println!("|---|---|---|---|---|");
    for w in [4, 8, 16, 32] {
        let n = if w <= 8 { w } else { SAMPLES };
        let mut tops: [HashMap<(i32, i32), i32>; 4] = Default::default();
        let (mut lifts, mut lost) = (Vec::new(), [Vec::new(), Vec::new()]);
        for cz in 0..SPAN / w {
            for cx in 0..SPAN / w {
                let (x0, z0) = (x_lo + cx * w, z_lo + cz * w);
                let rule = gen.cell_top_sampled(x0, z0, w, n);
                let mean = sampled_mean(&height, x0, z0, w, n);
                let capped = [rule.min(mean + w / 2), rule.min(mean + w / 4)];
                for (map, top) in tops.iter_mut().zip([rule, mean, capped[0], capped[1]]) {
                    map.insert((cx, cz), top);
                }
                if rule > mean {
                    lifts.push(rule - mean);
                    lost[0].push(rule - capped[0]);
                    lost[1].push(rule - capped[1]);
                }
            }
        }
        let spires = |map: &HashMap<(i32, i32), i32>| {
            map.iter()
                .filter(|(&(cx, cz), &top)| {
                    [(-1, 0), (1, 0), (0, -1), (0, 1)]
                        .iter()
                        .all(|(dx, dz)| map.get(&(cx + dx, cz + dz)).is_some_and(|&n| top - n >= w))
                })
                .count()
        };
        let stat = |v: &[i32]| {
            format!(
                "{:.1}, {}",
                v.iter().sum::<i32>() as f64 / v.len().max(1) as f64,
                v.iter().max().copied().unwrap_or(0)
            )
        };
        println!(
            "| {w} | {} of {} cells ({}) | {} / {} / {} / {} | {} | {} |",
            lifts.len(),
            tops[0].len(),
            stat(&lifts),
            spires(&tops[0]),
            spires(&tops[1]),
            spires(&tops[2]),
            spires(&tops[3]),
            stat(&lost[0]),
            stat(&lost[1]),
        );
    }
}

/// **What the grey patches on a slope are** (the same zoom test: "grey
/// stone patches that level 3 shows are missing"): not a riser's material.
/// Over 1,024 blocks square around the peak, level 0's risers are almost
/// never stone, while one column in twenty shows stone from above: the floor
/// of a pit a cave has opened. A coarse cell is a cave only where all its
/// ground is (`WorldGen::row_is_cave`, PHASE1 section 8.6), so the pits thin
/// out cell width by cell width, and the far terrain -- a height field
/// (`WorldGen::ground_faces`: "caves are not in it") -- has none.
///
/// The share of the ground every column of whose cell a cave opens is an
/// upper bound on what a level of that cell width can still show as a pit;
/// it is not the voxel rule itself, which judges row by row.
///
/// ```text
/// cargo test --release -p cubara-render --test lod_consistency -- --ignored --nocapture pits_by_cell_width_report
/// ```
#[test]
#[ignore]
fn pits_by_cell_width_report() {
    let gen = WorldGen::new(cubara_world::World::new().seed());
    const SPAN: i32 = 1024;
    let (x_lo, z_lo) = (ZOOM_PEAK[0] - 512, ZOOM_PEAK[1] - 512);
    // A column a cave opens: its surface block carved away; and how far
    // down the first block left is.
    let depth_of = |x: i32, z: i32| -> Option<i32> {
        let surface = gen.surface_height(x, z);
        if gen.is_solid(x, surface, z) {
            return None;
        }
        Some(
            (1..64)
                .find(|&d| gen.is_solid(x, surface - d, z))
                .unwrap_or(64),
        )
    };
    let open: Vec<Option<i32>> = (0..SPAN * SPAN)
        .map(|k| depth_of(x_lo + k % SPAN, z_lo + k / SPAN))
        .collect();
    let columns = open.len() as f64;
    let opened = open.iter().flatten().count() as f64;
    let stone = open
        .iter()
        .flatten()
        .filter(|&&d| d > cubara_world::SOIL_DEPTH)
        .count() as f64;
    let risers = gen.ground_faces(x_lo, z_lo, SPAN, SPAN);
    let by_layer: Vec<f64> = (0..3)
        .map(|l| risers.sides.iter().map(|s| s[l] as f64).sum())
        .collect();
    let faces: f64 = by_layer.iter().sum();
    println!(
        "level 0, {SPAN} blocks square around the peak: risers {:.2}% surface block, {:.2}% soil, {:.3}% stone; a cave opens {:.2}% of the columns, and under {:.2}% of all columns the first block down is stone",
        100.0 * by_layer[0] / faces,
        100.0 * by_layer[1] / faces,
        100.0 * by_layer[2] / faces,
        100.0 * opened / columns,
        100.0 * stone / columns,
    );
    println!("| cell width | share of the ground in cells whose every column a cave opens |");
    println!("|---|---|");
    for w in [1, 2, 4, 8, 16] {
        let n = SPAN / w;
        let all = (0..n * n)
            .filter(|k| {
                let (cx, cz) = (k % n, k / n);
                (0..w * w)
                    .all(|c| open[((cz * w + c / w) * SPAN + cx * w + c % w) as usize].is_some())
            })
            .count();
        println!("| {w} | {:.2}% |", 100.0 * all as f64 / (n * n) as f64);
    }
}

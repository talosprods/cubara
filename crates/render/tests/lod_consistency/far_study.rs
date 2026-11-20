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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FarRule {
    /// What `WorldGen::cell_top_sampled` is today: the mean of the samples,
    /// the highest of them where the quad holds a summit among its eight
    /// neighbours.
    TopRule,
    /// The mean of the samples, half to even: no summits.
    Mean,
    /// Summit judged among the 24 cells within two cells.
    Summit5,
    /// TopRule, the lift of a summit over the mean at most `w / 4` blocks.
    CapQuarter,
    /// The same at most `w / 8` blocks.
    CapEighth,
    /// Summit among 24, lift at most `w / 4`.
    Summit5CapQuarter,
    /// TopRule over 8 x 8 samples instead of 4 x 4 (four times the heights).
    TopRule8,
}

pub const FAR_RULES: [FarRule; 7] = [
    FarRule::TopRule,
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
            FarRule::TopRule => "TopRule (today)",
            FarRule::Mean => "plain mean",
            FarRule::Summit5 => "summit among 24 cells",
            FarRule::CapQuarter => "summit lift <= w/4",
            FarRule::CapEighth => "summit lift <= w/8",
            FarRule::Summit5CapQuarter => "summit among 24, lift <= w/4",
            FarRule::TopRule8 => "TopRule, 8 x 8 samples",
        }
    }
    /// How many cells either side a summit is judged among, and the cap on
    /// its lift as a fraction of the width (0: none).
    fn parts(self) -> (i32, i32, bool) {
        match self {
            FarRule::TopRule => (1, 0, true),
            FarRule::Mean => (1, 0, false),
            FarRule::Summit5 => (2, 0, true),
            FarRule::CapQuarter => (1, 4, true),
            FarRule::CapEighth => (1, 8, true),
            FarRule::Summit5CapQuarter => (2, 4, true),
            FarRule::TopRule8 => (1, 0, true),
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
    let (reach, cap_div, summits) = if w <= 8 {
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
    for b in 0..count {
        for a in 0..count {
            let h = sample(0, 0, a, b);
            sum += h as i64;
            if h > high {
                (high, first) = (h, (a, b));
            }
        }
    }
    let mean = div_round_half_even(sum, (count * count) as i64) as i32;
    if mean == high || !summits {
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
    if cap_div > 0 && w > 8 {
        mean + (high - mean).min(w / cap_div)
    } else {
        high
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

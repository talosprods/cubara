//! The far terrain's rows of the consistency measurement (issue #422 on
//! #423's definition): every seeded footprint at every far quad width that
//! fits one, against level 0 over the same ground -- measured from the faces
//! the game's far mesher makes (`harness::measure_far`).

use super::*;

/// One footprint at one far level, against level 0 split on its quads.
pub struct FarOne {
    pub kind: Terrain,
    pub fp: Footprint,
    pub level: u32,
    pub level0: Faces,
    pub faces: Faces,
    pub delta: [Ranked; 5],
    pub shape: Shape,
    /// Whether the footprint's highest column is a summit (`holds_a_summit`,
    /// over every column: the far terrain draws no caves): the far quads of 8
    /// and under hold its height exactly.
    pub summit: bool,
}

impl FarOne {
    pub fn at(&self) -> String {
        format!(
            "{:?} at ({}, {}), far quads of {} (seed {FOOTPRINT_SEED:#x})",
            self.kind,
            self.fp.x0,
            self.fp.z0,
            1 << self.level
        )
    }

    /// The stone share of the side faces at level 0 and here, where level 0
    /// has the faces to rank a direction by and the far terrain has any.
    fn stone(&self) -> Option<(f64, f64)> {
        let sides = |f: &Faces| f.area[1..].iter().sum::<f64>();
        (sides(&self.level0) >= 2.0 * Rep::Far(self.level).min_area() && sides(&self.faces) > 0.0)
            .then(|| {
                (
                    self.level0.side_share(STONE).expect("sides"),
                    self.faces.side_share(STONE).expect("sides"),
                )
            })
    }
}

/// Every footprint at every far level, by `measure`, compared as `rep_of`
/// says a far level is.
fn far_rows(measure: fn(&Setup, Footprint, u32) -> Measure, rep_of: fn(u32) -> Rep) -> Vec<FarOne> {
    let setup = Setup::new();
    let footprints = draw_footprints(&setup.gen, FOOTPRINT_SEED, PER_KIND);
    assert_eq!(
        footprints.len(),
        PER_KIND * TERRAINS.len(),
        "seed {FOOTPRINT_SEED:#x}"
    );
    let mut all = Vec::new();
    for (kind, fp) in footprints {
        let drawn = reference(&setup, fp);
        for level in FAR_LEVELS {
            let level0 = drawn.measure(&setup, 0, Rep::Far(level).mask());
            let other = measure(&setup, fp, level);
            let Compared { delta, shape } = compare(&setup, fp, rep_of(level), &level0, &other);
            all.push(FarOne {
                kind,
                fp,
                level,
                level0: level0.faces,
                faces: other.faces,
                delta,
                shape,
                // Over the columns the shape is taken on: the quad's
                // surface term (`compare`).
                summit: holds_a_summit(
                    &setup,
                    fp,
                    &setup.on_the_surface(fp, Rep::Far(level).mask()),
                ),
            });
        }
    }
    all
}

/// The far terrain as the game draws it, measured once for the tests below.
fn far_footprints() -> &'static [FarOne] {
    static ALL: std::sync::OnceLock<Vec<FarOne>> = std::sync::OnceLock::new();
    ALL.get_or_init(|| far_rows(measure_far, Rep::Far))
}

fn print_rows(title: &str, all: &[FarOne]) {
    let sorted = |mut v: Vec<f64>, signed: bool| {
        v.sort_by(f64::total_cmp);
        v.iter()
            .map(|v| {
                if signed {
                    format!("{v:+.2}")
                } else {
                    format!("{v:.2}")
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    };
    println!("\n######## {title} ########");
    for kind in TERRAINS {
        for level in FAR_LEVELS {
            let ones: Vec<&FarOne> = all
                .iter()
                .filter(|o| o.kind == kind && o.level == level)
                .collect();
            println!("\n== {kind:?}, far quads of {} ==", 1 << level);
            let (mut level0, mut faces) = (Faces::default(), Faces::default());
            for o in &ones {
                level0.add(&o.level0);
                faces.add(&o.faces);
            }
            for d in 0..5 {
                let deltas: Vec<f64> = ones
                    .iter()
                    .filter_map(|o| match o.delta[d] {
                        Ranked::Delta(delta) => Some(delta),
                        _ => None,
                    })
                    .collect();
                let missing = ones
                    .iter()
                    .filter(|o| matches!(o.delta[d], Ranked::Missing { .. }))
                    .count();
                let pooled = match (level0.mean_lit(d), faces.mean_lit(d)) {
                    (Some(a), Some(b)) => format!("{:.2}", colour_delta(a, b)),
                    _ => "-".to_string(),
                };
                println!(
                    "  T3 {}: pooled {pooled}; {} ranked, {missing} missing, {} over {T3_TOLERANCE}; area x{:.2} of level 0's; each: {}",
                    direction_name(d),
                    deltas.len(),
                    deltas.iter().filter(|v| **v > T3_TOLERANCE).count(),
                    faces.area[d] / level0.area[d],
                    sorted(deltas.clone(), false),
                );
            }
            let stone: Vec<String> = ones
                .iter()
                .filter_map(|o| o.stone())
                .map(|(a, b)| format!("{:.1}->{:.1}", a * 100.0, b * 100.0))
                .collect();
            println!(
                "  stone share of sides, %, level 0 -> here, each: {}",
                stone.join(" ")
            );
            let share = |f: &Faces, m: usize| f.side_share(m).map_or(f64::NAN, |v| v * 100.0);
            println!(
                "  sides pooled, % surface block / soil / stone: level 0 {:.1} / {:.1} / {:.1}, here {:.1} / {:.1} / {:.1}",
                share(&level0, SURFACE),
                share(&level0, SOIL),
                share(&level0, STONE),
                share(&faces, SURFACE),
                share(&faces, SOIL),
                share(&faces, STONE),
            );
            let each = |f: &dyn Fn(&FarOne) -> f64, signed: bool| {
                sorted(ones.iter().map(|o| f(o)).collect(), signed)
            };
            println!(
                "  shape, mean column: {}",
                each(&|o| spread(&o.shape.column).0, true)
            );
            println!(
                "  shape, skyline mean distance: {}",
                each(&|o| spread(&o.shape.skyline).1, false)
            );
            println!(
                "  shape, skyline mean (signed): {}",
                each(&|o| spread(&o.shape.skyline).0, true)
            );
            println!(
                "  shape, peak: {}",
                each(&|o| spread(&o.shape.peak).0, true)
            );
            println!(
                "  undrawn columns: {}",
                ones.iter().map(|o| o.shape.undrawn).sum::<usize>()
            );
        }
    }
}

/// The numbers of record for the far rows, footprint by footprint:
///
/// ```text
/// cargo test -p cubara-render --test lod_consistency -- --ignored --nocapture far_rows_report
/// ```
#[test]
#[ignore]
fn far_rows_report() {
    println!("footprint seed {FOOTPRINT_SEED:#x}; the surface term; T3 in /255");
    print_rows("AFTER: blocks (#422)", far_footprints());
}

/// The far quads the game draws that fit a footprint at least sixteen times
/// over: 8 blocks (the join: level 3's cell, and the finest quad there is,
/// [`cubara_world::far::FINEST_LEVEL`]) and 16. Wider quads are in the
/// report as an indication: a footprint holds four of 32 and one of 64.
const FAR_HELD: [u32; 2] = [3, 4];

/// **The far terrain's colours are level 0's** (#422 on #423's T3), in every
/// single footprint of every kind of terrain: the lit colour of every face
/// direction within the bound of level 3 (`T3_ALLOWED`'s 3.0 for the tops
/// and 3.25 for the sides -- the far terrain meets level 3 at the join and
/// is held to its standard), a direction level 0 has and the far terrain
/// lacks failing outright; **the stone share of the sides within one and a
/// quarter points of level 0's** in every footprint (the voxel levels'
/// bound); the soil share per kind of terrain within [`SOIL_POINTS`].
///
/// **Chosen, not derived**, just over the measured worst on this seed (tops
/// 2.59 and sides 2.42 at quads of 16 on a steep footprint, stone 1.1
/// points, soil 3.8 points on the cliffs: a cliff's walls are few).
#[test]
fn the_far_terrain_shows_level_0s_colours() {
    const TOPS: f64 = 3.0;
    const SIDES: f64 = 3.25;
    const STONE_POINTS: f64 = 0.0125;
    const SOIL_POINTS: f64 = 0.05;
    let all = far_footprints();
    let (mut ranked, mut stone_held, mut missing) = (0, 0, 0);
    for one in all.iter().filter(|o| FAR_HELD.contains(&o.level)) {
        for (d, delta) in one.delta.iter().enumerate() {
            let allowed = if d == 0 { TOPS } else { SIDES };
            // A far quad of 16 is wide enough for level 0's steps to be
            // inside it: a direction level 0 has can leave it no wall of
            // its own (one footprint of forty on this seed, a cliff's +Z).
            // The quads of 8 are level 3's cells and lose none.
            if one.level == 4 && matches!(delta, Ranked::Missing { .. }) {
                missing += 1;
                continue;
            }
            let Some(delta) = delta.delta() else {
                continue;
            };
            ranked += 1;
            assert!(
                delta <= allowed,
                "{}, {}: {delta:.2}/255 off level 0's, allowed {allowed} ({:?})",
                one.at(),
                direction_name(d),
                one.delta[d]
            );
        }
        if let Some((near, far)) = one.stone() {
            stone_held += 1;
            assert!(
                (far - near).abs() <= STONE_POINTS,
                "{}: {:.1}% of the walls are stone, level 0 has {:.1}%",
                one.at(),
                far * 100.0,
                near * 100.0
            );
        }
    }
    assert!(
        missing <= 2,
        "{missing} directions level 0 has the far terrain lacks"
    );
    assert!(
        ranked >= 150 && stone_held >= 30,
        "{ranked} directions ranked, {stone_held} footprints with the walls to compare"
    );
    for kind in TERRAINS {
        for level in FAR_HELD {
            let share = |f: fn(&FarOne) -> &Faces| {
                let (mut soil, mut area) = (0.0, 0.0);
                for one in all.iter().filter(|o| o.kind == kind && o.level == level) {
                    area += f(one).area[1..].iter().sum::<f64>();
                    soil += f(one).material[1..].iter().map(|m| m[SOIL]).sum::<f64>();
                }
                soil / area
            };
            let (near, far) = (share(|o| &o.level0), share(|o| &o.faces));
            assert!(
                (far - near).abs() <= SOIL_POINTS,
                "{kind:?}, far quads of {}: {:.1}% of the walls are soil, level 0 has {:.1}% (seed {FOOTPRINT_SEED:#x})",
                1 << level,
                far * 100.0,
                near * 100.0
            );
        }
    }
}

/// **The far terrain keeps level 0's shape** (#423 decision 4 on the far
/// quads), in every single footprint: every column drawn; at quads of 8 --
/// the same function as level 3's cells -- **a summit stands where level 3's
/// does**: never above level 0's, and under it by no more than level 3's
/// bounds on a footprint and in the mean (`a_coarse_level_keeps_level_0s_shape`'s
/// `SUMMIT`: a coarse summit stands at the mean of its highest half-width
/// cell since #437, where its highest column drew towers), and the ground
/// leans neither way nor off the skyline more than level 3 allows (`SHAPE`'s
/// third row); at quads of 16, sampled four to a side, within the bounds
/// below.
///
/// **Chosen, not derived**, just over the worst footprint measured on this
/// seed: at 16 the mean column stands -1.31 to +1.91 blocks off level 0's
/// and the skyline's mean distance reaches 17.3 (cliffs); at 8 the fifteen
/// footprints that hold a summit stand 2.13 blocks under level 0's in the
/// mean. The swap that matters, between far quad widths, is `far_study`'s,
/// in pixels.
#[test]
fn the_far_terrain_keeps_level_0s_shape() {
    /// Per far level held: the most the mean column may stand off level 0,
    /// and the most the skyline's mean distance may be.
    const SHAPE: [(u32, f64, f64); 2] = [(3, 0.3, 7.5), (4, 2.5, 18.0)];
    /// At quads of 8: the most a summit may stand under level 0's on any one
    /// footprint, and in the mean over the footprints that hold one --
    /// level 3's bounds, to the number.
    const SUMMIT: (f64, f64) = (6.0, 2.5);
    let (mut summits, mut summit_sum) = (0, 0.0);
    for one in far_footprints() {
        let Some(&(_, lean_bound, skyline_bound)) = SHAPE.iter().find(|s| s.0 == one.level) else {
            continue;
        };
        let at = one.at();
        assert_eq!(
            one.shape.undrawn, 0,
            "{at}: columns of the surface not drawn"
        );
        let (peak, ..) = spread(&one.shape.peak);
        if one.summit && one.level == 3 {
            summits += 1;
            summit_sum += peak;
            assert!(
                (-SUMMIT.0..=0.0).contains(&peak),
                "{at}: the summit stands {peak:+.2} blocks off level 0's, allowed {} under and none over",
                SUMMIT.0
            );
        }
        let (lean, ..) = spread(&one.shape.column);
        assert!(
            lean.abs() <= lean_bound,
            "{at}: the ground stands {lean:+.2} blocks off level 0's on average, allowed {lean_bound}"
        );
        let (_, skyline, ..) = spread(&one.shape.skyline);
        assert!(
            skyline <= skyline_bound,
            "{at}: the skyline is {skyline:.2} blocks off, allowed {skyline_bound}"
        );
    }
    assert!(summits >= 8, "{summits} footprints hold a summit");
    let mean = summit_sum / summits as f64;
    println!("far quads of 8: {summits} summits, {mean:+.2} blocks off level 0's in the mean");
    assert!(
        mean >= -SUMMIT.1,
        "far quads of 8: the summits stand {mean:+.2} blocks off level 0's in the mean, allowed {} under",
        SUMMIT.1
    );
}

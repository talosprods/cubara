//! **Every level shows the same ground as level 0** (issue #423): the
//! definition, as a measurement.
//!
//! Over a seeded set of footprints -- flat land, gentle grassy slopes, steep
//! mountain faces and cliffs -- each coarse voxel level (1-3) and each far
//! terrain level is measured against the level-0 blocks over the same
//! footprint:
//!
//! - **shape**: the drawn top over each column, the skyline and the peak,
//!   against level 0's, in blocks (#423 decision 4);
//! - **T3**: the area-weighted lit colour per face direction (+Y, ±X, ±Z)
//!   (#423 decision 3, `docs/PROPOSAL_LOD_TRANSITIONS.md` §6);
//! - **the stone share** per face direction -- the owner's play-test of
//!   2026-11-14: "from afar it is mostly stone; right in front of it, much
//!   more grass".
//!
//! No GPU. `harness.rs` says how each representation is measured and why
//! only sky-visible faces count.
//!
//! The full table is an `#[ignore]`d report:
//!
//! ```text
//! cargo test -p cubara-render --test lod_consistency -- --ignored --nocapture consistency_report
//! ```

mod harness;

use cubara_render::aggregate::face_light;
use glam::Vec3;
use harness::*;

/// The seed the footprints are drawn from. Printed by every failure.
const FOOTPRINT_SEED: u64 = 0x0423_C0FF_EE00_0001;

/// Footprints of each kind of terrain: forty in all.
const PER_KIND: usize = 10;

/// The coarse voxel levels.
const VOXEL_LEVELS: [u32; 3] = [1, 2, 3];

/// The 2/255 of #423 decision 3, on the scale `colour_delta` reports.
const T3_TOLERANCE: f64 = 2.0;

/// The far terrain's levels measured: quads of 1 to 64 blocks. Past 16
/// blocks a footprint holds fewer than sixteen quads, so those rows are an
/// indication rather than a statistic.
const FAR_LEVELS: [u32; 7] = [0, 1, 2, 3, 4, 5, 6];

/// The least face area a colour is taken from, in square blocks: a handful
/// of faces is not a colour. **Chosen, not derived.**
const MIN_AREA: f64 = 64.0;

/// The fewest of a level's own cell faces a colour is taken from: each is
/// one material and one shade, so a mean over a few of them is a sample of
/// a few. A level-3 cell's side is 32 square blocks, and [`MIN_AREA`] of
/// them would be two faces. **Chosen, not derived.**
const MIN_FACES: f64 = 16.0;

/// One direction of one footprint, one representation against level 0 over
/// the same ground.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Ranked {
    /// Level 0 has too few faces that way for there to be a colour to hold
    /// the level to: under twice [`Rep::min_area`].
    Unranked,
    /// Level 0 has the faces and the level does not (under
    /// [`Rep::min_area`] of them): **a failure**, not a direction to pass
    /// over -- a level that draws no risers on a hillside is not showing the
    /// same ground (second review of #428, point 3).
    Missing { level0: f64, area: f64 },
    /// T3: the difference of the two lit colours, /255.
    Delta(f64),
}

impl Ranked {
    /// The colour difference, a missing direction as an infinite one;
    /// `None` when unranked.
    fn delta(self) -> Option<f64> {
        match self {
            Ranked::Unranked => None,
            Ranked::Missing { .. } => Some(f64::INFINITY),
            Ranked::Delta(d) => Some(d),
        }
    }
}

/// One footprint's numbers for one representation, against level 0.
struct Compared {
    /// T3 per direction on the surface term.
    delta: [Ranked; 5],
    shape: Shape,
}

/// `other`, a measurement of `rep`, against `level0` -- level 0 over the
/// same footprint **split on `rep`'s cells** ([`Rep::mask`]), so that both
/// cover the same ground.
fn compare(setup: &Setup, fp: Footprint, rep: Rep, level0: &Measure, other: &Measure) -> Compared {
    let mut delta = [Ranked::Unranked; 5];
    let min_area = rep.min_area();
    for (d, slot) in delta.iter_mut().enumerate() {
        let (a, b) = (level0.faces.area[d], other.faces.area[d]);
        *slot = if a < 2.0 * min_area {
            Ranked::Unranked
        } else if b < min_area {
            Ranked::Missing { level0: a, area: b }
        } else {
            Ranked::Delta(colour_delta(
                level0.faces.mean_lit(d).expect("area"),
                other.faces.mean_lit(d).expect("area"),
            ))
        };
    }
    // The shape on the same ground: a voxel level draws its own caves, so
    // it is compared over the columns of its surface term; the far terrain
    // and the stepped models draw none, and neither does the height field,
    // so every column of theirs is the surface.
    let field = height_field(&setup.gen, fp);
    let shape = match rep {
        Rep::Voxel(_) => Shape::over(&field, other, &setup.on_the_surface(fp, rep.mask())),
        Rep::Far(_) | Rep::Stepped(..) => Shape::of(&field, other),
    };
    Compared { delta, shape }
}

/// One representation's numbers for one kind of terrain: pooled over its
/// footprints, and the worst single footprint beside each (review of #428,
/// finding B3: a pooled mean can hide a footprint that is far off).
#[derive(Clone, Default)]
struct Pool {
    /// The surface term: level 0's, and this representation's.
    level0: Faces,
    faces: Faces,
    /// The cave term, reported only.
    cave_level0: Faces,
    cave: Faces,
    shape: Shape,
    /// The worst single footprint's colour difference per direction, and
    /// in how many footprints the direction is missing ([`Ranked::Missing`]).
    worst_delta: [f64; 5],
    missing: [usize; 5],
    /// In how many footprints the direction was ranked at all.
    ranked: [usize; 5],
    /// The worst single footprint's shape: the largest magnitude of the
    /// mean column difference, the largest mean skyline distance, the
    /// largest peak difference.
    worst_column: f64,
    worst_skyline: f64,
    worst_peak: f64,
}

impl Pool {
    fn add(&mut self, setup: &Setup, fp: Footprint, rep: Rep, level0: &Measure, other: &Measure) {
        self.level0.add(&level0.faces);
        self.faces.add(&other.faces);
        self.cave_level0.add(&level0.cave);
        self.cave.add(&other.cave);
        let one = compare(setup, fp, rep, level0, other);
        for d in 0..5 {
            match one.delta[d] {
                Ranked::Unranked => {}
                Ranked::Missing { .. } => self.missing[d] += 1,
                Ranked::Delta(delta) => {
                    self.ranked[d] += 1;
                    self.worst_delta[d] = self.worst_delta[d].max(delta);
                }
            }
        }
        let finite = |v: f64| if v.is_nan() { 0.0 } else { v.abs() };
        self.worst_column = self.worst_column.max(finite(spread(&one.shape.column).0));
        self.worst_skyline = self.worst_skyline.max(finite(spread(&one.shape.skyline).1));
        self.worst_peak = self.worst_peak.max(finite(spread(&one.shape.peak).0));
        self.shape.add(&one.shape);
    }

    /// The pooled colour difference of direction `d`, on the 0-255 scale.
    fn delta(&self, d: usize) -> Option<f64> {
        Some(colour_delta(
            self.level0.mean_lit(d)?,
            self.faces.mean_lit(d)?,
        ))
    }
}

/// Which representation a row of the report is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Rep {
    Voxel(u32),
    Far(u32),
    /// A model, not something the game draws: a stepped height field of
    /// squares this wide, its tops by this rule.
    Stepped(i32, TopRule),
}

impl std::fmt::Display for Rep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Rep::Voxel(l) => write!(f, "voxel L{l} (cell {}x{})", 1 << l, cell_height(*l)),
            Rep::Far(l) => write!(f, "far L{l} (quad {})", 1 << l),
            Rep::Stepped(cell, rule) => write!(f, "stepped {cell}: {rule:?}"),
        }
    }
}

fn cell_height(level: u32) -> i32 {
    cubara_world::node::NodeKey::new(level, [0, 0, 0]).cell_height_blocks()
}

impl Rep {
    /// The width of the cells this representation's surface term is split
    /// on, in blocks -- and so the width level 0 is split on when it is the
    /// reference for it. (A stepped model has no faces to split.)
    fn mask(self) -> i32 {
        match self {
            Rep::Voxel(l) | Rep::Far(l) => (1 << l).min(FOOTPRINT),
            Rep::Stepped(..) => 1,
        }
    }

    /// The face area a direction needs for its colour to be taken, in
    /// square blocks: [`MIN_FACES`] of this representation's own cell faces
    /// and never under [`MIN_AREA`] -- 64 at voxel levels 0 to 2, 512 at
    /// level 3 (sixteen sides of 8 x 4). The far terrain has no faces: its
    /// shading is continuous, and [`MIN_AREA`] of it is as much of a colour
    /// as the same area of blocks.
    fn min_area(self) -> f64 {
        match self {
            Rep::Voxel(l) => MIN_AREA.max(MIN_FACES * ((1 << l) * cell_height(l)) as f64),
            Rep::Far(_) | Rep::Stepped(..) => MIN_AREA,
        }
    }
}

type Pools = std::collections::BTreeMap<(Terrain, Rep), Pool>;

/// Measure every footprint at every representation in `reps`.
fn measure_all(setup: &Setup, footprints: &[(Terrain, Footprint)], reps: &[Rep]) -> Pools {
    let mut pools = Pools::new();
    for &(kind, fp) in footprints {
        let level0 = reference(setup, fp);
        // Level 0 once per width it is split on.
        let mut split = std::collections::BTreeMap::<i32, Measure>::new();
        for &rep in reps {
            let other = match rep {
                Rep::Voxel(l) => measure_voxels(setup, fp, l),
                Rep::Far(l) => measure_far(setup, fp, l),
                Rep::Stepped(cell, rule) => measure_stepped(&setup.gen, fp, cell, rule),
            };
            let level0 = split
                .entry(rep.mask())
                .or_insert_with(|| level0.measure(setup, 0, rep.mask()));
            pools
                .entry((kind, rep))
                .or_default()
                .add(setup, fp, rep, level0, &other);
        }
    }
    pools
}

/// **The reference every representation is compared with: level 0**, the
/// game's own chunks at full detail, as the game meshes them --
/// `reference(..).measure(setup, 0, rep.mask())` is level 0 over the ground
/// `rep`'s surface term covers. Its surface term is what is asserted
/// against; `level_0_draws_its_surface` holds this very function to the
/// world, block by block.
fn reference(setup: &Setup, fp: Footprint) -> Drawn {
    draw_voxels(setup, fp, 0, &|node| {
        setup.world.node_at(node, setup.blocks)
    })
}

fn percent(v: Option<f64>) -> String {
    v.map_or("    -".into(), |v| format!("{:5.1}", v * 100.0))
}

fn report(pools: &Pools) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let models = |r: &Rep| matches!(r, Rep::Stepped(..));
    for kind in TERRAINS {
        let rows: Vec<_> = pools.iter().filter(|((k, _), _)| *k == kind).collect();
        // The whole surface: a row whose reference is split column by column.
        let Some((_, first)) = rows
            .iter()
            .find(|((_, r), _)| r.mask() == 1)
            .or(rows.first())
        else {
            continue;
        };
        let sides = |f: &Faces| f.area[1..].iter().sum::<f64>();
        let all = |f: &Faces| f.area.iter().sum::<f64>();
        writeln!(s, "\n== {kind:?} ==").unwrap();
        writeln!(
            s,
            "level 0, SURFACE (the reference): stone share of sides {}%, of all faces {}%; side area per unit of ground {:.3}",
            percent(first.level0.side_share(STONE)),
            percent(first.level0.total_share(STONE)),
            sides(&first.level0) / first.level0.area[0],
        )
        .unwrap();
        writeln!(
            s,
            "level 0, CAVES (reported, not asserted): {:.1}% of the sky-visible area is on or beside carved ground; stone share of its sides {}%",
            all(&first.cave_level0) * 100.0 / (all(&first.level0) + all(&first.cave_level0)),
            percent(first.cave_level0.side_share(STONE)),
        )
        .unwrap();
        writeln!(
            s,
            "\n  shape, drawn - level 0's height field, in blocks, a voxel level over its surface term's columns; pooled, then the worst single footprint"
        )
        .unwrap();
        writeln!(
            s,
            "  {:<24} {:>30} {:>30} {:>14}   worst: |column mean| skyline peak",
            "", "column: mean |mean| p95 max", "skyline: mean |mean| p95 max", "peak: mean max"
        )
        .unwrap();
        for ((_, rep), p) in &rows {
            let (c, k, pk) = (
                spread(&p.shape.column),
                spread(&p.shape.skyline),
                spread(&p.shape.peak),
            );
            writeln!(
                s,
                "  {:<24} {:>+7.2} {:>6.2} {:>6.2} {:>7.2} {:>+8.2} {:>6.2} {:>6.2} {:>7.2} {:>+7.2} {:>6.2}   {:>6.2} {:>6.2} {:>6.2}{}",
                rep.to_string(),
                c.0, c.1, c.2, c.3, k.0, k.1, k.2, k.3, pk.0, pk.3,
                p.worst_column, p.worst_skyline, p.worst_peak,
                match (p.shape.undrawn, p.shape.left_out) {
                    (0, 0) => String::new(),
                    (u, l) => format!("  ({u} columns undrawn, {l} on carved ground left out)"),
                },
            )
            .unwrap();
        }
        writeln!(
            s,
            "\n  T3 on the surface: lit colour against level 0 over the same ground per direction, /255 pooled (worst ranked footprint, - with none ranked); area / level 0's; !n = missing in n footprints"
        )
        .unwrap();
        for ((_, rep), p) in rows.iter().filter(|((_, r), _)| !models(r)) {
            write!(s, "  {:<24}", rep.to_string()).unwrap();
            for d in 0..5 {
                match p.delta(d) {
                    Some(delta) => write!(
                        s,
                        " {} {:5.2} ({}) x{:4.2}{}",
                        direction_name(d),
                        delta,
                        match p.ranked[d] {
                            0 => "    -".to_string(),
                            _ => format!("{:5.2}", p.worst_delta[d]),
                        },
                        p.faces.area[d] / p.level0.area[d],
                        match p.missing[d] {
                            0 => String::new(),
                            n => format!(" !{n}"),
                        }
                    )
                    .unwrap(),
                    None => write!(s, " {}     -               ", direction_name(d)).unwrap(),
                }
            }
            writeln!(s).unwrap();
        }
        writeln!(
            s,
            "\n  stone share on the surface, % of area: level 0 over the same ground -> this level, per direction; all sides; all faces; then the cave term's share of this level's area"
        )
        .unwrap();
        for ((_, rep), p) in rows.iter().filter(|((_, r), _)| !models(r)) {
            write!(s, "  {:<24}", rep.to_string()).unwrap();
            for d in 1..5 {
                write!(
                    s,
                    " {} {}->{}",
                    direction_name(d),
                    percent(p.level0.share(d, STONE)),
                    percent(p.faces.share(d, STONE))
                )
                .unwrap();
            }
            writeln!(
                s,
                "  sides {}->{}  all {}->{}  caves {:.1}% (stone of sides {}%)",
                percent(p.level0.side_share(STONE)),
                percent(p.faces.side_share(STONE)),
                percent(p.level0.total_share(STONE)),
                percent(p.faces.total_share(STONE)),
                all(&p.cave) * 100.0 / (all(&p.faces) + all(&p.cave)),
                percent(p.cave.side_share(STONE)),
            )
            .unwrap();
        }
    }
    s
}

/// The tops the far terrain's quads are to take from `WorldGen` (#422 on
/// #423's rule): the exact `cell_top` up to the level-3 cell's 8 blocks, so
/// the join at 1,024 m is the same number on both sides, and
/// `cell_top_sampled` with [`FAR_SAMPLES`] a side above that.
fn far_quad_tops() -> impl Iterator<Item = Rep> {
    [2, 4, 8, 16, 32, 64].into_iter().map(|quad| {
        Rep::Stepped(
            quad,
            if quad <= 8 {
                TopRule::Max
            } else {
                TopRule::SampledMax(FAR_SAMPLES)
            },
        )
    })
}

/// Sample columns a side for a far quad wider than 8 blocks: the far
/// terrain's own choice (#422), measured here.
const FAR_SAMPLES: i32 = 4;

fn all_reps() -> Vec<Rep> {
    // Level 0 against itself is the harness's own zero, and its shape row is
    // what caves do to the drawn surface.
    [0].iter()
        .chain(&VOXEL_LEVELS)
        .map(|&l| Rep::Voxel(l))
        .chain(FAR_LEVELS.iter().map(|&l| Rep::Far(l)))
        .chain(far_quad_tops())
        .collect()
}

/// The numbers of record: every level against level 0, over every footprint.
#[test]
#[ignore]
fn consistency_report() {
    let setup = Setup::new();
    let footprints = draw_footprints(&setup.gen, FOOTPRINT_SEED, PER_KIND);
    println!(
        "footprint seed {FOOTPRINT_SEED:#x}, world seed {:#x}, {} footprints of {FOOTPRINT} blocks",
        setup.world.seed(),
        footprints.len()
    );
    for (kind, fp) in &footprints {
        let (mean, max) = step_stats(&setup.gen, *fp);
        println!(
            "  {kind:?} at ({}, {}): mean step {mean:.2}, largest {max}",
            fp.x0, fp.z0
        );
    }
    let pools = measure_all(&setup, &footprints, &all_reps());
    println!("{}", report(&pools));
}

/// **For the far terrain's decision** (#423, #422): what a stepped height
/// field would draw under each rule for a quad's top, per quad width. The
/// max keeps every peak and can only raise the ground; the mean moves it
/// both ways and lowers the ridges. Measured, so the choice is not a guess.
#[test]
#[ignore]
fn stepped_top_rule_report() {
    let setup = Setup::new();
    let footprints = draw_footprints(&setup.gen, FOOTPRINT_SEED, PER_KIND);
    let reps: Vec<Rep> = [2, 4, 8, 16, 32, 64]
        .into_iter()
        .flat_map(|cell| {
            let mut rules = vec![TopRule::Max, TopRule::Mean];
            if cell > 8 {
                rules.push(TopRule::SampledMax(8));
            }
            if cell > 4 {
                rules.push(TopRule::SampledMax(FAR_SAMPLES));
            }
            rules.into_iter().map(move |rule| Rep::Stepped(cell, rule))
        })
        .collect();
    println!("footprint seed {FOOTPRINT_SEED:#x}");
    println!("{}", report(&measure_all(&setup, &footprints, &reps)));
}

/// What the kinds of terrain are chosen from: the spread of level-0 steps
/// over random footprints of the default world.
#[test]
#[ignore]
fn print_the_terrain_spread() {
    let setup = Setup::new();
    let mut rng = Rng(FOOTPRINT_SEED);
    let mut stats = Vec::new();
    for _ in 0..3000 {
        let fp = Footprint {
            x0: (rng.next() % 12_000) as i32 / FOOTPRINT * FOOTPRINT - 6000,
            z0: (rng.next() % 12_000) as i32 / FOOTPRINT * FOOTPRINT - 6000,
        };
        stats.push(step_stats(&setup.gen, fp));
    }
    stats.sort_by(|a, b| a.0.total_cmp(&b.0));
    for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99] {
        let (mean, max) = stats[stats.len() * pct / 100];
        println!("p{pct:02}: mean step {mean:.3}, largest step {max}");
    }
    for kind in TERRAINS {
        let n = stats
            .iter()
            .filter(|(mean, max)| classify(*mean, *max) == Some(kind))
            .count();
        println!("{kind:?}: {:.1}%", n as f64 * 100.0 / stats.len() as f64);
    }
}

/// The harness measures level 0 right, held to the generator: every column's
/// drawn top is its surface (but where a cave mouth has carved it away, and
/// those are few), and the sky sees exactly one top face per drawn column.
#[test]
fn level_0_draws_its_surface() {
    let setup = Setup::new();
    let footprints = draw_footprints(&setup.gen, FOOTPRINT_SEED, 2);
    assert_eq!(
        footprints.len(),
        2 * TERRAINS.len(),
        "seed {FOOTPRINT_SEED:#x}"
    );
    let mut grassy = 0;
    for (kind, fp) in footprints {
        // Through `reference`, the function every comparison takes level 0
        // from: this test is what says that reference IS level 0.
        let terms = reference(&setup, fp).measure(&setup, 0, 1);
        // Everything the sky sees, surface and caves together, for the
        // counts against the world; the two terms are held apart below.
        let m = {
            let mut all = terms.clone();
            all.faces.add(&terms.cave);
            all
        };
        let field = height_field(&setup.gen, fp);
        let (mut carved, mut drawn) = (0, 0);
        for (top, surface) in m.tops.iter().zip(&field.tops) {
            if top.is_nan() {
                carved += 1;
                continue;
            }
            drawn += 1;
            assert!(
                top <= surface,
                "{kind:?} {fp:?}: a top at {top} above its surface {surface} (seed {FOOTPRINT_SEED:#x})"
            );
            if top != surface {
                carved += 1;
            }
        }
        let columns = (FOOTPRINT * FOOTPRINT) as usize;
        assert!(
            carved * 2 < columns,
            "{kind:?} {fp:?}: {carved} of {columns} columns are not at their surface"
        );
        assert_eq!(
            m.faces.area[0], drawn as f64,
            "{kind:?} {fp:?}: one top face per drawn column"
        );
        // The tops' colour is the grass top's, lit from above, darkened by
        // the baked AO and by nothing else: `ao_floor + (1 - ao_floor) *
        // open` of it, to within what the two triangles of a quad shift the
        // mean of its four corners by.
        let open = m.faces.open[0] / m.faces.area[0];
        let floor = setup.lighting.ao_floor as f64;
        let unshaded = setup.far_materials.top * face_light(Vec3::Y, &setup.lighting);
        let lit = m.faces.mean_lit(0).expect("tops");
        // (Where every top is grass: a cave mouth's floor is soil or stone.)
        let all_grass = m.faces.material[0][SURFACE] == m.faces.area[0];
        grassy += all_grass as usize;
        for c in (0..3).filter(|_| all_grass) {
            let expected = unshaded[c] as f64 * (floor + (1.0 - floor) * open);
            assert!(
                (lit[c] - expected).abs() <= 0.02 * unshaded[c] as f64,
                "{kind:?} {fp:?}: tops lit {} against {expected} (openness {open:.3})",
                lit[c]
            );
        }
        if kind == Terrain::Steep {
            assert!(
                open < 0.95,
                "{kind:?} {fp:?}: a steep slope's tops are shaded"
            );
        }
        // A side face is counted on every solid block of a column that
        // stands above the top of the column beside it: counted here from
        // the world block by block, which the mesh walk must agree with
        // exactly. (Not simply the difference of the two tops: a cave under
        // a one-block crust leaves air in between.)
        let mut expected = 0.0;
        let mut beside_undrawn = false;
        for j in 0..FOOTPRINT {
            for i in 0..FOOTPRINT {
                for (di, dj) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    let (ni, nj) = (i + di, j + dj);
                    if !(0..FOOTPRINT).contains(&ni) || !(0..FOOTPRINT).contains(&nj) {
                        continue;
                    }
                    let (own, front) = (m.top(i, j), m.top(ni, nj));
                    if own.is_nan() || front.is_nan() {
                        beside_undrawn = true;
                        continue;
                    }
                    for y in front as i32..own as i32 {
                        if setup
                            .world
                            .is_solid_at(fp.x0 + i, y, fp.z0 + j, setup.blocks)
                        {
                            expected += 1.0;
                        }
                    }
                }
            }
        }
        let sides: f64 = m.faces.area[1..].iter().sum();
        if beside_undrawn {
            // A column beside one with no top in the generated band shows
            // its whole side, which the count above leaves out.
            assert!(sides >= expected, "{kind:?} {fp:?}: {sides} < {expected}");
        } else {
            assert_eq!(sides, expected, "{kind:?} {fp:?}: side faces");
        }

        // **The two terms.** The surface term is the uncarved columns'
        // faces: one grass top each, nothing else on top...
        let carved = setup.carved(fp);
        let uncarved = carved.iter().filter(|c| !**c).count();
        assert_eq!(terms.faces.area[0], uncarved as f64, "{kind:?} {fp:?}");
        // ...**and its sides are the height field's, face for face and
        // layer for layer**: between two uncarved columns, one face per
        // block the higher stands above the lower, made of what
        // `ground_layer` says at its depth below the higher one's surface.
        // Counted here from `surface_height` alone, with no mesh.
        {
            use cubara_world::{GroundLayer, WorldGen};
            let is_carved = |i: i32, j: i32| carved[(j * FOOTPRINT + i) as usize];
            let height = |i: i32, j: i32| setup.gen.surface_height(fp.x0 + i, fp.z0 + j);
            let mut expected = [[0f64; 3]; 5];
            for j in 0..FOOTPRINT {
                for i in 0..FOOTPRINT {
                    for (d, face) in DIRECTIONS.iter().enumerate().skip(1) {
                        let n = face.normal();
                        let (ni, nj) = (i + n[0] as i32, j + n[2] as i32);
                        if !(0..FOOTPRINT).contains(&ni)
                            || !(0..FOOTPRINT).contains(&nj)
                            || is_carved(i, j)
                            || is_carved(ni, nj)
                        {
                            continue;
                        }
                        let own = height(i, j);
                        for y in height(ni, nj) + 1..=own {
                            let material = match WorldGen::ground_layer(own - y) {
                                GroundLayer::Surface => SURFACE,
                                GroundLayer::Soil => SOIL,
                                GroundLayer::Stone => STONE,
                            };
                            expected[d][material] += 1.0;
                        }
                    }
                }
            }
            for (d, expected) in expected.iter().enumerate().skip(1) {
                assert_eq!(
                    terms.faces.material[d],
                    *expected,
                    "{kind:?} {fp:?}: the {} sides of the surface term (stone, soil, surface block) against the height field's",
                    direction_name(d)
                );
                assert_eq!(
                    terms.faces.area[d],
                    expected.iter().sum::<f64>(),
                    "{kind:?} {fp:?}: {} side area",
                    direction_name(d)
                );
            }
        }
        assert_eq!(
            terms.faces.material[0][SURFACE], terms.faces.area[0],
            "{kind:?} {fp:?}: a surface top that is not the surface block"
        );
        // ...and every side face is lit as a side: its own layers' colours
        // under the light of its own direction, darkened by AO and by
        // nothing else. (Lit as a top it would be 1.8 to 3 times this.)
        for (d, face) in DIRECTIONS.iter().enumerate().skip(1) {
            let f = &terms.faces;
            if f.area[d] < MIN_AREA {
                continue;
            }
            let light = face_light(Vec3::from(face.normal()), &setup.lighting);
            let colours = [
                setup.far_materials.stone,
                setup.far_materials.soil,
                setup.far_materials.side,
            ];
            for c in 0..3 {
                let unshaded: f64 = [STONE, SOIL, SURFACE]
                    .iter()
                    .map(|&m| f.material[d][m] * (colours[m][c] * light[c]) as f64)
                    .sum();
                let floor = setup.lighting.ao_floor as f64;
                assert!(
                    f.lit[d][c] <= unshaded * 1.001 && f.lit[d][c] >= unshaded * floor,
                    "{kind:?} {fp:?}: {} faces lit {} against {unshaded} unshaded",
                    direction_name(d),
                    f.lit[d][c]
                );
            }
        }
        // The claim the split rests on: a steep slope's *surface* is almost
        // stone-free, and its stone is in the cave mouths.
        if kind == Terrain::Steep {
            let stone = terms.faces.side_share(STONE).unwrap_or(0.0);
            assert!(
                stone < 0.02,
                "{kind:?} {fp:?}: {:.1}% of the surface's sides are stone",
                stone * 100.0
            );
        }
    }
    assert!(grassy >= 2, "only {grassy} footprints with all-grass tops");
}

/// The measurement moves when the ground is wrong: a level-1 node whose soil
/// has been turned to stone measures as more stone and a different colour,
/// and one whose cells are all a row lower measures a block lower.
#[test]
fn the_harness_sees_a_wrong_node() {
    use cubara_voxel::{AnisoChunk, BlockId, NodeChunk};
    let setup = Setup::new();
    let (_, fp) = draw_footprints(&setup.gen, FOOTPRINT_SEED, 1)
        .into_iter()
        .find(|(k, _)| *k == Terrain::Steep)
        .expect("a steep footprint");
    let real = measure_voxels(&setup, fp, 1);
    let rebuilt = |f: &dyn Fn(&AnisoChunk, usize, usize, usize) -> BlockId| {
        measure_voxels_from(&setup, fp, 1, &|node| {
            let Some(NodeChunk::Aniso(chunk)) = setup.world.node_at(node, setup.blocks) else {
                return None;
            };
            Some(NodeChunk::Aniso(AnisoChunk::from_fn(
                chunk.dims(),
                chunk.scale(),
                |x, y, z| f(&chunk, x, y, z),
            )))
        })
    };

    let same = rebuilt(&|c, x, y, z| c.get(x, y, z));
    assert_eq!(same.faces, real.faces, "a rebuilt node measures the same");
    assert_eq!(same.tops, real.tops);

    let stony = rebuilt(&|c, x, y, z| match c.get(x, y, z) {
        b if b == setup.blocks.soil => setup.blocks.stone,
        b => b,
    });
    let (was, is) = (
        real.faces.side_share(STONE).unwrap(),
        stony.faces.side_share(STONE).unwrap(),
    );
    assert!(is > was + 0.05, "stone share of sides {was:.3} -> {is:.3}");
    assert_eq!(stony.faces.area, real.faces.area, "the same faces");
    let moved = (1..5)
        .filter_map(|d| {
            Some(colour_delta(
                real.faces.mean_lit(d)?,
                stony.faces.mean_lit(d)?,
            ))
        })
        .fold(0.0, f64::max);
    assert!(moved > 1.0, "the sides' colour moved by {moved:.2}/255");

    // Every cell one row down (the bottom row of a node is lost, which is
    // far below the surface band the harness generates).
    let lower = rebuilt(&|c, x, y, z| {
        if y + 1 < c.dims()[1] {
            c.get(x, y + 1, z)
        } else {
            BlockId::AIR
        }
    });
    let dropped = real
        .tops
        .iter()
        .zip(&lower.tops)
        .filter(|(a, b)| **a - **b == 1.0)
        .count();
    assert!(
        dropped * 10 > real.tops.len() * 9,
        "{dropped} of {} tops dropped a block",
        real.tops.len()
    );
}

/// **Both sides of a comparison cover the same ground** (second review of
/// #428, point 1). A cell with one carved column under it leaves the
/// surface term whole, so a level of wider cells leaves more ground out;
/// level 0, split on that level's cells, must leave out exactly as much:
/// one top face per column of the ground the level's surface term covers,
/// and the level's own tops over the same ground. The split only moves
/// faces between the two terms.
#[test]
fn both_sides_of_a_comparison_cover_the_same_ground() {
    let setup = Setup::new();
    let all = |m: &Measure, d: usize| m.faces.area[d] + m.cave.area[d];
    let mut narrowed = 0;
    for (kind, fp) in draw_footprints(&setup.gen, FOOTPRINT_SEED, 1) {
        let level0 = reference(&setup, fp);
        let whole = level0.measure(&setup, 0, 1);
        let ground = |w: i32| setup.on_the_surface(fp, w).iter().filter(|on| **on).count() as f64;
        assert_eq!(whole.faces.area[0], ground(1), "{kind:?} {fp:?}");
        for level in VOXEL_LEVELS {
            let rep = Rep::Voxel(level);
            let at = format!("{kind:?} {fp:?}, level {level} (seed {FOOTPRINT_SEED:#x})");
            let split = level0.measure(&setup, 0, rep.mask());
            for d in 0..5 {
                assert_eq!(
                    all(&split, d),
                    all(&whole, d),
                    "{at}: {}",
                    direction_name(d)
                );
                assert!(
                    split.faces.area[d] <= whole.faces.area[d],
                    "{at}: wider cells put more of {} on the surface",
                    direction_name(d)
                );
            }
            assert_eq!(
                split.faces.area[0],
                ground(rep.mask()),
                "{at}: level 0's tops"
            );
            let other = measure_voxels(&setup, fp, level);
            assert_eq!(
                other.faces.area[0],
                ground(rep.mask()),
                "{at}: the level's own tops"
            );
            narrowed += (split.faces.area[0] < whole.faces.area[0]) as usize;
        }
        // The far terrain: one unit of top per unit of ground, on the quads
        // no cave has carved.
        let far = measure_far(&setup, fp, 2);
        assert_eq!(
            far.faces.area[0],
            ground(4),
            "{kind:?} {fp:?}: far quads of 4"
        );
    }
    // The split bites: wider cells do leave more ground out.
    assert!(
        narrowed >= 6,
        "the wider masks left out more ground in only {narrowed} of 12 comparisons"
    );
}

/// **A direction level 0 has and a level lacks is a failure, not a
/// direction to pass over** (second review of #428, point 3), and how much
/// area a direction needs follows the level's own faces: sixteen of them,
/// and never under 64 square blocks.
#[test]
fn a_direction_level_0_has_and_a_level_lacks_is_missing() {
    let setup = Setup::new();
    let (_, fp) = draw_footprints(&setup.gen, FOOTPRINT_SEED, 1)[0];
    // A measurement with `area` of +X faces and nothing else.
    let with = |area: f64| {
        let mut faces = Faces::default();
        faces.area[1] = area;
        faces.lit[1] = [area * 0.5; 3];
        Measure {
            faces,
            cave: Faces::default(),
            tops: height_field(&setup.gen, fp).tops,
        }
    };
    let ranked = |rep: Rep, a: f64, b: f64| compare(&setup, fp, rep, &with(a), &with(b)).delta[1];
    let missing = |level0: f64, area: f64| Ranked::Missing { level0, area };

    let l1 = Rep::Voxel(1);
    assert_eq!(l1.min_area(), 64.0);
    assert_eq!(ranked(l1, 127.0, 0.0), Ranked::Unranked);
    assert_eq!(ranked(l1, 128.0, 0.0), missing(128.0, 0.0));
    assert_eq!(ranked(l1, 128.0, 63.0), missing(128.0, 63.0));
    assert_eq!(ranked(l1, 128.0, 64.0), Ranked::Delta(0.0));

    // Level 3: a cell's side is 8 x 4, so sixteen of them are 512.
    let l3 = Rep::Voxel(3);
    assert_eq!(l3.min_area(), 512.0);
    assert_eq!(ranked(l3, 1023.0, 0.0), Ranked::Unranked);
    assert_eq!(ranked(l3, 1024.0, 511.0), missing(1024.0, 511.0));
    assert_eq!(ranked(l3, 1024.0, 512.0), Ranked::Delta(0.0));
    assert_eq!(Rep::Voxel(2).min_area(), 64.0);
    assert_eq!(Rep::Far(5).min_area(), 64.0);

    // What an asserted tolerance reads: no tolerance admits a missing
    // direction, and an unranked one is not held to any.
    assert_eq!(missing(128.0, 0.0).delta(), Some(f64::INFINITY));
    assert_eq!(Ranked::Unranked.delta(), None);
    assert_eq!(Ranked::Delta(1.5).delta(), Some(1.5));
}

/// **A level-3 node is measured in blocks, not in cells** (review of #428,
/// finding B2): its cells are 8 wide and 4 tall, and nothing else here has
/// a row taller than a block. A node made by hand -- a slab eleven rows
/// high, its east half a plateau two rows higher, all stone -- has tops and
/// side faces that are known exactly.
#[test]
fn a_level_3_node_is_measured_in_blocks() {
    use cubara_voxel::{AnisoChunk, BlockId, NodeChunk};
    use cubara_world::node::NodeKey;
    let setup = Setup::new();
    let (_, fp) = draw_footprints(&setup.gen, FOOTPRINT_SEED, 1)[0];
    // One row of nodes, which the footprint's own ground is in.
    let row = setup.gen.surface_height(fp.x0, fp.z0).div_euclid(128);
    let base = row * 128;
    let m = measure_voxels_from(&setup, fp, 3, &|node: NodeKey| {
        (node.pos[1] == row).then(|| {
            let origin = node.world_origin();
            NodeChunk::Aniso(AnisoChunk::from_fn([16, 32, 16], [8, 4, 8], |lx, ly, _| {
                let east = origin[0] + lx as i32 * 8 >= fp.x0 + FOOTPRINT / 2;
                if ly <= 10 + 2 * east as usize {
                    setup.blocks.stone
                } else {
                    BlockId::AIR
                }
            }))
        })
    });
    // Tops: one above the highest solid row, in blocks.
    for j in 0..FOOTPRINT {
        for i in 0..FOOTPRINT {
            let rows = if i >= FOOTPRINT / 2 { 13 } else { 11 };
            assert_eq!(m.top(i, j), (base + rows * 4) as f64, "column ({i}, {j})");
        }
    }
    // Faces, the two terms together (the split is on the real ground's
    // caves, which this node knows nothing of): a top over every column,
    // and the plateau's one riser -- 8 cells long, 2 rows of 4 blocks high,
    // 8 blocks a cell -- facing -X. Nothing else inside the footprint.
    let area = |d: usize| m.faces.area[d] + m.cave.area[d];
    let stone = |d: usize| m.faces.material[d][STONE] + m.cave.material[d][STONE];
    assert_eq!(area(0), (FOOTPRINT * FOOTPRINT) as f64, "tops");
    assert_eq!(direction_name(2), "-X");
    assert_eq!(area(2), (8 * 8 * 2 * 4) as f64, "the plateau's riser");
    assert_eq!(stone(2), area(2), "all stone");
    for d in [1, 3, 4] {
        assert_eq!(area(d), 0.0, "{} faces", direction_name(d));
    }
}

/// `Shape` and `spread` on tops made by hand: a field raised by two blocks
/// everywhere is two blocks off in every column, on the skyline and at the
/// peak; one that flattens a single peak moves the peak and nothing else by
/// much.
#[test]
fn shape_is_the_difference_in_blocks() {
    let n = (FOOTPRINT * FOOTPRINT) as usize;
    let flat = |h: f64| Measure {
        faces: Faces::default(),
        cave: Faces::default(),
        tops: vec![h; n],
    };
    let raised = Shape::of(&flat(10.0), &flat(12.0));
    assert_eq!(spread(&raised.column), (2.0, 2.0, 2.0, 2.0));
    assert_eq!(spread(&raised.skyline), (2.0, 2.0, 2.0, 2.0));
    assert_eq!(raised.peak, vec![2.0]);

    let mut peaked = flat(10.0);
    peaked.tops[n / 2] = 30.0;
    let lost = Shape::of(&peaked, &flat(10.0));
    assert_eq!(lost.peak, vec![-20.0]);
    let (mean, mean_mag, p95, max) = spread(&lost.column);
    assert!(mean < 0.0 && mean_mag < 0.01 && p95 == 0.0 && max == 20.0);
    // The peak is on one row's skyline and one line's.
    assert_eq!(lost.skyline.iter().filter(|d| **d == -20.0).count(), 2);

    let mut holed = flat(10.0);
    holed.tops[7] = f64::NAN;
    assert_eq!(Shape::of(&flat(10.0), &holed).undrawn, 1);

    // Over some columns only: the lost peak is not seen with its column
    // left out, on either side, and is with any other column left out.
    let mut counted = vec![true; n];
    counted[n / 2] = false;
    let masked = Shape::over(&peaked, &flat(10.0), &counted);
    assert_eq!((masked.peak.clone(), masked.left_out), (vec![0.0], 1));
    assert_eq!(spread(&masked.column), (0.0, 0.0, 0.0, 0.0));
    assert_eq!(spread(&masked.skyline), (0.0, 0.0, 0.0, 0.0));
    counted[n / 2] = true;
    counted[0] = false;
    assert_eq!(
        Shape::over(&peaked, &flat(10.0), &counted).peak,
        vec![-20.0]
    );
}

/// **What already holds today**, pinned so the fix cannot lose it: on the
/// rolling hills level 1 shows level 0's surface colours in every direction
/// -- **in every single footprint**, not only pooled (review of #428,
/// finding B3), within [`T3_HILLS_PER_FOOTPRINT`]; and pooled within T3's
/// 2/255. (Levels 2 and 3, and every level on a slope, do not: that is
/// issue #423, and `consistency_report` has the numbers.)
#[test]
fn level_1_matches_level_0_on_the_rolling_hills() {
    /// T3's own 2/255 holds footprint by footprint here: the worst single
    /// footprint and direction measured on this seed is 0.59.
    const T3_HILLS_PER_FOOTPRINT: f64 = T3_TOLERANCE;
    let setup = Setup::new();
    let footprints: Vec<_> = draw_footprints(&setup.gen, FOOTPRINT_SEED, PER_KIND)
        .into_iter()
        .filter(|(k, _)| *k == Terrain::Flat)
        .collect();
    assert_eq!(footprints.len(), PER_KIND, "seed {FOOTPRINT_SEED:#x}");
    let (mut worst, mut checked) = (0f64, 0);
    for &(_, fp) in &footprints {
        let rep = Rep::Voxel(1);
        let level0 = reference(&setup, fp).measure(&setup, 0, rep.mask());
        let one = compare(&setup, fp, rep, &level0, &measure_voxels(&setup, fp, 1));
        for (d, ranked) in one.delta.iter().enumerate() {
            // A direction level 0 has and level 1 lacks is infinitely off.
            let Some(delta) = ranked.delta() else {
                continue;
            };
            checked += 1;
            worst = worst.max(delta);
            assert!(
                delta <= T3_HILLS_PER_FOOTPRINT,
                "level 1, {fp:?}, {}: {delta:.2}/255 off level 0, {ranked:?} (seed {FOOTPRINT_SEED:#x})",
                direction_name(d)
            );
        }
    }
    println!("level 1 on the hills: worst footprint and direction {worst:.2}/255 of {checked}");
    assert!(
        checked >= 2 * PER_KIND,
        "only {checked} directions had the area"
    );
    let pools = measure_all(&setup, &footprints, &[Rep::Voxel(1)]);
    let pool = &pools[&(Terrain::Flat, Rep::Voxel(1))];
    for d in 0..5 {
        let delta = pool.delta(d).expect("faces in every direction");
        assert!(
            delta <= T3_TOLERANCE,
            "level 1, {}, pooled: {delta:.2}/255 off level 0 (seed {FOOTPRINT_SEED:#x})",
            direction_name(d)
        );
    }
}

/// **The far rows measure the far terrain, not something beside it**
/// (review of #428, finding B2). Two things a wrong far measurement would
/// get wrong, each held on every steep footprint (that the reference it
/// is compared with is level 0 is `level_0_draws_its_surface`'s to hold):
///
/// - its tops follow the ground: at quads of four blocks the mean drawn top
///   is within [`FAR_TOP`] blocks of the height field's;
/// - its risers face downhill: where level 0's risers along an axis face one
///   way at least twice as much as the other, the far terrain's larger share
///   faces the same way.
#[test]
fn the_far_rows_follow_the_ground() {
    /// **Chosen, not derived**: three times the largest magnitude measured
    /// on this seed's steep footprints (0.16).
    const FAR_TOP: f64 = 0.5;
    let setup = Setup::new();
    let footprints: Vec<_> = draw_footprints(&setup.gen, FOOTPRINT_SEED, PER_KIND)
        .into_iter()
        .filter(|(k, _)| *k == Terrain::Steep)
        .collect();
    assert_eq!(footprints.len(), PER_KIND, "seed {FOOTPRINT_SEED:#x}");
    let (mut worst_top, mut axes, mut top_gap) = (0f64, 0, 0f64);
    for &(_, fp) in &footprints {
        let rep = Rep::Far(2);
        let level0 = reference(&setup, fp).measure(&setup, 0, rep.mask());
        let far = measure_far(&setup, fp, 2);
        let one = compare(&setup, fp, rep, &level0, &far);
        let (mean, ..) = spread(&one.shape.column);
        worst_top = worst_top.max(mean.abs());
        assert!(
            mean.abs() <= FAR_TOP,
            "{fp:?}: the far tops stand {mean:+.2} blocks off the ground (seed {FOOTPRINT_SEED:#x})"
        );
        for (plus, minus) in [(1, 2), (3, 4)] {
            let (a, b) = (level0.faces.area[plus], level0.faces.area[minus]);
            if a.max(b) < 4.0 * MIN_AREA || a.max(b) < 2.0 * a.min(b) {
                continue;
            }
            axes += 1;
            let (fa, fb) = (far.faces.area[plus], far.faces.area[minus]);
            assert_eq!(
                a > b,
                fa > fb,
                "{fp:?}: level 0's risers face {} ({a} against {b}), the far terrain's the other way ({fa} against {fb})",
                direction_name(if a > b { plus } else { minus })
            );
        }
        if let Ranked::Delta(delta) = one.delta[0] {
            top_gap = top_gap.max(delta);
        }
    }
    println!("far, quads of 4: worst mean top {worst_top:.2} blocks; {axes} axes checked; tops up to {top_gap:.1}/255 off level 0's");
    assert!(
        axes >= PER_KIND,
        "only {axes} axes had a clear downhill side"
    );
}

/// **`WorldGen::ground_faces` is what the level-0 mesher draws.** The far
/// terrain is to colour its walls from that function (#422), so it is held
/// here to the real thing: the game's level-0 chunks, meshed by the game's
/// mesher, counted face by face over the same square.
///
/// Where no cave touches the surface the two agree **exactly** on every
/// count: top faces, and riser area per direction and per layer.
///
/// The openness agrees to within [`OPENNESS_TOLERANCE`], and the test says
/// where the rest goes: the mesher bakes a chunk's ambient occlusion without
/// looking into the next chunk, so a face at a chunk's edge (or whose air
/// side is in the chunk above) comes out more open than the height field
/// says. Asked the same question with that blindness put in -- the corner
/// rule itself, `face_ao_of`, over the same height field -- the answer is
/// the mesher's to the last bit. So `ground_faces` is the mesher's rule on
/// the true neighbours, and what differs is the mesher's chunk edges.
#[test]
fn ground_faces_are_what_the_level_0_mesher_draws() {
    use cubara_voxel::face_ao_of;
    use cubara_world::{GroundLayer, GROUND_DIRECTIONS};
    /// **Chosen, not derived**: above the largest gap measured over this
    /// seed's footprints (0.059 on tops, 0.106 on one direction's risers,
    /// both on steep ground, where a chunk edge hides the most).
    const OPENNESS_TOLERANCE: (f64, f64) = (0.08, 0.14);
    /// Counted this far inside the footprint, so that every counted face's
    /// front column is in the footprint too.
    const INNER: i32 = 8;

    let setup = Setup::new();
    let mut checked = std::collections::BTreeMap::<Terrain, usize>::new();
    let (mut worst_top, mut worst_side) = (0f64, 0f64);
    // Twice the usual draw: a cave under the surface somewhere in a
    // footprint is the common case, and those are passed over.
    for (kind, fp) in draw_footprints(&setup.gen, FOOTPRINT_SEED, 2 * PER_KIND) {
        let whole = measure_voxels(&setup, fp, 0);
        // Only footprints the height field describes completely: every top
        // at its surface, and no cave under a crust (a riser with air
        // behind part of it), which the step sum shows.
        let field = height_field(&setup.gen, fp);
        let mut steps = 0.0;
        for j in 0..FOOTPRINT {
            for i in 0..FOOTPRINT {
                for (di, dj) in [(1, 0), (0, 1)] {
                    if i + di < FOOTPRINT && j + dj < FOOTPRINT {
                        steps += (field.top(i, j) - field.top(i + di, j + dj)).abs();
                    }
                }
            }
        }
        if whole.tops != field.tops || whole.faces.area[1..].iter().sum::<f64>() != steps {
            continue;
        }
        *checked.entry(kind).or_default() += 1;

        let m = measure_voxels_within(
            &setup,
            fp,
            0,
            &|node| setup.world.node_at(node, setup.blocks),
            INNER,
        );
        let w = FOOTPRINT - 2 * INNER;
        let (x0, z0) = (fp.x0 + INNER, fp.z0 + INNER);
        let ground = setup.gen.ground_faces(x0, z0, w, w);
        let at = format!("{kind:?} {fp:?} (seed {FOOTPRINT_SEED:#x})");
        assert_eq!(ground.columns as f64, m.faces.area[0], "{at}: top faces");

        // The same faces with the mesher's blindness: nothing outside the
        // chunk of the block a face belongs to is solid.
        let chunk = |a: i32| a.div_euclid(16);
        let mut blind = [0f64; 5];
        // The footprint's heights, worked out once: every block asked about
        // is within a block of a counted column, and those are `INNER` in.
        let height = |x: i32, z: i32| field.top(x - fp.x0, z - fp.z0) as i32 - 1;
        for z in z0..z0 + w {
            for x in x0..x0 + w {
                let h = height(x, z);
                let solid_from = |own: [i32; 3]| {
                    move |bx: i32, by: i32, bz: i32| {
                        [bx, by, bz].map(chunk) == own.map(chunk) && by <= height(bx, bz)
                    }
                };
                let sum = |ao: [u8; 4]| ao.iter().map(|&a| a as f64).sum::<f64>() / 12.0;
                blind[0] += sum(face_ao_of(&solid_from([x, h, z]), 1, 2, 0, h + 1, z, x));
                for (d, [dx, dz]) in GROUND_DIRECTIONS.into_iter().enumerate() {
                    for y in height(x + dx, z + dz) + 1..=h {
                        let solid = solid_from([x, y, z]);
                        blind[d + 1] += sum(if dx != 0 {
                            face_ao_of(&solid, 0, 1, 2, x + dx, y, z)
                        } else {
                            face_ao_of(&solid, 2, 0, 1, z + dz, x, y)
                        });
                    }
                }
            }
        }

        for (d, [gx, gz]) in GROUND_DIRECTIONS.into_iter().enumerate() {
            // The harness's directions are +Y first, then these four.
            let h = d + 1;
            assert_eq!(
                Vec3::from(DIRECTIONS[h].normal()),
                Vec3::new(gx as f32, 0.0, gz as f32)
            );
            for (layer, material) in [
                (GroundLayer::Surface, SURFACE),
                (GroundLayer::Soil, SOIL),
                (GroundLayer::Stone, STONE),
            ] {
                assert_eq!(
                    ground.sides[d][layer as usize] as f64,
                    m.faces.material[h][material],
                    "{at}: {} risers, {layer:?}",
                    direction_name(h)
                );
            }
            assert!(
                (blind[h] - m.faces.open[h]).abs() < 1e-6,
                "{at}: {} risers' openness, chunk-blind: {} against the mesher's {}",
                direction_name(h),
                blind[h],
                m.faces.open[h]
            );
            if m.faces.area[h] >= 64.0 {
                let gap = (ground.side_open[d] as f64 - m.faces.open[h] / m.faces.area[h]).abs();
                worst_side = worst_side.max(gap);
                assert!(
                    gap <= OPENNESS_TOLERANCE.1,
                    "{at}: {} risers' openness is {gap:.3} off the mesher's",
                    direction_name(h)
                );
            }
        }
        assert!(
            (blind[0] - m.faces.open[0]).abs() < 1e-6,
            "{at}: the tops' openness, chunk-blind: {} against the mesher's {}",
            blind[0],
            m.faces.open[0]
        );
        let gap = (ground.top_open as f64 - m.faces.open[0] / m.faces.area[0]).abs();
        worst_top = worst_top.max(gap);
        assert!(
            gap <= OPENNESS_TOLERANCE.0,
            "{at}: the tops' openness is {gap:.3} off the mesher's"
        );
    }
    println!(
        "checked {checked:?}; worst openness gap: tops {worst_top:.4}, risers {worst_side:.4}"
    );
    // Every kind of terrain, cliffs included (stone is there), and enough
    // of them in all: a kind that dropped out would leave `ground_faces`
    // unchecked exactly where it matters.
    let total: usize = checked.values().sum();
    assert!(
        total >= 5 && TERRAINS.iter().all(|k| checked.contains_key(k)),
        "too few footprints without a cave at the surface: {checked:?} (seed {FOOTPRINT_SEED:#x})"
    );
}

/// **What a node costs to generate and to build, per level** -- the number
/// #423's rule is weighed against (a coarse cell's top over every column it
/// covers is up to 64 surface heights where one sample of four used to do).
///
/// The nodes are the ones that hold the ground of three footprints of each
/// kind of terrain. `generate` is `World::node_at` alone; `build` is
/// `mesh::build_node` (generation, the mesh with its border samples, the
/// visibility links, the trees' sites).
///
/// **Noisy on a shared machine**: wall-clock means over a few hundred nodes,
/// good for a ratio between two runs made back to back and for nothing
/// finer. `cubara-world` is built optimised in the test profile; the mesher
/// (`cubara-voxel`) is not, so `build` overstates meshing unless this is run
/// with `--release`:
///
/// ```text
/// cargo test --release -p cubara-render --test lod_consistency -- --ignored --nocapture generation_cost_report
/// ```
#[test]
#[ignore]
fn generation_cost_report() {
    use cubara_world::node::NodeKey;
    use std::time::Instant;
    let setup = Setup::new();
    let footprints = draw_footprints(&setup.gen, FOOTPRINT_SEED, 3);
    let snapshot = setup.world.snapshot();
    println!("level  nodes  generate us/node  build us/node");
    for level in 0..=3u32 {
        let size = 16 * (1 << level);
        let mut nodes = std::collections::BTreeSet::new();
        for (_, fp) in &footprints {
            let field = height_field(&setup.gen, *fp);
            let lo = field.tops.iter().fold(f64::INFINITY, |a, &b| a.min(b)) as i32;
            let hi = field.tops.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) as i32;
            for nz in fp.z0.div_euclid(size)..=(fp.z0 + FOOTPRINT - 1).div_euclid(size) {
                for nx in fp.x0.div_euclid(size)..=(fp.x0 + FOOTPRINT - 1).div_euclid(size) {
                    for ny in (lo - 8).div_euclid(size)..=(hi + 8).div_euclid(size) {
                        nodes.insert(NodeKey::new(level, [nx, ny, nz]));
                    }
                }
            }
        }
        // Three passes, the fastest kept: the first warms the caches, and a
        // neighbour's build landing in one pass should not decide the row.
        let (mut generate, mut build) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..3 {
            let start = Instant::now();
            for &node in &nodes {
                std::hint::black_box(setup.world.node_at(node, setup.blocks));
            }
            generate = generate.min(start.elapsed().as_secs_f64());
            let start = Instant::now();
            for &node in &nodes {
                std::hint::black_box(cubara_world::mesh::build_node(
                    &snapshot,
                    &setup.registry,
                    &|name| setup.layer_of(name),
                    node,
                    setup.blocks,
                ));
            }
            build = build.min(start.elapsed().as_secs_f64());
        }
        let n = nodes.len() as f64;
        println!(
            "L{level}     {:5}  {:16.1}  {:13.1}",
            nodes.len(),
            generate * 1e6 / n,
            build * 1e6 / n
        );
    }
}

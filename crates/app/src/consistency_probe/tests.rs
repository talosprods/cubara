use super::*;

/// A frame of one colour, `w` x `h`.
fn flat(w: u32, h: u32, rgb: [u8; 3]) -> Frame {
    Frame {
        width: w,
        height: h,
        pixels: (0..w * h)
            .flat_map(|_| [rgb[0], rgb[1], rgb[2], 255])
            .collect(),
    }
}

fn copy(frame: &Frame) -> Frame {
    Frame {
        width: frame.width,
        height: frame.height,
        pixels: frame.pixels.clone(),
    }
}

fn set(frame: &mut Frame, x: u32, y: u32, rgb: [u8; 3]) {
    let i = ((y * frame.width + x) * 4) as usize;
    frame.pixels[i..i + 3].copy_from_slice(&rgb);
}

const GRASS: [u8; 3] = [60, 140, 50];
const MAGENTA: [u8; 3] = [255, 0, 255];

/// Land below row `top(x)` in every column, magenta above.
fn ridge(w: u32, h: u32, top: impl Fn(u32) -> u32) -> Frame {
    let mut f = flat(w, h, MAGENTA);
    for x in 0..w {
        for y in top(x)..h {
            set(&mut f, x, y, GRASS);
        }
    }
    f
}

fn all(frame: &Frame) -> Vec<bool> {
    vec![true; (frame.width * frame.height) as usize]
}

#[test]
fn forced_nodes_tile_the_box_exactly_at_every_level() {
    let (min, ext) = ([-16, 0, 8], [24, 16, 8]);
    let chunks = |level: u32| {
        let mut seen = std::collections::HashSet::new();
        for n in forced_nodes(level, min, ext) {
            assert_eq!(n.level, level);
            let o = n.chunk_origin();
            let e = n.extent_chunks();
            for x in 0..e {
                for y in 0..e {
                    for z in 0..e {
                        assert!(
                            seen.insert([o.x + x, o.y + y, o.z + z]),
                            "level {level}: two nodes cover one chunk"
                        );
                    }
                }
            }
        }
        seen
    };
    let reference = chunks(0);
    assert_eq!(reference.len(), 24 * 16 * 8);
    for level in 1..=3 {
        assert_eq!(
            chunks(level),
            reference,
            "level {level} covers another footprint"
        );
    }
}

#[test]
#[should_panic(expected = "aligned")]
fn forced_nodes_refuse_a_box_off_the_level_3_grid() {
    forced_nodes(3, [4, 0, 0], [8, 8, 8]);
}

#[test]
fn the_aligned_box_contains_its_columns_and_the_height() {
    let (min, ext) = aligned_box(&[[-100, 40], [300, 90]], 96, -5, 250);
    for a in 0..3 {
        assert_eq!(min[a] % 8, 0);
        assert_eq!(ext[a] % 8, 0);
    }
    let blocks = |a: usize| (min[a] * 16, (min[a] + ext[a]) * 16);
    assert!(blocks(0).0 <= -100 - 96 && 300 + 96 < blocks(0).1);
    assert!(blocks(2).0 <= 40 - 96 && 90 + 96 < blocks(2).1);
    assert!(blocks(1).0 <= -5 && 250 < blocks(1).1);
}

#[test]
fn silhouette_is_zero_for_an_image_against_itself_and_exact_for_a_shift() {
    let a = ridge(200, 48, |x| 20 + x / 8);
    let m = all(&a);
    let same = silhouette_diff(&a, &a, &m);
    assert_eq!((same.max_px, same.p99_px, same.columns), (0.0, 0.0, 200));
    // Every column 3 rows lower, one column 7 rows lower.
    let b = ridge(200, 48, |x| 20 + x / 8 + if x == 10 { 7 } else { 3 });
    let d = silhouette_diff(&b, &a, &m);
    assert_eq!(d.max_px, 7.0);
    assert_eq!(
        d.p99_px, 3.0,
        "one column in 200 is not the 99th percentile"
    );
}

#[test]
fn silhouette_counts_only_columns_where_the_box_is_the_silhouette() {
    let a = ridge(8, 16, |_| 8);
    let b = ridge(8, 16, |x| if x < 4 { 5 } else { 8 });
    // The mask covers the left four columns only.
    let mut mask = vec![false; 8 * 16];
    for x in 0..4u32 {
        for y in 0..16u32 {
            mask[(y * 8 + x) as usize] = true;
        }
    }
    let d = silhouette_diff(&b, &a, &mask);
    assert_eq!(d.columns, 4);
    assert_eq!(d.max_px, 3.0);
    // A column with no land at all is the whole height away.
    let sky = flat(8, 16, MAGENTA);
    let d = silhouette_diff(&sky, &a, &all(&a));
    assert_eq!(d.max_px, 8.0);
}

#[test]
fn tiles_are_zero_against_themselves_and_read_a_known_tint() {
    let a = flat(64, 64, GRASS);
    let m = all(&a);
    let same = tile_diff(&a, &a, &m, 16);
    assert_eq!((same.tiles, same.max), (16, 0.0));
    // Greener by 51/255 = 0.2 in one tile of four.
    let mut b = flat(64, 64, GRASS);
    for y in 0..32 {
        for x in 0..32 {
            set(&mut b, x, y, [GRASS[0], GRASS[1] + 51, GRASS[2]]);
        }
    }
    let d = tile_diff(&b, &a, &m, 32);
    assert_eq!(d.tiles, 4);
    assert!((d.max - 0.2).abs() < 1e-6, "max {}", d.max);
    assert!(
        d.p50 < 1e-6,
        "three tiles of four are identical, p50 {}",
        d.p50
    );
}

#[test]
fn tiles_with_sky_in_either_image_are_not_a_colour_error() {
    let a = flat(32, 32, GRASS);
    let mut b = flat(32, 32, GRASS);
    set(&mut b, 3, 3, MAGENTA);
    let d = tile_diff(&b, &a, &all(&a), 16);
    assert_eq!(
        d.tiles, 3,
        "the tile with a sky pixel is the silhouette's business"
    );
}

#[test]
fn tiles_outside_the_box_are_not_counted() {
    let a = flat(32, 32, GRASS);
    let mut mask = vec![false; 32 * 32];
    for y in 0..16u32 {
        for x in 0..16u32 {
            mask[(y * 32 + x) as usize] = true;
        }
    }
    assert_eq!(tile_diff(&a, &a, &mask, 16).tiles, 1);
}

/// Three well-separated endmembers, in linear light, for the synthetic tests.
fn ends() -> Endmembers {
    Endmembers::from_colours([0.20, 0.21, 0.24], [0.05, 0.28, 0.08], [0.18, 0.07, 0.03]).unwrap()
}

/// The sRGB byte of a linear-light value.
fn to_srgb(v: f32) -> u8 {
    let c = if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (c.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// A pixel that is `fractions` of stone, grass and soil under lighting `s`.
fn mixed(fractions: [f32; 3], s: f32) -> [u8; 3] {
    let colours = [[0.20, 0.21, 0.24], [0.05, 0.28, 0.08], [0.18, 0.07, 0.03]];
    std::array::from_fn(|c| to_srgb(s * (0..3).map(|i| fractions[i] * colours[i][c]).sum::<f32>()))
}

#[test]
fn unmixing_recovers_known_mixtures_at_any_brightness() {
    let ends = ends();
    for lighting in [0.4, 0.7, 1.0] {
        for want in [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.2, 0.3, 0.5],
        ] {
            let px = mixed(want, lighting);
            let (got, fits) = ends.unmix(&[px[0], px[1], px[2], 255]).expect("not dark");
            assert!(fits, "{want:?} at {lighting}: should fit the model");
            for k in 0..3 {
                // 8-bit quantisation of a dark pixel is the limit, not the model.
                assert!(
                    (got[k] - want[k]).abs() < 0.06,
                    "{want:?} at {lighting}: got {got:?}"
                );
            }
        }
    }
}

/// What a hue test got wrong: grass and soil stripes average to an olive grey.
/// Unmixing must call that grass and soil, not stone.
#[test]
fn a_grass_and_soil_mix_is_not_stone() {
    let ends = ends();
    let px = mixed([0.0, 0.5, 0.5], 0.8);
    let (got, _) = ends.unmix(&[px[0], px[1], px[2], 255]).unwrap();
    assert!(got[0] < 0.05, "stone share {}", got[0]);
}

#[test]
fn material_share_is_the_mean_of_the_land_pixels_in_the_mask() {
    let ends = ends();
    let (stone, grass, soil) = (
        mixed([1.0, 0.0, 0.0], 0.8),
        mixed([0.0, 1.0, 0.0], 0.8),
        mixed([0.0, 0.0, 1.0], 0.8),
    );
    let mut f = flat(4, 1, MAGENTA);
    set(&mut f, 0, 0, stone);
    set(&mut f, 1, 0, grass);
    set(&mut f, 2, 0, soil);
    let share = material_share(&f, &[true, true, true, true], &ends);
    assert_eq!(share.land, 3, "the magenta pixel is sky");
    for (name, v) in [
        ("stone", share.stone),
        ("grass", share.grass),
        ("soil", share.soil),
    ] {
        assert!((v - 1.0 / 3.0).abs() < 0.03, "{name} {v}");
    }
    // Outside the mask it does not count.
    let only_grass = material_share(&f, &[false, true, false, false], &ends);
    assert!(
        only_grass.stone < 0.03 && only_grass.grass > 0.97,
        "{only_grass:?}"
    );
}

#[test]
fn the_box_mask_is_what_differs_from_the_backdrop() {
    let backdrop = flat(4, 1, GRASS);
    let mut image = flat(4, 1, GRASS);
    set(&mut image, 2, 0, MAGENTA);
    assert_eq!(box_mask(&image, &backdrop), vec![false, false, true, false]);
}

#[test]
fn the_tile_grows_with_the_level_and_never_drops_under_16() {
    let tiles: Vec<u32> = (0..=3)
        .map(|l| tile_px(Rep::Level(l), 240.0, 720))
        .collect();
    assert!(tiles.windows(2).all(|w| w[0] <= w[1]), "{tiles:?}");
    assert!(tiles.iter().all(|&t| t >= 16 && t % 4 == 0), "{tiles:?}");
    // A cell of level 3 is 8 blocks: 8 * 720 / (2 tan 30 deg * 240) px.
    assert!(tiles[3] as f32 >= 8.0 * 720.0 / (2.0 * (FOV_Y / 2.0).tan() * 240.0));
    assert_eq!(tile_px(Rep::Far, 240.0, 720), 32);
}

/// The real textures' endmembers (cheap: three small PNGs).
fn real_ends() -> Endmembers {
    Endmembers::from_textures()
}

fn metrics_of(image: &Frame, reference: &Frame) -> Metrics {
    let backdrop = flat(image.width, image.height, [10, 10, 10]);
    measure(
        Rep::Level(2),
        image,
        reference,
        &backdrop,
        240.0,
        &real_ends(),
    )
}

const TEST_TARGETS: Targets = Targets {
    silhouette_max_px: 2.0,
    silhouette_p99_px: 1.0,
    tile_p50: 0.02,
    tile_p99: 0.05,
    tile_max: 0.08,
    stone_delta_pp: 5.0,
};

#[test]
fn the_targets_pass_an_image_against_itself() {
    let reference = ridge(128, 96, |x| 30 + x / 16);
    let same = metrics_of(&reference, &reference);
    assert!(
        TEST_TARGETS.violations(&same).is_empty(),
        "{:?}",
        TEST_TARGETS.violations(&same)
    );
}

/// The owner's case: grass drawn as grey stone.
#[test]
fn the_targets_fail_grass_drawn_as_stone() {
    let reference = ridge(128, 96, |x| 30 + x / 16);
    let mut stone = copy(&reference);
    for p in stone.pixels.as_chunks_mut::<4>().0 {
        if !shows_nothing(&p[..]) {
            p[..3].copy_from_slice(&[110, 110, 110]);
        }
    }
    let violations = TEST_TARGETS.violations(&metrics_of(&stone, &reference));
    assert!(
        violations.iter().any(|v| v.contains("stone share")),
        "a stone-for-grass swap must break the share target: {violations:?}"
    );
    assert!(
        violations.iter().any(|v| v.contains("tile")),
        "{violations:?}"
    );
}

#[test]
fn the_targets_fail_a_shifted_ridge() {
    let reference = ridge(128, 96, |x| 30 + x / 16);
    let shifted = ridge(128, 96, |x| 35 + x / 16);
    let violations = TEST_TARGETS.violations(&metrics_of(&shifted, &reference));
    assert!(
        violations.iter().any(|v| v.contains("silhouette")),
        "{violations:?}"
    );
}

/// The unmixing's error on the real textures: each texture, lit at the
/// brightnesses the shaders produce (ambient + sun + AO: roughly 0.45..1.1 of
/// the texel), unmixes mostly into itself.
#[test]
fn unmixing_reads_the_terrain_textures_as_themselves() {
    let ends = real_ends();
    for (name, index) in [("stone", 0), ("grass_top", 1), ("soil", 2)] {
        let tile = cubara_render::load_icon(name).expect("texture");
        for scale in [0.45f32, 0.7, 1.0] {
            let texels = tile.as_chunks::<4>().0;
            let mut share = 0.0f32;
            for t in texels {
                let lit = [0, 1, 2].map(|c| to_srgb(srgb_lin(t[c]) * scale));
                share += ends
                    .unmix(&[lit[0], lit[1], lit[2], 255])
                    .map_or(0.0, |(f, _)| f[index]);
            }
            share /= texels.len() as f32;
            println!(
                "unmixing: {name} at x{scale}: {:.1}% of its texels' mass is {name}",
                100.0 * share
            );
            assert!(
                share >= UNMIX_MIN_RIGHT,
                "{name} at x{scale}: only {:.1}% reads as itself",
                100.0 * share
            );
        }
    }
}

fn srgb_lin(byte: u8) -> f32 {
    let c = f32::from(byte) / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// A texture must unmix at least this much into itself (chosen from the
/// measurement the test above prints; the texels' own noise is the limit).
const UNMIX_MIN_RIGHT: f32 = 0.8;

/// The eye stands well outside the box in both scenes (a forced level 3 is
/// never drawn that close in the game), above the ground, looking at the subject.
#[test]
fn the_eye_stands_outside_the_box_in_both_scenes() {
    let setup = Setup::new(false);
    let slope = slope_scene(&setup.gen, setup.blocks).expect("a gentle grassy slope near spawn");
    for scene in [mountain_scene(&setup.gen), slope] {
        assert!(
            scene.eye_to_box() >= MIN_EYE_TO_BOX,
            "{}: the eye is only {:.0} blocks from the box",
            scene.name,
            scene.eye_to_box()
        );
        let ground = setup
            .gen
            .surface_height(scene.eye[0] as i32, scene.eye[2] as i32);
        assert!(
            scene.eye[1] >= f64::from(ground) + 2.0,
            "{}: under ground",
            scene.name
        );
        assert!((scene.look.length() - 1.0).abs() < 1e-4);
    }
}

#[test]
#[ignore = "scouting: prints the two scenes"]
fn print_scenes() {
    let setup = Setup::new(false);
    let m = mountain_scene(&setup.gen);
    println!("{m:#?}");
    let s = slope_scene(&setup.gen, setup.blocks);
    println!("{s:#?}");
}

// ---------------------------------------------------------------------------
// The GPU runs: minutes in release, so all `#[ignore]`d
// ---------------------------------------------------------------------------

/// Every image of one scene, and the backdrop that tells box content from the
/// far terrain around it.
struct Run {
    scene: Scene,
    backdrop: Frame,
    images: Vec<(Rep, Frame)>,
}

/// Render the backdrop and the five representations. `None` without a GPU.
fn run_scene(setup: &Setup, scene: &Scene) -> Option<Run> {
    let far_set = FarSet::build(setup, scene);
    let backdrop = render(setup, &far_set, scene, None)?;
    let mut images = Vec::new();
    for rep in Rep::ALL {
        images.push((rep, render(setup, &far_set, scene, Some(rep))?));
    }
    // A scene whose box is out of view measures nothing and would pass
    // anything: the level-0 box content must fill a real share of the image
    // (chosen: a tenth).
    let content = box_mask(&images[0].1, &backdrop)
        .iter()
        .filter(|b| **b)
        .count();
    assert!(
        content * 10 >= (SIZE.0 * SIZE.1) as usize,
        "{}: the box is only {:.1}% of the image",
        scene.name,
        100.0 * content as f32 / (SIZE.0 * SIZE.1) as f32
    );
    Some(Run {
        scene: scene.clone(),
        backdrop,
        images,
    })
}

impl Run {
    fn reference(&self) -> &Frame {
        &self.images[0].1
    }

    /// Level 0's own share of the box content: the baseline of the stone delta.
    fn reference_share(&self) -> Share {
        material_share(
            self.reference(),
            &box_mask(self.reference(), &self.backdrop),
            &real_ends(),
        )
    }

    /// Every representation but level 0, against level 0.
    fn metrics(&self) -> Vec<Metrics> {
        self.images[1..]
            .iter()
            .map(|(rep, image)| {
                measure(
                    *rep,
                    image,
                    self.reference(),
                    &self.backdrop,
                    self.scene.subject_distance,
                    &real_ends(),
                )
            })
            .collect()
    }

    fn save(&self, dir: &Path, tag: &str) -> Vec<PathBuf> {
        let mut paths = Vec::new();
        let mut put = |name: String, frame: &Frame| {
            let path = dir.join(format!("{}-{tag}-{name}.png", self.scene.name));
            save_png(frame, &path);
            paths.push(path);
        };
        put("backdrop".to_string(), &self.backdrop);
        for (rep, image) in &self.images {
            put(rep.name(), image);
        }
        paths
    }
}

fn describe(scene: &Scene) -> String {
    format!(
        "{}: eye ({:.0}, {:.0}, {:.0}), look ({:.2}, {:.2}, {:.2}), box chunks min {:?} ext {:?}, {}x{}",
        scene.name,
        scene.eye[0],
        scene.eye[1],
        scene.eye[2],
        scene.look.x,
        scene.look.y,
        scene.look.z,
        scene.box_min,
        scene.box_ext,
        SIZE.0,
        SIZE.1
    )
}

fn table(run: &Run) -> String {
    let base = run.reference_share();
    let mut out = format!(
        "| representation | tile px | sil max px | sil p99 px | tile p50 | tile p99 | tile max | stone | grass | soil | stone delta pp | unmodelled | land px |\n\
         |---|---|---|---|---|---|---|---|---|---|---|---|---|\n\
         | level0 (reference) | - | 0 | 0 | 0 | 0 | 0 | {:.1}% | {:.1}% | {:.1}% | 0 | {:.1}% | {} |\n",
        100.0 * base.stone,
        100.0 * base.grass,
        100.0 * base.soil,
        100.0 * base.unmodelled,
        base.land
    );
    for m in run.metrics() {
        out += &format!(
            "| {} | {} | {:.0} | {:.0} | {:.3} | {:.3} | {:.3} | {:.1}% | {:.1}% | {:.1}% | {:+.1} | {:.1}% | {} |\n",
            m.rep.name(),
            m.tiles.tile_px,
            m.silhouette.max_px,
            m.silhouette.p99_px,
            m.tiles.p50,
            m.tiles.p99,
            m.tiles.max,
            100.0 * m.share.stone,
            100.0 * m.share.grass,
            100.0 * m.share.soil,
            m.stone_delta_pp,
            100.0 * m.share.unmodelled,
            m.share.land
        );
    }
    out
}

/// Today's numbers: five representations, both scenes, every metric, and the
/// images. `trees` on and off, since trees are a separate question from the
/// ground (level 3 draws boxes).
#[test]
#[ignore = "a measurement: renders both scenes at five representations, minutes in release"]
fn print_consistency_report() {
    let dir = output_dir();
    // Oaks matter little (measured once: every number within 1 point), so the
    // default run is the ground alone; `CUBARA_PROBE_TREES=1` adds the oaks.
    let with_trees = std::env::var_os("CUBARA_PROBE_TREES").is_some();
    for trees in [false, true].into_iter().filter(|t| !t || with_trees) {
        let setup = Setup::new(trees);
        let slope =
            slope_scene(&setup.gen, setup.blocks).expect("a gentle grassy slope near spawn");
        let scenes = [mountain_scene(&setup.gen), slope];
        for scene in &scenes {
            let started = std::time::Instant::now();
            let Some(run) = run_scene(&setup, scene) else {
                cubara_render::headless::skip_or_require_gpu("print_consistency_report");
                return;
            };
            let tag = if trees { "trees" } else { "ground" };
            println!(
                "\n### {} ({}), {:.0} s\n{}\n{}",
                scene.name,
                if trees { "oaks on" } else { "oaks off" },
                started.elapsed().as_secs_f32(),
                describe(scene),
                table(&run)
            );
            for path in run.save(&dir, tag) {
                println!("image: {}", path.display());
            }
        }
    }
}

/// The targets. **Chosen, not derived**, from the first measurement (the table
/// in the PR): a silhouette within about 4 blocks (1 block is 1.5 px at the
/// mountain's 420 blocks), a tile's mean colour within 8% at the 99th
/// percentile and 12% at the worst, a median tile within 2%, the stone share
/// within 5 points of level 0. Loose enough that a correct level-1 passes and
/// today's level 2 and 3 and far terrain do not; the fix PRs tighten them as
/// they land. Held against the ground (oaks off) of both scenes.
const TARGETS: Targets = Targets {
    silhouette_max_px: 6.0,
    silhouette_p99_px: 4.0,
    tile_p50: 0.02,
    tile_p99: 0.08,
    tile_max: 0.12,
    stone_delta_pp: 5.0,
};

/// Every representation shows the same ground as level 0, within [`TARGETS`].
/// Red today: the coarse levels and the far terrain disagree with level 0
/// (#423, #422). The fix PRs un-ignore it.
#[test]
#[ignore = "red until #423 / #422 land"]
fn levels_show_the_same_ground() {
    let setup = Setup::new(false);
    let slope = slope_scene(&setup.gen, setup.blocks).expect("a gentle grassy slope near spawn");
    let mut violations = Vec::new();
    for scene in [mountain_scene(&setup.gen), slope] {
        let Some(run) = run_scene(&setup, &scene) else {
            cubara_render::headless::skip_or_require_gpu("levels_show_the_same_ground");
            return;
        };
        for m in run.metrics() {
            violations.extend(
                TARGETS
                    .violations(&m)
                    .into_iter()
                    .map(|v| format!("{}: {v}", scene.name)),
            );
        }
    }
    assert!(
        violations.is_empty(),
        "the levels disagree with level 0:\n{}",
        violations.join("\n")
    );
}

/// The instrument can pass and fail, on the real renderer: level 0 against a
/// second render of itself is exactly zero (it passes, and the render is
/// deterministic), a deliberately miscoloured copy of it fails, and a forced
/// level 3 is not zero (the instrument is not blind).
#[test]
#[ignore = "GPU: renders the slope scene, minutes in release"]
fn the_instrument_passes_itself_and_fails_a_wrong_image() {
    let setup = Setup::new(false);
    let scene = slope_scene(&setup.gen, setup.blocks).expect("a gentle grassy slope near spawn");
    let far_set = FarSet::build(&setup, &scene);
    let Some(backdrop) = render(&setup, &far_set, &scene, None) else {
        cubara_render::headless::skip_or_require_gpu("the_instrument_passes_itself");
        return;
    };
    let level0 = render(&setup, &far_set, &scene, Some(Rep::Level(0))).unwrap();
    let again = render(&setup, &far_set, &scene, Some(Rep::Level(0))).unwrap();
    let measure_against_level0 = |rep, image: &Frame| {
        measure(
            rep,
            image,
            &level0,
            &backdrop,
            scene.subject_distance,
            &real_ends(),
        )
    };

    let same = measure_against_level0(Rep::Level(0), &again);
    assert!(
        TARGETS.violations(&same).is_empty()
            && same.silhouette.max_px == 0.0
            && same.tiles.max == 0.0,
        "level 0 against a second render of itself must be exactly zero: {same:?}"
    );
    assert!(
        same.tiles.tiles > 0 && same.silhouette.columns > 0,
        "nothing was measured"
    );

    // Miscoloured on purpose: every box pixel recoloured grey (stone for grass).
    let mask = box_mask(&level0, &backdrop);
    let mut grey = copy(&level0);
    for (p, in_box) in grey.pixels.as_chunks_mut::<4>().0.iter_mut().zip(&mask) {
        if *in_box && !shows_nothing(&p[..]) {
            let v = ((u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2])) / 3) as u8;
            p[..3].copy_from_slice(&[v, v, v]);
        }
    }
    let wrong = measure_against_level0(Rep::Level(0), &grey);
    let violations = TARGETS.violations(&wrong);
    assert!(
        violations.iter().any(|v| v.contains("stone share")),
        "a grey copy must fail the targets: {violations:?} ({wrong:?})"
    );

    let level3 = render(&setup, &far_set, &scene, Some(Rep::Level(3))).unwrap();
    let coarse = measure_against_level0(Rep::Level(3), &level3);
    assert!(
        coarse.tiles.max > 0.0 || coarse.silhouette.max_px > 0.0,
        "level 3 must not be identical to level 0: {coarse:?}"
    );
}

#[test]
fn percentile_is_nearest_rank() {
    let v: Vec<f32> = (1..=100).map(|i| i as f32).collect();
    assert_eq!(percentile(v.clone(), 500), 50.0);
    assert_eq!(percentile(v.clone(), 990), 99.0);
    assert_eq!(percentile(v, 1000), 100.0);
    assert_eq!(percentile(Vec::new(), 990), 0.0);
}

/// A colour outside the stone/grass/soil cone (a saturated blue) is reported as
/// not fitting, and what is reported for it is still a valid set of shares.
#[test]
fn a_colour_outside_the_materials_does_not_fit_and_still_sums_to_one() {
    let ends = ends();
    let mut misfits = 0;
    for rgb in [[10, 10, 250], [250, 10, 10], [250, 10, 250], [10, 250, 250]] {
        // A colour so far outside that the total goes negative reads as "too
        // dark to say"; the rest must say they do not fit.
        let Some((shares, fits)) = ends.unmix(&[rgb[0], rgb[1], rgb[2], 255]) else {
            continue;
        };
        misfits += usize::from(!fits);
        assert!(shares.iter().all(|v| *v >= 0.0), "{rgb:?}: {shares:?}");
        assert!(
            (shares.iter().sum::<f32>() - 1.0).abs() < 1e-4,
            "{rgb:?}: {shares:?}"
        );
    }
    assert!(
        misfits > 0,
        "no saturated colour was reported as not fitting"
    );
}

// ---------------------------------------------------------------------------
// The approach sequence
// ---------------------------------------------------------------------------

mod approach_tests {
    use super::super::approach::{self, bands, distances};
    use super::*;
    use cubara_world::node::DEFAULT_RING_SCHEDULE;

    #[test]
    fn the_bands_follow_the_ring_schedule() {
        let b = bands(DEFAULT_RING_SCHEDULE);
        let want = [
            ("level0", 0.0, 160.0),
            ("level1", 160.0, 288.0),
            ("level2", 288.0, 512.0),
            ("level3", 512.0, 1024.0),
            ("far", 1024.0, 2048.0),
        ];
        assert_eq!(b.len(), want.len());
        for (band, (name, from, to)) in b.iter().zip(want) {
            assert_eq!((band.name.as_str(), band.from, band.to), (name, from, to));
        }
        // A different schedule gives different bands: nothing is hard-coded.
        let other = bands(&[(0, 4), (2, 8)]);
        assert_eq!(other[1].to, 128.0);
    }

    #[test]
    fn one_eye_in_each_band_and_two_far() {
        let d = distances(DEFAULT_RING_SCHEDULE);
        let names: Vec<&str> = d.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            ["level0", "level1", "level2", "level3", "far", "far"]
        );
        let b = bands(DEFAULT_RING_SCHEDULE);
        for ((name, dist), band) in d.iter().zip(&b).take(4) {
            assert!(
                band.from < *dist && *dist < band.to,
                "{name} at {dist} is outside {band:?}"
            );
        }
        assert_eq!(d[4].1, 1200.0);
        assert_eq!(d[5].1, 2048.0);
    }

    #[test]
    fn the_approach_is_straight_level_and_toward_the_peak() {
        let setup = Setup::new(false);
        let a = approach::approach(&setup.gen, DEFAULT_RING_SCHEDULE);
        let steps = &a.steps;
        assert_eq!(steps.len(), 6);
        let y = steps[0].scene.eye[1];
        let h = glam::Vec2::new(steps[0].scene.look.x, steps[0].scene.look.z).normalize();
        for s in steps {
            assert_eq!(s.scene.eye[1], y, "the approach is level");
            let to_peak = glam::DVec2::new(
                f64::from(a.peak[0]) - s.scene.eye[0],
                f64::from(a.peak[2]) - s.scene.eye[2],
            );
            assert!((to_peak.length() - s.distance).abs() < 1e-6, "{s:?}");
            let dir = to_peak.normalize();
            assert!((dir.x as f32 - h.x).abs() < 1e-4 && (dir.y as f32 - h.y).abs() < 1e-4);
            assert!(
                s.scene.look.y.abs() <= 25f32.to_radians().sin() + 1e-4,
                "pitch is bounded"
            );
        }
        // Nearest first, strictly farther each time.
        assert!(steps.windows(2).all(|w| w[0].distance < w[1].distance));
        // One footprint for every eye, holding the peak.
        let hole = steps[0].scene.hole();
        assert!(steps.iter().all(|s| s.scene.hole() == hole));
        assert!(hole.min[0] <= f64::from(a.peak[0]) && f64::from(a.peak[0]) < hole.max[0]);
        assert!(hole.min[2] <= f64::from(a.peak[2]) && f64::from(a.peak[2]) < hole.max[2]);
    }

    fn prepare() -> (Setup, approach::Approach, u32) {
        let setup = Setup::new(false);
        let a = approach::approach(&setup.gen, DEFAULT_RING_SCHEDULE);
        let truth = approach::truth_level(&setup, &a.steps[0].scene);
        (setup, a, truth)
    }

    /// The owner's question, as numbers: at every distance the normal view of
    /// the mountain against every block rendered at the same eye, and the
    /// images of both.
    #[test]
    #[ignore = "a measurement: renders the approach, minutes in release"]
    fn print_approach_report() {
        let dir = output_dir();
        let (setup, a, truth) = prepare();
        let ends = real_ends();
        println!(
            "\n### approach to the peak at {:?}: eye height {:.0}, distances {:?}, truth = level {truth}{}",
            a.peak,
            a.steps[0].scene.eye[1],
            a.steps.iter().map(|s| s.distance).collect::<Vec<_>>(),
            if truth == 0 { "" } else { " (level 0 does not fit the arena)" },
        );
        println!(
            "bands (blocks): {:?}",
            approach::bands(DEFAULT_RING_SCHEDULE)
        );
        let mut rows = Vec::new();
        for step in &a.steps {
            let Some(r) = approach::run_step(&setup, step, a.peak, truth, &ends) else {
                cubara_render::headless::skip_or_require_gpu("print_approach_report");
                return;
            };
            for (frame, kind) in [(&r.normal, "normal"), (&r.truth, "truth")] {
                let path = dir.join(format!("approach-{:04.0}-{kind}.png", step.distance));
                save_png(frame, &path);
                println!("image: {}", path.display());
            }
            rows.push(r.row);
        }
        println!("{}", approach::table(&rows.iter().collect::<Vec<_>>()));
    }

    /// What the normal view of the mountain may differ from every block
    /// rendered, at any distance. **Chosen, not derived**: the same numbers as
    /// the forced-level targets, since "looks the same from afar as up close"
    /// is the same claim.
    #[test]
    #[ignore = "red until #423 / #422 land"]
    fn a_mountain_looks_the_same_from_afar_as_up_close() {
        let (setup, a, truth) = prepare();
        let ends = real_ends();
        let mut violations = Vec::new();
        for step in &a.steps {
            let Some(r) = approach::run_step(&setup, step, a.peak, truth, &ends) else {
                cubara_render::headless::skip_or_require_gpu("a_mountain_looks_the_same");
                return;
            };
            violations.extend(
                TARGETS
                    .violations(&r.row.metrics)
                    .into_iter()
                    .map(|v| format!("{:.0} blocks: {v}", step.distance)),
            );
        }
        assert!(
            violations.is_empty(),
            "the mountain changes as you approach:\n{}",
            violations.join("\n")
        );
    }
}

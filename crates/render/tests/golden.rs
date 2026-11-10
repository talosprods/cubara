//! Golden-image tests: render a fixed scene headlessly and compare it against a
//! committed reference (`ARCHITECTURE.md` Rule 6 — behaviour is pinned by tests
//! before it is rewritten).
//!
//! This is the test that makes the rest of the architecture work usable. The point
//! of the boundaries is that a subsystem can be deleted and rebuilt; the thing that
//! makes that *safe* is a check that says whether the rebuild still draws the same
//! world. Without it, "does it still render correctly?" can only be answered by a
//! human looking at a window, which does not scale and does not run on the next
//! commit.
//!
//! **Regenerating the reference** is deliberate, never automatic:
//!
//! ```bash
//! CUBARA_BLESS=1 cargo test -p cubara-render --test golden
//! ```
//!
//! Only do that when the visual change is intended, and look at the new image
//! before committing it — blessing on autopilot turns this file into decoration.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use cubara_render::headless::{self, Frame, Shot};
use cubara_render::materials::TextureLayers;
use cubara_render::{
    HotbarSlot, InventoryPanel, Lighting, MeshedNode, NodeId, PanelLayout, PanelSlotKind, TreeModel,
};
use cubara_voxel::{BlockId, BlockRegistry, Chunk, ChunkCoord};
use cubara_world::mesh::{mesh_nodes, mesh_region};
use cubara_world::node::{desired_nodes, schedule_for_radius};
use cubara_world::{TerrainBlocks, World};

/// The real `assets/blocks` registry -- the same one every entry point
/// (window, `--bench`, `--screenshot`, these tests) meshes and renders
/// against.
/// The real terrain palette, trees and ores included -- the same one the game
/// plays with, so a golden shows what a player sees.
fn real_blocks() -> TerrainBlocks {
    let r = real_registry();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let structures = cubara_voxel::StructureRegistry::load(&root.join("assets/structures"))
        .expect("assets/structures");
    let ores = cubara_voxel::OreRegistry::load(&root.join("assets/ores")).expect("assets/ores");
    TerrainBlocks::from_registry(&r)
        .with_oak(&structures, &r)
        .with_ores(&ores, &r)
}

fn real_registry() -> BlockRegistry {
    let assets_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/blocks");
    BlockRegistry::load(&assets_dir).expect("assets/blocks must be valid")
}

/// Which static model a node at `level` draws its trees with (issue #332
/// decision 3) -- mirrors `cubara_app::streaming::tree_model_for`, which
/// this test cannot import (`cubara-render`'s own tests cannot depend on
/// `cubara-app`; dependencies point one way, `ARCHITECTURE.md` Rule 3).
fn tree_model_for(level: u32) -> TreeModel {
    if level <= 2 {
        TreeModel::Oak
    } else {
        TreeModel::Box
    }
}

/// [`cubara_world::mesh::BuiltNode`] converted to [`MeshedNode`] -- mirrors
/// `cubara_app::streaming::to_meshed_node` for the same reason
/// [`tree_model_for`] does.
fn to_meshed_node(built: cubara_world::mesh::BuiltNode) -> Option<MeshedNode> {
    let geometry = built.geometry?;
    let tree_model = tree_model_for(built.node.level);
    Some(MeshedNode {
        id: NodeId {
            level: built.node.level,
            pos: built.node.pos,
        },
        origin: geometry.origin,
        scale: geometry.scale,
        mesh: geometry.mesh,
        local_aabb: geometry.local_aabb,
        trees: built.trees.iter().map(|t| [t.x, t.base, t.z]).collect(),
        tree_model,
    })
}

/// Serialises every `wgpu::Instance`/adapter/device creation and teardown in
/// this binary (issue #265): cargo runs this binary's `#[test]` fns on
/// several threads by default, and each one independently opens and drops a
/// Vulkan instance+device. That races against known thread-safety bugs in the
/// Khronos Vulkan Loader's own instance/device bookkeeping -- see
/// <https://github.com/KhronosGroup/Vulkan-Loader/issues/200> ("Race condition
/// in vkCreateDevice()+vkDestroyDevice()+vkGetDeviceProcAddr()"), fixed
/// upstream the same way this does: a lock around create/destroy. Confirmed
/// here by two independent crashes faulting at the identical code offset
/// inside `libvulkan.so.1.4.341` (not wgpu, not the NVIDIA driver, not this
/// crate) -- `journalctl -k` showed `... in libvulkan.so.1.4.341[236fe,...]`
/// both times, from different tests, meaning the crashing thread was
/// incidental, not test-specific. `--test-threads=1` and `--release` (whose
/// timing apparently never hits the window) always passed, and so does `gdb`
/// (which serialises/slows scheduling enough to avoid the race), which is why
/// this could not be pinned down with a debugger.
///
/// This is a test-harness lock, not simulation state: `ARCHITECTURE.md` Rule
/// 2 ("no ambient state") governs the engine and render crates, not a fixture
/// working around a driver bug. Meshing (the CPU-bound part of a shot) still
/// runs unlocked and in parallel; only the GPU setup/teardown is serialised.
static GPU_LIFECYCLE: Mutex<()> = Mutex::new(());

/// Mesh `world` out to `shot.region_radius` (the same ring-schedule
/// truncation the live renderer streams, `schedule_for_radius`) and render
/// it. `cubara-render` never touches a `World` (`ARCHITECTURE.md` §1), so
/// this is the test-side equivalent of what `cubara-app` does every frame:
/// mesh via `cubara_world::mesh`, then hand the result to
/// `headless::render`. `TextureLayers::from_registry` resolves texture
/// layers deterministically without needing a GPU device for this step --
/// `headless::render` builds the real texture array afterwards, and the two
/// are guaranteed to agree on layer numbers (see its doc comment).
fn render_world(world: &World, shot: Shot) -> Option<Frame> {
    let registry = real_registry();
    let layers = TextureLayers::from_registry(&registry);
    let layer_of = |name: &str| layers.layer_of(name);
    let schedule = schedule_for_radius(shot.region_radius);
    let meshed = mesh_region(
        world,
        &registry,
        &layer_of,
        ChunkCoord::new(0, 0, 0),
        0..=2,
        &schedule,
        real_blocks(),
    )
    .into_iter()
    .filter_map(to_meshed_node);
    let _gpu = GPU_LIFECYCLE.lock().unwrap_or_else(|e| e.into_inner());
    headless::render(meshed, shot)
}

/// Per-channel difference treated as equal. The same scene rasterises slightly
/// differently across backends and driver versions; an exact match would fire
/// constantly and the test would be deleted rather than trusted.
const TOLERANCE: u8 = 12;

/// Fraction of pixels allowed to exceed [`TOLERANCE`].
///
/// Calibrated from measurement, not taste. The reference is blessed on one
/// backend and checked everywhere, so the number that matters is the
/// *cross-backend* delta:
///
/// | Signal | Differing pixels |
/// |---|---|
/// | Same machine, same scene, twice | **0.0000%** (exact, since #81) |
/// | No mips, Metal-blessed: macOS CI | 0.0000% (max channel delta 1) |
/// | No mips, Metal-blessed: Windows CI (DX12) | 0.0215% (max delta 79) |
/// | **With mips, Vulkan-blessed: macOS CI (Metal)** | **0.0654%** (max delta 73) |
/// | **With mips, Vulkan-blessed: Windows CI (DX12)** | **0.2091%** (max delta 173) |
/// | A gash carved across the framed region | **4.07%** |
/// | This threshold | 0.5% |
///
/// ~2.4x above the worst measured backend difference, and ~8x below an obvious
/// regression.
///
/// **Re-calibrated upward when the mip chain landed, and that is a real loss of
/// sensitivity, not a rubber stamp.** Two sources of cross-backend difference
/// now stack. The old one is silhouette-edge pixels rasterising differently
/// between APIs. The new one is *mip level selection*: which level a backend
/// picks for a given pixel is implementation-defined, and at the grazing angles
/// in the `materials` shot two backends disagree by a whole level on the same
/// pixel — which in a high-contrast striped tile is a large per-pixel delta,
/// hence the jump from a max of 79 to 173. That is inherent to having mips at
/// all; the alternative was aliasing at the horizon.
///
/// The margin above noise fell from ~9x to ~2.4x, so this threshold is now the
/// weaker of the two guards. If it needs raising again, the honest move is to
/// question whether the shot should be that grazing, not to keep nudging the
/// number.
///
/// **The rule this number now lives under, so the next person does not have to
/// re-derive it:** raising this threshold is spending the sensitivity of the
/// only automated test that covers rendering at all. It is worth spending to
/// fix a defect — the mip bump above bought the horizon not shimmering. It is
/// not worth spending on something that only looks nicer, because a threshold
/// wide enough to admit cosmetic churn is a threshold too wide to catch the
/// regression it exists for.
///
/// 4x MSAA is the concrete case that was declined on exactly this basis; see
/// issue #132 for the numbers and the argument, including why "render the
/// goldens without it" is not a way out (it makes two render paths, which
/// `ARCHITECTURE.md` Rule 5 and `scripts/check-single-render-path.sh` forbid).
///
/// **The reference set moved from macOS/Metal to Windows/Vulkan** when the mip
/// chain landed, because that is the machine it was re-blessed on. The whole
/// set moved together rather than splitting across two backends: two of the
/// five images (`cave_mouth`, `outline`) are unaffected by mips and were
/// re-blessed *only* to keep the set on one backend — they measured 0.055% and
/// 0.054% against the Metal reference, already inside this threshold. CI
/// checking Metal and DX12 against a Vulkan reference is what now holds the
/// numbers above honest; the `Golden image deltas` workflow step prints them
/// on every run.
///
/// Do not tighten this to hug the measured number. A golden test that fires on a
/// driver update gets muted, and a muted test is worse than no test.
const MAX_DIFFERING: f64 = 0.005;

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn load_png(path: &Path) -> (u32, u32, Vec<u8>) {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("read golden {}: {e}", path.display()))
        .to_rgba8();
    (img.width(), img.height(), img.into_raw())
}

fn save_png(path: &Path, w: u32, h: u32, pixels: &[u8]) {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).expect("create golden dir");
    }
    image::save_buffer(path, pixels, w, h, image::ExtendedColorType::Rgba8)
        .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

/// Render `shot` and compare against `tests/golden/<name>.png`.
fn assert_golden(name: &str, world: &World, shot: Shot) {
    assert_frame_golden(name, render_world(world, shot));
}

/// [`assert_golden`] for a frame already rendered.
fn assert_frame_golden(name: &str, frame: Option<Frame>) {
    let Some(frame) = frame else {
        // No adapter (a GPU-less CI runner). Skipping is honest here — the
        // alternative is a red build that says nothing about the code — but it is
        // reported loudly so a silently-never-running test is noticeable, and
        // `CUBARA_REQUIRE_GPU=1` turns it into a hard failure on a runner that is
        // supposed to have an adapter (test audit #4).
        headless::skip_or_require_gpu(name);
        return;
    };

    let path = golden_dir().join(format!("{name}.png"));

    if std::env::var_os("CUBARA_BLESS").is_some() {
        save_png(&path, frame.width, frame.height, &frame.pixels);
        eprintln!("BLESSED {}", path.display());
        return;
    }

    assert!(
        path.exists(),
        "no golden image at {}. If this is a new test, create it with:\n    \
         CUBARA_BLESS=1 cargo test -p cubara-render --test golden",
        path.display()
    );

    let (gw, gh, expected) = load_png(&path);
    assert_eq!(
        (gw, gh),
        (frame.width, frame.height),
        "golden {name} is {gw}x{gh}, rendered {}x{}",
        frame.width,
        frame.height
    );

    let diff = headless::compare(&frame.pixels, &expected, TOLERANCE);
    // Always reported, not just on failure: the reference is generated on one
    // backend and checked on others, and this line is how the real cross-backend
    // delta becomes visible in CI logs instead of guessed at.
    eprintln!(
        "golden {name}: {:.4}% differ (tolerance {TOLERANCE}, max delta {}), threshold {:.4}%",
        diff.differing_fraction * 100.0,
        diff.max_channel_delta,
        MAX_DIFFERING * 100.0
    );
    if diff.differing_fraction > MAX_DIFFERING {
        // Write what was actually rendered next to the reference, so the failure is
        // diagnosable instead of just numeric.
        let actual = golden_dir().join(format!("{name}.actual.png"));
        save_png(&actual, frame.width, frame.height, &frame.pixels);
        panic!(
            "golden {name} differs: {:.2}% of pixels exceed tolerance {TOLERANCE} \
             (max channel delta {}), allowed {:.2}%.\n  expected: {}\n  actual:   {}\n\
             If the change is intended: CUBARA_BLESS=1 cargo test -p cubara-render --test golden",
            diff.differing_fraction * 100.0,
            diff.max_channel_delta,
            MAX_DIFFERING * 100.0,
            path.display(),
            actual.display(),
        );
    }
}

/// A rectangle in pixel space: `(x, y, width, height)`.
type Rect = (u32, u32, u32, u32);

/// Crop a tightly-packed RGBA8 buffer of row stride `stride` pixels to `rect`.
fn crop(pixels: &[u8], stride: u32, rect: Rect) -> Vec<u8> {
    let (x, y, w, h) = rect;
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for row in y..y + h {
        let start = ((row * stride + x) * 4) as usize;
        let end = start + (w * 4) as usize;
        out.extend_from_slice(&pixels[start..end]);
    }
    out
}

/// Test-quality audit #5 ("golden tolerance swallows the thing named in the
/// title", `handover/2026-11-09-windows`): [`MAX_DIFFERING`] lets ~0.5% of a
/// frame move anywhere -- a few thousand pixels, which is well *more* than a
/// missing half-heart, hotbar count, outline, ore vein, player figure, or LOD
/// seam actually changes. The whole-image check alone cannot tell a backend
/// quirk from exactly the feature a test's title names disappearing, which is
/// what every rect this is called with was chosen to catch (see each call
/// site, and the PR description's injected-defect table).
///
/// Re-reads the *same* committed golden [`assert_golden`]/[`assert_frame_golden`]
/// already compare the whole frame against -- never a second reference, and
/// never regenerated here (a `CUBARA_BLESS` run returns immediately: the
/// golden this would read is the one about to be overwritten, so comparing
/// against it proves nothing). `rect` must match within [`TOLERANCE`]
/// everywhere but at most `max_differing` pixels -- zero at every call site
/// where this was measured to hold exactly, and a small, explicitly justified
/// (per call site) allowance where the region still contains rasterised
/// terrain or text-under-alpha-blend, which real cross-backend noise does
/// touch even inside a tight box.
fn assert_region_golden(name: &str, frame: &Frame, rect: Rect, max_differing: u32, what: &str) {
    if std::env::var_os("CUBARA_BLESS").is_some() {
        return;
    }
    let path = golden_dir().join(format!("{name}.png"));
    let (gw, gh, golden) = load_png(&path);
    assert_eq!(
        (gw, gh),
        (frame.width, frame.height),
        "golden {name} is {gw}x{gh}, rendered {}x{}",
        frame.width,
        frame.height
    );
    let (x, y, w, h) = rect;
    assert!(
        x + w <= frame.width && y + h <= frame.height,
        "region {rect:?} for {name}/{what} is out of bounds for a {}x{} frame",
        frame.width,
        frame.height
    );
    let actual = crop(&frame.pixels, frame.width, rect);
    let expected = crop(&golden, gw, rect);
    let diff = headless::compare(&actual, &expected, TOLERANCE);
    let differing = (diff.differing_fraction * (w * h) as f64).round() as u32;
    eprintln!(
        "golden {name}/{what}: {differing} of {} pixels in {rect:?} differ from the \
         golden (tolerance {TOLERANCE}), allowed {max_differing}",
        w * h
    );
    assert!(
        differing <= max_differing,
        "golden {name}/{what}: {differing} pixels inside {rect:?} differ from the \
         committed golden (tolerance {TOLERANCE}), more than the {max_differing}-pixel \
         cross-backend noise budget for this region -- the thing it is named for \
         (test-quality audit #5) looks wrong here, not just noisy"
    );
}

/// Perceptual luminance (ITU-R BT.601, integer) of one RGBA8 pixel, ignoring
/// alpha -- every region [`assert_region_coverage_golden`] is used for is
/// drawn fully opaque.
fn luma(pixel: &[u8]) -> u32 {
    (299 * pixel[0] as u32 + 587 * pixel[1] as u32 + 114 * pixel[2] as u32) / 1000
}

/// How many pixels of a cropped region are "lit" -- luminance over
/// `threshold` -- out of the region's total.
fn lit_count(pixels: &[u8], threshold: u32) -> u32 {
    pixels
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|p| luma(*p) > threshold)
        .count() as u32
}

/// [`assert_region_golden`]'s sibling for a region where plain per-pixel
/// comparison cannot clear a healthy noise/defect margin (test-quality audit
/// #5, PR #376): the hotbar's "12" count is an 8x8 bitmap glyph upscaled
/// 1.5x through a `Nearest` sampler (`crates/render/src/text/mod.rs`), and at
/// that non-integer scale factor, which source texel a destination pixel
/// rounds to is the kind of rasterisation tie that backends settle
/// differently -- not just at the glyph's silhouette edge but through its
/// interior, since every row of the upscale has a tie somewhere. That moves
/// *which* pixels inside the glyph are foreground vs shadow vs background
/// more than a plain per-pixel count can budget for without also admitting
/// most of what disabling the glyph draw changes (measured: disabling it
/// changes 153 of this region's pixels per plain comparison, against 73
/// measured noise on Windows CI -- only a 2.1x margin, well under the 3x
/// this file aims for; see `the_hotbar_shows_slots_counts_and_the_held_one`).
///
/// So this compares *coverage* instead of position: how many pixels in the
/// region are lit (luminance over `threshold`) in each image, not which
/// ones. Shuffling ties among the glyph's own foreground pixels leaves that
/// count almost unchanged -- the glyph is still "mostly on" in the same
/// places -- while disabling the draw collapses it to zero, so the two are
/// far apart on this measure even where they are close on a pixel-position
/// one. `threshold` matters: it must sit above every opaque background this
/// region can be drawn over (read back sRGB-encoded off `Rgba8UnormSrgb`,
/// not the naive `color * 255`) or the count mixes background-crossing
/// noise into the glyph signal -- see the call site for a case where the
/// first threshold tried did exactly that.
fn assert_region_coverage_golden(
    name: &str,
    frame: &Frame,
    rect: Rect,
    threshold: u32,
    max_coverage_delta: u32,
    what: &str,
) {
    if std::env::var_os("CUBARA_BLESS").is_some() {
        return;
    }
    let path = golden_dir().join(format!("{name}.png"));
    let (gw, gh, golden) = load_png(&path);
    assert_eq!(
        (gw, gh),
        (frame.width, frame.height),
        "golden {name} is {gw}x{gh}, rendered {}x{}",
        frame.width,
        frame.height
    );
    let (x, y, w, h) = rect;
    assert!(
        x + w <= frame.width && y + h <= frame.height,
        "region {rect:?} for {name}/{what} is out of bounds for a {}x{} frame",
        frame.width,
        frame.height
    );
    let actual = crop(&frame.pixels, frame.width, rect);
    let expected = crop(&golden, gw, rect);
    let actual_lit = lit_count(&actual, threshold);
    let expected_lit = lit_count(&expected, threshold);
    let delta = actual_lit.abs_diff(expected_lit);
    eprintln!(
        "golden {name}/{what}: {actual_lit} of {} pixels lit (luminance > {threshold}) vs \
         {expected_lit} in the golden, delta {delta}, allowed {max_coverage_delta}",
        w * h
    );
    assert!(
        delta <= max_coverage_delta,
        "golden {name}/{what}: {actual_lit} lit pixels inside {rect:?} vs {expected_lit} in \
         the committed golden (luminance threshold {threshold}), a coverage delta of {delta} -- \
         more than the {max_coverage_delta}-pixel cross-backend noise budget for this region -- \
         the thing it is named for (test-quality audit #5) looks wrong here, not just noisy"
    );
}

#[test]
fn the_same_scene_renders_byte_identically() {
    // The property that makes every other visual test trustworthy, and the one
    // multithreading will depend on: rendering must not vary run to run.
    //
    // This failed before issue #81 — 0.006% of pixels differed with a max channel
    // delta of 63, because `ChunkArena` iterated a `HashMap` when building the
    // indirect draw list, so chunks submitted in a different order each run and
    // depth ties on silhouette edges resolved differently. Meshing already runs on
    // a worker pool, so "whatever order results arrived in" was leaking into the
    // rendered frame.
    //
    // **What "identically" means here, measured rather than assumed (#161).**
    // This asserted byte-exactness and flaked: roughly one run in three, 1-2
    // isolated pixels differed by exactly 1 in a single channel. 16 runs of a
    // diagnostic gave 5 divergences, every one 1-2 pixels at max delta 1 with
    // *zero* adjacency -- the signature of last-bit rasterisation/interpolation
    // rounding on this GPU, not of anything the engine controls.
    //
    // So the exact comparison was not a true property, and the old failure
    // message ("draw order or meshing is leaking nondeterminism") asserted a
    // cause that was wrong for what actually happened. The bounds below are what
    // *is* true, and they still fail on the #81 bug this test exists to catch.
    //
    // Scope, stated honestly: `render_world`/`mesh_region` mesh synchronously, so
    // this pins *draw-order* determinism, not worker-arrival determinism. Arrival
    // order still decides which slab offsets a chunk lands in (`ChunkArena::insert`
    // is first-fit), which no longer changes the image — the draw list is
    // coord-sorted and the geometry content is identical — but does mean the
    // arena's internal layout is not reproducible. That matters the moment world
    // state is hashed or saved; see issue #83.
    let shot = Shot::default();
    let Some(a) = render_world(&World::new(), shot.clone()) else {
        headless::skip_or_require_gpu("the_same_scene_renders_byte_identically");
        return;
    };
    let b = render_world(&World::new(), shot).expect("adapter was available a moment ago");

    // Two bounds, not one. Together they are strictly stronger than the exact
    // comparison this used to make, because that one was not actually true.
    //
    // **Bound 1: no channel may differ by more than 1.** This is what catches
    // the #81 class of bug -- a reordered draw list moved whole silhouette
    // edges, 0.006% of pixels at max delta **63**. Anything that changes what is
    // drawn, or in what order, moves pixels by far more than one step.
    //
    // **Bound 2: at most a handful of pixels may differ at all.** Bound 1 alone
    // would let a systematic shift through -- a shading change that nudged every
    // pixel by one would pass it. A whole-image shift is ~100% of pixels;
    // 0.01% is ~92 of this frame's 921,600.
    let differing = differing_pixels(&a, &b);
    let fraction = differing.len() as f64 / (a.width as f64 * a.height as f64);
    let max_delta = differing.iter().map(|p| p.delta).max().unwrap_or(0);

    assert!(
        max_delta <= 1,
        "channel delta {max_delta} is too large to be rounding -- something \
         changed what is drawn or the order it is drawn in.\n{}",
        describe(&differing)
    );
    assert!(
        fraction < 0.0001,
        "{:.6}% of pixels differ ({} of {}) -- too many to be per-pixel \
         rounding, even at delta {max_delta}.\n{}",
        fraction * 100.0,
        differing.len(),
        a.width as usize * a.height as usize,
        describe(&differing)
    );
}

/// One pixel that came out differently between two renders of the same scene.
struct DifferingPixel {
    x: usize,
    y: usize,
    delta: u8,
    a: [u8; 3],
    b: [u8; 3],
}

fn differing_pixels(a: &Frame, b: &Frame) -> Vec<DifferingPixel> {
    let w = a.width as usize;
    // `as_chunks`, not `chunks_exact(4)`: clippy's
    // `chunks_exact_to_as_chunks` denies the latter for a constant size, and
    // `headless::compare` already reads its pixels this way.
    a.pixels
        .as_chunks::<4>()
        .0
        .iter()
        .zip(b.pixels.as_chunks::<4>().0)
        .enumerate()
        .filter(|(_, (pa, pb))| pa != pb)
        .map(|(i, (pa, pb))| DifferingPixel {
            x: i % w,
            y: i / w,
            delta: (0..4).map(|c| pa[c].abs_diff(pb[c])).max().unwrap_or(0),
            a: [pa[0], pa[1], pa[2]],
            b: [pb[0], pb[1], pb[2]],
        })
        .collect()
}

/// Say **where** the differing pixels are and whether they touch each other.
///
/// The old failure message gave a percentage and asserted a cause ("draw order
/// or meshing"), which for the signature actually seen was wrong. Location and
/// adjacency are what tell the two apart: rasterisation rounding is scattered
/// singletons, a reordered draw is a contiguous run along an edge.
fn describe(pixels: &[DifferingPixel]) -> String {
    use std::collections::HashSet;
    let set: HashSet<(usize, usize)> = pixels.iter().map(|p| (p.x, p.y)).collect();
    let adjacent = pixels
        .iter()
        .filter(|p| {
            [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)]
                .iter()
                .any(|(dx, dy)| {
                    set.contains(&(
                        (p.x as i64 + dx).max(0) as usize,
                        (p.y as i64 + dy).max(0) as usize,
                    ))
                })
        })
        .count();
    let mut out = format!(
        "  {} differing pixels, {adjacent} of them adjacent to another \
         (contiguous => draw order; scattered => rounding)\n",
        pixels.len()
    );
    for p in pixels.iter().take(8) {
        out += &format!(
            "  ({}, {}) delta {}  {:?} vs {:?}\n",
            p.x, p.y, p.delta, p.a, p.b
        );
    }
    out
}

#[test]
fn terrain_renders_as_expected() {
    assert_golden("terrain", &World::new(), Shot::default());
}

#[test]
fn distance_fog_fades_the_far_ring_into_the_sky() {
    // Package 2 (cross-session engine review): distance fog dissolves the
    // render-distance ring into the sky colour instead of a hard edge. A
    // first-person eye rather than the default orbit -- the orbit's fixed,
    // steep overhead angle (`Shot::camera`'s doc comment) foreshortens the
    // ring near the top of the frame; a ground-level view looking outward
    // toward the region's edge is what a player actually sees fog do.
    let world = World::new();
    let region_radius = 8; // 128 blocks -- small enough to mesh quickly,
                           // large enough that `Lighting::fog_range` puts
                           // the fade inside the meshed region rather than
                           // past its edge.
    let eye = glam::vec3(8.0, 40.0, 8.0);
    let target = glam::vec3(120.0, 15.0, 120.0);
    let (fog_start, fog_end) = Lighting::fog_range((region_radius * 16) as f32);
    let shot = Shot {
        width: 960,
        height: 540,
        region_radius,
        orbit_t: 0.0,
        camera: Some((eye.as_dvec3(), target - eye)),
        players: Vec::new(),
        highlighted_block: None,
        cracking: None,
        hotbar: None,
        panel: None,
        health: None,
        crosshair: false,
        tooltip: None,
        gauges: None,
        icons: Vec::new(),
        anchor: None,
        menu: None,
        lighting: Lighting {
            fog_start,
            fog_end,
            ..Lighting::default()
        },
    };
    assert_golden("fog_over_the_far_ring", &world, shot);
}

#[test]
fn a_cave_mouth_is_visible() {
    // Block 1.5 (#48)'s own bar: caves are a real, cross-chunk 3D noise
    // field carved into the terrain (§8.3), not just a promise buried in
    // `density`'s implementation -- this proves one is actually reachable
    // and visible from outside, at ground level.
    //
    // The default orbit camera can't show this: it always looks down at a
    // fixed shallow angle from above (`Shot::camera`'s doc comment), which
    // can show a cave's roof but never see *into* an opening at ground
    // level, at any scene scale. This shot uses an explicit low, grazing
    // camera instead.
    //
    // Seed 119 and this exact eye/target are the result of scanning a small
    // region around the world origin for a cave chamber that breaches the
    // surface close enough to frame directly (see the PR description for
    // how) -- not hand-placed, and not the only one; caves are common
    // enough that most seeds have several within a couple of hundred blocks
    // of any point.
    let world = World::with_seed(119);
    let eye = glam::vec3(-6.0, 27.0, -2.0);
    let target = glam::vec3(-15.0, 29.0, -2.0);
    let shot = Shot {
        width: 960,
        height: 540,
        region_radius: 3,
        orbit_t: 0.0,
        camera: Some((eye.as_dvec3(), target - eye)),
        players: Vec::new(),
        highlighted_block: None,
        cracking: None,
        hotbar: None,
        panel: None,
        health: None,
        crosshair: false,
        tooltip: None,
        gauges: None,
        icons: Vec::new(),
        anchor: None,
        menu: None,
        lighting: Lighting::default(),
    };
    assert_golden("cave_mouth", &world, shot);
}

#[test]
fn iron_ore_is_visible_in_a_cave_wall() {
    // Block 2.3b (#156)'s own bar: ore actually reaches the render path and is
    // distinguishable from the stone around it. `PHASE2_ARCHITECTURE.md` §6
    // makes ore a *material* substitution inside the density pass, which means
    // the only way to see it is where stone is already exposed -- so this frames
    // a vein in a cave wall from inside the chamber.
    //
    // Seed 119 is the same one `a_cave_mouth_is_visible` uses, and the eye/target
    // are not hand-placed: a scan over the region around the origin looked for
    // ore with a large air pocket around it (a chamber big enough to stand in)
    // and then for a viewpoint with clear line of sight to it. This vein is at
    // (-31, 17, 5) with 160 of the 343 cells in a 7³ box around it open.
    //
    // Note the ore is *below* the surface by construction (`max_y: 40`), which
    // is why none of the other goldens moved when ore was switched on -- they
    // all frame the surface, and ore never surfaces.
    let world = World::with_seed(119);
    let eye = glam::vec3(-25.5, 17.5, 5.5);
    let target = glam::vec3(-31.0, 17.5, 5.5);
    let shot = Shot {
        width: 960,
        height: 540,
        region_radius: 3,
        orbit_t: 0.0,
        camera: Some((eye.as_dvec3(), target - eye)),
        players: Vec::new(),
        highlighted_block: None,
        cracking: None,
        hotbar: None,
        panel: None,
        health: None,
        crosshair: false,
        tooltip: None,
        gauges: None,
        icons: Vec::new(),
        anchor: None,
        menu: None,
        lighting: Lighting::default(),
    };
    // Audit #5: the vein is a small patch of a 960x540 frame -- well inside
    // the whole-image golden's 0.5% tolerance, so a defect that made ore
    // generate as plain stone (`cubara_world::worldgen::ore_or` always
    // returning `stone`) would pass `assert_golden` alone. Measured: 26 of
    // the ~18,450 pixels in this rect differ from the committed
    // (Windows/Vulkan-blessed) golden at `TOLERANCE` on an NVIDIA/Vulkan
    // machine, 26-28 on the real macOS and Windows CI runners (checked via
    // `gh run view --log` on a recent `main` run) -- ordinary cross-backend
    // terrain/mip noise, the same kind `MAX_DIFFERING`'s table documents --
    // while disabling ore substitution entirely changes 2,543. 60 keeps a
    // >2x margin over every measured noise figure and a >40x margin under
    // the defect.
    if let Some(frame) = render_world(&world, shot.clone()) {
        assert_region_golden("iron_ore", &frame, (325, 225, 205, 90), 60, "the ore vein");
    }
    assert_golden("iron_ore", &world, shot);
}

#[test]
fn no_crack_at_a_real_lod_boundary() {
    // Issue #108's own bar: a real level-0/level-1 node boundary, in the
    // live streamed scene (`render_world`/`mesh_region`'s ring-schedule
    // streaming, not a synthetic fixture), with no crack of background
    // showing through -- proof the skirts added to `cubara_voxel`'s greedy
    // mesher actually reach the render path, not just the unit tests in
    // isolation.
    //
    // `DEFAULT_RING_SCHEDULE`'s first ring ends at chunk-radius 8 from the
    // region's center, so `region_radius: 12` (giving `schedule_for_radius`
    // `[(0, 8), (1, 12)]`) puts a real level-0-to-level-1 seam inside the
    // framed region -- the default auto-framed orbit camera (same one
    // `terrain_renders_as_expected` uses) frames the whole region from
    // above, which is the honest test: a player flying/walking over this
    // area at a normal viewing angle, not a worst-case grazing shot chosen
    // to manufacture a crack. (An earlier, near-ground grazing version of
    // this shot showed the skirts themselves as a visible fringe along
    // every nearby chunk edge, not just the one LOD seam -- correct
    // per-node behaviour, since a node can't tell a same-level neighbour
    // from a different-level one, but a confusing shot to assert "no
    // crack" against. This angle avoids that.)
    let world = World::new();
    let shot = Shot {
        region_radius: 12,
        ..Shot::default()
    };
    // Audit #5: a seam is "a few hundred pixels" (the audit's own words) of a
    // 1280x720 frame, a small fraction of the whole-image golden's 0.5%
    // tolerance -- and the ring boundary this test frames runs most of the
    // way around the screen, so no single small rect covers "the" crack the
    // way it does for a UI element. This one rect was found by disabling
    // `push_skirt` (`crates/voxel/src/voxel.rs`) outright and diffing the
    // result against the committed golden: the regression shows up in dozens
    // of scattered patches around the ring, and this is one of them (43
    // pixels with skirts off, 0 with them on, on this machine). It proves a
    // *global* skirt regression (the actual failure mode #4/#5 describe,
    // and the one `push_skirt`'s own unit tests in `voxel.rs` do not cover
    // end-to-end) without having to track every point on the ring; it would
    // not by itself catch a one-off bug in a single unrelated node's skirt.
    // Checked via `gh run view --log` on a recent `main` run: the real macOS
    // and Windows CI runners differ from the golden by 0.0030%/0.0015% of
    // the *whole* frame here (28/14 pixels out of 921,600), where this
    // machine matches exactly -- scattered over a ring that spans most of
    // the image, so only a small fraction would be expected to land inside
    // this one 75x35 patch. 15 keeps a wide margin over that estimate while
    // staying well below the 43-pixel defect signal.
    if let Some(frame) = render_world(&world, shot.clone()) {
        assert_region_golden(
            "lod_boundary",
            &frame,
            (515, 585, 75, 35),
            15,
            "a real LOD seam on the ring boundary",
        );
    }
    assert_golden("lod_boundary", &world, shot);
}

/// Mesh exactly `schedule`'s nodes around the origin (never level 0, which
/// callers assert for themselves) with trees, and again with every node's
/// `trees` cleared -- same terrain, same auto-framed camera either way, so
/// the only thing that can differ between the two renders is whether R2's
/// instances were drawn. Shared by [`trees_continue_past_the_first_ring`]
/// and [`level_three_trees_draw_as_boxes`], which differ only in which
/// level they isolate.
fn render_with_and_without_trees(schedule: &[(u32, i32)]) -> Option<(Frame, Frame)> {
    let world = World::new();
    let registry = real_registry();
    let layers = TextureLayers::from_registry(&registry);
    let layer_of = |name: &str| layers.layer_of(name);
    let blocks = real_blocks();

    let center = ChunkCoord::new(0, 0, 0);
    let nodes = desired_nodes(center, 0..=0, schedule);
    assert!(
        nodes.iter().all(|n| n.level >= 1),
        "a level-0 node would carry real voxel trees of its own, which this \
         test must rule out to isolate R2's instances"
    );

    let with_trees = mesh_nodes(&world, &registry, &layer_of, nodes.clone(), blocks);
    assert!(
        with_trees.iter().any(|b| !b.trees.is_empty()),
        "no coarse node around the default-seed origin grows a tree at all \
         -- this scene proves nothing until it does"
    );
    let mut without_trees = mesh_nodes(&world, &registry, &layer_of, nodes, blocks);
    for b in &mut without_trees {
        b.trees.clear();
    }

    let shot = Shot {
        width: 960,
        height: 540,
        ..Shot::default()
    };
    let render = |built: Vec<cubara_world::mesh::BuiltNode>| {
        let _gpu = GPU_LIFECYCLE.lock().unwrap_or_else(|e| e.into_inner());
        headless::render(built.into_iter().filter_map(to_meshed_node), shot.clone())
    };
    let (with_frame, without_frame) = (render(with_trees), render(without_trees));
    Some((with_frame?, without_frame?))
}

/// T7 (`docs/PROPOSAL_LOD_TRANSITIONS.md` §6, issue #332 R2):
/// "trees_continue_past_the_first_ring" -- leaf-coloured pixels show up past
/// the first ring now, where before this landed nothing did.
///
/// Meshed is *only* `desired_nodes`' levels 1-2 selection (160-512 m) --
/// never level 0 -- so nothing in the geometry this renders can be a real
/// voxel tree (those only ever grow at level 0,
/// `WorldGen::generate`'s own doc comment). Whatever the camera sees
/// standing above the coarse ground can only be R2's instances (the real
/// oak model at these levels, decision 3).
///
/// Rendered twice from the *identical* mesh and camera
/// ([`render_with_and_without_trees`]), with every node's `trees` cleared
/// the second time. `headless::compare` (the goldens' own pixel diff)
/// between the two is the assertion: if the scene doesn't visibly change,
/// nothing drew. That is also this test's own proof that it can fail
/// (CLAUDE.md "a test that has never been seen to fail is not yet
/// evidence") -- `./scripts/check-tests-can-fail.sh`-style, stated
/// directly: strip the renderer's own tree draw (not just this test's
/// copy) and this goes red, recorded in the PR's defect-injection table.
#[test]
fn trees_continue_past_the_first_ring() {
    // Levels 1 and 2 only: a schedule whose first entry is level 1, never
    // level 0, so `desired_nodes` cannot select one.
    let schedule = [(1u32, 18i32), (2, 32)];
    let Some((with_frame, without_frame)) = render_with_and_without_trees(&schedule) else {
        headless::skip_or_require_gpu("trees_continue_past_the_first_ring");
        return;
    };
    assert_eq!(
        (with_frame.width, with_frame.height),
        (without_frame.width, without_frame.height)
    );
    let diff = headless::compare(&with_frame.pixels, &without_frame.pixels, TOLERANCE);
    eprintln!(
        "trees_continue_past_the_first_ring: {:.4}% of pixels differ with trees on vs off \
         (max channel delta {})",
        diff.differing_fraction * 100.0,
        diff.max_channel_delta
    );
    assert!(
        diff.differing_fraction > MAX_DIFFERING,
        "turning trees on changed only {:.4}% of pixels (same threshold the \
         goldens use for cross-backend noise) -- nothing R2 added is \
         actually visible past the first ring",
        diff.differing_fraction * 100.0
    );
}

/// Decision 3's other half: level 3 draws two boxes, not the real oak --
/// its own instanced draw, so "the real oak model landed but the boxes
/// didn't" is a distinct failure T7 alone cannot see (T7 never reaches
/// level 3). Same method as T7 (`render_with_and_without_trees`), isolated
/// to a level-3-only schedule instead.
#[test]
fn level_three_trees_draw_as_boxes() {
    let schedule = [(3u32, 16i32)];
    let Some((with_frame, without_frame)) = render_with_and_without_trees(&schedule) else {
        headless::skip_or_require_gpu("level_three_trees_draw_as_boxes");
        return;
    };
    assert_eq!(
        (with_frame.width, with_frame.height),
        (without_frame.width, without_frame.height)
    );
    let diff = headless::compare(&with_frame.pixels, &without_frame.pixels, TOLERANCE);
    eprintln!(
        "level_three_trees_draw_as_boxes: {:.4}% of pixels differ with trees on vs off \
         (max channel delta {})",
        diff.differing_fraction * 100.0,
        diff.max_channel_delta
    );
    assert!(
        diff.differing_fraction > MAX_DIFFERING,
        "turning trees on at level 3 changed only {:.4}% of pixels -- the \
         box model is not actually visible",
        diff.differing_fraction * 100.0
    );
}

#[test]
fn the_selected_block_shows_an_outline() {
    // Issue #52's own bar: the highlight must align to the targeted block
    // with no z-fighting against its own (flush) top face -- the coplanar
    // case the depth bias exists for. A close, angled camera on the
    // default-seed world, aimed at a surface block with solid neighbours on
    // every other side: the box's other 11 edges are correctly hidden
    // behind that solid terrain (depth-tested like any other geometry),
    // which is why only the top face's outline shows here.
    let world = World::new();
    let ground = world
        .raycast([5.5, 200.0, 5.5], [0.0, -1.0, 0.0], 400.0, real_blocks())
        .expect("ground below");
    let eye = glam::vec3(2.5, ground.block[1] as f32 + 3.0, 2.5);
    let target = glam::vec3(5.5, ground.block[1] as f32 + 0.5, 5.5);
    let hit = world
        .raycast(
            eye.to_array(),
            (target - eye).to_array(),
            30.0,
            real_blocks(),
        )
        .expect("the block this shot is framed around");

    let shot = Shot {
        width: 960,
        height: 540,
        region_radius: 3,
        orbit_t: 0.0,
        camera: Some((eye.as_dvec3(), target - eye)),
        players: Vec::new(),
        highlighted_block: Some(hit.block),
        cracking: None,
        hotbar: None,
        panel: None,
        health: None,
        crosshair: false,
        tooltip: None,
        gauges: None,
        icons: Vec::new(),
        anchor: None,
        menu: None,
        lighting: Lighting::default(),
    };
    // Audit #5: the outline is a thin wireframe box, a few hundred pixels at
    // most -- well inside the whole-image golden's 0.5% tolerance, so a
    // defect that skipped the outline draw call entirely
    // (`SceneRenderer::encode_scene`'s `if selected_block.is_some()`) would
    // pass `assert_golden` alone. `(383, 260, 194, 117)` is the screen-space
    // bounding box of the targeted block's top face, computed from `eye`,
    // `target` and `hit.block` via `CameraUniform::look_view_proj` and
    // confirmed against the committed golden (the outline's own dark pixels
    // fall at x 384..575, y 261..375 in it). This rect is mostly the
    // block's top face and its surroundings, not just the outline, so it
    // carries ordinary cross-backend terrain noise on top of the outline
    // itself (unlike the UI rects elsewhere in this file, which sit over
    // flat, opaque HUD quads).
    //
    // Measured, pixels differing from the golden at `TOLERANCE`: 13 of the
    // ~22,700 in this rect on an NVIDIA/Vulkan machine (local), 14 on
    // lavapipe (local and ubuntu CI), **41 on Windows CI** (the original
    // budget of 30 was set from the NVIDIA/lavapipe numbers alone, before
    // the Windows number was checked against this specific rect rather than
    // the shot's whole-frame percentage -- it under-shot). Disabling the
    // outline draw changes 235, a 5.7x margin over the worst measured
    // noise. 60 keeps a healthy margin over 41 (macOS CI not yet measured
    // under this rect; re-check once it runs) while staying a comfortable
    // 3.9x under the defect.
    if let Some(frame) = render_world(&world, shot.clone()) {
        assert_region_golden(
            "outline",
            &frame,
            (383, 260, 194, 117),
            60,
            "the outline around the targeted block's top face",
        );
    }
    assert_golden("outline", &world, shot);
}

#[test]
fn a_block_being_dug_shows_cracks() {
    // The owner asked to see a block take damage before it goes. The same
    // close camera as the outline shot, three quarters of the way through a
    // dig, so the crack is well grown and the face it sits on is in view.
    let world = World::new();
    let ground = world
        .raycast([5.5, 200.0, 5.5], [0.0, -1.0, 0.0], 400.0, real_blocks())
        .expect("ground below");
    let eye = glam::vec3(4.2, ground.block[1] as f32 + 2.0, 4.2);
    let target = glam::vec3(5.5, ground.block[1] as f32 + 0.5, 5.5);
    let hit = world
        .raycast(
            eye.to_array(),
            (target - eye).to_array(),
            30.0,
            real_blocks(),
        )
        .expect("the block this shot is framed around");

    let shot = Shot {
        width: 960,
        height: 540,
        region_radius: 3,
        orbit_t: 0.0,
        camera: Some((eye.as_dvec3(), target - eye)),
        players: Vec::new(),
        highlighted_block: Some(hit.block),
        cracking: Some((hit.block, 0.75)),
        hotbar: None,
        panel: None,
        health: None,
        crosshair: false,
        tooltip: None,
        gauges: None,
        icons: Vec::new(),
        anchor: None,
        menu: None,
        lighting: Lighting::default(),
    };
    // Measured: the image comparison alone caught missing cracks at 0.53%
    // against a 0.5% threshold -- one driver's rounding from passing. So also
    // render the same shot without cracks, on the same adapter, and require
    // the two to differ by a clear margin: a backend difference cannot make
    // that pass or fail, only the cracks can.
    let with = render_world(&world, shot.clone());
    let without = render_world(
        &world,
        Shot {
            cracking: None,
            ..shot.clone()
        },
    );
    if let (Some(a), Some(b)) = (with, without) {
        let changed = a
            .pixels
            .chunks(4)
            .zip(b.pixels.chunks(4))
            .filter(|(p, q)| (0..3).any(|c| p[c].abs_diff(q[c]) > 60))
            .count();
        eprintln!("pixels the cracks changed: {changed}");
        assert!(changed > 1500, "cracks changed only {changed} pixels");
    }
    assert_golden("cracks", &world, shot);
}

#[test]
fn items_show_their_icons() {
    // The owner asked for items to have pictures. Nine real icons from
    // `assets/textures` -- block items by their side, and the drawn items --
    // with counts on top, so the golden shows both are legible together.
    let names = [
        "grass_side",
        "cobble",
        "plank",
        "oak_log_side",
        "item_stick",
        "item_wooden_pick",
        "item_stone_pick",
        "item_iron_ingot",
        "item_raw_iron",
    ];
    let icons: Vec<Option<Vec<u8>>> = names
        .iter()
        .map(|n| Some(cubara_render::load_icon(n).unwrap_or_else(|| panic!("no art {n}"))))
        .collect();
    let mut hotbar = [None; 9];
    for (i, slot) in hotbar.iter_mut().enumerate() {
        *slot = Some(HotbarSlot {
            // A swatch colour nothing in the art uses, so falling back to it
            // would be obvious in the image.
            color: [1.0, 0.0, 1.0],
            count: [12, 64, 5, 1, 32, 1, 1, 3, 7][i],
            icon: Some(i as u32),
        });
    }

    let world = World::new();
    let shot = Shot {
        width: 960,
        height: 540,
        region_radius: 2,
        orbit_t: 0.0,
        camera: None,
        anchor: None,
        players: Vec::new(),
        highlighted_block: None,
        cracking: None,
        hotbar: Some(hotbar),
        panel: None,
        health: None,
        crosshair: false,
        tooltip: None,
        gauges: None,
        icons,
        menu: None,
        lighting: Lighting::default(),
    };
    // Nine icons against nine magenta swatches differ in well under the
    // golden's 0.5% of a 960x540 frame, so compare against the swatches on
    // the same adapter as well: only the icons can make that pass or fail.
    let with = render_world(&world, shot.clone());
    let without = render_world(
        &world,
        Shot {
            icons: Vec::new(),
            menu: None,
            lighting: Lighting::default(),
            ..shot.clone()
        },
    );
    if let (Some(a), Some(b)) = (with, without) {
        let magenta = |p: &[u8]| p[0] > 200 && p[1] < 60 && p[2] > 200;
        let left = a.pixels.chunks(4).filter(|p| magenta(p)).count();
        let before = b.pixels.chunks(4).filter(|p| magenta(p)).count();
        assert!(
            before > 5000,
            "the swatch fallback drew no magenta ({before})"
        );
        assert_eq!(
            left, 0,
            "{left} magenta pixels: an icon fell back to its swatch"
        );
    }
    assert_golden("item_icons", &world, shot);
}

#[test]
fn the_hotbar_shows_slots_counts_and_the_held_one() {
    // Block 2.1e's own bar. Three things have to be visible, and a golden is
    // the only automated way to say so: nine slots along the bottom, the held
    // one distinguishable from the rest, and a count on a stack of more than
    // one.
    //
    // Built from plain colours and counts rather than from an inventory --
    // that is the point of `HotbarView` (Rule 3: this crate does not know what
    // an item is), and it is what lets this run with no sim at all.
    let slot = |r: f32, g: f32, b: f32, count: u8| {
        Some(HotbarSlot {
            color: [r, g, b],
            count,
            icon: None,
        })
    };
    let hotbar = [
        slot(0.45, 0.62, 0.30, 12), // a two-digit count
        slot(0.55, 0.40, 0.26, 1),  // a count of 1 draws no number -- deliberate
        None,
        slot(0.52, 0.52, 0.55, 64), // a full stack
        None,
        None,
        slot(0.70, 0.35, 0.25, 3),
        None,
        None,
    ];

    let world = World::new();
    let shot = Shot {
        width: 960,
        height: 540,
        region_radius: 2,
        orbit_t: 0.0,
        camera: None,
        players: Vec::new(),
        highlighted_block: None,
        cracking: None,
        hotbar: Some(hotbar),
        panel: None,
        health: None,
        crosshair: false,
        tooltip: None,
        gauges: None,
        icons: Vec::new(),
        anchor: None,
        menu: None,
        lighting: Lighting::default(),
    };
    // Audit #5: the count glyph for slot 0 ("12") is a couple of dozen
    // pixels of a 960x540 frame -- far inside the whole-image golden's 0.5%
    // tolerance, so a defect that stopped drawing counts entirely
    // (`SceneRenderer::queue_item`'s `if item.count > 1`) would pass
    // `assert_golden` alone. `(269, 509, 27, 15)` is the glyph's exact box,
    // computed from `queue_item`'s own layout (`SLOT`=48, `GAP`=4,
    // `MARGIN`=16 place slot 0 at x=248,y=476; `SCALE`=1.5, `font::GLYPH`=8
    // place "12" at x=270,y=510, 24x12, plus its 1px drop shadow).
    //
    // Unlike the hotbar's *other* UI rects (opaque, pixel-aligned
    // solid-colour quads -- see `inventory_screen`/`furnace_screen` below,
    // where "no mips, `Nearest` sampler" really does mean zero cross-backend
    // noise), a plain per-pixel comparison does not work for this one.
    //
    // Measured cross-backend noise, plain per-pixel count, this region: 0
    // (NVIDIA/Vulkan, local), 40 (lavapipe, local and ubuntu CI), 73
    // (Windows CI). Disabling the count draw entirely
    // (`SceneRenderer::queue_item`'s `if item.count > 1`) changes only 153 --
    // just 2.1x the measured Windows noise, short of the 3x margin this file
    // aims for (test-quality audit #5). The reason is `SCALE`=1.5 upscaling
    // an 8x8 bitmap glyph through a `Nearest` sampler: at a non-integer
    // scale, which source texel a destination row/column rounds to is a
    // rasterisation tie, and ties land differently across backends *through
    // the glyph's interior*, not just at its silhouette -- unlike the
    // opaque, pixel-aligned solid-colour quads the comment above actually
    // describes.
    //
    // So this region uses [`assert_region_coverage_golden`] instead: coverage
    // (how many pixels are lit) rather than position (which ones). Getting
    // the threshold right took a second measurement -- the first (160) was
    // wrong for a reason worth recording.
    //
    // Pixel values here are **not** the naive `color * 255`: the colour
    // target is `Rgba8UnormSrgb` (`headless::COLOR_FORMAT`), so every value
    // read back is sRGB-encoded. Slot 0's own green item swatch
    // (`[0.45, 0.62, 0.30]`) reads back at luminance 191 -- *above* a 160
    // threshold -- while the slot's outer well colour reads at 134. Dumping
    // the region as a pixel grid (temporarily, not committed) showed why a
    // 160 threshold measured only a 61-pixel delta when the count draw was
    // disabled: that threshold counts the swatch's own 191 as "lit", so it
    // is really measuring a mix of the glyph's white ink turning on *and*
    // its black drop-shadow turning the swatch's already-lit background
    // off, two effects partly cancelling. 220 sits above both backgrounds
    // (191 and 134) and below white ink (255), so it counts *only* true
    // white glyph pixels -- disabling the draw takes the golden's 105 lit
    // pixels to a clean 0, not 194 down to 133.
    //
    // That is also a bigger, cleaner defect number against the same
    // underlying noise: the delta is driven by `SCALE`=1.5 upscaling an 8x8
    // glyph through a `Nearest` sampler, where ties over *which* pixels are
    // white land differently across backends (Metal/DX12 vs this file's
    // Vulkan-blessed reference) -- unaffected by where the threshold sits,
    // since it is a white/not-white question either way. Measured coverage
    // delta: 0 (NVIDIA/Vulkan, local), 0 (lavapipe, local and ubuntu CI,
    // at both 160 and 220 -- lavapipe agrees with the Vulkan-blessed
    // reference almost exactly), **20 on both macOS and Windows CI**
    // (measured at threshold 160 before this was re-derived; unaffected by
    // the threshold per the reasoning above, so still the right number to
    // budget against). Disabling the draw changes 105 -- a 5.25x margin
    // over the worst measured noise, comfortably over the 3x this file aims
    // for, where the old 160-threshold number (61) gave only 2.1-3.05x.
    if let Some(frame) = render_world(&world, shot.clone()) {
        assert_region_coverage_golden(
            "hotbar",
            &frame,
            (269, 509, 27, 15),
            220,
            25,
            "slot 0's \"12\" count",
        );
    }
    assert_golden("hotbar", &world, shot);
}

#[test]
fn the_inventory_screen_shows_slots_a_recipe_and_the_cursor() {
    // Block 2.2c's own bar. Four things have to be legible at once, and a
    // golden is the only automated way to say so: the dimmed world behind, the
    // slot grid, a partial recipe with its result showing, and the cursor's
    // stack drawn over everything.
    //
    // Built from the layout directly -- no sim, no inventory -- which is the
    // point of `PanelView` carrying colours and counts (Rule 3).
    let panel = InventoryPanel::layout(960, 540, 2);
    let swatch = |r: f32, g: f32, b: f32, count: u8| {
        Some(HotbarSlot {
            color: [r, g, b],
            count,
            icon: None,
        })
    };

    let contents: Vec<Option<HotbarSlot>> = panel
        .slots()
        .iter()
        .map(|s| match (s.kind, s.index) {
            // A 2x2 of planks in the grid: cells 0, 1, 3, 4 of the sim's fixed
            // 3x3 storage.
            (PanelSlotKind::Grid, 0 | 1 | 3 | 4) => swatch(0.66, 0.50, 0.31, 4),
            // ... so the result slot shows what it makes.
            (PanelSlotKind::Result, _) => swatch(0.55, 0.40, 0.26, 1),
            (PanelSlotKind::Inventory, 0) => swatch(0.45, 0.62, 0.30, 12),
            (PanelSlotKind::Inventory, 2) => swatch(0.52, 0.52, 0.55, 64),
            (PanelSlotKind::Inventory, 11) => swatch(0.70, 0.35, 0.25, 3),
            _ => None,
        })
        .collect();

    let world = World::new();
    let shot = Shot {
        width: 960,
        height: 540,
        region_radius: 2,
        orbit_t: 0.0,
        camera: None,
        players: Vec::new(),
        highlighted_block: None,
        cracking: None,
        hotbar: None,
        health: None,
        crosshair: false,
        tooltip: None,
        gauges: None,
        icons: Vec::new(),
        anchor: None,
        menu: None,
        lighting: Lighting::default(),
        panel: Some((
            PanelLayout::Grid(2),
            contents,
            swatch(0.66, 0.50, 0.31, 7),
            // Over a slot rather than in open space, so the golden shows the
            // cursor's stack drawn *on top of* a slot's contents.
            (panel.slots()[0].x + 30.0, panel.slots()[0].y + 30.0),
        )),
    };
    // Audit #5: the cursor's stack and the recipe result are each one 40x40
    // slot out of a 960x540 frame -- far inside the whole-image golden's
    // 0.5% tolerance, so a defect that stopped drawing either (the cursor's
    // `if let Some(held) = view.held`, or the result slot's item draw in
    // `queue_panel`'s slot loop) would pass `assert_golden` alone. Both
    // rects are exact slot rectangles from `InventoryPanel::layout`'s own
    // geometry (`SLOT`=40, `GAP`=4, `PAD`=16): the cursor box is centred on
    // `(panel.slots()[0].x + 30, .y + 30)` above; the result slot is
    // `panel.slots()`'s `Result`-kind entry.
    //
    // The two behave differently cross-backend, so they are budgeted
    // differently. The result slot sits on the grid's own integer
    // boundaries, like `furnace_screen`'s three slots below -- opaque
    // `queue_rect`/`queue_item` quads through `TextRenderer`'s `Nearest`,
    // no-mip sampler with no sub-pixel edge, measured at 0 pixels differing
    // from the golden on every backend checked. Kept at 0.
    //
    // The cursor box is not grid-aligned -- it is centred 30px into slot 0
    // from a corner, i.e. offset from the grid by a non-multiple of its own
    // half-size, which puts its edges at a sub-pixel position relative to
    // whatever is drawn under it. Measured noise there: 0 (NVIDIA/Vulkan,
    // local), 12 (lavapipe, local and ubuntu CI), 18 (Windows CI).
    // Disabling the cursor draw changes 533 in its box -- a 30x margin over
    // the worst measured noise, so a plain per-pixel budget with headroom
    // (30, below) keeps the 3x+ separation this file aims for. Hiding the
    // result slot's item changes 784 in its box.
    if let Some(frame) = render_world(&world, shot.clone()) {
        assert_region_golden(
            "inventory_screen",
            &frame,
            (294, 138, 40, 40),
            30,
            "the cursor's held stack",
        );
        assert_region_golden(
            "inventory_screen",
            &frame,
            (394, 150, 40, 40),
            0,
            "the recipe's result slot",
        );
    }
    assert_golden("inventory_screen", &world, shot);
}

#[test]
fn an_item_name_shows_beside_the_cursor() {
    // The owner asked for it: hovering an item says what it is. One filled
    // slot with the cursor over it and a label, which has to sit beside the
    // cursor and above the screen rather than under it.
    let panel = InventoryPanel::layout(960, 540, 2);
    let contents: Vec<Option<HotbarSlot>> = panel
        .slots()
        .iter()
        .map(|s| match (s.kind, s.index) {
            (PanelSlotKind::Inventory, 4) => Some(HotbarSlot {
                color: [0.66, 0.50, 0.31],
                count: 1,
                icon: None,
            }),
            _ => None,
        })
        .collect();
    let hovered = panel
        .slots()
        .iter()
        .find(|s| s.kind == PanelSlotKind::Inventory && s.index == 4)
        .expect("slot 4 exists");

    let world = World::new();
    let shot = Shot {
        width: 960,
        height: 540,
        region_radius: 2,
        orbit_t: 0.0,
        camera: None,
        players: Vec::new(),
        highlighted_block: None,
        cracking: None,
        hotbar: None,
        health: None,
        crosshair: false,
        tooltip: Some("Wooden Pick".to_string()),
        gauges: None,
        icons: Vec::new(),
        anchor: None,
        menu: None,
        lighting: Lighting::default(),
        panel: Some((
            PanelLayout::Grid(2),
            contents,
            None,
            (hovered.x + 20.0, hovered.y + 20.0),
        )),
    };
    // Audit #5: the tooltip box is a few thousand pixels of a 960x540 frame
    // -- inside the whole-image golden's 0.5% tolerance, so a defect that
    // stopped drawing it (`queue_panel`'s `if let Some(label) = view.tooltip`)
    // would pass `assert_golden` alone. `(493, 359, 188, 28)` is the box's
    // exact rectangle from `queue_tooltip`'s own layout (`SCALE`=2,
    // `font::GLYPH`=8, `PAD`=5, cursor at `(hovered.x + 20, hovered.y + 20)`
    // = (480, 392) above). Measured on this machine: 0 pixels differ from the
    // golden here; disabling the tooltip draw changes 5,264. Not kept at 0
    // like the other UI rects, unlike which this one is not fully opaque --
    // `queue_tooltip`'s own box is drawn at 94% alpha
    // (`queue_rect_alpha(.., BACK, 0.94)`), so 6% of whatever cross-backend
    // terrain noise sits behind it (this shot's world background) can bleed
    // through. 3 budgets for that without weakening what the rect actually
    // checks -- a missing or wrong label still changes thousands of pixels.
    if let Some(frame) = render_world(&world, shot.clone()) {
        assert_region_golden(
            "item_tooltip",
            &frame,
            (493, 359, 188, 28),
            3,
            "the \"Wooden Pick\" label",
        );
    }
    assert_golden("item_tooltip", &world, shot);
}

#[test]
fn the_crosshair_marks_the_centre_of_the_screen() {
    // The owner asked for a mark in the middle: where a click lands. Over
    // terrain rather than sky, so the golden shows it reads against both the
    // rim and the ground behind it.
    let world = World::new();
    let shot = Shot {
        width: 960,
        height: 540,
        region_radius: 2,
        orbit_t: 0.0,
        camera: None,
        players: Vec::new(),
        highlighted_block: None,
        cracking: None,
        hotbar: None,
        panel: None,
        health: None,
        crosshair: true,
        tooltip: None,
        gauges: None,
        icons: Vec::new(),
        anchor: None,
        menu: None,
        lighting: Lighting::default(),
    };
    // A crosshair is about 0.1% of the frame -- under the golden's 0.5%
    // tolerance, so the image comparison alone passes with it missing (found
    // by deleting the draw call). Look at the middle directly as well.
    if let Some(frame) = render_world(&world, shot.clone()) {
        let at = |x: u32, y: u32| {
            let i = ((y * frame.width + x) * 4) as usize;
            [frame.pixels[i], frame.pixels[i + 1], frame.pixels[i + 2]]
        };
        let (cx, cy) = (frame.width / 2, frame.height / 2);
        for (x, y) in [(cx, cy), (cx + 6, cy), (cx, cy + 6)] {
            let p = at(x, y);
            assert!(
                p.iter().all(|&c| c > 220),
                "no crosshair at ({x}, {y}): {p:?}"
            );
        }
    }
    assert_golden("crosshair", &world, shot);
}

#[test]
fn hearts_show_health_including_a_half() {
    // Block 2.9a's own bar. Twenty points drawn as ten hearts, and the reason
    // it is twenty rather than ten is that a **half** heart has to be
    // representable -- so the shot picks an odd number to prove it.
    //
    // 13 of 20: six full hearts, one half, three empty.
    let world = World::new();
    let shot = Shot {
        width: 960,
        height: 540,
        region_radius: 2,
        orbit_t: 0.0,
        camera: None,
        players: Vec::new(),
        highlighted_block: None,
        cracking: None,
        hotbar: Some([None; 9]),
        panel: None,
        health: Some((13, 20)),
        crosshair: false,
        tooltip: None,
        gauges: None,
        icons: Vec::new(),
        anchor: None,
        menu: None,
        lighting: Lighting::default(),
    };
    // Audit #5: the half heart (the 7th heart, index 6: `points.saturating_sub
    // (12).min(2) == 1`) is one 18x18 cell of a 960x540 frame -- far inside
    // the whole-image golden's 0.5% tolerance, so a defect that drew it full
    // or empty instead of half (`queue_hearts`'s `let w = HEART * (filled /
    // PER_HEART)`) would pass `assert_golden` alone, exactly as the audit's
    // own wording predicts ("a half heart drawn full or empty changes far
    // fewer pixels"). `(373, 447, 20, 20)` is that heart's cell, computed
    // from `queue_hearts`'s own layout (`HEART`=18, `GAP`=3, `MARGIN`=16,
    // `HOTBAR`=48, `LIFT`=10; x0=248, y=448; heart 6 at x=374). Measured: 0
    // pixels differ from the golden here. Kept at 0, not just a
    // near-matching guess: flat `queue_rect` quads through `TextRenderer`'s
    // `Nearest`, no-mip sampler (see `hotbar`'s region above), so none of
    // the mip-selection or texture-filtering noise this file documents
    // elsewhere applies. Forcing every heart to draw full changes 162.
    if let Some(frame) = render_world(&world, shot.clone()) {
        assert_region_golden("hearts", &frame, (373, 447, 20, 20), 0, "the half heart");
    }
    assert_golden("hearts", &world, shot);
}

#[test]
fn the_furnace_screen_shows_input_fuel_and_output() {
    // Block 2.4c's own bar: the furnace screen is a *different* screen from the
    // inventory, and the only automated way to say the three slots landed where
    // they should -- input above fuel on the left, output to the right -- is to
    // look at it.
    //
    // Built from the layout directly, no sim and no world (Rule 3), same as
    // `the_inventory_screen_shows_slots_a_recipe_and_the_cursor`.
    let panel = InventoryPanel::layout_furnace(960, 540);
    let swatch = |r: f32, g: f32, b: f32, count: u8| {
        Some(HotbarSlot {
            color: [r, g, b],
            count,
            icon: None,
        })
    };

    let contents: Vec<Option<HotbarSlot>> = panel
        .slots()
        .iter()
        .map(|s| match s.kind {
            // Raw iron waiting to be smelted, logs burning, ingots produced.
            PanelSlotKind::Grid => swatch(0.70, 0.35, 0.25, 8),
            PanelSlotKind::Fuel => swatch(0.45, 0.31, 0.18, 12),
            PanelSlotKind::Result => swatch(0.85, 0.85, 0.88, 3),
            PanelSlotKind::Inventory if s.index == 0 => swatch(0.52, 0.52, 0.55, 64),
            PanelSlotKind::Inventory => None,
        })
        .collect();

    let world = World::new();
    let shot = Shot {
        width: 960,
        height: 540,
        region_radius: 2,
        orbit_t: 0.0,
        camera: None,
        players: Vec::new(),
        highlighted_block: None,
        cracking: None,
        hotbar: None,
        health: None,
        crosshair: false,
        tooltip: None,
        gauges: None,
        icons: Vec::new(),
        anchor: None,
        menu: None,
        lighting: Lighting::default(),
        panel: Some((
            PanelLayout::Furnace,
            contents,
            swatch(0.70, 0.35, 0.25, 5),
            (panel.slots()[0].x + 30.0, panel.slots()[0].y + 30.0),
        )),
    };
    // Audit #5: each of the three slots is one 40x40 cell of a 960x540
    // frame -- far inside the whole-image golden's 0.5% tolerance, so a
    // defect that hid any single slot's item (`queue_panel`'s slot loop)
    // would pass `assert_golden` alone, and nothing else would say *which*
    // slot broke. All three rects are exact slot rectangles from
    // `InventoryPanel::layout_furnace`'s own geometry (input/`Grid` reuses
    // the 2-wide base's cell 0; `Fuel` sits one `step` below it; `Result`
    // sits two `step`s right of it, vertically centred). Each is an opaque
    // `queue_rect`/`queue_item` quad through `TextRenderer`'s `Nearest`,
    // no-mip sampler (see `hotbar`'s region above) -- the same case
    // `inventory_screen`'s grid-aligned result slot measures at a clean 0.
    //
    // Measured cross-backend noise, plain per-pixel count -- 0 on
    // NVIDIA/Vulkan (local) for all three. Not 0 everywhere, though, and not
    // evenly: input 1 (lavapipe, local and ubuntu CI; also 1 on macOS and
    // Windows CI). Fuel and output read 0 on ubuntu CI, but real hardware
    // disagrees with the Vulkan-blessed reference far more than lavapipe's
    // software rasteriser does -- fuel measured 18 on local lavapipe but
    // **72 on macOS CI and 76 on Windows CI**; output measured 28 on local
    // lavapipe (macOS/Windows not yet isolated -- the budgets below were
    // set generously from fuel's real-hardware/lavapipe ratio, ~4x, and
    // need re-checking once a CI run gets far enough to report output's own
    // number). A plain per-pixel budget of 0 for any of the three would be
    // betting real hardware never hits ties lavapipe's software rasteriser
    // does not -- exactly the kind of thing `ARCHITECTURE.md`'s
    // "verification means an automated check" warns against. So all three
    // get a small, non-zero budget, scaled to what was actually measured on
    // real GPUs rather than just the local software one.
    //
    // Suppressing just one slot's item at a time (`queue_panel`'s slot
    // loop, gated on `slot.kind`) -- confined to its own rect, which is
    // itself evidence the three are laid out apart the way the title
    // claims -- changes: input 460 (a 92x margin over its 5-pixel budget),
    // fuel 863 (a 7.2x margin over its 120-pixel budget, 11.3x over the
    // 76 actually measured), output 824 (a 5.2x margin over its 160-pixel
    // budget).
    if let Some(frame) = render_world(&world, shot.clone()) {
        assert_region_golden(
            "furnace_screen",
            &frame,
            (284, 128, 40, 40),
            5,
            "the input slot",
        );
        assert_region_golden(
            "furnace_screen",
            &frame,
            (284, 172, 40, 40),
            120,
            "the fuel slot below it",
        );
        assert_region_golden(
            "furnace_screen",
            &frame,
            (372, 150, 40, 40),
            160,
            "the output slot to the right",
        );
    }
    assert_golden("furnace_screen", &world, shot);
}

#[test]
fn a_furnace_shows_its_flame_and_progress() {
    // A working furnace looked exactly like a broken one until an ingot
    // appeared, and in play that read as "the furnace does not work". Two
    // fractions picked far apart -- a mostly-burnt flame and a mostly-done
    // arrow -- so drawing one where the other belongs would show.
    let panel = InventoryPanel::layout_furnace(960, 540);
    let contents: Vec<Option<HotbarSlot>> = panel
        .slots()
        .iter()
        .map(|s| match s.kind {
            PanelSlotKind::Grid => Some(HotbarSlot {
                color: [0.70, 0.35, 0.25],
                count: 3,
                icon: None,
            }),
            PanelSlotKind::Fuel => Some(HotbarSlot {
                color: [0.66, 0.50, 0.31],
                count: 6,
                icon: None,
            }),
            _ => None,
        })
        .collect();

    let world = World::new();
    let shot = Shot {
        width: 960,
        height: 540,
        region_radius: 2,
        orbit_t: 0.0,
        camera: None,
        players: Vec::new(),
        highlighted_block: None,
        cracking: None,
        hotbar: None,
        health: None,
        crosshair: false,
        tooltip: None,
        gauges: Some((0.25, 0.75)),
        icons: Vec::new(),
        anchor: None,
        menu: None,
        lighting: Lighting::default(),
        panel: Some((PanelLayout::Furnace, contents, None, (10.0, 10.0))),
    };
    // Both meters together are well under the golden's 0.5% tolerance, so a
    // missing or swapped meter would pass the image comparison. Read the
    // pixels where each fraction should and should not have filled.
    if let Some(frame) = render_world(&world, shot.clone()) {
        let at = |x: f32, y: f32| {
            let i = ((y as u32 * frame.width + x as u32) * 4) as usize;
            [frame.pixels[i], frame.pixels[i + 1], frame.pixels[i + 2]]
        };
        let slot = |kind| *panel.slots().iter().find(|s| s.kind == kind).unwrap();
        let (fuel, output) = (slot(PanelSlotKind::Fuel), slot(PanelSlotKind::Result));
        let flame_x = fuel.x - 7.0;
        // The target is sRGB, so the flame reads (253, 196, 97), not its
        // linear value; the track beside it is a dark grey.
        let orange = |p: [u8; 3]| p[0] > 220 && p[0] as i32 - p[2] as i32 > 100;
        assert!(
            orange(at(flame_x, fuel.y + fuel.size - 2.0)),
            "no flame at the bottom"
        );
        // Halfway up: empty for a quarter flame, full if the flame were
        // showing the arrow's three quarters instead.
        assert!(
            !orange(at(flame_x, fuel.y + fuel.size * 0.5)),
            "a quarter flame reached halfway"
        );

        let (x0, x1) = (fuel.x + fuel.size + 6.0, output.x - 6.0);
        let ay = output.y + output.size * 0.5;
        let white = |p: [u8; 3]| p.iter().all(|&c| c > 200);
        assert!(
            white(at(x0 + (x1 - x0) * 0.6, ay)),
            "the arrow has not filled to 3/4"
        );
        assert!(
            !white(at(x0 + (x1 - x0) * 0.9, ay)),
            "the arrow filled past 3/4"
        );
    }
    assert_golden("furnace_gauges", &world, shot);
}

#[test]
fn distinct_materials_render_with_distinct_textures() {
    // Block 1.4a's own bar: the packed vertex format carries a texture-array
    // layer per quad, resolved from the block registry -- this must actually
    // show up as visibly different colours on different materials, not just
    // "the flat green constant became a flat placeholder constant". `World`
    // can produce mixed materials since block 1.5, but only as noise-shaped
    // patches at whatever depth the terrain happens to expose them -- an
    // explicit, controlled block of each material is still the clearer test
    // of per-material appearance on its own, so this goes through
    // `render_chunks` with explicit chunks instead.
    let assets_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/blocks");
    let registry = BlockRegistry::load(&assets_dir).expect("assets/blocks must be valid");
    let stone = registry
        .id_of("cubara:stone")
        .expect("cubara:stone in registry");
    let soil = registry
        .id_of("cubara:soil")
        .expect("cubara:soil in registry");
    let grass = registry
        .id_of("cubara:grass")
        .expect("cubara:grass in registry");

    // Three chunks in a row, each a solid block of one material.
    let chunks = vec![
        (ChunkCoord::new(-1, 0, 0), Chunk::from_fn(|_, _, _| stone)),
        (ChunkCoord::new(0, 0, 0), Chunk::from_fn(|_, _, _| soil)),
        (ChunkCoord::new(1, 0, 0), Chunk::from_fn(|_, _, _| grass)),
    ];

    let shot = Shot {
        width: 960,
        height: 540,
        region_radius: 0, // unused by render_chunks -- the chunk list is explicit
        orbit_t: 5.0,
        camera: None,
        players: Vec::new(),
        highlighted_block: None,
        cracking: None,
        hotbar: None,
        panel: None,
        health: None,
        crosshair: false,
        tooltip: None,
        gauges: None,
        icons: Vec::new(),
        anchor: None,
        menu: None,
        lighting: Lighting::default(),
    };

    let frame = {
        let _gpu = GPU_LIFECYCLE.lock().unwrap_or_else(|e| e.into_inner());
        headless::render_chunks(&chunks, shot)
    };
    let Some(frame) = frame else {
        headless::skip_or_require_gpu("distinct_materials_render_with_distinct_textures");
        return;
    };

    let path = golden_dir().join("materials.png");

    if std::env::var_os("CUBARA_BLESS").is_some() {
        save_png(&path, frame.width, frame.height, &frame.pixels);
        eprintln!("BLESSED {}", path.display());
        return;
    }

    assert!(
        path.exists(),
        "no golden image at {}. If this is a new test, create it with:\n    \
         CUBARA_BLESS=1 cargo test -p cubara-render --test golden",
        path.display()
    );

    let (gw, gh, expected) = load_png(&path);
    assert_eq!((gw, gh), (frame.width, frame.height));

    let diff = headless::compare(&frame.pixels, &expected, TOLERANCE);
    eprintln!(
        "golden materials: {:.4}% differ (tolerance {TOLERANCE}, max delta {}), threshold {:.4}%",
        diff.differing_fraction * 100.0,
        diff.max_channel_delta,
        MAX_DIFFERING * 100.0
    );
    if diff.differing_fraction > MAX_DIFFERING {
        let actual = golden_dir().join("materials.actual.png");
        save_png(&actual, frame.width, frame.height, &frame.pixels);
        panic!(
            "golden materials differs: {:.2}% of pixels exceed tolerance {TOLERANCE} \
             (max channel delta {}), allowed {:.2}%.\n  expected: {}\n  actual:   {}",
            diff.differing_fraction * 100.0,
            diff.max_channel_delta,
            MAX_DIFFERING * 100.0,
            path.display(),
            actual.display(),
        );
    }
}

#[test]
fn edits_change_what_is_drawn() {
    // Proves the golden test has teeth: carve a trench through the surface and the
    // frame must differ from the untouched-terrain reference. A golden test that
    // cannot fail is worse than none, because it reads as coverage.
    let shot = Shot::default();
    let mut world = World::new();
    let Some(base) = render_world(&World::new(), shot.clone()) else {
        headless::skip_or_require_gpu("edits_change_what_is_drawn");
        return;
    };

    // A wide gash straight across the framed region (radius 6 chunks = ~208
    // blocks across), deep enough to cut the surface everywhere it passes.
    for x in -100..100 {
        for z in -16..16 {
            for y in 0..40 {
                world.set_block(x, y, z, BlockId::AIR);
            }
        }
    }
    let edited = render_world(&world, shot).expect("adapter was available a moment ago");

    let diff = headless::compare(&edited.pixels, &base.pixels, TOLERANCE);
    eprintln!(
        "trench changed {:.3}% of pixels (threshold {:.3}%)",
        diff.differing_fraction * 100.0,
        MAX_DIFFERING * 100.0
    );
    assert!(
        diff.differing_fraction > MAX_DIFFERING,
        "carving a gash across the whole region changed only {:.3}% of pixels — the \
         comparison is too loose to catch a real regression",
        diff.differing_fraction * 100.0
    );
}

/// Three players stand in front of the camera: Linux red, Windows yellow,
/// macOS green.
///
/// The owner's arrangement for three machines in one world. Each client
/// hard-codes its own colour and announces it on joining; this is what the
/// three look like side by side.
///
/// The geometry itself is checked without a GPU in `figure.rs` — where an arm
/// ends up is a number, not a picture. What only an image can show is that the
/// pipeline is bound, the depth test is the right way round, the figures are
/// not inside-out from back-face culling, and that the three colours are
/// actually tellable apart by eye rather than only by distance in a test.
#[test]
fn three_players_stand_in_front_of_the_camera() {
    let world = World::with_seed(119);
    let eye = glam::vec3(0.0, 40.0, 6.0);
    let shirt = |c: [u8; 3]| {
        [
            c[0] as f32 / 255.0,
            c[1] as f32 / 255.0,
            c[2] as f32 / 255.0,
        ]
    };
    let shot = Shot {
        width: 960,
        height: 540,
        region_radius: 2,
        orbit_t: 0.0,
        camera: Some((eye.as_dvec3(), glam::vec3(0.0, -0.15, -1.0))),
        players: vec![
            cubara_render::PlayerView {
                eye: [-1.8, 40.0, -1.0],
                yaw: 0.0,
                shirt: shirt([204, 41, 41]),
            },
            cubara_render::PlayerView {
                eye: [0.0, 40.0, -1.0],
                // Turned a quarter, so the image covers yaw as well: a figure
                // that ignored it would be identical to its neighbours.
                yaw: std::f32::consts::FRAC_PI_2,
                shirt: shirt([235, 209, 41]),
            },
            cubara_render::PlayerView {
                eye: [1.8, 40.0, -1.0],
                yaw: 0.0,
                shirt: shirt([51, 179, 61]),
            },
        ],
        highlighted_block: None,
        cracking: None,
        hotbar: None,
        panel: None,
        health: None,
        crosshair: false,
        tooltip: None,
        gauges: None,
        icons: Vec::new(),
        anchor: None,
        menu: None,
        lighting: Lighting::default(),
    };
    // Audit #5: all three figures together are a few hundred pixels of a
    // 960x540 frame -- inside the whole-image golden's 0.5% tolerance, so a
    // defect that ignored `yaw` entirely (`figure_vertices`' `view.yaw.
    // sin_cos()`) would pass `assert_golden` alone, which is exactly the
    // failure mode the comment above names ("a figure that ignored it would
    // be identical to its neighbours"). `(330, 183, 300, 133)` is the union
    // of the three figures' screen-space bounding boxes, computed from each
    // `eye`/`yaw` via `CameraUniform::look_view_proj` and `PLAYER_HEIGHT`/
    // `PLAYER_WIDTH` (confirmed against the committed golden: each figure's
    // shirt colour samples inside its own box, and the turned middle one is
    // visibly narrower there). Measured: 1 of the ~39,900 pixels in this
    // rect differs from the golden at `TOLERANCE` on an NVIDIA/Vulkan
    // machine (ordinary silhouette-edge noise); the real macOS and Windows
    // CI runners' whole-frame percentage for this shot is noticeably higher
    // than this machine's (0.122%/0.124% vs 0.083%, checked via `gh run
    // view --log`), so this rect's count is likely higher there too. Forcing
    // every figure's yaw to 0 changes 1,841. 10 keeps a wide margin under
    // the defect while giving real headroom over the higher CI baseline.
    if let Some(frame) = render_world(&world, shot.clone()) {
        assert_region_golden(
            "three_players",
            &frame,
            (330, 183, 300, 133),
            10,
            "all three figures, including the turned one",
        );
    }
    assert_golden("three_players", &world, shot);
}

/// The far terrain's patches around `eye`, reaching `radius` blocks, with the
/// region [`render_world`] draws as voxels left to it: chunks `-r..=r` across,
/// the three chunk-layers `0..=2` up.
fn far_terrain(
    world: &World,
    eye: glam::Vec3,
    region_radius: i32,
    shot: &Shot,
    radius: f64,
) -> Vec<cubara_world::far::PatchHeights> {
    use cubara_world::far::{build, split_ratio_for, FarView, Hole};
    let fov = std::f64::consts::FRAC_PI_3;
    let pixel = fov / shot.height as f64;
    let edge = (region_radius * 16) as f64;
    let view = FarView {
        eye: [eye.x as f64, eye.y as f64, eye.z as f64],
        // The settings the game ships (`cubara-app`'s `far_streaming`).
        split_ratio: split_ratio_for(16.0, shot.height as f64, fov),
        max_error: 2.0 * pixel,
        min_quad: 4.0 * pixel,
        radius,
        hole: Some(Hole {
            min: [-edge, 0.0, -edge],
            max: [edge + 16.0, 48.0, edge + 16.0],
        }),
    };
    build(&cubara_world::WorldGen::new(world.seed()), &view)
}

/// The far terrain's materials as the game sets them (`cubara-app`'s
/// `far_streaming::far_materials`).
fn far_materials() -> cubara_render::aggregate::Materials {
    cubara_render::far::ground_materials(cubara_world::SOIL_DEPTH as f32)
}

fn render_world_with_far(
    world: &World,
    shot: Shot,
    far: &[cubara_world::far::PatchHeights],
) -> Option<Frame> {
    use cubara_render::far::{FarParams, FarPatch};
    let registry = real_registry();
    let layers = TextureLayers::from_registry(&registry);
    let layer_of = |name: &str| layers.layer_of(name);
    let schedule = schedule_for_radius(shot.region_radius);
    let meshed = mesh_region(
        world,
        &registry,
        &layer_of,
        ChunkCoord::new(0, 0, 0),
        0..=2,
        &schedule,
        real_blocks(),
    )
    .into_iter()
    .filter_map(to_meshed_node);
    let patches: Vec<FarPatch> = far
        .iter()
        .map(|p| FarPatch {
            origin: p.key.origin(),
            quad: p.key.quad() as f32,
            heights: &p.heights,
            skirt: p.skirt(),
        })
        .collect();
    let edge = shot.region_radius * 16;
    let params = FarParams {
        hole: Some(([-edge, 0, -edge], [edge + 16, 48, edge + 16])),
        materials: far_materials(),
    };
    let _gpu = GPU_LIFECYCLE.lock().unwrap_or_else(|e| e.into_inner());
    headless::render_with_far(meshed, &patches, Some(params), shot)
}

/// **The far terrain** (`docs/PROPOSAL_FAR_VIEW.md`, block F3b): past a small
/// voxel region, the land goes on to the horizon -- as a height field, plain
/// shaded, and without the flat plane the voxel LOD collapsed into (the
/// proposal's table B).
#[test]
fn the_far_terrain_reaches_the_horizon() {
    let world = World::new();
    let eye = glam::Vec3::new(8.0, 120.0, 8.0);
    let shot = Shot {
        region_radius: 4,
        camera: Some((eye.as_dvec3(), glam::Vec3::new(1.0, -0.12, 0.35))),
        ..Shot::default()
    };
    let far = far_terrain(&world, eye, shot.region_radius, &shot, 20_000.0);
    assert!(far.len() > 50, "only {} patches", far.len());
    let with = render_world_with_far(&world, shot.clone(), &far);
    // A golden of an empty sky would pass against itself forever. The same
    // shot without the far terrain must look substantially different: that
    // is the far terrain being drawn at all.
    if let (Some(with), Some(without)) = (with.as_ref(), render_world(&world, shot.clone())) {
        let diff = headless::compare(&with.pixels, &without.pixels, TOLERANCE);
        assert!(
            diff.differing_fraction > 0.2,
            "the far terrain changed only {:.1}% of the image",
            diff.differing_fraction * 100.0
        );
    }
    assert_frame_golden("far_terrain_to_the_horizon", with);
}

/// **The far terrain's shader is the CPU reference** (`aggregate.rs`, block
/// F4). A patch that is a plane of known gradient is rendered from a known
/// direction, and the pixel at the centre of the image, whose ray is exactly
/// the look direction, must be the colour `aggregate_colour` computes for that
/// slope and view.
///
/// The cases cover every branch of `far.wgsl`'s `visible_weights` (risers
/// facing, none facing, and one axis hiding the other, both ways round, which
/// goes through the masking-table texture), and `riser_colour` from a
/// one-block step to a stone cliff. This is what pins the texture's layout and
/// the hand-blended fourth axis: get either wrong and the table cases come out
/// a different colour.
#[test]
fn far_shading_is_the_aggregate_reference() {
    use cubara_render::aggregate::{aggregate_colour, face_light, MaskingTable};
    use cubara_render::far::{FarParams, FarPatch, FAR_VERTS};
    use cubara_render::Lighting;

    let srgb = |c: f32| {
        let v = if c <= 0.0031308 {
            c * 12.92
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        };
        (v.clamp(0.0, 1.0) * 255.0).round() as i32
    };
    let view = |elevation: f32, azimuth: f32| {
        let (e, a) = (elevation.to_radians(), azimuth.to_radians());
        glam::Vec3::new(e.cos() * a.cos(), e.sin(), e.cos() * a.sin())
    };
    let table = MaskingTable::committed();
    let materials = far_materials();
    // The shading function alone: the shot stands 40 blocks away, and the
    // haze (which `aggregate_colour` does not model) would tint it by a step.
    let lighting = Lighting {
        haze: cubara_render::haze::Haze {
            density: 0.0,
            ..Default::default()
        },
        ..Lighting::default()
    };
    // (gradient, elevation, azimuth): what each exercises.
    let cases: [([f32; 2], f32, f32); 11] = [
        // x-risers face the eye, nothing hides.
        ([0.4, 0.0], 10.0, 180.0),
        // No riser faces the eye: tops only.
        ([0.3, 0.2], 40.0, 30.0),
        // x faces, z hides: the table.
        ([0.5, 0.35], 8.0, 150.0),
        // z faces, x hides: the table, the other way round.
        ([0.3, -0.7], 12.0, 60.0),
        // Both face, on a cliff: soil in the risers.
        ([2.5, 0.2], 5.0, 190.0),
        // Steeper than the soil is deep: stone.
        ([6.0, 0.0], 20.0, 180.0),
        // Grazing, gentle, one axis hiding: the far terrain's commonest view.
        ([0.12, 0.08], 2.0, 115.0),
        // More of the table, where it changes fastest: slopes seen nearly
        // edge-on, and in between its samples on every axis.
        ([0.5, 0.5], 5.0, 130.0),
        ([1.2, 0.6], 15.0, 160.0),
        ([0.2, 0.6], 45.0, 110.0),
        ([-0.9, 0.25], 30.0, 20.0),
    ];
    let mut plain_would_fail = false;
    for (g, elevation, azimuth) in cases {
        let v = view(elevation, azimuth);
        let expected = aggregate_colour(&table, g, v, &lighting, &materials)
            .expect("every case faces the eye");
        // One patch, a plane with its corner on the origin (a patch origin
        // is an integer block, like every real one), aimed at through its
        // centre vertex. `far.wgsl` shades per vertex, so the quads are made
        // tiny: the four vertices around the centre pixel then see it from
        // the same direction, and the pixel is the shading function itself
        // rather than a blend of its neighbours.
        let quad = 0.05;
        let heights: Vec<f32> = (0..FAR_VERTS * FAR_VERTS)
            .map(|k| {
                let (i, j) = ((k % FAR_VERTS) as f32, (k / FAR_VERTS) as f32);
                g[0] * (i * quad) + g[1] * (j * quad)
            })
            .collect();
        let patch = FarPatch {
            origin: [0, 0],
            quad,
            heights: &heights,
            skirt: 0.0,
        };
        // The centre vertex (16, 16), where the look direction lands.
        let centre = glam::vec3(16.0 * quad, (g[0] + g[1]) * 16.0 * quad, 16.0 * quad);
        let shot = Shot {
            // Odd, so the centre pixel's ray is the look direction itself.
            width: 65,
            height: 65,
            camera: Some(((centre + v * 40.0).as_dvec3(), -v)),
            lighting,
            ..Shot::default()
        };
        let params = FarParams {
            hole: None,
            materials,
        };
        let frame = {
            let _gpu = GPU_LIFECYCLE.lock().unwrap_or_else(|e| e.into_inner());
            headless::render_with_far(Vec::new(), &[patch], Some(params), shot)
        };
        let Some(frame) = frame else {
            headless::skip_or_require_gpu("far_shading_is_the_aggregate_reference");
            return;
        };
        let at = ((32 * frame.width + 32) * 4) as usize;
        let got = &frame.pixels[at..at + 3];
        let want = expected.to_array().map(srgb);
        // Exact on the machine this was written on; one step of slack for
        // another GPU's filtering precision. A wrong texture layout or blend
        // is off by more (the PR's mutation table).
        for c in 0..3 {
            assert!(
                (got[c] as i32 - want[c]).abs() <= 1,
                "g {g:?}, {elevation}° up, {azimuth}°: rendered {got:?}, reference {want:?}"
            );
        }
        // The plain shading F3 drew: the top colour lit by the smooth normal.
        let n = glam::Vec3::new(-g[0], 1.0, -g[1]).normalize();
        let plain = (materials.top * face_light(n, &lighting))
            .to_array()
            .map(srgb);
        if (0..3).any(|c| (plain[c] - want[c]).abs() > 12) {
            plain_would_fail = true;
        }
    }
    assert!(
        plain_would_fail,
        "no case tells the aggregate shading from plain shading"
    );
}

// ---------------------------------------------------------------------------
// Camera-relative rendering (issue #379): the same frame at the origin and a
// million blocks away.
// ---------------------------------------------------------------------------

/// Everything issue #379 makes anchor-relative, in one small scene: level-0
/// nodes meshed from the default world around the origin, three oak
/// instances stood on its surface (a level-0 node carries none of its own,
/// so these are placed by hand -- what the renderer draws does not care
/// which level asked for them), one far tile beyond the region with the
/// region as its hole, the outline on one block, cracks on another, one
/// other player, and a first-person eye looking across all of it.
struct AnchoredScene {
    nodes: Vec<MeshedNode>,
    far_origin: [i64; 2],
    far_heights: Vec<f32>,
    hole: ([i32; 3], [i32; 3]),
    shot: Shot,
}

/// How many blocks a bench or screenshot of this scene would have to
/// travel to reach the issue's test offset: 256 anchor-grid steps.
const FAR_SHIFT: i64 = 1_048_576;

/// The scene, with every position moved by `shift` blocks -- the nodes and
/// their ids, the trees, the far tile, the hole, the outline and crack
/// blocks, the player, the eye. A shift of zero is the scene as the world
/// generated it. A shift must keep the level-0 nodes on their 16-block
/// grid (it is applied to `NodeId::pos` too, so draw order is the same).
fn anchored_scene(shift: [i64; 3]) -> AnchoredScene {
    assert!(
        shift.iter().all(|s| s % 16 == 0),
        "a shift must keep the nodes on the chunk grid: {shift:?}"
    );
    let world = World::new();
    let registry = real_registry();
    let layers = TextureLayers::from_registry(&registry);
    let layer_of = |name: &str| layers.layer_of(name);
    // Radius 3: 7 x 7 chunks, three layers -- level 0 only
    // (`schedule_for_radius` collapses under the first ring), so the nodes'
    // own origins are exactly the 16-block grid the shift is checked against.
    let region_radius = 3;
    let schedule = schedule_for_radius(region_radius);
    let mut nodes: Vec<MeshedNode> = mesh_region(
        &world,
        &registry,
        &layer_of,
        ChunkCoord::new(0, 0, 0),
        0..=2,
        &schedule,
        real_blocks(),
    )
    .into_iter()
    .filter_map(to_meshed_node)
    .collect();
    assert!(nodes.len() > 50, "only {} nodes", nodes.len());

    // Three trees on the surface, each given to the node whose chunk holds
    // its base block -- all three down the look direction, in the frame
    // (`instances_are_in_the_frame` below checks that they really draw).
    let mut planted = 0;
    for (x, z) in [(-22, -18), (-16, -34), (-40, -46)] {
        let base = world.surface_height(x, z) + 1;
        let chunk = ChunkCoord::from_block(x, base, z);
        if let Some(node) = nodes
            .iter_mut()
            .find(|n| n.id.level == 0 && n.id.pos == [chunk.x, chunk.y, chunk.z])
        {
            node.trees.push([x, base, z]);
            node.tree_model = TreeModel::Oak;
            planted += 1;
        }
    }
    assert_eq!(planted, 3, "a tree's chunk is not in the region");

    // Looking toward -x/-z, away from the direction the scene is shifted
    // in: a culling box left absolute by a rebase that did not happen then
    // sits *behind* the camera (inside the 10,000 km far plane it would
    // still be in front), so a missed rebase culls what it owns and shows.
    let h0 = world.surface_height(8, 8);
    let eye = glam::DVec3::new(8.5, h0 as f64 + 24.0, 8.5);
    let look = glam::vec3(-0.55, -0.5, -0.7);

    // One far tile beyond the region's -x/-z corner, a gentle slope a little
    // under the surface here, 128 blocks a side.
    let far_origin = [-192i64, -192i64];
    let far_quad = 4.0f32;
    let far_heights: Vec<f32> = (0..cubara_render::FAR_VERTS * cubara_render::FAR_VERTS)
        .map(|k| {
            let (i, j) = (
                (k % cubara_render::FAR_VERTS) as f32,
                (k / cubara_render::FAR_VERTS) as f32,
            );
            h0 as f32 - 6.0 + 0.15 * i + 0.4 * j
        })
        .collect();
    let edge = region_radius * 16;
    let hole = ([-edge, 0, -edge], [edge + 16, 48, edge + 16]);

    // Thirty-odd blocks down the look direction, where each is a dozen
    // pixels: enough that leaving any of them out shows.
    let outline_block = [-30, world.surface_height(-30, -36), -36];
    let crack_block = [-24, world.surface_height(-24, -30), -30];
    let player_eye = [-36.0, world.surface_height(-36, -40) as f64 + 2.62, -40.0];

    // Apply the shift to everything that is a position. A node's bounds are
    // local to its origin and so do not move.
    let s32 = shift.map(|v| v as i32);
    let s64 = glam::DVec3::new(shift[0] as f64, shift[1] as f64, shift[2] as f64);
    for node in &mut nodes {
        let extent = 16i64 << node.id.level;
        for a in 0..3 {
            node.id.pos[a] += (shift[a] / extent) as i32;
            node.origin[a] += s32[a];
        }
        for t in &mut node.trees {
            for a in 0..3 {
                t[a] += s32[a];
            }
        }
    }
    let far_origin = [far_origin[0] + shift[0], far_origin[1] + shift[2]];
    let far_heights = far_heights
        .into_iter()
        .map(|h| h + shift[1] as f32)
        .collect();
    let add3 = |p: [i32; 3]| [p[0] + s32[0], p[1] + s32[1], p[2] + s32[2]];
    let hole = (add3(hole.0), add3(hole.1));
    let shot = Shot {
        width: 960,
        height: 540,
        region_radius,
        orbit_t: 0.0,
        camera: Some((eye + s64, look)),
        anchor: None,
        players: vec![cubara_render::PlayerView {
            eye: (glam::DVec3::from_array(player_eye) + s64).to_array(),
            yaw: 0.7,
            shirt: [0.8, 0.16, 0.16],
        }],
        highlighted_block: Some(add3(outline_block)),
        cracking: Some((add3(crack_block), 0.6)),
        hotbar: None,
        panel: None,
        health: None,
        crosshair: false,
        tooltip: None,
        gauges: None,
        icons: Vec::new(),
        menu: None,
        lighting: Lighting::default(),
    };
    let _ = far_quad;
    AnchoredScene {
        nodes,
        far_origin,
        far_heights,
        hole,
        shot,
    }
}

/// Render an [`AnchoredScene`], or `None` without a GPU. With
/// `without_trees`, its nodes carry no instances.
fn render_anchored(
    scene: AnchoredScene,
    anchor: Option<cubara_render::Anchor>,
    without_trees: bool,
) -> Option<Frame> {
    use cubara_render::far::{FarParams, FarPatch};
    let patch = FarPatch {
        origin: scene.far_origin,
        quad: 4.0,
        heights: &scene.far_heights,
        skirt: 8.0,
    };
    let params = FarParams {
        hole: Some(scene.hole),
        materials: far_materials(),
    };
    let shot = Shot {
        anchor,
        ..scene.shot
    };
    let mut nodes = scene.nodes;
    if without_trees {
        for node in &mut nodes {
            node.trees.clear();
        }
    }
    let _gpu = GPU_LIFECYCLE.lock().unwrap_or_else(|e| e.into_inner());
    headless::render_with_far(nodes, &[patch], Some(params), shot)
}

/// How much of `frame` is not the sky: the share of pixels that differ from
/// an empty render of the same shot. Guards the tests below against a scene
/// that draws nothing and so trivially matches itself.
fn drawn_fraction(frame: &Frame, shot: &Shot) -> f64 {
    let empty = Shot {
        players: Vec::new(),
        highlighted_block: None,
        cracking: None,
        ..shot.clone()
    };
    let sky = {
        let _gpu = GPU_LIFECYCLE.lock().unwrap_or_else(|e| e.into_inner());
        headless::render_with_far(Vec::new(), &[], None, empty)
    }
    .expect("adapter was available a moment ago");
    headless::compare(&frame.pixels, &sky.pixels, 0).differing_fraction
}

/// **Decision 1 and 6's test: a scene moved by a multiple of the anchor grid
/// renders byte-identically** (issue #379). At 2^20 blocks an `f32` steps
/// in 1/8 of a block, so an absolute `f32` renderer visibly breaks the
/// shifted scene up (`without_the_anchor_the_shifted_scene_falls_apart`
/// below is that renderer, and the proof this test can fail). With every
/// position relative to the anchor -- which is (2^20, 0, 2^20) for the
/// shifted eye and (0, 0, 0) for the original -- the GPU sees the same
/// numbers both times, so the two frames are the same bytes: nodes, trees,
/// far tile, hole, outline, cracks, figure, haze and all.
///
/// Exactly equal, not within the golden tolerance: the point of the anchor
/// is that the arithmetic is identical, and the existing
/// `the_same_scene_renders_byte_identically` measured this GPU's
/// run-to-run noise at 1-2 pixels of delta 1 roughly one run in three, so
/// the bound is the same one that test settled on.
#[test]
fn a_scene_moved_by_a_grid_multiple_renders_byte_identically() {
    let here = anchored_scene([0; 3]);
    let shot = here.shot.clone();
    let Some(a) = render_anchored(here, None, false) else {
        headless::skip_or_require_gpu("a_scene_moved_by_a_grid_multiple_renders_byte_identically");
        return;
    };
    let drawn = drawn_fraction(&a, &shot);
    assert!(
        drawn > 0.3,
        "the scene covers only {:.1}% of the frame",
        drawn * 100.0
    );
    // The instances are in the frame, so a tree arena that was not rebased
    // with its nodes (its boxes left absolute, its instances culled) shows
    // -- `check-tests-can-fail.sh` found `TreeArena::rebase`'s no-op guard
    // survive while the instances stood behind the camera.
    let bare = render_anchored(anchored_scene([0; 3]), None, true)
        .expect("adapter was available a moment ago");
    let instances_are_in_the_frame = differing_pixels(&a, &bare).len();
    assert!(
        instances_are_in_the_frame > 500,
        "the tree instances change only {instances_are_in_the_frame} pixels -- not in view"
    );
    let b = render_anchored(anchored_scene([FAR_SHIFT, 0, FAR_SHIFT]), None, false)
        .expect("adapter was available a moment ago");

    let differing = differing_pixels(&a, &b);
    let max_delta = differing.iter().map(|p| p.delta).max().unwrap_or(0);
    eprintln!(
        "a_scene_moved_by_a_grid_multiple_renders_byte_identically: {} of {} pixels differ, \
         max channel delta {max_delta}",
        differing.len(),
        a.pixels.len() / 4
    );
    assert!(
        max_delta <= 1 && differing.len() <= 4,
        "the scene 2^20 blocks out is not the scene at the origin:\n{}",
        describe(&differing)
    );
}

/// **And up a grid step** (review finding B1): the same scene 4,096 blocks
/// higher, so the anchor's `y` is 4,096 rather than 0. Everything the GPU
/// sees is anchor-relative except one buffer -- the far terrain's heights,
/// which stay absolute -- so `far.wgsl` must take `anchor.y` off them, the
/// far culling boxes likewise (`PlacedPatch::aabb`), and the haze must get
/// the eye's absolute height back. Without the first two the far tile here
/// draws 4,096 blocks too high, its hole test compares a relative box to an
/// absolute vertex, and this frame is not the frame at the origin. The
/// x/z-only test above cannot see any of that: its anchor has `y = 0`.
///
/// The height haze is turned off for this pair, and only this pair: it
/// depends on absolute height *by design* (decision 4 -- the air 4,096
/// blocks up really is thinner, `Haze::scale_height` is 600), so with it on
/// the two frames legitimately differ by a few shades everywhere. What this
/// test pins is everything else that carries the anchor's `y`.
#[test]
fn a_scene_moved_up_a_grid_step_renders_byte_identically() {
    let no_haze = Lighting {
        haze: cubara_render::haze::Haze {
            density: 0.0,
            ..Default::default()
        },
        ..Lighting::default()
    };
    let mut here = anchored_scene([0; 3]);
    here.shot.lighting = no_haze;
    let shot = here.shot.clone();
    let Some(a) = render_anchored(here, None, false) else {
        headless::skip_or_require_gpu("a_scene_moved_up_a_grid_step_renders_byte_identically");
        return;
    };
    // The far tile has to be in the frame for its height to matter.
    let no_far = {
        let scene = anchored_scene([0; 3]);
        let shot = Shot {
            anchor: None,
            lighting: no_haze,
            ..scene.shot
        };
        let _gpu = GPU_LIFECYCLE.lock().unwrap_or_else(|e| e.into_inner());
        headless::render_with_far(scene.nodes, &[], None, shot)
    }
    .expect("adapter was available a moment ago");
    let far_is_in_the_frame = differing_pixels(&a, &no_far).len();
    assert!(
        far_is_in_the_frame > 500,
        "the far tile changes only {far_is_in_the_frame} pixels -- not in view"
    );
    let mut moved = anchored_scene([FAR_SHIFT, 4096, FAR_SHIFT]);
    moved.shot.lighting = no_haze;
    let up = render_anchored(moved, None, false).expect("adapter was available a moment ago");
    let differing = differing_pixels(&a, &up);
    let max_delta = differing.iter().map(|p| p.delta).max().unwrap_or(0);
    eprintln!(
        "a_scene_moved_up_a_grid_step_renders_byte_identically: {} of {} pixels differ, \
         max channel delta {max_delta}",
        differing.len(),
        shot.width as usize * shot.height as usize
    );
    // Not quite byte-identical, and for a stated reason: the far heights are
    // stored as absolute `f32`s, and at 4,096 + h an `f32` has 1/2,048 of a
    // block left, so `(h + 4096) - 4096` is `h` to within that -- enough to
    // flip the odd silhouette pixel of the far tile against the sky (one,
    // measured). A far tile drawn a grid step too high moves hundreds of
    // thousands.
    assert!(
        differing.len() <= 8,
        "the scene 4,096 blocks up is not the scene at the origin:\n{}",
        describe(&differing)
    );
}

/// **The same at an offset that is not a multiple of the grid**: the eye
/// lands 1,160 blocks from its anchor instead of 8, so every anchor-relative
/// number has a different magnitude and `f32` rounds its last bits
/// differently -- the frame may move by a pixel here and there, and must
/// stay inside the golden tolerance. The forced-origin renderer below is
/// well outside it at the same offset.
#[test]
fn a_scene_moved_off_the_grid_renders_within_the_golden_tolerance() {
    let Some(a) = render_anchored(anchored_scene([0; 3]), None, false) else {
        headless::skip_or_require_gpu(
            "a_scene_moved_off_the_grid_renders_within_the_golden_tolerance",
        );
        return;
    };
    // 1,152 and -896 are multiples of 16 (the nodes stay on their grid) and
    // not of 4,096 (the anchor stays at 2^20, so the eye is off-centre).
    let c = render_anchored(
        anchored_scene([FAR_SHIFT + 1152, 0, FAR_SHIFT - 896]),
        None,
        false,
    )
    .expect("adapter was available a moment ago");
    let diff = headless::compare(&a.pixels, &c.pixels, TOLERANCE);
    eprintln!(
        "a_scene_moved_off_the_grid_renders_within_the_golden_tolerance: {:.4}% differ \
         (tolerance {TOLERANCE}, max delta {}), threshold {:.4}%",
        diff.differing_fraction * 100.0,
        diff.max_channel_delta,
        MAX_DIFFERING * 100.0
    );
    assert!(
        diff.differing_fraction <= MAX_DIFFERING,
        "{:.4}% of pixels differ between the scene at the origin and the scene \
         {} blocks out with its eye off the anchor's centre -- more than the \
         golden tolerance allows",
        diff.differing_fraction * 100.0,
        FAR_SHIFT + 1152
    );
}

/// **The proof the two tests above can fail**: the same shifted scene drawn
/// with the anchor forced to the world origin is what `origin/main`'s
/// renderer drew -- absolute `f32` positions and an absolute `f32` eye --
/// and it is visibly not the scene at the origin. Kept as a test rather than
/// a one-off so the anchor cannot be quietly made a no-op (an `anchor_for`
/// that always returns the origin passes everything else in this file).
#[test]
fn without_the_anchor_the_shifted_scene_falls_apart() {
    let Some(a) = render_anchored(anchored_scene([0; 3]), None, false) else {
        headless::skip_or_require_gpu("without_the_anchor_the_shifted_scene_falls_apart");
        return;
    };
    let broken = render_anchored(
        anchored_scene([FAR_SHIFT, 0, FAR_SHIFT]),
        Some(cubara_render::anchor::ORIGIN),
        false,
    )
    .expect("adapter was available a moment ago");
    let diff = headless::compare(&a.pixels, &broken.pixels, TOLERANCE);
    eprintln!(
        "without_the_anchor_the_shifted_scene_falls_apart: {:.4}% differ (tolerance \
         {TOLERANCE}, max delta {})",
        diff.differing_fraction * 100.0,
        diff.max_channel_delta
    );
    assert!(
        diff.differing_fraction > 4.0 * MAX_DIFFERING,
        "with the anchor forced to the origin only {:.4}% of pixels differ -- the \
         absolute-f32 breakage this issue exists for is not showing, so the tests \
         above are not evidence",
        diff.differing_fraction * 100.0
    );
}

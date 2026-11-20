//! The render-consistency probe: renders the same eye and frame with the
//! ground forced to **one representation at a time** -- level-0 voxels, the
//! coarse voxel levels 1, 2 and 3, and the far terrain alone -- and compares
//! each rendered image with the level-0 one, in pixels (issues #423, #422,
//! #380's last comment).
//!
//! **Why pixels.** `lod_probe` and the consistency harness of #423 check the
//! CPU-side data (heights, colour per face). They cannot see what the shaders
//! do on top: lighting, baked AO, texture tiling (a coarse top tiles its
//! 16x16 texture once per cell, 2^L blocks), fog and haze. #380's measurement
//! found that at today's worst swap every changed pixel was land to land,
//! from exactly those. This probe measures the images the player sees.
//!
//! **One render path.** Every image goes through
//! [`cubara_render::headless::render_with_far`], which runs the one
//! `SceneRenderer::encode_scene` the window runs (`ARCHITECTURE.md` Rule 5).
//! There is no pass of its own here. The one hook in the render crate is
//! `render_with_far_cleared`: a colour nothing else draws (magenta, as `--bench flight`
//! uses) so that "sky" is exact per pixel instead of a guess from the sky's
//! gradient.
//!
//! **What is forced, and what is not.** The ground inside a *box* of chunks
//! (aligned to the level-3 grid, so every level tiles exactly the same
//! footprint -- [`forced_nodes`], a parameter, never a global) is drawn by one
//! representation. Outside the box the same far terrain is drawn in every
//! image, as the game draws it beyond its voxel rings, with the box as the
//! far terrain's hole. The "far alone" image has no voxels and no hole, so
//! the far terrain also draws the box. A *backdrop* image (far terrain with
//! the hole, no voxels) tells which pixels the box touches: any pixel an
//! image shows differently from the backdrop is "box content"
//! ([`box_mask`]). Everything below is measured inside that mask only, so
//! the identical far backdrop does not dilute a number.
//!
//! **Metrics** (each is a pure function over two RGBA frames and a mask, with
//! unit tests, so it can be shown to fail):
//!
//! * [`silhouette_diff`] -- per pixel column, the row of the topmost land
//!   pixel, level L against level 0; difference in pixels, max and p99.
//!   Only columns whose silhouette is the box's own (the topmost land pixel of
//!   either image is box content) count.
//! * [`tile_diff`] -- the mean colour per screen tile, difference in /255
//!   (largest channel), p50, p99, max. **The tile is chosen, not derived**:
//!   at least the level's cell in pixels at the scene's distance
//!   ([`tile_px`]), never below 16 px, 32 px for the far terrain (its quads
//!   are at most 16 px wide at High, so a tile holds two). Only tiles that
//!   are land in both images and mostly box content count, so a silhouette
//!   shift is not also charged as a colour error.
//! * [`material_share`] -- the share of stone, grass and soil among the land
//!   pixels of the box content. **Unmixed by colour, not read from a material
//!   id**: the renderer has no id buffer and adding one would be a second
//!   render pass. The method and its measured error are in [`Endmembers`].
//!
//! **Not a benchmark**, and nothing here runs in the game. The images and the
//! table come from an `#[ignore]`d test:
//!
//! ```text
//! CUBARA_PROBE_OUT=<dir> cargo test --release -p cubara --bin cubara \
//!     consistency_probe::print_consistency_report -- --ignored --nocapture
//! ```
//!
//! and the assertion on the metrics, red until #423 / #422 land, is
//! `levels_show_the_same_ground`.

use std::path::{Path, PathBuf};

use cubara_render::headless::{self, Frame};
use cubara_render::materials::TextureLayers;
use cubara_render::{load_registry, Lighting, MeshedNode, Shot};
use cubara_world::far::{self, Hole, PatchHeights};
use cubara_world::mesh::mesh_nodes;
use cubara_world::node::NodeKey;
use cubara_world::{TerrainBlocks, World, WorldGen};
use glam::Vec3;

use crate::far_streaming::{far_view, FarQuality};
use crate::flight::{shows_nothing, tallest_surface, NOTHING};
use crate::streaming::to_meshed_node;

/// The image size every probe image has. Fixed, like the seed and the eyes.
pub const SIZE: (u32, u32) = (1280, 720);

/// The camera's vertical field of view, as `render.rs` projects it
/// (`far_streaming::FOV_Y`).
const FOV_Y: f32 = std::f32::consts::FRAC_PI_3;

/// What draws the ground inside the box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rep {
    /// Voxel nodes, every one of this level (0 = chunks, 1..=3 coarse).
    Level(u32),
    /// The far terrain alone: no voxels, no hole.
    Far,
}

impl Rep {
    /// The five representations, level 0 first (the reference).
    pub const ALL: [Rep; 5] = [
        Rep::Level(0),
        Rep::Level(1),
        Rep::Level(2),
        Rep::Level(3),
        Rep::Far,
    ];

    pub fn name(self) -> String {
        match self {
            Rep::Level(l) => format!("level{l}"),
            Rep::Far => "far".to_string(),
        }
    }
}

/// A fixed view: where the eye is, where it looks, and the box of chunks the
/// forced representation covers. `box_min` and `box_ext` are multiples of 8
/// (the level-3 node's extent in chunks), so every level tiles the box exactly.
#[derive(Clone, Debug, PartialEq)]
pub struct Scene {
    pub name: &'static str,
    pub eye: [f64; 3],
    pub look: Vec3,
    pub box_min: [i32; 3],
    pub box_ext: [i32; 3],
    /// Roughly how far the scene's subject is from the eye, in blocks: what
    /// [`tile_px`] sizes a tile by.
    pub subject_distance: f32,
}

impl Scene {
    /// The box as the far terrain's hole, in blocks.
    pub fn hole(&self) -> Hole {
        let block = |c: i32| (c * 16) as f64;
        Hole {
            min: self.box_min.map(block),
            max: std::array::from_fn(|a| block(self.box_min[a] + self.box_ext[a])),
        }
    }
}

/// Every node of exactly `level` that tiles the box of chunks
/// `box_min .. box_min + box_ext` -- the forced representation. The ring
/// schedule (`node::select_nodes`) cannot do this: its footprint depends on
/// the level through rounding, and the comparison needs the same footprint at
/// every level. Panics if the box is not aligned to the level-3 grid, which is
/// what makes it tile at every level.
pub fn forced_nodes(level: u32, box_min: [i32; 3], box_ext: [i32; 3]) -> Vec<NodeKey> {
    const ALIGN: i32 = 8;
    assert!(level <= 3, "the probe forces levels 0..=3");
    for a in 0..3 {
        assert!(
            box_min[a] % ALIGN == 0 && box_ext[a] % ALIGN == 0 && box_ext[a] > 0,
            "the box must be aligned to the level-3 grid ({ALIGN} chunks)"
        );
    }
    let extent = 1i32 << level;
    let lo: [i32; 3] = box_min.map(|c| c / extent);
    let hi: [i32; 3] = std::array::from_fn(|a| (box_min[a] + box_ext[a]) / extent);
    let mut nodes = Vec::new();
    for x in lo[0]..hi[0] {
        for y in lo[1]..hi[1] {
            for z in lo[2]..hi[2] {
                nodes.push(NodeKey::new(level, [x, y, z]));
            }
        }
    }
    nodes
}

/// The smallest box aligned to the level-3 grid that contains every block
/// column in `columns` (with `margin` blocks around them) and every height up
/// to `top_block`. Returns `(box_min, box_ext)` in chunks.
fn aligned_box(
    columns: &[[i32; 2]],
    margin: i32,
    bottom_block: i32,
    top_block: i32,
) -> ([i32; 3], [i32; 3]) {
    const ALIGN: i32 = 8;
    let chunk = |b: i32| b.div_euclid(16);
    let lo_c = |a: usize| columns.iter().map(|c| chunk(c[a] - margin)).min().unwrap();
    let hi_c = |a: usize| columns.iter().map(|c| chunk(c[a] + margin)).max().unwrap();
    let floor = |c: i32| c.div_euclid(ALIGN) * ALIGN;
    let ceil = |c: i32| (c + ALIGN - 1).div_euclid(ALIGN) * ALIGN;
    let x = (floor(lo_c(0)), ceil(hi_c(0) + 1));
    let z = (floor(lo_c(1)), ceil(hi_c(1) + 1));
    let y = (floor(chunk(bottom_block)), ceil(chunk(top_block) + 1));
    ([x.0, y.0, z.0], [x.1 - x.0, y.1 - y.0, z.1 - z.0])
}

/// The eye `lift` blocks above the standing height (two blocks above the
/// ground under it).
fn eye_above_ground(gen: &WorldGen, x: f64, z: f64, lift: f64) -> [f64; 3] {
    let ground = gen.surface_height(x.floor() as i32, z.floor() as i32);
    [x, f64::from(ground) + 2.0 + lift, z]
}

/// Surface heights over the box, sampled every 8 blocks: `(lowest, highest)`.
fn height_range(gen: &WorldGen, columns: &[[i32; 2]], margin: i32) -> (i32, i32) {
    let (mut lo, mut hi) = (i32::MAX, i32::MIN);
    let x0 = columns.iter().map(|c| c[0]).min().unwrap() - margin;
    let x1 = columns.iter().map(|c| c[0]).max().unwrap() + margin;
    let z0 = columns.iter().map(|c| c[1]).min().unwrap() - margin;
    let z1 = columns.iter().map(|c| c[1]).max().unwrap() + margin;
    for x in (x0..=x1).step_by(8) {
        for z in (z0..=z1).step_by(8) {
            let h = gen.surface_height(x, z);
            lo = lo.min(h);
            hi = hi.max(h);
        }
    }
    (lo, hi)
}

/// How far the mountain's summit is from the eye, in blocks (chosen).
const MOUNTAIN_DISTANCE: f64 = 420.0;
/// How far the box reaches to each side of the line from the eye to the
/// subject, in blocks (chosen).
const BOX_LATERAL: i32 = 160;
/// How far the box reaches past the subject and in front of its near edge, in
/// blocks (chosen; rounding out to the level-3 grid adds up to 127 more).
const BOX_MARGIN: i32 = 48;
/// The box starts at least this far in front of the eye, in blocks (chosen):
/// a forced level 3 is 8 blocks per cell and is never drawn that close in the
/// game, so the eye stands outside the box (rounding the box out to the
/// level-3 grid moves its edge toward the eye by up to 127 blocks, which
/// [`scene_from`] allows for, and a test pins).
const NEAR_GAP: i32 = 288;
/// What the eye must keep from the box after rounding, in blocks.
pub const MIN_EYE_TO_BOX: f64 = 128.0;
/// How far below the summit the mountain view aims, in blocks (chosen).
const MOUNTAIN_AIM_BELOW: i32 = 10;
/// How far below the summit the mountain's eye is, in blocks (chosen).
const MOUNTAIN_EYE_BELOW: i32 = 140;
/// How far above standing height the slope's eye floats, in blocks (chosen):
/// high enough that the slope fills the lower half of the frame instead of
/// showing edge-on.
const SLOPE_EYE_LIFT: f64 = 160.0;
/// Blocks above the highest surface the box keeps, for oaks (chosen: the
/// oak's tallest trunk plus canopy is well under this).
const TREE_ROOM: i32 = 40;

/// A scene looking along a compass axis (`forward` is one of `[+-1, 0]` or
/// `[0, +-1]` in x, z), the eye `eye_lift` blocks above standing height at
/// `eye_xz` (or at `eye_y` if given), looking at `target`. The box starts [`NEAR_GAP`] blocks in front of the eye and
/// reaches [`BOX_MARGIN`] past the target and to either side.
fn scene_from(
    name: &'static str,
    gen: &WorldGen,
    eye_xz: [f64; 2],
    eye_lift: f64,
    forward: [i32; 2],
    target: [i32; 3],
    subject_distance: f32,
) -> Scene {
    let eye = eye_above_ground(gen, eye_xz[0], eye_xz[1], eye_lift);
    let near = [
        eye[0] as i32 + forward[0] * (NEAR_GAP + BOX_MARGIN),
        eye[2] as i32 + forward[1] * (NEAR_GAP + BOX_MARGIN),
    ];
    // The line from the near edge to the target, widened to either side.
    let side = [forward[1], -forward[0]];
    let columns: Vec<[i32; 2]> = [near, [target[0], target[2]]]
        .into_iter()
        .flat_map(|c| {
            [-1, 1].map(move |k| {
                [
                    c[0] + k * BOX_LATERAL * side[0],
                    c[1] + k * BOX_LATERAL * side[1],
                ]
            })
        })
        .collect();
    let (lo, hi) = height_range(gen, &columns, BOX_MARGIN);
    let (box_min, box_ext) = aligned_box(&columns, BOX_MARGIN, (lo - 16).max(0), hi + TREE_ROOM);
    let look = (Vec3::new(target[0] as f32, target[1] as f32, target[2] as f32)
        - Vec3::new(eye[0] as f32, eye[1] as f32, eye[2] as f32))
    .normalize();
    Scene {
        name,
        eye,
        look,
        box_min,
        box_ext,
        subject_distance,
    }
}

impl Scene {
    /// How far the eye is from the box, horizontally, in blocks (0 inside it).
    pub fn eye_to_box(&self) -> f64 {
        let hole = self.hole();
        let gap = |a: usize| {
            (hole.min[a] - self.eye[a])
                .max(self.eye[a] - hole.max[a])
                .max(0.0)
        };
        gap(0).hypot(gap(2))
    }
}

/// The side of a peak at `(px, pz)` facing spawn, snapped to one compass axis
/// (the direction from the peak toward the eye).
fn facing_spawn(px: i32, pz: i32) -> [i32; 2] {
    if px.abs() >= pz.abs() {
        [-px.signum(), 0]
    } else {
        [0, -pz.signum()]
    }
}

/// The mountain: the peak `--bench flight --toward-peak` flies at (the tallest
/// surface within 4 km of spawn), seen from [`MOUNTAIN_DISTANCE`] blocks away
/// along the compass axis closest to the side facing spawn.
pub fn mountain_scene(gen: &WorldGen) -> Scene {
    let (h, px, pz) = tallest_surface(gen);
    let facing = facing_spawn(px, pz);
    let eye = [
        px as f64 + f64::from(facing[0]) * MOUNTAIN_DISTANCE,
        pz as f64 + f64::from(facing[1]) * MOUNTAIN_DISTANCE,
    ];
    // Looking back at the peak, so forward is minus `facing`. The eye is
    // `MOUNTAIN_EYE_BELOW` under the summit (or on the ground, if that is
    // higher), so the ridge stands against the sky and the box's silhouette
    // is the image's own.
    let ground_eye = eye_above_ground(gen, eye[0], eye[1], 0.0)[1];
    let lift = (f64::from(h - MOUNTAIN_EYE_BELOW) - ground_eye).max(0.0);
    scene_from(
        "mountain",
        gen,
        eye,
        lift,
        [-facing[0], -facing[1]],
        [px, h - MOUNTAIN_AIM_BELOW, pz],
        MOUNTAIN_DISTANCE as f32,
    )
}

/// The grass the owner walks up: how far the slope runs from its foot, in
/// blocks, and the gradient window it must have (rise per block run), and the
/// share of its columns that must show grass. All chosen.
const SLOPE_RUN: i32 = 224;
const SLOPE_GRADIENT: (f32, f32) = (0.10, 0.30);
const SLOPE_GRASS_SHARE: f32 = 0.95;

/// Whether the column at `(x, z)` shows grass on top (no stone, no snow, no
/// bare rock): the surface block level 0 draws there.
fn grass_on_top(gen: &WorldGen, blocks: TerrainBlocks, x: i32, z: i32) -> bool {
    let h = gen.surface_height(x, z);
    gen.terrain_block_at(x, h, z, blocks) == Some(blocks.grass)
}

/// A gentle grassy slope: the first spot, scanning outward from spawn on a
/// 64-block grid in eight directions, where the ground rises over
/// [`SLOPE_RUN`] blocks at a gradient inside [`SLOPE_GRADIENT`] and is almost
/// all grass on top. Deterministic for a seed. `None` if the world has none
/// within 3 km (the test then fails loudly rather than pick something else).
pub fn slope_scene(gen: &WorldGen, blocks: TerrainBlocks) -> Option<Scene> {
    // Compass axes only: the box is axis-aligned and the eye stands outside it.
    const DIRS: [(i32, i32); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];
    let mut candidates: Vec<(i32, i32)> = Vec::new();
    for x in (-3072..=3072).step_by(64) {
        for z in (-3072..=3072).step_by(64) {
            candidates.push((x, z));
        }
    }
    candidates.sort_by_key(|&(x, z)| (i64::from(x).pow(2) + i64::from(z).pow(2), x, z));
    for (x, z) in candidates {
        for (dx, dz) in DIRS {
            let at = |t: i32| (x + dx * t, z + dz * t);
            let (x1, z1) = at(SLOPE_RUN);
            let rise = (gen.surface_height(x1, z1) - gen.surface_height(x, z)) as f32;
            let gradient = rise / SLOPE_RUN as f32;
            if !(SLOPE_GRADIENT.0..=SLOPE_GRADIENT.1).contains(&gradient) {
                continue;
            }
            let samples: Vec<(i32, i32)> = (0..=SLOPE_RUN).step_by(4).map(at).collect();
            let grass = samples
                .iter()
                .filter(|&&(sx, sz)| grass_on_top(gen, blocks, sx, sz))
                .count();
            if (grass as f32) < SLOPE_GRASS_SHARE * samples.len() as f32 {
                continue;
            }
            // Rising the whole way: no dips deeper than a few blocks, which
            // would hide the slope's own foot from the eye.
            let heights: Vec<i32> = samples
                .iter()
                .map(|&(sx, sz)| gen.surface_height(sx, sz))
                .collect();
            let monotone = heights.windows(2).all(|w| w[1] >= w[0] - 3);
            if !monotone {
                continue;
            }
            let (tx, tz) = at(SLOPE_RUN * 3 / 4);
            // The eye stands NEAR_GAP blocks back from the slope's foot, so
            // the box begins about where the slope does.
            let (ex, ez) = at(-NEAR_GAP);
            return Some(scene_from(
                "slope",
                gen,
                [f64::from(ex), f64::from(ez)],
                SLOPE_EYE_LIFT,
                [dx, dz],
                [tx, gen.surface_height(tx, tz), tz],
                NEAR_GAP as f32 + SLOPE_RUN as f32 * 0.75,
            ));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// What the probe renders against: the world, the block registry and the
/// terrain blocks (with or without oaks). Passed in, never global (Rule 2).
pub struct Setup {
    pub world: World,
    pub gen: WorldGen,
    pub blocks: TerrainBlocks,
}

impl Setup {
    /// The default seed's world, ores on as in the game; oaks on or off. Oaks
    /// are off for the ground comparison: a tree is a different question
    /// (level 3 draws boxes) and its green would muddy the stone/grass share.
    pub fn new(trees: bool) -> Self {
        let world = World::new();
        let registry = load_registry();
        let blocks = TerrainBlocks::from_registry(&registry)
            .with_ores(&crate::game::load_ore_registry(), &registry);
        let blocks = if trees {
            blocks.with_oak(&crate::game::load_structure_registry(), &registry)
        } else {
            blocks
        };
        let gen = WorldGen::new(world.seed());
        Self { world, gen, blocks }
    }
}

/// The far terrain's patches for one scene, with and without the box as a hole.
pub struct FarSet {
    pub with_hole: Vec<PatchHeights>,
    pub without: Vec<PatchHeights>,
}

impl FarSet {
    pub fn build(setup: &Setup, scene: &Scene) -> Self {
        let make = |hole: Option<Hole>| {
            let view =
                far_view(scene.eye, SIZE.1, hole, FarQuality::High).expect("High is not Off");
            far::build(&setup.gen, &view)
        };
        Self {
            with_hole: make(Some(scene.hole())),
            without: make(None),
        }
    }
}

/// Render `scene` with `rep` forced inside the box. `None` without a GPU.
pub fn render(setup: &Setup, far_set: &FarSet, scene: &Scene, rep: Option<Rep>) -> Option<Frame> {
    let registry = load_registry();
    let layers = TextureLayers::from_registry(&registry);
    let layer_of = |name: &str| layers.layer_of(name);
    // `None` is the backdrop: the far terrain with the box as its hole, no voxels.
    let (meshed, far_patches, hole) = match rep {
        Some(Rep::Level(level)) => {
            let nodes = forced_nodes(level, scene.box_min, scene.box_ext);
            let built = mesh_nodes(&setup.world, &registry, &layer_of, nodes, setup.blocks);
            (
                built
                    .into_iter()
                    .filter_map(to_meshed_node)
                    .collect::<Vec<_>>(),
                &far_set.with_hole,
                Some(scene.hole()),
            )
        }
        Some(Rep::Far) => (Vec::new(), &far_set.without, None),
        None => (Vec::new(), &far_set.with_hole, Some(scene.hole())),
    };
    check_fits_the_arena(scene, rep, &meshed);
    draw(scene.eye, scene.look, meshed, far_patches, hole)
}

/// The one draw call of every probe image: `meshed` voxels and `far_patches`
/// with `hole` cut out of the far terrain, from `eye` looking along `look`,
/// fog and haze as the game's far view has them (`screenshot::run`), the
/// frame cleared to magenta so "sky" is exact.
fn draw(
    eye: [f64; 3],
    look: Vec3,
    meshed: Vec<MeshedNode>,
    far_patches: &[PatchHeights],
    hole: Option<Hole>,
) -> Option<Frame> {
    let patches: Vec<_> = far_patches
        .iter()
        .map(crate::far_streaming::to_far_patch)
        .collect();
    // The hole parameter of the shader: the box, or nothing.
    let params = match hole {
        Some(h) => crate::far_streaming::to_far_params(h, crate::far_streaming::far_materials()),
        None => cubara_render::FarParams {
            hole: None,
            materials: crate::far_streaming::far_materials(),
        },
    };
    // Fog and haze as the game's far view draws them (`screenshot::run`).
    let (fog_start, fog_end) = Lighting::fog_range(crate::far_streaming::FAR_VIEW_RADIUS as f32);
    let shot = Shot {
        width: SIZE.0,
        height: SIZE.1,
        camera: Some((glam::DVec3::from(eye), look)),
        lighting: Lighting {
            fog_start,
            fog_end,
            ..Lighting::default()
        },
        ..Shot::default()
    };
    headless::render_with_far_cleared(meshed, &patches, Some(params), shot, Some(NOTHING))
}

/// Panics if `meshed` is more than the arena holds: it would drop nodes
/// silently and the image would show holes that are the probe's, not the
/// renderer's. (The first version of this probe did exactly that at level 0.)
fn check_fits_the_arena(scene: &Scene, rep: Option<Rep>, meshed: &[MeshedNode]) {
    let (vertices, indices, nodes) = cubara_render::ARENA_CAPACITY;
    let used_vertices: usize = meshed.iter().map(|n| n.mesh.vertices.len()).sum();
    let used_indices: usize = meshed.iter().map(|n| n.mesh.indices.len()).sum();
    assert!(
        used_vertices <= vertices as usize
            && used_indices <= indices as usize
            && meshed.len() <= nodes as usize,
        "{} {rep:?} does not fit the arena (vertices {used_vertices}/{vertices}, indices \
         {used_indices}/{indices}, nodes {}/{nodes}): shrink the scene's box",
        scene.name,
        meshed.len(),
    );
}

/// Write a frame as a PNG.
pub fn save_png(frame: &Frame, path: &Path) {
    image::save_buffer(
        path,
        &frame.pixels,
        frame.width,
        frame.height,
        image::ExtendedColorType::Rgba8,
    )
    .expect("write png");
}

// ---------------------------------------------------------------------------
// Metrics: pure functions over RGBA frames
// ---------------------------------------------------------------------------

fn px(frame: &Frame, x: u32, y: u32) -> &[u8] {
    let i = ((y * frame.width + x) * 4) as usize;
    &frame.pixels[i..i + 4]
}

fn is_land(frame: &Frame, x: u32, y: u32) -> bool {
    !shows_nothing(px(frame, x, y))
}

/// Which pixels `image` draws differently from the `backdrop` -- the box's
/// content as seen in `image` (including box content that is *sky* where the
/// backdrop has land: it differs too). Exact: both are the same renderer on
/// the same far geometry, so outside the box they are bit-identical.
pub fn box_mask(image: &Frame, backdrop: &Frame) -> Vec<bool> {
    assert_eq!(
        (image.width, image.height),
        (backdrop.width, backdrop.height)
    );
    image
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .zip(backdrop.pixels.as_chunks::<4>().0)
        .map(|(a, b)| a != b)
        .collect()
}

/// `a | b`, per pixel.
pub fn union(a: &[bool], b: &[bool]) -> Vec<bool> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(a, b)| *a || *b).collect()
}

/// The `permille`/1000 percentile (nearest rank) of `values`; 0 if empty.
pub fn percentile(mut values: Vec<f32>, permille: usize) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let rank = (values.len() * permille)
        .div_ceil(1000)
        .clamp(1, values.len());
    values[rank - 1]
}

/// The topmost land row of column `x`; `height` if the column is all sky.
fn top_land_row(frame: &Frame, x: u32) -> u32 {
    (0..frame.height)
        .find(|&y| is_land(frame, x, y))
        .unwrap_or(frame.height)
}

/// Sky/land boundary difference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SilhouetteDiff {
    /// Columns counted: those whose silhouette is the box's in either image.
    pub columns: usize,
    pub max_px: f32,
    pub p99_px: f32,
}

/// The sky/land boundary per pixel column of `image` against `reference`, in
/// pixels. A column counts only if the topmost land pixel of either image is
/// box content (`mask`): elsewhere the silhouette is the identical far
/// backdrop and would only dilute the number.
pub fn silhouette_diff(image: &Frame, reference: &Frame, mask: &[bool]) -> SilhouetteDiff {
    let w = image.width;
    let in_mask = |x: u32, y: u32| y < image.height && mask[(y * w + x) as usize];
    let mut diffs = Vec::new();
    for x in 0..w {
        let (a, b) = (top_land_row(image, x), top_land_row(reference, x));
        if in_mask(x, a) || in_mask(x, b) {
            diffs.push(a.abs_diff(b) as f32);
        }
    }
    SilhouetteDiff {
        columns: diffs.len(),
        max_px: diffs.iter().copied().fold(0.0, f32::max),
        p99_px: percentile(diffs, 990),
    }
}

/// The screen tile for a level, in pixels. **Chosen, not derived**: the cell
/// of level `L` is `2^L` blocks wide; a tile covers at least that many pixels
/// at `subject_distance` (one block is `height / (2 tan(fov/2) d)` pixels
/// there), rounded up to a multiple of 4, and never under 16. The far terrain
/// gets 32: its quads are at most 16 px wide at High (`far_streaming`), so a
/// tile holds two of them.
pub fn tile_px(rep: Rep, subject_distance: f32, height: u32) -> u32 {
    tile_px_with_floor(rep, subject_distance, height, 16)
}

/// [`tile_px`] with the floor given: the approach sequence's far eyes see a
/// block under a pixel, and a tile of 8 is what leaves it a few dozen tiles.
pub fn tile_px_with_floor(rep: Rep, subject_distance: f32, height: u32, min: u32) -> u32 {
    let min = min.max(1);
    match rep {
        Rep::Far => min.max(32),
        Rep::Level(level) => {
            let px_per_block = height as f32 / (2.0 * (FOV_Y / 2.0).tan() * subject_distance);
            let cell = ((1u32 << level) as f32 * px_per_block).ceil() as u32;
            cell.next_multiple_of(4).max(min)
        }
    }
}

/// Colour-mix difference over screen tiles.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileDiff {
    pub tile_px: u32,
    /// Tiles counted.
    pub tiles: usize,
    /// Difference of the tile's mean colour, in /255 (the largest channel).
    pub p50: f32,
    pub p99: f32,
    pub max: f32,
}

/// The mean colour of each `tile`x`tile` tile, `image` against `reference`.
///
/// A tile counts only if **every** pixel is land in both images (so a
/// silhouette shift is charged to [`silhouette_diff`], not here) and at least
/// half of its pixels are box content in `mask`. The mean is of the sRGB bytes
/// as stored, per channel; the tile's difference is its largest channel's
/// `|mean_a - mean_b| / 255`.
pub fn tile_diff(image: &Frame, reference: &Frame, mask: &[bool], tile: u32) -> TileDiff {
    let (w, h) = (image.width, image.height);
    let mut diffs = Vec::new();
    for ty in 0..h / tile {
        for tx in 0..w / tile {
            let mut sum_a = [0u64; 3];
            let mut sum_b = [0u64; 3];
            let (mut land_in_both, mut masked) = (true, 0u32);
            'tile: for y in ty * tile..(ty + 1) * tile {
                for x in tx * tile..(tx + 1) * tile {
                    if !is_land(image, x, y) || !is_land(reference, x, y) {
                        land_in_both = false;
                        break 'tile;
                    }
                    masked += u32::from(mask[(y * w + x) as usize]);
                    let (a, b) = (px(image, x, y), px(reference, x, y));
                    for c in 0..3 {
                        sum_a[c] += u64::from(a[c]);
                        sum_b[c] += u64::from(b[c]);
                    }
                }
            }
            if !land_in_both || masked * 2 < tile * tile {
                continue;
            }
            let n = f64::from(tile * tile);
            let d = (0..3)
                .map(|c| ((sum_a[c] as f64 - sum_b[c] as f64) / n).abs() / 255.0)
                .fold(0.0, f64::max);
            diffs.push(d as f32);
        }
    }
    TileDiff {
        tile_px: tile,
        tiles: diffs.len(),
        p50: percentile(diffs.clone(), 500),
        p99: percentile(diffs.clone(), 990),
        max: diffs.iter().copied().fold(0.0, f32::max),
    }
}

/// The three ground materials as the light-averaged colour of their textures,
/// and the inverse that unmixes a pixel into them.
///
/// **Why unmixing, not a hue test.** Beyond a few blocks a pixel covers many
/// texels, and the GPU averages them in linear light (an sRGB texture, trilinear
/// mips): a pixel over grass and soil stripes is an olive grey that a hue test
/// reads as stone. The first version of this probe classified by hue and
/// reported 49% stone on a slope that is nearly all grass and soil. So a pixel
/// is modelled as what it is: `pixel = s * (f_stone * stone + f_grass * grass +
/// f_soil * soil)` in linear light, where `s` is the lighting (the sun, the
/// ambient and the AO scale all three alike) and `f` the area fractions. Three
/// colour channels, three unknowns: `pixel = a_stone * stone + a_grass * grass
/// + a_soil * soil` is a 3x3 solve, and `f_i = a_i / sum(a)`.
///
/// **This stands in for a material id.** The renderer has no id buffer, and
/// adding one would be a second render pass (Rule 5). Its error is measured in
/// `unmixing_recovers_known_mixtures`, and the share of pixels that do not fit
/// the model at all (fog tint, ore, anything else) is reported as `unmodelled`.
#[derive(Clone, Copy, Debug)]
pub struct Endmembers {
    /// Inverse of the matrix whose columns are the stone, grass and soil
    /// textures' mean colours in linear light.
    inverse: [[f32; 3]; 3],
}

fn srgb_to_linear(byte: u8) -> f32 {
    let c = f32::from(byte) / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// The inverse of a 3x3 matrix, or `None` if it is singular.
fn invert(m: [[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let c = |r: usize, k: usize| {
        let (r1, r2) = ((r + 1) % 3, (r + 2) % 3);
        let (k1, k2) = ((k + 1) % 3, (k + 2) % 3);
        m[r1][k1] * m[r2][k2] - m[r1][k2] * m[r2][k1]
    };
    let det = m[0][0] * c(0, 0) + m[0][1] * c(0, 1) + m[0][2] * c(0, 2);
    if det.abs() < 1e-9 {
        return None;
    }
    // inverse = adjugate / det; adjugate = transpose of the cofactor matrix.
    Some(std::array::from_fn(|i| {
        std::array::from_fn(|j| c(j, i) / det)
    }))
}

impl Endmembers {
    /// From three linear-light colours: stone, grass, soil.
    pub fn from_colours(stone: [f32; 3], grass: [f32; 3], soil: [f32; 3]) -> Option<Self> {
        // Columns are the endmembers.
        let m = [
            [stone[0], grass[0], soil[0]],
            [stone[1], grass[1], soil[1]],
            [stone[2], grass[2], soil[2]],
        ];
        Some(Self {
            inverse: invert(m)?,
        })
    }

    /// From the block textures (`assets/textures`): `stone`, `grass_top`, `soil`.
    pub fn from_textures() -> Self {
        let mean = |name: &str| {
            cubara_render::materials::mean_color(name).unwrap_or_else(|| panic!("texture {name}"))
        };
        Self::from_colours(mean("stone"), mean("grass_top"), mean("soil"))
            .expect("the three ground textures are independent colours")
    }

    /// The `(stone, grass, soil)` fractions of an sRGB pixel, summing to 1, and
    /// whether the pixel fits the model (all three amounts non-negative within
    /// 15% of the total). `None` for a pixel too dark to say anything.
    pub fn unmix(&self, rgb: &[u8]) -> Option<([f32; 3], bool)> {
        let lin = [
            srgb_to_linear(rgb[0]),
            srgb_to_linear(rgb[1]),
            srgb_to_linear(rgb[2]),
        ];
        let a: [f32; 3] = std::array::from_fn(|i| {
            self.inverse[i][0] * lin[0] + self.inverse[i][1] * lin[1] + self.inverse[i][2] * lin[2]
        });
        let total: f32 = a.iter().sum();
        if total < 0.01 {
            return None;
        }
        let fits = a.iter().all(|v| *v >= -0.15 * total);
        let clamped = a.map(|v| v.max(0.0));
        let sum: f32 = clamped.iter().sum();
        Some((clamped.map(|v| v / sum), fits))
    }
}

/// The share of each material among a frame's land pixels inside a mask.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Share {
    pub land: usize,
    pub stone: f32,
    pub grass: f32,
    pub soil: f32,
    /// The share of land pixels the three-material model does not fit.
    pub unmodelled: f32,
}

/// Stone, grass and soil as shares of the land pixels of `image` that are box
/// content in `mask`: the mean of every pixel's unmixed fractions.
pub fn material_share(image: &Frame, mask: &[bool], ends: &Endmembers) -> Share {
    let (mut sum, mut land, mut unfit) = ([0.0f64; 3], 0usize, 0usize);
    for (i, p) in image.pixels.as_chunks::<4>().0.iter().enumerate() {
        if !mask[i] || shows_nothing(&p[..]) {
            continue;
        }
        let Some((f, fits)) = ends.unmix(&p[..]) else {
            continue;
        };
        land += 1;
        unfit += usize::from(!fits);
        for k in 0..3 {
            sum[k] += f64::from(f[k]);
        }
    }
    let mean = |v: f64| {
        if land == 0 {
            0.0
        } else {
            (v / land as f64) as f32
        }
    };
    Share {
        land,
        stone: mean(sum[0]),
        grass: mean(sum[1]),
        soil: mean(sum[2]),
        unmodelled: if land == 0 {
            0.0
        } else {
            unfit as f32 / land as f32
        },
    }
}

/// Everything measured for one representation against level 0.
#[derive(Clone, Copy, Debug)]
pub struct Metrics {
    pub rep: Rep,
    pub silhouette: SilhouetteDiff,
    pub tiles: TileDiff,
    pub share: Share,
    /// Stone share minus level 0's, in percentage points.
    pub stone_delta_pp: f32,
}

/// Compare `image` (drawn with `rep`) with the level-0 `reference`.
pub fn measure(
    rep: Rep,
    image: &Frame,
    reference: &Frame,
    backdrop: &Frame,
    subject_distance: f32,
    ends: &Endmembers,
) -> Metrics {
    let mask_image = box_mask(image, backdrop);
    let mask_ref = box_mask(reference, backdrop);
    let both = union(&mask_image, &mask_ref);
    let share = material_share(image, &mask_image, ends);
    let ref_share = material_share(reference, &mask_ref, ends);
    Metrics {
        rep,
        silhouette: silhouette_diff(image, reference, &both),
        tiles: tile_diff(
            image,
            reference,
            &both,
            tile_px(rep, subject_distance, image.height),
        ),
        share,
        stone_delta_pp: 100.0 * (share.stone - ref_share.stone),
    }
}

/// The targets the consistency work is held to. **Chosen, not derived**: set
/// from what was measured on the first run (see the PR), as the loosest values
/// that still make today's inconsistency fail, and to be tightened by the fix
/// PRs as they land. Level 0 against itself is exactly zero on every metric.
#[derive(Clone, Copy, Debug)]
pub struct Targets {
    /// Silhouette difference, in pixels, max.
    pub silhouette_max_px: f32,
    /// Silhouette difference, in pixels, p99.
    pub silhouette_p99_px: f32,
    /// Colour-mix difference per tile, in /255: p50, p99, max.
    pub tile_p50: f32,
    pub tile_p99: f32,
    pub tile_max: f32,
    /// Stone share difference from level 0, in percentage points (absolute).
    pub stone_delta_pp: f32,
}

impl Targets {
    /// The violations of `self` by `m`, one line each; empty when it passes.
    pub fn violations(&self, m: &Metrics) -> Vec<String> {
        let mut out = Vec::new();
        let mut check = |what: &str, got: f32, limit: f32| {
            if got > limit {
                out.push(format!("{}: {what} {got:.3} > {limit:.3}", m.rep.name()));
            }
        };
        check(
            "silhouette max px",
            m.silhouette.max_px,
            self.silhouette_max_px,
        );
        check(
            "silhouette p99 px",
            m.silhouette.p99_px,
            self.silhouette_p99_px,
        );
        check("tile p50 /255", m.tiles.p50, self.tile_p50);
        check("tile p99 /255", m.tiles.p99, self.tile_p99);
        check("tile max /255", m.tiles.max, self.tile_max);
        check(
            "stone share delta pp",
            m.stone_delta_pp.abs(),
            self.stone_delta_pp,
        );
        out
    }
}

/// Where the probe writes its images: `CUBARA_PROBE_OUT`, else a scratch
/// directory under the system temp dir. Never `saves/`.
pub fn output_dir() -> PathBuf {
    let dir = std::env::var_os("CUBARA_PROBE_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("cubara-consistency-probe"));
    std::fs::create_dir_all(&dir).expect("create the probe output directory");
    dir
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod approach;

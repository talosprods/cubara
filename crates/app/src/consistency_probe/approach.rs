//! The approach sequence: does a mountain look the same from afar as up close?
//!
//! Comparing images taken at different distances directly is confounded by
//! parallax, so the ground truth is **"every block rendered" at the same
//! eye** (the owner's rule for the far view, `ROADMAP.md`). One eye per band
//! where a representation is normally used, on a straight, level approach
//! toward the mountain's peak. At each eye the probe renders
//!
//! * (a) the normal selection, exactly what a player sees there
//!   (`desired_nodes_3d` at the game's `DEFAULT_RING_SCHEDULE`, the far
//!   terrain beyond its hole), and
//! * (b) the truth: the finest level that fits the arena forced over the
//!   mountain's footprint, with the far terrain around it.
//!
//! and compares them inside the footprint's pixels (the pixels where (b)
//! differs from the far-only backdrop), with the metrics of the parent module.
//! The bands are derived from the schedule, never written down.

use cubara_voxel::ChunkCoord;
use cubara_world::far;
use cubara_world::node::{covered_box_3d, desired_nodes_3d, DEFAULT_RING_SCHEDULE};

use super::*;
use crate::far_streaming::hole_in_blocks;
use crate::streaming::VERTICAL_LOD_SQUASH;

/// Where one representation is normally used, in blocks from the eye
/// (horizontally, Chebyshev as the schedule measures).
#[derive(Clone, Debug, PartialEq)]
pub struct Band {
    pub name: String,
    pub from: f64,
    pub to: f64,
}

/// The bands of a ring schedule, nearest first: one per level, from the
/// previous ring's radius to its own (chunks times 16), then the far terrain
/// from the last ring's edge outward (`to` is twice that edge: the far
/// terrain's own bound is the 262 km radius, far beyond anything probed).
pub fn bands(schedule: &[(u32, i32)]) -> Vec<Band> {
    let mut out = Vec::new();
    let mut from = 0.0;
    for &(level, radius) in schedule {
        let to = f64::from(radius) * 16.0;
        out.push(Band {
            name: format!("level{level}"),
            from,
            to,
        });
        from = to;
    }
    out.push(Band {
        name: "far".to_string(),
        from,
        to: from * 2.0,
    });
    out
}

/// One eye on the approach.
#[derive(Clone, Debug)]
pub struct Step {
    pub band: String,
    /// Horizontal distance from the eye to the peak, in blocks.
    pub distance: f64,
    pub scene: Scene,
}

/// The distances probed: the middle of each voxel band, then 1,200 blocks
/// (the first far eye: the last ring's edge plus 176) and twice the edge
/// (about 2 km). The far eyes are the lead's numbers ("~1.2 km and ~2 km"); the
/// voxel ones follow from the schedule.
pub fn distances(schedule: &[(u32, i32)]) -> Vec<(String, f64)> {
    let all = bands(schedule);
    let (far_band, voxel) = all.split_last().expect("a far band");
    let mut out: Vec<(String, f64)> = voxel
        .iter()
        .map(|b| (b.name.clone(), 0.5 * (b.from + b.to)))
        .collect();
    out.push(("far".to_string(), far_band.from + 176.0));
    out.push(("far".to_string(), far_band.to));
    out
}

/// The mountain's footprint and the eyes approaching it, level at one height.
pub struct Approach {
    pub peak: [i32; 3],
    pub steps: Vec<Step>,
}

/// How far above and below the level of flight the approach may pitch to keep
/// the peak in view, in degrees (chosen).
const MAX_PITCH_DEG: f32 = 25.0;
/// The footprint box reaches this far from the peak, in blocks (chosen; the
/// box is rounded out to the level-3 grid).
const FOOTPRINT_REACH: i32 = 160;

/// The approach toward the peak [`mountain_scene`] looks at: the same side,
/// the same axis, the eye at a fixed height (the one [`mountain_scene`] uses,
/// or the highest ground under any eye if that is higher), pitched toward the
/// summit by at most [`MAX_PITCH_DEG`].
pub fn approach(gen: &WorldGen, schedule: &[(u32, i32)]) -> Approach {
    let (h, px, pz) = tallest_surface(gen);
    let facing = facing_spawn(px, pz);
    let stops = distances(schedule);
    let eyes_xz: Vec<[f64; 2]> = stops
        .iter()
        .map(|(_, d)| {
            [
                f64::from(px) + f64::from(facing[0]) * d,
                f64::from(pz) + f64::from(facing[1]) * d,
            ]
        })
        .collect();
    let y = eyes_xz
        .iter()
        .map(|e| eye_above_ground(gen, e[0], e[1], 0.0)[1])
        .fold(f64::from(h - MOUNTAIN_EYE_BELOW), f64::max);
    let columns = [[px, pz]];
    let (lo, hi) = height_range(gen, &columns, FOOTPRINT_REACH);
    let (box_min, box_ext) =
        aligned_box(&columns, FOOTPRINT_REACH, (lo - 16).max(0), hi + TREE_ROOM);
    let aim_y = f64::from(h - MOUNTAIN_AIM_BELOW);
    let steps = stops
        .iter()
        .zip(&eyes_xz)
        .map(|((band, d), e)| {
            let pitch = ((aim_y - y).atan2(*d) as f32)
                .clamp(-MAX_PITCH_DEG.to_radians(), MAX_PITCH_DEG.to_radians());
            let look = Vec3::new(
                -facing[0] as f32 * pitch.cos(),
                pitch.sin(),
                -facing[1] as f32 * pitch.cos(),
            );
            Step {
                band: band.clone(),
                distance: *d,
                scene: Scene {
                    name: "approach",
                    eye: [e[0], y, e[1]],
                    look,
                    box_min,
                    box_ext,
                    subject_distance: *d as f32,
                },
            }
        })
        .collect();
    Approach {
        peak: [px, h, pz],
        steps,
    }
}

/// Whether `meshed` fits the arena (see [`check_fits_the_arena`]).
fn fits_the_arena(meshed: &[MeshedNode]) -> bool {
    let (vertices, indices, nodes) = cubara_render::ARENA_CAPACITY;
    meshed.iter().map(|n| n.mesh.vertices.len()).sum::<usize>() <= vertices as usize
        && meshed.iter().map(|n| n.mesh.indices.len()).sum::<usize>() <= indices as usize
        && meshed.len() <= nodes as usize
}

/// The finest level whose nodes over the footprint fit the arena: level 0 if it
/// can be forced, else the next finer that can (the truth is then "every
/// block" only as finely as that, and the table says so).
pub fn truth_level(setup: &Setup, footprint: &Scene) -> u32 {
    let registry = load_registry();
    let layers = TextureLayers::from_registry(&registry);
    let layer_of = |name: &str| layers.layer_of(name);
    for level in 0..=3 {
        let nodes = forced_nodes(level, footprint.box_min, footprint.box_ext);
        let built = mesh_nodes(&setup.world, &registry, &layer_of, nodes, setup.blocks);
        let meshed: Vec<MeshedNode> = built.into_iter().filter_map(to_meshed_node).collect();
        if fits_the_arena(&meshed) {
            return level;
        }
    }
    3
}

/// The level of the node a player at `scene.eye` sees the peak drawn by, or
/// `None` where the far terrain draws it.
fn level_at(nodes: &[cubara_world::node::NodeKey], peak: [i32; 3]) -> Option<u32> {
    let chunk = [
        peak[0].div_euclid(16),
        peak[1].div_euclid(16),
        peak[2].div_euclid(16),
    ];
    nodes
        .iter()
        .find(|n| n.pos == chunk.map(|c| c.div_euclid(1 << n.level)))
        .map(|n| n.level)
}

/// (a): the normal selection at `scene.eye`, as the game draws it. Returns the
/// frame and what draws the peak. `None` without a GPU.
pub fn render_normal(setup: &Setup, scene: &Scene, peak: [i32; 3]) -> Option<(Frame, Rep)> {
    let registry = load_registry();
    let layers = TextureLayers::from_registry(&registry);
    let layer_of = |name: &str| layers.layer_of(name);
    let centre = ChunkCoord::from_world_pos(scene.eye.map(|c| c as f32));
    let nodes = desired_nodes_3d(centre, VERTICAL_LOD_SQUASH, DEFAULT_RING_SCHEDULE);
    let rep = level_at(&nodes, peak).map_or(Rep::Far, Rep::Level);
    let built = mesh_nodes(&setup.world, &registry, &layer_of, nodes, setup.blocks);
    let meshed: Vec<MeshedNode> = built.into_iter().filter_map(to_meshed_node).collect();
    check_fits_the_arena(scene, None, &meshed);
    let hole = hole_in_blocks(covered_box_3d(centre, DEFAULT_RING_SCHEDULE));
    let view = far_view(scene.eye, SIZE.1, Some(hole), FarQuality::High).expect("High is not Off");
    let patches = far::build(&setup.gen, &view);
    draw(scene.eye, scene.look, meshed, &patches, Some(hole)).map(|f| (f, rep))
}

/// One row of the sequence's table.
pub struct Row {
    pub band: String,
    pub distance: f64,
    /// The level forced for (b).
    pub truth: u32,
    pub metrics: Metrics,
    pub stone_normal: f32,
    pub stone_truth: f32,
    pub mask_px: usize,
}

/// Both images of one step and the numbers between them.
pub struct StepResult {
    pub row: Row,
    pub normal: Frame,
    pub truth: Frame,
}

/// Render and compare one step. `None` without a GPU.
pub fn run_step(
    setup: &Setup,
    step: &Step,
    peak: [i32; 3],
    truth: u32,
    ends: &Endmembers,
) -> Option<StepResult> {
    let (normal, rep) = render_normal(setup, &step.scene, peak)?;
    let hole_only = FarSet {
        with_hole: {
            let view = far_view(
                step.scene.eye,
                SIZE.1,
                Some(step.scene.hole()),
                FarQuality::High,
            )
            .expect("High is not Off");
            far::build(&setup.gen, &view)
        },
        without: Vec::new(),
    };
    let backdrop = render(setup, &hole_only, &step.scene, None)?;
    let truth_frame = render(setup, &hole_only, &step.scene, Some(Rep::Level(truth)))?;
    let mask = box_mask(&truth_frame, &backdrop);
    let tile = tile_px_with_floor(rep, step.scene.subject_distance, SIZE.1, 8);
    let share_normal = material_share(&normal, &mask, ends);
    let share_truth = material_share(&truth_frame, &mask, ends);
    let metrics = Metrics {
        rep,
        silhouette: silhouette_diff(&normal, &truth_frame, &mask),
        tiles: tile_diff(&normal, &truth_frame, &mask, tile),
        share: share_normal,
        stone_delta_pp: 100.0 * (share_normal.stone - share_truth.stone),
    };
    Some(StepResult {
        row: Row {
            band: step.band.clone(),
            distance: step.distance,
            truth,
            metrics,
            stone_normal: share_normal.stone,
            stone_truth: share_truth.stone,
            mask_px: mask.iter().filter(|b| **b).count(),
        },
        normal,
        truth: truth_frame,
    })
}

/// The table, one row per distance.
pub fn table(rows: &[&Row]) -> String {
    let mut out = String::from(
        "| distance | band | peak drawn by | truth | tile px | sil max px | sil p99 px | tile p50 | tile p99 | tile max | tiles | stone normal | stone truth | delta pp | box px |\n\
         |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n",
    );
    for r in rows {
        let m = &r.metrics;
        out += &format!(
            "| {:.0} | {} | {} | level{} | {} | {:.0} | {:.0} | {:.3} | {:.3} | {:.3} | {} | {:.1}% | {:.1}% | {:+.1} | {} |\n",
            r.distance,
            r.band,
            m.rep.name(),
            r.truth,
            m.tiles.tile_px,
            m.silhouette.max_px,
            m.silhouette.p99_px,
            m.tiles.p50,
            m.tiles.p99,
            m.tiles.max,
            m.tiles.tiles,
            100.0 * r.stone_normal,
            100.0 * r.stone_truth,
            m.stone_delta_pp,
            r.mask_px
        );
    }
    out
}

//! The measurements `docs/PROPOSAL_LOD_TRANSITIONS.md` is decided by: what the
//! voxel rings draw at each level (M1), how big a terrain step gets at a ring
//! edge (T1, as a measurement), where the trees are (M6), and -- the newest
//! one -- a CPU reference image deciding ring 3's own treatment (M4, issue
//! #335).
//!
//! **CPU only, and it changes nothing the game draws.** It uses the real node
//! selection (`desired_nodes_3d`, at the game's `VERTICAL_LOD_SQUASH`), the real
//! meshing (`mesh_nodes`' `build_node`), the real visibility search, and the
//! renderer's own frustum and face-facing tests, over the yaw path the bench
//! walks (`bench.rs`: a full turn, pitched down by `-0.25`). Nothing here
//! needs a GPU, so it runs in `cargo test`.
//!
//! The big tables are an `#[ignore]`d test, because they mesh a whole world:
//!
//! ```text
//! cargo test --release -p cubara lod_probe -- --ignored --nocapture
//! ```
//!
//! The fast tests pin the counting on a tiny world and on hand-made nodes.
//!
//! **M4** (bottom of this file, `m4_*`) is a separate, smaller harness: it
//! never meshes anything. It walks the *height field* only --
//! `WorldGen::surface_height`/`cell_surface_height` for the exact-blocks
//! reference and candidate (i), `cubara_world::far` (the production far
//! terrain code, just with a nearer hole than the game uses today) for
//! candidate (ii) -- and shades what it finds with `cubara_render::aggregate`
//! (the same CPU reference `far.wgsl` is already checked against). See its
//! module doc below for what it does and does not model.
//!
//! **T4** (`ring_edges_match_a_finer_ring`, issue #381 as amended) runs on
//! that same harness, in `cargo test`: each ring edge against a finer-ring
//! reference, the vertical cell height a parameter so the pre-R1 state is
//! the test's own positive control. Its section in `m4` has the
//! calibration table.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use cubara_render::aggregate::{
    aggregate_colour, face_light, riser_colour, MaskingTable, Materials,
};
use cubara_render::materials::TextureLayers;
use cubara_render::{faces_facing, Aabb, CameraUniform, Frustum, Lighting};
use cubara_voxel::ChunkCoord;
use cubara_world::far::{self as far_world, FarView, Hole as FarHole, PatchHeights};
use cubara_world::mesh::{build_node, BuiltNode};
use cubara_world::node::{desired_nodes_3d, NodeKey, RingSchedule, DEFAULT_RING_SCHEDULE};
use cubara_world::{TerrainBlocks, TreeSite, World, WorldGen};

use crate::bench::{DEFAULT_SIZE, GATE_EYES};
use crate::far_streaming::{self, FarQuality};
use crate::streaming::VERTICAL_LOD_SQUASH;

/// Levels a selection can hold: `NodeKey::containing` goes to 8 elsewhere, the
/// schedules in use stop at 3. Fixed so a table row is an array slot.
const LEVELS: usize = 4;

/// How a frame is projected: what turns blocks into pixels.
#[derive(Clone, Copy, Debug)]
pub struct ViewSpec {
    /// Vertical field of view, radians.
    pub fov_y: f32,
    /// Rows of pixels.
    pub height_px: u32,
}

impl ViewSpec {
    /// The gate's: 1920 x 1080 at the renderer's 60 degrees
    /// (`CameraUniform::look_view_proj`).
    pub fn gate() -> Self {
        Self {
            fov_y: 60f32.to_radians(),
            height_px: DEFAULT_SIZE.1,
        }
    }

    /// The angle one pixel covers, radians (0.97 mrad at the gate). The small
    /// angle figure the proposal's tables use: it is exact on the view axis and
    /// within a few percent over the middle of the frame.
    pub fn pixel_angle(self) -> f32 {
        self.fov_y / self.height_px as f32
    }
}

/// **T1, as a measurement.** The on-screen height, in pixels, of a vertical
/// terrain step one **cell** of `level` tall, seen at `distance` blocks.
///
/// A coarse cell's height is [`NodeKey::cell_height_blocks`] for `level` --
/// one source, read here rather than duplicated: `v = 1` at levels 0-2 and
/// `v = 4` at level 3 (R1b/R1c, issue #327, landed in #329). Wherever the
/// terrain climbs, the riser a node draws is that tall. Just outside a ring
/// edge this is the tallest step the ring draws.
///
/// **Before R1b/R1c this doubled at every edge** (a cell was `2^level` blocks
/// tall regardless of level, so going up a level always doubled the riser:
/// 6.4 -> 12.9 px at 160 m). That was the diagnosis in
/// `docs/PROPOSAL_LOD_TRANSITIONS.md` section 1. Now that coarse cells keep
/// block-height rows through level 2, the 160 m and 288 m edges (both bridge
/// levels <= 2, so `v` is 1 on both sides) draw the *same* step on both
/// sides: [`no_ring_edge_doubles_a_step_on_screen`] asserts that, at issue
/// #327's Done-when bound (1.25x and 7 px). The 512 m edge (level 2's
/// 1-block cell meeting level 3's 4-block one) still jumps -- that is ring
/// 3's own decision (M4), measured by that test but not asserted by it.
pub fn max_step_px(view: ViewSpec, level: u32, distance: f32) -> f32 {
    let v = NodeKey::new(level, [0, 0, 0]).cell_height_blocks() as f32;
    v / distance / view.pixel_angle()
}

/// A ring edge: where the level changes, in blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RingEdge {
    pub edge_blocks: i32,
    pub inside: u32,
    pub outside: u32,
}

/// The edges between consecutive rings of `schedule`, from the radii
/// (a ring's outer radius, in chunks, is the next ring's inner one).
pub fn ring_edges(schedule: &RingSchedule) -> Vec<RingEdge> {
    schedule
        .windows(2)
        .map(|w| RingEdge {
            edge_blocks: w[0].1 * 16,
            inside: w[0].0,
            outside: w[1].0,
        })
        .collect()
}

/// A step's height just inside and just outside an edge, in pixels.
#[derive(Clone, Copy, Debug)]
pub struct EdgeStep {
    pub edge: RingEdge,
    pub inside_px: f32,
    pub outside_px: f32,
}

/// [`max_step_px`] on both sides of every edge of `schedule`.
pub fn edge_steps(view: ViewSpec, schedule: &RingSchedule) -> Vec<EdgeStep> {
    ring_edges(schedule)
        .into_iter()
        .map(|edge| {
            let d = edge.edge_blocks as f32;
            EdgeStep {
                edge,
                inside_px: max_step_px(view, edge.inside, d),
                outside_px: max_step_px(view, edge.outside, d),
            }
        })
        .collect()
}

/// The camera at yaw sample `i` of `samples` around a full turn: the bench's
/// direction (`bench.rs`, `look_view_proj` with `(cos, -0.25, sin)`).
fn frustum_at(view: ViewSpec, eye: [f32; 3], i: usize, samples: usize) -> Frustum {
    let yaw = std::f32::consts::TAU * i as f32 / samples as f32;
    let dir = glam::vec3(yaw.cos(), -0.25, yaw.sin());
    let aspect = DEFAULT_SIZE.0 as f32 / view.height_px as f32;
    Frustum::from_view_proj(CameraUniform::look_view_proj(
        aspect,
        glam::Vec3::from(eye),
        dir,
    ))
}

/// Every frustum of the yaw path.
fn yaw_path(view: ViewSpec, eye: [f32; 3], samples: usize) -> Vec<Frustum> {
    (0..samples)
        .map(|i| frustum_at(view, eye, i, samples))
        .collect()
}

/// What one frame draws at one level.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Drawn {
    pub nodes: u64,
    pub triangles: u64,
}

/// What the arena would draw for `built` from `frustum`: only nodes the
/// visibility search reached, only those in the frustum, and of those only the
/// face directions that can face the camera -- the same three tests
/// `ChunkArena::prepare` makes -- per level.
pub fn count_drawn(
    built: &[BuiltNode],
    visible: &HashSet<NodeKey>,
    frustum: &Frustum,
) -> [Drawn; LEVELS] {
    let eye = frustum.eye();
    let mut out = [Drawn::default(); LEVELS];
    for b in built {
        let Some(g) = b.geometry.as_ref() else {
            continue;
        };
        if !visible.contains(&b.node) || !frustum.intersects_aabb(&g.aabb) {
            continue;
        }
        let level = (b.node.level as usize).min(LEVELS - 1);
        let tris = if g.mesh.is_grouped_by_face() {
            // `face_indices` is in `Face::DRAW_ORDER` (#313); `faces_facing`
            // answers in `Face` order.
            let facing = faces_facing(&g.aabb, eye);
            g.mesh
                .face_indices
                .iter()
                .zip(cubara_voxel::Face::DRAW_ORDER)
                .filter(|&(_, face)| facing[face as usize])
                .map(|(&n, _)| n as u64 / 3)
                .sum()
        } else {
            g.mesh.indices.len() as u64 / 3
        };
        out[level].nodes += 1;
        out[level].triangles += tris;
    }
    out
}

/// One level's row of M1.
#[derive(Clone, Copy, Debug, Default)]
pub struct LevelRow {
    /// Nodes the selection asks for, empty ones included.
    pub desired: usize,
    /// Of those, the ones with geometry.
    pub meshed: usize,
    /// Of those, the ones the visibility search reaches.
    pub reached: usize,
    pub nodes_avg: f64,
    pub nodes_max: u64,
    pub tris_avg: f64,
    pub tris_max: u64,
}

/// M1 for one eye.
#[derive(Clone, Debug)]
pub struct EyeTable {
    pub rows: [LevelRow; LEVELS],
    /// Per-frame totals over the yaw path.
    pub total_nodes_avg: f64,
    pub total_tris_avg: f64,
    pub total_tris_max: u64,
}

/// Select, mesh and search around `eye`, then count what each yaw of the path
/// draws, per level. Meshing runs on every core; it is the same `build_node`
/// the game's workers call.
pub fn measure_eye(
    world: &World,
    blocks: TerrainBlocks,
    eye: [f32; 3],
    schedule: &RingSchedule,
    view: ViewSpec,
    yaw_samples: usize,
) -> EyeTable {
    let center = ChunkCoord::from_world_pos(eye);
    let nodes = desired_nodes_3d(center, VERTICAL_LOD_SQUASH, schedule);
    let built = mesh_parallel(world, blocks, &nodes);

    let desired: HashSet<NodeKey> = built.iter().map(|b| b.node).collect();
    let links: HashMap<_, _> = built.iter().map(|b| (b.node, b.links)).collect();
    let visible =
        cubara_world::visibility::visible_nodes(center, &desired, |n| links.get(&n).copied());

    let mut rows = [LevelRow::default(); LEVELS];
    for b in &built {
        let r = &mut rows[(b.node.level as usize).min(LEVELS - 1)];
        r.desired += 1;
        if b.geometry.is_some() {
            r.meshed += 1;
            if visible.contains(&b.node) {
                r.reached += 1;
            }
        }
    }
    let mut totals = [Drawn::default(); LEVELS];
    let (mut sum_nodes, mut sum_tris, mut max_tris) = (0u64, 0u64, 0u64);
    for f in yaw_path(view, eye, yaw_samples) {
        let drawn = count_drawn(&built, &visible, &f);
        let frame_tris: u64 = drawn.iter().map(|d| d.triangles).sum();
        sum_nodes += drawn.iter().map(|d| d.nodes).sum::<u64>();
        sum_tris += frame_tris;
        max_tris = max_tris.max(frame_tris);
        for (l, d) in drawn.iter().enumerate() {
            rows[l].nodes_max = rows[l].nodes_max.max(d.nodes);
            rows[l].tris_max = rows[l].tris_max.max(d.triangles);
            totals[l].nodes += d.nodes;
            totals[l].triangles += d.triangles;
        }
    }
    let n = yaw_samples.max(1) as f64;
    for (r, t) in rows.iter_mut().zip(totals) {
        r.nodes_avg = t.nodes as f64 / n;
        r.tris_avg = t.triangles as f64 / n;
    }
    EyeTable {
        rows,
        total_nodes_avg: sum_nodes as f64 / n,
        total_tris_avg: sum_tris as f64 / n,
        total_tris_max: max_tris,
    }
}

/// `nodes` built on every core, in `NodeKey` order.
fn mesh_parallel(world: &World, blocks: TerrainBlocks, nodes: &[NodeKey]) -> Vec<BuiltNode> {
    let registry = cubara_render::load_registry();
    let layers = TextureLayers::from_registry(&registry);
    let layer_of = |name: &str| layers.layer_of(name);
    let mut nodes = nodes.to_vec();
    nodes.sort();
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(nodes.len().max(1));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut built: Vec<BuiltNode> = std::thread::scope(|s| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&node) = nodes.get(i) else { break };
                        mine.push(build_node(world, &registry, &layer_of, node, blocks));
                    }
                    mine
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|w| w.join().expect("a meshing worker panicked"))
            .collect()
    });
    built.sort_by_key(|b| b.node);
    built
}

/// M1 as the table that goes in the PR.
pub fn format_m1(name: &str, eye: [f32; 3], t: &EyeTable) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "M1 {name} eye {eye:?}");
    let _ = writeln!(
        s,
        "| level | desired | meshed | reached | drawn nodes avg (max) | drawn tris avg (max) | share of tris |"
    );
    let _ = writeln!(s, "|---|---|---|---|---|---|---|");
    for (l, r) in t.rows.iter().enumerate() {
        if r.desired == 0 {
            continue;
        }
        let share = if t.total_tris_avg > 0.0 {
            100.0 * r.tris_avg / t.total_tris_avg
        } else {
            0.0
        };
        let _ = writeln!(
            s,
            "| {l} | {} | {} | {} | {:.0} ({}) | {:.0} ({}) | {share:.0}% |",
            r.desired, r.meshed, r.reached, r.nodes_avg, r.nodes_max, r.tris_avg, r.tris_max
        );
    }
    let _ = writeln!(
        s,
        "| all | | | | {:.0} | {:.0} ({}) | 100% |",
        t.total_nodes_avg, t.total_tris_avg, t.total_tris_max
    );
    s
}

/// One level's row of M6: the trees whose trunk stands in a node of that level.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TreeRow {
    /// Trees in the ring, all round.
    pub trees: usize,
    /// Trees in view per frame, averaged over the yaw path.
    pub in_view_avg: f64,
    /// Mean on-screen height of those, pixels (0 when none).
    pub px_mean: f64,
    pub px_min: f64,
    pub px_max: f64,
}

/// M6, from tree sites already tagged with the level of the node they stand in
/// (see [`tree_sites`]): per level, how many trees, how many are in view, and
/// how tall they are on screen. Pure -- the sites and the frustums are inputs.
pub fn tree_rows(
    sites: &[(TreeSite, u32)],
    frustums: &[Frustum],
    view: ViewSpec,
) -> [TreeRow; LEVELS] {
    let mut rows = [TreeRow::default(); LEVELS];
    let mut px_sum = [0.0f64; LEVELS];
    let mut views = [0u64; LEVELS];
    let mut min = [f64::MAX; LEVELS];
    let mut max = [0.0f64; LEVELS];
    for (_, level) in sites {
        rows[(*level as usize).min(LEVELS - 1)].trees += 1;
    }
    for f in frustums {
        let eye = f.eye();
        for (t, level) in sites {
            let l = (*level as usize).min(LEVELS - 1);
            let r = t.canopy_radius as f32;
            let aabb = Aabb::new(
                glam::vec3(t.x as f32 - r, t.base as f32, t.z as f32 - r),
                glam::vec3(
                    t.x as f32 + r + 1.0,
                    (t.base + t.height) as f32,
                    t.z as f32 + r + 1.0,
                ),
            );
            if !f.intersects_aabb(&aabb) {
                continue;
            }
            let mid = glam::vec3(
                t.x as f32 + 0.5,
                t.base as f32 + t.height as f32 * 0.5,
                t.z as f32 + 0.5,
            );
            let px = (t.height as f32 / (mid - eye).length() / view.pixel_angle()) as f64;
            views[l] += 1;
            px_sum[l] += px;
            min[l] = min[l].min(px);
            max[l] = max[l].max(px);
        }
    }
    let n = frustums.len().max(1) as f64;
    for l in 0..LEVELS {
        rows[l].in_view_avg = views[l] as f64 / n;
        if views[l] > 0 {
            rows[l].px_mean = px_sum[l] / views[l] as f64;
            rows[l].px_min = min[l];
            rows[l].px_max = max[l];
        }
    }
    rows
}

/// Every tree within the outermost ring, tagged with the level of the selected
/// node its trunk base stands in. Trees outside the selection are left out.
/// Uses [`WorldGen::tree_site`], the placement `trees_near` is built from, so
/// these are the trees generation really grows.
pub fn tree_sites(
    gen: &WorldGen,
    blocks: TerrainBlocks,
    center: ChunkCoord,
    selected: &HashSet<NodeKey>,
    schedule: &RingSchedule,
) -> Vec<(TreeSite, u32)> {
    let top = schedule.last().map_or(0, |&(level, _)| level);
    let radius = schedule.last().map_or(0, |&(_, r)| r);
    let mut out = Vec::new();
    for cz in center.z - radius..=center.z + radius {
        for cx in center.x - radius..=center.x + radius {
            let Some(site) = gen.tree_site(cx, cz, blocks) else {
                continue;
            };
            let chunk = ChunkCoord::new(cx, site.base.div_euclid(16), cz);
            // Oldest-first reading: the finest node that holds it.
            if let Some(level) =
                (0..=top).find(|&l| selected.contains(&NodeKey::containing(chunk, l)))
            {
                out.push((site, level));
            }
        }
    }
    out
}

/// M6 as the table that goes in the PR.
///
/// Two columns answer two different questions, not one: "grown there"
/// (real voxel blocks, `WorldGen::generate`/`generate_aniso` -- still only
/// level 0) and "instanced" (issue #332, R2's static models, now levels
/// 1-3). Before R2 the second column would have read "no" everywhere but
/// level 0, which is exactly the tree line at 160 m this landed to remove.
pub fn format_m6(name: &str, eye: [f32; 3], rows: &[TreeRow; LEVELS]) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "M6 {name} eye {eye:?}");
    let _ = writeln!(
        s,
        "| level | grown there today | instanced (R2) | trees | in view (avg) | on-screen height px mean (min-max) |"
    );
    let _ = writeln!(s, "|---|---|---|---|---|---|");
    for (l, r) in rows.iter().enumerate() {
        if r.trees == 0 {
            continue;
        }
        // No test asserts this table's exact text (`check-tests-can-fail.sh`
        // found the "instanced" column's cell survive a mutation that always
        // prints "no") -- it is PR-report prose, not program behaviour; the
        // thing it reports (which model a level actually draws with) is
        // `cubara_app::streaming::tree_model_for`, which does have one.
        let _ = writeln!(
            s,
            "| {l} | {} | {} | {} | {:.0} | {:.1} ({:.1}-{:.1}) |",
            if l == 0 { "yes" } else { "no" },
            if l == 0 { "no" } else { "yes" },
            r.trees,
            r.in_view_avg,
            r.px_mean,
            r.px_min,
            r.px_max
        );
    }
    s
}

/// The tables, as one string.
pub fn report(yaw_samples: usize) -> String {
    let view = ViewSpec::gate();
    let schedule = DEFAULT_RING_SCHEDULE;
    let world = World::new();
    let registry = cubara_render::load_registry();
    let blocks = TerrainBlocks::from_registry(&registry)
        .with_oak(&crate::game::load_structure_registry(), &registry)
        .with_ores(&crate::game::load_ore_registry(), &registry);
    let mut s = String::new();
    let _ = writeln!(
        s,
        "T1 (measurement) step height at each ring edge, 1080p / 60 deg"
    );
    let _ = writeln!(
        s,
        "| edge | cell height inside -> outside (blocks) | step px inside -> outside |"
    );
    let _ = writeln!(s, "|---|---|---|");
    let cell_height = |level: u32| NodeKey::new(level, [0, 0, 0]).cell_height_blocks();
    for e in edge_steps(view, schedule) {
        let _ = writeln!(
            s,
            "| {} m | {} -> {} | {:.1} -> {:.1} |",
            e.edge.edge_blocks,
            cell_height(e.edge.inside),
            cell_height(e.edge.outside),
            e.inside_px,
            e.outside_px
        );
    }
    for g in GATE_EYES {
        let t = measure_eye(&world, blocks, g.eye, schedule, view, yaw_samples);
        let _ = writeln!(s, "\n{}", format_m1(g.name, g.eye, &t));
        let center = ChunkCoord::from_world_pos(g.eye);
        let selected: HashSet<_> = desired_nodes_3d(center, VERTICAL_LOD_SQUASH, schedule)
            .into_iter()
            .collect();
        let sites = tree_sites(
            &WorldGen::new(world.seed()),
            blocks,
            center,
            &selected,
            schedule,
        );
        let rows = tree_rows(&sites, &yaw_path(view, g.eye, yaw_samples), view);
        let _ = writeln!(s, "{}", format_m6(g.name, g.eye, &rows));
    }
    s
}

/// M2 (`PROPOSAL_LOD_TRANSITIONS.md` §5): what R1's block-height cells would
/// cost in walls, from the height field alone. For every column of `level`
/// cells (2^level blocks wide) in `nodes`, and its +x and +z neighbours, the
/// cell's surface ([`WorldGen::cell_surface_height`]) quantized to `v` blocks:
/// returns the edges whose heights differ (a riser) and the risers' material
/// bands, each riser counting at most 3 (grass side, soil, stone). `v` =
/// 2^level is today's coarse cell; `v` = 1 is R1.
pub fn riser_counts(gen: &WorldGen, nodes: &[NodeKey], level: u32, v: i32) -> (u64, u64) {
    let step = 1i32 << level;
    let columns: HashSet<(i32, i32)> = nodes
        .iter()
        .filter(|n| n.level == level)
        .map(|n| (n.pos[0], n.pos[2]))
        .collect();
    let (mut edges, mut bands) = (0u64, 0u64);
    for &(nx, nz) in &columns {
        let o = NodeKey::new(level, [nx, 0, nz]).world_origin();
        let cell = |i: i32, j: i32| {
            gen.cell_surface_height(o[0] + i * step, o[2] + j * step, step)
                .div_euclid(v)
        };
        for i in 0..16 {
            for j in 0..16 {
                let here = cell(i, j);
                for there in [cell(i + 1, j), cell(i, j + 1)] {
                    let d = (here - there).unsigned_abs() as u64;
                    if d > 0 {
                        edges += 1;
                        bands += d.min(3);
                    }
                }
            }
        }
    }
    (edges, bands)
}

/// M2's table at `eye`: per level, risers and bands at `v` = today's cell
/// height, 4, 2 and 1, and the multiplier R1 (`v` = 1) puts on today's.
pub fn format_m2(name: &str, eye: [f32; 3]) -> String {
    let gen = WorldGen::new(World::new().seed());
    let nodes = desired_nodes_3d(
        ChunkCoord::from_world_pos(eye),
        VERTICAL_LOD_SQUASH,
        DEFAULT_RING_SCHEDULE,
    );
    let mut out = String::new();
    let _ = writeln!(
        out,
        "M2 {name} eye {eye:?}: walls from the height field, per level"
    );
    let _ = writeln!(
        out,
        "| level | v (blocks) | edges with a riser | riser bands | bands vs today |"
    );
    let _ = writeln!(out, "|---|---|---|---|---|");
    for level in 1..LEVELS as u32 {
        let today = 1i32 << level;
        let (_, base) = riser_counts(&gen, &nodes, level, today);
        for v in [today, 4, 2, 1]
            .into_iter()
            .filter(|&v| v <= today)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .rev()
        {
            let (e, b) = riser_counts(&gen, &nodes, level, v);
            let _ = writeln!(
                out,
                "| {level} | {v} | {e} | {b} | {:.2}x |",
                b as f64 / base.max(1) as f64
            );
        }
    }
    out
}

// ===========================================================================
// M4 (`docs/PROPOSAL_LOD_TRANSITIONS.md` §5, issue #335): deciding ring 3
// (512 m - 1,024 m) by a CPU reference image.
// ===========================================================================
//
// Three variants, differing **only** in the height (and shading) lookup
// beyond the eye's own rings 0-2, which every variant treats the same way
// today already does (`m4::coarse_top`, `v = 1`):
//
// - **Reference**: the exact block height field everywhere
//   (`WorldGen::surface_height`), with no coarsening at all -- the ground
//   truth M4's numbers are measured against.
// - **(i) ring 3 at `v`**: today's state at `v = 4`; `m4::coarse_top` also
//   takes `v = 1, 2` (the same sensitivity M2 already prints; candidate
//   (i) itself is `v = 4`).
// - **(ii) the far terrain from 512 m**: `cubara_world::far` (the same
//   `far::select`/`far::generate` the game calls beyond 1,024 m today,
//   through `far_streaming::far_view`), just with a hole that stops 512 m
//   out instead of 1,024 m. Nothing about the live game changes -- this
//   harness builds its own patches and reads them, which is what the issue's
//   "do not build (ii) itself" leaves alone.
//
// **R5 is not implemented.** The proposal's R5(a) asks for the risers'
// block-scale roughness split both ways (`a± = (E|g| ± E[g]) / 2`) *and* a
// re-derivation of the masking cases (`aggregate::MaskingTable`) against
// real worldgen staircases instead of planes -- R5's own text calls that out
// as separate work, not a shading tweak. Doing a partial version here (the
// roughness correction without re-deriving the masking) would score
// candidate (ii) against an invented model nobody has checked, which is a
// worse risk than not doing it: the issue's own fallback is "emulate C and
// say so" (Design decisions 1), so that is what this does. Candidate (ii) is
// therefore measured as **C alone** -- which, per §3 R7, already has the
// "lighter" defect R5 exists to fix, so a pass here is if anything a
// stricter test of (ii) than the real R5-corrected version would face, and a
// fail does not by itself rule out C+R5 (the decision rule's "(i) stays"
// default is the safe outcome either way).
//
// **Row and column mapping is approximate, by design and on purpose.**
// Columns span the camera's true horizontal FOV (tangent-correct, one-time
// per column). Rows use the same small-angle approximation
// [`ViewSpec::pixel_angle`] already stands for everywhere else in this file
// (`max_step_px`, T1): `row = height_px / 2 - (elevation - pitch) /
// pixel_angle`, not a full perspective divide. That is accurate to a few
// percent near the frame's middle (documented on `pixel_angle` already) --
// exactly where the 512 m edge sits (tens of rows off-centre, never near a
// corner) -- and, because the *same* formula is used for the reference and
// both variants, any residual bias is common to all three images and cancels
// out of the comparison between them, which is what the decision actually
// rests on.
//
// **The visibility search is a running-maximum sweep per azimuth column**,
// not one ray per pixel: walk outward in ground distance, and whenever a
// sample's row is smaller than every row reached so far (a higher point on
// screen), it is newly visible and paints the rows between. That is the same
// idea `aggregate::Staircase::scan` already uses to find which faces are
// visible, one level up (whole columns of a height field instead of
// sub-pixel face shares) -- and it is what lets one azimuth sweep produce a
// whole column's pixels (and the skyline, for free, as the final minimum
// row) in one pass over distance instead of one pass over distance *per
// row*.
mod m4 {
    use super::*;

    /// How far a sweep marches before giving up (open sky beyond this is
    /// read as sky in every variant alike, so it does not bias the
    /// comparison) -- a few times the 1,024 m outer ring, enough to find
    /// most skylines without the cost of the full 262 km far view.
    pub const MAX_DISTANCE: f32 = 3000.0;

    /// Rows either side of the 512 m edge's own row that the band metric
    /// looks at -- wide enough to hold the 4x4 low-pass's context and both
    /// neighbouring rings, per the gate's own `rows below horizon` table
    /// (`PROPOSAL_LOD_TRANSITIONS.md` §1.1: 10-64 rows at the gate eyes).
    pub const BAND_HALF_HEIGHT: i32 = 90;

    /// A camera for M4 and T4: a world `yaw` (0 looks down +x, M4's only
    /// yaw; T4 adds 45 degrees), a `pitch` in `frustum_at`'s own raw-vector
    /// convention (the `y` alongside `(cos, sin)` of unit horizontal
    /// magnitude, not a literal radian angle, though for a small `pitch`
    /// the two nearly coincide).
    #[derive(Clone, Copy, Debug)]
    pub struct Camera {
        pub eye: [f32; 3],
        pub pitch: f32,
        /// World yaw of the view axis, radians.
        pub yaw: f32,
        pub view: ViewSpec,
        pub aspect: f32,
    }

    impl Camera {
        pub fn gate(eye: [f32; 3], pitch: f32) -> Self {
            Self {
                eye,
                pitch,
                yaw: 0.0,
                view: ViewSpec::gate(),
                aspect: DEFAULT_SIZE.0 as f32 / DEFAULT_SIZE.1 as f32,
            }
        }

        /// The azimuth (world yaw, radians) of column `col` of `columns`,
        /// tangent-correct across the camera's true horizontal FOV, around
        /// the camera's own `yaw`.
        fn azimuth(self, columns: usize, col: usize) -> f32 {
            let tan_x = (self.view.fov_y * 0.5).tan() * self.aspect;
            let ndc = (col as f32 + 0.5) / columns as f32 * 2.0 - 1.0;
            self.yaw + (ndc * tan_x).atan()
        }

        /// Azimuth samples for this camera: one per [`COLUMN_STRIDE_PX`]
        /// pixels of the viewport's width (`aspect * height_px`), so a
        /// wider or taller viewport gets proportionally more columns --
        /// `COLUMNS` (240) at the gate's 1920 x 1080.
        pub fn columns(self) -> usize {
            ((self.aspect * self.view.height_px as f32) / COLUMN_STRIDE_PX as f32).round() as usize
        }

        /// The screen row a point at ground distance `d` (blocks, along the
        /// column's own azimuth) and world height `y` falls on -- see the
        /// module doc on why this is a small-angle approximation, not a
        /// projective divide.
        fn row_for(self, d: f32, y: f32) -> f32 {
            let elevation = (y - self.eye[1]) / d.max(0.05) - self.pitch;
            self.view.height_px as f32 / 2.0 - elevation / self.view.pixel_angle()
        }
    }

    /// `cell_surface_height`-based drawn top for a coarse ring node,
    /// generalized from [`WorldGen::generate_aniso`]'s own
    /// `block_at_on_surface_v` quantization -- **not**
    /// [`WorldGen::drawn_top`], which samples the *exact* column for R2's
    /// trees (that function's doc comment explains why trees and the node's
    /// own mesh read two different surfaces, and §3 R2's own risk note is
    /// exactly that mismatch). `step_h` is the node's horizontal cell width
    /// (`2^level`); `v` is `NodeKey::cell_height_blocks()`, left as a
    /// parameter so M2's `v` sweep (1, 2, 4) can reuse this too.
    pub fn coarse_top(gen: &WorldGen, x: i32, z: i32, step_h: i32, v: i32) -> i32 {
        let (cx, cz) = (x.div_euclid(step_h) * step_h, z.div_euclid(step_h) * step_h);
        let surface = gen.cell_surface_height(cx, cz, step_h);
        if v <= 1 {
            return surface;
        }
        let bound = surface + 1 - (v - 1) / 2;
        let row = (bound - 1).div_euclid(v);
        row * v + (v - 1)
    }

    /// Which ring schedule entry (its level and its outer edge in blocks)
    /// covers ground distance `d` from the eye -- `None` beyond every ring
    /// in `schedule` (the far terrain's own territory, in every variant).
    fn ring_level_for(schedule: &RingSchedule, d: f32) -> Option<(u32, i32)> {
        schedule
            .iter()
            .find(|&&(_, r)| d < (r * 16) as f32)
            .copied()
    }

    /// The exact per-block height field's drawn top FACE (one above the top
    /// solid block) at `(x, z)` -- M4's reference, everywhere.
    pub fn exact_face_height(gen: &WorldGen, x: i32, z: i32) -> f64 {
        gen.surface_height(x, z) as f64 + 1.0
    }

    /// The inverse of a drawn top face's `+ 1.0`: the solid block a face at
    /// `face_y` sits directly above. Named and called from both sides of a
    /// riser comparison (`sweep_column`'s `top_blocks`/`new_top`) rather
    /// than written out twice, so a wrong offset is one function to pin,
    /// not two expressions that happen to read alike.
    fn block_below_face(face_y: f64) -> i32 {
        (face_y - 1.0).round() as i32
    }

    /// The row range `[lo, hi)` a sample landing on screen row `row` fills,
    /// given everything is already painted down to `running_min` -- clamped
    /// to the screen (`height_px` rows). Pulled out of `sweep_column`'s
    /// `paint` closure so the clamp itself is directly testable: `row` is
    /// almost always positive in practice (a riser far below the horizon),
    /// so a `max`/`min` slip in the lower clamp would only show up on a
    /// sample that goes negative -- exactly the case a test has to build by
    /// hand, not one that turns up by chance.
    fn paint_range(row: f32, running_min: f32, height_px: usize) -> (usize, usize) {
        let lo = row.floor().max(0.0) as usize;
        let hi = (running_min.min(height_px as f32).ceil() as usize).min(height_px);
        (lo, hi)
    }

    /// A level's vertical cell height in blocks, as a parameter (issue
    /// #381 as amended, decision 2): the harness never forks
    /// [`NodeKey::cell_height_blocks`] behind `cfg(test)`; it takes the
    /// `v` rule as an argument and the production rule is one of the two
    /// below.
    pub type CellHeight = fn(u32) -> i32;

    /// The cell height the game meshes with -- [`NodeKey::cell_height_blocks`]
    /// read, not copied: `1` at levels 0-2, `4` at level 3 (R1b/R1c, #329).
    pub fn production_cell_height(level: u32) -> i32 {
        NodeKey::new(level, [0, 0, 0]).cell_height_blocks()
    }

    /// Option A, the state before R1 (`docs/PROPOSAL_LOD_TRANSITIONS.md`
    /// §1): a cell `2^level` blocks tall at every level, so every ring edge
    /// doubles the riser. T4 must fail this by a clear margin.
    pub fn option_a_cell_height(level: u32) -> i32 {
        1 << level
    }

    /// A ring schedule's drawn top face at `(x, z)`, from whichever entry
    /// of `schedule` covers ground distance `d` from `eye_xz`, at that
    /// level's horizontal cell width (`2^level`) and the vertical cell
    /// height `cell_height(level)`. `None` beyond every entry in `schedule`
    /// -- the caller switches to the far terrain there, or stops.
    /// `schedule` is handed in (rather than always `DEFAULT_RING_SCHEDULE`)
    /// so candidate (ii) can pass a 3-entry schedule (levels 0-2 only,
    /// ring 3 dropped) and T4 can pass its finer-ring reference schedule.
    fn ring_face_height(
        gen: &WorldGen,
        schedule: &RingSchedule,
        eye_xz: (f32, f32),
        x: i32,
        z: i32,
        cell_height: &dyn Fn(u32) -> i32,
    ) -> Option<f64> {
        let d = ((x as f32 - eye_xz.0).powi(2) + (z as f32 - eye_xz.1).powi(2)).sqrt();
        let (level, _) = ring_level_for(schedule, d)?;
        let step_h = 1i32 << level;
        Some(coarse_top(gen, x, z, step_h, cell_height(level)) as f64 + 1.0)
    }

    /// The far terrain's own footprint-mean height field
    /// (`cubara_world::far`), built once for a hole of `hole_radius_blocks`
    /// around `eye` -- the production machinery
    /// (`far_streaming::far_view` + `far::build`, `FarQuality::High`, the
    /// quality §3 R7's own measurements used), just with a nearer hole than
    /// the game draws today (1,024 m), so candidate (ii) can sample it from
    /// 512 m. Nothing about the live game changes: this is built and read
    /// only inside this harness.
    pub struct FarField {
        patches: Vec<PatchHeights>,
    }

    impl FarField {
        pub fn build(
            gen: &WorldGen,
            eye: [f64; 3],
            hole_radius_blocks: f64,
            scan_radius: f64,
        ) -> Self {
            let hole = FarHole {
                min: [
                    eye[0] - hole_radius_blocks,
                    -8192.0,
                    eye[2] - hole_radius_blocks,
                ],
                max: [
                    eye[0] + hole_radius_blocks,
                    8192.0,
                    eye[2] + hole_radius_blocks,
                ],
            };
            let base = far_streaming::far_view(eye, DEFAULT_SIZE.1, Some(hole), FarQuality::High)
                .expect("FarQuality::High always returns a view");
            let view = FarView {
                radius: scan_radius,
                ..base
            };
            Self {
                patches: far_world::build(gen, &view),
            }
        }

        /// The smooth face height at `(x, z)`, or `None` outside every
        /// patch (shouldn't happen inside `scan_radius`).
        pub fn height_at(&self, x: f64, z: f64) -> Option<f64> {
            self.patches
                .iter()
                .find(|p| {
                    let [ox, oz] = p.key.origin();
                    let s = p.key.size() as f64;
                    (ox as f64..ox as f64 + s).contains(&x)
                        && (oz as f64..oz as f64 + s).contains(&z)
                })
                .map(|p| p.height_at(x, z))
        }

        /// Central-difference gradient at `(x, z)`, one block either side.
        pub fn gradient_at(&self, x: f64, z: f64) -> Option<[f32; 2]> {
            let h = |dx: f64, dz: f64| self.height_at(x + dx, z + dz);
            let (hx0, hx1, hz0, hz1) = (h(-1.0, 0.0)?, h(1.0, 0.0)?, h(0.0, -1.0)?, h(0.0, 1.0)?);
            Some([((hx1 - hx0) / 2.0) as f32, ((hz1 - hz0) / 2.0) as f32])
        }
    }

    /// The ground truth plus the two candidates, as one closure-driving
    /// enum -- the only thing a sweep has to switch on.
    pub enum Variant<'a> {
        /// The exact block height field, blocky, with no far terrain at all
        /// (bounded by `MAX_DISTANCE` like everything else).
        Reference,
        /// Today's state (`ring3_v = 4`) or a sensitivity row (1, 2) --
        /// blocky through 1,024 m, the real far terrain (today's hole)
        /// beyond it.
        RingThree { ring3_v: i32, far: &'a FarField },
        /// Candidate (ii): blocky through 512 m (levels 0-2 only), the far
        /// terrain (hole at 512 m) beyond it.
        FarFromFiveTwelve { far: &'a FarField },
        /// T4: a ring schedule drawn at `cell_height` per level, out to the
        /// schedule's last ring and sky beyond (no far terrain -- T4
        /// compares two schedules that agree past the ring under test, and
        /// the far terrain's 1,024 m hole never reaches the band around a
        /// nearer edge).
        Rings {
            schedule: &'a RingSchedule,
            cell_height: CellHeight,
        },
    }

    impl Variant<'_> {
        fn blocky_limit(&self) -> f32 {
            match self {
                Variant::Reference => MAX_DISTANCE,
                Variant::RingThree { .. } => 1024.0,
                Variant::FarFromFiveTwelve { .. } => 512.0,
                Variant::Rings { schedule, .. } => {
                    schedule.last().map_or(0.0, |&(_, r)| (r * 16) as f32)
                }
            }
        }

        fn blocky_face_height(
            &self,
            gen: &WorldGen,
            eye_xz: (f32, f32),
            x: i32,
            z: i32,
        ) -> Option<f64> {
            match self {
                Variant::Reference => Some(exact_face_height(gen, x, z)),
                Variant::RingThree { ring3_v, .. } => {
                    let cell_height = |level: u32| if level < 3 { 1 } else { *ring3_v };
                    ring_face_height(gen, DEFAULT_RING_SCHEDULE, eye_xz, x, z, &cell_height)
                }
                Variant::FarFromFiveTwelve { .. } => {
                    ring_face_height(gen, &DEFAULT_RING_SCHEDULE[..3], eye_xz, x, z, &|_| 1)
                }
                Variant::Rings {
                    schedule,
                    cell_height,
                } => ring_face_height(gen, schedule, eye_xz, x, z, cell_height),
            }
        }

        fn far(&self) -> Option<&FarField> {
            match self {
                Variant::Reference | Variant::Rings { .. } => None,
                Variant::RingThree { far, .. } | Variant::FarFromFiveTwelve { far } => Some(far),
            }
        }
    }

    /// A grid DDA through integer `(x, z)` columns along `dir` from
    /// `eye_xz`, out to `limit` blocks -- pulled out of `sweep_column`'s
    /// blocky zone so the stepping itself (not just height-field-dependent
    /// behaviour) can be pinned without a `WorldGen`.
    struct Dda {
        cx: i32,
        cz: i32,
        step: [i32; 2],
        delta: [f32; 2],
        next: [f32; 2],
        d_enter: f32,
        limit: f32,
    }

    impl Dda {
        fn new(eye_xz: (f32, f32), dir: (f32, f32), limit: f32) -> Self {
            let cx = eye_xz.0.floor() as i32;
            let cz = eye_xz.1.floor() as i32;
            let step = [
                if dir.0 >= 0.0 { 1 } else { -1 },
                if dir.1 >= 0.0 { 1 } else { -1 },
            ];
            let delta = [
                if dir.0 == 0.0 {
                    f32::INFINITY
                } else {
                    1.0 / dir.0.abs()
                },
                if dir.1 == 0.0 {
                    f32::INFINITY
                } else {
                    1.0 / dir.1.abs()
                },
            ];
            let next = [
                if dir.0 > 0.0 {
                    (cx as f32 + 1.0 - eye_xz.0) / dir.0
                } else if dir.0 < 0.0 {
                    (eye_xz.0 - cx as f32) / -dir.0
                } else {
                    f32::INFINITY
                },
                if dir.1 > 0.0 {
                    (cz as f32 + 1.0 - eye_xz.1) / dir.1
                } else if dir.1 < 0.0 {
                    (eye_xz.1 - cz as f32) / -dir.1
                } else {
                    f32::INFINITY
                },
            ];
            Self {
                cx,
                cz,
                step,
                delta,
                next,
                d_enter: 0.0,
                limit,
            }
        }

        /// The current column, and the ground distance it was entered at.
        fn here(&self) -> (i32, i32, f32) {
            (self.cx, self.cz, self.d_enter)
        }

        /// The ground distance the current column is exited at (the
        /// nearer of the two axis boundaries, capped at `limit`).
        fn exit_distance(&self) -> f32 {
            let a = if self.next[0] < self.next[1] { 0 } else { 1 };
            self.next[a].min(self.limit)
        }

        /// Advance to the next column. `None` once `limit` is reached (the
        /// current column was the last one); otherwise the ground distance
        /// the boundary was crossed at and which axis (0 = x, 1 = z).
        fn advance(&mut self) -> Option<(f32, usize)> {
            let a = if self.next[0] < self.next[1] { 0 } else { 1 };
            if self.next[a] >= self.limit {
                return None;
            }
            let d_boundary = self.next[a];
            self.next[a] += self.delta[a];
            if a == 0 {
                self.cx += self.step[0];
            } else {
                self.cz += self.step[1];
            }
            self.d_enter = d_boundary;
            Some((d_boundary, a))
        }
    }

    /// `mesh.wgsl`'s lighting applied to a riser facing `normal`, shown
    /// `rise` blocks tall (`aggregate::riser_colour`'s own material
    /// banding -- grass side, then soil, then stone).
    fn riser_colour_lit(
        materials: &Materials,
        lighting: &Lighting,
        normal: glam::Vec3,
        rise: i32,
    ) -> glam::Vec3 {
        riser_colour(materials, rise as f32) * face_light(normal, lighting)
    }

    /// One column's sweep: the row-indexed colours it paints (`None` where
    /// nothing was visible -- open sky), and the skyline row (the smallest
    /// row reached, i.e. the screen's own top rows if nothing was ever hit).
    pub fn sweep_column(
        cam: Camera,
        azimuth: f32,
        variant: &Variant,
        gen: &WorldGen,
        materials: &Materials,
        lighting: &Lighting,
    ) -> (Vec<Option<[f32; 3]>>, f32) {
        let height_px = cam.view.height_px as usize;
        let mut rows: Vec<Option<glam::Vec3>> = vec![None; height_px];
        let mut running_min = f32::INFINITY;

        let dir = (azimuth.cos(), azimuth.sin());
        let eye_xz = (cam.eye[0], cam.eye[2]);

        let mut paint = |d: f32, y: f32, colour: glam::Vec3| {
            let row = cam.row_for(d, y);
            if row < running_min {
                let (lo, hi) = paint_range(row, running_min, height_px);
                if lo < hi {
                    for slot in &mut rows[lo..hi] {
                        *slot = Some(colour);
                    }
                }
                running_min = row;
            }
        };

        let top_colour = materials.top * face_light(glam::Vec3::Y, lighting);

        // --- The blocky zone: a column DDA along `dir`, from the eye out to
        // this variant's own `blocky_limit` (512 m, 1,024 m, or the whole
        // sweep for the reference). ---
        let blocky_limit = variant.blocky_limit().min(MAX_DISTANCE);
        {
            let mut dda = Dda::new(eye_xz, dir, blocky_limit);
            'blocky: loop {
                let (cx, cz, d_enter) = dda.here();
                let Some(face_y) = variant.blocky_face_height(gen, eye_xz, cx, cz) else {
                    break 'blocky;
                };
                let top_blocks = block_below_face(face_y);
                let d_exit = dda.exit_distance();
                paint(d_enter.max(0.05), face_y as f32, top_colour);
                paint(d_exit.max(0.05), face_y as f32, top_colour);
                let Some((d_boundary, a)) = dda.advance() else {
                    break 'blocky;
                };
                if let Some(new_face_y) = variant.blocky_face_height(gen, eye_xz, dda.cx, dda.cz) {
                    let new_top = block_below_face(new_face_y);
                    if new_top > top_blocks {
                        let mut normal = glam::Vec3::ZERO;
                        normal[a] = -(dda.step[a] as f32);
                        let colour =
                            riser_colour_lit(materials, lighting, normal, new_top - top_blocks);
                        paint(d_boundary.max(0.05), new_face_y as f32, colour);
                    }
                }
            }
        }

        // --- The smooth zone beyond it, if this variant has a far terrain
        // (every variant but the reference). ---
        if let Some(far) = variant.far() {
            let table = MaskingTable::committed();
            let eye3 = glam::Vec3::from(cam.eye);
            // `.max(1.0)`, not `.min`: a mutation that starts `d` near 0
            // instead of at `blocky_limit` has no test pinning it directly,
            // and does not need one -- `measure_eye` always builds `far`'s
            // hole at exactly `blocky_limit` (the two variants' own
            // construction couples them), so `far.height_at` returns `None`
            // for every `d` short of `blocky_limit` regardless (nothing is
            // selected inside a far-terrain hole, `far::tests`), and
            // nothing is painted there either way. Starting too early only
            // wastes iterations on a region proven empty; it cannot change
            // what the sweep shows.
            let mut d = blocky_limit.max(1.0);
            while d < MAX_DISTANCE {
                let x = eye_xz.0 as f64 + d as f64 * dir.0 as f64;
                let z = eye_xz.1 as f64 + d as f64 * dir.1 as f64;
                if let (Some(h), Some(g)) = (far.height_at(x, z), far.gradient_at(x, z)) {
                    let point = glam::vec3(x as f32, h as f32, z as f32);
                    if let Some(view_dir) = (eye3 - point).try_normalize() {
                        if let Some(colour) =
                            aggregate_colour(&table, g, view_dir, lighting, materials)
                        {
                            paint(d, h as f32, colour);
                        }
                    }
                }
                d += (d * 0.02).max(2.0);
            }
        }

        // The `else` arm (open sky, nothing ever painted) is unreachable
        // through every variant this harness builds, not untested by
        // oversight: the very first blocky column is the eye's own, which
        // `exact_face_height`/`ring_face_height` always answer for (ring 0
        // always covers distance 0) -- so `running_min` becomes finite on
        // the sweep's first sample, before this line ever runs on an
        // infinite value. It stays as a defensive fallback for a variant
        // that might one day *not* have that guarantee, not as behaviour
        // this harness exercises.
        let skyline_row = if running_min.is_finite() {
            running_min
        } else {
            height_px as f32
        };
        (
            rows.into_iter().map(|c| c.map(|v| v.to_array())).collect(),
            skyline_row,
        )
    }

    /// One eye's rendered column set: every column's full row array (for
    /// the band metric, sliced to the band afterwards) and skyline row (for
    /// the skyline error), at `columns` azimuths spanning the camera's own
    /// horizontal FOV.
    pub struct ColumnSet {
        pub rows: Vec<Vec<Option<[f32; 3]>>>,
        pub skyline_rows: Vec<f32>,
    }

    pub fn render_eye(
        cam: Camera,
        variant: &Variant,
        gen: &WorldGen,
        materials: &Materials,
        lighting: &Lighting,
        columns: usize,
    ) -> ColumnSet {
        let mut rows = Vec::with_capacity(columns);
        let mut skyline_rows = Vec::with_capacity(columns);
        for c in 0..columns {
            let az = cam.azimuth(columns, c);
            let (col_rows, sky) = sweep_column(cam, az, variant, gen, materials, lighting);
            rows.push(col_rows);
            skyline_rows.push(sky);
        }
        ColumnSet { rows, skyline_rows }
    }

    // -----------------------------------------------------------------------
    // The band metric (`PROPOSAL_LOD_TRANSITIONS.md` §6, T4's own recipe,
    // reused here for M4): linear luminance (Rec. 709 weights), a 4x4 box
    // low-pass, D(row) = mean |LP(F) - LP(R)| along a row, then the band's
    // maximum and mean of D.
    // -----------------------------------------------------------------------

    /// Linear luminance of a linear-light RGB colour (Rec. 709 weights --
    /// the file does not otherwise pick one, so this says which).
    pub fn luminance(c: [f32; 3]) -> f32 {
        0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    }

    /// A `None` pixel (sky/unpainted) reads as the sky colour's luminance --
    /// both images agree on what sky looks like, so this cannot manufacture
    /// a difference the metric would wrongly report.
    fn pixel_luminance(c: Option<[f32; 3]>, lighting: &Lighting) -> f32 {
        c.map_or_else(|| luminance(lighting.fog_color.to_array()), luminance)
    }

    /// A 4x4 box low-pass over a row-major `width` x `height` grid,
    /// clamped at the edges (the box is shrunk rather than wrapping or
    /// zero-padding, so an edge pixel is the mean of what actually exists
    /// around it).
    pub fn box_low_pass_4x4(grid: &[f32], width: usize, height: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; grid.len()];
        for y in 0..height {
            for x in 0..width {
                let y0 = y.saturating_sub(1);
                let y1 = (y + 2).min(height);
                let x0 = x.saturating_sub(1);
                let x1 = (x + 2).min(width);
                let mut sum = 0.0f32;
                let mut n = 0u32;
                for yy in y0..y1 {
                    for xx in x0..x1 {
                        sum += grid[yy * width + xx];
                        n += 1;
                    }
                }
                out[y * width + x] = sum / n.max(1) as f32;
            }
        }
        out
    }

    /// `D(row)`, the band's maximum of `D` and its mean -- `F` and `R` are
    /// row-major `width` x `height` luminance grids, already low-passed.
    #[derive(Clone, Debug, PartialEq)]
    pub struct BandMetric {
        pub per_row: Vec<f32>,
        pub max: f32,
        pub mean: f32,
    }

    pub fn band_metric(lp_f: &[f32], lp_r: &[f32], width: usize, height: usize) -> BandMetric {
        assert_eq!(lp_f.len(), width * height);
        assert_eq!(lp_r.len(), width * height);
        let mut per_row = Vec::with_capacity(height);
        for y in 0..height {
            let row = &lp_f[y * width..(y + 1) * width];
            let row_r = &lp_r[y * width..(y + 1) * width];
            let d: f32 = row
                .iter()
                .zip(row_r)
                .map(|(f, r)| (f - r).abs())
                .sum::<f32>()
                / width.max(1) as f32;
            per_row.push(d);
        }
        let max = per_row.iter().copied().fold(0.0f32, f32::max);
        let mean = per_row.iter().sum::<f32>() / per_row.len().max(1) as f32;
        BandMetric { per_row, max, mean }
    }

    /// The band metric between two [`ColumnSet`]s, over `band` rows
    /// (absolute row indices), built from the per-column row arrays
    /// directly (handles the low-pass and luminance conversion).
    pub fn band_metric_between(
        f: &ColumnSet,
        r: &ColumnSet,
        band: std::ops::Range<usize>,
        lighting: &Lighting,
    ) -> BandMetric {
        let width = f.rows.len();
        assert_eq!(width, r.rows.len());
        let height = band.len();
        let (lf, lr) = (
            luminance_grid(f, band.clone(), lighting),
            luminance_grid(r, band, lighting),
        );
        let (lp_f, lp_r) = (
            box_low_pass_4x4(&lf, width, height),
            box_low_pass_4x4(&lr, width, height),
        );
        band_metric(&lp_f, &lp_r, width, height)
    }

    /// The band's luminance as a row-major `columns` x `band.len()` grid.
    /// The natural storage is column-major (one `Vec` per azimuth), so
    /// this transposes on the way in.
    fn luminance_grid(
        set: &ColumnSet,
        band: std::ops::Range<usize>,
        lighting: &Lighting,
    ) -> Vec<f32> {
        let width = set.rows.len();
        let height = band.len();
        let mut g = vec![0.0f32; width * height];
        for (c, col) in set.rows.iter().enumerate() {
            for (i, row) in band.clone().enumerate() {
                g[i * width + c] = pixel_luminance(col.get(row).copied().flatten(), lighting);
            }
        }
        g
    }

    /// T4's high-frequency energy (`PROPOSAL_LOD_TRANSITIONS.md` §6): the
    /// band's mean `|image - LP(image)|`, with the same 4x4 box low-pass
    /// the band metric uses. The real-over-reference ratio of this is what
    /// catches a pattern- or texture-scale step: a ring that draws its
    /// risers twice as tall but half as often moves little in `D` once
    /// low-passed, and a lot here.
    pub fn high_frequency_energy(
        set: &ColumnSet,
        band: std::ops::Range<usize>,
        lighting: &Lighting,
    ) -> f32 {
        let width = set.rows.len();
        let height = band.len();
        let grid = luminance_grid(set, band, lighting);
        let lp = box_low_pass_4x4(&grid, width, height);
        grid.iter()
            .zip(&lp)
            .map(|(g, l)| (g - l).abs())
            .sum::<f32>()
            / grid.len().max(1) as f32
    }

    // -----------------------------------------------------------------------
    // Skyline error per azimuth: pixels between a variant's skyline row and
    // the reference's, at each azimuth column.
    // -----------------------------------------------------------------------

    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct SkylineError {
        pub mean: f32,
        pub p75: f32,
        pub max: f32,
    }

    pub fn skyline_error(variant: &ColumnSet, reference: &ColumnSet) -> SkylineError {
        assert_eq!(variant.skyline_rows.len(), reference.skyline_rows.len());
        let mut errs: Vec<f32> = variant
            .skyline_rows
            .iter()
            .zip(&reference.skyline_rows)
            .map(|(v, r)| (v - r).abs())
            .collect();
        errs.sort_by(f32::total_cmp);
        let n = errs.len().max(1);
        let mean = errs.iter().sum::<f32>() / n as f32;
        let p75 = errs[(errs.len() * 3 / 4).min(errs.len().saturating_sub(1))];
        let max = errs.last().copied().unwrap_or(0.0);
        SkylineError { mean, p75, max }
    }

    // -----------------------------------------------------------------------
    // The decision rule (issue #335, Design decisions 2): (ii) wins only if
    // its band metric is within a threshold of (i)'s **and** its skyline
    // error is no worse. The threshold is the proposal's §6 calibration --
    // "about 2.5x the reference-vs-reference noise" -- measured here as the
    // band metric between the reference rendered at the eye and the
    // reference rendered from an eye shifted by half a block (the CPU has no
    // cross-backend noise to measure against, so this stands in for it: the
    // same scene, a sub-block camera jitter, nothing about the decision).
    pub const CALIBRATION_FACTOR: f32 = 2.5;

    #[derive(Clone, Debug, PartialEq)]
    pub struct Decision {
        pub band_i: BandMetric,
        pub band_ii: BandMetric,
        pub skyline_i: SkylineError,
        pub skyline_ii: SkylineError,
        pub noise: BandMetric,
        /// `2.5x` the noise's own max and mean -- two thresholds, not one,
        /// so the band metric's `max` is judged against a max-shaped noise
        /// figure and its `mean` against a mean-shaped one, rather than
        /// blending the two into a single number.
        pub threshold_max: f32,
        pub threshold_mean: f32,
        pub ii_wins: bool,
    }

    pub fn decide(
        band_i: BandMetric,
        band_ii: BandMetric,
        skyline_i: SkylineError,
        skyline_ii: SkylineError,
        noise: BandMetric,
    ) -> Decision {
        let threshold_max = CALIBRATION_FACTOR * noise.max;
        let threshold_mean = CALIBRATION_FACTOR * noise.mean;
        let band_ok = band_ii.max <= band_i.max + threshold_max
            && band_ii.mean <= band_i.mean + threshold_mean;
        let skyline_ok = skyline_ii.mean <= skyline_i.mean && skyline_ii.max <= skyline_i.max;
        Decision {
            band_i,
            band_ii,
            skyline_i,
            skyline_ii,
            noise,
            threshold_max,
            threshold_mean,
            ii_wins: band_ok && skyline_ok,
        }
    }

    // -----------------------------------------------------------------------
    // The eyes, the report, and the `#[ignore]`d test that prints it.
    // -----------------------------------------------------------------------

    /// The gate eyes' own pitch for M4: level (0), looking straight down +x
    /// -- the 512 m edge then sits close to the frame's own middle rows
    /// (§1.1's "rows below horizon" table is small at both gate eyes), which
    /// is where the row approximation documented on [`Camera`] is most
    /// accurate.
    pub const GATE_PITCH: f32 = 0.0;

    /// `docs/PROPOSAL_FAR_VIEW.md`'s F6 names no eye of its own -- it is
    /// "whatever F1-F5's measurements say is still missing" -- and neither
    /// it nor `BENCHMARKS.md` defines a numeric "low eye" anywhere in the
    /// repository. Issue #335's own fallback is used here instead, and
    /// recorded as exactly that: `[8, 60, 8]`, looking along +x pitched
    /// `-0.15`.
    pub const LOW_EYE: [f32; 3] = [8.0, 60.0, 8.0];
    pub const LOW_EYE_PITCH: f32 = -0.15;

    /// Columns for the report -- "a few hundred" per the issue.
    pub const COLUMNS: usize = 240;

    /// One azimuth sample per this many pixels of viewport width
    /// ([`Camera::columns`]): `1920 / 8 = COLUMNS` at the gate, so T4 and
    /// M4 sample the same way there, and a viewport of another width gets
    /// its own count rather than the gate's.
    pub const COLUMN_STRIDE_PX: usize = 8;

    /// The row band around the 512 m edge, from the *reference's* own exact
    /// height there (so the window is centred where the edge actually falls
    /// in this scene, not where it would on perfectly flat ground).
    fn band_window(cam: Camera, gen: &WorldGen, eye_xz: (f32, f32)) -> std::ops::Range<usize> {
        let x = (eye_xz.0 + 512.0).round() as i32;
        let z = eye_xz.1.round() as i32;
        let y = exact_face_height(gen, x, z) as f32;
        let row = cam.row_for(512.0, y);
        band_rows(row, BAND_HALF_HEIGHT as f32, cam.view.height_px as usize)
    }

    /// The rows `half` either side of `row`, clamped to the screen's
    /// `height_px` rows and never empty.
    fn band_rows(row: f32, half: f32, height_px: usize) -> std::ops::Range<usize> {
        let lo = (row - half).max(0.0) as usize;
        let hi = ((row + half).max(0.0) as usize + 1).min(height_px);
        lo.min(height_px.saturating_sub(1))..hi.max(lo + 1).min(height_px)
    }

    /// Everything measured at one eye: the three column sets, the band
    /// metric and skyline error for each candidate against the reference,
    /// the noise calibration, and the decision.
    pub struct EyeReport {
        pub name: &'static str,
        pub eye: [f32; 3],
        pub band_i: BandMetric,
        pub band_ii: BandMetric,
        pub skyline_i: SkylineError,
        pub skyline_ii: SkylineError,
        pub decision: Decision,
    }

    /// Measure one eye: build the reference, candidate (i) at `ring3_v`, and
    /// candidate (ii), then the band metric, skyline error and decision.
    pub fn measure_eye(
        gen: &WorldGen,
        name: &'static str,
        eye: [f32; 3],
        pitch: f32,
        ring3_v: i32,
    ) -> EyeReport {
        let cam = Camera::gate(eye, pitch);
        let materials = far_streaming::far_materials();
        let lighting = Lighting::default();
        let eye_xz = (eye[0], eye[2]);
        let eye64 = [eye[0] as f64, eye[1] as f64, eye[2] as f64];

        let reference = render_eye(
            cam,
            &Variant::Reference,
            gen,
            &materials,
            &lighting,
            COLUMNS,
        );

        let far_i = FarField::build(gen, eye64, 1024.0, MAX_DISTANCE as f64 + 64.0);
        let variant_i = Variant::RingThree {
            ring3_v,
            far: &far_i,
        };
        let set_i = render_eye(cam, &variant_i, gen, &materials, &lighting, COLUMNS);

        let far_ii = FarField::build(gen, eye64, 512.0, MAX_DISTANCE as f64 + 64.0);
        let variant_ii = Variant::FarFromFiveTwelve { far: &far_ii };
        let set_ii = render_eye(cam, &variant_ii, gen, &materials, &lighting, COLUMNS);

        let band = band_window(cam, gen, eye_xz);
        let band_i = band_metric_between(&set_i, &reference, band.clone(), &lighting);
        let band_ii = band_metric_between(&set_ii, &reference, band.clone(), &lighting);
        let skyline_i = skyline_error(&set_i, &reference);
        let skyline_ii = skyline_error(&set_ii, &reference);

        // The calibration noise: the same reference, rendered again from an
        // eye jittered half a block sideways -- see `decide`'s doc comment.
        let jittered_cam = Camera::gate([eye[0], eye[1], eye[2] + 0.5], pitch);
        let jittered = render_eye(
            jittered_cam,
            &Variant::Reference,
            gen,
            &materials,
            &lighting,
            COLUMNS,
        );
        let noise = band_metric_between(&jittered, &reference, band, &lighting);

        let decision = decide(
            band_i.clone(),
            band_ii.clone(),
            skyline_i,
            skyline_ii,
            noise,
        );
        EyeReport {
            name,
            eye,
            band_i,
            band_ii,
            skyline_i,
            skyline_ii,
            decision,
        }
    }

    /// One eye's row, for the table [`report`] prints.
    fn format_eye(r: &EyeReport) -> String {
        format!(
            "| {} {:?} | {:.4} / {:.4} | {:.4} / {:.4} | {:.2} / {:.2} / {:.2} | {:.2} / {:.2} / {:.2} | {:.4} / {:.4} | {} |",
            r.name,
            r.eye,
            r.band_i.max,
            r.band_i.mean,
            r.band_ii.max,
            r.band_ii.mean,
            r.skyline_i.mean,
            r.skyline_i.p75,
            r.skyline_i.max,
            r.skyline_ii.mean,
            r.skyline_ii.p75,
            r.skyline_ii.max,
            r.decision.threshold_max,
            r.decision.threshold_mean,
            if r.decision.ii_wins { "(ii)" } else { "(i) stays" },
        )
    }

    /// M4's results table, over the gate eyes plus the low eye, at ring 3's
    /// candidate (i) = `v = 4` (today's).
    pub fn report() -> String {
        let gen = WorldGen::new(World::new().seed());
        let mut s = String::new();
        let _ = writeln!(
            s,
            "M4 ({COLUMNS} columns, {MAX_DISTANCE} m sweep): band D max/mean (i / ii), skyline mean/p75/max px (i / ii), threshold, decision"
        );
        let _ = writeln!(
            s,
            "| eye | band D max/mean (i) | band D max/mean (ii) | skyline mean/p75/max px (i) | skyline mean/p75/max px (ii) | threshold max/mean | decision |"
        );
        let _ = writeln!(s, "|---|---|---|---|---|---|---|");
        // The issue's own eyes: ground and hill, not the flight eye -- at
        // 3,000 m up the game already draws nothing but the far terrain
        // (`GATE_EYES`' own doc comment), so ring 3 is moot there.
        for g in GATE_EYES
            .iter()
            .filter(|g| g.name == "ground" || g.name == "hill")
        {
            let r = measure_eye(&gen, g.name, g.eye, GATE_PITCH, 4);
            let _ = writeln!(s, "{}", format_eye(&r));
        }
        let low = measure_eye(&gen, "low (F6 fallback)", LOW_EYE, LOW_EYE_PITCH, 4);
        let _ = writeln!(s, "{}", format_eye(&low));
        s
    }

    // -----------------------------------------------------------------------
    // T4 (`PROPOSAL_LOD_TRANSITIONS.md` §6; issue #381 as amended on
    // 2026-11-10): ring edges match a finer ring -- on this harness, not
    // on GPU frames.
    //
    // For each ring edge of `DEFAULT_RING_SCHEDULE` (160, 288 and 512 m),
    // each eye (ground, low, hill) and yaws 0 and 45 degrees, the **real**
    // schedule is ray-marched against a **reference** schedule in which
    // the ring outside the edge is drawn at the inner ring's level out to
    // the next edge (`finer_ring_schedule`: the finer ring's own height
    // lookup, `coarse_top` with the inner level's `step_h` and `v`). Both
    // agree inside the edge and beyond the next one, so what the band
    // metric sees is the edge itself: what the coarser ring draws that the
    // finer one would not.
    //
    // The camera pitches to the edge: the view axis meets the reference's
    // own drawn top just outside the edge, so the edge sits on the frame's
    // middle row -- where `Camera::row_for`'s approximation is exact and
    // the band (`BAND_HALF_ANGLE` either side) is centred on it. At the
    // hill eye the near edges need a steep pitch (the 160 m edge is 59
    // degrees down from 300 m up); the row convention then overstates
    // every riser's rows alike, in the real image, the reference and the
    // noise, and the ratios the verdict rests on are unchanged.
    //
    // **Thresholds are calibrated, not chosen, and in this viewport's own
    // pixels.** The threshold is `CALIBRATION_FACTOR` (2.5) times the
    // reference-vs-reference noise -- the same reference schedule rendered
    // again from an eye `NOISE_JITTER_BLOCKS` (0.5) sideways, perpendicular
    // to the view axis, the calibration M4 already uses (not between yaws:
    // 0 and 45 degrees look at different terrain, so their difference is
    // scene, not noise). It is measured per case, on the viewport under
    // test, so a 360p or 4K `ViewSpec` (#394) gets a threshold in its own
    // pixels rather than the gate's. Measured at the gate viewport (1920 x
    // 1080, 60 degrees; Linux, 2026-11-10, release), the numbers
    // `ring_edges_match_a_finer_ring` was calibrated on -- noise and
    // threshold under the production rule, `D` as band max / band mean, A's
    // own threshold in parentheses:
    //
    // | edge | eye | yaw | noise | threshold (2.5x) | R1 D | R1 HF | A D (A's threshold) | A / threshold |
    // |---|---|---|---|---|---|---|---|---|
    // | 160 m | ground | 0 | 0.0190 / 0.0135 | 0.0475 / 0.0338 | 0.0178 / 0.0042 | 0.999 | 0.0587 / 0.0120 (0.0475 / 0.0335) | 1.2x / 0.4x |
    // | 160 m | ground | 45 | 0.0224 / 0.0114 | 0.0559 / 0.0284 | 0.0194 / 0.0040 | 0.994 | 0.0592 / 0.0125 (0.0559 / 0.0274) | 1.1x / 0.5x |
    // | 160 m | low | 0 | 0.0212 / 0.0160 | 0.0529 / 0.0400 | 0.0307 / 0.0095 | 1.020 | 0.0708 / 0.0213 (0.0529 / 0.0400) | 1.3x / 0.5x |
    // | 160 m | low | 45 | 0.0256 / 0.0178 | 0.0641 / 0.0445 | 0.0342 / 0.0147 | 1.018 | 0.0739 / 0.0357 (0.0641 / 0.0440) | 1.2x / 0.8x |
    // | 160 m | hill | 0 | 0.0188 / 0.0132 | 0.0470 / 0.0330 | 0.0235 / 0.0066 | 0.939 | 0.0300 / 0.0097 (0.0470 / 0.0330) | 0.6x / 0.3x |
    // | 160 m | hill | 45 | 0.0359 / 0.0249 | 0.0898 / 0.0623 | 0.0319 / 0.0127 | 0.916 | 0.0392 / 0.0188 (0.0898 / 0.0623) | 0.4x / 0.3x |
    // | 288 m | ground | 0 | 0.0172 / 0.0102 | 0.0430 / 0.0256 | 0.0183 / 0.0026 | 0.996 | 0.0403 / 0.0055 (0.0456 / 0.0233) | 0.9x / 0.2x |
    // | 288 m | ground | 45 | 0.0213 / 0.0074 | 0.0531 / 0.0186 | 0.0184 / 0.0040 | 0.994 | 0.0333 / 0.0061 (0.0552 / 0.0149) | 0.6x / 0.4x |
    // | 288 m | low | 0 | 0.0222 / 0.0130 | 0.0554 / 0.0326 | 0.0293 / 0.0081 | 0.985 | 0.0801 / 0.0195 (0.0482 / 0.0259) | 1.7x / 0.8x |
    // | 288 m | low | 45 | 0.0201 / 0.0116 | 0.0501 / 0.0289 | 0.0306 / 0.0103 | 0.969 | 0.0694 / 0.0196 (0.0384 / 0.0194) | 1.8x / 1.0x |
    // | 288 m | hill | 0 | 0.0175 / 0.0104 | 0.0437 / 0.0261 | 0.0210 / 0.0060 | 0.964 | 0.0318 / 0.0097 (0.0335 / 0.0190) | 0.9x / 0.5x |
    // | 288 m | hill | 45 | 0.0228 / 0.0153 | 0.0570 / 0.0381 | 0.0385 / 0.0159 | 0.864 | 0.0507 / 0.0232 (0.0453 / 0.0294) | 1.1x / 0.8x |
    // | 512 m | ground | 0 | 0.0167 / 0.0069 | 0.0418 / 0.0171 | 0.0523 / 0.0056 | 1.017 | 0.0750 / 0.0073 (0.0397 / 0.0159) | 1.9x / 0.5x |
    // | 512 m | ground | 45 | 0.0133 / 0.0030 | 0.0334 / 0.0075 | 0.0331 / 0.0047 | 1.029 | 0.0374 / 0.0055 (0.0249 / 0.0065) | 1.5x / 0.9x |
    // | 512 m | low | 0 | 0.0166 / 0.0076 | 0.0415 / 0.0190 | 0.0628 / 0.0121 | 0.956 | 0.0884 / 0.0166 (0.0336 / 0.0140) | 2.6x / 1.2x |
    // | 512 m | low | 45 | 0.0143 / 0.0039 | 0.0357 / 0.0098 | 0.0339 / 0.0083 | 0.958 | 0.0434 / 0.0100 (0.0243 / 0.0069) | 1.8x / 1.5x |
    // | 512 m | hill | 0 | 0.0118 / 0.0081 | 0.0295 / 0.0204 | 0.0555 / 0.0236 | 0.753 | 0.0734 / 0.0299 (0.0226 / 0.0135) | 3.2x / 2.2x |
    // | 512 m | hill | 45 | 0.0193 / 0.0148 | 0.0481 / 0.0370 | 0.0391 / 0.0217 | 0.804 | 0.0494 / 0.0276 (0.0386 / 0.0268) | 1.3x / 1.0x |
    //
    // The real schedule sits at 0.3-0.7x its threshold on the band max and
    // 0.1-0.5x on the band mean at every asserted case. The same scene at
    // 360, 720 and 2160 rows (same aspect) moves every `D` by up to 2x
    // between the extremes -- R1's and A's together, their ratio within
    // about 10% -- which is why the threshold is measured on the viewport
    // under test rather than carried over from the gate's.
    //
    // **Option A is the positive control, in the same test.** The vertical
    // cell `v` is a parameter (`CellHeight`): `production_cell_height`
    // reads `NodeKey::cell_height_blocks` and `option_a_cell_height` is
    // `2^level`, the state before R1 -- no `cfg(test)` fork anywhere. The
    // test asserts that A fails its own calibrated threshold at the 160 m
    // edge and at the 288 m edge: at each, the worst case over eyes and
    // yaws has a band max of at least `OPTION_A_MARGIN` times its
    // threshold. That is the issue's "seen to fail" requirement built in,
    // and the guard that keeps a per-case threshold honest: a noise
    // inflated enough to admit anything fails the test from the other side.
    //
    // **What the margin is, and is not.** Issue #381 asks for A to fail by
    // 2x. Measured, the §6 metric separates A from R1 by less than that.
    // The band *mean* cannot tell A from a half-block eye jitter at all (A
    // sits at 0.2-1.0x its mean threshold in every 160/288 m case: the
    // mean is diluted over a band most of whose rows the two schedules
    // agree on, while the jitter moves every row). The band *max* puts A
    // at 1.3x (160 m, low eye, 0 degrees) and 1.8x (288 m, low eye, 45
    // degrees) over its threshold at the best case, and under 1x at the
    // hill eye looking 60 degrees down, where a taller riser is
    // foreshortened to nothing -- and where the game's own 3D selection
    // never draws levels 0-1 in the first place (M1's hill-eye table). Every
    // noise model tried (M4's perpendicular half-block eye jitter; a
    // half-pixel row offset; a half-column azimuth offset; the band
    // narrowed to the ring's own rows) lands in the same 1.2-1.8x range on
    // the max, because any whole-image perturbation moves every edge in the
    // frame by about as much as A moves the surface in one ring: half a
    // block. So the margin asserted here is `OPTION_A_MARGIN` = 1.2 on the
    // band max per edge -- A goes red at both edges, with 8% headroom at
    // 160 m and 50% at 288 m -- and the 2x the issue asked for is recorded
    // as not reached by this metric rather than quietly redefined: setting
    // the constant to 2.0 turns this test red on today's code.
    //
    // **The 512 m edge is measured, not asserted**, as in T1: level 2's
    // 1-block cell meeting level 3's 4-block one is ring 3's accepted
    // decision (M4, #335). The real schedule already exceeds its threshold
    // there in four of the six cases -- the metric agreeing with T1 about
    // where the one remaining step is.
    // -----------------------------------------------------------------------

    /// The band around an edge, as an angle so it is the same *view* at
    /// every viewport: 5 degrees either side of the edge's own row -- 90
    /// rows at the gate (M4's own [`BAND_HALF_HEIGHT`], see
    /// `the_t4_band_is_m4s_band_at_the_gate`), and that many of *this*
    /// viewport's pixels elsewhere ([`band_half_rows`]).
    pub const BAND_HALF_ANGLE: f32 = 5.0 * std::f32::consts::PI / 180.0;

    /// [`BAND_HALF_ANGLE`] in rows of `view`.
    pub fn band_half_rows(view: ViewSpec) -> f32 {
        (BAND_HALF_ANGLE / view.pixel_angle()).round()
    }

    /// How far sideways (perpendicular to the view axis, in blocks) the
    /// reference is re-rendered from for the noise the threshold is 2.5x
    /// of -- M4's own half a block.
    pub const NOISE_JITTER_BLOCKS: f32 = 0.5;

    /// Option A's worst case per asserted edge must exceed its own
    /// threshold (band max) by this factor. Issue #381 asks for 2; the
    /// metric gives 1.3x and 1.8x at the gate viewport (the T4 section
    /// above says why), so this sits under the measured minimum and the
    /// shortfall is on record there.
    pub const OPTION_A_MARGIN: f32 = 1.2;

    /// The high-frequency energy ratio (real / reference) T4 allows:
    /// `[0.8, 1.25]` (`PROPOSAL_LOD_TRANSITIONS.md` §6).
    pub const HF_RATIO_RANGE: (f32, f32) = (0.8, 1.25);

    /// T4's yaws, degrees.
    pub const T4_YAWS_DEG: [f32; 2] = [0.0, 45.0];

    /// T4's reference for edge `edge` of `schedule` (an index into
    /// [`ring_edges`]): the same schedule with the ring outside the edge
    /// drawn at the inner ring's level, out to the next edge.
    pub fn finer_ring_schedule(schedule: &RingSchedule, edge: usize) -> Vec<(u32, i32)> {
        let mut s = schedule.to_vec();
        s[edge + 1].0 = s[edge].0;
        s
    }

    /// T4's eyes: the gate's ground and hill eyes and M4's low eye.
    pub fn t4_eyes() -> [(&'static str, [f32; 3]); 3] {
        [
            (GATE_EYES[0].name, GATE_EYES[0].eye),
            ("low", LOW_EYE),
            (GATE_EYES[1].name, GATE_EYES[1].eye),
        ]
    }

    /// One edge, one eye, one yaw, one cell-height rule: the real schedule
    /// against the finer-ring reference, and the noise its threshold is
    /// calibrated from.
    #[derive(Clone, Debug)]
    pub struct EdgeMeasurement {
        /// The rows the metric was taken over.
        pub band: std::ops::Range<usize>,
        /// The pitch that put the edge on the frame's middle row.
        pub pitch: f32,
        /// Real schedule vs the finer-ring reference.
        pub real: BandMetric,
        /// Reference vs the reference from `NOISE_JITTER_BLOCKS` aside.
        pub noise: BandMetric,
        /// `CALIBRATION_FACTOR` times the noise's own max and mean.
        pub threshold_max: f32,
        pub threshold_mean: f32,
        /// Real over reference high-frequency energy in the band.
        pub hf_ratio: f32,
    }

    impl EdgeMeasurement {
        /// T4's band assertion: `D`'s band maximum and band mean both
        /// within their thresholds.
        pub fn within_threshold(&self) -> bool {
            self.real.max <= self.threshold_max && self.real.mean <= self.threshold_mean
        }

        /// How far over (or under) its own threshold the band max sits:
        /// `real.max / threshold_max`, the positive control's figure. The
        /// `.max(EPSILON)` only keeps a zero threshold from dividing to
        /// infinity; `excess_is_the_band_max_over_its_threshold` pins that
        /// it is a floor and not a cap (a cap would make every excess
        /// astronomically large and the positive control vacuous).
        pub fn excess(&self) -> f32 {
            self.real.max / self.threshold_max.max(f32::EPSILON)
        }

        /// T4's high-frequency assertion.
        pub fn hf_ratio_in_range(&self) -> bool {
            (HF_RATIO_RANGE.0..=HF_RATIO_RANGE.1).contains(&self.hf_ratio)
        }
    }

    /// Where T4 looks from: an eye, a world yaw (radians), and the viewport
    /// (`view` at `aspect`) whose pixels the measurement is in. The pitch
    /// is not here -- `measure_edge` derives it per edge.
    #[derive(Clone, Copy, Debug)]
    pub struct EyeView {
        pub view: ViewSpec,
        pub aspect: f32,
        pub eye: [f32; 3],
        pub yaw: f32,
    }

    /// Measure edge `edge` (an index into [`ring_edges`]`(schedule)`) from
    /// `at`, with every level's vertical cell from `cell_height`.
    pub fn measure_edge(
        gen: &WorldGen,
        at: EyeView,
        schedule: &RingSchedule,
        edge: usize,
        cell_height: CellHeight,
    ) -> EdgeMeasurement {
        let EyeView {
            view,
            aspect,
            eye,
            yaw,
        } = at;
        let e = ring_edges(schedule)[edge];
        let d = e.edge_blocks as f32;
        let (dx, dz) = (yaw.cos(), yaw.sin());
        // The point on the view axis just outside the edge, at the
        // reference's own drawn top there (the inner level's lookup): the
        // camera pitches to it, so `row_for(d, face_y)` is the middle row.
        let (x, z) = (
            (eye[0] + d * dx).round() as i32,
            (eye[2] + d * dz).round() as i32,
        );
        let face_y = coarse_top(gen, x, z, 1 << e.inside, cell_height(e.inside)) as f32 + 1.0;
        let pitch = (face_y - eye[1]) / d;
        let cam = Camera {
            eye,
            pitch,
            yaw,
            view,
            aspect,
        };
        let height_px = view.height_px as usize;
        let band = band_rows(height_px as f32 / 2.0, band_half_rows(view), height_px);

        let reference_schedule = finer_ring_schedule(schedule, edge);
        let reference = Variant::Rings {
            schedule: &reference_schedule,
            cell_height,
        };
        let real = Variant::Rings {
            schedule,
            cell_height,
        };
        let materials = far_streaming::far_materials();
        let lighting = Lighting::default();
        let columns = cam.columns();
        let ref_set = render_eye(cam, &reference, gen, &materials, &lighting, columns);
        let real_set = render_eye(cam, &real, gen, &materials, &lighting, columns);
        // Perpendicular to the view axis: `(-sin, cos)` of the yaw, which
        // at yaw 0 is M4's own `z + 0.5`.
        let jittered = Camera {
            eye: [
                eye[0] - dz * NOISE_JITTER_BLOCKS,
                eye[1],
                eye[2] + dx * NOISE_JITTER_BLOCKS,
            ],
            ..cam
        };
        let jittered_set = render_eye(jittered, &reference, gen, &materials, &lighting, columns);

        let real_m = band_metric_between(&real_set, &ref_set, band.clone(), &lighting);
        let noise = band_metric_between(&jittered_set, &ref_set, band.clone(), &lighting);
        let hf_ratio = high_frequency_energy(&real_set, band.clone(), &lighting)
            / high_frequency_energy(&ref_set, band.clone(), &lighting);
        EdgeMeasurement {
            band,
            pitch,
            real: real_m,
            threshold_max: CALIBRATION_FACTOR * noise.max,
            threshold_mean: CALIBRATION_FACTOR * noise.mean,
            noise,
            hf_ratio,
        }
    }

    /// One T4 case: an edge, an eye and a yaw, measured under the
    /// production cell heights and under option A.
    #[derive(Clone, Debug)]
    pub struct T4Row {
        pub eye_name: &'static str,
        pub eye: [f32; 3],
        pub yaw_deg: f32,
        pub edge: RingEdge,
        pub production: EdgeMeasurement,
        pub option_a: EdgeMeasurement,
    }

    /// Every T4 case of `DEFAULT_RING_SCHEDULE` on `view` at `aspect`: 3
    /// edges x 3 eyes x 2 yaws, each under both cell-height rules, in
    /// edge-major order. The cases are independent, so they run on every
    /// core; the order of the result is fixed regardless.
    pub fn t4_rows(gen: &WorldGen, view: ViewSpec, aspect: f32) -> Vec<T4Row> {
        let schedule = DEFAULT_RING_SCHEDULE;
        let mut cases = Vec::new();
        for edge in 0..ring_edges(schedule).len() {
            for (name, eye) in t4_eyes() {
                for yaw_deg in T4_YAWS_DEG {
                    cases.push((edge, name, eye, yaw_deg));
                }
            }
        }
        let next = std::sync::atomic::AtomicUsize::new(0);
        // The clamp is a thread count, nothing more: `check-tests-can-fail.sh`
        // found `.max(1)` -> `.min(1)` survive, which runs every case on one
        // thread and changes no result -- the same clamp `mesh_parallel`
        // carries, and no test needs to see it break.
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(cases.len().max(1));
        let mut done: Vec<(usize, T4Row)> = std::thread::scope(|s| {
            let workers: Vec<_> = (0..threads)
                .map(|_| {
                    s.spawn(|| {
                        let mut mine = Vec::new();
                        loop {
                            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let Some(&(edge, name, eye, yaw_deg)) = cases.get(i) else {
                                break;
                            };
                            let at = EyeView {
                                view,
                                aspect,
                                eye,
                                yaw: yaw_deg.to_radians(),
                            };
                            let measure =
                                |cell_height| measure_edge(gen, at, schedule, edge, cell_height);
                            mine.push((
                                i,
                                T4Row {
                                    eye_name: name,
                                    eye,
                                    yaw_deg,
                                    edge: ring_edges(schedule)[edge],
                                    production: measure(production_cell_height),
                                    option_a: measure(option_a_cell_height),
                                },
                            ));
                        }
                        mine
                    })
                })
                .collect();
            workers
                .into_iter()
                .flat_map(|w| w.join().expect("a T4 worker panicked"))
                .collect()
        });
        done.sort_by_key(|(i, _)| *i);
        done.into_iter().map(|(_, row)| row).collect()
    }

    /// T4's table: one line per case, both rules side by side, with the
    /// noise and thresholds the verdict was calibrated on.
    pub fn format_t4(view: ViewSpec, rows: &[T4Row]) -> String {
        let mut s = String::new();
        let _ = writeln!(
            s,
            "T4 ({} rows, {} px band half-height, columns per {COLUMN_STRIDE_PX} px): band D max/mean, noise and threshold ({CALIBRATION_FACTOR}x), HF ratio -- real (R1) and option A",
            view.height_px,
            band_half_rows(view)
        );
        let _ = writeln!(
            s,
            "| edge | eye | yaw | pitch | noise max/mean | threshold max/mean | R1 D max/mean | R1 HF | A noise max/mean | A threshold max/mean | A D max/mean | A HF | A / threshold max/mean |"
        );
        let _ = writeln!(s, "|---|---|---|---|---|---|---|---|---|---|---|---|---|");
        for r in rows {
            let (p, a) = (&r.production, &r.option_a);
            let _ = writeln!(
                s,
                "| {} m | {} {:?} | {}° | {:.3} | {:.4} / {:.4} | {:.4} / {:.4} | {:.4} / {:.4} | {:.3} | {:.4} / {:.4} | {:.4} / {:.4} | {:.4} / {:.4} | {:.3} | {:.1}x / {:.1}x |",
                r.edge.edge_blocks,
                r.eye_name,
                r.eye,
                r.yaw_deg,
                p.pitch,
                p.noise.max,
                p.noise.mean,
                p.threshold_max,
                p.threshold_mean,
                p.real.max,
                p.real.mean,
                p.hf_ratio,
                a.noise.max,
                a.noise.mean,
                a.threshold_max,
                a.threshold_mean,
                a.real.max,
                a.real.mean,
                a.hf_ratio,
                // Report prose, not program behaviour (as `format_m6` notes
                // of its own cells): `check-tests-can-fail.sh` found the
                // `.max(EPSILON)` floor here survive a flip to `.min`, and
                // nothing asserts this table's text. The figure the test
                // acts on is `EdgeMeasurement::excess`, which has its own
                // unit test.
                a.excess(),
                a.real.mean / a.threshold_mean.max(f32::EPSILON),
            );
        }
        s
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// `Camera::azimuth` is 0 at the middle column and antisymmetric
        /// around it -- the centre column looks straight down the camera's
        /// own yaw (0 here), and column `columns - 1 - c` mirrors column
        /// `c`. A mistake in the NDC mapping (an off-by-one, a wrong sign, a
        /// wrong scale) breaks at least one of these.
        ///
        /// (The `< 0.2` bound's own `<`/`<=` is not pinned, and does not
        /// need to be: `azimuth(3)` and `azimuth(4)` are `atan` of an
        /// irrational ratio, never exactly `0.2`, so the two operators
        /// agree on every real input -- true of any tolerance check,
        /// `check-tests-can-fail.sh` included.)
        #[test]
        fn camera_azimuth_is_zero_at_the_centre_and_antisymmetric() {
            let cam = Camera::gate([0.0, 0.0, 0.0], 0.0);
            let columns = 8;
            // Columns 0..8: the centre sits between columns 3 and 4 (NDC 0
            // falls at col 3.5), so those two are closest to, and
            // symmetric around, zero. Checked independently (not combined
            // through a `max`/`min`) so a mix-up between the two can't
            // hide behind the other column's own margin.
            assert!(
                cam.azimuth(columns, 3).abs() < 0.2,
                "azimuth(3) = {}, too far from 0",
                cam.azimuth(columns, 3)
            );
            assert!(
                cam.azimuth(columns, 4).abs() < 0.2,
                "azimuth(4) = {}, too far from 0",
                cam.azimuth(columns, 4)
            );
            for c in 0..columns {
                let mirror = columns - 1 - c;
                assert!(
                    (cam.azimuth(columns, c) + cam.azimuth(columns, mirror)).abs() < 1e-5,
                    "column {c} and its mirror {mirror} are not antisymmetric"
                );
            }
            // Monotonic across the frame: a later column is never a
            // smaller azimuth (this camera's FOV is under 180 degrees).
            let azimuths: Vec<f32> = (0..columns).map(|c| cam.azimuth(columns, c)).collect();
            assert!(azimuths.is_sorted(), "{azimuths:?}");
        }

        /// `ring_face_height` is exactly `coarse_top(...) + 1` (the block
        /// surface's top FACE, not the block itself) -- pinned at a known
        /// point so a wrong offset (`+2`, or none at all) is caught.
        #[test]
        fn ring_face_height_is_one_above_coarse_top() {
            let gen = WorldGen::new(World::new().seed());
            let (x, z) = (20, -5);
            let got = ring_face_height(
                &gen,
                DEFAULT_RING_SCHEDULE,
                (0.0, 0.0),
                x,
                z,
                &production_cell_height,
            )
            .expect("inside the first ring");
            // Both sides do the identical integer arithmetic on the same
            // input, so this is an equality, not a tolerance.
            let want = coarse_top(&gen, x, z, 1, 1) as f64 + 1.0;
            assert_eq!(got, want);
        }

        /// `exact_face_height` is exactly `surface_height + 1` -- the exact
        /// per-block reference's own version of the same "+1 for the top
        /// face" offset `ring_face_height_is_one_above_coarse_top` pins for
        /// the coarse path.
        #[test]
        fn exact_face_height_is_one_above_the_exact_surface() {
            let gen = WorldGen::new(World::new().seed());
            for (x, z) in [(0, 0), (37, -19), (128, 300), (-50, 12)] {
                let want = gen.surface_height(x, z) as f64 + 1.0;
                assert_eq!(exact_face_height(&gen, x, z), want);
            }
        }

        /// `block_below_face` undoes that `+ 1.0` exactly -- the inverse
        /// `sweep_column`'s `top_blocks`/`new_top` both rely on.
        #[test]
        fn block_below_face_is_the_exact_inverse_of_the_plus_one() {
            let gen = WorldGen::new(World::new().seed());
            for (x, z) in [(0, 0), (37, -19), (128, 300), (-50, 12)] {
                let face = exact_face_height(&gen, x, z);
                assert_eq!(block_below_face(face), gen.surface_height(x, z));
            }
            // And directly, for values with no world behind them.
            for face_y in [1.0f64, 25.0, -8.0, 100.5] {
                assert_eq!(block_below_face(face_y), (face_y - 1.0).round() as i32);
            }
        }

        /// The DDA crosses *both* axes when `dir` points diagonally -- not
        /// just the one it happens to start on. A delta stuck at infinity
        /// on one axis (as if that axis never moved) would freeze that
        /// axis after its first crossing; this counts how many distinct
        /// values each axis actually reaches and requires both to move
        /// close to what the direction's own ratio predicts.
        #[test]
        fn the_dda_advances_on_both_axes_for_a_diagonal_direction() {
            // Every quadrant: `Dda::new`'s initial `next[]` has a separate
            // branch per sign of `dir`, so a direction with both signs
            // negative (or one of each) exercises branches the first,
            // all-positive-only version of this test never reached.
            for dir in [
                (0.8f32, 0.6f32),
                (-0.8f32, 0.6f32),
                (0.8f32, -0.6f32),
                (-0.8f32, -0.6f32),
            ] {
                let limit = 40.0;
                let mut dda = Dda::new((0.0, 0.0), dir, limit);
                let (mut x_crossings, mut z_crossings) = (0u32, 0u32);
                loop {
                    match dda.advance() {
                        Some((_, 0)) => x_crossings += 1,
                        Some((_, 1)) => z_crossings += 1,
                        Some(_) => unreachable!("only two axes exist"),
                        None => break,
                    }
                }
                // Expected crossings are `limit * |dir|` per axis (a
                // standard grid DDA): about 32 on x and 24 on z. A frozen
                // axis (the mutation this pins against: one branch of
                // `next[]`'s setup always taken) would stop at 0 or 1,
                // nowhere near either bound.
                assert!(
                    (28..=34).contains(&x_crossings),
                    "dir {dir:?}: x crossings {x_crossings}"
                );
                assert!(
                    (20..=26).contains(&z_crossings),
                    "dir {dir:?}: z crossings {z_crossings}"
                );
            }
        }

        /// Every boundary distance the DDA reports -- both `exit_distance`
        /// (before advancing) and the distance `advance` itself returns --
        /// is non-negative, for every quadrant of `dir`. The wrong branch
        /// of `Dda::new`'s per-sign `next[]` setup produces a *negative*
        /// initial threshold (worked out by hand for `dir.0 < 0`: the
        /// always-taken `dir.0 > 0` formula divides by a negative number),
        /// which the crossing-count test above cannot tell apart from the
        /// correct, merely phase-shifted sequence.
        #[test]
        fn the_dda_never_reports_a_negative_distance() {
            for dir in [
                (0.8f32, 0.6f32),
                (-0.8f32, 0.6f32),
                (0.8f32, -0.6f32),
                (-0.8f32, -0.6f32),
            ] {
                // Off a cell corner, so the very first threshold isn't
                // trivially zero regardless of which branch ran.
                let mut dda = Dda::new((0.3, 0.7), dir, 40.0);
                for _ in 0..30 {
                    let exit = dda.exit_distance();
                    assert!(
                        exit >= 0.0,
                        "dir {dir:?}: exit_distance reported a negative distance {exit}"
                    );
                    let Some((d, _)) = dda.advance() else { break };
                    assert!(
                        d >= 0.0,
                        "dir {dir:?}: advance reported a negative distance {d}"
                    );
                }
            }
        }

        /// `exit_distance` predicts exactly the boundary `advance` is about
        /// to use -- not the sweep's own distance limit. Swapping its
        /// `min`/`max` against `limit` would make it report the limit on
        /// almost every call instead.
        #[test]
        fn exit_distance_matches_the_boundary_advance_uses() {
            let mut dda = Dda::new((0.0, 0.0), (0.8, 0.6), 40.0);
            for _ in 0..20 {
                let exit = dda.exit_distance();
                assert!(exit < 40.0, "exit_distance reported the limit: {exit}");
                let Some((d, _)) = dda.advance() else { break };
                assert_eq!(exit, d, "exit_distance didn't predict advance's boundary");
            }
        }

        /// The DDA's first two boundaries, against hand-worked values --
        /// catches a wrong constant in *either* axis's `next[]` setup
        /// (`+1.0` is "one block ahead of this column's own origin", not
        /// `+2.0` or anything else) that a self-consistency check like the
        /// one above, or a loose crossing-count range, cannot: both would
        /// still agree with each other, and still land inside a wide
        /// enough range, on a value that is simply wrong.
        #[test]
        fn the_dda_first_two_boundaries_match_hand_worked_values() {
            // Eye at (0.3, 0.7), dir (0.8, 0.6): the x boundary is
            // (1.0 - 0.3) / 0.8 = 0.875; the z boundary is
            // (1.0 - 0.7) / 0.6 = 0.5. z is nearer, so it crosses first.
            let mut dda = Dda::new((0.3, 0.7), (0.8, 0.6), 40.0);
            assert!(
                (dda.exit_distance() - 0.5).abs() < 1e-5,
                "{}",
                dda.exit_distance()
            );
            let (d, axis) = dda.advance().expect("well inside the limit");
            assert!((d - 0.5).abs() < 1e-5, "{d}");
            assert_eq!(axis, 1, "z should cross first");
            assert!(
                (dda.exit_distance() - 0.875).abs() < 1e-5,
                "{}",
                dda.exit_distance()
            );
        }

        /// `paint_range`'s lower clamp keeps a positive row where it is (a
        /// riser far below the horizon is the common case) and only floors
        /// a negative one to 0 -- a `max`/`min` slip would instead clamp
        /// *every* row to 0, including the common positive case.
        #[test]
        fn paint_range_only_clamps_a_negative_row_to_zero() {
            assert_eq!(paint_range(50.0, 100.0, 1080), (50, 100));
            assert_eq!(paint_range(-5.0, 100.0, 1080), (0, 100));
        }

        /// `band_window`'s range is centred and `2 * BAND_HALF_HEIGHT + 1`
        /// rows wide away from the screen edges -- not collapsed to a
        /// single row (a `max`/`min` slip in the clamp would do exactly
        /// that).
        #[test]
        fn band_window_is_the_full_band_wide_away_from_the_edges() {
            let gen = WorldGen::new(World::new().seed());
            // The ground eye's own 512 m point sits well short of the
            // screen's bottom edge (T1/§1.1: the 512 m edge is ~20-64 rows
            // below the horizon at the gate eyes, nowhere near row 1080).
            let cam = Camera::gate([8.0, 40.0, 8.0], 0.0);
            let band = band_window(cam, &gen, (8.0, 8.0));
            assert!(band.start < band.end, "{band:?}");
            let width = band.end - band.start;
            assert_eq!(
                width,
                (2 * BAND_HALF_HEIGHT + 1) as usize,
                "band {band:?} is not the full window"
            );
        }

        /// Identical images (after the low-pass) give zero everywhere --
        /// the metric cannot manufacture a difference out of nothing.
        #[test]
        fn identical_images_give_zero_difference() {
            let (width, height) = (6, 4);
            let grid: Vec<f32> = (0..width * height)
                .map(|i| ((i as f32) * 0.37).sin().abs())
                .collect();
            let lp = box_low_pass_4x4(&grid, width, height);
            let m = band_metric(&lp, &lp, width, height);
            assert_eq!(m.max, 0.0);
            assert_eq!(m.mean, 0.0);
            assert!(m.per_row.iter().all(|&d| d == 0.0));
        }

        /// A known step, worked out by hand: `F` is 0 for the first four
        /// columns and 1 for the next four, `R` is flat 0, one row. After the
        /// 4x4 low-pass (which shrinks to fit at the edges, since there is
        /// only one row): `D` has a known value, exactly representable in
        /// `f32`, so this is an exact assertion, not a tolerance.
        #[test]
        fn a_known_step_gives_a_known_difference() {
            let f = [0.0f32, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
            let r = [0.0f32; 8];
            let (lp_f, lp_r) = (box_low_pass_4x4(&f, 8, 1), box_low_pass_4x4(&r, 8, 1));
            // By hand: a 4x4 box shrunk to fit a single row is a 3-wide
            // (2-wide at the very edges) box along it alone. At x = 0, 1, 2
            // the box never reaches the step (columns 0-3 are 0): 0. At
            // x = 3 it first reaches one `1` (column 4) of its 3: 1/3. At
            // x = 4 it holds two (columns 3, 5 besides the step... no,
            // columns 3, 4, 5 = 0, 1, 1): 2/3. From x = 5 on, every column
            // the box reaches is past the step: 1.
            let expected = [0.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0];
            for (got, want) in lp_f.iter().zip(expected) {
                assert!((got - want).abs() < 1e-6, "{lp_f:?}");
            }
            // One row: the band's maximum of `D` and its mean coincide.
            let m = band_metric(&lp_f, &lp_r, 8, 1);
            let expected_mean = expected.iter().sum::<f32>() / expected.len() as f32;
            assert!((m.max - expected_mean).abs() < 1e-6, "{m:?}");
            assert!((m.mean - expected_mean).abs() < 1e-6, "{m:?}");
            assert!((m.mean - 0.5).abs() < 1e-6, "{m:?}");
        }

        /// The band's maximum of `D` and its mean are genuinely different
        /// quantities, not the same number read twice: two rows, one far
        /// from the reference everywhere, one matching it exactly -- a
        /// metric that silently averaged instead of taking a max (or vice
        /// versa) would fail this even though the single-row test above
        /// cannot tell the two apart.
        ///
        /// (`per_row[0]`'s own `< 1e-6` is summed from three copies of the
        /// literal `0.8f32`, which floating addition is not guaranteed to
        /// divide back to bit-for-bit -- unlike `per_row[1]`, below, which
        /// sums literal zeros. A tolerance, not an equality, on purpose.)
        #[test]
        fn band_metric_reports_the_bands_max_separately_from_its_mean() {
            let lp_f = [0.8f32, 0.8, 0.8, 0.0, 0.0, 0.0];
            let lp_r = [0.0f32; 6];
            let m = band_metric(&lp_f, &lp_r, 3, 2);
            assert!((m.per_row[0] - 0.8).abs() < 1e-6, "{m:?}");
            // Exactly representable (both rows are literally identical),
            // so this is an equality, not a tolerance -- `<` vs `<=` on a
            // tolerance can't tell `0.0` apart from itself either way.
            assert_eq!(m.per_row[1], 0.0, "{m:?}");
            assert!((m.max - 0.8).abs() < 1e-6, "{m:?}");
            assert!((m.mean - 0.4).abs() < 1e-6, "{m:?}");
        }

        /// A uniform offset (no edge at all) gives that same offset
        /// everywhere -- so the metric is not accidentally only sensitive to
        /// edges.
        #[test]
        fn a_uniform_offset_gives_a_uniform_difference() {
            let f = [0.3f32; 16];
            let r = [0.1f32; 16];
            let (lp_f, lp_r) = (box_low_pass_4x4(&f, 4, 4), box_low_pass_4x4(&r, 4, 4));
            let m = band_metric(&lp_f, &lp_r, 4, 4);
            assert!((m.max - 0.2).abs() < 1e-5, "{m:?}");
            assert!((m.mean - 0.2).abs() < 1e-5, "{m:?}");
        }

        /// The low-pass of a uniform grid is that same value everywhere --
        /// pins that the box filter does not shift or scale a flat input
        /// (a broken normalization would). A tolerance, not an equality,
        /// for the same reason as the test above: summing nine copies of
        /// `0.42` and dividing by 9 is not guaranteed to land back on the
        /// exact bit pattern.
        #[test]
        fn a_flat_grid_is_unchanged_by_the_low_pass() {
            let grid = [0.42f32; 9];
            let lp = box_low_pass_4x4(&grid, 3, 3);
            assert!(lp.iter().all(|&v| (v - 0.42).abs() < 1e-6), "{lp:?}");
        }

        #[test]
        fn skyline_error_is_zero_when_rows_match() {
            let set = |rows: Vec<f32>| ColumnSet {
                rows: vec![Vec::new(); rows.len()],
                skyline_rows: rows,
            };
            let a = set(vec![10.0, 20.0, 30.0]);
            let b = set(vec![10.0, 20.0, 30.0]);
            let e = skyline_error(&a, &b);
            assert_eq!(e, SkylineError::default());
        }

        /// A known offset in one column only: pins `mean`, `p75` and `max`
        /// against a value worked out by hand, so a broken percentile or a
        /// mean that silently became a sum cannot pass.
        #[test]
        fn skyline_error_reports_the_known_offsets() {
            let set = |rows: Vec<f32>| ColumnSet {
                rows: vec![Vec::new(); rows.len()],
                skyline_rows: rows,
            };
            let a = set(vec![0.0, 10.0, 0.0, 0.0]);
            let b = set(vec![0.0, 0.0, 0.0, 0.0]);
            let e = skyline_error(&a, &b);
            assert_eq!(e.max, 10.0);
            // Exactly representable ((0 + 10 + 0 + 0) / 4), so an equality,
            // not a tolerance.
            assert_eq!(e.mean, 2.5, "{e:?}");
            // Sorted errors: [0, 0, 0, 10]; `len * 3 / 4` = 3 -> the last one.
            assert_eq!(e.p75, 10.0);
        }

        /// The decision rule as the issue states it: (ii) wins only within
        /// the threshold on the band metric *and* no worse on skyline error.
        /// Each of the four ways it can lose, pinned.
        #[test]
        fn the_decision_rule_requires_both_conditions() {
            let good_band = BandMetric {
                per_row: vec![],
                max: 0.01,
                mean: 0.005,
            };
            let bad_band = BandMetric {
                per_row: vec![],
                max: 0.5,
                mean: 0.4,
            };
            let good_sky = SkylineError {
                mean: 1.0,
                p75: 1.0,
                max: 1.0,
            };
            let bad_sky = SkylineError {
                mean: 10.0,
                p75: 10.0,
                max: 10.0,
            };
            let noise = BandMetric {
                per_row: vec![],
                max: 0.02,
                mean: 0.01,
            };
            assert!(
                decide(
                    good_band.clone(),
                    good_band.clone(),
                    good_sky,
                    good_sky,
                    noise.clone()
                )
                .ii_wins
            );
            assert!(
                !decide(
                    good_band.clone(),
                    bad_band.clone(),
                    good_sky,
                    good_sky,
                    noise.clone()
                )
                .ii_wins
            );
            assert!(
                !decide(
                    good_band.clone(),
                    good_band.clone(),
                    good_sky,
                    bad_sky,
                    noise.clone()
                )
                .ii_wins
            );
            assert!(!decide(good_band, bad_band, good_sky, bad_sky, noise).ii_wins);
        }

        /// `coarse_top` at `v = 1` is exactly `cell_surface_height` (no
        /// vertical rounding at all); finer cells (smaller `step_h`) find
        /// the same exact surface a plain level-0 query would.
        #[test]
        fn coarse_top_at_v_one_is_the_cell_surface_height() {
            let gen = WorldGen::new(World::new().seed());
            for (x, z) in [(0, 0), (37, -12), (900, 450)] {
                assert_eq!(
                    coarse_top(&gen, x, z, 1, 1),
                    gen.surface_height(x, z),
                    "step_h 1: exact"
                );
                assert_eq!(
                    coarse_top(&gen, x, z, 4, 1),
                    gen.cell_surface_height(x.div_euclid(4) * 4, z.div_euclid(4) * 4, 4)
                );
            }
        }

        /// `coarse_top` at a taller `v` rounds to the top of whichever
        /// `v`-tall row holds the surface -- within half a cell either way
        /// (never more, R2 decision 1's own bound, pinned for `drawn_top`
        /// itself by `drawn_top_is_within_half_a_cell_at_v_four` in
        /// `worldgen.rs`), and always the top block of a `v`-aligned row.
        #[test]
        fn coarse_top_quantizes_to_a_v_aligned_row_within_half_a_cell() {
            let gen = WorldGen::new(World::new().seed());
            for v in [2, 4, 8] {
                for (x, z) in [(10, 10), (-500, 300), (123, 456)] {
                    let top = coarse_top(&gen, x, z, 1, v);
                    assert_eq!(top.rem_euclid(v), v - 1, "v {v} ({x},{z}): top {top}");
                    let exact = gen.surface_height(x, z);
                    assert!(
                        (top - exact).abs() <= v / 2,
                        "v {v} ({x},{z}): top {top} vs exact {exact}, more than half a cell off"
                    );
                }
            }
        }

        /// One column, end to end, on the reference (exact blocks): the row
        /// array is the view's full height, the skyline is finite and in
        /// range, and something was actually painted (this is a real-world
        /// world seed, not a hand-built one, so a broken sweep that paints
        /// nothing would be silent without this).
        #[test]
        fn one_reference_column_paints_something_and_finds_a_skyline() {
            let gen = WorldGen::new(World::new().seed());
            let materials = Materials {
                top: glam::Vec3::new(0.3, 0.5, 0.2),
                side: glam::Vec3::new(0.35, 0.4, 0.2),
                soil: glam::Vec3::new(0.4, 0.3, 0.18),
                stone: glam::Vec3::new(0.45, 0.45, 0.45),
                soil_depth: 3.0,
            };
            let lighting = Lighting::default();
            let cam = Camera::gate([8.0, 40.0, 8.0], 0.0);
            let (rows, sky) =
                sweep_column(cam, 0.0, &Variant::Reference, &gen, &materials, &lighting);
            assert_eq!(rows.len(), cam.view.height_px as usize);
            assert!(sky.is_finite() && sky >= 0.0 && sky <= cam.view.height_px as f32);
            assert!(rows.iter().any(|r| r.is_some()), "nothing was painted");
        }

        /// T4's reference schedule is the real one with exactly the ring
        /// outside the edge re-levelled to the inner ring's level -- radii
        /// untouched, nothing else re-levelled.
        #[test]
        fn the_finer_ring_schedule_relevels_only_the_ring_outside_the_edge() {
            let s = finer_ring_schedule(DEFAULT_RING_SCHEDULE, 0);
            assert_eq!(s, vec![(0, 10), (0, 18), (2, 32), (3, 64)]);
            let s = finer_ring_schedule(DEFAULT_RING_SCHEDULE, 1);
            assert_eq!(s, vec![(0, 10), (1, 18), (1, 32), (3, 64)]);
            let s = finer_ring_schedule(DEFAULT_RING_SCHEDULE, 2);
            assert_eq!(s, vec![(0, 10), (1, 18), (2, 32), (2, 64)]);
        }

        /// The two cell-height rules T4 compares: production reads
        /// `NodeKey::cell_height_blocks` (1, 1, 1, 4) and option A is
        /// `2^level` (1, 2, 4, 8) -- they agree only at level 0.
        #[test]
        fn the_cell_height_rules_are_production_and_two_to_the_level() {
            for level in 0..4 {
                assert_eq!(
                    production_cell_height(level),
                    NodeKey::new(level, [0, 0, 0]).cell_height_blocks()
                );
                assert_eq!(option_a_cell_height(level), 1 << level);
            }
            assert_eq!(production_cell_height(1), 1);
            assert_eq!(option_a_cell_height(1), 2);
            assert_eq!(production_cell_height(3), 4);
            assert_eq!(option_a_cell_height(3), 8);
        }

        /// `excess` is the band max over its threshold, exactly (2.0 here,
        /// representable), and the `EPSILON` guard is a floor: a threshold
        /// of zero gives a finite, large figure rather than infinity, and a
        /// real threshold is used as is (a `.min` cap would turn every
        /// excess into `max / EPSILON`, and the positive control would pass
        /// anything).
        #[test]
        fn excess_is_the_band_max_over_its_threshold() {
            let m = |max: f32, threshold_max: f32| EdgeMeasurement {
                band: 0..1,
                pitch: 0.0,
                real: BandMetric {
                    per_row: vec![],
                    max,
                    mean: 0.0,
                },
                noise: BandMetric {
                    per_row: vec![],
                    max: 0.0,
                    mean: 0.0,
                },
                threshold_max,
                threshold_mean: 0.0,
                hf_ratio: 1.0,
            };
            assert_eq!(m(0.05, 0.025).excess(), 2.0);
            assert_eq!(m(0.01, 0.04).excess(), 0.25);
            let unguarded = m(0.05, 0.0).excess();
            assert!(unguarded.is_finite() && unguarded > 1e3, "{unguarded}");
        }

        /// T4's angular band is M4's 90-row band at the gate viewport, and
        /// scales with the rows of another: twice the rows, twice the band.
        #[test]
        fn the_t4_band_is_m4s_band_at_the_gate() {
            let gate = ViewSpec::gate();
            assert_eq!(band_half_rows(gate), BAND_HALF_HEIGHT as f32);
            let tall = ViewSpec {
                height_px: 2 * gate.height_px,
                ..gate
            };
            assert_eq!(band_half_rows(tall), 2.0 * BAND_HALF_HEIGHT as f32);
        }

        /// `Camera::columns` is M4's `COLUMNS` at the gate and follows the
        /// viewport's width elsewhere (a 32:9 viewport of the same height
        /// has twice the columns of a 16:9 one).
        #[test]
        fn columns_follow_the_viewport_width() {
            let cam = Camera::gate([0.0; 3], 0.0);
            assert_eq!(cam.columns(), COLUMNS);
            let wide = Camera {
                aspect: 2.0 * cam.aspect,
                ..cam
            };
            assert_eq!(wide.columns(), 2 * COLUMNS);
        }

        /// A yawed camera's azimuths are the unyawed ones shifted by the
        /// yaw, column for column.
        #[test]
        fn the_yaw_shifts_every_azimuth_by_the_yaw() {
            let cam = Camera::gate([0.0; 3], 0.0);
            let yawed = Camera {
                yaw: 45f32.to_radians(),
                ..cam
            };
            for c in 0..8 {
                let want = cam.azimuth(8, c) + 45f32.to_radians();
                assert!(
                    (yawed.azimuth(8, c) - want).abs() < 1e-6,
                    "column {c}: {} vs {want}",
                    yawed.azimuth(8, c)
                );
            }
        }

        /// A flat band has no high-frequency energy; a band that
        /// alternates between two luminances row by row has a known amount
        /// -- worked out by hand against the shrunk-at-the-edges box.
        #[test]
        fn high_frequency_energy_is_zero_for_a_flat_band_and_known_for_a_stripe() {
            let lighting = Lighting::default();
            let flat = ColumnSet {
                rows: vec![vec![Some([0.2, 0.2, 0.2]); 4]; 3],
                skyline_rows: vec![0.0; 3],
            };
            // A tolerance, not an equality: the box sums twelve copies of
            // one luminance and divides back, which `f32` does not land on
            // bit for bit (`a_flat_grid_is_unchanged_by_the_low_pass`
            // makes the same allowance). Its `<` versus `<=` is not pinned
            // and cannot be -- the value is never exactly `1e-6` -- as with
            // every tolerance check in this file.
            let hf_flat = high_frequency_energy(&flat, 0..4, &lighting);
            assert!(hf_flat.abs() < 1e-6, "{hf_flat}");
            // Three columns, two rows: row 0 is luminance 1 (white), row 1
            // is 0 (black). With only two rows the vertical box holds both
            // rows at every position (a 4-tall box shrunk to 2), so every
            // low-passed value is the mean of the two rows' box columns:
            // 0.5 everywhere. |image - LP| is then 0.5 at every pixel.
            let stripe = ColumnSet {
                rows: vec![vec![Some([1.0, 1.0, 1.0]), Some([0.0, 0.0, 0.0])]; 3],
                skyline_rows: vec![0.0; 3],
            };
            let hf = high_frequency_energy(&stripe, 0..2, &lighting);
            assert!((hf - 0.5).abs() < 1e-6, "{hf}");
        }

        /// `Variant::Rings` is the schedule it is handed: it stops at the
        /// schedule's last ring (sky beyond, no far terrain) and reads each
        /// ring at `coarse_top` with the rule's own `v` -- option A's level
        /// 1 is 2 blocks tall, production's is 1.
        #[test]
        fn the_rings_variant_draws_the_schedule_at_the_rules_cell_height() {
            let gen = WorldGen::new(World::new().seed());
            let production = Variant::Rings {
                schedule: DEFAULT_RING_SCHEDULE,
                cell_height: production_cell_height,
            };
            let option_a = Variant::Rings {
                schedule: DEFAULT_RING_SCHEDULE,
                cell_height: option_a_cell_height,
            };
            assert_eq!(production.blocky_limit(), 1024.0);
            assert!(production.far().is_none());
            // 200 m out along +x: ring 1 (160-288 m), level 1.
            let (x, z) = (208, 0);
            let eye_xz = (8.0, 0.0);
            assert_eq!(
                production.blocky_face_height(&gen, eye_xz, x, z),
                Some(coarse_top(&gen, x, z, 2, 1) as f64 + 1.0)
            );
            assert_eq!(
                option_a.blocky_face_height(&gen, eye_xz, x, z),
                Some(coarse_top(&gen, x, z, 2, 2) as f64 + 1.0)
            );
            // Beyond the last ring: nothing (the sweep stops).
            assert_eq!(production.blocky_face_height(&gen, eye_xz, 1100, 0), None);
            // Under the finer-ring reference for the 160 m edge, 200 m out
            // is drawn at level 0: the exact column.
            let reference_schedule = finer_ring_schedule(DEFAULT_RING_SCHEDULE, 0);
            let reference = Variant::Rings {
                schedule: &reference_schedule,
                cell_height: production_cell_height,
            };
            assert_eq!(
                reference.blocky_face_height(&gen, eye_xz, x, z),
                Some(exact_face_height(&gen, x, z))
            );
        }

        /// Candidate (ii)'s far field agrees with the real production
        /// machinery it is built from: at a point well inside its hole,
        /// nothing is covered (`height_at` returns `None`), and just beyond
        /// it, the smooth height is within a few blocks of the exact one --
        /// footprint-mean, not a copy of the exact field.
        #[test]
        fn the_far_field_from_512_m_matches_the_production_far_terrain() {
            let gen = WorldGen::new(World::new().seed());
            let far = FarField::build(&gen, [8.0, 40.0, 8.0], 512.0, 2000.0);
            assert_eq!(far.height_at(8.0, 8.0), None, "inside the 512 m hole");
            let h = far.height_at(600.0, 8.0).expect("600 m out, past the hole");
            let exact = exact_face_height(&gen, 600, 8);
            assert!(
                (h - exact).abs() < 16.0,
                "footprint-mean {h} vs exact {exact}: too far apart to be the same hill"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubara_voxel::Mesh;
    use cubara_world::mesh::NodeGeometry;

    fn near(a: f32, b: f32) -> bool {
        (a - b).abs() <= 0.2
    }

    /// **T1, turned around (R1d, issue #327's Done-when).** Before R1b/R1c
    /// (#329) a coarse cell was `2^level` blocks tall, so every ring edge
    /// doubled the riser (`a_coarse_step_doubles_at_every_ring_edge_today`,
    /// this test's previous shape, pinned 6.4 -> 12.9 px at 160 m and so on).
    /// Since then a cell's height is `NodeKey::cell_height_blocks`: 1 block at
    /// levels 0-2, 4 at level 3. The 160 m and 288 m edges both bridge levels
    /// <= 2, so the cell height is 1 on either side of them: the step drawn
    /// just outside must be no taller on screen than the one just inside, at
    /// most 1.25x and at most 7 px (the Done-when bound).
    ///
    /// **The 512 m edge is measured here but not asserted.** It bridges level
    /// 2 (1 block) to level 3 (4 blocks), which still jumps 4x -- that is
    /// ring 3's own decision (`PROPOSAL_LOD_TRANSITIONS.md` §3 R1, candidate
    /// (i) level-3 `v = 4` versus candidate (ii) C+R5), owned by M4, not by
    /// this issue. Printing it keeps the number on record without this test
    /// pretending the decision has already been made.
    #[test]
    fn no_ring_edge_doubles_a_step_on_screen() {
        let steps = edge_steps(ViewSpec::gate(), DEFAULT_RING_SCHEDULE);
        assert_eq!(steps.len(), 3, "160 m, 288 m and 512 m");

        let asserted = [(160, 6.4), (288, 3.6)];
        for (s, (edge, px)) in steps[..2].iter().zip(asserted) {
            assert_eq!(s.edge.edge_blocks, edge);
            assert!(near(s.inside_px, px), "{edge} m inside {}", s.inside_px);
            assert!(near(s.outside_px, px), "{edge} m outside {}", s.outside_px);
            assert!(
                s.outside_px <= 1.25 * s.inside_px + 1e-3,
                "{edge} m: outside {} is more than 1.25x inside {}",
                s.outside_px,
                s.inside_px
            );
            assert!(
                s.outside_px <= 7.0,
                "{edge} m: outside {} exceeds 7 px",
                s.outside_px
            );
        }

        // The 512 m edge: level 2 (v = 1) meeting level 3 (v = 4). Recorded,
        // not asserted against the Done-when ratio -- see the doc comment.
        let edge_512 = &steps[2];
        assert_eq!(edge_512.edge.edge_blocks, 512, "it is the third edge");
        println!(
            "512 m (M4/ring 3 owns this, not asserted here): inside {:.1} px, outside {:.1} px",
            edge_512.inside_px, edge_512.outside_px
        );
    }

    /// **T4** (`PROPOSAL_LOD_TRANSITIONS.md` §6, issue #381 as amended):
    /// `ring_edges_match_a_finer_ring`, on the M4 CPU harness. For every
    /// case of `m4::t4_rows` (3 edges x 3 eyes x 2 yaws) the real
    /// schedule's band metric against the finer-ring reference is within
    /// 2.5x the reference-vs-reference noise (band max and band mean, each
    /// against its own), and the high-frequency energy ratio is within
    /// `[0.8, 1.25]` -- at the 160 m and 288 m edges. The same case under
    /// option A (`m4::option_a_cell_height`, `v = 2^level`, the state
    /// before R1) must fail by at least 2x its own threshold: the positive
    /// control that shows the metric can tell, built into the test. The
    /// 512 m edge is measured and printed, not asserted (ring 3's accepted
    /// 4x step, M4/#335). See the `m4` module's T4 section for the
    /// calibration table and the reasoning.
    ///
    /// Thresholds, band and columns are all in the pixels of the
    /// `ViewSpec` handed in -- the gate's here; #394 runs this over its
    /// viewport sweep by passing another.
    #[test]
    fn ring_edges_match_a_finer_ring() {
        let gen = WorldGen::new(World::new().seed());
        let view = ViewSpec::gate();
        let aspect = DEFAULT_SIZE.0 as f32 / DEFAULT_SIZE.1 as f32;
        let rows = m4::t4_rows(&gen, view, aspect);
        println!("\n{}", m4::format_t4(view, &rows));
        assert_eq!(rows.len(), 3 * 3 * 2, "3 edges x 3 eyes x 2 yaws");

        let mut asserted = 0;
        for r in &rows {
            let label = format!(
                "{} m edge, {} eye {:?}, yaw {}°",
                r.edge.edge_blocks, r.eye_name, r.eye, r.yaw_deg
            );
            if r.edge.edge_blocks == 512 {
                println!(
                    "{label}: measured, not asserted (ring 3 owns the 4x step): rows {:?}, R1 D \
                     {:.4} / {:.4} against threshold {:.4} / {:.4}, HF {:.3}; A D {:.4} / {:.4}",
                    r.production.band,
                    r.production.real.max,
                    r.production.real.mean,
                    r.production.threshold_max,
                    r.production.threshold_mean,
                    r.production.hf_ratio,
                    r.option_a.real.max,
                    r.option_a.real.mean,
                );
                continue;
            }
            let p = &r.production;
            assert!(
                p.within_threshold(),
                "{label}: the real schedule's band D {:.4} / {:.4} (max / mean) exceeds \
                 {:.4} / {:.4}, 2.5x the reference-vs-reference noise -- the coarser ring \
                 draws a step the finer one does not",
                p.real.max,
                p.real.mean,
                p.threshold_max,
                p.threshold_mean
            );
            assert!(
                p.hf_ratio_in_range(),
                "{label}: high-frequency energy ratio {:.3} outside {:?} -- a pattern- or \
                 texture-scale step at the edge",
                p.hf_ratio,
                m4::HF_RATIO_RANGE
            );
            asserted += 1;
        }
        assert_eq!(
            asserted, 12,
            "the 160 m and 288 m edges at every eye and yaw"
        );

        // The positive control: option A, the state before R1, goes red at
        // each asserted edge -- its worst case over eyes and yaws exceeds
        // its own threshold by `OPTION_A_MARGIN`. Without this, a noise
        // inflated enough to admit anything would pass the assertions
        // above unnoticed.
        for edge in [160, 288] {
            let (worst, label) = rows
                .iter()
                .filter(|r| r.edge.edge_blocks == edge)
                .map(|r| {
                    (
                        r.option_a.excess(),
                        format!("{} eye, yaw {}°", r.eye_name, r.yaw_deg),
                    )
                })
                .max_by(|a, b| a.0.total_cmp(&b.0))
                .expect("six cases per edge");
            println!("{edge} m: option A's worst case is the {label} at {worst:.2}x its threshold");
            assert!(
                worst >= m4::OPTION_A_MARGIN,
                "{edge} m edge: option A (v = 2^level) at its worst case (the {label}) scores \
                 only {worst:.2}x its own threshold, under the {}x margin -- either the metric \
                 went blind or the noise was inflated",
                m4::OPTION_A_MARGIN
            );
        }
    }

    #[test]
    fn a_step_shrinks_with_distance_and_grows_with_resolution() {
        let v = ViewSpec::gate();
        assert!((max_step_px(v, 0, 100.0) / max_step_px(v, 0, 200.0) - 2.0).abs() < 1e-4);
        let tall = ViewSpec {
            height_px: 2160,
            ..v
        };
        assert!((max_step_px(tall, 0, 100.0) / max_step_px(v, 0, 100.0) - 2.0).abs() < 1e-4);
        // Level 3's cell is 4 blocks tall (`NodeKey::cell_height_blocks`);
        // levels 0-2 are 1 block. Before R1b/R1c this ratio was 8.0 (`2^3`,
        // `2^level` regardless of level); now it is the cell-height ratio,
        // 4.0.
        assert!((max_step_px(v, 3, 100.0) / max_step_px(v, 0, 100.0) - 4.0).abs() < 1e-4);
    }

    /// A node holding one box of geometry whose index runs are `faces`
    /// (given in `Face` order: +x -x +y -y +z -z, stored in
    /// `Face::DRAW_ORDER` as a mesher would), 3 indices per triangle.
    fn node(level: u32, pos: [i32; 3], lo: [f32; 3], hi: [f32; 3], faces: [u32; 6]) -> BuiltNode {
        let mesh = Mesh {
            indices: vec![0; faces.iter().sum::<u32>() as usize],
            face_indices: cubara_voxel::Face::DRAW_ORDER.map(|f| faces[f as usize]),
            ..Mesh::default()
        };
        BuiltNode {
            node: NodeKey::new(level, pos),
            geometry: Some(NodeGeometry {
                mesh,
                aabb: Aabb::new(lo.into(), hi.into()),
                origin: lo.map(|v| v as i32),
                scale: 1.0,
            }),
            links: cubara_world::visibility::OPEN,
            trees: Vec::new(),
        }
    }

    #[test]
    fn counting_buckets_by_level_and_skips_what_cannot_be_drawn() {
        // The camera at the origin looks down +x (yaw 0), pitched down 0.25.
        let view = ViewSpec::gate();
        let f = frustum_at(view, [0.0, 0.0, 0.0], 0, 4);
        let built = vec![
            // Ahead, wholly in front of the camera: its -x faces point at it.
            // +x 6, -x 12, +y 0, -y 3, +z 0, -z 0 indices: it sees -x only
            // (and -y? the camera is above its box, so +y would face; it has
            // none) = 12 / 3 = 4 triangles.
            node(
                0,
                [0, 0, 0],
                [20.0, -10.0, -5.0],
                [30.0, -2.0, 5.0],
                [6, 12, 0, 3, 0, 0],
            ),
            // Same box, a level-1 node: counts under level 1.
            node(
                1,
                [0, 0, 0],
                [40.0, -10.0, -5.0],
                [60.0, -2.0, 5.0],
                [0, 9, 0, 0, 0, 0],
            ),
            // Behind the camera: culled by the frustum.
            node(
                0,
                [1, 0, 0],
                [-50.0, -10.0, -5.0],
                [-40.0, -2.0, 5.0],
                [3, 3, 3, 3, 3, 3],
            ),
            // In front but not reached by the visibility search.
            node(
                2,
                [0, 0, 0],
                [70.0, -10.0, -5.0],
                [90.0, -2.0, 5.0],
                [0, 30, 0, 0, 0, 0],
            ),
        ];
        let visible: HashSet<_> = built[..3].iter().map(|b| b.node).collect();
        let d = count_drawn(&built, &visible, &f);
        assert_eq!(
            d[0],
            Drawn {
                nodes: 1,
                triangles: 4
            }
        );
        assert_eq!(
            d[1],
            Drawn {
                nodes: 1,
                triangles: 3
            }
        );
        assert_eq!(d[2], Drawn::default(), "an unreached node draws nothing");
        assert_eq!(d[3], Drawn::default());
    }

    /// Every direction a different count, and the camera off to one side, so
    /// counting the wrong directions gives a different number: the mismatch
    /// between `Face` order and `Face::DRAW_ORDER` (#313) cannot hide here.
    #[test]
    fn the_faces_counted_are_the_ones_facing_the_camera() {
        let view = ViewSpec::gate();
        let f = frustum_at(view, [0.0, 0.0, 0.0], 0, 4);
        // Ahead (+x), below and to the +z side: the camera sees its -x, +y
        // and -z faces: 6 + 12 + 96 indices = 38 triangles.
        let built = vec![node(
            0,
            [0, 0, 0],
            [20.0, -10.0, 2.0],
            [30.0, -2.0, 12.0],
            [3, 6, 12, 24, 48, 96],
        )];
        let visible: HashSet<_> = built.iter().map(|b| b.node).collect();
        let d = count_drawn(&built, &visible, &f);
        assert_eq!(
            d[0],
            Drawn {
                nodes: 1,
                triangles: 38
            }
        );
    }

    /// The real pipeline on a world small enough for a debug build: the counts
    /// add up, and each level that exists draws something.
    #[test]
    fn a_tiny_world_counts_every_level() {
        let world = World::new();
        let registry = cubara_render::load_registry();
        let blocks = TerrainBlocks::from_registry(&registry);
        let schedule: &RingSchedule = &[(0, 2), (1, 4)];
        let eye = [8.0, 40.0, 8.0];
        let t = measure_eye(&world, blocks, eye, schedule, ViewSpec::gate(), 8);
        let nodes = desired_nodes_3d(
            ChunkCoord::from_world_pos(eye),
            VERTICAL_LOD_SQUASH,
            schedule,
        );
        let by_level = |l: u32| nodes.iter().filter(|n| n.level == l).count();
        assert_eq!(t.rows[0].desired, by_level(0));
        assert_eq!(t.rows[1].desired, by_level(1));
        assert_eq!(t.rows[0].desired + t.rows[1].desired, nodes.len());
        for l in 0..2 {
            assert!(t.rows[l].tris_avg > 0.0, "level {l} draws nothing");
            assert!(t.rows[l].reached <= t.rows[l].meshed);
            assert!(t.rows[l].meshed <= t.rows[l].desired);
            assert!(t.rows[l].tris_avg <= t.rows[l].tris_max as f64);
        }
        let sum: f64 = t.rows.iter().map(|r| r.tris_avg).sum();
        assert!((sum - t.total_tris_avg).abs() < 1e-6);
        assert!(format_m1("tiny", eye, &t).contains("| 1 |"));
    }

    fn site(x: i32, z: i32, base: i32, height: i32) -> TreeSite {
        TreeSite {
            x,
            z,
            base,
            height,
            canopy_radius: 2,
        }
    }

    #[test]
    fn trees_are_counted_per_level_and_sized_by_distance() {
        let view = ViewSpec::gate();
        // Looking down +x with the first frustum of four.
        let frustums = yaw_path(view, [0.0, 10.0, 0.0], 4);
        let sites = [
            (site(100, 0, 10, 8), 0),  // ahead, 100 m
            (site(200, 0, 10, 8), 1),  // ahead, 200 m
            (site(400, 0, 10, 8), 1),  // ahead, 400 m
            (site(-100, 0, 10, 8), 0), // behind
        ];
        let rows = tree_rows(&sites, &frustums, view);
        assert_eq!((rows[0].trees, rows[1].trees, rows[2].trees), (2, 2, 0));
        // Four yaws a quarter turn apart each see one of the four directions'
        // trees; the 90 degree spacing is wider than the 92 degree frustum
        // only barely, so every tree is seen in one frame.
        assert!((rows[0].in_view_avg - 0.5).abs() < 1e-9, "{rows:?}");
        assert!((rows[1].in_view_avg - 0.5).abs() < 1e-9);
        // 8 blocks at about 100 / 200 / 400 m: 8/d/0.97 mrad.
        assert!(rows[0].px_mean > 70.0 && rows[0].px_mean < 90.0, "{rows:?}");
        assert!(rows[1].px_max > 2.0 * rows[1].px_min * 0.9, "{rows:?}");
        assert!(rows[1].px_min < 22.0 && rows[1].px_max > 35.0, "{rows:?}");
    }

    #[test]
    fn real_trees_stand_in_the_nodes_they_are_tagged_with() {
        let world = World::new();
        let registry = cubara_render::load_registry();
        let blocks = TerrainBlocks::from_registry(&registry)
            .with_oak(&crate::game::load_structure_registry(), &registry);
        let schedule: &RingSchedule = &[(0, 6), (1, 12)];
        let center = ChunkCoord::new(0, 2, 0);
        let selected: HashSet<_> = desired_nodes_3d(center, VERTICAL_LOD_SQUASH, schedule)
            .into_iter()
            .collect();
        let sites = tree_sites(
            &WorldGen::new(world.seed()),
            blocks,
            center,
            &selected,
            schedule,
        );
        assert!(!sites.is_empty(), "a 25 x 25 chunk area has trees");
        assert!(
            sites.iter().any(|(_, l)| *l == 0) && sites.iter().any(|(_, l)| *l == 1),
            "both rings hold trees"
        );
        for (s, level) in &sites {
            let chunk = ChunkCoord::new(
                s.x.div_euclid(16),
                s.base.div_euclid(16),
                s.z.div_euclid(16),
            );
            assert!(selected.contains(&NodeKey::containing(chunk, *level)));
            assert!(
                *level == 0 || !selected.contains(&NodeKey::containing(chunk, 0)),
                "the finest node is the one reported"
            );
        }
        // A world without oaks has no tree sites.
        let bare = TerrainBlocks::from_registry(&registry);
        assert!(tree_sites(
            &WorldGen::new(world.seed()),
            bare,
            center,
            &selected,
            schedule
        )
        .is_empty());
    }

    /// The tables at the three gate eyes, 1080p, over 72 yaws (every 5
    /// degrees). Minutes in release; what the PR body prints.
    #[test]
    #[ignore = "meshes the whole 64-chunk world three times"]
    fn print_the_gate_tables() {
        println!("\n{}", report(72));
    }

    /// M2 around the ground eye: what R1 would cost in walls. The columns
    /// depend only on x and z, so the hill eye's table is the same. Seconds
    /// in release, no meshing.
    #[test]
    #[ignore = "a measurement: prints M2's table"]
    fn print_r1_wall_multipliers() {
        let ground = GATE_EYES[0];
        println!("\n{}", format_m2(ground.name, ground.eye));
    }

    /// M4 (issue #335): the band metric and skyline error deciding ring 3's
    /// own treatment, at the gate eyes and the low (F6 fallback) eye.
    /// Minutes in release -- it ray-marches a height field, not meshes, but
    /// the reference variant's blocky DDA runs to `m4::MAX_DISTANCE` with no
    /// far-terrain shortcut, at `m4::COLUMNS` azimuths, for every eye.
    #[test]
    #[ignore = "a measurement: prints M4's results table"]
    fn print_m4_results() {
        println!("\n{}", m4::report());
    }

    /// Risers are counted where quantized heights differ: block-height cells
    /// find strictly more than 4-block ones near spawn, some of them taller
    /// than a block (more bands than risers), and nodes of another level are
    /// not counted.
    #[test]
    fn finer_cells_find_more_risers_than_coarse_ones() {
        let gen = WorldGen::new(World::new().seed());
        let nodes: Vec<NodeKey> = (0..3).map(|x| NodeKey::new(2, [x, 0, 0])).collect();
        let (_, coarse) = riser_counts(&gen, &nodes, 2, 4);
        let (fine_edges, fine) = riser_counts(&gen, &nodes, 2, 1);
        assert!(coarse > 0, "the default world near spawn has relief");
        // Strictly more: block-height cells see the risers 4-block cells
        // round away. Equal would mean `v` was not applied.
        assert!(
            fine > coarse,
            "v = 1 found {fine} bands, v = 4 found {coarse}"
        );
        // Some risers are taller than a block, so carry more than one band.
        assert!(fine > fine_edges, "{fine} bands on {fine_edges} risers");
        // Nodes of another level are not counted at this one.
        assert_eq!(riser_counts(&gen, &nodes, 1, 1), (0, 0));
    }
}

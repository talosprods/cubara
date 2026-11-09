//! The measurements `docs/PROPOSAL_LOD_TRANSITIONS.md` is decided by: what the
//! voxel rings draw at each level (M1), how big a terrain step gets at a ring
//! edge (T1, as a measurement), and where the trees are (M6).
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

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use cubara_render::materials::TextureLayers;
use cubara_render::{faces_facing, Aabb, CameraUniform, Frustum};
use cubara_voxel::ChunkCoord;
use cubara_world::mesh::{build_node, BuiltNode};
use cubara_world::node::{desired_nodes_3d, NodeKey, RingSchedule, DEFAULT_RING_SCHEDULE};
use cubara_world::{TerrainBlocks, TreeSite, World, WorldGen};

use crate::bench::{DEFAULT_SIZE, GATE_EYES};
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
/// terrain step one cell of `level` tall, seen at `distance` blocks.
///
/// A coarse cell is `2^level` blocks tall (`worldgen.rs` judges a cell by its
/// middle), so wherever the terrain climbs, the riser it draws is that tall.
/// Just outside a ring edge this is the tallest step the ring draws.
///
/// **Today it doubles at every edge** (6.4 -> 12.9 px at 160 m). That is the
/// diagnosis in `docs/PROPOSAL_LOD_TRANSITIONS.md` section 1, pinned below.
/// **The fix (R1, block-height cells) turns this into the T1 assertion**:
/// `no_ring_edge_doubles_a_step_on_screen`, where the cell height is the
/// block's 1 and the outside figure may be at most 1.25x the inside one. It is
/// a function of the view and the ring edges, so that test needs no GPU.
pub fn max_step_px(view: ViewSpec, level: u32, distance: f32) -> f32 {
    (1u32 << level) as f32 / distance / view.pixel_angle()
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
pub fn format_m6(name: &str, eye: [f32; 3], rows: &[TreeRow; LEVELS]) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "M6 {name} eye {eye:?}");
    let _ = writeln!(
        s,
        "| level | grown there today | trees | in view (avg) | on-screen height px mean (min-max) |"
    );
    let _ = writeln!(s, "|---|---|---|---|---|");
    for (l, r) in rows.iter().enumerate() {
        if r.trees == 0 {
            continue;
        }
        let _ = writeln!(
            s,
            "| {l} | {} | {} | {:.0} | {:.1} ({:.1}-{:.1}) |",
            if l == 0 { "yes" } else { "no" },
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
        "| edge | cell inside -> outside (blocks) | step px inside -> outside |"
    );
    let _ = writeln!(s, "|---|---|---|");
    for e in edge_steps(view, schedule) {
        let _ = writeln!(
            s,
            "| {} m | {} -> {} | {:.1} -> {:.1} |",
            e.edge.edge_blocks,
            1 << e.edge.inside,
            1 << e.edge.outside,
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

#[cfg(test)]
mod tests {
    use super::*;
    use cubara_voxel::Mesh;
    use cubara_world::mesh::NodeGeometry;

    fn near(a: f32, b: f32) -> bool {
        (a - b).abs() <= 0.2
    }

    /// T1's three edges, as they are today. The R1 fix changes the outside
    /// figures to equal the inside ones; this is the test that then moves.
    #[test]
    fn a_coarse_step_doubles_at_every_ring_edge_today() {
        let steps = edge_steps(ViewSpec::gate(), DEFAULT_RING_SCHEDULE);
        let expect = [(160, 6.4, 12.9), (288, 7.2, 14.3), (512, 8.1, 16.1)];
        assert_eq!(steps.len(), expect.len());
        for (s, (edge, inside, outside)) in steps.iter().zip(expect) {
            assert_eq!(s.edge.edge_blocks, edge);
            assert!(near(s.inside_px, inside), "{edge} m inside {}", s.inside_px);
            assert!(
                near(s.outside_px, outside),
                "{edge} m outside {}",
                s.outside_px
            );
            assert!(
                (s.outside_px / s.inside_px - 2.0).abs() < 1e-3,
                "it doubles"
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
        assert!((max_step_px(v, 3, 100.0) / max_step_px(v, 0, 100.0) - 8.0).abs() < 1e-4);
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
                origin: lo,
                scale: 1.0,
            }),
            links: cubara_world::visibility::OPEN,
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
}

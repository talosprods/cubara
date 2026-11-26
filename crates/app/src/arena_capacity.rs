//! How big the chunk arena has to be (issue #434).
//!
//! It was a constant, tried once at the default settings. What it has to
//! hold is **the selection** -- every node [`node::desired_nodes_3d`] picks
//! for the active settings -- so the capacity is worked out from that same
//! selection (Rule 5: the streamer's own code, not a second opinion about
//! what is selected), when the settings are set, never per frame.
//!
//! Counting nodes is exact and cheap; what a node costs in vertices is not
//! known without meshing it. So the rule is the count of selected nodes per
//! level times [`COLUMN_COST`], what a node of that level costs on the steepest
//! ground found, times [`SWAP_OVERLAP`]; and what keeps the table honest is
//! a test that meshes the selection around seeded mountains at the extreme
//! settings and holds the answer against it
//! (`the_capacity_holds_the_worst_selection_at_every_setting`).

use cubara_render::ArenaCapacity;
use std::collections::HashSet;

use cubara_voxel::ChunkCoord;
use cubara_world::node::{self, NodeKey, RingSchedule};

use crate::streaming::VERTICAL_LOD_SQUASH;

/// How much more than the selection the arena holds, for the nodes that are
/// resident twice while levels swap: a node and what replaces it stay until
/// the replacement is up (`crate::uploads` in the renderer). **Chosen, not
/// derived**: a quarter more, the allowance #432 introduced and #423's test
/// held the arena to.
pub const SWAP_OVERLAP: f64 = 1.25;

/// Levels in the cost table: 0 to 3, the game's own. A level past it costs
/// what the last one does (a coarser node meshes the same lattice, not more).
const LEVELS: usize = 4;

/// What the terrain of one **column of nodes** of each level costs,
/// `(vertices, indices)`, at most, on the steepest ground found (the seeded
/// mountains of the capacity test). A column is one `(x, z)` position of
/// nodes of that level, however many of them stand above each other: the
/// ground is a surface, it crosses a column about once, and the nodes above
/// and below it mesh to nothing -- so what a column costs does not depend on
/// how many nodes the vertical squash stacks in it, and it is the quantity
/// that stays put when the squash changes (a node's own mean does not: it
/// doubles from squash 1 to 4). **Chosen, not derived**: the most any
/// column cost in the capacity test's measurement (per level, vertices
/// 6,448 / 4,440 / 5,947 / 3,722 over every setting and mountain; a mesh of
/// quads has 1.5 indices to a vertex), plus 6% and rounded up, so it is a
/// measurement with a margin and the test says when it stops being one.
/// It is the same at every squash, which is conservative for the coarse
/// ones: level 0 at squash 4 costs half of this, as the surface crosses
/// less of a flatter box.
const COLUMN_COST: [(u32, u32); LEVELS] = [
    (6_850, 10_275),
    (4_700, 7_050),
    (6_300, 9_450),
    (3_950, 5_925),
];

/// The chunk offsets the selection is counted around, to find the alignment
/// of the player against the octree that needs the most nodes: the count
/// differs by a few nodes with where the eye is in the coarsest node. Eight
/// corners and the middle of one node of the top level (8 chunks). **Chosen,
/// not derived**; the test counts every alignment of a smaller schedule and
/// holds this set to the true maximum.
const ALIGNMENTS: [(i32, i32, i32); 9] = [
    (0, 0, 0),
    (0, 0, 4),
    (0, 4, 0),
    (0, 4, 4),
    (4, 0, 0),
    (4, 0, 4),
    (4, 4, 0),
    (4, 4, 4),
    (2, 6, 3),
];

/// How many columns of nodes the selection has at each level: distinct
/// `(x, z)` positions among the nodes of the level (levels past the table's
/// last count with it).
pub fn selection_columns(nodes: &[NodeKey]) -> [u64; LEVELS] {
    let mut seen: HashSet<(usize, i32, i32)> = HashSet::new();
    let mut columns = [0u64; LEVELS];
    for n in nodes {
        let level = (n.level as usize).min(LEVELS - 1);
        if seen.insert((level, n.pos[0], n.pos[2])) {
            columns[level] += 1;
        }
    }
    columns
}

/// What the columns of one selection cost together: `(vertices, indices)`.
pub fn selection_cost(nodes: &[NodeKey]) -> (u64, u64) {
    selection_columns(nodes)
        .into_iter()
        .zip(COLUMN_COST)
        .fold((0, 0), |(v, i), (columns, (cv, ci))| {
            (v + columns * cv as u64, i + columns * ci as u64)
        })
}

/// What a selection by `schedule` and `squash` needs, before the floor and
/// the ceiling: `(vertices, indices)`, the worst of the selections around
/// the [`ALIGNMENTS`], times [`SWAP_OVERLAP`].
pub fn required_for(schedule: &RingSchedule, squash: i32) -> (u64, u64) {
    let (mut vertices, mut indices) = (0u64, 0u64);
    for (x, y, z) in ALIGNMENTS {
        let nodes = node::desired_nodes_3d(ChunkCoord::new(x, y, z), squash, schedule);
        let (v, i) = selection_cost(&nodes);
        (vertices, indices) = (vertices.max(v), indices.max(i));
    }
    let with_overlap = |n: u64| (n as f64 * SWAP_OVERLAP).ceil() as u64;
    (with_overlap(vertices), with_overlap(indices))
}

/// The arena for a selection by `schedule` and `squash`: [`required_for`],
/// from [`ArenaCapacity::FLOOR`] up to what one buffer can be.
pub fn capacity_for(schedule: &RingSchedule, squash: i32) -> ArenaCapacity {
    let (vertices, indices) = required_for(schedule, squash);
    ArenaCapacity::new(vertices, indices)
}

/// The arena the game itself needs: its ring schedule and its vertical
/// squash. The game's settings do not change the selection today; when one
/// does (the start-up check of #434's context), this is called again with
/// the new schedule and `Renderer::set_arena_capacity` is called with it.
pub fn game_capacity() -> ArenaCapacity {
    capacity_for(node::DEFAULT_RING_SCHEDULE, VERTICAL_LOD_SQUASH)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubara_render::ChunkArena;
    use cubara_voxel::BlockRegistry;
    use cubara_world::mesh::mesh_node;
    use cubara_world::{TerrainBlocks, World};

    /// The settings the selection depends on, at their extremes, and how
    /// many of the steepest seeded mountains each is meshed around:
    /// `(render distance in chunks, vertical squash, mountains)`. The render
    /// distance is what `schedule_for_radius` turns into rings (64 is the
    /// schedule's own outer edge, a radius over it asks for no more; 10 is
    /// the first ring alone, with no coarser level). The squash is
    /// `--squash`, any whole number from 1: 1 is the most detail above and
    /// below the eye and so the most nodes, 2 is the game's
    /// (`VERTICAL_LOD_SQUASH`), 4 the coarsest its study measured.
    /// **Chosen, not derived**: there is no options screen for either yet
    /// (the far terrain's quality is the only choice there, and it does not
    /// change what the arena holds), so these are the extremes of what the
    /// code accepts, with the game's own setting between them; the largest
    /// gets the most mountains. (Squash 4 and radius 32 were measured too,
    /// and are what `COLUMN_COST` was read off; they are left out here for
    /// the minutes they add, each is one more mesh of about 21,000 nodes.)
    const SETTINGS: [(i32, i32, usize); 4] = [(64, 1, 2), (64, 2, 1), (18, 1, 1), (10, 1, 1)];

    /// The swap allowance the tests hold the capacity to: a quarter more than
    /// the selection (#432's value), written again here and not read from
    /// [`SWAP_OVERLAP`], so lowering the one the arena is built with is a
    /// change this test sees. **Chosen, not derived.**
    const SWAP_ALLOWANCE: f64 = 1.25;

    /// The arena before #434, to show what the largest setting needs
    /// against it.
    const OLD_VERTEX_CAPACITY: u64 = 8_000_000;
    const OLD_INDEX_CAPACITY: u64 = 12_000_000;

    /// What meshing a whole selection around one eye came to.
    #[derive(Default, Clone, Copy)]
    struct Measured {
        vertices: u64,
        indices: u64,
        /// Nodes that mesh to something, which are the ones that take an
        /// arena slot.
        non_empty: u64,
        /// Nodes the selection had, with or without geometry.
        selected: u64,
        /// Per level 0..=3: nodes selected, vertices, indices.
        levels: [(u64, u64, u64); 4],
        /// [`selection_columns`] of what was meshed.
        columns: [u64; 4],
    }

    /// The steepest chunk columns of a seeded draw of the default world --
    /// where the selection is largest (the method of #423's test).
    struct Mountains {
        world: World,
        registry: BlockRegistry,
        blocks: TerrainBlocks,
        columns: Vec<(i32, i32)>,
    }

    impl Mountains {
        fn seeded(centres: usize) -> Self {
            const CANDIDATES: usize = 64;
            const SEED: u64 = 0x0423_A7E4_A000_0001;
            let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
            let registry = BlockRegistry::load(&root.join("assets/blocks")).expect("assets/blocks");
            let structures = cubara_voxel::StructureRegistry::load(&root.join("assets/structures"))
                .expect("assets/structures");
            let ores =
                cubara_voxel::OreRegistry::load(&root.join("assets/ores")).expect("assets/ores");
            let blocks = TerrainBlocks::from_registry(&registry)
                .with_oak(&structures, &registry)
                .with_ores(&ores, &registry);
            let world = World::new();
            // SplitMix64, so the columns are the seed's and not picked.
            let mut state = SEED;
            let mut next = || {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^ (z >> 31)
            };
            // How steep the ground is around a column: the mean step between
            // surface heights 16 blocks apart, over the 512 blocks either
            // side (the three finest rings, where the triangles are).
            let steepness = |cx: i32, cz: i32| -> i64 {
                let at =
                    |i: i32, j: i32| world.surface_height((cx + i) * 16 + 8, (cz + j) * 16 + 8);
                let mut sum = 0i64;
                for j in -32..32 {
                    for i in -32..32 {
                        sum += ((at(i + 1, j) - at(i, j)).abs() + (at(i, j + 1) - at(i, j)).abs())
                            as i64;
                    }
                }
                sum
            };
            let mut columns: Vec<(i64, i32, i32)> = (0..CANDIDATES)
                .map(|_| {
                    let (cx, cz) = ((next() % 750) as i32 - 375, (next() % 750) as i32 - 375);
                    (steepness(cx, cz), cx, cz)
                })
                .collect();
            columns.sort();
            columns.reverse();
            let columns = columns[..centres].iter().map(|&(_, x, z)| (x, z)).collect();
            Self {
                world,
                registry,
                blocks,
                columns,
            }
        }

        /// Mesh every node the game selects for `squash` and `schedule`
        /// around the ground of `column`, as a mesh job meshes it.
        fn measure(&self, column: (i32, i32), squash: i32, schedule: &RingSchedule) -> Measured {
            let ground = self
                .world
                .surface_height(column.0 * 16 + 8, column.1 * 16 + 8)
                + 2;
            let centre = ChunkCoord::new(column.0, ground.div_euclid(16), column.1);
            let nodes = node::desired_nodes_3d(centre, squash, schedule);
            let columns = selection_columns(&nodes);
            let workers = std::thread::available_parallelism().map_or(1, |n| n.get());
            std::thread::scope(|scope| {
                // Every `workers`-th node each: the list is ordered by
                // level, so a contiguous split would hand one worker all of
                // the heavy ones.
                let handles: Vec<_> = (0..workers)
                    .map(|k| {
                        let nodes = &nodes;
                        scope.spawn(move || {
                            let layer_of = |_: &str| 0;
                            let mut m = Measured::default();
                            for &n in nodes.iter().skip(k).step_by(workers) {
                                m.levels[n.level as usize].0 += 1;
                                if let Some(g) = mesh_node(
                                    &self.world,
                                    &self.registry,
                                    &layer_of,
                                    n,
                                    self.blocks,
                                ) {
                                    m.vertices += g.mesh.vertices.len() as u64;
                                    m.indices += g.mesh.indices.len() as u64;
                                    m.non_empty += 1;
                                    m.levels[n.level as usize].1 += g.mesh.vertices.len() as u64;
                                    m.levels[n.level as usize].2 += g.mesh.indices.len() as u64;
                                }
                            }
                            m
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("a mesh worker"))
                    .fold(
                        Measured {
                            columns,
                            selected: nodes.len() as u64,
                            ..Measured::default()
                        },
                        |a, b| {
                            let mut levels = a.levels;
                            for (l, other) in levels.iter_mut().zip(b.levels) {
                                *l = (l.0 + other.0, l.1 + other.1, l.2 + other.2);
                            }
                            Measured {
                                vertices: a.vertices + b.vertices,
                                indices: a.indices + b.indices,
                                non_empty: a.non_empty + b.non_empty,
                                selected: a.selected,
                                levels,
                                columns: a.columns,
                            }
                        },
                    )
            })
        }
    }

    /// **The capacity holds the selection around the steepest ground, at
    /// every setting** (issue #434): every node the game selects for the
    /// setting is meshed as a mesh job meshes it, around the seeded
    /// mountains, and the capacity [`capacity_for`] gives -- the one the game
    /// builds its arena with -- must hold the vertices and indices of that,
    /// with the swap allowance over it, and the nodes that took a slot must
    /// fit the node slots. Every node is counted, seen or not: the game
    /// uploads only what its visibility search reaches, so this is more than
    /// is ever resident -- an upper bound for an eye on the ground, and not
    /// one by construction for an eye raised above the mountain, where
    /// nearly everything selected is in sight (#423's measurement: 3.46
    /// million resident against 4.28 million selected then).
    #[test]
    fn the_capacity_holds_the_worst_selection_at_every_setting() {
        let ground = Mountains::seeded(3);
        let mut largest = Measured::default();
        for (radius, squash, mountains) in SETTINGS {
            let schedule = node::schedule_for_radius(radius);
            let (need_v, need_i) = required_for(&schedule, squash);
            assert!(
                !ArenaCapacity::needs_more_than_a_buffer(need_v, need_i),
                "radius {radius}, squash {squash} needs {need_v} vertices and {need_i} indices: \
                 more than one buffer can be ({:?})",
                ArenaCapacity::CEILING
            );
            let capacity = capacity_for(&schedule, squash);
            for &column in &ground.columns[..mountains] {
                let m = ground.measure(column, squash, &schedule);
                println!(
                    "radius {radius} squash {squash} around {column:?}: {} vertices, {} indices, {} nodes with geometry \
                     of {} selected; capacity {} vertices, {} indices",
                    m.vertices, m.indices, m.non_empty, m.selected, capacity.vertices, capacity.indices
                );
                for (what, used, room) in [
                    ("vertices", m.vertices, capacity.vertices),
                    ("indices", m.indices, capacity.indices),
                ] {
                    assert!(
                        used as f64 * SWAP_ALLOWANCE <= room as f64,
                        "radius {radius}, squash {squash}, around {column:?}: the selection has \
                         {used} {what}; with {SWAP_ALLOWANCE} for a swap that is over the capacity {room}"
                    );
                }
                assert!(
                    m.non_empty as f64 * SWAP_ALLOWANCE <= ChunkArena::NODES_CAPACITY as f64,
                    "radius {radius}, squash {squash}, around {column:?}: {} nodes with geometry, \
                     with a swap more than the {} node slots",
                    m.non_empty,
                    ChunkArena::NODES_CAPACITY
                );
                if (radius, squash) == (64, 1) && m.vertices > largest.vertices {
                    largest = m;
                }
            }
        }
        // The largest setting is what the old constant had to hold: what it
        // takes there, against the 8 and 12 million the arena had.
        println!(
            "largest setting, 64 chunks at squash 1: {} vertices x {SWAP_ALLOWANCE} = {:.0} of the old {OLD_VERTEX_CAPACITY}; {} indices x {SWAP_ALLOWANCE} = {:.0} of the old {OLD_INDEX_CAPACITY}",
            largest.vertices,
            largest.vertices as f64 * SWAP_ALLOWANCE,
            largest.indices,
            largest.indices as f64 * SWAP_ALLOWANCE,
        );
    }

    /// The eye's place in the coarsest node changes how many nodes a
    /// selection has (a few percent); the capacity is for the alignment with
    /// the most, which [`ALIGNMENTS`] must find: every one of the 8 x 8 x 8
    /// the top level has costs no more than the worst of those counted.
    #[test]
    fn the_alignments_counted_include_the_one_with_the_most_nodes() {
        for (radius, squash) in [(64, 1), (64, 2), (64, 4), (32, 1), (18, 2)] {
            let schedule = node::schedule_for_radius(radius);
            let counted = required_for(&schedule, squash);
            let mut most = (0u64, 0u64);
            for x in 0..8 {
                for y in 0..8 {
                    for z in 0..8 {
                        let nodes =
                            node::desired_nodes_3d(ChunkCoord::new(x, y, z), squash, &schedule);
                        let (v, i) = selection_cost(&nodes);
                        most = (most.0.max(v), most.1.max(i));
                    }
                }
            }
            let allowed = |n: u64| (n as f64 * SWAP_ALLOWANCE).ceil() as u64;
            assert!(
                (allowed(most.0), allowed(most.1)) <= counted,
                "radius {radius}, squash {squash}: an alignment of the eye needs {most:?}, \
                 more than the {counted:?} counted"
            );
        }
    }

    /// A larger render distance never needs a smaller arena, and the game's
    /// own is between the floor and what one buffer can be.
    #[test]
    fn a_larger_selection_needs_no_smaller_an_arena() {
        let mut last = (0, 0);
        for radius in [4, 10, 14, 18, 24, 32, 48, 64] {
            let need = required_for(&node::schedule_for_radius(radius), 2);
            assert!(
                need >= last,
                "radius {radius} needs {need:?}, less than {last:?}"
            );
            last = need;
        }
        let game = game_capacity();
        assert!(game.vertices >= ArenaCapacity::FLOOR.vertices);
        assert!(game.vertices <= ArenaCapacity::CEILING.vertices);
        assert!(game.indices <= ArenaCapacity::CEILING.indices);
        // A tiny setting still gets the floor, never an arena a few edits fill.
        assert_eq!(
            capacity_for(&node::schedule_for_radius(1), 2),
            ArenaCapacity::FLOOR
        );
    }

    /// Prints what a node of each level costs where the ground is steepest:
    /// the measurement `COLUMN_COST` is read off. Not a check (the next test
    /// is), so it does not run unless asked.
    #[test]
    #[ignore = "prints the measurement COLUMN_COST is read off"]
    fn print_what_a_node_costs() {
        let ground = Mountains::seeded(2);
        for (radius, squash, _) in SETTINGS {
            let schedule = node::schedule_for_radius(radius);
            for &column in &ground.columns {
                let m = ground.measure(column, squash, &schedule);
                println!(
                    "radius {radius} squash {squash} column {column:?}: {} vertices {} indices {} non-empty; per level (nodes, vertices, indices): {:?}; columns {:?}",
                    m.vertices, m.indices, m.non_empty, m.levels, m.columns
                );
            }
        }
    }
}

//! The distance morph between levels of detail: CDLOD geomorphing (Strugar,
//! "Continuous Distance-Dependent Level of Detail for Rendering Heightmaps",
//! 2009) on the voxel height field. **A study prototype** (issue #380, the
//! amendment of 2026-11-20): good enough to measure, not yet a decided
//! design.
//!
//! **Why it applies here.** Since issue #423 every coarse level draws the
//! ground as one top per cell (`WorldGen::cell_top`), and a cell of level
//! `L + 1` covers four of level `L`. So a level-`L` node whose column tops
//! are moved to their parent cell's top *is* the parent's geometry: the
//! risers inside a parent cell shrink to nothing, the ones on its edges
//! become the parent's, and nothing moves sideways. CDLOD moves every
//! vertex to where the coarser level puts it over a band of distance before
//! the switch, so that when the switch comes the two meshes are the same and
//! the swap shows nothing.
//!
//! **This module is the pure half**: where each level's band lies
//! ([`morph_bands`]), the factor ([`morph_factor`] over [`morph_metric`]),
//! and each column's target ([`node_columns`]). The mesher bakes a target
//! per vertex (`cubara_voxel::MorphColumns`) and `mesh.wgsl` moves it by
//! the factor -- a copy of [`morph_factor`] the shader must keep, as
//! `aggregate.rs`'s CPU twins do for its lighting.
//!
//! **What does not morph** (and so still swaps as today): a column with a
//! cave mouth or an overhang among the rows it shows, or with anything
//! standing on it (a tree, a player's block) -- and every column beside
//! one, so that a moving face never shares an edge with a face that stays.
//! **A pit morphs**: its floor is its column's top and its wall a riser
//! between two tops (`WorldGen::column_top`, `docs/PHASE3_ARCHITECTURE.md`
//! §9.9), so a node's pit slides to its parent's wider, shallower one like
//! any slope -- but for a pit that widens under its rim, whose wall is an
//! overhang. Level 3, whose next level is the far terrain (#422's join). And
//! no column morphs across a floor or ceiling of its parent level's nodes
//! ([`node_columns`]): there the node across may be a coarser one that draws
//! none of this level's faces, so a column whose parent stands across such
//! a floor stops at it, and swaps the rest.

use std::f32::consts::SQRT_2;

use cubara_voxel::MorphColumns;

use crate::node::{NodeKey, RingSchedule, NEAR_FIELD_CHUNKS};
use crate::{TerrainBlocks, WorldSnapshot};

/// Blocks per chunk, as a float.
const CHUNK: f32 = 16.0;

/// The levels whose nodes morph: those whose next coarser level is a voxel
/// level of the game's schedule. Level 3 meets the far terrain, whose join
/// is #422's.
pub const MORPH_LEVELS: std::ops::RangeInclusive<u32> = 0..=2;

/// The widest band, in blocks of the morph metric. **Chosen, not derived**:
/// wider spreads a change over more frames, and leaves less of the ring at
/// its own detail; 128 blocks is 5.3 s at the flight speed of 24 blocks/s.
/// At the default schedule level 0's band is [`NEAR_EXACT`] to its ring's
/// end (see there); levels 1 and 2 get less than this anyway (32 and 64
/// blocks): [`morph_bands`] gives each level all the room its ring leaves.
pub const MAX_BAND: f32 = 128.0;

/// How far from the eye the finest level's terrain stands exactly on the
/// block grid: level 0's band never starts nearer, in blocks of the metric
/// -- and, since the metric counts height once out to here
/// ([`morph_metric`]), in blocks of true distance too, in every direction.
/// **Chosen, not derived**: near the player a block is tens to hundreds of
/// pixels tall, so any fraction of one shows -- a band that started at the
/// eye (as #380's first one did, 0 to 128) moved a vertex at 20 blocks 16%
/// of the way to its parent's height, and the owner's play test of
/// 2026-11-25 saw thin slabs on the terraces and slivers at the foot of
/// walls. The value is the measurement of that day (`docs/PHASE3_ARCHITECTURE.md`
/// §9.6.1): the transition study at the 160 m edge, against today's band.
/// It is the selection's near field ([`NEAR_FIELD_CHUNKS`]), in blocks: one
/// number, so the morph and the selection cannot disagree about it.
/// [`morph_bands`] keeps it at most three quarters of the way to the band's
/// end, so a short ring (a weak PC's) still has a band.
pub const NEAR_EXACT: f32 = CHUNK * NEAR_FIELD_CHUNKS as f32;

/// A level's morph band, in blocks of [`morph_metric`]: its vertices start
/// moving to their parent's height at `start` and have arrived at `end`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MorphBand {
    pub start: f32,
    pub end: f32,
}

/// How far before its ring edge a level's band must end, in blocks.
/// **Derived**: the selection is made from the chunk the eye is in
/// (`node::select_nodes`), so a node swaps while the eye is anywhere in that
/// chunk -- up to a chunk's diagonal nearer than the ring's radius says
/// horizontally (`16 * sqrt 2`), and with the vertical squash `16 * squash`
/// in the metric. A vertex of a node about to swap is never nearer than
/// that, so its morph has finished.
pub fn outer_margin(squash: i32) -> f32 {
    CHUNK * SQRT_2.max(squash.max(1) as f32)
}

/// How far past the inner edge of its ring a level-`level` node reaches, in
/// blocks of the metric. **Derived**: a node is split when its *nearest*
/// chunk is within the finer ring's radius, so the node beside a finer one
/// can reach a whole node further: its diagonal horizontally, its height
/// (times the squash) vertically. Its vertices there must not have started
/// to morph, because the finer node beside it ends its own morph on exactly
/// this node's unmorphed geometry.
pub fn inner_reach(level: u32, squash: i32) -> f32 {
    CHUNK * (1u32 << level) as f32 * SQRT_2.max(squash.max(1) as f32)
}

/// Each level's morph band for `schedule` (indexed by level, four levels):
/// ending [`outer_margin`] before the level's ring edge, and starting at most
/// [`MAX_BAND`] before that but never closer than [`inner_reach`] past the
/// ring's inner edge (for level 0: than [`NEAR_EXACT`] from the eye). `None` for the last ring (its join is the far
/// terrain's), for a level the schedule does not have, and for a ring with
/// no room for a band at all -- whose swap the morph cannot hide.
pub fn morph_bands(schedule: &RingSchedule, squash: i32) -> [Option<MorphBand>; 4] {
    let mut bands = [None; 4];
    for (k, &(level, radius)) in schedule.iter().enumerate() {
        if k + 1 >= schedule.len() || level as usize >= bands.len() {
            continue;
        }
        let end = CHUNK * radius as f32 - outer_margin(squash);
        let floor = match k {
            0 => NEAR_EXACT.min(0.75 * end),
            _ => CHUNK * schedule[k - 1].1 as f32 + inner_reach(level, squash),
        };
        let start = (end - MAX_BAND).max(floor);
        if start < end {
            bands[level as usize] = Some(MorphBand { start, end });
        }
    }
    bands
}

/// The morph metric of a vertex at `rel` (its position minus the eye's, in
/// blocks): the selection's own distance (`node::node_distance_components`),
/// continuous -- horizontal distance, or the vertical one as the selection
/// counts it ([`squashed_height`]), whichever is larger.
///
/// **Never more than the true distance inside the near field**: height
/// counts once out to [`NEAR_EXACT`], so a point fewer than [`NEAR_EXACT`]
/// blocks from the eye in a straight line has a metric under it, whatever
/// the direction -- and level 0's band starts there. With height counted
/// `squash` times all the way to the eye, ground 75 blocks under a peak was
/// at metric 150 and drawn at its parent's heights (§9.6.2).
pub fn morph_metric(rel: [f32; 3], squash: i32) -> f32 {
    rel[0]
        .hypot(rel[2])
        .max(squashed_height(rel[1].abs(), squash))
}

/// A vertical distance of `v` blocks (`>= 0`) as the morph metric counts it:
/// [`crate::node::squashed_gap`] continued between chunks,
/// `max(v, squash * (v - 16 * shift))` with [`crate::node::vertical_shift`]'s
/// `shift` -- `v` itself out to [`NEAR_EXACT`] at the game's squash, twice
/// as fast beyond. `morph.wgsl`'s `morph_factor` does exactly this, with the
/// shift in blocks from `MorphParams::shift` ([`vertical_shift_blocks`]).
pub fn squashed_height(v: f32, squash: i32) -> f32 {
    let s = squash.max(1) as f32;
    v.max(s * (v - vertical_shift_blocks(squash)))
}

/// [`crate::node::vertical_shift`] in blocks: how far every ring's vertical reach
/// lies further from the eye than a plain squash would put it. What the
/// renderer is handed with the squash (`MorphParams::shift`).
pub fn vertical_shift_blocks(squash: i32) -> f32 {
    CHUNK * crate::node::vertical_shift(squash) as f32
}

/// How far a vertex at metric `metric` has moved to its parent's height: 0
/// before the band, 1 after it, linear in between. `mesh.wgsl`'s `vs_main`
/// does exactly this.
pub fn morph_factor(metric: f32, band: MorphBand) -> f32 {
    ((metric - band.start) / (band.end - band.start).max(f32::MIN_POSITIVE)).clamp(0.0, 1.0)
}

/// How many rings of columns around a node [`node_columns`] looks at: the
/// ring the mesher reads, the ring whose purity that needs, and its
/// neighbours' tops.
const PAD: i32 = 3;

/// `node`'s columns and the ring just outside them, as the mesher wants them
/// ([`MorphColumns`], node-local face rows): each column's drawn top, where
/// it is at full morph, and -- in the ring -- the lowest top a neighbour may
/// show there. `None` for a node that does not morph ([`MORPH_LEVELS`]).
///
/// Everything here is a property of the column, worked out from the world
/// alone and never from the node's own blocks, so that two nodes that meet
/// -- side by side, or one above the other where a riser crosses from one
/// into the next -- move a shared edge identically.
///
/// A column **morphs** where it and its eight neighbours are *pure*: solid
/// in every row it shows to a neighbour and the row under them (no cave
/// mouth or overhang there), with nothing standing on it at level 0 (no
/// tree, no player's block). A column's top is what the sky sees of it
/// (`WorldGen::column_top`): the floor of a pit is a top, and a pit with
/// whole walls is pure. Its target is then its parent cell's top,
/// held inside the rows of the parent-level node that holds its own top --
/// and a top lying on such a node's floor keeps its height (see the body);
/// any other column's target is its own top.
pub fn node_columns(
    world: &WorldSnapshot,
    node: NodeKey,
    blocks: TerrainBlocks,
) -> Option<MorphColumns> {
    if !MORPH_LEVELS.contains(&node.level) {
        return None;
    }
    let level = node.level;
    let w = node.extent_chunks();
    let n = 16i32;
    let [ox, oy, oz] = node.world_origin();
    let side = n + 2 * PAD;
    let at = |cx: i32, cz: i32| ((cz + PAD) * side + (cx + PAD)) as usize;
    // The y of the top block a cell `width` blocks wide at corner (x, z)
    // draws: `cell_top`, which is `surface_height` at width 1.
    let top_of = |x: i32, z: i32, width: i32| world.cell_surface_height(x, z, width);
    let corner = |cx: i32, cz: i32| (ox + cx * w, oz + cz * w);

    let mut tops = vec![0i32; (side * side) as usize];
    for cz in -PAD..n + PAD {
        for cx in -PAD..n + PAD {
            let (x, z) = corner(cx, cz);
            tops[at(cx, cz)] = top_of(x, z, w);
        }
    }
    let parent_top = |cx: i32, cz: i32| {
        let (x, z) = corner(cx, cz);
        let pw = 2 * w;
        top_of(x.div_euclid(pw) * pw, z.div_euclid(pw) * pw, pw)
    };

    // Purity, two rings out: solid from the lowest neighbour's top to its own.
    let ring = PAD - 1;
    let mut pure = vec![false; (side * side) as usize];
    for cz in -ring..n + ring {
        for cx in -ring..n + ring {
            let top = tops[at(cx, cz)];
            let lowest = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                .iter()
                .map(|&(dx, dz)| tops[at(cx + dx, cz + dz)])
                .fold(top, i32::min);
            let (x, z) = corner(cx, cz);
            pure[at(cx, cz)] = if level == 0 {
                // (`top` for the column's surface: under a pit the top is
                // the floor, and at or under a column's top the two give
                // the same answer.)
                (lowest..=top).all(|y| world.certainly_solid_at(x, y, z, top))
                    && !world.is_solid_at(x, top + 1, z, blocks)
            } else {
                let cell = world.cell_ground(x, z, w);
                (lowest..=top).all(|y| world.certainly_solid_at_v(y, level, &cell))
            };
        }
    }
    let morphs =
        |cx: i32, cz: i32| (-1..=1).all(|dz| (-1..=1).all(|dx| pure[at(cx + dx, cz + dz)]));

    let row = |y_top_block: i32| y_top_block + 1 - oy;
    // No column morphs across a floor or ceiling of its parent level's
    // nodes (multiples of their height, `32 * w` blocks): its target is its
    // parent's top clamped into the rows of the parent-level node that holds
    // its own top. The node across such a floor may be drawn a level coarser
    // than this one and not moving (at the top and bottom of a ring, which
    // the vertical squash makes flat boxes, so a ring's floor cuts a slope
    // along a contour) -- and a top that crossed into it would be drawn by
    // nobody there: the node it came from shows it at its morphed height
    // only in its own rows, the coarse node shows its own ground and nothing
    // of this level's. What is left is a step the swap shows, on the columns
    // whose parent stands across such a floor, instead of a hole. Inside one
    // parent-level node the floors between siblings are always crossed by
    // nodes of one level, which move a shared top alike, so a column still
    // morphs all the way there.
    //
    // A top lying exactly on such a floor does not move at all. It is the
    // node below's, but the risers of the columns beside it in the node
    // above stand on it, and their feet follow it: down into the node below,
    // which may be the coarse one that draws none of this level's risers.
    let parent_rows = 32 * w;
    let morph_target = |top: i32, parent: i32| {
        if (top + 1).rem_euclid(parent_rows) == 0 {
            return top;
        }
        let k = top.div_euclid(parent_rows);
        parent.clamp(k * parent_rows - 1, (k + 1) * parent_rows - 1)
    };
    let mut columns = Vec::with_capacity(((n + 2) * (n + 2)) as usize);
    let mut morphing = Vec::with_capacity(((n + 2) * (n + 2)) as usize);
    for cz in -1..=n {
        for cx in -1..=n {
            let top = tops[at(cx, cz)];
            let parent = parent_top(cx, cz);
            morphing.push(morphs(cx, cz));
            let target = if morphs(cx, cz) {
                morph_target(top, parent)
            } else {
                top
            };
            let inside = (0..n).contains(&cx) && (0..n).contains(&cz);
            let (floor, target_floor) = if inside {
                (top, target)
            } else {
                // A neighbour outside the node may be drawn a level coarser
                // (its parent's top) or, above level 0, a level finer: the
                // two finer cells of it that touch the node.
                let mut floor = top.min(parent);
                if level > 0 {
                    let half = w / 2;
                    let (x, z) = corner(cx, cz);
                    let finer: [(i32, i32); 2] = if cx < 0 {
                        [(x + half, z), (x + half, z + half)]
                    } else if cx >= n {
                        [(x, z), (x, z + half)]
                    } else if cz < 0 {
                        [(x, z + half), (x + half, z + half)]
                    } else {
                        [(x, z), (x + half, z)]
                    };
                    for (fx, fz) in finer {
                        floor = floor.min(top_of(fx, fz, half));
                    }
                }
                (floor, target.min(parent))
            };
            columns.push([row(top), row(target), row(floor), row(target_floor)]);
        }
    }
    Some(MorphColumns {
        n,
        level,
        columns,
        morphs: morphing,
        parent: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{desired_nodes_3d, DEFAULT_RING_SCHEDULE};
    use cubara_voxel::{
        riser_map, BlockRegistry, ChunkCoord, DropRule, Faces, Interact, Material, Shape,
    };

    /// The game's vertical squash (`cubara-app`'s `VERTICAL_LOD_SQUASH`).
    const SQUASH: i32 = 2;

    /// The terrain's three materials by name (`TerrainBlocks::from_registry`),
    /// no trees: what `node_columns` asks `is_solid_at` about.
    fn blocks() -> TerrainBlocks {
        TerrainBlocks::from_registry(&registry())
    }

    fn registry() -> BlockRegistry {
        let material = |name: &str| {
            (
                std::path::PathBuf::from("test-fixture.ron"),
                Material {
                    name: name.to_string(),
                    solid: true,
                    faces: Faces::All(name.to_string()),
                    shapes: vec![Shape::Full],
                    drops: DropRule::SameName,
                    requires_tier: 0,
                    hardness: Some(1),
                    interact: Interact::None,
                },
            )
        };
        BlockRegistry::from_materials(vec![
            material("cubara:grass"),
            material("cubara:soil"),
            material("cubara:stone"),
        ])
        .expect("fixture registry is valid")
    }

    /// The nodes of a morphing `level` around the steep mountain of #432
    /// (chunk column (336, 213)): the one holding the surface there and its
    /// eight neighbours on the same row.
    fn mountain_nodes(snapshot: &WorldSnapshot, level: u32) -> Vec<NodeKey> {
        let (px, pz) = (336 * 16, 213 * 16);
        let extent = 16 << level;
        let mut nodes = Vec::new();
        for dz in -1..=1 {
            for dx in -1..=1 {
                let (x, z) = (px + dx * extent, pz + dz * extent);
                let y = snapshot.surface_height(x, z);
                nodes.push(NodeKey::new(
                    level,
                    [
                        x.div_euclid(extent),
                        y.div_euclid(extent),
                        z.div_euclid(extent),
                    ],
                ));
            }
        }
        nodes
    }

    /// **The mesh the game draws is closed at every factor, and at full
    /// morph it is the parent's** -- the mesher's own output, not the rule:
    /// real nodes of every morphing level on the steep mountain are meshed,
    /// every vertex is moved by one factor `f` as `mesh.wgsl` moves it, and
    /// then, for every column whose riser rows lie inside the node:
    ///
    /// - its top face stands at `lerp(top, target)`;
    /// - on every boundary with a neighbour whose morphed top is lower, the
    ///   column's riser quads facing that neighbour cover the whole height
    ///   between the two morphed tops, with no gap.
    ///
    /// Goes red when the mesher keys a riser's rows wrongly, leaves out a
    /// latent riser, bakes a delta from the wrong row, or merges faces that
    /// move differently.
    #[test]
    fn a_morphed_node_is_closed_and_stands_on_its_parent() {
        let world = crate::World::new();
        let snapshot = world.snapshot();
        let nodes = MORPH_LEVELS
            .flat_map(|level| mountain_nodes(&snapshot, level))
            .collect();
        let Closed {
            tops,
            risers,
            skipped,
            ..
        } = closed_at_every_factor(&snapshot, nodes);
        eprintln!("checked {tops} tops and {risers} risers; {skipped} pairs not a height field");
        assert!(
            tops > 1000 && risers > 1000,
            "checked {tops} tops and {risers} risers"
        );
        assert!(skipped * 10 < risers, "{skipped} pairs skipped of {risers}");
    }

    /// Whether a cave has opened a column of the cell `w` blocks wide at
    /// `(x0, z0)`: a pit, or part of one.
    fn holds_a_pit(snapshot: &WorldSnapshot, x0: i32, z0: i32, w: i32) -> bool {
        (0..w * w).any(|k| {
            let column = snapshot.column(x0 + k % w, z0 + k / w);
            column.top < column.surface
        })
    }

    /// Nodes of a morphing `level` that hold the pits near the spawn: those
    /// holding the floor of a pit column of the 192 blocks square from the
    /// origin, the first `count` in key order.
    fn pit_nodes(snapshot: &WorldSnapshot, level: u32, count: usize) -> Vec<NodeKey> {
        let extent = 16 << level;
        let mut nodes = std::collections::BTreeSet::new();
        for z in 0..192 {
            for x in 0..192 {
                let column = snapshot.column(x, z);
                if column.top < column.surface {
                    nodes.insert(NodeKey::new(
                        level,
                        [
                            x.div_euclid(extent),
                            column.top.div_euclid(extent),
                            z.div_euclid(extent),
                        ],
                    ));
                }
            }
        }
        nodes.into_iter().take(count).collect()
    }

    /// **A pit is ground, and the morph slides it** (`PHASE3_ARCHITECTURE.md`
    /// §9.9): on the nodes that hold the pits near the spawn, the mesh is
    /// closed at every factor and stands on its parent at full morph, as on
    /// the mountain -- and the tops and the risers checked include the
    /// floors of pits and their walls, moving. Until §9.9 no column a cave
    /// had opened morphed, nor any beside one: a level's pit was swapped
    /// for its parent's flat ground at the ring edge. Goes red when a top is
    /// the surface again (a pit's floor is then no column's top, and
    /// nothing of a pit moves).
    #[test]
    fn a_morphed_pit_is_closed_and_stands_on_its_parent() {
        let world = crate::World::new();
        let snapshot = world.snapshot();
        let nodes = MORPH_LEVELS
            .flat_map(|level| pit_nodes(&snapshot, level, 6))
            .collect();
        let Closed {
            tops,
            risers,
            skipped,
            pit_tops,
            pit_risers,
            pit_cells,
            moves,
        } = closed_at_every_factor(&snapshot, nodes);
        eprintln!(
            "checked {tops} tops and {risers} risers, {pit_tops} and {pit_risers} of them a moving pit's; {skipped} pairs not a height field; cells that hold a pit, and morph, by level: {pit_cells:?}"
        );
        for (level, (pit, other)) in moves.into_iter().enumerate() {
            let describe = |mut v: Vec<i32>| {
                v.sort_unstable();
                let at = |p: usize| v.get((v.len() * p / 100).min(v.len().max(1) - 1)).copied();
                format!(
                    "{} cells, {} moving 3 blocks or more; median {:?}, 95th percentile {:?}, most {:?}",
                    v.len(),
                    v.iter().filter(|&&m| m >= 3).count(),
                    at(50),
                    at(95),
                    v.last()
                )
            };
            eprintln!(
                "level {level}: morphing cells that hold a pit: {}; the others: {}",
                describe(pit),
                describe(other)
            );
        }
        assert!(
            pit_tops >= 100 && pit_risers >= 200,
            "only {pit_tops} moving tops and {pit_risers} moving risers of pits were checked"
        );
        // Measured: 64, 51 and 71% of the cells that hold a pit morph at
        // levels 0, 1 and 2. What does not is a pit that widens under its
        // rim: a column with an overhang among the rows it shows stays
        // still, and so does every column beside it (#436's limit).
        for (level, (held, morphing)) in pit_cells.into_iter().enumerate() {
            assert!(
                held >= 50 && 5 * morphing >= 2 * held,
                "level {level}: {morphing} of {held} cells that hold a pit morph"
            );
        }
    }

    /// What [`closed_at_every_factor`] checked.
    struct Closed {
        tops: usize,
        risers: usize,
        /// Pairs of columns that are not a height field as drawn.
        skipped: usize,
        /// Of `tops`, those of cells that hold a pit and move.
        pit_tops: usize,
        /// Of `risers`, those between two such cells, or one and its rim.
        pit_risers: usize,
        /// Per level, the cells that hold a pit and have their top in the
        /// node, and how many of them morph.
        pit_cells: [(usize, usize); 3],
        /// Per level, how far each morphing cell's top moves at full morph,
        /// in blocks: those of cells that hold a pit, and of the others.
        moves: [(Vec<i32>, Vec<i32>); 3],
    }

    /// The check of `a_morphed_node_is_closed_and_stands_on_its_parent`
    /// over `nodes`.
    fn closed_at_every_factor(snapshot: &WorldSnapshot, nodes: Vec<NodeKey>) -> Closed {
        let registry = registry();
        let blocks = TerrainBlocks::from_registry(&registry);
        let layer = |_: &str| 0u32;
        let (mut tops, mut risers, mut skipped) = (0, 0, 0);
        let (mut pit_tops, mut pit_risers) = (0, 0);
        let mut pit_cells = [(0, 0); 3];
        let mut moves: [(Vec<i32>, Vec<i32>); 3] = Default::default();
        {
            for node in nodes {
                let level = node.level;
                let w = 1i32 << level;
                let origin = node.world_origin();
                let moving_pit = |cx: i32, cz: i32, column: [i32; 4]| {
                    column[1] != column[0]
                        && holds_a_pit(snapshot, origin[0] + cx * w, origin[2] + cz * w, w)
                };
                let Some(geometry) =
                    crate::mesh::mesh_node(snapshot, &registry, &layer, node, blocks)
                else {
                    continue;
                };
                let columns = node_columns(snapshot, node, blocks).expect("a morphing level");
                let chunk = snapshot.node_at(node, blocks).expect("a drawn node");
                let ny = 16 * w; // rows of one block
                for cz in 0..16 {
                    for cx in 0..16 {
                        let [top, target, _, _] = columns.at(cx, cz);
                        if !(1..=ny).contains(&top) {
                            continue;
                        }
                        let pit = holds_a_pit(snapshot, origin[0] + cx * w, origin[2] + cz * w, w);
                        if pit {
                            pit_cells[level as usize].0 += 1;
                            pit_cells[level as usize].1 += columns.morphs_at(cx, cz) as usize;
                        }
                        if columns.morphs_at(cx, cz) {
                            let of = &mut moves[level as usize];
                            if pit { &mut of.0 } else { &mut of.1 }.push((target - top).abs());
                        }
                    }
                }
                // Whether column (cx, cz) is drawn as the height field says
                // over rows `from..to`: solid under `top`, air from it -- no
                // cave or pit there (a tree's leaves above `to` are not
                // asked). The closure is held on that ground; where a column
                // is not, the morph must leave it still (below).
                let height_field = |cx: i32, cz: i32, from: i32, top: i32, to: i32| {
                    (from.max(0)..to.min(ny)).all(|y| {
                        let solid = chunk.drawn(cx as usize, y as usize, cz as usize)
                            != cubara_voxel::BlockId::AIR;
                        solid == (y < top)
                    })
                };
                let quads: Vec<[cubara_voxel::Vertex; 4]> = geometry
                    .mesh
                    .vertices
                    .chunks(4)
                    .map(|q| [q[0], q[1], q[2], q[3]])
                    .collect();
                for f in [0.0f32, 0.25, 0.5, 1.0] {
                    let y = |v: cubara_voxel::Vertex| v.y() as f32 + v.morph_delta() * f;
                    let lerp = |s: i32, t: i32| s as f32 + (t - s) as f32 * f;
                    for cz in 0..16 {
                        for cx in 0..16 {
                            let [top, target, _, _] = columns.at(cx, cz);
                            // Tops: one face over the column's square, at
                            // the morphed top.
                            if (1..=ny).contains(&top)
                                && height_field(cx, cz, top - 1, top, top + 1)
                            {
                                let (x0, z0) = (cx * w, cz * w);
                                let found = quads.iter().any(|q| {
                                    q[0].face() == cubara_voxel::Face::PosY
                                        && q[0].y() as i32 == top
                                        && q.iter()
                                            .all(|v| (y(*v) - lerp(top, target)).abs() < 1e-3)
                                        && q.iter().map(|v| v.x() as i32).min() <= Some(x0)
                                        && q.iter().map(|v| v.x() as i32).max() >= Some(x0 + w)
                                        && q.iter().map(|v| v.z() as i32).min() <= Some(z0)
                                        && q.iter().map(|v| v.z() as i32).max() >= Some(z0 + w)
                                });
                                assert!(
                                    found,
                                    "level {level} {node:?} column ({cx}, {cz}): no top face at \
                                     {} (top {top}, target {target}, f {f})",
                                    lerp(top, target)
                                );
                                tops += 1;
                                pit_tops +=
                                    (f == 1.0 && moving_pit(cx, cz, columns.at(cx, cz))) as usize;
                            }
                            // Risers toward the +x and +z neighbours, both ways.
                            for (dx, dz) in [(1, 0), (0, 1)] {
                                let (nx, nz) = (cx + dx, cz + dz);
                                if nx >= 16 || nz >= 16 {
                                    continue;
                                }
                                let a = columns.at(cx, cz);
                                let b = columns.at(nx, nz);
                                let (ta, tb) = (lerp(a[0], a[1]), lerp(b[0], b[1]));
                                if (ta - tb).abs() < 1e-3 {
                                    continue;
                                }
                                // The higher one's riser, facing the lower.
                                let (high, low, sign) =
                                    if ta > tb { (a, b, 1) } else { (b, a, -1) };
                                let span = [high[0], high[1], low[0], low[1]];
                                if span.iter().any(|&r| !(1..=ny).contains(&r)) {
                                    continue; // crosses into the node above or below
                                }
                                let (from, to) = (a[0].min(b[0]) - 1, a[0].max(b[0]) + 1);
                                if !height_field(cx, cz, from, a[0], to)
                                    || !height_field(nx, nz, from, b[0], to)
                                {
                                    // A cave, a pit or an overhang the world's
                                    // purity rule must have seen too: neither
                                    // column moves.
                                    assert!(
                                        a[1] == a[0] && b[1] == b[0],
                                        "level {level} {node:?} ({cx},{cz})/({nx},{nz}) is not a \
                                         height field as drawn, yet morphs: {a:?} {b:?}"
                                    );
                                    // ...and neither does the riser between them.
                                    for (own, nb) in [(a, b), (b, a)] {
                                        if let Some([s0, s1, t0, t1]) = riser_map(own, nb) {
                                            assert!(
                                                (s0, s1) == (t0, t1),
                                                "level {level} {node:?} ({cx},{cz})/({nx},{nz}): a \
                                                 riser beside a cave moves: {own:?} {nb:?}"
                                            );
                                        }
                                    }
                                    skipped += 1;
                                    continue;
                                }
                                let axis = if dx == 1 { 0 } else { 2 };
                                let face = cubara_voxel::Face::from_axis_sign(axis, sign);
                                let plane = if dx == 1 { (cx + 1) * w } else { (cz + 1) * w };
                                let along = if dx == 1 { cz * w } else { cx * w };
                                let mut cover: Vec<(f32, f32)> = quads
                                    .iter()
                                    .filter(|q| q[0].face() == face)
                                    .filter(|q| {
                                        let p = |v: &cubara_voxel::Vertex| {
                                            (if axis == 0 { v.x() } else { v.z() }) as i32
                                        };
                                        let s = |v: &cubara_voxel::Vertex| {
                                            (if axis == 0 { v.z() } else { v.x() }) as i32
                                        };
                                        q.iter().all(|v| p(v) == plane)
                                            && q.iter().map(s).min().unwrap() <= along
                                            && q.iter().map(s).max().unwrap() >= along + w
                                    })
                                    .map(|q| {
                                        let ys = q.iter().map(|v| y(*v));
                                        (
                                            ys.clone().fold(f32::MAX, f32::min),
                                            ys.fold(f32::MIN, f32::max),
                                        )
                                    })
                                    .collect();
                                cover.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap());
                                let (lo, hi) = (ta.min(tb), ta.max(tb));
                                let mut reached = lo;
                                for (bottom, top) in cover {
                                    if bottom <= reached + 1e-3 {
                                        reached = reached.max(top);
                                    }
                                }
                                assert!(
                                    reached >= hi - 1e-3,
                                    "level {level} {node:?} ({cx},{cz})->({nx},{nz}) at f {f}: \
                                     tops {ta}/{tb}, risers reach only {reached}"
                                );
                                risers += 1;
                                pit_risers += (f == 0.5
                                    && (moving_pit(cx, cz, a) || moving_pit(nx, nz, b)))
                                    as usize;
                            }
                        }
                    }
                }
            }
        }
        Closed {
            tops,
            risers,
            skipped,
            pit_tops,
            pit_risers,
            pit_cells,
            moves,
        }
    }

    /// `geometry`'s quads in world blocks, every vertex moved by factor `f`
    /// as `mesh.wgsl` moves it. A quad the morph has turned over -- a corner
    /// that started above another (or level with it and moving further up)
    /// now below it -- is left out: culled, it covers nothing.
    fn morphed_quads(
        geometry: &crate::mesh::NodeGeometry,
        f: f32,
    ) -> Vec<(cubara_voxel::Face, [[f32; 3]; 4])> {
        let o = geometry.origin;
        geometry
            .mesh
            .vertices
            .chunks(4)
            .filter_map(|q| {
                let p: Vec<[f32; 3]> = q
                    .iter()
                    .map(|v| {
                        [
                            (o[0] + v.x() as i32) as f32,
                            (o[1] + v.y() as i32) as f32 + v.morph_delta() * f,
                            (o[2] + v.z() as i32) as f32,
                        ]
                    })
                    .collect();
                let key = |v: &cubara_voxel::Vertex| (v.y() as f32, v.morph_delta());
                let turned = (0..4)
                    .any(|i| (0..4).any(|j| key(&q[i]) > key(&q[j]) && p[i][1] < p[j][1] - 1e-4));
                (!turned).then(|| (q[0].face(), [p[0], p[1], p[2], p[3]]))
            })
            .collect()
    }

    /// **A node and the node above or below it are closed together** -- the
    /// joins [`a_morphed_node_is_closed_and_stands_on_its_parent`] leaves
    /// out. A riser crosses a node's floor or ceiling wherever the ground
    /// crosses a multiple of the node's height, and each node draws only its
    /// own rows of it. The node across may be the same level, morphing by the
    /// same factor; or, at the top and bottom of a ring (the vertical squash
    /// makes the rings flat boxes, so on a mountain slope a ring's floor
    /// cuts the ground along a contour), one level coarser and not moving at
    /// all, while this one is fully morphed -- its own geometry, which is
    /// what this node's full morph must meet.
    ///
    /// For real nodes on the steep mountain, the ones above and below the
    /// surface, and the pair traced at frame 1796 of the toward-peak flight
    /// (a level-0 node on a level-1 one): both meshes together, each vertex
    /// moved by its own node's factor, must hold the riser between every two
    /// neighbouring columns of which one moves over the whole height
    /// between their morphed tops, within the two nodes' rows; and a top
    /// face over every moving column at its morphed top.
    ///
    /// Goes red when a riser vertex on a node's floor or ceiling moves where
    /// the node across does not follow it (frame 1796: a riser's lowest row
    /// in the upper node was lifted 1.5 blocks off the floor, and the coarse
    /// node below draws nothing above its own ceiling).
    #[test]
    fn a_morphed_node_closes_with_the_nodes_above_and_below() {
        let registry = registry();
        let blocks = TerrainBlocks::from_registry(&registry);
        let world = crate::World::new();
        let snapshot = world.snapshot();
        let layer = |_: &str| 0u32;
        let mesh = |n: NodeKey| crate::mesh::mesh_node(&snapshot, &registry, &layer, n, blocks);
        let mut fine = vec![NodeKey::new(0, [-190, 10, -201])];
        for level in MORPH_LEVELS {
            // The mountain's nodes, and nodes that hold the pits near the
            // spawn: a pit's wall crosses a node's floor as a cliff does,
            // and since §9.9 it moves.
            let nodes = mountain_nodes(&snapshot, level)
                .into_iter()
                .chain(pit_nodes(&snapshot, level, 3));
            for n in nodes {
                for dy in -1..=1 {
                    fine.push(NodeKey::new(level, [n.pos[0], n.pos[1] + dy, n.pos[2]]));
                }
            }
        }
        let (mut risers, mut tops) = (0usize, 0usize);
        let mut failures = Vec::new();
        for node in fine {
            let Some(columns) = node_columns(&snapshot, node, blocks) else {
                continue;
            };
            let Some(own) = mesh(node) else {
                continue;
            };
            let level = node.level;
            let w = 1i32 << level;
            let [ox, oy, oz] = node.world_origin();
            for dy in [-1, 1] {
                let [px, py, pz] = node.pos;
                let mut partners = vec![(NodeKey::new(level, [px, py + dy, pz]), false)];
                if (py + dy).div_euclid(2) != py.div_euclid(2) {
                    let up = |c: i32| c.div_euclid(2);
                    partners.push((NodeKey::new(level + 1, [up(px), up(py + dy), up(pz)]), true));
                }
                for (other, coarse) in partners {
                    let other_mesh = mesh(other);
                    let oy2 = other.world_origin()[1];
                    let slab = (
                        oy.min(oy2) as f32,
                        (oy + 16 * w).max(oy2 + (16 << other.level)) as f32,
                    );
                    // What this node must show: beside a coarse node only
                    // its own rows, since the coarse node shows its own
                    // ground in its rows (the parent's, which this node's
                    // full morph meets); beside one of its own level, the
                    // two together show the morphed ground in both.
                    let (lo_row, hi_row) = if coarse {
                        (oy as f32, (oy + 16 * w) as f32)
                    } else {
                        (slab.0 + 1e-3, slab.1)
                    };
                    let factors: &[f32] = if coarse {
                        &[1.0]
                    } else {
                        &[0.0, 0.25, 0.5, 0.75, 1.0]
                    };
                    for &f in factors {
                        let mut quads = morphed_quads(&own, f);
                        if let Some(g) = &other_mesh {
                            quads.extend(morphed_quads(g, if coarse { 0.0 } else { f }));
                        }
                        let moves = |c: [i32; 4]| c[1] != c[0];
                        let h = |c: [i32; 4]| (oy + c[0]) as f32 + (c[1] - c[0]) as f32 * f;
                        // Drawn by one of the two nodes: a top face belongs
                        // to the node holding the block under it.
                        let ours = |c: [i32; 4]| {
                            [c[0], c[1]].iter().all(|&r| {
                                let r = (oy + r) as f32;
                                r > slab.0 && r <= slab.1
                            })
                        };
                        for cz in 0..16 {
                            for cx in 0..16 {
                                let a = columns.at(cx, cz);
                                let (x0, z0) = ((ox + cx * w) as f32, (oz + cz * w) as f32);
                                let ha = h(a);
                                if moves(a) && ours(a) && ha >= lo_row && ha <= hi_row {
                                    let found = quads.iter().any(|(face, q)| {
                                        *face == cubara_voxel::Face::PosY
                                            && q.iter().all(|p| (p[1] - ha).abs() < 1e-3)
                                            && q.iter().map(|p| p[0]).fold(f32::MAX, f32::min) <= x0
                                            && q.iter().map(|p| p[0]).fold(f32::MIN, f32::max)
                                                >= x0 + w as f32
                                            && q.iter().map(|p| p[2]).fold(f32::MAX, f32::min) <= z0
                                            && q.iter().map(|p| p[2]).fold(f32::MIN, f32::max)
                                                >= z0 + w as f32
                                    });
                                    tops += 1;
                                    if !found {
                                        failures.push(format!(
                                            "{node:?} with {other:?} at f {f}: column ({cx},{cz}) \
                                             {a:?} has no top face at {ha}"
                                        ));
                                    }
                                }
                                for (dx, dz) in [(1, 0), (0, 1)] {
                                    let (nx, nz) = (cx + dx, cz + dz);
                                    if nx >= 16 || nz >= 16 {
                                        continue;
                                    }
                                    let b = columns.at(nx, nz);
                                    if !moves(a) && !moves(b) || !ours(a) || !ours(b) {
                                        continue;
                                    }
                                    let hb = h(b);
                                    let (lo, hi) = (ha.min(hb).max(lo_row), ha.max(hb).min(hi_row));
                                    if hi - lo < 1e-3 {
                                        continue;
                                    }
                                    let axis = if dx == 1 { 0 } else { 2 };
                                    let sign = if ha > hb { 1 } else { -1 };
                                    let face = cubara_voxel::Face::from_axis_sign(axis, sign);
                                    let plane = if dx == 1 {
                                        x0 + w as f32
                                    } else {
                                        z0 + w as f32
                                    };
                                    let along = if dx == 1 { z0 } else { x0 };
                                    let other_axis = 2 - axis;
                                    let mut cover: Vec<(f32, f32)> = quads
                                        .iter()
                                        .filter(|(fc, q)| {
                                            *fc == face
                                                && q.iter().all(|p| p[axis] == plane)
                                                && q.iter()
                                                    .map(|p| p[other_axis])
                                                    .fold(f32::MAX, f32::min)
                                                    <= along
                                                && q.iter()
                                                    .map(|p| p[other_axis])
                                                    .fold(f32::MIN, f32::max)
                                                    >= along + w as f32
                                        })
                                        .map(|(_, q)| {
                                            let ys = q.iter().map(|p| p[1]);
                                            (
                                                ys.clone().fold(f32::MAX, f32::min),
                                                ys.fold(f32::MIN, f32::max),
                                            )
                                        })
                                        .collect();
                                    cover.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap());
                                    let mut reached = lo;
                                    for (bottom, top) in cover {
                                        if bottom <= reached + 1e-3 {
                                            reached = reached.max(top);
                                        }
                                    }
                                    risers += 1;
                                    if reached < hi - 1e-3 {
                                        failures.push(format!(
                                            "{node:?} with {other:?} at f {f}: ({cx},{cz}) {a:?} \
                                             / ({nx},{nz}) {b:?}: tops {ha}/{hb}, risers cover \
                                             {lo}..{reached} of ..{hi}"
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        eprintln!(
            "checked {tops} tops and {risers} risers; {} gaps",
            failures.len()
        );
        for line in failures.iter().take(40) {
            eprintln!("  {line}");
        }
        assert!(failures.is_empty(), "{} gaps", failures.len());
        assert!(
            tops > 1000 && risers > 1000,
            "checked {tops} tops, {risers} risers"
        );
    }

    #[test]
    fn the_default_schedule_has_a_band_for_levels_0_to_2_and_none_for_3() {
        let bands = morph_bands(DEFAULT_RING_SCHEDULE, SQUASH);
        // 160 m edge: ends 32 before it; the cap (128) would reach the eye,
        // but the near field (96 blocks) stands exactly on the grid.
        assert_eq!(
            bands[0],
            Some(MorphBand {
                start: NEAR_EXACT,
                end: 128.0
            })
        );
        // 288 m edge: the ring leaves 32 blocks between the inner reach of a
        // level-1 node (160 + 64) and the outer margin (288 - 32).
        assert_eq!(
            bands[1],
            Some(MorphBand {
                start: 224.0,
                end: 256.0
            })
        );
        // 512 m edge: 288 + 128 to 512 - 32.
        assert_eq!(
            bands[2],
            Some(MorphBand {
                start: 416.0,
                end: 480.0
            })
        );
        assert_eq!(bands[3], None, "level 3 meets the far terrain");
    }

    /// **The near field is exactly on the block grid** (the owner's play
    /// test, 2026-11-25: thin slabs on the terraces and slivers at the foot
    /// of walls, which were level-0 vertices a few percent of the way to
    /// their parent's height). Real level-0 nodes around eyes standing on
    /// the mountain's terraces, and hovering above them: every vertex that
    /// has a morph target and is within [`NEAR_EXACT`] of the eye, by the
    /// morph metric (horizontal, or twice the vertical), has factor 0 --
    /// its drawn height is the block's own. Goes red when the band starts
    /// nearer than [`NEAR_EXACT`] (main's started at the eye).
    #[test]
    fn level_0_stands_exactly_on_the_grid_near_the_eye() {
        let registry = registry();
        let blocks = TerrainBlocks::from_registry(&registry);
        let world = crate::World::new();
        let snapshot = world.snapshot();
        let layer = |_: &str| 0u32;
        // The floor of the measured candidates (64, 80, 96): the radius
        // cannot be set back toward the eye without this failing to build.
        const _: () = assert!(NEAR_EXACT >= 64.0, "the near field is at least 64 blocks");
        let band = morph_bands(DEFAULT_RING_SCHEDULE, SQUASH)[0].expect("level 0 morphs");
        let ground = snapshot.surface_height(336 * 16, 213 * 16);
        let (mut near_moving, mut far_moving) = (0, 0);
        for up in [2.6f32, 6.0, 20.0] {
            let eye = [336.0 * 16.0 + 0.5, ground as f32 + up, 213.0 * 16.0 + 0.5];
            let at = ChunkCoord::from_world_pos(eye);
            for node in desired_nodes_3d(at, SQUASH, DEFAULT_RING_SCHEDULE) {
                if node.level != 0 {
                    continue;
                }
                let (min, _) = metric_range(node, eye);
                if min > 140.0 {
                    continue;
                }
                let Some(geometry) =
                    crate::mesh::mesh_node(&snapshot, &registry, &layer, node, blocks)
                else {
                    continue;
                };
                let o = geometry.origin;
                for v in &geometry.mesh.vertices {
                    if v.morph_delta() == 0.0 {
                        continue;
                    }
                    let rel = [
                        (o[0] + v.x() as i32) as f32 - eye[0],
                        (o[1] + v.y() as i32) as f32 - eye[1],
                        (o[2] + v.z() as i32) as f32 - eye[2],
                    ];
                    let metric = morph_metric(rel, SQUASH);
                    if metric < NEAR_EXACT {
                        near_moving += 1;
                        assert_eq!(
                            morph_factor(metric, band),
                            0.0,
                            "{node:?} vertex {rel:?} (metric {metric}, eye {up} up) has a \
                             morph target but is inside the near radius and would stand off \
                             the block grid"
                        );
                    } else {
                        far_moving += 1;
                    }
                }
            }
        }
        assert!(
            near_moving > 100 && far_moving > 100,
            "near {near_moving}, far {far_moving} moving vertices: the test saw too little"
        );
    }

    /// **Within [`NEAR_EXACT`] blocks in a straight line, the ground is level
    /// 0 and does not move -- looking down from a peak and up a face too**
    /// (the acceptance tour of 2026-11-27, §9.6.2). Real nodes of the game's
    /// selection around two eyes the tour stood at: on the highest peak near
    /// the origin, and at the foot of the steepest face beside it. Every node
    /// that comes that near is level 0, and every vertex of it that has a
    /// morph target and lies that near has factor 0. Guarded by how many of
    /// those vertices are more than half of [`NEAR_EXACT`] above or below
    /// the eye -- the ones the squash used to move. Goes red with height
    /// squashed from the eye: in the selection (a level-1 node 81 blocks
    /// under the peak) or in the metric (factor 1 at 75 blocks down).
    #[test]
    fn level_0_stands_exactly_on_the_grid_within_the_near_field_in_true_distance() {
        // The acceptance pass asked for 96 blocks: the near field cannot be
        // set back under it without this failing to build.
        const _: () = assert!(NEAR_EXACT >= 96.0, "the near field is at least 96 blocks");
        let registry = registry();
        let blocks = TerrainBlocks::from_registry(&registry);
        let world = crate::World::new();
        let snapshot = world.snapshot();
        let layer = |_: &str| 0u32;
        let band = morph_bands(DEFAULT_RING_SCHEDULE, SQUASH)[0].expect("level 0 morphs");
        let (mut near, mut steep) = (0, 0);
        for eye in [[-960.5f32, 243.62, -3335.5], [-886.5, 70.62, -3137.5]] {
            let at = ChunkCoord::from_world_pos(eye);
            for node in desired_nodes_3d(at, SQUASH, DEFAULT_RING_SCHEDULE) {
                let o = node.world_origin();
                let size = CHUNK * node.extent_chunks() as f32;
                let gap = |k: usize| {
                    (o[k] as f32 - eye[k])
                        .max(eye[k] - o[k] as f32 - size)
                        .max(0.0)
                };
                if (gap(0).powi(2) + gap(1).powi(2) + gap(2).powi(2)).sqrt() >= NEAR_EXACT {
                    continue;
                }
                assert_eq!(
                    node.level, 0,
                    "{node:?} comes within {NEAR_EXACT} of {eye:?}"
                );
                let Some(geometry) =
                    crate::mesh::mesh_node(&snapshot, &registry, &layer, node, blocks)
                else {
                    continue;
                };
                for v in &geometry.mesh.vertices {
                    if v.morph_delta() == 0.0 {
                        continue;
                    }
                    let rel = [
                        (o[0] + v.x() as i32) as f32 - eye[0],
                        (o[1] + v.y() as i32) as f32 - eye[1],
                        (o[2] + v.z() as i32) as f32 - eye[2],
                    ];
                    let distance = (rel[0].powi(2) + rel[1].powi(2) + rel[2].powi(2)).sqrt();
                    if distance >= NEAR_EXACT {
                        continue;
                    }
                    near += 1;
                    if 2.0 * rel[1].abs() > NEAR_EXACT {
                        steep += 1;
                    }
                    let metric = morph_metric(rel, SQUASH);
                    assert_eq!(
                        morph_factor(metric, band),
                        0.0,
                        "{node:?} vertex {rel:?}, {distance} blocks from {eye:?}, has metric \
                         {metric} and moves"
                    );
                }
            }
        }
        assert!(
            near > 10_000 && steep > 1_000,
            "{near} vertices with a target in the near field, {steep} of them steeply above \
             or below: the test saw too little"
        );
    }

    #[test]
    fn a_ring_too_narrow_for_its_nodes_has_no_band() {
        // Every radius halved: the level-1 ring is 4 chunks wide, but a
        // level-1 node beside level 0 reaches 4 chunks past its inner edge
        // and the band must end 2 chunks before its outer one.
        let half = [(0u32, 5i32), (1, 9), (2, 16), (3, 32)];
        let bands = morph_bands(&half, SQUASH);
        assert_eq!(
            bands[0],
            Some(MorphBand {
                start: 36.0,
                end: 48.0
            })
        );
        assert_eq!(bands[1], None);
        assert_eq!(bands[2], None);
    }

    #[test]
    fn the_factor_is_zero_before_the_band_one_after_and_linear_in_it() {
        let b = MorphBand {
            start: 100.0,
            end: 140.0,
        };
        assert_eq!(morph_factor(0.0, b), 0.0);
        assert_eq!(morph_factor(100.0, b), 0.0);
        assert_eq!(morph_factor(110.0, b), 0.25);
        assert_eq!(morph_factor(140.0, b), 1.0);
        assert_eq!(morph_factor(1e9, b), 1.0);
    }

    /// **The morph's metric is the selection's, continued between chunks**:
    /// at every chunk boundary its vertical part is the selection's
    /// ([`crate::node::squashed_gap`], times 16), at every squash; inside
    /// the near field height counts once, beyond it `squash` times as fast.
    /// Goes red when either side changes its vertical distance alone -- the
    /// selection would then swap a node where the morph has not finished
    /// (`every_node_is_fully_morphed_at_the_frame_it_swaps`).
    #[test]
    fn the_metric_is_the_selections_distance() {
        assert_eq!(morph_metric([3.0, 0.0, 4.0], 2), 5.0);
        assert_eq!(morph_metric([3.0, -2.0, 4.0], 2), 5.0);
        // Inside the near field, height counts once...
        assert_eq!(morph_metric([3.0, 10.0, 4.0], 2), 10.0);
        assert_eq!(morph_metric([0.0, -NEAR_EXACT, 0.0], 2), NEAR_EXACT);
        // ...and beyond it twice as fast: 100 blocks down is 96 + 2 * 4.
        assert_eq!(morph_metric([0.0, -100.0, 0.0], 2), 104.0);
        assert_eq!(morph_metric([0.0, 200.0, 0.0], 2), 304.0);
        for squash in 1..=4 {
            for dy in 0..64 {
                let v = CHUNK * dy as f32;
                assert_eq!(
                    squashed_height(v, squash),
                    CHUNK * crate::node::squashed_gap(dy, squash) as f32,
                    "squash {squash}, {dy} chunks"
                );
            }
        }
    }

    /// **Within [`NEAR_EXACT`] blocks of the eye, in a straight line and in
    /// any direction, level 0 does not move** (the acceptance tour of
    /// 2026-11-27: from a peak, ground 75 blocks down and 90 away was drawn
    /// at level 1's heights, because height counted twice all the way to
    /// the eye). Every point of the ball, straight down, straight up and
    /// everything between, has factor 0 in level 0's band at the game's
    /// schedule and squash; and just outside it, straight down, the band has
    /// begun. Goes red with the squash applied from the eye (factor 0.75 at
    /// 60 blocks down).
    #[test]
    fn level_0_does_not_move_within_the_near_field_in_true_distance() {
        let band = morph_bands(DEFAULT_RING_SCHEDULE, SQUASH)[0].expect("level 0 morphs");
        assert_eq!(band.start, NEAR_EXACT);
        // A grid over the half-plane of horizontal distance and height,
        // from straight up to straight down: every point of it inside the
        // ball (a quarter block apart), in three horizontal directions.
        let (mut inside, mut steep) = (0, 0);
        for i in 0..=384 {
            for j in -384..=384 {
                let (h, v) = (i as f32 * 0.25, j as f32 * 0.25);
                let r = h.hypot(v);
                if r >= NEAR_EXACT {
                    continue;
                }
                inside += 1;
                if 2.0 * v.abs() > NEAR_EXACT {
                    steep += 1;
                }
                for rel in [[h, v, 0.0], [0.0, v, h], [h * 0.6, v, h * 0.8]] {
                    let metric = morph_metric(rel, SQUASH);
                    assert_eq!(
                        morph_factor(metric, band),
                        0.0,
                        "{rel:?}, {r} blocks from the eye, has metric {metric}: inside {band:?}"
                    );
                }
            }
        }
        // The guard: the points the squash used to move were checked.
        assert!(
            inside > 200_000 && steep > 40_000,
            "{inside} points, {steep} steep"
        );
        let below = morph_metric([0.0, -(NEAR_EXACT + 4.0), 0.0], SQUASH);
        assert!(
            morph_factor(below, band) > 0.0,
            "100 blocks down is in the band"
        );
    }

    /// The least and greatest [`morph_metric`] of any point of `node` from
    /// `eye`: the box is a rectangle times an interval, so the horizontal
    /// and vertical parts are found apart.
    fn metric_range(node: NodeKey, eye: [f32; 3]) -> (f32, f32) {
        let o = node.world_origin();
        let size = CHUNK * node.extent_chunks() as f32;
        let lo = [o[0] as f32, o[1] as f32, o[2] as f32];
        let near = |k: usize| (lo[k] - eye[k]).max(eye[k] - (lo[k] + size)).max(0.0);
        let far = |k: usize| (eye[k] - lo[k]).abs().max((eye[k] - lo[k] - size).abs());
        let min = morph_metric([near(0), near(1), near(2)], SQUASH);
        let max = morph_metric([far(0), far(1), far(2)], SQUASH);
        (min, max)
    }

    /// A small deterministic generator for eye positions (Rule 1: seeded).
    fn eyes(count: usize) -> Vec<[f32; 3]> {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f32 / (1u64 << 53) as f32
        };
        (0..count)
            .map(|_| {
                [
                    next() * 4000.0 - 2000.0,
                    next() * 400.0 - 100.0,
                    next() * 4000.0 - 2000.0,
                ]
            })
            .collect()
    }

    /// **The band ends before every swap.** From eyes all over, stepping a
    /// flight's frame (0.4 blocks) in every direction: every node a step
    /// swaps out for its parent or its children -- on either side of the
    /// step -- is fully morphed at the eye it is drawn from, at every point
    /// of its box. Goes red with a band that ends at the ring edge (an outer
    /// margin of 0) or with the vertical squash left out of the margin.
    #[test]
    fn every_node_is_fully_morphed_at_the_frame_it_swaps() {
        let bands = morph_bands(DEFAULT_RING_SCHEDULE, SQUASH);
        let mut checked = 0;
        for eye in eyes(60) {
            for step in [
                [0.4f32, 0.0, 0.0],
                [-0.4, 0.0, 0.0],
                [0.0, 0.4, 0.0],
                [0.0, -0.4, 0.0],
                [0.0, 0.0, 0.4],
                [0.0, 0.0, -0.4],
                [0.28, 0.0, 0.28],
            ] {
                // Put the step across a chunk boundary, where swaps happen:
                // 0.2 before the boundary on every axis the step moves along.
                let place = |c: f32, s: f32| {
                    let boundary = (c / CHUNK).floor() * CHUNK + CHUNK;
                    if s > 0.0 {
                        boundary - 0.2
                    } else if s < 0.0 {
                        boundary + 0.2
                    } else {
                        c
                    }
                };
                let a = [
                    place(eye[0], step[0]),
                    place(eye[1], step[1]),
                    place(eye[2], step[2]),
                ];
                let b = [a[0] + step[0], a[1] + step[1], a[2] + step[2]];
                let set = |e: [f32; 3]| -> std::collections::HashSet<NodeKey> {
                    desired_nodes_3d(ChunkCoord::from_world_pos(e), SQUASH, DEFAULT_RING_SCHEDULE)
                        .into_iter()
                        .collect()
                };
                let (sa, sb) = (set(a), set(b));
                for (drawn, other, e) in [(&sa, &sb, a), (&sb, &sa, b)] {
                    for &n in drawn.difference(other) {
                        // A merge: n's parent is drawn on the other side.
                        if !other.contains(&n.parent()) {
                            continue;
                        }
                        let Some(band) = bands[n.level as usize] else {
                            continue;
                        };
                        let (min, _) = metric_range(n, e);
                        assert!(
                            morph_factor(min, band) == 1.0,
                            "{n:?} swaps at eye {e:?} with its nearest point at metric {min} \
                             but the band {band:?} has not ended there"
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 50, "only {checked} swaps were checked");
    }

    /// The greatest [`morph_metric`] from `eye` of any point of `node`'s face
    /// on `axis` at side `s` (-1 low, +1 high).
    fn face_metric_max(node: NodeKey, axis: usize, s: i32, eye: [f32; 3]) -> f32 {
        let o = node.world_origin();
        let size = CHUNK * node.extent_chunks() as f32;
        let lo = [o[0] as f32, o[1] as f32, o[2] as f32];
        let far = |k: usize| {
            if k == axis {
                let plane = if s < 0 { lo[k] } else { lo[k] + size };
                (plane - eye[k]).abs()
            } else {
                (eye[k] - lo[k]).abs().max((eye[k] - lo[k] - size).abs())
            }
        };
        morph_metric([far(0), far(1), far(2)], SQUASH)
    }

    /// **The band starts after every node beside a finer one.** Where a
    /// level-`L` node shares a face with a level `L - 1` node, the finer one
    /// has finished its own morph onto this node's unmorphed geometry, so
    /// this one must not have started on that face. Goes red with a band
    /// that starts at the ring's inner edge.
    #[test]
    fn no_node_beside_a_finer_one_has_started_to_morph() {
        let bands = morph_bands(DEFAULT_RING_SCHEDULE, SQUASH);
        let mut checked = 0;
        for eye in eyes(40) {
            let nodes: std::collections::HashSet<NodeKey> = desired_nodes_3d(
                ChunkCoord::from_world_pos(eye),
                SQUASH,
                DEFAULT_RING_SCHEDULE,
            )
            .into_iter()
            .collect();
            for &n in &nodes {
                let Some(band) = bands[n.level as usize] else {
                    continue;
                };
                if n.level == 0 {
                    continue;
                }
                // Every face of n with a finer node across it.
                for axis in 0..3 {
                    for s in [-1, 1] {
                        let mut p = n.pos;
                        p[axis] += s;
                        let across = NodeKey::new(n.level, p);
                        if !across.children().iter().any(|c| nodes.contains(c)) {
                            continue;
                        }
                        let max = face_metric_max(n, axis, s, eye);
                        assert!(
                            morph_factor(max, band) == 0.0,
                            "{n:?}'s face {axis}/{s} beside a finer node reaches metric {max} \
                             from {eye:?}, inside {band:?}"
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 50, "only {checked} nodes were checked");
    }

    /// **The morphed staircase is closed.** For every pair of neighbouring
    /// columns of real nodes, at every factor: the two risers between them
    /// ([`riser_map`], each column's own toward the other) together cover
    /// the whole height between the two morphed tops -- so no gap opens
    /// between a moving top and the riser that should meet it. And at full
    /// morph each pure column stands at its parent's top.
    #[test]
    fn the_morphed_staircase_is_closed_and_ends_on_the_parent() {
        let world = crate::World::new();
        let snapshot = world.snapshot();
        let blocks = blocks();
        let (px, pz) = (336 * 16, 213 * 16); // the steep mountain of #432
        let surface = snapshot.surface_height(px, pz);
        let mut pairs = 0;
        for level in MORPH_LEVELS {
            let extent = 16 << level;
            let node = NodeKey::new(
                level,
                [
                    px.div_euclid(extent),
                    surface.div_euclid(extent),
                    pz.div_euclid(extent),
                ],
            );
            let columns = node_columns(&snapshot, node, blocks).expect("a morphing level");
            let morphing = columns.columns.iter().filter(|c| c[1] != c[0]).count();
            assert!(morphing > 0, "level {level}: nothing morphs on a mountain");
            for z in 0..16 {
                for x in 0..16 {
                    for (dx, dz) in [(1, 0), (0, 1)] {
                        let (a, b) = (columns.at(x, z), columns.at(x + dx, z + dz));
                        if !(0..16).contains(&(x + dx)) || !(0..16).contains(&(z + dz)) {
                            continue;
                        }
                        for k in 0..=8 {
                            let f = k as f32 / 8.0;
                            let lerp = |s: i32, t: i32| s as f32 + (t - s) as f32 * f;
                            let (ta, tb) = (lerp(a[0], a[1]), lerp(b[0], b[1]));
                            let span = |own: [i32; 4], nb: [i32; 4]| {
                                riser_map(own, nb)
                                    .map(|[s0, s1, t0, t1]| (lerp(s0, t0), lerp(s1, t1)))
                            };
                            let (lo, hi) = (ta.min(tb), ta.max(tb));
                            if hi - lo < 1e-4 {
                                continue;
                            }
                            // The higher column's riser must reach the lower top.
                            let higher = if ta >= tb { span(a, b) } else { span(b, a) };
                            let (bottom, top) = higher.expect("a riser where tops differ");
                            assert!(
                                bottom <= lo + 1e-4 && (top - hi).abs() < 1e-4,
                                "level {level} ({x},{z})+({dx},{dz}) at f {f}: tops {ta}/{tb}, \
                                 riser {bottom}..{top}"
                            );
                            pairs += 1;
                        }
                    }
                }
            }
        }
        assert!(pairs > 1000, "only {pairs} pairs were checked");
    }

    /// **No corner that moves sits inside an edge that moves.** The factor
    /// is not linear along an edge (the metric's `max` bends it where
    /// horizontal and doubled vertical distance are equal), so a vertex in
    /// the middle of another quad's edge -- a T-junction -- leaves that edge
    /// when both move, by up to half a pixel at the 288 m band: single pixels
    /// of sky, which the first flights with the morph showed. On real nodes
    /// of every morphing level on the steep mountain: no vertex with a delta
    /// lies strictly inside a horizontal edge of a quad that moves. Goes red
    /// when moving faces may merge across cells again.
    #[test]
    fn no_moving_corner_sits_inside_a_moving_edge() {
        let registry = registry();
        let blocks = TerrainBlocks::from_registry(&registry);
        let world = crate::World::new();
        let snapshot = world.snapshot();
        let layer = |_: &str| 0u32;
        let mut edges_checked = 0usize;
        for level in MORPH_LEVELS {
            for node in mountain_nodes(&snapshot, level) {
                let Some(geometry) =
                    crate::mesh::mesh_node(&snapshot, &registry, &layer, node, blocks)
                else {
                    continue;
                };
                let quads: Vec<&[cubara_voxel::Vertex]> =
                    geometry.mesh.vertices.chunks(4).collect();
                // Horizontal edges of quads that move, by height: (y, a, b)
                // with a and b the (x, z) ends.
                type Edge = ([u32; 2], [u32; 2]);
                let mut edges: std::collections::HashMap<u32, Vec<Edge>> =
                    std::collections::HashMap::new();
                for q in &quads {
                    if q.iter().all(|v| v.morph_delta() == 0.0) {
                        continue;
                    }
                    for k in 0..4 {
                        let (a, b) = (q[k], q[(k + 1) % 4]);
                        if a.y() == b.y() && (a.x(), a.z()) != (b.x(), b.z()) {
                            edges
                                .entry(a.y())
                                .or_default()
                                .push(([a.x(), a.z()], [b.x(), b.z()]));
                        }
                    }
                }
                for q in &quads {
                    for v in q.iter().filter(|v| v.morph_delta() != 0.0) {
                        let p = [v.x(), v.z()];
                        for &(a, b) in edges.get(&v.y()).map_or(&[][..], |e| &e[..]) {
                            edges_checked += 1;
                            // Strictly inside an axis-aligned edge.
                            let inside = if a[0] == b[0] {
                                p[0] == a[0] && p[1] > a[1].min(b[1]) && p[1] < a[1].max(b[1])
                            } else {
                                p[1] == a[1] && p[0] > a[0].min(b[0]) && p[0] < a[0].max(b[0])
                            };
                            assert!(
                                !inside,
                                "level {level} {node:?}: a moving corner at {p:?}, row {}, sits \
                                 inside the moving edge {a:?}-{b:?}",
                                v.y()
                            );
                        }
                    }
                }
            }
        }
        assert!(edges_checked > 10_000, "only {edges_checked} pairs checked");
    }

    /// **What the morph costs in geometry** (issue #380's study): the
    /// game's selection around an eye (radius 64, squash 2, the default
    /// schedule, the real assets), meshed with and without the morph --
    /// triangles and vertices per level, how many of the added quads are
    /// latent risers, and the largest delta a vertex carries. Every node of
    /// the selection, seen or not (as `geometry_census` and #432's tables).
    /// `MORPH_CENSUS_AT=x,y,z` picks the eye (default: the gate's ground eye,
    /// (8, 40, 8)).
    #[test]
    #[ignore = "meshes a whole selection twice: cargo test -p cubara-world --release morph_cost_census -- --ignored --nocapture"]
    fn morph_cost_census() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let registry = BlockRegistry::load(&root.join("assets/blocks")).expect("assets/blocks");
        let structures = cubara_voxel::StructureRegistry::load(&root.join("assets/structures"))
            .expect("assets/structures");
        let ores = cubara_voxel::OreRegistry::load(&root.join("assets/ores")).expect("assets/ores");
        let blocks = TerrainBlocks::from_registry(&registry)
            .with_oak(&structures, &registry)
            .with_ores(&ores, &registry);
        let world = crate::World::new();
        let snapshot = world.snapshot();
        let layer = |_: &str| 0u32;
        let eye: [f32; 3] = std::env::var("MORPH_CENSUS_AT")
            .ok()
            .and_then(|t| {
                let v: Vec<f32> = t.split(',').filter_map(|p| p.trim().parse().ok()).collect();
                (v.len() == 3).then(|| [v[0], v[1], v[2]])
            })
            .unwrap_or([8.0, 40.0, 8.0]);
        let nodes = desired_nodes_3d(
            ChunkCoord::from_world_pos(eye),
            SQUASH,
            DEFAULT_RING_SCHEDULE,
        );
        // Per level: [triangles, vertices] without and with the morph, latent
        // quads, quads that move, nodes. No mesh times: wall-clock time has
        // no place in a simulation crate (`scripts/check-architecture.sh`,
        // rule 1), and the cost that matters here is geometry.
        let mut rows = [[0u64; 7]; 4];
        let mut largest = 0f32;
        for &node in &nodes {
            let l = node.level as usize;
            let plain =
                crate::mesh::mesh_node_with(&snapshot, &registry, &layer, node, blocks, false);
            let morphed =
                crate::mesh::mesh_node_with(&snapshot, &registry, &layer, node, blocks, true);
            rows[l][6] += 1;
            if let Some(g) = plain {
                rows[l][0] += g.mesh.triangle_count() as u64;
                rows[l][1] += g.mesh.vertices.len() as u64;
            }
            if let Some(g) = morphed {
                rows[l][2] += g.mesh.triangle_count() as u64;
                rows[l][3] += g.mesh.vertices.len() as u64;
                for q in g.mesh.vertices.chunks(4) {
                    if q[0].is_latent() {
                        rows[l][4] += 1;
                    }
                    if q.iter().any(|v| v.morph_delta() != 0.0) {
                        rows[l][5] += 1;
                    }
                    for v in q {
                        largest = largest.max(v.morph_delta().abs());
                    }
                }
            }
        }
        println!(
            "morph cost census around {eye:?}: {} nodes, radius 64, squash {SQUASH}",
            nodes.len()
        );
        println!("| level | nodes | triangles, plain | triangles, morph | change | vertices, plain | vertices, morph | change | latent quads | quads that move |");
        println!("|---|---|---|---|---|---|---|---|---|---|");
        let mut total = [0u64; 4];
        for (l, r) in rows.iter().enumerate() {
            if r[6] == 0 {
                continue;
            }
            for k in 0..4 {
                total[k] += r[k];
            }
            let pct = |a: u64, b: u64| 100.0 * (b as f64 - a as f64) / (a.max(1) as f64);
            println!(
                "| {l} | {} | {} | {} | {:+.1}% | {} | {} | {:+.1}% | {} | {} |",
                r[6],
                r[0],
                r[2],
                pct(r[0], r[2]),
                r[1],
                r[3],
                pct(r[1], r[3]),
                r[4],
                r[5]
            );
        }
        let pct = |a: u64, b: u64| 100.0 * (b as f64 - a as f64) / (a.max(1) as f64);
        println!(
            "| all | {} | {} | {} | {:+.1}% | {} | {} | {:+.1}% | | |",
            nodes.len(),
            total[0],
            total[2],
            pct(total[0], total[2]),
            total[1],
            total[3],
            pct(total[1], total[3])
        );
        println!(
            "largest delta a vertex carries: {largest} blocks (the format holds {})",
            cubara_voxel::MAX_MORPH_DELTA
        );
        assert!(
            largest < cubara_voxel::MAX_MORPH_DELTA,
            "a delta was clamped"
        );
    }

    #[test]
    fn a_morphing_columns_target_is_its_parents_top() {
        let world = crate::World::new();
        let snapshot = world.snapshot();
        let blocks = blocks();
        let (px, pz) = (336 * 16, 213 * 16);
        let surface = snapshot.surface_height(px, pz);
        for level in MORPH_LEVELS {
            let extent = 16 << level;
            let w = 1 << level;
            let node = NodeKey::new(
                level,
                [
                    px.div_euclid(extent),
                    surface.div_euclid(extent),
                    pz.div_euclid(extent),
                ],
            );
            let o = node.world_origin();
            let columns = node_columns(&snapshot, node, blocks).unwrap();
            for z in 0..16 {
                for x in 0..16 {
                    let [top, target, _, _] = columns.at(x, z);
                    let (cx, cz) = (o[0] + x * w, o[2] + z * w);
                    assert_eq!(top, snapshot.cell_surface_height(cx, cz, w) + 1 - o[1]);
                    if target != top {
                        let pw = 2 * w;
                        let parent = snapshot.cell_surface_height(
                            cx.div_euclid(pw) * pw,
                            cz.div_euclid(pw) * pw,
                            pw,
                        ) + 1
                            - o[1];
                        // Face rows, world: the parent-level node rows the
                        // top face is in, which the target never leaves.
                        let rows = 32 * w;
                        let face = top + o[1];
                        assert_ne!(face.rem_euclid(rows), 0, "a top on a floor moved");
                        let floor = (face - 1).div_euclid(rows) * rows - o[1];
                        assert_eq!(
                            target,
                            parent.clamp(floor, floor + rows),
                            "level {level} ({x}, {z})"
                        );
                    }
                }
            }
            // The ring of neighbours outside the node: each may be drawn a
            // level coarser, so its floor is the lower of its own top and its
            // parent's, and at full morph the lower of its target and its
            // parent's. Counted over the whole ring, since a flat spot would
            // not tell the right floor from the wrong one.
            let mut lowered = 0;
            for k in -1..=16 {
                for (x, z) in [(k, -1), (k, 16), (-1, k), (16, k)] {
                    let [top, target, floor, target_floor] = columns.at(x, z);
                    let (cx, cz) = (o[0] + x * w, o[2] + z * w);
                    let pw = 2 * w;
                    let parent = snapshot.cell_surface_height(
                        cx.div_euclid(pw) * pw,
                        cz.div_euclid(pw) * pw,
                        pw,
                    ) + 1
                        - o[1];
                    if level == 0 {
                        assert_eq!(floor, top.min(parent), "ring ({x}, {z})");
                        assert_eq!(target_floor, target.min(parent), "ring ({x}, {z})");
                    }
                    assert!(floor <= top && target_floor <= target);
                    lowered += (floor < top) as i32;
                }
            }
            println!("level {level}: {lowered} of 68 ring columns draw lower than their own top");
        }
    }

    /// **At full morph a node wears what its parent draws** (the colour
    /// morph, `docs/PHASE3_ARCHITECTURE.md` §9.6), on the real ground of the
    /// steep mountain at levels 0 to 2: every face that carries a parent
    /// lies, at full morph, on a face of the parent's own mesh facing the
    /// same way, in that face's texture layer, and each of its corners
    /// carries the openness the parent's triangle has there. Found here by
    /// a brute-force search of the parent's triangles -- not by
    /// `ParentFaces`, whose answer this checks. And most of the ground that
    /// morphs does carry a parent: of the tops of morphing columns, moving
    /// or not, at least [`MIN_PARENTED`].
    ///
    /// Goes red when the parent is read in the wrong frame (offset, axis),
    /// a face is given the wrong layer or corner, a riser row is mapped
    /// stretched rather than whole, or the keying leaves faces out.
    #[test]
    fn a_fully_morphed_node_wears_what_its_parent_draws() {
        /// The least share of morphing columns' tops that must carry a
        /// parent. Chosen, not derived: what does not is where the parent
        /// draws no top over the column at full morph (beside a pit the
        /// parent shows, under a node's floor), a few percent here.
        const MIN_PARENTED: f64 = 0.9;
        let registry = registry();
        let blocks = TerrainBlocks::from_registry(&registry);
        let world = crate::World::new();
        let snapshot = world.snapshot();
        // A layer per material, so a wrong layer shows.
        let layer = |name: &str| match name {
            "cubara:grass" => 1u32,
            "cubara:soil" => 2,
            "cubara:stone" => 3,
            _ => 0,
        };
        type Tri = [([f32; 3], f32); 3];
        let (mut checked, mut morphing_tops, mut parented_tops) = (0usize, 0usize, 0usize);
        for level in MORPH_LEVELS {
            for node in mountain_nodes(&snapshot, level) {
                let Some(child) =
                    crate::mesh::mesh_node(&snapshot, &registry, &layer, node, blocks)
                else {
                    continue;
                };
                let parent = node.parent();
                let Some(drawn) = crate::mesh::mesh_node_with(
                    &snapshot, &registry, &layer, parent, blocks, false,
                ) else {
                    continue;
                };
                let columns = node_columns(&snapshot, node, blocks).expect("a morphing level");
                let w = 1i32 << level;
                let (po, co) = (parent.world_origin(), node.world_origin());
                let offset = [po[0] - co[0], po[1] - co[1], po[2] - co[2]];
                // The parent's triangles by (face, plane), in the child's frame.
                let mut tris: std::collections::HashMap<(u8, i32), Vec<(u32, Tri)>> =
                    std::collections::HashMap::new();
                for quad in drawn.mesh.indices.chunks(6) {
                    let vs: Vec<_> = quad
                        .iter()
                        .map(|&i| drawn.mesh.vertices[i as usize])
                        .collect();
                    let face = vs[0].face();
                    let d = face as usize / 2;
                    let at = |v: cubara_voxel::Vertex| {
                        let p = [
                            v.x() as i32 + offset[0],
                            v.y() as i32 + offset[1],
                            v.z() as i32 + offset[2],
                        ];
                        (p.map(|c| c as f32), v.openness())
                    };
                    let plane = match d {
                        0 => vs[0].x() as i32 + offset[0],
                        1 => vs[0].y() as i32 + offset[1],
                        _ => vs[0].z() as i32 + offset[2],
                    };
                    for t in [[0, 1, 2], [3, 4, 5]] {
                        tris.entry((face as u8, plane))
                            .or_default()
                            .push((vs[0].tex_layer(), t.map(|k| at(vs[k]))));
                    }
                }
                // The parent's (layer, openness) at `p` on `face`, read off the
                // triangle that holds `toward`.
                let read = |face: cubara_voxel::Face, p: [f32; 3], toward: [f32; 3]| {
                    let d = face as usize / 2;
                    let (ua, va) = ((d + 1) % 3, (d + 2) % 3);
                    let plane = p[d].round() as i32;
                    let candidates = tris.get(&(face as u8, plane))?;
                    let bary = |t: &Tri, q: [f32; 3]| {
                        let (a, b, c) = (t[0].0, t[1].0, t[2].0);
                        let det =
                            (b[va] - c[va]) * (a[ua] - c[ua]) + (c[ua] - b[ua]) * (a[va] - c[va]);
                        if det.abs() < 1e-9 {
                            return None;
                        }
                        let w0 = ((b[va] - c[va]) * (q[ua] - c[ua])
                            + (c[ua] - b[ua]) * (q[va] - c[va]))
                            / det;
                        let w1 = ((c[va] - a[va]) * (q[ua] - c[ua])
                            + (a[ua] - c[ua]) * (q[va] - c[va]))
                            / det;
                        Some([w0, w1, 1.0 - w0 - w1])
                    };
                    candidates.iter().find_map(|(layer, t)| {
                        let w = bary(t, toward)?;
                        if w.iter().any(|&x| x < -1e-4) {
                            return None;
                        }
                        let w = bary(t, p)?;
                        let open = w[0] * t[0].1 + w[1] * t[1].1 + w[2] * t[2].1;
                        Some((*layer, open.clamp(0.0, 1.0)))
                    })
                };
                for q in child.mesh.vertices.chunks(4) {
                    let full: Vec<[f32; 3]> = q
                        .iter()
                        .map(|v| [v.x() as f32, v.y() as f32 + v.morph_delta(), v.z() as f32])
                        .collect();
                    let centre = full.iter().fold([0.0f32; 3], |a, c| {
                        [a[0] + c[0] / 4.0, a[1] + c[1] / 4.0, a[2] + c[2] / 4.0]
                    });
                    let face = q[0].face();
                    if face == cubara_voxel::Face::PosY {
                        let cx = q.iter().map(|v| v.x() as i32).min().unwrap() / w;
                        let cz = q.iter().map(|v| v.z() as i32).min().unwrap() / w;
                        let [top, ..] = columns.at(cx, cz);
                        if columns.morphs_at(cx, cz) && q[0].y() as i32 == top {
                            morphing_tops += 1;
                            parented_tops += q[0].parent().is_some() as usize;
                        }
                    }
                    for (v, p) in q.iter().zip(&full) {
                        let Some((layer, open)) = v.parent() else {
                            continue;
                        };
                        let toward = std::array::from_fn(|k| p[k] + 0.01 * (centre[k] - p[k]));
                        let Some((drawn_layer, drawn_open)) = read(face, *p, toward) else {
                            panic!(
                                "level {level} {node:?}: a {face:?} corner at {p:?} carries a \
                                 parent, and the parent draws no such face there"
                            );
                        };
                        assert_eq!(
                            layer, drawn_layer,
                            "level {level} {node:?} {face:?} at {p:?}"
                        );
                        assert!(
                            (open - drawn_open).abs() <= 1.0 / 255.0 + 1e-4,
                            "level {level} {node:?} {face:?} at {p:?}: carries {open}, the \
                             parent draws {drawn_open}"
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 10_000, "only {checked} corners checked");
        let share = parented_tops as f64 / morphing_tops as f64;
        println!(
            "{checked} corners checked; {parented_tops} of {morphing_tops} tops of morphing columns \
             carry a parent"
        );
        assert!(
            share >= MIN_PARENTED,
            "{share:.3} of morphing columns' tops carry a parent"
        );
    }
}

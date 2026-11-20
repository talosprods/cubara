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
//! cave mouth, a pit or an overhang among the rows it shows, or with
//! anything standing on it (a tree, a player's block) -- and every column
//! beside one, so that a moving face never shares an edge with a face that
//! stays. Level 3, whose next level is the far terrain (#422's join).

use std::f32::consts::SQRT_2;

use cubara_voxel::MorphColumns;

use crate::node::{NodeKey, RingSchedule};
use crate::{TerrainBlocks, WorldSnapshot};

/// Blocks per chunk, as a float.
const CHUNK: f32 = 16.0;

/// The levels whose nodes morph: those whose next coarser level is a voxel
/// level of the game's schedule. Level 3 meets the far terrain, whose join
/// is #422's.
pub const MORPH_LEVELS: std::ops::RangeInclusive<u32> = 0..=2;

/// The widest band, in blocks of the morph metric. **Chosen, not derived**:
/// wider spreads a change over more frames, and leaves less of the ring at
/// its own detail; 64 blocks is 2.7 s at the flight speed of 24 blocks/s.
/// Levels 1 and 2 get less than this anyway at the default schedule (32 and
/// 64 blocks): [`morph_bands`] gives each level all the room its ring leaves.
pub const MAX_BAND: f32 = 64.0;

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
/// ring's inner edge. `None` for the last ring (its join is the far
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
            0 => 0.0,
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
/// continuous -- horizontal distance, or `squash` times the vertical one,
/// whichever is larger.
pub fn morph_metric(rel: [f32; 3], squash: i32) -> f32 {
    rel[0].hypot(rel[2]).max(squash.max(1) as f32 * rel[1].abs())
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
/// mouth, pit or overhang there), with nothing standing on it at level 0
/// (no tree, no player's block). Its target is then its parent cell's top;
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
                (lowest..=top).all(|y| world.certainly_solid_at(x, y, z, top))
                    && !world.is_solid_at(x, top + 1, z, blocks)
            } else {
                let cell = world.cell_ground(x, z, w);
                (lowest..=top).all(|y| world.certainly_solid_at_v(y, level, &cell))
            };
        }
    }
    let morphs = |cx: i32, cz: i32| {
        (-1..=1).all(|dz| (-1..=1).all(|dx| pure[at(cx + dx, cz + dz)]))
    };

    let row = |y_top_block: i32| y_top_block + 1 - oy;
    let mut columns = Vec::with_capacity(((n + 2) * (n + 2)) as usize);
    for cz in -1..=n {
        for cx in -1..=n {
            let top = tops[at(cx, cz)];
            let parent = parent_top(cx, cz);
            let target = if morphs(cx, cz) { parent } else { top };
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
        let registry = BlockRegistry::from_materials(vec![
            material("cubara:grass"),
            material("cubara:soil"),
            material("cubara:stone"),
        ])
        .expect("fixture registry is valid");
        TerrainBlocks::from_registry(&registry)
    }

    #[test]
    fn the_default_schedule_has_a_band_for_levels_0_to_2_and_none_for_3() {
        let bands = morph_bands(DEFAULT_RING_SCHEDULE, SQUASH);
        // 160 m edge: ends 32 before it, 64 wide (the cap).
        assert_eq!(bands[0], Some(MorphBand { start: 64.0, end: 128.0 }));
        // 288 m edge: the ring leaves 32 blocks between the inner reach of a
        // level-1 node (160 + 64) and the outer margin (288 - 32).
        assert_eq!(bands[1], Some(MorphBand { start: 224.0, end: 256.0 }));
        // 512 m edge: 288 + 128 to 512 - 32.
        assert_eq!(bands[2], Some(MorphBand { start: 416.0, end: 480.0 }));
        assert_eq!(bands[3], None, "level 3 meets the far terrain");
    }

    #[test]
    fn a_ring_too_narrow_for_its_nodes_has_no_band() {
        // Every radius halved: the level-1 ring is 4 chunks wide, but a
        // level-1 node beside level 0 reaches 4 chunks past its inner edge
        // and the band must end 2 chunks before its outer one.
        let half = [(0u32, 5i32), (1, 9), (2, 16), (3, 32)];
        let bands = morph_bands(&half, SQUASH);
        assert_eq!(bands[0], Some(MorphBand { start: 0.0, end: 48.0 }));
        assert_eq!(bands[1], None);
        assert_eq!(bands[2], None);
    }

    #[test]
    fn the_factor_is_zero_before_the_band_one_after_and_linear_in_it() {
        let b = MorphBand { start: 100.0, end: 140.0 };
        assert_eq!(morph_factor(0.0, b), 0.0);
        assert_eq!(morph_factor(100.0, b), 0.0);
        assert_eq!(morph_factor(110.0, b), 0.25);
        assert_eq!(morph_factor(140.0, b), 1.0);
        assert_eq!(morph_factor(1e9, b), 1.0);
    }

    #[test]
    fn the_metric_is_the_selections_distance() {
        assert_eq!(morph_metric([3.0, 0.0, 4.0], 2), 5.0);
        assert_eq!(morph_metric([3.0, 10.0, 4.0], 2), 20.0);
        assert_eq!(morph_metric([3.0, -2.0, 4.0], 2), 5.0);
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
            let nodes: std::collections::HashSet<NodeKey> =
                desired_nodes_3d(ChunkCoord::from_world_pos(eye), SQUASH, DEFAULT_RING_SCHEDULE)
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
                                riser_map(own, nb).map(|[s0, s1, t0, t1]| (lerp(s0, t0), lerp(s1, t1)))
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
                        );
                        assert_eq!(target, parent + 1 - o[1], "level {level} ({x}, {z})");
                    }
                }
            }
        }
    }
}

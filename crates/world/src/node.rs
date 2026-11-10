//! LOD node addressing and node-based streaming policy
//! (`docs/PHASE1_ARCHITECTURE.md` §6.1, §6.3, issue #38 sub-issue #105).
//!
//! An LOD node is one mesh, one arena allocation, and one draw, covering
//! `2^level × 2^level × 2^level` chunks — level 0 is a single chunk, level 3
//! covers 512 chunks in one draw. That is where the ~25× draw-count
//! reduction radius 64 needs comes from (§2: draws, not triangles, are the
//! binding constraint; today's per-chunk LOD only reduces triangles while
//! still issuing one draw per chunk).
//!
//! This module is addressing and streaming policy only — pure book-keeping,
//! no generation, no meshing, no GPU, mirroring [`crate::streaming`]'s own
//! shape exactly. It does not yet drive the live renderer: the existing
//! chunk-based [`crate::streaming::desired_chunks`]/`lod_for`/`plan_updates`
//! keep doing that until a later sub-issue in the #38 arc supersedes them.

use std::collections::HashSet;
use std::ops::RangeInclusive;

use cubara_voxel::{Chunk, ChunkCoord};

/// The address of one LOD node: `level` (0 = one chunk, doubling in extent
/// each level up) and `pos`, the node's own position on *that level's*
/// grid — not a chunk coordinate. A level-`L` node's chunk-space footprint
/// is `pos * 2^L .. pos*2^L + 2^L` on each axis.
///
/// `Ord` is derived (fields compared in declaration order: `level` then
/// `pos`) — total and fixed, which is what node meshing (a later sub-issue)
/// needs for a merge order that never depends on which worker finishes
/// first (issue #83).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct NodeKey {
    pub level: u32,
    pub pos: [i32; 3],
}

impl NodeKey {
    pub const fn new(level: u32, pos: [i32; 3]) -> Self {
        Self { level, pos }
    }

    /// The node one level coarser that contains this one.
    pub fn parent(self) -> Self {
        Self::new(self.level + 1, self.pos.map(|p| p.div_euclid(2)))
    }

    /// The eight nodes one level finer that tile this one, in the order of
    /// their sub-block in it (`x` fastest, then `y`, then `z` -- the order
    /// `visibility::NodeLinks` is in). Empty at level 0.
    pub fn children(self) -> Vec<Self> {
        let Some(finer) = self.level.checked_sub(1) else {
            return Vec::new();
        };
        let [x, y, z] = self.pos;
        (0..8)
            .map(|i| {
                Self::new(
                    finer,
                    [2 * x + (i & 1), 2 * y + (i >> 1 & 1), 2 * z + (i >> 2)],
                )
            })
            .collect()
    }

    /// How many chunks this node spans on each axis: `2^level`.
    pub fn extent_chunks(self) -> i32 {
        1 << self.level
    }

    /// This level's vertical cell height in blocks (R1b, issue #327 decision
    /// 1): `v = 1` at levels 1-2, `v = 4` at level 3 and beyond (ring 3's
    /// candidate (i) -- M4 decides whether that holds past it). Level 0's
    /// cells are always 1 block on every axis regardless of this (it stays a
    /// plain [`Chunk`](cubara_voxel::Chunk), decision 6), so this is only
    /// meaningful for `level >= 1`.
    ///
    /// A node's **horizontal** cell width is `2^level` blocks, unchanged --
    /// that's [`extent_chunks`](Self::extent_chunks) itself, numerically.
    pub fn cell_height_blocks(self) -> i32 {
        if self.level <= 2 {
            1
        } else {
            4
        }
    }

    /// This node's own lattice cell counts `[nx, ny, nz]` (R1b): 16 on x/z,
    /// always, and `16 * extent_chunks() / cell_height_blocks()` on y -- the
    /// same formula [`crate::worldgen::WorldGen::generate_aniso`] sizes its
    /// lattice with, so a caller that needs to know a node's own vertical
    /// cell count (`cubara_world::mesh::border_samples`,
    /// `visibility::node_links`) doesn't need to generate it to find out.
    /// `[16, 16, 16]` at level 0 -- never actually consulted there, since
    /// level 0 is a plain [`Chunk`](cubara_voxel::Chunk), but correct by the
    /// same formula regardless (`extent_chunks`/`cell_height_blocks` are
    /// both 1).
    pub fn lattice_dims(self) -> [i32; 3] {
        let ny = (16 * self.extent_chunks()) / self.cell_height_blocks();
        [16, ny, 16]
    }

    /// The node's minimum-corner chunk.
    pub fn chunk_origin(self) -> ChunkCoord {
        let extent = self.extent_chunks();
        ChunkCoord::new(
            self.pos[0] * extent,
            self.pos[1] * extent,
            self.pos[2] * extent,
        )
    }

    /// The node's minimum-corner world-block coordinate — what
    /// `WorldGen::generate`'s `origin` parameter wants (a later sub-issue).
    pub fn world_origin(self) -> [i32; 3] {
        let size = Chunk::SIZE as i32;
        let c = self.chunk_origin();
        [c.x * size, c.y * size, c.z * size]
    }

    /// The node at `level` that contains `chunk` — floor division
    /// (`div_euclid`, correct for negative coordinates) by `2^level`, the
    /// same convention `ChunkCoord::from_block`/`cubara_world::region`
    /// already established for block 1.9's save/load.
    pub fn containing(chunk: ChunkCoord, level: u32) -> Self {
        let extent = 1i32 << level;
        Self {
            level,
            pos: [
                chunk.x.div_euclid(extent),
                chunk.y.div_euclid(extent),
                chunk.z.div_euclid(extent),
            ],
        }
    }
}

/// The load/unload deltas needed to turn a resident node set into the
/// desired one — the node-keyed equivalent of
/// [`crate::streaming::StreamUpdates`].
#[derive(Debug, Default, PartialEq, Eq)]
pub struct NodeStreamUpdates {
    pub to_load: Vec<NodeKey>,
    pub to_unload: Vec<NodeKey>,
}

/// A `(level, outer_radius_in_chunks)` ring: everything out to
/// `outer_radius_in_chunks` (Chebyshev, horizontal) is at least this
/// detailed. Radii must increase with each successive entry; levels need
/// not be contiguous, though every current use is (§6.3's example: level 0
/// to 8, level 1 to 16, level 2 to 32, level 3 to 64).
pub type RingSchedule = [(u32, i32)];

/// Tuned against the real radius-64 draw budget (§2, issue #109): resident
/// node count at radius 64 must stay under 2,000. Measured (not guessed) at
/// four different world positions -- origin plus three far-apart points, to
/// rule out a schedule that only happens to work near `(0,0,0)` -- landing
/// at **1,585-1,613 resident nodes**, a comfortable ~20% margin under
/// budget while widening the full-resolution near field from chunk-radius 8
/// (§6.3's illustrative example) to 10 and level 1 from 16 to 18, the two
/// changes that matter most for how coarse the ground right around the
/// player looks. Levels 2/3 (and the radius-64 ceiling itself) are
/// untouched: they're already far enough out that widening them buys much
/// less visible quality per node than widening the near field does. A
/// tighter table (11/19/33/64) measured 1,935-1,956 -- technically still
/// under 2,000, but only a 2-3% margin, too close to the edge to trust
/// across world positions and seeds this table wasn't measured against; not
/// used for exactly that reason. See `BENCHMARKS.md`'s issue-#109 rows for
/// the full `--bench 64` numbers this table produces.
pub const DEFAULT_RING_SCHEDULE: &[(u32, i32)] = &[(0, 10), (1, 18), (2, 32), (3, 64)];

/// Every node that should be resident around `center`, across the vertical
/// band `y_range` (in chunks, same convention as
/// [`crate::streaming::desired_chunks`]'s own `y_range`), per `schedule`.
///
/// **Built as an octree, so the rings tile space exactly.** Start from the
/// coarsest level's nodes within its radius, and split any node that reaches
/// inside the next finer ring's radius into its eight children, down to level
/// 0. Every chunk inside the outer radius then belongs to exactly one node.
///
/// It used to resolve each ring on its own grid and exclude the finer ring by
/// a rounded-down node radius, which left chunks no ring covered -- chunk 11
/// and 36-39 out from the player, 603 of the 14,641 chunks within 60 -- and
/// others two rings both drew. The gaps looked like trenches while every node
/// walled its borders; once covered border faces were left out they became
/// holes straight through the ground.
///
/// **Neighbours differ by at most one level** wherever rings are far enough
/// apart for it, which the default schedule is -- and which leaving covered
/// border faces out relies on (`crate::mesh`). A test holds both properties.
pub fn desired_nodes(
    center: ChunkCoord,
    y_range: RangeInclusive<i32>,
    schedule: &RingSchedule,
) -> Vec<NodeKey> {
    select_nodes(center, Vertical::Band(y_range), schedule)
}

/// Every node that should be resident around `center`, **in all three
/// dimensions**: no vertical band, and detail falling off with height and
/// depth the way it does with horizontal distance.
///
/// **How far is the same in every direction**: the outer radius is a cube, so
/// a player 300 blocks up still sees the ground below them as far as they see
/// it ahead. **How detailed falls off faster vertically**: a node is split
/// towards level 0 by `max(|dx|, |dz|, squash * |dy|)`, so `squash` says how
/// many times sooner detail coarsens with height and depth than with
/// horizontal distance.
///
/// Both halves matter. Squashing the radius as well was measured first, and at
/// `squash` 8 a camera 300 blocks up drew nothing at all -- the render distance
/// straight down had become 128 blocks.
pub fn desired_nodes_3d(center: ChunkCoord, squash: i32, schedule: &RingSchedule) -> Vec<NodeKey> {
    select_nodes(center, Vertical::Squash(squash.max(1)), schedule)
}

/// The box of chunks [`desired_nodes_3d`] fills around `center`, as
/// `(min, max)` with `max` exclusive: every chunk inside is covered by exactly
/// one node, and nothing outside it is. What the far terrain leaves alone
/// (`crate::far::Hole`), because the voxel nodes draw it themselves.
///
/// Not a cube of the schedule's radius exactly: the coarsest nodes sit on
/// their own grid, and every one of them within reach is kept whole.
pub fn covered_box_3d(center: ChunkCoord, schedule: &RingSchedule) -> ([i32; 3], [i32; 3]) {
    let Some(&(top, radius)) = schedule.last() else {
        let c = [center.x, center.y, center.z];
        return (c, c);
    };
    let extent = 1i32 << top;
    let lo = |c: i32| (c - radius).div_euclid(extent) * extent;
    let hi = |c: i32| ((c + radius).div_euclid(extent) + 1) * extent;
    (
        [lo(center.x), lo(center.y), lo(center.z)],
        [hi(center.x), hi(center.y), hi(center.z)],
    )
}

/// How [`select_nodes`] treats the vertical axis.
enum Vertical {
    /// Only nodes overlapping these chunk-layers, at any height distance.
    Band(RangeInclusive<i32>),
    /// Vertical distance counts this many times horizontal distance.
    Squash(i32),
}

/// The octree both selections share -- one implementation (Rule 5).
fn select_nodes(center: ChunkCoord, vertical: Vertical, schedule: &RingSchedule) -> Vec<NodeKey> {
    let Some(&(top, top_radius)) = schedule.last() else {
        return Vec::new();
    };
    // The finest level a node at `level` must be split down towards: the ring
    // radius of the level below it, or none for the finest ring.
    let radius_of = |level: u32| schedule.iter().find(|&&(l, _)| l == level).map(|&(_, r)| r);
    let finest = schedule[0].0;
    let (y_lo, y_hi, squash) = match &vertical {
        Vertical::Band(r) => (*r.start(), *r.end(), 0),
        Vertical::Squash(k) => (center.y - top_radius, center.y + top_radius, *k),
    };
    // What is drawn at all is a cube in 3D (vertical distance counted once),
    // measured the old way (Chebyshev) so the outer reach -- and therefore
    // `covered_box_3d` and the far terrain's hole -- does not move (R3a keeps
    // the cube reach, issue #327 §3). How finely it is drawn, horizontally,
    // is measured by Euclidean distance instead: Chebyshev rings are
    // squares, reaching `sqrt(2)` further on the diagonals than straight
    // ahead, so a node there split by mistake. A circle doesn't -- see
    // `node_distance_components`'s doc comment for why only the horizontal
    // half changes shape, with the vertical one still combined by `max`.
    let reach = |node: NodeKey| node_distance(node, center, squash.min(1));
    let splits_within = |node: NodeKey, r: i32| {
        let (h_sq, v) = node_distance_components(node, center, squash);
        v <= r && h_sq <= i64::from(r) * i64::from(r)
    };

    let extent = 1i32 << top;
    let first = |c: i32| (c - top_radius).div_euclid(extent);
    let last = |c: i32| (c + top_radius).div_euclid(extent);
    let mut pending: Vec<NodeKey> = Vec::new();
    for nx in first(center.x)..=last(center.x) {
        for nz in first(center.z)..=last(center.z) {
            for ny in y_lo.div_euclid(extent)..=y_hi.div_euclid(extent) {
                let node = NodeKey::new(top, [nx, ny, nz]);
                if reach(node) <= top_radius {
                    pending.push(node);
                }
            }
        }
    }

    let mut nodes = Vec::new();
    while let Some(node) = pending.pop() {
        let finer = node.level.checked_sub(1).filter(|&l| l >= finest);
        let split = finer
            .and_then(radius_of)
            .is_some_and(|r| splits_within(node, r));
        if !split {
            nodes.push(node);
            continue;
        }
        let level = node.level - 1;
        for dx in 0..2 {
            for dy in 0..2 {
                for dz in 0..2 {
                    let child = NodeKey::new(
                        level,
                        [
                            node.pos[0] * 2 + dx,
                            node.pos[1] * 2 + dy,
                            node.pos[2] * 2 + dz,
                        ],
                    );
                    // A band keeps only children overlapping it. In 3D the
                    // parent was already inside the render distance, so all
                    // eight fill it -- dropping one would be a gap.
                    let o = child.chunk_origin().y;
                    let in_band = o + child.extent_chunks() > y_lo && o <= y_hi;
                    if squash > 0 || in_band {
                        pending.push(child);
                    }
                }
            }
        }
    }
    nodes.sort();
    nodes
}

/// Chebyshev distance in chunks from `center` to the nearest chunk of `node`,
/// with vertical distance counted `squash` times -- `0` ignores height, which
/// is what a band selection wants.
///
/// Used only for [`select_nodes`]'s outer cube (`reach`): the top-level
/// radius that bounds the whole selection, and therefore [`covered_box_3d`]
/// and the far terrain's hole, stays exactly what it is today (R3a keeps the
/// cube reach, issue #327 §3 R3a). [`node_distance_components`] below is the split
/// test's own distance, and the two are deliberately different functions
/// rather than one with a flag, so a future change to one cannot silently
/// reach into the other.
fn node_distance(node: NodeKey, center: ChunkCoord, squash: i32) -> i32 {
    let o = node.chunk_origin();
    let e = node.extent_chunks();
    let axis = |c: i32, lo: i32| (lo - c).max(c - (lo + e - 1)).max(0);
    axis(center.x, o.x)
        .max(axis(center.z, o.z))
        .max(axis(center.y, o.y) * squash)
}

/// `(horizontal distance squared, squashed vertical distance)` from `center`
/// to the nearest chunk of `node`'s box -- the split test's own distance
/// (R3a, issue #327 §3 R3a), split into its two components rather than
/// combined into one number.
///
/// [`node_distance`]'s Chebyshev distance makes a horizontal ring a square: a
/// node on the diagonal is still "within radius `r`" at up to `r * sqrt(2)`
/// out, so the finer ring reaches further there than straight ahead --
/// diagonal views get more detail than the schedule intends, for free.
/// Comparing the squared horizontal distance against `r * r` instead makes
/// the ring a circle: the worst-case (on-axis) cell size the schedule
/// promises is unchanged, and about `1 - pi/4` (~21%) of each inner ring's
/// *area* -- the part only the diagonals were reaching -- splits one level
/// coarser instead.
///
/// The vertical component stays linear and un-squared, combined with the
/// horizontal one by [`select_nodes`] via **`max`, exactly as
/// [`node_distance`] combines all three axes** -- not folded into one 3D
/// Euclidean sum. A full 3D Euclidean distance was tried first and measured
/// to break `nodes_that_share_a_face_differ_by_at_most_one_level`'s 3D
/// counterpart (`in_three_dimensions_the_rings_still_tile_and_stay_balanced`)
/// at `squash` 4: a vertical and a horizontal gap that each individually sit
/// right at a ring's radius add enough under Euclidean's sum-of-squares to
/// push a node past it when neither alone would, pulling the split boundary
/// in by more than one level between some vertically- and
/// horizontally-adjacent nodes. Circular *only* on the horizontal plane (what
/// the proposal's R3a actually measures -- it is about the ground-level
/// diagonal view, not the squash) keeps the vertical axis's contribution
/// exactly as it was, which keeps that test passing.
///
/// The horizontal half is returned squared, and in `i64`, so comparing it
/// against `r * r` is exact integer arithmetic with no `sqrt` and no risk of
/// a 32-bit overflow (a chunk-space gap can reach into the low thousands once
/// squared) -- matching Rule 1's "no floating-point where integers will do":
/// `sqrt` is IEEE-754 exact, but squaring both sides of the comparison avoids
/// it entirely rather than relying on that.
fn node_distance_components(node: NodeKey, center: ChunkCoord, squash: i32) -> (i64, i32) {
    let o = node.chunk_origin();
    let e = node.extent_chunks();
    let axis = |c: i32, lo: i32| (lo - c).max(c - (lo + e - 1)).max(0);
    let dx = i64::from(axis(center.x, o.x));
    let dz = i64::from(axis(center.z, o.z));
    let dy = axis(center.y, o.y) * squash;
    (dx * dx + dz * dz, dy)
}

/// Compute the [`NodeStreamUpdates`] that move `resident` to exactly the
/// set [`desired_nodes`] wants around `center`. `to_load` is nearest-first
/// (Chebyshev, in that node's own level-scaled units) so nodes around the
/// camera stream in before the fringe -- mirrors
/// [`crate::streaming::plan_updates`] exactly.
pub fn plan_node_updates(
    resident: &HashSet<NodeKey>,
    center: ChunkCoord,
    y_range: RangeInclusive<i32>,
    schedule: &RingSchedule,
) -> NodeStreamUpdates {
    plan_updates_to(resident, center, desired_nodes(center, y_range, schedule))
}

/// [`plan_node_updates`] for a 3D selection ([`desired_nodes_3d`]).
pub fn plan_node_updates_3d(
    resident: &HashSet<NodeKey>,
    center: ChunkCoord,
    squash: i32,
    schedule: &RingSchedule,
) -> NodeStreamUpdates {
    plan_updates_to(resident, center, desired_nodes_3d(center, squash, schedule))
}

/// The updates that turn `resident` into exactly `desired`, loading nearest
/// first.
fn plan_updates_to(
    resident: &HashSet<NodeKey>,
    center: ChunkCoord,
    desired: Vec<NodeKey>,
) -> NodeStreamUpdates {
    let desired_set: HashSet<NodeKey> = desired.iter().copied().collect();

    let mut to_load: Vec<NodeKey> = desired
        .into_iter()
        .filter(|n| !resident.contains(n))
        .collect();
    to_load.sort_by_key(|n| node_dist_sq(*n, center));

    let to_unload: Vec<NodeKey> = resident
        .iter()
        .filter(|n| !desired_set.contains(n))
        .copied()
        .collect();

    NodeStreamUpdates { to_load, to_unload }
}

/// [`DEFAULT_RING_SCHEDULE`], truncated at `radius`: every ring whose outer
/// radius is under `radius` is kept as-is, and the first ring that would
/// reach or exceed it is clamped to end exactly at `radius` instead. Turns a
/// single flat `radius` parameter (what a bench/screenshot/golden-test scene
/// asks for) into a real, distance-falloff node schedule rather than a flat
/// level-0 one -- `radius` under the first ring's own outer radius collapses
/// to a single `[(0, radius)]` entry (a small scene stays level-0 only,
/// pixel-identical to meshing it as plain chunks), while a large `radius`
/// (e.g. matching the schedule's own outer ring) reproduces
/// `DEFAULT_RING_SCHEDULE` exactly.
pub fn schedule_for_radius(radius: i32) -> Vec<(u32, i32)> {
    let mut schedule = Vec::new();
    for &(level, outer) in DEFAULT_RING_SCHEDULE {
        if outer >= radius {
            schedule.push((level, radius));
            break;
        }
        schedule.push((level, outer));
    }
    if schedule.is_empty() {
        schedule.push((0, radius));
    }
    schedule
}

/// Sort `nodes` so the ones nearest `center` come first -- the order to load
/// them in, so what is around the camera arrives before the fringe.
pub fn sort_nearest_first(nodes: &mut [NodeKey], center: ChunkCoord) {
    nodes.sort_by_key(|n| (node_dist_sq(*n, center), *n));
}

/// Squared horizontal chunk-space distance from a node's own center to the
/// center of `center` (the player's chunk, not its minimum corner) -- the
/// sort key [`plan_node_updates`] loads nearest-first by. Both centers are
/// computed in the same doubled-integer units (`origin*2 + extent`, so a
/// half-chunk offset stays exact without floats); comparing a node's real
/// center against `center`'s own *corner* instead would make the two chunks
/// straddling that corner tie in distance and sort arbitrarily.
fn node_dist_sq(node: NodeKey, center: ChunkCoord) -> i64 {
    let extent = node.extent_chunks() as i64;
    let origin = node.chunk_origin();
    let node_center_x = origin.x as i64 * 2 + extent;
    let node_center_y = origin.y as i64 * 2 + extent;
    let node_center_z = origin.z as i64 * 2 + extent;
    let center_x = center.x as i64 * 2 + 1;
    let center_y = center.y as i64 * 2 + 1;
    let center_z = center.z as i64 * 2 + 1;
    let dx = node_center_x - center_x;
    let dy = node_center_y - center_y;
    let dz = node_center_z - center_z;
    // Height counts too: with nothing limiting the render distance vertically,
    // the ground under a player flying high must not wait behind sky.
    dx * dx + dy * dy + dz * dz
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn children_tile_their_parent_in_sub_block_order() {
        for node in [NodeKey::new(2, [0, 0, 0]), NodeKey::new(1, [-3, 5, -1])] {
            let children = node.children();
            assert_eq!(children.len(), 8);
            let o = node.chunk_origin();
            let half = node.extent_chunks() / 2;
            for (i, child) in children.iter().enumerate() {
                assert_eq!(child.parent(), node, "{child:?}");
                let c = child.chunk_origin();
                let at = [(c.x - o.x) / half, (c.y - o.y) / half, (c.z - o.z) / half];
                assert_eq!(at[0] as usize + 2 * at[1] as usize + 4 * at[2] as usize, i);
            }
        }
        assert!(NodeKey::new(0, [4, 4, 4]).children().is_empty());
    }

    #[test]
    fn extent_chunks_doubles_per_level() {
        assert_eq!(NodeKey::new(0, [0, 0, 0]).extent_chunks(), 1);
        assert_eq!(NodeKey::new(1, [0, 0, 0]).extent_chunks(), 2);
        assert_eq!(NodeKey::new(3, [0, 0, 0]).extent_chunks(), 8);
    }

    /// R1b's decision 1, pinned directly: `v = 1` at levels 0-2, `v = 4` at
    /// level 3. Nothing else in this crate's test suite fails if level 3's
    /// `4` regresses to `1` -- `coarse_tops_are_the_surface` (world.rs)
    /// derives its own tolerance from this same function, so a bug here
    /// moves its expectation along with its answer; `a_coarse_node_averages_
    /// like_its_blocks` (`cubara-render`) only asserts rings 1-2 strictly,
    /// and ring 3's golden scenes don't happen to put a riser where the
    /// difference crosses tolerance. This is the one test standing between
    /// "level 3's v matches the issue" and nothing noticing.
    #[test]
    fn cell_height_blocks_is_one_except_level_three() {
        assert_eq!(NodeKey::new(0, [0, 0, 0]).cell_height_blocks(), 1);
        assert_eq!(NodeKey::new(1, [0, 0, 0]).cell_height_blocks(), 1);
        assert_eq!(NodeKey::new(2, [0, 0, 0]).cell_height_blocks(), 1);
        assert_eq!(NodeKey::new(3, [0, 0, 0]).cell_height_blocks(), 4);
    }

    #[test]
    fn chunk_origin_scales_pos_by_extent() {
        assert_eq!(
            NodeKey::new(2, [1, -1, 3]).chunk_origin(),
            ChunkCoord::new(4, -4, 12)
        );
        assert_eq!(
            NodeKey::new(0, [5, 6, 7]).chunk_origin(),
            ChunkCoord::new(5, 6, 7),
            "level 0 is exactly a chunk coordinate"
        );
    }

    #[test]
    fn world_origin_scales_by_chunk_size_too() {
        assert_eq!(
            NodeKey::new(1, [1, 0, -1]).world_origin(),
            [32, 0, -32] // chunk (2, 0, -2) * 16
        );
    }

    #[test]
    fn containing_floors_toward_negative_infinity() {
        // Level 1 (extent 2): chunk -1 belongs to node covering chunks [-2, -1].
        assert_eq!(
            NodeKey::containing(ChunkCoord::new(-1, 0, 0), 1),
            NodeKey::new(1, [-1, 0, 0])
        );
        // Chunk -2 starts a new node.
        assert_eq!(
            NodeKey::containing(ChunkCoord::new(-2, 0, 0), 1),
            NodeKey::new(1, [-1, 0, 0])
        );
        assert_eq!(
            NodeKey::containing(ChunkCoord::new(-3, 0, 0), 1),
            NodeKey::new(1, [-2, 0, 0])
        );
    }

    #[test]
    fn containing_is_the_inverse_of_chunk_origin_at_the_corner() {
        for level in 0..4u32 {
            for &c in &[-40, -17, -1, 0, 1, 16, 39] {
                let key = NodeKey::containing(ChunkCoord::new(c, c, c), level);
                let origin = key.chunk_origin();
                let extent = key.extent_chunks();
                assert!(
                    (origin.x..origin.x + extent).contains(&c),
                    "level {level} c {c}: origin {origin:?} extent {extent}"
                );
            }
        }
    }

    #[test]
    fn level_0_desired_nodes_matches_chunk_streaming_shape() {
        // At level 0 only (a single-ring schedule), node == chunk 1:1, so
        // this should match `streaming::desired_chunks`'s own count exactly.
        let schedule = [(0u32, 2i32)];
        let nodes = desired_nodes(ChunkCoord::new(0, 0, 0), 0..=1, &schedule);
        assert_eq!(nodes.len(), 5 * 5 * 2, "(2*2+1)^2 columns * 2 y layers");
        for n in &nodes {
            assert_eq!(n.level, 0);
        }
    }

    /// Which node covers each chunk, for chunks within the band.
    /// The far terrain's hole is this box, so it must be exactly what the
    /// nodes cover: a chunk inside it that no node covers is a hole in the
    /// world, and a node outside it is ground drawn twice.
    #[test]
    fn covered_box_3d_is_exactly_what_the_nodes_cover() {
        let schedule = [(0u32, 3i32), (1, 6), (2, 12)];
        for centre in [
            ChunkCoord::new(0, 0, 0),
            ChunkCoord::new(-5, 3, 9),
            ChunkCoord::new(13, -7, -2),
        ] {
            let nodes = desired_nodes_3d(centre, 2, &schedule);
            let map = coverage(&nodes);
            let (min, max) = covered_box_3d(centre, &schedule);
            for x in min[0]..max[0] {
                for y in min[1]..max[1] {
                    for z in min[2]..max[2] {
                        let n = map.get(&(x, y, z)).map_or(0, Vec::len);
                        assert_eq!(n, 1, "{centre:?}: chunk ({x}, {y}, {z}) covered {n} times");
                    }
                }
            }
            let inside = (max[0] - min[0]) * (max[1] - min[1]) * (max[2] - min[2]);
            assert_eq!(
                map.len() as i32,
                inside,
                "{centre:?}: nodes reach outside the box"
            );
        }
    }

    fn coverage(nodes: &[NodeKey]) -> std::collections::HashMap<(i32, i32, i32), Vec<NodeKey>> {
        let mut map: std::collections::HashMap<(i32, i32, i32), Vec<NodeKey>> = Default::default();
        for &n in nodes {
            let (o, e) = (n.chunk_origin(), n.extent_chunks());
            for x in o.x..o.x + e {
                for y in o.y..o.y + e {
                    for z in o.z..o.z + e {
                        map.entry((x, y, z)).or_default().push(n);
                    }
                }
            }
        }
        map
    }

    const CENTRES: [ChunkCoord; 4] = [
        ChunkCoord::new(0, 0, 0),
        ChunkCoord::new(3, 1, -5),
        ChunkCoord::new(-17, -2, 29),
        ChunkCoord::new(7, 0, 7),
    ];

    /// **No chunk twice, no chunk missing** anywhere inside the render
    /// distance. A gap is a hole through the ground; an overlap is two levels
    /// of the same terrain fighting for the same pixels.
    #[test]
    fn the_rings_tile_every_chunk_within_the_radius_exactly_once() {
        let outer = DEFAULT_RING_SCHEDULE.last().unwrap().1;
        for centre in CENTRES {
            let band = (centre.y - 2)..=(centre.y + 2);
            let map = coverage(&desired_nodes(centre, band.clone(), DEFAULT_RING_SCHEDULE));
            for x in (centre.x - outer)..=(centre.x + outer) {
                for z in (centre.z - outer)..=(centre.z + outer) {
                    for y in band.clone() {
                        let n = map.get(&(x, y, z)).map_or(0, Vec::len);
                        assert_eq!(
                            n, 1,
                            "centre {centre:?}: chunk ({x}, {y}, {z}) covered {n} times"
                        );
                    }
                }
            }
        }
    }

    /// Leaving out a border face its neighbour covers is only safe against a
    /// neighbour at most one level coarser or finer (`crate::mesh`).
    #[test]
    fn nodes_that_share_a_face_differ_by_at_most_one_level() {
        for centre in CENTRES {
            let band = (centre.y - 2)..=(centre.y + 2);
            let nodes = desired_nodes(centre, band, DEFAULT_RING_SCHEDULE);
            let map = coverage(&nodes);
            for &(x, y, z) in map.keys() {
                let here = map[&(x, y, z)][0].level as i32;
                for d in [
                    (1, 0, 0),
                    (-1, 0, 0),
                    (0, 1, 0),
                    (0, -1, 0),
                    (0, 0, 1),
                    (0, 0, -1),
                ] {
                    if let Some(there) = map.get(&(x + d.0, y + d.1, z + d.2)) {
                        let gap = (here - there[0].level as i32).abs();
                        assert!(
                            gap <= 1,
                            "centre {centre:?}: chunk ({x},{y},{z}) level {here} beside level {}",
                            there[0].level
                        );
                    }
                }
            }
        }
    }

    /// The same two properties with no band: every chunk within the 3D render
    /// distance exactly once, and face-neighbours at most one level apart --
    /// for every vertical squash worth using.
    #[test]
    fn in_three_dimensions_the_rings_still_tile_and_stay_balanced() {
        let schedule = &[(0u32, 6), (1, 12), (2, 24), (3, 40)];
        for squash in 1..=4 {
            for centre in [ChunkCoord::new(0, 0, 0), ChunkCoord::new(-9, 13, 4)] {
                let nodes = desired_nodes_3d(centre, squash, schedule);
                let map = coverage(&nodes);
                let r = 40;
                for x in (centre.x - r)..=(centre.x + r) {
                    for z in (centre.z - r)..=(centre.z + r) {
                        for y in (centre.y - r)..=(centre.y + r) {
                            let n = map.get(&(x, y, z)).map_or(0, Vec::len);
                            assert_eq!(n, 1, "squash {squash}, centre {centre:?}: ({x},{y},{z}) covered {n} times");
                        }
                    }
                }
                for (&(x, y, z), here) in &map {
                    for d in [
                        (1, 0, 0),
                        (-1, 0, 0),
                        (0, 1, 0),
                        (0, -1, 0),
                        (0, 0, 1),
                        (0, 0, -1),
                    ] {
                        if let Some(there) = map.get(&(x + d.0, y + d.1, z + d.2)) {
                            let gap = (here[0].level as i32 - there[0].level as i32).abs();
                            assert!(
                                gap <= 1,
                                "squash {squash}: ({x},{y},{z}) level {} beside {}",
                                here[0].level,
                                there[0].level
                            );
                        }
                    }
                }
            }
        }
    }

    /// **R3a, pinned directly** (issue #327 §3 R3a): splitting by Euclidean
    /// distance instead of Chebyshev leaves an on-axis node's level exactly
    /// where it was, and coarsens a diagonal one that used to split under
    /// the old square-ring rule.
    ///
    /// Schedule `[(0, 4), (1, 8)]`: level 1 (extent 2) splits a child into
    /// level 0 when the child's nearest chunk is within 4. Band mode (no
    /// squash) isolates the horizontal effect R3a changes -- `squash = 0`
    /// zeroes the vertical component of `node_distance_components`, so the
    /// split test below is exactly the horizontal one.
    ///
    /// - On-axis: the level-1 node at chunk origin (4, 0, 0) is exactly 4
    ///   chunks out along x alone. Chebyshev and Euclidean agree at a pure
    ///   axis offset (`dz = 0`), so this still splits to level 0, same as
    ///   today -- R3a's "the worst-case cell size is unchanged" claim.
    /// - Diagonal: the level-1 node at chunk origin (4, 0, 4) is 4 chunks out
    ///   on *both* x and z. Chebyshev distance is `max(4, 4) = 4`, so today
    ///   this also splits to level 0. Euclidean distance is
    ///   `sqrt(4^2 + 4^2) ~= 5.66`, outside radius 4, so under R3a it is left
    ///   one level coarser (level 1) -- exactly the ~21% of the inner ring's
    ///   diagonal area the proposal says this reclaims.
    #[test]
    fn circular_splitting_keeps_the_axis_but_coarsens_the_diagonal() {
        let schedule = [(0u32, 4i32), (1, 8)];
        let center = ChunkCoord::new(0, 0, 0);
        let map = coverage(&desired_nodes(center, 0..=0, &schedule));
        assert_eq!(
            map[&(4, 0, 0)][0].level,
            0,
            "on-axis at the inner radius: unchanged, still finest"
        );
        assert_eq!(
            map[&(4, 0, 4)][0].level,
            1,
            "on the diagonal at the same Chebyshev distance: now left coarser"
        );
    }

    /// **R3a keeps the cube reach** (issue #327 §3 R3a): the *outer* radius
    /// that bounds the whole selection -- and therefore `covered_box_3d` and
    /// the far terrain's hole -- is unaffected by the split test becoming
    /// circular. Computed independently of node selection (the same formula
    /// `covered_box_3d` uses), so a reach that quietly became circular too
    /// would show up as a mismatch against `desired_nodes_3d`'s actual
    /// coverage, exactly as `covered_box_3d_is_exactly_what_the_nodes_cover`
    /// (above) already checks generally -- this pins the same property with
    /// a schedule small enough to read by hand.
    #[test]
    fn the_outer_reach_is_still_the_chebyshev_cube() {
        let schedule = [(0u32, 4i32), (1, 8)];
        let center = ChunkCoord::new(0, 0, 0);
        let nodes = desired_nodes_3d(center, 2, &schedule);
        let map = coverage(&nodes);
        let (min, max) = covered_box_3d(center, &schedule);
        // The formula `covered_box_3d` uses: every level-1 node (extent 2)
        // whose chunk-origin corner is within Chebyshev 8 is kept whole, so
        // the box rounds the radius-8 cube out to the next extent-2
        // boundary past it (`max` is exclusive): [-8, 10) on every axis.
        assert_eq!(min, [-8, -8, -8]);
        assert_eq!(max, [10, 10, 10]);
        // A corner at exactly the old Chebyshev reach (8, 0, 8) -- which a
        // *circular* reach of radius 8 would have excluded, since
        // `sqrt(8^2 + 8^2) ~= 11.3 > 8` -- is still covered by exactly one
        // node: the reach test did not change shape even though the split
        // test (above) did.
        assert_eq!(map.get(&(8, 0, 8)).map_or(0, Vec::len), 1);
    }

    #[test]
    fn rings_do_not_double_count_the_same_node() {
        let nodes = desired_nodes(ChunkCoord::new(0, 0, 0), 0..=0, DEFAULT_RING_SCHEDULE);
        let unique: HashSet<NodeKey> = nodes.iter().copied().collect();
        assert_eq!(nodes.len(), unique.len(), "no node emitted by two rings");
    }

    #[test]
    fn every_ring_beyond_the_first_contains_at_least_one_node() {
        // Sanity: a schedule with strictly increasing radii must actually
        // grow the resident set at each level, not silently contribute
        // nothing (which would mean the exclusion math ate the whole ring).
        for level in 1..DEFAULT_RING_SCHEDULE.len() {
            let schedule = &DEFAULT_RING_SCHEDULE[..=level];
            let nodes = desired_nodes(ChunkCoord::new(0, 0, 0), 0..=0, schedule);
            let at_this_level = nodes.iter().filter(|n| n.level as usize == level).count();
            assert!(at_this_level > 0, "level {level} contributed no nodes");
        }
    }

    #[test]
    fn coarser_rings_cover_further_out_than_finer_ones() {
        let nodes = desired_nodes(ChunkCoord::new(0, 0, 0), 0..=0, DEFAULT_RING_SCHEDULE);
        let center = ChunkCoord::new(0, 0, 0);
        // Every level-0 node's chunk origin is within the first ring's own
        // outer radius; every level-3 node reaches meaningfully further out
        // than every level-0 node does.
        let max_chunk_dist = |level: u32| -> i32 {
            nodes
                .iter()
                .filter(|n| n.level == level)
                .map(|n| {
                    let o = n.chunk_origin();
                    (o.x - center.x).abs().max((o.z - center.z).abs())
                })
                .max()
                .unwrap()
        };
        assert!(max_chunk_dist(3) > max_chunk_dist(0));
    }

    #[test]
    fn staying_put_is_a_no_op() {
        let nodes = desired_nodes(ChunkCoord::new(5, 0, 5), 0..=1, DEFAULT_RING_SCHEDULE);
        let resident: HashSet<NodeKey> = nodes.into_iter().collect();
        let updates = plan_node_updates(
            &resident,
            ChunkCoord::new(5, 0, 5),
            0..=1,
            DEFAULT_RING_SCHEDULE,
        );
        assert_eq!(updates, NodeStreamUpdates::default());
    }

    #[test]
    fn from_empty_everything_loads_nothing_unloads() {
        let schedule = [(0u32, 1i32)];
        let updates =
            plan_node_updates(&HashSet::new(), ChunkCoord::new(0, 0, 0), 0..=0, &schedule);
        assert_eq!(updates.to_load.len(), 3 * 3);
        assert!(updates.to_unload.is_empty());
        // Nearest-first: the center node loads before its neighbours.
        assert_eq!(updates.to_load[0], NodeKey::new(0, [0, 0, 0]));
    }

    #[test]
    fn moving_unloads_nodes_no_longer_desired() {
        let schedule = [(0u32, 1i32)];
        let resident: HashSet<NodeKey> = desired_nodes(ChunkCoord::new(0, 0, 0), 0..=0, &schedule)
            .into_iter()
            .collect();
        let updates = plan_node_updates(&resident, ChunkCoord::new(5, 0, 5), 0..=0, &schedule);
        assert!(!updates.to_unload.is_empty());
        assert!(updates.to_unload.iter().all(|n| resident.contains(n)));
    }

    #[test]
    fn schedule_for_radius_collapses_to_level_0_under_the_first_ring() {
        // Every current golden test uses a radius well under the first ring's
        // own outer radius (10) -- this must stay a single level-0 entry, so
        // meshing that region as nodes is pixel-identical to meshing it as
        // plain chunks.
        assert_eq!(schedule_for_radius(6), vec![(0, 6)]);
        assert_eq!(schedule_for_radius(0), vec![(0, 0)]);
    }

    #[test]
    fn schedule_for_radius_reproduces_the_default_schedule_at_its_own_outer_radius() {
        let last = DEFAULT_RING_SCHEDULE.last().unwrap();
        assert_eq!(schedule_for_radius(last.1), DEFAULT_RING_SCHEDULE.to_vec());
    }

    #[test]
    fn schedule_for_radius_clamps_the_ring_it_lands_in() {
        // A radius strictly inside a later ring keeps every earlier ring
        // untouched and clamps only the one it falls in.
        let schedule = schedule_for_radius(15);
        assert_eq!(schedule, vec![(0, 10), (1, 15)]);
    }
}

//! The render anchor (issue #379): the integer block every position the GPU
//! sees is measured from, so a frame drawn a million blocks from the world
//! origin is the same frame as one drawn at it.
//!
//! The simulation is exact -- positions are `FixedVec3`, an `i64` in
//! 1/65536 of a block. The GPU is not: an `f32` at 50,000 blocks steps in
//! 1/256 of a block, and a block one block from the eye jitters by several
//! pixels. So nothing absolute reaches the GPU. Node origins
//! (`crate::arena`), far tile origins and the voxels' hole (`crate::far`),
//! the outline, the cracks, the figures (`crate::scene`) and the eye in the
//! frame uniform are all **relative to the anchor**, and `view_proj` is
//! built from `eye - anchor`. Node and tile origins are integers, so their
//! subtraction is exact; the eye is `f64` (`crate::CameraPose`), and
//! `eye - anchor` is under [`ANCHOR_GRID`] / 2 + [`REBASE_HYSTERESIS`], so
//! its `f32` error is at most 1/4,096 of a block.
//!
//! The anchor is `[i32; 3]`, not `i64`: the world's block coordinates are
//! `i32` everywhere (`ChunkCoord`, `NodeKey::world_origin`, an edit's
//! block), so an eye the anchor cannot reach cannot stand in the world
//! either; [`anchor_for`] saturates rather than wraps, so even an absurd
//! `f64` eye yields a valid anchor.
//!
//! **Rebasing** (decision 2, amended 2026-11-10 after review finding F16):
//! the anchor snaps to a [`ANCHOR_GRID`]-block grid and moves only when the
//! eye is more than [`REBASE_DISTANCE`] from it on some axis -- the 256
//! blocks of hysteresis stop it thrashing when the eye hovers on a grid
//! boundary (the nearest grid point is never more than 2,048 away, so after
//! a rebase the eye has 256 blocks to go before the next). A rebase must not
//! be a hitch: it rewrites one `vec4` per resident node, the far tiles'
//! 16-byte records in one bounded write, and the frame uniform -- never a
//! tree instance, which is node-relative (`crate::trees::TreeInstance`).
//! Between rebases the cost per frame is zero. Near the origin the anchor
//! is `(0, 0, 0)` and every subtraction is the identity, which is why the
//! goldens did not change (decision 6).

use glam::{DVec3, Vec3};

use crate::arena::ChunkArena;
use crate::culling::Aabb;
use crate::far::FarTerrain;
use crate::trees::TreeArena;

/// The anchor grid, in blocks: the anchor is always a multiple of this on
/// every axis.
pub const ANCHOR_GRID: i32 = 4096;

/// How far past the half-grid the eye may stray before the anchor moves.
pub const REBASE_HYSTERESIS: f64 = 256.0;

/// The eye-to-anchor distance, on any one axis, that triggers a rebase.
pub const REBASE_DISTANCE: f64 = ANCHOR_GRID as f64 / 2.0 + REBASE_HYSTERESIS;

/// An integer block position on the [`ANCHOR_GRID`].
pub type Anchor = [i32; 3];

/// The largest anchor coordinate: the last grid line inside `i32`, so a
/// saturated anchor is still a grid multiple (and so still exact in `f32`,
/// [`anchor_f32`]).
pub const MAX_ANCHOR: i32 = (i32::MAX / ANCHOR_GRID) * ANCHOR_GRID;

/// The origin of the world as an anchor: what every renderer starts with
/// near spawn, and the value under which every subtraction here is the
/// identity.
pub const ORIGIN: Anchor = [0; 3];

/// The grid point nearest `eye` -- the anchor a renderer adopts when it
/// rebases (or starts). Saturates to [`MAX_ANCHOR`] (a grid line, never
/// `i32::MAX` itself) rather than wrapping; a NaN becomes the origin.
pub fn anchor_for(eye: DVec3) -> Anchor {
    let grid = ANCHOR_GRID as f64;
    let lines = (MAX_ANCHOR / ANCHOR_GRID) as f64;
    let snap = |v: f64| {
        // Round to the nearest grid line in f64 (exact: grid lines are
        // integers well inside f64's range), clamp to the last line that
        // fits, then convert -- `as` saturates for float-to-int and maps a
        // NaN to 0, but by then the value is an in-range integer anyway.
        ((v / grid).round().clamp(-lines, lines) * grid) as i32
    };
    [snap(eye.x), snap(eye.y), snap(eye.z)]
}

/// Whether `eye` has strayed more than [`REBASE_DISTANCE`] from `anchor` on
/// any axis. A boundary wobble never trips this: the eye is at most 2,048
/// from the grid point it last snapped to, and needs 256 more.
pub fn needs_rebase(anchor: Anchor, eye: DVec3) -> bool {
    let d = eye - DVec3::new(anchor[0] as f64, anchor[1] as f64, anchor[2] as f64);
    d.abs().max_element() > REBASE_DISTANCE
}

/// The anchor to move to for `eye`, or `None` to stay: the pure rebase
/// rule. A jump (`/tp`, a new world) rebases on its first frame; a walk
/// rebases once per [`ANCHOR_GRID`] blocks of travel, never on a wobble.
pub fn next_anchor(anchor: Anchor, eye: DVec3) -> Option<Anchor> {
    needs_rebase(anchor, eye).then(|| anchor_for(eye))
}

/// `eye - anchor`, as the `f32` the GPU gets. Small by construction (see
/// [`needs_rebase`]: under 2,304 on every axis), so the conversion rounds by
/// at most half an `f32` step at that magnitude, 1/8,192 of a block.
pub fn relative_eye(anchor: Anchor, eye: DVec3) -> Vec3 {
    (eye - DVec3::new(anchor[0] as f64, anchor[1] as f64, anchor[2] as f64)).as_vec3()
}

/// `block - anchor` for an integer block position, exact: **the one
/// subtraction** every integer origin goes through -- a node's
/// (`crate::arena`), a far tile's (`crate::far`, whose origins are `i64`,
/// hence the parameter type), the outline's, a crack's. In `i64` so two
/// coordinates at opposite ends of the world cannot overflow on the way to
/// a difference that is small; the result is an `f32` the GPU holds exactly
/// whenever the difference is below 2^24 blocks.
pub fn offset_i64(block: [i64; 3], anchor: Anchor) -> Vec3 {
    let d = |a: i64, b: i32| (a - b as i64) as f32;
    Vec3::new(
        d(block[0], anchor[0]),
        d(block[1], anchor[1]),
        d(block[2], anchor[2]),
    )
}

/// [`offset_i64`] for an `i32` block, which is what every block coordinate
/// in the world is (`ChunkCoord`, `NodeKey::world_origin`, an edit).
pub fn offset(block: [i32; 3], anchor: Anchor) -> Vec3 {
    offset_i64(block.map(i64::from), anchor)
}

/// [`offset`] kept as integers, for a block the CPU still has to build
/// geometry around (the cracks) before it becomes floats. The difference
/// is small (the block is one the player can reach), so the narrowing back
/// to `i32` is exact.
pub fn block_offset(block: [i32; 3], anchor: Anchor) -> [i32; 3] {
    let d = |a: i32, b: i32| (a as i64 - b as i64) as i32;
    [
        d(block[0], anchor[0]),
        d(block[1], anchor[1]),
        d(block[2], anchor[2]),
    ]
}

/// The anchor as the `vec4` the frame uniform carries (`.w` unused). Exact:
/// a multiple of 4,096 is representable in `f32` to well past `i32`.
pub fn anchor_f32(anchor: Anchor) -> [f32; 4] {
    [anchor[0] as f32, anchor[1] as f32, anchor[2] as f32, 0.0]
}

/// Where a node stands, kept so its culling box can be re-expressed against
/// any anchor without loss: the node's integer origin, and its bounds
/// relative to that origin.
///
/// This is how both arenas (`ChunkArena`, `TreeArena`) store a box, so the
/// CPU frustum test runs in the same anchor-relative space as `view_proj`
/// (decision 5) through one piece of arithmetic. The local box is what the
/// mesher produces (`cubara_voxel::build_mesh_bounded`,
/// `cubara_world::mesh::NodeGeometry::local_aabb`): no absolute `f32`
/// position is ever formed and subtracted again, so the box is as exact
/// 2^30 blocks out as at the origin.
#[derive(Clone, Copy, Debug)]
pub struct NodeBox {
    origin: [i32; 3],
    local: Aabb,
}

impl NodeBox {
    /// From a node's integer origin and its bounds **relative to that
    /// origin**.
    pub fn new(origin: [i32; 3], local: Aabb) -> Self {
        Self { origin, local }
    }

    /// The node's integer origin, in world blocks.
    pub fn origin(&self) -> [i32; 3] {
        self.origin
    }

    /// The bounds relative to the node's own origin, as given.
    pub fn local(&self) -> Aabb {
        self.local
    }

    /// The bounds relative to `anchor`: the integer `origin - anchor`
    /// (exact) added to the local box.
    pub fn relative_to(&self, anchor: Anchor) -> Aabb {
        self.local.translated(offset(self.origin, anchor))
    }

    /// The bounds in world space, for framing a camera around everything
    /// resident (`ChunkArena::bounds`).
    pub fn absolute(&self) -> Aabb {
        self.relative_to(ORIGIN)
    }
}

/// Move `anchor` to where `eye` wants it, if it has strayed far enough
/// ([`next_anchor`]), and rewrite everything that holds anchor-relative
/// positions: the arena's node origins and culling boxes, the trees' culling
/// boxes (their instances are node-relative and untouched), and the far
/// tiles' records, boxes and hole. Returns the new anchor when it moved.
///
/// The one implementation of "what a rebase touches" (`ARCHITECTURE.md`
/// Rule 5): the window's [`crate::Renderer`] and `--bench`'s hand-rolled
/// frame both call this, so a buffer that gains an absolute position later
/// is added here and nowhere else.
pub fn rebase_if_needed(
    anchor: &mut Anchor,
    eye: DVec3,
    queue: &wgpu::Queue,
    arena: &mut ChunkArena,
    trees: &mut TreeArena,
    far: Option<&mut FarTerrain>,
) -> Option<Anchor> {
    let next = next_anchor(*anchor, eye)?;
    *anchor = next;
    arena.rebase(queue, next);
    trees.rebase(next);
    if let Some(far) = far {
        far.rebase(queue, next);
    }
    Some(next)
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: f64 = ANCHOR_GRID as f64;

    #[test]
    fn the_anchor_is_the_nearest_grid_point_on_every_axis() {
        assert_eq!(anchor_for(DVec3::new(8.0, 120.0, 8.0)), ORIGIN);
        assert_eq!(anchor_for(DVec3::new(2047.9, -2047.9, 0.0)), ORIGIN);
        assert_eq!(
            anchor_for(DVec3::new(2049.0, -2049.0, 3.0 * G + 1.0)),
            [ANCHOR_GRID, -ANCHOR_GRID, 3 * ANCHOR_GRID]
        );
        // The issue's test offset: 256 grid steps.
        let far = DVec3::new(1_048_576.0 + 8.0, 120.0, 1_048_576.0 + 8.0);
        assert_eq!(anchor_for(far), [1_048_576, 0, 1_048_576]);
        // The world's edge: saturates to the last grid line, never wraps,
        // and stays a grid multiple (`i32::MAX` is not one).
        let edge = anchor_for(DVec3::splat(1e12));
        assert_eq!(edge, [MAX_ANCHOR; 3]);
        assert_eq!(edge[0] % ANCHOR_GRID, 0, "{edge:?} is not on the grid");
        assert!(edge[0] > i32::MAX - ANCHOR_GRID, "{edge:?} stopped short");
        assert_eq!(anchor_for(DVec3::splat(-1e12)), [-MAX_ANCHOR; 3]);
        assert_eq!(anchor_for(DVec3::splat(f64::NAN)), ORIGIN);
    }

    #[test]
    fn a_boundary_wobble_never_rebases() {
        // Standing exactly on a grid boundary, half a grid from the anchor,
        // and wobbling either way by less than the hysteresis.
        let anchor = ORIGIN;
        for x in [2047.0, 2048.0, 2049.0, 2300.0, 2304.0] {
            let eye = DVec3::new(x, 60.0, -x);
            assert_eq!(
                next_anchor(anchor, eye),
                None,
                "a wobble to {x} must not rebase ({REBASE_DISTANCE} is the line)"
            );
        }
        // And once rebased to the next grid point, wobbling back does not
        // rebase again either: the eye is now within 2,048 of the new one.
        let moved = [ANCHOR_GRID, 0, 0];
        for x in [2049.0, 2304.0, 2305.0, 4096.0, 6000.0] {
            assert_eq!(next_anchor(moved, DVec3::new(x, 60.0, 0.0)), None);
        }
    }

    #[test]
    fn walking_past_the_hysteresis_rebases_to_the_next_grid_point() {
        let eye = DVec3::new(REBASE_DISTANCE + 0.5, 60.0, 10.0);
        assert_eq!(next_anchor(ORIGIN, eye), Some([ANCHOR_GRID, 0, 0]));
        // Any one axis is enough, the sign does not matter.
        let eye = DVec3::new(10.0, 60.0, -(REBASE_DISTANCE + 0.5));
        assert_eq!(next_anchor(ORIGIN, eye), Some([0, 0, -ANCHOR_GRID]));
        // Right on the line is not past it.
        let eye = DVec3::new(REBASE_DISTANCE, 60.0, 10.0);
        assert_eq!(next_anchor(ORIGIN, eye), None);
    }

    #[test]
    fn a_teleport_rebases_at_once_however_far() {
        let eye = DVec3::new(1_048_576.0 + 8.0, 120.0, -3.0 * G + 100.0);
        assert_eq!(
            next_anchor(ORIGIN, eye),
            Some([1_048_576, 0, -3 * ANCHOR_GRID])
        );
        // After it, the eye is inside the half-grid of the new anchor.
        let after = next_anchor(ORIGIN, eye).unwrap();
        assert!(!needs_rebase(after, eye));
        let rel = relative_eye(after, eye);
        assert!(rel.abs().max_element() <= G as f32 / 2.0, "{rel}");
    }

    #[test]
    fn the_relative_eye_is_exact_where_the_absolute_one_is_not() {
        // 2^20 + 8.1: the f32 step there is 1/8 of a block, so the absolute
        // f32 eye rounds to 8.125; relative to the anchor it is 8.1 to f32's
        // own precision.
        let eye = DVec3::new(1_048_576.0 + 8.1, 120.0, 1_048_576.0 + 8.1);
        let anchor = anchor_for(eye);
        let rel = relative_eye(anchor, eye);
        assert_eq!(rel, Vec3::new(8.1, 120.0, 8.1));
        assert_ne!((eye.x as f32) - 1_048_576.0, 8.1f32);
    }

    #[test]
    fn integer_offsets_are_exact_and_do_not_overflow() {
        let anchor = [1_048_576, 0, -1_048_576];
        assert_eq!(
            offset([1_048_576 + 16, -32, -1_048_576 + 128], anchor),
            Vec3::new(16.0, -32.0, 128.0)
        );
        assert_eq!(
            block_offset([1_048_576 + 16, -32, -1_048_576 + 128], anchor),
            [16, -32, 128]
        );
        // Opposite ends of i32 would overflow an i32 subtraction; the
        // difference is computed in i64 and only then narrowed.
        let far = offset([i32::MAX, 0, 0], [i32::MIN + 1, 0, 0]);
        assert!(far.x > 4.0e9, "{far}");
        // The far terrain's `i64` origins go through the same subtraction.
        assert_eq!(
            offset_i64([1_048_576 + 64, 0, -1_048_576 - 128], anchor),
            Vec3::new(64.0, 0.0, -128.0)
        );
    }

    #[test]
    fn a_node_box_re_expressed_against_the_shifted_anchor_is_the_same_box() {
        // The mesher's box is local to the node, so the same box serves the
        // node at the origin and the node 2^20 -- or 2^30 -- blocks out.
        let local = Aabb::new(Vec3::new(0.0, 1.0, 0.0), Vec3::new(16.0, 9.0, 16.0));
        let at_origin = NodeBox::new([16, 32, -64], local);
        let shift = 1_048_576;
        let shifted = NodeBox::new([16 + shift, 32, -64 + shift], local);
        let a = at_origin.relative_to(ORIGIN);
        let b = shifted.relative_to([shift, 0, shift]);
        assert_eq!((a.min, a.max), (b.min, b.max));
        assert_eq!(a.min, Vec3::new(16.0, 33.0, -64.0));
        let abs = shifted.absolute();
        assert_eq!(
            abs.min,
            Vec3::new((16 + shift) as f32, 33.0, (-64 + shift) as f32)
        );
        // Beyond 2^24, where an absolute f32 box would have been rounded
        // to even blocks before it could be subtracted from.
        let far = 1 << 30;
        let c = NodeBox::new([16 + far, 32, -64 + far], local).relative_to([far, 0, far]);
        assert_eq!((c.min, c.max), (a.min, a.max));
    }

    #[test]
    fn the_anchor_vec4_is_exact() {
        let a = anchor_f32([1_048_576, -ANCHOR_GRID, MAX_ANCHOR]);
        assert_eq!(a, [1_048_576.0, -4096.0, 2_147_479_552.0, 0.0]);
        assert_eq!(a[2] as i64, MAX_ANCHOR as i64);
    }
}

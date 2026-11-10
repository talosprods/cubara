//! The geometric shape of a tree: trunk column plus a rounded canopy blob,
//! relative to the trunk's base block. One answer to "what shape is a tree"
//! (`ARCHITECTURE.md` Rule 5), shared by [`cubara_world::worldgen`]'s real
//! per-voxel placement (`tree_block_at`) and `cubara-render`'s static
//! instance model (issue #332, R2) -- the model has to be built from
//! *exactly* this shape, or a tree would look different just inside and just
//! outside the first LOD ring, which is the one thing R2's decision 3
//! forbids.
//!
//! Pure geometry, relative to the trunk's base block at `(0, 0, 0)`: no
//! noise, no seed, no world position. Placement (which columns grow a tree,
//! how tall) is a different question and stays in `cubara_world::worldgen`,
//! which alone knows the seed.

/// Which part of a tree one relative cell belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreePart {
    Trunk,
    Canopy,
}

/// Whether the cell `dy` blocks above the trunk's base, and `dx`/`dz` blocks
/// away from the trunk column (both already absolute -- the shape is
/// symmetric on x and z), is trunk, canopy, or nothing, for a trunk
/// `height` blocks tall with a canopy reaching `canopy_radius` blocks from
/// it.
///
/// Extracted unchanged from the single-tree body of
/// `cubara_world::worldgen::WorldGen::tree_block_at` (issue #332): that
/// function still decides which of several overlapping trees' blocks wins
/// (trunk over canopy, across trees) -- the part that needs the trees'
/// absolute positions -- while this is the shape of *one* tree alone, which
/// needs none of that and so can live below both `cubara-world` and
/// `cubara-render` in the dependency graph.
pub fn tree_part_at(
    dx: i32,
    dz: i32,
    dy: i32,
    height: i32,
    canopy_radius: i32,
) -> Option<TreePart> {
    let top = height - 1;
    if dx == 0 && dz == 0 && dy >= 0 && dy <= top {
        return Some(TreePart::Trunk);
    }
    let r = canopy_radius;
    if dx <= r && dz <= r && dy >= top - r && dy <= top + 1 {
        // Trim the corners so the canopy reads as round rather than a cube,
        // and drop the ring at the very top so it comes to a point.
        let shrink = if dy >= top { 1 } else { 0 };
        if dx + dz <= r + 1 - shrink {
            return Some(TreePart::Canopy);
        }
    }
    None
}

/// The local bounding box a tree of `height`/`canopy_radius` fits in,
/// relative to the trunk's base block: `(min, max)` inclusive, in blocks.
/// What a caller building a static model (the renderer's oak mesh, or its
/// level-3 boxes) sizes its volume from, so the bound and the shape this
/// module draws can never disagree about how far the canopy reaches.
pub fn tree_bounds(height: i32, canopy_radius: i32) -> ([i32; 3], [i32; 3]) {
    let top = height - 1;
    let r = canopy_radius;
    // The canopy's topmost row is `top + 1` (the point above the trunk).
    ([-r, 0, -r], [r, top + 1, r])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_trunk_is_a_single_column_up_to_its_height() {
        for dy in 0..4 {
            assert_eq!(tree_part_at(0, 0, dy, 4, 2), Some(TreePart::Trunk));
        }
        assert_eq!(tree_part_at(0, 0, -1, 4, 2), None, "below the base");
        // One above the trunk's own top is the canopy's own apex cell (see
        // `the_canopy_narrows_right_at_and_above_the_trunks_top` below), not
        // nothing -- the canopy's reach ends one row further up than that.
        assert_eq!(
            tree_part_at(0, 0, 5, 4, 2),
            None,
            "two above the top -- past the canopy's own reach too"
        );
    }

    #[test]
    fn the_trunk_range_is_never_canopy() {
        // Within the trunk's own height range, directly over the trunk is
        // always `Trunk`, never `Canopy` -- the one place the two shapes
        // could otherwise disagree about who owns a cell. (One row *above*
        // the range is genuinely the canopy's own apex -- see
        // `the_trunk_is_a_single_column_up_to_its_height`.)
        let (height, r) = (5, 2);
        for dy in 0..height {
            assert_eq!(tree_part_at(0, 0, dy, height, r), Some(TreePart::Trunk));
        }
    }

    /// The canopy is a diamond (`dx + dz <= some radius`, clipped to
    /// `dx <= r`/`dz <= r`) that narrows by one at and above the trunk's own
    /// top -- "drop the ring at the very top so it comes to a point"
    /// (`tree_part_at`'s own doc comment). `(2, 1)` sits exactly on the
    /// wider diamond's edge, which is what makes it the cell that tells the
    /// two tiers apart; `(2, 2)` (both axes at the full radius) is cut off
    /// at every row, narrower or not.
    #[test]
    fn the_canopy_narrows_right_at_and_above_the_trunks_top() {
        let (height, r) = (4, 2); // top = 3
        assert_eq!(tree_part_at(2, 1, 1, height, r), Some(TreePart::Canopy));
        assert_eq!(tree_part_at(2, 1, 2, height, r), Some(TreePart::Canopy));
        assert_eq!(
            tree_part_at(2, 1, 3, height, r),
            None,
            "the top row narrows: (2, 1) no longer fits"
        );
        assert_eq!(
            tree_part_at(2, 1, 4, height, r),
            None,
            "one above the top stays at the narrower radius"
        );
        assert_eq!(
            tree_part_at(2, 2, 1, height, r),
            None,
            "the far corner is trimmed everywhere"
        );
        assert_eq!(tree_part_at(2, 2, 3, height, r), None);
        // The canopy's own apex, directly above the trunk -- this is the
        // "point" the doc comment means.
        assert_eq!(tree_part_at(0, 0, 4, height, r), Some(TreePart::Canopy));
        // And nothing at all once even the apex's row is behind us.
        assert_eq!(
            tree_part_at(0, 0, 5, height, r),
            None,
            "above the whole tree"
        );
    }

    #[test]
    fn nothing_is_both_trunk_and_canopy() {
        for dx in 0..=3 {
            for dz in 0..=3 {
                for dy in -2..8 {
                    // The function can only ever return one variant; this is
                    // really just exercising the whole grid for a panic/
                    // logic-contradiction, since the type system already
                    // forbids returning both.
                    let _ = tree_part_at(dx, dz, dy, 5, 2);
                }
            }
        }
    }
}

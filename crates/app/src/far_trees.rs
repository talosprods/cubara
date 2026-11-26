//! The trees on the far terrain between the world and the renderer (issue
//! #433 item 1): which patches carry their trees one by one, and the
//! renderer's form of them. `cubara_world::far_trees` places them and
//! `cubara_render::far_trees` draws them; this crate is the one that knows
//! both (`ARCHITECTURE.md` Rule 3).

use cubara_render::far_trees::{px_per_block, reach, FarTreePlacement};
use cubara_world::far::{FarView, PatchHeights, PatchKey, WithTrees, FINEST_LEVEL};
use cubara_world::TerrainBlocks;

/// The renderer's form of a patch's trees: the same trees, field for field.
pub fn to_far_trees(p: &PatchHeights) -> Vec<FarTreePlacement> {
    p.trees
        .iter()
        .map(|t| FarTreePlacement {
            base: [t.x, t.base, t.z],
            height: t.height,
        })
        .collect()
}

/// The tallest tree the world grows, in blocks from its foot to the top of
/// its canopy -- the stand-in's height (`cubara_render::far_trees::StandIn::
/// tall`, from the same `tree_bounds`) -- or `None` without trees.
fn tallest(blocks: TerrainBlocks) -> Option<f32> {
    let oak = blocks.oak?;
    let (_, max) = cubara_voxel::tree_bounds(oak.height.1, oak.canopy_radius);
    Some(max[1] as f32 + 1.0)
}

/// **The coarsest patch level that can hold a tree drawn one by one**, for
/// `view` on a screen `screen_height` pixels tall: a patch is kept whole
/// only while the eye is at least `split_ratio` times its width away
/// (`far::select`), and a tree is drawn out to its [`reach`]; a level whose
/// patches are never that near draws none of its trees, and carries none.
///
/// The same for every screen height: a taller screen splits patches as much
/// further out as it draws trees (`the_tree_level_is_every_screens`).
pub fn tree_level(view: &FarView, screen_height: u32, tallest: f32) -> u32 {
    // A tree stands up to its height above its patch's box (which reaches
    // the terrain's highest top), and a little further is no harm.
    let reach = reach(tallest, px_per_block(screen_height)) + 2.0 * tallest;
    let mut level = FINEST_LEVEL;
    while view.split_ratio * (PatchKey::new(level + 1, 0, 0).size() as f64) < reach as f64 {
        level += 1;
    }
    level
}

/// What a far patch is generated with to draw `blocks`' trees for `view`
/// ([`tree_level`]); `None` for a world without trees.
pub fn with_trees(blocks: TerrainBlocks, view: &FarView, screen_height: u32) -> Option<WithTrees> {
    let tallest = tallest(blocks)?;
    Some(WithTrees {
        blocks,
        up_to_level: tree_level(view, screen_height, tallest),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::far_streaming::{far_view, FarQuality};

    fn blocks() -> TerrainBlocks {
        let registry = cubara_render::load_registry();
        TerrainBlocks::from_registry(&registry)
            .with_oak(&crate::game::load_structure_registry(), &registry)
    }

    /// At the game's quad scale a tree is drawn out to about 3.3 km at
    /// 1080 rows, and patches of quads of 128 (level 7) are kept from about
    /// 3.1 km: they carry trees, and the next level (from 6.1 km) none.
    /// Low splits less, so a level more carries them there.
    #[test]
    fn the_tree_level_is_where_patches_come_within_reach() {
        let blocks = blocks();
        let view = |quality, height| far_view([0.0; 3], height, None, quality).expect("drawn");
        let level = |quality| with_trees(blocks, &view(quality, 1080), 1080).map(|t| t.up_to_level);
        assert_eq!(level(FarQuality::Medium), Some(7));
        assert_eq!(level(FarQuality::High), Some(7));
        assert_eq!(level(FarQuality::Low), Some(8));
        let bare = TerrainBlocks { oak: None, ..blocks };
        assert_eq!(with_trees(bare, &view(FarQuality::Medium, 1080), 1080), None);
    }

    #[test]
    fn the_tree_level_is_every_screens() {
        let blocks = blocks();
        for quality in [FarQuality::Low, FarQuality::Medium, FarQuality::High] {
            let at = |h: u32| {
                let view = far_view([0.0; 3], h, None, quality).expect("drawn");
                with_trees(blocks, &view, h).map(|t| t.up_to_level)
            };
            for h in [720, 1080, 1440, 1664, 2160] {
                assert_eq!(at(h), at(1080), "{quality:?} at {h} rows");
            }
        }
    }
}

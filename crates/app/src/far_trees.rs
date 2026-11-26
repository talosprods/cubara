//! The trees on the far terrain between the world and the renderer (issue
//! #433 item 1): which patches carry their trees one by one, and the
//! renderer's form of them. `cubara_world::far_trees` places them and
//! `cubara_render::far_trees` draws them; this crate is the one that knows
//! both (`ARCHITECTURE.md` Rule 3).

use cubara_render::far_trees::{px_per_block, reach, Fade, FarTreePlacement};
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
    let reach = reach(tallest, px_per_block(screen_height), Fade::GAME) + 2.0 * tallest;
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

    /// At the game's quad scale the tallest tree is drawn out to about
    /// 2.6 km at 1080 rows, and patches of quads of 64 (level 6) are kept
    /// from about 1.5 km and of 128 from 3.1 km: level 6 carries trees, 7
    /// none. Low splits less, so a level more carries them there.
    #[test]
    fn the_tree_level_is_where_patches_come_within_reach() {
        let blocks = blocks();
        let view = |quality, height| far_view([0.0; 3], height, None, quality).expect("drawn");
        let level = |quality| with_trees(blocks, &view(quality, 1080), 1080).map(|t| t.up_to_level);
        assert_eq!(level(FarQuality::Medium), Some(6));
        assert_eq!(level(FarQuality::High), Some(6));
        assert_eq!(level(FarQuality::Low), Some(7));
        let bare = TerrainBlocks {
            oak: None,
            ..blocks
        };
        assert_eq!(
            with_trees(bare, &view(FarQuality::Medium, 1080), 1080),
            None
        );
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

    /// **A far patch's trees are drawn while it is, and go with it**: the
    /// window's wiring (`FarStreaming::update` hands the renderer each
    /// patch's trees with the patch, `Renderer::far_remove` drops them with
    /// it). Streamed round the gate's hill eye until settled, the renderer
    /// holds exactly the trees of the patches it draws, and some; a new
    /// world holds none.
    #[test]
    fn a_patchs_trees_come_and_go_with_it() {
        use crate::far_streaming::{voxel_hole, FarStreaming};
        let eye = [8.0, 300.0, 8.0];
        let camera = cubara_render::CameraPose {
            eye: glam::DVec3::from_array(eye.map(f64::from)),
            look_dir: glam::Vec3::new(1.0, -0.25, 0.0),
        };
        let Some((mut renderer, _assets)) = cubara_render::Renderer::offscreen(320, 180, camera)
        else {
            cubara_render::headless::skip_or_require_gpu("a_patchs_trees_come_and_go_with_it");
            return;
        };
        let seed = cubara_world::World::new().seed();
        let mut far = FarStreaming::new(seed, FarQuality::Medium).with_trees(blocks());
        let hole = voxel_hole(eye);
        for _ in 0..4_000 {
            far.update(&mut renderer, eye, hole);
            if far.settled() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(far.settled(), "the far terrain did not settle");
        let held = renderer.far_trees_drawn().1;
        assert_eq!(held, far.trees_of_drawn_patches());
        assert!(held > 10_000, "only {held} far trees held");
        far.reset(seed, &mut renderer);
        assert_eq!(
            renderer.far_trees_drawn().1,
            0,
            "a new world kept the old one's trees"
        );
    }
}

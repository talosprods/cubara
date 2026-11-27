//! Headless single-frame screenshot, for visual verification without a window.
//!
//! A thin wrapper over [`cubara_render::headless::render`] — the same code the
//! golden-image tests use, which in turn goes through the one scene-render path the
//! window uses (`ARCHITECTURE.md` Rule 5). This file deliberately contains no
//! rendering of its own; when it did, it drifted from the game and stopped proving
//! anything.
//!
//! Run with: `cargo run --release -- --screenshot out.png`, or from a place:
//! `--screenshot out.png --eye X,Y,Z --look DX,DY,DZ [--radius N] [--size WxH]
//! [--far-quad-scale F]`, or from where a player standing on block column
//! `X,Z` has their eye ([`standing_eye`]), at the game's own render distance
//! unless `--radius` says otherwise: `--screenshot out.png --stand X,Z
//! [--look DX,DY,DZ] [--radius N] [--size WxH]`.

use cubara_render::materials::TextureLayers;
use cubara_render::{headless, load_registry, Lighting, Shot};
use cubara_sim::physics::EYE_HEIGHT;
use cubara_voxel::{ChunkCoord, Fixed};
use cubara_world::mesh::{mesh_nodes, mesh_region};
use cubara_world::node::{desired_nodes_3d, schedule_for_radius};
use cubara_world::{World, WorldGen};

use crate::streaming::{to_meshed_node, VERTICAL_LOD_SQUASH};

/// Where a screenshot looks from. `None` is the default shot: the orbit over
/// the old three-layer slab around the origin, as it always was.
#[derive(Clone, Copy, Debug)]
pub struct View {
    pub eye: [f32; 3],
    pub look: [f32; 3],
    /// Render distance in chunks.
    pub radius: i32,
    pub size: (u32, u32),
    /// The far terrain's quads, error bound and floor this many times as
    /// wide on screen as this PC's quality names them
    /// (`far_streaming::far_view_scaled`): the game's
    /// (`far_streaming::FAR_QUAD_SCALE`) unless `--far-quad-scale` asked
    /// for another, to compare what it would look like (issue #422).
    pub far_quad_scale: f64,
    /// Whether the far terrain draws the world's trees (issue #433 item 1):
    /// yes, unless `--far-trees off` asks for bare far ground.
    pub far_trees: bool,
}

/// The eye of a player standing on block column `(x, z)` (`--stand X,Z`):
/// over the middle of the column, [`EYE_HEIGHT`] above the top they stand
/// on -- the highest block of ground the sky sees there,
/// [`WorldGen::column_top`] (the surface block, or the floor of a pit), so
/// the feet are on that block's top face. Before trees and edits, like the
/// top itself: a column a trunk stands on puts the eye inside the trunk.
///
/// A pure function of the seed, so a stand names the same eye on every
/// machine -- the one the simulation settles a player at there
/// (`a_stand_is_where_the_simulation_stands_a_player`).
pub fn standing_eye(gen: &WorldGen, x: i32, z: i32) -> [f32; 3] {
    let feet = Fixed::from_blocks(gen.column_top(x, z) + 1);
    [x as f32 + 0.5, (feet + EYE_HEIGHT).to_f32(), z as f32 + 0.5]
}

/// `--stand X,Z`: a block column, as two whole numbers.
pub fn parse_column(text: &str) -> Option<(i32, i32)> {
    let (x, z) = text.split_once(',')?;
    Some((x.trim().parse().ok()?, z.trim().parse().ok()?))
}

/// Fog for `shot`'s own `region_radius` -- not `streaming::render_radius_chunks()` (chunks; times 16 for blocks)
/// (the *window*'s fixed schedule), since `--screenshot --radius N` can ask
/// for a region smaller or larger than the game streams: reusing a fixed
/// number here would put fog past this shot's own meshed region exactly the
/// way `FAR_PLANE` put it past the window's, for a different fixed value.
/// A screenshot is meant to show what the game looks like, fog included --
/// without this, `--screenshot` was the one entry point the review found
/// where it never did.
fn shot_with_fog(shot: Shot) -> Shot {
    let radius_blocks = (shot.region_radius * 16) as f32;
    let (fog_start, fog_end) = Lighting::fog_range(radius_blocks);
    Shot {
        lighting: Lighting {
            fog_start,
            fog_end,
            ..Lighting::default()
        },
        ..shot
    }
}

pub fn run(path: &str, view: Option<View>, menu: Option<String>, paths: &crate::paths::SavePaths) {
    let world = World::new();
    let registry = load_registry();
    let layers = TextureLayers::from_registry(&registry);
    let layer_of = |name: &str| layers.layer_of(name);
    let blocks = cubara_world::TerrainBlocks::from_registry(&registry)
        .with_oak(&crate::game::load_structure_registry(), &registry)
        .with_ores(&crate::game::load_ore_registry(), &registry);
    // The far terrain, for a shot from an eye: what the player would see
    // beyond the voxels (`docs/PROPOSAL_FAR_VIEW.md`). The orbit frames the
    // voxel region from outside and gets none, as the bench's orbit does.
    let mut far = Vec::new();
    let mut far_params = None;
    let mut far_trees = false;
    let (shot, built) = match view {
        None => {
            let shot = Shot::default();
            let schedule = schedule_for_radius(shot.region_radius);
            let built = mesh_region(
                &world,
                &registry,
                &layer_of,
                ChunkCoord::new(0, 0, 0),
                0..=2,
                &schedule,
                blocks,
            );
            (shot_with_fog(shot), built)
        }
        // The nodes the game streams around a player standing at `eye`.
        Some(v) => {
            let centre = ChunkCoord::from_world_pos(v.eye);
            let nodes =
                desired_nodes_3d(centre, VERTICAL_LOD_SQUASH, &schedule_for_radius(v.radius));
            let built = mesh_nodes(&world, &registry, &layer_of, nodes, blocks);
            let shot = Shot {
                width: v.size.0,
                height: v.size.1,
                region_radius: v.radius,
                camera: Some((glam::Vec3::from(v.eye).as_dvec3(), glam::Vec3::from(v.look))),
                // The distance morph, as the game draws it (without it a
                // shot showed every node at its own heights and could not
                // show what the morph does near the player, #380).
                morph: Some(crate::streaming::morph_params_for(
                    &schedule_for_radius(v.radius),
                    VERTICAL_LOD_SQUASH,
                )),
                ..Shot::default()
            };
            let (min, max) =
                cubara_world::node::covered_box_3d(centre, &schedule_for_radius(v.radius));
            let hole = cubara_world::far::Hole {
                min: min.map(|c| (c * 16) as f64),
                max: max.map(|c| (c * 16) as f64),
            };
            // This PC's quality, as the game would draw it.
            let quality = crate::settings::load(&paths.settings)
                .unwrap_or_default()
                .far;
            let eye = v.eye.map(|c| c as f64);
            if let Some(view) = crate::far_streaming::far_view_scaled(
                eye,
                v.size.1,
                Some(hole),
                quality,
                v.far_quad_scale,
            ) {
                // With the trees on it (issue #433 item 1), as the game
                // draws them.
                far = cubara_world::far::build_with(
                    &cubara_world::WorldGen::new(world.seed()),
                    &view,
                    crate::far_trees::with_trees(blocks, &view, v.size.1).filter(|_| v.far_trees),
                );
                far_trees = v.far_trees;
            }
            far_params = Some(crate::far_streaming::to_far_params(
                hole,
                crate::far_streaming::far_materials(),
            ));
            // Fog at the far terrain's edge, as in the game.
            let (fog_start, fog_end) =
                Lighting::fog_range(crate::far_streaming::FAR_VIEW_RADIUS as f32);
            let shot = Shot {
                lighting: Lighting {
                    fog_start,
                    fog_end,
                    ..Lighting::default()
                },
                ..shot
            };
            (shot, built)
        }
    };
    let shot = Shot { menu, ..shot };
    let meshed = built.into_iter().filter_map(to_meshed_node);
    let faces: Vec<_> = far.iter().map(crate::far_streaming::to_far_faces).collect();
    let patches: Vec<_> = far
        .iter()
        .zip(&faces)
        .map(|(p, f)| crate::far_streaming::to_far_patch(p, f))
        .collect();
    let trees: Vec<_> = far
        .iter()
        .flat_map(crate::far_trees::to_far_trees)
        .collect();
    let density = if far.is_empty() || !far_trees {
        Vec::new()
    } else {
        cubara_world::far_trees::density(blocks)
    };
    let Some(frame) = headless::render_with_far_trees(
        meshed,
        &patches,
        far_params,
        headless::FarTreesShot::game(&trees, &density),
        shot,
    ) else {
        log::error!("no suitable GPU adapter — cannot render a screenshot");
        return;
    };

    image::save_buffer(
        path,
        &frame.pixels,
        frame.width,
        frame.height,
        image::ExtendedColorType::Rgba8,
    )
    .expect("write png");
    log::info!(
        "screenshot written to {path} ({}x{})",
        frame.width,
        frame.height
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubara_voxel::FixedVec3;

    /// `--stand X,Z` shows what a player standing there sees only if its eye
    /// is the one the simulation gives them, so this asks the simulation: a
    /// real `Server`'s player, dropped onto the column from a block above
    /// the eye, lands with its eye exactly at `standing_eye` -- feet on the
    /// top face, `EYE_HEIGHT` over them, over the column's middle.
    ///
    /// Two columns, each pinning what the other cannot: the spawn, where the
    /// server itself puts a player on a column (so its `x`/`z` are the
    /// server's, not a re-derivation), and a pit, whose top is its floor and
    /// not the surface the cave carved away -- a stand on the surface height
    /// would hang in the air over it. Both are open to the sky, checked
    /// first, so a tree in the way fails as that and not as a wrong eye.
    #[test]
    fn a_stand_is_where_the_simulation_stands_a_player() {
        let mut server = cubara_server::Server::new();
        server.open(std::path::Path::new("cubara-nonexistent-stand-fixture"));
        let terrain = server.terrain.expect("assets are loaded");
        let gen = WorldGen::new(server.world.seed());
        let who = server.local.expect("a local player");
        let spawn = server.world_spawn().expect("a spawn");
        let [sx, _, sz] = spawn.floor_block();
        let open = |x: i32, z: i32, from: i32, to: i32| {
            (from..=to).all(|y| !server.world.is_solid_at(x, y, z, terrain))
        };
        let pit = (0..64 * 64)
            .map(|k| (k % 64 - 32, k / 64 - 32))
            .find(|&(x, z)| {
                let c = gen.column(x, z);
                c.top < c.surface && open(x, z, c.top + 1, c.surface + 4)
            })
            .expect("a pit open to the sky within 32 blocks of the origin");

        let eye = standing_eye(&gen, sx, sz);
        assert_eq!(
            [eye[0], eye[2]],
            [spawn.x.to_f32(), spawn.z.to_f32()],
            "a stand on the spawn column is not where the server stands a player on it"
        );
        let columns = [(sx, sz), pit];
        for (x, z) in columns {
            let top = gen.column_top(x, z);
            assert!(open(x, z, top + 1, top + 5), "({x}, {z}): not open above");
        }
        for (x, z) in columns {
            let eye = standing_eye(&gen, x, z);
            let p = server.sim.player_mut(who);
            p.pos = FixedVec3::from_f32([eye[0], eye[1] + 1.0, eye[2]]);
            p.velocity = FixedVec3::ZERO;
            for _ in 0..120 {
                server.tick_sim(&cubara_sim::InputFrame::default());
            }
            let p = server.sim.get(who).expect("the player");
            assert!(p.on_ground, "({x}, {z}): never landed");
            assert_eq!(
                p.pos.to_f32(),
                eye,
                "({x}, {z}): the simulation stands a player with its eye elsewhere"
            );
        }
    }

    #[test]
    fn a_column_is_two_whole_numbers() {
        assert_eq!(parse_column("5376,3408"), Some((5376, 3408)));
        assert_eq!(parse_column(" -120 , 48 "), Some((-120, 48)));
        for bad in ["", "12", "1,2,3", "1.5,2", "x,2", "1,"] {
            assert_eq!(parse_column(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn shot_with_fog_stays_inside_its_own_region_regardless_of_radius() {
        // The bug class this exists to avoid repeating: a fixed fog range
        // reused across shots of different sizes lands outside a smaller
        // one's meshed geometry, same as `FAR_PLANE` did for the window.
        for region_radius in [3, 6, 32, 64] {
            let shot = shot_with_fog(Shot {
                region_radius,
                ..Shot::default()
            });
            let radius_blocks = (region_radius * 16) as f32;
            assert!(
                shot.lighting.fog_end < radius_blocks,
                "region_radius {region_radius}: fog_end must land inside this shot's own region"
            );
        }
    }
}

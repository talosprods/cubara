//! The tick must not pay for a mesh job's snapshot (issue #387).
//!
//! `Server::tick_sim_all` calls `Arc::make_mut(&mut self.world)` every tick.
//! A mesh job used to hold a clone of that `Arc<World>`, so the tick found
//! its `Arc` shared and copied the whole world -- every edit ever made --
//! sixty times a second while meshing was in flight. Jobs hold a
//! `WorldSnapshot` now, which shares the edit overlay by reference, and this
//! test is what keeps the tick from ever paying for it again.

use std::sync::Arc;

use cubara_server::Server;
use cubara_sim::{InputFrame, PlayerId, PlayerInputs};
use cubara_world::{RegionEdits, RegionKey};

/// Edits spread over ten regions (512 blocks apart on x), deep underground,
/// where the player standing at the spawn never touches them.
fn edit_ten_regions(server: &mut Server) {
    let stone = server.terrain().stone;
    let world = Arc::make_mut(&mut server.world);
    for region in 0..10 {
        for i in 0..20 {
            world.set_block(region * 512 + i, -40, i, stone);
        }
    }
}

fn region_maps(server: &Server) -> Vec<(RegionKey, Arc<RegionEdits>)> {
    server
        .world
        .edit_regions()
        .map(|(key, map)| (key, Arc::clone(map)))
        .collect()
}

/// Done-when (a): hold the snapshot an in-flight mesh job holds, tick once,
/// and the tick copies nothing -- not the world (`clone_generation`), not a
/// single region's edits (`Arc::ptr_eq`). Failed before #387: the job held an
/// `Arc<World>`, and `make_mut` cloned the whole world on the tick.
#[test]
fn a_held_mesh_snapshot_does_not_make_the_tick_clone_the_edits() {
    let mut server = Server::new();
    server.open(std::path::Path::new(
        "cubara-nonexistent-mesh-snapshot-fixture",
    ));
    server.place_player_on_ground();
    edit_ten_regions(&mut server);
    assert_eq!(server.world.edit_count(), 200);

    // What `MeshPool::request` is handed, and what a job keeps until it is
    // done: taken from the world as it is now.
    let held = Arc::new(server.world.snapshot());
    let generation_before = server.world.clone_generation();
    let regions_before = region_maps(&server);
    assert_eq!(regions_before.len(), 10, "ten regions were edited");

    let inputs = PlayerInputs::one(PlayerId::LOCAL, InputFrame::default());
    server.tick_sim_all(&inputs);

    assert_eq!(
        server.world.clone_generation(),
        generation_before,
        "the tick cloned the world: something held an Arc<World> across it"
    );
    let regions_after = region_maps(&server);
    assert_eq!(regions_after.len(), regions_before.len());
    for ((key_before, before), (key_after, after)) in regions_before.iter().zip(&regions_after) {
        assert_eq!(key_before, key_after);
        assert!(
            Arc::ptr_eq(before, after),
            "the tick copied region {key_before:?}'s edits, which it did not touch"
        );
    }
    // And the held snapshot still shares every one of them with the world:
    // nothing was copied on either side.
    for ((key, world_map), (snap_key, snap_map)) in regions_after.iter().zip(held.edit_regions()) {
        assert_eq!(*key, snap_key);
        assert!(
            Arc::ptr_eq(world_map, snap_map),
            "the snapshot and the world hold different copies of region {key:?}"
        );
    }
    drop(held);
}

/// The other half of the bargain: the snapshot a job holds is what the world
/// was, and the world the tick runs on keeps moving. A tick that edits a
/// block (here, the server's own `set_block` between two ticks, as a flood
/// step in #346 will) copies that block's region and leaves the other nine
/// shared with the snapshot.
#[test]
fn an_edit_between_ticks_copies_one_region_and_the_snapshot_keeps_the_rest() {
    let mut server = Server::new();
    server.open(std::path::Path::new(
        "cubara-nonexistent-mesh-snapshot-fixture",
    ));
    server.place_player_on_ground();
    edit_ten_regions(&mut server);
    let stone = server.terrain().stone;

    let held = Arc::new(server.world.snapshot());
    let blocks = server.terrain();
    let dug = [3 * 512 + 100, -40, 100];
    assert!(
        held.is_solid_at(dug[0], dug[1], dug[2], blocks),
        "the test digs out rock; this is not rock"
    );
    let inputs = PlayerInputs::one(PlayerId::LOCAL, InputFrame::default());
    server.tick_sim_all(&inputs);
    // An edit lands in region 3 (x = 3 * 512 .. 4 * 512): a block dug out.
    Arc::make_mut(&mut server.world).set_block(dug[0], dug[1], dug[2], cubara_voxel::BlockId::AIR);
    server.tick_sim_all(&inputs);

    let edited: RegionKey = (3, -1, 0);
    let mut changed = Vec::new();
    for ((key, world_map), (_, snap_map)) in server.world.edit_regions().zip(held.edit_regions()) {
        if !Arc::ptr_eq(world_map, snap_map) {
            changed.push(key);
        }
    }
    assert_eq!(
        changed,
        vec![edited],
        "exactly the edited region was copied"
    );
    assert_eq!(server.world.edit_count(), held.edit_count() + 1);
    assert!(
        held.is_solid_at(dug[0], dug[1], dug[2], blocks),
        "the snapshot saw an edit made after it was taken"
    );
    assert!(!server.world.is_solid_at(dug[0], dug[1], dug[2], blocks));
    assert_eq!(
        server.world.block_at(0, -40, 0, blocks),
        stone,
        "an older edit, still there"
    );
}

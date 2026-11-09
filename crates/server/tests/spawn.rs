//! Spawn stands on the surface under open sky, never in a cave.
//!
//! Found by play-testing the real window: New World with seed
//! 1881928537856268101 put the player at y = 12 at the bottom of a shaft,
//! because spawn raycast down the origin column and took the first solid block
//! it met, which was a cave floor.

use cubara_server::Server;
use cubara_voxel::FixedVec3;

const CAVE_SEED: u64 = 1881928537856268101;

fn world_with_seed(seed: u64) -> Server {
    let mut s = Server::new();
    s.open(std::path::Path::new("cubara-nonexistent-spawn-fixture"));
    s.new_world(seed);
    s
}

fn column(spawn: FixedVec3) -> (i32, i32, i32) {
    let b = |f: cubara_voxel::Fixed| f.to_f32().floor() as i32;
    (b(spawn.x), b(spawn.y), b(spawn.z))
}

/// Assert the spawn's feet stand two blocks over its column's generated
/// surface, on a solid surface block, with air from the feet upward.
fn assert_on_surface(s: &Server, seed: u64) {
    let terrain = s.terrain.expect("assets are loaded");
    let (x, feet, z) = column(s.world_spawn().expect("a spawn"));
    let surface = s.world.surface_height(x, z);
    assert_eq!(feet, surface + 2, "seed {seed}: not on the surface");
    assert!(
        s.world.is_solid_at(x, surface, z, terrain),
        "seed {seed}: no ground under the feet"
    );
    for y in surface + 1..=surface + 32 {
        assert!(
            !s.world.is_solid_at(x, y, z, terrain),
            "seed {seed}: solid block at y = {y} above the spawn"
        );
    }
}

/// What spawn did before: the first solid block under a ray from y = 200.
fn old_spawn_feet(s: &Server) -> i32 {
    let hit = s
        .world
        .raycast(
            [0.5, 200.0, 0.5],
            [0.0, -1.0, 0.0],
            400.0,
            s.terrain.unwrap(),
        )
        .expect("ground");
    hit.block[1] + 2
}

#[test]
fn spawn_is_on_the_surface_not_in_a_cave() {
    let s = world_with_seed(CAVE_SEED);
    assert_on_surface(&s, CAVE_SEED);
    let (_, feet, _) = column(s.world_spawn().unwrap());
    assert!(feet > 12, "still at the bottom of the shaft: y = {feet}");
}

#[test]
fn spawn_is_on_the_surface_for_many_seeds() {
    let mut underground_before = 0;
    for seed in 0..1000u64 {
        let s = world_with_seed(seed);
        assert_on_surface(&s, seed);
        if old_spawn_feet(&s) != s.world.surface_height(0, 0) + 2 {
            underground_before += 1;
        }
    }
    println!("OLD_SPAWN_NOT_ON_ORIGIN_SURFACE: {underground_before} of 1000");
}

#[test]
fn spawn_is_deterministic() {
    for seed in [0, 7, CAVE_SEED] {
        let a = world_with_seed(seed).world_spawn();
        let b = world_with_seed(seed).world_spawn();
        assert_eq!(a, b, "seed {seed}");
    }
}

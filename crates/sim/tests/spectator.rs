//! The three play modes (owner's decision, 2026-11-09; `ROADMAP.md`, "Recorded,
//! 2026-11-09 -- play modes").
//!
//! - **Survival** and **Creative** walk, with collision. Neither flies.
//! - **Spectator** is a camera that flies through everything; F4 switches
//!   between your normal mode and it, and leaving puts you back exactly where
//!   you pressed F4.
//!
//! Through public API only, like `determinism.rs`: `Sim::new`, `Sim::tick`,
//! `WorldHash::compute`.

use cubara_sim::{InputFrame, Player, PlayerId, PlayerInputs, Sim, WorldHash};
use cubara_voxel::{Angle, BlockId, ChunkCoord, FixedVec3};
use cubara_world::{OreSet, TerrainBlocks, World};

const P: PlayerId = PlayerId::LOCAL;

fn blocks() -> TerrainBlocks {
    TerrainBlocks {
        oak: None,
        ores: OreSet::EMPTY,
        grass: BlockId(1),
        soil: BlockId(2),
        stone: BlockId(3),
    }
}

fn region() -> Vec<ChunkCoord> {
    (-3..=3)
        .flat_map(|x| (0..=2).flat_map(move |y| (-3..=3).map(move |z| ChunkCoord::new(x, y, z))))
        .collect()
}

/// The mouse motion these scripts are written in, as the client converts it.
fn pixels(px: f32) -> Angle {
    Angle::from_raw((px * cubara_sim::SENSITIVITY_PER_PIXEL as f32) as i32)
}

fn tick(sim: &mut Sim, world: &mut World, input: InputFrame) {
    sim.tick(world, &PlayerInputs::one(P, input), blocks());
}

fn idle() -> InputFrame {
    InputFrame::default()
}

fn f4() -> InputFrame {
    InputFrame {
        toggle_spectate: true,
        ..InputFrame::default()
    }
}

/// A player standing on the real terrain at the origin column, eye height
/// `above` blocks over the surface, at rest after 150 ticks. Returns the world
/// too, with the surface's y.
fn standing(creative: bool, above: f32) -> (Sim, World, i32) {
    let mut world = World::with_seed(7);
    let ground = world
        .raycast([0.5, 200.0, 0.5], [0.0, -1.0, 0.0], 400.0, blocks())
        .expect("ground below")
        .block[1];
    let mut sim = Sim::new(
        7,
        Player::new(
            FixedVec3::from_f32([0.5, ground as f32 + above, 0.5]),
            Angle::ZERO,
            Angle::ZERO,
        ),
    );
    sim.player_mut(P).set_creative(creative);
    for _ in 0..150 {
        tick(&mut sim, &mut world, idle());
    }
    assert!(sim.player(P).on_ground, "the fixture player never landed");
    (sim, world, ground)
}

/// **F4 round trip.** Press F4, fly 50 blocks through solid terrain, press F4.
/// The body is bit-identical, and the world hash is unchanged except for what
/// is tick-driven.
///
/// The body starts **in mid-air, falling**: a frozen body that kept falling
/// during the flight would then have landed (and been hurt) somewhere else, so
/// this fails if the body is not truly frozen, where a body at rest would
/// hide it.
#[test]
fn f4_round_trip_leaves_the_body_bit_identical_and_the_world_unchanged() {
    let blocks = blocks();
    let world = World::with_seed(7);
    let ground = world
        .raycast([0.5, 200.0, 0.5], [0.0, -1.0, 0.0], 400.0, blocks)
        .expect("ground below")
        .block[1];
    let start = FixedVec3::from_f32([0.5, ground as f32 + 30.0, 0.5]);
    let new_sim = || Sim::new(7, Player::new(start, Angle::from_radians(0.7), Angle::ZERO));

    // The control: the same ticks with F4 never pressed.
    let (mut control, mut control_world) = (new_sim(), World::with_seed(7));
    // The subject.
    let (mut sim, mut subject_world) = (new_sim(), World::with_seed(7));

    // Both fall for 10 ticks: velocity, fall_distance and on_ground are all
    // non-trivial by now.
    for _ in 0..10 {
        tick(&mut control, &mut control_world, idle());
        tick(&mut sim, &mut subject_world, idle());
    }
    let falling = *sim.player(P);
    assert!(
        falling.velocity.y < FixedVec3::ZERO.y && !falling.on_ground,
        "the fixture body is not mid-fall"
    );

    // F4 on: the camera starts at the eye, and the body does not move.
    tick(&mut sim, &mut subject_world, f4());
    assert!(sim.player(P).is_spectating());
    let body = |sim: &Sim| {
        let p = sim.player(P);
        (
            p.pos,
            p.velocity,
            p.on_ground,
            p.fall_distance,
            p.health,
            p.ticks_since_damage,
            p.yaw(),
            p.pitch(),
        )
    };
    let frozen = body(&sim);
    assert_eq!(
        frozen,
        (
            falling.pos,
            falling.velocity,
            falling.on_ground,
            falling.fall_distance,
            falling.health,
            falling.ticks_since_damage,
            falling.yaw(),
            falling.pitch()
        ),
        "pressing F4 moved the body"
    );

    // Fly 50+ blocks straight down through the ground, looking down.
    let dive = InputFrame {
        move_axes: [0.0, 0.0, 1.0],
        look_delta: [Angle::ZERO, pixels(-100_000.0)],
        ..InputFrame::default()
    };
    let mut went_through_solid = false;
    let mut max_distance = 0.0f32;
    for i in 0..400 {
        let input = if i == 0 {
            dive
        } else {
            InputFrame {
                look_delta: [Angle::ZERO, Angle::ZERO],
                ..dive
            }
        };
        tick(&mut sim, &mut subject_world, input);
        assert_eq!(
            body(&sim),
            frozen,
            "the body changed on spectator tick {i}: it must be frozen"
        );
        let cam = sim.player(P).spectating().expect("still spectating").pos;
        let [x, y, z] = [cam.x.to_f32(), cam.y.to_f32(), cam.z.to_f32()];
        went_through_solid |=
            subject_world.is_solid_at(x.floor() as i32, y.floor() as i32, z.floor() as i32, blocks);
        let [bx, by, bz] = frozen.0.to_f32();
        let d = ((x - bx).powi(2) + (y - by).powi(2) + (z - bz).powi(2)).sqrt();
        max_distance = max_distance.max(d);
        if max_distance > 55.0 && went_through_solid {
            break;
        }
    }
    assert!(
        went_through_solid,
        "the camera never entered solid terrain -- the fixture proves nothing"
    );
    assert!(max_distance > 50.0, "the camera only flew {max_distance}");
    assert_eq!(
        sim.player(P).health,
        falling.health,
        "no damage while flying"
    );

    // F4 off: that tick runs one tick of body physics, like the control's
    // 11th tick. The two worlds now differ by nothing but time spent.
    tick(&mut sim, &mut subject_world, f4());
    assert!(!sim.player(P).is_spectating());
    tick(&mut control, &mut control_world, idle()); // control's tick 11
    assert_eq!(
        *sim.player(P),
        *control.player(P),
        "the body after leaving is not the body of a player who never pressed F4"
    );

    // Tick-driven state is the one thing allowed to differ: the counters.
    sim.tick = control.tick;
    let hash = |s: &Sim, w: &World| WorldHash::compute(s, w, &region(), blocks, 1);
    assert_eq!(
        hash(&sim, &subject_world),
        hash(&control, &control_world),
        "the world hash changed across the F4 round trip"
    );
}

/// The camera is in the hash while it exists -- two worlds that differ only by
/// where a spectator's camera is are different worlds (Rule 1; prediction
/// reconciles against this state).
#[test]
fn the_camera_is_part_of_the_world_hash() {
    let (mut a, mut world_a, _) = standing(false, 3.0);
    let (mut b, mut world_b, _) = standing(false, 3.0);
    tick(&mut a, &mut world_a, f4());
    tick(&mut b, &mut world_b, f4());
    let fly = InputFrame {
        move_axes: [0.0, 1.0, 0.0],
        ..InputFrame::default()
    };
    tick(&mut a, &mut world_a, fly);
    tick(&mut b, &mut world_b, idle());
    let h = |s: &Sim, w: &World| WorldHash::compute(s, w, &region(), blocks(), 1);
    assert_ne!(
        h(&a, &world_a),
        h(&b, &world_b),
        "the camera's position did not reach the hash"
    );
}

/// **Creative and Survival walk.** With no F4, holding forward into a wall does
/// not pass through it, and jumping does not fly -- in either mode.
#[test]
fn creative_and_survival_walk_into_a_wall_and_do_not_fly() {
    for creative in [false, true] {
        let mode = if creative { "creative" } else { "survival" };
        let (mut sim, mut world, ground) = standing(creative, 3.0);
        assert!(!sim.player(P).is_spectating(), "{mode}: a camera appeared");

        let stand = *sim.player(P);
        let (px, pz) = (stand.pos.x.floor_block(), stand.pos.z.floor_block());
        let feet = stand.pos.y.floor_block() - 1;
        let wall_z = pz - 6;
        // A wall across the whole way, far taller than any jump, and thick
        // enough that "through" cannot mean "out the other side of a thin one".
        for x in px - 6..=px + 6 {
            for y in feet - 2..=feet + 12 {
                for z in wall_z - 2..=wall_z {
                    world.set_block(x, y, z, blocks().stone);
                }
            }
        }

        // Straight at it for 600 ticks (~10 s at 5 b/s), jumping every tick.
        for _ in 0..600 {
            tick(
                &mut sim,
                &mut world,
                InputFrame {
                    move_axes: [0.0, 0.0, 1.0],
                    jump: true,
                    ..InputFrame::default()
                },
            );
        }
        let p = sim.player(P);
        let z = p.pos.z.to_f32();
        assert!(
            z > (wall_z + 1) as f32,
            "{mode}: walked through the wall (z = {z}, wall face at {})",
            wall_z + 1
        );
        assert!(
            z < (wall_z + 1) as f32 + 3.0,
            "{mode}: never reached the wall (z = {z}); the test proves nothing"
        );
        assert!(
            p.pos.y.to_f32() < (feet + 8) as f32,
            "{mode}: climbed over a twelve-block wall"
        );
        let _ = ground;
    }

    // Jumping alone: a hop, never flight.
    for creative in [false, true] {
        let mode = if creative { "creative" } else { "survival" };
        let (mut sim, mut world, _) = standing(creative, 3.0);
        let rest = sim.player(P).pos.y.to_f32();
        let mut highest = rest;
        for _ in 0..600 {
            tick(
                &mut sim,
                &mut world,
                InputFrame {
                    jump: true,
                    move_axes: [0.0, 1.0, 0.0], // "up" does nothing off Spectator
                    ..InputFrame::default()
                },
            );
            highest = highest.max(sim.player(P).pos.y.to_f32());
        }
        for _ in 0..90 {
            tick(&mut sim, &mut world, idle()); // let the last hop finish
        }
        assert!(
            highest - rest < 2.0,
            "{mode}: rose {} blocks by jumping -- that is flying",
            highest - rest
        );
        assert!(
            (sim.player(P).pos.y.to_f32() - rest).abs() < 0.01,
            "{mode}: did not come back down"
        );
    }
}

/// No damage of any kind while spectating, however it is asked for.
#[test]
fn a_spectator_takes_no_damage_and_regenerates_nothing() {
    let (mut sim, mut world, _) = standing(false, 3.0);
    sim.player_mut(P).take_damage(6);
    let hurt = *sim.player(P);
    assert!(hurt.health < cubara_sim::MAX_HEALTH);

    tick(&mut sim, &mut world, f4());
    assert!(
        !sim.player_mut(P).take_damage(200),
        "a spectator was killed"
    );
    for _ in 0..(cubara_sim::REGEN_INTERVAL * 3) {
        tick(&mut sim, &mut world, idle());
    }
    let p = sim.player(P);
    assert_eq!(p.health, hurt.health, "a spectator healed or was hurt");
    assert_eq!(
        p.ticks_since_damage, hurt.ticks_since_damage,
        "the regeneration clock ticked while spectating"
    );
}

/// `free_fly_never_hurts_however_far_you_descend`, as it was before the play
/// modes, for the camera: however far a spectator descends, neither the camera
/// nor the body is hurt.
#[test]
fn a_spectator_never_hurts_however_far_it_descends() {
    let (mut sim, mut world, _) = standing(false, 3.0);
    let before = *sim.player(P);
    tick(&mut sim, &mut world, f4());
    let down = InputFrame {
        move_axes: [0.0, -1.0, 0.0],
        ..InputFrame::default()
    };
    for _ in 0..600 {
        tick(&mut sim, &mut world, down);
    }
    let cam = sim.player(P).spectating().expect("spectating").pos;
    assert!(
        cam.y.to_f32() < before.pos.y.to_f32() - 100.0,
        "did not descend"
    );
    tick(&mut sim, &mut world, f4());
    assert_eq!(sim.player(P).health, before.health);
    assert_eq!(
        sim.player(P).pos,
        before.pos,
        "the body was moved by the dive"
    );
}

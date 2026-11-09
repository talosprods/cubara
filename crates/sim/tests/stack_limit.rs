//! Lowering an item's `max_stack` must never delete saved items.
//!
//! A saved stack is rebuilt with `ItemStack::new(.., max_stack)`, which rejects
//! a count above the limit. Loading used to `.ok()` that away, so an asset edit
//! from 64 to 16 made every saved stack of 17..=64 vanish without a log line.
//! Now the slot keeps what fits and the rest moves to free inventory slots, and
//! only when there are none, onto the floor.

use cubara_sim::{load_world, save_world, Player, PlayerId, Sim};
use cubara_voxel::{
    Angle, BlockRegistry, DropRule, Faces, FixedVec3, Interact, ItemDef, ItemRegistry, Material,
    Rarity, Shape,
};
use cubara_world::{TerrainBlocks, World};
use std::path::PathBuf;

const P: PlayerId = PlayerId::LOCAL;

fn registry() -> BlockRegistry {
    let block = |name: &str| {
        let short = name.trim_start_matches("cubara:");
        (
            PathBuf::from(format!("{short}.ron")),
            Material {
                name: name.to_string(),
                solid: true,
                faces: Faces::All(short.to_string()),
                shapes: vec![Shape::Full],
                drops: DropRule::SameName,
                requires_tier: 0,
                hardness: Some(1),
                interact: Interact::None,
            },
        )
    };
    BlockRegistry::from_materials(vec![
        block("cubara:grass"),
        block("cubara:soil"),
        block("cubara:stone"),
    ])
    .expect("fixture registry is valid")
}

fn items(max_stack: u8) -> ItemRegistry {
    ItemRegistry::from_defs(vec![(
        PathBuf::from("stone.ron"),
        ItemDef {
            name: "cubara:stone".to_string(),
            max_stack,
            durability: None,
            tier: 0,
            speed: None,
            burn_ticks: None,
            rarity: Rarity::Common,
        },
    )])
    .expect("fixture items are valid")
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "cubara-stack-limit-{name}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// Save with limit 64, load with `new_limit`; return the loaded sim.
fn round_trip(name: &str, fill: impl Fn(&mut Player, &ItemRegistry), new_limit: u8) -> Sim {
    let reg = registry();
    let blocks = TerrainBlocks::from_registry(&reg);
    let old = items(64);
    let world = World::with_seed(7);
    let mut sim = Sim::new(
        7,
        Player::new(
            FixedVec3::from_f32([0.5, 50.0, 0.5]),
            Angle::ZERO,
            Angle::ZERO,
        ),
    );
    fill(sim.player_mut(P), &old);
    let dir = scratch(name);
    save_world(&dir, &sim, &world, &reg, &old, blocks).expect("save");
    let (loaded, _) = load_world(&dir, &reg, &items(new_limit), blocks).expect("load");
    let _ = std::fs::remove_dir_all(&dir);
    loaded
}

fn count_in_inventory(sim: &Sim) -> u32 {
    sim.player(P)
        .inventory
        .slots()
        .flatten()
        .map(|s| u32::from(s.count()))
        .sum()
}

fn count_on_floor(sim: &Sim) -> u32 {
    sim.entities
        .sorted()
        .iter()
        .map(|(_, d)| u32::from(d.stack.count()))
        .sum()
}

#[test]
fn a_stack_over_a_lowered_limit_is_split_into_free_slots_not_deleted() {
    let loaded = round_trip(
        "split",
        |p, items| {
            let stone = items.id_of("cubara:stone").unwrap();
            p.inventory
                .set_slot(0, Some(items.new_stack(stone, 64).unwrap()));
        },
        16,
    );
    assert_eq!(
        count_in_inventory(&loaded) + count_on_floor(&loaded),
        64,
        "lowering max_stack deleted saved items"
    );
    assert_eq!(count_in_inventory(&loaded), 64, "there was room in the bag");
    assert!(
        loaded
            .player(P)
            .inventory
            .slots()
            .flatten()
            .all(|s| s.count() <= 16),
        "a stack still exceeds the new limit"
    );
}

#[test]
fn with_no_free_slot_the_excess_lands_on_the_floor_not_in_the_void() {
    let loaded = round_trip(
        "spill",
        |p, items| {
            let stone = items.id_of("cubara:stone").unwrap();
            // Every slot holds a full 64: after the limit drops there is
            // nowhere in the bag for the 48 extra each one now carries.
            for i in 0..36 {
                p.inventory
                    .set_slot(i, Some(items.new_stack(stone, 64).unwrap()));
            }
        },
        16,
    );
    assert_eq!(
        count_in_inventory(&loaded) + count_on_floor(&loaded),
        36 * 64,
        "items were deleted when the bag had no room for the excess"
    );
    assert!(count_on_floor(&loaded) > 0, "nothing reached the floor");
}

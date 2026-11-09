//! A worn tool put in a furnace comes back as worn as it went in.
//!
//! Furnace slots are `(ItemId, count)`, which cannot say how worn an item is, so
//! every path that rebuilt a stack from a slot rebuilt a *new* tool: put a
//! one-use pickaxe in, click again, hold a fresh one. A free repair, on every
//! path -- the click, the spill when the furnace is broken, and the save file.
//!
//! Each test uses a **damaged** tool (3 uses left of 400) and asserts the number,
//! because a full-durability tool is exactly what the broken code produces: a
//! test that only checked "a pickaxe came back" would pass on it.

use cubara_server::{Action, FurnaceSlot, Server};
use cubara_sim::PlayerId;
use cubara_voxel::{ItemId, ItemStack, ItemState};

const WORN: u16 = 3;

/// A world with assets, its player, and a furnace two blocks away (in reach).
fn fixture() -> (Server, PlayerId, [i32; 3], ItemId) {
    let mut s = Server::new();
    s.open(std::path::Path::new(
        "cubara-nonexistent-furnace-wear-fixture",
    ));
    let who = s.local.expect("a local client");
    let p = s.sim.player(who).pos;
    let at = [p.x.floor_block() + 2, p.y.floor_block(), p.z.floor_block()];
    s.add_furnace(at);
    let pick = s
        .items
        .as_ref()
        .expect("assets loaded")
        .id_of("cubara:stone_pick")
        .expect("stone pick is an item");
    (s, who, at, pick)
}

fn worn(s: &Server, pick: ItemId, remaining: u16) -> ItemStack {
    let items = s.items.as_ref().expect("assets loaded");
    ItemStack::new(
        pick,
        1,
        ItemState::Durability { remaining },
        items.max_stack(pick),
    )
    .expect("a tool is a stack of one")
}

fn click(s: &mut Server, at: [i32; 3], slot: FurnaceSlot) {
    s.apply(Action::ClickFurnace { pos: at, slot });
}

fn held(s: &Server, who: PlayerId) -> Option<ItemStack> {
    s.sim.player(who).crafting.held()
}

#[test]
fn a_worn_tool_comes_back_out_of_the_input_slot_as_worn() {
    let (mut s, who, at, pick) = fixture();
    let stack = worn(&s, pick, WORN);
    s.sim.player_mut(who).crafting.set_held(Some(stack));
    click(&mut s, at, FurnaceSlot::Input);
    assert!(held(&s, who).is_none(), "the furnace did not take the tool");
    assert_eq!(s.world.furnace_at(at).unwrap().input_wear, Some(WORN));

    click(&mut s, at, FurnaceSlot::Input);
    assert_eq!(
        s.world.furnace_at(at).unwrap().input_wear,
        None,
        "an emptied slot kept its tool's wear (two equal furnaces would differ)"
    );
    assert_eq!(
        held(&s, who).map(|t| t.state()),
        Some(ItemState::Durability { remaining: WORN }),
        "the tool came back with different wear than it went in with"
    );
}

#[test]
fn a_worn_tool_comes_back_out_of_the_fuel_slot_as_worn() {
    let (mut s, who, at, pick) = fixture();
    let stack = worn(&s, pick, WORN);
    s.sim.player_mut(who).crafting.set_held(Some(stack));
    click(&mut s, at, FurnaceSlot::Fuel);
    click(&mut s, at, FurnaceSlot::Fuel);
    assert_eq!(
        held(&s, who).map(|t| t.state()),
        Some(ItemState::Durability { remaining: WORN }),
    );
}

/// Swapping hands both ways: the tool that leaves keeps *its* wear and the one
/// that enters brings *its own*. Two different wear values, so reading the wrong
/// one is visible.
#[test]
fn swapping_two_worn_tools_keeps_each_ones_wear() {
    let (mut s, who, at, pick) = fixture();
    let first = worn(&s, pick, 3);
    let second = worn(&s, pick, 250);
    s.sim.player_mut(who).crafting.set_held(Some(first));
    click(&mut s, at, FurnaceSlot::Input);
    s.sim.player_mut(who).crafting.set_held(Some(second));
    click(&mut s, at, FurnaceSlot::Input);
    assert_eq!(
        held(&s, who).map(|t| t.state()),
        Some(ItemState::Durability { remaining: 3 }),
        "the tool swapped out lost its wear"
    );
    click(&mut s, at, FurnaceSlot::Input);
    assert_eq!(
        held(&s, who).map(|t| t.state()),
        Some(ItemState::Durability { remaining: 250 }),
        "the tool swapped in lost its wear"
    );
}

/// A plain stack must not pick up the wear of whatever was in the slot before.
#[test]
fn wear_does_not_leak_onto_the_next_stack_in_the_slot() {
    let (mut s, who, at, pick) = fixture();
    let stack = worn(&s, pick, WORN);
    s.sim.player_mut(who).crafting.set_held(Some(stack));
    click(&mut s, at, FurnaceSlot::Input);
    click(&mut s, at, FurnaceSlot::Input);
    s.sim.player_mut(who).crafting.set_held(None);

    let items = s.items.as_ref().expect("assets loaded");
    let log = items.id_of("cubara:oak_log").expect("oak log");
    let logs = items.new_stack(log, 5).expect("a stack");
    s.sim.player_mut(who).crafting.set_held(Some(logs));
    click(&mut s, at, FurnaceSlot::Input);
    click(&mut s, at, FurnaceSlot::Input);
    assert_eq!(
        held(&s, who).map(|t| (t.count(), t.state())),
        Some((5, ItemState::None))
    );
}

/// Breaking the furnace spills its contents; a worn tool spills worn.
#[test]
fn a_worn_tool_spilled_from_a_broken_furnace_is_still_worn() {
    let (mut s, who, at, pick) = fixture();
    let stack = worn(&s, pick, WORN);
    s.sim.player_mut(who).crafting.set_held(Some(stack));
    click(&mut s, at, FurnaceSlot::Input);

    s.break_at_as(who, at);

    let spilled: Vec<_> = s
        .sim
        .entities
        .sorted()
        .into_iter()
        .map(|(_, d)| d.stack)
        .filter(|st| st.item() == pick)
        .collect();
    assert_eq!(spilled.len(), 1, "the tool was not spilled");
    assert_eq!(
        spilled[0].state(),
        ItemState::Durability { remaining: WORN }
    );
}

/// Through a save and a load.
#[test]
fn a_worn_tool_in_a_furnace_survives_a_save_and_load() {
    let (mut s, who, at, pick) = fixture();
    let stack = worn(&s, pick, WORN);
    s.sim.player_mut(who).crafting.set_held(Some(stack));
    click(&mut s, at, FurnaceSlot::Input);
    let fuel_stack = worn(&s, pick, 77);
    s.sim.player_mut(who).crafting.set_held(Some(fuel_stack));
    click(&mut s, at, FurnaceSlot::Fuel);

    let dir = std::env::temp_dir().join(format!("cubara-furnace-wear-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    assert!(s.save_to(&dir), "the world did not save");

    let mut loaded = Server::new();
    assert!(loaded.open(&dir), "the saved world did not load");
    let _ = std::fs::remove_dir_all(&dir);

    let f = loaded.world.furnace_at(at).expect("the furnace came back");
    assert_eq!(f.input, Some((pick, 1)));
    assert_eq!(f.input_wear, Some(WORN), "input wear lost in the save");
    assert_eq!(f.fuel_wear, Some(77), "fuel wear lost in the save");

    let who = loaded.local.expect("a local client");
    click(&mut loaded, at, FurnaceSlot::Input);
    assert_eq!(
        held(&loaded, who).map(|t| t.state()),
        Some(ItemState::Durability { remaining: WORN }),
    );
}

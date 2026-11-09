//! Spectator on the server (owner's decision, 2026-11-09; `ROADMAP.md`,
//! "Recorded, 2026-11-09 -- play modes").
//!
//! The rules here are the server's, not the client's: hiding the hotbar is not a
//! rule. Every test drives a real [`Session`] through an in-process link, the
//! same `ClientMessage`s a remote client sends, and each rejection test **first
//! proves the same action works for a player who is not spectating** -- a
//! rejection test that has never seen the action succeed might be rejecting
//! everything.

use cubara_server::headless::{Config, Session};
use cubara_server::net::Link;
use cubara_server::wire::{ClientMessage, ServerMessage};
use cubara_server::{Action, Effect, FurnaceSlot};
use cubara_sim::{InputFrame, PlayerId, SlotRef};
use cubara_voxel::{BlockId, ChunkCoord, FixedVec3};

const SKY: i32 = 400;

struct Client {
    link: Link<ClientMessage, ServerMessage>,
    id: PlayerId,
    seq: u64,
    heard: Vec<Effect>,
}

struct Arena {
    session: Session,
    cfg: Config,
    clients: Vec<Client>,
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

impl Arena {
    fn new() -> Self {
        let cfg = Config {
            world: std::path::PathBuf::from("cubara-nonexistent-spectator-fixture"),
            autosave_ticks: 0,
            ..Config::default()
        };
        let session = Session::open(&cfg);
        let mut arena = Self {
            session,
            cfg,
            clients: Vec::new(),
        };
        // A stone platform in empty sky, so nobody falls while the test runs
        // and a raycast hits what the test put there and nothing else.
        let stone = arena.stone();
        for x in -3..=8 {
            for z in -3..=3 {
                arena.session.server.set_block([x, SKY - 2, z], stone);
            }
        }
        arena
    }

    fn stone(&self) -> BlockId {
        self.session
            .server
            .blocks_registry
            .as_deref()
            .and_then(|r| r.id_of("cubara:stone"))
            .expect("stone is a block")
    }

    fn cobble(&self) -> BlockId {
        self.session
            .server
            .blocks_registry
            .as_deref()
            .and_then(|r| r.id_of("cubara:cobble"))
            .expect("cobble is a block")
    }

    /// Join a client by the route a remote one takes and stand it at `x` on the
    /// platform, looking down -Z. Returns its index.
    fn join(&mut self, x: i32) -> usize {
        let mut link = self.session.attach();
        let mut heard = Vec::new();
        self.session.advance(1, &self.cfg);
        let mut id = None;
        for msg in link.poll() {
            match msg {
                ServerMessage::Welcome { you, .. } => id = Some(you),
                ServerMessage::Effects(fx) => heard.extend(fx),
                _ => {}
            }
        }
        let id = id.expect("the server never welcomed the client");
        let p = self.session.server.sim.player_mut(id);
        p.pos = FixedVec3::from_blocks(x, SKY, 0);
        p.velocity = FixedVec3::ZERO;
        p.fall_distance = cubara_voxel::Fixed::ZERO;
        self.clients.push(Client {
            link,
            id,
            seq: 0,
            heard,
        });
        let i = self.clients.len() - 1;
        for _ in 0..5 {
            self.step_all(&vec![idle(); self.clients.len()]);
        }
        i
    }

    /// One tick: every client sends `frames[i]` as its numbered input.
    fn step_all(&mut self, frames: &[InputFrame]) {
        for (c, f) in self.clients.iter_mut().zip(frames) {
            c.link.send(ClientMessage::Input {
                seq: c.seq,
                frame: *f,
            });
            c.seq += 1;
        }
        self.session.advance(1, &self.cfg);
        for c in &mut self.clients {
            for msg in c.link.poll() {
                if let ServerMessage::Effects(fx) = msg {
                    c.heard.extend(fx);
                }
            }
        }
    }

    /// One tick in which client `who` sends `frame` and everybody else idles.
    fn step(&mut self, who: usize, frame: InputFrame) {
        let mut frames = vec![idle(); self.clients.len()];
        frames[who] = frame;
        self.step_all(&frames);
    }

    /// Client `who` sends one action, over the wire, and a tick passes.
    fn act(&mut self, who: usize, action: Action) {
        self.clients[who].link.send(ClientMessage::Act(action));
        self.step(who, idle());
    }

    fn id(&self, who: usize) -> PlayerId {
        self.clients[who].id
    }

    fn block(&self, pos: [i32; 3]) -> BlockId {
        self.session
            .server
            .world
            .block_at(pos[0], pos[1], pos[2], self.session.server.terrain())
    }

    fn spectate(&mut self, who: usize) {
        self.step(who, f4());
        assert!(
            self.session.server.sim.player(self.id(who)).is_spectating(),
            "F4 over the wire did not start spectating"
        );
    }

    fn creative(&mut self, who: usize) {
        self.act(who, Action::SetCreative(true));
    }

    fn give(&mut self, who: usize, item: &str, count: u8) {
        let id = self.id(who);
        let items = self.session.server.items.clone().expect("assets");
        let stack = items
            .new_stack(items.id_of(item).expect("a shipped item"), count)
            .expect("a stack");
        let p = self.session.server.sim.player_mut(id);
        p.inventory.set_slot(0, Some(stack));
        p.inventory.select(0);
    }
}

// ---------------------------------------------------------------------------
// Rejections
// ---------------------------------------------------------------------------

/// Break: held over the wire as `InputFrame::breaking`.
#[test]
fn a_spectators_break_is_rejected() {
    let target = [0, SKY, -3];

    // The control: a creative player who is not spectating breaks it at once.
    let mut a = Arena::new();
    let c = a.join(0);
    let stone = a.stone();
    a.session.server.set_block(target, stone);
    a.creative(c);
    for _ in 0..4 {
        a.step(
            c,
            InputFrame {
                breaking: true,
                ..idle()
            },
        );
    }
    assert_eq!(
        a.block(target),
        BlockId::AIR,
        "the control never broke the block, so the rejection below proves nothing"
    );

    // The subject: the same, spectating, for ten times as long.
    let mut a = Arena::new();
    let c = a.join(0);
    let stone = a.stone();
    a.session.server.set_block(target, stone);
    a.creative(c);
    a.spectate(c);
    for _ in 0..40 {
        a.step(
            c,
            InputFrame {
                breaking: true,
                ..idle()
            },
        );
    }
    assert_eq!(a.block(target), stone, "a spectating player broke a block");
}

#[test]
fn a_spectators_place_is_rejected() {
    let target = [0, SKY, -3];
    let beside = [0, SKY, -2];

    let mut a = Arena::new();
    let c = a.join(0);
    let (stone, cobble) = (a.stone(), a.cobble());
    a.session.server.set_block(target, stone);
    a.give(c, "cubara:cobble", 5);
    a.act(c, Action::Place);
    assert_eq!(
        a.block(beside),
        cobble,
        "the control never placed a block, so the rejection below proves nothing"
    );

    let mut a = Arena::new();
    let c = a.join(0);
    let stone = a.stone();
    a.session.server.set_block(target, stone);
    a.give(c, "cubara:cobble", 5);
    a.spectate(c);
    a.act(c, Action::Place);
    assert_eq!(
        a.block(beside),
        BlockId::AIR,
        "a spectating player placed a block"
    );
    let held = a
        .session
        .server
        .sim
        .player(a.id(c))
        .inventory
        .slot(0)
        .map(|s| s.count());
    assert_eq!(held, Some(5), "a refused placement still spent the item");
}

/// A furnace click names its target, so it is its own wire path.
#[test]
fn a_spectators_furnace_click_is_rejected() {
    let furnace = [0, SKY, -3];
    let fuel = |a: &Arena| a.session.server.world.furnace_at(furnace).expect("it").fuel;
    let hold_log = |a: &mut Arena, c: usize| {
        let id = a.id(c);
        let items = a.session.server.items.clone().expect("assets");
        let log = items.id_of("cubara:oak_log").expect("oak log");
        let stack = items.new_stack(log, 1).expect("a stack");
        a.session
            .server
            .sim
            .player_mut(id)
            .crafting
            .set_held(Some(stack));
    };
    let click = Action::ClickFurnace {
        pos: furnace,
        slot: FurnaceSlot::Fuel,
    };

    let mut a = Arena::new();
    let c = a.join(0);
    a.session.server.add_furnace(furnace);
    hold_log(&mut a, c);
    a.act(c, click.clone());
    assert!(
        fuel(&a).is_some(),
        "the control never fuelled the furnace, so the rejection below proves nothing"
    );

    let mut a = Arena::new();
    let c = a.join(0);
    a.session.server.add_furnace(furnace);
    hold_log(&mut a, c);
    a.spectate(c);
    a.act(c, click);
    assert_eq!(fuel(&a), None, "a spectating player fuelled a furnace");
    assert!(
        a.session
            .server
            .sim
            .player(a.id(c))
            .crafting
            .held()
            .is_some(),
        "the refused click still took the item out of the cursor"
    );
}

/// A real furnace block, so `Interact` has something to use.
fn place_furnace(a: &mut Arena, at: [i32; 3]) {
    let furnace = a
        .session
        .server
        .blocks_registry
        .as_deref()
        .and_then(|r| r.id_of("cubara:furnace"))
        .expect("the furnace is a block");
    a.session.server.set_block(at, furnace);
}

/// Crafting and inventory clicks, and using what is under the crosshair.
#[test]
fn a_spectators_inventory_clicks_and_interactions_are_rejected() {
    let furnace = [0, SKY, -3];
    let click = Action::ClickSlot {
        slot: SlotRef::Inventory(0),
        right: false,
    };
    let opened = |a: &Arena, c: usize| {
        a.clients[c]
            .heard
            .iter()
            .any(|e| matches!(e, Effect::Open(_)))
    };

    let mut a = Arena::new();
    let c = a.join(0);
    place_furnace(&mut a, furnace);
    a.give(c, "cubara:oak_log", 3);
    a.act(c, click.clone());
    assert!(
        a.session
            .server
            .sim
            .player(a.id(c))
            .crafting
            .held()
            .is_some(),
        "the control's click never picked the stack up"
    );
    a.act(c, Action::Interact);
    assert!(opened(&a, c), "the control never opened the furnace");

    let mut a = Arena::new();
    let c = a.join(0);
    place_furnace(&mut a, furnace);
    a.give(c, "cubara:oak_log", 3);
    a.spectate(c);
    a.act(c, click);
    let p = a.session.server.sim.player(a.id(c));
    assert!(p.crafting.held().is_none(), "a spectator picked up a stack");
    assert_eq!(
        p.inventory.slot(0).map(|s| s.count()),
        Some(3),
        "a spectator's click moved items"
    );
    a.act(c, Action::Interact);
    assert!(!opened(&a, c), "a spectator opened a furnace");
}

/// Pickups: a dropped item at the frozen body's feet stays where it is.
#[test]
fn a_spectator_does_not_pick_things_up() {
    let drop_at = |a: &mut Arena, c: usize| {
        let id = a.id(c);
        let at = a.session.server.sim.player(id).pos;
        let items = a.session.server.items.clone().expect("assets");
        let stack = items
            .new_stack(items.id_of("cubara:oak_log").expect("oak log"), 2)
            .expect("a stack");
        a.session
            .server
            .sim
            .entities
            .spawn_item(stack, at, FixedVec3::ZERO);
    };
    let carried = |a: &Arena, c: usize| {
        a.session
            .server
            .sim
            .player(a.id(c))
            .inventory
            .slots()
            .flatten()
            .count()
    };

    let mut a = Arena::new();
    let c = a.join(0);
    drop_at(&mut a, c);
    for _ in 0..30 {
        a.step(c, idle());
    }
    assert_eq!(
        carried(&a, c),
        1,
        "the control never picked the drop up, so the rejection below proves nothing"
    );

    let mut a = Arena::new();
    let c = a.join(0);
    a.spectate(c);
    drop_at(&mut a, c);
    for _ in 0..30 {
        a.step(c, idle());
    }
    assert_eq!(carried(&a, c), 0, "a spectator picked an item up");
    assert_eq!(
        a.session.server.sim.entities.len(),
        1,
        "the item left the world without anyone taking it"
    );
}

// ---------------------------------------------------------------------------
// Who sees what
// ---------------------------------------------------------------------------

/// **Others see the body, not the camera.** A second client sees the spectating
/// player's figure at the frozen body, and never at the camera -- both while it
/// is watching and when it arrives afterwards.
#[test]
fn others_see_the_frozen_body_and_never_the_camera() {
    let mut a = Arena::new();
    let spectator = a.join(0);
    let watcher = a.join(4);
    let body = a.session.server.sim.player(a.id(spectator)).pos;

    a.spectate(spectator);
    let fly = InputFrame {
        move_axes: [0.0, 0.5, 1.0],
        ..idle()
    };
    for _ in 0..300 {
        a.step(spectator, fly);
    }
    let camera = a
        .session
        .server
        .sim
        .player(a.id(spectator))
        .spectating()
        .expect("still spectating")
        .pos;
    let apart = camera.to_f32();
    let frozen = body.to_f32();
    let d = ((apart[0] - frozen[0]).powi(2) + (apart[2] - frozen[2]).powi(2)).sqrt();
    assert!(
        d > 50.0,
        "the camera only flew {d} blocks; nothing is proven"
    );
    assert_eq!(
        a.session.server.sim.player(a.id(spectator)).pos,
        body,
        "the body moved while spectating"
    );

    // Everybody the watcher was told about the spectator, ever, is the body.
    let who = a.id(spectator);
    let seen_by = |a: &Arena, c: usize| -> Vec<FixedVec3> {
        a.clients[c]
            .heard
            .iter()
            .filter_map(|e| match e {
                Effect::PlayerMoved { who: w, pos, .. } if *w == who => Some(*pos),
                _ => None,
            })
            .collect()
    };
    let seen = seen_by(&a, watcher);
    assert!(
        !seen.is_empty(),
        "the watcher was never told about the spectator"
    );
    for pos in &seen {
        assert_eq!(
            *pos, body,
            "the watcher was told the spectator is not at its body"
        );
        assert_ne!(*pos, camera, "the watcher was told where the camera is");
    }

    // And somebody who arrives after the flight, who has to be backfilled.
    let late = a.join(2);
    for _ in 0..3 {
        a.step_all(&vec![idle(); a.clients.len()]);
    }
    let seen = seen_by(&a, late);
    assert!(
        !seen.is_empty(),
        "a client arriving while somebody spectates was never shown their body"
    );
    assert!(
        seen.iter().all(|p| *p == body),
        "a late arrival was shown the camera, not the body: {seen:?}"
    );
}

/// **Interest follows the camera** while spectating, and the body again once
/// you leave.
#[test]
fn interest_follows_the_camera_and_returns_to_the_body() {
    let mut a = Arena::new();
    let c = a.join(0);
    let id = a.id(c);
    let chunk = |p: FixedVec3| ChunkCoord::from_world_pos(p.to_f32());
    let body = a.session.server.sim.player(id).pos;
    assert_eq!(
        a.session.server.view(id).unwrap().centre(),
        Some(chunk(body))
    );

    a.spectate(c);
    let fly = InputFrame {
        move_axes: [0.0, 0.0, 1.0],
        ..idle()
    };
    for _ in 0..300 {
        a.step(c, fly);
    }
    let camera = a
        .session
        .server
        .sim
        .player(id)
        .spectating()
        .expect("spectating")
        .pos;
    assert_ne!(
        chunk(camera),
        chunk(body),
        "the camera stayed in the body's chunk"
    );
    assert_eq!(
        a.session.server.view(id).unwrap().centre(),
        Some(chunk(camera)),
        "the spectator's interest is still around its body"
    );

    a.step(c, f4());
    assert!(!a.session.server.sim.player(id).is_spectating());
    assert_eq!(
        a.session.server.view(id).unwrap().centre(),
        Some(chunk(body)),
        "leaving Spectator did not re-centre interest on the body"
    );
}

/// The correction a client gets about itself carries the camera, so prediction
/// can reconcile against it.
#[test]
fn the_players_own_state_reports_the_camera() {
    let mut a = Arena::new();
    let c = a.join(0);
    a.spectate(c);
    a.step(
        c,
        InputFrame {
            move_axes: [0.0, 0.0, 1.0],
            ..idle()
        },
    );
    let state = a.clients[c]
        .heard
        .iter()
        .rev()
        .find_map(|e| match e {
            Effect::SelfState { state, .. } => Some(*state),
            _ => None,
        })
        .expect("a self state");
    assert_eq!(
        state.spectating,
        a.session.server.sim.player(a.id(c)).spectating().copied(),
        "the server's own-state correction left the camera out"
    );
    assert!(state.spectating.is_some());
}

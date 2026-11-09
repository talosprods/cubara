# Audit: crates/sim + crates/server (read-only, branch gui-test)

Coverage: wire.rs, net.rs, headless.rs, lib.rs (server), predict.rs, join.rs, view.rs, coordinator.rs, sim lib/physics/player/inventory/crafting/save, entity (partly). Not read in full: others.rs, shard.rs, hash.rs, rng.rs, entity pickup logic, tests/. Nothing was built or run.

## Findings

### 1. Dropped connections are never closed: thread and fd leak (HIGH, CONFIRMED)
net.rs:133-211, headless.rs admit_pending (silent_connections), accept_new_clients (MAX_PENDING refusal).
`Link` has no Drop/shutdown. The reader thread holds a `try_clone` of the socket and blocks in `read()`. When the server drops a Link (never said Hello within 600 ticks, pending queue full, `seat` returns early with no spawn), the socket stays open and the reader thread lives until the peer hangs up.
Scenario: attacker opens N TCP connections and sends nothing. Each is "dropped" after 10 s, but each keeps a thread and an fd. The Acceptor has no cap, so threads/fds grow without bound.
Fix: shut the stream down on drop, and cap total connections.

### 2. Client that never reads: unbounded server memory (HIGH, CONFIRMED)
net.rs:149,194-204. The outbound channel is an unbounded mpsc. The writer thread blocks in `write_all` once TCP buffers fill, and `Link::send` keeps succeeding, so `closed` is never set.
Scenario: connect, send Hello, stop reading. The server queues Tick + SelfState + effects every tick forever, and the player is never dropped (`is_closed` never true). The Welcome handshake of all edits within radius 64 is queued too.
Fix: bounded queue, drop the client on backlog, write timeout.

### 3. Client-sent `toggle_fly` grants free-fly in survival (MED/HIGH, CONFIRMED)
sim/src/lib.rs:259. `if input.toggle_fly { free_fly = !free_fly }` is not gated on creative; `sanitized()` leaves the bool alone and `ClientMessage::Input` passes it through.
Scenario: a survival client sends toggle_fly=true. It flies, and free-fly zeroes `fall_distance`, so no fall damage. tests/untrusted.rs:23 mentions this, so it may be a known gap.
Related: `SetCreative(bool)` (lib.rs:1415) and `/give`, `/tp`, `/gamemode` (lib.rs:1444+) accept any client with no permission check; the creative loadout is granted on request.

### 4. `/tp` with huge or inf coordinates (MED, PLAUSIBLE)
lib.rs:1456 and Fixed::from_f32 (fixed.rs:77). `/tp 1e30 0 0` parses as f32; `* 65536 as i64` saturates to i64::MAX and the position is set unchecked.
Debug builds: the next `physics::step` overflows (`feet.x + HALF_WIDTH`) and panics the server tick.
Release: wraps; `floor_block` truncates to i32, so `move_axis` can iterate a ~2^31-wide `u_lo..=u_hi`, hanging the tick.
Fix: clamp or reject coordinates beyond about +-3e7 blocks.

### 5. Disconnect discards the player: inventory loss and ghost players (HIGH functionally, CONFIRMED by code; probably a known pre-2.14 gap)
headless.rs drop_client (`sim.leave`), seat (always `sim.join(new Player)`), save.rs plan_players (its `remove` list deletes `players/<id>.ron` for ids absent from the sim).
Scenario: a client collects items and disconnects. The Player (inventory, grid, held) is dropped and its save file deleted at the next save. Reconnecting gives a fresh empty player.
On a server restart that loads a save, previously saved players re-enter as ghosts nobody drives, visible to others and in the hash.
Also `centre_player()` (lib.rs:616) picks the lowest-id player (a ghost or the first client). `tick_furnaces` and `hash_region` therefore simulate and hash only around that one player. A second player's furnaces next to them stay dormant until catch-up, and `Server::hash` covers the wrong region.

### 6. Server applies only the last Input per tick, and default input on ticks with none (MED, CONFIRMED by code; effect on feel PLAUSIBLE)
headless.rs collect_input: `inputs.set(who, frame)` overwrites while `last_seq` takes the max seq of all.
A: two inputs (seq 5, 6) arrive in one tick. Only 6 is simulated, but 5 and 6 are acked. The client predicted two steps, so it snaps on reconcile (missing input 5's look_delta and movement) on every jitter burst.
B: a tick with no input still steps the player with `InputFrame::default()` (gravity, velocity zeroed). The client never predicted that step.
Fix: queue inputs per player and consume one per tick.

### 7. Ack ambiguity at seq 0 (LOW, CONFIRMED)
headless.rs flush_effects uses `last_seq.get().unwrap_or(0)`; predict.rs reconcile pops pending with `s <= seq`. Before the server applied input 0, `SelfState{seq:0}` makes the client discard its pending input 0 without replay.

### 8. Furnace slots drop item durability: repair exploit (MED, CONFIRMED)
lib.rs click_furnace_as (~1928-1946) stores `(item.id, count)`; `Furnace` slots carry no ItemState, so the worn tool comes back as `items.new_stack(id, c)` with full durability.
Scenario: put a 1-durability pickaxe in the furnace input, click again to take it back: a fresh pickaxe. Same for contents spilled in `break_at_as`.
save.rs `stack_of` (~214) also rebuilds with `ItemState::None`.

### 9. `ClickSlot{Inventory(i >= 36)}` destroys the held stack (LOW, self-inflicted, CONFIRMED)
crafting.rs click, `SlotRef::Inventory(i)` is unbounded. `inventory.slot(i)` is None, `exchange` moves held into the "slot", `set_slot` ignores the index, so the held stack is gone. The wire sends `u32 as usize` unvalidated. Server should reject out-of-range indices.

### 10. Stack-limit mismatch silently deletes items on load or exchange (LOW/MED, PLAUSIBLE)
save.rs from_saved and crafting `rebuild` use `ItemStack::new(..., max_stack).ok()` (rejects count > max). If an asset lowers max_stack (64 to 16), every saved stack above 16 vanishes on load with no log. Unknown items are logged only at debug.

### 11. `put_str` truncates length to u16; decode ignores trailing bytes (LOW, CONFIRMED)
wire.rs:200 `bytes.len() as u16`; `ClientMessage::decode` / `ServerMessage::decode` never check the cursor reached the end. A 64 KB `/unknown...` command name is echoed in `CommandReply`; the length wraps, the frame is misparsed, and the receiving client drops the connection. Affects only the sender. Frames with extra trailing bytes are accepted.

### 12. Reader reassembly O(n^2); MAX_FRAME 16 MB for client frames (LOW, CONFIRMED)
net.rs:182 `buf.drain(..used)` per message. Many connections trickling near-16 MB prefixes hold ~16 MB each (see 1: connections are uncapped). A ~64 KB cap on client-to-server frames would be safe.

### 13. Sim path uses float (LOW/MED, CONFIRMED; acknowledged in wire.rs)
physics.rs step: `horizontal_axes` Vec3, `wish.normalize()`, `component * WALK_SPEED as f32`. `apply_free_fly` uses normalize * SPEED * dt with `from_f32`. IEEE ops are likely stable cross-platform, but it is formally a Rule 1 exception. No HashMap iteration reaches sim results (the only HashMap is the save.rs `remap` lookup). The wall clock appears only in clock.rs (Pacer).

### 14. Minor
- `Sim::keep` / `load_world` / `Entities::restore_item`: `id + 1` overflows on a u64::MAX id from a hand-edited save.
- `fall_distance` is not saved.
- `Server::load_from` does not clear `mining`, `last_pose`, `last_items`: stale mining progress, and a possibly suppressed `SelfItems` resend.
- `admit_pending` discards Input/Act messages that arrived in the same poll as Hello.
- `seat` returning early (no spawn) drops an accepted link silently.

## Not found
No panics from malformed wire bytes: Cursor is checked, tags are validated, and `get_stacks` / Effects loops are bounded by buffer length. Inventory::add and crafting merge arithmetic stay within u8. No duplication found in `exchange` or `take_result`; `/give` terminates.
Unread, not cleared: others.rs, shard.rs, hash.rs, entity pickup.

## Tests that cannot fail
Not audited in depth. The wire round-trip tests are sound. I saw no test that would catch findings 1, 2, 6 or 8, so those paths are unpinned.

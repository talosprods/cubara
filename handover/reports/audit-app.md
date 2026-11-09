# Audit: crates/app/src (branch gui-test), read-only

Not read line by line: bench.rs, flight.rs, caps.rs, most of the game.rs tests. Streaming.rs, far_streaming.rs, main.rs, settings.rs, paths.rs, capture.rs and the screen/save parts of game.rs were read in full or traced.

## Findings

### 1. A joined client panics on close or F5 (HIGH, CONFIRMED)
- main.rs:508 (CloseRequested) and main.rs:588 (F5) call `Game::save()` unconditionally.
- game.rs:871 `save()` calls `self.server()`, which is `.expect("this client is not hosting a world")` (game.rs:591-597).
- `join_over` sets `self.host = None` (game.rs ~548).
- Scenario: `cubara --connect host:25650`, then close the window or press F5. The game panics and `event_loop.exit()` is never reached.
- Fix: guard with `if self.game.hosting()`. `cracking()` already does this kind of check.

### 2. Key releases are swallowed while the console or options screen is open (MED, CONFIRMED)
- main.rs:402-404 (console_key) and main.rs:366-368 (options_key) both return on `!pressed`.
- The sim keeps ticking, and `open_console` / `toggle_pause` do not clear forward/back/left/right/up/down.
- Scenario: hold W, tap `/`, release W while typing, press Enter. `Game.forward` stays true and the player walks forward until W is pressed and released again.
- Same for Esc, then O (options), then releasing a movement key.
- Winit's synthetic releases on focus loss go through the same dispatch, so alt-tab while the console is open leaves keys stuck too.
- Fix: forward releases to `game.key_input` in all branches, or clear held keys when a screen opens.

### 3. Mining sticks when the mouse button is released with a screen open (MED, CONFIRMED)
- main.rs:699: `set_breaking(false)` is only sent `if self.cursor_captured`.
- Opening the inventory or pause menu un-captures the cursor via `follow_screen`.
- Scenario: hold RMB to mine, press E or Esc, release RMB. The release is ignored and `Game.breaking` stays true.
- `toggle_inventory` and `toggle_pause` do not clear it (only `new_world_in` does).
- `advance` keeps sending `breaking: true` every tick, so the block under the crosshair is dug while the menu is open, and mining continues after closing until the next RMB click.
- Fix: always send the release, or clear `breaking` when a screen opens.

### 4. The tune child can overwrite the player's choice (MED, CONFIRMED)
- main.rs:490 starts the child automatically on first run. `options_key` (main.rs:383-389) then saves the player's pick with `by_hand: true`.
- The child (main.rs:975-983) later writes `Settings { by_hand: false, tuned: Some(..) }` over it, and `poll_tuning` (main.rs:319-322) loads and applies it.
- Scenario: first launch, open options, press [4] while the benchmark runs. When the child exits, `by_hand` is gone and the quality is replaced.
- Pressing [B] on a by-hand setting also clears `by_hand`.
- `settings::save` (settings.rs:191-198) is a plain truncating `fs::write`, not atomic. A crash mid-write gives a file that does not read, which falls back to defaults.
- Fix: have the game ignore the child's result when `by_hand` is set, or have the child write only its answer.

### 5. The tune child is orphaned on exit (LOW-MED, CONFIRMED)
- `App.tuning: Option<Child>` (main.rs:99) is never killed or waited. CloseRequested just saves and exits (main.rs:503-510).
- A `--bench tune` process with its own wgpu device keeps running after the game is gone and writes settings later.
- Scenario: close the game within the benchmark's first minutes. Every relaunch before the first tune finishes spawns another child, so several run at once and the last writer wins.
- Fix: `kill()` and `wait()` on close and on drop.

### 6. `--world <dir>` can still hit the real settings file (MED, CONFIRMED by reading)
- paths.rs:36-39: `settings = world/../settings.ron`.
- Scenario: a GUI test with `--world saves/gui-test` gives `settings` = the owner's real `saves/settings.ron`. The game (options, `watch_frame` step-down) and the tune child (`child_args` passes `--world`, so it computes the same path) all write it.
- `--world .` resolves to the parent directory's `settings.ron`.
- The owner's real world is untouched, but the owner's tuned or by-hand setting is clobbered.
- Fix: make the settings path explicit (for example `--settings`), or keep it inside the world dir for non-standard worlds.
- Not a hit: a `--world` pointing at a temp dir, and the unit tests. `SavePaths::default()` is `for_tests()` under `cfg!(test)`, and the tests that call `new_world*` use `scratch_dir`.
- `for_tests()` is one fixed temp dir shared by all parallel tests. It cannot reach the real saves but is racy between tests.

### 7. `place_block` drops dirty chunks (MED when joined, CONFIRMED path; harmless in singleplayer)
- game.rs:1051-1054: `self.act(Place); self.sync();` discards the returned `Vec<ChunkCoord>`.
- The replica world is updated by `apply_effects`, but no re-mesh is requested.
- In-process nothing is in the link between host ticks, so it is harmless.
- Joined over a socket, messages arrive on a thread at any time. Any `Effects::Edit` that lands between the last frame's `sync` and a left click is applied to the world and its re-mesh is lost.
- Result: the mesh disagrees with collision and the replica until something else invalidates that chunk.
- Fix: return the dirty chunks, or stash them in a pending list that `advance` returns.

### 8. Save failure on exit is only logged (LOW, CONFIRMED)
- `save_to` returns a bool and logs `log::error!`. `Game::save` ignores the result and CloseRequested exits anyway.
- It also returns false silently if assets are not set yet.
- The owner sees nothing on screen when the world is not saved.

### 9. The simulation is not paused by a menu (LOW, design note)
- The pause menu, inventory and console all leave `advance` running: held keys keep moving the player, and mining and fall damage continue. This is what makes findings 2 and 3 dangerous. It may be intended.

### 10. Per-tick clone of the whole replica World while meshing is in flight (MED perf, CONFIRMED mechanism)
- game.rs `advance` calls `Arc::make_mut(&mut self.world)` every tick for `predict`.
- `MeshPool::request` (world crate mesh.rs:409) clones the `Arc<World>` into every queued or running job. Cancelled jobs also still run to completion (mesh.rs:435).
- While any job holds the Arc, `make_mut` deep-clones the World (edits BTreeMap, block entities, chunk states; world.rs:47) at 60 Hz.
- Cost grows with the number of player edits. After thousands of edits, walking around (constant meshing) copies the map every tick.
- Fix: do not take `&mut World` when nothing changes, or give the mesher a snapshot separate from the predicted world.

### 11. New World leaves stale old-world jobs queued (LOW, CONFIRMED)
- `forget_world` cancels in-flight tickets, and `poll` correctly drops their results. No old-world mesh is applied to the new world; the ticket check is sound.
- The queued jobs still execute FIFO on the mesher threads, each holding the old `Arc<World>`. The new world's first meshes wait behind up to a full render distance of dead work.

## Checked, no defect found
- Channels: search `done` and mesh `result_rx` are drained every frame, and the job queue is bounded by the visible set. `FarStreaming::reset` bumps the generation and drops stale results. No join or deadlock on exit: workers are detached and exit on sender drop.
- `Focused(false)` releases the cursor, calls `set_breaking(false)` and clears `alt_held`. The click that recaptures is consumed before it reaches the world.
- Slash opens the console only when no screen is open.
- No test in this crate can reach `saves/world` through `Game::new()`. `hosted_elsewhere` uses a relative nonexistent dir.
- The tune child receives `--world` via `child_args`, and `main` parses paths before the `--bench` branch.
- Theoretical only: if the search thread died, `searching` would stay true and visibility updates would freeze silently.

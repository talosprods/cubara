# Test-quality audit (read-only, branch gui-test)

Scope: sampled, not exhaustive. Read in full: scripts/*.sh, the golden.rs harness, client_view, acting_player, prediction, sharding, coordinator, survival/determinism fixtures, join.rs, parts of worldgen/physics/game.rs. A scan for tests with 0-1 assertions located the golden-only tests. Nothing was run or mutated, so every "would pass" below comes from reading the code. Confirm with check-tests-can-fail.sh.

## Ranked findings

### 1. a_correction_wins_when_the_client_was_wrong - crates/server/tests/prediction.rs:342 (HIGH, certain)
The harness's `forward()` is move_axes z=+1 at yaw 0, which walks toward -Z (angle.rs horizontal_axes: forward = [sy, 0, -cy]; the free-fly test and acting_player.rs agree). The test builds its wall at `p.z + dz` for dz in 1..4, which is +Z, behind the walker. The player never meets the wall, the client never mispredicts, and `believed == authoritative` holds trivially. It passes with no collision at all, and with a reconcile that does nothing.
Fix: put the wall at `p.z - dz`; assert before settling that `h.mispredictions > 0` and that the final z stopped short of the wall.

### 2. Gate dormant test: a_dormant_chunk_ends_where_a_continuously_ticked_one_would - crates/app/src/game.rs:3647 (HIGH)
- Compares only `.output` and `.input`. `fuel`, `burning` and `progress` are never compared, so a catch-up that loses fuel or resets progress passes.
- Never asserts the chunk went dormant in the `slept` run (no chunk_states check; :3706 does that separately). If the 4000-block teleport did not sleep the chunk, both runs tick continuously and are equal by construction.
- No non-triviality guard: at total=50 or 199 both outputs may be empty.
Fix: `assert_eq!(continuous, slept)` on the whole Furnace; assert the dormant state mid-run; assert `continuous.output.is_some()` for total >= 201.

### 3. Gate script runs `cargo test <name>`: scripts/check-phase-gate.sh (every `run ... cargo test ... <name>` line) (HIGH)
A filter that matches 0 tests exits 0 ("running 0 tests"). Rename any gated test (the_fixture_reaches_a_known_hash..., a_cave_mouth_is_visible, two_clients_in_one_world..., etc.) and its criterion still prints PASS. The golden criterion passes three names in one command, so one rename goes unnoticed while the other two pass.
Fix: parse "N passed" per name (or use `--exact` and a per-name run) and FAIL on 0.

### 4. Golden and GPU tests SKIP-and-pass without a GPU - golden.rs:210 (assert_frame_golden), gpu_timestamps.rs:53, mesh_arena_integration.rs:80 and :164, offscreen.rs:48, app/src/bench.rs:1473 (HIGH)
`eprintln!("SKIP")` then return is green. The gate's "golden images" line and any CI row go green having compared zero images if the runner has no adapter. Nothing asserts that at least one golden ran. (ubuntu CI uses lavapipe but is not a required check.)
Fix: `CUBARA_REQUIRE_GPU=1` in CI and in check-phase-gate.sh; panic on SKIP when set.

### 5. Golden tolerance swallows the thing named in the title - golden.rs:122-183 (TOLERANCE 12, MAX_DIFFERING 0.5%) (HIGH)
0.5% of 960x540 is about 2,600 pixels, with 12 levels of per-channel slack. These tests rely on the golden alone (no other assertion):
- hearts_show_health_including_a_half :977: a half heart drawn full or empty changes far fewer pixels.
- no_crack_at_a_real_lod_boundary :543: a thin seam crack is a few hundred pixels, and the test never asserts "no crack".
- iron_ore_is_visible_in_a_cave_wall :501, the_selected_block_shows_an_outline :573, an_item_name_shows_beside_the_cursor :880, the_hotbar_shows_slots_counts_and_the_held_one :764 (counts are a few glyphs), the_inventory_screen... :816, the_furnace_screen... :1007, three_players_stand_in_front_of_the_camera :1299.
The file already hit this twice (crosshair :932, furnace gauges :1066 gained explicit pixel asserts) and did not apply it to the rest.
Fix: per-test region assertion (differing pixels inside the named rect, threshold 0), or a ~0.02% threshold for UI shots.

### 6. Spawn: default seed only, no cave check - game.rs:3901 the_game_does_not_start_by_killing_the_player, :3935 the_spawn_point_is_somewhere_survivable; worldgen.rs:1133 a_new_world_starts_in_the_hills_with_mountains_in_view; Server::world_spawn (server/src/lib.rs:564) (HIGH)
These do use the real start-up path (good), but assert only `on_ground` and full health, both true for a player standing in a cave pocket. All use the default seed, while New World and --world give other seeds. world_spawn takes the first solid from y=200, so a tree canopy would pass too.
Fix: loop over ~200 seeds; assert air from spawn.y up to highest_generated_y, the block below is not leaf/log, and spawn y equals surface_height + 2.

### 7. prediction_converges_on_what_the_server_later_says - prediction.rs:257 (MED)
The guard `believed().pos != FixedVec3::ZERO` is always true because the player starts on the ground at y>0, so "the player never moved" cannot fire. Equality also holds if the client just copies the server each tick (the :290 test is the real one).
Fix: record the start, assert at least N blocks of z displacement.

### 8. the_fixture_reaches_a_known_hash_regardless_of_worker_count - sim/tests/determinism.rs:239 (MED; a phase 1 and phase 2 gate line)
"Multi-threaded" varies only the worker count of the hash reducer (`WorldHash::compute(.., 1)` vs `(.., 6)`). The simulation ticks single-threaded in `replay_with_edits`, and the fixture uses `fixture_blocks()`, not the real registry. The gate label claims more than is tested.
Fix: rename the gate text to "hash reduction is order-independent", or add a threaded tick/meshing variant when one exists.

### 9. two_clients_in_one_world_leave_one_authoritative_state - server/tests/client_view.rs:57 (MED; gate line)
Runs the same script on two servers in one process and compares hashes. It passes if `break_at` is a no-op, if players are not in the hash, and if `hash_with_workers(6)` ignores its argument. It never asserts the two blocks became AIR. No client or wire is involved.
Fix: assert both blocks AIR; `assert_ne!` against a hash with the second player removed; make the second player act via `apply_as`.

### 10. the_join_handshake_carries_no_terrain - client_view.rs:174 (MED)
Structural by its own comment: `Effect` has no terrain variant. With a single edit, `handshake.len() == edit_count()` also passes when both are 1 for the wrong reason.
Fix: edit 3 blocks, one far away; assert exactly the in-view subset.

### 11. trees_do_not_grow_on_stone_or_in_caves - worldgen.rs:1304 (MED)
The title names caves; the body only sets `grows_on` to a block id nobody uses (99). The cave branch is never exercised, and the test passes if `trees_near` always returns empty (no positive control in the test).
Fix: use the real grows_on and assert each returned tree has solid grass below and air above, over a region with known cave columns.

### 12. Gate line "a player AABB never passes through a solid voxel" - sim/src/physics.rs:424 (MED)
Only vertical falls onto an infinite flat floor (`flat_floor`: y <= 0). No horizontal sprint into a wall, ceiling, corner or one-block-thick slab entered from below. Only the ledge test (:525) touches a wall, at walking speed.
Fix: sweep speeds x axes x a one-block wall/ceiling, asserting no overlap.

### 13. a_peer_that_conjures_an_item_is_caught - server/tests/coordinator.rs:114 (LOW-MED)
One lie in one of the three compared fields (output = 64 ingots). A lie in `blocks` or `entities`, or "one ingot too many", is never tried.
Fix: parametrize the lie over each field, including +1.

### 14. Registry fingerprint: join.rs tests (:120-230) cover `accept()` in isolation (LOW-MED)
No end-to-end negative: nothing drives a client with a different registry against a server. Making `Game::join_over` (game.rs:544) ignore the Err from `accept`, or the server send equal fake fingerprints, is caught only by positive-path side effects.
Fix: a two_processes test with a client whose assets differ by one item, expecting refusal.

### 15. sharding: a_region_simulated_elsewhere_lands_where_it_would_have_locally - server/tests/sharding.rs:96 (LOW)
The non-triviality guard covers only `block_entities`; the `entities` and `blocks` equalities may compare untouched data.
Fix: also assert `locally.entities != untouched.entities`, or state that nothing drops.

### 16. #[ignore]d tests - render/src/aggregate.rs:853 and :1031, world/src/far.rs:767 and :868
The aggregate ones are a measurement and a table generator (fine). far.rs:767 `print_error_per_config` and :868 `what_the_gate_eyes_select` have zero assertions; the latter reads as pinning what the gate eyes select, which the perf gate depends on.
Fix: turn it into an asserting test (node counts per eye in a range) and un-ignore it.

### 17. different_seeds_produce_different_terrain - worldgen.rs:1159 (LOW)
One column, seeds 1 and 2. It cannot detect a seed that affects only part of generation (caves, ores, trees).
Fix: compare chunk hashes over several seeds, per feature.

### 18. Gate-script greps (check-architecture.sh, check-single-render-path.sh) (LOW-MED, by design)
String greps that catch only the shapes already seen:
- Rule 1 misses HashMap/HashSet iteration order (the real determinism risk) and `UNIX_EPOCH`.
- Rule 2 misses `static X: AtomicU64`.
- Rule 5: `begin_render_pass` or `build_pipeline(` split across lines, direct `create_render_pipeline` calls, or a renamed duplicate of `encode_scene` all pass.
- Rule 4 checks Cargo.toml `^wgpu` only at line start (misses a renamed dependency).
- check-benchmark-history.sh checks footnote bookkeeping only, not whether the numbers are real.
Fix: ban HashMap/HashSet in sim crates, add Atomic* to Rule 2, and add a self-test fixture per rule that must trip it.

### 19. perf_gate - check-phase-gate.sh (LOW)
Parses `GATE: MET` (good, not the exit code), but nothing requires that all three eyes were evaluated, so a bench that evaluates fewer still reports MET.
Fix: require the eye count in the GATE line.

## Checked and fine
acting_player.rs (distant players, different items, asserts which one moved); survival_replay.rs (station register guard); bounded_catch_up_agrees_with_the_one_tick_reference (7 states x 14 counts); bytes_to_one_client_do_not_grow... (has the crowded > few guard); fingerprint.rs unit tests; handshake.rs.

## Systemic notes
- Nearly every fixture repositions players ("pair()", Harness). Only the two game.rs tests use the real start, and only with the default seed.
- A guard that cannot fire recurs (#7, #10, #15). A cheap sweep: grep tests for `!= FixedVec3::ZERO` and `!is_empty()` on data both sides share.

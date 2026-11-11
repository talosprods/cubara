# 388 PR A code review

Verdict: open after the listed fixes. No blocker.

## Blocker
None. The present mode is still reported: `renderer.present_mode()` reads `config.present_mode` (main.rs report block), which is the mode actually obtained. `present_mode_for(None, ..)` is the old `chosen_present_mode` logic (test covers it), and the `--present` fallback is warned about. Removing `chosen_present_mode` lost nothing.

## Major
1. Row 13 is unpinned (render.rs, `queue.present` + `Instant::now()`). It can be pinned without a GPU: extract a generic `fn present_timed<F>(frame, acquire, present: impl FnOnce(F)) -> Presented` and call it from `render`. Test it with a closure that sleeps 20 ms and assert `at - before >= 20 ms`. Moving the Instant before the call turns it red.
2. `the_walk_script_walks_and_jumps` (game.rs) never asserts a jump. Deleting the Space logic in `WalkScript::before` and `after` keeps it green. Assert max eye.y rises above the start (or record the Space key state).
3. The `Some`/`None` accounting in main.rs (skipped vs presents/acquires) and the `--window` without `--world` exit path have no test. The `refuse_owners_world` function is tested, but the call in `main` is not. At minimum extract the "does this arg set require --world" decision into a pure function and test it.

## Minor
- flight.rs ~1040-1050: the doc of `walk()` ("With `fly`, F4 first...") now sits on top of `WalkScript`, and `walk()` has no doc. Move it back.
- render.rs: the figure-pipeline doc ("Depth-tested...") sits above `present_mode_for`. That was already misplaced on main, but `Presented` now sits between it and its function. Fix it while you are there.
- main.rs rebuild block: `renderer.resize` is called directly and winit then delivers `Resized`, so the rebuild can run twice. The time logged and the "rebuild" gap may cover the second one. Verify with the log, or drop the direct call.
- The run ends only on a RedrawRequested after `seconds`. An occluded window can stop redrawing and hang. Add a wall-clock exit in `about_to_wait`, or document it.
- Neutrality when off: two extra `Instant::now()` calls per window frame and a returned `Option`. Negligible. Every caller (flight.rs x2, offscreen test) ignores the result, which is fine (not `#[must_use]`). `device_event` returns early only when the bench is on.

## Checked and fine
- Rule 5: one `RedrawRequested` -> `render`. `WalkScript` is shared with the headless `walk()`.
- Skipped, Outdated and offscreen frames return `None` and are not counted. The acquire is timed around `get_current_texture` only.
- Portability: `wgpu` and `winit` are already in the app's Cargo.toml, there are no macOS-only APIs, and `Occluded` and `Fullscreen` are cross-platform. I saw no unused-variable risk.
- `refuse_owners_world` compares both the world and the settings path through `same_path`.
- Rerun `check-tests-can-fail.sh` after the fixes.

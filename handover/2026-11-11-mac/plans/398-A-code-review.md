# Review: PR A of #398 (refactor/398-far-plan)

Verdict: merge after the listed fixes. No blocker.

## Correctness (verified by reading old `update` against `FarPlan::step` + `FarStreaming::update`)
- Retain-before-insert is safe. Every key in `arrive` is in `draw`, `to_keep(wanted, draw)` keeps all of `draw`,
  and `arrive` keys are ready wanted patches, which are in `heights`. So `patch(key)` cannot miss. Same for a failed
  insert (slot full): the key stays out of `slots` and is offered again next step, exactly as before.
- Order of renderer calls is unchanged: jobs sent, then removes, then inserts. `set_far_params` still runs before
  the early return. The Off path matches (slots drained, `selected_at` set, `arrived` cleared, `in_flight` and `heights` untouched).
- `clear()` mirrors old `reset` (hole and screen_height are not cleared, as before; `selected_at = None` forces a pass).
  In-flight jobs of the old generation are dropped by the generation check in `update`. OK.
- Start-up: `screen_height` 0 != real height, but `selected_at` is None anyway. Teleport: the hole moves, so it reselects.

## Major
1. Behaviour change not covered by "no behaviour change": `screen_moved` (far_streaming.rs, `step`) means a window
   resize now reselects. Main did not. During a drag-resize that is a select per frame. State it in the PR body
   (plan Step 2 says "new, harmless"), or drop it from A and add it in B.
2. Tests cannot see the "ground never uncovered" rule. Replace `draw_set` with `wanted.filter(ready)` and
   nothing goes red: `Sim::frame` only asserts leave/arrive consistency. Add a coverage assertion in `fly`
   (every ready-wanted ancestor or descendant of the eye's selection is covered by `shown`, or at least no ground
   gap per frame vs `exact`'s footprint), and write down that it was broken on purpose and went red.
   Likewise `swap`/`to_keep` are only covered indirectly (settle would panic on oscillation; confirm by injection).

## Minor
3. `today_the_live_selection_trails_the_exact_one` pins a lag bug and asserts `behind > 0`. It is fine as a
   characterisation test, but its doc says PR B replaces it. Make sure B deletes it, otherwise B fails.
4. Stale cross-reference: the cost-report doc cites "#398 §6.2", which is a section of a scratchpad plan, not the repo.
   Cite the issue comment or `docs/`. The `service_frames = 2` placeholder is documented and only feeds the ignored
   report, so it can't affect the checks. OK, but file a note on the issue so the placeholder is replaced.
5. `a_new_hole_screen_height_or_reselect...` never tests `arrived` alone triggering a pass (only indirectly via settle).
   `the_eye_moving_past...` checks the 0.5x/2x distances only. Fine.
6. `Sim.exact` re-implements the `far::build` loop. It is test-only, but note it as a second copy (rule 5 in spirit); a
   comment points at it, so acceptable.
7. Pin test (world/src/far.rs): hash and key count over 12 cases, recorded constants. Constants cannot be proven
   "recorded on main" from the diff alone; the commit order / PR body should show the test passing on main's `select`
   (run it against origin/main in a throwaway worktree). Pin does catch duplicate push or `>`/`>=` changes.
8. No dead code found; `selects` is cfg(test) only. No globals (memo is per-Sim), dependencies still one way, FarPlan
   has no wgpu types (it sits in the app crate, rule 4 concerns data/sim crates only). OK.

## check-tests-can-fail.sh "nothing to check": the script's defect
`scripts/check-tests-can-fail.sh` sets `test_cutoff` to the first line matching `^#\[cfg(test)\]` in the file, assuming
that is `mod tests`. far_streaming.rs has `#[cfg(test)] use ChunkCoord;` at line 15 (also on main) and
`mod tests` at line 553, so every added line (all >15) is treated as test code. The file is legitimate Rust; the
fix is in the script: use the first `#[cfg(test)]` immediately followed by `mod `, e.g.
`awk '/^#\[cfg\(test\)\]/{p=NR;next} p&&/^(pub\(crate\) )?mod /{print p;exit} {p=0}'`.
Add a selftest fixture for it. Until then, PR A's mutation evidence is the hand-run 16 injections only (and the
non-test lines of `step` are the ones the script should have mutated).

## Merge as is / after fixes
Merge after fixing 1 and 2 (or stating 1 plainly in the PR body and adding the coverage assertion for 2).

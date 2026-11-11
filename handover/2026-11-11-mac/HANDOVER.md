# Handover: Mac lead session → Linux session, 2026-11-11 08:00 CET

From 03:00 to 08:00 the Mac session was the lead. From 08:00 it starts nothing new.
It finishes one item still in flight (#388 PR A, below) and then stops.

## Merged overnight

| PR | What | Evidence |
|---|---|---|
| #415 | M3 gate and 1080p flight at 0ab0cc3, after #379's split and #405 | BENCHMARKS.md ⁸⁹ |
| #416 | #398 PR A: the far terrain's streaming decisions in a GPU-free `FarPlan` | A/B on the M3 (⁹⁰); also fixes `check-tests-can-fail.sh` |
| #417 | `next-block.sh` with no argument uses the active phase | — |

Numbers behind them:
- **#415 (⁸⁹):** quiet run 1,155 / 1,135 / 3,461 FPS, `MET at Medium`. That is +0.6% / +0.4% / +1.7% against 874c771, inside the run-to-run ramp.
- **#416 (⁹⁰):** CPU/frame and STALE equal to main. Its one behaviour change, a window-resize reselect, is stated in the PR.
- **#416's script fix:** `check-tests-can-fail.sh` cut the test code at the first `#[cfg(test)]` rather than the `#[cfg(test)] mod`, so on files with an early `#[cfg(test)] use` it mutated nothing.

#412 is closed. Its done-when evidence, three green ubuntu `tree_culling` runs, is in a comment.

## In flight at 08:00

- **#388 PR A, branch `feat/388-window-walk`.** This is the windowed present-to-present instrument, `--bench walk --window`, about 830 lines in two crates. A Sonnet code review found no blocker. A Sonnet agent is applying its fixes now:
  - pin the present instant with a GPU-free `present_timed`;
  - add the missing jump assertion;
  - test the frame accounting and the refusal without `--world`;
  - add a watchdog against an occluded-window hang under Fifo.

  When it reports, the Mac reviews the diff, opens the PR with `--auto --squash`, and stops. No windowed run was made. That is PR B, under CLAUDE.md's standing-permission checks.

## Plans: all in `plans/` on this branch

Each plan was written by one model, reviewed by a second, and revised. Read the plan and its review together.

| Issue | Plan | Decisions on the issue | State |
|---|---|---|---|
| #398 | `398.md`, `398-review.md`, `398-step1-results.md` | posted (plan; step 1 results and the lead's reading) | PR A merged (#416). **PR B waits for the owner** (questions below). |
| #380 | `380.md`, `380-review.md`, `380-amendment.md` | posted (amendment; lead decisions C–F) | Ready for PR 1 (the decision-8 table). Nothing built. |
| #394 | `394.md` (revision 2), `394-review.md` | **not posted.** Read it before building; it was revised after an Opus review. | Seven PRs, the new PR 1b grows the arena and far slots first. Nothing built. |
| #388 | `388.md`, `388-review.md`, `388-A-code-review.md` | posted | PR A in flight. PR B is the M3 runs. PR C, a ROADMAP line, **needs the owner's yes**. |
| #410 | none | — | The planner was stopped for budget at 01:25. No plan. |

## Findings

- **Background agents distort M3 benchmarks.** Four read-only agents streaming in the Claude app cost 11–14% FPS and doubled p99 on the M3, while no process exceeded the 25%-CPU quiet check (⁸⁹). From then on the Mac session stopped every agent during a run. The standing-permission check "no process holding a core" does not catch this. Whether to tighten CLAUDE.md is for the owner, since he approved that section.
- **#398 step 1 (diagnosis):**
  - Rules 1 and 2 said stop. The mechanism is confirmed: reselecting every block takes STALE from 11.3 to 1.7, and exact selection leaves 0% "neither".
  - The stale pixels hit at 816–2,604 blocks, not only at 1.5–1.9 km.
  - The `sync` residue is one frame of voxel late-swap or loading.
- **Low-flight HOLES is intermittent on main at t ≈ 10.7 s,** eye ≈ [253, 83, 82]. Main's runs at 1080p: 874c771 7, 6; 0ab0cc3 0, 6; 16d153f 0, 0, 0, 5, 0. This is a timing race, recorded on #318, and nobody owns it.
- **Far slots at 2560x1664 (the M3 panel):** 2,320 of 4,096. At 4K, 3,729 (91%). Overflow shows as holes plus a warning, with no panic.
- **The Mac's own power settings:** `pmset sleep` is 1 minute, so unattended work needs a `caffeinate -i -m -s` hold and the lid open. A `caffeinate` runs until 08:30.

## Owner questions (carried, not decided)

1. **#398:**
   - Is 0 stale kept frames at 1080p the bar on the M3? On this evidence the fix reaches about 0–2 frames, not provably 0.
   - With exact selection, a player hovering on a threshold may see a patch swap back and forth. Hysteresis would stop that but conflicts with STALE = 0.
2. **#380 against STALE:** R6's rise shows sky where the finished frame has land. That is STALE by definition, so R6 and gate item 0's STALE term may conflict by design.
3. **#394:** should a FarQuality step also scale the ring constants? It changes what `MET at Medium` means.
4. **#388:** the ROADMAP line in PR C.
5. **#386:** whether block 3.0 closes. Nothing was written into ROADMAP.md.

## Usage

- The 5-hour window reset at 06:20; it stood at 16% at 08:01.
- The weekly budget reset at 03:00 and is at 3%.
- Before 03:00 the shared window was the binding limit. Both machines' agents emptied it to about 89%.

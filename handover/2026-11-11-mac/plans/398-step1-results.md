# #398 step 1 results (diagnosis, Mac M3, 2026-11-11)

Measured commit: 572113d (main after #406), temporary edits to `crates/app/src/far_streaming.rs` only (logging; half: RESELECT_BLOCKS 16; tiny: 1; sync: reselect when eye != selected_at, select+generate loop in place, no pool). File restored after each build; worktree clean.
Scripts/logs: `398-patch.py`, `398-run.sh`, `398-extract.py`, `398-logs/`, `398-frames/{base,sync}`, binaries `398-bin/`.

## Conditions
- AC Power in all 28 pmset snapshots (before+after each of 14 runs); `pmset -g therm`: no thermal / performance / CPU-power warning recorded (the platform reports none; not proof of no throttling).
- Last build finished 02:24:03; first run started 02:29:25 (5 min 22 s after, not the plan's 6 minutes; the task allowed >= 5). Runs 02:29-03:07, ~2 min apart, alternated base/half/tiny/sync x3, then the two uncounted `--frames` runs.
- Quiet check at 02:29:17: nothing above 12% (launchd 11.6%). Deviation: the snapshot at the start of base-1 (02:29:25) showed XprotectService at 34.5% CPU (presumably scanning the new binaries). Every other run's top process was <= 4.2%. base-1 is therefore not fully quiet; its STALE count (11) is in line with base-2/3.
- Built in the worktree with CARGO_TARGET_DIR=target (not the plan's per-variant target dirs); binaries run from the worktree root, `RUST_LOG=info ... --bench flight --eye 8,120,8 --look 1,-0.15,0.3 --seconds 20`, default 1920x1080.
- Latency logging: "frames" = `update` calls between request and arrival (includes the settled pass's calls).

## Table: STALE kept frames (worst %; shares late swap / loading node / neither)
| variant | run 1 | run 2 | run 3 | mean kept |
|---|---|---|---|---|
| base | 11 (0.007%; 0/0/100) | 12 (0.008%; 0/0/100) | 11 (0.008%; 0/0/100) | 11.33 |
| half | 11 (0.010%; 6/18/77) | 11 (0.010%; 6/17/77) | 11 (0.010%; 6/17/77) | 11.00 |
| tiny | 1 (0.000%; 0/0/100) | 3 (0.010%; 14/44/42) | 1 (0.007%; 0/0/100) | 1.67 |
| sync | 1 (0.010%; 26/74/0) | 1 (0.010%; 25/75/0) | 1 (0.010%; 24/76/0) | 1.00 |
Uncounted `--frames`: base 11 (0.008%; 0/0/100), sync 0.
Sync work p99 17.6 ms, over 16.7 ms: 15 frames, over 33.3: 3 (sync-1; work times meaningless as the plan says, but the stalled main thread is the likely cause of sync's late-swap/loading shares).

## "Neither" hit ranges (blocks away; per kept frame: t, px, range)
- base-1: 1.0 (3px) 1300-1304; 1.25 (1) 1296; 1.75 (26) 2372-2604; 2.0 (25) 2368-2600; 2.25 (28) 2360-2592; 2.5 (28) 2356-2588; 4.75 (5) 1860-2476; 5.0 (6) 1856-2472; 5.25 (6) 1848-2464; 10.75 (155) 820-1156; 13.5 (82) 1080-1268. Total 365 px.
- base-2: same, plus 9.26 (145) 1952-2576; 10.76 (156) 816-1232; 13.51 (85) 1080-1268. Total 515 px.
- base-3: as base-1 with 10.76 (157) 816-1232, 13.51 (83) 1080-1268. Total 369 px.
- half: 1.0, 1.25, 1.75, 2.0, 4.75 as base; 9.26 (~146) 1952-2576; 9.51 (~154) 2264-2572; 10.76 (155) 816-1232; 13.51 (~85) 1080-1268; 13.75 (~85) 1076-1264. ~683 px.
- tiny-1: 4.76 (5) 1860-2476. tiny-2: 4.75 (5) 1860-2476; 9.26 (145) 1952-2576. tiny-3: 9.26 (145) 1952-2576.
- sync: none, all three runs.
All "neither" pixels are classified by flight as "ground beyond the voxel nodes' reach (far terrain)", 0 at nothing. The log gives a min..max per frame, not per pixel; no frame straddles 1400.
Note: this differs from the issue's 1,530-1,900 blocks: the two biggest base frames (t = 10.75 and 13.5, 82-157 px each) hit 816-1268 blocks.

## Request->arrival latency (FARARR)
All arrivals (2830/2832 per run; dominated by the load burst at start, where the queue is thousands deep): p50 620-647 ms (192-239 frames), p99 1077-1161 ms (349-411 frames), max up to 1198 ms.
Requests made with queue length < 4 (steady state, ~200-260 per run):
| run | p50 ms | p99 ms | p50 frames | p99 frames |
|---|---|---|---|---|
| base 1/2/3/frames | 5.8/5.8/5.9/5.7 | 56.8/18.1/18.2/18.9 | 1 | 3 |
| half 1/2/3 | 6.5/7.3/6.2 | 22.7/21.4/18.1 | 1 | 4/4/3 |
| tiny 1/2/3 | 9.1/8.5/8.9 | 17.8/20.2/18.8 | 1 | 3 |

## Median `generate` time (arrival latency, queue < 4)
base 5.8/5.7/5.9 ms; half 6.5/7.3/6.2; tiny 9.5/8.6/8.9 (more concurrent work in the tiny runs). So ~6-9 ms, i.e. <= 1 frame at 60 Hz (the plan's service_frames: 1).

## Reselection triggers (all selections per run, incl. settled pass and startup): moved / arrival / hole
base: 1066/1016/12, 1070/1023/12, 1063/1040/12 (frames run 1080/989/12). half: 1095/1068/12, 1081/1054/12, 1092/1032/11. tiny: 1429/1046/10, 1434/1054/11, 1452/1027/10. sync: 2227/0/4, 2242/0/4, 2209/0/4 (sync-frames 2224/0/4).
Blocks moved per reselection, base p50/p99/max ~0.35/2.4-2.8/32.4: most selections are arrival-triggered, only a few at the full 32-block step.

## Residue check (~400-700 blocks)
No "neither" hits below 816 blocks in any of the 14 runs (base, half, tiny, sync). So no residue of the kind the issue describes was seen in the far-terrain term, and sync (zero lag) shows no "neither" pixels at all. In sync, the 1 surviving stale kept frame per run is under late swaps (24-26%) or loading nodes (74-76%), i.e. voxel-node loading, not far terrain (sync-frames: 0). I did not inspect the dump PNGs.

## Rules (literal)
1. "Premise. In the base runs, >= 80% of the 'neither' pixels hit beyond 1,400 blocks. Otherwise stop and report." Base pixels in frames hitting >= 1400: 124/365 (34%), 270/515 (52%), 125/369 (34%). The remaining pixels hit 816-1268 (frames at t = 9.26-13.5 s). **Not met -> STOP.**
2. "Ceiling. Every sync run has STALE = 0. Any sync run > 0: stop and report with the sync dumps." sync = 1, 1, 1. **Not met -> STOP** (sync dumps: `398-frames/sync`, from the uncounted run, which had 0). Caveat for the owner: sync's 1 frame is 0% "neither"; it is voxel late-swap/loading, coinciding with sync's stalled main thread.
3. "A is warranted if mean(base) - mean(tiny) >= 2 kept frames." 11.33 - 1.67 = 9.67 -> holds.
4. "B is warranted if any tiny run is > 0, or the logged p99 request->arrival exceeds one frame (16.7 ms)." All tiny runs > 0 (1, 3, 1); steady-state p99 17.8-20.2 ms > 16.7 ms (all-arrival p99 ~1.1 s) -> holds. Sizing: if the all-arrival p99 (1.08-1.14 s) is the figure, `1024*m/(1+m) >= 2*p99*24*2` needs ~105-110 blocks, i.e. m = 1/8 (covers 1.18 s), not 1/16 (0.63 s). If the steady-state p99 (~0.02 s) is, 1/16 covers it many times over. The rule does not say which; the plan's own wording ("logged p99") is the all-arrival one. Not decided here.
5. "If neither 3 nor 4 holds while base > 0, stop." Does not apply.
6/7. Do not apply (3 and 4 both hold).

## Conclusion
**Stop and report** (rules 1 and 2 both fail as written). Had they passed: rules 3 and 4 both hold -> build A and B (with prefetch).
Facts for the owner/lead: tiny nearly eliminates the stale frames (11.3 -> 1.7), half does nothing (11.0), confirming reselection distance; the remaining tiny pixels hit 1860-2576 blocks (latency tail, consistent with B). But rule 1's premise misses because ~half the base pixels are in two frames at 816-1268 blocks (not 1400+), and rule 2's ceiling misses by one frame of voxel-loading origin under sync.

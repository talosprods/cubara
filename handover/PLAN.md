# Cubara — autonomous work state

Source of truth for autonomous sessions (the owner said on 2026-11-09: "werk
zelfstandig door, test grondig, ook in de echte GUI; gebruik Sonnet/Haiku voor
implementatie en lezen, Opus alleen voor plannen"). Local only, not committed.

**Read first, every session:** `CLAUDE.md`, `ARCHITECTURE.md`, then this file.
Always `git fetch` and compare with `origin/main` before touching anything.

## Rules for these sessions

- Model economy: Opus plans and reviews; implementation goes to a `general-purpose`
  subagent with `model: "sonnet"`; bulk reading/searching to `Explore` with
  `model: "haiku"`. Keep the main context small.
- Usage: check `mcp__ccd_session_mgmt__get_usage`. At >= 85% of the 5-hour
  window, finish the current step, update this file, and stop. The scheduled
  task `cubara-autonomous-resume` restarts the work after the reset.
- GUI: CLAUDE.md forbids *synthetic OS input* as verification. Looking is fine:
  start the real game (never against the owner's `saves/world`), take
  screenshots, compare with the headless/golden output. Never close a Cubara
  window you did not start yourself (the owner may be playing).
- Never touch `saves/world` or `saves/settings.ron` from tests or tools.
- Gameplay/content decisions are the owner's: write them under "Questions" and
  keep working on something else.

## Owner decisions, 2026-11-09 (in this session, his own words)

- Warm-up: "pas gerust de tests aan als je merkt dat wat je meet niet
  representatief is. Graag zelfs ... zo vroeg mogelijk in het proces" -> PR #298.
- GUI: "Ook besturen" -> computer-use control of the game window allowed (test
  world only). CLAUDE.md text parked in `.claude/autonomous/claude-md-gui.patch`; lands with the
  `--world <dir>` flag PR (the game has no way to use a separate world yet:
  `world_dir()` = repo_root/saves/world, compile-time).
- Yardstick for engineering choices: performance, efficiency, clean extensible code.

## Queue (in order)

- [x] 1. MERGED 2026-11-09. PR #298 bench warm-up (>=200 frames AND >=1 s). Measured + ⁷⁵ written
      (ground 3,443->3,650, hill 3,426->3,617, flight 4,892->5,427 FPS; CPU/frame
      ~halves). Remaining: mutation run, PR body, un-draft, CI, merge.
- [~] 2. Review fixes pushed (454af24), main merged in, Windows rows added, auto-merge ON. PR #297 flight streaming. Windows flight bench run1: low 4.18% -> 0.21%,
      down 17.49% -> 0.42% (confirms M3). Review found a real bug: uploads.rs
      `desired` set lets a stale Insert queued before a Remove resurrect after a
      re-Insert (New World shows old world briefly) -> per-id generation fix;
      also drain_uploads ignores arena-full `false`; wrong assert message in
      visibility test. Fix on the branch, then merge. Add Windows rows.
- [~] 2b. PR #300 open, auto-merge ON. `--world <dir>` for the game window (+ settings beside it), CLAUDE.md
      GUI-testing text (patch: `.claude/autonomous/claude-md-gui.patch`).
- [~] 3. F5 haze — issue #299. Implemented on `feat/far-haze` (agent), golden
      `far_terrain_to_the_horizon` regenerated (7.91% px, expected). Bench (Windows,
      3 alternating runs vs main+warmup): ground -1%, hill +0.5%, flight +2.6% = noise.
      Images in `f5-images/`. Gemini + own look: 1/40000,H=2000 too weak at ground
      (mountains a few km away dark). Asked agent to render density 1/20000, H=600
      (`after-d20k-h600-*.png`) — thinner, lower layer: more ground haze AND farther
      aerial view. Also asked: 2nd-order series below |Δ|<1e-2, rebase, push.
      Remaining: pick values, regenerate golden, BENCHMARKS row ⁷⁶, proposal §6 note,
      comparison image `f5-images/far-view-f5-haze.jpg` (rebuild with the final
      values, commit as `docs/img/far-view-f5-haze.jpg`), PR, merge.
- [ ] 4. F6 — the 160 m – 1 km band. Prototypes exist:
      `origin/proto/band-b-block-tops`, `origin/proto/band-c-far-from-512m`.
      Judge from images + bench, write the decision into the proposal.
- [~] 4b. GUI play test DONE once (2026-11-09) with a test build: `.claude/autonomous/bin/cubara-gui.exe`
      + Start-menu shortcut "Cubara GUI-test" (computer-use grants by exe name `cubara-gui.exe`;
      screen frame is 1568x441, overlay text too small to read -> read saves instead).
      Findings: (1) New World spawn in a cave (seed 1881928537856268101, feet y=12):
      world_spawn raycast falls through cave mouth -> agent fixing on `fix/spawn-on-the-surface`.
      (2) Owner sees flicker bottom-left in the cave view (not caught in screenshots; node
      counts stable) -> asked him what it looks like. (3) Owner: LOD layers clearly visible
      -> F6 (agent rendering variants A/B/C/D into `f6-images/`, branches proto/f6-*).
      (4) Pause menu text tiny in top-left on a big window (note for report).
      Old: GUI play test (after #297 + #300 merge): release build, `--world $TEMP/cubara-gui-test/world`, computer-use: walk, fly, New World, options; turn findings into tests.
- [ ] 5. Gate run (`./scripts/check-phase-gate.sh`), BENCHMARKS rows, report.

## Reports to triage (2026-11-09, owner: "ga all out met tokens, laat agents designs bedenken en oude code controleren")

In `.claude/autonomous/reports/`: audit-voxel-world.md, audit-sim-server.md, audit-render.md,
audit-app.md, audit-tests.md, audit-docs.md (code audits, read-only), design-lod-transitions.md,
design-perf-roadmap.md, design-ui-scale.md, phase3-options-memo.md (Opus design studies).
Triage: CONFIRMED high/med findings -> GitHub issue per ISSUE_STANDARD (or a fix PR if small and
clearly a bug); designs -> pick the next blocks; the phase-3 memo is for the OWNER (not committed,
nothing in it is decided) -> show him the 5 questions.
Also running: flicker detector (branch bench/static-flicker, frames in `flicker/`), F6 variants
(`f6-images/`, branches proto/f6-B/C/D), haze variant d20k-h600, spawn fix PR #301 (auto-merge).

## Fix PRs in flight (2026-11-09 evening, all auto-merge)

- feat/far-haze (F5 final values 1/20000, H=600) - haze agent finishing PR
- fix/lighting-rebuild-once (render audit #1)
- docs/drift-2026-11-09 (docs audit)
- fix/app-audit-2026-11-09 (joined-client save panic, stuck keys, stuck mining, tune child, settings path)
- fix/server-net-hardening (link shutdown, connection cap, bounded outbound, strict decode, /tp clamp)
- fix/sim-audit-2026-11-09 (furnace durability, input acks, max_stack)
- fix/world-persistence-audit (broken trees respawn after reload!, corrupt save panic)
- test/tests-that-can-fail (audit-tests top 8, gate script 0-tests)
- bench/static-flicker (no PR; flicker hunt, lead: far.wgsl masked_top NaN with eye below surface)
Merged/in flight since: #303 lighting rebuild, #304 app fixes (auto-merge). Flicker hunt: NOT
reproduced (bench/static-flicker, `--bench still`): no steady-state flicker at 3136x860 or 1080p;
only loading holes (~12 frames in the cave after New World) -> most likely what the owner saw.
Design reports saved: design-ui-scale.md (7-PR plan, PR A-F neutral), design-perf-roadmap.md
(M3 is GPU-bound on far triangles ~1 ns each; order: M3 measurability + census, haze-aware far
error bound F-a, skirt mask F-e, floor by projected size F-b; F6(c) may COST on M3), and
phase3-options-memo.md (for the owner, 5 Dutch questions at the end; NOT a decision).
design-lod-transitions.md SAVED: real cause at 160/288 m = coarse cells are 2^L blocks TALL
(riser steps double on screen: 6.4->12.9 px at 160 m); trees stop at 160 m (level 0 only).
Plan: PR1 harness (agent on feat/lod-transition-harness: proposal doc + M1/M6 + T1 measurement),
then R1 block-height coarse cells (v=1 in rings 1-2) + ring-3 decision (v=4 or C+R5 far shading
fix) + R3a circular rings to pay for it; R2 trees beyond 160 m (after R1). Owner's calls listed in §7.
Still to do from audits: render #3 draw cap 16,384 vs 3 draws/node, render #2 camera-relative
(f32 at 1e5 blocks), app #7/#world #4 Arc::make_mut deep clone per tick (perf), world #3 LOD
neighbour remesh, frustum far plane NaN (render #5).

## Notes for sessions

- `check-benchmark-history.sh` fails locally in Git Bash on main too (prints `� �`):
  run it as `LC_ALL=C.UTF-8 ./scripts/check-benchmark-history.sh`. CI is fine.
- Agents share `CARGO_TARGET_DIR=target-agents` (excluded in .git/info/exclude);
  their worktrees live in `.claude/worktrees/`.
- Flight bench on Windows (#297), 2 runs each: low 4.18/4.45% -> 0.21/0.22% mean
  loading, worst 22-27% -> 3.2-3.3%, >1%: 48-49% -> 8%; down 17.5/18.5% ->
  0.42/0.44%, worst 97.4% -> 24.5%, >1%: 35-37% -> 2-3%. No "arena full" seen.

## Owner decisions, 2026-11-09 evening

- PLAY MODES (issue #305, agent building on feat/play-modes): Creative walks with collision
  (no flying), Survival same, Spectator = noclip camera that takes no part. F4 toggles normal <->
  Spectator "overal" (also Survival); leaving returns you "terug waar je was" (body frozen).
  Answers question 1 below (F4 in survival) -> it is allowed, by design.
- Owner bought Claude Max (2026-11-09). Still pace agents.

## Other sessions

- Linux HP laptop session "Cubara spel lokaliseren" (bridge:session_019m63KAQBEDS9UuLb92Htjp)
  reported in; asked it to: (1) Linux gate rows under new warm-up (footnote ⁷⁷), (2) Linux flight
  bench rows, (3) far-triangle census tool (perf study's first lever), (4) face-group order D1.
  ⁷⁶ reserved for F5 haze. Delivery of my message was not confirmed.

## Questions for the owner (open)

1. [ANSWERED by play-modes decision: F4 = Spectator, overal] Vrij vliegen (F4, zonder botsing) werkt ook in Survival, en elke client kan het via het netwerk
   aanzetten; /give, /tp en /gamemode hebben geen rechtencheck. Moet F4 alleen in Creative?
   Moeten commands in Survival uit (of alleen voor de host)? (audit-sim-server #3)
2. Een speler die de verbinding verbreekt verliest zijn inventaris, en de volgende save verwijdert
   players/<id>.ron. Moet een terugkerende speler zijn spullen terugkrijgen (dan is er een
   identiteit nodig: naam/sleutel)? (audit-sim-server #5)
3. F6 band: optie (c) (ver terrein vanaf 512 m) scheelt 19-47% voxel-driehoeken maar toont een
   gladde band die minder 'elk blok' oogt; (b) doet visueel niets. De ringranden op 160/288 m
   blijven bij beide. -> design-lod-transitions.md moet dit beslechten (engineering), geen vraag
   tenzij het rapport een smaakkeuze oplevert.

## Laatste stand

2026-11-09 — sessie gestart, plan opgesteld, hervat-taak en SessionStart-hook
ingesteld. Bezig met stap 1.

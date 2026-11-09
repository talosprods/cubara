# Docs vs code consistency audit

Read-only audit of branch `gui-test`. Nothing edited, cargo not run. `gh` used for issue and PR state.

Legend: "likely right" is my judgement from the code and git history, not a verdict.

## 1. Stale or wrong statements in docs

| # | Doc file:line | Code file:line | Doc says | Code says | Likely right |
|---|---|---|---|---|---|
| 1 | `docs/PROPOSAL_FAR_VIEW.md:3` | `crates/app/src/far_streaming.rs:22`, `bench.rs:707`, `far_streaming.rs:16-111` | "Status: proposed ... nothing is built yet" | F3 (far terrain, streamed in window/bench/screenshot), F4 (per-vertex aggregate shading, `far.wgsl`) and the quality levels are built (see the doc's own §6b, §6c, line 287). Only F5 haze (#299, open) is unbuilt. | Code. Status line is stale. |
| 2 | `docs/PROPOSAL_FAR_VIEW.md:36` | `crates/render/src/render.rs:35` | "Far plane: `FAR_PLANE` = 2,000 blocks" | `FAR_PLANE: f32 = 1.0e7` | Code. The row predates the far-terrain change. |
| 3 | `docs/PROPOSAL_FAR_VIEW.md:38` | `crates/app/src/main.rs:799-802`, `far_streaming.rs:22` | "Fog: `Lighting::fog_range(render_radius_blocks())`, ends inside 1,024 blocks" | Window fog is `fog_range(FAR_VIEW_RADIUS)` = 262,144 blocks, so fog sits at the far-terrain edge. Its own comment says the haze is not yet in. The `render_radius_blocks()` function does not exist in the code. | Code. |
| 4 | `docs/PROPOSAL_FAR_VIEW.md:39` | `crates/world/src/node.rs:137,180` | "a cube: 1,024 blocks up or down ... so **from 3 km up nothing is drawn**" | Reach is the 64-chunk cube = 1,024 blocks in all directions. 1,024 blocks is about 1 km, so "3 km" contradicts the same row. | Probably "1 km". The row was already corrected once (512 to 1,024) and the 3 km figure looks like a leftover. |
| 5 | `ROADMAP.md:114` | `scripts/check-phase-gate.sh` `perf_gate` | Phase 1 criterion: "`cargo run --release -- --bench 64` reports ≥ 1000 FPS" | Gate runs `--bench gate` (amended at ROADMAP:116-124). | Code. The bullet was not updated after the amendment. |
| 6 | `ROADMAP.md:370-374` ("Open, and deliberately not closed") | `crates/app/src/main.rs:1056` (`--connect`), `crates/render/src/figure.rs` (players drawn) | "`--connect` does not exist ... `Effect::PlayerMoved` is a no-op ... Two players still cannot see each other" | `--connect` exists (PR #226 merged). Other players are drawn (PR #225 merged). `ROADMAP.md:397-400` in the same doc says these landed. | Code. The "Open" paragraph is stale and contradicts the table in the same document. |
| 7 | `ROADMAP.md:121` | `docs/PROPOSAL_FAR_VIEW.md` F3 | "the flight eye, until `docs/PROPOSAL_FAR_VIEW.md`'s block F3" | F3 is built, and BENCHMARKS.md has flight-eye rows with far-only geometry (`0 voxel`, `far only`). | Code. |
| 8 | `ROADMAP.md:464` | `crates/app/src/options.rs`, `main.rs:592-597` | Pause menu "Options (empty for now)" | Options screen exists with keys 1-4 (quality), A (auto), B (benchmark). | Code. |
| 9 | `crates/render/src/render.rs:28-29` (doc comment) | `render.rs:31-35` | "The far plane covers radius 64's diagonal (~1,448 corner to corner) with room over." | Same constant now `1.0e7`, with a newer comment below it saying it reaches hundreds of km. The first comment is stale. | Code. |
| 10 | `crates/app/src/screenshot.rs:32` (doc comment) | - | Refers to `streaming::render_radius_blocks()` | No such function exists (only `render_radius_chunks()` in `streaming.rs:41`). | Doc comment: rename the reference. |
| 11 | `docs/PHASE2_ARCHITECTURE.md` furnace section (~lines 431-438) | `crates/sim` (chunk catch-up, issues #58/#59 closed) | "This design does **not** attempt that now. Block 2.7 ... is what makes it resumable" | Block 2.7 landed (#58, #59 closed; `docs/PHASE2_ARCHITECTURE.md` ~649 says dormancy is now correct). Written in the future tense for a shipped feature. | Code. Doc is a 2.4c-era design; could add an "as built" note. |
| 12 | `scripts/check-phase-gate.sh` (phase-2 block comment) | ROADMAP:297 (16 passed, 0 failed) | "Wired up while block 2.7a was landing, deliberately RED" | Phase 2 gate recorded green at `b5dcfec`. | Probably a stale comment. |
| 13 | `README.md` ("There is no network transport yet") | `crates/server/src/net.rs`, `main.rs:1056`, ROADMAP 2.12 (#202 merged) | Server "does not yet serve one to anybody" | `--listen` and `--connect` exist and are documented in `headless.rs` USAGE and CONTRIBUTING.md:136. | Code. README is stale. |
| 14 | `docs/PROPOSAL_FAR_VIEW.md` header / `CLAUDE.md` (project file, out of scope but relevant) | `BENCHMARKS.md:208-209`, `bench.rs:48` (`WARMUP_FRAMES = 200`, `WARMUP_MIN = 1 s`) | "the 200-frame warmup is too short to reach steady clocks" | Warm-up now waits for at least 200 frames and 1 s (commit 2ff8fee, BENCHMARKS row 2026-11-09). | Code. The caveat in `BENCHMARKS.md` (lines ~208-214) and the CLAUDE.md performance note are out of date. |
| 15 | `PLAN.md` §7 M0-M8 | ROADMAP:547-549 | PLAN milestone table "what ships when" | ROADMAP says PLAN §7 is superseded. | Expected; PLAN is historical. Low priority. |

## 2. Constants that agree (checked, no finding)

- Ring schedule: `DEFAULT_RING_SCHEDULE = [(0,10),(1,18),(2,32),(3,64)]` (`node.rs:137`). Outer 64 chunks = 1,024 blocks matches PROPOSAL §1 and §2A.
- `VERTICAL_LOD_SQUASH = 2` (`streaming.rs:36`). The "keeps 80 blocks / 40 blocks" figures match 10/squash chunks x 16.
- `FarQuality` levels and `(quad_px, error_px, floor_px)`: High (16,2,4), Medium (16,4,4), Low (32,6,8), Off = voxels only. Matches PROPOSAL §6c. Keys 1-4 / A / B match `options.rs:50-60`.
- Step-down: 3 s window, 20% misses, 1.5x budget, 10 s settle (`options.rs:88-100`). Matches the §6c text.
- `REPLY_TICKS = 5 s` (`game.rs:44`). Matches ROADMAP:486.
- Gate eyes and 1000 FPS target: `try_qualities(1000.0)` (`main.rs:~870`). Matches ROADMAP amendment.
- `--bench tune` default 60 (`main.rs:969-972`), `--target`, `--far`, `--eye`, `--squash`, `--fog`, `--gpu-timing`, `--size`, `--caps`, `--connect`, `--screenshot`, `--menu`, `--look`, `--radius`: all exist. Flags documented in README, CONTRIBUTING, ROADMAP and PROPOSAL are all real.
- `server` flags (`--world --tps --autosave --ticks --listen --help`): match README and `headless.rs` USAGE. Autosave default 18000 = 5 min at 60 tps.
- Inventory: `SLOT_COUNT = 36`, `HOTBAR_WIDTH = 9`, `MAX_GRID = 3`, `REGEN_INTERVAL = 300` ticks, despawn 18,000 / 108,000 ticks, `oak_log` burn 800, max_stack 64. All match PHASE2.
- `inventory::add` doc comment (lowest-indexed merge, return leftover) matches the body.
- `bench::try_qualities` doc (stop at first quality that holds everywhere) matches the `break` at `bench.rs:989`.
- `cubara-server` binary and `cubara server` subcommand both exist; README build commands are valid.
- Feature `profile` exists in `crates/app/Cargo.toml:13-15`.

## 3. Flags and keys documented nowhere (or undocumented real ones)

- `--bench flight` (with `--eye --look --speed --seconds --fps --every --frames`) exists in `main.rs:~945-963`. Not in README, CONTRIBUTING or ROADMAP.
- `--bench tune`: documented in PROPOSAL §6c, not in README or CONTRIBUTING.
- `--gpu-timing`, `--squash`, `--band`, `--overlay`, `--fog`: exist in `main.rs`, undocumented in the top-level docs.
- `/time`: a console command that answers "not built" (`server/src/lib.rs:1486`). Mentioned in ROADMAP only.
- Console command set is `/tp /give /gamemode /seed` (`server/src/lib.rs:1450-1486`). Matches ROADMAP:484.
- Key `F4` (fly toggle), `F3`, `F5`, `O`, `C`, `N`: in code (`game.rs:~690`, `main.rs:585-612`). Not documented in any of the audited docs.

## 4. Doc comments on public functions

Spot-checked; they agree with the body except the items in section 1 (#3, #9, #10). Not every `///` in `crates/*/src` was read, so this is a sample, not a full sweep.

## 5. Open-issue table (ROADMAP "Where the existing open issues land", line 537)

gh state, `gh issue list --state all --limit 300`:

| Listed in ROADMAP table | gh state | Finding |
|---|---|---|
| #89, #83, #46, #54, #43, #44, #55, #48, #57, #53, #52, #90, #60, #38 (Phase 1) | all CLOSED | Table is titled "existing open issues" but every phase 1 entry is closed. Stale. |
| #56, #47, #58, #59 (Phase 2) | all CLOSED | Same. Stale. |
| #28, #32, #33, #42 (engine, phase 3) | all OPEN | Correct. |
| #36 (engine, phase 3) | CLOSED ("Add native Linux (Vulkan) to CI") | Listed as open. Stale. |

Open issues not in the table: #299 (haze, F5), #271 (naga/HLSL DX12), #264 (horizon from mountain; ROADMAP discusses it as answered but does not list it), #132 (4x MSAA deferred), #131 (front-to-back draw order), #130 (hierarchical frustum cull). The table does not claim to be exhaustive, but #264 is the one with a recorded owner decision.

Note: #123, #124, #125, #209, #221, #222, #224, #225, #226, #233 cited in ROADMAP are PRs, not issues (`gh pr list`: all MERGED). ROADMAP writes them as `#NNN` without saying so. Minor naming ambiguity, not a contradiction.

## 6. Summary of which side is likely right

- Code is right in every item in section 1 except #4 (the "3 km" wording is a doc error, probably "1 km").
- The docs most in need of update: `docs/PROPOSAL_FAR_VIEW.md` (status, FAR_PLANE, fog, reach), `ROADMAP.md` ("Open" paragraph at 370-374, gate bullet at 114, table at 537), `README.md` (network transport line), `BENCHMARKS.md` warm-up caveat (208-214).
- No CLI flag documented in the audited docs is missing from `main.rs`.

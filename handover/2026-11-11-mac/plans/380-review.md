# Review of `plans/380.md` (#380, R6: a detail swap slides over 250 ms)

Second-model review, read-only. Checked against `origin/main` = `572113d`
(#406, merged 2026-11-11 00:30Z) and the plan's own base `b151790`. Nothing was
compiled, run or computed beyond reading. Sources: `CLAUDE.md`,
`ARCHITECTURE.md`, #380 (body; **0 comments**, last edit 2026-11-10 09:43Z),
#379's PRs #403/#404/#407, #337/#318/#384–#398, the Linux handover
(`origin/handover/2026-11-11-linux:handover/2026-11-11-linux.md`, which holds
the lessons list and the S2 wording), `docs/PHASE3_ARCHITECTURE.md` §7,
`docs/PROPOSAL_LOD_TRANSITIONS.md` §3 R4/R6, and every file the plan cites.

**Verdict: amend the issue first, then execute with the corrections below.**
The mechanism is sound. Its per-node lift lives in `NodeSlot`, rises and sinks
by rate, holds for H, and Δ comes from the caller. I re-derived the no-hole
argument and refinement B's counter-example, and both are right. Three things
are not ready:

- **The acceptance test is not pinned** (B1). As written, the 4× can be reached
  by changing the frame step. Nobody has said which of three numbers decides.
- **R6 makes the tree hand-over visible, and nobody has decided that** (B2).
  Today the hand-over is invisible: the same oak stands at the same site on
  both sides of a 0↔1 or 1↔2 swap.
- **The go/no-go bar sits below what the 4× arithmetically needs** (M1).

---

## 0. The four points the planner raised, settled

### 0.1 "S2": it exists, and the plan complies with it

S2 is **not an amendment on #380**. It is code-review finding **S2 of #379's
part-3 PR, #407**. That PR's findings table reads: `| S2 | documented: R6's
lift must live in NodeSlot | — |`. The "fix" was documentation, in two places:

- `docs/PHASE3_ARCHITECTURE.md` §7 (main, line ~249): *"#380's R6 lift of
  `node_origins[i].y` must live in `NodeSlot` and go through an arena method
  that `insert` and `rebase` both apply, never a `write_buffer` from outside."*
- `crates/render/src/arena.rs:496-503`, the doc of `rebase`. It names the
  method shape (`set_origin_lift(id, dy)`) and the failure: a write from
  outside "would be silently undone by the next rebase and leave the mirror
  stale".

The handover restates it as one of #380's binding amendments (line 61). S1
(#403) and S3–S6 (#407) are the same review's other findings.

**Compliance: yes, with either wording.**

- `NodeSlot.lift` holds the lift.
- `origin_entry`/`drawn_box` are the single helpers, used by `insert` (lift 0)
  and `rebase` (`slot.lift`).
- `set_lift` is an arena method that writes `origins[i]` and the buffer.
  `Renderer` never touches `origins_buffer`.

M2 below changes *when* the buffer is written (once per frame) and keeps it
inside the arena, so S2 still holds.

**Correction to the plan, §1, "Not found — say so":** replace the paragraph with
*"S2 is #407's review finding, recorded in `PHASE3_ARCHITECTURE.md` §7 and
`arena.rs:496-503` and quoted above; it binds and this plan follows it. #380
itself has no comments."* In §7, drop the stop condition "An S2 amendment …".
Keep the instruction to re-read #380's comments for the amendment below.

### 0.2 Refinements A (decision 3) and B (decision 5): the planner is right

**A is right.** In `streaming.rs:742-784`, `drain_meshes` hands the renderer
whatever the mesh pool finished that frame, plus the held nodes that are now
releasable, in one `apply_node_updates`. So:

- the children of one split arrive over several frames, mostly in batches with
  no Remove;
- the parent's Remove rides the batch of whichever child completes the set;
- a batch's Inserts also carry new edge nodes and edit re-meshes;
- `follow_visibility` (`:613-625`) sends Remove-only batches.

Read literally, decision 3 lifts an arbitrary subset: some children, plus edge
nodes that replace nothing. Δ also cannot be computed in the renderer. The
level is in `NodeId`, but the cell-height rule lives in
`NodeKey::cell_height_blocks` (`node.rs:79-85`). Duplicating it in
`cubara-render` breaks Rule 5, and `NodeId` is opaque by its own doc
(`arena.rs:30-45`, Rule 3).

**B is right.** I checked the counter-example step by step. Take O at hold
until t_O+H, and N placed at or before t_O and leaving at t_N, with
t_O < t_N < t_N0+H. If N freezes below −BIAS at t_N, then on [t_O+H, t_N+H)
nothing over N's footprint is at full height:

- O is sinking;
- N is frozen low;
- N′ may still be rising.

The literal text, "sinks from where it is", admits both readings. Only the
"keep rising through the hold" reading satisfies the property the same
decision demands. Under it, N finishes its rise by t_N0+H ≤ t_O+H, so it is
at −BIAS when its own sink starts.

A side effect, too small to block on: while capped at −BIAS during its hold,
N is coplanar with its own held parent O for at most t_O − t_N0. That is a few
ms, and only in a three-level cascade inside 125 ms. Note it in the docs.

Per CLAUDE.md ("a decision the issue does not contain is a defect in the issue
— fix the issue first"), these must be on #380 **before** PR 2 starts, not as
the executor's step 0. The text, ready to post, is in §5 below. It also covers
the points that B2, B1, M1 and M4 add.

### 0.3 "R6 slides height only": a real gap, in the issue first and the plan second

The planner's risk 1 is real, and larger than "AO".

- **Texture.** The mesher lays **one texture tile per lattice cell**:
  `crates/voxel/src/voxel.rs:595-617`, "one tile per grid cell".
  `mesh.wgsl:192` passes `uv` through unscaled. A level-L top therefore tiles
  its texture once per 2^L blocks. R4's per-block UVs were never built (no
  multiply in `mesh.wgsl`).
- **AO** is width-scaled per cell (`ao_width_log2`, `mesh.wgsl:174-186`).
- **The result.** Where the fine and coarse tops coincide, the hand-over is a
  one-frame **texture-scale and AO switch over the whole flat area**. It happens
  at the frame the incoming node's lift passes −BIAS.

At the 0↔1 edge (~160 m, about 6 px per block at 1080p), that area can be many
times the silhouette band. The issue's metric counts it. The pop paragraph says
"largest single-frame **pixel** change in the swapped area", which is
`changed_pixels`, not a silhouette. Whether R6 can pass its own acceptance test
is therefore genuinely open, and may be **no** on flat ground.

**Where the gap sits.** It is first a gap in the **issue**:

- decision 2 chose a height-only slide with a one-frame coplanar hand-over
  ("a bias of 0.01 block avoids z-fighting at the hand-over");
- decision 7 chose a metric that counts everything that changes.

The plan sees it (§8 risk 1) and turns it into a stop condition, which is the
right shape. But the plan leaves the measure undefined (B1), so "less than a
4× drop" cannot be read the same way twice. The plan also misses trees (B2),
a second un-slidable component that R6 makes *worse*.

**What F14's done-when requires, exactly:**

1. **The "Pop metric" paragraph:**
   - camera still;
   - a forced swap;
   - the largest single-frame pixel change *in the swapped area*;
   - **≥ 4× lower** with the fade;
   - the same final image.
2. **The owner's eye.**
3. **HOLES must not regress.** Decision 6: "must stay 0". On the low flight
   today that is "not more than today".
4. **STALE is reported before and after.** It is "not expected to move".

It does **not** require gate item 0 or POPS p99 to change. POPS p99 is already
0, so the plan's honesty note is correct. #384, which the amendment said to
wait for, has **landed**: its three terms are printed by every flight. The
acceptance number is still the issue's own harness, because POPS samples the
frame a node *leaves the arena*. With R6 that frame is the end of the sink, so
POPS cannot see R6 at all.

**What the plan should do (folded into B1):**

- pin the measure;
- report its composition (silhouette flips vs land→land changes vs tree
  pixels);
- have the lead decide **now**, not the executor at 05:00, what happens when
  the flips drop ≥ 4× and `changed_pixels` does not. Proposed text is in §5 D.

### 0.4 PR 1's go/no-go: chosen, labelled as chosen, and inconsistent with the 4×

The plan labels the 50% rule "a chosen reading". That satisfies the lesson.
But a **derivation exists**, from the issue's own acceptance test, and 50% fails
it.

**Per cell, what slides and what pops:**

- With |d| = |fine − coarse| and Δ the cell height, min(|d|, Δ) slides.
- max(|d| − Δ, 0) pops in one frame:
  - at **insert**, where fine is higher (the new node starts Δ low, already
    above the old by |d| − Δ);
  - at **removal**, where fine is lower (the old node ends Δ low, still above
    the new by |d| − Δ).

**The covered share needed:**

- At the flights' 60 Hz, a half-fade is k = 0.125 s × 60 = **7.5 frames**.
- Let U↑, C↑ be the uncovered and covered sums over cells where fine is above
  coarse, and U↓, C↓ the same where fine is below. Let T = U↑+C↑+U↓+C↓.
- Height alone, the best fade-on maximum is about max(U↑ + C↑/k, U↓ + C↓/k).
- The 4× needs that to be ≤ T/4.
- **One-sided** (all ↑ or all ↓): U + C/7.5 ≤ (U+C)/4. That needs covered
  ≥ 1/(1 + (1/4 − 1/7.5)/(3/4)) = **86.5%**.
- **Split evenly** between ↑ and ↓: U/2 + C/15 ≤ (U+C)/4. That needs covered
  ≥ **57.7%**.

So with 50% covered, the 4× is unreachable **on height alone**, before texture,
AO or trees. PR 2 would be built to fail its acceptance test. Correction: §4,
M1.

---

## 1. Findings, ranked

### BLOCKERS: fix before anything is built

**B1. The acceptance test can be passed by changing the measurement, and the
plan never names the measure.**

1. **The synthetic GPU test** (`a_swap_slides_instead_of_popping`):
   - It draws at `frame_dt = 1/120 s`. That is 15 frames per half-fade, 2× easier
     than the flights' 60 Hz (`HOLE_MAX_FRAMES`' doc: "at the flight's 60 Hz").
   - Its fallback, "If (2) is marginal … halve `frame_dt` — never lower the 4",
     **is** lowering the bar. Halving the step halves every per-frame change.
   - Assertion (3), "first frames equal … between the two runs", is ambiguous.
     The first frame *after* `apply_node_swaps` differs by design: fade-off
     drops 1 block, fade-on draws the new node hidden.
2. **`print_swap_pop`:**
   - It draws "until settled" with the wall clock, so the number of frames a
     fade spans depends on how fast the machine reads back.
   - It measures "over the whole screen", where moving the streaming eye also
     changes the far terrain and new edge nodes. None of that is a swap, and
     none of it is spread by the fade.
   - It runs at 1280×720, while gate item 0 is judged at 1080p (handover).
   - It prints three measures and never says which one decides.
3. **Stop condition 3** ("less than a 4× drop") inherits all of the above.

**Corrected text, replacing §5's two pop tests and §7 bullet 3:**

> **Synthetic (CI):** `frame_dt` is fixed at **1/60 s**, the flights' display
> rate, and is never changed to make the test pass. If (2) is marginal, raise
> the resolution, never the frame rate. "First frame" is the frame drawn
> **before** `apply_node_swaps`; assert that it, and the last frame after
> `fades_active() == 0 && uploads_pending() == 0`, are **byte-equal** between
> the fade-on and fade-off runs.
>
> **Real terrain (`print_swap_pop`, ignored, reported):**
> - `Streamed` gets one `frame_at(flight, stream_eye, camera_eye, frame_dt)`;
>   `frame()` calls it with equal eyes and its wall-clock dt, so there is still
>   one frame path.
> - The measurement passes a **fixed 1/60 s** `frame_dt` and runs at
>   **1920×1080**.
> - **Swapped area** = the union, over the whole step and over *both* runs, of
>   the screen boxes of every box `swapped_out()` reported. Use the full-height
>   box per 2.2; boxes project with the renderer's anchor, as `fly()` does.
> - **The acceptance measure is `changed_pixels(…, POP_TOLERANCE)` inside that
>   union**, maximised over every frame of the eight steps.
> - Also printed, as diagnostics only: land↔nothing flips, `silhouette_shift`,
>   and the land→land share of the fade-on maximum frame.
> - Three runs per arm. **Pass: median(fade-on max) × 4 ≤ median(fade-off
>   max)**, all six maxima in the PR (chosen, not derived: median of three).
>   Final frames per step byte-equal on vs off; any difference reported with
>   its pixel count.
>
> **Stop condition 3:** when the acceptance measure is under 4×, open the PR as
> a draft with all three measures and the land→land share. If the lead has
> pre-decided this case on #380 (§5 D), follow that instead.

Optional and cheap: land `print_swap_pop`'s **fade-off arm in PR 1**, on `main`
with no R6 code. It gives the land→land share of today's swap frames, an upper
bound on what no height slide can spread.

- If that share is ≤ ~15%, the 4× is safe from texture and AO.
- If it is ≥ 25%, tell the lead before PR 2.

This also moves about 200 lines out of PR 2 (M3).

**B2. Trees: a decision the issue does not contain, which R6 makes worse than
today.**

`trees.wgsl:75` adds the node's `node_origins` entry, so instanced trees slide
with their node. Level-0 voxel trees are part of the chunk mesh, so they slide
too.

At **0↔1** and **1↔2**, the outgoing and incoming nodes draw the **same oak at
the same site**:

- `WorldGen::node_trees` (`worldgen.rs:654-675`) places one site per chunk;
- `TreeModel::Oak` is "the real oak at levels 1-2" (`trees.rs`, `streaming.rs:204-217`);
- level 0 draws that oak as voxels.

So **today's tree hand-over is invisible**. With the plan as written:

- the incoming node's instances go in at insert, Δ low, under the old ones, so
  a leafy band Δ tall appears under every canopy in one frame;
- the outgoing node's instances stay until the sink ends, so a Δ band vanishes
  in one frame at removal;
- in between, trees are doubled.

At 160 m, Δ = 1 block is about 6 px × the canopy width per tree. That is a
**new** one-frame change on every tree. It is not spread, and it counts in the
acceptance measure.

The plan mentions trees only for culling boxes (§8 risk 6).

**Fix:** amendment C in §5, and a plan change if the lead adopts it.

- In `ArenaSink::try_insert`, when `lift != 0`, keep the
  `(node_index, origin, tree_model, trees, local_aabb)` in the `Fades` entry.
  `TreeArena::insert` them when the rise completes.
- In `advance`, at the end of a leaver's hold, `trees.remove(id)`.
- Tests: `an_incoming_nodes_trees_appear_when_its_rise_ends` and
  `a_leavers_trees_go_when_its_hold_ends`. Break each by inserting or removing
  at the old moment.
- Level-0 voxel trees cannot be separated without a shader change, which
  decision 2 rules out. Report them, do not work around them.

---

### MAJOR

**M1. PR 1's go/no-go rule:** replace step 7 of PR 1. Derivation in §0.4.

> 7. **Go / no-go for PR 2 (derived from decision 7's 4×; chosen inputs named).**
>    `SwapResidual` carries signed sums: `u_up/c_up` over cells with
>    `d = fine − coarse > 0` (`Σ max(d−Δ,0)`, `Σ min(d,Δ)`), and `u_down/c_down`
>    over `d < 0`. For each of 0↔1 and 1↔2, over the four flights' total:
>    `best = max(u_up + c_up/7.5, u_down + c_down/7.5)`, with
>    `T = u_up+c_up+u_down+c_down`. Here 7.5 = frames per half-fade at the
>    flights' 60 Hz.
>    - **Go** if `best ≤ T/4` for both pairs.
>    - Otherwise **stop**. Post the table and say decision 2 is to be revisited
>      (the issue's own instruction).
>
>    State in the comment:
>    - this is a **necessary, not sufficient** condition (texture, AO and trees
>      come on top);
>    - per-cell block heights stand in for pixels (a proxy);
>    - the literal "routinely > Δ" is also printed: the % of swaps with
>      `max_abs > Δ`.
>
>    The 2↔3 row is reported, never acted on.

Also in PR 1:

- Factor the arithmetic into a pure
  `fn summarize(d: &[i32], delta: i32) -> SwapResidual`, so the hand-made
  `|d|=3, Δ=1 → covered 1` test (step 5a) can exist without a `WorldGen`.
- Convert to pixels with the **3-D** eye-to-top distance, not the horizontal
  one. On the down flight, from 450 m, horizontal distance under-reads by the
  height.
- In `swaps_between`, say "**any** child in `after`".

**M2. Per-node, per-frame `write_buffer`, and no CPU/frame budget.**

`BENCHMARKS.md` ⁸⁷ counts 16,197–28,435 swapped-out nodes per 15–30 s flight,
about 800–1,300 a second. With splits and merges balanced, arrivals are about
the same number. Concurrent fades are therefore about 1,000 × (0.25 + 0.125) s,
roughly 400 on average and several times that in a chunk-crossing burst.

The plan's `set_lift` issues **one 16-byte `queue.write_buffer` per fading node
per frame**. That is hundreds to over a thousand calls a frame. At about 1 µs
each (`uploads.rs:419`: 1,000 node uploads, 3 writes each, in 3.6 ms on an M3),
that is 0.4–1 ms a frame, against the M3 low flight's work p99 of 2.7–2.9 ms
(⁸⁸). Decision 6 says "a few vec4 per frame". The lessons say every
render/perf issue carries a CPU/frame budget, and neither the issue nor the
plan has one.

**Fix (S2-compliant, still inside the arena):**

> `set_lift` updates `slot.lift`, `slot.aabb` and `origins[i]`, and widens a
> dirty range `[lo, hi]`. `ChunkArena::prepare`, which runs every frame on
> every path, writes `origins[lo..=hi]` in **one** `write_buffer` when the
> range is non-empty, then clears it. `insert` keeps its immediate write, and
> `rebase` its prefix write; both read the mirror, so any order is consistent.
>
> **Budget (chosen, not derived):** R6's CPU per frame (`Fades::advance` plus
> the flush) ≤ **0.05 ms at 2,000 fading nodes** on the M3.
> - Instrument: an ignored `fade_cost_report` in `tests/offscreen.rs`, the
>   `rebase_cost_report` pattern. Fill 2,000 nodes, start their fades, time
>   `render` with fades against `set_swap_fade(ZERO)`.
> - Plus the low flight's work p99, fade on vs off, from the same binary
>   (§6 run 3).

**M3. PR 2 is over the size lesson, and the reason given for not splitting is
wrong.**

Estimate:

- `fade.rs` with unit and property tests: about 650 lines.
- `arena`, `uploads` and `render` changes: about 300.
- The synthetic GPU test and the rebase test: about 220.
- `streaming`, `flight`, `main` and `print_swap_pop`: about 400.
- Docs and `BENCHMARKS.md`: about 140.

Total: **about 1,700 lines**, against "split a PR over ~1,200 lines" (handover
lessons).

"The state machine without its caller would be dead code" is false.
`Renderer::apply_node_swaps` is `pub`, so `Fades` is reachable and exercised
by the synthetic test. Split:

> **PR 2a `feat/380a-swap-slide-render`** (render crate, plus the three
> `render()` call sites):
> - `fade.rs`, the arena lift, the uploads `sink`, `Arriving` and
>   `apply_node_swaps`;
> - `frame_dt`, `set_swap_fade` and `fades_active`;
> - the tree timing (B2) and the dirty-range flush (M2);
> - every render-crate test, including the synthetic GPU test and
>   `fade_cost_report`.
>
> The game is unchanged, because no caller passes Δ yet. Gate A/B only.
>
> **PR 2b `feat/380b-swap-slide-streaming`:**
> - `rise_for`/`sink_for` and the wiring, the `Streamed` changes,
>   `--swap-fade-ms` and `print_swap_pop`;
> - the flight measurements, `BENCHMARKS.md` and PHASE3 §9.
>
> Stop condition 3 applies here.

**M4. STALE can rise by design, and the plan has no rule for it.**

In decision 1, the incoming node starts Δ low. Where the fine surface is above
the coarse one, the live frame shows the lower coarse silhouette for up to H
**after** the swap. The settled pass (fade 0) shows the fine land.
`stale_pixels` (`flight.rs:572-601`) counts exactly that wherever the band is
≥ 2 px tall. It lengthens the late-swap window (p99 183–350 ms, ⁸⁷) by up to
125 ms.

#406's attribution files those pixels under "**neither**": the swap lag closed
at release. #406 reads "neither" as the far terrain. So an increase would also
be mis-attributed.

Lesson: "never accept a PR that moves a gate metric the wrong way". Add to §6
and §7:

> Report STALE (kept frames, worst, the three shares, and #406's hit-distance
> line), low flight, **fade on vs `--swap-fade-ms 0`, two runs each, same
> binary**. Stale pixels whose rays land inside the voxels' reach with fade on
> are R6's rise, not far terrain; say so. If STALE kept frames with fade on
> exceed both fade-off runs, open as a draft and stop (amendment F).

**M5. Two pieces of logic have no test that can fail.**

1. **The property test cannot catch mutation (iii)** ("returning id restarts at
   −Δ"). I traced it: the returning P re-rises over H while the children it
   replaces hold at −BIAS for H, so coverage never breaks. Only the named unit
   test catches it.

   Coverage also does not pin the issue's actual goal, "never a one-frame
   change". Add to the property test, checked after every step:

   > (a) **continuity:** for every id present in consecutive steps,
   > `|lift_k+1 − lift_k| ≤ max_rate·dt + BIAS + 1e-4`. The `+BIAS` is the
   > one-time drop when a hold starts.
   > (b) **hidden removal:** a node with `sink > BIAS` leaves the fake arena
   > only at `lift ≤ floor + 1e-4`.

   With these, (iii) and an "early remove" mutation are caught by the property.
   Update the §5 table's "break it" column to match.
2. **The streaming wiring has no CI pin.** Swap `self.resident.insert` above
   `rise_for` in `drain_meshes`, or pass `0.0` for `sink`. Every arrival then
   looks like an edit and nothing ever fades. Only the ignored `print_swap_pop`
   would notice.

   Fix: extract a pure
   `fn swap_updates(built: &[NodeKey], released: &[NodeKey], resident: &mut HashSet<NodeKey>, visible: &HashSet<NodeKey>) -> (Vec<(NodeKey, f32)>, Vec<(NodeKey, f32)>)`,
   used by both `drain_meshes` and `follow_visibility`. Test it on a split
   delivered over two batches: the first child, then the last child plus the
   parent's release. Assert both children get Δ and the parent's sink is Δ.

---

### MINOR

**m1. Three things that fail CI as written** (`RUSTFLAGS: "-D warnings"`,
`ci.yml:16`):

- **(a) Dead code.** `pub fn offset` and `pub enum Role` sit in the private
  `mod fade` and are not re-exported, so the lib build warns that they are
  never used. Make them `#[cfg(test)]`. They are the oracle;
  `Fades::advance`, pure and clock-free, is decision 4's pure function. That
  also leaves one implementation of the timeline in production (Rule 5).
- **(b) Two closures borrowing `arena`.** In 2.4,
  `advance(dt, |id,l| arena.set_lift(..), |id| arena.remove(..))` takes two
  closures that both need `&mut arena`, which is error E0499. "Split the
  borrows with locals" does not help. `render.rs:1970-1973` and
  `uploads.rs:76-79` exist precisely because of this. Use one callback:
  `advance(dt, mut f: impl FnMut(NodeId, Step))` with
  `enum Step { Lift(f32), HoldEnded, Remove }`. `HoldEnded` carries B2's tree
  removal.
- **(c) Argument count.** `render()` gains an eighth input. Counting `self`,
  that trips `clippy::too_many_arguments` (the precedent is
  `text/mod.rs:346`'s allow on `self` plus seven). Add
  `#[allow(clippy::too_many_arguments)]` with a one-line reason, as
  `scene.rs:320` does.

**m2. Stale citations.** #406 merged after `b151790`, and the plan's §8 risk 9
predicted this. Function names hold; the line numbers on `main` are:

| Plan | Is on `main` (`572113d`) |
|---|---|
| `flight.rs:1175-1193` (pop) | `1379-1397` |
| `flight.rs:878-903` / `:887` (walk) | `1064-1089` / `1073` |
| `flight.rs:963-1035` `Streamed`, `:998` render | `1149-1221`, `1184` |
| `flight.rs:1016-1034` settle/settled | `1202-1220` |
| `flight.rs:1262` settled pass | `1474` |
| `flight.rs:1417` test `Flight` | `1709-1719` |
| `streaming.rs:738-780`, `:779` | `742-784`, `783` |
| `streaming.rs:570-637`, `:620-621` | `574-641`, `624-625` |
| `streaming.rs:492-498` invalidate | `496-502` |
| `check-phase-gate.sh:335-338`, `:181-204` | `336-339`, `180-203` |

All other citations were opened and are correct:

- `uploads.rs` (all of them);
- `render.rs` 946-948, 964-971, 979-983, 1058-1079, 1086-1094, 1271-1334,
  1974-2039, 2052;
- `arena.rs` 30-45, 203-229, 382, 496-513, 636-643, 670-676, 783-813;
- `mesh.wgsl` 150-151 and `trees.wgsl` 75;
- `node.rs` 79-85 and 212;
- `lod_probe.rs` 53, 71, 81, 778, 801, 837;
- `flight.rs` 30, 267, 278, 301, 392, 408;
- `main.rs` 756, 828, 978, 1030;
- `check-architecture.sh` 99-110;
- `check-phase-gate.sh` 33-34.

**m3. The #390 recovery path.** In `drain`'s full-arena pass
(`uploads.rs:366-375`), `pop_ready_remove` exists to **free space**. With
`Leave::Later` it frees nothing, yet `freed_any` is set and the retry fails.
Decide it: pass `0.0` at `:370` (apply at once, documented as "space beats
smoothness on a full arena"), and change the planned test to assert exactly
that.

**m4. Measurement protocol** (CLAUDE.md "Standing permissions"). Add:

- the machine is **not in use by the owner**;
- if apps are closed, the PR says which;
- per side, **build → ≥ 5 min idle → two runs** (main and the branch are
  separate builds);
- `--bench walk --fly --seconds 30` ×2 before and after (a gate-item-0 frame
  term that goes through `Renderer`);
- `print_swap_pop` and `fade_cost_report` at 1080p.

The `m3-remeasure.sh` path is in this session's scratchpad. Inline its rules
(AC check, `ps` filter, 120 s gaps, `pmset` before and after) rather than rely
on the file existing for a later session.

Also: "the gate path never touches `Fades`" is true, but it does touch
`NodeSlot`, which grows by 4 bytes and is iterated in `prepare`. "Within noise"
is still the expectation; give the right reason.

**m5. Block order and closing.** `next-block.sh`'s order puts #318, #397, #398
and #388 (3.0a) before #380. The plan should say #380 is taken because the
lead assigned it (handover, "Yours: 1. #380"), so step 1 of "go ahead" does not
contradict it.

Done-when includes the owner's eye, so PR 2b says **"Part of #380"**, not
"Closes". After merge, comment with the numbers and either label #380
`blocked` (awaiting the owner's eye; `next-block.sh` skips it) or let the lead
close it. Say which.

Also follow the handover's subagent rule if Sonnet runs as a subagent: its own
worktree and `CARGO_TARGET_DIR`, push the branch, open no PR, and the lead
opens it with `--auto --squash`. Or state that this plan overrides it.

**m6. Small test gaps:**

- `ChunkArena::remove` returning the **full-height** box (2.2) has no pin. Add:
  `set_lift(-1)` then `remove` returns `place.relative_to(anchor)`. Break:
  return `slot.aabb`.
- `set_lift` must update `slot.aabb`. Add it to the rebase test. Break: skip
  the aabb update.
- `Streamed::settled` waiting on `fades_active()` is unpinned; low value, note
  only.

**m7.** In `PHASE3_ARCHITECTURE.md` §7, turn the forward-looking sentence on
#380 into a pointer to the new §9, together with replacing `arena.rs:496-503`.
Use "the next unused §" rather than a hard-coded §9: #394's planner may add
sections too.

---

## 2. Tests: can each one fail?

| Test (plan) | Fails when the code is wrong? |
|---|---|
| `offset_*` (2) | Pins the **oracle** only (m1a). Fine as cfg(test). |
| `the_tracker_follows_offset…` | Yes: catches a wrong `rise_rate`, or a missing or broken hold. |
| `an_edit_does_not_fade`, `…during_a_rise_keeps_rising` | Yes |
| `a_returning_id_cancels_its_sink…` | Yes, and the **only** catcher of (iii) until M5(a). |
| `…leave_mid_rise_is_full_before_it_sinks` | Yes (freeze) |
| `…goes_at_once` / `fade_zero…` / `a_hitch…` | Yes |
| property test (coverage) | Catches (i) and (ii); **not (iii)**; blind to one-frame jumps. Fix: M5(a). |
| `uploads` sink pass-through | Yes; m3 changes what it asserts at `:370`. |
| `origin_entry` / `drawn_box` pure | Yes |
| `rise_for` / `sink_for`, incl. 2↔3 both ways | Yes. The 2↔3 rows are the only ones that tell "coarser" from "own level"; good. |
| **wiring `drain_meshes` → renderer** | **No test.** M5(2). |
| `a_rebase_keeps_a_fading_nodes_lift` (GPU) | Yes for origins; add aabb (m6). |
| `a_swap_slides_instead_of_popping` (GPU) | Yes, **only once B1's text is applied**: fixed 1/60 s and no frame-step escape hatch. |
| `ArenaSink` re-insert keeps lift 0 | Yes: catches `was` passed wrong. |
| `print_swap_pop` (ignored) | A measurement; B1 defines what it decides. |
| PR 1 `swaps_between` / `top_face` / hand-made covered | Yes; needs M1's pure `summarize`. The `≤` invariants are near-tautological; keep them only as smoke tests. |

## 3. Goldens

Correct as planned:

- `golden.rs` renders through `headless::render` (`from_meshed` → `insert`),
  never through `Renderer`'s fades.
- `y + 0.0` and `min.y += 0.0` are bit-identities for every integer-derived
  finite `f32`.

No reference may change, and none may be regenerated. The synthetic offscreen
test is this change's automated rendering pin. The issue rules out a mid-fade
golden.

## 4. Architecture rules

- **Rule 1:** the renderer is not the sim; the wall-clock dt passed in is
  allowed.
- **Rule 2:** `Fades` is a `Renderer` field, with no statics; the dt is passed
  in (decision 4).
- **Rule 3:** Δ is computed in `cubara-app`. The renderer gets an f32 and never
  a level rule. Good.
- **Rule 4:** `fade.rs` is pure and its tests need no GPU.
- **Rule 5:** one helper each forms the origin entry and the culling box. Two
  timelines (`offset` vs `Fades`) are resolved by m1a.
  `apply_node_updates` forwards to `apply_node_swaps`, which is fine.
  `Renderer` keeps its own `Instant` for the FPS read-out and upload budget
  beside the passed `frame_dt`. Acceptable, and the plan says so (§8 risk 8).
- **Rule 6:** gaps are M5 and m6.
- **Rule 7:** needs M2's budget and instrument.
- **S2:** complies (§0.1).

## 5. Amendment for #380: post before anything is built

A and B are engineering and the lead can post them. C, D, E and F decide what
"done" means and how the look trades off. They are the lead's call, and the
owner's for anything he wants to see first. If C–F are not adopted, the
executor needs at least a one-line rule for each.

> ## Amended 2026-11-11, before implementation: two readings made exact, trees, and what the pop test measures
>
> Nothing above is deleted.
>
> **A. Decision 3, made per node.** *"Only batches that carry Removes fade, and
> only their new ids rise"* is replaced by:
> - The streaming caller states, per node, whether it is part of a swap and the
>   swap's Δ.
> - Each arriving mesh carries `rise`: Δ when it replaces resident nodes (its
>   parent or its children), else 0.
> - Each leaving id carries `sink`: Δ when something visible replaces it (its
>   children or its parent), else 0.
> - Δ is `cell_height_blocks()` of the coarser level, computed in `cubara-app`.
> - An Insert for an id that is already resident and not fading never fades,
>   whatever `rise` says. That is an edit (`NodeStreaming::invalidate`).
>
> *Why:* a batch is not a swap. The children of one split arrive over several
> frames, mostly in batches with no Remove. The parent's Remove rides the batch
> of whichever child completes the set. A batch's Inserts also include new edge
> nodes and edit re-meshes. Read literally, the rule makes an arbitrary subset
> rise, including edge nodes that replace nothing, and leaves the rest to pop.
> The renderer cannot compute Δ either: the level rule lives in `cubara-world`
> (Rule 3, and Rule 5 against a copy). What decision 3 protects is kept
> exactly: edits never fade, only swaps do.
>
> **B. Decision 5, "sinks from where it is", made exact.** A node whose Remove
> becomes ready while it is still rising keeps rising through its hold half. It
> is capped 0.01 below full height, as every leaving node is. It then sinks from
> where it is, which is full height. It is never frozen mid-rise.
>
> *Why:* freezing breaks this decision's own property. Parent O is replaced by
> child N, so N rises while O holds until t_O + H. N's children N′ land at t_N,
> with t_O < t_N < t_O + H. If N froze below full height at t_N, then on
> [t_O + H, t_N + H) O is sinking, N is low, and N′ may still be rising, so
> nothing is at full height over N's footprint. Continuing the rise closes the
> gap. Nothing visible changes, because N is under O while it finishes.
> Decision 9's property test pins this interleaving by name.
>
> **C. Trees (not covered above).** A node's tree instances (`trees.wgsl`) and
> its voxel trees (level 0) slide with it. At 0↔1 and 1↔2 the outgoing and
> incoming nodes draw the same oak at the same site (`WorldGen::node_trees`),
> so today's tree hand-over is invisible. A plain slide would make it visible:
> trees doubled for 250 ms, and a band Δ tall appearing at insert and vanishing
> at removal.
>
> Rule:
> - an incoming node's instances are inserted when its rise completes;
> - an outgoing node's instances are removed when its hold ends.
>
> Both moments have old and new at full height, so identical trees hand over
> invisibly. Level-0 voxel trees cannot be separated from their mesh without a
> shader change, which decision 2 excludes. At the ring-0 edge they still show
> a Δ band for one frame. The pop test reports it; it is not worked around
> here.
>
> **D. What the pop test measures (decision 7, made exact).**
> - **Setup:** camera fixed at the low flight's start; the streaming eye moved
>   +16 x eight times.
> - **Frame step and size:** fixed at 1/60 s (the flights' display rate; never
>   shorter) at 1920×1080.
> - **Measure:** `changed_pixels` at `POP_TOLERANCE` between consecutive
>   frames, inside the union of the screen boxes of every node that left the
>   arena during the step. The same union is used for both arms.
> - **Pass:** median over three runs of (fade-on maximum) × 4 ≤ median of
>   (fade-off maximum). All six maxima are in the PR, and the final frames are
>   byte-equal on vs off.
> - **Diagnostics:** land↔nothing flips, `silhouette_shift`, and the land→land
>   share are printed beside it.
>
> *Known limit:* coarse tops tile their texture once per cell (2^L blocks) and
> scale AO per cell. Where fine and coarse tops coincide, the hand-over is
> therefore a one-frame texture-scale and AO switch that no height slide
> spreads. **If the flips drop ≥ 4× and `changed_pixels` does not**, the PR
> merges as an improvement with both numbers, and the owner's eye decides
> whether a texture cross-fade (a shader change, outside decision 2) is worth
> its own issue. *[Lead: confirm, or say the PR stays a draft.]*
>
> **E. Decision 8's "routinely", derived from D.** Per cell, min(|d|, Δ) slides
> and max(|d| − Δ, 0) pops: at insert where fine is higher, at removal where
> fine is lower. At 60 Hz a half-fade is 7.5 frames. For 0↔1 and 1↔2, over the
> four flights, implementation goes ahead only if
> max(U↑ + C↑/7.5, U↓ + C↓/7.5) ≤ (U↑ + C↑ + U↓ + C↓)/4, where U is the
> uncovered and C the covered sum on each side. That is, the 4× must be
> reachable on height alone: a necessary condition, not a sufficient one. If it
> is not reachable, decision 2 is revisited here with the table.
>
> **F. STALE.** The incoming node starts Δ low, so where it is higher than the
> outgoing one, the live frame lags the finished frame by up to 125 ms. The
> STALE term counts that. The PR reports STALE on the low flight, fade on vs
> `--swap-fade-ms 0`, two runs each from the same binary. An increase beyond
> both fade-off runs is a regression (decision 6's "must not regress", extended
> to STALE), and the PR stays a draft.

## 6. Decisions the plan still leaves to the executor (all fixed above)

| Decision | Fixed by |
|---|---|
| Which pop measure decides, over which area, at which frame step and size, and how three runs combine | B1 / D |
| When trees come and go | B2 / C |
| What to do if STALE rises | M4 / F |
| The `advance` callback shape | m1b |
| `too_many_arguments` | m1c |
| `offset`'s status | m1a |
| The #390 recovery-path sink | m3 |
| What "first frames equal" means | B1 |
| How `print_swap_pop` separates the camera eye from the streaming eye | B1, `frame_at` |
| Whether PR 2 says "Closes" | m5 |
| Whether #380 is labelled `blocked` | m5 |
| Whether the subagent opens PRs | m5 |
| The PR 2 split | M3 |
| PR 1's aggregator factoring | M1 |

## 7. What I could not verify (nothing was built or run)

- Whether clippy counts `self` in `too_many_arguments`. I believe it does
  (HIR `inputs`), and `text/mod.rs:346` is consistent with that.
- The size of the coplanar texture/AO switch on real terrain. The mesher code
  says it exists; how many pixels exceed `POP_TOLERANCE` is the B1 measurement.
- That the voxel oak and `TreeModel::Oak` are pixel-identical. Both are built
  from the same structure, and `drawn_top` gives the exact column at v = 1.
- The ~1 µs per `write_buffer` figure. It is inferred from `uploads.rs:419`.
  M2's instrument measures it.

**Verdict: amend the issue first (§5: A and B must be posted; C–F decided by
the lead), then execute with B1, B2 and M1–M5 applied to the plan, and the
minors as listed.**

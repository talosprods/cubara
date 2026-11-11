# Review of `plans/398.md` (#398, the far terrain's selection lag at 1.5–1.9 km)

Second-model review, read-only. Checked against `origin/main` = `572113d`
(PR #406 merged 2026-11-11 00:30:46Z) and against `3151b87`, the plan's own
base. Nothing was compiled or run. Sources read: `CLAUDE.md`, `ARCHITECTURE.md`,
#398 body + re-scope comment, the Linux handover
(`origin/handover/2026-11-11-linux:handover/2026-11-11-linux.md`, which holds
the "lessons" list), and the code cited below.

**Verdict: execute with the listed corrections.** The design holds up:
exact selection at the eye, far patches generated early but kept on the CPU,
and the decisions moved into a GPU-free `FarPlan`. I re-derived the band
arithmetic and the claim that the converged selection does not depend on the
path taken, and both are right. The defects are in the evidence. One test
design cannot see the fix being deleted from the place it matters (B1).
Step 1 cannot tell generation latency from level choice, and builds both PRs
anyway when the result is ambiguous (M1). The measurement protocol would
compare a loaded "before" with an unloaded "after" (M2). There is no CPU/frame
budget, and no instrument that can see this change's cost (M3). The harness
cannot see queueing (M4). `select` is rewritten without a pin that can fail
(M5).

---

## 0. State of play

- #406 has **merged** (`572113d`). Step 0.1 is satisfied, and §2.6 ("open,
  auto-merge on") is out of date.
- `crates/app/src/far_streaming.rs`, `crates/world/src/far.rs` and
  `crates/render/src/far.rs` are byte-identical between `3151b87` and
  `572113d` (`git diff --stat` is empty). Their citations hold. `flight.rs`
  and `streaming.rs` moved when #406 merged, so their citations are stale
  (§1).

## 1. Citations and claims about today's code

**Correct** (opened and confirmed): `far_streaming.rs` :61 (High = 16/2/4 px),
:95 `RESELECT_BLOCKS`, :100 `far_view`, :184 `draw_set`, :250 `cores / 2`,
:359 `settled`, :363 `reselect` doc, :374–453 `update`, :384–398 (moved / hole /
arrival test), :413–422 (requests at first want), :614/:627 (the
`keeping_the_ancestors…` test and its converge loop). `world/src/far.rs` :145
`select`, :150 `c as i64`, :176 `walk`, :186 hole drop, :190 `near`, :191
floor, :193 rough, :142–144 doc, :294 `footprint_mean`. `Cargo.toml:48`
(world at opt-level 3). `bench.rs:607` (`--bench 64` uses `far::build`, never
`FarStreaming`). `check-phase-gate.sh:34`, :191. `split_ratio` ≈ 2.0143. A
level-4 patch sits in the 1,031–2,063 band by distance. The rough threshold
is 2.97–3.68 blocks at 1,530–1,900. One `generate` costs about 85k
`surface_height` calls (17.4k vertex + ~68k `finer_gap`). The far terrain has
no geomorphing (`far.wgsl` has none) and no patch upload queue (`insert` is
two `write_buffer`s). There are 81 kept frames. The seed literal equals
`World::new()`'s `DEFAULT_SEED` (`world.rs:35`).

**Stale** (right at `3151b87`, wrong on the `main` the executor will check out):

| Plan says | On `572113d` |
|---|---|
| `flight.rs:1129` `fly` | `:1329` |
| `flight.rs:1280–1282` settled pass | `:1493–1494` |
| `flight.rs:453` `stale_pixels` | `:572` |
| `flight.rs:994` `Streamed::frame` | `:1180` |
| `streaming.rs:657` `far_hole` | `:661` |
| `check-phase-gate.sh:335` low flight | `:336` |
| PROVISIONAL text "lines 22–27" | lines 22–32 |
| `render/src/far.rs:412` `insert` | `:411` (also at `3151b87`) |
| "main = `3151b87`", #406 "open" | main = `572113d`, #406 merged |

**Wrong:**

1. §3a: *"The extra ring of roots is always beyond the radius, so `walk`
   drops them and the output is unchanged."* Not at `FAR_VIEW_RADIUS`. That
   radius equals the level-13 root size (262,144), so `reach == root_size`.
   - At an eye with x = 0, `lo = -1`. Root −2's near edge is then exactly
     262,144 away. `walk` keeps a box whose `ground_distance > radius` is
     false, so walking the extra ring adds a level-11 patch, 65,536 wide, at
     the radius.
   - At x = −0.5, `c as i64` truncates to 0. Today's loop already skips a root
     whose edge is 262,143.5 away, inside the radius. The extra ring adds it.

   So the output changes at those eyes. No golden or test eye is affected:
   all use (8, ·, 8), and the goldens use radius 20,000. But the claim the
   rewrite rests on is false, and nothing pins it (M5).
2. §7: *"#392 is voxel node selection / `streaming.rs`."* #392 is R2's +23%
   CPU/frame. Its body cites `trees.rs`, `arena.rs`, `culling.rs` and
   `tests/tree_culling.rs`.
3. §7: *"No open issue owns [the low flight's HOLES] as far as I can see."*
   #318 ("3.0a: no holes while flying") is open. It is the umbrella for
   exactly that item 0 term.
4. §5.4: *"`select_with_slack(v, e).0 ==` the pre-change `select` order (pin
   with the existing tiling/neighbour tests)."* After the change, `select` *is*
   `select_with_slack(..).0`, so the equality cannot fail. The existing far
   tests (`the_patches_tile_the_ground_exactly_once`,
   `neighbours_differ_by_at_most_one_level`, …) check properties, not the
   output. None of them would notice the change in item 1 (M5).
5. §2.3(1): the timing fit ("selections fall near t ≈ 1.35, 2.70, 4.05 s").
   Every arrival sets `stale = true` (`far_streaming.rs:381`), which forces a
   reselection at the current eye (:396–399) and resets `selected_at`. The
   live selection is therefore every 32 blocks *plus* one shortly after each
   batch of arrivals. The fit to t = 3.00–4.00 is suggestive. It does not
   confirm anything, and the text should say so.

## 2. The diagnosis

**What is supported.**

- The band arithmetic places the stale hits where 4→3 roughness splits
  happen, and the floor (`children_wide_enough`, :191) stops further splits
  past ~2,063 blocks. So at 1.5–1.9 km at most one roughness level is pending,
  plus the 5→4 distance split at ~2,063.
- "Level choice" is correctly ruled out *at convergence*. In a converged
  selection, every visited non-near node has a known error: a non-near node
  with an unknown error is a leaf, and a leaf gets generated. Each visited
  node's decision is then fixed top-down, so the result is unique.
- Live and settled use the same `far_view`, size and quality. The hole only
  drops whole patches.
- "Upload pacing" does not exist.

**What is not supported.** The split between reselection distance and
generation latency.

- Latency is unmeasured.
- In the real flight, latency includes FIFO queueing. The pool is one channel
  read through a shared `Mutex<Receiver>` (:254–264), and anything sent
  earlier delays a later urgent request. One example is the burst when the
  hole steps, every 128 blocks of x (~5.6 s at this heading).

**What would falsify it.**

- (i) With RESELECT_BLOCKS = 1, STALE stays at the base count. Then
  reselection distance is not the cause.
- (ii) A run whose live far terrain is the converged selection at the exact
  eye on *every* frame still shows stale frames. Then the cause is neither
  reselection nor latency, and the plan's fix cannot reach 0.
- (iii) At stale kept frames, live's `wanted` already holds the finer patches.
  Then the cause is latency.

**Does halving RESELECT_BLOCKS discriminate?** Only weakly. Per crossing, the
expected stale kept frames are about (R/2 ÷ 24 b/s + L) ÷ 0.25 s.

- With latency L = 0.1 s: 3.1, 1.7 and 0.5 for R = 32, 16 and 1.
- With L = 0.5 s (queueing): 4.7, 3.3 and 2.1.

Either way the halved run lands at 55–70% of base. With n = 3 and ±2 noise,
"about half" does not separate the hypotheses. The R = 1 run separates
reselection distance from *the rest*, and nothing in the plan separates latency
from level choice. Table row 2 ("base ≈ half ≈ tiny") then builds both PRs
before knowing.

The table also has gaps. Take base 12, half 9, tiny 5: tiny is not ≤ 3, and
the three are not within ±2, so no row applies, and Sonnet would have to
decide. Run with `--frames` (the plan says to), all of this is also
confounded by the PNG encoder thread (M2).

---

## 3. Findings

### Blocker

**B1. Tests 5.2, 5.3 and 5.5b record `plan.drawn`, the plan's belief, instead
of the effect of the `Step`s it hands to the renderer. A `step` that emits an
empty or wrong `arrive`/`leave` stays green in every planned test.**

- In §4 step 2, `step` sets `self.drawn = draw` itself.
- The harness (§5.1) "record[s] `plan.drawn`".
- 5.5b (`prefetch_uploads_nothing`: every `arrive` key is in `exact`) is
  satisfied by `arrive: vec![]`.

So the mutation "`Step { arrive: vec![], .. }`" survives all of §5. So does
"`leave` and `arrive` swapped". In the game, either one means nothing new is
ever drawn. This is the failure the existing comment at
`far_streaming.rs:484–485` records ("skipping every insert survived"), moved
one layer up. Corrected §5.1 harness text:

> - `fly(plan, eyes, hole, pool) -> Vec<HashSet<PatchKey>>`: the harness holds
>   `shown: HashSet<PatchKey>`, what a renderer would hold, and per frame:
>   1. delivers what the modelled pool finished this frame (`plan.take`);
>   2. `let step = plan.step(eye, 1080, hole)`;
>   3. **applies it**: every `leave` key must be in `shown` (assert) and is
>      removed; every `arrive` key must not be in `shown` and must have
>      `plan.patch(key)` (assert), and is inserted;
>   4. queues `step.request`, in order;
>   5. records `shown` and asserts `shown == plan.drawn`.
> - 5.2, 5.3 and 5.5 compare **`shown`**, never `plan.drawn`, with `exact(eye)`.
>
> Add to 5.3's break table: `step` returns `arrive: vec![]` → red at the
> first split; `leave`/`arrive` swapped → red (the apply asserts).

### Major

**M1. Step 1 never measures the fix's target state, and its decision table
leaves gaps and a "build anyway" branch.**

A temporary edit that is ~10 lines in `far_streaming.rs` alone can produce the
state PR B aims for, live = converged exact selection on every frame, at zero
latency. That bounds what A+B can reach before anything is built. The owner's
re-scope asks to "show whether the lag is reselection distance, patch upload
pacing, or level choice". Each of these four runs isolates one term. Corrected
Step 1:

> ### Step 1 — diagnosis. Nothing is committed.
>
> Four variants, temporary edits to `crates/app/src/far_streaming.rs` only:
>
> | Variant | Edit | Isolates |
> |---|---|---|
> | base | none | today |
> | half | `RESELECT_BLOCKS` 32 → 16 | the owner's run; reported, does not decide |
> | tiny | `RESELECT_BLOCKS` → 1 | exact selection with today's latency |
> | sync | reselect whenever `selected_at != Some(eye)`, and in `update` generate missing patches on the spot until `select` asks for nothing new (`far::build`'s loop, over `self.heights`) | live = converged exact selection every frame, zero latency: PR B's target |
>
> Every variant also carries temporary `info` logging: per arrival, the ms and
> frames since its request and `in_flight.len()` when it was requested; per
> reselection, the blocks since the previous one and the trigger (moved /
> hole / arrival). The "sync" run's work times are meaningless; only its
> STALE counts.
>
> Build each variant once into its own `CARGO_TARGET_DIR`
> (`$SCRATCH/398/t-<v>`), then `git checkout crates/app/src/far_streaming.rs`.
> Run each binary from the repo root (as `cargo run` does),
> `RUST_LOG=info … --bench flight --eye 8,120,8 --look 1,-0.15,0.3
> --seconds 20`, **without `--frames`**. Alternate the variants (base, half,
> tiny, sync, base, …), three rounds, under M2's protocol. Then one extra base
> run *with* `--frames` for the dumps and the residue check; it is not counted.
>
> Decide (thresholds chosen, not derived):
> 1. **Premise.** In the base runs, ≥ 80% of the "neither" pixels hit beyond
>    1,400 blocks. Otherwise stop and report.
> 2. **Ceiling.** Every sync run has STALE = 0. Any sync run > 0: stop and
>    report, with dumps from one sync `--frames` run. The fix's target state
>    does not reach 0, so the residual is level choice or not the far terrain.
> 3. **A** (exact selection) is warranted if mean(base) − mean(tiny) ≥ 2 kept
>    frames.
> 4. **B** (generate ahead) is warranted if any tiny run is > 0, or the logged
>    p99 request→arrival exceeds one frame. Size `PREFETCH_MARGIN` so that the
>    lead at 1,024 blocks, 1,024·m/(1+m), is ≥ 2 levels × p99 latency ×
>    24 b/s × 2, and give the numbers. Feed the median generation time into
>    the harness (M4).
> 5. If neither 3 nor 4 holds, stop and report.
>
> Put the table (the four variants × three runs, latency p50/p99, the
> reselection-trigger counts) in PR B's body and a #398 comment.

This replaces the plan's outcome table, including the row-2 branch "do steps
2–3, then measure", which builds both PRs on a falsified hypothesis.

**M2. The measurement protocol compares unlike runs.**

- §6 says "Before = … step 1's base runs count", but step 1 runs with
  `--frames`. `--frames` creates `FrameKeeper` (`flight.rs:1263–1271`), whose
  own thread clones and PNG-encodes every candidate frame during the live
  pass. On the M3 it competes with the four far workers, so it inflates
  generation latency and work times. A before/after across that line flatters
  the fix.
- Runs are grouped (3 base, then 3 half, then 3 tiny) and not alternated. The
  Air is fanless and drifts over an hour.
- "Build once" is false once the source is edited, since `cargo run` rebuilds.
- "2 minutes after any build" is weaker than the lead's own protocol
  (`scratchpad/m3-remeasure.sh`: AC check, 6 minutes, quiet wait, a `pmset`
  snapshot per run).
- §6 takes one "before" run for the high, down and peak flights.

Corrected §6 preamble:

> Every run of record (step 1 and §6) follows the lead's `m3-remeasure.sh`
> shape:
> - **AC power**, checked with `pmset -g ps` (abort on battery), and a clean
>   tree.
> - Binaries built into per-variant `CARGO_TARGET_DIR`s and run from the repo
>   root.
> - **6 minutes** after the last build, and then until no other process uses
>   more than 25% CPU.
> - A `pmset -g ps`/`therm` snapshot per run.
> - Variants **alternated**, never grouped.
> - **Every run in the row** (⁸⁸: never the better of two).
> - **No `--frames`** in a run of record: a `--frames` run is a diagnostic.
>
> Before = `main` (`572113d` or later) without `--frames`; after = PR B.
>
> | Flight | Runs before / after |
> |---|---|
> | low, 1080p | 3 / 3 |
> | high, down, toward a peak | 2 / 2 each |
> | `walk --fly` | 3 / 3 |
> | `--bench 64` | 1 / 1 |
>
> The 720p low-flight rows are optional: the handover made 1080p the size of
> record, and the M3's idle time is shared tonight.

**M3. No CPU/frame budget, and no instrument that can see this change's
cost.**

- The handover's lesson: *"every render or perf issue carries a CPU/frame
  budget beside the GPU one"*. The re-scope adds *"keeping the far terrain's
  cost within the gate"*.
- The plan's cost checks cannot see the change:
  - `--bench 64` never runs `FarStreaming` (the plan says so itself).
  - The flight's `work p99` and the walk's `frame p99.9` print to 0.1 ms
    (`flight.rs:866`, `:1019`).
  - The prediction "±0.3 ms" and the stop at "~1 ms" are invented and not
    labelled.
- The plan's cost model ("proportional to movement") leaves out what B
  introduces. `prefetch_due` includes `arrived`, so once prefetched patches
  stream in continuously, *every* arrival costs three full `select`s plus the
  swap machinery.

Add to §6:

> **CPU budget** (handover rule; numbers chosen, not derived).
> `FarStreaming::update` runs on the main thread every frame.
> - **Standing still:** no `select` (5.5's counter).
> - **Flying:** over the 5.3 flight at `FAR_VIEW_RADIUS` (600 frames), the
>   summed `FarPlan::step` time is ≤ 3× PR A's (today's behaviour), and no
>   single `step` exceeds 1.0 ms on the M3.
>
> **Instrument:** an `#[ignore]`d `far_plan_cost_report` in
> `far_streaming.rs`, run with `cargo test --release -p cubara --bin cubara --
> --ignored --nocapture far_plan_cost_report`. The precedent is
> `rebase_cost_report`, PHASE3 §7.
> - Warm the memo first, so generation is not timed.
> - Print sum, mean, p99 and max of `step`, and the counts of exact, ahead and
>   behind selects.
> - Run it on PR A's commit and PR B's, alternating, 3 each; the median goes
>   in the PR and the footnote.
>
> The flight and walk frame terms remain the *gate's* terms. They are not
> this budget's instrument.

An optional lever, if the budget is missed: an arrival can change the exact
selection only if the arrived key is in `wanted`. A non-near node with an
unknown error is necessarily a `wanted` leaf, and a prefetched child of a
`wanted` leaf is never visited by the exact walk. So `exact_due` on arrival
can be "an arrived key ∈ `wanted`". Take this only with 5.3 green.

**M4. B shares one FIFO with urgent requests, and the harness's fixed
per-request latency cannot see queueing.**

- §4 step 8 orders requests *within* one step: wanted, then behind, then
  ahead. The pool is a single FIFO channel, though, so a prefetch backlog sent
  in step k is served before a wanted request sent in step k+1.
- §5.1 delivers every request exactly `latency_frames` later, which is
  infinite workers. 5.3 can therefore be green while the game is late.
- `latency_frames = 6`, and "not below 3", are invented.

Corrected §5.1 pool model:

> The pool is modelled with `workers` workers over one FIFO, in request order,
> each patch taking `service_frames` frames. `workers = 4` (the M3's
> `cores / 2`). `service_frames` is step 1's measured median generation time,
> in frames, rounded up (at least 1); the doc comment gives the number and
> its source.
>
> If 5.3 then goes red because a wanted patch waited behind prefetch requests,
> give `Pool` two queues. The wanted queue is served first: one
> `Mutex<(VecDeque, VecDeque)>` and a `Condvar` replace the shared `Receiver`.
> Model the same in the harness. Then the mutation "one queue for both" must
> go red.
>
> The §5.3 levers stay as written, but "lowering `latency_frames`" becomes
> "lowering `service_frames` below the measured value: not permitted".

**M5. `select`'s loop is rewritten without a pin that can fail (Rule 6), and
the root-ring change alters output (§1 wrong-claims 1 and 4).** `select` is
the one function the game, `far::build`, the screenshot, `far_census` and the
bench share. In practice the blast radius is tiny: edge patches 262 km out,
under fog. The problem is that the plan asserts "unchanged" and pins it with
an equality that cannot fail. Corrected §3a bullet:

> `select` computes `lo`/`hi` from `c.floor() as i64`; today's `c as i64`
> truncates toward zero, which skips a root inside the radius for eyes with a
> coordinate in (−1, 0) at `FAR_VIEW_RADIUS`. It walks `lo..=hi` as today.
> The ring just outside (`lo − 1`, `hi + 1`, in x or z) is **not walked**: for
> each of those roots, only `ground_distance(eye, box) − radius` goes into the
> slack. The output then equals today's at every eye except those
> negative-fraction eyes, where it gains today's missing edge patches. The PR
> says so.
>
> This is sound. A root enters the walked range only as its ground distance
> drops below `reach`, and its note goes to 0 as that distance reaches the
> radius.

Add to PR A, in `crates/world/src/far.rs`'s tests, before `select` is touched:

> `select_output_is_pinned` records `select`'s output for these eyes and
> radii, with `error_of` the fixed closure of 5.4:
> - eyes (8,120,8), (8,400,8), (8,450,8), (0,120,0), (−0.5,120,−0.5) and
>   (100000.3,300,−70000.6);
> - radii 20,000 and 262,144.
>
> It stores the key count and an FNV-1a hash of the keys in order, written out
> in the test, not `DefaultHasher`, whose algorithm is not promised across
> releases. Each must equal the constant recorded when the test was added.
>
> PR B leaves every entry unchanged except (−0.5, …) at 262,144, and explains
> that change (CLAUDE.md: a changed reference is explained, never silently
> regenerated).
>
> 5.4's eye set adds (0,·,0) and (−0.5,·,−0.5) at `FAR_VIEW_RADIUS`: that is
> where the root set changes.

PR A then touches two files: one is a test-only addition. Still small.

### Minor

- **m1. 5.5a's named break does not go red.** After settle the slack is
  positive, so removing `at != eye` changes nothing, and the counter stays at
  0. Correct the text:
  > Break: `exact_due = true` → the counter moves → red.

  To pin the `at != eye` guard, which matters only when the slack is exactly
  0, extract the due test as a pure `fn reselect_due(at, eye, slack) -> bool`
  and add `assert!(!reselect_due(Some(e), e, 0.0))`. Name the survivor in the
  mutation table: `slack * 0.01` stays green in every test, because it is
  correct and only costs work. M3's report catches it.
- **m2. Invented thresholds without the label.**
  - test parameters: `TEST_RADIUS = 4096`, `latency_frames = 6` (now M4), "not
    below 3", the "≥ 3 changes" guard;
  - measurement: "±0.3 ms" and "~1 ms" (now M3), "mostly beyond ~1,400" (now
    M1's 80%);
  - lever caps: 1/8 and 2.

  Each gets "chosen, not derived" in the PR, or a measured source.
- **m3. Two sources of truth for "what is drawn".** `FarPlan.drawn` sits
  beside `FarStreaming.slots`; `streaming.rs` itself warns that "two sources
  of truth for one fact is how they drift apart". B1's `shown == plan.drawn`
  assertion is the cheap guard. The cleaner form is
  `step(…, drawn: impl FnOnce() -> HashSet<PatchKey>)`, built from
  `slots.keys()` only when a step proceeds, as today (:425).
- **m4. A screen-height change does not trigger a reselection.** That is true
  today too, but with the slack nothing reselects until the eye moves past the
  old view's slack. `FarPlan` gets `screen_height` every step anyway: store it
  and treat a change like `hole_moved`. #394 (in the Mac's own queue) will turn
  it into a `Viewport` and rewrite `far_view` and the far tests; say so in
  §7's overlap.
- **m5. §7 overlap facts.**
  - #392 is trees/arena, not `streaming.rs`.
  - #318 owns the low flight's HOLES term.
  - #415 (open, M3 rows) will also take a `BENCHMARKS.md` footnote.
  - #397, if it lands on "a node meeting the far terrain", may edit
    `PatchHeights::skirt` in `world/src/far.rs`: the same file as PR B, a
    different function.
  - #385 (3.3a) turns the hole into a footprint. The slack's "the hole test
    does not depend on the eye" must be re-checked then; put that in the
    doc comment.
- **m6. PHASE3 doc format.** The doc asks each block's PR for its own
  `## §N` section. That makes this `## §9 3.0a — the far terrain is selected
  at the eye (#398)`, not "one paragraph under Later blocks".
- **m7. Test time and Rule 5.** Each generating test has its own memo; a
  shared static memo would trip `check-architecture.sh` Rule 2. Fold 5.5b
  into 5.3's flight, and keep 5.2's flights short. `exact()` re-implements
  `far::build`'s loop. Either give `far` a
  `build_with(gen, view, &mut HashMap)` that `build` wraps (PR B touches
  `far.rs` anyway), or cite the existing `converge` closure as precedent.
- **m8. PR A's "no behaviour change" has no evidence beyond review.** One
  low-flight run on PR A, under M2's protocol, inside step 1's base range is
  cheap. Alternatively, PR A's cost report must print the same select counts
  as a quick count on `main`.
- **m9. Use `cubara_world::World::new().seed()`** in the harness instead of
  the literal, so the two cannot drift.
- **m10. §1 wording on prefetch.** The note distinguishing far-patch
  generate-ahead from the rejected voxel prefetch is good. Add that B is built
  only on step 1's measured latency (M1 rule 4). That is what answers the
  owner's "prefetch is not the lever" with a measurement rather than an
  argument (handover: a diagnosis that flips cites the measurement that
  flipped it).

---

## 4. The seven questions, briefly

1. **Citations.** 9 stale (the table in §1, all from #406's merge plus one
   off by one) and 5 wrong (§1 list).
2. **Diagnosis.**
   - Supported: the band, the ruling-out of level choice at convergence, and
     the absence of upload pacing.
   - Not separated: reselection distance from latency. The timing fit
     ignores arrival-triggered reselections.
   - The halving run discriminates weakly, and nothing distinguishes latency
     from level choice. M1's "sync" run is the discriminating experiment.
3. **Contradictions.** None with the issue's decisions; decision 2's
   "never uploaded early" is followed and tested. Rules 2, 3 and 4 are fine:
   the counter lives on the plan, and the world crate stays pure. Against the
   handover's lessons:
   - Rule 6: M5.
   - The CPU/frame budget: missing (M3).
   - "Chosen, not derived": partly (m2).
   - The split rule: met, at 2 PRs and ≤ 2 crates each.
4. **Tests.** 5.3, 5.2 and 5.5b stay green with the renderer-facing half of
   the fix deleted (B1). 5.5a's named break survives (m1). The 5.4 "order"
   clause cannot fail (M5). The fixed-latency model hides queueing (M4).
5. **Measurement.**
   - AC power: only in §6, now everywhere.
   - Idle: 2 minutes, raised to the lead's 6.
   - Warm-up: the flight starts after the settle, which is acceptable.
   - Both runs recorded: required now.
   - 1080p: correct.
   - CPU/frame against main: absent; M3 adds it.
6. **Decisions left to Sonnet.**
   - the step 1 table's gaps and its "build anyway" branch (M1);
   - the budget (M3);
   - `latency_frames` (M4);
   - how to alternate builds (M2);
   - whether `--frames` runs count (M2);
   - what "mostly" means (M1);
   - the pin's form (M5).

   All are now decided in the corrected text.
7. **Overlap.** No code overlap with #397, #392 or #412 in the files the
   plan edits, and M1's step-1 edits stay in `far_streaming.rs`. The shared
   resources are `BENCHMARKS.md` footnotes (also #415) and the M3's idle
   time; the plan handles both. #394 and #385 are later collisions (m4, m5).

## 5. Verdict

**Execute with the listed corrections**: B1, M1–M5, and the minors that
change text (m1, m2, m5, m6). No re-plan is needed. The approach survives
review, and the corrections are confined to step 1, the harness, §6 and one
bullet of 3a. If M1's sync run comes back non-zero, the plan's own stop
condition applies, and nothing beyond step 1 gets built.

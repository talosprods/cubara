# Review of the #394 plan (`plans/394.md`): second-model, read-only

This review checks the plan against `origin/main` at `572113d` (2026-11-11), the commit Sonnet will
branch from. The plan was written at `3151b87`. Since then #406 has landed: `flight.rs` +553 lines,
`streaming.rs` +138 lines. Nothing was compiled or run for this review. Every number below is either
read from the repo or worked out in my head from the code's own formulas, and the text says which.

**Verdict: execute with corrections.** The overall shape holds: one `Viewport` type and one LOD
scalar, ring radii from a pure function, N passes over one resident set, and the live wiring left
until last. But one PR has to be split and one widened, and five findings are major:

- **Split PR 6.** Capacity work moves into its own PR ("PR 1b"), landed before PR 2. It covers the
  voxel arena and the far-patch slots, both growing on demand.
- **Widen PR 4.** The per-view step becomes one function. `Renderer::render`, `headless` and `bench`
  all call it.
- **Major findings.**
  - S1 measures the wrong number.
  - S2 is likely to stop the session at PR 2 for a problem that is already in production.
  - The "same capacity at 1080 rows" claim is false as written.
  - Multi-view is only built in `headless`.
  - The far-join scope item is missing.

---

## 0. Answers to the eight questions, in short

1. **Citations.** Accurate at `3151b87`. Line numbers have drifted for `flight.rs` and `streaming.rs`
   on current main, and the plan misses three sites: `flight.rs:618-627` (`pixel_ray`, a 4th copy of
   the FOV), `flight.rs:1613` (a new `DEFAULT_RING_SCHEDULE` use from #406), and the
   `crates/world/examples/*` and `crates/render/tests/offscreen.rs` callers of
   `schedule_for_radius`. Two of its doc statements are wrong (§3 "pixels at the view axis", §1 F-2).
2. **The PR cut.** The order 1 → {2, 3, 4} → 5 → 6 is right in spirit. PR 6 is the only PR that
   bundles ring growth with capacity growth. It is the largest PR, the riskiest, and it touches three
   crates. Split the capacity work out and land it **first**. Then no PR can ship ring growth without
   capacity, and the far-slot exposure that exists today gets fixed rather than tripping S2 (§2).
3. **The arena.** The conclusion holds and the proxy is wrong. Geometry grows with rows² (about 2.4×
   at 1664 rows), not with the desired-node count (about rows³, which gives "3×"). The last recorded
   occupancy is 45%, so the fixed 4M/6M arena overflows from about **1,600 rows**. That is **at the
   M3's own native fullscreen**. So no order may let derived rings reach the live game before
   capacity grows. The plan respects that only because PR 6 bundles the two (§3).
4. **The FarQuality question.** It is the **owner's**, as the plan says. But the plan should ask it
   *now* through the contact session, as a defect in decision 4's premise, and not discover it in
   PR 6 (§4).
5. **Seen-to-fail (b).** The planner's doubt is right on the code. The replacement is the resize
   defect, which *does* reproduce on main, plus a metric statement the lead must confirm (§5).
6. **Can the tests fail?**
   - One test cannot fail at all (PR 4's identity test).
   - Two mutations point at tests that are not specified.
   - One test measures a proxy for the quantity that overflows.
   - Six thresholds are invented and not labelled.
   - S1 watches CPU/frame for a change that only moves GPU time (§6).
7. **ARCHITECTURE.md.** Rule 5 is not met for N views: the per-view step already exists three times,
   and PR 4 multiplies only one of the copies. Rules 2 and 3 are fine (§7).
8. **Decisions left to the executor.** Listed in §8. One of them deviates from a binding issue
   decision: the T4 sweep may drop below 64 viewports, which decision 3 forbids. That must not be
   the executor's call.

---

## 1. Citations and claims about today's code

Verified on `origin/main` (`572113d`):

| Plan says | Code | Verdict |
|---|---|---|
| `node.rs:169` `DEFAULT_RING_SCHEDULE` | `crates/world/src/node.rs:169` | ✓ |
| `node.rs:450-463` `schedule_for_radius` | `:450-463` | ✓ |
| `streaming.rs:41-46`, `:373-378`, `:659` | `:41-46`, **`:377-380`**, **`:663`** | drifted by +4 (#406) |
| `flight.rs:1140,1182,1286` | **`:1340, :1386, :1499`** | drifted by about +200 (#406) |
| `arena.rs:57,61,79,90`; "radius-64 peak 2,488,824 indices" | ✓, but see §3: that peak is from #111, and the newer recorded occupancy is 2,705,796 indices / 1,803,864 vertices (`BENCHMARKS.md:1764`, `3b52c49`, 2026-09-13) | the number is stale |
| `render.rs:40` `FAR_PATCH_CAPACITY = 4096` | ✓. Its own doc: "measured selection at 262 km is about 1,800 … room for twice that" | ✓ |
| `far_streaming.rs:90` `FOV_Y`; `:97-116` `far_view`; `:374-402` `update` | ✓. `update` calls `far_view(eye64, renderer.size().1, …)` at `:402` | ✓ (see F-1) |
| `render.rs:246-255` `60f32.to_radians()` | ✓ `:246-248` | ✓ |
| `mesh.wgsl:72` `TAN_HALF_FOV`; `:66-68` overrides; `:125-128` `CRACK_SEAL` | ✓ (`:128`, used at `:172`) | ✓ |
| `render.rs:1463-1491` `mesh_pipeline_constants`; test `:2223`, `:2233-2256` | ✓ | ✓ |
| `lod_probe.rs:60-84` `ViewSpec`; `:159` `DEFAULT_SIZE.0 / view.height_px`; `:714`; `:1781`; `:2803`; `:2853` | ✓ | ✓ |
| `golden.rs:2104-2127` `far_terrain` | ✓ | ✓ |
| `far_census.rs:28` `SIZE` | ✓. It builds the full selection at `:300` (`far::build`), so the census can print the *selected* count S2 needs, not only the drawn count | ✓ |
| `main.rs:1125` `.unwrap_or((1920, 1080))` | ✓ | ✓ |
| `trees.rs:580-612` contributor cache | ✓ (`last_oak_contributors`/`last_box_contributors`): correct per view, re-uploads when views alternate | ✓ |

**Missed sites.** The executor would find all of these with grep, but the plan's claims are wrong
without them.

- **F-1 (minor). `flight.rs:618-627` `pixel_ray`** builds rays with `30f32.to_radians().tan()`. That
  is a fourth copy of the 60° FOV, and it feeds the HOLES/POPS detector used for gate item 0. §3's
  claim that `Viewport` is "the only place a size becomes an angle" is false until this is converted.
  The plan's own `1080|1920` grep cannot catch it. **Add to PR 1, step 5:**
  `pixel_ray(look, viewport: Viewport, x, y)` uses `viewport.tan_half_fov_y()` and
  `viewport.aspect()`. **Add to PR 5, step 5:** a second grep block, with a fixture, for
  `to_radians\(\)\.tan\(\)|FRAC_PI_3|60f(32|64)\.to_radians` in `crates/*/src` outside
  `viewport.rs`, test modules and `// viewport sample` lines.
- **F-2 (minor). `flight.rs:1613`** reads `node::DEFAULT_RING_SCHEDULE` for the far/voxel
  attribution added by #406. PR 6 step 5 must route it through the streamed schedule (flight already
  owns a `NodeStreaming`).
- **F-3 (minor). Other `schedule_for_radius` callers PR 6 must handle.**
  `crates/world/examples/geometry_census.rs:66`, `crates/world/examples/visibility_grain.rs:142` and
  `crates/render/tests/offscreen.rs:224` all call `schedule_for_radius`. CI builds `--all-targets`,
  so deleting the function from non-test code breaks the examples. Keep `truncate_schedule` public
  and give each of them a local sample.
- **F-4 (minor). `crates/world/tests/radius_64_smoke.rs:84-87`** mirrors
  `VERTEX_CAPACITY`/`INDEX_CAPACITY` ("If those change, check whether this bound should move too").
  It stays valid if `MEASURED` keeps today's values; say so in PR 1b.
- **F-5 (minor, but it is the definition of the engine's one scalar).** §3 says "'Pixels of the
  viewport' means pixels at the view axis, i.e. an angle divided by `rad_per_px()` (`fov_y /
  height_px`)". That is wrong. On a flat projection, an axial pixel subtends
  `2·tan(fov_y/2)/height_px`: 1.069 mrad at 1080 rows and 60°. `fov_y/height_px` (0.970 mrad) is the
  **mean** vertical pixel, about 10% finer than the axis at 60° and 1.65× finer at 120°.
  `ViewSpec::pixel_angle`'s doc ("exact on the view axis") has the same error. The scalar is fine:
  it reproduces today's radii exactly. Only its documented meaning is wrong. Corrected text for
  `Viewport::rad_per_px` and the PHASE3 section:

  > *The LOD scalar is the mean vertical pixel angle, `fov_y / height_px`. A pixel on the view axis
  > subtends `2·tan(fov_y/2)/height_px`: 10% more at 60°, so on the axis the bound is about 10%
  > conservative. Toward the frame's edges a pixel subtends less angle (×cos²θ horizontally). The
  > bound is angular on purpose: a bound in edge pixels grows without limit as a view approaches
  > 180°, and decision 2a has no FOV limit.*

  Also drop the "would make the LOD depend on view direction" argument in §3. A minimum over each
  view's whole field is direction-free. The boundedness argument alone carries the decision.

---

## 2. The PR cut

### Size and crate count (my estimates)

| PR | Crates | Est. lines | Note |
|---|---|---|---|
| 1 | render, app | ~600 | about 25 projection call sites (`look_view_proj`/`view_proj` × 23 on main) |
| 2 | world, app, render(tests) | ~700 | |
| 3 | world, app (+docs) | 1,000–1,300 | 52 `ViewSpec`/aspect sites in a 3,228-line `lod_probe.rs`, plus T1/T4 sweeps and the proposal rewrite. Near the limit: put the docs rewrite (step 6) in PR 5 if over. |
| 4 | render (+app if widened, below) | ~700 → ~900 | |
| 5 | render, app, scripts (+world test literals) | ~500 | |
| 6 | render, app, world | **~1,000–1,300** | arena refactor + streaming + settle + bench sweep + 8 call sites + tests + rows |

### Does each PR stand alone and keep behaviour where it says it does?

- **PR 1** ✓. It changes a `const` to an `override` in the shader, which the plan measures. But see
  S1 in §6: it measures the wrong quantity.
- **PR 2** ✓ for drawing at a fixed size. Two corrections:
  - **S2 is likely to stop the session here for a pre-existing condition (major, §2a).**
  - The far reselect on resize happens per frame during a drag. Feed it the same `SettledViewport`
    PR 6 adds, so one policy decides when a viewport counts as changed (minor). That means moving
    `view_settle.rs` from PR 6 into PR 2.
- **PR 3** says "no production change", but its `truncate_schedule` extension changes
  `schedule_for_radius(r)` for `r > 64`: today it returns `[10,18,32,64]`, afterwards
  `[10,18,32,r]`. That changes what `--bench N` and `--screenshot --radius N` build for N > 64.
  **Fix (minor):** keep `schedule_for_radius` exactly as it is today, and put the
  extend-to-radius rule only in the new `truncate_schedule`, used from PR 6 on. Or state the change
  in the PR.
- **PR 4** stands alone, but under-delivers 2a and Rule 5 (§7).
- **PR 5** ✓. The crack seal is 0.0516 px × 0.9696 mrad = 5.003e-5 at the gate, which is not
  bit-equal to today's `5e-5`. The plan says so; it is harmless.
- **PR 6** claims "same capacity at 1080 rows", which is false as written (§3, finding C-2).

### Order: capacity before rings (major)

The plan keeps derived rings out of production until PR 6 and bundles capacity growth into the same
PR. That is the only reason no intermediate state overflows (§3). Bundling the riskiest change with
the largest refactor in one ~1,200-line, three-crate PR is the wrong place for it. **Re-cut PR 6:**

- **PR 1b — capacity as a value, growing on demand. Behaviour is unchanged at the gate. Lands after
  PR 1.** Render crate only, plus the app's far-insert call site.
  - `ArenaCapacity` replaces the four constants. `MEASURED` holds today's values.
  - **Grow on exhaustion, with no model:** when a slab allocation fails, allocate buffers 1.5× the
    size (capped at `device.limits().max_buffer_size` and at 65,536 nodes), `copy_buffer_to_buffer`
    the used prefix, extend both `SlabAllocator`s with a free tail, and rebuild the origins bind
    group. Offsets and node indices are unchanged, so **nothing re-streams**.
  - Give `FarTerrain` the same treatment: when `insert` would return `None`, grow its slots.
  - Tests:
    - a GPU test in `mesh_arena_integration.rs` inserts past `MEASURED`, checks that `usage()`
      reports no exhaustion, and reads back a frame equal to the same scene drawn in an arena built
      big from the start;
    - the same for far slots.
  - Mutations: grow without copying (the frame is red); grow without extending the allocator (the
    insert after growth fails).
  - Gate pair on the M3: the canonical rows must not move. At 45% occupancy nothing grows at 1080
    rows.
- **PR 6 (now smaller)** — the live game streams derived rings. No arena resize-and-reset path is
  needed.

Why grow-on-demand rather than the plan's `for_footprint` plus `ensure_arena_capacity` plus
`streaming.reset`:

1. The plan's path **drops every resident node** when the window grows: going fullscreen on the M3
   would show a reload with holes. Block 3.0a's goal was *"never a hole"*.
2. It needs a capacity model (area scaling, the `+16`, the 0.25 margin), and the model has to be
   proven. On-demand growth follows the real demand: mountains, caves, any resolution.
3. It also fixes a far-slot exposure that exists today (§2a), instead of stopping on it.

If the lead prefers the plan's `for_footprint` design anyway, at least apply C-2 and C-3 from §3.

### 2a. S2 will probably fire at PR 2, and the cause is already in production (major)

`FarStreaming::update` already sizes the far view from the real window:
`far_view(eye64, renderer.size().1, …)` at `far_streaming.rs:402`. PR 2 does not change the far
patch count at any size. Its only behavioural change is reselecting on resize.

The distance split keeps quads at a constant angle, so the selection grows with
`split_ratio² ∝ rows²`:

- With the "about 1,800" that `render.rs:37` records, 1080 → 1664 rows gives ≈ 4,270 patches,
  which is already over the 4,096 slots at the M3's native fullscreen *today*. At 2160 rows it is
  ≈ 7,200.
- With the ≈ 730 implied by `PROPOSAL_FAR_VIEW.md:184` (1.5M triangles at 16 px / 2 px ÷ 2,048 per
  patch), it fits at 2160 rows.

Either way, the census at 3840×2160 has a real chance of tripping S2. And S2 says "stop", which
would end the overnight session at block 2 of 6, over a capacity that is engineering's to size.
Meanwhile the plan *does* size the voxel arena itself in PR 6: the two cases are treated
inconsistently.

**Corrected S2:**

> *S2 (PR 2): the census prints, per sweep size, the **selected** patch count
> (`far::build(..).len()`) as well as the drawn count. A selection over the far slots is recorded
> as a finding that already exists on `main` (the far view already follows `renderer.size()`). PR 1b
> grows the slots on demand, so the census also asserts that no sweep size logged "every slot is
> taken". It is not a stop.*

---

## 3. The arena estimate, re-derived

All arithmetic here is by hand from the code's formulas.

**Ring radii.** The plan's formula is `R_L = ceil(2^(L+1) / (N_L · p) / 16)` with
`N = [12.90, 14.33, 16.12, 16.12]`:

- **1080 rows** (`p = 9.696e-4`): 9.9935, 17.9925, 31.989, 63.98 → `[10, 18, 32, 64]` ✓
- **1664 rows** (`p = 6.293e-4`): 15.40, 27.72, 49.29, 98.57 → **`[16, 28, 50, 99]`**
- **1200 rows** (the M3's default 1600×1200 window): `[12, 20, 36, 72]`
- **2160 rows:** `[20, 36, 64, 128]` ✓
- **180 rows**, with the floor: `[2, 6, 14, 30]` ✓

**What grows how.**

- **Desired nodes**, what the plan's Python counted, are dominated by **volume**:
  - the level-3 cube holds `(2·R3/8)³` nodes, which is proportional to R3³;
  - each inner ring is a squashed cylinder, roughly `π(R_L+2^L)²·(R_L/2+2^L)/8^L` nodes, which is
    proportional to `(2^L/p)³/8^L`, i.e. to `1/p³`.

  So desired nodes grow with rows³. My own volume estimate gives ≈ 15,400 at 1080 rows (plan:
  14,265), ≈ 52,000 at 1664 (plan: ≈ 45,000) and ≈ 101,000 at 2160 (plan: 95,600). The plan's
  3.2× and 6.7× are this volumetric ratio.
- **Arena geometry** lives on the terrain surface (a 2-D sheet, plus caves). Ring L contributes
  about `π(R_L² − R_{L−1}²) · 256/4^L` surface cells, which is proportional to `(2^L/p)²/4^L`,
  i.e. to `1/p²`, **for every ring alike**. So vertices and indices grow with **rows²**. Per-ring
  area ratios at 1664 rows: 2.56, 2.42, 2.44, 2.39. That is **≈ 2.4×** (the plan's figure is 3×).
  At 2160 rows it is **≈ 4.0×** (the plan's is 6.7×).
- **Resident nodes and draws** also grow with surface: about 1,568 × 2.4 ≈ 3,800 nodes and 2,219 ×
  2.4 ≈ 5,300 draws at 1664 rows. Both are far below 16,384, so **neither is binding**. Only
  vertices and indices are.

**Headroom.** The last recorded occupancy is v 1,803,864 / 4,000,000 and i 2,705,796 / 6,000,000:
45% for both (`BENCHMARKS.md:1764`, M3, `3b52c49`, 2026-09-13). Overflow therefore comes at
geometry × 2.22, i.e. at **1080 × √2.22 ≈ 1,609 rows**:

- 1,664 rows gives ≈ 2.4 × 1.80M ≈ 4.3M vertices, which is over 4M;
- 1,600 rows (fullscreen below the notch) gives ≈ 2.3 × 1.80M ≈ 4.1M, also over.

This is before counting the held-while-replaced nodes and the slab fragmentation.

**"Overflows near 1664 rows" therefore holds**, on a better proxy: from about 1,600 rows, i.e. at
the M3's own native fullscreen. The 45% figure is two months old (R1b, R3a and phase-3 terrain have
landed since), so the crossover could sit anywhere from about 1,530 rows (50% occupancy) to about
1,710 rows (40%).

**C-1 (minor, free).** Take today's occupancy from PR 1's own gate pair: `--bench gate` prints
`ARENA:` (`bench.rs:1261`). Record it in PR 1's footnote with the formula
`overflow rows ≈ 1080·√(1/occupancy)`.

**Consequence for the order.** With fixed capacity and derived rings, the live game overflows on the
machine the gate binds on. That would be a **blocker** for any order that ships ring growth without
capacity growth. The plan as written does not do that (PR 6 bundles the two), so it is not a blocker
here. PR 1b removes the hazard structurally.

**C-2 (major, if `for_footprint` is kept).** PR 6 step 4 calls
`for_footprint(2·R3 + 1 + 16, …)`, but `MEASURED_COLUMNS = 129·129`. At the gate viewport the
footprint is 2·64 + 1 + 16 = 145, which gives **(145/129)² = 1.26× `MEASURED`**. That contradicts
"same capacity at 1080 rows" (PR 6, Measurement), and the test `for_footprint(129) == MEASURED`
passes regardless. **Fix:** a single `fn footprint_side(schedule) -> u32`, used both by the call
site and by the normalisation, and pin it with
`for_footprint(footprint_side(&SAMPLE_1080)) == MEASURED  // viewport sample`.

**C-3 (major, if `for_footprint` is kept).** The capacity sweep test checks
"desired nodes × 0.25 ≤ `for_footprint(..).nodes`". Desired nodes grow with rows³, but node
capacity is not what overflows. Vertices and indices are, and they grow with rows². The 0.25 is
invented. The test can pass while the vertex arena overflows. **Replace it with:**

- (a) pure unit tests of `for_footprint`: scales with area, never below `MEASURED`, caps respected;
- (b) an `#[ignore]`d measurement that meshes `desired_nodes_3d` at the derived schedule for 1080,
  1664 and 2160 rows at the gate ground eye, using `crates/world/examples/geometry_census.rs`'s
  machinery (it already counts quads GPU-free), and prints vertices and indices against capacity —
  run once, numbers in the PR;
- (c) S6(ii), the bench `ARENA:` line at every sweep size, which remains the real check.

---

## 4. The FarQuality question (plan's Q1): the owner's, and ask it now

**Recommendation: the owner's.** Three reasons:

1. **The issue reserves it explicitly.** Its out-of-scope section says: *"The ring schedule's
   constants `N_L` (what step size is acceptable is the owner's eye and the gate)"*. A quality step
   that scales `N_L` is precisely a choice of a different acceptable step per quality.
2. **It changes what the gate means.** Today `FarQuality` only changes `quad_px/error_px/floor_px`
   (`far_streaming.rs:56-68`). `GATE: MET at Medium` (`bench.rs:1064`; the M3's ⁸⁸ rows) means
   "today's rings plus Medium's far budgets". If Medium also coarsened the rings, the canonical gate
   scene would depend on the tune result. Equally, the voxel scene the gate's 1,118/1,148 FPS was
   measured on would no longer be what Medium draws. Defining the gate is the owner's call
   (CLAUDE.md: "Do not move the exit gate"; standing permissions: "the exit gate … still goes to
   him").
3. **Decision 4 rests on a false premise.** It says *"`--bench tune` / the far-quality setting
   (which already pick the budget per PC) are what trade it"*. No existing setting trades ring
   cost. CLAUDE.md calls this "a defect in the issue — fix the issue first, or stop and ask if it is
   a gameplay decision". It is one.

**The plan's handling is right in substance** (PR 6 measures; it adds no lever). Two corrections:

- **Ask now, not at PR 6.** Send Q1 to the contact session at the start of the run. Per
  CLAUDE.md's multi-session rules, nobody waits on a question: keep working while it is open. Frame
  it with the derived numbers from §3. Geometry grows ≈ 1.3× at the M3's 1200-row default window,
  ≈ 2.4× at 1664 rows and ≈ 4× at 2160 rows; desired nodes grow ≈ 3.2× and ≈ 6.7×.
  Engineering can lay out the options without choosing one:
  - (a) rings follow the pixel and nothing trades them;
  - (b) quality steps scale `N_L`;
  - (c) a separate ring-detail step that `--bench tune` tries after the far quality, with the gate
    holding it at 1.0.
- **S6(iii) is nearly vacuous on the M3.** The M3 Air panel is 60 Hz. At 1080 rows the gate runs
  1,118 FPS, so even 2.4× the geometry and 2.4× the pixels leaves it far above 60. Say so, so
  nobody mistakes "held refresh" for "cost absorbed". The real cost question is what the 1,000-FPS
  gate would say at 1664 rows. That is Q1 territory: record it and do not judge it.

---

## 5. Seen-to-fail (b): the planner is right, and here is the replacement

**What the issue asks.** *"a 32:9 viewport at 60° vertical: today at least one far patch inside the
horizontal frustum is culled or over its `max_error` (the far view is derived from height only)"*.

**Culled: it does not reproduce.**

- `far::select` (`world/src/far.rs:146-203`) is eye-centred, with no frustum.
- Per-frame culling (`render/src/far.rs:462-469`) uses `self.frustum`, built at
  `render.rs:1316-1317` from `camera.view_proj(self.scene.aspect(), …)`: the real aspect.
- The far pixel comes from `renderer.size().1`.

Nothing in the far path assumes 16:9.

**Over its `max_error`: this depends on what "pixel" means.**

- In the angular metric the plan adopts (F-5), it holds by construction at every aspect.
- In screen pixels *at the patch's position*, it is exceeded today at every aspect, not only 32:9.
  For a vertical error, the stretch relative to the mean pixel is `0.907/cos θ`. At the edge of a
  16:9 view that is ≈ **1.30×** (θ = 45.75°); at the edge of a 32:9 view it is ≈ **2.07×**
  (θ = 64.03°). It grows without bound toward 180°.
- No eye-centred selection can satisfy (b) in screen pixels without using the finest edge pixel of
  every view, which is unbounded under 2a. So under that reading, (b) contradicts decision 2a.

**Replace (b) with three things:**

1. **The far-path defect that does reproduce on `main`.** `FarStreaming::update` returns early
   unless `moved || hole_moved || stale` (`far_streaming.rs:383-398`). After a resize, the old pixel
   budget therefore stays until the eye has moved 32 blocks. Sequence:
   - Extract `needs_reselect(moved, hole_moved, stale, pixel_changed)` **with main's body**, which
     ignores `pixel_changed`.
   - Commit `a_resize_reselects_at_once` and **see it red**.
   - Then fix it.

   That is an honest before/after for "the far view is derived from the viewport". The plan has the
   test, but only as mutation (c).
2. **The per-view far bound over the sweep** (PR 2, step 6). Two fixes so it can fail when
   production is wrong:
   - **Sample the test's rays independently of the projection matrix**, by trigonometry from
     `fov_y` and `aspect`. Rays built by inverting the same `view_proj` agree with any wrong
     frustum.
   - **Apply mutation (a) to production** (`Viewport::projection` or `aspect()` at a fixed 16/9),
     not to the test's own frustum.
3. **Q4 to the lead, with the metric included.** The amended (b) should state the angular
   definition from F-5, so that "pixels of each view" (decision 1 and 2a) has one meaning. The
   plan's Q4 asks only whether to amend.

---

## 6. Tests, goldens, budgets and thresholds

### Tests that cannot fail, or that are not specified

- **T-1 (major). PR 4 `one_view_through_render_views_is_the_same_frame`.** `render()` will call
  `render_views` with one direction (step 3), so this compares the function with itself and cannot
  fail. **Replace with** `views_do_not_leak_into_each_other`:
  `render_views(&[a, b, c])[1]` is byte-identical to `render_views(&[b])[0]`, and likewise for
  index 2. It catches a stale indirect list, an unflushed camera uniform, and the tree cache serving
  the previous view. Mutation (b) (one submit) turns it red as well.
- **T-2 (minor). PR 3 mutation (d)** names a test, `columns_follow_the_viewport_width`, that no step
  writes. **Add it to PR 3, step 3:** `m4::Camera`'s column count at `Viewport::new(3840, 1080)` is
  2× its count at the gate sample, and `frustum_at`'s horizontal extent matches `fov_x()`.
- **T-3 (minor). PR 5 mutation (a)** needs `crack_seal` asserted in the constants test. **Add to
  PR 5, step 1:** `mesh_pipeline_constants_carry_…` asserts `crack_seal` at two viewports, and that
  it halves when the height doubles.
- **T-4 (minor). PR 1 `tan_half_fov`.** If `mesh.wgsl` keeps reading the old `const`, nothing goes
  red: every golden is at 60°. Say plainly in the PR that the line needs no assertion while
  `fov_y` cannot change, as CLAUDE.md asks.
- **T-5 (major).** C-3 above: the capacity guard tests a proxy.

### Goldens

- No existing golden is blessed: ✓.
- New goldens are named and explained: ✓ (PR 4: one; PR 5: three).
- **PR 5's pins render at golden-scale region radii** (< 10 chunks). For most pin sizes,
  `truncate_schedule(&ring_schedule_for(..), r)` collapses to `[(0, r)]`. At 1280×720,
  R0 = 7 ≥ r = 6, so the pin is level 0 only; only the 1920×540 pin (R0 = 5) shows a level-1 ring.
  The pins then show projection and Hor+, not rings. **Say that in each pin's explanation**, so
  nobody reads them as ring pins.
- **PR 4's seam threshold is "2.5× the value measured on this scene", measured on Metal.** Goldens
  are blessed on one backend and checked on lavapipe and DX12 (`golden.rs:142-203`'s table shows
  cross-backend deltas up to 0.21% of pixels and a max channel delta of 173). **Calibrate the seam
  threshold on the three CI backends** from the `Golden image deltas` step's `--nocapture` output,
  not on the M3 alone, and label it "chosen, not derived (2.5× the worst backend's measured seam)".

### Per-PR CPU/frame budget

**S1 measures the wrong number (major).** PR 1 and PR 5 change `mesh.wgsl`. The M3 sensitivity they
guard against is **GPU-side**: the occupancy cliff that `render.rs:1453-1462` measured at
~400 FPS. On the gate scene, CPU/frame is only half the frame: ground 0.444 ms against
1/1,118 s = 0.894 ms. And Metal reports no GPU/frame. So S1 can stay green while the M3 gate goes
red. The margin there is 1.12× (1,118/1,148 FPS against 1,000).

**Corrected S1:**

> *S1 (PR 1, PR 5, and PR 1b and PR 4 if they touch the bench path): on the M3, two runs of
> `--bench gate`, both recorded. Stop if either run's line is not `GATE: MET at Medium`, or if at
> any gate eye **both** runs' FPS fall below the lower ⁸⁸ run by more than ⁸⁸'s ±4% run-to-run.
> CPU/frame is reported beside it. It is not the criterion here, because the change is in the
> shader and Metal has no GPU/frame. (The 4% is ⁸⁸'s measured spread; the rule is chosen, not
> derived.)*

Other budget points:

- **PR 4, if widened as §7 says,** changes the bench's per-frame path, so it needs the same gate
  pair.
- **PR 2** changes only a per-frame comparison. A sentence saying so is enough.

### Invented thresholds to label "chosen, not derived"

| Threshold | Where |
|---|---|
| S1's 10% | replaced above |
| `SettledViewport`'s 30 frames | PR 6 |
| ×0.25 | PR 6 (C-3) |
| 2.5× seam | PR 4 |
| 30 s / 60 s / 120 s wall-time fallbacks | PRs 2 and 3 |
| `MIN_RING_NODES = 2`, then 3 | PR 3; derived from the balance rule, but the 3 retry is a choice |

The ring constants `N_L` are **derived** (back-solved from today's radii): ✓, say so. The
`CRACK_SEAL_PX = 0.0516` value is derived from today's 5e-5: ✓.

---

## 7. ARCHITECTURE.md

### Rule 5 with N views (major)

The per-view step — camera, frustum, three `prepare`s, `set_camera`, `encode_scene` — **already
exists three times** on `main`:

- `Renderer::render` and `update` (`render.rs:1143-1151`, `:1316-1335`);
- `headless.rs:463-532`;
- `bench.rs:743-805`.

`encode_scene` is single, which is all `check-single-render-path.sh` checks. PR 4 turns only the
`headless` copy into a loop. Consequences:

- The window's `Renderer` cannot draw N views after #394, although 2a says *"the renderer draws a
  frame as one or more views"* and the issue's scope is *"makes the engine correct for N views"*.
  Only "several real windows" is deferred.
- The windowing follow-up would have to write the per-frame/per-view contract a second time, in
  `Renderer`.

**Corrected PR 4, step 3:** add to the render crate one function, e.g.
`SceneRenderer::draw_view(&mut self, device, queue, encoder, target, view: &View, eye, anchor, arena, trees, far, overlay…) -> u32`.
It does `set_camera` → `Frustum::for_view` → the three `prepare`s → `encode_scene`.

- `Renderer::render` loops over `self.views` (one in the window today).
- `headless::render_views` loops and submits once per view.
- `bench.rs` calls it once per frame.

Extend `check-single-render-path.sh` with one block: `\.prepare\(&` on `arena`/`trees`/`far`
appears only in that function. Add a fixture under
`scripts/selftest/single_render_path_prepare/`. Expose `Renderer::lod_pixel_angle()`
(`= lod_pixel_angle(&self.views)`), and have PR 2's far streaming and PR 6's ring streaming read
**that**, not `viewport().rad_per_px()`. Then the multi-window follow-up touches neither. The cost is
a gate pair for PR 4 (S1); it fits in two crates.

### Other rules

- **Rule 2:** ✓. `Viewport` is passed in, `SettledViewport` is a field, and `sweep_seed()` only
  reads an environment variable (test input, not mutable global state). Keep `sweep`/`sweep_seed`
  documented as test support.
- **Rule 3/4:** ✓.
  - World takes `pixel_angle: f64`, never `Viewport`.
  - `render/tests/viewport_sweep.rs` uses the existing `cubara-world` dev-dependency.
  - Moving `Detail::{LOW, MEDIUM, HIGH}` into `cubara-world::far` is the only place both `app` and
    the render goldens can share them. Fine.
- **Rule 5, viewport-change policy (minor).** PR 2 reselects the far terrain on every pixel change;
  PR 6 settles over 30 frames for rings. That is two policies for one event. Use one
  `SettledViewport`, moved into PR 2, for both.

---

## 8. Decisions left to the executor

| Decision | Plan | Should be |
|---|---|---|
| **T4 sweep under 64 viewports if over 120 s** (PR 3, step 5) | executor drops to 8 and notes it in the PR | **Not the executor's.** Decision 3 is binding ("at least 64 per test run"). Engineering alternative: keep 64, mark it `#[ignore]`, and add a CI step `cargo test --release -p cubara -- --ignored <sweep>` (app tests are opt-level 0 in CI; only `cubara-world` is opt 3, `Cargo.toml:48`). CI is engineering's. If even that does not fit, it is a lead question. |
| Default sweep seed | fixed constant, other seeds by hand | Acceptable, but consider `CUBARA_SWEEP_SEED` set from the commit SHA in CI: new viewports per commit, reproducible from the printed seed. Engineering's call; state it. |
| Sweep corners | 320×180, 7680×2160, 2160×2160, 640×180 | Add 320×320, the smallest 1:1, which is in range. |
| `rad_per_px()` precision | `rad_per_px_y as f64`, an f32 division | Compute `fov_y as f64 / height_px as f64`. Ring margins are about 6e-4 relative, so it is safe either way, but f64 makes the gate-sample test and production compute the same thing. |
| Stale search answers after `set_schedule` (PR 6) | not addressed | A search started under the old schedule can return afterwards and be applied (`SearchDone` carries `generation` and `center`, not the schedule; `streaming.rs:347-355`). Carry the schedule in `SearchJob`/`SearchDone` and drop answers whose schedule differs from the current one. |
| Far-join tolerance | **absent** | **Scope item 2 and decision 3** ("no node/far seam above tolerance at the join"). Major gap. Add to PR 6 a pure property over the sweep: at the hole edge, the voxel top-level cell (8 blocks at `R3·16` blocks) in `rad_per_px` pixels and the far quad bound there are both the same number of pixels at every sweep viewport, up to `ceil`'s rounding (that resolution-invariance is not an invented threshold). Seen to fail with `DEFAULT_RING_SCHEDULE` forced: the voxel cell at the join grows from 8.06 px at 1080 rows to 16.1 px at 2160. |
| What PR 2 does if S2 trips | stop | Record and continue (§2a). |
| `+16` footprint margin | call site only | One function (C-2), or moot under PR 1b. |
| Standing-permission conditions | AC power, quiet, no process holding a core | List all of CLAUDE.md's: also **"the machine is not in use by him"**, and the power and thermal state written in the row itself. Or simply point to CLAUDE.md § Standing permissions. |
| UI split, Q2, Q3 | lead / owner | ✓ correctly routed. |
| Linux/Windows sweep rows | other sessions | ✓. #394 stays open (done-when requires them). |

---

## 9. Ranked findings

**Blockers:** none in the plan as written. The arena overflow at the M3's native fullscreen would be
a blocker for any order that wires rings before capacity. The plan avoids that only by bundling the
two in PR 6 (§3).

**Major**

1. **Re-cut.** Capacity as grow-on-demand (arena and far slots) in **PR 1b**, before PR 2.
   - PR 6 wires rings only.
   - This removes the resize-reset reload (holes on going fullscreen), the footprint model, C-2,
     C-3 and S2's likely stop.
2. **S1 watches CPU/frame for a GPU-side shader change** on a GPU-bound machine with a 1.12× gate
   margin. Use the corrected S1 (§6).
3. **S2 will likely stop the session at PR 2** for a far-slot overflow that already exists in
   production (`far_streaming.rs:402` uses `renderer.size().1`). Record it; do not stop; PR 1b fixes
   it (§2a).
4. **PR 4 builds N views in `headless` only.** The per-view step is triplicated today. Make it one
   function used by `Renderer`, `headless` and `bench`, with a check and a gate pair (§7).
5. **The far-join tolerance property is missing** (scope item 2, decision 3); add it to PR 6 (§8).
6. If `for_footprint` is kept:
   - **C-2:** `+16` gives 1.26× `MEASURED` at the gate, so "same capacity at 1080 rows" is false.
   - **C-3:** the capacity guard tests desired nodes × 0.25 (∝ rows³, invented) instead of
     vertices and indices (∝ rows², the quantity that overflows).
7. **PR 4's identity test cannot fail** (T-1). Replace it with the view-leak test.
8. **The T4 sweep's fallback below 64 viewports overrides binding decision 3** (§8). Use the
   release-mode `--ignored` CI step instead.

**Minor:** F-1 to F-5 (`pixel_ray` FOV copy plus a FOV grep; `flight.rs:1613`; the examples and
`offscreen.rs`; the `radius_64_smoke` mirror; the "view axis" wording); T-2 to T-4; C-1 (record
`ARENA:` in PR 1); PR 3's `schedule_for_radius(r > 64)` change; one settle policy for far and
rings; stale search answers; sweep corner 320×320; f64 `rad_per_px`; labels on the six invented
thresholds; PR 5's pins do not show rings (say so); seam threshold calibrated across backends;
standing-permission conditions incomplete; line drift on `flight.rs`/`streaming.rs` (+4 to +200
since `3151b87`).

**Owner questions:** Q1 is the owner's; ask it now, via the contact session, framed as decision 4's
false premise, with the §3 numbers. Q4 goes to the lead with the angular metric included (§5).

**Verdict: execute with corrections.** Split PR 6 into PR 1b (capacity, grow-on-demand, landed after
PR 1) and a smaller PR 6 (rings). Widen PR 4 to a single per-view function. Apply the corrected S1
and S2. Add the far-join property. Every other item is a text edit to the plan.

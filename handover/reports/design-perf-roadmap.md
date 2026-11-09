# Engine performance roadmap: design study (2026-11-09)

Read-only study. Sources: `ARCHITECTURE.md`, `ROADMAP.md` (phase 3, "Recorded, 2026-09-26"), `docs/PROPOSAL_FAR_VIEW.md` §2/§3/§6b/§6c, `BENCHMARKS.md` (the newest rows and footnotes ⁴⁵ ⁵² ⁵⁵ ⁵⁷ ⁶⁰ ⁶²–⁷⁵), issues #42 #32 #28 #33 #130 #131 #132 #271, the render crate, `crates/world/src/far.rs`, `crates/app/src/far_streaming.rs`, the F5 branch `origin/feat/far-haze`, and `.claude/autonomous/reports/audit-render.md`. I added the coordinator's data from 2026-11-09: the Windows gate after #298, and F6(c)'s voxel-triangle cuts.

## 1. Where the frame goes

| Machine, eye (far terrain at High) | FPS | CPU/frame | GPU/frame | Far triangles |
|---|---|---|---|---|
| M3, ground | 826–873 (warm, §6b); 597 (⁶⁷, 19% battery) | not recorded (0.266 ms voxels only, ⁶²) | not available (#267) | ~720k |
| M3, hill | 878–1,105; 809 | (0.189 ms voxels only) | not available | ~728k |
| M3, flight | 1,546–1,634 | – | not available | ~548k |
| RTX 4060, ground / hill / flight (after #298) | ~3,650 / 3,617 / 5,427 | 0.07–0.10 ms | 0.15–0.23 ms | same counts |
| GTX 1060, ground (⁶⁵) | 1,610 | 0.340 ms | 0.573 ms | same |

**Both tight machines are GPU-bound at every eye.** CPU-side work cannot raise gate FPS on the M3 or on the 4060. It still lowers CPU/frame, which counts toward the owner's "efficiency".

**The far terrain is limited by geometry, not pixels, on the M3.** I subtracted the voxel-only frame and divided by far triangles, using the §6b warm numbers:

| Eye | Far cost per frame | Per triangle |
|---|---|---|
| Ground | (1/850 − 1/3,100) ≈ 0.85 ms | ~1.2 ns |
| Hill | (1/990 − 1/3,500) ≈ 0.72 ms | ~1.0 ns |
| Flight (whole frame, minus ~0.1 ms fixed) | ~0.5 ms | ~0.9 ns |

- At ground the far terrain is a thin band near the horizon; in flight it is most of the screen.
- The cost per triangle is the same either way, so pixel shading is not what it pays for. It pays for vertex shading, triangle setup and tiling. On a tile-based GPU, tiling means sorting each triangle into screen tiles, and every vertex output is written to memory.
- §6b's "several triangles in each pixel" is the mechanism. The F4 fragment shader is trivial.

**The M3 is specifically weak at this:**

| | Voxels only, ground | Far cost, ground |
|---|---|---|
| M3 | 3,121 FPS | 0.85 ms |
| GTX 1060 | 2,380 FPS | 0.21 ms |
| RTX 4060 | – | 0.07 ms |

Tiny triangles on a tile-based GPU are the M3's weak point.

**What the M3 needs.** High holds 1,000 FPS at ground once the far terrain's cost falls by 18–25%, and at hill once it falls by up to 16%. Flight already passes. Medium read 907 / 1,225 / 2,175 in ⁶⁷ on a depressed machine. Scaled by High's warm-to-depressed ratio (~1.4×), it probably already passes when the M3 is warm and charged. That is unconfirmed.

**The owner's play goal is a different problem.** The goal is never dropping below the monitor's refresh rate. Averages are far from that limit: 1.2 ms against 6.9–16.7 ms per frame. The risk is the tail (hitches), so it gets its own track (T).

**Resolution scaling.** Far-terrain detail is set in pixels, so its triangle count grows roughly as the resolution to the power 1.3–2. `--bench tune` measures at the window's size, so the quality chosen for play accounts for that. A High that passes the gate at 1080p is no promise at the M3's native resolution.

## 2. Ranking (M3 gain per unit of effort)

| Rank | Candidate | Bottleneck it attacks | M3 gain | RTX 4060 gain | Effort |
|---|---|---|---|---|---|
| 1 | M0: make the M3 measurable (⁵⁵/⁶⁷/#267), plus the far census | (enabler) | Decides whether Medium already passes; makes 5–10% effects visible | – | 1–2 small PRs + one Mac session |
| 2 | F-a: haze-aware error bound and floor | Geometry | −10 to −45% far triangles → ground at High ~950–1,300 FPS | +3–12% | 2 PRs (after F5) |
| 2 | F-e: skirts only where the level changes | Geometry | −3 to −8% | 0–2% | 1 PR |
| 2 | F-b: floor in projected size, not width | Geometry (edge-on slivers) | −20 to −50% at ground (overlaps F-a) | small | 1 PR |
| 3 | F-f: index order for vertex reuse | Vertex shading / parameter-buffer memory | 0–15% | 0–5% | tiny |
| 3 | F-g: lighter far vertex outputs and loads | Parameter-buffer memory / vertex | 0–10% | ~0 | 1 small PR |
| 4 | D1: face-group order | CPU submit | 0 FPS; CPU −10–15% | 0 FPS | tiny |
| 5 | F-c: 16×16 patches | Geometry | −10 to −25% (needs F-e) | small | 2 PRs |
| 6 | F-d: anisotropic patches | Geometry | further −15 to −30% after F-a/F-b (uncertain) | small | 3–5 PRs |
| 7 | V1: u16 indices / vertex compression | Bandwidth, memory | 0–3%; −12 MB | 0–2% | 1 PR |
| 8 | #131 front-to-back | Fragments (immediate-mode GPUs only) | ~0 | 0–5% of voxel GPU time | 1 PR |
| 9 | J: waste at the voxel/far join | Geometry, raster | 1–3% | ~0 | 1 PR |
| – | F6(c), far terrain from 512 m | Trade: voxel vs far triangles | sign unknown (−10% to +15%) | +5–15% | a decision, not a build |
| parked | #130 hierarchical cull | CPU | 0 FPS | 0 FPS | 1–2 PRs |
| parked | #32/#28/#33 GPU culling | CPU submit (Vulkan only) | 0 or worse (measured regression) | 0 FPS | 3+ PRs + a path per backend |
| close | #42 hi-Z | Raster / fragments | −34 to −60% (measured) | measured loss | built already |
| T | Frame-time tail (refresh goal) | Hitches | none in FPS; removes stalls | – | small |

## 3. Candidates

### M0: make the M3 measurable (⁵⁵, ⁶⁷, #267, #298)

- **Problem:**
  - The same code reads ±15–20% apart from run to run.
  - Hot against cold is 2.3× (⁵⁵).
  - ⁶⁷ read 30% low at 19% battery.
  - Metal GPU timestamps come back invalid (#267), so FPS is the only metric.
  - §6b dropped "skirts only where needed" and the 3–4 px bound as "within noise". On this machine that means anything under ~15% is lost.
  - The #298 warm-up (≥1 s) may move the M3 the other way from Windows.
- **Gain:** none directly. It tells whether Medium already passes, and makes everything below decidable.
- **PR 1, small:** the bench prints the run's conditions next to SUMMARY on macOS: power source and battery % (`pmset -g batt`), thermal state (`pmset -g therm`, or NSProcessInfo.thermalState), and the change in pageouts (`vm_stat`). It warns on battery, below 80%, or under thermal pressure.
- **PR 2, small:** `scripts/bench-ab.sh A B N`. It alternates two binaries with a cool-down and prints medians and spread, the way ⁶⁹, ⁷⁰ and ⁷⁵ were done by hand.
- **Mac session:**
  - Three alternated runs of `--bench tune --target 100000`, which prints every quality at every eye.
  - One `xcrun xctrace record --template 'Metal System Trace'` at the ground eye, High. It splits vertex, tiler and fragment time with no code change.
  - One 60-second sustained run, to see where clocks settle under throttling.
- **Confirming experiment:** fit frame_ms = a + b × far_triangles over 3 eyes × 3 qualities.
  - If b holds near ~1 ns, the far triangle count is the lever and the ranking stands.
  - If b tracks screen coverage, pixel work matters, and F-g and J move up.
  - Also add a fixed 0.1 ms CPU spin per frame. If FPS drops, CPU and GPU are not overlapping and the CPU-side items gain value.
- **Fallback for GPU timing:** read MTLCommandBuffer GPUStartTime/GPUEndTime through wgpu's backend-level (hal) access. Only if xctrace is not enough.
- **Risks:** macOS-only code in the bench (keep it to a reported string, behind cfg). If the 60-second plateau sits well below the 2-second number, what the gate should measure is the owner's call (the #261 rule). The same goes for `check-phase-gate.sh` running the bench right after `cargo test --all`.

### Census: far triangles in view (CPU only, any machine; feeds F-a to F-d)

- **What:** extend `crates/world/src/far.rs`'s `print_error_per_config`. At the three gate eyes, turning, with the real frustum, report far triangles split by:
  - distance octave;
  - projected quad height in px;
  - front- or back-facing;
  - why the patch was split: distance, error, or stopped at the floor;
  - triangles whose three vertices are all inside the hole.
- **Effort:** one small PR (a test or example). No GPU needed.
- **Why it matters:** every geometric gain below depends on where the triangles are.

### F-a: haze-aware error bound and floor

- **Bottleneck:** geometry.
- **Mechanism:** a height error of e px behind haze h changes a pixel by about (1 − h) × a full-contrast e px error. Scale the 2 px bound and the 4 px floor by 1 / max(1 − h, c_min), using each patch's nearest point (the least haze).
  - The CPU mirror `Haze::amount` (`haze.rs`, F5) is already pinned against `haze.wgsl` by tests.
  - The world crate stays ignorant of haze: give `select` a `contrast_of` closure alongside `error_of`.
- **M3 gain, estimated with F5's candidate density of 1/20,000 per block and a 600-block scale height:**

  | Distance | Haze, ground eye |
  |---|---|
  | 10 km | ~37% |
  | 20 km | ~61% |
  | 40 km | ~85% |

  - Hills (~20 blocks of relief) stop needing splits beyond ~10 km. Mountains need them out to ~100 km. So triangles likely concentrate within 10 km.
  - **−10 to −15%** if the census puts most triangles inside 5 km; **−25 to −45%** if a third or more lie beyond 15 km.
  - Flight gains less: the haze is thin looking down from 3 km.
- **RTX 4060 gain:** far terrain is ~0.05–0.07 ms of a ~0.2 ms GPU frame, so +3–12%.
- **Effort:**
  - PR 1: the closure in `select`, with tests (hazier never selects more; error × contrast stays within the bound).
  - PR 2: app wiring, golden images at the three eyes, BENCHMARKS rows. Do this after F5's values are fixed.
- **Risks:**
  - The level of detail now follows the haze the owner tunes, so a haze change must invalidate `settings.ron`'s tuned answer.
  - Popping moves, though at lower contrast.
- **Confirming experiment:** the census with contrast weighting (triangles against contrast-weighted p99 error). Then images at the three eyes against plain High: at most one 8-bit step of difference on ≥99.x% of pixels. Then M3 A/B.

### F-b: a floor in projected size, not width (the cheap half of anisotropy)

- **Mechanism:** the floor stops splits by quad *width* (4 px). Seen edge-on, such a quad is 0.05–0.8 px tall, which is §6b's "several triangles in each pixel". Bound the projected minor axis or the area instead: the vertical angle a quad spans along the line of sight, plus a relief term.
- **Gain:** M3 −20 to −50% at ground where the census shows edge-on patches stopped at the floor; less at hill; ~0 in flight. RTX 4060 small.
- **Effort:** 1 PR (`walk()`, a test, golden images).
- **Risk:** in exactly these patches the 2 px height bound is broken. Seen from a low eye, hill crests are silhouettes, so ridges can move by several px. Judge by image difference with ridge-line displacement measured; possibly keep the tight bound for patches near the local horizon.
- **Confirming experiment:** the census and the image comparison, as for F-a.

### F-d: anisotropic geometry (coarser along the line of sight)

- **Mechanism:** separate quad sizes along x and z (binary splits along the axis that points more toward the eye), or a polar grid around the camera. Detail across the view ray (silhouette shape) stays; rows along it shrink. Crest height needs samples at about the crest's curvature, not 4 px slivers.
- **Gain:** 2–4× fewer triangles on edge-on land at equal crest accuracy. After F-a and F-b, perhaps a further −15 to −30%. Uncertain. RTX 4060 small.
- **Effort, 3–5 PRs:**
  - `PatchKey` with (level_x, level_z), and per-axis split and error;
  - rectangular footprints in `generate`/`finer_gap`;
  - `PatchGpu.quad` as xy, and `far.wgsl` to match;
  - skirts between neighbours of different shapes;
  - `far_streaming`'s `ancestors`/`draw_set` logic, which assumes four children.
  This is the largest change in the list.
- **Risks:** complexity in streaming's swap logic, and cracks. Base the split on the eye's position, not its direction, or every turn of the camera churns the selection.
- **Confirming experiment:** a CPU-only prototype of the per-axis `select` with triangle counts in the census. Build it only if F-a + F-b leave the M3 short.

### F-c: 16×16 patches (refine locally)

- **Mechanism:** a 32×32 patch refines all its 1,024 quads at once when its p99 gap fails, often because of a few columns of peak.
- **Gain:** −10 to −25% triangles (estimate).
  - The skirt share rises from 11.4% (264/2,312) to ~21% unless F-e lands first.
  - There are 4× the patches to select, generate and stream on the CPU.
- **Effort:** 2 PRs. The constant is asserted in 3 places, plus budgets and capacity.
- **Confirming experiment:** the census with `PATCH_QUADS` = 16.

### F-e: skirts only where neighbours differ in level

- **Mechanism:** 264 of 2,312 triangles per patch are skirt. Neighbours at the same level share identical edge heights, so they leave no crack. Give each patch a 4-bit edge mask; the vertex shader collapses unneeded skirt vertices onto the edge, which keeps one draw. Zero-area triangles are dropped before rasterization.
- **Gain:** M3 3–8% (zero-area triangles still cost vertex shading and setup); RTX 4060 0–2%. The §6b "within noise" result is consistent with a real gain of this size.
- **Effort:** 1 PR.
- **Risk:** cracks if a mask is stale for a frame. Compute masks from the *drawn* set, not the wanted set.
- **Confirming experiment:** a throwaway with skirts off entirely, A/B on the M3. That gives the upper bound.

### F-f: index order for vertex reuse

- **Mechanism:** `grid_indices()` walks rows of 35 vertices. That is longer than typical vertex-reuse windows, so each vertex is likely shaded twice (~2,450 per patch instead of 1,225). On a tile-based GPU each shaded vertex is also written to the parameter buffer. Reorder into strips 8–16 cells wide, or use a vertex-cache optimiser order; the index buffer is shared by all patches.
- **Gain:** M3 0–15% (how Apple's GPU reuses vertices is unknown); RTX 4060 0–5%.
- **Effort:** tiny: one function and its test (every cell exactly once, counter-clockwise).
- **Risk:** none visual.
- **Confirming experiment:** M3 A/B at the three eyes.

### F-g: lighter far vertex outputs and loads

- **Mechanism:** each vertex outputs clip position plus a vec3<f32> colour, 28 B in the M3's parameter memory. ⁶⁰ found one extra f32 output cost the M3 up to ~7%. Each vertex also reads 5 heights.
- **Options:**
  - A colour gradient precomputed per vertex on the CPU, packed as 2×f16 into one u32: 2 loads instead of 5. Works on every backend.
  - An f16 colour output (`SHADER_F16`), only if every CI backend supports it; otherwise it means a second shader path.
- **Gain:** M3 0–10%; RTX 4060 ~0.
- **Confirming experiment:** a throwaway upper bound: position only, constant colour, `shade()` removed, A/B on the M3. It also answers ⁷⁰'s open "re-measure F4 on the M3". Drop F-g if the bound is below 10%.

### D1: face-group order (merging draws)

- **Mechanism:** `Mesh::group_by_face` stores the six directions in `Face` order (+X, −X, +Y, −Y, +Z, −Z), so the two directions of one axis sit side by side, and a camera outside the box never sees both.
  - Average runs per node = 3 − ¼ × (adjacent pairs from different axes).
  - Today that is 3 − 0.5 = **2.5**. Any order without same-axis neighbours, for example +X, +Y, +Z, −X, −Y, −Z, gives 3 − 1.25 = **1.75**.
  - Measured draws per node are 2.24 (ground) and 2.32 (hill), so expect −25 to −30% voxel draws.
- **Bottleneck:** CPU submit. wgpu on Metal turns `multi_draw_indexed_indirect` into one draw call per entry, ≈113 ns each from the M3 rows: (0.266 − 0.189 ms) / (1,325 − 644 draws).
- **Gain:**
  - M3: no gate FPS. CPU/frame −0.03 to −0.05 ms at ground; in the 4,975-draw orbit, −0.1 ms or more.
  - RTX 4060: ~0, because Vulkan multi-draw is native.
  - Second-order: on a shared-power chip, less CPU work leaves more thermal budget for the GPU.
- **Effort:** tiny. One shared ordering constant in the voxel crate, used by `group_by_face` and `arena::prepare`, plus a test over the 8 octants.
- **Risk:** none visible. Opaque, depth-tested geometry should render the same golden images.

### V1: vertex and index compression (voxels)

- **Mechanism:** u16 indices with base_vertex work if every node stays under 65,536 vertices. The checkerboard worst case for a 16³ node is 49,152 plus skirts, which needs a mesher bound test. An 8-byte vertex would need node_index taken out of the vertex, and §5.3's instance_index route is unreliable.
- **Gain:** the gate eyes fetch ~2–3 MB of voxel vertices and indices per frame, so ≤3% on the M3 and ≤2% on the 4060. Memory drops by 12 MB, relevant on an 8 GB machine that pages (⁵⁵), but small against 1–1.4 GiB RSS.
- **Effort:** 1 PR.

### #131: front-to-back voxel draws

- **Already partly there:** `NodeId` order is level first, so the finest, nearest ring is drawn first.
- **M3:** Apple's hidden-surface removal shades only visible opaque fragments, and `mesh.wgsl` never discards, so ~0.
- **RTX 4060:** 0–5% of voxel GPU time, for a sort of 600–1,300 nodes per frame (10–30 µs) or a sort only on chunk crossings.
- **Confirming experiment:** a throwaway sort; A/B GPU/frame on the 4060. Keep deferred unless it gains ≥5%.

### J: waste and overdraw at the voxel/far join

- **Mechanism:** patches that straddle the hole are drawn whole, with their inside sunk to the hole floor. Those triangles are shaded, set up and rasterized, then hidden.
- **Estimate:** 5–15 such patches in view, about half inside: 5k–20k triangles, 1–3% of far triangles.
- **Fixes considered:**
  - Splitting straddling patches makes their children 4× finer, so it adds triangles.
  - Collapsing "deep inside" vertices outside the frustum risks stretched triangles.
- **Verdict:** low priority. Measure with the census's all-inside count. F6(c) moves the join anyway.

### F6(c): far terrain from 512 m, as a performance decision

- **New data:** it cuts voxel triangles 19 / 25 / 47% at ground / low / hill.
- **The catch:** the far terrain then covers 512–1,024 m. There its floor (4 px ≈ 2 blocks) and 2 px bound (≈1 block) are finer than the 13–16 px voxel cells it replaces.
- **Hill-eye estimate:** −48k voxel triangles against +80k–300k far triangles for that ring in view. On the M3, far triangles are the binding cost, so the net may be a loss there and a gain on the 4060.
- **Experiment:** run the prototype branch with `--bench tune --target 100000` on both machines, recording voxel triangles, far triangles and FPS at every eye. Decide F6 with M3 numbers, not with the voxel cut alone.

### Parked or close

- **#130 hierarchical cull:** 13% of 0.10 ms CPU on Windows, no FPS on either GPU-bound machine. Keep its revisit trigger.
- **#32/#28/#33 GPU culling:**
  - Metal has no draw-count variant (`multi_draw_indirect_count`), and wgpu emulates multi-draw with a CPU loop. The prototype measured a regression (0.520 vs 0.439 ms).
  - On Vulkan the 4060 is GPU-bound, so CPU savings do not raise FPS.
  - It adds a path per backend, and GPU-written indirect arguments reopen ⁶⁸'s validation trade.
  - Reopen if wgpu gains native multi-draw on Metal or the gate becomes CPU-bound.
- **#42 hi-Z:** measured −34 to −60% on the M3. Far patches hidden by terrain were only 8 of 312. Close it.
- **Watch item:** if wgpu's experimental mesh shaders reach Metal, the far terrain is the natural first user (per-meshlet back-face and small-triangle culling).

### T: frame-time tail (the refresh-rate goal)

- **T1, a real bug.** `scene.rs` `set_lighting` compares only against `pipeline_lighting`, which changes only when a rebuild lands (scene.rs:437–454, audit-render.md #1). While a rebuild is in flight, every frame starts another compile thread and drops the previous receiver. With the ~40 ms first compile, dozens of threads pile up and the pipeline may never land. Fix: remember the target in flight. Tiny PR.
- **T2, F5 before merge.**
  - Constant haze lives in `Lighting` as a uniform, so it triggers no rebuilds.
  - But `mesh.wgsl` gains an output component (`ao_haze`), and ⁶⁰ showed one output costing the M3 ~7%.
  - Windows measured only noise; measure F5 on the M3 before merging.
- **T3, frame times in play.** Report p99 and max frame time against the refresh budget in play-like runs (extend `--bench flight`). Covers first-use pipeline compiles and upload bursts.
- **Minor, on the RTX 4060:** the frame interval (~0.27 ms) exceeds the timed GPU pass (0.15–0.23 ms). The ~0.05–0.1 ms outside the main pass is unexplained (overlay pass, staging copies, submit). It does not affect the gate.

## 4. Recommended order and what to measure first

**First three:**
1. **M0 plus the census.** The census runs on any machine, M0 on the Mac, in parallel. The M0 Mac runs also re-measure F5 on the M3 and the F6(c) prototype.
2. **Far triangle budget: F-a and F-e, plus F-b if the census says so.** The census picks the variant: triangles against contrast-weighted error. Golden images then judge the look, and M3 A/B runs confirm the FPS. Target: −20 to −25% far triangles, which puts the M3 at High above 1,000.
3. **One Mac session of throwaway tests:** F-f (index order), the F-g upper bound, and skirts off. Keep whatever gains ≥5% in A/B.

**Alongside:** D1 (tiny, CPU) and the T1 bug fix. F-d only if steps 2 and 3 fall short.

**Measure first, before any optimisation:**
- On the M3: AC power, battery ≥80%, idle 5 minutes, nothing else running. Run `--bench tune --target 100000` three times alternated with the same binary, recording FPS, CPU/frame and far triangles at every eye and quality. Take one Metal System Trace at the ground eye, High.
  - Fit frame time against far triangles.
  - If the cost per triangle is constant (~1 ns), the ranking above stands.
  - The same runs say whether Medium already holds 1,000.
- On any machine: the far-triangle census. It says where the triangles are, which decides between F-a, F-b and F-d.

### Critical Files for Implementation
- crates\world\src\far.rs (`select`/`walk`: split rules, floor, error; the census harness)
- crates\render\src\far.rs (`grid_indices`, `PatchGpu`, skirts, `prepare`)
- crates\render\src\shaders\far.wgsl (vertex outputs, height loads, skirt collapse)
- crates\app\src\far_streaming.rs (`FarQuality` detail, `far_view` wiring, the haze closure)
- crates\app\src\bench.rs (run-condition reporting, far triangle counts, M0)

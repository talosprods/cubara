# Proposal: LOD transitions that do not show

Status: **proposal for block 3.0b** (`ROADMAP.md`, Phase 3, block 3.0 "The engine first", item (b)). Written 2026-11-09 as a design study, after a play test that day in which the owner found the detail layers clearly visible. Numbers marked *est.* are estimates; §5 names the measurement that replaces each one. M1, M2, M4, M6 and T1 have been measured since, by `crates/app/src/lod_probe.rs` (§5.1); the rest is still to be measured. M4 (issue #335) decided ring 3: level 3 stays at v = 4 (§3 R1, §5.1).

The owner's call is §7. Everything else is engineering, decided here and open to change only with a measurement.

Related, from block 3.0a (#318): flying, a coarse node replaced at the skyline drops the ridge by a pixel or two for a few frames. That is the same step this proposal shrinks (R1, §1.2 item 8), and R6 smooths what is left.

## 0. The answer

What changes at 160 m and 288 m is not mainly texture. Prototype B is almost identical to A, and the grass top tile's luminance varies by only 13%. It is not the far terrain either, because prototype C leaves those edges as they are.

It is the vertical size of a cell. Every ring edge doubles the height of every step in the terrain:
- From any eye above the ground, a hill is read through its risers, the contour lines.
- At an edge those lines become twice as thick and half as many, along a straight line across the landscape.
- When the ring moves, they jump.

Trees are also generated only at level 0, so they stop at 160 m.

The fix for both edges is to keep coarse cells coarse horizontally but make them one block tall. That has to be done together with a decision for 512–1,024 m, and trees have to continue past the first ring. Everything else below is ranked against that and the M3 budget.

## 1. Diagnosis

### 1.1 Ring edges on screen

At 1080p with a 60° vertical FOV, one pixel is 0.97 mrad.

| Edge | Cell beyond | Cell px, inside → outside | Riser px today, inside → outside | Max height error beyond (blocks / px) | Rows below horizon, ground eye |
|---|---|---|---|---|---|
| 160 m | 2 | 6.4 → 12.9 | 6.4 → 12.9 | 1 / 6.4 | 64 |
| 288 m | 4 | 7.2 → 14.3 | 7.2 → 14.3 | 2 / 7.2 | 36 |
| 512 m | 8 | 8.1 → 16.1 | 8.1 → 16.1 | 4 / 8.1 | 20 |
| 1,024 m | far terrain | 8.1 → smooth (a block is 1.0 px) | 8.1 → smooth | 4 / 4.0 | 10 |

- **Height error:** a coarse cell is solid when its middle is under the surface (`density_at`), so its top is the surface rounded to the cell, which is up to half a cell off.
- **Ground eye:** it is about 10 blocks above the ground, so every ring edge lies within 64 px of the horizon.
- **Raised eye:** at 50 blocks up the 160/288/512 m edges are 313, 178 and 101 rows down. The 160–512 m bands then fill much of the screen, which is why A-low shows the 160 m edge as a straight line across the terraces.

### 1.2 What changes at an edge, largest first

1. **The riser pattern (dominant).**
   - With `mesh.wgsl`'s default lighting and the textures' mean luminance, a lit grass top is 0.253 (linear). A grass-side riser is 0.083 sunlit and 0.046 shaded, so tops are 3 to 5.5 times brighter.
   - At an edge the step height doubles. Riser area is unchanged because the geometry scales uniformly, but the pattern's scale doubles along a line, and that is what the eye detects.
   - A riser-share error of 0.10 is about a 7.5% luminance error.

2. **Trees.**
   - `WorldGen::generate` grows no trees at step > 1 (PHASE1 §8.4).
   - An oak is about 8 blocks tall: 52 px at 160 m, 29 px at 288 m, 16 px at 512 m.
   - The density is one per 12 chunk columns (`oak.ron`). That gives about 1,300 trees in the 160–1,024 m square ring and about 330 in a 92° horizontal frustum (*est.*).
   - The result is a tree line at 160 m, and trees popping in and out as the ring moves.

3. **Elevated eyes see coarser cells than the schedule intends.**
   - `VERTICAL_LOD_SQUASH = 2` counts vertical distance twice.
   - From the hill eye (y 300, about 265 above the ground) all ground in view is level 2 out to 512 m and level 3 beyond.
   - At the bottom of the bench's view (381 m slant) a 4-block cell is 11 px; looking straight down it is 16 px.
   - With squash 1 the same ground would be level 1, at 5 to 8 px.

4. **Riser materials.**
   - A coarse riser wears its cell's material, stretched. The top cell wears grass (#296), so an 8-block riser shows the grass-side tile stretched 8 times: its green band, the top 28% of the tile, becomes 2.2 blocks tall. Below it is stone.
   - Real columns show 1 block of grass side, then 3 of soil, then stone.
   - The far terrain's `riser_colour` gets this right, so the join at 1,024 m differs here too.

5. **Texture scale (prototype B's subject).**
   - UVs count lattice cells, so a level-L face shows its tile 2^L times too large, and the sampler picks a mip level L levels too fine.
   - From the ground eye, tops are already at the 1×1 mip beyond about 50 m (at 160 m one pixel row covers 2.5 blocks of depth), so nothing changes there.
   - From the hill eye, level-3 tops show the tile pattern at about mip 1, where real blocks would show their mean colour.
   - How visible this is depends on the tile's contrast. The coefficient of variation of linear luminance is 0.13 for grass_top, 0.12 for soil and 0.11 for stone, so B looks the same as A. grass_side is the contrasty tile (0.47), and B leaves side faces alone.

6. **AO extent.** AO gradients span one cell. Today the mean comes out right because riser count shrinks by the same factor; only the pattern scale doubles, which is part of item 1.

7. **The far join** at 1,024 m, or at 512 m in C, is smooth and lighter.
   - *Lighter (estimated cause):* `far.wgsl` takes the gradient of the smoothed surface (vertex heights are footprint means, gradient across two quads). A staircase's risers are set by E|∇h| at block scale, and E|∇h| ≥ |∇E[h]|.
   - The terrain fbm (4 octaves, lacunarity 2, gain 0.5) gives each octave about the same slope. With roughly 16-block quads at the join, the shader keeps about one octave in four, so the riser share is underestimated by up to about 2× (*est.*).
   - AO is also left out on purpose (`aggregate.rs`).
   - *Smooth:* at 512 m a block is 2 px, and no smooth surface can show 2-px steps.

8. **Temporal pops.** A swap happens on one frame (`uploads.rs`). At today's edges it moves heights by up to 2^(L−1) blocks (6 to 8 px) and redraws the riser pattern.

## 2. Budget

- **RTX 4060 gate frame** (after ⁷⁵): about 0.10 ms CPU and 0.23 ms GPU.
  - It is fill-bound: the hill eye's GPU time (0.229 ms) is higher than the ground eye's (0.178 ms) with 40% fewer drawn triangles (⁶⁴).
  - Each draw costs about 50 ns of CPU (⁶⁸).
- **M3 at Low** (⁶⁷, measured before ⁷⁵ on a battery at 19%):

  | Eye | FPS | Frame | Headroom to 1,000 FPS |
  |---|---|---|---|
  | ground | 1,090 | 0.917 ms | 0.083 ms |
  | hill | 1,563 | 0.64 ms | 0.36 ms |
  | flight | 4,039 | – | – |

- **Where the M3's time goes:** at the ground eye, voxels only run at 3,121 FPS (0.32 ms, 168k triangles drawn). The far terrain is therefore about 0.6 ms of the M3's ground frame at Low.
- **M3 cost per drawn voxel triangle:** 0.6 to 1.4 ns, derived from the ⁶² rows (ground 168k at 3,121 FPS, hill 103k at 3,530, orbit 481k at 1,341).
- **Drawn-triangle split, measured (M1, §5.1):** at the ground eye, 150k voxel triangles are drawn on average and 187k at most. Level 0 has 40% of them, and levels 1, 2 and 3 have 21%, 20% and 19% (32k, 29k and 28k). The assumed split above holds, at about 1.2 times the size. At the hill eye, all 77k come from levels 2 (53%, 41k) and 3 (47%, 36k). The table is in §5.1.

## 3. Techniques, ranked

### R1. Block-height coarse cells (vertical precision of 1 block) — the core fix

**What it is**
- Coarse nodes keep their 2^L-block horizontal cells, but their vertical cells are v blocks tall, with v = 1 in rings 1–2.
- The node stays the same cube: same draws, same octree, same `covered_box_3d`, same far hole.
- Its lattice becomes 16 × 16·2^L/v × 16. Only the surface band (the coarse cell holding each column's surface) is generated at full vertical resolution, so generation costs about 2× today's, not 2^L×.
- Positions are emitted in blocks with `node_origins.w = 1`. A level-3 node spans 128 blocks, which fits the 10-bit position fields; u/v reach at most 128, which fits their 8 bits. UVs are then per block on every face, which is option (b) including side faces, with no shader multiply.

**What it fixes**
- The riser step at 160 m and 288 m: risers are 1 block on both sides (6.4 → 6.4 px at 160 m, 3.6 → 3.6 px at 288 m).
- Height error drops to at most 0.5 block plus the footprint-averaging error.
- Risers get per-block materials (grass side, soil, stone), matching `riser_colour`.
- Pops shrink, and skirts get smaller.

**What it does not fix:** the contour lines are still jagged horizontally at 2 or 4 blocks. That is foreshortened from low eyes and visible from high ones.

**Triangle cost**
- Model: wall edges per node ≈ 512·min(1, 2^L·g/v), against 512·g today (g is the slope), with at most 3 material bands per edge. That gives ×2 for level 1, ×4 for level 2, and ×6–8 for level 3 at v = 1.
- With v = 1 in levels 1–2: about +106k triangles drawn at the ground eye. That is +0.01 to 0.03 ms of GPU on the 4060 and +0.06 to 0.15 ms on the M3 (*est.*).
- CPU frame cost: none, because no draws are added.
- Worker threads: about 2× the generation and 2–4× the meshing for coarse nodes. `--bench flight` must stay where it is.
- Vertex arena: +30 to 50% meshed triangles; the arena is at 41% today.

**Required alongside it**
- **AO on top faces divided by the cell width.** Riser count no longer shrinks with level, so without this each riser foot gets an AO band 2^L blocks wide and the coarse rings come out too dark. Two of the 16 spare bits in vertex word 2 can carry log2 of the width.

**Changes elsewhere**
- `greedy_mesh` and `push_quad` take (nx, ny, nz) and a scale per axis. A uniform-scale call stays pixel-identical, so the existing goldens hold.
- `border_samples` and `visibility::node_links` adapt to the anisotropic grid.
- §6.2's "uniform node cost" changes. Record that under ARCHITECTURE's "When a rule blocks real work".

**Ring 3 must be decided in the same step.** With level 2 at v = 1 and level 3 left as it is, the 512 m step becomes 2 → 16 px risers, which is worse than today's 8 → 16. There are two candidates, and M4 decides between them:
- (i) Level 3 at v = 4: 2 → 8 px risers, about +32k triangles.
- (ii) C plus R5 (the corrected far shading): about −32k triangles. Its silhouettes are closer to the truth than 8-block columns: the far terrain's p99 is 2.7 px, while 8-block columns make 16-px jaggies at 512 m.

**Decided (M4, issue #335, 2026-11-10): (i) stays.** The CPU reference image (§5.1) scores candidate (ii) -- measured as C alone, R5 not built (see §5.1's own note on why) -- against the decision rule's skyline condition and finds it fails at the ground and low eyes: the smooth footprint-mean height from 512 m loses real relief the blocky, still-per-block candidate (i) keeps, enough to move the skyline 44-78 px at its worst (against (i)'s 34-36 px, the same height field's own averaging error). The band metric alone would not have ruled (ii) out -- at two of the three eyes its band D is *lower* than (i)'s -- but the decision rule requires both conditions, and skyline is the one that holds. Ring 3 is level 3 at v = 4, as it is today; no change lands from this issue.

### R2. Trees beyond the first ring

**What it is**
- The same placement (`trees_near`, a pure function) evaluated per coarse node.
- Drawn as instances in one draw: the real tree mesh out to 512 m, and two boxes (trunk and canopy) beyond.

**What it fixes:** the tree line at 160 m, popping trees, and objects on the skyline.

**Cost**
- About 330 instances in view, roughly 10–30k triangles, plus one draw.
- Under 0.01 ms on the 4060 and under 0.03 ms on the M3 (*est.*).
- One new pipeline, built inside `encode_scene` (Rule 5).

**Risk:** trees float over coarse tops that sit up to 4 blocks too low today. Land R1 first, or set each tree on its coarse cell's top.

### R3. Ring placement in screen space

**R3a. Circular rings for the split, keeping the cube reach.**
- Rings are Chebyshev squares, so detail reaches √2 farther along the diagonals.
- A cell's size on screen depends on Euclidean distance. Splitting by Euclidean distance keeps the worst-case cell size the same and removes 1 − π/4, about 21%, of each inner ring's area. Net, about 15–20% fewer voxel triangles and draws (*est.*): roughly −35k drawn at the ground eye, which is −0.02 to 0.05 ms on the M3.
- Keeping the cube reach means the far hole doesn't change.
- Cost: diagonal views get the same detail as axis views, which is up to 1.4× coarser than today but never coarser than today's worst case.
- Change: `node_distance`. The tiling and level-balance tests in `node.rs` must still pass.

**R3b. Squash 1 for the split, keeping the reach.**
- This puts cells under elevated eyes at the size the schedule intends.
- Cost lands only on high eyes: the hill eye goes from 869 nodes towards about 1,900 and gains about +70k drawn triangles, about +0.04 to 0.10 ms on the M3 against 0.36 ms of headroom.
- Underground eyes cost more too (see the measurements in `streaming.rs`'s comment).
- It can be measured today with `--bench 64 --eye 8,300,8 --squash 1`.

**R3c. Ring radii as a function of resolution and FOV.**
- Formula: R_L = 2^L / (N_L · p), with p = fov_y / height in rows. Today's schedule corresponds to N = 12.9, 14.3 and 16.1 at 1080p and 60°.
- At 1440p the cells beyond the 160 m edge are 17 px; at 4K they are 26 px. Holding 13 px at 4K would need R_1 = 317 m and about 2.5× the draws, so this can't be the default.
- Build it as a pure function that returns today's radii at 1080p/60°, so the gate doesn't change and a future FOV setting moves the rings with it.
- The game window renders at the display's resolution, so above 1080 rows the seams the owner sees are 1.3 to 2× larger than in the gate's images.

### R4. Texture-scale continuity (option b) and its mip/aliasing behaviour

- **With R1** it comes free.
- **Without R1:**
  - Tops: B's per-block multiply on top faces, which costs nothing.
  - Sides: the layer is chosen by depth below the quad's top: within 1 block, the surface side; within 1 + `SOIL_DEPTH`, soil; deeper, stone. This needs a small layer table and one select per fragment.
- **Mip and aliasing:**
  - Per-block UVs make the derivatives 2^L× larger, so the hardware picks a mip L levels coarser.
  - The texels per pixel are then the same on both sides of an edge, so the texture-scale step disappears by construction.
  - Aliasing can only go down, because the coarser mips are box-filtered in linear light (`materials.rs`).
  - Isotropic trilinear filtering over-blurs at grazing angles, but it does so exactly as level 0 does.
- **Visible gain today:** almost none (B looks the same as A). Keep it because it is free and the step would show with higher-contrast textures, such as stone on level-2/3 cliffs.

### R5. Matching colour and shading at the far join

- **(a) Roughness.**
  - Store per far vertex the block-scale mean |Δh| along x and along z: two floats, either sampled alongside the heights (about 3× the `surface_height` calls) or computed analytically from the octave spectrum (no calls).
  - Use them for risers facing both ways: a± = (E|g| ± E[g]) / 2.
  - Re-derive the masking cases against `reference_weights` walking real worldgen staircases instead of planes.
- **(b) AO.** Add an AO term to the aggregate, counted by the CPU reference.
- **Cost:** about 0 GPU (per-vertex shading, as F4 measured in ⁷⁰), +8 bytes per vertex, and patch generation CPU ×1 to ×3.
- **Fixes:** the lighter band. It is a prerequisite for any form of C.

### R6. Temporal popping: a sliding cross-fade (recommended) or a dithered fade (fallback)

**What it is**
- When nodes swap, both are drawn for about 250 ms of wall time.
- First half: the incoming node starts Δ (one vertical cell) low and rises.
- Second half: the outgoing node sinks by Δ and is then removed.
- What is seen is the higher of the two surfaces, moving continuously from old to new.

**Why it suits this engine**
- Only `node_origins.y` changes; the CPU writes a few vec4 a frame.
- No shader change and no discard, so hidden-surface removal on the M3 stays on.
- One of the two nodes is always at full height over the same footprint, so no cracks open.
- A bias of 0.01 block avoids z-fighting at the hand-over. That also removes today's frame or two of two coplanar surfaces in `uploads.rs`.

**Cost**
- Nothing on the static gate.
- During a swap, the outgoing nodes' triangles again for about 250 ms (*est.* 10–20k triangles per chunk crossing, under 0.02 ms on the M3).
- Measure it with `--bench flight` p99.

**Fallback:** a dithered fade needs a discard pipeline for the fading nodes (on the M3 this turns hidden-surface removal off for those draws) and a second indirect list in `arena.rs`.

### R7. Option (c), the far terrain from 512 m, weighed against the rendering agent's findings

- **Measured by the agent:** −19%, −25% and −47% voxel triangles at the ground, low and hill views, but a smooth, lighter, rippled band from 512 m.
- **Fixable:** the lighter look (R5). The ripple is real relief shaded smoothly, and R5 partly restores it.
- **Not fixable:** at 512 m a block is 2 px. C breaks §3.1's criterion between 512 m and about 1,030 m at 1080p, and more at higher resolutions, where the join should move outward (1,375 m at 1440p, 2,060 m at 4K).
- **It does nothing for 160 m or 288 m.**
- **Verdict:** not a look improvement. It is the largest budget lever there is, and it is candidate (ii) for 512–1,024 m in R1. Adopt it only with R5, and only if it scores within threshold on the band metric. D (B+C) behaves like C at the edges.

### R8. Silhouette and occlusion

- **Skyline today:** a ridge crossing a ring edge steps by up to 2^(L−1) blocks, which is 6.4, 7.2 and 8.1 px at 160, 288 and 512 m. With R1 that drops to 0.5 block plus the averaging error.
  - Averaging lowers narrow peaks.
  - M4 measures the skyline error per azimuth for the mean, the max of the 4 samples, and the 75th percentile.
  - Use the same estimator as the far terrain's footprint mean at the join; otherwise a step appears there.
- **Trees:** R2.
- **Occlusion:** ridges up to 4 blocks too low today hide less of what is behind them. The far terrain isn't affected, because it sinks inside the box.
- **Cave mouths:** these are majority-sampled at coarse levels and can open or close at an edge. That is small, and it is not addressed here.

### R9. Geomorphing — not recommended

- A level-0 chunk and its level-1 parent have different greedy quad sets, so there is no vertex-to-vertex correspondence to morph between.
- Making it work would need a coarse target on every vertex, greedy merges cut at every coarse boundary (2–4× the top quads), and morph data on every level-0 node, because meshes are built once while the morph zone moves.
- After R1 the remaining difference is horizontal, and axis-aligned faces cannot morph horizontally without tearing.
- R6 gets the temporal benefit for a fraction of the cost.

## 4. Recommended first combination, and why

**PR 1, the harness, with no change in what the game draws:** T1 and T4 with A failing, plus the M1 and M4 probes.

**PR 2, the main fix:**
- R1 at v = 1 in rings 1–2, with R4 and the AO scaling.
- Ring 3 replaced by C+R5 or by level 3 at v = 4, whichever M4 picks.
- R3a to pay for it.

**PR 3, in parallel:** R2 (trees).

**Then:** R6, R5 if it hasn't landed yet, R3b, R3c.

**Why this combination:**
- R1 is the only technique that removes what changes at 160 m and 288 m; B and C measured nothing there.
- It adds no draws, keeps the octree and the far hole, and its cost lands in the rings the owner is looking at.
- R3a takes back part of that cost without losing worst-case detail.
- R2 is cheap and removes the other thing that stops at 160 m.

**M3 budget at the ground eye, Low quality (*est.*):**

| Item | M3 cost |
|---|---|
| R1 in levels 1–2 | +0.06 to 0.15 ms |
| Ring 3 as C+R5 | −0.02 to −0.045 ms |
| R3a | −0.02 to −0.05 ms |
| R2 | under +0.03 ms |
| **Net** | **about +0.03 to +0.07 ms** |
| Headroom (measured, before ⁷⁵, low battery) | 0.083 ms |

M0 comes first, because that headroom number is the one everything is weighed against.

## 5. What to measure before building

- **M0.** Run `--bench gate` on the M3 at current main (after ⁷⁵) on a full battery.
- **M1.** Drawn triangles and nodes per level at each gate eye: a CPU probe using `desired_nodes_3d`, `mesh_nodes` and the frustum over the bench's yaw path, or a per-level counter in `--bench`. This replaces the assumed split in §2.
- **M2.** R1's triangle multipliers from the height field alone: count neighbouring coarse columns whose v-quantized heights differ. Calibrate against M1 at v = 2^L. No new mesher is needed.
- **M3.** The M3's cost per voxel triangle: `--bench 64 --eye` with one denser schedule at quality Off. The ⁶² rows bound it to 0.6–1.4 ns today.
- **M4.** A CPU reference image:
  - Ray-march the exact block height field from the gate eyes and the F6 low eye (a few hundred pixel columns).
  - Shade with the textures' mean colours and `mesh.wgsl`'s lighting.
  - Emulate A, R1 (v = 1, 2 and 4) and C/C+R5 by changing only the height lookup.
  - Apply the §6 band metric and the skyline error per azimuth.
  - This decides ring 3's treatment and the height estimator.
- **M5.** The size of the far-shading gap: E|Δh| against the shader's gradient near the join, run through `aggregate_colour`. Compare against the C images.
- **M6.** Tree count and pixel size per ring (pure function).
- **M7.** `--bench 64 --eye 8,300,8 --squash 1` on Windows and the M3 (the flag already exists).
- **M8.** Pop size: pixels changed in the ring bands at a chunk crossing, A against R1, from M4.

### 5.1 Measured so far (`lod_probe.rs`)

`cargo test --release -p cubara --bin cubara lod_probe -- --ignored --nocapture`, CPU only, with the real node selection, meshing, visibility search, frustum and face-facing tests over the bench's yaw path. Linux, 2026-11-09:

**T1** (turned around, R1d, #327: `no_ring_edge_doubles_a_step_on_screen` asserts the 160 m and 288 m rows below; the 512 m row is measured, not asserted -- ring 3's own decision, M4):

| edge | cell height inside -> outside (blocks) | step px inside -> outside |
|---|---|---|
| 160 m | 1 -> 1 | 6.4 -> 6.4 |
| 288 m | 1 -> 1 | 3.6 -> 3.6 |
| 512 m | 1 -> 4 | 2.0 -> 8.1 |

The cell height is now `NodeKey::cell_height_blocks` (1 block at levels 0-2, 4 at level 3), not `2^level` -- which is why 288 m's step dropped from 7.2 to 3.6 px rather than staying level: it was never about the 288 m edge doubling, it was about every cell height matching its neighbour's up through level 2.

**M1, ground eye** (8, 40, 8):

| level | desired | meshed | reached | drawn nodes avg (max) | drawn tris avg (max) | share of tris |
|---|---|---|---|---|---|---|
| 0 | 5808 | 1596 | 778 | 222 (321) | 60467 (83778) | 40% |
| 1 | 3274 | 1458 | 374 | 104 (135) | 32225 (40462) | 21% |
| 2 | 2740 | 459 | 414 | 111 (152) | 29447 (43966) | 20% |
| 3 | 4508 | 412 | 374 | 103 (121) | 28281 (32136) | 19% |
| all | | | | 540 | 150420 (187422) | 100% |

**M6, ground eye**:

| level | grown there today | trees | in view (avg) | on-screen height px mean (min-max) |
|---|---|---|---|---|
| 0 | yes | 40 | 11 | 49.7 (18.1-165.5) |
| 1 | no | 91 | 23 | 18.6 (9.4-31.7) |
| 2 | no | 270 | 67 | 9.7 (4.8-17.9) |
| 3 | no | 887 | 221 | 5.7 (2.9-11.6) |

**M1, hill eye** (8, 300, 8). Only levels 2 and 3 are reached, as §1.2 item 3 predicts:

| level | desired | meshed | reached | drawn nodes avg (max) | drawn tris avg (max) | share of tris |
|---|---|---|---|---|---|---|
| 0 | 5808 | 0 | 0 | 0 (0) | 0 (0) | 0% |
| 1 | 3274 | 0 | 0 | 0 (0) | 0 (0) | 0% |
| 2 | 2740 | 406 | 402 | 101 (181) | 40741 (72412) | 53% |
| 3 | 4508 | 482 | 467 | 132 (154) | 36179 (49814) | 47% |
| all | | | | 233 | 76920 (121002) | 100% |

**M6, hill eye**:

| level | grown there today | trees | in view (avg) | on-screen height px mean (min-max) |
|---|---|---|---|---|
| 2 | no | 401 | 82 | 9.0 (4.6-16.2) |
| 3 | no | 887 | 235 | 5.5 (2.9-10.3) |

The flight eye (8, 3,000, 8) draws no voxels: everything it sees is the far terrain.

**M2**: R1's walls, from the height field. For each level, it counts the coarse columns in the rings around spawn (`desired_nodes_3d`; the columns depend only on x and z, so every eye gives the same table). Each column's surface (`cell_surface_height`) is quantized to v blocks. The count is the risers against the +x and +z neighbours, with each riser counting at most 3 material bands.

| level | v (blocks) | edges with a riser | riser bands | bands vs today |
|---|---|---|---|---|
| 1 | 2 (today) | 36,642 | 36,643 | 1.00× |
| 1 | 1 | 72,090 | 73,399 | **2.00×** |
| 2 | 4 (today) | 40,963 | 44,874 | 1.00× |
| 2 | 2 | 69,558 | 85,934 | 1.92× |
| 2 | 1 | 104,102 | 153,602 | **3.42×** |
| 3 | 8 (today) | 38,618 | 43,703 | 1.00× |
| 3 | 4 | 63,436 | 84,961 | 1.94× |
| 3 | 2 | 93,045 | 145,356 | 3.33× |
| 3 | 1 | 117,070 | 226,655 | **5.19×** |

R1's wall multipliers are therefore ×2.0 at level 1 and ×3.4 at level 2. §3 estimated ×2 and ×4; ring 3 at v = 4 is ×1.9.

An upper bound on the cost at the ground eye: assume every drawn triangle at levels 1 and 2 (M1: 32k and 29k) were a wall, since tops do not multiply. Then R1 at v = 1 in rings 1–2 adds at most 32k + 2.42 × 29k ≈ **+102k** drawn triangles, against §3's estimate of +106k. Ring 3 at v = 4 (candidate (i)) adds at most 0.94 × 28k ≈ +27k. The real figures are lower by the top faces' share, which only meshing R1 will tell.

**M4** (issue #335, `lod_probe.rs`'s `m4` module): a CPU reference image, deciding ring 3 (512 m – 1,024 m) between candidate (i) (level 3 at v = 4, today's) and candidate (ii) (the far terrain's footprint-mean height from 512 m, R7/C). CPU only, no GPU, no meshing -- it walks the height field directly (`WorldGen::surface_height`/`cell_surface_height` for the exact-blocks reference and (i); `cubara_world::far` -- the production far-terrain code, just with a nearer hole than the game uses today -- for (ii)), shaded with `cubara_render::aggregate` (the same CPU reference `far.wgsl` is already checked against) and `mesh.wgsl`'s lighting.

**(ii) is measured as C alone, without R5.** R5(a) asks for both a roughness-split riser share (`a± = (E|g| ± E[g]) / 2`) *and* a re-derivation of `aggregate::MaskingTable`'s masking cases against real worldgen staircases instead of planes -- R5's own text treats that as separate work, not a shading tweak folded into a measurement harness. A partial version (the roughness correction alone, against an un-rederived masking table nobody has checked) would score (ii) against an invented model, which is a worse risk than not doing it, so this harness falls back to the issue's own fallback: "emulate C and say so". That means a fail here does not by itself rule out C+R5 -- but a pass would have been a strong result, and the decision rule's default ("otherwise (i) stays") is the safe outcome either way.

Ray-marched at 240 columns (azimuths spanning the camera's true horizontal FOV, yaw 0 only -- T4's own convention of 0°/45° is left for a fuller harness), a 3,000 m sweep, at the ground eye, the hill eye, and a low eye. `docs/PROPOSAL_FAR_VIEW.md`'s F6 names no eye of its own and neither it nor `BENCHMARKS.md` defines a numeric "low eye" anywhere in the repository, so this uses issue #335's own fallback: `[8, 60, 8]`, looking along +x pitched −0.15.

Linux, 2026-11-10:

| eye | band D max/mean (i) | band D max/mean (ii) | skyline mean/p75/max px (i) | skyline mean/p75/max px (ii) | threshold max/mean | decision |
|---|---|---|---|---|---|---|
| ground [8, 40, 8] | 0.0861 / 0.0178 | 0.0916 / 0.0172 | 8.48 / 11.96 / 33.92 | 11.01 / 14.10 / 77.80 | 0.0411 / 0.0219 | (i) stays |
| hill [8, 300, 8] | 0.0565 / 0.0471 | 0.0471 / 0.0386 | 9.18 / 13.41 / 35.82 | 9.15 / 13.41 / 35.88 | 0.0487 / 0.0363 | (i) stays |
| low [8, 60, 8] (F6 fallback) | 0.0656 / 0.0259 | 0.0411 / 0.0235 | 8.89 / 13.07 / 34.06 | 10.16 / 14.32 / 54.53 | 0.0471 / 0.0259 | (i) stays |

"Band D" is the §6 band metric (linear luminance, 4×4 box low-pass, mean |LP(F) − LP(R)| per row, around the 512 m edge), (i)/(ii) against the exact-blocks reference. "Skyline" is pixels between a variant's skyline and the reference's, per azimuth. The threshold is 2.5× (the proposal's own §6 calibration figure, used because the issue gives no other value) the reference-vs-reference noise -- here, the same reference rendered again from an eye jittered half a block sideways, since the CPU has no cross-backend variance to measure against.

**The decision (Design decisions 2): (ii) wins only within the threshold on the band metric *and* no worse on skyline error.** The band metric alone would not have ruled (ii) out: at the hill and low eyes its D is *lower* than (i)'s, i.e. closer to the exact-blocks reference, not farther. The skyline error is what decides it -- at the ground and low eyes (ii)'s worst-case skyline error (77.80 and 54.53 px) is more than twice (i)'s (33.92 and 34.06 px, which is itself just the height field's own footprint-averaging error); at the hill eye the two are close enough (35.88 vs 35.82 px) that it reads as a tie, not a win. **(i) stays**: ring 3 is level 3 at v = 4, as it is today. No change lands from this issue; §3 R1's ring-3 paragraph records the same decision.

## 6. How it is tested

**Without a GPU** (in `cubara-world` and `cubara-render`):
- **T1** `no_ring_edge_doubles_a_step_on_screen`: at 1080p and 60°, the tallest vertical step drawn outside each edge is at most 1.25× the step inside, and at most 7 px. Today fails it (12.9 against 6.4 px).
- **T2** `coarse_tops_are_the_surface`: the drawn top minus (surface + 1) is at most 0.5 block, over sampled columns in every ring.
- **T3** `a_coarse_node_averages_like_its_blocks`: area-weighted lit colour, with AO and its scaling, per face direction, against the level-0 chunks over the same footprint, within 2/255.

**Headless, in `golden.rs`:**
- **T4** `ring_edges_match_a_finer_ring`:
  - For each edge and each eye (ground, low, hill), at yaws 0° and 45°, meshing only the nodes in the frustum.
  - The real schedule is compared with a reference in which the ring outside the edge is drawn at the inner ring's level out to the next edge.
  - Compare in linear luminance after a 4×4 box low-pass. D(row) is the mean |LP(F) − LP(R)| along a row. Assert the maximum of D and the band mean of D.
  - The ratio of high-frequency energy (mean |image − LP(image)|) between the two must stay within [0.8, 1.25]. That is the pattern-scale and texture-scale step.
- **T5** `crossing_a_chunk_does_not_pop`: eye at x = 15.9 and 16.1. Render the second eye once with its own node set and once with the first eye's. Count pixels in the ring bands that differ by more than `TOLERANCE`. With R6, also check fade progress 0, 0.5 and 1.
- **T6** `the_far_join_matches_the_voxels`: the low-pass luminance step across the join, minus the step in M4's reference, must stay under the threshold.
- **T7** `trees_continue_past_the_first_ring`: leaf-coloured pixels in the 160–512 m band, against the reference.

**Thresholds:** calibrate as `TOLERANCE` was: set them about 2.5× above the reference-vs-reference noise across backends, and A must fail them by a clear margin (a mutation check).

**Also:**
- Golden images at the gate eyes, committed after looking at them.
- `--bench gate` rows on Windows and the M3, with a BENCHMARKS footnote.
- `--bench flight` rows for R1 and R6.

## 7. Owner's calls

1. **If the M3 at Low can't pay for R1 in both rings**, the options are:
   - (a) at Low, cells in ring 2 are 2 blocks tall, leaving a step at 288 m half today's size;
   - (b) at Low, the far terrain starts at 512 m (C+R5);
   - (c) the M3 waits for the far-terrain cost-per-triangle work.

   If every PC must get the same fix, the answer is (c).
2. **Displays taller than 1080 rows:**
   - (a) rings stay fixed, so seams are 1.3 to 2× larger in pixels than in the gate's images;
   - (b) rings scale with resolution under a node cap, at the cost of more draws;
   - (c) ring scaling becomes part of the quality setting.

## 8. Critical files

- `crates/voxel/src/voxel.rs` (`greedy_mesh`, `push_quad`, `push_skirt`, `face_ao`)
- `crates/world/src/worldgen.rs` (`generate`, `block_at_on_surface`, `cell_surface_height`, `trees_near`, `tree_site`)
- `crates/world/src/mesh.rs` (`mesh_chunk`, `border_samples`)
- `crates/world/src/node.rs` (`node_distance`, `select_nodes`, `DEFAULT_RING_SCHEDULE`)
- `crates/render/src/shaders/mesh.wgsl` and `far.wgsl`, `crates/world/src/far.rs`, `crates/render/tests/golden.rs`, `crates/render/src/uploads.rs`, `crates/app/src/streaming.rs`
- `crates/app/src/lod_probe.rs` (M1, M6, T1)

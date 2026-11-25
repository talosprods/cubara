# Phase 3 — Architecture (started)

The design [`ROADMAP.md`](../ROADMAP.md)'s phase-3 blocks implement. Same role
[`PHASE1_ARCHITECTURE.md`](PHASE1_ARCHITECTURE.md) and
[`PHASE2_ARCHITECTURE.md`](PHASE2_ARCHITECTURE.md) play for phases 1 and 2: not
a summary of the block list — the decisions the block list executes, with the
reasons, so no issue has to re-open one.

## Scope of this document

This is a **start**, not the full design. It carries over only the decisions
that already span more than one block, recorded in [#337](../../issues/337)
(the phase-3 block plan, written 2026-11-10 from the phase 3 the owner designed
on 2026-11-09), before the first world block (3.1, water) lands. They are
engineering decisions inside his design, each made in the issue named with it;
what is his to decide is listed as open questions in #337, not settled here. Each later block's PR adds its
own section here, the same way phases 1 and 2 grew theirs one block at a time;
this file is not finished until block 3.10 closes.

The feature set itself — what worldgen contains, in what order — is
`ROADMAP.md`'s phase-3 section and
[`docs/PROPOSAL_CLIMATE.md`](PROPOSAL_CLIMATE.md); this document does not
repeat it. What follows is the engineering machinery every block is built on.

---

## §1 Sea level

```rust
/// Blocks 3.1+ (`crates/world`). Altitude for every climate, hydrology and
/// biome calculation is `y - SEA_LEVEL`, never raw `y` -- so a change to one
/// constant moves the whole world's water line consistently instead of
/// requiring every consumer to know where "sea" is.
pub const SEA_LEVEL: i32 = 20;
```

Decided in [#340](../../issues/340) (block 3.1b). Everywhere altitude matters —
the lapse rate, the tree line, snow lines, river mouths, biome bands — reads
`y - SEA_LEVEL`, not `y`, so the sea can only move by changing this one
constant.

## §2 The column record — one source, two consumers

**One function answers "what is here", read by the voxels and the far terrain
alike**, so the far terrain at 262 km can keep agreeing with what the voxels
draw underfoot ([`PROPOSAL_FAR_VIEW.md`](PROPOSAL_FAR_VIEW.md)'s join
requirement) without a second implementation (`ARCHITECTURE.md` Rule 5).

```rust
/// `crates/world`. The exact per-block answer, used by chunk generation.
fn column(x: i32, z: i32) -> Column;

/// The same column as a far-footprint mean at quality `q`, used by the far
/// terrain and by anything averaging over an area rather than a single block.
fn column_lod(x: i32, z: i32, q: Quality) -> Column;
```

Every world feature from block 3.1 onward — water, rivers, lakes, biomes,
seasons — is a field of `Column` or a pure function of one, never a second
code path that happens to agree today. Decided in
[#340](../../issues/340)/[#350](../../issues/350) (blocks 3.1b, 3.3b); the exact
field list grows with each block that adds one, recorded in that block's own
section below.

## §3 Purity and the cache rule

**Everything is a pure function of seed and position** (and tick, for
seasons) — the same keystone `PROPOSAL_CLIMATE.md` §2 states for climate,
extended to every later system. Consequences:

- A region cache (rivers, lakes, biomes) is an **accelerator only**. It may be
  empty, cold, or rebuilt from scratch on any machine, at any time, and the
  result must be bit-identical either way. A cache is never consulted for a
  decision a pure function could not also make; it exists purely so the
  decision is not re-paid on every read.
- The standing test shape this produces, wired per feature as it lands: **a
  cold-cache run and a cache warmed from far away agree, bit for bit.**
  `ROADMAP.md`'s exit gate item 2 ("no seams") is this property, measured.

Decided in [#353](../../issues/353)/[#354](../../issues/354) (blocks 3.4a,
3.4b), stated as the general rule here because 3.6's biome cache and 3.8's
seasonal cache both lean on it rather than re-deciding it.

## §4 Integer climate

**Climate, seasons and year anomalies are integers after one quantisation
step**, so every machine's gameplay decisions built on them — ice forming,
growth windows opening — agree exactly, with no float-rounding divergence
between a Windows runner and the M3.

This sits inside the wider float rule `PROPOSAL_LOD_TRANSITIONS.md` and
[#349](../../issues/349) already state for worldgen generally (floats only as
`+ − × /`, `sqrt`, `floor`; the architecture check extends to ban the rest) —
climate narrows it further: temperature, humidity and the year's anomaly are
computed once as floats from the noise fields, then quantised to integers at
the boundary, and every decision downstream of that boundary (does it snow
here this tick, is this the growing season) reads only the integer.

Decided in [#361](../../issues/361) (climate fields), carried into
[#370](../../issues/370)/[#372](../../issues/372) (seasonal windows, year
anomalies).

## §5 Seasonal windows

**A year is 64 steps, and a season is a window of them** — the
same numbers read by the simulation (ice, growth) and by the shader (the
seasonal look), so the two cannot disagree about what season it is.

The vertex format's 13 spare bits (`PHASE1_ARCHITECTURE.md` §5) carry the
season window directly; this is why a later per-vertex need — a climate tint
for grass, say — would cost a fourth vertex word rather than fitting in what
seasons leave over ([#337](../../issues/337) risk 7: "the vertex bit budget is
used up by seasons").

**The bit budget**, every spare bit assigned, from [#371](../../issues/371)
(seasons), [#341](../../issues/341) and [#347](../../issues/347) (water,
block 3.1–3.2) as they stand today. Opaque and water vertices are two
disjoint meshes (`NodeMesh { opaque, water }`, #341 decision 1), never the
same vertex, so the same bit range carries a different, non-conflicting
meaning in each:

| bit           | opaque-vertex owner                     | water-vertex owner           | issue              |
|---------------|------------------------------------------|-------------------------------|--------------------|
| word 1 bit 31 | season kind (0 = ground, 1 = foliage)    | unused                         | #371 decision 1    |
| word 2 bit 20 | window `on` bit 0 (of `on:6`, bits 20–25)| `drop_16ths` bit 0 (of bits 20–23) | #371 d1 / #341 d2 |
| word 2 bit 21 | window `on` bit 1                        | `drop_16ths` bit 1             | #371 d1 / #341 d2  |
| word 2 bit 22 | window `on` bit 2                        | `drop_16ths` bit 2             | #371 d1 / #341 d2  |
| word 2 bit 23 | window `on` bit 3                        | `drop_16ths` bit 3             | #371 d1 / #341 d2  |
| word 2 bit 24 | window `on` bit 4                        | `flow_dir` bit 0 (of bits 24–27)| #371 d1 / #347 d2  |
| word 2 bit 25 | window `on` bit 5                        | `flow_dir` bit 1               | #371 d1 / #347 d2  |
| word 2 bit 26 | window `len` bit 0 (of `len:6`, bits 26–31)| `flow_dir` bit 2             | #371 d1 / #347 d2  |
| word 2 bit 27 | window `len` bit 1                       | `flow_dir` bit 3               | #371 d1 / #347 d2  |
| word 2 bit 28 | window `len` bit 2                       | unused                         | #371 decision 1    |
| word 2 bit 29 | window `len` bit 3                       | unused                         | #371 decision 1    |
| word 2 bit 30 | window `len` bit 4                       | unused                         | #371 decision 1    |
| word 2 bit 31 | window `len` bit 5                       | unused                         | #371 decision 1    |

That accounts for the 13 bits named above (1 + 12), all of them, with no two
owners claiming the same bit for the same vertex kind. The ice window is a
separate case, not one of these spare bits: the top water face's `tex_layer`
field (word 1, 12 bits — unused for water because water has one texture, a
pipeline constant) carries it as `on:6`/`len:6` instead (#371 decision 2).
Opaque vertices keep using `tex_layer` for its original purpose, the texture
array layer.

Decided in [#370](../../issues/370)/[#371](../../issues/371) (blocks 3.8a,
3.8b).

## §6 Cost budgets per issue

**Every block that touches generation measures its own cost against a baseline
taken before the terrain changes**, so a regression is caught at the block
that caused it rather than discovered later as "gate item 0 or 1 went red,
somewhere".

- `--bench worldgen` (added in [#349](../../issues/349), block 3.3a): per-column,
  per-chunk and per-far-patch generation cost.
- `--bench gate` / `--bench flight`: the existing frame-level budgets
  (`ROADMAP.md` exit gate items 0 and 1), which every block's `BENCHMARKS.md`
  row is measured against.

A block that regresses either number reports the delta in its PR rather than
shipping silently faster-or-slower; `ARCHITECTURE.md` Rule 7 applies to
worldgen exactly as it does to rendering.

---

## Later blocks (filled in as each lands)

Each block's PR adds its own `## §N <block>` section here, as phases 1 and 2
did. Block 3.1 (water) is the next world block; 3.0c below is the engine
carry-over that lands before it.

## §7 3.0c — Camera-relative rendering

**Nothing absolute reaches the GPU.** Every position the renderer uploads is
relative to a *render anchor* (`crates/render/src/anchor.rs`), an integer
block `[i32; 3]` snapped to a 4,096-block grid, so what is drawn does not
depend on how far the eye is from the world origin. Decided in
[#379](../../issues/379); the test is the frame itself: a synthetic scene
rendered at the origin and again shifted by (2^20, 0, 2^20) is byte-identical
(`golden.rs`, `a_scene_moved_by_a_grid_multiple_renders_byte_identically`),
and with the anchor forced to the origin — `origin/main`'s arithmetic — 47%
of its pixels move.

Why this had to come before the continents (`ROADMAP.md` 3.0 (c),
`PROPOSAL_FAR_VIEW.md`): the simulation is exact (`FixedVec3`, `i64` in
1/65536 of a block) but an `f32` at 50,000 blocks steps in 1/256 of a block,
and phase 3 puts coasts an hour's walk away. Standing there, a block one
block from the eye jittered by several pixels.

The decisions, as built:

- **Integers subtract exactly.** Node origins (`NodeKey::world_origin`,
  carried as `[i32; 3]` through `NodeGeometry` → `MeshedNode` →
  `ChunkArena::insert`), far tile origins (`PatchKey::origin`, `[i64; 2]`)
  and the voxels' hole are integers; `origin - anchor` is an `i64`
  subtraction and then a small `f32`. The eye is `f64` (`CameraPose::eye:
  DVec3`, from `FixedVec3::to_f64`, which is exact); `eye - anchor` is under
  2,048 + 256 blocks, so its `f32` error is at most 1/4,096 of a block.
  `view_proj` is built from that difference. Streaming keeps its `[f32; 3]`
  eye: it is chunk-granular.
- **One space for drawing and culling** (Rule 5). Both arenas and the far
  terrain store a `NodeBox` — integer origin plus bounds local to it — and
  the CPU frustum, built from the same anchor-relative `view_proj`, tests
  the box re-expressed through the one function `NodeBox::relative_to`.
  `Frustum::eye` is therefore the relative eye, and `faces_facing`'s 0.05
  margin is meaningful again at any distance. The local bounds come from
  the mesher as such (`cubara_voxel::build_mesh_bounded` returns node-local
  bounds, `NodeGeometry::local_aabb`, `MeshedNode::local_aabb`): no
  absolute `f32` box is ever formed and subtracted from again, so the cull
  is as exact at 2^30 blocks as at the origin. Every frame hands the anchor
  it draws against to the one scene pass explicitly (`SceneFrame::anchor`),
  which checks it against the arena's, the far terrain's and the tree
  arena's in debug builds — the anchor is held in four places and this is
  what fails when they disagree.
- **The one buffer that stays absolute.** The far terrain's heights are
  absolute world heights (a rebase never touches them), so `far.wgsl` takes
  `Frame.anchor.y` off every vertex and `PlacedPatch::aabb` off every cull
  box. Found in review (B1) and by the gate's flight eye at y = 3,000, where
  the anchor's y is 4,096 and the far terrain drew a grid step too high;
  pinned by `a_scene_moved_up_a_grid_step_renders_byte_identically`.
- **Rebase rarely, and cheaply** (decision 2, amended after review finding
  F16). The anchor moves only when the eye is more than 2,304 blocks from
  it on some axis — `anchor::next_anchor`, a pure function with its own
  tests: a boundary wobble never rebases, a `/tp` rebases on its first
  frame. A rebase rewrites one `vec4` per resident node from a CPU mirror
  of the origins buffer in a single `write_buffer`, the far tiles' 16-byte
  records in one bounded 64 KiB write (not spread over frames: a tile drawn
  against the old anchor while the eye is measured from the new one would
  be 4,096 blocks off — the very thing the anchor prevents; and not
  deferred to each tile's next upload, which for the horizon may be
  minutes away), the culling boxes, and the frame uniform. **Tree instances
  are node-relative** (`TreeInstance` carries a position relative to its
  node plus the node's slot in `node_origins`; `trees.wgsl` adds the origin
  from the same storage buffer `mesh.wgsl` reads), so a rebase touches no
  instance. Between rebases the cost per frame is zero: the check is three
  subtractions, and the clock is only read on the frame that rebases. What
  node-relative instances cost instead: `TreeInstance` is 16 bytes rather
  than 12, `trees.wgsl` reads one `vec4` from `node_origins` per tree
  vertex, and the tree pipeline binds a third bind group — the arena's
  origins, shared with `mesh.wgsl`.
  `anchor::rebase_if_needed` is the one place a rebase touches everything;
  the window and `--bench` both call it. A tree instance indexes its node's
  slot, so the slot and the trees are freed together — also when an edit's
  re-insert frees the slot and then fails on a full arena
  (`ArenaSink::try_insert`; review finding S1, pinned by
  `a_failed_reinsert_takes_the_nodes_trees_with_its_freed_slot`). The
  arena's CPU mirror of `node_origins` is the truth for the buffer: #380's
  R6 lift of `node_origins[i].y` must live in `NodeSlot` and go through an
  arena method that `insert` and `rebase` both apply, never a
  `write_buffer` from outside. Measured by `Renderer::last_rebase` through
  the ignored `rebase_cost_report` in `tests/offscreen.rs`
  (`cargo test -p cubara-render --release --test offscreen -- --ignored
  --nocapture rebase_cost_report`): 781 resident nodes (216 with trees) plus
  the far terrain's full 4,096 patches rewritten in **0.09–0.10 ms** on a GTX
  1060 (0.13 µs per node, so about 0.25 ms at the gate's ~1,940 nodes),
  against the 1 ms the decision allows. The lead's quiet-machine `--bench
  gate` A/B is the number of record in `BENCHMARKS.md`.
- **Absolute height where a shader needs it** (decision 4). The frame
  uniform carries the anchor; `haze.wgsl` adds its `y` back to the relative
  eye for the extinction's absolute height. Fog distance is relative
  already.
- **Goldens unchanged** (decision 6). Near the origin the anchor is
  `(0, 0, 0)` and every subtraction is the identity; the 18 reference
  images rendered byte-identically before and after on the same machine.

Out of scope, as the issue says: far terrain *generation* at huge
coordinates (worldgen noise precision) is its own question if it shows;
physics and the network were already fixed-point.

did. Block 3.1 (water) is next.

## §8 The world snapshot a mesh job holds (block 3.0c, [#387](../../issues/387))

An engine carry-over from the critical review of 2026-11-10 (F4), landed
before [#346](../../issues/346) because the flood step there would have paid
for it on every tick.

**What was wrong.** `World` was `derive(Clone)` and owned every edit; a mesh
job held an `Arc<World>`; `Server::tick_sim_all` and the client's `predict`
call `Arc::make_mut(&mut world)` every tick. While a job was in flight the
`Arc` was shared, so the tick copied every edit ever made. Measured
(`--bench edits`, 10,000 edits, a job requested every tick, as streaming
does while the player moves): 60 clones/s, 8.4 MB/s, the tick 7–8× the
empty world's.

**The design, binding on what follows.**

1. A mesh job holds a **`WorldSnapshot`** (`World::snapshot()`): the seed and
   the edit overlay by reference. Not block entities, not chunk states, not
   the `FluidQueue` #346 adds -- nothing that meshes reads them. One
   snapshot per request batch (`NodeStreaming::follow_visibility`,
   `invalidate`), shared by its jobs.
2. **Edits are filed by region behind `Arc`s** (`EditMap`: save/load's
   32-chunk region as the key, each region's `BTreeMap` behind its own
   `Arc`). `set_block` does `Arc::make_mut` on the outer map and on the one
   region it lands in. So an edit while a snapshot is held copies one entry
   per edited region plus that one region's edits, never the whole set; with
   no snapshot held it copies nothing.
3. `World` **derefs to** `WorldSnapshot`: every read of terrain and edits is
   the snapshot's, once (Rule 5). `World` adds what only the simulation owns.
4. **Nothing holds an `Arc<World>` across a tick.** `World::clone_generation`
   counts the clones; `--bench flight`, `--bench walk` and `--bench edits`
   print them, and `crates/server/tests/mesh_snapshot.rs` asserts the tick
   leaves every region map's pointer where it was.

**For #346.** The `FluidQueue` lives on `World` beside the edits (its decision
1 is unchanged) but is not part of `WorldSnapshot`, so a held mesh snapshot
never copies it; a flood step edits through `set_block` and so touches only
its own regions' maps. #346's PR keeps `a_held_mesh_snapshot_does_not_make_the_tick_clone_the_edits`
green and adds its own variant with the queue populated.

Iteration order over the edits is still total and fixed by the contents
(Rule 1) -- regions ascending, positions within each -- but no longer plain
position order; nothing depended on that (`dirty_chunks` sorts, the hash
reads chunks, a client applies a snapshot's edits in any order), and the
cross-platform fixture hashes are unchanged.

## §9 Render consistency: one ground for every level (block 3.0b, [#423](../../issues/423))

The owner's play-test of 2026-11-14: approaching a mountain, it changes --
*"from afar it is mostly stone; right in front of it, much more grass"*, and
*"the shape of the mountain is really different"*. Three renderers draw the
same ground (level-0 blocks, the coarse voxel levels 1-3, the far terrain),
and each derived it its own way. This section is the audit of those ways,
the measurement that defines "the same ground", and the one rule that
replaces them. Written with PR 1 (the harness and the API, no behaviour
change); PR 2 (the voxels on the rule) and #422 (the far terrain on it) add
their measured results below.

### 9.1 Audit: how each representation derived the ground (main at `4fe57c4`)

| | height and shape | material per face | light, AO, texture |
|---|---|---|---|
| **Level 0** (`WorldGen::block_at`, `Chunk`, `mesh.wgsl`) | `surface_height(x, z)` per column | by depth below **that column's** surface: grass, `SOIL_DEPTH` of soil, stone (`material_at`) | `face_light` per face; AO baked per block corner (`face_ao`); one tile per block |
| **Voxel levels 1-3** (`generate_aniso`, same mesher and shader) | `cell_surface_height`: the mean of four columns of the cell, rounded down; at level 3 rounded again to rows of 4 | `material_at` by depth below the **cell's** surface at the row's own `y` (`probe`), the top cell forced to grass (`top_cell`, #296) | the same light; top-face AO scaled by the cell's width (R1c), wall AO not; one tile per cell, so stretched 2^L times |
| **Far terrain** (`far::generate`, `aggregate.rs`, `far.wgsl`) | per vertex, the mean of up to 4 x 4 columns over a quad-wide square; drawn smooth | per vertex: `riser_colour` of the **smoothed** gradient across two quads -- a staircase model, layered by depth below each step | `face_light`, weighted by `visible_weights`; **no AO**; mean colours, no texture |

Every divergence that follows from that table:

1. **Height.** A coarse cell's top is a mean of four samples, the far
   terrain's a mean of sixteen: neither is level 0's, and both sit under
   every peak and ridge. Level 3 rounds its mean to rows of four as well.
2. **Riser material (the owner's stone).** A cell `w` blocks wide on a slope
   of `s` meets its neighbour at one riser `w * s` tall, and wears soil and
   stone below its first block by the *riser's* depth. Level 0 covers the
   same slope with `w` risers of `s`, each the grass side of its own column.
   The far terrain has the opposite error: its smoothed gradient is below
   the block-scale one (`E|grad h| >= |grad E[h]|`), so it shows no stone
   short of a mean slope of 4.
3. **Riser area.** Coarse levels and the far terrain draw less riser than
   level 0 (bumps inside a cell or under a quad vanish).
4. **Top-face AO.** On a slope every level-0 top carries the shade of the
   step above it. A coarse top's shade is scaled away with the cell's width
   (right for one cliff, wrong for a staircase of `w` steps), and the far
   terrain has none. This is the largest colour difference on any slope.
5. **Wall AO.** A level-0 riser is shaded at its foot, every block of a
   staircase; a coarse riser once, at the foot of the whole wall.
6. **Texture scale.** One tile per cell rather than per block. The mean
   colour is the same; the pattern is 2^L times larger (#380 found this and
   the AO to be most of what changes land-to-land at a swap).
7. **Caves.** Level 0 carves them at every depth and through the surface; a
   coarse cell was cave by a vote -- at levels 1 and 2 the noise at one of
   its blocks, at level 3 most of eight samples near its middle -- and only
   within `DISTANT_CAVE_DEPTH` of the surface; the far terrain has none. So
   a level could open a pit over ground the next finer level drew whole.
   PR 1 left it as it was (PHASE1 §8.6) and the harness counts it apart from
   the rest; PR 2 makes coarse caves nest (§9.4).
8. **Trees** stand on `drawn_top` of their own column, not on the cell that
   draws the ground under them (R2's recorded risk).

### 9.2 The definition: `crates/render/tests/lod_consistency`

"Consistent" is a measurement, GPU-free, over a **seeded** set of 64-block
footprints -- ten each of rolling hills, gentle grassy slopes (the owner's
case), steep mountain faces and cliffs, drawn by a printed seed, not picked:

- **Voxels** are measured from the real thing: the node the game generates
  and the quads the game's mesher makes of it, each quad its texture's mean
  colour, lit by `aggregate::face_light` and darkened by its own baked AO
  through `aggregate::ao_fraction` -- the shader's CPU twins. Only what the
  sky can see is counted (a top that is the highest in its column, a side
  whose front cell is above that column's top), each level judged against
  its own drawn tops. `level_0_draws_its_surface` holds that walk to a
  block-by-block count of the world, exactly.
- **The far terrain** is measured from what `far.wgsl`'s `shade` says is
  there: one unit of top per unit of ground, `|g|` of riser along each axis,
  coloured by `riser_colour`, the gradient taken as the shader takes it.
- Per representation, against level 0 over the same footprint: **shape**
  (the drawn top over each column, the skyline, the peak, in blocks; level
  0 is the height field `surface_height + 1`; a voxel level is compared
  over the columns of its surface term, below), **T3** (the area-weighted
  lit colour per face direction, /255 in linear light as T3 has been
  measured since R1c), and **the stone share** per direction.

- **The reference is the surface; caves are a separate term** (review of
  #428, finding B1). What the sky sees is split on level 0's own columns:
  the *surface* term is the faces over ground whose level-0 columns all
  stand at the height field, and the *cave* term is the faces of, or
  facing, ground where a cave has carved a column's top away (the floor of
  a cave mouth and the walls it bares). **Every level is held to the
  surface term, and only to it.** The cave term is reported and never
  asserted, because the levels differ there by design: level 0 draws every
  cave mouth, a coarse cell is cave only where all of its ground is
  (§9.4; by a vote, when this was first written) and only within
  `DISTANT_CAVE_DEPTH` of the surface, the far terrain draws none. A cave
  mouth is detail. It is 4 to 12% of level 0's sky-visible area and about
  half stone, so counted in with the surface it made a steep slope look
  4.0% stone where its surface is 0.1%. (Since §9.9 a pit is ground at
  every level: a cell over one sinks by its share of it. The split stands
  -- the harness's colours and shapes are still held on the surface term
  -- and what a level draws of the pits is held by `pit_study.rs`: the
  air, the pits shown, the nodes themselves.)
- **A column is carved where level 0 is not the height field**: its drawn
  top is not `surface_height + 1`, or a block of a side that a lower
  neighbour bares is missing (a cave under an intact top). Everywhere else
  level 0's faces *are* the height field's, face for face and layer for
  layer, and `level_0_draws_its_surface` asserts exactly that from
  `surface_height` and `ground_layer` alone.
- **Both sides of a comparison cover the same ground** (second review,
  point 1). A cell or a far quad with one carved column under it goes to
  the cave term whole, so wider cells leave more ground out (steep: 6% of
  level 0's area, 23% of level 3's). Level 0 is therefore measured again
  for each comparison, **split on the cells of the level it is compared
  with** (`Drawn::measure`'s `mask`), and the shape of a voxel level is
  taken over the same columns (`Shape::over`).
  `both_sides_of_a_comparison_cover_the_same_ground` holds the two sides to
  the same count of columns. It moved the pooled numbers by at most 0.1/255
  and 0.1 point of stone share; the shape rows moved more (a cave under a
  footprint's peak had been read as a lost peak, at level 0 too).
- **Exactly the same ground for the tops; not quite for the sides** (third
  review of #428, S3). A side face goes to the cave term when the mask cell
  of its own column *or of the column in front* is carved. A level's riser
  always stands between two cells, so it needs both whole; level 0, split
  on the same mask, keeps every riser between two columns of *one* whole
  cell. Where a whole cell has a carved neighbour, level 0 keeps the
  staircase inside it and the level loses the one riser that stands for
  it. The tops are not touched (`compare` holds the two sides to the same
  count of them). The side *area* is: on steep ground level 3 showed x0.75
  of level 0's side area on the surface term where over every face it
  showed x0.81 (level 2: x0.89 against x0.92; measured on the first version
  of PR 2). So the report's `area / level 0's` column understates a level's
  risers by six to eight percent at level 3 and three to four at level 2.
  T3 is a mean colour and hardly moves with it. **Riser area is reported
  and not bounded, and must not be bounded with that ratio as it stands.**
- **Per footprint, not only pooled** (finding B3): every comparison is made
  footprint by footprint, the report prints the worst single footprint
  beside each pooled number, and the assertions hold the worst footprint.
- **A direction is ranked by level 0, and a level that lacks it fails**
  (second review, point 3). A direction of a footprint is held to a
  tolerance when level 0 has at least twice the area a colour is taken
  from; a level with under that area there is `Missing`, which no tolerance
  admits. The area is sixteen of the level's own cell faces and never under
  64 square blocks: 64 at levels 0 to 2, 512 at level 3, whose cell side is
  8 x 4 (all chosen, not derived; a colour from two faces is two samples).

`consistency_report` (ignored; `cargo test -p cubara-render --test
lod_consistency -- --ignored --nocapture consistency_report`) prints all of
it. **Before** (main at `4fe57c4`; the surface term, pooled over ten
footprints a kind, with the worst single footprint in brackets):

| steep mountain | L1 | L2 | L3 | far, quad 8 |
|---|---|---|---|---|
| stone share of side faces (level 0: 0.1%) | 6.6% | 30.3% | 39.8% | 0.0% |
| T3, tops (+Y), /255 | 6.9 (7.9) | 11.3 (12.6) | 14.1 (15.1) | 16.5 (18.0) |
| T3, worst side direction, /255 | 2.0 (4.3) | 8.1 (13.9) | 10.4 (15.7) | 4.4 (6.2) |
| peak, blocks | -1.1 | -3.0 | -6.1 | -3.9 |
| skyline, mean, blocks | -0.9 | -2.0 | -3.8 | -2.7 |

| | L1 | L2 | L3 |
|---|---|---|---|
| stone share of sides, cliffs (level 0: 0.7%) | 11.2% | 37.7% | 47.1% |
| stone share of sides, gentle slope (level 0: 0.0%) | 0.0% | 4.0% | 9.5% |
| T3 tops, gentle slope | 2.5 (4.5) | 5.3 (8.2) | 8.0 (10.7) |
| T3 tops, rolling hills | 0.26 (0.4) | 1.2 (1.5) | 3.6 (4.0) |

The cave term, for the record (share of each level's sky-visible area on or
beside carved ground, steep mountain): level 0 6.1%, L1 8.2%, L2 12.9%, L3
23.3% -- it grows with the cell, since one carved column puts its whole cell
there -- and 61% of level 0's cave side faces are stone.

On the rolling hills level 1 was within 2/255 in every direction of every
single footprint (worst 0.59); that was the one thing the harness could
assert of main's coarse levels
(`level_1_matches_level_0_on_the_rolling_hills`). R1c's own T3
(`t3_coarse_ao.rs`) passed because it sampled around the origin, which
`MOUNTAIN_SEED_MIX` keeps in the hills; PR 2 removes it, since it worked
R1c's shade out for itself and the harness measures what the mesher made.

**The measurement is itself seen to fail** (finding B2): non-ignored tests
go red when the level-0 reference is swapped for a coarse level, when every
face is lit as a top (`level_0_draws_its_surface`), when a far row's riser
direction is swapped or its tops are shifted
(`the_far_rows_follow_the_ground`), and when a level-3 node's side area
forgets its rows are four blocks tall or its top is placed as if they were
one (`a_level_3_node_is_measured_in_blocks`, a node made by hand).

### 9.3 The rule

**One pure column function, and every level derives its ground from it**
(Rule 5; #423 decisions 1, 2). All in `cubara_world::WorldGen`:

```rust
/// The height field: the y of a column's surface block, before caves.
pub fn surface_height(&self, x: i32, z: i32) -> i32;

/// The column function, and level 0 itself: the y of the highest block of
/// ground the sky sees in a column -- its surface block, or the floor of
/// the pit a cave has opened there (§9.9). `column` gives both.
pub fn column_top(&self, x: i32, z: i32) -> i32;
pub fn column(&self, x: i32, z: i32) -> GroundColumn; // { surface, top }

/// What a column is made of `depth` blocks below ITS OWN surface.
pub fn ground_layer(depth: i32) -> GroundLayer; // Surface | Soil | Stone

/// A cell's drawn top block over the columns [x0, x0 + w) x [z0, z0 + w):
/// the MEAN of their column_top, rounded to the nearest block (a half
/// to the even one) -- and, where the cell holds a summit (the first
/// highest column of the nine cells around it is one of its own), the
/// highest MEAN OF ITS FOUR HALF-WIDTH CELLS. cell_top(x, z, 1) is
/// column_top(x, z).
pub fn cell_top(&self, x0: i32, z0: i32, w: i32) -> i32;

/// The same over an n x n grid of the cell's columns, centred in the cell,
/// the nine cells around each sampled on its own grid: the single
/// implementation (cell_top is this at n = w).
pub fn cell_top_sampled(&self, x0: i32, z0: i32, w: i32, n: i32) -> i32;

/// What level 0 shows over a square: riser area per direction and layer,
/// and how open (un-occluded) its tops and risers are.
pub fn ground_faces(&self, x0: i32, z0: i32, w: i32, n: i32) -> GroundFaces;
```

- **Height: the mean of the covered columns, to the block, with summits
  lifted; at every level.** A coarse cell's top, a far quad's top and a
  level-0 column's top are `cell_top` at three widths. **A column stands
  where the sky sees it** (`column_top`): at its surface block, or, where
  a cave has carved that away, at the floor of the pit -- so a pit is
  ground like any other, and a cell over one sinks by the share of its
  columns that are pit times their depth. (Until §9.9 the mean was of the
  *surfaces*, and a pit showed only where a cave had carved every column
  of a cell: it appeared out of flat grass on the way in. §9.9 has the
  study that chose this.) The mean is rounded to the nearest block, a
  half to the even one, so that rounding lifts nothing. A cell that **holds a summit** -- the first highest column (`z`,
  then `x`) of the nine cells around it is one of its own -- stands at the
  **highest mean of the four half-width cells it splits into** instead
  (the summit study below; until then it stood at that highest column).
  The reasons, in §9.4's study and the summit study:
  - a swap replaces a cell by the four finer cells under it, so what
    changes on screen is the coarser top minus the finer ones. Under the
    mean that difference has no direction: as many columns stand above
    the coarser cell as below it, and a slope neither sinks nor rises as a
    ring edge passes over it;
  - the mean alone lowers what stands out, and a peak would grow back
    edge by edge: six blocks under level 0 at level 3 on steep ground;
  - the highest column keeps a peak exactly but turns one high column
    into a tower the cell's width, and a crest into a row of them (the
    lead's mountain zoom test, 2026-11-21: "pillars and spires along the
    crests" at level 3 that level 1 does not show). The mean of the
    highest half-width cell is what the next finer level would draw there
    by the mean alone, so a coarse summit is never more than twice as wide
    as ground that really stands that high -- at level 1 its halves are
    single columns and it stands at the highest column, as before -- and a
    peak erodes a step per level instead of jumping from its column to
    the whole cell (`a_lone_high_column_raises_no_tower`,
    `a_coarse_crest_grows_few_towers`,
    `the_owners_mountain_keeps_its_peak`).

  **Decision 1 of #423 named the max.** PR 2 was first built on it and its
  review measured what it does: every slope stands higher the coarser the
  level and sinks 7 to 9 pixels at each ring edge on the way in. The max is
  not what is built; the study and the reason are in §9.4, and the lead
  asked for the rule to be chosen by that measurement (2026-11-20).

  **The tops do not nest as the max's did.** Nothing may assume a coarser
  level stands at or above a finer one: `border_samples` asks every
  neighbour it could have what it draws, and that is what keeps the seams
  closed. Nor may anything assume a coarser level shows every opening a
  finer one does: the visibility search did, of a node that had just
  split, and that was a hole (§9.5).

  **Level 3 stores rows of four blocks and is drawn to the block all the
  same** (PR 2, §9.4): the row that holds a column's top is cut off at it.
  PR 1 gave `cell_top` a row height and rounded to it, because level 3 drew
  its rows whole: -1, +2, +1 or 0 blocks off by the top's height mod 4, so
  on a plain at 1 mod 4 level 3 stood two blocks above level 2, a wall
  ringing the player 512 blocks out (the lead's audit, 2026-11-20). The
  parameter is gone with the rounding.
- **Material: by depth below each column's own surface.** `ground_layer` is
  the only depth rule; `material_at` reads it. A coarse cell wears the
  layer of one of the level-0 side faces its columns show at its rows --
  a column shows one at every row above a neighbour's top and at or under
  its own, the wall of a pit among them (§9.9), each made of
  `ground_layer` at its depth below its **own** column's surface --
  picked by a hash of the cell's position, so a gentle slope's
  coarse riser is grass side nearly row after row, and stone shows where a
  covered column really stands more than `1 + SOIL_DEPTH` blocks proud
  (PR 2, §9.4). A far wall, which cannot band, takes the mix
  `ground_faces(..).sides[d]` gives for its quad's footprint (#422).
- **Light: level 0's openness.** `ground_faces` carries how open level 0's
  tops and risers are, from the mesher's own corner rule
  (`cubara_voxel::face_ao_of`, which `face_ao` now delegates to), so a
  representation with no blocks to occlude each other can still be lit
  `ao_floor + (1 - ao_floor) * open`, as `mesh.wgsl` lights a block.
- **`ground_faces` is held to the real mesher**:
  `ground_faces_are_what_the_level_0_mesher_draws` counts the game's level-0
  quads over the same square and requires every count equal -- tops, and
  riser area per direction and layer -- wherever no cave touches the
  surface. (Since §9.9 it counts the columns' tops, so a pit's walls and
  the risers of its floor are in it, stone under the soil: what a far wall
  beside a sunk quad is mixed from.) The openness is within 0.08 on tops and 0.14 on a direction's
  risers, and the test shows where the rest goes: **the mesher bakes a
  chunk's AO without looking into the next chunk**, so faces at a chunk's
  edge come out more open than the height field says (up to 0.06 of
  openness on steep ground, about 2/255 on a grass top). Asked with that
  blindness put in, the same corner rule gives the mesher's number to the
  last bit. `ground_faces` reports the true neighbours; the chunk-edge
  blindness is a level-0 mesher property, recorded here and not fixed by
  this block.

**The far-terrain interface** (#422 calls these; nothing else decides a far
quad's height or a far face's colour):

- quad top face = `cell_top_sampled(x0, z0, q, n) + 1` (`far::quad_top`,
  the one place a quad's height is decided), with `n = q` up to 8 blocks and
  `n = 4` above. At `q = 8` a far quad and a level-3 cell cover the same 64
  columns and get the same number, so the 1,024 m join is exact (both are on
  the 8-block grid) -- and there is no row height left to get wrong.
- above 8 blocks the exact form costs `q^2` heights a quad (and a look at
  the quads around one that might hold a summit), so the far terrain calls
  `cell_top_sampled` with `n = 4`: the same rule over 4 x 4 columns centred
  in the quad (a summit at the highest mean of the grid's four 2 x 2
  quarters). Its mean is a sample of the quad's, it can miss a peak
  narrower than the spacing, and its summits do not nest (a quad half as
  wide looks at columns its parent skips); the harness's `stepped` rows
  measure it per quad width, below, and §9.8 measures it against level 0.
- wall colour facing direction `d` = the `Materials` mixed by
  `ground_faces(x0, z0, q, n).sides[d]`, times the openness. Not
  `riser_colour` of the wall's own height, which is divergence 2 again.
- top colour = what a voxel cell's top wears over the same columns
  (`top_layer(x0, z0, q, n, top)`, the far terrain's form of `top_block`;
  §9.6, §9.8): `Materials.top`, or the soil or stone of a pit's floor,
  times `top_open`.

**What the rule does to the shape, per cell width** -- measured, not guessed
(`stepped_top_rule_report`; drawn minus level 0 on steep ground, mean over
columns, with the highest column -- the first version of #423 -- beside it):

| cell width | `cell_top` | `cell_top_sampled`, `n = 4` | the highest column | `cell_top` with a summit at its highest column (#432) |
|---|---|---|---|---|
| 2 (level 1) | +0.01 | | +1.2 | +0.01 |
| 4 (level 2) | +0.02 | | +3.5 | +0.02 |
| 8 (level 3, far join) | +0.06 | +0.09 | +8.0 | +0.09 |
| 16 | +0.3 | +0.3 | +16.6 | +0.6 |
| 32 | +2.0 | +1.4 | +30.9 | +3.8 |
| 64 | +11.4 | +10.2 | +50.1 | +24.7 |

Up to 8 blocks the ground stands where level 0's does, to a tenth of a
block. **Past 16 blocks the summits take over**: one cell in nine is the
highest of its nine at any width, and a cell 64 blocks wide that holds a
summit stands at the mean of its highest 32-block half, which on steep
ground lifts the columns 11 blocks over level 0 in the mean (25 when it
stood at its highest column). That is the max's swelling again, in fewer
cells. Whether a far
quad that wide should keep its summit, or stand at its mean, was #422's to
measure with these rows (§9.8: it keeps the rule); the voxel levels stop at 8.

**How high a summit stands: the summit study** (2026-11-21, after #432).
#432 stood a summit at its highest column. The lead's mountain zoom test
(the far view cropped and enlarged against the near view) found "tall
vertical pillars and spires along the crests" at level 3, 700 blocks out,
that level 1 at 220 does not show: one high column at level 0 is a needle
one block wide, and the cell that holds it drew a tower the cell's width.
The candidates were each worked out from the height field alone (the
default seed, 1,792 x 1,792 columns around the owner's mountain at about
(-3072, -3232), its peak 239), and the best of them through the study of
§9.4 as well:

- **towers**: cells standing at least their own width above all four cells
  beside them -- a block as tall as it is wide, or taller, standing alone;
- **summit loss**: over the cells that hold a summit standing above their
  mean, the highest column minus the drawn top, mean / most;
- **the owner's peak**: the highest drawn top minus level 0's;
- **step**: how far a column's drawn top moves from one width to the next
  (what a ring swap changes, and what a height morph has to slide). Its
  50th and 95th percentiles are 0 / 1, 0 / 2, 1 / 4 and 1 / 8 blocks from
  width 1 to 2, 2 to 4, 4 to 8 and 8 to 16 under *every* rule: slopes set
  them, not summits. The largest is what a rule decides;
- **tip rise**: at each level-3 summit's highest column, the most its drawn
  top rises at one swap on the way in (the highest column never moves);
- **steep footprints**: §9.4's study, the summit at levels 1 / 2 / 3 against
  level 0 in blocks, and the outline's mean change at the 160 / 288 /
  512-block edges in pixels at 1080 rows.

| summit stands at | towers, w = 2 / 4 / 8 / 16 | summit loss, w = 4; 8; 16 | owner's peak, w = 2 / 4 / 8 / 16 | largest step, 1→2 / 2→4 / 4→8 / 8→16 | tip rise, most | steep: summit L1 / L2 / L3 | steep: outline, px |
|---|---|---|---|---|---|---|---|
| its highest column (#432) | 170 / 87 / 53 / 34 | 0 | 0 / 0 / 0 / 0 | 5 / 14 / 28 / 56 | 0 | 0 / 0 / 0 | -0.9 / -0.8 / -0.8 |
| its mean (no summit) | 34 / 4 / 3 / 1 | 1.4/8; 2.3/16; 4.9/35 | -1 / -2 / -4 / -12 | 4 / 7 / 13 / 22 | 11 | -0.8 / -2.0 / -6.0 | -1.6 / -1.7 / -2.3 |
| highest, lift capped at w/2 | 134 / 43 / 18 / 10 | 0.2/6; 0.5/12; 1.3/27 | 0 / -1 / 0 / -4 | 4 / 9 / 16 / 29 | 9 | 0 / -0.4 / -3.8 | -1.0 / -1.2 / -1.6 |
| highest, lift capped at w/4 | 34 / 15 / 9 / 5 | 0.4/7; 0.9/14; 2.2/31 | -1 / -2 / -2 / -8 | 4 / 8 / 14 / 25 | | | |
| #432's rule over the four children's tops (recursive) | 170 / 89 / 52 / 35 | 0.0/2; 0.0/5; 0.0/3 | 0 / 0 / 0 / 0 | 5 / 14 / 29 / 56 | | | |
| the 75th percentile of its columns | 64 / 19 / 13 / 5 | 0.9/5; 1.4/10; 2.9/21 | -1 / -2 / -2 / -6 | 4 / 10 / 21 / 35 | 6 | -0.6 / -1.0 / -3.0 | -1.3 / -1.4 / -1.5 |
| the 90th percentile | 170 / 55 / 25 / 11 | 0.3/2; 0.7/5; 1.5/12 | 0 / -1 / -1 / -2 | 5 / 14 / 25 / 47 | | 0 / -0.4 / -1.6 | -0.9 / -1.0 / -1.3 |
| the highest w/2 x w/2 block's lowest column | 170 / 15 / 8 / 3 | 0.9/5; 1.8/12; 4.2/34 | 0 / -2 / -4 / -11 | 5 / 11 / 16 / 25 | | | |
| the highest mean of any w/2 x w/2 window in it | 170 / 41 / 18 / 9 | 0.5/3; 0.9/6; 2.0/14 | 0 / -1 / -2 / -4 | 5 / 13 / 23 / 43 | | | |
| the mean of the half-width cell holding its highest column | 170 / 31 / 12 / 9 | 0.7/3; 1.2/8; 2.5/15 | 0 / -1 / -3 / -4 | 5 / 13 / 23 / 43 | 6 | | |
| the highest mean of its w/4-wide cells | 170 / 87 / 42 / 19 | 0/0; 0.4/2; 1.0/6 | 0 / 0 / -1 / -2 | 5 / 14 / 27 / 52 | | | |
| **the highest mean of its four half-width cells (chosen)** | **170 / 33 / 12 / 9** | **0.5/3; 1.0/6; 2.3/15** | **0 / -1 / -2 / -4** | 5 / 13 / 23 / 43 | **4** | **0 / -0.8 / -2.2** | -0.9 / -1.3 / -1.4 |

The surface's mean change at every edge stays 0.0 to +0.1 pixel under
every rule here: summits are one cell in nine, and the cells that hold
none stand at their mean whatever the summit rule.

What the table says:

- **The tower count is set by how far one column can lift a cell.** The
  highest column lifts it all the way; the recursive form of the same rule
  ends at the same column (the child that holds it holds a summit too) and
  changes nothing. A cap on the lift is blind to how wide the high ground
  is: capped at w/2 it still leaves 18 towers at level 3, and drops a
  summit 12 blocks where the high ground is a single column.
- **A mean over part of the cell is what tells a needle from a peak.** One
  column counts for a quarter, a sixteenth and a sixty-fourth of its height
  at widths 4, 8 and 16 (`a_lone_high_column_raises_no_tower`), while a
  peak whose high ground fills half the cell keeps nearly all of it. The
  four half-width cells beat the 75th percentile on every count at level 3
  (12 towers against 13, a summit 1.0 / 6 blocks under against 1.4 / 10)
  and keep level 1's summit exact, which the percentile does not; the
  stricter tests (a whole w/2 block, or the plain mean) lose the owner's
  peak by 4 blocks at level 3, which is the old main the lead's zoom test
  rated worst.
- **On the way in a summit rises a step per swap.** Its tip rises at most
  4 blocks at one swap (3 in 95 of 100), where the mean's grew by up to 11.
  It sinks at some swap at 42 of 623 level-3 summits whose tip moves: a
  sibling half-width cell can stand higher by its mean than every quarter
  of the one that holds the highest column. The variant that takes the
  half holding the highest column sinks at 7, but loses more (1.2 / 8 at
  level 3, the owner's peak -3).
- **What is left are wide high grounds.** Two towers stand at level 3 in
  the 512 x 512 blocks around the owner's peak (`a_coarse_crest_grows_few_towers`
  holds the count; #432 drew 12). Each stands at the mean of one of its
  half-width cells, so level 2 draws that ground at least as high: it is
  there on the way in, narrower, not a tower that melts.


### 9.4 The voxels on the rule (PR 2)

Built in two rounds. The first put the coarse levels on #423's decisions as
written -- a cell's top at its highest column -- and fixed colour, material,
shade, peaks and the step at the 512-block ring. Its review measured what
was left: the *shape* of a slope still moved at every ring, by as many
pixels as before and now always the same way. The second round chose the
height rule by that measurement, made coarse caves nest, and closed the
review's findings. What follows is the tree as built.

What changed, all of it behind the functions of §9.3:

- **Height.** `cell_surface_height` is `cell_top`: the mean of the covered
  columns, or the cell's summit. `generate_aniso` (and `generate` at a step
  above one, which is the same rule at `v = step`, drawn whole, and built
  only by tests) asks every column of the node's footprint once
  (`node_ground`), keeps each cell's columns sorted, and works each cell's
  top out with the one body the standalone functions use (`top_over`). A
  cell at the node's edge that might hold a summit asks the cells beyond it
  as well.
- **Level 3's top is cut off at the exact top.** It still stores rows of
  four blocks (`NodeKey::cell_height_blocks`, #327 decision 1: generation
  and caves are per row of four). A row is solid while it holds any ground,
  and the one that holds a column's top is drawn only up to it: the node
  carries its columns' exact tops (`AnisoChunk::with_tops`) and its mesh is
  built on block rows (`mesh_dims`, `NodeKey::drawn_row_blocks`). So every
  level's top face is `cell_top + 1`, a riser between two level-3 cells is
  as tall as their tops differ, and level 3 joins the far terrain exactly.
  `border_samples` asks every level about the block, not the row: a
  neighbour's cell is solid at a block when the block is at or under its
  top and its row is not cave (`WorldGen::cell_draws_solid`, held to the
  generated node block by block).
- **Caves nest.** A coarse cell-row is cave only where **every block of
  ground in it is cave at level 0** (`WorldGen::row_is_cave`): every block
  of the cell's columns, in those rows, that is at or under its own
  column's surface and the cell's top. One block of rock and the cell stays
  whole. A finer cell inside it covers some of the same blocks, so whatever
  a level draws open every finer level draws open too, and **a pit only
  ever opens on approach** (so it was until §9.9: a pit is now in the
  *tops*, which never nested, and is drawn at every level; what nests is
  the caves under a top). The depth limit nests as well: a row is carved
  only if its lowest block is within `DISTANT_CAVE_DEPTH` of the cell's
  highest column. It replaces a vote (§9.1 item 7; PHASE1 §8.6 is amended).
  The first block of rock ends the count, so a cell still costs one noise
  sample nearly always.
- **Material.** `coarse_block` replaces `block_at_on_surface_v`, `top_cell`
  and `probe`. At level 0 a column shows a side face at every row above a
  neighbour's surface and at or under its own, made of `ground_layer` at
  that row's depth below its **own** surface. A coarse cell wears the layer
  of **one of the level-0 side faces its columns show at its rows, picked by
  a hash of the cell's position**, so over many cells every layer shows in
  the share it has at level 0. (Two rules came before and each was a
  measured error: depth below the cell's own top, which is main's stone;
  and the lowest covered surface at or above the row, which never showed
  stone and hardly any soil either.) The cell that holds the top wears the
  surface block. **A mean top's riser reaches below every column the cell
  covers**, down to the top of the next cell: rows level 0 shows from that
  cell's columns. Such a row wears one of *all* the cell's level-0 faces, by
  the same hash; by depth alone those rows came out as bands of soil and
  stone along the foot of every riser. Under the riser's foot a row is the
  inside of the ground and wears its depth.
- **Top-face AO.** The shade a riser casts on the coarse top at its foot is
  `min(riser height, cell width)` blocks wide, not always one (`top_ao` in
  `cubara_voxel`): a riser `k` blocks tall on a slope stands for `k` level-0
  steps, each shading its own column. R1c's rule (#327 decision 5) was right
  for a single step and made every coarse slope lighter than its blocks. The
  band is a power of two (the vertex has two bits), a tie going down; it is
  set by the riser *beside* the corner, a diagonal one only where it stands
  alone; and an occluder darkens a corner's band only if it is at least half
  as tall as the riser that sets it.
- **Wall AO.** A coarse riser's foot shade is carried up the wall, as many
  rows as the riser is tall and no further than the cell is wide, no darker
  than one occluder (`wall_ao`): at level 0 every step of the staircase the
  riser stands for is shaded along its own foot. It needs no vertex data:
  the rows merge into one quad and one ramp, whose mean is the mean of a
  ramp on each row. (A riser one row tall is its foot, shaded as level 0
  shades it.)
- **Texture scale.** A coarse face tiles once per block (`push_quad`
  multiplies the tile coordinate by the cell's size), so it shows the tile
  at level 0's size and picks level 0's mip (R4).
- **Trees** stand on the top of the cell that draws the ground under them
  (`node_trees` calls `cell_top` for the node's cell). `drawn_top` is gone.
- **Visibility links are those of what is drawn.** `node_links` floods the
  lattice a node is drawn on (`NodeChunk::drawn` over `mesh_dims`). A
  level-3 node's stored rows of four are solid in the one to three blocks
  of air above every top; worked out on those, the links called air rock,
  which is the unsafe direction for a search that must never leave out
  what a line of sight reaches (review of #432, B2: the search left out
  nodes the drawn lattice keeps from 20 of 48 eye positions; no hole was
  found). `the_links_of_a_node_are_those_of_what_it_draws` compares the
  two on 109 generated level-3 nodes, 28 of which the stored rows get
  wrong.

**The height rule was chosen by what a swap changes on screen**
(`height_rule_study_report`, `crates/render/tests/lod_consistency/study.rs`).
The owner's requirement, 2026-11-20: approaching a mountain, it must not be
seen to shift or change at all; only detail may appear. A swap replaces a
coarse cell by the finer cells under it, so per level-0 column what changes
at a ring edge is the coarser level's top minus the finer one's. Each
candidate rule is worked out from the height field alone over the harness's
forty seeded footprints, and judged in pixels at the distance of the edge
(60 degrees of vertical field of view; one block is 6.4, 3.6 and 2.0 pixels
at 160, 288 and 512 blocks on 1080 rows, and 1.33 and 1.54 times that on
the 1440 and 1664 rows the owner plays at):

- **the surface**: the difference over every column. Its mean is the
  systematic part -- the whole slope sinking (+) or rising (-) as the edge
  passes over it; its 95th percentile is the size of what changes at all;
- **the outline**: the same for the skyline, the highest top along a line
  of sight (along x, along z and both diagonals, in a window of 192 blocks),
  on lines where level 0 has a crest inside the window;
- **the summit**: a level's highest top against level 0's, where a
  footprint holds one.

Steep mountain, pixels at 1080 rows, at the 160 / 288 / 512-block edges:

| rule | surface: mean | surface: 95th pct | outline: mean | outline: 95th pct | summit, blocks, L1 / L2 / L3 |
|---|---|---|---|---|---|
| highest column (the first version) | +7.5 / +8.3 / +9.0 | 19.3 / 25.1 / 26.2 | +5.3 / +5.5 / +5.5 | 19.3 / 17.9 / 20.1 | 0 / 0 / 0 |
| mean, rounded down (as main, of all columns) | -2.2 / -0.4 / 0.0 | 12.9 / 14.3 / 14.1 | -3.9 / -2.1 / -2.3 | 12.9 / 10.7 / 12.1 | -1.2 / -2.2 / -6.2 |
| mean, a half rounded up | +1.1 / -0.4 / -0.1 | 12.9 / 14.3 / 14.1 | -0.5 / -2.2 / -2.3 | 12.9 / 10.7 / 12.1 | -0.8 / -2.0 / -6.0 |
| mean, a half to the even block | 0.0 / 0.0 / 0.0 | 12.9 / 14.3 / 14.1 | -1.6 / -1.7 / -2.3 | 12.9 / 10.7 / 12.1 | -0.8 / -2.0 / -6.0 |
| median | -3.0 / +0.6 / +0.4 | 19.3 / 14.3 / 14.1 | -3.8 / -0.9 / -1.7 | 12.9 / 10.7 / 12.1 | -0.8 / -2.0 / -5.4 |
| 75th percentile | +3.0 / +3.1 / +3.5 | 19.3 / 14.3 / 18.1 | +1.0 / +1.4 / +0.9 | 12.9 / 10.7 / 12.1 | -0.6 / -1.0 / -2.6 |
| midrange | +1.7 / 0.0 / 0.0 | 12.9 / 14.3 / 14.1 | +0.2 / -2.4 / -3.2 | 12.9 / 10.7 / 12.1 | -0.4 / -2.0 / -6.8 |
| highest column where the cell's is above both neighbours' along a line, else mean | +1.7 / +1.4 / +3.6 | 12.9 / 17.9 / 26.2 | +4.8 / +5.4 / +5.3 | 19.3 / 17.9 / 20.1 | 0 / 0 / 0 |
| highest column once the cell's own slope is taken out | +2.2 / +1.0 / +2.0 | 12.9 / 14.3 / 16.1 | +0.9 / +0.8 / +1.0 | 12.9 / 10.7 / 12.1 | -0.2 / -0.4 / -2.8 |
| the same, with summits kept | +0.6 / +1.9 / +2.2 | 12.9 / 14.3 / 16.1 | -0.1 / +2.0 / +1.7 | 12.9 / 14.3 / 14.1 | 0 / 0 / 0 |
| **mean, a half to even, with summits kept (chosen)** | **0.0 / +0.1 / +0.1** | 12.9 / 14.3 / 14.1 | **-0.9 / -0.8 / -0.8** | 12.9 / 10.7 / 14.1 | **0 / 0 / 0** |

The chosen rule on the other kinds of ground, surface mean then outline
mean: cliffs 0.0 / +0.1 / +0.2 and -1.0 / -1.0 / -0.7 (the highest column:
+7.8 / +8.7 / +9.8 and +5.8 / +6.2 / +6.5); gentle slopes 0.0 / +0.1 / 0.0
and +0.1 / +0.2 / +0.2 (+2.9 / +3.3 / +3.6 and +3.5 / +3.7 / +4.0); rolling
hills 0.0 / 0.0 / +0.1 and +0.2 / +0.1 / 0.0 (+1.1 / +1.2 / +1.3 and +0.8 /
+0.9 / +0.8).

What the table says:

- **The systematic part is what a rule decides.** With the highest column a
  slope sank 7.5 to 9 pixels at every edge, its outline 5.5. With the
  chosen rule the surface does not move in the mean (at most 0.2 pixel on
  any kind of ground) and the outline by under a pixel, the other way.
- **The mean alone gives up the peaks**: six blocks at level 3, growing
  back edge by edge. Keeping the summit costs nothing on the surface and
  halves the outline's mean (-1.6 / -1.7 / -2.3 to -0.9 / -0.8 / -0.8).
- **Deciding "ridge or slope" cell by cell does not work.** The rule that
  takes the highest column wherever a cell stands above both neighbours
  along some line keeps every peak and leaves the outline as swollen as
  the max did: on a slope the height field's own roughness makes a cell a
  crest along the contour as often as not.
- **No rule gets the 95th percentile near 2 pixels.** It is 13 to 14 pixels
  at every edge on steep ground whatever the rule (19 to 26 with the
  highest column), 6 to 7 on a gentle slope, and on the rolling hills one
  block, which at 160 blocks is 6.4 pixels by itself. That is the size of
  a cell: a flat top a cell wide stands off a slope by a quarter of its
  width times the slope, and cells are sized in pixels (a level-1 cell at
  160 blocks is 13 pixels wide, a level-3 cell at 512 is 16). **What is
  left is detail arriving all at once, and only a blend between levels
  over distance can spread it**; that is a later block, not this one.
- **How each nests.** The highest column nests exactly and upward: a parent
  is never below a child, so every swap is a drop. The mean nests in the
  mean: a parent is the mean of its children up to rounding, so a swap has
  no direction and about 33 / 39 / 45% of a steep slope's columns stand
  above the coarser cell that covered them. A summit at its highest
  column nested exactly; at the mean of its highest half (the summit
  study, §9.3) it rises a step per swap on the way in.

The four mountain images say the same (`mountain_level_0..3.png`; the
outline is the first land pixel of each image column, against level 0's
from the same eye): the level's outline stands +0.1 / +0.2 / +0.6 pixel
from level 0's in the mean at levels 1 / 2 / 3, where the highest column
stood +1.6 / +4.5 / +8.3 and main -0.9 / -1.2 / -2.2. The mean *distance* is
1.0 / 2.4 / 5.3 pixels (the highest column 1.6 / 4.6 / 10.1, main 1.3 / 2.7 /
5.4): the direction is gone, the size of a cell is not.

**After** (same forty footprints, the surface term, pooled over ten
footprints a kind with the worst single footprint in brackets; §9.2's
"before" beside it):

| steep mountain | L1 | L2 | L3 |
|---|---|---|---|
| stone share of side faces (level 0: 0.1%) | 0.1% (was 6.6) | 0.1% (30.3) | 0.1% (39.8) |
| T3, tops (+Y), /255 | 1.09 (1.50), was 6.9 (7.9) | 0.33 (0.86), was 11.3 (12.6) | 0.68 (2.88), was 14.1 (15.1) |
| T3, worst side direction, /255 | 0.85 (1.14), was 2.0 (4.3) | 0.74 (1.18), was 8.1 (13.9) | 1.08 (2.21), was 10.4 (15.7) |
| mean column, drawn - level 0, blocks | +0.01 | +0.02 | +0.07 |
| summit, blocks | 0 | 0 | 0 (at #432; since the summit study, 0 / -0.8 / -2.2 on steep ground, §9.3) |

| | L1 | L2 | L3 |
|---|---|---|---|
| stone share of sides, cliffs (level 0: 0.7%, 0.6%, 0.6%) | 0.9% (was 11.2) | 0.8% (37.7) | 0.8% (47.1) |
| stone share of sides, gentle slope (level 0: 0.0%) | 0.0% (0.0) | 0.0% (4.0) | 0.0% (9.5) |
| soil share of sides, steep (level 0: 33.5%, 33.2%, 32.2%) | 30.2% | 32.5% | 26.6% |
| T3 tops, gentle slope | 0.31 (0.88), was 2.5 (4.5) | 0.86 (1.66), was 5.3 (8.2) | 1.05 (2.30), was 8.0 (10.7) |
| T3 tops, rolling hills | 0.08 (0.14), was 0.26 (0.4) | 0.40 (0.56), was 1.2 (1.5) | 0.96 (1.36), was 3.6 (4.0) |

(§9.2's "peak" and "skyline" rows are not repeated: a footprint cut out of
a slope has its highest column at its cut edge, with higher ground beyond,
so those rows read -0.6 / -1.4 / -3.6 and -0.5 / -1.5 / -3.6 blocks for a
rule that is at level 0's height wherever there is a summit. The study's
outline, taken on real crests in a wider window, is -0.1 / -0.3 / -0.7
blocks against level 0 on steep ground; the highest column's was +0.8 /
+2.4 / +5.1.)

Riser *area* is 0.96 to 1.00 of level 0's at level 1, 0.86 to 0.93 at level
2 and 0.67 to 0.81 at level 3 on the surface term, which understates it
(§9.2): bumps inside a cell have no riser. It is reported and not bounded
(§9.1 item 3).

**#423 decision 3's 2/255 holds at levels 1 and 2, in every direction of
every single footprint, and not at level 3.** Worst footprint and direction
measured, tops then sides: level 1 1.50 and 1.51; level 2 1.66 and 1.62;
level 3 **2.88** (nine of forty footprints are over 2.0) and **3.18** (two
of 63 ranked directions: 3.18 and 2.21). The harness asserts 2.0 and 2.0,
2.0 and 2.0, 3.0 and 3.25 (`T3_ALLOWED`; level 3's are chosen, just over
the measured worst). The stone share of the sides is within 1.15 points of
level 0's in every footprint (asserted at 1.25), the soil share within 3.7,
0.7 and 5.6 points per kind (asserted at 4.5, 2 and 6.5). What is left at
level 3, in order (`top_shade_by_riser_report` takes the first apart):

1. **A coarse top knows the risers beside it and nothing of the steps
   inside it.** Its shade is cast by its neighbours' heights; level 0's, by
   every step among the 64 columns under it. A footprint is off by
   whichever it has more of.
   **The lever:** shade a coarse top by level 0's openness over its own
   columns -- `ground_faces(..).top_open`, the number the far terrain is to
   use (§9.3) -- carried per column the way the exact tops are. Not built.
2. **A cell is one material over four rows**, and the one that holds the
   top is the surface block over as many of them as are drawn, where level
   0 shows one row of it and soil under it: that is the soil level 3 is
   short of, and a direction with a few dozen faces is a sample of level
   0's mix (the 3.18 is one cliff's +X, 2.45 of it the materials alone).
   **The lever:** let the cut-off row wear the surface block and the rest
   of its cell what the hash picks. Not built.
3. **Level 0's own chunk-edge AO blindness** (§9.3, #430), which is about
   2/255 on steep tops by itself and is in the reference every level is
   held to.

(Pits were item 3 of this list: up to 267 columns of a footprint at level
3. There are none left on the surface term at any level, and
`a_coarse_level_keeps_level_0s_shape` holds it at none.)

**The shape, asserted** (`a_coarse_level_keeps_level_0s_shape`, every
footprint): the mean column stands within 0.1, 0.1 and 0.3 blocks of level
0 at levels 1, 2 and 3 (measured: 0.03, 0.06 and 0.26, the last on cliffs,
where a summit cell at a cliff's edge stands over its foot); where a
footprint's highest column is a summit every level's highest top is level
0's, exactly; the skyline's mean distance is within 1.5, 3.5 and 7.5 blocks
(measured 1.31, 3.26, 7.18, on cliffs); no column is drawn under its own
cell's top. `coarse_caves_nest` holds the cave rule itself: in every
footprint every block a level draws as cave is air at the next finer level,
down to the world's own blocks (475,297 / 145,592 / 100,112 / 29,632 blocks
drawn as cave at levels 0 to 3).

**What level 0 itself changed by.** A level-0 node leaves out a border face
only where every neighbour it could have draws solid there, the level-1
cell beside it included (`border_samples`). What that cell draws changed
twice over: its top is the mean of all four columns with a half to the even
block, where main took the mean rounded down, and it is cave only where all
four of its blocks are, where main asked one of them. In the game the
level-1 cell stands there. In a scene drawn at level 0 alone it shows at the
outer edge of what is drawn, as strips of wall one block tall that are kept
or left out. That is all of what moved in the level-0 scenes among the
golden images (`hearts`, `crosshair`, `hotbar`, `item_icons`, `cave_mouth`
and `mountain_level_0`), 0.3 to 0.45% of their pixels.

**Triangles** (`geometry_census`: meshed triangles per level in the game's
own selection around each gate eye -- radius 64, squash 2 -- deterministic;
every node of the selection, not what a frame draws):

| ground eye (8, 40, 8) | main | the first version (highest column) | **as built** |
|---|---|---|---|
| level 0 | 390,624 | 387,756 | 387,742 |
| level 1 | 270,072 | 265,902 | 259,124 |
| level 2 | 332,042 | 306,506 | 287,264 |
| level 3 | 325,368 | 541,754 (+66.5%) | **448,390 (+37.8%)** |
| all | 1,318,106 | 1,501,918 (+13.9%) | **1,382,520 (+4.9%)** |

| hill eye (8, 300, 8) | main | the first version | **as built** |
|---|---|---|---|
| level 2 | 445,828 | 416,796 | 394,714 |
| level 3 | 330,822 | 547,056 (+65.4%) | **449,276 (+35.8%)** |
| all | 776,650 | 963,852 (+24.1%) | **843,990 (+8.7%)** |

(The flight eye at 3,000 blocks has no voxel node in its selection.)

Level 3's rise is the cut: an exact top means two neighbouring 8-block
cells differ by a block or three almost everywhere, so tops that merged
into terraces of four no longer do and every cell edge has a riser. Flat
ground costs nothing more. It is a sixth less than with the highest column;
part of that is caves, of which fewer are drawn at a distance (7% of level
3's triangles are underground where 15% were).

**Decided by the lead, 2026-11-20: the cut, with a riser's foot row merged
into its shade band.** The requirement it rests on, as the lead put the
owner's: approaching a mountain, it must not shift or change at all, and
that must hold on a weak PC too, where an automatic quality setting chooses
distance and detail. Rounding between levels is therefore not acceptable at
any quality, and what exact heights cost is to be recovered elsewhere
(per-PC quality, engine optimisations), not by rounding. The same
requirement is what the height rule was then chosen by. What the count does
to a frame is the gate benchmark's to say (rows in `BENCHMARKS.md`, taken by
the lead on a quiet machine).

**Generation and build cost** (`generation_cost_report`; microseconds per
node, fastest of three passes, test profile with `cubara-world` optimised
and the mesher and the link flood not, on a shared machine -- good for the
ratio and nothing finer):

| | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| generate, main | 373 | 285 | 675 | 398 |
| generate, as built | 342 | 351 | 1,073 | 1,725 |
| | 0.9x (the same code: this is the measurement's noise) | 1.2x | 1.6x | 4.3x |
| build (generate, mesh, links, trees), main | 1,098 | 2,146 | 5,757 | 3,634 |
| build, as built | 1,028 | 2,409 | 7,266 | 14,964 |
| | 0.9x | 1.1x | 1.3x | 4.1x |

A level-3 node asks 16,900 surface heights where it asked 1,024, counts
level 0's faces for its cells near the surface, and its mesh is swept, its
border sampled and its links flooded on four times the rows. It is worker
time, not frame time. The first version was 4.0 and 3.6 times main; the
summit test and the nested cave rule add 0.3 to generation, and flooding
the links on the drawn lattice most of the rest of the build (2.6 ms of the
15 in this unoptimised profile, where the stored rows took a quarter of
that). **It is a known cost, to be judged against the flight benchmark's
swap lag** (lead, 2026-11-20). **The levers:** cache a node column's height
grid across its vertical stack of nodes -- an accelerator only, by §3 --
which roughly halves generation; and sample a level-3 border per stored
row of four wherever the block is not in the row that holds the
neighbour's top. Neither is built.

**The chunk arena.** Every resident node's mesh lives in one vertex and one
index buffer of fixed capacity (`cubara_render`'s `ChunkArena`), and a node
that does not fit is skipped: a hole. It held 4.0 million vertices, sized
from the scene around the origin when the node tree landed (#111). PR 2
raises it to 8 million vertices and 12 million indices (lead, 2026-11-20;
96 + 48 MB of buffers, from 48 + 24). Two numbers stand behind that, and
they are not the same thing:

- **every node of the game's selection**, seen or not: 4,280,540 vertices
  standing on a mountain (chunk column (336, 213) of the default seed), and
  2,765,040 around the origin;
- **what is resident**: the game meshes and uploads only the nodes its
  visibility search reaches. On the ground at that column that is 2,474,840
  (58%). The most found from any eye tried is **3,462,428**, nine chunks
  above the same mountain, where nearly all of a smaller selection is in
  sight (eleven eyes over three mountains and the origin).

So what is shown is that 4.0 million had no margin left: 3.46 million
resident before any swap is 87% of it, and over it with a quarter more for
the nodes a swap keeps resident twice. **What is not shown is that main
overflowed**: main's whole selection at that column is 4,233,148 vertices,
which counts nodes it never uploads, and what main keeps resident there was
not measured. `streaming::tests::the_arena_holds_the_selection_around_a_mountain`
holds the first number -- above everything found resident -- at the three
steepest of 64 seeded columns, times 1.25 for a swap (chosen, not derived),
under the capacity. A capacity derived from the selection, instead of a
constant held to it, is the lead's follow-up.

### 9.5 What a split node hides: the low flight's hole (PR 2, after its benchmark)

The lead's flight benchmark of PR 2 (M3, 1920 x 1080, 2026-11-21): on the
low flight (`--bench flight --eye 8,120,8 --look 1,-0.15,0.3 --seconds 20`)
main shows no hole and PR 2 showed **9 of 1,200 frames with one**, the same
nine in both runs, about 25 pixels at (786, 833). The flight toward a
mountain showed none on either. The gate wants none.

**What was missing was a node, not a seam.** The pixels look at the top
faces of five blocks, (352..356, 15, 80): the floor of a pit, in the **top
row** of the level-0 node (22, 0, 5). No face of that node was drawn for ten
frames after its level-1 parent (11, 0, 2) had split, because the node had
not been asked for:

- the pit is a cave open to the sky (the ground's own surface there is at
  24). At level 0 its floor is at 15 in the row `z = 80` and at 16 in the
  row behind it;
- the parent draws one cell over each two by two columns, and **a coarse
  cell is cave only where all of its ground is** (§9.4): the cell's row at
  16 holds two blocks of air and two of rock, so it stays whole. The
  parent drew that floor at 16, a block higher, which is in the node
  above. (Main asked one block of the cell, called the row cave, drew the
  floor at 15, and showed no hole here.)
- the visibility search goes through a node that has just split, and is
  not generated yet, on a guess (`visibility::stand_in`). The guess was
  "as its parent drew it", and as the parent drew it no air touched the
  plane between the two nodes: the search did not step down, and the node
  with the floor was not in its answer;
- the streamer lets a stale node go when the replacements *that answer
  sees* have arrived (`can_unload`). The parent went; the floor was sky
  until the sibling's own links had arrived, another search had run (they
  are ten frames apart) and the node had been meshed.

So PR 2's own rules did it, but not through the geometry: every seam was
closed (`border_samples` asks each neighbour what it draws). **The
assumption that broke is that a finer node shows no opening its ancestor
does not.** Caves now only ever open on approach (§9.4), and a top is a
mean with columns below it (§9.3): a finer node is the more open one, by
design.

**How often, measured without a GPU**: the streamer's searches replayed
along that flight's 41 chunk crossings, with every mesh arriving at once
(a throwaway probe, not committed). A node is *left out* when the settled
search at a chunk sees it, the first search there does not, and something
was drawn in its place before:

| | nodes left out | in crossings | nodes generated by the first searches |
|---|---|---|---|
| main (`df1aaf3`) | 45 | 19 of 41 | 12,519 |
| PR 2 as it was | 62 | 21 of 41 | 12,409 |
| a split node is open (**as built**) | **0** | 0 | 15,315 (+23%) |
| open unless the ancestor drew rock there | 16 | 3 | 13,722 (+11%) |
| as the ancestor drew it, and the nodes beside a guessed node seen | 22 | 3 | 14,151 (+14%) |
| every node with a history open, merged ones too | 0 | 0 | 17,767 (+43%) |

Main has the defect too, a little less of it; it happened not to show in
that flight. One node in 62 was in view, at the bottom of a pit, for ten
frames.

**What is built**, two rules, each where its question is answered:

1. **A node that has split is searched as open** until its own links are
   known (`visibility::stand_in`). Open joins everything its links can, so
   the first search after a crossing asks for everything that will be seen
   through it. The second and fourth rows of the table are the cheaper
   guesses that do not hold; the last is the dearer one that adds nothing
   measured (a merged node was still searched as its children drew it;
   since §9.9, where a coarser node's dent is wider than its children's
   pit, it is searched as open too).
2. **A stale node also waits for the bare nodes beside it**
   (`cubara-app`'s `can_unload`). The node seen through a new opening may
   be one nothing drew before -- under the pit's floor, in another parent.
   It is in the same answer, but it replaces nothing, so nothing waited
   for it. A stale node now leaves when its replacements have arrived
   **and** no node beside it (a shared face) is still loading with nothing
   drawn in its place. In the low flight's twenty seconds that holds 101
   nodes back, each for one frame (counted with a log line, since removed);
   without it each of those left while a bare node beside it was still
   being meshed. None of them showed as a hole in two flights flown with
   the first rule alone: the second makes certain what was likely.

**What it costs.** The first searches ask for 23% more nodes over that
flight (level 0 +21%, level 1 +30%, level 2 +32%, level 3 +4%): nodes
behind a split node that turn out to be hidden. Each is generated once,
found hidden by the search that knows the split node's links, and dropped;
its links are kept, so it is not generated again. That is worker time, to
be judged with §9.4's generation cost against the flight benchmark's swap
lag (lead; rows in `BENCHMARKS.md`). One effect on the benchmark's own
terms: a hidden node that is dropped counts as a swap in `POPS:`, so that
line counts more swaps and more changed pixels per swap (pixels that moved
because the camera did); its silhouette term is unchanged. `--bench gate`
is not touched: its scene knows every node's links, so no guess is made.
**The lever, not built:** a node seen only through a guess needs its links
to be judged, not its mesh. Generating it for its links first, and meshing
it only if the search that knows still sees it, would leave the generation
and drop the meshing and the upload of the hidden ones.

**What is not closed.** A node behind a node *nothing is known about* --
neither it nor an ancestor nor a child was ever generated -- is still seen
only when that node's links arrive: the rule that lets the streamer
generate outward along what can be seen instead of everything. For that to
be a hole, a line of sight has to pass clean through a node that opened in
this very crossing; none was found in the replay (no node below level 3
arrives late).

**Tests** (`crates/world/src/visibility.rs`, `crates/app/src/streaming.rs`;
no GPU): `the_node_under_a_pits_floor_is_asked_for_when_its_parent_splits`
is that crossing in the default world;
`the_first_search_after_a_crossing_leaves_out_nothing_that_was_drawn`
replays twelve crossings from each of four seeded places, with nodes
splitting and merging at every level;
`a_node_split_or_merged_since_the_last_search_hides_nothing_behind_it` lost
its exception for nodes under the ground's surface, which was this hole;
`a_node_stays_until_the_bare_node_beside_it_has_arrived` and its two
companions hold the second rule, and
`a_held_node_is_not_released_while_the_pool_meshes_a_bare_node_beside_it`
and `a_search_answer_holds_a_node_for_the_bare_node_it_sees_beside_it` hold
it to the two places a node is let go (`release_held`, `leave_or_hold`: the
streamer's bookkeeping, taken apart from the GPU for that).

### 9.6 The colour morph: a ring edge swaps no colour either

The distance morph (#380, #436) slides a node's heights to its parent's over
a band before its ring edge, so that the swap moves nothing. What the swap
still changed was **colour and shade**. The transition study
(`crates/app/src/transition_study.rs`, the toward-peak flight, the worst
single frame of each edge, pixels over the goldens' tolerance) put it at
16,398, 12,933 and 21,080 pixels at the 160, 288 and 512 m edges on main
(`b2fc847`), and took it apart with switches in the shader, each run on the
same flight (study binary of this branch's first commit, since removed):

| what was held equal at the swap | 160 m | 288 m | 512 m |
|---|---|---|---|
| nothing (main) | 16,398 | 12,933 | 21,080 |
| a riser's texture laid by world height (no stretch) | 11,075 | 10,218 | 16,201 |
| ... and every wall one texture | 8,286 | 6,987 | 12,346 |
| ... and walls unshaded instead | 10,052 | 9,869 | 15,649 |
| ... and tops unshaded instead | 8,788 | 7,743 | 14,049 |
| ... walls one texture and unshaded | 4,994 | 4,868 | 8,163 |
| ... one wall texture, no shade at all (**the floor**) | 3,027 | 3,657 | 6,594 |

Four things, then, each worth a quarter to a third: a morphed riser's
texture **stretched** with it (one tile over the three blocks a riser grew
to); its **material** was its own where the parent's riser wears another
(the hash of §9.4 picks per cell); and the **shade** of tops and of walls
was its own -- most visibly the band a riser casts on the top at its foot,
which a coarse cell draws as one ramp over the cell. What is left at the
floor is ground that does not morph (beside a cave, a pit or a tree, across
a node's floor) and, at 512 m, the trees' switch to boxes (#433).

**The rule: at the end of its band a node draws its parent's colours as
well as its parent's shape.** Three parts:

- **A riser's rows move with its top, one for one** (`Morph::target`,
  `cubara_voxel`). They used to move on the straight line from the rows as
  built to the parent's, so a riser that grew was stretched. Moved whole,
  every row lands on one of the parent's rows, tile on tile; a row that
  would pass below the new foot stops there and closes up, and where the
  riser grows the room left at its foot is laid as a latent riser, flat as
  built, one piece per run of parent rows, its tile one per block at full
  morph. The shape is the same at every instant -- only where rows lie
  inside the riser's plane differs -- and the closure tests of #436 hold it.
- **Every face of a morphing column carries what its parent draws where the
  morph takes it**: the parent face's texture layer and its openness there
  (word 3 of the vertex, `Vertex::with_parent`; `mesh.wgsl` lights a vertex
  by `mix(ao_floor, 1, openness)`). It is **read off the parent's own
  mesh** (`cubara_voxel::ParentFaces`, built by `cubara_world::mesh`'s
  `parent_faces` from the parent node generated and meshed by the game's
  own generator and mesher): which quad covers the face's middle at full
  morph, and that quad's openness at each corner, interpolated over the
  triangle that holds the face, as the rasterizer does. Worked out again
  from the parent's cells it would be a second copy of the shade rules
  (Rule 5), and a wrong one: a coarse wall's shade band is one ramp over
  the quad the mesher *merges* (§9.4), a property of no single cell. A face
  that moves nowhere still carries it (its column morphs; its colour does),
  and cells merge only where they show the same of their parent -- the same
  quad, or two quads equally open all over.
- **`mesh.wgsl` blends by the morph factor**: the light toward the parent's
  openness over the whole band, and, where the parent's layer is another,
  the colour toward that texture **over the band's second half**, sampled
  with the gradients of the fragment's own (so at the same mip) under a
  branch most fragments skip. The second half because a second sample is
  the one real cost here and the first half is the near ground, most of
  the frame: over the whole band the gate's ground eye lost 6.4% against
  4.1% (below).
- **A parent is made once for its children** (`cubara_world::mesh`'s
  `ParentCache`, shared by the mesh pool's workers): its eight children
  are requested together, and each needs the same parent generated and
  meshed. An accelerator only (§3): keyed by the snapshot and the node,
  and a node built with it is built exactly as without
  (`a_parent_is_made_once_for_its_children_and_changes_nothing`).

**The fourth vertex word was measured before it was built.** No spare bits
were left in three words (#436 took the last); a padded 16-byte vertex
against the 12-byte one on the M3, `--bench gate` twice each, alternating:
ground 1,125 and 1,128 FPS against 1,130 and 1,125, hill 1,082 and 1,079
against 1,089 and 1,082 -- nothing to measure. The arena's 8 million
vertices take 128 MB instead of 96.

**A coarse top wears what level 0 shows from above one of its columns**
(`WorldGen::top_block`, `from_above`), picked by a hash of the cell as its
riser rows are: the surface block over a whole column, the floor of the pit
over one a cave has opened. A coarse level showed a pit only where all of
a cell is one (§9.4; until §9.9 sank the cell), so on the default seed's mountains the pits that open
6.1% of level 0's columns were 3.4% of level 2's ground and 1.3% of level
3's: the grey patches a mountain lost from afar in the zoom pairs. Partly
pit cells now wear the floor in the share their columns are pit
(`a_coarse_top_shows_pit_floors_in_the_share_level_0_does`: 58 and 156
such tops against 61.6 and 150.2 expected at widths 4 and 8). `from_above`
is the one rule for a column's colour from above; the far terrain reads the
same floor through `top_layer` (#422, §9.8). The surface term of the
consistency harness does not move: a cell over a pit is in the cave term
(§9.2).

**The vertex stage had no pipeline constants.** `mesh.wgsl` has resolved AO
in the vertex stage since R1c, with `ao_floor`, but the mesh pipeline passed
its constants to the fragment stage only, so the vertex stage lit with the
shader's default floor whatever `Lighting` said; it matched only because
the two defaults agree. The colour morph lights by `ao_floor` there too; the
vertex stage now gets the constants (`the_ao_floor_reaches_the_vertex_stage`).

**Measured** (the transition study, the same flight and harness, main
`b2fc847` against this rule; the morph's worst single frame per edge,
pixels over the tolerance, and its luminance in pixel-equivalents):

| edge | main | the colour morph | luminance, main -> colour |
|---|---|---|---|
| 0<->1 (160 m) | 16,398 | **4,959** (-70%) | 2,368 -> 786 |
| 1<->2 (288 m) | 12,933 | **4,735** (-63%) | 2,020 -> 1,003 |
| 2<->3 (512 m) | 21,080 | **9,217** (-56%) | 3,982 -> 1,597 |

The floor of the table above is 3,027, 3,657 and 6,594: what is left is
what does not morph. At 512 m a little of it is new -- a cell partly over a
pit wears the floor now, and its children beside the pit do not morph, so
they swap from the floor's colour rather than the surface's (9,217 against
8,641 with the colour morph alone). The holes the morph opens are those of
#436, unchanged (210 of 450 frames checked, at most 34 pixels; the same
count as main's). `a_fully_morphed_level_draws_its_parents_colours` (GPU)
draws a mountain at levels 0, 1 and 2 fully morphed against the level above
unmorphed: 20,558, 22,204 and 21,383 pixels of 921,600 over the tolerance
with the parent's colours, 52,043, 50,104 and 47,546 without -- what is left
is ground beside caves and pits and along the node floors that cut a slope
(#436's limit), which neither colour nor shape can morph.

**What it costs** (M3, not a quiet machine; alternating with main, two
rounds each). `--bench gate` stays MET at Medium: ground 1,127 and 1,130 FPS
on main against 1,078 and 1,079 (-4.4%), hill 1,082 and 1,083 against 1,059
and 1,059 (-2.1%). About half of it is the second texture sample (light
only, no texture blend: 1,084 and 1,060 in one round each), the rest the
quads the parent's look cuts (+2.6% of the ground eye's vertices, +2.2% of
the hill's) and the vertex stage's work. The flights keep HOLES at 0 (low:
0 of 1,195 and 0 of 1,193; toward the mountain: 0 of 1,796 and 0 of 1,799)
and their pops halve (p50 3,745 and 4,332 px on main against 1,665 and 1,512
on the low flight). A child's parent is generated and meshed once for its
siblings; the low flight's swap lag p50 / p90 / p99 is 100 / 151 / 251 and
101 / 151 / 235 ms against main's 84 / 134 / 167 and 84 / 134 / 184 (119 /
237 / 339 before the cache), the flight toward the mountain's 84 / 100 / 150
against 84 / 100 / 134.

#### 9.6.1 The near field stands exactly on the block grid (play test, 2026-11-25)

The distance morph's level-0 band started **at the eye** (0 to 128 blocks,
§9.5's study), so every level-0 vertex at metric m had moved m/128 of the way
to its parent's height: 4% at 5 blocks, 16% at 20, 31% at 40. Near the
player a block is tens to hundreds of pixels tall, and a few percent of one
read as a misaligned slab: on the owner's play test of 2026-11-25, thin
grass slabs a fraction of a block thick on the terraces and thin brown
slivers at the foot of the nearest walls. Reproduced from the ground
(`--screenshot`, an eye on the terraces at the steep mountain of #432, 1544 x
1146): the slivers and slabs are in the render of main, and gone from the
one with the band starting at `NEAR_EXACT`.

**Level 0's band starts no closer than `morph::NEAR_EXACT` (96 blocks,
chosen, not derived), and ends where it did (128: 32 before the ring's
edge).** Measured with the transition study (the toward-peak flight,
`CUBARA_STUDY_VARIANTS=morph`, the 160 m edge; `hole`: pixels of nothing the
morph opens, over 450 frames checked):

| level 0's band | worst single frame (px over tol.) | its luminance | frames that change | total over the approach | holes (frames with any / most px) | worst off-grid offset in the band |
|---|---|---|---|---|---|---|
| 0 to 128 (before) | 4,926 | 783 | 190 | 22,988 | 251 / 32 | 18.6 px |
| 64 to 128 | 4,929 | 784 | 190 | 25,373 | 251 / 32 | 14.6 px |
| 80 to 128 | 4,929 | 784 | 191 | 26,525 | 251 / 32 | 13.5 px |
| **96 to 128** | **4,933** | **785** | **192** | **28,140** | **251 / 32** | **12.5 px** |

(The off-grid offset is the largest `delta * min(f, 1 - f) * 935 / distance`
over the real level-0 vertices around the mountain's terraces, in pixels at
1080 rows and 60 degrees; `delta` reaches 3 blocks inside the near field and
4 beyond it.) The worst frame and the holes do not move (+0.1%); only the
sum over the approach grows (+22%), because the same change is spread over
a narrower band. 96 is the largest candidate and does not make the worst
frame worse, so it is the one. Levels 1 and 2 keep their bands. A ring too
short for it (a weak PC's) keeps a band: the start is at most three quarters
of the way to the band's end.

Pinned by `morph::tests::level_0_stands_exactly_on_the_grid_near_the_eye`:
real level-0 nodes around eyes on the terraces and above them; every vertex
with a morph target within `NEAR_EXACT` of the eye, by the metric (the
vertical part included), has factor 0.

### 9.7 One tree at every distance (#433, items 2 to 4)

The ground was made the same at every level by §9.3 to §9.6; the trees on
it were not. Level 0 draws a tree as blocks in its chunk's mesh, levels 1
and 2 as an instance of one oak model, level 3 as two boxes (#332). An
audit of the render path (2026-11-20) found what the owner sees change at
a ring edge while flying toward a wood: every instance stood five blocks
tall, whatever the tree's own trunk (four to six); instances were lit
flat, without the shade level 0 bakes under the canopy and at the trunk's
foot; and at 512 m a square box replaced the canopy's rounded outline.
Trees past the voxel rings (#433 item 1) wait on the far terrain (#422)
and are not part of this.

**The measurement** (`crates/render/tests/lod_consistency/trees.rs`, no
GPU): 48 trees the default world grows, drawn from a printed seed, sixteen
each on level ground (the 25 columns under the canopy at one height),
gentle (one or two blocks between them) and steep (three or more). Each is
drawn as every representation draws it -- level 0 from the game's own
chunk meshes, an instance from the game's own model where the game places
it -- and compared tree by tree: the tree's top and foot in blocks; its
outline from five views (three level, two from above) rasterised at an
eighth of a block, as the area one drawing covers and the other does not;
and its brightness, as the mean colour each view sees of it and per face
direction (T3 per tree), /255 in linear light. Each ring edge is compared
as it swaps: the near level at the end of its band against the far level.

**Before** (main at `c5b37e6`; mean over the trees / the worst tree):

| | top, blocks | outline, worst view, square blocks | seen colour, /255 | T3, worst direction |
|---|---|---|---|---|
| 160 m: level 1 against level 0 | 0.65 / 2 (27 of 48 move) | 9.8 / 25.8 | 4.0 / 6.7 | 3.8 / 6.3 |
| 288 m: level 2 against level 1 at full morph | 0.40 / 2 (16 move) | 4.6 / 23.0 | 0.0 | 0.0 |
| 512 m: level 3 against level 2 at full morph | 0.85 / 6 (25 move) | 18.5 / 61.2 | 1.5 / 1.6 | 1.5 / 1.5 |

On level ground the 160 m edge moved 9 of 16 canopies a block (the height)
and brightened every tree by 4/255 (the shade); the 512 m edge changed
every outline by 14 square blocks (the box's corners). The 288 m edge
changed only where the ground did: the distance morph moved the ground
under a tree to its parent's height and left the tree where it was, and
the swap put it back on its ground.

**The rule: one tree shape, meshed once per height by the mesher level 0
uses, riding the ground it stands on.** (`cubara_render::trees`,
`cubara_world::mesh::node_trees`.)

- **Its own height** (item 2). `TreeShapes` reads the oak's trunk heights
  from `oak.ron`, and there is one model of each: the oak `h` blocks tall,
  `tree_part_at` meshed by the voxel mesher (`build_oak_mesh`). An
  instance carries its tree's height (`TreePlacement`), from the site the
  world places (`WorldGen::node_trees`, which already had it), and the
  arena files it under that height's model: one draw per model and
  height with instances. No second generator (Rule 5): the shape is
  `tree_part_at`, the height and the canopy are the site's.
- **Its own shade** (item 3). The mesher bakes AO by the corner rule a
  level-0 tree gets (`face_ao_of`), and the model keeps it, per vertex, as
  `Vertex::openness`; `trees.wgsl` lights it by `mix(ao_floor, 1, open)`
  as `mesh.wgsl` lights a block. The tree is meshed standing on a layer of
  the block it grows on, whose faces are then left out, so the foot of its
  trunk is shaded as level 0 shades it on level ground.
- **Its own outline** (item 4). Level 3 draws the oak's **silhouette**
  (`build_silhouette_mesh`): the oak's surface merged into as few quads as
  it takes, the same faces and so the same outline from every side, each
  face direction and texture one flat colour -- the texture's mean, shaded
  by the area mean of the oak's openness there. `mix` is linear in the
  openness, so per direction it is the oak's own colour. Two boxes the size
  of the tree had a square canopy where the tree's is rounded.
- **Riding the ground** (the 288 m and 512 m jumps). A coarse node's tree
  takes the morph of the column its trunk stands on, read off the node's
  own `MorphColumns` -- the ones its mesh is built with, worked out once
  for both (`NodeTree::morph`) -- and `trees.wgsl` moves it by the factor
  `mesh.wgsl` moves the ground by. That factor is now one WGSL function
  both shaders include (`morph.wgsl`); `mesh.wgsl` calls it where it wrote
  it out (two pixels of the goldens moved by one level). The instance packs
  its node's slot, the morph's level and its lift into one word laid out as
  a vertex's word 2, so it stays 16 bytes.

**After** (this rule; the same 48 trees):

| | top, blocks | outline, worst view | seen colour | T3, worst direction |
|---|---|---|---|---|
| 160 m: level 1 against level 0 | 0.25 / 1 (12 move) | 3.3 / 23.0 | 0.6 / 4.2 | 1.2 / 4.2 |
| ... level ground, in one chunk (10 trees) | 0 | 0.0 | 0.0 | 0.0 |
| 288 m: level 2 against level 1 at full morph | 0.02 / 1 (1 moves) | 0.2 / 11.5 | 0.0 | 0.0 |
| 512 m: level 3 against level 2 at full morph | 0 | 0.0 | 0.8 / 0.9 | 0.0 |

**On screen** (`golden.rs`'s `a_forest_swaps_nothing_at_its_ring_edges`,
the M3): the densest wood near spawn (15 trees in one level-3 node), each
edge's two frames, the trees' pixels found by drawing each again without
trees. The share of the trees' pixels drawn by one side only, and their
mean colour change on the image's 0-255 scale: 160 m 18.6% and 1.2 on main,
9.1% and 0.8 now; 288 m 10.1% and 1.8, now 2 pixels and 0.0; 512 m 66.1%
and 7.9, now 0.3% and 1.7 (the flat colours against the texture, at a view
where each tree is tens of pixels; at 512 m it is a few, and the texture's
mip is its mean). `forest_level_0..3.png` are the four images.

**What is left, and why it is not here:**

- **The ground under a tree at 160 m.** Level 0 stands a tree on its own
  column, level 1 on its cell's mean, and nothing that stands on a column
  morphs at level 0 (§9.6: "ground that does not morph (beside ... a
  tree)"). On sloping ground a tree moves a block with its ground at the
  swap: 12 of 48 trees, never more than one block, and the harness holds
  the foot to exactly the ground's step (`check`). Moving level-0 trees
  with their column is a change to the morph's rule, its own issue.
- **Level 0's chunk floors.** The level-0 mesher bakes a chunk's AO
  without looking into the next chunk (§9.3), so a tree whose blocks cross
  a chunk's floor (15 of 48) is up to 4.2/255 more open there at level 0
  than the corner rule says; the instance, meshed whole, follows the rule.
  Reported and not held, as the ground harness reports its caves; fixing
  it is the level-0 mesher's.
- **A slope's own blocks.** On a slope level 0 leaves out the faces of a
  tree that touch the ground beside it, and shades its trunk's foot by the
  uphill column: up to 3.4/255 per direction, and outlines of 1 to 11
  square blocks that the ground itself hides on screen. Held only on level
  ground; the foot and the height are held everywhere, exactly.

**What it costs.** Per tree, the oak models are 170, 178 and 186
triangles (four, five and six blocks tall; 340 to 372 vertices) and the
silhouette 110 for every height (220 vertices), which replaces the two boxes
(24 triangles, 48 vertices). That is the one increase: at the gate's ground
eye (M6, `lod_probe`) 247 level-3 trees are in view on average, so 27,200
triangles (54,300 vertices) where the boxes were 5,900 (11,900), and at the
hill eye 264 of them. Levels 1 and 2 draw 20 and 46 trees in view there,
about 12,000 triangles, as before (the same oak, one model per height now:
three draws where there was one). Six draws where there were two. CPU: one
`MorphColumns` lookup per tree when a node is built (the columns were
already worked out for the mesh); the instance stays 16 bytes. No FPS
number is given here: the gate and flight rows are the lead's, on a quiet
machine on AC power (`--bench gate` and `--bench flight`); the counts above
are what they will show, and the direction to expect is a small cost at
level 3.

**Tests.** `a_tree_is_the_same_tree_at_every_distance` (the check over the
48 trees, bounds in `trees.rs`'s `bounds`, each chosen) and
`the_tree_harness_sees_a_wrong_tree` (one height for all, no shade, a
block aside, a block up, the two boxes: each makes the check fail);
`a_tree_rides_the_ground_under_it` (the tree's morph is the lift of the
top face under its trunk in the game's own node mesh, at levels 1 and 2);
`trees.rs`' unit tests (every model is its tree to the block; the
silhouette has the oak's area and shade per direction in fewer
triangles; the oak is shaded at its foot; the packed word round-trips);
the forest goldens and `a_forest_swaps_nothing_at_its_ring_edges`.

### 9.8 The far terrain on the rule (#422)

The far terrain is drawn as blocks: every quad a flat top at
`quad_top`, a wall wherever two neighbouring tops differ, exactly as tall as
the step, coloured from `ground_faces` and `top_layer`. No second height
function exists and no second colour rule (Rule 5);
`far_study::the_chosen_far_rule_is_the_games` holds the study's copy of the
height rule to `WorldGen::cell_top_sampled`.

**The far height rule is the voxels' rule, whatever that is** (`TopRule`:
the mean, and a summit at the mean of its highest half since #437). Studied
by the swap: coarser minus finer top, per level-0 column, in pixels at the
distance each quad width is first drawn at with quads of 16 pixels at 1080
rows (8 to 16 at 1,032 blocks, then 2,064, 4,128, 8,256; the pixels scale
with `FAR_QUAD_SCALE`, below), on this seed's 40 footprints of four kinds of
terrain (`far_height_rule_study_report`). Pooled over the kinds, per edge:

| rule | surface mean, px | outline mean, px | outline p95, px | peak against level 0, steep ground, blocks (w 8 / 16 / 32 / 64) |
|---|---|---|---|---|
| **TopRule (built; #437's summit)** | +0.1 / +0.1 / +0.2 / +0.2 | -1.9 / -1.2 / -1.4 / -1.2 | 10 / 10.5 / 8.5 / 6.6 | -2.5 / -5.5 / -16 / -40 |
| a summit at its highest sample (#432; built until #437) | +0.1 / +0.2 / +0.4 / +0.4 | -1.1 / -0.6 / -0.6 / -1.0 | 11 / 12.5 / 11.7 / 8.6 | -2.5 / +0.0 / -2.5 / -29 |
| plain mean | -0.0 / -0.0 / +0.0 / +0.0 | -3.0 / -2.2 / -2.0 / -1.4 | 11 / 9.5 / 7.2 / 5.5 | -2.5 / -16.5 / -33 / -48.5 |
| highest, summit among 24 cells | +0.0 / +0.2 / +0.2 / +0.2 | -1.7 / -0.8 / -1.2 / -1.4 | 11 / 11.5 / 11.7 / 9.9 | -2.5 / +0.0 / -2.5 / -29 |
| highest, lift capped at w/4 | +0.0 / +0.1 / +0.1 / +0.3 | -2.3 / -1.6 / -1.5 / -0.9 | 11 / 9.5 / 7.5 / 6.1 | -2.5 / -16.5 / -25 / -40.5 |
| highest, lift capped at w/8 | +0.0 / +0.1 / +0.1 / +0.1 | -2.6 / -1.9 / -1.8 / -1.2 | 11 / 9.5 / 7.2 / 5.5 | -2.5 / -16.5 / -29 / -44.5 |
| TopRule over 8 x 8 samples | +0.1 / +0.1 / +0.2 / +0.2 | -2.0 / -1.2 / -1.3 / -1.2 | 10 / 10 / 8.2 / 5.9 | -2.5 / -6.5 / -19 / -33 |

(Up to 8 blocks every row is the voxels' rule, so the first column of the
peaks is one number.) Every rule keeps the systematic part of the swap under
a pixel; the 95th percentile of the surface's change is 5 to 10 pixels for
all of them and is the slope's own steps, not the rule. The study first
chose the voxels' rule unchanged when that stood a summit at its highest
column: best outline, peaks kept. #437 then lowered the voxels' summit to
the mean of its highest half -- the highest column drew towers the cell's
width on every crest (§9.3) -- and the far terrain follows, because a far
quad of 8 *is* level 3's cell and a second rule above 8 would be a seam at
16. What that costs the far terrain, from the table: a far outline rises
1.2 to 1.9 pixels a swap on the way in where it rose 0.6 to 1.1, and a
sharp peak stands 5.5, 16 and 40 blocks low at quads of 16, 32 and 64
(2 to 5 pixels at the distances those are drawn), growing back a step per
swap. What it buys: the far crests' towers. Around this seed's highest peak
(1,792 blocks square, `summit_lift_report`) the quads standing their own
width or more above all four neighbours were 53, 24 and 9 at quads of 8, 16
and 32 and are 12, 8 and 5 (`a_far_crest_grows_few_towers`); the outline's
95th percentile falls at every edge. The plain mean and the caps still lose
more of a summit than the rule does. The thresholds in the study (region
256, crest margin 32, every 8th line of sight) are chosen, not derived.

**The join.** `far::FINEST_LEVEL` (the coarsest voxel level, 3: cells of 8)
is the finest far patch level at any screen height and quality: on a 1664-row
display the pixel floor let quads of 4 reach out to 1,589 blocks, a fine band
between two coarse ones. `the_far_terrain_is_never_finer_than_the_join`.

**What a top wears.** The voxels' rule for it (`top_block`, §9.6), read
through `WorldGen::top_layer`: what level 0 shows from above one of the
quad's columns, the column picked by the cell's hash (`top_pick`, the one
pick for both) -- the surface block, or the floor of a pit a cave has opened
(`pit_floor`, the one place a floor is found), soil or stone by its depth.
A far quad of 8 visits the 64 columns the level-3 cell does, in the same
order, with the same hash: **it wears what that cell wears, cell for cell**
(`a_far_top_wears_what_the_voxel_cell_wears`, against the node the game
generates), so the hand-over to level 3 changes no top's colour. Wider
quads pick among their 4 x 4 sample columns, as their tops are sampled:
1,010, 261 and 66 quads of 8, 16 and 32 show a floor where the ground's
6.17% of pit columns says 1,010.5, 252.6 and 63.2
(`a_far_top_shows_pit_floors_in_the_share_level_0_does`). Before this the
far terrain drew every top as the surface block, and a mountain's grey
patches appeared at the join. A quad is one pick, not a blend of its
samples: a quad of 32 that shows a floor is grey over all 32 blocks where
level 3 will show four or five grey cells of 8, so the share is right at
every width and the grain is the quad's. A wall is still the blend of its
footprint's risers (below). An ore is stone to every far face.

**What is guaranteed** (every claim is a test; the numbers are this seed's
worst footprint, `far_rows_report`):

| | far quads of 8 (= level 3's cells) | far quads of 16 |
|---|---|---|
| top = level 0's ground by the rule | exactly `cell_top + 1` | `cell_top_sampled(.., 4) + 1` |
| a summit | level 3's bounds: never above level 0's, at most 6 blocks under on a footprint and 2.5 in the mean (measured 2.13 over 15; `the_far_terrain_keeps_level_0s_shape`) | not guaranteed: sampled, 3 to 37 blocks under on mountain footprints |
| mean column against level 0 | within 0.3 block | within 2.5 (measured -1.3 to +1.9) |
| skyline mean distance | within 7.5 blocks | within 18 (measured 17.3, cliffs) |
| colour T3, tops / sides | 2.4 / 1.1 of 255 | 2.6 / 2.4 (bounds 3.0 / 3.25, level 3's) |
| stone share of the walls, per footprint | within 0.3 points | within 1.1 (bound 1.25) |
| soil share, per kind | within 2.1 points | within 3.8 (bound 5) |
| what a top wears | the level-3 cell's, cell for cell | a floor in the share the ground is pit, within a fifth |

**Approximated, and how far.** A wall's colour is one mix per quad and
direction from 4 x 4 sample columns (2 x 2 above 16 blocks), by depth below
each column's own surface (#423 decision 2), so a tall far wall is not banded
as the blocks behind it are. At quads of 32 the stone and soil share of the
walls are up to 5 points off on mountain footprints, and the quads of 64 have
too few walls in a footprint to measure; both are indications, not
guarantees. A quad of 16 can leave a direction level 0 has without a wall of
its own (one cliff footprint of forty). Far quads of 2 and 4 exist only in
tests: the join clamp never draws them. The consistency harness's surface
term does not see the tops over pits (a quad over a carved column is in the
cave term, §9.2); the two tests above are what hold them.

**What the far terrain costs, and why** (M3, 1920x1080, Medium, the gate's
eyes; single runs on a machine that was not quiet, so the differences count
and the absolute numbers do not -- the formal rows are `BENCHMARKS.md`'s).
The blocks cost 0.99 ms a frame at the ground eye and 1.15 ms at the hill
eye, where the smooth surface they replace cost about 0.29 ms. Found by
switching parts of the pass off, since this backend's GPU timestamps are
invalid:

| the far pass, changed | ground eye, ms | hill eye, ms |
|---|--:|--:|
| as built | 0.99 | 1.15 |
| every vertex sent off screen (full vertex shader, nothing rasterised) | 0.82 | |
| the vertex shader returning at once (no buffer read, no light, no fog) | 0.84 | 0.86 |
| the same, one vertex a face (the index buffer pointing all six at it) | 0.25 at the flight eye, of 0.55 | |
| half the pixels (960x540, the same patches) | unchanged at the flight eye | |
| a shared-corner mesh of the same quads, trivial shader (below) | 0.51 | 0.52 |

So it is **the number of vertices the GPU is asked to shade, at about 0.4 ns
each whatever the shader does** -- not the storage-buffer fetch, not the
light or the fog, not overdraw or fill (0.15 to 0.29 ms is all the
rasteriser's), not the draws (five to seven). A face is four vertices of its
own, a quad has about two faces that can face the eye, and a run is padded
to its class: about 9 vertices a quad, 1.96 million a frame, where the
smooth surface shared one vertex among four quads. Vertices are reused only within
some 64 of the index stream (measured: the second triangle of each face
pointed 16 faces back costs next to nothing, 32 faces back half again).

**What was done about it: one number, `far_streaming::FAR_QUAD_SCALE`.**
The owner left the size of a far quad to the engine (2026-11-21: "as long
as it is good"). The constant multiplies every pixel size a quality names
-- its quads, its error bound and its floor -- so Medium's quads of 16
pixels are quads of 48 at 3; `--far-quad-scale F` overrides it for the
window, `--bench`, a flight and `--screenshot`, which is how the sizes below
were measured. It is the only lever of the three found that both holds the
gate and is watertight:

| scale | Medium, ground / hill FPS | High, ground / hill FPS | the gate | patches selected / in view (Medium, ground eye) | far triangles | far vertices issued | holes, low flight / toward the mountain |
|---|--:|--:|---|--:|--:|--:|---|
| x1 (the quality's own sizes; as first built) | 636 / 577 | | not met at any quality | | | 1.96 M | |
| x2.5 | 1,080 / 1,035 | 988 / 944 | met at Medium | 163 / 56 | 244,604 | 591,064 | 0 of 1,195 / 0 of 1,798 |
| **x3 (the default)** | 1,152 / 1,124 | 1,028 / 983 | met at Medium | 127 / 44 | 192,333 | 467,201 | 0 of 1,194 / 0 of 1,798 |
| x3.5 | not run: the gate stops at High | 1,059 / 1,013 | met at High | 181 / 61 at High | 263,342 at High | 637,284 at High | 0 of 1,195 / 0 of 1,799 |

(M3 on battery, the machine in use and other sessions running: **not
quiet**, single runs, so the patch, triangle and vertex counts are exact and
the frame rates are not `BENCHMARKS.md` rows. The flights fly High. At 2.5
the hill eye held the gate by 3.5% there.)

**The quiet bench chose 3** (`BENCHMARKS.md` footnote 96: AC power, no
agents, five quiet minutes, two runs of each, alternating; Medium, ground /
hill FPS). Main: 1,052 / 1,074 and 1,049 / 1,055. At 2.5: 1,070 / 1,081 and
1,036 / 1,034 -- it holds, by 3.5% at the hill eye, and stands below main
there. At 3: 1,153 / 1,161 and 1,121 / 1,130, 7% over main at the hill eye.
Both fly with no hole and with 1 or 2 stale frames where main has 16.

A shared-vertex mesh (`feat/422-far-mesh-cost`, not landed) issued fewer
vertices at 1.5 than this mesh at 2.5 (488,357 against 595,612 at the hill
eye, Medium) but twice the indices (1.88 million against 0.89), and measured
860 / 860 FPS at the hill eye against 1,036 / 1,034 in the same quiet window:
the far pass's cost follows the triangles it draws, not the vertices it
shades. A finer far terrain needs fewer triangles.

- **Shared corners** (four vertices a quad in one fixed index list, each
  face's colour carried flat from a corner of its own) were built on a
  branch after the prototype above (`wip/435-shared-corners`): 1.96 to 0.93
  million vertices a frame, the gate's ground / hill eyes from 636 / 577 to
  739 / 691 FPS at x1. Not landed: risers of different heights that share
  a column's corners leave T-junctions, and the low flight showed holes in
  22 to 27 of 1,200 frames where the mesh here shows none. Four vertices a
  quad is the least blocks can have, and it still needs quads twice as wide
  to hold the gate.
- **Not drawing what is hidden** helps one eye only: from the ground eye 11
  to 21% of the patches the frustum passes leave a pixel on screen (two
  directions, at High), from the hill eye 83 to 92% -- and the hill eye is
  the one the gate turns on.

What a wider quad changes besides the count: the quads of 8 that equal
level 3's cells are drawn only where the error bound splits that far (a
quad of 8 is 8 pixels wide at the join at 1080 rows; at 2.5 the distance
rule alone draws quads of 32 there), so the hand-over to level 3 is from
quads two levels coarser over smooth ground. The exactness of a quad of 8
is a property of the rule and stands; how much ground is drawn with it is
what the scale trades.

**The plan kept too little at a wider quad.** `far_streaming::to_keep` holds
on to every ancestor of a wanted patch, since an ancestor's error is what
split it -- but only up to the coarsest *wanted* level. A root too far to be
split by distance and split for its roughness alone is no wanted patch's
level once all the roots are split; it was dropped, selected again unsplit,
generated, split and dropped, without end. Quads of 16 pixels split every
root of the tests' radius by distance, so it took the wider quads to show
it; the game's own roots (262 km) are too large to be rough at their
distance at any scale tried. Ancestors are now kept up to the level the
selection walks from (`far::root_level_for`);
`a_plan_at_the_games_quad_scale_settles_on_the_exact_selection` was red
with "the plan never settled".

**Blocks turned the far selection's lag into holes, and the selection is
now worked out every block.** The plan worked its selection out again only
when the eye had moved `RESELECT_BLOCKS` = 32 blocks (1.3 s of flight), so
for up to that long a far mountain could stand a level coarser than the
selection at the eye asks for. That lag is #398's, measured there as STALE
on main's smooth far terrain (10 to 13 kept frames on the low flight) and
never as a hole: a smooth surface a level coarser moves its outline by less
than the error bound. A block surface a level coarser has quads twice as
wide, and where a far mountain's foot meets the ground behind it a pixel
that was land shows sky until the finer patch comes: the flight's HOLES.
Measured on the low flight (`--bench flight --eye 8,120,8 --look
1,-0.15,0.3 --seconds 20`, M3, not quiet):

| far quad scale | every 32 blocks: frames with a hole / stale kept frames | every block |
|---|---|---|
| x2.5 | 26 of 1,194 / 8 | 0 of 1,195 / 1 |
| x3 | 7 of 1,194 / 7 | 0 of 1,194 / 1 |
| x3.5 | 0 of 1,194 / 5 | 0 of 1,195 / 2 |

The 26 are one event of 0.4 s, a pixel or two at the foot of one mountain;
the flight toward the mountain shows none at either distance.

So `RESELECT_BLOCKS` is 1 (chosen, not derived: #398's "tiny" variant; the
eye flies 0.4 block a frame). It is not #398's fix -- its plan replaces the
constant with the slack `select` can report, and generates ahead -- and it
does not make the live selection exact: a patch asked for still takes its
frames to generate. What the plan costs the main thread
(`far_plan_cost_report`, 600 frames of the low flight at the game's radius
and quad scale, 232 patches shown): 200 selections where 32 blocks made 25,
0.013 ms a frame in the mean, 0.041 ms at the 99th percentile and 0.046 at
most. A standing eye selects nothing, as before.

**What the zoom test found in the far terrain, and what became of it**
(2026-11-21; `far_study.rs`'s `summit_lift_report` and
`pits_by_cell_width_report`):

- *Spires on peaks and ridges* were the summit's lift to its highest
  column, the rule's own at every width -- a cell of 8 is level 3's and the
  far terrain's alike, so the far terrain had no rule of its own to correct.
  #437 changed the rule for the voxels and with it for the far terrain (the
  counts are above).
- *Grey patches on slopes* are not a riser's material: level 0's risers
  there are 79% surface block, 21% soil and 0.08% stone. They are the floors
  of pits a cave has opened -- 6.1% of the columns, stone under 5.0% -- and a
  coarse cell was a cave only where all its ground is, so they thinned out
  with the cell and the far terrain drew none. #438 gave a coarse top the
  floor in the share its columns are pit, and the far tops read the same
  rule (above). A wall keeps the blend of its footprint's risers: one layer
  a wall, picked by a hash as a voxel cell must, was tried and puts a
  footprint's walls 4.4/255 off level 0's colour at quads of 16, where the
  bound is 3.25 and the blend measures 2.4.
### 9.9 A pit at every distance: the ground is what the sky sees

After §9.3 to §9.8 the thing the lead's zoom pairs still rated worst
(2026-11-25, the forest line of §9.7: severity 4 at the level 1 -> 0, 2 -> 1
and 3 -> 2 edges alike) was "missing pit / depression with exposed grey
stone". A pit is a cave open to the sky: a column whose surface block the
cave noise has carved away, down to its floor (`WorldGen::pit_floor`).
Level 0 draws every one. A coarse cell was cave only where all of its
ground is (§9.4), and its top was the mean of its columns' *surfaces*, so a
pit showed at a level only where it filled a cell: it appeared out of flat
grass on the way in, and #438 could only paint its colour on the grass.

**Pits are not rare.** Around the spawn a cave opens 6.98% of the columns
(1,024 blocks square: 293 pits, half of them over 149 columns, the largest
4,053); along the forest line 7.14% (seven squares of 512 blocks: 575
pits); on the owner's mountain 5.91%. A pit column is 9 blocks deep at the
median and 26 to 28 at the 90th percentile, and seven to eight in ten are
deeper than the soil: their floor is stone.

**The study** (`crates/render/tests/lod_consistency/pit_study.rs`,
`pit_rule_study_report`). Every candidate is worked out from the world's own
columns (`surface_height`, `pit_floor`, `solid_on`) over that ground, per
cell width -- 2, 4 and 8 blocks for levels 1 to 3 (level 3 in its rows of
four), and far quads of 16 and 32 on the far terrain's 4 x 4 samples -- and
judged against level 0:

- **the depression from above**: the columns drawn at least a block under
  what the same level draws of the ground with no caves (`TopRule` over the
  surfaces), and the air between the two, as a multiple of level 0's;
- **presence**: the share of level 0's pits (columns that touch along an
  edge; under 16 columns, 16 to 63, 64 and over) the level shows anything
  of. What is left appears out of nothing at a finer level;
- **the swap**: per level-0 column a pit touches, the coarser level's drawn
  top minus the finer one's, in blocks and in pixels at the edge (1080
  rows); summed, the blocks moved per 1,000 columns of ground; and the
  signed mean over *every* column, the systematic part;
- **what an eye sees**: parallel rays from 15, 30 and 60 degrees above the
  horizon, four ways round, half a block apart -- how many of them travel a
  block or more further than over the same level's ground with no caves;
- **the trees** standing on a cell the rule sinks.

Along the forest line (the spawn and the mountain agree to a few points;
the report prints all three), cell width 2 / 4 / 8, then far 16 / 32:

What is drawn (the forest line; level 0's pit column is 11.3 blocks deep in
the mean):

| rule | area from above | air over the pits | mean depth, blocks | pits shown at level 3: small / middling / large, % | at far 16 |
|---|---|---|---|---|---|
| the mean of the surfaces; a row cave where all its ground is (main) | x0.88 / x0.69 / x0.31 / 0 / 0 | x0.78 / x0.56 / x0.19 / 0 / 0 | 10.0 / 9.2 / 7.1 | 0 / 0 / 54 | 0 / 0 / 0 |
| **the mean of what the sky sees (built)** | x1.10 / x1.30 / x1.71 / x2.49 / x4.18 | **x1.03 / x1.03 / x1.01 / x0.99 / x0.98** | 10.7 / 9.0 / 6.7 / 4.5 / 2.7 | **54 / 89 / 100** | 42 / 70 / 99 |
| the surfaces' top, sunk by share x depth | x1.10 / x1.30 / x1.70 / x2.54 / x4.18 | x1.04 / x1.04 / x1.01 / x1.00 / x0.99 | 10.7 / 9.0 / 6.8 / 4.5 / 2.7 | 48 / 83 / 100 | 40 / 63 / 99 |
| a row cave where at least half its ground is | x1.05 / x1.01 / x0.95 / x0.89 / x0.48 | x0.96 / x0.91 / x0.82 / x0.64 / x0.21 | 10.3 / 10.2 / 9.8 / 8.2 / 5.0 | 21 / 44 / 98 | 8 / 7 / 60 |
| ... at least a quarter | x1.12 / x1.19 / x1.35 / x1.55 / x1.68 | x1.03 / x1.10 / x1.29 / x1.34 / x1.14 | 10.4 / 10.5 / 10.8 / 9.8 / 7.7 | 34 / 63 / 100 | 16 / 19 / 86 |
| a cell at least half pit stands at its floors' mean | x1.06 / x1.02 / x0.95 / x0.84 / x0.30 | x1.05 / x1.03 / x0.97 / x0.86 / x0.31 | 11.2 / 11.5 / 11.6 / 11.7 / 11.8 | 2 / 22 / 98 | 2 / 0 / 58 |
| what the sky sees, a pit's bottom at its lowest half | x1.16 / x1.38 / x1.83 / x2.65 / x4.31 | x1.04 / x1.05 / x1.11 / x1.43 / x1.66 | 10.2 / 8.6 / 6.9 / 6.1 / 4.4 | 57 / 90 / 100 | 44 / 74 / 99 |

What a swap changes, at the 160 / 288 / 512-block edges:

| rule | on a pit's columns: mean; 95th percentile, blocks | the mean, pixels | blocks moved per 1,000 columns of ground | every column, pixels | an eye at 15 degrees: rays into a pit, levels 1 / 2 / 3 | trees on a sunk cell, of 562 |
|---|---|---|---|---|---|---|
| main | 2.6 / 2.6 / 4.4; 12 / 13 / 17 | 16.7 / 9.4 / 8.9 | 186 / 189 / 318 | +1.19 / +0.64 / +0.61 | x0.88 / x0.67 / x0.30 | 0 |
| **what the sky sees (built)** | **1.4 / 2.1 / 3.1; 7 / 9 / 11** | **9.3 / 7.7 / 6.3** | 114 / 204 / 399 | **-0.14 / +0.02 / +0.06** | x1.10 / x1.27 / x1.59 | 7 / 11 / 29 |
| sunk by share x depth | 1.5 / 2.2 / 3.2; 7 / 9 / 11 | 9.4 / 7.7 / 6.3 | 116 / 205 / 398 | -0.18 / +0.02 / +0.05 | x1.11 / x1.27 / x1.59 | 7 / 9 / 23 |
| cave at half | 2.2 / 2.1 / 4.0; 11 / 11 / 17 | 14.0 / 7.4 / 8.1 | 167 / 170 / 355 | +0.24 / +0.16 / +0.16 | x1.05 / x0.99 / x0.94 | 5 / 3 / 4 |
| cave at a quarter | 2.4 / 2.0 / 4.1; 12 / 11 / 18 | 15.5 / 7.3 / 8.3 | 194 / 185 / 449 | -0.12 / -0.21 / -0.28 | x1.11 / x1.18 / x1.34 | 8 / 9 / 15 |
| half pit: the floors' mean | 1.3 / 2.3 / 4.2; 6 / 11 / 18 | 8.5 / 8.4 / 8.6 | 102 / 192 / 373 | -0.24 / +0.09 / +0.10 | x1.07 / x1.01 / x0.94 | 3 / 3 / 2 |
| a pit's bottom at its lowest half | 1.4 / 2.0 / 3.1; 6 / 8 / 12 | 9.0 / 7.3 / 6.2 | 118 / 214 / 434 | -0.19 / -0.01 / -0.08 | x1.14 / x1.32 / x1.66 | 11 / 14 / 32 |


What the table says:

- **Main draws a pit in pieces and the far terrain not at all**: 88, 69 and
  31% of its area at levels 1 to 3, a fifth of its air at level 3, none of
  the pits under 16 columns from level 2 on, and half of even the large
  ones at level 3. Every edge deepens the ground on the way in (+1.14,
  +0.64 and +0.61 pixels over *every* column).
- **A vote keeps the area and drops the pit.** Carving a row where half its
  ground is cave draws level 0's area to five points at every voxel level
  -- and shows 61, 19 and 21% of the small pits and 44% of the middling
  ones at level 3: a cell flips whole, so what is under half a cell is not
  there and arrives at once. Its swap is no smaller than main's at 288 and
  512 blocks. A quarter overdraws (x1.30 of the air at level 3) and still
  drops a third of the middling pits; a cell standing at its pit floors'
  mean once half of it is pit is the same all-or-nothing with a deeper
  step. And a vote is a carved row: it is not part of a top, so the
  distance morph cannot slide it.
- **The mean of what the sky sees keeps the air and the pit.** x1.03, x1.03
  and x1.01 of level 0's air at levels 1 to 3 and x0.99 and x0.98 on far
  quads of 16 and 32; every large pit at every voxel level and 99% of them
  at far 16; the smallest step at every edge on the columns a pit touches
  (1.4, 2.1 and 3.1 blocks in the mean, 7, 9 and 11 at the 95th percentile,
  where the votes are at 11, 11 and 17), and no direction left in it
  (-0.14, +0.02 and +0.06 pixels over every column).
- **Summed over the ground it moves less at the near edge and more at the
  far ones**: 114, 204 and 399 blocks per 1,000 columns against main's 186,
  189 and 318. Main moves a pit's own columns the whole depth at once; the
  mean moves the pit's cells and the cells round its rim, each a part of
  it. More columns move, each by less (9.3, 7.7 and 6.3 pixels where main's
  move 16.7, 9.4 and 8.9) -- and what moves is a *top*, which the distance
  morph slides (below). A pit that arrives as a carved row is swapped.
- **What it costs is width.** A cell a quarter pit sinks a quarter of the
  depth, all of it: the depression is drawn x1.10, x1.30 and x1.71 as wide
  at 10.7, 9.0 and 6.7 blocks deep where level 0's is 11.3, and from 15
  degrees an eye sees x1.10, x1.27 and x1.59 as many rays go into a pit as
  at level 0. A pit from afar is a wider, shallower dent that narrows and
  deepens on the way in. That is the size of a cell, as §9.4's 95th
  percentile is on a slope, and it is what the morph spreads over distance.
- **Sinking the surfaces' top by share times depth** is the same rule
  rounded twice, with a second notion of height beside the first: the same
  numbers to a point, fewer of the small pits (68 / 64 / 48% against 80 /
  77 / 54).
- **A pit's bottom at its lowest half** (the summit rule turned over)
  overdraws: x1.11 of the air at level 3, x1.41 and x1.64 on the far
  quads. A summit is lifted because an outline is seen against the sky;
  nothing sees a pit's bottom from afar.

**The rule: the column function is what the sky sees**
(`WorldGen::column`, `column_top`). A column stands at the highest block of
ground the sky sees in it: its surface block, or the floor of the pit a
cave has opened there. That *is* level 0's drawn top, block for block, and
every coarser top is §9.3's rule over it, unchanged: the mean, a half to
the even block, a summit at the mean of its highest half. Nothing else was
added; what follows from it:

- **A cell over a pit sinks by the share of its columns that are pit times
  their depth** (`a_cell_over_a_pit_sinks_by_its_share_of_the_depth`: a pit
  4 x 4 and 20 deep is 320 blocks of air at widths 1 to 8, and 256 at 16,
  whose block and a quarter is drawn as one).
- **The cave rule is §9.4's, on what is left**: a row at or under a cell's
  top is cave where every block of ground in it is, a column's ground being
  what is at or under its own top. The air of a pit is no longer a cave the
  rule has to find; it is where the top is. Caves still nest
  (`coarse_caves_nest`); tops never did.
- **A riser is level 0's faces**, as before: a column shows a side at every
  row above a neighbour's top and at or under its own, so the wall of a pit
  is among the faces a coarse cell picks from, by its depth below its
  column's own *surface*: stone under the soil
  (`a_coarse_riser_is_made_of_level_0s_faces`) -- under the cell's own
  risers, that is: the slope's come first (below, "Measured after").
  `ground_faces` counts the same faces per direction, so a far wall beside
  a sunk quad is mixed from them (`ground_faces_are_level_0s_risers`).
- **A top wears what §9.6 gave it**: what level 0 shows from above one of
  the cell's columns, picked by the cell's hash -- now off the column
  itself (`from_above`), with no second search for the floor.
- **The far terrain follows with no change of its own**: `far::quad_top` is
  `cell_top_sampled`, so a far quad of 8 over a pit stands where the
  level-3 cell does, and a wider one sinks by what its 4 x 4 samples see of
  the pit (`a_quads_top_is_the_voxel_cells_top`). One rule (Rule 5).
- **The distance morph slides a pit as it slides a slope** (#436). A pit's
  floor is its column's top and its wall is a riser between two tops, so a
  node's pit is moved to its parent's over the band and the swap shows
  nothing. `morph::node_columns` reads `cell_top`; nothing in it had to
  change.
- **A tree stands in the dent its cell draws** (`node_trees` reads
  `cell_top`), and rides it through the band as it rides any ground (§9.7).

**What the morph makes of it.** Until now no column a cave had opened
morphed, nor any beside one: a level's pit was swapped for its parent's
ground at the ring edge. On the nodes that hold the pits near the spawn
(`a_morphed_pit_is_closed_and_stands_on_its_parent`) 121 of 190 cells that
hold a pit morph at level 0, 77 of 151 at level 1 and 50 of 70 at level 2,
their meshes closed at every factor and standing on the parent at the end
of the band. What does not morph is #436's limit, unchanged: **a pit that
widens under its rim**. Its wall is an overhang, a column with one among
the rows it shows stays still, and so does every column beside it -- so the
pits that slide are the bowls, whose rims step down a block or two (the
morphing pit cells move 0 / 1 / 2 blocks at the median at levels 0 / 1 / 2
and 4 / 5 / 6 at most), and the rim of a pit with a sheer or overhung wall
still swaps, by the step of the table above.

On screen (`transition_study`, the toward-peak flight, the morph's worst
single frame per edge in pixels over the goldens' tolerance, and its
luminance; main `ca3f5bc` beside it -- level 0's band from 96 blocks,
§9.6.1 -- the same machine and day):

| edge | main | pits in the ground | luminance |
|---|---|---|---|
| 0<->1 (160 m) | 4,933 | **2,889** (-41%) | 785 -> 500 |
| 1<->2 (288 m) | 4,747 | **3,531** (-26%) | 1,005 -> 829 |
| 2<->3 (512 m) | 8,993 | **8,455** (-6%) | 1,576 -> 1,473 |

The holes the morph opens are #436's, as before: 265 of 450 frames checked
with any, at most 32 pixels (main: 251, at most 32). (Everything else in
this section was measured against main `9ec668a`, before §9.6.1 landed; it
moves where level 0's band starts and no mesh, so the counts and the
studies stand. A level-0 pit within 96 blocks of the eye stands exactly
as its blocks do.)

**A merged node is searched as open** (`visibility::stand_in`; §9.5 had it
searched as its children drew it, "not a proof"). A
coarser node is not the less open one any more: a cell beside a pit sinks
where the finer cells under it stood whole, so its dent reaches a face of
the node that the children's narrower pit did not. The seeded crossings of
`the_first_search_after_a_crossing_leaves_out_nothing_that_was_drawn` found
it at once: stepping from the mountain column (336, 213), the level-2 node
(82, 0, 53) merges; as its one known child drew it its sub-block joined one
pair of faces, as the node is, three, and the node beside it was seen only
after another search. With a known child a node is now open until its own
links arrive, as a split node is. What that costs, replayed without a GPU
along the low flight's 41 chunk crossings at the game's rings (a throwaway
probe, as §9.5's was): the first searches ask for 15,393 nodes where they
asked for 15,300 (+0.6%), and neither leaves a node out there. **The
lever, not built:** a node's parent is already generated for its colour
morph (§9.6); keeping that parent's links would make a node that merges
known before it is asked for (15,271 asked for in the same replay).

**Measured after** (`consistency_report` and the studies of §9.3 and
§9.4, each run on main `9ec668a` and on this rule):

- **The surface term has not moved in shape or in stone.** Every voxel
  level's mean column, skyline and summit rows are main's to 0.01 block,
  and the stone share of the sides main's to 0.1 point. That took the
  order the faces are picked in: with a pit's wall among the faces a cell
  picks from, a cell beside a pit wore stone on the side that faces the
  slope too (2.4% of a steep footprint's sides at level 3, level 0: 0.1%).
  A row that is part of the cell's own riser wears the slope; the wall of
  a pit is picked only under it (`coarse_block`).
- **The tops are a little lighter beside a pit.** A cell a pit has sunk no
  longer shades the top of the whole cell below it, where level 0's rim
  columns still do: the worst footprint's T3 on tops went 1.50 -> 1.48,
  1.66 -> 1.93 and 2.88 -> 2.98 at levels 1 to 3 (bounds 2.0, 2.0 and 3.0;
  pooled, at most +0.2).
- **The pitted ground wears level 0's stone again.** The stone share of the
  sides the sky sees on or beside carved ground (the harness's cave term,
  reported, not held), main -> now:

  | | level 0 | L1 | L2 | L3 | far 8 | far 16 |
  |---|---|---|---|---|---|---|
  | rolling hills | 69.1% | 67.3 -> 68.2 | 52.3 -> 62.3 | 23.3 -> 41.8 | 0.0 -> 44.0 | 0.0 -> 19.4 |
  | gentle slope | 58.8% | 52.3 -> 58.1 | 31.1 -> 52.5 | 10.4 -> 28.5 | 0.0 -> 20.6 | 0.0 -> 8.5 |
  | steep mountain | 60.9% | 37.0 -> 50.8 | 17.4 -> 33.9 | 3.6 -> 10.9 | 0.3 -> 16.0 | 0.5 -> 7.4 |
  | cliffs | 49.4% | 35.1 -> 48.8 | 16.5 -> 31.2 | 4.1 -> 10.9 | 0.7 -> 12.9 | 0.3 -> 7.7 |

  What is still missing at level 3 on a slope is the price of the order
  above and of the width: a dent 6.7 blocks deep has under three rows
  below its soil where level 0's 11.3 have seven, and its top rows wear
  the slope.
- **The swap on the whole ground** (`height_rule_study_report`, now over
  what the sky sees): the systematic part stays within 0.1 pixel at every
  edge on every kind of ground; the outline's mean moves by at most 0.3
  pixel (gentle slopes: -0.1 / -0.4 / -0.4 where it was +0.1 / -0.1 /
  -0.2); the 95th percentile at the 512-block edge is a block more on
  three kinds (16.1 pixels on steep ground, 14.1 before). Summits are
  where they were (0 / -0.8 / -2.2 on steep ground).
- **Towers**, the cells standing their own width over all four beside
  them around the owner's mountain (`summit_lift_report`): 34 / 16 / 9 / 5
  at widths 4 / 8 / 16 / 32 where main counts 33 / 12 / 8 / 5. The ones
  more each stand beside a cell a pit has sunk -- the pit's wall, which
  level 0 has too -- so the two tests that hold the count
  (`a_coarse_crest_grows_few_towers`, `a_far_crest_grows_few_towers`) now
  hold a cell against what the cells beside it draw of the ground with no
  cave in it, and count what they counted.

**Triangles** (`geometry_census`, every node of the game's selection
around a gate eye, radius 64, squash 2; main `9ec668a` -> now):

| | ground eye (8, 40, 8) | hill eye (8, 300, 8) |
|---|---|---|
| level 0 | 467,648 -> 481,164 (+2.9%) | |
| level 1 | 340,166 -> 354,294 (+4.2%) | |
| level 2 | 376,534 -> 393,144 (+4.4%) | 522,086 -> 541,538 (+3.7%) |
| level 3 | 448,078 -> 468,994 (+4.7%) | 448,964 -> 471,124 (+4.9%) |
| all | 1,632,426 -> **1,697,596 (+4.0%)** | 971,050 -> **1,012,662 (+4.3%)** |

A dent has walls, and a top that moves is not merged across cells (§9.6),
which is level 0's share. The far terrain at the game's quad scale
(`far_census`'s count through `FAR_QUAD_SCALE`, triangles facing the eye a
frame, Medium / High): ground eye 191,621 -> 192,703 and 283,758 -> 285,349,
hill eye 193,641 -> 194,533 and 286,600 -> 287,790, flight eye 149,745 ->
150,361 and 209,646 -> 210,257: **+0.3 to +0.6%**. What these do to a frame
is the gate benchmark's to say (the lead's, on a quiet machine).

**Generation and build cost** (`generation_cost_report`, microseconds a
node, fastest of three passes, test profile; main -> now):

| | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| generate | 367 -> 361 | 357 -> 423 (1.19x) | 1,088 -> 1,329 (1.22x) | 1,807 -> 2,650 (1.47x) |
| build (generate, mesh, links, trees) | 3,786 -> 3,984 (1.05x) | 9,294 -> 10,346 (1.11x) | 25,798 -> 31,572 (1.22x) | 15,585 -> 19,012 (1.22x) |

Every column asked is now one cave sample over its surface height (and a
walk down the pit under one column in fourteen). It is worker time, to be
judged against the flight benchmark's swap lag with §9.4's. **The lever,
not built:** the cave noise is one octave of value noise, an interpolation
of the eight lattice values around a block, so a lattice cell (22 blocks
a side) none of whose corners is over the threshold holds no cave, and one
look at its corners spares every sample in it. How many cells that is was
not measured. An accelerator only (§3).

**What is left, and why.**

- **A pit is drawn wider and shallower from afar** (x1.3 and x1.7 the area
  at levels 2 and 3). That is one top a cell; a rule that keeps the area
  drops the pit instead (the votes above).
- **The rim of a pit with a sheer or overhung wall still swaps**: a third
  to a half of the cells that hold a pit do not morph (#436's purity
  limit). Letting an overhang ride its column is a change to the morph,
  not to the ground.
- **A tree beside a pit at 160 m.** A coarse tree stands in the dent its
  cell draws and rides it through the band; a level-0 tree does not morph
  (§9.7's first open item). Along the forest line 7 of 562 trees stand on
  a sunk cell at level 1 (11 and 29 at levels 2 and 3), 5.3 blocks down in
  the mean and 14 at most, and those 7 step up at the 160 m swap where
  they moved a block at most before.
- **A sunk cell's top is one pick** (§9.6): the floor in the share its
  columns are pit, the surface block in the rest, so a cell three quarters
  pit is a grass ledge deep in the pit one time in four. The share is
  level 0's; the ledge is not. Wearing the floor by how far the cell has
  sunk would trade the one for the other and was not measured.
- **A far quad over 8 blocks samples its pits**: 42 / 70 / 99% of the
  small, middling and large ones show at quads of 16 and 25 / 38 / 90% at
  32. What it shows has the right air in the sum (x0.99, x0.98).
- **A pit deeper than `PIT_FLOOR_SEARCH`** (256 blocks) stands at the block
  the search stops at. It was 48, and a cave can run on under a pit's
  mouth: the deepest pit columns found are 73 blocks around the spawn, 80
  along the forest line and 118 on the owner's mountain. A colour could
  stop short of the floor; a height every level is held to may not.

**Tests** (no GPU unless said). `crates/world`:
`a_cell_over_a_pit_sinks_by_its_share_of_the_depth` (by hand);
`a_cells_top_is_its_columns_mean_and_its_summit` (the rule over the columns'
tops, cells a pit sinks among them); `a_pits_floor_is_the_first_ground_under_the_carved_surface`
(a column's top is level 0's); `a_coarse_riser_is_made_of_level_0s_faces`
(slope first, then a pit's wall); `ground_faces_are_level_0s_risers` (pit
walls and floor risers among them); `a_coarse_cell_is_cave_only_where_all_its_ground_is`;
`a_quads_top_is_the_voxel_cells_top` (sunk quads among them);
`a_morphed_pit_is_closed_and_stands_on_its_parent` and
`a_morphed_node_closes_with_the_nodes_above_and_below` (pit nodes added);
`a_merged_node_is_searched_as_open` and
`the_first_search_after_a_crossing_leaves_out_nothing_that_was_drawn`.
`crates/render/tests/lod_consistency/pit_study.rs`:
`the_chosen_pit_rule_is_the_games` (the study's copy is the game's);
`every_level_draws_level_0s_pits` (the air within 6% at the voxel widths
and 15% at far 16 and 32, every large pit shown at every voxel level, a
swap with no direction -- and main's rule measured beside it, red);
`the_nodes_draw_the_pits_where_the_rule_says` (the generated nodes, cell by
cell, down to level 0's own blocks). GPU: the goldens of the mountain and
the forest at levels 1 to 3, `lod_boundary`, `ring_edge_288m`,
`morph_level_1` and the two far scenes were drawn again -- each shows its
pits as dents with stone walls where it showed flat ground, or flat paint
(#438) -- and `a_fully_morphed_level_draws_its_parents_colours` and
`a_forest_swaps_nothing_at_its_ring_edges` pass unchanged.

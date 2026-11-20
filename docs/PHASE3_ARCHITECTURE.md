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
   coarse cell is cave where most of it is and only within
   `DISTANT_CAVE_DEPTH` of the surface; the far terrain has none. Left as it
   is (PHASE1 §8.6); the harness counts it and keeps it apart from the rest.
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
  cave mouth, a coarse cell is cave where most of it is and only within
  `DISTANT_CAVE_DEPTH` of the surface, the far terrain draws none. A cave
  mouth is detail. It is 4 to 12% of level 0's sky-visible area and about
  half stone, so counted in with the surface it made a steep slope look
  4.0% stone where its surface is 0.1%.
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
/// The column function, and level 0 itself: the y of a column's surface block.
pub fn surface_height(&self, x: i32, z: i32) -> i32;

/// What a column is made of `depth` blocks below ITS OWN surface.
pub fn ground_layer(depth: i32) -> GroundLayer; // Surface | Soil | Stone

/// A cell's drawn top block: the MAX of surface_height over the columns
/// [x0, x0 + w) x [z0, z0 + w). cell_top(x, z, 1) is surface_height(x, z).
pub fn cell_top(&self, x0: i32, z0: i32, w: i32) -> i32;

/// The same over an n x n grid of the cell's columns: the single
/// implementation (cell_top is this at n = w).
pub fn cell_top_sampled(&self, x0: i32, z0: i32, w: i32, n: i32) -> i32;

/// What level 0 shows over a square: riser area per direction and layer,
/// and how open (un-occluded) its tops and risers are.
pub fn ground_faces(&self, x0: i32, z0: i32, w: i32, n: i32) -> GroundFaces;
```

- **Height: the max, to the block, at every level.** A coarse cell's top, a
  far quad's top and a level-0 column's top are `cell_top` at three widths.
  A peak keeps its height exactly, and a parent cell's top is the highest of
  its children's (`exact_tops_nest`), so a coarser level never shows sky
  where a finer one has land. **Level 3 stores rows of four blocks and is
  drawn to the block all the same** (PR 2, §9.4): the row that holds a
  column's top is cut off at it. PR 1 gave `cell_top` a row height and
  rounded to it, because level 3 drew its rows whole: -1, +2, +1 or 0
  blocks off by the top's height mod 4, so on a plain at 1 mod 4 level 3
  stood two blocks above level 2, a wall ringing the player 512 blocks out
  (the lead's audit, 2026-11-20). The parameter is gone with the rounding.
- **Material: by depth below each column's own surface.** `ground_layer` is
  the only depth rule; `material_at` reads it. A coarse riser's row at
  height `y` wears `ground_layer(s - y)`, where `s` is the lowest covered
  surface that is at or above `y` -- level 0's staircase projected onto the
  riser, so a gentle slope's coarse riser is grass side row after row, and
  stone shows where a covered column really stands more than `1 +
  SOIL_DEPTH` blocks proud (PR 2). A far wall, which cannot band, takes the
  mix `ground_faces(..).sides[d]` gives for its quad's footprint (#422).
- **Light: level 0's openness.** `ground_faces` carries how open level 0's
  tops and risers are, from the mesher's own corner rule
  (`cubara_voxel::face_ao_of`, which `face_ao` now delegates to), so a
  representation with no blocks to occlude each other can still be lit
  `ao_floor + (1 - ao_floor) * open`, as `mesh.wgsl` lights a block.
- **`ground_faces` is held to the real mesher**:
  `ground_faces_are_what_the_level_0_mesher_draws` counts the game's level-0
  quads over the same square and requires every count equal -- tops, and
  riser area per direction and layer -- wherever no cave touches the
  surface. The openness is within 0.08 on tops and 0.14 on a direction's
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

- quad top face = `cell_top(x0, z0, q) + 1` for `q <= 8`. At `q = 8` a far
  quad and a level-3 cell cover the same 64 columns and get the same number,
  so the 1,024 m join is exact (both are on the 8-block grid) -- and there
  is no row height left to get wrong: until PR 2 the join needed the far
  quad to round to level 3's rows of four as well.
- above 8 blocks the exact max costs `q^2` heights a quad, so the far
  terrain calls `cell_top_sampled` with `n = 4`. It can only be lower than
  the exact top, and it does not nest (a child can stand above its parent);
  the harness's `stepped` rows measure it per quad width.
- wall colour facing direction `d` = the `Materials` mixed by
  `ground_faces(x0, z0, q, n).sides[d]`, times the openness; top colour =
  `Materials.top` times `top_open`. Not `riser_colour` of the wall's own
  height, which is divergence 2 again.

**What the max costs in shape** -- measured, not guessed
(`stepped_top_rule_report`; drawn minus level 0, mean over columns, with the
skyline's mean in brackets and the peak last):

| steep mountain, cell width | max (#423 d1) | mean of the columns, for comparison |
|---|---|---|
| 2 (level 1) | +1.2 (+0.6), peak 0 | 0.0 (-0.3), peak -0.7 |
| 4 (level 2) | +3.5 (+1.8), peak 0 | 0.0 (-1.4), peak -1.8 |
| 8 (level 3, far join) | +8.0 (+3.9), peak 0 | 0.0 (-3.5), peak -5.5 |
| 16 | +16.5 (+7.9), peak 0 | 0.0 (-7.6), peak -13 |
| 32 | +31 (+15), peak 0 | 0.0 (-15), peak -27 |
| 64 | +50 (+23), peak 0 | 0.0 (-27), peak -50 |

The max keeps every peak and raises slopes by about half a cell width times
the slope; a mean keeps the bulk and lowers every peak and ridge by as much.
On screen either is about half a cell, since cells are sized in pixels. The
max is the decision (#423 d1, confirmed by the lead on 2026-11-15: a top
never below level 0 never shows sky where level 0 has land). That distant
mountains come out slightly fuller is the owner's to look at, with PR 2's
images; changing it is one aggregator in `cell_top_sampled`.

### 9.4 The voxels on the rule (PR 2)

What changed, all of it behind the functions of §9.3:

- **Height.** `cell_surface_height` is `cell_top`: the highest covered
  column. `generate_aniso` (and `generate` at a step above one, which is the
  same rule at `v = step`) asks every column of the node's footprint once
  (`covered_surfaces`) and keeps each cell's heights sorted.
- **Level 3's top is cut off at the exact top.** It still stores rows of
  four blocks (`NodeKey::cell_height_blocks`, #327 decision 1: generation,
  caves and the visibility links are per row of four, as before). A row is
  now solid while it holds any ground, and the one that holds a column's top
  is drawn only up to it: the node carries its columns' exact tops
  (`AnisoChunk::with_tops`) and its mesh is built on block rows
  (`mesh_dims`, `NodeKey::drawn_row_blocks`). So every level's top face is
  `cell_top + 1`, a riser between two level-3 cells is as tall as their tops
  differ, and level 3 nests in level 2 and joins the far terrain exactly.
  `border_samples` asks every level about the block, not the row: a
  neighbour's cell is solid at a block when the block is at or under its
  exact top and its row is not cave.
- **Material.** `coarse_block` replaces `block_at_on_surface_v`, `top_cell`
  and `probe`. At level 0 a column shows a side face at every row above a
  neighbour's surface and at or under its own, made of `ground_layer` at
  that row's depth below its **own** surface. A coarse cell wears the layer
  of **one of the level-0 side faces its columns show at its rows, picked by
  a hash of the cell's position**, so over many cells every layer shows in
  the share it has at level 0. (Two rules came before and each was a
  measured error: depth below the cell's own top, which is main's stone;
  and the lowest covered surface at or above the row, the first version of
  this PR, which never showed stone and hardly any soil either.) A level-3
  cell is one block id over four rows: its face is picked among all four
  rows' faces, and the topmost cell wears the surface block.
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

**After** (same forty footprints, the surface term, pooled over ten
footprints a kind with the worst single footprint in brackets; §9.2's
"before" beside it):

| steep mountain | L1 | L2 | L3 |
|---|---|---|---|
| stone share of side faces (level 0: 0.1%) | 0.1% (was 6.6) | 0.1% (30.3) | 0.0% (39.8) |
| T3, tops (+Y), /255 | 1.02 (1.41), was 6.9 (7.9) | 0.12 (0.89), was 11.3 (12.6) | 1.26 (3.40), was 14.1 (15.1) |
| T3, worst side direction, /255 | 0.93 (1.29), was 2.0 (4.3) | 0.85 (1.42), was 8.1 (13.9) | 1.21 (2.25), was 10.4 (15.7) |
| peak, blocks | 0.0 (was -1.1) | 0.0 (-3.0) | 0.0 (-6.1) |
| skyline, mean, blocks | +0.6 (was -0.9) | +1.7 (-2.0) | +3.6 (-3.8) |
| mean column, drawn - level 0, blocks | +1.2 | +3.5 | +7.8 |

| | L1 | L2 | L3 |
|---|---|---|---|
| stone share of sides, cliffs (level 0: 0.7%) | 0.5% (was 11.2) | 0.6% (37.7) | 0.6% (47.1) |
| stone share of sides, gentle slope (level 0: 0.0%) | 0.0% (0.0) | 0.0% (4.0) | 0.0% (9.5) |
| T3 tops, gentle slope | 0.37 (1.13), was 2.5 (4.5) | 0.85 (1.93), was 5.3 (8.2) | 0.49 (3.19), was 8.0 (10.7) |
| T3 tops, rolling hills | 0.11 (0.31), was 0.26 (0.4) | 0.46 (0.62), was 1.2 (1.5) | 0.99 (1.55), was 3.6 (4.0) |

Soil is the other half of "made of the same": on steep ground a third of
level 0's side faces are soil (33%), and the levels show 28%, 28% and 25%
(asserted within 6, 6 and 8 points; the first version of the rule showed 1
to 19%). Riser *area* is 0.96 to 0.99 of level 0's at level 1, 0.87 to 0.95
at level 2 and 0.71 to 0.88 at level 3 on slopes: bumps inside a cell have
no riser. It is reported and not bounded (§9.1 item 3).

**#423 decision 3's 2/255 holds at levels 1 and 2, in every direction of
every single footprint, and not at level 3.** Worst footprint and direction
measured, tops then sides: level 1 1.44 and 1.29; level 2 1.93 and 1.53;
level 3 **3.40** (nine of forty footprints are over 2.0) and **2.25** (two
of 63 ranked directions). The harness asserts 2.0 and 2.0, 2.0 and 2.0, 3.5
and 2.5 (`T3_ALLOWED`; level 3's are chosen, just over the measured worst).
What is left at level 3, in order (`top_shade_by_riser_report` takes the
first apart):

1. **A coarse top knows the risers beside it and nothing of the steps
   inside it.** Its shade is cast by its neighbours' heights; level 0's, by
   every step among the 64 columns under it. A cell with no riser beside it
   keeps 0.96 to 0.99 of its light where its blocks keep 0.80 to 0.94, and
   one under a riser as tall as the cell is wide keeps 0.77 where they keep
   0.79 to 0.83: a footprint is off by whichever it has more of (the worst
   one too light, the next too dark).
   **The lever:** shade a coarse top by level 0's openness over its own
   columns -- `ground_faces(..).top_open`, the number the far terrain is to
   use (§9.3) -- carried per column the way the exact tops are. Not built.
2. **A cell below the top one is one material over four rows**, where level
   0 changes layer row by row: 1.3 to 2.1/255 of the worst side directions
   is material alone.
3. **Pits.** A coarse cell is cave where most of it is (§9.1 item 7), so
   level 3 opens a pit under ground that level 0 leaves whole: up to 267 of
   a footprint's 3,700 columns (39 at level 2, 10 at level 1), bounded by
   `a_coarse_level_keeps_level_0s_shape`.
4. **Level 0's own chunk-edge AO blindness** (§9.3), which is about 2/255 on
   steep tops by itself and is in the reference every level is held to.

**The shape, and what the max costs.** Peaks keep their height exactly at
every level (asserted; one gentle footprint's peak stands over a level-3
pit, a block down). The ground is never lower in the bulk. Slopes swell, as
§9.3's table said they would: the worst footprint's mean column stands
+1.77, +5.30 and +12.01 blocks above level 0 at levels 1, 2 and 3, and its
skyline 1.08, 3.23 and 7.27 blocks off (both on cliffs; asserted at 2.0,
5.5, 12.5 and 1.25, 3.5, 7.5). That a distant mountain is slightly fuller,
rather than lower and stonier, is the owner's to judge from
`crates/render/tests/golden/mountain_level_0..3.png` against
`docs/images/423/mountain_level_0..3_before.png`.

**What level 0 itself changed by.** A level-0 node leaves out a border face
only where every neighbour it could have draws solid there, the level-1
cell beside it included (`border_samples`). That cell is now solid up to
the highest of its four columns, where it was solid up to their mean, so a
few more border faces are left out. In the game the level-1 cell stands
there. In a scene drawn at level 0 alone it shows at the outer edge of what
is drawn: strips of wall at the edge that were kept are gone. That is all
of what moved in the four HUD goldens and in `mountain_level_0` (with
`cell_surface_height` alone put back on the mean, they render the same
pixels as before).

**Triangles** (`geometry_census`: meshed triangles per level in the game's
own selection around each gate eye -- radius 64, squash 2 -- deterministic;
resident, not what a frame draws):

| ground eye (8, 40, 8) | main | no cut (rows of four) | **cut, as built** | cut, foot row apart |
|---|---|---|---|---|
| level 0 | 390,624 | 387,756 | 387,756 | 387,756 |
| level 1 | 270,072 | 268,740 | 265,902 | 269,614 |
| level 2 | 332,042 | 327,454 | 306,506 | 323,098 |
| level 3 | 325,368 | 323,584 (-0.5%) | **541,754 (+66.5%)** | 619,834 (+90.5%) |
| all | 1,318,106 | 1,307,534 (-0.8%) | **1,501,918 (+13.9%)** | 1,600,302 (+21.4%) |

| hill eye (8, 300, 8) | main | no cut | **cut, as built** | cut, foot row apart |
|---|---|---|---|---|
| level 2 | 445,828 | 442,074 | 416,796 | 436,816 |
| level 3 | 330,822 | 327,480 (-1.0%) | **547,056 (+65.4%)** | 626,892 (+89.5%) |
| all | 776,650 | 769,554 (-0.9%) | **963,852 (+24.1%)** | 1,063,708 (+37.0%) |

(The flight eye at 3,000 blocks has no voxel node in its selection. The
two columns that were not built were counted at the commits where they
existed; the top shade's corner rule, added after them, moves a level's
count by under 0.1%.)

Level 3 grows by two thirds, and all of it is the cut: an exact top means
two neighbouring 8-block cells differ by a block or three almost
everywhere, so tops that merged into terraces of four no longer do and
every cell edge has a riser. Flat ground costs nothing more.

**Decided by the lead, 2026-11-20: the cut, with a riser's foot row merged
into its shade band** (the bold column). The requirement it rests on, as the
lead put the owner's: approaching a mountain, it must not shift or change at
all, and that must hold on a weak PC too, where an automatic quality setting
chooses distance and detail. Rounding between levels is therefore not
acceptable at any quality; exact heights make "no shift" true by
construction, and what they cost is to be recovered elsewhere (per-PC
quality, engine optimisations), not by rounding. What the count does to a
frame is the gate benchmark's to say (rows in `BENCHMARKS.md`, taken by the
lead on a quiet machine).

**Generation and build cost** (`generation_cost_report`; microseconds per
node, fastest of three passes, test profile with `cubara-world` optimised
and the mesher not, on a shared machine -- good for the ratio and nothing
finer):

| | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| generate, before | 373 | 285 | 675 | 398 |
| generate, after | 377 | 328 | 999 | 1,583 |
| | 1.0x | 1.2x | 1.5x | 4.0x |
| build (generate, mesh, links, trees), before | 1,098 | 2,146 | 5,757 | 3,634 |
| build, after | 1,100 | 2,512 | 7,333 | 12,908 |
| | 1.0x | 1.2x | 1.3x | 3.6x |

A level-3 node asks 16,900 surface heights where it asked 1,024, counts
level 0's faces for its cells near the surface (a cell at or under its
columns' lowest neighbour shows none and is spared the count: 2,113 without
that), and its mesh is swept, and its border sampled, on four times the
rows. It is worker time, not frame time. The exact max is what makes peaks,
the nesting of level 2 inside level 3 and the 8-block far join exact, and
was accepted at 3.3 times (lead, 2026-11-16); **at 4.0 times for generation
it is a known cost, to be judged against the flight benchmark's swap lag**
(lead, 2026-11-20). **The levers:** cache a node column's height grid across
its vertical stack of nodes -- an accelerator only, by §3 -- which roughly
halves generation; and sample a level-3 border per stored row of four
wherever the block is not in the row that holds the neighbour's top. Neither
is built.

**The chunk arena.** Every resident node's mesh lives in one vertex and one
index buffer of fixed capacity (`cubara_render`'s `ChunkArena`), and a node
that does not fit is skipped: a hole. It held 4.0 million vertices, sized
from the scene around the origin when the node tree landed (#111). Standing
on a mountain the game's selection is larger than that **on main already**:
4,233,148 vertices around chunk column (336, 213) of the default seed,
steady state, before the nodes a swap keeps resident twice. With level 3's
exact tops the same selection is 4,583,520. So PR 2 raises the arena to 8
million vertices and 12 million indices (lead, 2026-11-20; 96 + 48 MB of
buffers, from 48 + 24), and
`streaming::tests::the_arena_holds_the_selection_around_a_mountain` holds
the worst of three seeded mountains, times 1.25 for a swap (chosen, not
derived), under it. A capacity derived from the selection, instead of a
constant held to it, is the lead's follow-up.

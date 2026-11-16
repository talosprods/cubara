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
  (the drawn top over every column, the skyline, the peak, in blocks; level
  0 is the height field `surface_height + 1`), **T3** (the area-weighted lit
  colour per face direction, /255 in linear light as `t3_coarse_ao.rs` has
  always measured it), and **the stone share** per direction.

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
  mouth is detail. It is 4 to 12% of level 0's sky-visible area and more
  than half stone, so counted in with the surface it made a steep slope
  look 4.0% stone where its surface is 0.1%.
- **Per footprint, not only pooled** (finding B3): every comparison is made
  footprint by footprint, the report prints the worst single footprint
  beside each pooled number, and the assertions hold the worst footprint.

`consistency_report` (ignored; `cargo test -p cubara-render --test
lod_consistency -- --ignored --nocapture consistency_report`) prints all of
it. **Before** (main at `4fe57c4`; the surface term, pooled over ten
footprints a kind, with the worst single footprint in brackets):

| steep mountain | L1 | L2 | L3 | far, quad 8 |
|---|---|---|---|---|
| stone share of side faces (level 0: 0.1%) | 6.6% | 30.3% | 39.7% | 0.0% |
| T3, tops (+Y), /255 | 6.9 (7.9) | 11.2 (12.6) | 14.1 (15.3) | 16.4 (18.0) |
| T3, worst side direction, /255 | 2.0 (4.3) | 8.1 (13.9) | 10.4 (15.7) | 4.4 (6.2) |
| peak, blocks | -1.3 | -2.6 | -5.7 | -3.9 |
| skyline, mean, blocks | -1.2 | -2.2 | -4.0 | -2.7 |

| | L1 | L2 | L3 |
|---|---|---|---|
| stone share of sides, cliffs (level 0: 0.7%) | 11.2% | 37.8% | 47.1% |
| stone share of sides, gentle slope (level 0: 0.0%) | 0.0% | 4.0% | 9.5% |
| T3 tops, gentle slope | 2.5 (4.4) | 5.3 (8.0) | 8.0 (10.4) |
| T3 tops, rolling hills | 0.25 (0.4) | 1.2 (1.5) | 3.6 (4.0) |

The cave term, for the record (share of each level's sky-visible area on or
beside carved ground, steep mountain): level 0 5.8%, L1 7.8%, L2 12.6%, L3
23.2% -- it grows with the cell, since one carved column puts its whole cell
there -- and 65% of level 0's cave side faces are stone.

On the rolling hills level 1 is within 2/255 in every direction of every
single footprint (worst 0.59); that is the one thing the harness asserts of
today's coarse levels (`level_1_matches_level_0_on_the_rolling_hills`). The
existing T3 (`t3_coarse_ao.rs`) passes because it samples around the
origin, which `MOUNTAIN_SEED_MIX` keeps in the hills.

**The measurement is itself seen to fail** (finding B2): non-ignored tests
go red when the level-0 reference is swapped for a coarse level, when every
face is lit as a top (`level_0_draws_its_surface`), and when a far row's
riser direction is swapped or its tops are shifted
(`the_far_rows_follow_the_ground`).

### 9.3 The rule

**One pure column function, and every level derives its ground from it**
(Rule 5; #423 decisions 1, 2). All in `cubara_world::WorldGen`:

```rust
/// The column function, and level 0 itself: the y of a column's surface block.
pub fn surface_height(&self, x: i32, z: i32) -> i32;

/// What a column is made of `depth` blocks below ITS OWN surface.
pub fn ground_layer(depth: i32) -> GroundLayer; // Surface | Soil | Stone

/// A cell's drawn top block: the MAX of surface_height over the columns
/// [x0, x0 + w) x [z0, z0 + w), rounded to rows of v. cell_top(x, z, 1, 1)
/// is surface_height(x, z).
pub fn cell_top(&self, x0: i32, z0: i32, w: i32, v: i32) -> i32;

/// The same over an n x n grid of the cell's columns: the single
/// implementation (cell_top is this at n = w).
pub fn cell_top_sampled(&self, x0: i32, z0: i32, w: i32, v: i32, n: i32) -> i32;

/// What level 0 shows over a square: riser area per direction and layer,
/// and how open (un-occluded) its tops and risers are.
pub fn ground_faces(&self, x0: i32, z0: i32, w: i32, n: i32) -> GroundFaces;
```

- **Height: the max, rounded the same way at every level.** A coarse cell's
  top, a far quad's top and a level-0 column's top are `cell_top` at three
  widths. A peak keeps its height at every level, and a parent cell's exact
  top is the highest of its children's (`exact_tops_nest`), so a coarser
  level never shows sky where a finer one has land. The rounding to rows of
  `v` is the one `drawn_top` already had; `drawn_top` is now `cell_top` at
  `w = 1`.
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

- quad top face = `cell_top(x0, z0, q, v) + 1` for `q <= 8`. With `q = 8,
  v = 4` a far quad and a level-3 cell cover the same 64 columns and get the
  same number, so the 1,024 m join is exact (both are on the 8-block grid).
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

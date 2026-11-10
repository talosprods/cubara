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

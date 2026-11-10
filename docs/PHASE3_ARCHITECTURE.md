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

Nothing below this line yet — block 3.1 (water) is next. Each block's PR adds
its own `## §N <block>` section here, as phases 1 and 2 did.

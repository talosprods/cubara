# Audit: crates/voxel/src + crates/world/src (branch gui-test; origin/main identical for these paths)

Read in full or in the load-bearing parts: worldgen, noise, node, visibility, far (non-test), world, raycast, region, durable, chunk_state, streaming, world/mesh, voxel/storage, voxel.rs (chunk, greedy mesher), connectivity, coord, bounds. Not read in depth: item/recipe/smelt/registry/angle/fixed/block_entity bodies. No cargo was run.

Determinism review: noise is integer hash plus plain IEEE f32 (no FMA, no unseeded RNG, no wall clock). Ore seed uses FNV. Edits/block entities/chunk states are BTreeMaps. No violation found in the worldgen path.

## Findings

### 1. [HIGH-MED, CONFIRMED] load_chunk_edits ignores trees: broken tree blocks respawn after load; tree voxels become spurious edits
world/src/world.rs:455-473 (`load_chunk_edits`), with worldgen.rs:959 (`generate` deliberately has no trees) vs world.rs:386-412 (`build_chunk` includes trees).
Scenario: the real server terrain has oak (server/src/lib.rs:485). Player breaks a log -> edits[pos]=AIR. Save writes `edited_chunk_at` (trees included, AIR at pos). Load compares loaded(AIR) with `generate(...)` (no trees, AIR) -> equal -> no edit recorded -> `build_chunk` puts the log back. The trunk/leaves reappear after every reload. Same path in `Server::install_shard` (server/src/shard.rs:213). Inverse effect: every un-broken tree voxel in a dirty chunk differs from `generate` and is stored as a permanent edit, so edit_count and World equality/state hash differ before vs after a reload, and the multiplayer snapshot `edits()` ships tree blocks.
Why untested: both load_chunk_edits tests (world.rs ~834-890) use `oak: None`.
Fix direction: compare against `build_chunk` without edits (terrain + trees), not `generate`; add a test with oak and a broken log.

### 2. [MED, CONFIRMED] Corrupt chunk payload panics instead of returning an error
voxel/src/storage.rs:251-278 (`read_payload`), then `get` at :136-144.
Scenario: tag 1, palette len 0, or packed indices >= palette.len() (hand-edited/corrupt file; bits=1 with 2-entry palette but a data word of 0b11..) passes decode, then the first `Chunk::get` does `palette[idx]` out of bounds -> panic. The doc claims "corruption is always a hard error". Also trailing bytes are not rejected, and `decode_region` (region.rs:196-205) accepts `local >= 32768` (maps to a coord outside the region) and duplicate indices.
Fix: validate palette non-empty, every index < len (scan data), consumed == bytes.len(), local < 32768.

### 3. [LOW-MED, PLAUSIBLE] LOD node borders depend on level-0 edits but are not remeshed on edit
world/src/mesh.rs:140-160 (`chunks_affected_by_edit`) returns only the level-0 chunk and level-0 border neighbours. `border_samples` (mesh.rs:193-267) for a coarser node samples the finer neighbour via `certainly_solid_at(.., level-1 == 0 ..)`, which consults edits (world.rs:142-149). Scenario: another player (or a camera far from the editor) digs a border block of a level-0 chunk next to a level-1 node; the level-1 node dropped its covered face and is never remeshed -> hole into terrain. Needs an edit at a ring boundary, so rare for the local player.

### 4. [LOW-MED, PLAUSIBLE] Whole-World clone cost and unbounded chunk_states
world.rs:46-64 (`World: Clone`, "an edit publishes a snapshot by cloning"), server/src/lib.rs ~2016 (`Arc::make_mut(&mut self.world)` every tick). If a mesh job or any reader holds the Arc, make_mut deep-copies edits + block_entities + chunk_states (O(all edits)) each time. Separately `ChunkStates` (chunk_state.rs:99-187) never removes entries (Unloaded unreachable), and `active()` scans every entry on each `keep_simulating` (only on chunk change, so mitigated). Memory and clone cost grow with explored area, contradicting the doc comment at world.rs:231-234.

### 5. [LOW, PLAUSIBLE] Fixed temp name in write_atomic
world/src/durable.rs:74-78. Two writers of the same path (async autosave thread, headless.rs:657, plus a shutdown save) share `<name>.writing`; File::create truncates the other's in-progress file, the first rename can publish interleaved bytes and the second fails NotFound. Comment assumes a single saver; no lock enforces it. No fsync before rename (documented).

### 6. [LOW, CONFIRMED] schedule_for_radius silently caps large radii
node.rs:354-367. radius > 64 (outer ring) returns the default schedule ending at 64, not radius; docs say it "clamps". radius < 0 yields (0, negative).

### 7. [LOW, PLAUSIBLE] Perf: allocation in visibility BFS inner loop
visibility.rs:142-184 `Index::neighbours` returns a fresh `Vec` per call, called per (node, sub-block, face, arrival variant); `stand_in` (:264) allocates `children()` per unknown node. Runs on every camera-node change over ~1.6k+ nodes x 8 subs x 6 faces. Use a small fixed array/SmallVec.

### 8. [LOW, CONFIRMED but unreachable today] Overflow/hang edges
- raycast.rs:70-96: `max_dist` of INFINITY/NaN with all remaining t_max infinite loops (t > max_dist never true); `voxel[a] += step` overflows i32 only at +-2^31.
- far.rs:168-174 `root_level_for(INFINITY)`: `32 << level` shift reaches 64 -> debug panic. Real radii (262144) are fine.
- far.rs:66-67 `PatchKey::children` underflows `level - 1` at level 0 if called directly (walk guards it).
- node.rs `select_nodes` / `chunk_origin` `pos * extent` are fine at 262 km (chunk coord ~16k).

### 9. [LOW, cosmetic] Canopy replaces solid terrain on slopes
worldgen.rs:896-897 / world.rs:403-407: tree blocks win over terrain, and canopy spans y >= top-r over adjacent columns whose surface can be higher than the trunk's, so leaves overwrite hillside stone/grass. Consistent across `block_at`, `build_chunk`, `is_solid_with_trees`, so no desync; just odd terrain. Structure canopy.radius is not validated (structure.rs validates density/height only); radius >= ~16 breaks the "tree reaches only adjacent columns" bound used by trees_near (worldgen.rs:674-700).

### 10. [LOW] Edits never pruned
world.rs:161-164 `set_block` always inserts; placing back exactly what generation produces (or AIR over air) keeps a permanent entry, so overlay/save size grows with actions, not with net difference.

## Tests that cannot fail / weak
- load_chunk_edits tests (world.rs:833-890): never use oak, so finding 1 is invisible.
- visibility.rs:611-647 `a_node_split_or_merged...`: `lost` is filtered to `stand_in.is_some() && !underground`, exempting most of the cases it could catch; the failure window is narrow by design but unverifiable for caves.
- region/storage tests round-trip only well-formed payloads; no corrupt-index test (finding 2).

## Checked and found sound (no defect)
Face-pair bit table (connectivity.rs:33-40, test `every_pair_has_its_own_bit`); visibility `arrived` intersection terminates (strictly shrinking 6-bit set); `stand_in` child/sub-block indexing matches `children()` order; octree `select_nodes` tiling and 1-level balance (tests cover); `ChunkCoord::from_block` div_euclid; region local_index/coord_from_local; greedy mesher slice bounds and skirt guards; storage repack width logic; `cell_surface_height` rounding via div_euclid; ore/structure registry validation (density 0, freq <= 0, threshold range).

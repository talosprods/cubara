# Phase 3: options memo

**This is not a decision.** It is a menu, written so the design conversation can start from the code rather than from a blank page. Nothing in it is admitted, scheduled or settled. Every "option" below is a question for the owner (`CLAUDE.md`: "a plausible-sounding invention is worse than a question"). Drafted 2026-11-09 from the repo at `248fc03`. It is local only and not committed.

The admission rule (`ROADMAP.md`, phase 3) is applied to every candidate:
- (a) which existing systems it deepens, and how
- (b) what it makes possible that was not possible before
- (c) what it replaces or removes

---

## 1. The systems that exist now, and how they connect

```
assets/*.ron  (10 blocks, 16 items, 7 recipes, 1 smelt recipe, 1 ore, 1 tree)
    │  names are identity, numbers are per-world
    ▼
cubara-voxel: BlockRegistry · ItemRegistry · RecipeBook · SmeltBook · OreSet · Fixed · Angle
    │
    ├──► cubara-world: WorldGen (pure in seed,x,y,z)
    │      terrain + mountain ranges (peaks ~y 240) + 3D caves + ore threshold + oak structure pass
    │      World = worldgen + edits overlay + block entities (furnaces only) + chunk states
    │      node tree (voxel LOD, to 1 km) · far.rs height-field patches (1 km to 262 km)
    │
    ├──► cubara-sim: Sim at 60 Hz · players by id (fixed-point positions, binary angles)
    │      physics (AABB sweep) · inventory (36 slots) · crafting (2x2 / 3x3 at a bench)
    │      entities (hecs: dropped items only, despawn by rarity) · health 20, fall damage, regen
    │      WorldHash · save (FORMAT_VERSION 6, refuses a different WORLDGEN_VERSION)
    │
    └──► cubara-server (authority): Action in -> rules -> Effect out
           mining ticks = ceil(hardness / speed) · drops (DropRule + requires_tier)
           wear on a break that yields · furnace ticking + bounded dormant catch-up
           commands /tp /give /gamemode /seed (/time refuses) · interest (VIEW_RADIUS 64,
           SIM_RADIUS_CHUNKS 4) · wire · prediction · shards · coordinator (audit by replay)

cubara-render (knows nothing of the world): arena + mesh.wgsl (lighting as pipeline override
  constants) · far.wgsl (block-aggregate shading, per vertex) · figures (shirts) · panels
cubara-app: Game (client replica) · streaming · far streaming · far quality per PC
  (Off/Low/Medium/High, benchmarked) · pause menu · creative · console
```

**The loop phase 2 built.** Phase 3 deepens this or it doesn't belong:

log → plank (x4, 2x2) → stick (x4) and bench (4 planks) → wooden pick (tier 1, speed 2, durability 50) → stone drops cobble → stone pick (tier 2, speed 4, durability 150) → iron ore (tier 2, hardness 210, only below y 40) → raw iron → furnace (8 cobble). Wood is the only fuel: log 800, plank 200, stick 100 burn ticks. Raw iron becomes an ingot in 200 ticks (one plank per ingot), and three ingots make an iron pick (tier 3, speed 6, durability 400).

The only threat is fall damage: 3 blocks are safe, then 1 point per block, so a fall of 23 blocks kills. Death returns you to spawn with your things.

**Where phase 3 would attach.** Each of these is a single special case, or a hook nobody uses yet:

| # | Seam | Where |
|---|---|---|
| S1 | A block entity is always a furnace: `pub type BlockEntities = BTreeMap<[i32; 3], Furnace>`, `Effect::BlockEntity { furnace: Option<Furnace> }`, and placing a block checks `interact(block) == Interact::Furnace` | `world/src/block_entity.rs:313`, `server/src/lib.rs` `Effect`, placement code at ~2143 |
| S2 | The timer machinery has one user, and its return type is furnace-shaped: `TimedProcess::advance -> FurnaceOutcome`. `REQUIREMENTS.md` #4 founded it for "crops keep growing" | `block_entity.rs:84` |
| S3 | One smelt recipe. `stone.ron` says "placing stone back down is a later crafting step", and that step does not exist | `assets/smelting/`, `assets/blocks/stone.ron` |
| S4 | `Rarity::Treasured` has no item. It was built for "the pile you leave when you die", which §13.4 removed | `PHASE2_ARCHITECTURE.md` §10.5, §13.4 |
| S5 | `Shape` has one variant, `Full`. Shapes as ids in the flat id space were designed in phase 1 §3.2/§3.5 | `voxel/src/registry.rs:33` |
| S6 | Leaves `drops: Nothing`. Leaf decay, saplings, broken-tool remnants and a crafting preview are still open (§9) | `assets/blocks/oak_leaves.ron` |
| S7 | Iron ore `max_y: 40` while mountains peak around y 240. Below `SOIL_DEPTH` (3) it is stone at any depth | `assets/ores/iron.ron`, `worldgen.rs` |
| S8 | `SAFE_FALL`, `FALL_DAMAGE_PER_BLOCK`, `MAX_HEALTH` and `REGEN_INTERVAL` are `pub const` in code, although §13.3 says they "live in data" | `sim/src/player.rs:178-193` |
| S9 | `/time` replies "no time of day yet", and `Lighting::time_of_day` is never used. The sun is a pipeline override constant, because reading a uniform in the fragment stage cost the M3 ~400 FPS | `server/src/lib.rs` `apply_command`, `render/src/render.rs` `mesh_pipeline_constants` |
| S10 | The far terrain has one global material triple, built from the texture names `grass_top`, `soil` and `stone` | `render/src/far.rs:60` `ground_materials`, `app/src/far_streaming.rs:149` |
| S11 | Players are shirts. Nothing in the game is about there being other people. A disconnect drops the player and deletes its save file (audit `audit-sim-server.md` #5) | `server/src/others.rs`, `headless.rs` |
| S12 | One tool `speed` for every block (an iron pick chops oak at 6). Wear is spent on any break that yields something | `server/src/lib.rs:757`, `wear_held_tool` |

---

## 2. Candidates

Sizes are in **PR-sized blocks** (one issue, one PR, as in phase 2). Engine items **E1–E9** are described in §3.

### C1. Growth: saplings, and trees that grow back (the second timed process)

**(a) Deepens**
- **`TimedProcess` and dormancy (S2).** Growth is the case `REQUIREMENTS.md` #4 was written for. Its catch-up is one event (grown or not yet), so `advance(n)` costs O(1), and it is checked by the same rule as the phase gate's dormant test: dormant for N ticks equals ticked N times.
- **The structure pass.** A grown tree uses `assets/structures/oak.ron`, the same shape a generated tree uses, so there is one definition of what an oak is (Rule 5).
- **The furnace economy.** Wood is the only fuel (§7, the owner's call), and the wood near a base is finite today.

**(b) Makes possible**
- A renewable resource, and forestry as something a base does.
- The first change to the world made by the simulation rather than by a player.
- In multiplayer, the forest around spawn is not permanently stripped.

**(c) Replaces / removes**
- The furnace-only block-entity type (S1) becomes an enum of kinds.
- `FurnaceOutcome` stops being the trait's return type.
- Leaves' `drops: Nothing` goes.
- If block-entity slots become `ItemStack`, the audit's durability exploit closes as well (`audit-sim-server.md` #8: furnace slots store `(ItemId, u8)`, so a worn pick comes back new).

**Engine work**
- E1.
- Growth writes many blocks at once, possibly across a chunk border (the canopy radius is 2), so the shard that owns the write must be decided (`PHASE2_MULTIPLAYER.md` §7.4 #1).
- **Prerequisite:** the audit's tree bug (`audit-voxel-world.md` #1: broken logs reappear after a reload, because `load_chunk_edits` compares against `generate` without trees). Growth makes trees-as-edits the common case.
- A sapling can be a Full block (no engine work) or a plant shape (E3).
- Random drops use the seeded RNG (`Sim::roll`) and stay server-side.

**Size:** 4–5 blocks, plus 2–3 if crops are included.

**Owner decisions**
- Do leaves drop saplings? *(always / sometimes / never: saplings come from somewhere else)*
- How does a tree grow? *(all at once after N ticks / in stages you can watch / only when there is room, otherwise it waits)*
- Trees only? *(trees only / trees and one crop / crops wait for hunger, C7)*
- Leaf decay? *(no, as now / yes.)* Decay is a non-local process and is hard to make dormancy-correct.

### C2. Heat: charcoal and a fuel ladder (the first production chain)

**(a) Deepens**
- **The furnace.** Today it has one recipe and fuel means `burn_ticks` only. Fuel would gain a `heat` and recipes a `min_heat`. The event-jumping catch-up (§12.1) gains no new kind of event, because "this fuel is too cold for this recipe" is a stall, so bounded catch-up keeps its shape.
- **Tool tiers.** Whatever comes after iron is gated by a hotter process, not only by a new ore.
- **Trees,** together with C1: wood becomes charcoal, which drives a hotter smelt.
- **S3:** cobble back to stone as one data file.

**(b) Makes possible**
- The minimal Factorio pattern: a furnace's output is another furnace's fuel.
- A reason to run more than one furnace.

**(c) Replaces / removes**
- The scalar fuel model (`burn_ticks` alone).
- The dead end where stone is mined and can never be placed back.
- The owner's "no coal" decision stands: charcoal is made from wood.

**Engine work**
- No rendering work.
- `ItemDef` (`voxel/item.rs`) and `SmeltRecipeDef` (`voxel/smelt.rs`) change shape.
- The furnace gains one field, the heat of the fuel currently alight, so swapping fuel mid-burn stays deterministic.
- The property test `bounded_catch_up_agrees_with_the_one_tick_reference` is extended over heat.

**Size:** 2–3 blocks.

**Owner decisions**
- Does charcoal gate something wood cannot do? *(yes, it gates / it only burns longer / no charcoal)*
- Is there a tier above iron in phase 3? *(no, iron stays the top / one new metal, with an original name / a better iron from a hotter furnace, no new ore)*
- Cobble back to stone? *(furnace / bench / not at all)*

### C3. Logistics: items that move between block entities

**(a) Deepens**
- **Dormant catch-up, from one entity to a network.** Today each furnace is caught up alone (`advance_furnace`, server `lib.rs` ~2160). A chain (storage → furnace → storage) has to meet the same contract across coupled processes: `advance(n)` equals n single ticks. Event-jumping still works, because a network's next event is the earliest among its members, and the cost stays bounded by item counts, not by elapsed time. This is the hardest test Rule 1 and §12 would face, and it is the problem the Factorio standard names.
- **Client prediction.** Clients already run a furnace forward with `advance`; a network is the same thing at scale.
- **Sharding.** A chain that crosses a shard border is §7.4's "furnace exactly on the seam", made general.

**(b) Makes possible**
- Automation, and bases that produce while nobody is there.
- The Factorio-grade simulation becomes something you can see in play.

**(c) Replaces / removes**
- `Action::ClickFurnace` as the only way items enter or leave a block entity.
- Catch-up one entity at a time.
- The 36-slot inventory as the only storage (a chest arrives as part of this, not on its own).

**Engine work**
- E1 and E2.
- Transport you can see (items moving on blocks) would mean many moving things, which needs instanced rendering. That is the first profile that might ask for #32/#33 GPU culling. Transfer between adjacent blocks only needs no rendering work.

**Size:** 5–6 blocks with adjacent transfer only; 9–12 with visible transport.

**Owner decisions**
- Does automation belong in Cubara? *(yes, at its core / a little: a chest above a furnace feeds it / no, Factorio stays the engineering standard only.)* `ARCHITECTURE.md` says the Factorio reference is "about engineering, not gameplay", and this option would change that.
- Visible transport? *(adjacent only / pipe-like blocks with no visible items / belts with visible items)*
- Do networks run while their chunks are dormant? *(yes, by catch-up / only while loaded)*

### C4. Rock strata: depth as a progression axis, readable from afar

**(a) Deepens**
- **The vertical world (#252/#259).** Today stone is uniform at any depth: a test asserts there is solid stone however far down you go, and 1,000 blocks down looks and plays like 50.
- **Tool tiers and hardness.** `requires_tier` and `ceil(hardness / speed)` already exist per block; strata give them a geography.
- **Ores (S7).** Ores would declare a host layer instead of a `max_y`. `OreSet` holds 4.
- **The far view.** `riser_colour` already shows "the blocks a riser cuts through". Strata are more of exactly that: a mountain face seen from 10 km shows its bands.

**(b) Makes possible**
- Depth and altitude become decisions: dig where the layer you need is shallow.
- Mountains become exposed cross-sections. Prospecting by looking becomes real because of the far view.

**(c) Replaces / removes**
- The uniform stone below `SOIL_DEPTH`.
- Ore `max_y` cutoffs.
- The far terrain's hard-coded material triple (S10). One function giving the materials of a column would be called by both the generator and the far terrain. Today they agree only because both assume grass, soil and stone (Rule 5).

**Engine work**
- E7.
- `WORLDGEN_VERSION` goes from 3 to 4. `save.rs` refuses a world with a different worldgen version, so the owner's world would not load without a decision (see the third question below).
- Golden images regenerate, and the PR says so.
- More bands means more work in the per-vertex far shading, which must stay inside F4's measured cost.

**Size:** 4–6 blocks.

**Owner decisions**
- Layers by absolute height or by depth below the surface? *(absolute: mountains expose deep rock high up / relative: the same sequence everywhere / mixed)*
- Does harder rock need a higher tier? *(the tier gates the drop, like iron ore / only slower / per layer)*
- What happens to the current world? *(a new world, the old one kept as New World already does / keep the old generator loadable: two generators in the code / accept that unedited terrain changes)*

### C5. Water: sea level, safe landings, lakes on the horizon

**(a) Deepens**
- **Fall damage.** Landing in water resets `fall_distance`: the first counterplay to the only threat, on 240-block mountains.
- **The far view.** A sea-level plane in the patches gives coasts and lakes as landmarks to 262 km, with F5's haze over the water.
- **Worldgen.** Valleys fill to sea level.
- **The transparency seam** phase 1 §11 kept open: "a second arena and a second pass, sorted back-to-front". Leaves, solid and opaque today, could move to it.
- **Physics.** Swimming is written in `Fixed`.

**(b) Makes possible**
- Geography: coast, lake, valley.
- Safe ways down.
- Navigating by coastline.

**(c) Replaces / removes**
- `Material.solid: bool` as the whole story: water is neither solid nor opaque.
- Visibility (#253/#254) treating everything that isn't air as opaque.
- The dry lowlands.

**Engine work**
- E3 (non-solid blocks), E4, E7.
- Water that stays where worldgen put it is pure worldgen. Flowing water is deterministic but hard to make dormancy-correct: it settles, so catch-up means running until it settles, which is bounded by the volume of water, not by the size of the area it can spread to.
- Transparency means overdraw on the M3, which already holds 1,000 FPS only at Low.

**Size:** 6–8 blocks for water that doesn't flow; 3–5 more for flowing.

**Owner decisions**
- Static or flowing? *(placed by worldgen, moved with an item / flows within a limit / flows fully)*
- Does water flood caves below sea level? *(yes / no / only near the surface)*
- Swimming and drowning? *(swim, no drowning / swimming and breath / for now only a soft landing)*

### C6. Light: the time of day, and darkness underground

**(a) Deepens**
- **The tick.** Time of day = tick mod day length. That adds no world state: like terrain, it is derived and never sent.
- **S9:** `/time`, and `Lighting::time_of_day`, which is never used.
- **The far view.** Aggregate shading already takes the sun direction, so dusk would show long shadows on mountain faces and F5 haze could take its tint from the sun.
- **Caves and depth.** Today they are lit like the surface (a sun term plus ambient occlusion, no light propagation).
- **The furnace.** A lit furnace (`burning > 0`) could give light, and wood or charcoal could make a light source.

**(b) Makes possible**
- A rhythm to the day.
- Darkness as the cost of going deep.
- Navigating by the sun.
- A home for hostile creatures, if they are ever wanted.

**(c) Replaces / removes**
- The fixed `sun_dir` override constants in `mesh.wgsl`.
- `/time` refusing.
- The uniform ambient light underground.

**Engine work**
- **E6, the sun.** The voxel shader's fragment stage reads no per-frame data on purpose (a measured cliff on the M3), so a moving sun is one of three things: stepped (the existing background pipeline rebuild, every N seconds), moved to the vertex stage, or paid for in FPS.
- **E5, darkness.**
  - Light per vertex fits in the packed vertex (word 2 has 16 spare bits), but faces with different light cannot be greedily merged, so there are more triangles. Placing a light re-meshes the neighbouring chunks.
  - Skylight in a world with no height limit is non-local, because it depends on every block above. That conflicts with the purity of `generate(seed, coord)`.
  - A cheaper alternative: ambient light as a pure function of depth below `surface_height(x, z)`, plus light from blocks only.

**Size:** time of day 2–3 blocks; darkness 5–8.

**Owner decisions**
- Day/night: does night do anything? *(looks only / growth or danger depends on it / no cycle)*
- How dark is it underground? *(as now / darker with depth, light sources needed / full light propagation)*
- May it cost FPS on the M3? *(no, a stepped sun only / a small measured cost / whatever it takes)*

### C7. Hunger and food: the condition §13.2 left open

**(a) Deepens**
- `Player::tick_regeneration`, which has no condition today. §13.2 says that when hunger arrives it "becomes the gate", a change to one condition.
- **The furnace.** Cooking is a `SmeltRecipe`, the same machinery.
- C1 as the source of food.

**(b) Makes possible**
- A cost to long expeditions (mountains, deep layers, far landmarks).
- Supply as something to plan.

**(c) Replaces / removes**
- Regeneration that needs nothing.
- The phase-2 gate's survival replay could eat again. Eating was removed on 2026-09-05.

**Engine work:** a field in `PlayerState`, the hash and the save, so prediction stays exact; a nutrition field on `ItemDef`; a HUD bar.

**Size:** 3–4 blocks, plus C1.

**Owner decisions** (put to him on 2026-09-05, answered "not yet", still open)
- What is hunger? *(a bar that gates healing / food heals directly / no hunger)*
- Where does food come from? *(growth / animals, which needs mobs / both)*
- Starving? *(stops healing / slowly hurts / can kill)*

### C8. Death with a cost: the Treasured pile, and a place to come back to

**(a) Deepens**
- **Fall damage,** which costs nothing today.
- **`Rarity::Treasured` (S4).**
- **Dropped items.** The §10.4 path already spawns items in the world.
- **`Player::spawn`,** which is already per player, so a respawn block only has to set it.
- **Multiplayer.** Others can find your pile, and entities crossing shards already work (#221).

**(b) Makes possible**
- Risk, and preparing before a descent.
- Going back for your things.
- Players helping one another.

**(c) Replaces / removes**
- §13.4's "returns you with your things". That was the owner's own decision, so changing it is his call.
- The world spawn as the only respawn point.
- Moves S8's constants into data, as §13.3 already promised.

**Engine work:** small. Ownership of a pile needs a player who persists, which needs E9.

**Size:** 2–3 blocks.

**Owner decisions**
- What does dying cost? *(nothing / everything, in a pile that never despawns / part of it, or tool wear)*
- Who may take the pile? *(anyone / only you / you first, then anyone)*
- A respawn block? *(no / a craftable block / a block, with the world spawn as fallback)*

### C9. Shapes and climbing: slabs, stairs, ladders

**(a) Deepens**
- **The registry's material × shape expansion (S5).** Shapes are ids in the flat id space, so the save format needs nothing.
- **Fall damage.** A climbable block resets `fall_distance`.
- **Mountains with cliffs** (`mountains_have_cliffs`) and unlimited depth.
- **Building.**

**(b) Makes possible**
- Safe vertical travel.
- Architecture at the scale the world now has.

**(c) Replaces / removes**
- Pillaring and one-block jumps as the only way up.
- The full-cube assumption in the mesher, visibility and physics.

**Engine work:** E3, across:
- greedy merging and face culling (a slab does not hide its neighbour's face)
- `connectivity.rs` and visibility (a partial block lets sight through)
- a collision box per shape
- coarse LOD nodes
- `crack.rs`

The far terrain is unaffected.

**Size:** 5–7 blocks.

**Owner decisions**
- Which first? *(ladders, for travel / slabs and stairs, for building / all three)*
- Climbing as a block or a property? *(a ladder block / some blocks are climbable, logs for example / both)*

### C10. Claims: who may change what, in a shared world

**(a) Deepens**
- **Untrusted clients (2.14).** The server checks reach, speed, inventory and rate today, but not *permission*. All edits go through one path (`break_at_as`, `place_as`, `ClickFurnace`), so there is one place to check.
- **Sharding.** The owner's §7 words, "elke speler zijn eigen gebied actief kan houden", fit a claim as the unit a shard follows. `Server::assigned` already exists for regions that are simulated because they were assigned.
- **Dormancy.** Bounded catch-up already keeps furnaces correct while asleep, so keeping a region active matters only for C3's networks.

**(b) Makes possible**
- Public play with strangers. A world for 5,000 is otherwise one griefer from nothing.
- Shared bases.
- Ownership as a fact of the world.

**(c) Replaces / removes**
- Any client may edit any block within reach.
- Being near a player (`SIM_RADIUS_CHUNKS = 4`) as the only reason a region is active.
- The audit's `centre_player()` = the lowest id.

**Engine work**
- **E9 first.** `PHASE2_MULTIPLAYER.md` §8: "who is allowed to connect, and how that is proved, is a product decision nobody has made". Today a disconnect deletes the player.
- Claims become world state in the hash, the save and the wire, and interest management replicates them.
- E8 once players spread out (the eye is an `f32` `glam::Vec3`).

**Size:** 4–6 blocks, plus 2–3 for E9.

**Owner decisions**
- Friends or strangers? *(private: no claims / public: claims / a server option)*
- Identity? *(a name, trusted / a key the client keeps / accounts on a service)*
- How is land claimed? *(a block with a fixed radius / paid for with a resource / assigned by an admin)*

---

## 3. Shared engine work

| | Engine work | Needed by | Size |
|---|---|---|---|
| E1 | Block-entity kinds: an enum with `ItemStack` slots, save format 6 → 7, wire, hash, and the block declares its entity kind | C1, C2, C3 | 2 |
| E2 | Coupled catch-up for networks, with a one-tick reference stepper as §12.2 kept | C3 | 2–3 |
| E3 | Non-full and non-solid blocks in the mesher, visibility, physics and crack overlay | C9, C5, C1 (optional) | 3–4 |
| E4 | A transparent pass: a second arena, sorted back-to-front | C5, leaves | 2–3 |
| E5 | Light data in vertices, and propagation | C6 (darkness) | 4–6 |
| E6 | A moving sun without the M3 override cliff | C6 (time of day) | 1–2 |
| E7 | Far-terrain materials from the generator: one function, used by both sides | C4, C5 | 1–2 |
| E8 | Camera-relative rendering (`PROPOSAL_FAR_VIEW.md` §3.5) | C10, any long travel | 1–2 |
| E9 | A player identity that persists across disconnects | C10, C8, all multiplayer | 2–3 |

**Bugs to clear first** (the 2026-11-09 audit reports):
- Trees treated as edits after a reload: blocks C1.
- Furnace slots lose durability: fixed by E1.
- A disconnect drops the player: E9.

**Open render and perf issues.** None of the candidates' profiles asks for these yet:
- #42 (occlusion culling) was built and measured, and it does not pay (#256, `BENCHMARKS.md` ⁵⁵).
- #32/#33/#28 (GPU culling), #130 (a spatial index for culling) and #131 (front-to-back ordering) wait for C3's visible transport or C5's overdraw.

---

## 4. Three shapes for phase 3

### Shape I: The world keeps working (C1 + C2 + C3, with C7 through cooking)

`REQUIREMENTS.md` #4 founded the chunk system on "timers for while you're away, so crops keep growing", and after phase 2 exactly one process uses those timers: the furnace. This shape gives that machinery the job it was built for:
- Trees grow back (C1).
- The furnace becomes a ladder that feeds itself, from wood to charcoal to a hotter smelt (C2).
- Items move between processes (C3), so a base is a network that catches up while dormant.

Every piece deepens the same three systems: block entities, `TimedProcess` and dormancy. Each replaces a path that serves exactly one case: block entities that can only be furnaces, `ClickFurnace` as the only way items move, and catch-up one entity at a time. It is the purest test of the engineering thesis: Rule 1 across coupled processes, bounded catch-up for a network, and chains across shard seams. Multiplayer prediction already works for a furnace, and a chain is the same at scale. Most of the work is in the simulation, so the performance risk is low.

The cost is an identity question. It would make Factorio the gameplay reference too, which `ARCHITECTURE.md` deliberately did not claim. Roughly 12–16 blocks.

### Shape II: The deep and the high (C4 + C6 darkness + C9 + C8, with C5 optional)

The world with no height limit and the 262 km view were the two biggest things built after phase 2's gate, and neither has a gameplay consequence yet:
- Below three blocks of soil, the stone is the same forever and lit like noon.
- Iron stops at y 40 while mountains reach y 240.
- The only threat, falling, costs nothing because death keeps your things.

This shape turns the axis the world was rebuilt for into the progression axis:
- Strata give depth a ladder that tool tiers and hardness already know how to enforce.
- Darkness makes depth cost light.
- Ladders and stairs give fall damage a counterplay.
- A death pile gives a fall a price.
- The far terrain shows the strata in every cliff, so 262 km of view becomes information, not only scenery.

It replaces ore `max_y`, uniform stone, the far terrain's hard-coded materials, the full-cube-only mesher and free death. The cost: the heaviest mesher and render work of the three (E3, E5), a `WORLDGEN_VERSION` bump, and the M3's FPS budget. Roughly 18–22 blocks.

### Shape III: One world, many people (E9 + C10 + C8 + C1 + C6 time of day, with E8)

Phase 2 built the structure for 5,000 players in one world. What was played on 2026-09-11 was three people watching each other's edits. Nothing in the game is about there being other people yet. This shape makes multiplayer the system:
- An identity that survives a disconnect, which it does not today.
- Claims, so public play survives strangers and a region has an owner that sharding can follow (the owner's §7 words).
- Trees that grow back, so a crowded spawn is not stripped for good.
- Death piles other players can find.
- One clock for everyone, derived from the tick at zero bandwidth: the same principle as "the server never sends terrain".

It replaces activity by proximity alone, edits without permission, and the lowest id as the simulation centre. The cost: identity is a product decision the project has so far avoided, camera-relative rendering becomes necessary, and there is the least new to *do* for someone playing alone. Roughly 18–21 blocks.

**Overlap, as an observation and not a recommendation:**
- C1 appears in Shapes I and III, and C8 in Shapes II and III.
- E1 is needed by both Shape I and C1.
- C2 is the smallest admissible step (2–3 blocks) and deepens the furnace whatever shape is chosen.

---

## 5. Considered, and not proposed on their own

- **Hostile mobs.** Deferred by the owner on 2026-09-05. On their own they only add a threat. With C6's darkness they would have answers to (a), (b) and (c): darkness is where they spawn, and hecs and shard ownership of entities are the systems they deepen.
- **A chest on its own.** Only adds. It belongs to C3, through E1.
- **Axes and shovels.** Speed per kind of material is a data change to S12. Its (b) is thin: specialisation adds friction without making anything new possible. Tuning, not a feature.
- **Biomes as decoration.** Only adds, unless they are C4's or C5's column-material function, which the far terrain then shows.
- **A recipe book.** §3.1 already allows it as a later convenience. It is not a system.
- **Engine issues #42, #32/#33/#28, #130 and #131:** see the end of §3.

---

## 6. De vijf vragen die ik eerst zou stellen

1. Hoort automatisering (items die zelf tussen ovens en kisten gaan) bij Cubara, of blijft Factorio alleen de maatstaf voor de techniek?
2. Moet diepte iets betekenen, met hardere steenlagen en duisternis, of blijft de ondergrond overal hetzelfde?
3. Is fase 3 voor spelen met vrienden, of moet een publieke wereld met vreemden werken?
4. Moet doodgaan iets kosten, of houd je je spullen zoals nu?
5. Mag een feature FPS kosten op de M3 als hij het waard is, of moet alles binnen het huidige budget passen?

Every candidate in §2 has further owner questions. These five come first because their answers choose between the shapes in §4.

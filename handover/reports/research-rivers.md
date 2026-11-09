# Rivers in an infinite, deterministic, chunk-local world: research report

Date 2026-11-09. Verified = I read the page this session. Unverified = from memory or search snippet only.

## The core problem, stated by others

River flow at a point depends on its whole upstream catchment and its downstream route. A pure function of (seed, x, y, z) cannot see either. Three write-ups say this:

- Red Blob Games (mapgen4 rivers page): on an infinite map the generator must "limit river length", let rivers ignore downhill flow, or generate rivers before elevation. https://www.redblobgames.com/maps/rivers/ (verified)
- alcatrazEscapee's voxel-game gist sorts methods into context-free, bounded and unbounded. Of tracing to the nearest ocean it says it "will not work". https://gist.github.com/alcatrazEscapee/8bd54572c8d3fe070d1ed92704dd6aed (verified)
- The Vintage Story forum thread says the same. A tiny local basin can swallow a "massive multi-tributary river", so you need "continent-sized regions". https://www.vintagestory.at/forums/topic/20345-brainstorming-river-worldgen/ (verified)

Every working solution therefore does one of two things. Either it makes the river network a cheap, cached, coarse object, bounded in space, and derives terrain from it. Or it bakes the downhill guarantee into the terrain function itself.

## Approach 1: river-first, terrain follows (Génevaux et al. 2013 and descendants)

Génevaux, Galin, Guérin, Peytavie, Beneš, "Terrain generation using procedural models based on hydrology", ACM TOG 32(4), SIGGRAPH 2013. https://www.cs.purdue.edu/homes/bbenes/papers/Genevaux13ToG.pdf and https://hal.archives-ouvertes.fr/hal-01339224 (abstract verified; I could not extract the PDF body, so the details below are unverified).

- **How it works.** Rivers form a graph that partitions the terrain into patches. The paper builds watersheds and classifies river types and paths from it. Terrain comes from procedural primitives, blended with carving operators, and is analytic and continuous, so it supports LOD.
- **Mouths and tributaries.** From memory, the graph is grown from the coast inland. Nodes are added with slope constraints so elevation rises monotonically upstream. Branches are added by sampling candidate nodes, and Strahler order sets river class.
- **Open-source reimplementation.** `terrainHydrology` says it generates "the hydrological network first". Inputs are a shoreline mask plus river-slope and terrain-slope maps. The terrain is then built from primitives. It has no lakes or inland seas, and it is offline Python plus native code. https://github.com/wattzhikang/terrainHydrology (verified)
- **Cost, determinism, locality.** The output is deterministic. But it generates the whole graph for a bounded map in one pass, so it is global. For an infinite world you would have to run it per region (see approach 2 or the plate-local method below). Once the graph exists, evaluating terrain at a point is local: find nearby river segments with a spatial hash and blend.
- **Pitfalls.** The paper does not use river profiles as sculpting parameters; Peytavie 2019 added that. https://www.cs.purdue.edu/homes/bbenes/papers/Peytavie19CGF.pdf (snippet only). Two Red Blob Games experiments show the hard part: deriving elevation from a river tree was "purely ad-hoc" and he abandoned it. The ridges between basins need compatible heights on both sides. https://www.redblobgames.com/x/1724-elevation-from-rivers (verified) He also concluded that unconstrained random flow assignment gives rivers that loop or miss the sea. His breadth-first search inward from the border with noisy visit order is guaranteed acyclic and reaches the exterior. https://redblobgames.com/x/1723-procedural-river-growing (verified) That is the cleanest known way to get "always reaches the sea" on a graph.

## Approach 2: regional drainage on a coarse grid

- **Shape.** Take a coarse height field (for example 64–256 m cells). Fill depressions with priority-flood or Planchon-Darboux, assign each cell a single downhill receiver, and accumulate area in topological order. Cache the result per region.
- **Veloren as proof.** `erosion.rs` runs `fill_sinks` (Planchon-Darboux), `get_lakes` (lakes connected through their "pass" points), multi-receiver drainage, and a stream-power erosion solve (m=0.4, n=1). Rivers get width and depth from Gauckler-Manning-Strickler and a river-width paper. Width is capped, and wider rivers are made deeper. https://docs.veloren.net/src/veloren_world/sim/erosion.rs.html (verified, first 100k of 140k chars). Veloren does this over a finite world map pregenerated at startup, which is not chunk-local.
- **Seamlessness.** The honest answer is that there is no free seamlessness. Flow accumulation depends on the entire upstream basin and on the depression-fill spill points, which can propagate arbitrarily far. Bounding approaches:
  1. Make regions bigger than any basin that matters, and force the terrain to drain to a region edge or coast. This is the plate-local method below.
  2. Overlap margins. These only give approximate seams, because spill points and accumulation are non-local. I found no source claiming exact seamlessness (unverified, from my own reasoning).
  3. Rely on the terrain construction so that each region's boundary heights are fixed by a coarser, parent-level function. Then the child can only be wrong in detail, never in direction.
- **Cost.** Priority-flood is O(n log n), or near-linear with the Barnes 2014 variant (unverified). At 1 km cells, a 512x512-cell region is about 262k cells, so tens of milliseconds, cached per region.
- **Pitfalls.** The result is deterministic but only if floating-point and iteration order are fixed. Tie-breaking needs a hash. Small lakes at region seams are the usual defect.

## Approach 2b: plate-local river generation (the closest published fit)

- **Source.** Davis, "Plate-Local River Generation: Deterministic, Terrain-Aware, Downhill-Flowing Rivers for Infinite Procedural Worlds." It was posted to r/proceduralgeneration as "Locally Generated Procedural Rivers That Always Flow Downhill".
  - Repo: https://github.com/Davismk3/plate-local-river-generation (verified)
  - Reddit snapshot: https://reddit.sentinel-team.org/posts/1tw3bkg/snapshots/2026-06-04T17%3A41%3A06.368323Z (verified)
  - Summary: https://gamedev.net/news/6145-plate-local-river-generation-deterministic-terrain-aware-downhill-flowing/ (snippet only; the page returned 403)
  - arXiv ID and date: unverified.
- **How it works.** Plate seeds define the terrain fields and a bounded domain per plate. Rivers are traced downhill on a low-resolution grid inside that domain, from highlands to a coast, a minimum or a confluence. The segments are adjusted so stored heights always descend. The network is cached per plate on first use. At full resolution the engine takes the exact distance to the cached network and blends river height into the terrain on demand.
- **Seam guarantee.** Plate borders are noise-warped. Rivers stay inside straight-edged domains, and the "river border margin" exceeds the maximum warp displacement. So each channel lies on its own plate and the final terrain never rises along it. Example tests check this.
- **Costs.** Each viewer builds the 3x3 plates around the view. No timings are given.
- **Limits.** The repo does not discuss river width or lakes. One commenter asked "Are your lakes always one block deep?" and got no recorded answer. Reddit: https://reddit.com/r/proceduralgeneration/comments/1tw3bkg/locally_generated_procedural_rivers_that_always/
- **Caveat.** This is a single-author, recent, 14-upvote project. Treat it as an existence proof, not an authority.

## Approach 3: what the voxel-game mods do

- **Vanilla (verified via the gist).** Pre-1.18 rivers come from a cellular automaton; 1.18 and later use noise. Both lack direction, form cycles, have no sources or drains, and sit at sea level.
- **TerraFirmaCraft.** TFC 2 used finite hexagonal islands. "Streams" allows one river per 16x16-chunk region. TFC-TNG partitions into watersheds and generates rivers at very large scale, then refines them by midpoint bisection. All verified via the gist, which also notes that height cannot be included at that scale.
- **TerraForged / ReTerraForged.** I could not find documentation or source of the river algorithm. From memory it builds branching river networks per large tile and carves them into the heightmap, with performance complaints; this is unverified. A 1.21 port's blurb claims "branching river networks winding through valleys", with lakes and waterfalls. (link removed) (listing only). The mod uses tiled caching and has known complaints about being demanding, but I have no source for specific failure modes such as abrupt endings.
- **Tectonic and Terralith.** These are density-function datapacks. Rivers are noise-derived (separate river, hill and "weirdness" functions), so they are context-free. Tectonic carves rivers underground through mountains that are too tall, and the tunnels link to surface rivers. https://modrinth.com/mod/tectonic (changelog snippet). I found no evidence they guarantee reaching the sea.
- **Vintage Story "Rivers" mod.** This is the best-documented node-graph design. Rivers are built from 8–20 nodes in 3 segments, with fork angles, a `downhillError` and a noise distortion. Region size is 32768 blocks, 1024 chunks. It stores flow vectors and bank distances per column, only where a river is in range. Oceans are required, and lakes form at river ends (`lakeChance`). Users report "straight rivers, steep banks, high RAM use". https://mods.vintagestory.at/rivers (verified, first 100k chars)

## Approach 4: indie dev reports

- **Per-chunk drainage graph (rivergen).** Build a graph per chunk, search for a steep path, and carve later. Pitfalls it lists:
  - local minima ("cheese holes");
  - the discoverability paradox, where a river is missing when you come back;
  - memory;
  - generation reach beyond the player.
  - It states "Rivers must be constructed in their entirety". https://github.com/monoamin/rivergen (verified)
- **Vintage Story forum.** Chunk-by-chunk loading means a river's source or mouth may be unloaded. Flowing water simulation is "a serious hog". Steep rivers are easier to simulate than long, gentle ones. https://www.vintagestory.at/forums/topic/21825-river-generation-method/ (verified)
- **Tile erosion.** Eroding each tile separately leaves seams, because water and sediment crossing a boundary are not tracked. https://3dworldgen.blogspot.com/2017/12/terrain-erosion.html (snippet only)
- **Reddit, r/VoxelGameDev.** Chrome tools failed (the connection was lost), so I read no Reddit threads there. The one r/proceduralgeneration thread above came through a mirror. Gap: community opinions beyond these are thin.

## Approach 5: games with settlements

- **Red Blob mapgen2/mapgen4.** mapgen2 picked random source points and followed the corners with lowest elevation. mapgen4 computes rainfall everywhere and draws only the larger rivers; each node has one outflow, so rivers merge but never split. Rivers are binary trees on a dual mesh. https://www.redblobgames.com/maps/rivers/ and https://www.redblobgames.com/maps/mapgen4/ (verified). The rivers-then-elevation idea there is explicitly unfinished.
- **Dwarf Fortress.** I could not find a description of its river algorithm; it simulates erosion and drainage as a separate step, from a secondhand Reddit claim. Unverified. Community consensus (Vintage Story thread above) is that it is too heavy for just-in-time generation.
- **Settlement placement.** I found no source that documents how a game places settlements on rivers. What follows is my own design reasoning, unverified. Score sites by flow accumulation at confluences, river mouths, lake outlets and the lowest crossing on a river. Roads follow the valley floor along the river graph. Passes are the saddle between two opposing headwater basins, which is exactly the divide edge between two adjacent drainage trees.

## Practical details (mostly general hydrology, unverified by URL)

- **Width and depth.** Hydraulic geometry gives width roughly proportional to Q^0.5 and depth to Q^0.4, with Q proportional to catchment area. Veloren caps width and deepens beyond it (verified). A 1 m block grid needs a minimum river width of about 3 blocks for readability.
- **Meanders.** Add low-frequency noise displacement perpendicular to the path, scaled by width and inversely by slope. Do it after the graph is fixed, so topology never changes.
- **Waterfalls.** Where the descent per segment exceeds a threshold, stop smoothing and make a step. Tectonic and Vintage Story mod pages list them as features; there is no algorithm in either source.
- **Lakes.** Treat a lake as a graph node whose surface height equals its spill height. The outflow starts at the lowest rim point. Flood-fill lakes are the weak spot, as the Reddit commenter said: overflow can start a new river.
- **Valleys.** A carve profile with width proportional to river width gives valleys and passes for free once terrain is derived from river height.
- **Visibility at range.** The river is a vector network. It can be drawn in the far-LOD pass from the coarse graph, independent of block resolution.

## Fits Cubara's constraints best (ranking)

1. **Hierarchical river-first network on a large coarse region grid, with terrain derived from it** (approaches 1 + 2b combined, my synthesis).
   - Fix continent and coast fields from seed. Per large region (hundreds of km), build a river tree breadth-first inward from coast and lake outlets, with hash-noised visit order (Red Blob 1723). Cache it.
   - Make river heights strictly descending toward the mouth. Evaluate terrain as the carved blend of a base field and river height. A bounded margin, as in plate-local generation, makes the guarantee hold across seams.
   - Strengths: downhill and sea-reaching hold by construction, it is deterministic and cacheable, it is cheap per block (spatial hash of nearby segments), and it gives vector data for far LOD, settlements and roads.
   - Risks: the ridge-elevation problem (Red Blob 1724), region seams, and lakes.
2. **Coarse-grid priority-flood drainage per region (Veloren-style)**, used for hydrology on a coarse heightfield, with the large-region and boundary caveats above. It gives realistic dendritic networks and lakes. Seams are the cost, so it needs approach 1's fixed-boundary trick.
3. **Per-chunk drainage graph or noise rivers** (rivergen, Tectonic, Terralith). Cheap, but it cannot guarantee downhill flow or sea-reaching. This is exactly the failure the owner described, and the gist above rejects it.

Not recommended: any unbounded trace-to-ocean search (rejected by the gist), and full erosion simulation over the world (too heavy for just-in-time generation, per the Vintage Story thread).

## Unverified or open

- Génevaux PDF body, TerraForged internals, Dwarf Fortress river algorithm and settlement placement use: unverified.
- No r/VoxelGameDev threads were read (Chrome connection failed).
- The plate-local paper's arXiv ID and date are unconfirmed.
- Open design question: how to get coherent mountain ridges between opposing basins with river-first elevation. This is the hardest unsolved piece in the sources.

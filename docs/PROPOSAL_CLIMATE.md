# Proposal: climate, biomes and seasons

**Status:** proposal. The owner set the direction in a design conversation on
2026-11-09; his words are quoted in §0 and are decisions. Everything else here is
engineering's proposal for how to build it, and §6 lists what is still his to
decide. Nothing in this document is a scheduled block yet. It becomes blocks
when it is planned as a phase, and an issue per block (`docs/ISSUE_STANDARD.md`)
carries the details.

Research behind it: what players ask for in the genre, and how Vintage Story,
TerraFirmaCraft, Valheim, Don't Starve and the seasons mods do it. It was
gathered the same day and is summarised where it changed a choice (marked
*research*).

## §0 What the owner asked for and decided (2026-11-09)

- **Seasons that change the biomes, and altitude as climate.** *"seizoenen en
  dus biomes die met de seizoenen mee wisselen … hoogte, met uitgestrekte
  hoogvlaktes met bergmeren enzo, wellicht zelfs boven een bomengrens. echt
  realistisch groot terrein, waar je echt merkbaar kan afdalen naar de
  zeespiegel"*
- **Time.** *"een dag 40 min tot een uur … met een kwart van die tijd echt
  donker. dan is een seizoen minimaal 3 uur"*. He was unsure whether longer is
  still fun once crops are planted per season.
- **No temperature mechanic for the player.** *"geen kou en warmte effecten. dat
  lijkt me voor de speler niet nodig, maar effect op de omgeving kan wel leuk
  zijn"*, and later: *"de klim zelf [is] ook wel een uitdaging"*.
- **Two climate axes plus height.** *"de dubbele as van warmte/kou en
  droogte/nat … als een soort willekeurige mesh over de wereld … samen met de
  hoogte … heb je dan de biome en de soorten planten en terrein"*.
- **Water and seas.** *"water moet altijd redelijk in de buurt zijn, maar de zee
  op zeeniveau mag best een uur lopen zijn … zeeën ook groot, zodat het echt als
  een reis kan voelen"*. Boats: *"boten komen ooit"*.
- **No hemispheres.** *"halfronden zie ik niet echt zitten op een oneindige
  wereld … een soort funky mesh … van tropen, woestijnen en koude polen en alles
  ertussen in"*.
- **Biomes as a caricature.** *"een overdreven karikatuur versie van de huidige
  wereld biomes. met heftige vulkanen en bergen, mooie eilanden, grote bossen en
  bomen … geleidelijk van boomsoort wisselen naarmate je door het land gaat"*.
- **Variation between years.** *"variatie in de seizoenen … hoeveel sneeuw er
  komt en hoe ver in de winter … 1 x in de 10-20 jaar sneeuw in de woestijn"*.
- **Rivers may end in lakes; consistency over ambition; the look is Switzerland.**
  *"prima als het bij een meer uitkomt. een bergmeer is ook leuk. ik heb liever
  dat het consistent goed is, dan dat je vage fouten hebt en te erg probeert. het
  moet er vooral echt adembenemend uitzien. denk aan zwitserland enzo"*.
- **The last open questions, answered the same day:** water flows, the classic voxel game
  style (*"c"*). Climbing aids come later and peaks stay hard (*"een top is in
  het echt ook niet makkelijk bereikbaar"*). We make the textures, with Gemini.
  The current world may go. Caves may be under water and open under water, and
  the underground is a later, larger phase. This proposal is **phase 3**
  (`ROADMAP.md`).
- **Night.** *"schaduwen en contouren zijn gaaf, maar het moet niet welkom voelen
  om naar buiten te gaan"*.
- **Later, not now:** shadows (*"schaduwen komen op een gegeven moment wel"*),
  and **game rules per world** (*"een gamerule setting, waarbij je dit soort
  shit kan bijhouden en aanpassen per wereld"*).

## §1 The admission rule (`ROADMAP.md`, phase 3)

- **(a) What it deepens.**
  - The worldgen, which gets climate fields and real scale.
  - The vertical world (#175), where altitude becomes climate.
  - The far view (`PROPOSAL_FAR_VIEW.md`): snow caps, tree lines and seasons
    you watch change from 100 km away.
  - Trees, which get species and growth seasons.
  - `Lighting::time_of_day` and `/time`, which today only say "no time of day
    yet".
- **(b) What it makes possible.** Navigation and planning by climate and
  season. Journeys that change the world around you. A year with a rhythm, and
  years that differ from each other.
- **(c) What it replaces or removes.**
  - The single global climate: today it is grass, soil and stone everywhere,
    lit at a fixed noon.
  - The far terrain's hard-coded material triple, which becomes one
    column-material function shared by worldgen and the far terrain (Rule 5).
  - The fixed `sun_dir` override constants.

## §2 The model

Everything here is a **pure function of the seed, the position and the tick**,
like the terrain already is. That is the one engineering decision that makes
the rest cheap: deterministic (Rule 1), nothing to save, nothing to send, and no
catch-up for chunks nobody visited.

### 2.1 Climate fields

- **Temperature** `T(x, z)` and **humidity** `H(x, z)` are two low-frequency,
  domain-warped noise fields. The result is organic, coast-like zone shapes
  with gradual borders: the owner's "funky mesh".
- **Zones are large:** tens to hundreds of kilometres, so a change of climate
  is a journey. *Research: small, scattered biomes were the top complaint
  after the classic voxel game's 1.18 update.*
- **A "pole" is a cold region, not a place on the map.** An infinite world has
  many, as it has many deserts.
- **Altitude cools at the real lapse rate,** about 0.65 °C per 100 blocks, so a
  plateau at 2,000–2,500 blocks sits near or above the tree line. *Research:
  Vintage Story's 1.5 °C per 10 blocks suits a 256-block world, not one with no
  height limit.*
- **The tree line follows from temperature,** around the 10 °C warmest-month
  isotherm. Plateaus above it appear wherever the terrain is high enough.
- **Rain shadow (proposed):** drier on the lee side of a range. This is a cheap
  approximation from the terrain gradient along a prevailing wind, and gives
  dry high plateaus a reason to exist.

### 2.2 Continents, seas and inland water

- A **continentalness** field gives large seas with long, gentle descents to
  sea level. An hour's walk to the coast is allowed; the sea itself is a
  crossing.
- **Inland water is never far:**
  - **Lakes** in terrain basins, each with a fixed water level per basin. That
    is cheap and local, and includes mountain lakes on plateaus.
  - **Rivers**, first-class: see §2.2a.
- **Spawn is always temperate and near water,** so nobody starts in a desert or
  on an ice field. *Research: harshness falls hardest on newcomers and late
  joiners.*

### 2.2a Rivers: the backbone of the geography

The owner, 2026-11-09: *"rivieren lijken me wel heel mooi … de basis van
aardrijkskunde en beschaving. dan kan je villagers en routes (die later komen)
langs de rivieren hebben. ook kunnen die routes door bergen … en je kan ze dus
volgen om altijd bij de zee of een stad … uit te komen"*.

So rivers come before climate in the order (§5). The requirements:
- A river always flows downhill and **always reaches the sea or a lake**.
  Following one is a reliable way out.
- Rivers cut valleys, and two valleys meeting from either side of a range
  leave a **pass**: the route over the mountains.
- Rivers are visible from far away.
- All of it stays a pure function of seed and position, evaluated chunk by
  chunk in any order.

**Why the obvious ways fail.** *Research:*
- Rivers drawn by thresholding noise run uphill and form loops.
- Tracing a river from a chunk to the sea is unbounded in an infinite world, and
  "will not work".
- Deriving the *whole* terrain from a river tree failed at Red Blob Games: the
  ridges between neighbouring basins need heights that agree, and the tree
  alone does not provide them.
- Vintage Story's rivers mod (a node graph per 32 km region) works, but
  players complain of straight rivers, steep banks and memory use.

**The rule that makes chunks join exactly: every river decision is local.**

The owner asked whether this can be generated per chunk without seams and
without heavy precomputation. The answer is yes, but only if no river decision
depends on the *order* in which anything is computed. A network "grown from the
coast" per region does depend on order. Two neighbouring regions would then see
almost, not exactly, the same rivers in their overlap: a seam at the 50 km
scale. So:

1. **Base terrain stays as it is:** continents, ranges and plateaus from noise.
   It owns the ridges and watersheds.
2. **Drainage by strict descent on a coarse grid.**
   - Each coarse cell (a few hundred blocks) drains to the neighbour with the
     strictly lowest value of a smooth **potential** field: the large-scale
     elevation, with a hash of the cell as tie-breaker.
   - That is a pure function of the cell and its 8 neighbours. No region, no
     visit order, no seam.
   - Strictly decreasing means **no loops are possible**. Every path ends in the
     sea or in a basin (a local minimum).
3. **Basins are lakes.**
   - A basin, a local minimum of the potential, becomes a lake.
   - A lake may spill at its lowest rim cell and continue downstream, when the
     rim search finds an outlet within a bounded radius. Otherwise it is a
     terminal lake, such as a mountain lake or a lake in a dry basin.
   - The owner chose this explicitly: a river ends in the sea **or a lake**, and
     consistent correctness beats trying too hard. No rule exists to force every
     river to the sea.
4. **Levels, not one grid.**
   - Trunk rivers live on a very coarse grid. Tributaries live on finer grids
     and always drain into the coarser level's channel.
   - Each level is local. Width and depth come from the level and the position
     along the channel, so a tributary is never wider than the river it joins.
   - The density of rivers is a parameter per level.
5. **Per column, a chunk only asks:** which channel segments are near, how far
   am I from them, and how high is the bed at that point.
   - The bed is a property of the *segment*: sea level at the mouth, rising
     upstream with a minimum slope, clamped under the base terrain. Two chunks
     on opposite banks get exactly the same bed.
   - The terrain is lowered toward the bed within a valley profile around the
     channel.
   - Where the base terrain drops faster than the allowed bed slope, the river
     gets a **waterfall** instead of an unnatural cut.
6. **Meanders.** The coarse polyline is displaced by deterministic noise of the
   segment, never of the chunk. It is bounded so the channel stays in its
   valley. This cures "straight rivers".
7. **Passes** are where two headwaters from opposite sides of a ridge come
   closest: a query on the same local graph.
   - Routes and villages, later, read it too: confluences and mouths are
     natural village sites, and valleys are natural roads.

**"Cache, but never decide in the cache."** Coarse cells may be cached per
region to make worldgen fast. Every cached value must be computable identically
from any vantage point, so the cache is an accelerator, never a source of truth.
The up-front cost is small: a 64 km region at 256-block cells is ~60,000 cells
of a few noise samples each, a few milliseconds.

*Research behind this:*
- Rivers from thresholded noise run uphill and loop.
- An unbounded trace to the sea "will not work" in an infinite world.
- Deriving the whole terrain from a river tree fails, because ridges between
  basins need agreeing heights. Here the base terrain owns the ridges.
- Vintage Story's region-graph rivers mod draws complaints of straight rivers,
  steep banks and memory use.
- "Plate-Local River Generation" traces downhill per plate with a border margin.
  This proposal goes one step further: the descent rule is local, so it needs
  no margin.

**Engine consequences.**
- Water (§5 block 1) must exist first.
- Flowing *animation* is visual only: the water surface is static, and a
  shader scrolls it along the channel direction, which the graph provides.
  Real fluid simulation is not needed and not proposed.
- The far terrain reads the same river field, so a river shows as a line of
  water to the horizon.

**Checked by:**
- **Every river reaches sea level or a lake**, and the bed never rises
  downstream: a property test over many seeds and regions.
- **Seams cannot exist:**
  - a chunk computed with a cold cache, and with a cache warmed from far
    away, is bit-identical;
  - two neighbouring chunks computed independently have identical border
    columns;
  - the far terrain's river field matches the voxels' at the join.
- **Determinism:** the same seed gives the same network in any evaluation
  order and thread count.
- **Coverage:** within N blocks of any spawn there is water. The owner's
  "water always fairly near" becomes a number.
- **Performance:** one region's network costs at most X ms to build, and the
  per-column carve stays inside the worldgen budget. Both are `BENCHMARKS.md`
  rows.

### 2.2b The look: Switzerland

The owner's target is *"echt adembenemend … denk aan zwitserland"*. Alpine
landscapes are made of a few recognisable forms. Each becomes a terrain feature
with a test or a golden image, rather than a hope:

- **U-shaped valleys**: wide, flat floors between steep walls, as glaciers
  leave them (Lauterbrunnen). Main valleys get a U profile; small streams keep a
  V.
- **Hanging valleys and waterfalls**: side valleys that end high above the main
  valley floor, their stream falling over the wall. These follow from §2.2a's
  rule that a bed steeper than allowed becomes a waterfall. Lauterbrunnen's
  walls are the reference image.
- **Long lakes in valley floors**, like Lake Geneva or Lake Lucerne. In this
  design a lake is where a valley fills, so a glacial valley with a basin
  becomes a finger lake.
- **Sharp peaks and arêtes** above the tree line (the Matterhorn's horn), on top
  of today's ridged ranges.
- **Alpine meadows** between forest and rock, and the **seasonal snow line**
  (§2.5) moving over them.
- **Scale that reads:** valley floors a few hundred blocks wide, walls of
  500–1,500 blocks, peaks of 3,000–4,500. Big enough to look Alpine from the
  valley floor and from 100 km away. That is within the vertical world and the
  far view already built.

Judged from images at fixed eyes (a valley floor, a pass, a peak, a lake shore),
in the style F5's haze was judged, then pinned as golden images.

### 2.3 Biomes as a caricature

- A **biome table in data** (RON, like blocks and recipes) maps a climate range
  (T, H, altitude) to:
  - surface blocks
  - vegetation
  - tree species weights
  - terrain character
- A new biome is a data file, not code.
- **The starting set** is drawn from the Whittaker diagram and exaggerated, as
  the owner asked:
  - tundra and ice fields
  - taiga
  - giant temperate forest
  - grassland
  - savanna
  - desert
  - rainforest
  - swamp
  - alpine meadow above the tree line
  - volcanic regions
  - island chains
  The owner edits this list (§6).
- **Tree species blend with the climate** rather than switching at a border. A
  forest changes species gradually as you travel.
- **Terrain archetypes** (volcanoes, big ranges, islands) are terrain-noise
  features placed by the same fields, so a volcano region is also a climate
  region.

### 2.4 Time: day, night, calendar

- **The defaults are the owner's numbers.**
  - A day lasts 40–60 minutes, with a quarter of it dark.
  - A season lasts at least 3 hours, so roughly 4 days at 50 minutes.
  - Four seasons make a year.
- **The calendar is derived from the tick.** Every machine computes the same
  date.
- **These are game rules stored with the world**, not constants in code, so
  they can be tuned after play. The game-rules screen the owner wants later
  edits these values.
- **Night:** dark enough that going out does not feel welcoming, with moonlight
  contours so the world still reads. Light sources (torches, the furnace's
  glow) are a later block; they need light propagation.
- **The sun moves smoothly.**
  - Today the sun is a pipeline override constant, a measured necessity on the
    M3.
  - A stepped sun updated every few seconds through the existing background
    rebuild would be invisible *without* shadows.
  - Because shadows are coming, the design keeps the door open for a per-frame
    sun direction in the vertex stage, where the M3 cost is acceptable. That
    choice is measured before it is made.

### 2.5 Seasons, per climate zone

There is one global calendar, but each zone answers it in its own way:
- **Temperate:** four seasons, with snow, leaf colour and fall, and frozen
  lakes.
- **Cold:** a long winter and a short summer.
- **Tropical:** no winter. The season moves *humidity* instead, giving a wet
  and a dry season.
- **Desert:** little change.

The seasonal swing is an amplitude field, larger inland and at altitude and
smaller near the sea. The **snow line climbs and falls** with the season, which
is visible on the far terrain.

### 2.6 Years that differ

A **year anomaly** `A(x, z, year)` is noise over space *and* year, seeded from
the world seed. It shifts temperature and precipitation for a region for a
year:
- How much snow falls.
- How far the winter reaches.
- Snow in a desert, when the anomaly is extreme.

The rarity is calibrated in a test by sampling 1,000 years: snow in a desert
about once in 10–20 years, as the owner asked.

### 2.7 Derived, not stored

- **Snow cover, leaf colour and frozen water** are evaluated from (climate,
  height, tick) when meshing, in the shader, or when the sim queries them.
  They are never written as block edits.
- The **sim asks the same function** where it matters for play: ice is
  walkable, and growth happens only in season. One implementation (Rule 5).
- *Research: seasons mods that store real snow blocks cause lag, and colour
  changes need an explicit refresh of distant LODs.* Here the far terrain takes
  the season as a shader parameter, so no patch is regenerated.

## §3 What this does not do

- **No temperature effects on the player:** no clothing, no freezing (owner,
  §0).
- **No hemispheres** (owner, §0).
- **Boats, weather events (falling rain or snow), torches and light
  propagation, shadows, and the game-rules UI** are later blocks. They are
  named here so the model leaves room for them.

## §4 How it is checked

- **Pure-function tests:**
  - The climate fields are continuous: no step larger than ε between
    neighbouring columns.
  - They are deterministic across runs and thread counts.
  - The lapse rate holds.
  - Every spawn is temperate with water within N blocks, over 1,000 seeds.
  - The tree line sits at the isotherm.
  - The desert-snow frequency over 1,000 sampled years falls within 1/20 to
    1/10.
- **Golden images** of the same view in four seasons, plus a far-view image of
  a snow line, judged and then pinned.
- **`BENCHMARKS.md` rows** for every block that touches meshing or shaders.
- **A worldgen change bumps `WORLDGEN_VERSION`.** The current world does not
  load into the new generator; *New World* keeps it aside, as it already does.

## §5 Proposed order of blocks

1. **Water.** Static water, non-solid blocks and a transparent pass. Everything
   with lakes, rivers and seas rests on it.
2. **Continents and seas.** Continentalness, sea level and coasts, and terrain
   archetypes (volcanoes, ranges, islands).
3. **Rivers and lakes (§2.2a).**
   - First, the regional network as a pure, tested, GPU-free module, with the
     seam, downhill and coverage properties.
   - Then carving the valleys, waterfalls and meanders into worldgen.
   - Then the far terrain showing rivers.
   - Then the temperate, near-water spawn rule.
4. **Climate and biomes.** The T/H fields, the lapse rate, rain shadow, the
   biome table in data, surface and vegetation from climate, and tree species
   blending.
5. **Time.** Day and night, the calendar, game rules stored per world, and the
   moving sun.
6. **Seasons in the world.** Derived snow, leaf colour and ice (frozen rivers
   and lakes included). The far terrain reads the same column function.
7. **Year anomalies.**
8. **Growth seasons** for trees, and crops when the owner shapes them.
9. **Later:** routes and villages along the river graph, light sources,
   shadows, boats, weather events, and the game-rules screen.

## §6 Still the owner's to decide

- The biome list: add and remove from §2.3.
- Crops: which ones, and their seasons.
- How night darkness is tuned: judged from images.
- What winter means for gameplay beyond the visuals and growth windows: for
  example, does snow slow walking?

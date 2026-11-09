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

**The proposal: rivers first, terrain carved around them, in bounded regions.**

1. **Base terrain stays as it is:** continents, ranges and plateaus from noise.
   It owns the ridges and watersheds, which avoids the Red Blob failure.
2. **A coarse drainage network per large region.**
   - The region is something like 32–64 km, on a grid of a few hundred blocks
     per cell.
   - Over it, the network is grown **from the coast inland**: breadth-first,
     with a noisy visit order, always stepping to the neighbouring cell that
     rises.
   - The result is a tree, acyclic by construction, in which every cell drains
     to a mouth. *Research: this is how Red Blob's experiments got rivers that
     always reach the sea. "Plate-Local River Generation" uses the same idea per
     tectonic plate, with a border margin larger than the warp so seams stay
     downhill.*
   - **Regions overlap by a margin.** A region's network is computed from its
     own cells plus the margin, so neighbouring regions agree across the seam.
   - The network is **computed once per region, cached, and never saved**.
     It is pure in the seed, so every machine computes the same rivers, and
     memory stays bounded by the regions near players.
   - Basins with no outlet at this scale become **lakes**, nodes in the
     graph whose outflow continues from their lowest rim point.
3. **Flow and width from the tree.**
   - Flow accumulation is exact on a finite tree, so it is cheap.
   - Width and depth grow with catchment: a 2-block headwater stream becomes a
     mouth tens to hundreds of blocks wide.
   - Rivers below a threshold catchment are not drawn, so the density is
     tunable.
4. **Carving.**
   - The channel's bed elevation is defined *along the tree*: sea level at the
     mouth, rising upstream with a minimum slope. It is clamped under the base
     terrain, so it is monotonic by construction.
   - The terrain is lowered toward the bed within a valley profile around the
     channel, using the distance to the river line: a pure function, evaluated
     per column.
   - Where the base terrain drops faster than the allowed bed slope, the
     river gets a **waterfall** instead of an unnatural cut.
5. **Meanders and natural lines.** The coarse polyline is refined per chunk by
   deterministic noise displacement, bounded so the channel never leaves its
   valley. This is the cure for "straight rivers".
6. **Passes** are where two headwaters from opposite sides of a ridge come
   closest: the network knows both, so a pass is a query on the graph.
   Routes and villages, later, read the same graph:
   - confluences and mouths are natural village sites
   - valleys are natural roads

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
- **Seams agree:** two regions computed independently give the same channel at
  their shared border.
- **Determinism:** the same seed gives the same network in any evaluation
  order and thread count.
- **Coverage:** within N blocks of any spawn there is water. The owner's
  "water always fairly near" becomes a number.
- **Performance:** one region's network costs at most X ms to build, and the
  per-column carve stays inside the worldgen budget. Both are `BENCHMARKS.md`
  rows.

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

# Research: biomes, seasons, climate, large-scale terrain (2026-11-09)

Method: web search plus page fetches. old.reddit.com is blocked for automated fetches, so no Reddit threads were read. Player demand comes from the official feedback site, the original developer's bug tracker, the Vintage Story and Klei forums, Steam discussions and mod pages. Search results were summarized by a tool; anything I could not open myself is marked (unverified). The TFC English Field Guide pages for elevation and snow did not load fully.

## 1. What players ask for

- **Seasons are a long-standing request.** The feedback site has a long-running thread, "Seasons for most biomes", and a newer "Seasons" post. Proposals for season length range from 20 to 30-90 days. I could not open the thread (403), so vote counts are unknown.
- **Abrupt borders.** A suggestion asks for a snow-on-ground gradient at borders between snowy and non-snowy areas ("Biome blending for snow on the ground"). Another asks that plains pick up trees, snow, sand or coarse dirt from neighbours ("Biome variations depending on its neighbors"). A commenter on a 1.18 thread said the hot-to-cold gradient is "almost non-existent".
- **Biome scale after 1.18.** Reports of tiny 15x15 to 50x50 biomes, "weird" patches, rare biomes too common, and complaints that mountains took too much space. Modders explain this as structural: biomes with similar parameters compete for the same spot and become smaller and scattered.
- **Altitude should drive snow.** A mod and companion data pack exist because vanilla snow follows each biome's fixed temperature, not height, so a snowy peak can sit beside a bare jungle peak at the same altitude. The Forge 1.16.5 mod's page says "the classic voxel game defines constant climates for its biomes". It lists summer cooling with altitude, a small snow-chance increase with altitude, and a tree line as planned but not built.
- **Snow-height regression.** One thread asks for taiga snow heights to return to 1.17 levels.
- I found no verified "desert next to snow" thread specifically, and no the suggestions subreddit threads.

## 2. Games and mods

**Vintage Story.** Four seasons of 3 months, 27 days each, 108-day year. In the northern hemisphere the coldest point is mid-January and the hottest mid-July, reversed in the south. Mean temperature drops about 1.5 C per 10 blocks of elevation (wiki snippet; blocks are 1 m, so that is very steep and tuned to a small-height world). The wiki says snow depth of 2 lets mobs hop fences and stops grass harvesting. Below about -10 C crops struggle even in a greenhouse, which adds only 5 C. Complaints: a forum thread about winter arriving too early at spawn, and one observing that with 30-day months winter feels extended. Others answer: plant early, stock 50+ crocks. A calendar mod notes the calendar ignores real temperature, so a hot "winter" still reads as winter.

**TerraFirmaCraft.** Four seasons of about 3 months, split early/mid/late. Latitude (Z coordinate) is the largest temperature factor, then season and hour. Annual rainfall is 0-500 mm and feeds soil hydration, which gates crops. TFC 4.0 added wet and dry seasons and a hemisphere flip. Its changelog guarantees crops survive the 4 warmest months in any climate they spawn in, suggesting survival was softened after complaints (inference). Temperature gates crop growth and snow/ice formation.

**Serene Seasons / Fabric Seasons (voxel-game mods).** Serene Seasons has 4 seasons with early/mid/late sub-seasons. Grass and foliage colours change, temperature shifts, snow can fall where it normally does not, crops have fertile seasons shown on seed tooltips, and greenhouse glass lets you farm year-round. In tropical biomes only summer crops are in season. I found no verified complaints about season length. Performance pain is documented around snow: Season Cache exists because otherwise the world only gains or loses snow through the slow random-tick system, taking game days. Serene Seasons Plus 5.1.0 notes say snow work was reworked to avoid doing everything in one laggy tick. Fabric Seasons internals could not be verified.

**the classic voxel game's 1.18 update noise worldgen.** Five parameters (temperature, humidity, continentalness, erosion, weirdness) pick the biome. Continentalness sets general height (inland rises). Erosion sets rugged versus flat. Weirdness drives peaks-and-valleys relief. Temperature and humidity do not shape terrain directly (via mirrors of the wiki; check the original). Pre-release notes mention most biomes being made 4x larger.

**Terralith / Tectonic / Orogenesis.** Tectonic claims ranges thousands of blocks long, peaks above y300, optional height 640, and rivers carved underground where terrain is too high. Orogenesis claims mountains about 2x as high and 3x as wide as vanilla. No review data found on player satisfaction.

**Valheim.** Biomes are progression gates: Mountain needs frost resistance (armor or mead). Meadows to Plains is rated highly by a 1000-hour player. Mistlands drew complaints about fog, and a developer admitted the order felt wrong at first. Cold is a gear check, not a calendar event.

**Don't Starve.** Summer and winter are the "harsh" seasons, spring and autumn "friendly". Freezing is a recurring death for new players. Veterans call the harsh seasons boring or easy to skip. A forum poster says late joiners suffer because the world is already in an extreme season. Hamlet's year is 33 days (3 seasons x 11).

**Dwarf Fortress:** I did not research it. Unverified.

## 3. Technical approaches

- **Climate as noise vs simulation.** The classic voxel game uses noise parameters with no simulation. TFC and Vintage Story compute temperature from latitude, season and elevation and rainfall from noise. All of that is pure function of (x, z, y, tick), which fits a deterministic engine. This is my synthesis from the sources.
- **Lapse rate and tree line.** About 0.65 C per 100 m is the standard approximation, and tree line follows roughly the 10 C growing-season isotherm (Koerner and Paulsen 2004; the Slovak study cites the rate). It is not universal. Tree line also depends on more than temperature. The 10 C figure is a rule of thumb, so treat it as a tunable.
- **Long gradients to the sea.** The classic voxel game's continentalness is the model: a low-frequency field that raises terrain inland. A very low-frequency continent field plus a slow slope toward sea level gives long descents. No source gave concrete scales; choosing them is a design call.
- **Rivers and lakes.** Thresholding ridged noise gives loops and rivers that run uphill (Godotcraft note). Cheap and local: noise basins for lakes, and the known "plate-local" approach of a cached river network per bounded region with distance-to-network computed on demand (a 2025-2026 arXiv paper; only an aggregator summary was read, unverified). Needs global-ish work: true flow accumulation across a continent. The workable compromise is regional graphs with border margins.
- **Seasonal visuals without re-editing blocks.** The sources show both approaches. Serene Seasons stores real snow block changes, reconciled in batches (default 8 chunks per tick in Season Cache), which is where lag comes from. Seasonal leaf and grass colour is client-side and only needs a refresh when the season changes. For distant terrain, mods keep a per-chunk snow-coverage map for shaders. A derived approach (snow as a function of climate, height and tick at mesh or shader time) avoids block edits entirely. That is my inference, not a documented practice in these games.
- **Pitfalls:** per-tick snow replacement spikes, random-tick-only melting, and LOD terrain needing explicit season updates.

## 4. Linking seasons to other survival systems

- **Crops and food.** Seasonal crops, greenhouses and stockpiling are core to TFC, Vintage Story and Serene Seasons. TFC softened crop death. Vintage Story players accept winter prep as the game. Unreasonable harshness mostly hits newcomers.
- **Heat and clothing.** Cold Sweat is described as one of the best temperature mods and performant. Tough As Nails has a mixed "love hate" review (reason not seen). Valheim ties cold to a gear check.
- **Snow and ice.** Vintage Story's deep snow blocks harvesting and lets mobs cross fences. I found no player opinions on walkable ice or snow slowing movement (unverified).
- **Newcomer and late-joiner risk.** Don't Starve forum posts show that seasons in multiplayer punish people who join mid-season.
- **Migration:** nothing found.

## Sources

- https://the official feedback site/hc/en-us/community/posts/360009353652-Seasons-for-most-biomes
- https://the official feedback site/hc/en-us/community/posts/4407884905229-Biome-blending-for-snow-on-the-ground
- https://the official feedback site/hc/en-us/community/posts/4417305312653-Biome-variations-depending-on-its-neighbors
- https://the official feedback site/hc/en-us/community/posts/4411085932429-Lower-Snow-Heights-Back-to-1-17-Levels-For-Colder-Biomes
- https://the official feedback site/hc/en-us/community/posts/34487125753997-Seasons
- https://the official feedback site/hc/en-us/community/posts/4405400249357/comments/4405759014541 (1.18 biome comments)
- https://modrinth.com/project/k8l3M9My (altitude snow mod)
- (link removed)
- https://wiki.vintagestory.at/Snow, https://wiki.vintagestory.at/index.php/Temperature/en, https://wiki.vintagestory.at/index.php/Time
- https://www.vintagestory.at/forums/topic/13335-winter-too-early-in-my-world-way-to-fix-it-without-starting-over/
- https://www.vintagestory.at/forums/topic/14697-question-about-season-lengths/
- https://terrafirmacraft.github.io/Field-Guide/18/en_us/the_world/climate.html
- https://modrinth.com/mod/terrafirmacraft/changelog
- (link removed)
- (link removed)
- https://modrinth.com/mod/season-cache
- https://modrinth.com/mod/serene-seasons-x-distant-horizons
- (link removed) (and mirror (link removed))
- https://github.com/Apollounknowndev/tectonic, https://modrinth.com/datapack/orogenesis
- https://dontstarve.wiki.gg/wiki/Seasons
- https://forums.kleientertainment.com/forums/topic/151402-summer-and-spring-are-is-far-too-extreme-for-most-late-joiners-to-have-fun/
- https://www.gfinityesports.com/valheim/valheim-biome-order-progression-guide-areas-location-meadows-black-forest-swamp-mountain-plains
- https://gamesradar.com/valheims-mistlands-update-is-spellbindingly-pretty-and-utterly-deadly
- https://journals.savba.sk/index.php/cgg/article/view/5824 (10 C isotherm, 0.65 C/100 m)
- https://gamedev.net/news/6145-plate-local-river-generation-deterministic-terrain-aware-downhill-flowing/
- https://github.com/monoamin/rivergen
- https://git.allpurposem.at/mat/Godotcraft/commit/5cd0b0d53aaed36ff41791822d858f5b98689f4a
- https://moddex.gg/mod/cold-sweat/reviews/1887, https://mmcreviews.com/all/mods/tough-as-nails

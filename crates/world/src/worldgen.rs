//! Seeded terrain + cave generation (`docs/PHASE1_ARCHITECTURE.md` §8).
//!
//! [`WorldGen`] replaces the old fixed rolling-hills formula with a seeded
//! noise height field, plus a second 3D noise field subtracted from it for
//! caves (§8.3). Every sample [`WorldGen::generate`] produces is a pure
//! function of `(seed, x, y, z)` alone -- no neighbouring chunk's state, no
//! generation order, no shared scratch (§8.1) -- which is what lets an
//! unedited chunk be regenerated instead of saved (§7.4) and a far LOD node
//! sample its own volume with no neighbours to ask (§6).

use cubara_voxel::{AnisoChunk, BlockId, BlockRegistry, Chunk, OreRegistry, StructureRegistry};

use crate::noise::{fbm2, fbm3, ridged2};

/// The three block ids unedited terrain is layered with by depth below the
/// surface -- resolved by the caller from its own loaded registry by name
/// (e.g. `registry.id_of("cubara:grass")`) rather than assumed as fixed
/// numbers. `BlockId`s are assigned by sorted name per registry (§3.4), so a
/// hardcoded id would silently mean a different material the moment the
/// loaded registry's material set changed -- exactly the bug block 1.4b
/// found and fixed in the old placeholder terrain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TerrainBlocks {
    /// The exposed surface block of unedited terrain.
    pub grass: BlockId,
    /// The `SOIL_DEPTH` blocks directly beneath the surface.
    pub soil: BlockId,
    /// Everything deeper than that (including cave walls, which sit well
    /// past `SOIL_DEPTH`), and every player-placed block -- edits don't
    /// carry a material choice yet, since there's no inventory/build system
    /// to choose one from.
    pub stone: BlockId,
    /// The oak's trunk and canopy, and the shape it grows in. `None` when no
    /// structure data was supplied -- a world with no trees, which every test
    /// that predates block 2.3 still expects.
    pub oak: Option<Oak>,
    /// The ores that replace deep material, checked in slot order with the
    /// first match winning. Empty when no ore data was supplied -- a world of
    /// plain stone, which every test that predates block 2.3b still expects.
    pub ores: OreSet,
}

/// The ores a world generates, as a fixed-size set.
///
/// **Fixed-size rather than a `Vec` because [`TerrainBlocks`] is `Copy`** and
/// is passed by value through every per-voxel path in this file; a heap
/// allocation there would be a per-voxel cost for a list that never has more
/// than a handful of entries.
///
/// Four slots rather than one deliberately: a single `Option<Iron>` would be
/// smaller, but it would make a second ore a code change instead of a data
/// file, and `REQUIREMENTS.md` #3 asks for the opposite.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct OreSet {
    ores: [Option<OreGen>; MAX_ORES],
}

/// How many distinct ores one world can generate. Raising it costs a few
/// bytes in a `Copy` struct and nothing else.
pub const MAX_ORES: usize = 4;

impl OreSet {
    /// The ore-free set: plain stone everywhere.
    pub const EMPTY: Self = Self {
        ores: [None; MAX_ORES],
    };

    /// Append an ore, ignoring it once the set is full.
    ///
    /// Silently ignoring rather than panicking, for the same reason
    /// [`TerrainBlocks::with_oak`] tolerates missing data: a world missing its
    /// fifth ore is a worse world, not a broken one, and a data file should
    /// not be able to crash the game at startup.
    fn push(&mut self, ore: OreGen) {
        if let Some(slot) = self.ores.iter_mut().find(|s| s.is_none()) {
            *slot = Some(ore);
        }
    }

    fn iter(&self) -> impl Iterator<Item = &OreGen> {
        self.ores.iter().flatten()
    }

    pub fn is_empty(&self) -> bool {
        self.ores.iter().all(|s| s.is_none())
    }
}

/// One ore, resolved: ids instead of names, and the tuning numbers from
/// `assets/ores/*.ron` converted from the integers the file stores.
///
/// Ore is a *material* choice, never a density one -- it replaces a solid
/// block with a different solid block. That is what keeps it out of the
/// three-path problem trees have (`PHASE2_ARCHITECTURE.md` §6): solidity is
/// identical with and without ore, so `is_solid` need not know it exists, and
/// `generate` (LOD) can include it safely.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OreGen {
    pub block: BlockId,
    /// Only this material becomes ore. Deep material only, so ore can never
    /// surface and never hollows out a trunk.
    pub replaces: BlockId,
    /// The highest `y` this ore appears at.
    pub max_y: i32,
    /// Noise above this becomes ore, in **thousandths**, exactly as the data
    /// file stores it.
    ///
    /// Kept as the integer rather than converted at load, for two reasons that
    /// point the same way: it keeps this struct (and so [`TerrainBlocks`])
    /// `Eq`, which an `f32` field would forbid; and it puts the single
    /// IEEE-defined conversion at the one place the comparison happens, so
    /// there is no rounding step between the file and the decision. §8.5 wants
    /// that decision bit-identical across platforms.
    threshold_milli: i32,
    /// Noise frequency per 1000 blocks. See
    /// [`threshold_milli`](Self::threshold_milli).
    freq_milli: i32,
    /// Keeps two ores from generating in exactly the same places. Derived
    /// from the ore's name, so it is stable across runs and platforms.
    seed_mix: u64,
}

/// The oak, resolved: ids instead of names, and the shape numbers from
/// `assets/structures/oak.ron`.
///
/// Shape lives in data (`REQUIREMENTS.md` #3); *placement* does not, and is
/// the structure pass below (`PHASE1_ARCHITECTURE.md` §8.4) -- an algorithm
/// with a declared radius, not a number to tune.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Oak {
    pub log: BlockId,
    pub leaves: BlockId,
    /// Only grows on this block, which keeps trees off stone and out of caves.
    pub grows_on: BlockId,
    /// Inclusive trunk height range.
    pub height: (i32, i32),
    /// How far the canopy reaches from the trunk, in blocks.
    pub canopy_radius: i32,
    /// One in `density` chunk columns carries a tree. An integer so the
    /// decision is bit-identical across platforms (§8.5).
    pub density: u32,
}

impl TerrainBlocks {
    /// Add the oak, resolved from `structures` and `registry` by name.
    ///
    /// A builder rather than a parameter on [`from_registry`](Self::from_registry)
    /// so that the many callers who do not care about trees -- tests pinning
    /// terrain shape, save round-trips, the determinism fixture -- keep a
    /// one-argument constructor and a treeless world. The callers that render
    /// or play the real world opt in.
    ///
    /// Returns `self` unchanged if the structure or any of its blocks are
    /// missing: a world with no oaks is a worse world, not a broken one, and
    /// panicking here would make a missing data file fatal at startup for
    /// something the game can do without.
    pub fn with_oak(mut self, structures: &StructureRegistry, registry: &BlockRegistry) -> Self {
        let Some(def) = structures.get("cubara:oak") else {
            return self;
        };
        let (Some(log), Some(leaves), Some(grows_on)) = (
            registry.id_of(&def.trunk.block),
            registry.id_of(&def.canopy.block),
            registry.id_of(&def.grows_on),
        ) else {
            // No logging: this crate has no logger dependency, and a
            // missing block is caught by the registry tests long before here.
            return self;
        };
        self.oak = Some(Oak {
            log,
            leaves,
            grows_on,
            height: def.trunk.height,
            canopy_radius: def.canopy.radius,
            density: def.density,
        });
        self
    }

    /// Add every ore in `ores`, resolved from `registry` by name.
    ///
    /// A builder rather than a parameter on [`from_registry`](Self::from_registry)
    /// for exactly the reason [`with_oak`](Self::with_oak) is one: the many
    /// callers that pin terrain shape, round-trip a save, or seed the
    /// determinism fixture keep a one-argument constructor and an ore-free
    /// world, so this change cannot move a single existing expectation. Only
    /// the callers that render or play the real world opt in.
    ///
    /// Ores are taken in **name order** ([`OreRegistry::sorted`]) and checked
    /// in that order, first match winning, so which ore wins an overlap is a
    /// property of the data and not of hash iteration order (Rule 1).
    ///
    /// An ore whose blocks are missing from `registry` is skipped, not fatal:
    /// same reasoning as `with_oak`.
    pub fn with_ores(mut self, ores: &OreRegistry, registry: &BlockRegistry) -> Self {
        for def in ores.sorted() {
            let (Some(block), Some(replaces)) =
                (registry.id_of(&def.name), registry.id_of(&def.replaces))
            else {
                continue;
            };
            self.ores.push(OreGen {
                block,
                replaces,
                max_y: def.max_y,
                threshold_milli: def.threshold,
                freq_milli: def.freq,
                seed_mix: ore_seed_mix(&def.name),
            });
        }
        self
    }

    /// Resolve `cubara:grass`/`cubara:soil`/`cubara:stone` from `registry` by
    /// name -- the one place every real caller (the live renderer, the
    /// headless bench/screenshot paths) gets its `TerrainBlocks` from, so the
    /// "resolve by name, not a hardcoded id" rule lives in one spot instead
    /// of being repeated at each call site.
    pub fn from_registry(registry: &BlockRegistry) -> Self {
        let id = |name: &str| {
            registry
                .id_of(name)
                .unwrap_or_else(|| panic!("assets/blocks must define {name}"))
        };
        Self {
            oak: None,
            ores: OreSet::EMPTY,
            grass: id("cubara:grass"),
            soil: id("cubara:soil"),
            stone: id("cubara:stone"),
        }
    }
}

/// What a column of unedited ground is made of at one depth below **its own**
/// surface: the one answer every level of detail reads ([`WorldGen::ground_layer`];
/// issue #423 decision 2, `docs/PHASE3_ARCHITECTURE.md` "Render consistency").
/// The discriminants index [`GroundFaces::sides`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroundLayer {
    /// The surface block (grass): its top on a top face, its side on a riser.
    Surface = 0,
    /// The [`SOIL_DEPTH`] blocks under it.
    Soil = 1,
    /// Everything deeper.
    Stone = 2,
}

/// The four ways a riser can face, as `(dx, dz)`: the order of
/// [`GroundFaces::sides`] and [`GroundFaces::side_open`].
pub const GROUND_DIRECTIONS: [[i32; 2]; 4] = [[1, 0], [-1, 0], [0, 1], [0, -1]];

/// What level 0 shows over a square of ground, from its columns alone
/// ([`WorldGen::ground_faces`], over [`WorldGen::column`]): the faces a
/// coarser representation of that square stands for, so that it can show
/// the same mix.
///
/// A pit is in it -- its floor is its column's top, and its walls are the
/// risers of the columns around it, stone below the soil -- since every
/// level draws one (`docs/PHASE3_ARCHITECTURE.md` §9.9). A cave under an
/// intact top is not: these are the faces of what the sky sees, which is
/// what a level too coarse to draw a cave's inside can know.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct GroundFaces {
    /// Columns counted: one top face each.
    pub columns: u32,
    /// The riser area level 0 draws, in block faces, per direction
    /// ([`GROUND_DIRECTIONS`]) and per [`GroundLayer`]: a column standing
    /// `k` blocks above its neighbour shows that neighbour one surface-block
    /// side, then up to [`SOIL_DEPTH`] of soil, then stone -- each row by
    /// its depth below the column's own surface, so the wall of a pit under
    /// a floor column is stone from its first row.
    pub sides: [[f32; 3]; 4],
    /// How open level 0's top faces are to the sky, 0 to 1: the mean over
    /// their corners of the baked ambient occlusion level
    /// (`cubara_voxel::face_ao_of`) divided by 3. A renderer lights a face
    /// `ao_floor + (1 - ao_floor) * open` of the way (`mesh.wgsl`).
    pub top_open: f32,
    /// The same for each direction's risers, weighted by area. 1 where a
    /// direction has none.
    pub side_open: [f32; 4],
}

impl GroundFaces {
    /// The share of direction `d`'s risers that is `layer`, or `None` where
    /// level 0 has no riser facing that way.
    pub fn share(&self, d: usize, layer: GroundLayer) -> Option<f32> {
        let area: f32 = self.sides[d].iter().sum();
        (area > 0.0).then(|| self.sides[d][layer as usize] / area)
    }
}

/// How many blocks of soil separate the grass surface from stone beneath --
/// a simple, fixed depth rule (carried over from block 1.4c). Real layered
/// biomes with varying depth are out of phase 1's scope (§8's own "out of
/// scope" list). Public because the far terrain's colours follow it: a tall
/// riser shows soil, then stone (`cubara_render::aggregate::riser_colour`).
pub const SOIL_DEPTH: i32 = 3;

// Terrain shape. Frequency is in noise-cycles-per-block; a smaller number is
// broader, gentler hills. Tuned by eye against the old rolling-hills formula
// (`terrain_height`, ~±11 blocks of relief over a ~20-30 block wavelength)
// so the visible scene doesn't change wildly from block 1.4's golden images.
const TERRAIN_FREQ: f32 = 0.02;
const TERRAIN_OCTAVES: u32 = 4;
const TERRAIN_LACUNARITY: f32 = 2.0;
const TERRAIN_GAIN: f32 = 0.5;
const TERRAIN_AMPLITUDE: f32 = 14.0;
const TERRAIN_BASE_HEIGHT: i32 = 24;

// Mountains: ranges rising out of the hills above. A broad, slow noise
// decides *where* a range is (`MOUNTAIN_REGION_*`, hundreds of blocks
// across); inside one, ridged noise decides the peaks and the valleys between
// them. The edge of a range is a smooth ramp from nothing to full height, so
// hills turn into foothills rather than meeting a wall.
const MOUNTAIN_REGION_FREQ: f32 = 0.0012;
const MOUNTAIN_REGION_OCTAVES: u32 = 3;
/// Where the region noise starts and finishes raising a range.
const MOUNTAIN_EDGE: (f32, f32) = (0.05, 0.35);
const MOUNTAIN_RIDGE_FREQ: f32 = 0.005;
const MOUNTAIN_RIDGE_OCTAVES: u32 = 5;
/// How far a whole range is lifted, before any peak.
const MOUNTAIN_UPLIFT: f32 = 30.0;
/// The most a ridge adds on top of that.
const MOUNTAIN_PEAK: f32 = 190.0;
/// Arbitrary, as every mix constant is -- but picked, from 3,000 candidates,
/// as one that keeps the origin in the hills for the default seed (nothing
/// above y = 40 within 320 blocks) and for every seed a test or golden image
/// frames the origin with (within 128). The default seed's nearest range then
/// rises ~370 blocks from spawn, peaking ~220 up. So a new world starts where
/// iron is within reach and mountains are in view, and every test built around
/// the origin's terrain still shows the terrain it was built around.
const MOUNTAIN_SEED_MIX: u64 = 0x74D8_B1DA_9810_93EC;
const RIDGE_SEED_MIX: u64 = 0x5851_F42D_4C95_7F2D;

// Cave shape. `CAVE_FREQ` sets tunnel scale; `CAVE_THRESHOLD` sets how much
// of the noise field's ~[-1,1] range counts as "inside a tunnel" -- higher
// means rarer, narrower caves. `CAVE_SEED_MIX` keeps cave noise from being
// the exact same pattern as terrain noise reused at a different frequency.
const CAVE_FREQ: f32 = 0.045;
const CAVE_OCTAVES: u32 = 1;
const CAVE_LACUNARITY: f32 = 2.0;
const CAVE_GAIN: f32 = 0.5;
const CAVE_THRESHOLD: f32 = 0.6;
/// How far below the surface caves reach in a coarse node -- one farther
/// away than the finest detail ring. Nearby, caves are carved at every depth;
/// far off, only where they could be seen from above: the mouth and the start
/// of the passage, from a mountain or in flight. The rest of a distant cave
/// network is rock until the player comes close, where it has always been
/// hidden anyway, and it was most of what caves at every level cost to draw
/// (the owner: "in de verte niet super gedetailleerd en tot heel diep").
const DISTANT_CAVE_DEPTH: i32 = 24;
/// Large enough to flip `density` negative regardless of how far below the
/// surface a cell is -- `density`'s terrain term never approaches this
/// magnitude within any loaded region.
const CAVE_CARVE_AMOUNT: f32 = 1_000_000.0;
const CAVE_SEED_MIX: u64 = 0xD6E8_FEB8_6659_FD93;

/// Which of the level-0 side faces under a coarse cell speaks for it
/// ([`WorldGen::coarse_block`]): its own hash stream, so retuning caves or
/// trees cannot move it. Arbitrary, as every mix constant is.
const COARSE_FACE_SEED_MIX: u64 = 0x423B_10C5_A11E_7E15;

/// Which of a coarse cell's columns its top shows from above
/// ([`WorldGen::top_block`]): its own hash stream, apart from the sides'.
/// Arbitrary, as every mix constant is.
const COARSE_TOP_SEED_MIX: u64 = 0x7E0F_5A6B_2C19_D4E3;

/// How far [`WorldGen::pit_floor`] looks down a pit for its floor, in
/// blocks. **Chosen, not derived**: a pit deeper than this stands at the
/// block the search stops at. A pit column near the spawn is 9 blocks deep
/// at the median and 28 at the 90th percentile, but a cave can run on
/// under a pit's mouth: the deepest found on the default seed is 73 blocks
/// around the spawn, 80 along the forest line and 118 on the highest
/// mountain. This was 48 until the floor became the column's top
/// (`docs/PHASE3_ARCHITECTURE.md` §9.9): a colour could stop short of the
/// floor, a height that every level is held to may not. The search walks
/// only while it is in the cave, so the room costs nothing where a pit is
/// shallow.
const PIT_FLOOR_SEARCH: i32 = 256;

/// One level-0 column of ground, as every coarser level reads it
/// ([`WorldGen::column`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroundColumn {
    /// The `y` of its surface block: the height field
    /// ([`WorldGen::surface_height`]), before caves. What the column is
    /// made of is by depth below this ([`WorldGen::ground_layer`]).
    pub surface: i32,
    /// The `y` of the highest block of ground the sky sees in it
    /// ([`WorldGen::column_top`]): the surface block, or the floor of the
    /// pit a cave has opened there. Where the column stands, to every
    /// level.
    pub top: i32,
}

impl GroundColumn {
    /// A column no cave has opened: its top is its surface block.
    pub const fn whole(surface: i32) -> Self {
        Self {
            surface,
            top: surface,
        }
    }
}

/// One level-0 column under a coarse cell ([`NodeGround`]): the `y` of its
/// surface block (what it is made of is by depth below that), of its top
/// -- the highest block of ground the sky sees in it
/// ([`WorldGen::column_top`]) -- and of its four neighbours' tops and
/// surfaces, which is where it shows a side at level 0: every row above a
/// neighbour's top and at or under its own. Above the neighbour's
/// *surface* that side is a riser of the height field; from the surface
/// down to the neighbour's top it is the wall of the pit a cave has opened
/// there. And where in the cell it stands.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Covered {
    pub(crate) surface: i32,
    pub(crate) top: i32,
    /// The four neighbours' tops.
    pub(crate) around: [i32; 4],
    /// The four neighbours' surfaces.
    pub(crate) beside: [i32; 4],
    /// Blocks from the cell's corner, along x and along z.
    pub(crate) at: [u8; 2],
}

/// The ground under a coarse node, cell by cell: what
/// [`WorldGen::generate_aniso`] makes the node of.
pub(crate) struct NodeGround<'a> {
    /// The node's corner column, `(x, z)`, and its cells' width in blocks.
    corner: [i32; 2],
    step: i32,
    /// Any level-0 column, by its offset from the corner: the node's own and
    /// the ring around it asked once, anything further worked out when asked
    /// -- what the cave rule asks of the ground around a cell
    /// ([`WorldGen::cave_opens_only_where_level_0_does`]).
    column: Box<dyn Fn(i32, i32) -> GroundColumn + 'a>,
    /// The tops of cells other than the node's own, by corner and width, as
    /// they are worked out: an accelerator only (§3).
    other_tops: std::cell::RefCell<std::collections::HashMap<(i32, i32, i32), i32>>,
    /// Columns per cell: the cell's width squared.
    per_cell: usize,
    /// Every cell's level-0 columns, **sorted by top, ascending**, cell
    /// after cell (`z` outer). All of them say what the cell's sides are
    /// made of ([`WorldGen::coarse_block`]).
    columns: Vec<Covered>,
    /// Every cell's top ([`WorldGen::cell_top`]).
    tops: Vec<i32>,
    /// Per cell, the lowest top among its columns' neighbours: at or
    /// under it none of the cell's columns shows a side face at level 0.
    lows: Vec<i32>,
    /// Per cell, the lowest top among the four cells beside it **as they
    /// stand with no cave in the ground** (the rule over the surfaces):
    /// where its own risers end. Beside a cell of the next node, whose top
    /// this node does not work out, [`OUTSIDE_FOOT`] cells' widths under
    /// the lowest surface along that edge. Under this a row facing lower
    /// ground is the wall of a pit, not part of a riser.
    feet: Vec<i32>,
}

/// How far under the lowest column along a node's edge the foot of a riser
/// is taken to be, in cells' widths, where the cell beyond the edge belongs
/// to the next node and its top is not worked out. **Chosen, not derived**:
/// deep enough for a slope of four blocks to the block across that cell; a
/// row taken for part of a riser that is in fact buried is never seen.
const OUTSIDE_FOOT: i32 = 2;

/// One cell of a [`NodeGround`].
#[derive(Clone, Copy)]
struct CellColumns<'a> {
    columns: &'a [Covered],
    top: i32,
    low: i32,
    foot: i32,
}

impl<'a> NodeGround<'a> {
    /// The ground of a node of 16 x 16 cells, each `step` blocks wide, whose
    /// corner column is `corner`, over the columns `column(i, j)`, asked in
    /// blocks from that corner: from one cell's width outside the node to
    /// one cell's width past its far side to work the cells out (a column's
    /// neighbours, and the cells around a cell that might hold a summit),
    /// and further for the cave rule's cells around a cave. The world's own
    /// columns for a real node ([`WorldGen::node_ground`]); a test hands in
    /// ones made by hand.
    pub(crate) fn over(
        corner: [i32; 2],
        step: i32,
        column: impl Fn(i32, i32) -> GroundColumn + 'a,
    ) -> Self {
        let height = |i: i32, j: i32| column(i, j).top;
        let surface = |i: i32, j: i32| column(i, j).surface;
        let cells = Chunk::SIZE as i32;
        let per_cell = (step * step) as usize;
        let count = (cells * cells) as usize;
        let mut columns = Vec::with_capacity(count * per_cell);
        let (mut tops, mut lows) = (Vec::with_capacity(count), Vec::with_capacity(count));
        // Every cell's top with no cave in the ground: the feet's.
        let mut whole = Vec::with_capacity(count);
        for cz in 0..cells {
            for cx in 0..cells {
                let start = columns.len();
                for dz in 0..step {
                    for dx in 0..step {
                        let (i, j) = (cx * step + dx, cz * step + dz);
                        let own = column(i, j);
                        let near = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                            .map(|(di, dj): (i32, i32)| column(i + di, j + dj));
                        columns.push(Covered {
                            surface: own.surface,
                            top: own.top,
                            around: near.map(|c| c.top),
                            beside: near.map(|c| c.surface),
                            at: [dx as u8, dz as u8],
                        });
                    }
                }
                let cell = &mut columns[start..];
                // A fixed order among columns as high as each other: the
                // hash that picks a cell's face walks them in this order.
                cell.sort_unstable_by_key(|c| (c.top, c.at[1], c.at[0]));
                lows.push(
                    cell.iter()
                        .flat_map(|c| c.around)
                        .min()
                        .expect("a cell covers a column"),
                );
                let top = WorldGen::top_over(&height, cx * step, cz * step, step, step);
                tops.push(top);
                // The same over the surfaces: the same number wherever no
                // cave has opened a column of the cells around.
                whole.push(WorldGen::top_over(
                    &surface,
                    cx * step,
                    cz * step,
                    step,
                    step,
                ));
            }
        }
        // Where each cell's risers end: the lowest ground beside it.
        let feet = (0..count as i32)
            .map(|at| {
                let (cx, cz) = (at % cells, at / cells);
                GROUND_DIRECTIONS
                    .into_iter()
                    .enumerate()
                    .map(|(d, [dx, dz])| {
                        let (nx, nz) = (cx + dx, cz + dz);
                        if (0..cells).contains(&nx) && (0..cells).contains(&nz) {
                            return whole[(nz * cells + nx) as usize];
                        }
                        // The next node's cell: from the columns of it that
                        // stand along this edge.
                        let edge = |c: &&Covered| {
                            let along = c.at[if dx != 0 { 0 } else { 1 }] as i32;
                            along == if dx + dz > 0 { step - 1 } else { 0 }
                        };
                        columns[at as usize * per_cell..][..per_cell]
                            .iter()
                            .filter(edge)
                            .map(|c| c.beside[d])
                            .min()
                            .expect("a cell has a column on every edge")
                            - OUTSIDE_FOOT * step
                    })
                    .min()
                    .expect("four sides")
            })
            .collect();
        Self {
            corner,
            step,
            column: Box::new(column),
            other_tops: Default::default(),
            per_cell,
            columns,
            tops,
            lows,
            feet,
        }
    }

    /// The top ([`WorldGen::column_top`]) of the level-0 column `(x, z)`, in
    /// world blocks.
    fn column_top(&self, x: i32, z: i32) -> i32 {
        (self.column)(x - self.corner[0], z - self.corner[1]).top
    }

    /// The top ([`WorldGen::cell_top`]) of the cell `w` blocks wide at
    /// corner `(x0, z0)`, in world blocks, over this ground: one of the
    /// node's own cells as it was worked out, any other by the one rule
    /// ([`WorldGen::top_over`]).
    fn cell_top(&self, x0: i32, z0: i32, w: i32) -> i32 {
        let (i, j) = (x0 - self.corner[0], z0 - self.corner[1]);
        let cells = Chunk::SIZE as i32;
        if w == self.step
            && (0..cells).contains(&i.div_euclid(w))
            && (0..cells).contains(&j.div_euclid(w))
        {
            return self.tops[(j / w * cells + i / w) as usize];
        }
        if let Some(&top) = self.other_tops.borrow().get(&(x0, z0, w)) {
            return top;
        }
        let top = WorldGen::top_over(&|x, z| self.column_top(x, z), x0, z0, w, w);
        self.other_tops.borrow_mut().insert((x0, z0, w), top);
        top
    }

    fn cell(&self, at: usize) -> CellColumns<'_> {
        CellColumns {
            columns: &self.columns[at * self.per_cell..][..self.per_cell],
            top: self.tops[at],
            low: self.lows[at],
            foot: self.feet[at],
        }
    }
}

/// The level-0 columns one coarse cell covers and the top it draws
/// ([`WorldGen::cell_ground`]): what a neighbouring node needs to know what
/// the cell shows at a block, without generating its node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CellGround {
    /// The cell's corner column, and its width in blocks.
    pub x0: i32,
    pub z0: i32,
    pub w: i32,
    /// The `y` of the top block the cell draws ([`WorldGen::cell_top`]).
    pub top: i32,
    /// The `y` of the highest top among its columns.
    pub peak: i32,
    /// Each column's top ([`WorldGen::column_top`]), row by row (`z` outer).
    tops: Vec<i32>,
}

/// `a / b` for `b > 0`, rounded to the nearest whole number, **a half to
/// the even one**: rounding halves one way would lift (or lower) every cell
/// whose columns split evenly, and that is a level of the whole ground
/// standing a fraction of a block off the next.
fn div_round_half_even(a: i64, b: i64) -> i64 {
    let (q, r) = (a.div_euclid(b), a.rem_euclid(b));
    match (2 * r).cmp(&b) {
        std::cmp::Ordering::Less => q,
        std::cmp::Ordering::Greater => q + 1,
        std::cmp::Ordering::Equal => q + (q & 1),
    }
}

// Ore shape. Each ore carries its own frequency and threshold from its data
// file (`assets/ores/*.ron`); what is fixed here is the *kind* of noise, which
// is an algorithm rather than a number to tune. One octave, like caves: ore
// wants compact blobs, and extra octaves only add detail far below the size of
// a single block.
const ORE_OCTAVES: u32 = 1;
const ORE_LACUNARITY: f32 = 2.0;
const ORE_GAIN: f32 = 0.5;
/// Keeps ore noise from being cave noise reused -- without it, ore would line
/// every cave wall, since both would cross their thresholds in the same cells.
const ORE_SEED_MIX: u64 = 0x9E37_79B9_7F4A_7C15;

/// A per-ore seed offset derived from its name.
///
/// FNV-1a rather than [`std::collections::hash_map::DefaultHasher`]: the
/// standard hasher is explicitly not stable across Rust releases, and a world
/// whose ore positions move when the toolchain updates would violate Rule 1 in
/// a way no test on one machine would ever catch.
fn ore_seed_mix(name: &str) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for b in name.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// Bumped whenever a change in this file could change already-generated
/// terrain's shape -- any of the constants above, `density_at`,
/// `surface_height`, or the noise functions themselves. Save/load (block
/// 1.9, §7.2) refuses to load a world whose header disagrees with this: an
/// old save's *unedited* chunks are regenerated on load (§7.4), so a
/// changed generator would silently reshape the world around the player's
/// edits if this weren't checked -- the failure mode §7.4 names directly.
pub const WORLDGEN_VERSION: u32 = 3;

/// Seeded terrain + cave generator. See the module docs and
/// `docs/PHASE1_ARCHITECTURE.md` §8 for the contract this must hold to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorldGen {
    seed: u64,
}

/// One placed tree: where its trunk stands, and how tall it is.
///
/// Resolved from a hash of `(seed, chunk column)` alone, so it is a pure
/// function of the seed and never depends on what has been generated (§8.1).
#[derive(Clone, Copy, Debug)]
pub(crate) struct PlacedTree {
    x: i32,
    z: i32,
    base: i32,
    height: i32,
}

/// Where one tree stands and how big it is: the public face of the placement
/// [`WorldGen::trees_near`] is built from, for a caller that wants the trees
/// themselves rather than the blocks they make -- the LOD-transition probe
/// (`docs/PROPOSAL_LOD_TRANSITIONS.md`, M6) counting trees per ring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TreeSite {
    /// Trunk column, in blocks.
    pub x: i32,
    pub z: i32,
    /// The first trunk block's y (the surface block's y plus one).
    pub base: i32,
    /// Trunk height in blocks.
    pub height: i32,
    /// How far the canopy reaches from the trunk, in blocks.
    pub canopy_radius: i32,
}

/// The at-most-nine trees that can reach one chunk column.
///
/// A fixed array rather than a `Vec`: this is built per chunk (and per voxel
/// on the single-block paths), and allocating there would put a heap call in
/// the middle of worldgen. Nine is a proven bound, not a guess -- see
/// [`WorldGen::trees_near`].
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TreeSet {
    trees: [Option<PlacedTree>; 9],
    len: usize,
}

impl TreeSet {
    fn new() -> Self {
        Self {
            trees: [None; 9],
            len: 0,
        }
    }

    fn push(&mut self, t: PlacedTree) {
        if self.len < self.trees.len() {
            self.trees[self.len] = Some(t);
            self.len += 1;
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[cfg(test)]
    fn count_for_test(&self) -> usize {
        self.len
    }

    fn iter(&self) -> impl Iterator<Item = &PlacedTree> {
        self.trees[..self.len].iter().flatten()
    }
}

impl WorldGen {
    pub const fn new(seed: u64) -> Self {
        Self { seed }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The surface a cell of `step` blocks at corner `(x, z)` is judged
    /// against: its top, [`cell_top`](Self::cell_top). At `step` 1 it is the
    /// block's own column's top ([`column_top`](Self::column_top)).
    ///
    /// (Main took the mean of four of the cell's columns, rounded down; the
    /// first version of issue #423 the highest of all of them. Why neither
    /// is in `cell_top`'s own doc comment.)
    pub fn cell_surface_height(&self, x: i32, z: i32, step: i32) -> i32 {
        self.cell_top(x, z, step.max(1))
    }

    /// The terrain height field in blocks: a fractal noise surface that
    /// [`density`](Self::density)'s solid/air split (before caves) is
    /// measured against. A pure function of `(seed, x, z)` alone (§8.1).
    pub fn surface_height(&self, x: i32, z: i32) -> i32 {
        let n = fbm2(
            self.seed,
            x as f32 * TERRAIN_FREQ,
            z as f32 * TERRAIN_FREQ,
            TERRAIN_OCTAVES,
            TERRAIN_LACUNARITY,
            TERRAIN_GAIN,
        );
        (TERRAIN_BASE_HEIGHT as f32 + n * TERRAIN_AMPLITUDE + self.mountains(x, z)).round() as i32
    }

    /// How much a mountain range raises the column at `(x, z)`: zero outside
    /// every range, up to `MOUNTAIN_UPLIFT + MOUNTAIN_PEAK` on the highest
    /// peak.
    fn mountains(&self, x: i32, z: i32) -> f32 {
        let region = fbm2(
            self.seed ^ MOUNTAIN_SEED_MIX,
            x as f32 * MOUNTAIN_REGION_FREQ,
            z as f32 * MOUNTAIN_REGION_FREQ,
            MOUNTAIN_REGION_OCTAVES,
            2.0,
            0.5,
        );
        let (low, high) = MOUNTAIN_EDGE;
        let t = ((region - low) / (high - low)).clamp(0.0, 1.0);
        if t == 0.0 {
            // Most of the world: the ridged noise is not worth sampling.
            return 0.0;
        }
        let rise = t * t * (3.0 - 2.0 * t);
        let ridges = ridged2(
            self.seed ^ RIDGE_SEED_MIX,
            x as f32 * MOUNTAIN_RIDGE_FREQ,
            z as f32 * MOUNTAIN_RIDGE_FREQ,
            MOUNTAIN_RIDGE_OCTAVES,
            2.0,
            0.5,
        );
        rise * (MOUNTAIN_UPLIFT + ridges * MOUNTAIN_PEAK)
    }

    /// [`density`](Self::density), given an already-known `surface_height(x,
    /// z)` -- the shared body every per-block query below funnels through,
    /// so the 2D height-field noise (expensive: [`TERRAIN_OCTAVES`] octaves
    /// of hashing) is computed by the caller once per column and passed in,
    /// not recomputed per voxel. See [`generate`](Self::generate)'s doc
    /// comment for why that matters.
    ///
    /// **One block of level 0.** A coarse cell is not this function at a
    /// larger size: its top is [`cell_top`](Self::cell_top), and it is cave
    /// where [`row_is_cave`](Self::row_is_cave) says so.
    fn density_at(&self, x: i32, y: i32, z: i32, surface: i32) -> f32 {
        let terrain = (surface - y + 1) as f32;
        if terrain <= 0.0 {
            // Already air from the height field alone. `carved` below is
            // always >= 0 (it's either 0 or CAVE_CARVE_AMOUNT), so caves can
            // only ever *remove* solid mass, never add it -- this cell can't
            // become solid no matter what the cave noise says, so there's no
            // need to sample it. Skipping this is a real, load-bearing
            // optimization, not a micro one: roughly half of any generated
            // region's volume is above the surface, and the cave noise is by
            // far the most expensive part of this function (CAVE_OCTAVES
            // rounds of 3D value noise, 8 hashed corners each) -- see
            // `generate`'s doc comment for the regression this and the
            // surface-amortization above it together fix.
            return terrain;
        }
        let carved = if self.cave_at(x, y, z) {
            CAVE_CARVE_AMOUNT
        } else {
            0.0
        };
        terrain - carved
    }

    /// The ground under the coarse node at `(x0, z0)` whose cells are `step`
    /// blocks wide: every cell's level-0 columns, its top and where its
    /// columns show their sides ([`NodeGround`]).
    ///
    /// Every column of the node's footprint and of one ring around it is
    /// asked once ([`column`](Self::column): its surface, and whether a
    /// cave has opened it) -- 1,156 columns at level 1, 16,900 at level 3
    /// -- and a
    /// cell at the node's edge that might hold a summit asks the cells
    /// beyond it as well (up to one cell's width out; few do). That is what
    /// deriving every level from level 0's own columns costs, and
    /// `generation_cost_report` (`crates/render/tests/lod_consistency`)
    /// measures it.
    fn node_ground(&self, x0: i32, z0: i32, step: i32) -> NodeGround<'static> {
        // The node's columns and a cell's width around them, each worked
        // out when it is first asked for; the few the cave rule asks
        // further out (`cave_opens_only_where_level_0_does`), whenever they
        // are asked.
        let side = (Chunk::SIZE as i32 + 2) * step;
        let unasked = GroundColumn::whole(i32::MIN);
        let grid = vec![std::cell::Cell::new(unasked); (side * side) as usize];
        let gen = *self;
        NodeGround::over([x0, z0], step, move |i, j| {
            let (a, b) = (i + step, j + step);
            if !(0..side).contains(&a) || !(0..side).contains(&b) {
                return gen.column(x0 + i, z0 + j);
            }
            let slot = &grid[(b * side + a) as usize];
            if slot.get() == unasked {
                slot.set(gen.column(x0 + i, z0 + j));
            }
            slot.get()
        })
    }

    /// The block a coarse cell at corner `(x, y, z)` draws, or `None` for
    /// air: `v` blocks tall, over the level-0 columns of `cell`.
    ///
    /// **Solid** from the cell's top ([`cell_top`](Self::cell_top)) down: a
    /// row is solid while its lowest block is at or under the top, and the
    /// row that holds the top is cut off at it when the node is drawn
    /// ([`generate_aniso`](Self::generate_aniso)). **Air where a cave has
    /// carved all of it** ([`row_is_cave`](Self::row_is_cave)).
    ///
    /// **Made of what level 0's faces show there** (issue #423 decision 2:
    /// "a coarse riser face takes the mix of materials that level 0's faces
    /// show over the same footprint"). At level 0 a column shows a side face
    /// at every row above a neighbour's *surface* and at or under its own
    /// top -- a riser of the height field -- made of
    /// [`ground_layer`](Self::ground_layer) at that row's depth below the
    /// column's **own** surface. A coarse cell's rows stand for all the side
    /// faces its columns show at those rows, and the cell wears the layer of
    /// **one of them, picked by a hash of the cell's position** -- so over
    /// many cells every layer shows in the share it has at level 0,
    /// whichever way the slope runs. A gentle slope's coarse
    /// riser is then the surface block's side nearly row after row, as its
    /// many one-block risers are at level 0; a steep one is about a third
    /// soil, as level 0's two- and three-block risers are; stone shows where
    /// a column stands more than `1 + SOIL_DEPTH` above a neighbour, which
    /// is a cliff at level 0 too.
    ///
    /// Two rules came before, and each was a measured error
    /// (`crates/render/tests/lod_consistency`, on steep ground, where level
    /// 0's sides are 66% surface block, 33% soil, 0.1% stone):
    /// - by depth below the *cell's* surface (main until #423): one grass
    ///   row, three of soil, then stone down a riser as tall as the cell is
    ///   wide times the slope -- 7 to 40% stone, the owner's "from afar it
    ///   is mostly stone";
    /// - by depth below the lowest covered surface at or above the row (the
    ///   first version of this function): never stone, but the *least*
    ///   depth any of the cell's columns offers, so 81 to 99% surface block
    ///   -- a mountain all grass side from afar and a third soil up close.
    ///
    /// **Where no covered column shows a riser at these rows.** A cell's top
    /// is the mean of its columns, so on a slope its riser reaches below the
    /// lowest of them, down to the top of the next cell as it stands with
    /// no cave in the ground (`cell.foot`): rows that level 0 shows from
    /// that cell's columns. Such a row wears **one of all the risers the
    /// cell's columns show, at any row**, by the same hash: the same slope,
    /// so the same mix.
    ///
    /// **The wall of a pit, under the cell's own risers.** A column beside
    /// a pit shows a side from the pit's floor up to where that neighbour's
    /// surface was ([`column_top`](Self::column_top)): stone under the
    /// soil. At or under the foot a cell's row faces lower ground only
    /// where a pit has sunk the cell beside it, and wears one of *those*
    /// faces its columns show at these rows, by the same hash. Where there
    /// is none either -- the inside of the ground -- it wears the layer
    /// below the lowest covered column at or above it, which is stone in
    /// the end.
    ///
    /// **Why the slope comes first.** A cell is one block a row, so its row
    /// is one material on all four sides. Picked from the slope's risers
    /// and a pit's wall together -- the faces level 0 shows there, in their
    /// shares -- a cell beside a pit wore the wall's stone on the side that
    /// faces the slope as well: 2.4% of a steep footprint's sides at level
    /// 3 where level 0 has 0.1%, and still 1.6% with the pit's wall taken
    /// wherever the slope showed no riser at those very rows (measured when
    /// §9.9 was built: `a_mountain_is_made_of_the_same_from_afar`, the
    /// stone of issue #423 again). So a row that is part of the cell's own
    /// riser wears the slope, and the top rows of a pit's wall on a slope
    /// wear it too. (By depth alone, these rows came out as bands of soil and
    /// stone along the foot of every riser on a slope that steepens: the
    /// stone of issue #423 again.)
    ///
    /// **The cell that holds the top** wears the surface block, as every
    /// real column's top does (#296): its top face is most of what is seen
    /// of it.
    ///
    /// `cell.low` is the lowest top among the covered columns' neighbours:
    /// at or under it no covered column shows a side, so those cells --
    /// most of a node -- are spared the count.
    fn coarse_block(
        &self,
        [x, y, z]: [i32; 3],
        ground: &NodeGround,
        at: usize,
        blocks: TerrainBlocks,
        v: i32,
    ) -> Option<BlockId> {
        let CellColumns {
            columns: covered,
            top,
            low,
            foot,
        } = ground.cell(at);
        if y > top {
            return None;
        }
        let peak = covered
            .last()
            .expect("a cell covers at least one column")
            .top;
        // The columns that reach this row: the others have no ground in it.
        let reaching = &covered[covered.partition_point(|c| c.top < y)..];
        let reaching_blocks = reaching
            .iter()
            .map(|c| (x + c.at[0] as i32, z + c.at[1] as i32, c.top));
        if self.row_is_cave(reaching_blocks, peak, top, y, v)
            && self.cave_opens_only_where_level_0_does(
                &|x, z| ground.column_top(x, z),
                &|x0, z0, w| ground.cell_top(x0, z0, w),
                [x, z, ground.step],
                top,
                (y, v),
            )
        {
            return None;
        }
        // The cell that holds the top.
        if top < y + v {
            return Some(self.top_block(x, z, top, covered, blocks));
        }
        // The level-0 side faces at this cell's rows: one for every
        // neighbour a column stands above, at each row it reaches --
        // `beside` the neighbours' surfaces for the risers of the height
        // field, their tops for the walls of pits as well. One of them,
        // by the cell's hash; `None` where there is none.
        let hash = crate::noise::hash(self.seed ^ COARSE_FACE_SEED_MIX, x, y, z);
        let one_of_the_faces_here = |beside: fn(&Covered) -> [i32; 4]| -> Option<BlockId> {
            if y + v - 1 <= low {
                return None;
            }
            let faces = |c: &Covered, row: i32| -> u64 {
                if row > c.top {
                    return 0;
                }
                beside(c).iter().filter(|&&n| n < row).count() as u64
            };
            let shown: u64 = reaching
                .iter()
                .map(|c| (y..y + v).map(|row| faces(c, row)).sum::<u64>())
                .sum();
            if shown == 0 {
                return None;
            }
            let mut pick = hash % shown;
            for c in reaching {
                for row in y..y + v {
                    let n = faces(c, row);
                    if pick < n {
                        return Some(self.material_at(x, row, z, c.surface, blocks));
                    }
                    pick -= n;
                }
            }
            unreachable!("the pick is under the number of faces counted")
        };
        // A riser of the height field at these rows.
        if let Some(block) = one_of_the_faces_here(|c| c.beside) {
            return Some(block);
        }
        // No covered column shows a riser here.
        let row = y + v - 1;
        if row > foot {
            // Part of the cell's own riser all the same: one of all its
            // risers on the height field. A column shows a neighbour one
            // face for every block it stands above that neighbour's
            // surface, the first of its own top block.
            let every =
                |c: &Covered| -> u64 { c.beside.iter().map(|&n| (c.top - n).max(0) as u64).sum() };
            let all: u64 = covered.iter().map(every).sum();
            if all > 0 {
                let mut pick = hash % all;
                for c in covered {
                    for &n in &c.beside {
                        let tall = (c.top - n).max(0) as u64;
                        if pick < tall {
                            let depth = pick as i32;
                            return Some(self.material_at(x, c.top - depth, z, c.surface, blocks));
                        }
                        pick -= tall;
                    }
                }
                unreachable!("the pick is under the number of faces counted")
            }
        }
        // Under the cell's own risers: the wall of a pit, where a column
        // shows one here.
        if let Some(block) = one_of_the_faces_here(|c| c.around) {
            return Some(block);
        }
        let at = covered
            .partition_point(|c| c.top < row)
            .min(covered.len() - 1);
        Some(self.material_at(x, row, z, covered[at].surface, blocks))
    }

    /// The block a coarse cell's top wears, its top face being most of what
    /// is seen of it: **what level 0 shows from above over one of its
    /// columns, picked by a hash of the cell** ([`from_above`]) -- so over
    /// many cells each thing level 0 shows from above shows in the share it
    /// has there, as the cell's riser rows do for the sides.
    ///
    /// Until the colour morph every top cell wore the surface block (#296),
    /// and a coarse level showed a pit only where every column of a cell is
    /// one (`row_is_cave`, the nesting rule): on the default seed's
    /// mountains pits open 6.1% of level 0's columns, and the coarse levels
    /// showed 5.1, 3.4 and 1.3% of their ground as pit at cell widths 2, 4
    /// and 8 -- the grey patches a mountain lost from afar (the zoom pairs of
    /// 2026-11-21). A cell that is partly pit now wears the pit's floor in the
    /// share its columns are pit.
    ///
    /// [`from_above`]: Self::from_above
    fn top_block(
        &self,
        x: i32,
        z: i32,
        top: i32,
        covered: &[Covered],
        blocks: TerrainBlocks,
    ) -> BlockId {
        let c = covered[self.top_pick(x, z, top, covered.len())];
        let column = GroundColumn {
            surface: c.surface,
            top: c.top,
        };
        match self.from_above(x + c.at[0] as i32, z + c.at[1] as i32, column, blocks) {
            Some(floor) => floor,
            None => self.material_at(x, top, z, top, blocks),
        }
    }

    /// Which of the `columns` level-0 columns of the cell at `(x, z)`, its
    /// top block at `top`, the cell's top shows from above: an index into
    /// them **sorted by surface, then `z`, then `x`** (the order a
    /// [`NodeGround`] keeps a cell's columns in), by a hash of the cell.
    /// The one pick: a voxel cell's top block ([`top_block`](Self::top_block))
    /// and a far quad's top ([`top_layer`](Self::top_layer)) both ask it, so
    /// a far quad over the columns of a voxel cell shows the same column.
    fn top_pick(&self, x: i32, z: i32, top: i32, columns: usize) -> usize {
        (crate::noise::hash(self.seed ^ COARSE_TOP_SEED_MIX, x, top, z) % columns as u64) as usize
    }

    /// [`top_block`](Self::top_block) for a representation that has no
    /// blocks -- the far terrain (#422): **the layer the top of the cell
    /// `[x0, x0 + w) x [z0, z0 + w)` wears**, its top block at `top`
    /// ([`cell_top_sampled`](Self::cell_top_sampled) with the same `n`).
    /// The surface block, or the layer of the pit floor level 0 shows from
    /// above one of its columns ([`pit_floor`](Self::pit_floor)), the column
    /// picked by the cell's hash ([`top_pick`](Self::top_pick)).
    ///
    /// The columns are the `n` x `n` grid `cell_top_sampled` visits, all of
    /// them when `n >= w`: there this is the layer of the block
    /// `generate_aniso` gives the same cell's top, column for column
    /// (`a_far_top_wears_what_the_voxel_cell_wears` holds the two together;
    /// an ore is stone here, as it is to every far face). Wider cells are
    /// sampled, as their tops are: over many cells a pit's floor still shows
    /// in the share the sampled columns are pit.
    pub fn top_layer(&self, x0: i32, z0: i32, w: i32, n: i32, top: i32) -> GroundLayer {
        let mut columns: Vec<(i32, i32, i32, i32)> = Self::sample_columns(x0, z0, w, n)
            .map(|(x, z)| {
                let column = self.column(x, z);
                (column.top, z, x, column.surface)
            })
            .collect();
        columns.sort_unstable();
        let (floor, _, _, surface) = columns[self.top_pick(x0, z0, top, columns.len())];
        Self::ground_layer(surface - floor)
    }

    /// What level 0 shows from directly above `column`, the column at
    /// `(x, z)` ([`column`](Self::column)), where that is not the surface
    /// block: the floor of the pit a cave has opened there, made of what
    /// that depth below the column's surface is made of. `None` where the
    /// column's surface block stands (no cave carves it): it shows the
    /// surface block.
    ///
    /// The one rule a coarser representation reads a column's top colour
    /// from: the voxel levels as a block (this), the far terrain as a layer
    /// ([`top_layer`](Self::top_layer), #422), both off the same floor.
    pub fn from_above(
        &self,
        x: i32,
        z: i32,
        column: GroundColumn,
        blocks: TerrainBlocks,
    ) -> Option<BlockId> {
        (column.top < column.surface)
            .then(|| self.material_at(x, column.top, z, column.surface, blocks))
    }

    /// The column at `(x, z)` as every level of detail reads it
    /// ([`GroundColumn`]): its surface, and its top -- **the highest block
    /// of ground the sky sees in it**. That is the surface block, or, where
    /// a cave has carved the surface block away, the floor of the pit
    /// ([`pit_floor`](Self::pit_floor)).
    ///
    /// **This is level 0's drawn top**, block for block
    /// (`a_columns_top_is_what_level_0_draws_from_above`), and every
    /// coarser top is the one rule over it ([`cell_top`](Self::cell_top)):
    /// so a pit is ground like any other, a coarse cell over one sinks by
    /// the share of its columns that are pit times their depth, and the
    /// distance morph slides it as it slides a slope
    /// (`docs/PHASE3_ARCHITECTURE.md` §9.9). Until then a top was the mean
    /// of the *surfaces* and a coarse cell was open only where all of its
    /// ground is cave, so a pit appeared out of flat grass on the way in.
    ///
    /// Costs one cave sample over the surface height, and a walk down the
    /// pit where there is one (about one column in fifteen).
    pub fn column(&self, x: i32, z: i32) -> GroundColumn {
        let surface = self.surface_height(x, z);
        GroundColumn {
            surface,
            top: self.pit_floor(x, z, surface).unwrap_or(surface),
        }
    }

    /// The `y` of the highest block of ground the sky sees in the column
    /// `(x, z)`: [`column`](Self::column)`.top`. **Where a column stands,
    /// to every level**: `cell_top(x, z, 1)` is this.
    pub fn column_top(&self, x: i32, z: i32) -> i32 {
        self.column(x, z).top
    }

    /// The `y` of the floor of the pit a cave has opened in the column
    /// `(x, z)` whose surface block is at `surface`: the first block of
    /// ground under the carved surface, within [`PIT_FLOOR_SEARCH`] (a
    /// deeper pit shows the block the search stops at). `None` where no
    /// cave carves the surface block. **The one place a pit's floor is
    /// found**: [`column`](Self::column) reads it, and every top and every
    /// top's colour reads that.
    pub fn pit_floor(&self, x: i32, z: i32, surface: i32) -> Option<i32> {
        if !self.cave_at(x, surface, z) {
            return None;
        }
        Some(
            (1..=PIT_FLOOR_SEARCH)
                .map(|d| surface - d)
                .find(|&y| !self.cave_at(x, y, z))
                .unwrap_or(surface - PIT_FLOOR_SEARCH),
        )
    }

    /// Whether `block` is what level 0 shows from above one of the columns
    /// `[x0, x0 + w) x [z0, z0 + w)` that a cave has opened ([`from_above`]):
    /// the other thing a coarse top may wear besides the surface block.
    ///
    /// [`from_above`]: Self::from_above
    #[cfg(test)]
    pub(crate) fn a_pit_under_shows(
        &self,
        x0: i32,
        z0: i32,
        w: i32,
        block: BlockId,
        blocks: TerrainBlocks,
    ) -> bool {
        (0..w * w).any(|k| {
            let (x, z) = (x0 + k % w, z0 + k / w);
            self.from_above(x, z, self.column(x, z), blocks) == Some(block)
        })
    }

    /// Whether a cave has carved away the row of `v` blocks from `y` of a
    /// coarse cell whose top is `top` and whose highest column is `peak`:
    /// `columns` its level-0 columns as `(x, z, top)`, each one's top what
    /// the sky sees of it ([`column_top`](Self::column_top)).
    ///
    /// **Only where every block of ground in the row is cave at level 0**:
    /// every block of the cell's columns in these rows that is at or under
    /// its own column's top (and at or under the cell's top, above which
    /// the row is not drawn). One block of rock among them and the cell
    /// stays whole. (The air of a pit above its floor is not ground: the
    /// cell's top has sunk for it already, [`cell_top`](Self::cell_top).)
    ///
    /// **So caves nest, and a pit only ever opens on approach.** A finer
    /// cell inside this one covers some of the same blocks; if all of this
    /// cell's are cave, so are all of its, and level 0 is the blocks
    /// themselves. Whatever a level draws carved, every finer level draws
    /// open there too (`coarse_caves_nest`,
    /// `crates/render/tests/lod_consistency`). A cave narrower than a cell
    /// is detail: it shows when the level that can hold it is drawn.
    ///
    /// Until #423's second round a cell of rows one block tall was cave
    /// where the noise at its corner block was, and a level-3 cell where
    /// most of eight samples near its middle were: a cell-wide pit could
    /// open at level 3 over ground every finer level drew whole (up to 267
    /// columns of a 64-block footprint, 8 x 8 blocks at a time and up to 12
    /// deep), and close again as the player came nearer.
    ///
    /// **Only within [`DISTANT_CAVE_DEPTH`] of the highest column**: a row
    /// whose lowest block is deeper than that below `peak` is rock. Measured
    /// from the highest covered column and the row's lowest block so that
    /// this nests as well -- a finer cell's highest column is no higher and
    /// its row no lower.
    ///
    /// The first block of rock ends the count, and nearly every cell has
    /// one at the first block asked: one noise sample a cell, as a cave
    /// sample has always cost.
    fn row_is_cave(
        &self,
        columns: impl Iterator<Item = (i32, i32, i32)>,
        peak: i32,
        top: i32,
        y: i32,
        v: i32,
    ) -> bool {
        if y <= peak - DISTANT_CAVE_DEPTH {
            return false;
        }
        let last = (y + v - 1).min(top);
        let mut ground = false;
        for (x, z, column_top) in columns {
            for row in y..=last.min(column_top) {
                if !self.cave_at(x, row, z) {
                    return false;
                }
                ground = true;
            }
        }
        // No ground in the row at all is not a cave: the row is above the
        // top, and never asked.
        ground
    }

    /// Whether the cave [`row_is_cave`](Self::row_is_cave) has found in the
    /// block rows `rows` (`y` to `y + v - 1`) of the coarse cell `w` blocks
    /// wide at `(x0, z0)` (`cell`), its top at `top`, is drawn: **only where
    /// no level opens it through ground that level 0 draws**
    /// (`docs/PHASE3_ARCHITECTURE.md` §9.12). `column_top` is a level-0
    /// column's top ([`column_top`](Self::column_top)) and `cell_top` a
    /// cell's ([`cell_top`](Self::cell_top)), over the same ground.
    ///
    /// **Why.** At level 0 a cave under the ground is hidden by the ground
    /// around it. A coarse level draws that ground lower wherever a cell's
    /// mean stands under some of its columns -- by design (§9.3) -- and where
    /// that ground is what hid a cave, the cave is open: the owner's
    /// floating block, a level-2 cell at (5460, 3068) whose rows 135 and 136
    /// are cave, seen past the cell in front of it standing at its mean,
    /// 134, where level 0's columns stand at 136 to 138 and close the cave.
    /// The sky showed through it from 346 blocks away, and it closed at 288.
    /// The cave itself was right -- every block of it is cave at level 0 --
    /// and what opened it was a top. So every block row of the cave is asked,
    /// on each side and over its highest row:
    ///
    /// - **Beside it**: level 0 is open -- no block of ground -- across the
    ///   whole cell beside it in that row, so what a level opens there level
    ///   0 opens too; or every cell that may be drawn against that side
    ///   stands at or over the row: of every level from 1 to this one, the
    ///   cells that touch the side, and where the side is the node's edge
    ///   the cell one level coarser that touches it (rings are one level
    ///   apart; past the coarsest voxel level the far terrain's quads at the
    ///   join are that level's cells, §9.3).
    /// - **Over it**: level 0 is open across the cell's own columns in the
    ///   row over the cave, or every cell that may be drawn there stands at
    ///   or over that row: the cell itself, every finer cell in it, and
    ///   where that row is the next node's, the cell one level coarser.
    ///
    /// Otherwise the row stays rock, as a cave too narrow for the cell does:
    /// it shows when a level that keeps it closed where level 0 does is
    /// drawn.
    ///
    /// **It nests, so caves still do** (`coarse_caves_nest`): a finer cell in
    /// this one passes the same test in the same row. Its own ground is part
    /// of this cell's; a cell beside it is inside this one (open: all of
    /// this cell's ground in the row is cave) or inside the cell beside this
    /// one (open if that is, and among the cells that stand over the row if
    /// not); and the cells that stand over this one's rows stand over its.
    /// Asking only what the cell's own level draws beside it would not nest:
    /// tops do not nest, and a finer cell beside a cave can stand lower than
    /// the coarser one that hid it.
    ///
    /// Asked only of a row [`row_is_cave`](Self::row_is_cave) has found
    /// cave, which few rows are: level 0's blocks across a cell beside it
    /// first, which the first block of ground ends, and the cells' tops only
    /// where that is not open.
    fn cave_opens_only_where_level_0_does(
        &self,
        column_top: &dyn Fn(i32, i32) -> i32,
        cell_top: &dyn Fn(i32, i32, i32) -> i32,
        cell: [i32; 3],
        top: i32,
        rows: (i32, i32),
    ) -> bool {
        let [x0, z0, w] = cell;
        let (y, v) = rows;
        let level = w.trailing_zeros();
        // A node is 16 cells wide and deep, and 16 cells' widths tall.
        let (cells, node) = (Chunk::SIZE as i32, Chunk::SIZE as i32 * w);
        // Where a cell one level coarser may be drawn beside or over this
        // one; past the coarsest voxel level it is the far terrain, whose
        // quads at the join are this level's cells.
        let coarser = level < crate::far::FINEST_LEVEL;
        let highest = (y + v - 1).min(top);
        // Level 0 has no block of ground in row `r` of the cell `cw` wide at
        // `(cx, cz)`.
        let open = |cx: i32, cz: i32, cw: i32, r: i32| {
            (0..cw * cw).all(|k| {
                let (x, z) = (cx + k % cw, cz + k / cw);
                r > column_top(x, z) || self.cave_at(x, r, z)
            })
        };
        // The lowest top of the cells of widths 2 to `2^last` that hold any
        // of the columns `(at(t), t)`.
        let lowest = |at: &dyn Fn(i32) -> (i32, i32), count: i32, last: u32| -> i32 {
            (1..=last)
                .flat_map(|k| {
                    let cw = 1 << k;
                    (0..count).step_by(cw.min(count) as usize).map(move |t| {
                        let (x, z) = at(t);
                        (x.div_euclid(cw) * cw, z.div_euclid(cw) * cw, cw)
                    })
                })
                .map(|(cx, cz, cw)| cell_top(cx, cz, cw))
                .min()
                .expect("a level at least")
        };
        let (i, j) = (x0.div_euclid(w).rem_euclid(cells), z0.div_euclid(w).rem_euclid(cells));
        for [dx, dz] in GROUND_DIRECTIONS {
            // The columns just across this side, and whether it is the node's.
            let across = |t: i32| match (dx, dz) {
                (1, _) => (x0 + w, z0 + t),
                (-1, _) => (x0 - 1, z0 + t),
                (_, 1) => (x0 + t, z0 + w),
                _ => (x0 + t, z0 - 1),
            };
            let edge = match (dx, dz) {
                (1, _) => i == cells - 1,
                (-1, _) => i == 0,
                (_, 1) => j == cells - 1,
                _ => j == 0,
            };
            let last = if edge && coarser { level + 1 } else { level };
            let mut stands = None;
            for r in y..=highest {
                if open(x0 + dx * w, z0 + dz * w, w, r) {
                    continue;
                }
                if *stands.get_or_insert_with(|| lowest(&across, w, last)) < r {
                    return false;
                }
            }
        }
        let over = highest + 1;
        if !open(x0, z0, w, over) {
            let last = if over.rem_euclid(node) == 0 && coarser {
                level + 1
            } else {
                level
            };
            // The cells over it: the cell itself, the finer cells in it, and
            // the coarser one that holds it.
            let low = (1..=last)
                .flat_map(|k| {
                    let cw = 1i32 << k;
                    let per = (w / cw).max(1);
                    (0..per * per).map(move |q| {
                        let (x, z) = (x0 + q % per * cw, z0 + q / per * cw);
                        (x.div_euclid(cw) * cw, z.div_euclid(cw) * cw, cw)
                    })
                })
                .map(|(cx, cz, cw)| cell_top(cx, cz, cw))
                .min()
                .expect("a level at least");
            if low < over {
                return false;
            }
        }
        true
    }

    /// The level-0 columns the cell `w` blocks wide at corner `(x0, z0)`
    /// covers, and its top ([`CellGround`]): `w * w` surface heights, and
    /// for a cell that might hold a summit a look at the cells around it.
    pub fn cell_ground(&self, x0: i32, z0: i32, w: i32) -> CellGround {
        let w = w.max(1);
        let tops: Vec<i32> = (0..w * w)
            .map(|k| self.column_top(x0 + k % w, z0 + k / w))
            .collect();
        let top = Self::top_over(
            &|x: i32, z: i32| {
                if (x0..x0 + w).contains(&x) && (z0..z0 + w).contains(&z) {
                    tops[((z - z0) * w + x - x0) as usize]
                } else {
                    self.column_top(x, z)
                }
            },
            x0,
            z0,
            w,
            w,
        );
        CellGround {
            x0,
            z0,
            w,
            top,
            peak: *tops.iter().max().expect("a cell covers a column"),
            tops,
        }
    }

    /// Whether a coarse node whose rows are `v` blocks tall draws the block
    /// at row `y` of `cell` solid: at or under the cell's top, in a row no
    /// cave has carved away ([`row_is_cave`](Self::row_is_cave), the rule
    /// [`generate_aniso`](Self::generate_aniso) builds the node by). For
    /// judging what a neighbouring node of another level of detail draws,
    /// without generating it.
    ///
    /// `y` is **a block row**, not a stored row's corner: the node is cut
    /// off at the cell's top as it is drawn, so a block above the top is air
    /// whatever row it is stored in.
    pub fn cell_draws_solid(&self, cell: &CellGround, y: i32, v: i32) -> bool {
        if y > cell.top {
            return false;
        }
        let row = y.div_euclid(v) * v;
        let w = cell.w;
        let columns = cell
            .tops
            .iter()
            .enumerate()
            .map(|(k, &top)| (cell.x0 + k as i32 % w, cell.z0 + k as i32 / w, top));
        if !self.row_is_cave(columns, cell.peak, cell.top, row, v) {
            return true;
        }
        // The cell's own columns are at hand; any other is worked out.
        let column_top = |x: i32, z: i32| {
            let (i, j) = (x - cell.x0, z - cell.z0);
            if (0..w).contains(&i) && (0..w).contains(&j) {
                cell.tops[(j * w + i) as usize]
            } else {
                self.column_top(x, z)
            }
        };
        !self.cave_opens_only_where_level_0_does(
            &column_top,
            &|x0, z0, cw| Self::top_over(&column_top, x0, z0, cw, cw),
            [cell.x0, cell.z0, w],
            cell.top,
            (row, v),
        )
    }

    /// What a column is made of `depth` blocks below its own surface
    /// (`depth = surface - y`): the surface block at the surface, soil for
    /// the next [`SOIL_DEPTH`], stone beneath. **The one depth rule**: level
    /// 0's blocks ([`material_at`](Self::material_at)), a coarse riser and a
    /// far wall all read it, each from the surface of the column the face
    /// stands for and never from the height of the face itself (issue #423
    /// decision 2).
    pub fn ground_layer(depth: i32) -> GroundLayer {
        if depth <= 0 {
            GroundLayer::Surface
        } else if depth <= SOIL_DEPTH {
            GroundLayer::Soil
        } else {
            GroundLayer::Stone
        }
    }

    /// The `y` of the top block a cell `w` blocks wide and deep draws over
    /// the columns `[x0, x0 + w) x [z0, z0 + w)`: **the mean of their
    /// level-0 tops -- what the sky sees of each
    /// ([`column_top`](Self::column_top)) -- rounded to the nearest block
    /// (a half to the even one) -- and, where the cell holds a summit, the
    /// highest mean of the four half-width cells it splits into.** The top
    /// *face* is one above.
    ///
    /// **A cell holds a summit** when the highest column of the nine cells
    /// around -- itself and its eight neighbours -- is one of its own (the
    /// first of them, `z` then `x`, where several are as high).
    ///
    /// At `w = 1` this is [`column_top`](Self::column_top) itself: level 0
    /// is the same function, not a special case. At `w = 2` the half-width
    /// cells are single columns, so a level-1 summit stands at its highest
    /// column.
    ///
    /// **A column's top is its surface block, or the floor of the pit a
    /// cave has opened in it** -- so a cell over a pit stands lower by the
    /// share of its columns that are pit times how deep they are, and a pit
    /// is drawn at every level with the volume level 0 gives it, wider and
    /// shallower the coarser the cell (§9.9's study). Until then this was
    /// the mean of the *surfaces*, and a pit showed only where a cave had
    /// carved every column of a cell.
    ///
    /// **Why this rule** (issue #423, second round, and the summit study
    /// after it; `height_rule_study_report` in
    /// `crates/render/tests/lod_consistency`, the tables in
    /// `docs/PHASE3_ARCHITECTURE.md` §9.3 and §9.4). A swap replaces a cell
    /// by the four finer cells under it, so what changes on screen is the
    /// coarser top minus the finer ones.
    /// - The **mean** leaves that difference with no direction: the tops
    ///   under a cell stand as much above it as below, so a slope neither
    ///   sinks nor rises as a ring edge passes over it. The highest column
    ///   (the first version of #423) stood every slope higher the coarser
    ///   the level, and the slope sank 7 to 9 pixels at every edge; main's
    ///   mean of four columns rounded *down* stood it half a block low.
    /// - The mean alone lowers what stands out: a peak lost 6 blocks by
    ///   level 3 and grew back edge by edge. So a summit is lifted.
    /// - **Not to its highest column**: that turned one high column into a
    ///   tower the cell's width, and a crest into a row of them (18 and 12
    ///   cells standing alone at least their own width proud around the
    ///   owner's mountain at levels 2 and 3, against 6 and 2 now). The mean
    ///   of the highest half-width cell is what the next finer level would
    ///   draw there by the mean, so a coarse summit is never more than twice
    ///   as wide as ground that really stands that high, and a peak erodes
    ///   a step per level instead of jumping from its column to the cell:
    ///   0, 0.8 and 2.2 blocks under level 0 at levels 1 to 3 on steep
    ///   ground, where the plain mean was 0.8, 2.0 and 6.0.
    ///
    /// **It does not nest as the max did**: about half of a slope's columns
    /// stand above the coarser cell that covered them, half below, and a
    /// summit mostly rises a little as the player comes nearer. Nothing may
    /// assume a coarser level is at or above a finer one.
    ///
    /// **No rounding to rows.** Level 3 stores rows of four blocks, and the
    /// row that holds this top is cut off at it when the node is drawn
    /// ([`generate_aniso`](Self::generate_aniso)), so every level's top face
    /// is exactly one above this, and a far quad's is too.
    ///
    /// Costs `w * w` surface heights, and for a cell whose own columns are
    /// not level a look at its neighbours that stops at the first column
    /// above its highest. [`cell_top_sampled`](Self::cell_top_sampled) is
    /// the cheaper form for a cell too wide to visit every column of.
    pub fn cell_top(&self, x0: i32, z0: i32, w: i32) -> i32 {
        self.cell_top_sampled(x0, z0, w, w)
    }

    /// [`cell_top`](Self::cell_top) over an `n` x `n` grid of the cell's
    /// columns instead of all of them: the mean of `min(n, w)^2` column
    /// tops, or the highest mean of the four quarters of the sample grid
    /// where the first highest sample of the nine cells around, each
    /// sampled on its own grid, is one of this cell's. **The one implementation** ([`top_over`](Self::top_over)) --
    /// `cell_top` is this at `n = w`, and for any `n >= w` the two are the
    /// same number.
    ///
    /// The sample columns are fixed by the cell alone
    /// ([`sample_columns`](Self::sample_columns)): `n` to a side, `w / n`
    /// apart, **centred in the cell**, so that the mean of a slope's samples
    /// is the slope at the cell's middle. So the answer is a pure function
    /// of `(seed, x0, z0, w, n)`, the same on every machine and whichever
    /// neighbour asks.
    ///
    /// It can miss a peak narrower than the spacing, and its mean is a
    /// sample of the cell's; by how much is measured, per cell width, by
    /// `stepped_top_rule_report` in `crates/render/tests/lod_consistency`.
    /// A cell half as wide sampled with the same `n` looks at columns its
    /// parent skips.
    pub fn cell_top_sampled(&self, x0: i32, z0: i32, w: i32, n: i32) -> i32 {
        Self::top_over(&|x, z| self.column_top(x, z), x0, z0, w, n)
    }

    /// **The rule for a cell's top**, over any height field: `height(x, z)`
    /// a column's top. [`cell_top`](Self::cell_top) and
    /// [`cell_top_sampled`](Self::cell_top_sampled) are this over the
    /// world's own field, and a node's cells are this over the heights the
    /// node has already asked for ([`NodeGround`]) -- one body, whoever
    /// calls (`ARCHITECTURE.md` Rule 5).
    pub(crate) fn top_over(
        height: &impl Fn(i32, i32) -> i32,
        x0: i32,
        z0: i32,
        w: i32,
        n: i32,
    ) -> i32 {
        let (count, step, lead) = Self::sampling(w, n);
        // Sample `(a, b)` of the cell `(ci, cj)` cells from this one.
        let sample = |ci: i32, cj: i32, a: i32, b: i32| {
            height(
                x0 + ci * w.max(1) + lead + a * step,
                z0 + cj * w.max(1) + lead + b * step,
            )
        };
        let (mut sum, mut high, mut first) = (0i64, i32::MIN, (0, 0));
        // The four half-width cells the cell splits into -- the cells the
        // next finer level draws here: each one's sum and count of samples.
        let half = count / 2;
        let (mut quarter, mut quarter_count) = ([0i64; 4], [0i64; 4]);
        for b in 0..count {
            for a in 0..count {
                let h = sample(0, 0, a, b);
                sum += h as i64;
                let q = (a >= half) as usize + 2 * (b >= half) as usize;
                quarter[q] += h as i64;
                quarter_count[q] += 1;
                if h > high {
                    (high, first) = (h, (a, b));
                }
            }
        }
        let mean = div_round_half_even(sum, (count * count) as i64) as i32;
        if mean == high {
            // Level ground, or near enough that the summit changes nothing.
            return high;
        }
        // A summit: no sample of the eight cells around is higher, and none
        // that comes before this cell's first highest (`z`, then `x`) is as
        // high. The cells the highest sample leans towards are asked first:
        // on a slope the first sample uphill ends the search.
        let lean = (2 * first.0 - (count - 1), 2 * first.1 - (count - 1));
        let mut around = [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ];
        around.sort_by_key(|&(ci, cj)| -(ci * lean.0 + cj * lean.1));
        for (ci, cj) in around {
            for b in 0..count {
                for a in 0..count {
                    let h = sample(ci, cj, a, b);
                    let before = cj < 0 || (cj == 0 && (b < first.1 || (b == first.1 && ci < 0)));
                    if h > high || (h == high && before) {
                        return mean;
                    }
                }
            }
        }
        // A summit stands at the mean of its highest half-width cell: never
        // under its own mean (a mean of means), never over its highest
        // column. At two samples a side the halves are single columns, so
        // level 1 keeps the highest column itself. Every half has samples:
        // a cell whose mean is not its highest has at least two a side.
        (0..4)
            .map(|q| div_round_half_even(quarter[q], quarter_count[q]) as i32)
            .max()
            .expect("four halves")
    }

    /// How an `n` x `n` sampling of a cell `w` blocks wide is laid out: the
    /// samples to a side (`n`, but no more than `w`), the blocks between
    /// them (`w / n` rounded down, at least one) and the blocks from the
    /// cell's corner to the first (half of what one sample stands for, so
    /// the grid is centred in the cell).
    fn sampling(w: i32, n: i32) -> (i32, i32, i32) {
        let w = w.max(1);
        let count = n.clamp(1, w);
        let step = w / count;
        (count, step, (step - 1) / 2)
    }

    /// The columns an `n` x `n` sampling of a cell `w` blocks wide from
    /// `(x0, z0)` visits ([`sampling`](Self::sampling)): every column when
    /// `n >= w`. Shared by [`cell_top_sampled`](Self::cell_top_sampled) and
    /// [`ground_faces`](Self::ground_faces), so the top and the faces of one
    /// cell are judged on the same columns.
    fn sample_columns(x0: i32, z0: i32, w: i32, n: i32) -> impl Iterator<Item = (i32, i32)> {
        let (count, step, lead) = Self::sampling(w, n);
        (0..count).flat_map(move |b| {
            (0..count).map(move |a| (x0 + lead + a * step, z0 + lead + b * step))
        })
    }

    /// What level 0 shows over the columns `[x0, x0 + w) x [z0, z0 + w)`:
    /// see [`GroundFaces`]. An `n` x `n` grid of them is counted -- the
    /// columns [`cell_top_sampled`](Self::cell_top_sampled) visits, all of
    /// them when `n >= w` -- each with the eight columns around it: about
    /// `9 * n^2` surface heights sampled, `(w + 2)^2` exact.
    ///
    /// A riser is counted for the column it belongs to -- the higher one --
    /// whichever side of the square's edge its lower neighbour is on.
    ///
    /// Pure, like everything here: the far terrain's wall colours and any
    /// other aggregate of the ground are this, never a second rule that
    /// happens to agree (`ARCHITECTURE.md` Rule 5).
    pub fn ground_faces(&self, x0: i32, z0: i32, w: i32, n: i32) -> GroundFaces {
        // Heights are asked for more than once per column (a column is its
        // neighbours' neighbour); when every column is counted they are all
        // in the square and one ring around it, so they are worked out once.
        let exact = n >= w;
        let side = (w + 2) as usize;
        let grid: Vec<GroundColumn> = if exact {
            (0..side * side)
                .map(|k| self.column(x0 - 1 + (k % side) as i32, z0 - 1 + (k / side) as i32))
                .collect()
        } else {
            Vec::new()
        };
        let column = |x: i32, z: i32| -> GroundColumn {
            if exact {
                grid[(z - z0 + 1) as usize * side + (x - x0 + 1) as usize]
            } else {
                self.column(x, z)
            }
        };

        let mut out = GroundFaces::default();
        let (mut top_open, mut side_open) = (0f64, [0f64; 4]);
        for (x, z) in Self::sample_columns(x0, z0, w, n) {
            {
                // The nine tops this column's faces can be occluded by.
                let own = column(x, z);
                let mut near = [[0i32; 3]; 3];
                for (dz, row) in near.iter_mut().enumerate() {
                    for (dx, h) in row.iter_mut().enumerate() {
                        *h = if (dx, dz) == (1, 1) {
                            own.top
                        } else {
                            column(x + dx as i32 - 1, z + dz as i32 - 1).top
                        };
                    }
                }
                let h = own.top;
                // The ground as blocks: solid at and under a column's top.
                let solid = |bx: i32, by: i32, bz: i32| -> bool {
                    by <= near[(bz - z + 1) as usize][(bx - x + 1) as usize]
                };
                out.columns += 1;
                // The top face: the mesher's own sweep axes for `d = 1`.
                let ao = cubara_voxel::face_ao_of(&solid, 1, 2, 0, h + 1, z, x);
                top_open += ao.iter().map(|&a| a as f64).sum::<f64>() / 12.0;
                for (d, [dx, dz]) in GROUND_DIRECTIONS.into_iter().enumerate() {
                    let below = near[(1 + dz) as usize][(1 + dx) as usize];
                    for y in below + 1..=h {
                        out.sides[d][Self::ground_layer(own.surface - y) as usize] += 1.0;
                        let ao = if dx != 0 {
                            cubara_voxel::face_ao_of(&solid, 0, 1, 2, x + dx, y, z)
                        } else {
                            cubara_voxel::face_ao_of(&solid, 2, 0, 1, z + dz, x, y)
                        };
                        side_open[d] += ao.iter().map(|&a| a as f64).sum::<f64>() / 12.0;
                    }
                }
            }
        }
        out.top_open = if out.columns > 0 {
            (top_open / out.columns as f64) as f32
        } else {
            1.0
        };
        for (d, open) in side_open.into_iter().enumerate() {
            let area: f32 = out.sides[d].iter().sum();
            out.side_open[d] = if area > 0.0 {
                (open / area as f64) as f32
            } else {
                1.0
            };
        }
        out
    }

    /// Every tree whose trunk chunk column `node` covers, standing on the
    /// top of the cell of this node that draws the ground under it
    /// ([`cell_top`](Self::cell_top); issue #332, R2 decision 1, and #423)
    /// rather than on its own column's surface -- the same sites
    /// [`tree_site`](Self::tree_site)
    /// places at level 0, so a tree never pops in or out crossing a ring
    /// edge. `None` at level 0: its real voxel trees are grown by
    /// [`block_at`](Self::block_at)/[`is_solid_with_trees`](Self::is_solid_with_trees)
    /// already (decision 1's "one source: no second placement rule" --
    /// this is not a second one, it is the same `tree_site` evaluated over a
    /// node's footprint instead of one chunk column at a time).
    ///
    /// Pure, like every other placement in this file (§8.1): `cubara-app`'s
    /// streaming glue calls this once per resident coarse node and hands the
    /// positions to `cubara-render`, which never calls into this crate
    /// (`ARCHITECTURE.md` Rule 3).
    pub fn node_trees(&self, node: crate::node::NodeKey, blocks: TerrainBlocks) -> Vec<TreeSite> {
        if node.level == 0 {
            return Vec::new();
        }
        let w = node.extent_chunks();
        let origin = node.chunk_origin();
        let extent = node.extent_chunks();
        let mut out = Vec::new();
        for cz in origin.z..origin.z + extent {
            for cx in origin.x..origin.x + extent {
                let Some(site) = self.tree_site(cx, cz, blocks) else {
                    continue;
                };
                // The cell of this node that draws the ground under the trunk.
                let (cx, cz) = (site.x.div_euclid(w) * w, site.z.div_euclid(w) * w);
                let top = self.cell_top(cx, cz, w);
                out.push(TreeSite {
                    base: top + 1,
                    ..site
                });
            }
        }
        out
    }

    /// Whether the block at `(x, y, z)` is cave: where the cave noise
    /// crosses the threshold. **A block, and only a block**: a coarse cell is
    /// cave where every block of ground in it is
    /// ([`row_is_cave`](Self::row_is_cave)), which asks this of each.
    fn cave_at(&self, x: i32, y: i32, z: i32) -> bool {
        fbm3(
            self.seed ^ CAVE_SEED_MIX,
            x as f32 * CAVE_FREQ,
            y as f32 * CAVE_FREQ,
            z as f32 * CAVE_FREQ,
            CAVE_OCTAVES,
            CAVE_LACUNARITY,
            CAVE_GAIN,
        ) > CAVE_THRESHOLD
    }

    /// Terrain density at a world position: positive means solid. The
    /// height field minus a threshold-gated 3D cave noise field (§8.3) -- a
    /// pure function of `(seed, x, y, z)` alone, so no chunk this feeds
    /// depends on any other chunk, generated in any order (§8.1). Caves are
    /// noise, not carved tunnels, specifically so a cave crossing a chunk
    /// boundary agrees with itself on both sides for free.
    ///
    /// `+ 1` on the height-field term: `surface_height` names the *last
    /// solid row*, so at `y == surface_height` density must already be
    /// positive (solid), not exactly zero -- zero is reserved for the first
    /// *air* row, one above the surface. Without the offset "positive means
    /// solid" and "the surface block is solid" contradict each other by
    /// exactly one block.
    pub fn density(&self, x: i32, y: i32, z: i32) -> f32 {
        self.density_at(x, y, z, self.surface_height(x, z))
    }

    /// Shorthand for `density(x, y, z) > 0.0` -- what a single-block
    /// solidity query (a raycast step, [`World::is_solid_at`]) needs,
    /// without pulling in [`TerrainBlocks`] just to ask a yes/no question.
    ///
    /// [`World::is_solid_at`]: crate::World::is_solid_at
    pub fn is_solid(&self, x: i32, y: i32, z: i32) -> bool {
        self.density(x, y, z) > 0.0
    }

    /// The material a solid position should render as, given an
    /// already-known `surface_height(x, z)` (see [`density_at`](Self::density_at)):
    /// `blocks.grass` at the surface, `blocks.soil` for the next
    /// [`SOIL_DEPTH`] blocks, `blocks.stone` deeper than that -- including
    /// cave walls, which sit far enough below the surface height that this
    /// never mistakes one for topsoil (`SOIL_DEPTH` is shallow, caves are
    /// not near the surface except at the rare mouth where a tunnel
    /// breaches it, and a breach is exactly where grass at a cave wall
    /// would be correct anyway).
    fn material_at(&self, x: i32, y: i32, z: i32, surface: i32, blocks: TerrainBlocks) -> BlockId {
        match Self::ground_layer(surface - y) {
            GroundLayer::Surface => blocks.grass,
            GroundLayer::Soil => blocks.soil,
            GroundLayer::Stone => self.ore_or(x, y, z, blocks.stone, blocks.ores),
        }
    }

    /// `stone`, or an ore that replaces it at this position
    /// (`PHASE2_ARCHITECTURE.md` §6).
    ///
    /// **This is a material substitution and nothing else.** It is reached
    /// only where the caller has already decided the voxel is solid and deep,
    /// and it returns a solid block in every case -- so adding an ore cannot
    /// change any world's shape, which is what lets `generate` (LOD) call it
    /// safely where the structure pass deliberately is not called at all.
    ///
    /// Returns early on an empty set, which is every test that predates block
    /// 2.3b and every world built without `with_ores`: no noise is sampled, so
    /// an ore-free world costs exactly what it did before.
    fn ore_or(&self, x: i32, y: i32, z: i32, stone: BlockId, ores: OreSet) -> BlockId {
        if ores.is_empty() {
            return stone;
        }
        for ore in ores.iter() {
            if ore.replaces != stone || y > ore.max_y {
                continue;
            }
            let freq = ore.freq_milli as f32 * 0.001;
            let n = fbm3(
                self.seed ^ ORE_SEED_MIX ^ ore.seed_mix,
                x as f32 * freq,
                y as f32 * freq,
                z as f32 * freq,
                ORE_OCTAVES,
                ORE_LACUNARITY,
                ORE_GAIN,
            );
            if n > ore.threshold_milli as f32 * 0.001 {
                return ore.block;
            }
        }
        stone
    }

    /// The trees whose volume can reach anything in this chunk column: at most
    /// one per neighbouring chunk column, so at most nine.
    ///
    /// **Why this takes a chunk column rather than a voxel.** Which trees can
    /// reach a voxel depends only on its chunk, not on the voxel, so this is
    /// computed *once per chunk* by `World::build_chunk` and once per query by
    /// the single-voxel paths. One rule, hoisted differently -- a naive
    /// per-voxel version would cost nine hashes 8,192 times per chunk and blow
    /// `radius_64_smoke`'s budget.
    ///
    /// Nine is the bound because a tree reaches at most `canopy_radius` blocks
    /// sideways, and one chunk is 16 -- so only immediately adjacent chunk
    /// columns can ever overlap this one, and each carries at most one tree.
    pub(crate) fn trees_near(&self, chunk_x: i32, chunk_z: i32, blocks: TerrainBlocks) -> TreeSet {
        let mut out = TreeSet::new();
        if blocks.oak.is_none() {
            return out;
        }
        for cz in chunk_z - 1..=chunk_z + 1 {
            for cx in chunk_x - 1..=chunk_x + 1 {
                if let Some(t) = self.tree_in_column(cx, cz, blocks) {
                    out.push(t);
                }
            }
        }
        out
    }

    /// The tree that grows in chunk column `(chunk_x, chunk_z)`, if any: the
    /// same placement `trees_near` collects nine of, so what this reports is
    /// exactly what generation puts in the world.
    pub fn tree_site(&self, chunk_x: i32, chunk_z: i32, blocks: TerrainBlocks) -> Option<TreeSite> {
        let oak = blocks.oak?;
        let t = self.tree_in_column(chunk_x, chunk_z, blocks)?;
        Some(TreeSite {
            x: t.x,
            z: t.z,
            base: t.base,
            height: t.height,
            canopy_radius: oak.canopy_radius,
        })
    }

    /// The tree in one chunk column, if it has one.
    ///
    /// Integer arithmetic only: the placement decision has to produce the same
    /// bits on every platform (§8.5), and float comparisons are exactly what
    /// that rule exists to avoid.
    fn tree_in_column(
        &self,
        chunk_x: i32,
        chunk_z: i32,
        blocks: TerrainBlocks,
    ) -> Option<PlacedTree> {
        let oak = blocks.oak?;
        // A separate hash stream from the terrain's, so retuning one cannot
        // silently move the other.
        let h = crate::noise::hash(self.seed ^ 0x7EE5, chunk_x, 0, chunk_z);
        if !h.is_multiple_of(oak.density as u64) {
            return None;
        }
        // Where in the column, kept clear of the chunk edge by the canopy
        // radius so a tree never reaches further than the one chunk of slack
        // `trees_near` allows for.
        let size = Chunk::SIZE as i32;
        let margin = oak.canopy_radius;
        let span = (size - margin * 2).max(1) as u64;
        let x = chunk_x * size + margin + ((h >> 8) % span) as i32;
        let z = chunk_z * size + margin + ((h >> 24) % span) as i32;

        let surface = self.surface_height(x, z);
        // Only on the block it grows on, which keeps trees off stone and out
        // of caves -- and only where that surface block is actually there,
        // since a cave can carve the ground out from under it.
        // The real surface material, from the real palette.
        //
        // An earlier version compared against a *synthetic* palette whose
        // `grass` was already `grows_on` -- which made the check vacuous: it
        // could only ever pass. `trees_do_not_grow_on_stone_or_in_caves`
        // caught it, which is exactly what that test is for.
        //
        // The density check is the second half: a cave can carve the ground
        // out from under a surface height, and a tree must not grow on air.
        if self.material_at(x, surface, z, surface, blocks) != oak.grows_on
            || self.density_at(x, surface, z, surface) <= 0.0
        {
            return None;
        }

        let (lo, hi) = oak.height;
        let height = lo + ((h >> 40) % (hi - lo + 1).max(1) as u64) as i32;
        Some(PlacedTree {
            x,
            z,
            base: surface + 1,
            height,
        })
    }

    /// Whether one of `trees` occupies `(x, y, z)`, and with what.
    ///
    /// Trunk wins over canopy where they overlap, so the trunk reads as a
    /// continuous column rather than being swallowed by its own leaves.
    ///
    /// The single-tree shape test (trunk column, rounded canopy blob) is
    /// [`cubara_voxel::tree_part_at`], not duplicated here (issue #332) --
    /// this function's own job is only the part that needs trees' absolute
    /// positions: which of several overlapping trees wins where they
    /// disagree.
    pub(crate) fn tree_block_at(
        &self,
        trees: &TreeSet,
        x: i32,
        y: i32,
        z: i32,
        oak: Oak,
    ) -> Option<BlockId> {
        let mut leaves = false;
        for t in trees.iter() {
            let (dx, dz, dy) = ((x - t.x).abs(), (z - t.z).abs(), y - t.base);
            match cubara_voxel::tree_part_at(dx, dz, dy, t.height, oak.canopy_radius) {
                Some(cubara_voxel::TreePart::Trunk) => return Some(oak.log),
                Some(cubara_voxel::TreePart::Canopy) => leaves = true,
                None => {}
            }
        }
        leaves.then_some(oak.leaves)
    }

    /// The block at a world position given an already-known
    /// `surface_height(x, z)`, or `None` for air: one block of level 0.
    /// (A coarse cell is [`coarse_block`](Self::coarse_block).)
    fn block_at_on_surface(
        &self,
        x: i32,
        y: i32,
        z: i32,
        surface: i32,
        blocks: TerrainBlocks,
    ) -> Option<BlockId> {
        (self.density_at(x, y, z, surface) > 0.0)
            .then(|| self.material_at(x, y, z, surface, blocks))
    }

    /// The block at a world position, or `None` for air. Exposed for
    /// callers that only need one block (a `World` cell with no edit
    /// overriding it) rather than a whole chunk; [`generate`](Self::generate)
    /// does not call this; see its doc comment for why.
    /// The terrain at a position, **without** consulting structures.
    ///
    /// What a caller uses when it has already resolved the trees for a whole
    /// chunk once (`World::build_chunk`) -- so the nine-hash structure lookup
    /// happens per chunk rather than per voxel. Getting that wrong is not a
    /// correctness bug, it is a 5x frame-time regression, which is how it was
    /// found.
    /// Whether the generated block at `(x, y, z)` is solid, given its
    /// column's `surface_height` -- [`is_solid`](Self::is_solid) for a
    /// caller that has the height already.
    pub fn solid_on(&self, x: i32, y: i32, z: i32, surface: i32) -> bool {
        self.density_at(x, y, z, surface) > 0.0
    }

    /// The lowest and highest value [`surface_height`](Self::surface_height)
    /// can return, from the constants that shape it: `base ± amplitude` for
    /// the rolling land (fbm is normalised into [-1, 1]), and the mountains
    /// only ever add, up to `uplift + peak`. What a far-terrain patch whose
    /// heights are not generated yet is assumed to span (`crate::far`).
    pub fn surface_range() -> (i32, i32) {
        let amplitude = TERRAIN_AMPLITUDE.ceil() as i32;
        (
            TERRAIN_BASE_HEIGHT - amplitude,
            TERRAIN_BASE_HEIGHT + amplitude + (MOUNTAIN_UPLIFT + MOUNTAIN_PEAK).ceil() as i32,
        )
    }

    /// The lowest and highest value [`column_top`](Self::column_top) can
    /// return: [`surface_range`](Self::surface_range), its floor lowered by
    /// the deepest pit floor [`pit_floor`](Self::pit_floor) finds. Every
    /// level's top -- a mean of these, or a summit among them -- is one of
    /// them plus one, so no ground any level draws stands outside it
    /// (`crate::far::ground_range`).
    pub fn column_top_range() -> (i32, i32) {
        let (lo, hi) = Self::surface_range();
        (lo - PIT_FLOOR_SEARCH, hi)
    }

    /// No block the generator places is ever higher than this: the tallest
    /// surface the height field can produce, plus the tallest tree on it.
    ///
    /// What lets a render distance with no height limit skip the sky: a node
    /// wholly above this, with no edits in it, is air without generating a
    /// single cell. Derived from the constants and the tree data rather than
    /// written down, so tuning either moves it; `nothing_generates_above_the_highest_y`
    /// holds it to the truth.
    pub fn highest_generated_y(blocks: TerrainBlocks) -> i32 {
        // The height field is `base + fbm * amplitude`, and fbm is normalised
        // into [-1, 1].
        let surface = TERRAIN_BASE_HEIGHT
            + TERRAIN_AMPLITUDE.ceil() as i32
            + (MOUNTAIN_UPLIFT + MOUNTAIN_PEAK).ceil() as i32
            + 1;
        match blocks.oak {
            // A trunk starts one above the surface; the canopy reaches one
            // above the trunk's top. One more for good measure.
            Some(oak) => surface + 1 + oak.height.1 + oak.canopy_radius + 2,
            None => surface,
        }
    }

    pub fn terrain_block_at(
        &self,
        x: i32,
        y: i32,
        z: i32,
        blocks: TerrainBlocks,
    ) -> Option<BlockId> {
        self.block_at_on_surface(x, y, z, self.surface_height(x, z), blocks)
    }

    pub fn block_at(&self, x: i32, y: i32, z: i32, blocks: TerrainBlocks) -> Option<BlockId> {
        if let Some(oak) = blocks.oak {
            let size = Chunk::SIZE as i32;
            let trees = self.trees_near(x.div_euclid(size), z.div_euclid(size), blocks);
            // Trees sit *on* the terrain, so they win where they overlap air
            // and lose to nothing -- a trunk is never inside solid ground,
            // because it starts one block above the surface.
            if let Some(block) = self.tree_block_at(&trees, x, y, z, oak) {
                return Some(block);
            }
        }
        self.block_at_on_surface(x, y, z, self.surface_height(x, z), blocks)
    }

    /// Whether `(x, y, z)` is solid **including trees**, given the ids that
    /// say what a tree is made of.
    ///
    /// Separate from [`is_solid`](Self::is_solid) because that one answers a
    /// yes/no question without needing a [`TerrainBlocks`], and most callers
    /// (the density field itself) genuinely do not want trees. Walking and
    /// raycasting do: you must not walk through a trunk you can see.
    pub fn is_solid_with_trees(&self, x: i32, y: i32, z: i32, blocks: TerrainBlocks) -> bool {
        if let Some(oak) = blocks.oak {
            let size = Chunk::SIZE as i32;
            let trees = self.trees_near(x.div_euclid(size), z.div_euclid(size), blocks);
            if !trees.is_empty() && self.tree_block_at(&trees, x, y, z, oak).is_some() {
                return true;
            }
        }
        self.is_solid(x, y, z)
    }

    /// Fill a `16³` lattice of samples, `step` world-blocks apart starting
    /// at `origin` -- `step = 1` for a full-resolution chunk, `step = 8` for
    /// a level-3 LOD node sampling the same fixed lattice across a 128³
    /// volume (§6.2's "every node is sampled on a fixed 16³ lattice").
    /// Nothing this returns depends on any other chunk having been
    /// generated first, in any order (§8.1) -- pinned by the isolation
    /// test, `tests::generation_is_isolated_from_neighbours` below.
    ///
    /// Returns a [`Chunk`], not the `ChunkStorage` §8.1's signature sketch
    /// names: `ChunkStorage::from_ids` is a deliberately crate-private fast
    /// path inside `cubara-voxel` (see its own doc comment), and
    /// `Chunk::from_fn` is the existing public entry point that already
    /// does exactly this -- no new hole needed in `cubara-voxel`'s
    /// encapsulation for it.
    ///
    /// Not yet called from the live streamed scene at a non-unit `step` --
    /// `World::chunk_at` always requests `step = 1` today, and
    /// `Chunk::build_mesh_lod` still downsamples an already-generated
    /// full-resolution chunk for distant LOD. Wiring far nodes to call this
    /// directly at their target `step` (skipping the full-resolution
    /// generation `step` exists to avoid) is block 1.10's region node tree,
    /// not this one -- this block's job is to make `step` correct and
    /// tested, ready for that to call.
    ///
    /// Precomputes `surface_height` once per `(x, z)` column (256 calls, not
    /// one per voxel) before sampling the lattice: the naive per-voxel
    /// version called it up to twice per voxel (once for density, once for
    /// material) -- 8,192 calls per chunk of the *same* 16 values, each one
    /// several octaves of hashing. That redundancy was the actual cause of
    /// a real regression the radius-64 smoke test caught (a debug-build
    /// generation pass that used to finish in ~1.3s no longer finished in
    /// the test's 120s budget) -- see `BENCHMARKS.md`.
    /// **No trees here, deliberately.** `PHASE1_ARCHITECTURE.md` §8.4:
    /// structures run at level 0 only, because "a tree sampled every 8 blocks
    /// is one stray voxel". This is the LOD path (`step > 1`), so growing trees
    /// in it would put single floating leaf blocks on the horizon. The two
    /// per-voxel routes -- `block_at` and `is_solid_with_trees` -- are the ones
    /// that do. They differ by design, and this comment is the design.
    pub fn generate(&self, origin: [i32; 3], step: i32, blocks: TerrainBlocks) -> Chunk {
        if step > 1 {
            // A coarse lattice of cubes: the one coarse rule, at `v = step`,
            // drawn whole (no cut at the tops). Only tests build one.
            let ground = self.node_ground(origin[0], origin[2], step);
            return Chunk::from_fn(|lx, ly, lz| {
                let x = origin[0] + lx as i32 * step;
                let y = origin[1] + ly as i32 * step;
                let z = origin[2] + lz as i32 * step;
                self.coarse_block([x, y, z], &ground, lz * Chunk::SIZE + lx, blocks, step)
                    .unwrap_or(BlockId::AIR)
            });
        }
        let mut surfaces = [[0i32; Chunk::SIZE]; Chunk::SIZE];
        for (lz, row) in surfaces.iter_mut().enumerate() {
            let z = origin[2] + lz as i32;
            for (lx, cell) in row.iter_mut().enumerate() {
                *cell = self.surface_height(origin[0] + lx as i32, z);
            }
        }
        Chunk::from_fn(|lx, ly, lz| {
            let x = origin[0] + lx as i32;
            let y = origin[1] + ly as i32;
            let z = origin[2] + lz as i32;
            self.block_at_on_surface(x, y, z, surfaces[lz][lx], blocks)
                .unwrap_or(BlockId::AIR)
        })
    }

    /// [`generate`](Self::generate), for a coarse (level > 0) node's
    /// anisotropic lattice (R1b, issue #327 decisions 1/2): cells stay 16
    /// wide/deep at `step_h` (`2^level`) blocks each, but the 16 tall rows
    /// become `16 * step_h / v` rows of `v` blocks each.
    ///
    /// Every cell is [`coarse_block`](Self::coarse_block) over the level-0
    /// columns it covers ([`node_ground`](Self::node_ground)): its top is
    /// [`cell_top`](Self::cell_top), what it is made of is what level 0's
    /// faces show at its rows, and it is cave where all of its ground is
    /// (issue #423). Caves are asked only near the surface, so a node stays
    /// cheap deep below it regardless of how small `v` is.
    ///
    /// **Rows taller than a block are cut off at the top** (`v > 1`, level
    /// 3): the node carries each column's exact top
    /// ([`AnisoChunk::with_tops`]) and is drawn up to it and no further, so
    /// its top face is where the ground is, to the block, as at every other
    /// level.
    ///
    /// No trees, for the same reason [`generate`](Self::generate) has none.
    pub fn generate_aniso(
        &self,
        origin: [i32; 3],
        step_h: i32,
        v: i32,
        blocks: TerrainBlocks,
    ) -> AnisoChunk {
        let ground = self.node_ground(origin[0], origin[2], step_h);
        self.generate_aniso_over(origin, step_h, v, blocks, &ground)
    }

    /// [`generate_aniso`](Self::generate_aniso) over the ground `ground`
    /// instead of the world's own: the one body of the two, so that a test
    /// can hand in a height field made by hand -- a flat one, say
    /// ([`NodeGround::over`]) -- and get the node the game would make of it.
    pub(crate) fn generate_aniso_over(
        &self,
        origin: [i32; 3],
        step_h: i32,
        v: i32,
        blocks: TerrainBlocks,
        ground: &NodeGround,
    ) -> AnisoChunk {
        let nx = Chunk::SIZE;
        let nz = Chunk::SIZE;
        let ny = ((Chunk::SIZE as i32 * step_h) / v).max(1) as usize;
        assert_eq!(
            ground.per_cell,
            (step_h * step_h) as usize,
            "the ground of a node of this cell width"
        );
        let chunk = AnisoChunk::from_fn([nx, ny, nz], [step_h, v, step_h], |lx, ly, lz| {
            let x = origin[0] + lx as i32 * step_h;
            let y = origin[1] + ly as i32 * v;
            let z = origin[2] + lz as i32 * step_h;
            self.coarse_block([x, y, z], ground, lz * Chunk::SIZE + lx, blocks, v)
                .unwrap_or(BlockId::AIR)
        });
        if v == 1 {
            // Rows one block tall end at the top already.
            return chunk;
        }
        // Rows of `v` blocks: the one that holds a cell's top is cut off
        // there, so the top face is where the ground is, to the block.
        let tops = ground
            .tops
            .iter()
            .map(|top| (top - origin[1]).clamp(i16::MIN as i32, i16::MAX as i32) as i16)
            .collect();
        chunk.with_tops(tops)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `World::new`'s seed: the one a player gets.
    const DEFAULT_TEST_SEED: u64 = 0x005E_ED00_00C0_FFEE;

    /// `surface_range` is what the far terrain trusts before it has generated
    /// anything (`crate::far`), so a surface outside it would be a patch
    /// culled or split on the wrong distance.
    #[test]
    fn the_surface_stays_inside_its_range() {
        let gen = WorldGen::new(DEFAULT_TEST_SEED);
        let (lo, hi) = WorldGen::surface_range();
        let (mut seen_lo, mut seen_hi) = (i32::MAX, i32::MIN);
        for x in (-20_000..20_000).step_by(97) {
            for z in (-20_000..20_000).step_by(89) {
                let h = gen.surface_height(x, z);
                assert!(
                    (lo..=hi).contains(&h),
                    "{h} at ({x}, {z}) outside {lo}..={hi}"
                );
                seen_lo = seen_lo.min(h);
                seen_hi = seen_hi.max(h);
            }
        }
        // And the range is not so loose it means nothing: the sample reaches
        // well into both ends of it.
        assert!(seen_hi > hi - 60, "highest seen {seen_hi}, range to {hi}");
        assert!(seen_lo < lo + 10, "lowest seen {seen_lo}, range from {lo}");
    }

    fn test_blocks() -> TerrainBlocks {
        TerrainBlocks {
            oak: None,
            ores: OreSet::EMPTY,
            grass: BlockId(1),
            soil: BlockId(2),
            stone: BlockId(3),
        }
    }

    fn chunks_equal(a: &Chunk, b: &Chunk) -> bool {
        for z in 0..Chunk::SIZE {
            for y in 0..Chunk::SIZE {
                for x in 0..Chunk::SIZE {
                    if a.get(x, y, z) != b.get(x, y, z) {
                        return false;
                    }
                }
            }
        }
        true
    }

    #[test]
    fn terrain_layers_by_depth() {
        // Block 1.4c's depth rule, carried over verbatim: grass at the
        // surface, soil for the next SOIL_DEPTH blocks, stone beyond that.
        // Tests `material_at` directly (not `block_at`/`generate`) so this
        // is exactly the depth mapping, with no cave noise able to
        // interfere by carving the probed cell to air.
        let gen = WorldGen::new(42);
        let blocks = test_blocks();
        let surface = gen.surface_height(0, 0);

        assert_eq!(
            gen.material_at(0, surface, 0, surface, blocks),
            blocks.grass,
            "surface"
        );
        for depth in 1..=SOIL_DEPTH {
            assert_eq!(
                gen.material_at(0, surface - depth, 0, surface, blocks),
                blocks.soil,
                "depth {depth}"
            );
        }
        assert_eq!(
            gen.material_at(0, surface - SOIL_DEPTH - 1, 0, surface, blocks),
            blocks.stone,
            "below the soil layer"
        );
    }

    #[test]
    fn surface_is_solid_and_the_block_above_is_air() {
        // The density/is_solid boundary should land exactly where
        // surface_height says, not off by one in either direction.
        let gen = WorldGen::new(7);
        let surface = gen.surface_height(3, -5);
        assert!(gen.is_solid(3, surface, -5), "the surface block is solid");
        assert!(
            !gen.is_solid(3, surface + 1, -5),
            "one block above the surface is air"
        );
    }

    /// Surface heights over an 8,192-block square, every 32 blocks.
    fn surface_sample(gen: WorldGen) -> Vec<i32> {
        (-4096..4096)
            .step_by(32)
            .flat_map(|x| {
                (-4096..4096)
                    .step_by(32)
                    .map(move |z| gen.surface_height(x, z))
            })
            .collect()
    }

    #[test]
    fn mountains_rise_in_ranges_with_hills_between() {
        // The owner asked for serious mountains, and to see them at scale:
        // peaks far above the hills, but as ranges -- most of the world is
        // still hills, and a range is a region, not noise on every column.
        for seed in [DEFAULT_TEST_SEED, 0x5EED, 119] {
            let heights = surface_sample(WorldGen::new(seed));
            let share = |f: &dyn Fn(i32) -> bool| {
                heights.iter().filter(|&&h| f(h)).count() as f64 / heights.len() as f64
            };
            let tallest = *heights.iter().max().unwrap();
            let hills = share(&|h| h <= TERRAIN_BASE_HEIGHT + TERRAIN_AMPLITUDE as i32);
            let high = share(&|h| h > 100);
            assert!(
                tallest > 180,
                "seed {seed}: the tallest peak is only {tallest}"
            );
            assert!(
                hills > 0.5,
                "seed {seed}: only {:.0}% is hills",
                hills * 100.0
            );
            assert!(
                (0.03..0.3).contains(&high),
                "seed {seed}: {:.1}% of the world is above y = 100",
                high * 100.0
            );
        }
    }

    #[test]
    fn mountains_have_cliffs() {
        // Ridged, not rolling: somewhere a single step sideways climbs several
        // blocks at once.
        let gen = WorldGen::new(DEFAULT_TEST_SEED);
        let steepest = (-2048..2048)
            .step_by(4)
            .flat_map(|x| (-2048..2048).step_by(4).map(move |z| (x, z)))
            .map(|(x, z)| (gen.surface_height(x + 1, z) - gen.surface_height(x, z)).abs())
            .max()
            .unwrap();
        assert!(steepest >= 5, "the steepest step is {steepest} blocks");
    }

    #[test]
    fn a_new_world_starts_in_the_hills_with_mountains_in_view() {
        // What `MOUNTAIN_SEED_MIX` was picked for: the default seed's spawn is
        // not on a peak (iron, which lies below y = 40, stays within reach, and
        // the tests framed on the origin keep their terrain), and a range is
        // close enough to see.
        //
        // This is about the *shape of the land* around the origin. That the
        // spawn point itself stands on the surface under open sky (not in a
        // cave pocket) is `crates/server/tests/spawn.rs`'s job, over 1000
        // seeds, and the game's start-up path is checked by
        // `the_game_does_not_start_by_killing_the_player` in the app.
        let gen = WorldGen::new(DEFAULT_TEST_SEED);
        let mut nearest_peak = f64::MAX;
        for x in (-640..=640).step_by(8) {
            for z in (-640..=640).step_by(8) {
                let h = gen.surface_height(x, z);
                let d = ((x * x + z * z) as f64).sqrt();
                if d <= 320.0 {
                    assert!(h <= 40, "a mountain at ({x}, {z}), {d:.0} from spawn: {h}");
                }
                if h > 120 {
                    nearest_peak = nearest_peak.min(d);
                }
            }
        }
        assert!(
            nearest_peak < 600.0,
            "the nearest range is {nearest_peak:.0} blocks off"
        );
    }

    #[test]
    fn different_seeds_produce_different_terrain() {
        let a = WorldGen::new(1).surface_height(0, 0);
        let b = WorldGen::new(2).surface_height(0, 0);
        assert_ne!(
            a, b,
            "two different seeds landed on the same height by coincidence"
        );
    }

    /// The sky shortcut skips generating anything above this, so it has to be
    /// true: over a wide area with trees, nothing solid is ever above it --
    /// and the tallest thing found is close to it, or the bound is so loose it
    /// saves nothing.
    #[test]
    fn nothing_generates_above_the_highest_y() {
        let gen = WorldGen::new(0x5EED);
        let blocks = tree_blocks();
        let top = WorldGen::highest_generated_y(blocks);
        let mut tallest = i32::MIN;
        let check_column = |x: i32, z: i32, tallest: &mut i32| {
            let surface = gen.surface_height(x, z);
            for y in (surface - 2)..=(top + 4) {
                if gen.block_at(x, y, z, blocks).is_some() {
                    assert!(y <= top, "solid at ({x}, {y}, {z}), above {top}");
                    *tallest = (*tallest).max(y);
                }
            }
        };
        for x in (-400..400).step_by(3) {
            for z in (-400..400).step_by(3) {
                check_column(x, z, &mut tallest);
            }
        }
        // Mountains are hundreds of blocks apart, so the peaks have to be
        // looked for over a wider area: the tallest columns of it, and the
        // trees that may stand on them.
        let mut peaks: Vec<(i32, i32, i32)> = (-4096..4096)
            .step_by(16)
            .flat_map(|x| (-4096..4096).step_by(16).map(move |z| (x, z)))
            .map(|(x, z)| (gen.surface_height(x, z), x, z))
            .collect();
        peaks.sort_unstable_by(|a, b| b.cmp(a));
        for &(_, px, pz) in peaks.iter().take(8) {
            for x in (px - 16)..=(px + 16) {
                for z in (pz - 16)..=(pz + 16) {
                    check_column(x, z, &mut tallest);
                }
            }
        }
        // Looser than it was before mountains (12): the ridged noise reaches
        // its maximum only where every octave crests at once, which a
        // 8,192-block square measured 19 short of (239 against 258).
        assert!(
            top - tallest <= 40,
            "tallest {tallest} but the bound is {top}: too loose"
        );
    }

    /// A terrain palette with trees, for the structure-pass tests.
    fn tree_blocks() -> TerrainBlocks {
        TerrainBlocks {
            oak: Some(Oak {
                log: BlockId(10),
                leaves: BlockId(11),
                grows_on: BlockId(1),
                height: (4, 6),
                canopy_radius: 2,
                density: 4,
            }),
            ores: OreSet::EMPTY,
            grass: BlockId(1),
            soil: BlockId(2),
            stone: BlockId(3),
        }
    }

    /// Where this seed actually puts a trunk, found by looking rather than
    /// assumed: a hardcoded coordinate would silently stop testing anything
    /// the moment the terrain noise is retuned.
    fn find_a_trunk(gen: &WorldGen, blocks: TerrainBlocks) -> (i32, i32, i32) {
        let oak = blocks.oak.expect("this fixture has trees");
        for cz in -4..4 {
            for cx in -4..4 {
                let trees = gen.trees_near(cx, cz, blocks);
                for x in cx * 16..(cx + 1) * 16 {
                    for z in cz * 16..(cz + 1) * 16 {
                        let surface = gen.surface_height(x, z);
                        for y in surface..surface + 10 {
                            if gen.tree_block_at(&trees, x, y, z, oak) == Some(oak.log) {
                                return (x, y, z);
                            }
                        }
                    }
                }
            }
        }
        panic!("this seed grows no trees at all -- density is wrong");
    }

    #[test]
    fn trees_grow_and_are_solid() {
        // The walk-through-a-trunk bug. `is_solid` answers from the density
        // field, which knows nothing about trees, so the tree-aware query is
        // the one physics and raycasting have to use.
        let gen = WorldGen::new(0x77EE_5EED);
        let blocks = tree_blocks();
        let (x, y, z) = find_a_trunk(&gen, blocks);

        assert_eq!(
            gen.block_at(x, y, z, blocks),
            Some(blocks.oak.unwrap().log),
            "the trunk is there"
        );
        assert!(
            gen.is_solid_with_trees(x, y, z, blocks),
            "and you cannot walk through it"
        );
        assert!(
            !gen.is_solid(x, y, z),
            "while the raw density field still says air -- which is exactly why              the tree-aware query has to exist"
        );
    }

    #[test]
    fn the_same_seed_grows_the_same_trees() {
        let blocks = tree_blocks();
        let survey = |g: &WorldGen| {
            (-3..3)
                .flat_map(|cx| (-3..3).map(move |cz| (cx, cz)))
                .map(|(cx, cz)| g.trees_near(cx, cz, blocks).count_for_test())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            survey(&WorldGen::new(0x77EE_5EED)),
            survey(&WorldGen::new(0x77EE_5EED)),
            "the same seed grows the same trees"
        );
        assert_ne!(
            survey(&WorldGen::new(0x77EE_5EED)),
            survey(&WorldGen::new(0x77EE_5EEE)),
            "a different seed grows different ones -- otherwise the seed is not              reaching placement at all"
        );
    }

    #[test]
    fn trees_do_not_grow_on_stone_or_in_caves() {
        // `grows_on` is the whole rule. Point it at a block no surface ever
        // is, and nothing may grow anywhere.
        let gen = WorldGen::new(0x77EE_5EED);
        let mut blocks = tree_blocks();
        blocks.oak = Some(Oak {
            grows_on: BlockId(99),
            ..blocks.oak.unwrap()
        });
        for cz in -3..3 {
            for cx in -3..3 {
                assert!(
                    gen.trees_near(cx, cz, blocks).is_empty(),
                    "a tree grew on a surface it does not grow on, at chunk ({cx}, {cz})"
                );
            }
        }
    }

    #[test]
    fn lod_nodes_contain_no_tree_blocks() {
        // §8.4: structures run at level 0 only, because a tree sampled every
        // eight blocks is one stray voxel. `generate` is the LOD path.
        let gen = WorldGen::new(0x77EE_5EED);
        let blocks = tree_blocks();
        let oak = blocks.oak.unwrap();
        let node = gen.generate([0, 0, 0], 8, blocks);
        for z in 0..Chunk::SIZE {
            for y in 0..Chunk::SIZE {
                for x in 0..Chunk::SIZE {
                    let b = node.get(x, y, z);
                    assert_ne!(b, oak.log, "a log appeared in an LOD node");
                    assert_ne!(b, oak.leaves, "leaves appeared in an LOD node");
                }
            }
        }
    }

    #[test]
    fn node_trees_is_empty_at_level_zero() {
        // Level 0 already grows real voxel trees (`block_at`); `node_trees`
        // must not place a second, independent set on top of them.
        let gen = WorldGen::new(0x77EE_5EED);
        let blocks = tree_blocks();
        let node = crate::node::NodeKey::new(0, [0, 0, 0]);
        assert!(gen.node_trees(node, blocks).is_empty());
    }

    #[test]
    fn node_trees_is_exactly_one_per_tree_bearing_column_in_the_footprint() {
        let gen = WorldGen::new(0x77EE_5EED);
        let blocks = tree_blocks();
        let node = crate::node::NodeKey::new(2, [1, 0, -1]); // level 2, extent 4
        let trees = gen.node_trees(node, blocks);

        let origin = node.chunk_origin();
        let extent = node.extent_chunks();
        let mut expected = 0usize;
        for cz in origin.z..origin.z + extent {
            for cx in origin.x..origin.x + extent {
                if gen.tree_site(cx, cz, blocks).is_some() {
                    expected += 1;
                }
            }
        }
        assert_eq!(
            trees.len(),
            expected,
            "wrong count of trees in the footprint"
        );
        // Every tree's (x, z) is genuinely inside the node's own chunk
        // footprint, in blocks -- a tree outside it would be a column some
        // *other* node is responsible for, drawn twice.
        let size = Chunk::SIZE as i32;
        for t in &trees {
            assert!(
                (origin.x * size..(origin.x + extent) * size).contains(&t.x),
                "{t:?} outside the node's own x span"
            );
            assert!(
                (origin.z * size..(origin.z + extent) * size).contains(&t.z),
                "{t:?} outside the node's own z span"
            );
        }
    }

    /// **A tree stands on the ground its node draws under it** (issue #332,
    /// R2 decision 1; issue #423): at every coarse level its base is one
    /// above the top of the cell that holds its trunk -- the mean of the
    /// tops of the level-0 columns that cell covers (what the sky sees of
    /// each: a tree beside a pit stands in the dent its cell draws, §9.9),
    /// or its summit -- and not one above its own column's surface, which
    /// the cell's top face can stand over or under. Held on every tree of
    /// sixteen node columns a level, and each level must show trees for
    /// which the two differ, or the test could not tell them apart.
    #[test]
    fn a_tree_stands_on_the_top_of_the_cell_that_draws_its_ground() {
        let gen = WorldGen::new(0x77EE_5EED);
        let blocks = tree_blocks();
        for level in 1..=3u32 {
            let (mut trees, mut lifted) = (0, 0);
            for nx in -2..2 {
                for nz in -2..2 {
                    let node = crate::node::NodeKey::new(level, [nx, 0, nz]);
                    let w = node.extent_chunks();
                    for t in gen.node_trees(node, blocks) {
                        let (cx, cz) = (t.x.div_euclid(w) * w, t.z.div_euclid(w) * w);
                        let (top, _, _, _) = top_by_hand(&|x, z| gen.column_top(x, z), cx, cz, w);
                        assert_eq!(
                            t.base,
                            top + 1,
                            "level {level}: the tree at ({}, {}) does not stand on its cell's top",
                            t.x,
                            t.z
                        );
                        trees += 1;
                        if t.base != gen.surface_height(t.x, t.z) + 1 {
                            lifted += 1;
                        }
                    }
                }
            }
            assert!(
                trees >= 8 && lifted >= 2,
                "level {level}: {trees} trees, {lifted} of them on a cell top that is not their own column's"
            );
        }
    }

    /// **T332-edge**, the issue's own bar: a tree site exactly at the 160 m
    /// edge is a tree on both sides of a level-0/level-1 swap. Level 0 draws
    /// it as real voxel blocks (`tree_site`, the same rule `block_at` uses);
    /// a level-1 node covering the same chunk column must grow the identical
    /// tree (`node_trees`) -- same column, same height, same standing base
    /// (the ground its cell draws) -- or crossing that ring edge pops a tree
    /// in or out.
    #[test]
    fn a_tree_at_the_160m_edge_is_a_tree_on_both_sides_of_the_swap() {
        use crate::node::NodeKey;
        use cubara_voxel::ChunkCoord;

        let gen = WorldGen::new(0x77EE_5EED);
        let blocks = tree_blocks();
        // `node::DEFAULT_RING_SCHEDULE`'s first ring ends at chunk-radius 10
        // (160 m): level 0 up to there, level 1 picks up past it. Trees are
        // sparse (one in `density` columns), so search outward from the edge
        // chunk for one that actually has a tree.
        let (cx, cz) = (9..16)
            .flat_map(|cx| (-4..4).map(move |cz| (cx, cz)))
            .find(|&(cx, cz)| gen.tree_site(cx, cz, blocks).is_some())
            .expect("some chunk column at/near the 160m edge grows a tree");

        // The level-0 side: exactly what `block_at`'s per-voxel trees place.
        let level0 = gen.tree_site(cx, cz, blocks).expect("checked above");

        // The level-1 side: the coarse node that would replace this chunk
        // once it crosses the edge.
        let node = NodeKey::containing(ChunkCoord::new(cx, 0, cz), 1);
        let node_trees = gen.node_trees(node, blocks);
        let same_tree = node_trees
            .iter()
            .find(|t| t.x == level0.x && t.z == level0.z)
            .unwrap_or_else(|| {
                panic!("level-1 node {node:?} grows no tree at ({}, {}), level 0 does -- a tree would vanish crossing the edge", level0.x, level0.z)
            });

        // It stands on the ground the level-1 node draws under it: the top
        // of its 2-block cell, which is the mean of that cell's four columns
        // or its summit (issue #423) -- so within what the cell's own
        // columns differ of where level 0 stands it, above or below.
        let (cx2, cz2) = (level0.x.div_euclid(2) * 2, level0.z.div_euclid(2) * 2);
        assert_eq!(
            same_tree.base,
            gen.cell_top(cx2, cz2, 2) + 1,
            "a level-1 tree does not stand on its cell's top"
        );
        let columns = (0..2)
            .flat_map(|dx| (0..2).map(move |dz| (dx, dz)))
            .map(|(dx, dz)| gen.column_top(cx2 + dx, cz2 + dz));
        let (lowest, highest) = (columns.clone().min().unwrap(), columns.max().unwrap());
        assert!(
            (lowest + 1..=highest + 1).contains(&same_tree.base)
                && (lowest + 1..=highest + 1).contains(&level0.base),
            "a tree moved from {} to {} crossing the 160m edge, over columns {lowest}..={highest}",
            level0.base,
            same_tree.base
        );
        assert_eq!(
            same_tree.height, level0.height,
            "a tree changed size crossing the edge"
        );
        assert_eq!(same_tree.canopy_radius, level0.canopy_radius);
    }

    // ---- One ground for every level (issue #423) -------------------------

    /// Seeds and corners the ground tests walk: hills, and the default
    /// seed's mountains.
    const GROUND_CORNERS: [(i32, i32); 5] = [
        (0, 0),
        (-300, 170),
        (-848, 4832),
        (1200, 900),
        (-5000, 2040),
    ];

    #[test]
    fn ground_layer_is_the_depth_rule() {
        assert_eq!(WorldGen::ground_layer(-3), GroundLayer::Surface);
        assert_eq!(WorldGen::ground_layer(0), GroundLayer::Surface);
        for depth in 1..=SOIL_DEPTH {
            assert_eq!(WorldGen::ground_layer(depth), GroundLayer::Soil, "{depth}");
        }
        assert_eq!(WorldGen::ground_layer(SOIL_DEPTH + 1), GroundLayer::Stone);
        assert_eq!(WorldGen::ground_layer(500), GroundLayer::Stone);
    }

    /// A cell's top worked out the long way, from every column of the nine
    /// cells around it: `(top, mean, highest, holds a summit)`. The test's
    /// own arithmetic, sharing nothing with `top_over` but the rounding.
    fn top_by_hand(
        height: &dyn Fn(i32, i32) -> i32,
        x0: i32,
        z0: i32,
        w: i32,
    ) -> (i32, i32, i32, bool) {
        let own: Vec<i32> = (0..w * w).map(|k| height(x0 + k % w, z0 + k / w)).collect();
        let sum: i64 = own.iter().map(|&h| h as i64).sum();
        let mean = div_round_half_even(sum, (w * w) as i64) as i32;
        let highest = *own.iter().max().unwrap();
        // The first highest column of the 3w x 3w square, z then x.
        let mut first = (i32::MIN, 0, 0);
        for z in z0 - w..z0 + 2 * w {
            for x in x0 - w..x0 + 2 * w {
                let h = height(x, z);
                if h > first.0 {
                    first = (h, x, z);
                }
            }
        }
        let summit = (x0..x0 + w).contains(&first.1) && (z0..z0 + w).contains(&first.2);
        // A summit's top: the highest mean of the four half-width cells.
        let half = (w / 2).max(1);
        let best_half = (0..4)
            .filter_map(|q| {
                let (qx, qz) = (x0 + q % 2 * half, z0 + q / 2 * half);
                (qx < x0 + w && qz < z0 + w).then(|| {
                    let sum: i64 = (0..half * half)
                        .map(|k| height(qx + k % half, qz + k / half) as i64)
                        .sum();
                    div_round_half_even(sum, (half * half) as i64) as i32
                })
            })
            .max()
            .unwrap();
        (if summit { best_half } else { mean }, mean, highest, summit)
    }

    #[test]
    fn a_half_rounds_to_the_even_block() {
        for (a, b, want) in [
            (1, 2, 0),
            (3, 2, 2),
            (5, 2, 2),
            (-1, 2, 0),
            (-3, 2, -2),
            (1, 4, 0),
            (3, 4, 1),
            (-3, 4, -1),
            (6, 4, 2),
            (10, 4, 2),
            (7, 1, 7),
            (-7, 1, -7),
        ] {
            assert_eq!(div_round_half_even(a, b), want, "{a} / {b}");
        }
    }

    /// Level 0 is the same function, not a special case; and a cell's top is
    /// the mean of its columns' tops -- or, where it holds a summit, the
    /// highest mean of its four half-width cells -- found here by looking
    /// at every column of the nine cells around it. Both kinds must turn
    /// up, and a summit must stand above its own mean somewhere, or the
    /// test could not tell the rule from either half of it.
    ///
    /// **A column's top is what the sky sees of it** ([`WorldGen::column_top`]:
    /// under a pit, the floor), and cells that a pit sinks must turn up
    /// too: the same rule over the surfaces -- what was built until §9.9 --
    /// is red here on every one of them.
    #[test]
    fn a_cells_top_is_its_columns_mean_and_its_summit() {
        let gen = WorldGen::new(DEFAULT_TEST_SEED);
        let height = |x: i32, z: i32| gen.column_top(x, z);
        let surface = |x: i32, z: i32| gen.surface_height(x, z);
        let (mut summits, mut lifted, mut means, mut under, mut sunk) = (0, 0, 0, 0, 0);
        for (ox, oz) in GROUND_CORNERS {
            assert_eq!(gen.cell_top(ox, oz, 1), gen.column_top(ox, oz));
            for w in [2, 4, 8, 16] {
                for cz in -4..4 {
                    for cx in -4..4 {
                        let (x0, z0) = (ox + cx * w, oz + cz * w);
                        let (top, mean, highest, summit) = top_by_hand(&height, x0, z0, w);
                        assert_eq!(gen.cell_top(x0, z0, w), top, "w {w} at ({x0}, {z0})");
                        sunk += (top < top_by_hand(&surface, x0, z0, w).0) as u32;
                        if summit {
                            summits += 1;
                            lifted += (highest > mean) as u32;
                        } else {
                            means += 1;
                            under += (mean < highest) as u32;
                        }
                    }
                }
            }
        }
        assert!(
            summits >= 20 && lifted >= 10 && means >= 500 && under >= 200 && sunk >= 40,
            "{summits} summits ({lifted} above their mean), {means} means ({under} under their highest column), {sunk} cells a pit sinks"
        );
    }

    /// **A pit sinks a coarse cell by the share of its columns that are pit
    /// times their depth** (`docs/PHASE3_ARCHITECTURE.md` §9.9): a pit four
    /// columns square and twenty deep, in level ground. The cells that are
    /// all pit stand at its floor; a cell a quarter pit stands five blocks
    /// down; one a sixteenth pit, one block (1.25, rounded); and the cells
    /// beside it stay where the ground is. The air drawn over the pit is
    /// level 0's 320 blocks at every width, to the rounding (256 at width
    /// 16, whose block and a quarter is drawn as one). With the mean of the
    /// surfaces every cell stood at 64 and the pit was not there.
    #[test]
    fn a_cell_over_a_pit_sinks_by_its_share_of_the_depth() {
        const GROUND: i32 = 64;
        const DEEP: i32 = 20;
        let pit = |x: i32, z: i32| (32..36).contains(&x) && (48..52).contains(&z);
        let top = |x: i32, z: i32| GROUND - DEEP * pit(x, z) as i32;
        for (w, want, want_air) in [
            (1, 44, 320),
            (2, 44, 320),
            (4, 44, 320),
            (8, 59, 320),
            (16, 63, 256),
        ] {
            assert_eq!(
                WorldGen::top_over(&top, 32, 48, w, w),
                want,
                "the cell {w} wide at the pit's corner"
            );
            // Every cell around, and the air over all of them.
            let mut air = 0;
            for cz in -3..4 {
                for cx in -3..4 {
                    let (x0, z0) = (32 + cx * w, 48 + cz * w);
                    let drawn = WorldGen::top_over(&top, x0, z0, w, w);
                    let holds_pit = (0..w * w).any(|k| pit(x0 + k % w, z0 + k / w));
                    assert_eq!(
                        drawn < GROUND,
                        holds_pit,
                        "w {w}: the cell at ({x0}, {z0}) stands at {drawn}"
                    );
                    air += (GROUND - drawn) * w * w;
                }
            }
            assert_eq!(air, want_air, "w {w}: the air over the pit");
        }
        // A pit that straddles cells: each sinks by its own share.
        let pit = |x: i32, z: i32| (35..39).contains(&x) && (50..54).contains(&z);
        let top = |x: i32, z: i32| GROUND - DEEP * pit(x, z) as i32;
        for w in [2, 4, 8] {
            for cz in -2..3 {
                for cx in -2..3 {
                    let (x0, z0) = (32 + cx * w, 48 + cz * w);
                    let pits = (0..w * w).filter(|k| pit(x0 + k % w, z0 + k / w)).count() as i64;
                    assert_eq!(
                        WorldGen::top_over(&top, x0, z0, w, w) as i64,
                        GROUND as i64 - div_round_half_even(pits * DEEP as i64, (w * w) as i64),
                        "w {w}: the cell at ({x0}, {z0}), {pits} of its columns pit"
                    );
                }
            }
        }
    }

    /// **A lone high column raises no tower.** On level ground one column
    /// stands 32 blocks proud. Level 1 keeps it (its half-width cells are
    /// single columns); every coarser cell that holds it stands at the mean
    /// of its half-width cell, so the column counts for a quarter, a
    /// sixteenth, a sixty-fourth of its height at widths 4, 8 and 16 --
    /// wherever in the cell it stands. Under the highest column (#432's
    /// rule) the cell stood all 32 blocks up at every width: a tower `w`
    /// blocks wide where level 0 has a needle one block wide.
    #[test]
    fn a_lone_high_column_raises_no_tower() {
        const GROUND: i32 = 64;
        const PROUD: i32 = 32;
        for w in [2, 4, 8, 16] {
            let half = (w / 2) as i64;
            let want = div_round_half_even(GROUND as i64 * half * half + PROUD as i64, half * half);
            for (dx, dz) in [(0, 0), (w - 1, w - 1), (w / 2, w / 2 - 1), (0, w - 1)] {
                let (px, pz) = (3 * w + dx, 5 * w + dz);
                let height = |x: i32, z: i32| GROUND + PROUD * ((x, z) == (px, pz)) as i32;
                let (x0, z0) = (3 * w, 5 * w);
                let top = WorldGen::top_over(&height, x0, z0, w, w);
                assert_eq!(
                    top as i64, want,
                    "w {w}: the column {PROUD} proud at ({dx}, {dz}) in the cell"
                );
                for (nx, nz) in [(x0 - w, z0), (x0 + w, z0), (x0, z0 - w), (x0, z0 + w)] {
                    assert_eq!(WorldGen::top_over(&height, nx, nz, w, w), GROUND);
                }
            }
        }
    }

    /// The owner's mountain (the default seed's highest peak, at about
    /// (-3072, -3232)), 512 x 512 blocks around it.
    const PEAK_WINDOW: (i32, i32, i32) = (-3328, -3488, 512);

    /// Every cell's top at width `w` over [`PEAK_WINDOW`] and one cell
    /// around it, `z` outer.
    fn peak_window_tops(gen: &WorldGen, w: i32) -> (i32, Vec<i32>) {
        let (x0, z0, side) = PEAK_WINDOW;
        let n = side / w + 2;
        let tops = (0..n * n)
            .map(|k| gen.cell_top(x0 + (k % n - 1) * w, z0 + (k / n - 1) * w, w))
            .collect();
        (n, tops)
    }

    /// **The crests of a coarse mountain grow few towers**: over the owner's
    /// mountain, the cells that stand at least their own width above all
    /// four cells beside them -- a block as tall as it is wide, or taller,
    /// standing alone -- are counted at widths 4 and 8 (levels 2 and 3).
    /// Measured: 6 and 2 (the plain mean: 1 and 0). Under #432's rule, a
    /// summit at its highest column, they were 18 and 12: Gemini's
    /// "pillars and spires along the crests" at level 3.
    ///
    /// **Above the ground beside it, not above a pit beside it**: a cell is
    /// held against what the four beside it draw of the ground with no cave
    /// in it (the rule over the surfaces). A pit sinks the cell it is in
    /// (§9.9), and the cell next to a pit thirty blocks deep stands thirty
    /// blocks over it at level 0 too: that is the pit's wall, not a tower
    /// the level grew. Against the drawn tops the same window counts 6 and
    /// 4, the two more each beside a cell a pit has sunk.
    #[test]
    fn a_coarse_crest_grows_few_towers() {
        /// **Chosen, not derived**: the measured counts with a little room.
        const MOST: [(i32, usize); 2] = [(4, 8), (8, 3)];
        let gen = WorldGen::new(DEFAULT_TEST_SEED);
        let surface = |x: i32, z: i32| gen.surface_height(x, z);
        let (x0, z0, _) = PEAK_WINDOW;
        for (w, most) in MOST {
            let (n, tops) = peak_window_tops(&gen, w);
            let at = |i: i32, j: i32| tops[(j * n + i) as usize];
            // What the cell `(i, j)` draws of the ground with no cave in it.
            let ground = |i: i32, j: i32| {
                WorldGen::top_over(&surface, x0 + (i - 1) * w, z0 + (j - 1) * w, w, w)
            };
            let towers = (1..n - 1)
                .flat_map(|j| (1..n - 1).map(move |i| (i, j)))
                .filter(|&(i, j)| {
                    let beside = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)];
                    // The drawn tops first: they are at hand.
                    beside.iter().all(|&(a, b)| at(i, j) - at(a, b) >= w)
                        && beside.iter().all(|&(a, b)| at(i, j) - ground(a, b) >= w)
                })
                .count();
            println!("width {w}: {towers} towers");
            assert!(
                towers <= most,
                "width {w}: {towers} cells stand alone at least {w} blocks above all four beside them, allowed {most}"
            );
        }
    }

    /// **The owner's mountain keeps its peak**: the highest top over the
    /// window is level 0's at level 1, at most one block under it at level
    /// 2 and two at level 3 -- and never above it. Measured: 0, -1 and -2;
    /// the plain mean drew -1, -2 and -4, the coarse levels losing the peak
    /// that the summit rule exists to keep.
    #[test]
    fn the_owners_mountain_keeps_its_peak() {
        /// Per width, the most the drawn peak may stand under level 0's.
        /// **Chosen, not derived**: the measured numbers.
        const UNDER: [(i32, i32); 3] = [(2, 0), (4, 1), (8, 2)];
        let gen = WorldGen::new(DEFAULT_TEST_SEED);
        let (x0, z0, side) = PEAK_WINDOW;
        let level0 = (0..side * side)
            .map(|k| gen.column_top(x0 + k % side, z0 + k / side))
            .max()
            .unwrap();
        for (w, under) in UNDER {
            let (n, tops) = peak_window_tops(&gen, w);
            let drawn = (1..n - 1)
                .flat_map(|j| (1..n - 1).map(move |i| (j * n + i) as usize))
                .map(|k| tops[k])
                .max()
                .unwrap();
            println!("width {w}: peak {drawn}, level 0 {level0}");
            assert!(
                (level0 - under..=level0).contains(&drawn),
                "width {w}: the peak is drawn at {drawn}, level 0's is {level0} (allowed {under} under, none over)"
            );
        }
    }

    /// **A coarse cell stands as much above its columns as below them**:
    /// over the cells that hold no summit, the top minus the columns' mean
    /// comes to nothing, at every width. That is what leaves a swap with no
    /// direction -- the first version of #423 (the highest column) stood
    /// +1.2, +3.5 and +8.0 blocks over a steep slope's mean at widths 2, 4
    /// and 8, and main's mean rounded down stood half a block under.
    #[test]
    fn a_cells_top_leans_neither_up_nor_down() {
        /// **Chosen, not derived**: a twentieth of a block, over some
        /// thousand cells a width (measured: 0.02 at most).
        const LEAN: f64 = 0.05;
        let gen = WorldGen::new(DEFAULT_TEST_SEED);
        let height = |x: i32, z: i32| gen.column_top(x, z);
        // The default seed's mountains: slopes, where a lean would show.
        let (mx, mz) = (-848, 4832);
        for w in [2, 4, 8] {
            let (mut lean, mut cells) = (0f64, 0u32);
            for cz in 0..256 / w {
                for cx in 0..256 / w {
                    let (x0, z0) = (mx + cx * w, mz + cz * w);
                    let (_, _, _, summit) = top_by_hand(&height, x0, z0, w);
                    if summit {
                        continue;
                    }
                    let sum: i64 = (0..w * w)
                        .map(|k| height(x0 + k % w, z0 + k / w) as i64)
                        .sum();
                    lean += gen.cell_top(x0, z0, w) as f64 - sum as f64 / (w * w) as f64;
                    cells += 1;
                }
            }
            let lean = lean / cells as f64;
            println!("width {w}: {cells} cells, top - mean = {lean:+.4}");
            assert!(cells >= 900, "width {w}: only {cells} cells");
            assert!(
                lean.abs() <= LEAN,
                "width {w}: a cell's top stands {lean:+.3} blocks off its columns' mean"
            );
        }
    }

    /// The sampled top is the exact one once it samples every column; with
    /// fewer it is the rule over its own samples -- `n` to a side, centred in
    /// the cell, the nine cells around sampled alike -- and at one sample it
    /// is the column at the cell's middle.
    #[test]
    fn a_sampled_top_is_exact_at_full_count_and_the_rule_over_its_samples() {
        let gen = WorldGen::new(DEFAULT_TEST_SEED);
        let mut differs = 0;
        for (ox, oz) in GROUND_CORNERS {
            for w in [4, 8, 16, 32] {
                let exact = gen.cell_top(ox, oz, w);
                assert_eq!(gen.cell_top_sampled(ox, oz, w, w), exact);
                assert_eq!(gen.cell_top_sampled(ox, oz, w, 4 * w), exact);
                for n in [1, 2, 4] {
                    let sampled = gen.cell_top_sampled(ox, oz, w, n);
                    // The columns it says it visits, and no others: the
                    // sample grid as a height field of its own, `n` wide.
                    let step = w / n.min(w);
                    let lead = (step - 1) / 2;
                    let grid =
                        |a: i32, b: i32| gen.column_top(ox + lead + a * step, oz + lead + b * step);
                    let (by_hand, _, _, _) = top_by_hand(&grid, 0, 0, n.min(w));
                    assert_eq!(sampled, by_hand, "w {w} n {n}");
                    if sampled != exact {
                        differs += 1;
                    }
                }
            }
        }
        assert!(
            differs > 0,
            "no sampled top ever differed from the exact one: nothing was tested"
        );
    }

    /// `ground_faces` against counting level 0's faces block by block, from
    /// `block_at`-free arithmetic on the columns: a column shows its lower
    /// neighbour one face per block its top stands above the neighbour's,
    /// layered by depth below its own surface. **A top is what the sky
    /// sees** (`column`): the walls of a pit are among the faces, stone
    /// under the soil, and its floor's own risers stone from the first row
    /// -- which must turn up here, or the count could be the surfaces'.
    #[test]
    fn ground_faces_are_level_0s_risers() {
        let gen = WorldGen::new(DEFAULT_TEST_SEED);
        let mut stone = 0.0;
        let (mut pit_walls, mut floor_risers) = (0, 0);
        // And a cliff, found rather than assumed: the first column standing
        // six blocks over its neighbour.
        let cliff = (-2048..2048)
            .step_by(4)
            .flat_map(|x| (-2048..2048).step_by(4).map(move |z| (x, z)))
            .find(|&(x, z)| gen.surface_height(x, z) - gen.surface_height(x + 1, z) >= 6)
            .expect("a cliff within 2,048 blocks");
        for (x0, z0) in GROUND_CORNERS
            .into_iter()
            .chain([(cliff.0 - 8, cliff.1 - 8)])
        {
            let w = 16;
            let faces = gen.ground_faces(x0, z0, w, w);
            assert_eq!(faces.columns, (w * w) as u32);
            let mut expected = [[0f32; 3]; 4];
            let mut any_step = false;
            for z in z0..z0 + w {
                for x in x0..x0 + w {
                    let GroundColumn { surface, top: h } = gen.column(x, z);
                    for (d, [dx, dz]) in GROUND_DIRECTIONS.into_iter().enumerate() {
                        let beside = gen.column(x + dx, z + dz);
                        for y in (beside.top + 1)..=h {
                            any_step = true;
                            let layer = if y == surface {
                                0
                            } else if surface - y <= SOIL_DEPTH {
                                1
                            } else {
                                2
                            };
                            expected[d][layer] += 1.0;
                            // A face a whole column shows a pit beside it,
                            // under where the neighbour's surface was...
                            pit_walls += (h == surface && y <= beside.surface) as u32;
                            // ...and one a pit's own floor shows.
                            floor_risers += (h < surface) as u32;
                        }
                    }
                }
            }
            assert_eq!(faces.sides, expected, "at ({x0}, {z0})");
            stone += faces.sides.iter().map(|s| s[2]).sum::<f32>();
            assert!((0.0..=1.0).contains(&faces.top_open));
            for d in 0..4 {
                assert!((0.0..=1.0).contains(&faces.side_open[d]));
                let share: f32 = [GroundLayer::Surface, GroundLayer::Soil, GroundLayer::Stone]
                    .iter()
                    .filter_map(|&l| faces.share(d, l))
                    .sum();
                let area: f32 = faces.sides[d].iter().sum();
                assert!(area == 0.0 || (share - 1.0).abs() < 1e-5);
            }
            // A step above a top face shades it.
            assert_eq!(any_step, faces.top_open < 1.0, "at ({x0}, {z0})");
            // Sampling every column is the exact count.
            assert_eq!(gen.ground_faces(x0, z0, w, 4 * w), faces);
            // Sampling fewer counts fewer columns, on the sampled grid.
            let sampled = gen.ground_faces(x0, z0, w, 4);
            assert_eq!(sampled.columns, 16);
        }
        assert!(
            stone > 0.0,
            "no cliff in the sample: the stone layer was not tested"
        );
        assert!(
            pit_walls > 50 && floor_risers > 5,
            "{pit_walls} faces of pit walls and {floor_risers} of pit floors: the pits were not tested"
        );
    }

    /// Open ground is fully open, and a riser's foot is not: the openness is
    /// the mesher's corner rule, on a slope whose answer can be worked out
    /// by hand. A column one block under its +x neighbour and level with the
    /// rest has two of its top's four corners shaded one step by the wall and
    /// one more by the wall's continuation along z: `(3 + 3 + 1 + 1) / 12`.
    #[test]
    fn openness_is_the_meshers_corner_rule() {
        // A west-east staircase of the real generator is not to be had on
        // demand, so the rule is checked through `face_ao_of` on one, the
        // way `ground_faces` calls it.
        let stairs = |x: i32, y: i32, _z: i32| y <= x;
        // The top of the column at x = 5 (surface y = 5): air layer y = 6.
        let ao = cubara_voxel::face_ao_of(&stairs, 1, 2, 0, 6, 0, 5);
        let mut sorted = ao;
        sorted.sort_unstable();
        assert_eq!(sorted, [1, 1, 3, 3]);
        // Its riser facing -x, one block tall: the lower column's top is at
        // its foot, so its two bottom corners are shaded.
        let ao = cubara_voxel::face_ao_of(&stairs, 0, 1, 2, 4, 5, 0);
        let mut sorted = ao;
        sorted.sort_unstable();
        assert_eq!(sorted, [1, 1, 3, 3]);
    }

    #[test]
    fn generation_is_isolated_from_neighbours() {
        // §8.1's binding contract: generating a chunk reads nothing but the
        // seed and its own coordinate -- no neighbouring chunk's state, no
        // generation order, no shared scratch. Generate one chunk alone,
        // then generate it again after generating a shuffled handful of its
        // neighbours first: if `generate` ever reached for neighbour state,
        // this is exactly what would catch it.
        let gen = WorldGen::new(1234);
        let blocks = test_blocks();
        let origin = [32, 0, -16];

        let alone = gen.generate(origin, 1, blocks);

        // A fixed, deliberately-not-sorted handful of neighbouring chunk
        // origins (16 blocks apart, matching Chunk::SIZE).
        let neighbours = [
            [origin[0] + 16, origin[1], origin[2]],
            [origin[0], origin[1] + 16, origin[2] - 16],
            [origin[0] - 16, origin[1], origin[2] + 16],
            [origin[0], origin[1] - 16, origin[2]],
            [origin[0] + 16, origin[1], origin[2] + 16],
        ];
        for n in neighbours {
            let _ = gen.generate(n, 1, blocks);
        }

        let after_neighbours = gen.generate(origin, 1, blocks);
        assert!(
            chunks_equal(&alone, &after_neighbours),
            "chunk at {origin:?} changed after generating its neighbours -- \
             generation must not depend on what else was generated, or in \
             what order"
        );
    }

    #[test]
    fn step_one_is_exactly_block_at() {
        // `generate()` samples the same functions a block query does; at step
        // 1 that means exact agreement.
        let gen = WorldGen::new(99);
        let blocks = test_blocks();
        let fine = gen.generate([0, -8, 0], 1, blocks);
        for lz in 0..Chunk::SIZE {
            for ly in 0..Chunk::SIZE {
                for lx in 0..Chunk::SIZE {
                    let (x, y, z) = (lx as i32, ly as i32 - 8, lz as i32);
                    let expected = gen.block_at(x, y, z, blocks).unwrap_or(BlockId::AIR);
                    assert_eq!(fine.get(lx, ly, lz), expected, "step 1 ({lx},{ly},{lz})");
                }
            }
        }
    }

    /// **`generate` at a step of one is level 0: the per-block rule, block
    /// for block** (`terrain_block_at`), and never the coarse one. The two
    /// part where it matters most to a player underground: a coarse cell is
    /// not carved more than `DISTANT_CAVE_DEPTH` under the surface, a block
    /// is carved at any depth. So the chunks compared here reach well below
    /// that, and the test requires caves down there to have been compared.
    #[test]
    fn generate_at_step_one_is_the_per_block_rule() {
        let gen = WorldGen::new(DEFAULT_TEST_SEED);
        let blocks = test_blocks();
        let mut deep_air = 0;
        for (ox, oz) in [(0, 0), (-304, 176), (1200, 896)] {
            let surface = gen.surface_height(ox, oz);
            for oy in [surface - 96, surface - 64, surface - 16] {
                let origin = [ox, oy.div_euclid(16) * 16, oz];
                let chunk = gen.generate(origin, 1, blocks);
                for lz in 0..Chunk::SIZE {
                    for ly in 0..Chunk::SIZE {
                        for lx in 0..Chunk::SIZE {
                            let (x, y, z) = (
                                origin[0] + lx as i32,
                                origin[1] + ly as i32,
                                origin[2] + lz as i32,
                            );
                            let want = gen.terrain_block_at(x, y, z, blocks);
                            assert_eq!(
                                chunk.get(lx, ly, lz),
                                want.unwrap_or(BlockId::AIR),
                                "block ({x}, {y}, {z})"
                            );
                            if want.is_none() && y < gen.surface_height(x, z) - DISTANT_CAVE_DEPTH {
                                deep_air += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(
            deep_air > 100,
            "only {deep_air} blocks of cave deeper than a coarse cell is carved: nothing to tell the rules apart by"
        );
    }

    /// The top of a coarse column is the surface block, as the top of every
    /// real column is. Seen from far away, those tops are most of what the
    /// ground is; when a third of them wore soil (the cell's own top block
    /// sits under the surface), the distant fields came out patched with
    /// brown. **Or the floor of a pit under it** (the colour morph,
    /// [`WorldGen::top_block`]): what level 0 shows from above one of the
    /// cell's columns that a cave has opened, and nothing else.
    ///
    /// (This also held "and only the top: the cell under it is soil or
    /// stone" until issue #423. That was the bug: on a slope the cells under
    /// a coarse top are the tops of the lower columns it covers, and level 0
    /// shows their sides. `a_coarse_riser_is_made_of_level_0s_faces` is what
    /// holds the cells under the top now.)
    #[test]
    fn a_coarse_columns_top_is_the_surface_block() {
        let gen = WorldGen::new(0xCAFE);
        let blocks = test_blocks();
        for step in [2, 4, 8] {
            let (mut tops, mut soil) = (0u32, 0u32);
            for (ox, oz) in [(0, 0), (-300, 170), (500, 20), (-700, -640), (1200, 900)] {
                let x0 = ox;
                let z0 = oz;
                // A node whose middle holds the surface here.
                let surface = gen.cell_surface_height(x0, z0, step);
                let oy = surface - 8 * step;
                let node = gen.generate([ox, oy, oz], step, blocks);
                for lz in 0..Chunk::SIZE {
                    for lx in 0..Chunk::SIZE {
                        let (x, z) = (ox + lx as i32 * step, oz + lz as i32 * step);
                        let surface = gen.cell_surface_height(x, z, step);
                        // The cell the height field ends in, if this node
                        // holds it and something is left of it (a cave can
                        // carve it away).
                        let Some(ly) = (0..Chunk::SIZE).rev().find(|&ly| {
                            let y = oy + ly as i32 * step;
                            (y..y + step).contains(&surface)
                        }) else {
                            continue;
                        };
                        let block = node.get(lx, ly, lz);
                        if block == BlockId::AIR {
                            continue;
                        }
                        tops += 1;
                        if block != blocks.grass
                            && !gen.a_pit_under_shows(x, z, step, block, blocks)
                        {
                            soil += 1;
                        }
                    }
                }
            }
            assert!(tops > 500, "step {step}: only {tops} tops checked");
            assert_eq!(
                soil, 0,
                "step {step}: {soil} of {tops} coarse tops are not the surface block"
            );
        }
    }

    /// **A coarse riser is made of what level 0's faces show** (issue #423
    /// decision 2). At level 0 a column shows a side face at every row above
    /// a neighbour's top and at or under its own (`column_top`: the wall of
    /// a pit is such a side), made of `ground_layer` at that row's depth
    /// below its own surface. Every solid cell of a coarse node, at levels
    /// 1, 2 and 3:
    ///
    /// - wears a layer that one of the level-0 risers at its rows, within its
    ///   columns, shows -- counted here from the columns alone;
    /// - or, where no covered column shows a riser there but the row is part
    ///   of the cell's own (above the lowest top beside the cell, as that
    ///   stands with no cave in the ground), a layer that one of the cell's
    ///   risers at any row shows;
    /// - or, under that foot, where its columns show the wall of a pit (a
    ///   neighbour's top under the row, its surface at or over it), a layer
    ///   one of those faces shows;
    /// - or, at or under that foot, the layer below the lowest covered
    ///   column that reaches the row;
    /// - or, as the cell that holds the top, the surface block.
    ///
    /// **And the layers show in level 0's shares**: weighing each cell by
    /// the faces it stands for, the share of every layer among the cells is
    /// within `SHARE` of its share among the faces. On this mountain a third
    /// of level 0's faces are soil. Depth below the cell's own top gave
    /// mostly stone here, and the least depth any covered column offers gave
    /// nine tenths surface block; this test is red on both.
    /// **A coarse top shows a pit's floor in the share its columns are pit**
    /// (the colour morph, [`WorldGen::top_block`]): over the default seed's
    /// mountains, at cell widths 4 and 8 (levels 2 and 3, where the zoom
    /// pairs lost the grey patches), the tops that wear something other
    /// than the surface block number what the pit share of their cells'
    /// columns says they should -- within [`PIT_SHARE`] of it. Before the
    /// rule none did, which is red here.
    #[test]
    fn a_coarse_top_shows_pit_floors_in_the_share_level_0_does() {
        /// Chosen, not derived: the pick is a hash per cell, so the count is
        /// a sample of the expectation; a few hundred cells land within a
        /// fifth of it.
        const PIT_SHARE: f64 = 0.25;
        let gen = WorldGen::new(DEFAULT_TEST_SEED);
        let blocks = test_blocks();
        let (mx, mz): (i32, i32) = (-848, 4832);
        for (step, v) in [(4, 1), (8, 4)] {
            let size = 16 * step;
            let (mut expected, mut shown, mut tops) = (0f64, 0u32, 0u32);
            let mut nodes = std::collections::BTreeMap::new();
            for nz in -2..2 {
                for nx in -2..2 {
                    let (ox, oz) = (
                        mx.div_euclid(size) * size + nx * size,
                        mz.div_euclid(size) * size + nz * size,
                    );
                    for lz in 0..Chunk::SIZE as i32 {
                        for lx in 0..Chunk::SIZE as i32 {
                            let (x0, z0) = (ox + lx * step, oz + lz * step);
                            let top = gen.cell_surface_height(x0, z0, step);
                            let oy = top.div_euclid(size) * size;
                            let node = nodes.entry((ox, oy, oz)).or_insert_with(|| {
                                gen.generate_aniso([ox, oy, oz], step, v, blocks)
                            });
                            let ly = ((top - oy) / v) as usize;
                            let block = node.get(lx as usize, ly, lz as usize);
                            if block == BlockId::AIR {
                                continue; // the whole cell is pit
                            }
                            tops += 1;
                            let pit = (0..step * step)
                                .filter(|k| {
                                    let (x, z) = (x0 + k % step, z0 + k / step);
                                    gen.from_above(x, z, gen.column(x, z), blocks).is_some()
                                })
                                .count();
                            expected += pit as f64 / (step * step) as f64;
                            if block != blocks.grass {
                                shown += 1;
                            }
                        }
                    }
                }
            }
            println!(
                "width {step}: {shown} of {tops} tops show a pit's floor, {expected:.1} expected"
            );
            assert!(
                expected > 30.0,
                "width {step}: too few pits to measure ({expected:.1})"
            );
            assert!(
                (shown as f64 - expected).abs() <= PIT_SHARE * expected,
                "width {step}: {shown} tops show a pit's floor, {expected:.1} expected"
            );
        }
    }

    /// **A pit's floor is level 0's**: [`WorldGen::pit_floor`] is the first
    /// solid block under a surface block a cave has carved away -- every
    /// block above it, up to the surface, is air at level 0 -- and there is
    /// none where the surface block stands. And what shows from above there
    /// ([`WorldGen::from_above`], the voxels' form; [`WorldGen::top_layer`]
    /// reads the same floor) is what level 0's block at the floor is. Over
    /// the pitted mountains; soil floors and stone floors must both turn up,
    /// or a floor always one block down would pass.
    #[test]
    fn a_pits_floor_is_the_first_ground_under_the_carved_surface() {
        let gen = WorldGen::new(DEFAULT_TEST_SEED);
        let blocks = test_blocks();
        let (mx, mz): (i32, i32) = (-848, 4832);
        let (mut whole, mut soil, mut stone) = (0, 0, 0);
        for z in mz - 128..mz + 128 {
            for x in mx - 128..mx + 128 {
                let surface = gen.surface_height(x, z);
                let Some(floor) = gen.pit_floor(x, z, surface) else {
                    assert!(
                        gen.is_solid(x, surface, z),
                        "({x}, {z}): no pit, no surface"
                    );
                    assert_eq!(gen.column(x, z), GroundColumn::whole(surface));
                    assert_eq!(gen.from_above(x, z, gen.column(x, z), blocks), None);
                    whole += 1;
                    continue;
                };
                assert!(floor < surface, "({x}, {z}): the floor at {floor}");
                for y in floor + 1..=surface {
                    assert!(!gen.is_solid(x, y, z), "({x}, {y}, {z}) above the floor");
                }
                assert!(
                    gen.is_solid(x, floor, z) || floor == surface - PIT_FLOOR_SEARCH,
                    "({x}, {floor}, {z}): the floor is air"
                );
                assert_eq!(
                    gen.column(x, z),
                    GroundColumn {
                        surface,
                        top: floor
                    }
                );
                let shows = gen
                    .from_above(x, z, gen.column(x, z), blocks)
                    .expect("a pit");
                if gen.is_solid(x, floor, z) {
                    assert_eq!(Some(shows), gen.block_at(x, floor, z, blocks), "({x}, {z})");
                }
                if shows == blocks.soil {
                    soil += 1;
                } else {
                    assert_eq!(shows, blocks.stone, "({x}, {z}): {shows:?}");
                    stone += 1;
                }
            }
        }
        assert!(
            whole > 10_000 && soil > 50 && stone > 500,
            "{whole} whole columns, {soil} soil floors, {stone} stone floors"
        );
    }

    #[test]
    fn a_coarse_riser_is_made_of_level_0s_faces() {
        /// **Chosen, not derived**: a hash picks one face per cell, so the
        /// shares are a sample. Over the thousands of cells of one-block rows
        /// counted here they land within a point of level 0's; at level 3
        /// some 150 cells stand for the faces and they land within three
        /// (all printed below).
        const SHARE: [f64; 3] = [0.03, 0.03, 0.06];
        let gen = WorldGen::new(DEFAULT_TEST_SEED);
        let blocks = test_blocks();
        let layer_of = |block: BlockId| match block {
            b if b == blocks.grass => GroundLayer::Surface,
            b if b == blocks.soil => GroundLayer::Soil,
            b if b == blocks.stone => GroundLayer::Stone,
            other => panic!("a coarse cell of {other:?}"),
        };
        // The default seed's mountains (`far.rs` frames a patch here too).
        let (mx, mz): (i32, i32) = (-848, 4832);
        for ((step, v), allowed) in [(2, 1), (4, 1), (8, 4)].into_iter().zip(SHARE) {
            let size = 16 * step;
            let (ox, oz) = (mx.div_euclid(size) * size, mz.div_euclid(size) * size);
            // Level 0's faces per layer, and the same faces filed under the
            // layer the cell that stands for them wears.
            let (mut level0, mut coarse) = ([0u64; 3], [0u64; 3]);
            let (mut bare, mut reach, mut tops, mut walls) = (0u32, 0u32, 0u32, 0u32);
            let mut nodes = std::collections::BTreeMap::new();
            for lz in 0..Chunk::SIZE as i32 {
                for lx in 0..Chunk::SIZE as i32 {
                    let (x0, z0) = (ox + lx * step, oz + lz * step);
                    // Each column (`z` outer), and its four neighbours.
                    let columns: Vec<(GroundColumn, [GroundColumn; 4])> = (0..step)
                        .flat_map(|dz| (0..step).map(move |dx| (x0 + dx, z0 + dz)))
                        .map(|(x, z)| {
                            let around = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                                .map(|(dx, dz)| gen.column(x + dx, z + dz));
                            (gen.column(x, z), around)
                        })
                        .collect();
                    let top = gen.cell_surface_height(x0, z0, step);
                    // Every riser of the cell on the height field, at any
                    // row, by layer: by its depth below its own column's
                    // surface.
                    let mut all = [0u64; 3];
                    for (c, around) in &columns {
                        for n in around {
                            for row in n.surface + 1..=c.top {
                                all[WorldGen::ground_layer(c.surface - row) as usize] += 1;
                            }
                        }
                    }
                    // From under every face of the cell up to its top.
                    let low = columns
                        .iter()
                        .flat_map(|(c, around)| around.iter().map(|n| n.top).chain([c.top]))
                        .min()
                        .unwrap()
                        - 2 * v;
                    for oy in (low.div_euclid(size)..=top.div_euclid(size)).map(|n| n * size) {
                        let node = nodes
                            .entry(oy)
                            .or_insert_with(|| gen.generate_aniso([ox, oy, oz], step, v, blocks));
                        for ly in 0..node.dims()[1] {
                            let y = oy + ly as i32 * v;
                            if y + v <= low || y > top {
                                continue;
                            }
                            let block = node.get(lx as usize, ly, lz as usize);
                            if block == BlockId::AIR {
                                continue; // above the top, or a cave
                            }
                            let wears = layer_of(block);
                            let at = format!("step {step} cell ({x0}, {y}, {z0})");
                            if top < y + v {
                                if !gen.a_pit_under_shows(x0, z0, step, block, blocks) {
                                    assert_eq!(wears, GroundLayer::Surface, "{at}: the top cell");
                                }
                                tops += 1;
                                continue;
                            }
                            // The risers of the height field at these rows
                            // (over a neighbour's surface), and the walls
                            // of pits (over its top, under its surface).
                            let (mut here, mut pit) = ([0u64; 3], [0u64; 3]);
                            for row in y..y + v {
                                for (c, around) in &columns {
                                    if row <= c.top {
                                        let layer = WorldGen::ground_layer(c.surface - row);
                                        for n in around {
                                            here[layer as usize] += (n.surface < row) as u64;
                                            pit[layer as usize] +=
                                                (n.top < row && row <= n.surface) as u64;
                                        }
                                    }
                                }
                            }
                            let faces: u64 = here.iter().sum();
                            if faces == 0 {
                                let row = y + v - 1;
                                // Where the cell's risers end: the lowest
                                // top beside it as it stands with no cave
                                // in the ground, or -- across the node's
                                // edge -- two cells' widths under the
                                // lowest surface along that edge.
                                let foot = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                                    .into_iter()
                                    .enumerate()
                                    .map(|(d, (dx, dz)): (usize, (i32, i32))| {
                                        if (0..16).contains(&(lx + dx))
                                            && (0..16).contains(&(lz + dz))
                                        {
                                            return WorldGen::top_over(
                                                &|x, z| gen.surface_height(x, z),
                                                x0 + dx * step,
                                                z0 + dz * step,
                                                step,
                                                step,
                                            );
                                        }
                                        (0..step)
                                            .map(|k| {
                                                let (ex, ez) = match (dx, dz) {
                                                    (1, _) => (step - 1, k),
                                                    (-1, _) => (0, k),
                                                    (_, 1) => (k, step - 1),
                                                    _ => (k, 0),
                                                };
                                                columns[(ez * step + ex) as usize].1[d].surface
                                            })
                                            .min()
                                            .unwrap()
                                            - 2 * step
                                    })
                                    .min()
                                    .unwrap();
                                if row > foot && all.iter().sum::<u64>() > 0 {
                                    // Part of a riser: one of all the
                                    // cell's faces.
                                    assert!(
                                        all[wears as usize] > 0,
                                        "{at}: wears {wears:?}, which none of the cell's level-0 faces shows ({all:?})"
                                    );
                                    reach += 1;
                                    continue;
                                }
                                if pit.iter().sum::<u64>() > 0 {
                                    // Under the cell's own risers, and a
                                    // pit's wall: one of its faces.
                                    assert!(
                                        pit[wears as usize] > 0,
                                        "{at}: wears {wears:?}, which none of the pit walls there shows ({pit:?})"
                                    );
                                    walls += 1;
                                    continue;
                                }
                                // The lowest column that reaches the row (the
                                // first, `z` then `x`, of several as low).
                                let reaches = columns
                                    .iter()
                                    .map(|c| c.0)
                                    .enumerate()
                                    .filter(|(_, c)| c.top >= row)
                                    .min_by_key(|&(k, c)| (c.top, k))
                                    .map(|(_, c)| c.surface);
                                assert_eq!(
                                    wears,
                                    WorldGen::ground_layer(reaches.unwrap_or(top) - row),
                                    "{at}: no level-0 face at these rows"
                                );
                                bare += 1;
                                continue;
                            }
                            assert!(
                                here[wears as usize] > 0,
                                "{at}: wears {wears:?}, which none of the {faces} level-0 faces there shows ({here:?})"
                            );
                            for (sum, n) in level0.iter_mut().zip(here) {
                                *sum += n;
                            }
                            coarse[wears as usize] += faces;
                        }
                    }
                }
            }
            let total: u64 = level0.iter().sum();
            let share = |n: u64| n as f64 / total as f64;
            println!(
                "step {step}: {total} faces, level 0 {:.3?}, the cells {:.3?}; {bare} cells with none, {reach} under their lowest column, {walls} on a pit's wall, {tops} top cells",
                level0.map(share),
                coarse.map(share)
            );
            assert!(
                total > 1000,
                "step {step}: only {total} faces on this mountain"
            );
            assert!(
                share(level0[GroundLayer::Soil as usize]) > 0.2,
                "step {step}: this mountain's faces are not soil enough to tell the rules apart"
            );
            assert!(
                bare > 0 && reach > 0 && tops > 0 && walls > 0,
                "step {step}: {bare}, {reach}, {tops}, {walls}"
            );
            for (layer, (a, b)) in level0.into_iter().zip(coarse).enumerate() {
                assert!(
                    (share(a) - share(b)).abs() <= allowed,
                    "step {step}, layer {layer}: {:.3} of level 0's faces, {:.3} of the cells'",
                    share(a),
                    share(b)
                );
            }
        }
    }

    /// **A coarse cell is cave only where all of its ground is** (issue
    /// #423, second round): every solid-or-carved cell of coarse nodes, at
    /// levels 1, 2 and 3 as the game generates them, against the blocks it
    /// stands for.
    ///
    /// - At or under its top and within `DISTANT_CAVE_DEPTH` of its highest
    ///   column, a cell-row is air exactly when every block of ground in it
    ///   -- at or under its own column's top (what the sky sees of it: the
    ///   air of a pit is the cell's top sunk, not a cave), and at or under
    ///   the cell's top -- is cave at level 0.
    /// - Deeper than that it is rock, whatever its blocks are.
    /// - Above its top it is air.
    ///
    /// - And where all of it is cave, the row stays rock only where a level
    ///   that may be drawn beside it or over it would open it through ground
    ///   level 0 draws (`docs/PHASE3_ARCHITECTURE.md` §9.12,
    ///   [`cave_opens_only_where_level_0_does`]): a side or the row over it
    ///   where level 0 has a block of ground across the cell there, and a
    ///   cell of some level that may be drawn there -- 1 to this one, and
    ///   one coarser at the node's edge or top -- stands under the row.
    ///   Worked out here from the blocks and the cells' tops, not by the
    ///   rule's own body.
    ///
    /// The rule this replaces was a vote (a cell was cave where the noise at
    /// its corner was, or where most of eight samples in it were): a cell
    /// could open a pit over ground no finer level opened, and close it
    /// again as the player came nearer. This one can only ever show less
    /// cave than the level under it. Both outcomes must turn up here, and
    /// cells that the old vote would have carved, or the test could not
    /// tell the rules apart -- and rows kept whole for what is beside them.
    ///
    /// [`cave_opens_only_where_level_0_does`]: WorldGen::cave_opens_only_where_level_0_does
    #[test]
    fn a_coarse_cell_is_cave_only_where_all_its_ground_is() {
        let gen = WorldGen::new(0xCAFE);
        let blocks = test_blocks();
        let mut kept_for_what_is_beside = 0u32;
        for (step, v) in [(2i32, 1i32), (4, 1), (8, 4)] {
            let size = 16 * step;
            let level = step.trailing_zeros();
            // No block of ground in row `r` of the cell `step` wide at `(x, z)`.
            let open = |x: i32, z: i32, r: i32| {
                (0..step * step).all(|k| !gen.is_solid(x + k % step, r, z + k / step))
            };
            // Whether some cell of widths 2 to `2^last` holding one of the
            // columns `at` stands under `r`.
            let under = |at: &[(i32, i32)], last: u32, r: i32| {
                (1..=last).any(|k| {
                    let cw = 1 << k;
                    at.iter().any(|&(x, z)| {
                        gen.cell_top(x.div_euclid(cw) * cw, z.div_euclid(cw) * cw, cw) < r
                    })
                })
            };
            let (mut carved, mut whole_with_cave, mut deep) = (0u32, 0u32, 0u32);
            // (Twelve node columns: a pit is no longer a carved row but a
            // sunk top, §9.9, so carved rows are the caves under a top and
            // there are fewer of them to a node.)
            for (ox, oz) in [
                (0, 0),
                (-128, 64),
                (256, -256),
                (-768, -640),
                (512, 128),
                (128, 384),
                (-384, 256),
                (640, -512),
                (-256, -384),
                (384, 640),
                (-640, 128),
                (896, 256),
            ] {
                let (ox, oz): (i32, i32) = (ox / size * size, oz / size * size);
                for oy in [-size, 0, size] {
                    let node = gen.generate_aniso([ox, oy, oz], step, v, blocks);
                    for lz in 0..Chunk::SIZE {
                        for lx in 0..Chunk::SIZE {
                            let (x0, z0) = (ox + lx as i32 * step, oz + lz as i32 * step);
                            let top = gen.cell_top(x0, z0, step);
                            let columns: Vec<(i32, i32, i32)> = (0..step * step)
                                .map(|k| (x0 + k % step, z0 + k / step))
                                .map(|(x, z)| (x, z, gen.column_top(x, z)))
                                .collect();
                            let peak = columns.iter().map(|c| c.2).max().unwrap();
                            for ly in 0..node.dims()[1] {
                                let y = oy + ly as i32 * v;
                                let at = format!("step {step} cell ({x0}, {y}, {z0}), top {top}");
                                let air = node.get(lx, ly, lz) == BlockId::AIR;
                                if y > top {
                                    assert!(air, "{at}: solid above its top");
                                    continue;
                                }
                                // Its ground, block by block.
                                let (mut ground, mut cave) = (0u32, 0u32);
                                for &(x, z, column_top) in &columns {
                                    for row in y..=(y + v - 1).min(top).min(column_top) {
                                        ground += 1;
                                        cave += gen.block_at(x, row, z, blocks).is_none() as u32;
                                    }
                                }
                                assert!(ground > 0, "{at}: no ground at or under its top");
                                if y <= peak - DISTANT_CAVE_DEPTH {
                                    assert!(!air, "{at}: a distant cave below its depth");
                                    deep += (cave == ground) as u32;
                                    continue;
                                }
                                // Whether some level would open it through
                                // level 0's ground: beside it, or over it.
                                let highest = (y + v - 1).min(top);
                                let (i, j) = (
                                    x0.div_euclid(step).rem_euclid(16),
                                    z0.div_euclid(step).rem_euclid(16),
                                );
                                let line = |f: &dyn Fn(i32) -> (i32, i32)| -> Vec<(i32, i32)> {
                                    (0..step).map(f).collect()
                                };
                                let sides = [
                                    ((x0 + step, z0), i == 15, line(&|t| (x0 + step, z0 + t))),
                                    ((x0 - step, z0), i == 0, line(&|t| (x0 - 1, z0 + t))),
                                    ((x0, z0 + step), j == 15, line(&|t| (x0 + t, z0 + step))),
                                    ((x0, z0 - step), j == 0, line(&|t| (x0 + t, z0 - 1))),
                                ];
                                let beside = (y..=highest).any(|r| {
                                    sides.iter().any(|((nx, nz), edge, across)| {
                                        let last = level + (*edge && level < 3) as u32;
                                        !open(*nx, *nz, r) && under(across, last, r)
                                    })
                                });
                                let over = highest + 1;
                                let inside: Vec<(i32, i32)> = columns.iter().map(|c| (c.0, c.1)).collect();
                                let last = level + (over.rem_euclid(size) == 0 && level < 3) as u32;
                                let above = !open(x0, z0, over) && under(&inside, last, over);
                                let opens = beside || above;
                                assert_eq!(
                                    air,
                                    cave == ground && !opens,
                                    "{at}: {cave} of its {ground} blocks of ground are cave; a level would open it through level 0's ground: {opens}"
                                );
                                kept_for_what_is_beside += (cave == ground && opens) as u32;
                                carved += air as u32;
                                // What a vote would have opened: most of it cave.
                                whole_with_cave += (!air && 2 * cave >= ground) as u32;
                            }
                        }
                    }
                }
            }
            println!(
                "step {step}: {carved} cells carved, {whole_with_cave} whole though half cave or more, {deep} all cave but too deep"
            );
            assert!(
                carved >= 20 && whole_with_cave >= 20,
                "step {step}: {carved} carved, {whole_with_cave} kept whole: not enough to tell the rules apart"
            );
        }
        println!("{kept_for_what_is_beside} rows all cave, kept whole for what is beside them");
        assert!(
            kept_for_what_is_beside >= 10,
            "only {kept_for_what_is_beside} rows kept whole for what is beside them: not enough to tell the rules apart"
        );
    }

    #[test]
    fn cross_platform_block_array_is_bit_identical() {
        // §8.5: noise is floating point, thresholded to a BlockId here. The
        // FNV-1a hash below folds every cell of a fixed chunk at a fixed
        // seed into one constant; both macOS and Windows CI run this test,
        // so a cross-platform float divergence in the noise shows up as a
        // CI failure instead of silently drifting into different worlds for
        // different players. If this ever fails from a genuine platform
        // difference, the recorded fallback is fixed-point noise (§8.5) --
        // if it fails because worldgen intentionally changed, recompute and
        // update the constant only after confirming both platforms agree on
        // the new value.
        let gen = WorldGen::new(0xC0FFEE);
        let blocks = test_blocks();
        let chunk = gen.generate([0, 0, 0], 1, blocks);

        let mut hash: u64 = 0xcbf29ce484222325;
        for z in 0..Chunk::SIZE {
            for y in 0..Chunk::SIZE {
                for x in 0..Chunk::SIZE {
                    for b in chunk.get(x, y, z).0.to_le_bytes() {
                        hash ^= b as u64;
                        hash = hash.wrapping_mul(0x100000001b3);
                    }
                }
            }
        }
        assert_eq!(hash, 0x680f_9807_8f35_e325);
    }

    /// A world with one ore, tuned like `assets/ores/iron.ron` but with the
    /// synthetic palette the rest of these tests use.
    fn ore_blocks() -> TerrainBlocks {
        let mut ores = OreSet::EMPTY;
        ores.push(OreGen {
            block: BlockId(20),
            replaces: BlockId(3), // stone
            max_y: 40,
            threshold_milli: 800,
            freq_milli: 350,
            seed_mix: ore_seed_mix("cubara:iron_ore"),
        });
        TerrainBlocks {
            ores,
            ..test_blocks()
        }
    }

    const ORE: BlockId = BlockId(20);

    #[test]
    fn ore_generates_at_all() {
        // The tuning is data, but "some ore exists" is the floor below which
        // the whole block is pointless, and a typo in the threshold would sail
        // past every other test here.
        let gen = WorldGen::new(0x5EED);
        let blocks = ore_blocks();
        let found = (-40..40)
            .flat_map(|x| (0..40).map(move |y| (x, y)))
            .any(|(x, y)| gen.block_at(x, y, 7, blocks) == Some(ORE));
        assert!(found, "no ore anywhere in the probed volume");
    }

    #[test]
    fn ore_only_ever_replaces_stone() {
        // Never grass, never soil, never a tree block, never air. This is the
        // property that keeps ore from surfacing or hollowing out a trunk.
        let gen = WorldGen::new(0x5EED);
        let plain = ore_blocks();
        let bare = test_blocks();
        for x in -30..30 {
            for z in -30..30 {
                for y in 0..40 {
                    let with = gen.block_at(x, y, z, plain);
                    let without = gen.block_at(x, y, z, bare);
                    if with != without {
                        assert_eq!(with, Some(ORE), "at ({x},{y},{z})");
                        assert_eq!(
                            without,
                            Some(bare.stone),
                            "ore replaced something that was not stone at ({x},{y},{z})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn no_ore_above_its_max_y() {
        let gen = WorldGen::new(0x5EED);
        let blocks = ore_blocks();
        let max_y = blocks.ores.iter().next().unwrap().max_y;
        for x in -30..30 {
            for z in -30..30 {
                for y in (max_y + 1)..(max_y + 60) {
                    assert_ne!(
                        gen.block_at(x, y, z, blocks),
                        Some(ORE),
                        "ore above max_y at ({x},{y},{z})"
                    );
                }
            }
        }
    }

    #[test]
    fn ore_never_changes_whether_a_voxel_is_solid() {
        // The load-bearing property of §6, and the reason ore -- unlike a tree
        // -- is safe in `generate`/LOD: it is a material substitution, so
        // `is_solid` (raycast, player collision) need not know it exists. If
        // this ever fails, ore has become a structure and the three-path
        // problem is back.
        let gen = WorldGen::new(0x5EED);
        let plain = ore_blocks();
        let bare = test_blocks();
        for x in -30..30 {
            for z in -30..30 {
                for y in 0..50 {
                    assert_eq!(
                        gen.block_at(x, y, z, plain).is_some(),
                        gen.block_at(x, y, z, bare).is_some(),
                        "solidity changed at ({x},{y},{z})"
                    );
                }
            }
        }
    }

    #[test]
    fn an_ore_free_world_is_bit_identical_to_one_generated_before_ores_existed() {
        // `with_ores` is opt-in precisely so that every caller that does not
        // ask for ore -- the save fixture, the determinism harness, every
        // terrain-shape test -- generates exactly what it generated before.
        let gen = WorldGen::new(0x5EED);
        let a = gen.generate([0, 0, 0], 1, test_blocks());
        let b = gen.generate(
            [0, 0, 0],
            1,
            TerrainBlocks {
                ores: OreSet::EMPTY,
                ..test_blocks()
            },
        );
        assert!(chunks_equal(&a, &b));
    }

    #[test]
    fn the_same_seed_and_position_always_give_the_same_ore() {
        // Rule 1. The ore's seed mix comes from an FNV hash of its name rather
        // than `DefaultHasher`, which is not stable across Rust releases.
        let gen = WorldGen::new(0x5EED);
        let blocks = ore_blocks();
        let once = gen.generate([0, 0, 0], 1, blocks);
        let twice = gen.generate([0, 0, 0], 1, blocks);
        assert!(chunks_equal(&once, &twice));
        assert_eq!(
            ore_seed_mix("cubara:iron_ore"),
            ore_seed_mix("cubara:iron_ore")
        );
        assert_ne!(
            ore_seed_mix("cubara:iron_ore"),
            ore_seed_mix("cubara:coal_ore")
        );
    }

    #[test]
    fn lod_nodes_do_contain_ore() {
        // The deliberate opposite of `lod_nodes_contain_no_tree_blocks`. A
        // structure must not appear at level > 0 (one stray voxel); a material
        // must, or distant terrain would be a different colour than near
        // terrain and the LOD boundary would be visible as a seam.
        let gen = WorldGen::new(0x5EED);
        let blocks = ore_blocks();
        let node = gen.generate([-64, 0, -64], 8, blocks);
        let mut any = false;
        for z in 0..Chunk::SIZE {
            for y in 0..Chunk::SIZE {
                for x in 0..Chunk::SIZE {
                    if node.get(x, y, z) == ORE {
                        any = true;
                    }
                }
            }
        }
        assert!(any, "no ore in an LOD node -- §6 wants ore at every level");
    }
}

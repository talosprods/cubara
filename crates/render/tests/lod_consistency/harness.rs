//! The measurement behind `lod_consistency`: what each representation of the
//! ground shows over one footprint, as numbers that can be compared.
//!
//! **Voxels (levels 0-3)** are measured from the real thing: the node the
//! game generates (`World::node_at`) and the mesh the game's mesher makes of
//! it (`NodeChunk::build_mesh_with_neighbours`), quad by quad. A quad's
//! colour is its texture's mean, lit as `mesh.wgsl` lights it
//! ([`face_light`]) and darkened by its own baked ambient occlusion
//! ([`ao_fraction`]) -- the shader's two CPU twins, not a second formula.
//!
//! **Only what the sky can see is counted**: a top face is counted where it
//! is the highest in its column, and a side face where the cell in front of
//! it is above that column's top, each level judged against its own drawn
//! tops. That leaves out the undersides and every cave wall the sky cannot
//! reach.
//!
//! **What is counted is then split in two, and only the first is the
//! reference** (review of #428, finding B1):
//!
//! - **the surface** ([`Measure::faces`]): faces over ground whose level-0
//!   columns all stand at the height field (`surface_height`). This is the
//!   term every level is held to. On it level 0 is the height field's faces
//!   exactly -- `WorldGen::ground_faces` -- and almost stone-free on a slope.
//! - **the caves** ([`Measure::cave`]): faces of a cell, or facing a cell,
//!   under which a cave has carved some level-0 column's top away -- the
//!   floor of a cave mouth and the walls it bares. Reported, never asserted.
//!   A cave mouth is detail: level 0 draws every one, a coarse cell is cave
//!   only where all of its ground is (`WorldGen::row_is_cave`) and only near
//!   the surface, the far terrain draws none. Counted in with the surface,
//!   level 0's pits made a steep slope look 4% stone where its surface is
//!   0.1%, and every coarse level was being compared with that.
//!
//! The split is made on level 0's columns for every representation alike
//! ([`Setup::carved`]), cell by cell: a cell (or a far quad) with one carved
//! column under it goes to the cave term whole. A wider cell therefore
//! leaves more ground out -- on a steep slope 6% of level 0's sky-visible
//! area is cave term, 23% of level 3's -- so **level 0 is measured again for
//! every comparison, split on the cells of the level it is compared with**
//! ([`Drawn::measure`]'s `mask`): both sides of a comparison then cover the
//! same ground (second review of #428, point 1).
//!
//! **Exactly the same ground for the tops; not quite for the sides** (third
//! review of #428, S3). A side face goes to the cave term when the mask cell
//! of its own column *or of the column in front* is carved. A coarse level's
//! riser always stands between two cells, so it needs both whole. Level 0,
//! split on the same mask, keeps every riser between two columns of *one*
//! whole cell, whatever the cells around it are. So where a whole cell has
//! a carved neighbour, level 0 keeps the staircase inside it and the level
//! loses the one riser that stands for that staircase. The tops are not
//! touched by this (`compare` holds the two sides to the same count of
//! them). The side *area* is: on steep ground a level-3 node shows x0.75 of
//! level 0's side area on the surface term where over every face it shows
//! x0.81 (level 2: x0.89 against x0.92; cliffs at level 3: x0.83 against
//! x0.88 -- measured on the first version of #423, a throwaway probe). So
//! the report's `area / level 0's` column understates a level's risers by
//! six to eight percent at level 3 and three to four at level 2. T3 is a
//! mean colour and hardly moves with it. **Nothing may bound riser area
//! with the surface term's ratio as it stands**; it is reported, not
//! asserted.
//!
//! A column is *carved* where level 0 is not the height field: its drawn
//! top is not `surface_height + 1`, or a block of its side that a lower
//! neighbour bares is missing (a cave under an intact top opening a hole in
//! a riser). Everywhere else level 0's faces are the height field's, face
//! for face and layer for layer, which `level_0_draws_its_surface` asserts.
//!
//! Side faces on the footprint's own boundary are left out at every level:
//! what stands in front of them is outside the footprint. The boundary is a
//! multiple of [`FOOTPRINT`] blocks, which is a cell edge at every level, so
//! the same planes are left out everywhere.
//!
//! **The far terrain** is measured from the faces `far::mesh` makes -- the
//! tops and walls the game draws -- lit as `far.wgsl` lights them
//! ([`measure_far`]).

use std::collections::HashMap;

use cubara_render::aggregate::{ao_fraction, face_light, Materials};
use cubara_render::far::{face_colour, face_open};
use cubara_render::materials::mean_color;
use cubara_render::{FarDir, Lighting};
use cubara_voxel::{BlockRegistry, Face, MeshContext, NodeChunk};
use cubara_world::far::{self, PatchKey};
use cubara_world::node::NodeKey;
use cubara_world::{TerrainBlocks, World, WorldGen, SOIL_DEPTH};
use glam::Vec3;

#[path = "../common/far_faces.rs"]
mod far_faces_of;
use far_faces_of::far_faces;

/// The side of a footprint, in blocks: eight level-3 cells, and one
/// level-2 node. **Chosen, not derived**: large enough that a level-3
/// measurement has 64 cells in it, small enough that the level-0 chunks of
/// dozens of mountain footprints mesh in a debug `cargo test`.
pub const FOOTPRINT: i32 = 64;

/// The five directions the sky can see a face from, in the order every array
/// here uses.
pub const DIRECTIONS: [Face; 5] = [Face::PosY, Face::PosX, Face::NegX, Face::PosZ, Face::NegZ];

pub fn direction_name(d: usize) -> &'static str {
    ["+Y", "+X", "-X", "+Z", "-Z"][d]
}

fn direction_index(face: Face) -> Option<usize> {
    DIRECTIONS.iter().position(|&f| f == face)
}

/// A square of ground, [`FOOTPRINT`] blocks on a side, from `(x0, z0)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Footprint {
    pub x0: i32,
    pub z0: i32,
}

impl Footprint {
    fn holds(self, x: i32, z: i32) -> bool {
        (self.x0..self.x0 + FOOTPRINT).contains(&x) && (self.z0..self.z0 + FOOTPRINT).contains(&z)
    }
}

/// What a footprint's ground is like, from its level-0 surface heights.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Terrain {
    Flat,
    /// The owner's case: a grassy slope a player walks up.
    Gentle,
    Steep,
    Cliff,
}

pub const TERRAINS: [Terrain; 4] = [
    Terrain::Flat,
    Terrain::Gentle,
    Terrain::Steep,
    Terrain::Cliff,
];

/// The level-0 steps of a footprint: the mean and the largest height
/// difference between neighbouring columns, along x and along z.
pub fn step_stats(gen: &WorldGen, fp: Footprint) -> (f64, i32) {
    let h: Vec<i32> = (0..FOOTPRINT)
        .flat_map(|j| (0..FOOTPRINT).map(move |i| (i, j)))
        .map(|(i, j)| gen.surface_height(fp.x0 + i, fp.z0 + j))
        .collect();
    let at = |i: i32, j: i32| h[(j * FOOTPRINT + i) as usize];
    let (mut sum, mut n, mut max) = (0i64, 0i64, 0i32);
    for j in 0..FOOTPRINT {
        for i in 0..FOOTPRINT {
            for (di, dj) in [(1, 0), (0, 1)] {
                if i + di < FOOTPRINT && j + dj < FOOTPRINT {
                    let d = (at(i + di, j + dj) - at(i, j)).abs();
                    sum += d as i64;
                    n += 1;
                    max = max.max(d);
                }
            }
        }
    }
    (sum as f64 / n as f64, max)
}

/// Which kind of ground a footprint is, or `None` for one between the kinds.
///
/// **The bounds are chosen, not derived**, from the spread of `step_stats`
/// over the default world (`print_the_terrain_spread`): flat is the rolling
/// hills at or below their median (the world has nothing flatter), gentle a hillside whose steps are almost all one
/// block, steep a mountain face climbing more than a block per block, and a
/// cliff is anything holding a single step of six blocks or more.
pub fn classify(mean_step: f64, max_step: i32) -> Option<Terrain> {
    if max_step >= 6 {
        Some(Terrain::Cliff)
    } else if mean_step >= 1.0 {
        Some(Terrain::Steep)
    } else if (0.3..0.7).contains(&mean_step) && max_step <= 3 {
        Some(Terrain::Gentle)
    } else if mean_step <= 0.2 {
        Some(Terrain::Flat)
    } else {
        None
    }
}

/// SplitMix64: the harness's own seeded stream, so a failing footprint can be
/// named by the seed that drew it.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `-range..range`.
    fn within(&mut self, range: i32) -> i32 {
        (self.next() % (2 * range as u64)) as i32 - range
    }
}

/// How far from the origin footprints are drawn, in blocks. The default
/// seed's nearest range is about 370 blocks out and its mountains are
/// hundreds across, so this holds several.
const DRAW_RANGE: i32 = 6000;

/// `per_kind` footprints of each kind of terrain, drawn from `seed`: random
/// squares on the [`FOOTPRINT`] grid, each kept if its kind still has room.
/// Not hand-picked -- the seed decides, and a failure names it.
pub fn draw_footprints(gen: &WorldGen, seed: u64, per_kind: usize) -> Vec<(Terrain, Footprint)> {
    let mut rng = Rng(seed);
    let mut out: Vec<(Terrain, Footprint)> = Vec::new();
    let count = |out: &[(Terrain, Footprint)], t| out.iter().filter(|(k, _)| *k == t).count();
    for _ in 0..200_000 {
        if out.len() == per_kind * TERRAINS.len() {
            break;
        }
        let fp = Footprint {
            x0: rng.within(DRAW_RANGE).div_euclid(FOOTPRINT) * FOOTPRINT,
            z0: rng.within(DRAW_RANGE).div_euclid(FOOTPRINT) * FOOTPRINT,
        };
        // A cheap look first: most of the world is hills, and the full
        // statistics of a footprint are 4,096 heights.
        let corner_relief = {
            let hs = [0, FOOTPRINT / 2, FOOTPRINT - 1].map(|d| {
                [0, FOOTPRINT / 2, FOOTPRINT - 1].map(|e| gen.surface_height(fp.x0 + d, fp.z0 + e))
            });
            let flat: Vec<i32> = hs.iter().flatten().copied().collect();
            flat.iter().max().unwrap() - flat.iter().min().unwrap()
        };
        let wants_relief = count(&out, Terrain::Steep) < per_kind
            || count(&out, Terrain::Cliff) < per_kind
            || count(&out, Terrain::Gentle) < per_kind;
        let wants_flat = count(&out, Terrain::Flat) < per_kind;
        if (corner_relief >= 8 && !wants_relief) || (corner_relief < 8 && !wants_flat) {
            continue;
        }
        let (mean, max) = step_stats(gen, fp);
        let Some(kind) = classify(mean, max) else {
            continue;
        };
        if count(&out, kind) < per_kind && !out.iter().any(|(_, f)| *f == fp) {
            out.push((kind, fp));
        }
    }
    out.sort();
    out
}

impl PartialOrd for Footprint {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Footprint {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.x0, self.z0).cmp(&(other.x0, other.z0))
    }
}

/// The faces the sky can see over a footprint, summed per direction.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Faces {
    /// Face area, in square blocks.
    pub area: [f64; 5],
    /// The lit colour (linear light), times area.
    pub lit: [[f64; 3]; 5],
    /// The area that is stone, soil, and the surface block (its top or its
    /// side), in that order.
    pub material: [[f64; 3]; 5],
    /// How open the faces are, times area: the mean over a face's corners of
    /// its baked ambient-occlusion level, 0 to 1 -- `GroundFaces`' openness,
    /// as the mesher baked it.
    pub open: [f64; 5],
}

pub const STONE: usize = 0;
pub const SOIL: usize = 1;
pub const SURFACE: usize = 2;

impl Faces {
    pub fn add(&mut self, other: &Faces) {
        for d in 0..5 {
            self.area[d] += other.area[d];
            self.open[d] += other.open[d];
            for c in 0..3 {
                self.lit[d][c] += other.lit[d][c];
                self.material[d][c] += other.material[d][c];
            }
        }
    }

    /// The area-weighted lit colour of direction `d`, or `None` with no face.
    pub fn mean_lit(&self, d: usize) -> Option<[f64; 3]> {
        (self.area[d] > 0.0).then(|| self.lit[d].map(|c| c / self.area[d]))
    }

    /// The share of direction `d`'s area that is `material`.
    pub fn share(&self, d: usize, material: usize) -> Option<f64> {
        (self.area[d] > 0.0).then(|| self.material[d][material] / self.area[d])
    }

    /// The share of all side faces' area that is `material`.
    pub fn side_share(&self, material: usize) -> Option<f64> {
        let area: f64 = self.area[1..].iter().sum();
        let of: f64 = self.material[1..].iter().map(|m| m[material]).sum();
        (area > 0.0).then(|| of / area)
    }

    /// The share of every face's area, tops included, that is `material`.
    pub fn total_share(&self, material: usize) -> Option<f64> {
        let area: f64 = self.area.iter().sum();
        let of: f64 = self.material.iter().map(|m| m[material]).sum();
        (area > 0.0).then(|| of / area)
    }
}

/// The largest channel difference between two colours, on the 0-255 scale
/// (linear light, as T3 has been measured since R1c, issue #327).
pub fn colour_delta(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3)
        .map(|c| (a[c] - b[c]).abs() * 255.0)
        .fold(0.0, f64::max)
}

/// One representation of one footprint.
#[derive(Clone, Debug)]
pub struct Measure {
    /// The surface term: the reference, and what is asserted.
    pub faces: Faces,
    /// The cave term: faces on or beside ground a cave has carved at level
    /// 0. Reported only.
    pub cave: Faces,
    /// The drawn top face's height over each level-0 column, row by row (`z`
    /// outer): `surface + 1` at level 0. `NaN` where nothing is drawn.
    pub tops: Vec<f64>,
}

impl Measure {
    pub fn top(&self, i: i32, j: i32) -> f64 {
        self.tops[(j * FOOTPRINT + i) as usize]
    }
}

/// Per footprint (by its corner), which of its level-0 columns are carved.
type CarvedMasks = HashMap<(i32, i32), std::rc::Rc<Vec<bool>>>;

/// What the harness needs of the game's data, loaded once.
pub struct Setup {
    pub world: World,
    pub gen: WorldGen,
    pub registry: BlockRegistry,
    pub blocks: TerrainBlocks,
    pub lighting: Lighting,
    /// Texture names, in layer order.
    textures: Vec<String>,
    /// Each layer's mean colour, linear.
    colours: Vec<Vec3>,
    /// Each layer's material: [`STONE`], [`SOIL`] or [`SURFACE`].
    materials: Vec<Option<usize>>,
    pub far_materials: Materials,
    /// Per footprint, which level-0 columns a cave has carved ([`Setup::carved`]).
    carved: std::cell::RefCell<CarvedMasks>,
}

impl Setup {
    /// The default world, the real block registry and textures, the default
    /// lighting. No trees and no ores: both are other systems (R2 draws
    /// trees beyond level 0; ore never surfaces).
    pub fn new() -> Self {
        let assets = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/blocks");
        let registry = BlockRegistry::load(&assets).expect("assets/blocks must be valid");
        let blocks = TerrainBlocks::from_registry(&registry);
        let world = World::new();
        let gen = WorldGen::new(world.seed());
        let textures: Vec<String> = registry
            .texture_names()
            .into_iter()
            .map(str::to_string)
            .collect();
        let colours = textures
            .iter()
            .map(|name| Vec3::from(mean_color(name).unwrap_or_else(|| panic!("no texture {name}"))))
            .collect();
        let materials = textures
            .iter()
            .map(|name| match name.as_str() {
                "stone" => Some(STONE),
                "soil" => Some(SOIL),
                "grass_top" | "grass_side" => Some(SURFACE),
                _ => None,
            })
            .collect();
        Self {
            world,
            gen,
            registry,
            blocks,
            lighting: Lighting::default(),
            textures,
            colours,
            materials,
            far_materials: cubara_render::far::ground_materials(SOIL_DEPTH as f32),
            carved: Default::default(),
        }
    }

    /// Which level-0 columns of `fp` a cave has carved, row by row (`z`
    /// outer): those whose drawn top is not `surface_height + 1` (or that
    /// draw nothing near it), and those with a hole in a side a lower
    /// neighbour bares. From the real level-0 chunks, once per footprint;
    /// every representation's faces are split on it.
    pub fn carved(&self, fp: Footprint) -> std::rc::Rc<Vec<bool>> {
        if let Some(mask) = self.carved.borrow().get(&(fp.x0, fp.z0)) {
            return mask.clone();
        }
        // Drawing level 0 works the mask out and files it.
        draw_voxels(self, fp, 0, &|node| self.world.node_at(node, self.blocks));
        self.carved.borrow()[&(fp.x0, fp.z0)].clone()
    }

    /// Which level-0 columns of `fp` are in the surface term of a
    /// representation whose cells are `w` blocks wide, row by row (`z`
    /// outer): those whose cell stands on ground no cave has carved. The
    /// split the faces are counted by, for the shape ([`Shape::over`]).
    pub fn on_the_surface(&self, fp: Footprint, w: i32) -> Vec<bool> {
        let carved = self.carved(fp);
        let w = w.clamp(1, FOOTPRINT);
        (0..FOOTPRINT)
            .flat_map(|j| (0..FOOTPRINT).map(move |i| (i, j)))
            .map(|(i, j)| !any_carved(&carved, fp, fp.x0 + i / w * w, fp.z0 + j / w * w, w))
            .collect()
    }

    pub fn layer_of(&self, name: &str) -> u32 {
        self.textures
            .iter()
            .position(|t| t == name)
            .unwrap_or_else(|| panic!("no layer for {name}")) as u32
    }
}

/// A node's generated cells, for whoever stands in for `World::node_at`.
pub type NodeSource<'a> = &'a dyn Fn(NodeKey) -> Option<NodeChunk>;

/// The voxels of `level` over `fp`, as the game generates and meshes them,
/// split on the level's own cells.
pub fn measure_voxels(setup: &Setup, fp: Footprint, level: u32) -> Measure {
    measure_voxels_from(setup, fp, level, &|node| {
        setup.world.node_at(node, setup.blocks)
    })
}

/// [`measure_voxels`], with the nodes' cells from `source` -- so a test can
/// hand in a node that is wrong on purpose and see the measurement move.
pub fn measure_voxels_from(
    setup: &Setup,
    fp: Footprint,
    level: u32,
    source: NodeSource,
) -> Measure {
    let drawn = draw_voxels(setup, fp, level, source);
    drawn.measure(setup, 0, drawn.w)
}

/// One level's voxels over a footprint, generated and meshed: what
/// [`Drawn::measure`] counts, as often as there are masks to count it by.
pub struct Drawn {
    fp: Footprint,
    /// A cell's width, in blocks. (Its faces are walked in block rows: a
    /// mesh's positions are in blocks whatever rows the node stores.)
    pub w: i32,
    /// Each lattice column's drawn top: one above its highest solid cell.
    top: Vec<Option<i32>>,
    /// Every node's mesh, with the node's origin.
    meshes: Vec<([i32; 3], cubara_voxel::Mesh)>,
}

/// Generate and mesh the voxels of `level` over `fp`, the nodes' cells from
/// `source`.
pub fn draw_voxels(setup: &Setup, fp: Footprint, level: u32, source: NodeSource) -> Drawn {
    let w = NodeKey::new(level, [0, 0, 0]).extent_chunks();
    let node_size = 16 * w;
    let cells = FOOTPRINT / w;
    let mut top: Vec<Option<i32>> = vec![None; (cells * cells) as usize];
    let column = |x: i32, z: i32| -> Option<usize> {
        fp.holds(x, z)
            .then(|| (((z - fp.z0) / w) * cells + (x - fp.x0) / w) as usize)
    };

    let ctx = MeshContext {
        registry: &setup.registry,
        layer_of: &|name| setup.layer_of(name),
    };
    let mut meshes = Vec::new();
    let node_range =
        |lo: i32| lo.div_euclid(node_size)..=(lo + FOOTPRINT - 1).div_euclid(node_size);
    for nz in node_range(fp.z0) {
        for nx in node_range(fp.x0) {
            // The rows of nodes that can hold this node column's surface:
            // its level-0 heights over the part of it inside the footprint,
            // with room either side for a coarse level's own rounding and
            // for a cave mouth under the surface.
            let (mut lo, mut hi) = (i32::MAX, i32::MIN);
            for z in (nz * node_size).max(fp.z0)..((nz + 1) * node_size).min(fp.z0 + FOOTPRINT) {
                for x in (nx * node_size).max(fp.x0)..((nx + 1) * node_size).min(fp.x0 + FOOTPRINT)
                {
                    let h = setup.gen.surface_height(x, z);
                    lo = lo.min(h);
                    hi = hi.max(h);
                }
            }
            for ny in (lo - SURFACE_MARGIN).div_euclid(node_size)
                ..=(hi + SURFACE_MARGIN).div_euclid(node_size)
            {
                let node = NodeKey::new(level, [nx, ny, nz]);
                let Some(chunk) = source(node) else {
                    continue;
                };
                let origin = node.world_origin();
                // What the node draws, on the lattice it is drawn on: level
                // 3's rows of four are cut off at its exact tops.
                let (dims, rows) = (chunk.mesh_dims(), chunk.mesh_scale()[1]);
                for lz in 0..dims[2] {
                    for lx in 0..dims[0] {
                        let (x, z) = (origin[0] + lx as i32 * w, origin[2] + lz as i32 * w);
                        let Some(c) = column(x, z) else {
                            continue;
                        };
                        let solid = (0..dims[1])
                            .rev()
                            .find(|&ly| setup.registry.is_solid(chunk.drawn(lx, ly, lz)));
                        if let Some(ly) = solid {
                            let face = origin[1] + (ly as i32 + 1) * rows;
                            top[c] = Some(top[c].map_or(face, |t| t.max(face)));
                        }
                    }
                }
                // Every border face drawn and no skirts: what a neighbour
                // covers is decided below, from the tops, for every face
                // alike.
                let mesh = chunk.build_mesh_with_neighbours(&ctx, |_, _, _| false, |_, _, _| false);
                meshes.push((origin, mesh));
            }
        }
    }

    // The ground a cave has carved, from level 0's own columns: found here
    // and filed for every representation's split.
    if level == 0 {
        let mask: Vec<bool> = (0..FOOTPRINT)
            .flat_map(|j| (0..FOOTPRINT).map(move |i| (i, j)))
            .map(|(i, j)| {
                let (x, z) = (fp.x0 + i, fp.z0 + j);
                let surface = setup.gen.surface_height(x, z);
                if top[column(x, z).expect("inside the footprint")] != Some(surface + 1) {
                    return true;
                }
                // A hole in a side a lower neighbour bares: a cave under an
                // intact top.
                let lowest = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                    .into_iter()
                    .map(|(dx, dz)| setup.gen.surface_height(x + dx, z + dz))
                    .min()
                    .expect("four neighbours");
                (lowest + 1..surface).any(|y| !setup.world.is_solid_at(x, y, z, setup.blocks))
            })
            .collect();
        setup
            .carved
            .borrow_mut()
            .entry((fp.x0, fp.z0))
            .or_insert_with(|| std::rc::Rc::new(mask));
    }
    Drawn { fp, w, top, meshes }
}

/// [`measure_voxels_from`], counting only the faces of columns at least
/// `inner` blocks inside the footprint's edge. At `inner = 0` a side face
/// counts when the column in front of it is in the footprint too; further
/// in, every face of a counted column has its front column in the footprint,
/// so nothing is left out -- which is what lets the count be held against
/// `WorldGen::ground_faces` over the same inner square, face for face.
pub fn measure_voxels_within(
    setup: &Setup,
    fp: Footprint,
    level: u32,
    source: NodeSource,
    inner: i32,
) -> Measure {
    let drawn = draw_voxels(setup, fp, level, source);
    drawn.measure(setup, inner, drawn.w)
}

impl Drawn {
    /// What the sky sees of these voxels, in the two terms (`harness.rs`'s
    /// module doc): the faces of columns at least `inner` blocks inside the
    /// footprint, split on cells `mask` blocks wide -- the level's own width
    /// for the level itself, and **the width of the level it is compared
    /// with for level 0**, so that the reference and the level it is held
    /// against leave out exactly the same ground.
    pub fn measure(&self, setup: &Setup, inner: i32, mask: i32) -> Measure {
        let (fp, w) = (self.fp, self.w);
        assert!(
            mask >= w && mask % w == 0,
            "a mask of {mask} blocks cannot split cells of {w}"
        );
        let (top, meshes) = (&self.top, &self.meshes);
        let cells = FOOTPRINT / w;
        let column = |x: i32, z: i32| -> Option<usize> {
            fp.holds(x, z)
                .then(|| (((z - fp.z0) / w) * cells + (x - fp.x0) / w) as usize)
        };
        let counted = |x: i32, z: i32| {
            (fp.x0 + inner..fp.x0 + FOOTPRINT - inner).contains(&x)
                && (fp.z0 + inner..fp.z0 + FOOTPRINT - inner).contains(&z)
        };
        let carved = setup.carved(fp);
        let mask = mask.min(FOOTPRINT);
        let cell_carved = |x: i32, z: i32| {
            let (mx, mz) = (
                fp.x0 + (x - fp.x0).div_euclid(mask) * mask,
                fp.z0 + (z - fp.z0).div_euclid(mask) * mask,
            );
            any_carved(&carved, fp, mx, mz, mask)
        };

        let mut faces = Faces::default();
        let mut cave = Faces::default();
        for (origin, mesh) in meshes {
            for quad in mesh.indices.chunks(6) {
                let base = *quad.iter().min().expect("six indices") as usize;
                let corners = &mesh.vertices[base..base + 4];
                let face = corners[0].face();
                let Some(d) = direction_index(face) else {
                    continue;
                };
                let ao: Vec<f64> = corners
                    .iter()
                    .map(|c| ao_fraction(c.ao(), c.ao_width_log2(), setup.lighting.ao_floor) as f64)
                    .collect();
                // The mean of the two triangles the quad is drawn as: the corners
                // on the shared diagonal count in both.
                let ao_mean = if quad[0] as usize == base {
                    (2.0 * (ao[0] + ao[2]) + ao[1] + ao[3]) / 6.0
                } else {
                    (2.0 * (ao[1] + ao[3]) + ao[0] + ao[2]) / 6.0
                };
                let layer = corners[0].tex_layer() as usize;
                let light = face_light(Vec3::from(face.normal()), &setup.lighting);
                let lit = setup.colours[layer] * light;
                let pos = |c: &cubara_voxel::Vertex| {
                    [
                        origin[0] + c.x() as i32,
                        origin[1] + c.y() as i32,
                        origin[2] + c.z() as i32,
                    ]
                };
                let min = corners.iter().map(pos).fold([i32::MAX; 3], |a, p| {
                    [a[0].min(p[0]), a[1].min(p[1]), a[2].min(p[2])]
                });
                let max = corners.iter().map(pos).fold([i32::MIN; 3], |a, p| {
                    [a[0].max(p[0]), a[1].max(p[1]), a[2].max(p[2])]
                });
                // The area of this quad the sky sees, cell by cell.
                // ... on the surface, and on or beside carved ground.
                let mut seen = [0f64; 2];
                match face {
                    Face::PosY => {
                        for z in (min[2]..max[2]).step_by(w as usize) {
                            for x in (min[0]..max[0]).step_by(w as usize) {
                                if counted(x, z)
                                    && column(x, z).is_some_and(|c| top[c] == Some(min[1]))
                                {
                                    seen[cell_carved(x, z) as usize] += (w * w) as f64;
                                }
                            }
                        }
                    }
                    Face::PosX | Face::NegX | Face::PosZ | Face::NegZ => {
                        let axis = if matches!(face, Face::PosX | Face::NegX) {
                            0
                        } else {
                            2
                        };
                        let along = 2 - axis;
                        let positive = matches!(face, Face::PosX | Face::PosZ);
                        for a in (min[along]..max[along]).step_by(w as usize) {
                            // The cell behind the face, and the one in front.
                            let mut own = [0i32; 3];
                            let mut front = [0i32; 3];
                            own[along] = a;
                            front[along] = a;
                            own[axis] = if positive { min[axis] - w } else { min[axis] };
                            front[axis] = if positive { min[axis] } else { min[axis] - w };
                            let (Some(_), Some(f)) =
                                (column(own[0], own[2]), column(front[0], front[2]))
                            else {
                                continue;
                            };
                            if !counted(own[0], own[2]) {
                                continue;
                            }
                            for y in min[1]..max[1] {
                                // Open sky in front: the column there is drawn no
                                // higher than this block row's foot (or not at
                                // all). Row by row of blocks -- a top is at any
                                // block, whatever rows the node stores.
                                if top[f].is_none_or(|t| t <= y) {
                                    let beside_a_cave = cell_carved(own[0], own[2])
                                        || cell_carved(front[0], front[2]);
                                    seen[beside_a_cave as usize] += w as f64;
                                }
                            }
                        }
                    }
                    Face::NegY => unreachable!("filtered above"),
                }
                let open = corners.iter().map(|c| c.ao() as f64).sum::<f64>() / 12.0;
                for (term, area) in [(&mut faces, seen[0]), (&mut cave, seen[1])] {
                    if area == 0.0 {
                        continue;
                    }
                    term.area[d] += area;
                    term.open[d] += open * area;
                    for c in 0..3 {
                        term.lit[d][c] += lit[c] as f64 * ao_mean * area;
                    }
                    if let Some(m) = setup.materials[layer] {
                        term.material[d][m] += area;
                    }
                }
            }
        }

        let tops = (0..FOOTPRINT)
            .flat_map(|j| (0..FOOTPRINT).map(move |i| (i, j)))
            .map(|(i, j)| {
                let c = column(fp.x0 + i, fp.z0 + j).expect("inside the footprint");
                top[c].map_or(f64::NAN, |t| t as f64)
            })
            .collect();
        Measure { faces, cave, tops }
    }
}

impl Drawn {
    /// Per level-0 column (row by row, `z` outer): how much of its light the
    /// baked ambient occlusion leaves the sky-visible top face over it --
    /// the mean over the quad that covers it, as [`Drawn::measure`] takes
    /// it. `NaN` where the sky sees no top. For taking a level's shade apart
    /// (`top_shade_by_riser_report`), not for a tolerance.
    pub fn top_light(&self, setup: &Setup) -> Vec<f64> {
        let (fp, w) = (self.fp, self.w);
        let cells = FOOTPRINT / w;
        let mut out = vec![f64::NAN; (FOOTPRINT * FOOTPRINT) as usize];
        for (origin, mesh) in &self.meshes {
            for quad in mesh.indices.chunks(6) {
                let base = *quad.iter().min().expect("six indices") as usize;
                let corners = &mesh.vertices[base..base + 4];
                if corners[0].face() != Face::PosY {
                    continue;
                }
                let ao: Vec<f64> = corners
                    .iter()
                    .map(|c| ao_fraction(c.ao(), c.ao_width_log2(), setup.lighting.ao_floor) as f64)
                    .collect();
                let light = if quad[0] as usize == base {
                    (2.0 * (ao[0] + ao[2]) + ao[1] + ao[3]) / 6.0
                } else {
                    (2.0 * (ao[1] + ao[3]) + ao[0] + ao[2]) / 6.0
                };
                // Back out the floor: 1 is unshaded, 0 as dark as it gets.
                let lo = |f: fn(&cubara_voxel::Vertex) -> u32, o: i32| {
                    o + corners.iter().map(f).min().expect("four") as i32
                };
                let hi = |f: fn(&cubara_voxel::Vertex) -> u32, o: i32| {
                    o + corners.iter().map(f).max().expect("four") as i32
                };
                let y = lo(|c| c.y(), origin[1]);
                for z in lo(|c| c.z(), origin[2])..hi(|c| c.z(), origin[2]) {
                    for x in lo(|c| c.x(), origin[0])..hi(|c| c.x(), origin[0]) {
                        if !fp.holds(x, z) {
                            continue;
                        }
                        let cell = (((z - fp.z0) / w) * cells + (x - fp.x0) / w) as usize;
                        if self.top[cell] == Some(y) {
                            out[((z - fp.z0) * FOOTPRINT + (x - fp.x0)) as usize] = light;
                        }
                    }
                }
            }
        }
        out
    }
}

/// What `level` draws of `fp` block by block, as the game generates it: for
/// every column of the footprint (row by row, `z` outer) and every block row
/// of `rows`, whether the level draws air there. A coarse level's cell
/// speaks for every column it covers; level 0 is the world's own blocks.
pub fn drawn_air(
    setup: &Setup,
    fp: Footprint,
    level: u32,
    rows: std::ops::Range<i32>,
) -> Vec<bool> {
    let w = NodeKey::new(level, [0, 0, 0]).extent_chunks();
    let node_size = 16 * w;
    let height = rows.len();
    let mut air = vec![true; (FOOTPRINT * FOOTPRINT) as usize * height];
    let node_range =
        |lo: i32, len: i32| lo.div_euclid(node_size)..=(lo + len - 1).div_euclid(node_size);
    for nz in node_range(fp.z0, FOOTPRINT) {
        for nx in node_range(fp.x0, FOOTPRINT) {
            for ny in node_range(rows.start, height as i32) {
                let node = NodeKey::new(level, [nx, ny, nz]);
                // Nothing generated is nothing drawn: air.
                let Some(chunk) = setup.world.node_at(node, setup.blocks) else {
                    continue;
                };
                let origin = node.world_origin();
                let (dims, scale) = (chunk.mesh_dims(), chunk.mesh_scale());
                assert_eq!(scale, [w, 1, w], "level {level} is drawn in block rows");
                for lz in 0..dims[2] {
                    for lx in 0..dims[0] {
                        let (x, z) = (origin[0] + lx as i32 * w, origin[2] + lz as i32 * w);
                        for ly in 0..dims[1] {
                            let y = origin[1] + ly as i32;
                            if !rows.contains(&y)
                                || !setup.registry.is_solid(chunk.drawn(lx, ly, lz))
                            {
                                continue;
                            }
                            for cz in z.max(fp.z0)..(z + w).min(fp.z0 + FOOTPRINT) {
                                for cx in x.max(fp.x0)..(x + w).min(fp.x0 + FOOTPRINT) {
                                    let column = ((cz - fp.z0) * FOOTPRINT + cx - fp.x0) as usize;
                                    air[column * height + (y - rows.start) as usize] = false;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    air
}

/// Whether a cave has carved any level-0 column under the `w`-block cell at
/// `(x, z)` (clipped to the footprint; a cell outside it is not).
fn any_carved(carved: &[bool], fp: Footprint, x: i32, z: i32, w: i32) -> bool {
    (z.max(fp.z0)..(z + w).min(fp.z0 + FOOTPRINT)).any(|cz| {
        (x.max(fp.x0)..(x + w).min(fp.x0 + FOOTPRINT))
            .any(|cx| carved[((cz - fp.z0) * FOOTPRINT + (cx - fp.x0)) as usize])
    })
}

/// How far above and below a node column's level-0 surface its nodes are
/// generated, in blocks. **Chosen, not derived**: above, a cell's top is no
/// higher than its own highest column, so any margin does; below, a cave
/// mouth can lower a column's top, and one deeper than this is counted as
/// nothing drawn.
const SURFACE_MARGIN: i32 = 12;

/// The far terrain's patches of `level` (quads of `2^level` blocks, up to a
/// footprint's width) over `fp`, **as it is drawn** (issue #422): the game's
/// own patches (`far::generate`) and the faces its mesher made of them
/// (`far::mesh`), face by face -- each lit as `far.wgsl` lights it, by the
/// shader's CPU twins ([`face_colour`], [`face_open`], [`face_light`];
/// `golden.rs`'s `far_shading_is_the_face_reference` holds the shader to
/// them).
///
/// Counted as the voxels are: a top over each of its quads, a wall as tall
/// as the step it shows (a patch's skirt hangs under its neighbour's ground
/// and the sky never sees it), and only where the quad it belongs to and the
/// quad it looks out over are both in the footprint. A quad over ground a
/// cave has carved at level 0 goes to the cave term, and so does a wall of
/// it or facing it: the far terrain draws no cave, and what it draws there
/// is not the surface's.
pub fn measure_far(setup: &Setup, fp: Footprint, level: u32) -> Measure {
    let q = 1i64 << level;
    assert!(
        q <= FOOTPRINT as i64,
        "a quad of {q} blocks is wider than a footprint"
    );
    let carved = setup.carved(fp);
    let holds = |(x, z): (i64, i64)| fp.holds(x as i32, z as i32);
    let quad_carved = |(x, z): (i64, i64)| any_carved(&carved, fp, x as i32, z as i32, q as i32);

    let mut keys = std::collections::BTreeSet::new();
    for z in (fp.z0 as i64..(fp.z0 + FOOTPRINT) as i64).step_by(q as usize) {
        for x in (fp.x0 as i64..(fp.x0 + FOOTPRINT) as i64).step_by(q as usize) {
            keys.insert(PatchKey::containing(level, x, z));
        }
    }
    let mut faces = Faces::default();
    let mut cave = Faces::default();
    let mut tops = vec![f64::NAN; (FOOTPRINT * FOOTPRINT) as usize];
    for key in keys {
        let patch = far::generate(&setup.gen, key);
        let [ox, oz] = key.origin();
        for face in far_faces(&patch.faces) {
            // `FarDir` and `DIRECTIONS` are in one order.
            let d = face.dir as usize;
            let normal = face.dir.normal();
            debug_assert_eq!(Vec3::from(DIRECTIONS[d].normal()), normal);
            let lit = face_colour(&setup.far_materials, &face)
                * face_open(&face, setup.lighting.ao_floor)
                * face_light(normal, &setup.lighting);
            // What the face is made of: stone, soil, the surface block.
            let (surface, soil) = match face.dir {
                FarDir::Top => (1.0, 0.0),
                _ => (face.mix[0] as f64 / 255.0, face.mix[1] as f64 / 255.0),
            };
            let made_of = [1.0 - surface - soil, soil, surface];
            debug_assert_eq!((STONE, SOIL, SURFACE), (0, 1, 2));
            // The area of this face the sky sees: on the surface, and on or
            // beside carved ground.
            let mut seen = [0f64; 2];
            if face.dir == FarDir::Top {
                for j in face.j0..face.j1 {
                    for i in face.i0..face.i1 {
                        let quad = (ox + i as i64 * q, oz + j as i64 * q);
                        if !holds(quad) {
                            continue;
                        }
                        seen[quad_carved(quad) as usize] += (q * q) as f64;
                        for b in 0..q {
                            for a in 0..q {
                                let (ci, cj) =
                                    (quad.0 + a - fp.x0 as i64, quad.1 + b - fp.z0 as i64);
                                tops[(cj * FOOTPRINT as i64 + ci) as usize] = face.top as f64;
                            }
                        }
                    }
                }
            } else {
                // A wall stands on a quad line and runs along the other
                // axis: the quad behind it is its own, the one in front is
                // the lower ground it looks out over.
                let along_x = face.i0 == face.i1;
                let positive = matches!(face.dir, FarDir::PosX | FarDir::PosZ);
                let (line, run) = if along_x {
                    (ox + face.i0 as i64 * q, face.j0..face.j1)
                } else {
                    (oz + face.j0 as i64 * q, face.i0..face.i1)
                };
                let (own, front) = if positive {
                    (line - q, line)
                } else {
                    (line, line - q)
                };
                for k in run {
                    let (own, front) = if along_x {
                        let z = oz + k as i64 * q;
                        ((own, z), (front, z))
                    } else {
                        let x = ox + k as i64 * q;
                        ((x, own), (x, front))
                    };
                    if !(holds(own) && holds(front)) {
                        continue;
                    }
                    let beside_a_cave = quad_carved(own) || quad_carved(front);
                    seen[beside_a_cave as usize] += (face.rise as i64 * q) as f64;
                }
            }
            for (term, area) in [(&mut faces, seen[0]), (&mut cave, seen[1])] {
                if area == 0.0 {
                    continue;
                }
                term.area[d] += area;
                term.open[d] += face.open as f64 / 255.0 * area;
                for c in 0..3 {
                    term.lit[d][c] += lit[c] as f64 * area;
                    term.material[d][c] += made_of[c] * area;
                }
            }
        }
    }
    Measure { faces, cave, tops }
}

/// How a stepped height field of `cell`-block squares takes each square's
/// top from the level-0 columns it covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TopRule {
    /// `WorldGen::cell_top`: the mean of the covered columns, and the
    /// highest of them where the square holds a summit (#423).
    Exact,
    /// `WorldGen::cell_top_sampled` over an `n` x `n` grid of the covered
    /// columns: what a quad too large to visit every column of affords.
    Sampled(i32),
    /// For comparison only, and not in `WorldGen`: the highest covered
    /// column (the first version of #423).
    Max,
}

/// The tops a stepped height field of `cell`-block squares would draw over
/// `fp` under `rule`: a model of a proposal, from the height field alone, for
/// deciding with numbers (no faces: it has no colours yet).
pub fn measure_stepped(gen: &WorldGen, fp: Footprint, cell: i32, rule: TopRule) -> Measure {
    let h = |i: i32, j: i32| gen.surface_height(fp.x0 + i, fp.z0 + j);
    let mut tops = vec![f64::NAN; (FOOTPRINT * FOOTPRINT) as usize];
    for cj in (0..FOOTPRINT).step_by(cell as usize) {
        for ci in (0..FOOTPRINT).step_by(cell as usize) {
            let columns = |n: i32| {
                let step = (cell / n).max(1);
                (0..cell)
                    .step_by(step as usize)
                    .flat_map(move |b| (0..cell).step_by(step as usize).map(move |a| (a, b)))
            };
            let (x0, z0) = (fp.x0 + ci, fp.z0 + cj);
            let top = match rule {
                TopRule::Exact => gen.cell_top(x0, z0, cell),
                TopRule::Sampled(n) => gen.cell_top_sampled(x0, z0, cell, n),
                TopRule::Max => columns(cell)
                    .map(|(a, b)| h(ci + a, cj + b))
                    .max()
                    .expect("a square covers a column"),
            };
            for b in 0..cell {
                for a in 0..cell {
                    tops[((cj + b) * FOOTPRINT + ci + a) as usize] = top as f64 + 1.0;
                }
            }
        }
    }
    Measure {
        faces: Faces::default(),
        cave: Faces::default(),
        tops,
    }
}

/// How one representation's shape differs from level 0's, over one footprint
/// or several pooled.
///
/// **Level 0 here is the height field**: `surface_height + 1`, the top face
/// of every column's surface block, which is what #423's decisions 1 and 4
/// name. A cave mouth lowers a level-0 column's drawn top below that, and
/// caves have their own rule for coarse cells (`WorldGen::row_is_cave`: all
/// of the cell's ground), so they are not what is compared --
/// `level_0_draws_its_surface` holds the drawn level 0 to the height field
/// everywhere else.
#[derive(Clone, Debug, Default)]
pub struct Shape {
    /// `drawn - level 0`, per column, in blocks.
    pub column: Vec<f64>,
    /// The same for the skyline: the highest top along each row and each
    /// line of columns, which is the ridge a viewer level with the ground
    /// sees against the sky.
    pub skyline: Vec<f64>,
    /// The same for the footprint's single highest top: the peak.
    pub peak: Vec<f64>,
    /// Columns where this representation draws nothing.
    pub undrawn: usize,
    /// Columns left out of the comparison: on ground a cave has carved
    /// ([`Shape::over`]).
    pub left_out: usize,
}

/// The level-0 height field over `fp`, as a [`Measure`] with no faces.
pub fn height_field(gen: &WorldGen, fp: Footprint) -> Measure {
    Measure {
        faces: Faces::default(),
        cave: Faces::default(),
        tops: (0..FOOTPRINT)
            .flat_map(|j| (0..FOOTPRINT).map(move |i| (i, j)))
            .map(|(i, j)| gen.surface_height(fp.x0 + i, fp.z0 + j) as f64 + 1.0)
            .collect(),
    }
}

impl Shape {
    /// `other` against the level-0 height field `level0`, over every column.
    pub fn of(level0: &Measure, other: &Measure) -> Self {
        Self::over(level0, other, &[])
    }

    /// [`Shape::of`] over the columns `counted` only (row by row, `z` outer;
    /// empty for all of them) -- the same columns on both sides. With
    /// [`Setup::on_the_surface`] this is the shape of the surface term, the
    /// ground a level is held to: a cave mouth lowers a drawn top by as much
    /// as the cave is deep, and one under a footprint's highest column would
    /// otherwise be read as a lost peak (level 0 itself "lost" six blocks of
    /// one that way).
    pub fn over(level0: &Measure, other: &Measure, counted: &[bool]) -> Self {
        let counts = |k: usize| counted.is_empty() || counted[k];
        let mut s = Shape::default();
        for (k, (a, b)) in level0.tops.iter().zip(&other.tops).enumerate() {
            if !counts(k) {
                s.left_out += 1;
            } else if b.is_nan() {
                s.undrawn += 1;
            } else {
                s.column.push(b - a);
            }
        }
        let skyline = |m: &Measure, rows: bool, k: i32| {
            (0..FOOTPRINT)
                .map(|n| if rows { (n, k) } else { (k, n) })
                .filter(|&(i, j)| counts((j * FOOTPRINT + i) as usize))
                .map(|(i, j)| m.top(i, j))
                .filter(|t| !t.is_nan())
                .fold(f64::NEG_INFINITY, f64::max)
        };
        for rows in [true, false] {
            for k in 0..FOOTPRINT {
                let (a, b) = (skyline(level0, rows, k), skyline(other, rows, k));
                if a.is_finite() && b.is_finite() {
                    s.skyline.push(b - a);
                }
            }
        }
        let peak = |m: &Measure| {
            m.tops
                .iter()
                .enumerate()
                .filter(|(k, t)| counts(*k) && !t.is_nan())
                .fold(f64::NEG_INFINITY, |a, (_, &t)| a.max(t))
        };
        if peak(level0).is_finite() && peak(other).is_finite() {
            s.peak.push(peak(other) - peak(level0));
        }
        s
    }

    pub fn add(&mut self, other: &Shape) {
        self.column.extend_from_slice(&other.column);
        self.skyline.extend_from_slice(&other.skyline);
        self.peak.extend_from_slice(&other.peak);
        self.undrawn += other.undrawn;
        self.left_out += other.left_out;
    }
}

/// The mean, the mean magnitude, the 95th percentile of the magnitude and
/// the largest magnitude of `values`.
pub fn spread(values: &[f64]) -> (f64, f64, f64, f64) {
    if values.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let mut mags: Vec<f64> = values.iter().map(|v| v.abs()).collect();
    mags.sort_by(f64::total_cmp);
    let mean_mag = mags.iter().sum::<f64>() / mags.len() as f64;
    (
        mean,
        mean_mag,
        mags[(mags.len() * 95 / 100).min(mags.len() - 1)],
        mags[mags.len() - 1],
    )
}

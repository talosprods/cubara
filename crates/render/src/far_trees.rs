//! **Trees on the far terrain** (issue #433 item 1): what a far tree is
//! drawn as, how it fades by its size on screen, and which of them a frame
//! draws -- the GPU half. Which trees stand where is `cubara_world`'s
//! (`far_trees::patch_trees`), handed over as plain [`FarTreePlacement`]s
//! (Rule 3); this module never decides where a tree is.
//!
//! The far terrain reaches 262 km and about 18 million trees stand in view
//! beyond the voxel rings (`docs/PHASE3_ARCHITECTURE.md` §9.12), so a far
//! tree is drawn one by one only while it is large enough on screen to be
//! seen as one, and then hands over to a tint of the far terrain's tops:
//!
//! - **The stand-in** ([`StandIn`]): the oak as axis-aligned boxes -- its
//!   canopy's bounds and its trunk below them -- in the flat colours the
//!   level-3 silhouette has for each face direction (the same texture means,
//!   shaded by the oak's own openness, `trees::oak_shades`). Each box draws
//!   only the faces that face the eye, chosen in `far_trees.wgsl`: five faces
//!   a tree.
//! - **The fade by screen size** ([`tree_scale`]): a tree is drawn whole
//!   while it is [`FULL_PX`] pixels tall or more, shrinks about the foot of
//!   its trunk to nothing at [`GONE_PX`], and is not drawn below that. A
//!   distance, never: the pixel sizes are a screen's, so a taller screen
//!   draws trees further out.
//! - **The tint** ([`FarTreeTint`]): what the trees no longer drawn covered,
//!   the far terrain's top shows instead (`far.wgsl`): the share of the
//!   top's ground the missing tree area hides from the eye, in the colour
//!   that area shows. At [`FULL_PX`] it is nothing and below [`GONE_PX`]
//!   everything, so a far quad's colour is the same however much of its
//!   trees is drawn one by one.

use std::collections::BTreeMap;

use cubara_voxel::{BlockRegistry, Face, StructureRegistry};
use glam::Vec3;

use crate::anchor::{self, Anchor};
use crate::culling::{Aabb, Frustum};
use crate::trees::{oak_shades, TreeModelMesh, TreeShapes, TreeVertex};

/// **A far tree is drawn whole while it stands this many pixels tall or
/// more** (its own height, at the screen's height in pixels). **Chosen, not
/// derived**: the voxel rings end at 1,024 blocks, where a tree is 4.6 to
/// 6.4 pixels tall at 1080 rows; whole to 3 pixels (1.6 to 2.2 km) is
/// most of the band the census puts trees in view in
/// (`far_trees_census`).
pub const FULL_PX: f32 = 3.0;

/// **...and is gone at this many pixels**: shrunk to nothing about the foot
/// of its trunk, its ground's colour then wholly the tint's. **Chosen, not
/// derived**: the issue's "about 1 to 2 pixels", at the end that keeps the
/// trees in view at the gate's hill eye to about 1,500 (2.3 to 3.3 km).
pub const GONE_PX: f32 = 2.0;

/// How many pixels tall one block stands at a distance of one block, on a
/// screen `screen_height` pixels tall with the game's vertical field of view
/// ([`crate::render::FOV_Y_DEGREES`]): what a tree's height is multiplied by
/// and its distance divided by to give its size on screen.
pub fn px_per_block(screen_height: u32) -> f32 {
    screen_height as f32 / 2.0 / (crate::render::FOV_Y_DEGREES.to_radians() / 2.0).tan()
}

/// **How large a far tree is drawn, 0 to 1** (`far_trees.wgsl`'s
/// `tree_scale` is this, and the tint's fade is too): a tree `tall` blocks
/// tall at `distance` blocks from the eye stands `tall * px_per_block /
/// distance` pixels tall; it is drawn whole from [`FULL_PX`] up, not at all
/// from [`GONE_PX`] down, and between them at the share of the way it is.
pub fn tree_scale(tall: f32, distance: f32, px_per_block: f32) -> f32 {
    let px = tall * px_per_block / distance.max(1e-3);
    ((px - GONE_PX) / (FULL_PX - GONE_PX)).clamp(0.0, 1.0)
}

/// The farthest a tree `tall` blocks tall is drawn at all: where it is
/// [`GONE_PX`] tall.
pub fn reach(tall: f32, px_per_block: f32) -> f32 {
    tall * px_per_block / GONE_PX
}

/// One tree on the far terrain, as this crate needs it: its trunk's base
/// block, in world blocks -- the first trunk block, standing on the far
/// quad's top (`cubara_world::far_trees::FarTree`, field for field) -- and
/// its trunk's height.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FarTreePlacement {
    pub base: [i32; 3],
    pub height: i32,
}

/// One face of a stand-in box: its flat colour (linear, before light) and
/// how open it is to the sky, `0..=1` -- lit as `trees.wgsl` lights a
/// silhouette face, `mix(ao_floor, 1, open)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StandInFace {
    pub colour: [f32; 3],
    pub open: f32,
}

/// One box of a far tree's stand-in, in blocks from its trunk's base block
/// (the frame [`TreeVertex::position`] is in), and its faces in `Face` order
/// (+X -X +Y -Y +Z -Z): `None` for a face that is never seen (a trunk's top,
/// inside the canopy; its foot, on the ground).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StandInBox {
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub faces: [Option<StandInFace>; 6],
}

/// **What a far tree of one trunk height is drawn as**: a few boxes, each
/// drawing the faces of it that face the eye.
#[derive(Clone, Debug, PartialEq)]
pub struct StandIn {
    pub boxes: Vec<StandInBox>,
    /// The tree's height in blocks, its foot to the top of its canopy: what
    /// its size on screen is measured by ([`tree_scale`]).
    pub tall: f32,
}

/// The six face directions in `Face` order (+X -X +Y -Y +Z -Z): the order
/// every per-face array here is in.
const FACES: [Face; 6] = [
    Face::PosX,
    Face::NegX,
    Face::PosY,
    Face::NegY,
    Face::PosZ,
    Face::NegZ,
];

/// The most boxes a stand-in has: `far_trees.wgsl`'s `MAX_BOXES`.
pub const MAX_BOXES: usize = 3;
/// The most trunk heights a far tree can have: `far_trees.wgsl`'s
/// `MAX_HEIGHTS`. The oak has three (`oak.ron`).
pub const MAX_HEIGHTS: usize = 8;

impl StandIn {
    /// Every face the stand-in can show, as a tree model (local, standing
    /// on its trunk's base block): what `tests/lod_consistency` measures.
    /// The GPU draws, of each box, the faces that face the eye; the faces
    /// of a box that face the eye are those that are not back faces, so a
    /// measurement that culls back faces sees the same pixels.
    pub fn mesh(&self) -> TreeModelMesh {
        let mut out = TreeModelMesh::default();
        for b in &self.boxes {
            for (d, face) in b.faces.iter().enumerate() {
                let Some(face) = face else { continue };
                let [p, q] = face_corners(b.min, b.max, d);
                let (axis, normal) = (d / 2, Vec3::from(FACES[d].normal()));
                let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
                let corner = |a: f32, c: f32| {
                    let mut x = [0.0; 3];
                    x[axis] = p[axis];
                    x[u] = a;
                    x[v] = c;
                    x
                };
                let mut corners = [
                    corner(p[u], p[v]),
                    corner(q[u], p[v]),
                    corner(q[u], q[v]),
                    corner(p[u], q[v]),
                ];
                let c = corners.map(Vec3::from);
                if (c[1] - c[0]).cross(c[2] - c[0]).dot(normal) < 0.0 {
                    corners.reverse();
                }
                let base = out.vertices.len() as u32;
                for position in corners {
                    out.vertices.push(TreeVertex {
                        position,
                        normal: normal.to_array(),
                        uv: [0.0, 0.0],
                        layer: -1.0,
                        color: face.colour,
                        open: face.open,
                    });
                }
                out.indices
                    .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            }
        }
        out
    }

    /// Per face direction (`Face` order): the area of the stand-in's faces
    /// that way, the sum of area times colour, and of area times colour
    /// times openness -- what the tint lights the missing trees with
    /// ([`FarTreeTint`]): `mix(ao_floor, 1, open)` is linear in `open`, so
    /// the two sums light them exactly as the faces are lit.
    pub fn by_direction(&self) -> [(f32, [f32; 3], [f32; 3]); 6] {
        let mut out = [(0.0, [0.0; 3], [0.0; 3]); 6];
        for b in &self.boxes {
            for (d, face) in b.faces.iter().enumerate() {
                let Some(face) = face else { continue };
                let axis = d / 2;
                let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
                let area = (b.max[u] - b.min[u]) * (b.max[v] - b.min[v]);
                out[d].0 += area;
                for c in 0..3 {
                    out[d].1[c] += area * face.colour[c];
                    out[d].2[c] += area * face.colour[c] * face.open;
                }
            }
        }
        out
    }
}

/// A face's two opposite corners: `min` and `max` with the face's own axis
/// set to its plane.
fn face_corners(min: [f32; 3], max: [f32; 3], d: usize) -> [[f32; 3]; 2] {
    let axis = d / 2;
    let plane = if d % 2 == 0 { max[axis] } else { min[axis] };
    let (mut p, mut q) = (min, max);
    p[axis] = plane;
    q[axis] = plane;
    [p, q]
}

/// **The stand-in of the oak `height` blocks tall**, from `oak` -- its
/// model (`trees::build_oak_mesh`) -- and the tree shape it is meshed from
/// (`cubara_voxel::tree_part_at`): the canopy's bounds as one box, the
/// trunk below the canopy as another. Each face's colour is its texture's
/// mean (`colour_of`) and its openness the oak's own there
/// ([`oak_shades`]): what the level-3 silhouette shades that direction and
/// texture by, so a stand-in face is a silhouette face's colour.
///
/// One box for the canopy, not the oak's rounded outline: **chosen, not
/// derived** -- five faces a tree where the silhouette draws 55. Its outline
/// is the oak's exactly from along an axis and up to 8.4 square blocks
/// larger diagonally (its corners, which the oak trims), at the voxel
/// rings' edge 7 square pixels on a tree of 20 (`lod_consistency`'s far
/// tree check measures and holds it). Two crossed boxes would halve that
/// for a third more triangles (§9.12).
pub fn build_stand_in(
    registry: &BlockRegistry,
    structures: &StructureRegistry,
    layer_of: &dyn Fn(&str) -> u32,
    colour_of: &dyn Fn(u32) -> [f32; 3],
    height: i32,
    oak: &TreeModelMesh,
) -> Option<StandIn> {
    let def = structures.get("cubara:oak")?;
    let log = registry.id_of(&def.trunk.block)?;
    let leaves = registry.id_of(&def.canopy.block)?;
    let r = def.canopy.radius;
    let (min, max) = cubara_voxel::tree_bounds(height, r);
    // The canopy's lowest row: the first row any cell of which is canopy.
    let canopy_from = (min[1]..=max[1]).find(|&dy| {
        (0..=r).any(|dx| {
            (0..=r).any(|dz| {
                cubara_voxel::tree_part_at(dx, dz, dy, height, r)
                    == Some(cubara_voxel::TreePart::Canopy)
            })
        })
    })?;
    let shades = oak_shades(oak);
    let face_of = |block, d: usize| {
        let layer = layer_of(registry.texture_for_face(block, FACES[d])?);
        Some(StandInFace {
            colour: colour_of(layer),
            open: shades.get(&(d, layer)).copied().unwrap_or(1.0),
        })
    };
    let canopy = StandInBox {
        min: [min[0] as f32, canopy_from as f32, min[2] as f32],
        max: [
            max[0] as f32 + 1.0,
            max[1] as f32 + 1.0,
            max[2] as f32 + 1.0,
        ],
        faces: std::array::from_fn(|d| face_of(leaves, d)),
    };
    // The trunk's sides below the canopy: its top is inside the canopy,
    // its foot on the ground.
    let trunk = StandInBox {
        min: [0.0, 0.0, 0.0],
        max: [1.0, canopy_from as f32, 1.0],
        faces: std::array::from_fn(|d| if d / 2 == 1 { None } else { face_of(log, d) }),
    };
    Some(StandIn {
        boxes: vec![canopy, trunk],
        tall: max[1] as f32 + 1.0,
    })
}

/// Every stand-in, one per trunk height in [`TreeShapes::heights`] order,
/// from the oak models ([`crate::tree_model_meshes`]). Empty without the
/// oak's data.
pub fn stand_ins(
    registry: &BlockRegistry,
    structures: &StructureRegistry,
    layer_of: &dyn Fn(&str) -> u32,
) -> Vec<StandIn> {
    let names = registry.texture_names();
    let colour_of = |layer: u32| {
        names
            .iter()
            .find(|name| layer_of(name) == layer)
            .and_then(|name| crate::materials::mean_color(name))
            .unwrap_or([0.5, 0.5, 0.5])
    };
    let (oaks, _) = crate::trees::tree_model_meshes(registry, structures, layer_of);
    let shapes = TreeShapes::of(structures);
    shapes
        .heights
        .clone()
        .zip(&oaks)
        .filter_map(|(h, oak)| build_stand_in(registry, structures, layer_of, &colour_of, h, oak))
        .collect()
}

/// **What the far terrain's tops show of the trees no longer drawn one by
/// one** (`far.wgsl`'s `tree_tint`): per trunk height, how densely such
/// trees stand where every column may grow one, their stand-in's height and
/// its faces per direction ([`StandIn::by_direction`]). The tops' own
/// share of ground a tree may grow on (`FarFace::trees`) scales the density.
#[derive(Clone, Debug, PartialEq)]
pub struct FarTreeTint {
    /// Per trunk height: trees per square block where every column may grow
    /// one, the stand-in, in [`TreeShapes::heights`] order.
    pub heights: Vec<(f32, StandIn)>,
}

impl FarTreeTint {
    /// `densities` (per trunk height, in order: trees per square block,
    /// from the world's placement -- `cubara_world::far_trees::density`)
    /// with the stand-ins of the same heights.
    pub fn new(densities: &[f32], stand_ins: &[StandIn]) -> Self {
        Self {
            heights: densities
                .iter()
                .copied()
                .zip(stand_ins.iter().cloned())
                .take(MAX_HEIGHTS)
                .collect(),
        }
    }
}

/// The tint as `far.wgsl` reads it: per height, `[density, tall, 0, 0]`,
/// then per direction the area-weighted colour (`.w` the area) and the
/// colour times openness, for the five directions a top can see a tree from
/// (`Face` order without -Y: +X -X +Y +Z -Z).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TintGpu {
    /// `.x` the heights in use; `.y` [`FULL_PX`], `.z` [`GONE_PX`], `.w`
    /// pixels a block is tall at a distance of one block.
    pub fade: [f32; 4],
    pub height: [[f32; 4]; MAX_HEIGHTS],
    pub colour: [[[f32; 4]; 5]; MAX_HEIGHTS],
    pub open: [[[f32; 4]; 5]; MAX_HEIGHTS],
}

/// The five directions of [`TintGpu`]: `Face` order without -Y.
const TINT_DIRECTIONS: [usize; 5] = [0, 1, 2, 4, 5];

impl TintGpu {
    /// No tint: no trees, or a far terrain drawn for a world without them.
    pub const NONE: TintGpu = TintGpu {
        fade: [0.0; 4],
        height: [[0.0; 4]; MAX_HEIGHTS],
        colour: [[[0.0; 4]; 5]; MAX_HEIGHTS],
        open: [[[0.0; 4]; 5]; MAX_HEIGHTS],
    };

    pub fn of(tint: Option<&FarTreeTint>, px_per_block: f32) -> Self {
        let Some(tint) = tint else {
            return Self::NONE;
        };
        let mut out = Self::NONE;
        out.fade = [tint.heights.len() as f32, FULL_PX, GONE_PX, px_per_block];
        for (k, (density, stand_in)) in tint.heights.iter().enumerate() {
            out.height[k] = [*density, stand_in.tall, 0.0, 0.0];
            let dirs = stand_in.by_direction();
            for (slot, &d) in TINT_DIRECTIONS.iter().enumerate() {
                let (area, colour, open) = dirs[d];
                out.colour[k][slot] = [colour[0], colour[1], colour[2], area];
                out.open[k][slot] = [open[0], open[1], open[2], 0.0];
            }
        }
        out
    }
}

/// One far tree drawn this frame: where its trunk's foot stands, relative
/// to the render anchor -- the middle of its base block's floor, `(x + 0.5,
/// base, z + 0.5)`, exact in `f32` -- and which trunk height's stand-in it
/// is drawn with.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FarTreeInstance {
    pub foot: [f32; 3],
    pub height: u32,
}

/// The side of the squares, in blocks, a patch's trees are culled by
/// frame by frame -- against the frustum and against how far the tallest
/// tree is drawn ([`reach`]). **Chosen, not derived**: a patch of quads of
/// 128 is 4,096 blocks and holds about 5,000 trees, most of them out of
/// reach; squares of 128 blocks hold about five.
pub const TILE_BLOCKS: i32 = 128;

/// One square of a patch's trees: its box in world blocks, and where its
/// trees are in the patch's list.
#[derive(Clone, Debug)]
struct Tile {
    min: [i64; 3],
    max: [i64; 3],
    trees: std::ops::Range<usize>,
}

/// One patch's trees, by square ([`TILE_BLOCKS`]), and the box round all of
/// them.
#[derive(Clone, Debug)]
struct PatchTrees {
    trees: Vec<FarTreePlacement>,
    tiles: Vec<Tile>,
    min: [i64; 3],
    max: [i64; 3],
}

/// `(min, max)`, in world blocks, of `trees` standing as `tall` says, their
/// canopies `r` blocks round their trunks.
fn bounds_of(
    trees: &[FarTreePlacement],
    tall: &dyn Fn(i32) -> f32,
    r: i64,
) -> ([i64; 3], [i64; 3]) {
    let mut min = [i64::MAX; 3];
    let mut max = [i64::MIN; 3];
    for t in trees {
        let [x, y, z] = t.base.map(i64::from);
        let top = y + tall(t.height).ceil() as i64;
        min = [min[0].min(x - r), min[1].min(y), min[2].min(z - r)];
        max = [max[0].max(x + r + 1), max[1].max(top), max[2].max(z + r + 1)];
    }
    (min, max)
}

fn relative_box(min: [i64; 3], max: [i64; 3], anchor: Anchor) -> Aabb {
    Aabb::new(anchor::offset_i64(min, anchor), anchor::offset_i64(max, anchor))
}

/// How far `eye` is from the nearest point of `b`.
fn distance_to(eye: Vec3, b: &Aabb) -> f32 {
    (b.min - eye).max(eye - b.max).max(Vec3::ZERO).length()
}

/// **The far trees, by patch, and which of them a frame draws**: the CPU
/// half of [`FarTrees`], with no GPU, so a test can drive it.
#[derive(Debug, Default)]
pub struct FarTreeSet {
    by_patch: BTreeMap<u64, PatchTrees>,
    /// Per trunk height (in [`TreeShapes::heights`] order): the stand-in's
    /// height in blocks.
    talls: Vec<f32>,
    first_height: i32,
    canopy_radius: i64,
    /// The voxels' box, in world blocks: a tree whose ground is in it is
    /// the voxels' to draw (`FarParams::hole`).
    hole: Option<([i32; 3], [i32; 3])>,
}

impl FarTreeSet {
    /// For trees of `shapes`' heights, drawn as `stand_ins`.
    pub fn new(shapes: &TreeShapes, stand_ins: &[StandIn]) -> Self {
        Self {
            by_patch: BTreeMap::new(),
            talls: stand_ins.iter().map(|s| s.tall).collect(),
            first_height: *shapes.heights.start(),
            canopy_radius: shapes.canopy_radius as i64,
            hole: None,
        }
    }

    fn height_index(&self, height: i32) -> usize {
        let k = (height - self.first_height).max(0) as usize;
        debug_assert!(k < self.talls.len(), "a far tree of trunk height {height}");
        k.min(self.talls.len().saturating_sub(1))
    }

    fn tall(&self, height: i32) -> f32 {
        self.talls
            .get(self.height_index(height))
            .copied()
            .unwrap_or(0.0)
    }

    /// Record (or replace) the trees of the patch `id` -- the caller's key
    /// for it, the far terrain's slot in the game. An empty list removes
    /// it.
    pub fn insert(&mut self, id: u64, mut trees: Vec<FarTreePlacement>) {
        if trees.is_empty() || self.talls.is_empty() {
            self.by_patch.remove(&id);
            return;
        }
        let tile = |t: &FarTreePlacement| {
            (
                t.base[2].div_euclid(TILE_BLOCKS),
                t.base[0].div_euclid(TILE_BLOCKS),
            )
        };
        trees.sort_by_key(|t| (tile(t), t.base[2], t.base[0]));
        let tall = |h: i32| self.tall(h);
        let r = self.canopy_radius;
        let mut tiles = Vec::new();
        let mut start = 0;
        while start < trees.len() {
            let key = tile(&trees[start]);
            let end = start + trees[start..].iter().take_while(|t| tile(t) == key).count();
            let (min, max) = bounds_of(&trees[start..end], &tall, r);
            tiles.push(Tile {
                min,
                max,
                trees: start..end,
            });
            start = end;
        }
        let (min, max) = bounds_of(&trees, &tall, r);
        self.by_patch.insert(
            id,
            PatchTrees {
                trees,
                tiles,
                min,
                max,
            },
        );
    }

    pub fn remove(&mut self, id: u64) {
        self.by_patch.remove(&id);
    }

    pub fn clear(&mut self) {
        self.by_patch.clear();
    }

    /// The voxels' box, in world blocks (`FarParams::hole`), or `None`.
    pub fn set_hole(&mut self, hole: Option<([i32; 3], [i32; 3])>) {
        self.hole = hole;
    }

    /// Trees held.
    pub fn len(&self) -> usize {
        self.by_patch.values().map(|p| p.trees.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.by_patch.is_empty()
    }

    /// Whether the voxels draw the tree at `t`: the block its trunk stands
    /// on is inside their box -- the far terrain's own test for a top
    /// (`far.wgsl`: strictly inside), at the middle of that block.
    fn in_hole(&self, t: &FarTreePlacement) -> bool {
        self.hole.is_some_and(|(min, max)| {
            let p = [
                t.base[0] as f64 + 0.5,
                t.base[1] as f64 - 0.5,
                t.base[2] as f64 + 0.5,
            ];
            (0..3).all(|a| (min[a] as f64) < p[a] && p[a] < max[a] as f64)
        })
    }

    /// **The trees a frame draws**, as instances relative to `anchor`, in
    /// patch, square and tree order: every tree the frustum can see whose
    /// size on screen is over [`GONE_PX`] -- by its own height, from `eye`
    /// (anchor-relative), at `px_per_block` -- and whose ground is not the
    /// voxels'. Squares and patches out of the frustum or out of reach are
    /// passed over whole.
    pub fn visible(
        &self,
        frustum: &Frustum,
        eye: Vec3,
        anchor: Anchor,
        px_per_block: f32,
        out: &mut Vec<FarTreeInstance>,
    ) {
        out.clear();
        let farthest = self
            .talls
            .iter()
            .map(|&t| reach(t, px_per_block))
            .fold(0.0, f32::max);
        for patch in self.by_patch.values() {
            let b = relative_box(patch.min, patch.max, anchor);
            if distance_to(eye, &b) > farthest || !frustum.intersects_aabb(&b) {
                continue;
            }
            for tile in &patch.tiles {
                let b = relative_box(tile.min, tile.max, anchor);
                if distance_to(eye, &b) > farthest || !frustum.intersects_aabb(&b) {
                    continue;
                }
                for t in &patch.trees[tile.trees.clone()] {
                    let k = self.height_index(t.height);
                    let tall = self.talls[k];
                    let foot = anchor::offset(t.base, anchor) + Vec3::new(0.5, 0.0, 0.5);
                    let middle = foot + Vec3::Y * (tall / 2.0);
                    if tree_scale(tall, middle.distance(eye), px_per_block) <= 0.0
                        || self.in_hole(t)
                    {
                        continue;
                    }
                    out.push(FarTreeInstance {
                        foot: foot.to_array(),
                        height: k as u32,
                    });
                }
            }
        }
    }
}

/// One stand-in box as `far_trees.wgsl` reads it: its corners (`.w`
/// unused) and per face (`Face` order) its colour and openness, `.w` below
/// zero for a face it does not draw.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct BoxGpu {
    min: [f32; 4],
    max: [f32; 4],
    faces: [[f32; 4]; 6],
}

/// `far_trees.wgsl`'s `Params`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsGpu {
    /// `.x` boxes a stand-in has, `.y` [`FULL_PX`], `.z` [`GONE_PX`], `.w`
    /// pixels a block is tall at one block.
    fade: [f32; 4],
    /// Per trunk height, the stand-in's height in blocks.
    tall: [[f32; 4]; MAX_HEIGHTS],
    boxes: [[BoxGpu; MAX_BOXES]; MAX_HEIGHTS],
}

impl ParamsGpu {
    fn new(stand_ins: &[StandIn], px_per_block: f32) -> Self {
        let none = BoxGpu {
            min: [0.0; 4],
            max: [0.0; 4],
            faces: [[0.0, 0.0, 0.0, -1.0]; 6],
        };
        let mut out = ParamsGpu {
            fade: [0.0, FULL_PX, GONE_PX, px_per_block],
            tall: [[0.0; 4]; MAX_HEIGHTS],
            boxes: [[none; MAX_BOXES]; MAX_HEIGHTS],
        };
        let boxes = stand_ins.iter().map(|s| s.boxes.len()).max().unwrap_or(0);
        out.fade[0] = boxes.min(MAX_BOXES) as f32;
        for (k, s) in stand_ins.iter().take(MAX_HEIGHTS).enumerate() {
            out.tall[k][0] = s.tall;
            for (slot, b) in s.boxes.iter().take(MAX_BOXES).enumerate() {
                out.boxes[k][slot] = BoxGpu {
                    min: [b.min[0], b.min[1], b.min[2], 0.0],
                    max: [b.max[0], b.max[1], b.max[2], 0.0],
                    faces: b.faces.map(|f| {
                        f.map_or([0.0, 0.0, 0.0, -1.0], |f| {
                            [f.colour[0], f.colour[1], f.colour[2], f.open]
                        })
                    }),
                };
            }
        }
        out
    }
}

/// The indices of one tree: per box, per axis, the face of that axis that
/// faces the eye, four corners `far_trees.wgsl` makes from the vertex
/// index, drawn as two triangles.
fn tree_indices() -> Vec<u16> {
    (0..(MAX_BOXES * 3) as u16)
        .flat_map(|face| [0, 2, 1, 1, 2, 3].map(|i| face * 4 + i))
        .collect()
}

/// **The far trees on the GPU**: [`FarTreeSet`], the stand-ins' parameters,
/// and the instances a frame draws -- uploaded by [`prepare`](Self::prepare)
/// whenever they differ from the last frame's, and drawn in one instanced
/// draw.
pub struct FarTrees {
    set: FarTreeSet,
    stand_ins: Vec<StandIn>,
    params: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    indices: wgpu::Buffer,
    /// Indices a tree is drawn with: six a face, three faces a box.
    index_count: u32,
    instances: Option<wgpu::Buffer>,
    capacity: usize,
    /// This frame's instances, and how many of them were uploaded.
    drawn: Vec<FarTreeInstance>,
    uploaded: Vec<FarTreeInstance>,
    /// What the instances were made relative to, and for what screen.
    anchor: Anchor,
    px_per_block: f32,
}

impl FarTrees {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shapes: &TreeShapes,
        stand_ins: Vec<StandIn>,
        anchor: Anchor,
    ) -> Self {
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("far-trees-params"),
            size: std::mem::size_of::<ParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &params,
            0,
            bytemuck::bytes_of(&ParamsGpu::new(&stand_ins, 0.0)),
        );
        let layout = crate::render::far_trees_bind_group_layout(device);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("far-trees-bind-group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params.as_entire_binding(),
            }],
        });
        let list = tree_indices();
        let indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("far-trees-indices"),
            size: std::mem::size_of_val(list.as_slice()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&indices, 0, bytemuck::cast_slice(&list));
        let boxes = stand_ins.iter().map(|s| s.boxes.len()).max().unwrap_or(0);
        Self {
            set: FarTreeSet::new(shapes, &stand_ins),
            stand_ins,
            params,
            bind_group,
            indices,
            index_count: (boxes.min(MAX_BOXES) * 18) as u32,
            instances: None,
            capacity: 0,
            drawn: Vec::new(),
            uploaded: Vec::new(),
            anchor,
            px_per_block: 0.0,
        }
    }

    /// The trees a far patch carries (the caller's key, `id`), replacing any
    /// it carried before.
    pub fn insert(&mut self, id: u64, trees: Vec<FarTreePlacement>) {
        self.set.insert(id, trees);
    }

    pub fn remove(&mut self, id: u64) {
        self.set.remove(id);
    }

    pub fn clear(&mut self) {
        self.set.clear();
    }

    pub fn set_hole(&mut self, hole: Option<([i32; 3], [i32; 3])>) {
        self.set.set_hole(hole);
    }

    /// Trees held, drawn or not.
    pub fn len(&self) -> usize {
        self.set.len()
    }

    pub fn is_empty(&self) -> bool {
        self.set.is_empty()
    }

    /// The stand-ins the trees are drawn as, one per trunk height.
    pub fn stand_ins(&self) -> &[StandIn] {
        &self.stand_ins
    }

    /// Trees drawn this frame, from the last [`prepare`](Self::prepare).
    pub fn drawn(&self) -> usize {
        self.drawn.len()
    }

    /// What this frame's far trees cost: `(draws, triangles, vertices)` --
    /// one instanced draw, and per tree two triangles and four vertices a
    /// face drawn, the faces a box does not draw included (they are
    /// degenerate on the GPU, and its vertices are shaded all the same).
    pub fn drawn_cost(&self) -> (u32, u64, u64) {
        let trees = self.drawn.len() as u64;
        let faces = self.index_count as u64 / 6;
        (u32::from(trees > 0), trees * faces * 2, trees * faces * 4)
    }

    /// The anchor the instances drawn were made relative to.
    pub fn anchor(&self) -> Anchor {
        self.anchor
    }

    /// Work out which trees this frame draws ([`FarTreeSet::visible`]) and
    /// upload them if they differ from the last frame's: `eye` relative to
    /// `anchor`, which the instances are made relative to (so a rebase
    /// needs nothing but the next frame), on a screen `screen_height` pixels
    /// tall.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frustum: &Frustum,
        eye: Vec3,
        anchor: Anchor,
        screen_height: u32,
    ) {
        puffin::profile_function!();
        let px = px_per_block(screen_height);
        if px != self.px_per_block {
            self.px_per_block = px;
            queue.write_buffer(
                &self.params,
                0,
                bytemuck::bytes_of(&ParamsGpu::new(&self.stand_ins, px)),
            );
        }
        self.anchor = anchor;
        let mut drawn = std::mem::take(&mut self.drawn);
        self.set.visible(frustum, eye, anchor, px, &mut drawn);
        self.drawn = drawn;
        if self.drawn == self.uploaded || self.drawn.is_empty() {
            return;
        }
        if self.instances.is_none() || self.drawn.len() > self.capacity {
            self.capacity = self.drawn.len().next_power_of_two();
            self.instances = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("far-tree-instances"),
                size: (self.capacity * std::mem::size_of::<FarTreeInstance>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        queue.write_buffer(
            self.instances.as_ref().expect("just ensured"),
            0,
            bytemuck::cast_slice(&self.drawn),
        );
        self.uploaded.clone_from(&self.drawn);
    }

    /// Draw this frame's trees. The pipeline and bind group 0 (the camera)
    /// are the caller's to set.
    pub(crate) fn encode(&self, pass: &mut wgpu::RenderPass<'_>) {
        let Some(instances) = self.instances.as_ref() else {
            return;
        };
        if self.drawn.is_empty() || self.index_count == 0 {
            return;
        }
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, instances.slice(..));
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..self.index_count, 0, 0..self.drawn.len() as u32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tree_fades_between_full_and_gone_by_its_size_on_screen() {
        let k = px_per_block(1080);
        // 935 pixels a block at one block, at 1080 rows and 60 degrees.
        assert!((k - 935.3).abs() < 0.1, "{k}");
        let tall = 6.0;
        let at = |px: f32| tall * k / px;
        assert_eq!(tree_scale(tall, at(5.0), k), 1.0);
        assert_eq!(tree_scale(tall, at(FULL_PX), k), 1.0);
        assert!((tree_scale(tall, at(2.5), k) - 0.5).abs() < 1e-4);
        assert_eq!(tree_scale(tall, at(GONE_PX), k), 0.0);
        assert_eq!(tree_scale(tall, at(1.0), k), 0.0);
        // A taller screen draws a tree whole further out.
        let k2 = px_per_block(2160);
        assert_eq!(tree_scale(tall, at(FULL_PX) * 1.9, k2), 1.0);
        // How far: where it is gone.
        assert!((reach(tall, k) - at(GONE_PX)).abs() < 1e-2);
    }
}

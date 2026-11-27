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
use crate::far::FarRole;
use crate::trees::{oak_shades, TreeModelMesh, TreeShapes, TreeVertex};

/// **A far tree is drawn whole while it stands this many pixels tall or
/// more** (its own height, at the screen's height in pixels). **Chosen, not
/// derived**: the voxel rings end at 1,024 blocks, where a tree is 4.6 to
/// 6.4 pixels tall at 1080 rows; whole to 3.5 pixels is 1.3 to 1.9 km, and
/// a pixel of fade above [`GONE_PX`] lets a tree shrink over some hundred
/// blocks of flight rather than a few.
pub const FULL_PX: f32 = 3.5;

/// **...and is gone at this many pixels**: shrunk to nothing about the foot
/// of its trunk, its ground's colour then wholly the tint's (which keeps a
/// wood's colour, `golden.rs`'s `a_fading_wood_keeps_its_colour`).
/// **Chosen, not derived**: the issue asks for "about 1 to 2 pixels"; at 2
/// the gate's hill eye draws 1,587 far trees a frame (25,386 triangles with
/// the crossed canopy), at 2.5 892 (14,264) -- the count the 5 to 7% the
/// hill eye has left over 1,000 FPS can take (`far_trees_census`'s
/// `far_trees_drawn`, §9.12). Gone at 1.9 to 2.6 km.
pub const GONE_PX: f32 = 2.5;

/// How many pixels tall one block stands at a distance of one block, on a
/// screen `screen_height` pixels tall with the game's vertical field of view
/// ([`crate::render::FOV_Y_DEGREES`]): what a tree's height is multiplied by
/// and its distance divided by to give its size on screen.
pub fn px_per_block(screen_height: u32) -> f32 {
    screen_height as f32 / 2.0 / (crate::render::FOV_Y_DEGREES.to_radians() / 2.0).tan()
}

/// The pixel sizes a far tree fades between: whole from `full_px` up, gone
/// from `gone_px` down. The game's is [`Fade::GAME`]; a measurement may ask
/// for another to see what the fade changes (`golden.rs`: every tree whole,
/// or none and all of it tint), as [`crate::far::FarCulling`] is asked.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fade {
    pub full_px: f32,
    pub gone_px: f32,
}

impl Fade {
    /// [`FULL_PX`] and [`GONE_PX`].
    pub const GAME: Fade = Fade {
        full_px: FULL_PX,
        gone_px: GONE_PX,
    };
    /// Every tree whole wherever it stands: no tint.
    pub const WHOLE: Fade = Fade {
        full_px: 0.0,
        gone_px: -1.0,
    };
    /// No tree drawn one by one: all of them tint.
    pub const GONE: Fade = Fade {
        full_px: f32::MAX,
        gone_px: f32::MAX / 2.0,
    };
}

/// **How large a far tree is drawn, 0 to 1** (`far_trees.wgsl`'s
/// `tree_scale` is this, and the tint's fade is too): a tree `tall` blocks
/// tall at `distance` blocks from the eye stands `tall * px_per_block /
/// distance` pixels tall; it is drawn whole from `fade.full_px` up, not at
/// all from `fade.gone_px` down, and between them at the share of the way
/// it is.
pub fn tree_scale(tall: f32, distance: f32, px_per_block: f32, fade: Fade) -> f32 {
    let px = tall * px_per_block / distance.max(1e-3);
    ((px - fade.gone_px) / (fade.full_px - fade.gone_px)).clamp(0.0, 1.0)
}

/// The farthest a tree `tall` blocks tall is drawn at all: where it is
/// `fade.gone_px` tall -- everywhere, for a fade that never takes a tree.
pub fn reach(tall: f32, px_per_block: f32, fade: Fade) -> f32 {
    if fade.gone_px <= 0.0 {
        return f32::INFINITY;
    }
    tall * px_per_block / fade.gone_px
}

/// Which way the tint measures the ground the missing trees hide: the
/// game's, along the line of sight; or as seen from straight above, the
/// share their canopies cover -- what #433 first asked for, kept so that a
/// test can show what it does to a fading wood (`golden.rs`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TintView {
    #[default]
    LineOfSight,
    FromAbove,
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
/// The most face slots a stand-in draws -- a box and an axis along which it
/// has a face: `far_trees.wgsl`'s `MAX_SLOTS`.
pub const MAX_SLOTS: usize = 3 * MAX_BOXES;
/// The most trunk heights a far tree can have: `far_trees.wgsl`'s
/// `MAX_HEIGHTS`. The oak has three (`oak.ron`).
pub const MAX_HEIGHTS: usize = 8;

impl StandIn {
    /// The radius of a sphere about the tree's middle -- `tall / 2` above
    /// its foot -- that holds every box of it at every size it fades
    /// through: the farthest corner at full size, or the foot, whichever is
    /// further (a box shrinks about the foot, so each corner moves along the
    /// line to it). What `far_trees.wgsl` drops a tree off screen by.
    pub fn radius(&self) -> f32 {
        let middle = Vec3::new(0.5, self.tall / 2.0, 0.5);
        self.boxes
            .iter()
            .flat_map(|b| {
                (0..8).map(move |i| {
                    Vec3::new(
                        if i & 1 == 0 { b.min[0] } else { b.max[0] },
                        if i & 2 == 0 { b.min[1] } else { b.max[1] },
                        if i & 4 == 0 { b.min[2] } else { b.max[2] },
                    )
                })
            })
            .map(|corner| corner.distance(middle))
            .fold(self.tall / 2.0, f32::max)
    }

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
                out.indices.extend_from_slice(&[
                    base,
                    base + 1,
                    base + 2,
                    base,
                    base + 2,
                    base + 3,
                ]);
            }
        }
        out
    }

    /// **The faces a tree draws**, as `(box, axis)`: every axis along which
    /// a box has a face. Of each, the face facing the eye is drawn, so this
    /// is what a tree costs: two triangles a slot.
    pub fn slots(&self) -> Vec<(usize, usize)> {
        self.boxes
            .iter()
            .enumerate()
            .flat_map(|(b, bx)| {
                (0..3)
                    .filter(move |&axis| {
                        bx.faces[2 * axis].is_some() || bx.faces[2 * axis + 1].is_some()
                    })
                    .map(move |axis| (b, axis))
            })
            .collect()
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
    let plane = if d.is_multiple_of(2) {
        max[axis]
    } else {
        min[axis]
    };
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
/// **The canopy as two crossed boxes, not the oak's own outline nor one
/// box: chosen, not derived** -- eight faces a tree where the silhouette
/// draws 55. Their outline is the oak's exactly from along an axis and from
/// above, and up to 3.5 square blocks larger at the diagonals of its upper
/// canopy, which the oak narrows (`lod_consistency`'s far tree check holds
/// it): at the voxel rings' edge 12.6% of the trees' pixels change where one
/// box -- its corners, which the oak trims -- changed 33%, for 16 triangles a
/// tree where one box is 10 (§9.12).
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
    // The canopy as two crossed arms, each its bounds a block narrower on
    // one side of the trunk and the other: the oak's lower canopy (the
    // bounds without their four corner columns) exactly, and its upper one
    // a block wider at the diagonals.
    let arm = |a: usize| {
        let mut b = canopy;
        b.min[a] += 1.0;
        b.max[a] -= 1.0;
        b
    };
    Some(StandIn {
        boxes: vec![arm(2), arm(0), trunk],
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
    /// The fade the trees are drawn with: the tint is what it leaves out.
    pub fade: Fade,
    pub view: TintView,
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
            fade: Fade::GAME,
            view: TintView::LineOfSight,
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
    /// `.x` the heights in use; `.y` and `.z` the fade's pixel sizes
    /// ([`Fade`]), `.w` pixels a block is tall at a distance of one block.
    pub fade: [f32; 4],
    /// `.x` 1 to measure from straight above ([`TintView::FromAbove`]);
    /// `.y` the distance within which every trunk height is drawn whole.
    pub view: [f32; 4],
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
        view: [0.0; 4],
        height: [[0.0; 4]; MAX_HEIGHTS],
        colour: [[[0.0; 4]; 5]; MAX_HEIGHTS],
        open: [[[0.0; 4]; 5]; MAX_HEIGHTS],
    };

    pub fn of(tint: Option<&FarTreeTint>, px_per_block: f32) -> Self {
        let Some(tint) = tint else {
            return Self::NONE;
        };
        let mut out = Self::NONE;
        out.fade = [
            tint.heights.len() as f32,
            tint.fade.full_px,
            tint.fade.gone_px,
            px_per_block,
        ];
        out.view[0] = f32::from(u8::from(tint.view == TintView::FromAbove));
        // Nearer than this every trunk height is drawn whole and there is
        // nothing to tint: where the shortest stand-in is `full_px` tall
        // (everywhere, for a fade that never takes a tree).
        out.view[1] = tint
            .heights
            .iter()
            .map(|(_, stand_in)| stand_in.tall * px_per_block / tint.fade.full_px)
            .fold(f32::INFINITY, f32::min);
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
/// is drawn with, [`STANDIN_TREE`] set for a tree of the strip's stand-in.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FarTreeInstance {
    pub foot: [f32; 3],
    pub height: u32,
}

/// [`FarTreeInstance::height`]'s flag for a tree of a stand-in for voxel
/// nodes not drawn yet ([`FarRole::StandIn`]): drawn a hair further along
/// its line of sight than it stands, as the stand-in's ground is
/// ([`crate::far::STANDIN_BEHIND`]), so a level-3 node drawn meanwhile --
/// the same tree, as its silhouette -- is what the depth test keeps.
/// `far_trees.wgsl`'s `STANDIN_TREE`.
pub const STANDIN_TREE: u32 = 1 << 31;

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

/// One patch's trees, by square ([`TILE_BLOCKS`]), the box round all of
/// them, and what they are drawn as.
#[derive(Clone, Debug)]
struct PatchTrees {
    trees: Vec<FarTreePlacement>,
    tiles: Vec<Tile>,
    min: [i64; 3],
    max: [i64; 3],
    role: FarRole,
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
        max = [
            max[0].max(x + r + 1),
            max[1].max(top),
            max[2].max(z + r + 1),
        ];
    }
    (min, max)
}

fn relative_box(min: [i64; 3], max: [i64; 3], anchor: Anchor) -> Aabb {
    Aabb::new(
        anchor::offset_i64(min, anchor),
        anchor::offset_i64(max, anchor),
    )
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
    pub fn insert(&mut self, id: u64, trees: Vec<FarTreePlacement>) {
        self.insert_as(id, trees, FarRole::Terrain);
    }

    /// [`insert`](Self::insert), the trees drawn as `role`: those of a
    /// stand-in for voxel nodes not drawn yet ([`FarRole::StandIn`]) are
    /// the strip's own -- the caller has already kept those standing on
    /// its ground (`cubara-app`'s `standin_trees`) -- and are drawn inside
    /// the voxels' box, a hair behind where they stand ([`STANDIN_TREE`]).
    pub fn insert_as(&mut self, id: u64, mut trees: Vec<FarTreePlacement>, role: FarRole) {
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
                role,
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

    /// The box round a held patch's trees, in world blocks, if it is held.
    pub fn patch_box(&self, id: u64) -> Option<([i64; 3], [i64; 3])> {
        self.by_patch.get(&id).map(|p| (p.min, p.max))
    }

    /// The farthest any tree is drawn at all ([`reach`] of the tallest).
    pub fn farthest(&self, px_per_block: f32, fade: Fade) -> f32 {
        self.talls
            .iter()
            .map(|&t| reach(t, px_per_block, fade))
            .fold(0.0, f32::max)
    }

    /// Every held tree that `keep` keeps -- given its trunk height's index,
    /// its foot and middle relative to `anchor`, and whether it is a
    /// stand-in's -- of the patches and squares within `farthest` of `eye`
    /// (anchor-relative), as instances, in patch, square and tree order.
    /// A tree whose ground is the voxels' is never kept, but for a
    /// stand-in's, which stands in for theirs.
    fn instances(
        &self,
        (eye, anchor): (Vec3, Anchor),
        farthest: f32,
        keep: impl Fn(usize, Vec3, Vec3) -> bool,
        out: &mut Vec<FarTreeInstance>,
    ) {
        out.clear();
        for patch in self.by_patch.values() {
            let b = relative_box(patch.min, patch.max, anchor);
            if distance_to(eye, &b) > farthest {
                continue;
            }
            let standin = patch.role == FarRole::StandIn;
            for tile in &patch.tiles {
                let b = relative_box(tile.min, tile.max, anchor);
                if distance_to(eye, &b) > farthest {
                    continue;
                }
                for t in &patch.trees[tile.trees.clone()] {
                    let k = self.height_index(t.height);
                    let foot = anchor::offset(t.base, anchor) + Vec3::new(0.5, 0.0, 0.5);
                    let middle = foot + Vec3::Y * (self.talls[k] / 2.0);
                    if !keep(k, foot, middle) || (!standin && self.in_hole(t)) {
                        continue;
                    }
                    out.push(FarTreeInstance {
                        foot: foot.to_array(),
                        height: k as u32 | if standin { STANDIN_TREE } else { 0 },
                    });
                }
            }
        }
    }

    /// **The trees a frame may draw from an eye within `margin` of `eye`**
    /// (anchor-relative): every tree whose middle is within its own
    /// [`reach`] plus `margin` of `eye`, wherever the camera looks -- what
    /// [`FarTrees`] hands the GPU, which drops a tree that is off screen or
    /// faded out (`far_trees.wgsl`). It is worked out again only when the eye
    /// has moved `margin` or the trees, the voxels' box or the screen
    /// changed, so a frame touches no tree on the CPU.
    pub fn in_reach(
        &self,
        (eye, anchor): (Vec3, Anchor),
        (px_per_block, fade): (f32, Fade),
        margin: f32,
        out: &mut Vec<FarTreeInstance>,
    ) {
        let farthest = self.farthest(px_per_block, fade) + margin;
        self.instances(
            (eye, anchor),
            farthest,
            |k, _, middle| middle.distance(eye) <= reach(self.talls[k], px_per_block, fade) + margin,
            out,
        );
    }

    /// **The trees a frame shows**, as instances relative to `anchor`, in
    /// patch, square and tree order: every tree whose box the frustum can
    /// see and whose size on screen is over `fade`'s gone size -- by its own
    /// height, from `eye` (anchor-relative), at `px_per_block` -- and whose
    /// ground is not the voxels', but for a stand-in's, which stands in for
    /// theirs. The definition the GPU's drawing is held to (a tree it draws
    /// is one of [`in_reach`](Self::in_reach)'s that `far_trees.wgsl` does
    /// not drop), and what a census counts as drawn; no frame works it out.
    pub fn visible(
        &self,
        frustum: &Frustum,
        (eye, anchor): (Vec3, Anchor),
        (px_per_block, fade): (f32, Fade),
        out: &mut Vec<FarTreeInstance>,
    ) {
        let r = self.canopy_radius as f32 + 0.5;
        self.instances(
            (eye, anchor),
            self.farthest(px_per_block, fade),
            |k, foot, middle| {
                let tall = self.talls[k];
                tree_scale(tall, middle.distance(eye), px_per_block, fade) > 0.0
                    && frustum.intersects_aabb(&Aabb::new(
                        foot - Vec3::new(r, 0.0, r),
                        foot + Vec3::new(r, tall, r),
                    ))
            },
            out,
        );
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
    /// `.x` the face slots a tree draws, `.y` [`FULL_PX`], `.z` [`GONE_PX`],
    /// `.w` pixels a block is tall at one block.
    fade: [f32; 4],
    /// The face slots ([`StandIn::slots`]): `.x` the box, `.y` the axis.
    slots: [[u32; 4]; MAX_SLOTS],
    /// Per trunk height, the stand-in's height in blocks and the radius of
    /// the sphere about its middle that holds it ([`StandIn::radius`]).
    tall: [[f32; 4]; MAX_HEIGHTS],
    boxes: [[BoxGpu; MAX_BOXES]; MAX_HEIGHTS],
}

impl ParamsGpu {
    fn new(stand_ins: &[StandIn], px_per_block: f32, fade: Fade) -> Self {
        let none = BoxGpu {
            min: [0.0; 4],
            max: [0.0; 4],
            faces: [[0.0, 0.0, 0.0, -1.0]; 6],
        };
        let mut out = ParamsGpu {
            fade: [0.0, fade.full_px, fade.gone_px, px_per_block],
            slots: [[0; 4]; MAX_SLOTS],
            tall: [[0.0; 4]; MAX_HEIGHTS],
            boxes: [[none; MAX_BOXES]; MAX_HEIGHTS],
        };
        let slots = slots_of(stand_ins);
        out.fade[0] = slots.len() as f32;
        for (k, &(b, axis)) in slots.iter().enumerate() {
            out.slots[k] = [b as u32, axis as u32, 0, 0];
        }
        for (k, s) in stand_ins.iter().take(MAX_HEIGHTS).enumerate() {
            out.tall[k] = [s.tall, s.radius(), 0.0, 0.0];
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

/// The face slots every stand-in draws: the first one's
/// ([`StandIn::slots`]). Every trunk height's stand-in is the same boxes of
/// other sizes, so they are every one's.
fn slots_of(stand_ins: &[StandIn]) -> Vec<(usize, usize)> {
    let slots = stand_ins.first().map(StandIn::slots).unwrap_or_default();
    debug_assert!(
        stand_ins.iter().all(|s| s.slots() == slots),
        "the stand-ins of the trunk heights draw different faces"
    );
    debug_assert!(slots.len() <= MAX_SLOTS);
    slots.into_iter().take(MAX_SLOTS).collect()
}

/// The indices of one tree: per face slot, the face that faces the eye,
/// four corners `far_trees.wgsl` makes from the vertex index, drawn as two
/// triangles.
fn tree_indices() -> Vec<u16> {
    (0..MAX_SLOTS as u16)
        .flat_map(|face| [0, 2, 1, 1, 2, 3].map(|i| face * 4 + i))
        .collect()
}

/// **How far the eye may move before the trees handed to the GPU are worked
/// out again** ([`FarTreeSet::in_reach`]'s margin, in blocks). **Chosen, not
/// derived**: a frame at a standing eye -- every gate eye, the camera turning
/// -- then touches no far tree on the CPU; a flight works them out again
/// every 64 blocks it flies (a few hundred frames at the game's speed), and
/// hands the GPU the trees within 64 blocks more than their reach, which its
/// fade drops for nothing but a vertex shader's first lines.
pub const REACH_MARGIN: f32 = 64.0;

/// What the trees in reach were worked out for: where the eye was, in world
/// blocks, and the anchor, screen and fade they were made for.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ReachKey {
    eye: [f64; 3],
    anchor: Anchor,
    px_per_block: f32,
    fade: Fade,
}

/// **When the far trees handed to the GPU are worked out again** -- the one
/// decision a frame makes about them on the CPU ([`FarTrees::prepare`]),
/// with no GPU so a test can drive it: when the eye has moved further than
/// [`REACH_MARGIN`] from where they were worked out, the anchor, the
/// screen's height or the fade changed, the voxels' box moved, or a patch
/// came or went within reach of where they were.
#[derive(Clone, Debug, Default)]
pub struct ReachCache {
    at: Option<ReachKey>,
}

/// How far `p` is from the nearest point of the box `(min, max)`, all in
/// world blocks.
fn distance_to_box(p: [f64; 3], (min, max): ([i64; 3], [i64; 3])) -> f64 {
    (0..3)
        .map(|a| {
            let d = (min[a] as f64 - p[a]).max(p[a] - max[a] as f64).max(0.0);
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

impl ReachCache {
    /// Whether the trees in reach must be worked out again for an eye at
    /// `eye` (world blocks), made relative to `anchor`, on a screen of
    /// `px_per_block`, faded as `fade` says.
    pub fn stale(&self, eye: [f64; 3], anchor: Anchor, px_per_block: f32, fade: Fade) -> bool {
        self.at.is_none_or(|k| {
            k.anchor != anchor
                || k.px_per_block != px_per_block
                || k.fade != fade
                || (0..3).map(|a| (k.eye[a] - eye[a]).powi(2)).sum::<f64>().sqrt()
                    > f64::from(REACH_MARGIN)
        })
    }

    /// They were worked out, for what [`stale`](Self::stale) was asked.
    pub fn worked_out(&mut self, eye: [f64; 3], anchor: Anchor, px_per_block: f32, fade: Fade) {
        self.at = Some(ReachKey {
            eye,
            anchor,
            px_per_block,
            fade,
        });
    }

    /// A patch's trees came or went, within the box `(min, max)` (world
    /// blocks); `farthest(px_per_block, fade)` is how far any tree is drawn.
    /// The trees in reach change only if that box is within reach, plus the
    /// margin, of where they were worked out.
    pub fn patch_changed(
        &mut self,
        bounds: ([i64; 3], [i64; 3]),
        farthest: impl Fn(f32, Fade) -> f32,
    ) {
        if let Some(k) = self.at {
            let within = f64::from(farthest(k.px_per_block, k.fade) + REACH_MARGIN);
            if distance_to_box(k.eye, bounds) <= within {
                self.at = None;
            }
        }
    }

    /// Everything changed: the voxels' box, or every patch.
    pub fn forget(&mut self) {
        self.at = None;
    }
}

/// **The far trees on the GPU**: [`FarTreeSet`], the stand-ins' parameters,
/// and the instances of every tree in reach ([`FarTreeSet::in_reach`]),
/// uploaded when [`ReachCache`] says they changed -- not when the camera
/// turns -- and drawn in one instanced draw, `far_trees.wgsl` dropping the
/// ones off screen or faded out.
pub struct FarTrees {
    set: FarTreeSet,
    stand_ins: Vec<StandIn>,
    params: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    indices: wgpu::Buffer,
    /// Indices a tree is drawn with: six a face slot.
    index_count: u32,
    instances: Option<wgpu::Buffer>,
    capacity: usize,
    /// The trees in reach, as uploaded.
    handed: Vec<FarTreeInstance>,
    reach: ReachCache,
    /// How many times they were worked out and uploaded.
    workings_out: u64,
    /// What the instances were made relative to, and for what screen.
    anchor: Anchor,
    px_per_block: f32,
    /// How the trees fade: the game's, unless a measurement says otherwise
    /// ([`set_fade`](Self::set_fade)).
    fade: Fade,
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
            bytemuck::bytes_of(&ParamsGpu::new(&stand_ins, 0.0, Fade::GAME)),
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
        let slots = slots_of(&stand_ins).len();
        Self {
            set: FarTreeSet::new(shapes, &stand_ins),
            stand_ins,
            params,
            bind_group,
            indices,
            index_count: (slots * 6) as u32,
            instances: None,
            capacity: 0,
            handed: Vec::new(),
            reach: ReachCache::default(),
            workings_out: 0,
            anchor,
            px_per_block: 0.0,
            fade: Fade::GAME,
        }
    }

    /// Fade the trees as `fade` says from the next
    /// [`prepare`](Self::prepare): for a measurement of what the fade
    /// changes; the game's is [`Fade::GAME`].
    pub fn set_fade(&mut self, fade: Fade) {
        self.fade = fade;
        // Written again by the next `prepare`.
        self.px_per_block = 0.0;
    }

    /// The trees a far patch carries (the caller's key, `id`), replacing any
    /// it carried before.
    pub fn insert(&mut self, id: u64, trees: Vec<FarTreePlacement>) {
        self.insert_as(id, trees, FarRole::Terrain);
    }

    /// [`insert`](Self::insert), drawn as `role` ([`FarTreeSet::insert_as`]).
    pub fn insert_as(&mut self, id: u64, trees: Vec<FarTreePlacement>, role: FarRole) {
        self.patch_changed(id);
        self.set.insert_as(id, trees, role);
        self.patch_changed(id);
    }

    pub fn remove(&mut self, id: u64) {
        self.patch_changed(id);
        self.set.remove(id);
    }

    /// Tell the cache that the patch `id`'s trees (as held now) change.
    fn patch_changed(&mut self, id: u64) {
        if let Some(bounds) = self.set.patch_box(id) {
            let set = &self.set;
            self.reach
                .patch_changed(bounds, |px, fade| set.farthest(px, fade));
        }
    }

    pub fn clear(&mut self) {
        self.set.clear();
        self.reach.forget();
    }

    pub fn set_hole(&mut self, hole: Option<([i32; 3], [i32; 3])>) {
        if hole != self.set.hole {
            self.set.set_hole(hole);
            self.reach.forget();
        }
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

    /// Trees handed to the GPU, from the last [`prepare`](Self::prepare):
    /// every one in reach, of which it draws those on screen and not faded
    /// out ([`shown`](Self::shown) counts those).
    pub fn handed(&self) -> usize {
        self.handed.len()
    }

    /// How many times the trees in reach were worked out and uploaded
    /// ([`prepare`](Self::prepare)): once at a standing eye, however the
    /// camera turns.
    pub fn workings_out(&self) -> u64 {
        self.workings_out
    }

    /// What drawing the trees handed over costs: `(draws, vertex shader
    /// runs)` -- one instanced draw, and four corners a face slot
    /// ([`StandIn::slots`]) a tree. The runs for a tree off screen or faded
    /// out end in the shader's first lines.
    pub fn handed_cost(&self) -> (u32, u64) {
        let trees = self.handed.len() as u64;
        let faces = self.index_count as u64 / 6;
        (u32::from(trees > 0), trees * faces * 4)
    }

    /// How many trees a frame with this frustum shows ([`FarTreeSet::visible`],
    /// the definition the drawing is held to): for a census, never a frame.
    pub fn shown(&self, frustum: &Frustum, eye: Vec3, anchor: Anchor) -> usize {
        let mut out = Vec::new();
        self.set.visible(
            frustum,
            (eye, anchor),
            (self.px_per_block, self.fade),
            &mut out,
        );
        out.len()
    }

    /// The anchor the instances drawn were made relative to.
    pub fn anchor(&self) -> Anchor {
        self.anchor
    }

    /// Bring the trees handed to the GPU up to date for an eye at `eye`
    /// (relative to `anchor`, which the instances are made relative to) on a
    /// screen `screen_height` pixels tall: nothing, unless [`ReachCache`]
    /// says they changed -- then every tree in reach
    /// ([`FarTreeSet::in_reach`]), uploaded. Where the camera looks changes
    /// nothing here: `far_trees.wgsl` drops what is off screen.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
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
                bytemuck::bytes_of(&ParamsGpu::new(&self.stand_ins, px, self.fade)),
            );
        }
        self.anchor = anchor;
        let world = [0, 1, 2].map(|a| f64::from(eye[a]) + f64::from(anchor[a]));
        if !self.reach.stale(world, anchor, px, self.fade) {
            return;
        }
        self.reach.worked_out(world, anchor, px, self.fade);
        self.workings_out += 1;
        self.set.in_reach(
            (eye, anchor),
            (px, self.fade),
            REACH_MARGIN,
            &mut self.handed,
        );
        if self.handed.is_empty() {
            return;
        }
        if self.instances.is_none() || self.handed.len() > self.capacity {
            self.capacity = self.handed.len().next_power_of_two();
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
            bytemuck::cast_slice(&self.handed),
        );
    }

    /// Draw the trees handed over. The pipeline and bind group 0 (the
    /// camera) are the caller's to set.
    pub(crate) fn encode(&self, pass: &mut wgpu::RenderPass<'_>) {
        let Some(instances) = self.instances.as_ref() else {
            return;
        };
        if self.handed.is_empty() || self.index_count == 0 {
            return;
        }
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, instances.slice(..));
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..self.index_count, 0, 0..self.handed.len() as u32);
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
        let game = Fade::GAME;
        assert_eq!(tree_scale(tall, at(5.0), k, game), 1.0);
        assert_eq!(tree_scale(tall, at(FULL_PX), k, game), 1.0);
        let half = (FULL_PX + GONE_PX) / 2.0;
        assert!((tree_scale(tall, at(half), k, game) - 0.5).abs() < 1e-4);
        assert_eq!(tree_scale(tall, at(GONE_PX), k, game), 0.0);
        assert_eq!(tree_scale(tall, at(1.0), k, game), 0.0);
        // A taller screen draws a tree whole further out.
        let k2 = px_per_block(2160);
        assert_eq!(tree_scale(tall, at(FULL_PX) * 1.9, k2, game), 1.0);
        // How far: where it is gone.
        assert!((reach(tall, k, game) - at(GONE_PX)).abs() < 1e-2);
        // The measurements' fades: every tree whole, and none.
        assert_eq!(tree_scale(tall, 1e6, k, Fade::WHOLE), 1.0);
        assert_eq!(reach(tall, k, Fade::WHOLE), f32::INFINITY);
        assert_eq!(tree_scale(tall, 1.0, k, Fade::GONE), 0.0);
    }

    fn the_oak() -> (
        BlockRegistry,
        StructureRegistry,
        crate::materials::TextureLayers,
    ) {
        let registry = crate::render::load_registry();
        let layers = crate::materials::TextureLayers::from_registry(&registry);
        (registry, crate::render::load_structures(), layers)
    }

    fn the_stand_ins() -> Vec<StandIn> {
        let (registry, structures, layers) = the_oak();
        stand_ins(&registry, &structures, &|n: &str| layers.layer_of(n))
    }

    /// **The stand-in is the oak's bounds, in the silhouette's colours**:
    /// one per trunk height; the canopy box is the oak's own bounds
    /// (`tree_bounds`) from its lowest canopy row, the trunk below it; each
    /// face the colour of its texture as the silhouette has it, shaded by
    /// the oak's openness that way and in that texture (`oak_shades`); five
    /// face slots, the trunk's top and foot never drawn.
    #[test]
    fn the_stand_in_is_the_oaks_bounds_in_the_silhouettes_colours() {
        let (registry, structures, layers) = the_oak();
        let layer_of = |n: &str| layers.layer_of(n);
        let all = the_stand_ins();
        let shapes = TreeShapes::of(&structures);
        assert_eq!(all.len(), shapes.count());
        let (oaks, silhouettes) =
            crate::trees::tree_model_meshes(&registry, &structures, &layer_of);
        for ((h, s), (oak, silhouette)) in shapes
            .heights
            .clone()
            .zip(&all)
            .zip(oaks.iter().zip(&silhouettes))
        {
            let (min, max) = cubara_voxel::tree_bounds(h, shapes.canopy_radius);
            assert_eq!(s.tall, max[1] as f32 + 1.0, "height {h}");
            assert_eq!(
                s.slots(),
                vec![
                    (0, 0),
                    (0, 1),
                    (0, 2),
                    (1, 0),
                    (1, 1),
                    (1, 2),
                    (2, 0),
                    (2, 2)
                ]
            );
            let [along_x, along_z, trunk] = [s.boxes[0], s.boxes[1], s.boxes[2]];
            let bottom = (h - 1 - shapes.canopy_radius) as f32;
            let (lo, hi) = (min[0] as f32, max[0] as f32 + 1.0);
            // The oak's bounds, a block narrower each side across the arm.
            assert_eq!(along_x.min, [lo, bottom, lo + 1.0]);
            assert_eq!(along_x.max, [hi, max[1] as f32 + 1.0, hi - 1.0]);
            assert_eq!(along_z.min, [lo + 1.0, bottom, lo]);
            assert_eq!(along_z.max, [hi - 1.0, max[1] as f32 + 1.0, hi]);
            assert_eq!(trunk.min, [0.0, 0.0, 0.0]);
            assert_eq!(trunk.max, [1.0, bottom, 1.0]);
            assert!(trunk.faces[2].is_none() && trunk.faces[3].is_none());
            // Every face's colour and openness are a silhouette face's, of
            // the same direction.
            let shades = oak_shades(oak);
            for b in [along_x, along_z, trunk] {
                for (d, face) in b.faces.iter().enumerate() {
                    let Some(face) = face else { continue };
                    let found = silhouette.vertices.iter().any(|v| {
                        let n = Vec3::from(v.normal);
                        n.dot(Vec3::from(FACES[d].normal())) > 0.99
                            && v.color == face.colour
                            && (v.open - face.open).abs() < 1e-6
                    });
                    assert!(
                        found,
                        "height {h}: face {d} {face:?} is no silhouette face's"
                    );
                    assert!(shades.values().any(|&o| (o - face.open).abs() < 1e-6));
                }
            }
        }
    }

    /// A set of one patch of trees in a row along x, every 64 blocks from
    /// the eye, all of trunk height 5.
    fn a_row(eye: Vec3) -> (FarTreeSet, Vec<FarTreePlacement>) {
        let (_, structures, _) = the_oak();
        let mut set = FarTreeSet::new(&TreeShapes::of(&structures), &the_stand_ins());
        let trees: Vec<_> = (1..80)
            .map(|k| FarTreePlacement {
                base: [eye.x as i32 + 64 * k, 40, eye.z as i32],
                height: 5,
            })
            .collect();
        set.insert(7, trees.clone());
        (set, trees)
    }

    fn looking(eye: Vec3, dir: Vec3) -> Frustum {
        Frustum::from_view_proj(crate::CameraUniform::look_view_proj(16.0 / 9.0, eye, dir))
    }

    /// **A frame draws the trees in view, near enough to show, and not the
    /// voxels'**: of a row of trees running away from the eye, those up to
    /// where a tree of their height is [`GONE_PX`] tall, from in front of
    /// the eye; none when it looks away; none whose ground is in the
    /// voxels' box; and every one with the fade that never takes a tree.
    #[test]
    fn a_frame_draws_the_trees_in_view_and_in_reach() {
        let eye = Vec3::new(0.0, 60.0, 0.0);
        let (mut set, trees) = a_row(eye);
        let k = px_per_block(1080);
        let mut out = Vec::new();
        set.visible(
            &looking(eye, Vec3::X),
            (eye, [0; 3]),
            (k, Fade::GAME),
            &mut out,
        );
        let tall = 6.0;
        let near = trees
            .iter()
            .filter(|t| {
                let middle = Vec3::new(t.base[0] as f32 + 0.5, 40.0 + tall / 2.0, 0.5);
                tree_scale(tall, middle.distance(eye), k, Fade::GAME) > 0.0
            })
            .count();
        assert!(near > 20 && near < trees.len(), "{near} of {}", trees.len());
        assert_eq!(out.len(), near);
        // Each at its foot, relative to the anchor, of its height's stand-in.
        assert_eq!(out[0].foot, [64.5, 40.0, 0.5]);
        assert_eq!(out[0].height, 1);
        // Relative to another anchor, the same trees there.
        let anchor = [4096, 0, 0];
        let mut moved = Vec::new();
        set.visible(
            &looking(eye - Vec3::new(4096.0, 0.0, 0.0), Vec3::X),
            (eye - Vec3::new(4096.0, 0.0, 0.0), anchor),
            (k, Fade::GAME),
            &mut moved,
        );
        assert_eq!(moved.len(), out.len());
        assert_eq!(moved[0].foot, [64.5 - 4096.0, 40.0, 0.5]);
        // Looking away: none.
        set.visible(
            &looking(eye, -Vec3::X),
            (eye, [0; 3]),
            (k, Fade::GAME),
            &mut out,
        );
        assert!(out.is_empty());
        // Every one, with the fade that takes none.
        set.visible(
            &looking(eye, Vec3::X),
            (eye, [0; 3]),
            (k, Fade::WHOLE),
            &mut out,
        );
        assert_eq!(out.len(), trees.len());
        // The voxels' box over the first 1,000 blocks. A tree whose ground
        // block (39, under its base at 40) is in the box is the voxels': they
        // draw that block and the tree on it. One standing on ground above
        // the box is the far terrain's.
        set.set_hole(Some(([-16, 0, -16], [1000, 39, 16])));
        set.visible(
            &looking(eye, Vec3::X),
            (eye, [0; 3]),
            (k, Fade::GAME),
            &mut out,
        );
        assert_eq!(
            out.len(),
            near,
            "a tree standing above the box is the far terrain's"
        );
        set.set_hole(Some(([-16, 0, -16], [1000, 40, 16])));
        set.visible(
            &looking(eye, Vec3::X),
            (eye, [0; 3]),
            (k, Fade::GAME),
            &mut out,
        );
        let in_box = trees.iter().filter(|t| t.base[0] < 1000).count();
        assert_eq!(out.len(), near - in_box);
        assert!(out.iter().all(|t| t.foot[0] > 1000.0));
        // Removed: nothing.
        set.remove(7);
        set.visible(
            &looking(eye, Vec3::X),
            (eye, [0; 3]),
            (k, Fade::WHOLE),
            &mut out,
        );
        assert!(out.is_empty() && set.is_empty());
    }

    /// The tint's numbers for `far.wgsl`: per height the density and the
    /// stand-in's height, per direction its area and colours; nothing for a
    /// world without trees.
    #[test]
    fn the_tint_carries_the_stand_ins_by_direction() {
        let all = the_stand_ins();
        let density = [1.0e-4, 2.0e-4, 3.0e-4];
        let tint = FarTreeTint::new(&density, &all);
        let gpu = TintGpu::of(Some(&tint), 935.0);
        assert_eq!(gpu.fade, [3.0, FULL_PX, GONE_PX, 935.0]);
        assert_eq!(gpu.view[0], 0.0);
        for (k, s) in all.iter().enumerate() {
            assert_eq!(gpu.height[k], [density[k], s.tall, 0.0, 0.0]);
            let dirs = s.by_direction();
            // +Y: the canopy's tops, two arms of 15 square blocks of leaves.
            assert_eq!(gpu.colour[k][2][3], 30.0);
            for (slot, &d) in TINT_DIRECTIONS.iter().enumerate() {
                assert_eq!(gpu.colour[k][slot][3], dirs[d].0);
                assert_eq!(gpu.open[k][slot][..3], dirs[d].2);
            }
        }
        let above = FarTreeTint {
            view: TintView::FromAbove,
            ..tint
        };
        assert_eq!(TintGpu::of(Some(&above), 935.0).view[0], 1.0);
        assert_eq!(TintGpu::of(None, 935.0), TintGpu::NONE);
    }

    /// **A stand-in's trees are the strip's** (issue #433 item 1 with
    /// §9.11's stand-in): drawn inside the voxels' box -- which the far
    /// terrain's own trees there are not -- and flagged, so the shader draws
    /// them a hair behind where they stand, as the stand-in's ground is.
    #[test]
    fn a_stand_in_draws_its_trees_inside_the_box_behind() {
        let eye = Vec3::new(0.0, 60.0, 0.0);
        let (mut set, trees) = a_row(eye);
        let k = px_per_block(1080);
        // Every tree's ground in the box.
        set.set_hole(Some(([-16, 0, -16], [100_000, 40, 16])));
        let mut out = Vec::new();
        set.visible(
            &looking(eye, Vec3::X),
            (eye, [0; 3]),
            (k, Fade::GAME),
            &mut out,
        );
        assert!(
            out.is_empty(),
            "the far terrain's trees in the box are the voxels'"
        );
        set.remove(7);
        set.insert_as(9, trees, FarRole::StandIn);
        set.visible(
            &looking(eye, Vec3::X),
            (eye, [0; 3]),
            (k, Fade::GAME),
            &mut out,
        );
        assert!(out.len() > 20, "only {} of the stand-in's trees", out.len());
        assert!(out.iter().all(|t| t.height & STANDIN_TREE != 0));
        assert!(out.iter().all(|t| t.height & !STANDIN_TREE == 1));
    }

    /// The shader reads the flag and the push as this module and the far
    /// terrain's stand-in define them (`far.rs`'s `STANDIN_BEHIND`).
    #[test]
    fn the_shader_draws_a_stand_in_tree_as_the_stand_in() {
        let wgsl = crate::haze::FAR_TREES_WGSL;
        let constant = |name: &str| {
            wgsl.lines()
                .find_map(|l| l.strip_prefix(&format!("const {name}: ")))
                .unwrap_or_else(|| panic!("far_trees.wgsl has {name}"))
                .to_string()
        };
        let behind: f32 = constant("STANDIN_BEHIND")
            .strip_prefix("f32 = ")
            .and_then(|v| v.trim_end_matches(';').parse().ok())
            .expect("a number");
        assert_eq!(behind, crate::far::STANDIN_BEHIND);
        let flag = constant("STANDIN_TREE");
        assert_eq!(
            flag.trim_end_matches(';'),
            format!("u32 = {STANDIN_TREE:#010x}u")
        );
        for line in [
            "let behind = frame.eye.xyz + (p - frame.eye.xyz) * STANDIN_BEHIND;",
            "out.clip_pos = frame.view_proj * vec4<f32>(select(p, behind, standin), 1.0);",
            "let height = in.height & 0xFFu;",
        ] {
            assert!(wgsl.contains(line), "far_trees.wgsl no longer has: {line}");
        }
    }

    /// A small deterministic generator for the tests below (no crate for it).
    struct Xorshift(u64);

    impl Xorshift {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }
        fn between(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.next()
        }
    }

    /// **The trees handed to the GPU are worked out again only when they
    /// can change** ([`ReachCache`]): never while the eye stands or moves
    /// less than [`REACH_MARGIN`] -- however the camera turns -- and always
    /// when it moves further, the anchor, the screen or the fade changes, a
    /// patch comes or goes within reach of where they were worked out, or
    /// the voxels' box moves; a patch beyond reach changes nothing.
    #[test]
    fn the_trees_in_reach_are_worked_out_again_only_when_they_can_change() {
        let k = px_per_block(1080);
        let farthest = |px: f32, fade: Fade| reach(8.0, px, fade);
        let mut cache = ReachCache::default();
        let eye = [100.0, 300.0, -50.0];
        assert!(cache.stale(eye, [0; 3], k, Fade::GAME), "never worked out");
        cache.worked_out(eye, [0; 3], k, Fade::GAME);
        assert!(!cache.stale(eye, [0; 3], k, Fade::GAME), "standing");
        let near = [eye[0] + 40.0, eye[1] - 40.0, eye[2] + 20.0];
        assert!(!cache.stale(near, [0; 3], k, Fade::GAME), "within the margin");
        let far = [eye[0] + f64::from(REACH_MARGIN) + 0.5, eye[1], eye[2]];
        assert!(cache.stale(far, [0; 3], k, Fade::GAME), "moved past the margin");
        assert!(cache.stale(eye, [4096, 0, 0], k, Fade::GAME), "rebased");
        assert!(cache.stale(eye, [0; 3], px_per_block(1664), Fade::GAME), "a taller screen");
        assert!(cache.stale(eye, [0; 3], k, Fade::WHOLE), "another fade");
        // A patch beyond reach of where they were worked out changes nothing...
        let reach_far = f64::from(farthest(k, Fade::GAME) + REACH_MARGIN);
        let beyond = (eye[0] + reach_far + 2.0) as i64;
        cache.patch_changed(([beyond, 0, -50], [beyond + 256, 400, 206]), farthest);
        assert!(!cache.stale(eye, [0; 3], k, Fade::GAME), "a patch beyond reach");
        // ...one within it does.
        let within = (eye[0] + reach_far - 2.0) as i64;
        cache.patch_changed(([within, 0, -50], [within + 256, 400, 206]), farthest);
        assert!(cache.stale(eye, [0; 3], k, Fade::GAME), "a patch within reach");
        cache.worked_out(eye, [0; 3], k, Fade::GAME);
        cache.forget();
        assert!(cache.stale(eye, [0; 3], k, Fade::GAME), "the voxels' box moved");
    }

    /// **The trees handed over hold every tree any frame shows from an eye
    /// within [`REACH_MARGIN`] of where they were worked out**: over a
    /// field of trees round a voxels' box, with a strip's stand-in in it,
    /// for random eyes within the margin and random views at two screen
    /// heights, every tree [`FarTreeSet::visible`] shows is one
    /// [`FarTreeSet::in_reach`] handed over -- so a frame between two
    /// workings-out never misses one.
    #[test]
    fn the_trees_in_reach_hold_every_tree_a_frame_within_the_margin_shows() {
        let (_, structures, _) = the_oak();
        let mut set = FarTreeSet::new(&TreeShapes::of(&structures), &the_stand_ins());
        let mut trees = Vec::new();
        for i in -60..60 {
            for j in -60..60 {
                trees.push(FarTreePlacement {
                    base: [i * 48 + (j % 7) * 5, 40 + (i + j).rem_euclid(9), j * 48],
                    height: 4 + (i * 7 + j).rem_euclid(3),
                });
            }
        }
        set.insert(1, trees);
        // The strip's stand-in, inside the box along its +x side.
        let strip: Vec<_> = (0..40)
            .map(|j| FarTreePlacement {
                base: [1000, 44, -1000 + j * 50],
                height: 5,
            })
            .collect();
        set.insert_as(2, strip, FarRole::StandIn);
        set.set_hole(Some(([-1024, -64, -1024], [1024, 400, 1024])));
        let worked_out_at = Vec3::new(30.0, 120.0, -20.0);
        let mut rng = Xorshift(0x5eed_0f_7ee5);
        let (mut shown, mut trials) = (0usize, 0usize);
        for height in [1080, 1664] {
            let k = px_per_block(height);
            let mut handed = Vec::new();
            set.in_reach(
                (worked_out_at, [0; 3]),
                (k, Fade::GAME),
                REACH_MARGIN,
                &mut handed,
            );
            let key = |t: &FarTreeInstance| (t.foot.map(f32::to_bits), t.height);
            let handed: std::collections::HashSet<_> = handed.iter().map(key).collect();
            for _ in 0..200 {
                let offset = Vec3::new(
                    rng.between(-1.0, 1.0),
                    rng.between(-1.0, 1.0),
                    rng.between(-1.0, 1.0),
                )
                .normalize_or_zero()
                    * rng.between(0.0, REACH_MARGIN * 0.999);
                let eye = worked_out_at + offset;
                let yaw = rng.between(0.0, std::f32::consts::TAU);
                let pitch = rng.between(-0.7, 0.3);
                let dir = Vec3::new(yaw.cos(), pitch, yaw.sin());
                let frustum = Frustum::from_view_proj(crate::CameraUniform::look_view_proj(
                    16.0 / 9.0,
                    eye,
                    dir,
                ));
                let mut out = Vec::new();
                set.visible(&frustum, (eye, [0; 3]), (k, Fade::GAME), &mut out);
                for t in &out {
                    assert!(
                        handed.contains(&key(t)),
                        "{height} rows, eye {eye}: the tree at {:?} is shown but was not handed over",
                        t.foot
                    );
                }
                shown += out.len();
                trials += 1;
            }
        }
        assert!(
            shown > trials * 50,
            "{shown} trees shown over {trials} frames: this pins little"
        );
    }

    /// `far_trees.wgsl`'s test of a tree off screen, line for line: its
    /// middle in clip space, against the side planes widened by the radius
    /// times a bound on each plane's normal, and behind the eye.
    fn off_screen(view_proj: glam::Mat4, middle: Vec3, radius: f32) -> bool {
        let c = view_proj * middle.extend(1.0);
        let row = |i: usize| {
            Vec3::new(
                view_proj.x_axis[i],
                view_proj.y_axis[i],
                view_proj.z_axis[i],
            )
        };
        let kx = radius * (1.0 + row(0).length());
        let ky = radius * (1.0 + row(1).length());
        c.w < -radius || c.x.abs() > c.w + kx || c.y.abs() > c.w + ky
    }

    /// **The shader drops only trees off screen**: over random views at two
    /// aspects and random trees, a tree whose box the frustum can see
    /// (`Frustum::intersects_aabb`, the scene's own test, of the stand-in at
    /// full size) is never dropped -- with the radius `ParamsGpu` hands the
    /// shader, which holds every box of it at every size it fades through --
    /// while most of those it cannot see are; and the shader says it as
    /// [`off_screen`] does.
    #[test]
    fn the_shader_drops_only_trees_off_screen() {
        let stand_ins = the_stand_ins();
        let params = ParamsGpu::new(&stand_ins, px_per_block(1080), Fade::GAME);
        for (k, s) in stand_ins.iter().enumerate() {
            assert_eq!(params.tall[k][1], s.radius());
            // Every corner, at every size it fades through, within it.
            let middle = Vec3::new(0.5, s.tall / 2.0, 0.5);
            let pivot = Vec3::new(0.5, 0.0, 0.5);
            for b in &s.boxes {
                for i in 0..8 {
                    let corner = Vec3::new(
                        if i & 1 == 0 { b.min[0] } else { b.max[0] },
                        if i & 2 == 0 { b.min[1] } else { b.max[1] },
                        if i & 4 == 0 { b.min[2] } else { b.max[2] },
                    );
                    for step in 0..=10 {
                        let scale = step as f32 / 10.0;
                        let p = pivot + (corner - pivot) * scale;
                        assert!(p.distance(middle) <= s.radius() + 1e-4, "{k}: {p}");
                    }
                }
            }
        }
        let mut rng = Xorshift(0xf00d_cafe);
        let (mut outside, mut dropped) = (0usize, 0usize);
        for aspect in [16.0 / 9.0, 2560.0 / 1664.0] {
            for _ in 0..300 {
                let eye = Vec3::new(
                    rng.between(-500.0, 500.0),
                    rng.between(20.0, 900.0),
                    rng.between(-500.0, 500.0),
                );
                let yaw = rng.between(0.0, std::f32::consts::TAU);
                let dir = Vec3::new(yaw.cos(), rng.between(-0.9, 0.4), yaw.sin());
                let view_proj = crate::CameraUniform::look_view_proj(aspect, eye, dir);
                let frustum = Frustum::from_view_proj(view_proj);
                for _ in 0..60 {
                    let k = (rng.next() * stand_ins.len() as f32) as usize % stand_ins.len();
                    let s = &stand_ins[k];
                    let foot = Vec3::new(
                        rng.between(-3000.0, 3000.0).floor() + 0.5,
                        rng.between(0.0, 200.0).floor(),
                        rng.between(-3000.0, 3000.0).floor() + 0.5,
                    );
                    let middle = foot + Vec3::Y * (s.tall / 2.0);
                    let corners = s.boxes.iter().fold(
                        (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
                        |(lo, hi), b| {
                            (
                                lo.min(Vec3::from(b.min) - Vec3::new(0.5, 0.0, 0.5)),
                                hi.max(Vec3::from(b.max) - Vec3::new(0.5, 0.0, 0.5)),
                            )
                        },
                    );
                    let seen = frustum.intersects_aabb(&Aabb::new(foot + corners.0, foot + corners.1));
                    let drop = off_screen(view_proj, middle, s.radius());
                    assert!(!(seen && drop), "a tree at {foot} the frustum can see was dropped");
                    if !seen {
                        outside += 1;
                        dropped += usize::from(drop);
                    }
                }
            }
        }
        assert!(
            dropped * 10 > outside * 9,
            "only {dropped} of {outside} trees off screen dropped"
        );
        let wgsl = crate::haze::FAR_TREES_WGSL;
        for line in [
            "let c = frame.view_proj * vec4<f32>(middle, 1.0);",
            "let radius = params.tall[height].y;",
            "let kx = radius * (1.0 + length(vec3<f32>(vp[0].x, vp[1].x, vp[2].x)));",
            "let ky = radius * (1.0 + length(vec3<f32>(vp[0].y, vp[1].y, vp[2].y)));",
            "if c.w < -radius || abs(c.x) > c.w + kx || abs(c.y) > c.w + ky {",
        ] {
            assert!(wgsl.contains(line), "far_trees.wgsl no longer has: {line}");
        }
    }
}

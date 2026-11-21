//! The far terrain's GPU half (`docs/PROPOSAL_FAR_VIEW.md` §3.2): where the
//! patches' heights live on the GPU, which of them are drawn this frame, and
//! the parameters `far.wgsl` reads.
//!
//! Which patches exist and what heights they hold is decided elsewhere
//! (`cubara_world::far`), and handed over as plain numbers -- this crate does
//! not depend on the world crate (Rule 3), so a patch arrives as a
//! [`FarPatch`] and leaves as a [`FarSlot`].

use crate::aggregate::Materials;
use crate::anchor::{self, Anchor};
use crate::culling::{Aabb, Frustum};
use crate::render::far_bind_group_layout;
use glam::Vec3;

/// Quads along one side of a patch. Must match `far.wgsl`'s `QUADS` and
/// `cubara_world::far::PATCH_QUADS`; the app asserts the second.
pub const FAR_QUADS: usize = 32;

/// The most faces one patch can have: a top for every quad, and a wall on
/// every quad line beside every quad, both ways.
pub const FAR_MAX_FACES: usize = FAR_QUADS * FAR_QUADS + 2 * (FAR_QUADS + 1) * FAR_QUADS;

/// Which way a [`FarFace`] looks. `far.wgsl`'s `DIR_*` and `FACE_NORMALS`
/// are in this order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FarDir {
    Top = 0,
    PosX = 1,
    NegX = 2,
    PosZ = 3,
    NegZ = 4,
}

impl FarDir {
    pub const ALL: [FarDir; 5] = [
        FarDir::Top,
        FarDir::PosX,
        FarDir::NegX,
        FarDir::PosZ,
        FarDir::NegZ,
    ];

    /// The unit vector a face of this direction looks along.
    pub fn normal(self) -> Vec3 {
        match self {
            FarDir::Top => Vec3::Y,
            FarDir::PosX => Vec3::X,
            FarDir::NegX => Vec3::NEG_X,
            FarDir::PosZ => Vec3::Z,
            FarDir::NegZ => Vec3::NEG_Z,
        }
    }

    /// Whether this is a wall standing on a line of constant x.
    fn faces_x(self) -> bool {
        matches!(self, FarDir::PosX | FarDir::NegX)
    }
}

/// One rectangle of a patch drawn as blocks -- a flat top or a wall
/// (`cubara_world::far::Face`, which this mirrors field for field because
/// this crate does not depend on that one, Rule 3).
///
/// In quads from the patch's low corner. A top spans `i0..i1` by `j0..j1` at
/// height `top`; a wall stands on one quad line (`i0 == i1` facing along x,
/// `j0 == j1` facing along z), runs along the other axis, and hangs from
/// `top` down `rise` blocks -- and, on the patch's edge (`border`), the
/// patch's skirt below that.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FarFace {
    pub dir: FarDir,
    pub i0: u8,
    pub j0: u8,
    pub i1: u8,
    pub j1: u8,
    /// The height of a top, or of a wall's upper edge, in whole blocks.
    pub top: i32,
    /// A wall: the blocks of step it shows. Zero for a top.
    pub rise: u32,
    pub border: bool,
    /// A wall: the share of it, in 255ths, that is the surface block's side
    /// and that is soil; the rest is stone. Zero for a top.
    pub mix: [u8; 2],
    /// How open the face is to the sky, in 255ths: what ambient occlusion
    /// leaves of its light ([`face_open`]).
    pub open: u8,
}

/// One corner of a [`FarFace`]: what `far.wgsl` is handed per vertex.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FarVertex {
    /// The corner's place on the patch's grid of quad corners, `0..=32`.
    pub at: [i32; 2],
    pub dir: FarDir,
    /// Which corner of its face: bit 0, the high end of the face's ground
    /// axis (x for a top and for a wall facing along z, z for a wall facing
    /// along x); bit 1, the high end of the other (z for a top, **up** for a
    /// wall, so a wall's lower corners are the ones without it).
    pub corner: u8,
    pub top: i32,
    pub rise: u32,
    pub border: bool,
}

/// The heights a face can carry: 18 bits, biased.
pub const FAR_TOP_RANGE: std::ops::RangeInclusive<i32> = -131_072..=131_071;
/// The tallest step a face can carry: 14 bits.
pub const FAR_MAX_RISE: u32 = 16_383;

impl FarVertex {
    /// The corner's height, in blocks, in a patch whose skirt is `skirt`
    /// deep: the face's top, but for a wall's lower corners.
    pub fn y(&self, skirt: f32) -> f32 {
        let low = self.dir != FarDir::Top && self.corner & 2 == 0;
        let drop = if low {
            self.rise as f32 + if self.border { skirt } else { 0.0 }
        } else {
            0.0
        };
        self.top as f32 - drop
    }

    /// The direction `face.wgsl`'s `CRACK_SEAL` moves this corner in: out of
    /// its face along the face's two in-plane axes.
    pub fn outward(&self) -> Vec3 {
        let su = if self.corner & 1 != 0 { 1.0 } else { -1.0 };
        let sv = if self.corner & 2 != 0 { 1.0 } else { -1.0 };
        match self.dir {
            FarDir::Top => Vec3::new(su, 0.0, sv),
            d if d.faces_x() => Vec3::new(0.0, sv, su),
            _ => Vec3::new(su, sv, 0.0),
        }
    }
}

impl FarFace {
    /// The face's four corners, in the order that draws it: with
    /// [`QUAD_INDICES`], two triangles wound counter-clockwise as seen from
    /// the side the face looks out of (the far pipeline culls the back).
    pub fn corners(&self) -> [FarVertex; 4] {
        let v = |at: [u8; 2], corner: u8| FarVertex {
            at: [at[0] as i32, at[1] as i32],
            dir: self.dir,
            corner,
            top: self.top,
            rise: self.rise,
            border: self.border,
        };
        let (i0, j0, i1, j1) = (self.i0, self.j0, self.i1, self.j1);
        match self.dir {
            FarDir::Top => [
                v([i0, j0], 0),
                v([i1, j0], 1),
                v([i0, j1], 2),
                v([i1, j1], 3),
            ],
            // A wall: low end and high end of its run, bottom then top --
            // and the other way along the run for the two directions that
            // would otherwise wind clockwise.
            FarDir::PosX => [
                v([i0, j0], 0),
                v([i0, j1], 1),
                v([i0, j0], 2),
                v([i0, j1], 3),
            ],
            FarDir::NegX => [
                v([i0, j1], 1),
                v([i0, j0], 0),
                v([i0, j1], 3),
                v([i0, j0], 2),
            ],
            FarDir::PosZ => [
                v([i1, j0], 1),
                v([i0, j0], 0),
                v([i1, j0], 3),
                v([i0, j0], 2),
            ],
            FarDir::NegZ => [
                v([i0, j0], 0),
                v([i1, j0], 1),
                v([i0, j0], 2),
                v([i1, j0], 3),
            ],
        }
    }
}

/// The indices that draw a face from its four [`corners`](FarFace::corners).
pub const QUAD_INDICES: [u16; 6] = [0, 2, 1, 1, 2, 3];

/// **The one place a face's colour is decided** (the seam issue #422 asks
/// for; `far.wgsl`'s `face_colour` is this, and a golden test holds the two
/// together): a top wears the top of the surface block, and a wall the mix
/// of the surface block's side, soil and stone it carries -- which is what
/// level 0's own risers facing that way are made of over the wall's quad
/// (`cubara_world::far::QuadGround`, issue #423 decision 2), not what the
/// wall's height would suggest.
pub fn face_colour(materials: &Materials, face: &FarFace) -> Vec3 {
    if face.dir == FarDir::Top {
        return materials.top;
    }
    let surface = face.mix[0] as f32 / 255.0;
    let soil = face.mix[1] as f32 / 255.0;
    materials.side * surface + materials.soil * soil + materials.stone * (1.0 - surface - soil)
}

/// How much of its light a face keeps: `mesh.wgsl`'s ambient occlusion for
/// a face as open to the sky as level 0's faces over the same ground are on
/// average (`FarFace::open`), `ao_floor + (1 - ao_floor) * open`. The far
/// terrain's twin of `aggregate::ao_fraction` at a width of one block, with
/// the mean of the corners' levels in place of one corner's.
pub fn face_open(face: &FarFace, ao_floor: f32) -> f32 {
    ao_floor + (1.0 - ao_floor) * (face.open as f32 / 255.0)
}

/// One patch, as the renderer needs it: where it is, how coarse, and its
/// faces.
#[derive(Clone, Copy, Debug)]
pub struct FarPatch<'a> {
    /// The lowest `(x, z)` corner, in blocks -- the integer it is
    /// (`cubara_world::far::PatchKey::origin`), so the render anchor is
    /// subtracted from it exactly (issue #379).
    pub origin: [i64; 2],
    /// The width of one quad, in blocks.
    pub quad: f32,
    pub faces: &'a [FarFace],
    /// How far the walls on the patch's edge hang below the step they show,
    /// in blocks: deep enough to cover the gap to a neighbour of another
    /// level.
    pub skirt: f32,
}

/// Where a patch was put, to remove it by later.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FarSlot(u32);

/// What `far.wgsl` reads besides the vertices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FarParams {
    /// The box the voxel rings draw themselves, `(min, max)` in blocks --
    /// integers (chunk edges), subtracted from exactly like a tile origin.
    /// Nothing of the far terrain is drawn inside it. `None`: no hole.
    pub hole: Option<([i32; 3], [i32; 3])>,
    /// What the ground is made of: the mean colours of its faces
    /// ([`crate::materials::mean_color`]) and the soil's depth, for
    /// [`face_colour`].
    pub materials: Materials,
}

/// What the ground is made of, from the block textures: each face's texture
/// averaged ([`crate::materials::mean_color`]), falling back to plain
/// colours where a texture is missing, with the world's `soil_depth`
/// (`cubara_world::SOIL_DEPTH`, which this crate cannot name).
pub fn ground_materials(soil_depth: f32) -> Materials {
    let mean = |name: &str, fallback: Vec3| {
        crate::materials::mean_color(name).map_or(fallback, Vec3::from)
    };
    let p = PLACEHOLDER_MATERIALS;
    Materials {
        top: mean("grass_top", p.top),
        side: mean("grass_side", p.side),
        soil: mean("soil", p.soil),
        stone: mean("stone", p.stone),
        soil_depth,
    }
}

/// Plain grass-and-dirt colours, for before the real ones are set.
const PLACEHOLDER_MATERIALS: Materials = Materials {
    top: Vec3::new(0.2, 0.4, 0.15),
    side: Vec3::new(0.3, 0.3, 0.15),
    soil: Vec3::new(0.3, 0.2, 0.1),
    stone: Vec3::new(0.4, 0.4, 0.4),
    soil_depth: 3.0,
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PatchGpu {
    origin: [f32; 2],
    quad: f32,
    skirt: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsGpu {
    /// `w`: the height hidden vertices go to ([`hidden_height`]).
    hole_min: [f32; 4],
    hole_max: [f32; 4],
    top_color: [f32; 4],
    /// `w`: the soil's depth, in blocks.
    side_color: [f32; 4],
    soil_color: [f32; 4],
    stone_color: [f32; 4],
}

impl ParamsGpu {
    /// `p` with its hole made relative to `anchor` -- the space `far.wgsl`'s
    /// vertices are in.
    fn new(p: FarParams, anchor: Anchor, raised: bool) -> Self {
        // No hole: an empty box, min above max, which nothing is inside.
        let (min, max) = match p.hole {
            Some((min, max)) => (anchor::offset(min, anchor), anchor::offset(max, anchor)),
            None => (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
        };
        let m = p.materials;
        Self {
            // The two heights `hidden_height` chooses between.
            hole_min: [
                min.x,
                min.y,
                min.z,
                if raised { max.y + 1.0 } else { min.y - 1.0 },
            ],
            hole_max: [max.x, max.y, max.z, 0.0],
            top_color: m.top.extend(1.0).to_array(),
            side_color: m.side.extend(m.soil_depth).to_array(),
            soil_color: m.soil.extend(1.0).to_array(),
            stone_color: m.stone.extend(1.0).to_array(),
        }
    }
}

// ---------------------------------------------------------------------------
// The patch as the GPU holds it: shared corners (issue #422, §9.6 of
// `docs/PHASE3_ARCHITECTURE.md`).
// ---------------------------------------------------------------------------

/// Quads in a patch.
pub const FAR_CELLS: usize = FAR_QUADS * FAR_QUADS;

/// The vertices of a patch's quads: four a quad, the corners of its top, at
/// its height. **Every face of the patch is drawn from these**: a top is
/// its quad's four, and the riser between two neighbouring quads is the two
/// corners each has on the line between them -- so a riser needs no vertices
/// of its own, and a step of no height is a riser of no area that costs
/// indices only.
///
/// Why this and not four vertices a face: the far pass is bound by the
/// vertices the GPU shades, about 0.4 ns each whatever the shader does
/// (`docs/PHASE3_ARCHITECTURE.md` §9.6). As separate faces a quad cost about
/// nine of them (four a face, two facing walls, padding to the draw's size
/// class); as shared corners it costs four, and the measured cost of the pass
/// halved.
pub const FAR_CELL_VERTICES: usize = FAR_CELLS * 4;

/// The vertices under the walls on a patch's own edge, which hang a skirt
/// below the step they show and so have no quad to borrow a corner from: two
/// for every quad along each of the four edges.
pub const FAR_EDGE_VERTICES: usize = 4 * FAR_QUADS * 2;

/// Everything a patch draws from: [`FAR_CELL_VERTICES`] then
/// [`FAR_EDGE_VERTICES`]. Fits an index of 16 bits.
pub const FAR_PATCH_VERTICES: usize = FAR_CELL_VERTICES + FAR_EDGE_VERTICES;

/// Words one quad takes in the cells buffer: its height and the light its
/// top is open to (`far.wgsl`'s `cell_height`), then the three bytes -- the
/// share that is surface block, the share that is soil, how open -- of each
/// of the four walls it can stand behind.
pub const FAR_CELL_WORDS: usize = 4;

/// Words the edge walls' drops take: one half-word a quad along each edge.
const EDGE_WORDS: usize = 4 * FAR_QUADS / 2;

/// Where in a slot the edge walls' drops start, in words.
const EDGE_AT: usize = FAR_CELLS * FAR_CELL_WORDS;

/// Words one patch takes in the cells buffer.
pub const FAR_SLOT_WORDS: usize = EDGE_AT + EDGE_WORDS;

/// The edges of a patch, in the order of the edge vertices and of the
/// shader's `edge`: the wall faces `-x`, `+x`, `-z`, `+z`.
const EDGE_DIRS: [FarDir; 4] = [FarDir::NegX, FarDir::PosX, FarDir::NegZ, FarDir::PosZ];

/// **A patch as the GPU holds it**, from the faces the mesher made
/// (`cubara_world::far::mesh`): [`FAR_SLOT_WORDS`] words.
///
/// The faces are a description of what the patch looks like; the topology
/// the GPU draws from ([`patch_indices`]) is fixed, and what varies is a
/// height and the light of a few walls per quad. So this keeps, per quad, the
/// height of the top it is under and the material and openness of the four
/// walls it can stand behind, and per quad along the patch's edge how far its
/// edge wall hangs; the risers between neighbouring quads follow from the
/// heights. Panics if a quad has no top: the mesher gives every quad one.
pub fn dense_patch(faces: &[FarFace]) -> Vec<u32> {
    let mut words = vec![0u32; FAR_SLOT_WORDS];
    let mut covered = [false; FAR_CELLS];
    let cell_at = |i: usize, j: usize| (j * FAR_QUADS + i) * FAR_CELL_WORDS;
    for face in faces {
        let top = face.top.clamp(*FAR_TOP_RANGE.start(), *FAR_TOP_RANGE.end());
        let (i0, j0, i1, j1) = (
            face.i0 as usize,
            face.j0 as usize,
            face.i1 as usize,
            face.j1 as usize,
        );
        if face.dir == FarDir::Top {
            for j in j0..j1 {
                for i in i0..i1 {
                    covered[j * FAR_QUADS + i] = true;
                    let w = cell_at(i, j);
                    words[w] = (top - FAR_TOP_RANGE.start()) as u32 | (face.open as u32) << 18;
                }
            }
            continue;
        }
        // The quads along the wall, each with the wall it is behind: the
        // higher of the two it stands between (a wall on a patch's edge: the
        // one inside).
        let (line, along, by_x) = match face.dir {
            FarDir::PosX | FarDir::NegX => (i0, j0..j1, true),
            _ => (j0, i0..i1, false),
        };
        for k in along {
            let (i, j) = match (face.dir, by_x) {
                (FarDir::PosX, _) => (line - 1, k),
                (FarDir::NegX, _) => (line, k),
                (FarDir::PosZ, _) => (k, line - 1),
                _ => (k, line),
            };
            let w = cell_at(i, j);
            let byte = 3 * (face.dir as usize - 1);
            for (n, value) in [face.mix[0], face.mix[1], face.open].into_iter().enumerate() {
                let at = w + 1 + (byte + n) / 4;
                words[at] |= (value as u32) << (((byte + n) % 4) * 8);
            }
            if face.border {
                let edge = EDGE_DIRS.iter().position(|&d| d == face.dir).expect("a wall");
                let n = edge * FAR_QUADS + k;
                // `rise + 1`: nothing is zero.
                let drop = face.rise.min(FAR_MAX_RISE) + 1;
                words[EDGE_AT + n / 2] |= drop << ((n % 2) * 16);
            }
        }
    }
    assert!(
        covered.iter().all(|&c| c),
        "a patch's quads are all under a top"
    );
    words
}

/// How far the wall on `edge` beside quad `along` hangs below its top, less
/// the skirt: its step. `None`: the patch has no wall there.
pub fn edge_rise(words: &[u32], edge: usize, along: usize) -> Option<u32> {
    let n = edge * FAR_QUADS + along;
    let drop = (words[EDGE_AT + n / 2] >> ((n % 2) * 16)) & 0xFFFF;
    drop.checked_sub(1)
}

/// A quad's height in a patch's [`dense_patch`] words.
pub fn cell_height(words: &[u32], cell: usize) -> i32 {
    (words[cell * FAR_CELL_WORDS] & 0x3_FFFF) as i32 + FAR_TOP_RANGE.start()
}

/// Which corner of the patch's grid a vertex is, and whose, as `far.wgsl`'s
/// `vs_main` works it out from the vertex index: the first `4096` are the
/// corners of the quads (bit 0 of the corner is the high end of x, bit 1 of
/// z), then two a quad along each edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PatchVertex {
    /// The place on the patch's grid of quad corners, `0..=32`.
    pub at: [i32; 2],
    /// The quad it belongs to, `FAR_QUADS * j + i`.
    pub cell: usize,
    /// Which corner of that quad's footprint it is under or at.
    pub corner: u8,
    /// An edge vertex: the edge ([`EDGE_DIRS`]'s order) it hangs on.
    pub edge: Option<usize>,
}

impl PatchVertex {
    pub fn of(vertex: usize) -> Self {
        let q = FAR_QUADS;
        if vertex < FAR_CELL_VERTICES {
            let (cell, corner) = (vertex / 4, (vertex % 4) as u8);
            return Self {
                at: [
                    (cell % q) as i32 + (corner & 1) as i32,
                    (cell / q) as i32 + (corner >> 1) as i32,
                ],
                cell,
                corner,
                edge: None,
            };
        }
        let n = vertex - FAR_CELL_VERTICES;
        let (edge, along, end) = (n / (2 * q), (n / 2) % q, n % 2);
        let (cell, at, corner) = match edge {
            0 => (along * q, [0, (along + end) as i32], (end << 1) as u8),
            1 => (along * q + q - 1, [q as i32, (along + end) as i32], 1 | (end << 1) as u8),
            2 => (along, [(along + end) as i32, 0], end as u8),
            _ => ((q - 1) * q + along, [(along + end) as i32, q as i32], 2 | end as u8),
        };
        Self {
            at,
            cell,
            corner,
            edge: Some(edge),
        }
    }

    /// The height of this vertex, in blocks, in a patch of [`dense_patch`]
    /// `words` whose skirt is `skirt` deep: its quad's top, or for an edge
    /// vertex that less the step it shows and the skirt (or just the top,
    /// where there is no wall).
    pub fn y(&self, words: &[u32], skirt: f32) -> f32 {
        let top = cell_height(words, self.cell) as f32;
        let Some(edge) = self.edge else {
            return top;
        };
        let along = if edge < 2 {
            self.cell / FAR_QUADS
        } else {
            self.cell % FAR_QUADS
        };
        match edge_rise(words, edge, along) {
            Some(rise) => top - rise as f32 - skirt,
            None => top,
        }
    }
}

/// The index list every patch is drawn with -- the same one, whatever the
/// ground: the tops, and the risers between neighbouring quads, and the
/// walls on the edge.
///
/// - A top: `QUAD_INDICES` over its quad's four corners.
/// - The riser between a quad and the next along x: its two corners on the
///   line and the next one's two. Wound so that it faces `+x` when the first
///   quad is higher and `-x` when the next is -- the winding of a triangle
///   flips with the heights, and the far pipeline culls the back, so the one
///   index list shows each step from the side it faces and from no other.
///   Likewise along z.
/// - The wall on an edge: the quad's two corners on the edge, and the two
///   vertices under them.
///
/// **Every triangle starts at the vertex that says what it is**: a top and a
/// riser at a corner of the quad they are drawn from (a riser of the quad
/// before the line), an edge wall at one of its own vertices. `far.wgsl` reads
/// the triangle's material from that vertex (a flat varying, the first
/// vertex of the triangle) and its facing from the screen, so a quad's four
/// corners serve its top and the risers on four sides at once.
///
/// The quads are visited in bands of four rows, along a band and then the
/// next, so that the corners a riser reuses are still in the GPU's cache
/// (measured on the M3: any band from one to eight rows is as fast, and a
/// whole row is not).
pub fn patch_indices() -> Vec<u16> {
    let q = FAR_QUADS;
    const BAND: usize = 4;
    let v = |i: usize, j: usize, k: usize| ((j * q + i) * 4 + k) as u16;
    let e = |edge: usize, along: usize, end: usize| {
        (FAR_CELL_VERTICES + edge * 2 * q + along * 2 + end) as u16
    };
    let mut out = Vec::new();
    for band in (0..q).step_by(BAND) {
        for i in 0..q {
            for j in band..(band + BAND).min(q) {
                out.extend(QUAD_INDICES.map(|k| v(i, j, k as usize)));
                if i + 1 < q {
                    let (p1, p3, q0, q2) = (v(i, j, 1), v(i, j, 3), v(i + 1, j, 0), v(i + 1, j, 2));
                    out.extend([p1, p3, q0, p3, q2, q0]);
                }
                if j + 1 < q {
                    let (p2, p3, r0, r1) = (v(i, j, 2), v(i, j, 3), v(i, j + 1, 0), v(i, j + 1, 1));
                    out.extend([p2, r0, p3, p3, r0, r1]);
                }
            }
        }
    }
    for k in 0..q {
        // -x: the quad (0, k), corners 0 and 2.
        let (c0, c2) = (v(0, k, 0), v(0, k, 2));
        out.extend([e(0, k, 0), c2, c0, e(0, k, 1), c2, e(0, k, 0)]);
        // +x: the quad (31, k), corners 1 and 3.
        let (c1, c3) = (v(q - 1, k, 1), v(q - 1, k, 3));
        out.extend([e(1, k, 0), c1, c3, e(1, k, 1), e(1, k, 0), c3]);
        // -z: the quad (k, 0), corners 0 and 1.
        let (c0, c1) = (v(k, 0, 0), v(k, 0, 1));
        out.extend([e(2, k, 0), c0, c1, e(2, k, 1), e(2, k, 0), c1]);
        // +z: the quad (k, 31), corners 2 and 3.
        let (c2, c3) = (v(k, q - 1, 2), v(k, q - 1, 3));
        out.extend([e(3, k, 0), c3, c2, e(3, k, 0), e(3, k, 1), c3]);
    }
    out
}

/// The far terrain on the GPU: every held patch's quads in one storage
/// buffer ([`dense_patch`]), and the list of slots drawn this frame.
///
/// There is no vertex buffer. `far.wgsl` makes a vertex from its index and a
/// patch's quads from the instance; one draw issues every patch in view.
/// Both are sound because every draw here is direct
/// (`scripts/check-architecture.sh`, Rule 5).
pub struct FarTerrain {
    cells: wgpu::Buffer,
    patches: wgpu::Buffer,
    params: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    /// The slots drawn this frame, nearest first.
    slots: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    capacity: u32,
    /// Slots not holding a patch, handed out last-freed first.
    free: Vec<u32>,
    /// The voxels' hole, from the last [`set_params`](Self::set_params),
    /// relative to `anchor`.
    hole: Option<Aabb>,
    /// Each occupied slot's box relative to `anchor`, for culling; `None`
    /// where the slot is free.
    boxes: Vec<Option<Aabb>>,
    /// Each occupied slot's patch in world terms -- what `boxes` and the
    /// GPU records are re-derived from on a [`rebase`](Self::rebase).
    placed: Vec<Option<PlacedPatch>>,
    /// The last [`set_params`](Self::set_params), in world terms, for the
    /// same reason.
    params_world: FarParams,
    /// The render anchor (`crate::anchor`) the GPU records, `hole` and
    /// `boxes` are relative to.
    anchor: Anchor,
    /// How many slots [`prepare`](Self::prepare) found in view.
    drawn: u32,
    /// Whether the far terrain can be seen at all
    /// ([`set_shown`](Self::set_shown)).
    shown: bool,
    /// Whether hidden vertices were last sent above the box rather than
    /// under it ([`hidden_height`]): what `params` tells `far.wgsl`. Under
    /// it until [`prepare`](Self::prepare) says otherwise.
    raised: bool,
}

/// A patch as it was inserted, minus its quads (which the anchor does not
/// touch): enough to rewrite its `PatchGpu` record and its box, and to draw
/// it.
#[derive(Clone, Copy, Debug)]
struct PlacedPatch {
    origin: [i64; 2],
    quad: f32,
    skirt: f32,
    /// Lowest and highest point of any face, skirt included.
    lo: f32,
    hi: f32,
}

impl PlacedPatch {
    fn of(patch: &FarPatch<'_>) -> Self {
        let (lo, hi) = min_max(patch);
        Self {
            origin: patch.origin,
            quad: patch.quad,
            skirt: patch.skirt,
            lo,
            hi,
        }
    }

    /// The `(x, z)` origin relative to `anchor`: the one exact integer
    /// subtraction ([`anchor::offset_i64`]), then a small `f32`.
    fn origin_relative(&self, anchor: Anchor) -> [f32; 2] {
        let o = anchor::offset_i64([self.origin[0], 0, self.origin[1]], anchor);
        [o.x, o.z]
    }

    fn gpu(&self, anchor: Anchor) -> PatchGpu {
        PatchGpu {
            origin: self.origin_relative(anchor),
            quad: self.quad,
            skirt: self.skirt,
        }
    }

    /// The patch's box relative to `anchor`: its square of ground, from the
    /// bottom of its lowest wall to its highest top. The heights are
    /// absolute (a quad carries its own, and a rebase leaves them alone), so
    /// the anchor's `y` comes off here, as `far.wgsl` takes it off every
    /// vertex.
    fn aabb(&self, anchor: Anchor) -> Aabb {
        let [ox, oz] = self.origin_relative(anchor);
        let size = self.quad * FAR_QUADS as f32;
        let dy = anchor[1] as f32;
        Aabb {
            min: Vec3::new(ox, self.lo - dy, oz),
            max: Vec3::new(ox + size, self.hi - dy, oz + size),
        }
    }
}

/// Bytes a patch takes in the cells buffer.
const SLOT_BYTES: u64 = FAR_SLOT_WORDS as u64 * 4;

impl FarTerrain {
    /// Room for `capacity` patches, placed against the render anchor
    /// `anchor` until [`rebase`](Self::rebase) moves it.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, capacity: u32, anchor: Anchor) -> Self {
        let storage = |label, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(16),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let cells = storage("far-cells", capacity as u64 * SLOT_BYTES);
        let patches = storage(
            "far-patches",
            std::mem::size_of::<PatchGpu>() as u64 * capacity as u64,
        );
        let slots = storage("far-slots", 4 * capacity as u64);
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("far-params"),
            size: std::mem::size_of::<ParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_world = FarParams {
            hole: None,
            materials: PLACEHOLDER_MATERIALS,
        };
        queue.write_buffer(
            &params,
            0,
            bytemuck::bytes_of(&ParamsGpu::new(params_world, anchor, false)),
        );

        let list = patch_indices();
        let indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("far-patch-indices"),
            size: std::mem::size_of_val(list.as_slice()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&indices, 0, bytemuck::cast_slice(&list));

        let layout = far_bind_group_layout(device);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("far-bind-group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: patches.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cells.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: slots.as_entire_binding(),
                },
            ],
        });
        Self {
            cells,
            patches,
            params,
            indices,
            index_count: list.len() as u32,
            slots,
            bind_group,
            capacity,
            free: (0..capacity).rev().collect(),
            boxes: vec![None; capacity as usize],
            placed: vec![None; capacity as usize],
            params_world,
            anchor,
            drawn: 0,
            shown: true,
            raised: false,
            hole: None,
        }
    }

    /// A far terrain holding exactly `patches`, sized for them: what a
    /// one-shot caller wants (a screenshot, the bench, a golden test).
    pub fn holding(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        patches: &[FarPatch<'_>],
        anchor: Anchor,
    ) -> Self {
        let mut terrain = Self::new(device, queue, patches.len().max(1) as u32, anchor);
        for patch in patches {
            terrain
                .insert(queue, *patch)
                .expect("sized for these patches");
        }
        terrain
    }

    /// Whether any of the far terrain can be seen from where the eye is --
    /// the caller's to know (`cubara_world::visibility::Sight::open_to_sky`).
    /// While it cannot, [`prepare`](Self::prepare) culls nothing and draws
    /// nothing: an eye shut in underground pays for none of it (issue #424).
    /// The patches stay where they are, so it is back the frame this is set
    /// again.
    pub fn set_shown(&mut self, shown: bool) {
        self.shown = shown;
    }

    /// The render anchor the records, hole and boxes are relative to.
    pub fn anchor(&self) -> Anchor {
        self.anchor
    }

    /// Re-express every patch, the hole and their boxes against `anchor`
    /// (issue #379 decision 2). The quads are untouched -- they are whole
    /// quads from their patch's corner and absolute heights, and `far.wgsl`
    /// subtracts `Frame.anchor.y` from each; what is rewritten is one
    /// 16-byte record per slot -- all of them, in **one** `write_buffer` of
    /// `capacity * 16` bytes (64 KiB at the window's `FAR_PATCH_CAPACITY`),
    /// which costs less than a frame's own indirect upload. Not spread over
    /// frames: a patch drawn against the old anchor while the eye is
    /// measured from the new one would be 4,096 blocks off, the very thing
    /// the anchor exists to prevent; and not deferred to each patch's next
    /// upload, because a patch at the horizon may not be re-uploaded for
    /// minutes. The bounded single write is the cheapest form that is also
    /// correct on the rebase frame itself. A no-op for the anchor already in
    /// use.
    pub fn rebase(&mut self, queue: &wgpu::Queue, anchor: Anchor) {
        if anchor == self.anchor {
            return;
        }
        self.anchor = anchor;
        let records: Vec<PatchGpu> = self
            .placed
            .iter()
            .map(|p| p.map_or(bytemuck::Zeroable::zeroed(), |p| p.gpu(anchor)))
            .collect();
        queue.write_buffer(&self.patches, 0, bytemuck::cast_slice(&records));
        for (slot, placed) in self.placed.iter().enumerate() {
            self.boxes[slot] = placed.map(|p| p.aabb(anchor));
        }
        self.hole = self
            .params_world
            .hole
            .map(|(min, max)| hole_aabb(min, max, anchor));
        queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&ParamsGpu::new(self.params_world, anchor, self.raised)),
        );
    }

    /// Upload a patch. `None` when every slot is taken -- the caller decides
    /// what gives way, since it is the one that knows which patches matter.
    pub fn insert(&mut self, queue: &wgpu::Queue, patch: FarPatch<'_>) -> Option<FarSlot> {
        assert!(
            patch.faces.len() <= FAR_MAX_FACES,
            "a patch of {} faces",
            patch.faces.len()
        );
        let slot = self.free.pop()?;
        queue.write_buffer(
            &self.cells,
            slot as u64 * SLOT_BYTES,
            bytemuck::cast_slice(&dense_patch(patch.faces)),
        );
        let placed = PlacedPatch::of(&patch);
        queue.write_buffer(
            &self.patches,
            std::mem::size_of::<PatchGpu>() as u64 * slot as u64,
            bytemuck::bytes_of(&placed.gpu(self.anchor)),
        );
        self.boxes[slot as usize] = Some(placed.aabb(self.anchor));
        self.placed[slot as usize] = Some(placed);
        Some(FarSlot(slot))
    }

    /// Free a slot. The patch in it stops being drawn from the next
    /// [`prepare`](Self::prepare).
    pub fn remove(&mut self, slot: FarSlot) {
        if self.boxes[slot.0 as usize].take().is_some() {
            self.placed[slot.0 as usize] = None;
            self.free.push(slot.0);
        }
    }

    pub fn set_params(&mut self, queue: &wgpu::Queue, params: FarParams) {
        self.params_world = params;
        self.hole = params
            .hole
            .map(|(min, max)| hole_aabb(min, max, self.anchor));
        queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&ParamsGpu::new(params, self.anchor, self.raised)),
        );
    }

    /// Decide what is drawn this frame: every occupied slot whose box the
    /// frustum can see, nearest to `eye` first -- so near ground fills the
    /// depth buffer before the distant ground it hides is drawn. Returns how
    /// many patches. `frustum` and `eye` are anchor-relative, like the boxes
    /// (issue #379 decision 5). Patches wholly inside the voxels' hole are
    /// not drawn at all.
    pub fn prepare(&mut self, queue: &wgpu::Queue, frustum: &Frustum, eye: Vec3) -> u32 {
        // Where the hidden vertices go depends on the eye and on the ground
        // over the box, so it is settled here, per frame; it changes only
        // when the eye enters or leaves the box or the ground crosses its
        // floor, and is written only then.
        if let Some(hole) = self.hole.filter(|_| self.shown) {
            let lowest = lowest_over(&self.boxes, &hole);
            let raised = hidden_height(hole.min, hole.max, eye, lowest) > hole.max.y;
            if raised != self.raised {
                self.raised = raised;
                queue.write_buffer(
                    &self.params,
                    0,
                    bytemuck::bytes_of(&ParamsGpu::new(self.params_world, self.anchor, raised)),
                );
            }
        }
        let order = slots_to_draw(self.shown, &self.boxes, frustum, self.hole.as_ref(), eye);
        if !order.is_empty() {
            queue.write_buffer(&self.slots, 0, bytemuck::cast_slice(&order));
        }
        self.drawn = order.len() as u32;
        self.drawn
    }

    /// Patches held.
    pub fn len(&self) -> usize {
        self.boxes.iter().filter(|b| b.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Patches drawn this frame, from the last [`prepare`](Self::prepare).
    pub fn drawn(&self) -> u32 {
        self.drawn
    }

    /// Triangles issued this frame: every patch in view draws the whole of
    /// [`patch_indices`] -- the triangles that face away are culled after
    /// their vertices are shaded, which they share with the ones that do not.
    pub fn drawn_triangles(&self) -> u64 {
        self.drawn as u64 * (self.index_count / 3) as u64
    }

    /// What this frame's far terrain costs to issue: `(draws, vertices)` --
    /// one draw, and [`FAR_PATCH_VERTICES`] for every patch in it (the most
    /// the GPU shades: a vertex it still has from a neighbouring triangle is
    /// not shaded again).
    pub fn drawn_cost(&self) -> (u32, u64) {
        issued(self.drawn)
    }

    /// Draw this frame's patches. The pipeline and bind group 0 (the camera)
    /// are the caller's to set.
    pub(crate) fn encode(&self, pass: &mut wgpu::RenderPass<'_>) {
        if self.drawn == 0 {
            return;
        }
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint16);
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.draw_indexed(0..self.index_count, 0, 0..self.drawn);
    }
}

/// `(draws, vertices)` for `patches` patches in view.
pub fn issued(patches: u32) -> (u32, u64) {
    (
        u32::from(patches > 0),
        patches as u64 * FAR_PATCH_VERTICES as u64,
    )
}

/// `far.wgsl`'s `vs_main`, as far as a vertex's position before the crack
/// seal: where the vertex at grid point `at` and height `y` of a patch at
/// `origin` with `quad`-wide quads and a skirt `skirt` deep is put, given
/// the voxels' box `hole` (`None`: no box) and the height hidden vertices go
/// to ([`hidden_height`]). In world blocks; the shader does the same with the
/// render anchor taken off everything.
///
/// A vertex serves a quad's top and every riser beside it, so the rule is
/// the one a top's corner always had: strictly inside the box it is hung
/// from the box's side or hidden ([`join`]).
pub fn place(
    at: [i32; 2],
    corner: u8,
    y: f32,
    (origin, quad, skirt): ([f32; 2], f32, f32),
    hole: Option<(Vec3, Vec3)>,
    hidden: f32,
) -> Vec3 {
    let grid = |g: [i32; 2]| {
        [
            origin[0] + g[0] as f32 * quad,
            origin[1] + g[1] as f32 * quad,
        ]
    };
    let [x, z] = grid(at);
    let mut world = Vec3::new(x, y, z);
    let Some((min, max)) = hole else {
        return world;
    };
    if world.cmpgt(min).all() && world.cmplt(max).all() {
        let lo = [(x - min.x) / quad, (z - min.z) / quad];
        let hi = [(max.x - x) / quad, (max.z - z) / quad];
        match join(at, corner, lo, hi) {
            Join::Skirt(to) => {
                let [x, z] = grid(to);
                world = Vec3::new(x, y - skirt, z);
            }
            Join::Hidden => world.y = hidden,
        }
    }
    world
}

/// Where `far.wgsl` puts a vertex that lies strictly inside the voxels' box
/// -- the join of `docs/PROPOSAL_FAR_VIEW.md` §3.4, as a function so it can be
/// tested. `far.wgsl`'s `vs_main` is this, line for line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Join {
    /// The first ring inside: straight under the grid vertex `[i, j]` on the
    /// box's side, a skirt's depth down. With the vertices on the side
    /// itself that makes a skirt, which closes the seam to the voxels.
    Skirt([i32; 2]),
    /// Deeper: where the eye sees only its underside, which is culled --
    /// above the box for an eye inside it, under it otherwise.
    Hidden,
}

/// [`Join`] for the grid vertex `at` of a patch, `lo` quads inside the box's
/// low sides along x and z and `hi` quads inside its high ones; `corner` is
/// which corner of its quad it is (bit 0 the high end of x, bit 1 of z).
///
/// **A corner folds onto a side only along an axis where its quad touches
/// that side**: one quad inside it is the quad's own end towards the side
/// (the high end of x for the low side), and a quad's far end is hidden like
/// the rest. Shared corners are the reason: a corner serves the quad's top
/// and the risers on four sides, so one that the next quad in folds too would
/// stand a riser between two quads' skirts in the side's own plane, and a
/// hidden corner beside it would pull that riser up into the sky -- a spike
/// the eye inside the box sees the front of.
///
/// **Why not simply sink everything to the box's floor**, as this did until
/// issue #424: the triangles from the box's side down to the floor are a wall
/// a kilometre from the eye on every side, from the surface to a kilometre
/// below it, and its lit side faces inward. Above ground the voxels hide it.
/// From under the ground nothing does -- the owner's striped curtain. Raised
/// instead, the same triangles face away from an eye inside the box and
/// back-face culling drops them before a single fragment is shaded, which a
/// `discard` would not (`far.wgsl`).
pub fn join(at: [i32; 2], corner: u8, lo: [f32; 2], hi: [f32; 2]) -> Join {
    let quads = FAR_QUADS as i32;
    let mut to = at;
    let mut snapped = false;
    for a in 0..2 {
        // The step to the nearest side along this axis, in quads.
        let step = if hi[a] < lo[a] { hi[a] } else { -lo[a] }.round() as i32;
        let target = at[a] + step;
        let towards = (step < 0) == (corner >> a & 1 == 1);
        if step.abs() == 1 && towards && (0..=quads).contains(&target) {
            to[a] = target;
            snapped = true;
        }
    }
    if snapped {
        Join::Skirt(to)
    } else {
        Join::Hidden
    }
}

/// The height [`Join::Hidden`] vertices go to, for a box from `hole_min` to
/// `hole_max`, an eye at `eye`, and far terrain whose lowest point over the
/// box (skirts included) is `lowest`.
///
/// **Just above the box** when the eye is strictly inside it and all of that
/// ground is above its floor: the eye then sees only the underside of what
/// reaches up there, which is culled.
///
/// **Just under the box otherwise**, as it always was. For an eye outside
/// the box that is the side it cannot see. And where any ground lies under
/// the floor -- an eye a thousand blocks up, the floor cutting through the
/// hills -- that ground is not inside the box and stays where it is, so a
/// hidden vertex next to it has to stay next to it: raised, the triangle
/// between the two is a sheet two thousand blocks tall on the near side of
/// every hill (review of #427). Under the floor it is a step of a block. The
/// wall this leaves at the box's side is no taller than the ground stands
/// above the floor, and it is seen from above, where the voxels cover it.
pub fn hidden_height(hole_min: Vec3, hole_max: Vec3, eye: Vec3, lowest: f32) -> f32 {
    let eye_inside = eye.cmpgt(hole_min).all() && eye.cmplt(hole_max).all();
    if eye_inside && lowest > hole_min.y {
        hole_max.y + 1.0
    } else {
        hole_min.y - 1.0
    }
}

/// The lowest and highest point of a patch's faces: the bottom of its
/// deepest wall, skirt included, and its highest top.
fn min_max(patch: &FarPatch<'_>) -> (f32, f32) {
    patch
        .faces
        .iter()
        .flat_map(|f| f.corners())
        .map(|v| v.y(patch.skirt))
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), y| {
            (lo.min(y), hi.max(y))
        })
}

/// The voxels' hole as a culling box relative to `anchor`.
fn hole_aabb(min: [i32; 3], max: [i32; 3], anchor: Anchor) -> Aabb {
    Aabb {
        min: anchor::offset(min, anchor),
        max: anchor::offset(max, anchor),
    }
}

/// A patch's box: its square of ground, from the bottom of its skirt to its
/// highest vertex -- against the world origin, i.e. in world blocks.
#[cfg(test)]
fn patch_box(patch: &FarPatch<'_>) -> Aabb {
    PlacedPatch::of(patch).aabb(anchor::ORIGIN)
}

/// The lowest point, skirts included, of every patch standing over any of
/// the hole's ground -- [`hidden_height`]'s `lowest`. Infinity when none
/// does. A patch that only touches the hole's side counts too: that errs
/// towards "some ground is under the floor", which is the rule that is right
/// everywhere but underground.
pub fn lowest_over(boxes: &[Option<Aabb>], hole: &Aabb) -> f32 {
    boxes
        .iter()
        .flatten()
        .filter(|b| {
            b.min.x <= hole.max.x
                && b.max.x >= hole.min.x
                && b.min.z <= hole.max.z
                && b.max.z >= hole.min.z
        })
        .map(|b| b.min.y)
        .fold(f32::INFINITY, f32::min)
}

/// The slots drawn this frame, in the order they are drawn: none at all
/// while the far terrain is out of sight ([`FarTerrain::set_shown`]), else
/// what the frustum sees, nearest first, outside the hole.
fn slots_to_draw(
    shown: bool,
    boxes: &[Option<Aabb>],
    frustum: &Frustum,
    hole: Option<&Aabb>,
    eye: Vec3,
) -> Vec<u32> {
    if !shown {
        return Vec::new();
    }
    draw_order(boxes, visible_slots(boxes, frustum), hole, eye)
}

/// The occupied slots the frustum can see, in slot order.
fn visible_slots(boxes: &[Option<Aabb>], frustum: &Frustum) -> Vec<u32> {
    boxes
        .iter()
        .enumerate()
        .filter_map(|(slot, b)| {
            b.as_ref()
                .filter(|b| frustum.intersects_aabb(b))
                .map(|_| slot as u32)
        })
        .collect()
}

/// The visible slots to draw, nearest to `eye` first -- so near ground fills
/// the depth buffer before the distant ground it hides -- leaving out any
/// wholly inside the voxels' hole.
fn draw_order(
    boxes: &[Option<Aabb>],
    visible: Vec<u32>,
    hole: Option<&Aabb>,
    eye: Vec3,
) -> Vec<u32> {
    let bx = |slot: &u32| {
        boxes[*slot as usize]
            .as_ref()
            .expect("a visible slot is occupied")
    };
    let mut order: Vec<u32> = visible
        .into_iter()
        .filter(|slot| !hole.is_some_and(|h| contains(h, bx(slot))))
        .collect();
    let distance = |slot: &u32| {
        let b = bx(slot);
        (eye.clamp(b.min, b.max) - eye).length_squared()
    };
    order.sort_by(|a, b| distance(a).total_cmp(&distance(b)));
    order
}

fn contains(outer: &Aabb, inner: &Aabb) -> bool {
    outer.min.cmple(inner.min).all() && inner.max.cmple(outer.max).all()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cheap, fixed stream of numbers for the property tests.
    fn stream(seed: u64) -> impl FnMut(u32) -> u32 {
        let mut s = seed | 1;
        move |below: u32| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            ((s.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 33) % below as u64) as u32
        }
    }

    /// A face of each kind somewhere in a patch, from `next`.
    fn some_face(next: &mut impl FnMut(u32) -> u32) -> FarFace {
        let dir = FarDir::ALL[next(5) as usize];
        let (i0, j0) = (next(32) as u8, next(32) as u8);
        let (w, d) = (
            1 + next(32 - i0 as u32) as u8,
            1 + next(32 - j0 as u32) as u8,
        );
        let (i1, j1) = match dir {
            FarDir::Top => (i0 + w, j0 + d),
            FarDir::PosX | FarDir::NegX => (i0, j0 + d),
            FarDir::PosZ | FarDir::NegZ => (i0 + w, j0),
        };
        let wall = dir != FarDir::Top;
        let mix = if wall {
            [next(128) as u8, next(128) as u8]
        } else {
            [0, 0]
        };
        FarFace {
            dir,
            i0,
            j0,
            i1,
            j1,
            top: next(600) as i32 - 100,
            rise: if wall { 1 + next(40) } else { 0 },
            border: wall && next(4) == 0,
            mix,
            open: next(256) as u8,
        }
    }

    /// The shader's constants and this module's must agree, or a vertex
    /// reads another patch's quads and the join steps off the patch.
    #[test]
    fn the_shader_agrees_on_the_patch_layout() {
        let wgsl = crate::haze::FAR_WGSL;
        for line in [
            format!("const QUADS: i32 = {FAR_QUADS};"),
            format!("const QUADS_U: u32 = {FAR_QUADS}u;"),
            format!("const CELL_VERTICES: u32 = {FAR_CELL_VERTICES}u;"),
            format!("const CELL_WORDS: u32 = {FAR_CELL_WORDS}u;"),
            format!("const SLOT_WORDS: u32 = {FAR_SLOT_WORDS}u;"),
            format!("const EDGE_AT: u32 = {EDGE_AT}u;"),
        ] {
            assert!(wgsl.contains(&line), "far.wgsl no longer has: {line}");
        }
        // Two vertices a quad along each edge, `n / 64` the edge.
        assert_eq!(2 * FAR_QUADS, 64);
        // An index of 16 bits reaches every vertex of a patch.
        assert!(FAR_PATCH_VERTICES <= u16::MAX as usize + 1);
        // A slot is whole words of what it holds: 16 bytes a quad and 64
        // words of edge drops.
        assert_eq!(FAR_SLOT_WORDS, 4 * FAR_CELLS + 4 * FAR_QUADS / 2);
    }

    /// A patch of unit faces over `height`, as the mesher makes them for a
    /// patch of one level: a top on every quad, a wall of exactly the step
    /// between every two neighbours (standing on the higher, facing the
    /// lower), and a wall on every quad along the patch's edge hanging
    /// `edge_step` -- more where the ground inside is higher. What a wall is
    /// made of, and how open each face is, depends on where it is, so that a
    /// face read from the wrong quad shows.
    fn unit_faces(height: &dyn Fn(i32, i32) -> i32, edge_step: u32) -> Vec<FarFace> {
        let n = FAR_QUADS as i32;
        let mut faces = Vec::new();
        for j in 0..n {
            for i in 0..n {
                faces.push(FarFace {
                    open: (50 + (i * 5 + j * 3) % 200) as u8,
                    ..unit_top(i as u8, j as u8, height(i, j))
                });
            }
        }
        let wall = |dir: FarDir, line: i32, k: i32, top: i32, rise: u32, border: bool| {
            let (i0, j0, i1, j1) = match dir {
                FarDir::PosX | FarDir::NegX => (line, k, line, k + 1),
                _ => (k, line, k + 1, line),
            };
            FarFace {
                dir,
                i0: i0 as u8,
                j0: j0 as u8,
                i1: i1 as u8,
                j1: j1 as u8,
                top,
                rise,
                border,
                mix: [
                    (10 + (line * 7 + k * 3 + dir as i32 * 11) % 100) as u8,
                    (20 + (line * 5 + k * 13 + dir as i32 * 3) % 100) as u8,
                ],
                open: (30 + (line * 11 + k * 7 + dir as i32 * 5) % 200) as u8,
            }
        };
        for k in 0..n {
            for line in 1..n {
                let (lo, hi) = (height(line - 1, k), height(line, k));
                if lo > hi {
                    faces.push(wall(FarDir::PosX, line, k, lo, (lo - hi) as u32, false));
                } else if hi > lo {
                    faces.push(wall(FarDir::NegX, line, k, hi, (hi - lo) as u32, false));
                }
                let (lo, hi) = (height(k, line - 1), height(k, line));
                if lo > hi {
                    faces.push(wall(FarDir::PosZ, line, k, lo, (lo - hi) as u32, false));
                } else if hi > lo {
                    faces.push(wall(FarDir::NegZ, line, k, hi, (hi - lo) as u32, false));
                }
            }
            for (dir, line, i, j) in [
                (FarDir::NegX, 0, 0, k),
                (FarDir::PosX, n, n - 1, k),
                (FarDir::NegZ, 0, k, 0),
                (FarDir::PosZ, n, k, n - 1),
            ] {
                let top = height(i, j);
                // Each edge a different step, so one read from the wrong
                // edge shows.
                let rise = edge_step + (dir as u32) * 3 + (k as u32 % 2);
                faces.push(wall(dir, line, k, top, rise, true));
            }
        }
        faces
    }

    fn rolling(i: i32, j: i32) -> i32 {
        60 + 9 * ((i * 3 + j * 5) % 7) + ((i * j) % 5)
    }

    /// A patch survives the words it travels in: every quad's height and
    /// light, every wall's material and openness, every edge wall's step.
    #[test]
    fn a_patch_goes_up_as_its_quads() {
        let faces = unit_faces(&rolling, 2);
        let words = dense_patch(&faces);
        assert_eq!(words.len(), FAR_SLOT_WORDS);
        for f in &faces {
            let cells = |i: usize, j: usize| (j * FAR_QUADS + i, (j * FAR_QUADS + i) * FAR_CELL_WORDS);
            match f.dir {
                FarDir::Top => {
                    let (cell, w) = cells(f.i0 as usize, f.j0 as usize);
                    assert_eq!(cell_height(&words, cell), f.top);
                    assert_eq!((words[w] >> 18) & 0xFF, f.open as u32);
                }
                dir => {
                    let (i, j) = match dir {
                        FarDir::PosX => (f.i0 as usize - 1, f.j0 as usize),
                        FarDir::NegX => (f.i0 as usize, f.j0 as usize),
                        FarDir::PosZ => (f.i0 as usize, f.j0 as usize - 1),
                        _ => (f.i0 as usize, f.j0 as usize),
                    };
                    let (cell, w) = cells(i, j);
                    // This wall was made on the higher quad's height.
                    assert_eq!(cell_height(&words, cell), f.top, "{f:?}");
                    let byte = |n: usize| (words[w + 1 + n / 4] >> ((n % 4) * 8)) & 0xFF;
                    let at = 3 * (dir as usize - 1);
                    assert_eq!(
                        [byte(at), byte(at + 1), byte(at + 2)],
                        [f.mix[0] as u32, f.mix[1] as u32, f.open as u32],
                        "{f:?}"
                    );
                    if f.border {
                        let edge = EDGE_DIRS.iter().position(|&d| d == dir).unwrap();
                        let along = if edge < 2 { j } else { i };
                        assert_eq!(edge_rise(&words, edge, along), Some(f.rise), "{f:?}");
                    }
                }
            }
        }
        // A merged top is every quad it covers.
        let merged = [FarFace {
            i1: FAR_QUADS as u8,
            j1: FAR_QUADS as u8,
            ..unit_top(0, 0, 77)
        }];
        let words = dense_patch(&merged);
        assert!((0..FAR_CELLS).all(|c| cell_height(&words, c) == 77));
        assert_eq!(edge_rise(&words, 2, 5), None, "no wall, no step");
    }

    /// The mesher gives every quad a top, and the GPU's form relies on it: a
    /// patch that leaves one out is refused rather than drawn with a hole.
    #[test]
    #[should_panic(expected = "all under a top")]
    fn a_patch_with_a_quad_without_a_top_is_not_one() {
        dense_patch(&[unit_top(0, 0, 10)]);
    }

    /// The faces of a patch drawn with [`patch_indices`], as unit pieces:
    /// from the triangles that have area, and whose back is not the side they
    /// are seen from. Each piece is `(direction, the quad it stands on or
    /// belongs to along x, along z, its top, its height)`, with the area of
    /// triangles that make it.
    fn drawn_faces(
        words: &[u32],
        skirt: f32,
        quad: f32,
    ) -> std::collections::BTreeMap<(usize, i32, i32, i32, i32), f32> {
        let positions: Vec<Vec3> = (0..FAR_PATCH_VERTICES)
            .map(|v| {
                let pv = PatchVertex::of(v);
                place(pv.at, pv.corner, pv.y(words, skirt), ([0.0, 0.0], quad, skirt), None, 0.0)
            })
            .collect();
        let mut pieces = std::collections::BTreeMap::new();
        for t in patch_indices().chunks(3) {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| positions[i as usize]);
            let cross = (b - a).cross(c - a);
            let area = cross.length() / 2.0;
            if area < 1e-4 {
                continue;
            }
            let n = cross.normalize();
            let dir = FarDir::ALL
                .iter()
                .position(|d| d.normal().distance(n) < 1e-4)
                .unwrap_or_else(|| panic!("a triangle that faces no way: {a} {b} {c}"));
            let (lo, hi) = (a.y.min(b.y).min(c.y), a.y.max(b.y).max(c.y));
            // The grid line or quad it stands on, along x and z.
            let centre = (a + b + c) / 3.0;
            let key = |v: f32| (v / quad).floor() as i32;
            let (gx, gz) = match FarDir::ALL[dir] {
                FarDir::Top => (key(centre.x), key(centre.z)),
                FarDir::PosX | FarDir::NegX => ((centre.x / quad).round() as i32, key(centre.z)),
                _ => (key(centre.x), (centre.z / quad).round() as i32),
            };
            *pieces
                .entry((dir, gx, gz, hi as i32, (hi - lo).round() as i32))
                .or_insert(0.0) += area;
        }
        pieces
    }

    /// **The shared corners draw the faces the mesher makes**: the same tops
    /// and the same walls, as tall as the mesher says, on every quad and
    /// every edge -- and nothing else with an area, and no face twice. Seen
    /// from the side each faces: the winding of a riser flips with which of
    /// the two quads it stands between is higher.
    #[test]
    fn the_shared_corners_draw_the_faces_the_mesher_makes() {
        let (quad, skirt) = (4.0, 6.0);
        let faces = unit_faces(&rolling, 2);
        let words = dense_patch(&faces);
        let drawn = drawn_faces(&words, skirt, quad);
        let mut expected = std::collections::BTreeMap::new();
        for f in &faces {
            let dir = f.dir as usize;
            let (gx, gz) = match f.dir {
                FarDir::Top => (f.i0 as i32, f.j0 as i32),
                FarDir::PosX | FarDir::NegX => (f.i0 as i32, f.j0 as i32),
                _ => (f.i0 as i32, f.j0 as i32),
            };
            let height = if f.dir == FarDir::Top {
                0
            } else {
                f.rise as i32 + if f.border { skirt as i32 } else { 0 }
            };
            let area = if f.dir == FarDir::Top {
                quad * quad
            } else {
                quad * height as f32
            };
            *expected
                .entry((dir, gx, gz, f.top, height))
                .or_insert(0.0) += area;
        }
        let missing: Vec<_> = expected.keys().filter(|k| !drawn.contains_key(k)).collect();
        let extra: Vec<_> = drawn.keys().filter(|k| !expected.contains_key(k)).collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "{} faces are not drawn, e.g. {:?}; {} are drawn that are not faces, e.g. {:?}",
            missing.len(),
            missing.first(),
            extra.len(),
            extra.first()
        );
        for (k, area) in &expected {
            assert!(
                (drawn[k] - area).abs() < 1e-2 * area.max(1.0),
                "{k:?}: {} drawn, {area} wanted",
                drawn[k]
            );
        }
        assert!(expected.len() > 3000, "{} faces proves little", expected.len());
    }

    /// The owner a triangle's material is read from: the shader's rule
    /// (`fs_main`), as a function. `first` is the triangle's first vertex,
    /// `dir` the way the face looks.
    fn owner(first: usize, dir: FarDir) -> usize {
        let pv = PatchVertex::of(first);
        if pv.edge.is_some() {
            return pv.cell;
        }
        match dir {
            FarDir::NegX => pv.cell + 1,
            FarDir::NegZ => pv.cell + FAR_QUADS,
            _ => pv.cell,
        }
    }

    /// **Every face is the material of the quad it belongs to**: for every
    /// triangle that is seen, the quad the shader reads -- from the
    /// triangle's first vertex and which way it looks -- is the one the
    /// mesher made the face for (the higher of the two a riser stands
    /// between, the inner of an edge wall), and its record holds the
    /// face's own light.
    #[test]
    fn a_triangle_reads_the_quad_its_face_belongs_to() {
        let (quad, skirt) = (4.0, 6.0);
        let faces = unit_faces(&rolling, 2);
        let words = dense_patch(&faces);
        let positions: Vec<Vec3> = (0..FAR_PATCH_VERTICES)
            .map(|v| {
                let pv = PatchVertex::of(v);
                place(pv.at, pv.corner, pv.y(&words, skirt), ([0.0, 0.0], quad, skirt), None, 0.0)
            })
            .collect();
        let mut checked = 0;
        for t in patch_indices().chunks(3) {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| positions[i as usize]);
            let cross = (b - a).cross(c - a);
            if cross.length() < 1e-3 {
                continue;
            }
            let n = cross.normalize();
            let dir = FarDir::ALL
                .iter()
                .copied()
                .find(|d| d.normal().distance(n) < 1e-4)
                .unwrap();
            let cell = owner(t[0] as usize, dir);
            let w = cell * FAR_CELL_WORDS;
            let centre = (a + b + c) / 3.0;
            let (ci, cj) = ((cell % FAR_QUADS) as f32, (cell / FAR_QUADS) as f32);
            // The owner is a quad whose own edge the face is on, or whose
            // top it is.
            let (x, z) = (centre.x / quad, centre.z / quad);
            match dir {
                FarDir::Top => assert!(
                    (x - ci - 0.5).abs() <= 0.5 && (z - cj - 0.5).abs() <= 0.5,
                    "top {t:?}"
                ),
                FarDir::PosX => assert!((x - ci - 1.0).abs() < 1e-4 && (z - cj - 0.5).abs() <= 0.5),
                FarDir::NegX => assert!((x - ci).abs() < 1e-4 && (z - cj - 0.5).abs() <= 0.5),
                FarDir::PosZ => assert!((z - cj - 1.0).abs() < 1e-4 && (x - ci - 0.5).abs() <= 0.5),
                FarDir::NegZ => assert!((z - cj).abs() < 1e-4 && (x - ci - 0.5).abs() <= 0.5),
            }
            // And a wall's quad is the higher: its own top is the wall's.
            if dir != FarDir::Top {
                let top = a.y.max(b.y).max(c.y);
                assert_eq!(cell_height(&words, cell) as f32, top, "{dir:?} {t:?}");
                let byte = |n: usize| (words[w + 1 + n / 4] >> ((n % 4) * 8)) & 0xFF;
                let at = 3 * (dir as usize - 1);
                assert_ne!(byte(at + 2), 0, "{dir:?} of quad {cell} has no light");
            }
            checked += 1;
        }
        assert!(checked > 6_000, "only {checked} triangles");
    }

    /// The shader's vertex steps are the ones [`PatchVertex`] and
    /// [`dense_patch`] have, line for line: a pin on its text, as a shader
    /// cannot be called from here (`golden.rs` renders it).
    #[test]
    fn the_shader_makes_the_vertices_this_module_does() {
        // Every vertex, as the shader's branches say.
        for v in 0..FAR_PATCH_VERTICES {
            let pv = PatchVertex::of(v);
            if v < FAR_CELL_VERTICES {
                assert_eq!(pv.cell, v / 4);
                assert_eq!(pv.corner as usize, v % 4);
                assert_eq!(
                    pv.at,
                    [
                        (pv.cell % FAR_QUADS) as i32 + (v % 2) as i32,
                        (pv.cell / FAR_QUADS) as i32 + (v % 4 / 2) as i32
                    ]
                );
            } else {
                // An edge vertex is on the edge it names, at a corner of
                // its quad on that edge.
                let q = FAR_QUADS as i32;
                let edge = pv.edge.unwrap();
                match edge {
                    0 => assert_eq!(pv.at[0], 0),
                    1 => assert_eq!(pv.at[0], q),
                    2 => assert_eq!(pv.at[1], 0),
                    _ => assert_eq!(pv.at[1], q),
                }
                let corner = PatchVertex::of(pv.cell * 4 + pv.corner as usize);
                assert_eq!(corner.at, pv.at, "{v} is under a corner of its quad");
            }
        }
        let wgsl = crate::haze::FAR_WGSL;
        for line in [
            "let slot = slots[instance];",
            "let base = slot * SLOT_WORDS;",
            "cell = vertex / 4u;",
            "corner = vertex % 4u;",
            "i = i32(cell % QUADS_U + (corner & 1u));",
            "j = i32(cell / QUADS_U + (corner >> 1u));",
            "edge = n / 64u;",
            "along = (n / 2u) % QUADS_U;",
            "let end = n % 2u;",
            "cell = along * QUADS_U;",
            "corner = end << 1u;",
            "cell = along * QUADS_U + QUADS_U - 1u;",
            "corner = 1u | (end << 1u);",
            "cell = along;",
            "corner = end;",
            "cell = (QUADS_U - 1u) * QUADS_U + along;",
            "corner = 2u | end;",
            "let record = base + cell * CELL_WORDS;",
            // `cell_height`.
            "let top = f32(i32(cells[record] & 0x3FFFFu) - 131072) - frame.anchor.y;",
            // `edge_rise`, and `PatchVertex::y`.
            "let n = edge * QUADS_U + along;",
            "let drop = (cells[base + EDGE_AT + n / 2u] >> ((n % 2u) * 16u)) & 0xFFFFu;",
            "y = top - f32(drop - 1u) - tile.skirt;",
            // `dense_patch`'s bytes and the openness of a top.
            "return f32((cells[record + 1u + n / 4u] >> ((n % 4u) * 8u)) & 0xFFu) / 255.0;",
            "var open = f32((cells[record] >> 18u) & 0xFFu) / 255.0;",
            "let byte = 3u * (dir - 1u);",
            // `owner`.
            "var record = in.record & ~EDGE_BIT;",
            "if (in.record & EDGE_BIT) == 0u {",
            "if dir == DIR_NEG_X {",
            "record += CELL_WORDS;",
            "} else if dir == DIR_NEG_Z {",
            "record += QUADS_U * CELL_WORDS;",
            "out.record = record | select(0u, EDGE_BIT, edge_wall);",
            // The first vertex's, for the triangle.
            "@location(2) @interpolate(flat, first) record: u32,",
            // `FarVertex::outward`, for the quad's corners.
            "let outward = vec3<f32>(su, 0.0, sv);",
        ] {
            assert!(wgsl.contains(line), "far.wgsl no longer has: {line}");
        }
        // `FarDir`'s order is the shader's.
        for (dir, name) in [
            (FarDir::Top, "DIR_TOP"),
            (FarDir::PosX, "DIR_POS_X"),
            (FarDir::NegX, "DIR_NEG_X"),
            (FarDir::PosZ, "DIR_POS_Z"),
            (FarDir::NegZ, "DIR_NEG_Z"),
        ] {
            let line = format!("const {name}: u32 = {}u;", dir as u32);
            assert!(wgsl.contains(&line), "far.wgsl no longer has: {line}");
        }
        let normals: String = FarDir::ALL
            .iter()
            .map(|d| {
                let n = d.normal();
                format!("    vec3<f32>({:.1}, {:.1}, {:.1}),\n", n.x, n.y, n.z)
            })
            .collect();
        // Five lines in a row, so the line endings count: a checkout on
        // Windows has `\r\n` in the shader, and this was red there alone.
        assert!(
            wgsl.replace("\r\n", "\n").contains(&normals),
            "FACE_NORMALS is not FarDir's order"
        );
    }

    /// The index list is the same for every patch, reaches every vertex, and
    /// starts every triangle at the vertex that names the quad it is drawn
    /// from.
    #[test]
    fn the_index_list_is_one_for_every_patch() {
        let indices = patch_indices();
        let q = FAR_QUADS;
        assert_eq!(
            indices.len(),
            6 * q * q + 2 * 6 * q * (q - 1) + 4 * 6 * q,
            "a top a quad, a riser between every two, two triangles an edge quad"
        );
        let mut seen = vec![false; FAR_PATCH_VERTICES];
        for &i in &indices {
            seen[i as usize] = true;
        }
        assert!(seen.iter().all(|&s| s), "a vertex nothing draws");
        // An edge wall's triangles start on one of its own vertices; a top
        // and a riser on the quad's own corner.
        for t in indices.chunks(3) {
            let edge = t.iter().any(|&i| i as usize >= FAR_CELL_VERTICES);
            assert_eq!(
                edge,
                t[0] as usize >= FAR_CELL_VERTICES,
                "a wall on the edge starts on its own vertex: {t:?}"
            );
        }
    }

    /// Every face is drawn counter-clockwise as seen from the side it looks
    /// out of -- the side the pipeline does not cull -- and its corners are
    /// where the face is: a top at its height over its rectangle, a wall
    /// from its top down exactly its rise, and a skirt further on the edge.
    #[test]
    fn a_face_winds_towards_its_normal_and_spans_what_it_says() {
        let mut next = stream(0x51DE);
        let patch = ([64.0, -128.0], 4.0, 9.0);
        for _ in 0..2_000 {
            let face = some_face(&mut next);
            let p = face
                .corners()
                .map(|v| place(v.at, v.corner, v.y(patch.2), patch, None, 0.0));
            for [a, b, c] in [[0, 2, 1], [1, 2, 3]] {
                let normal = (p[b] - p[a]).cross(p[c] - p[a]);
                assert!(
                    normal.normalize().distance(face.dir.normal()) < 1e-5,
                    "{face:?}: triangle {a}{b}{c} looks along {normal}"
                );
            }
            let (lo, hi) = p.iter().fold(
                (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
                |(lo, hi), &v| (lo.min(v), hi.max(v)),
            );
            let x = |i: u8| 64.0 + i as f32 * 4.0;
            let z = |j: u8| -128.0 + j as f32 * 4.0;
            let depth = face.rise as f32 + if face.border { 9.0 } else { 0.0 };
            assert_eq!(
                lo,
                Vec3::new(x(face.i0), face.top as f32 - depth, z(face.j0))
            );
            assert_eq!(hi, Vec3::new(x(face.i1), face.top as f32, z(face.j1)));
            // And each corner is pushed out of the face, not into it.
            let centre = (lo + hi) / 2.0;
            for (v, at) in face.corners().iter().zip(p) {
                let out = v.outward();
                assert_eq!(out.dot(face.dir.normal()), 0.0, "{face:?}: out of plane");
                for axis in 0..3 {
                    if out[axis] != 0.0 {
                        assert_eq!(out[axis], (at[axis] - centre[axis]).signum(), "{face:?}");
                    }
                }
            }
        }
    }

    /// **The far terrain's face light is the voxels'** (issue #422 decision
    /// 2): `mesh.wgsl` and `far.wgsl` are both built with `face.wgsl` in
    /// front, both call its `face_brightness`, and neither has a diffuse or
    /// ambient term of its own left to drift.
    #[test]
    fn the_far_terrain_and_the_voxels_are_lit_by_one_function() {
        let face = crate::haze::FACE_WGSL;
        assert!(face.contains("fn face_brightness("));
        assert!(face.contains("let diffuse = max(dot(n, sun_dir), 0.0) * diffuse_weight;"));
        assert!(face.contains("let ambient = mix(ambient_low, ambient_high, n.y * 0.5 + 0.5);"));
        assert!(face.contains("return ambient + diffuse;"));
        let own = |wgsl: &str| wgsl.replacen(face, "", 1);
        for (name, wgsl) in [
            ("mesh.wgsl", crate::haze::MESH_WGSL),
            ("far.wgsl", crate::haze::FAR_WGSL),
        ] {
            assert!(wgsl.contains(face), "{name} is not built with face.wgsl");
            let own = own(wgsl);
            assert!(own.contains("face_brightness("), "{name} does not call it");
            for term in ["max(dot(n", "n.y * 0.5 + 0.5", "CRACK_SEAL: f32"] {
                assert!(!own.contains(term), "{name} has its own `{term}`");
            }
            assert!(
                own.contains("CRACK_SEAL"),
                "{name} does not seal its cracks"
            );
        }
        // The terms: the voxels' are pipeline constants, the far terrain's
        // the frame's -- the same `Lighting` fields, in the same places.
        assert!(crate::haze::MESH_WGSL
            .contains("face_brightness(n, sun_dir, ambient_low, ambient_high, diffuse_weight)"));
        for term in [
            "frame.sun_dir.xyz,",
            "frame.ambient.x,",
            "frame.ambient.y,",
            "frame.sun_color.w,",
        ] {
            assert!(crate::haze::FAR_WGSL.contains(term), "far.wgsl: {term}");
        }
    }

    /// A top is the surface block's top; a wall is the mix it carries, and
    /// either keeps as much light as it is open.
    #[test]
    fn a_face_wears_the_mix_it_carries() {
        let m = Materials {
            top: Vec3::new(0.1, 0.8, 0.1),
            side: Vec3::new(0.3, 0.5, 0.1),
            soil: Vec3::new(0.4, 0.3, 0.2),
            stone: Vec3::splat(0.5),
            soil_depth: 3.0,
        };
        let top = unit_top(3, 4, 20);
        assert_eq!(face_colour(&m, &top), m.top);
        for dir in [FarDir::PosX, FarDir::NegX, FarDir::PosZ, FarDir::NegZ] {
            let wall = |mix: [u8; 2]| FarFace {
                dir,
                rise: 9,
                mix,
                ..top
            };
            assert_eq!(face_colour(&m, &wall([255, 0])), m.side);
            assert_eq!(face_colour(&m, &wall([0, 255])), m.soil);
            assert_eq!(face_colour(&m, &wall([0, 0])), m.stone);
            // A fifth surface, two fifths soil, the rest stone -- whatever
            // the wall's own height.
            let mixed = m.side * 0.2 + m.soil * 0.4 + m.stone * 0.4;
            assert!(face_colour(&m, &wall([51, 102])).distance(mixed) < 1e-6);
            let tall = FarFace {
                rise: 900,
                ..wall([51, 102])
            };
            assert!(face_colour(&m, &tall).distance(mixed) < 1e-6);
        }
        // Wide open keeps all its light; shut in keeps the floor.
        assert_eq!(face_open(&top, 0.4), 1.0);
        assert_eq!(face_open(&FarFace { open: 0, ..top }, 0.4), 0.4);
        let half = face_open(&FarFace { open: 51, ..top }, 0.4);
        assert!((half - 0.52).abs() < 1e-6, "{half}");
    }

    fn unit_top(i: u8, j: u8, top: i32) -> FarFace {
        FarFace {
            dir: FarDir::Top,
            i0: i,
            j0: j,
            i1: i + 1,
            j1: j + 1,
            top,
            rise: 0,
            border: false,
            mix: [0, 0],
            open: 255,
        }
    }

    #[test]
    fn a_patch_box_spans_its_ground_and_its_skirt() {
        let faces = [
            unit_top(0, 0, 10),
            unit_top(5, 9, 16),
            FarFace {
                dir: FarDir::NegX,
                i0: 0,
                j0: 0,
                i1: 0,
                j1: 1,
                top: 10,
                rise: 0,
                border: true,
                mix: [0, 0],
                open: 255,
            },
        ];
        let b = patch_box(&FarPatch {
            origin: [64, -128],
            quad: 4.0,
            faces: &faces,
            skirt: 8.0,
        });
        assert_eq!(b.min, Vec3::new(64.0, 2.0, -128.0));
        assert_eq!(b.max, Vec3::new(64.0 + 128.0, 16.0, 0.0));
    }

    /// Issue #379: a patch's record and box against a shifted anchor are the
    /// record and box of the unshifted patch against the origin -- the exact
    /// integer subtraction the anchor relies on, at an offset (2^20) where
    /// an `f32` origin would already have no fractional bits to spare.
    #[test]
    fn a_patch_against_the_shifted_anchor_is_the_patch_at_the_origin() {
        let faces = [unit_top(0, 0, 10), unit_top(5, 9, 16)];
        let shift = 1_048_576i64;
        let here = PlacedPatch::of(&FarPatch {
            origin: [64, -128],
            quad: 4.0,
            faces: &faces,
            skirt: 8.0,
        });
        let there = PlacedPatch::of(&FarPatch {
            origin: [64 + shift, -128 - shift],
            quad: 4.0,
            faces: &faces,
            skirt: 8.0,
        });
        let anchor = [shift as i32, 0, -shift as i32];
        assert_eq!(there.gpu(anchor).origin, here.gpu(anchor::ORIGIN).origin);
        let (a, b) = (there.aabb(anchor), here.aabb(anchor::ORIGIN));
        assert_eq!((a.min, a.max), (b.min, b.max));
        // And the hole likewise.
        let h = hole_aabb(
            [-64 + shift as i32, 0, -64 - shift as i32],
            [80 + shift as i32, 48, 80 - shift as i32],
            anchor,
        );
        assert_eq!(h.min, Vec3::new(-64.0, 0.0, -64.0));
        assert_eq!(h.max, Vec3::new(80.0, 48.0, 80.0));

        // Up a grid step too (review finding B1): the heights are absolute,
        // so a patch 4,096 blocks higher against an anchor 4,096 higher is
        // the same box -- the anchor's y comes off the heights, as it does
        // off the hole and, in `far.wgsl`, off every vertex.
        let up = faces.map(|f| FarFace {
            top: f.top + 4096,
            ..f
        });
        let high = PlacedPatch::of(&FarPatch {
            origin: [64 + shift, -128 - shift],
            quad: 4.0,
            faces: &up,
            skirt: 8.0,
        });
        let anchor_up = [shift as i32, 4096, -shift as i32];
        assert_eq!(high.gpu(anchor_up).origin, here.gpu(anchor::ORIGIN).origin);
        let c = high.aabb(anchor_up);
        assert_eq!((c.min, c.max), (b.min, b.max));
    }

    /// Patches wholly inside the voxels' hole are not drawn; the rest are,
    /// nearest first.
    #[test]
    fn patches_are_drawn_nearest_first_and_none_inside_the_hole() {
        let b = |x: f32| {
            Some(Aabb {
                min: Vec3::new(x, 0.0, 0.0),
                max: Vec3::new(x + 10.0, 10.0, 10.0),
            })
        };
        let hole = Aabb {
            min: Vec3::new(-100.0, -100.0, -100.0),
            max: Vec3::new(100.0, 100.0, 100.0),
        };
        //           inside  crosses  clear     clear     crosses
        let boxes = [b(0.0), b(95.0), b(300.0), b(150.0), b(-105.0)];
        let order = draw_order(&boxes, vec![0, 1, 2, 3, 4], Some(&hole), Vec3::ZERO);
        assert_eq!(order, vec![1, 4, 3, 2]);
        let order = draw_order(&boxes, vec![2, 0, 1], None, Vec3::ZERO);
        assert_eq!(order, vec![0, 1, 2], "no hole: everything, nearest first");
    }

    /// Where every vertex of a patch is, in world blocks, for a patch of
    /// [`dense_patch`] `words` at `origin`. `sink`: every vertex strictly
    /// inside the box goes to `hidden`, which is what the far terrain did
    /// until issue #424 -- the control a test below tells apart.
    fn placed_patch(
        words: &[u32],
        patch: ([f32; 2], f32, f32),
        hole: (Vec3, Vec3),
        hidden: f32,
        sink: bool,
    ) -> Vec<Vec3> {
        (0..FAR_PATCH_VERTICES)
            .map(|v| {
                let pv = PatchVertex::of(v);
                let y = pv.y(words, patch.2);
                let world = place(pv.at, pv.corner, y, patch, Some(hole), hidden);
                let raw = place(pv.at, pv.corner, y, patch, None, 0.0);
                if sink && raw.cmpgt(hole.0).all() && raw.cmplt(hole.1).all() {
                    Vec3::new(raw.x, hidden, raw.z)
                } else {
                    world
                }
            })
            .collect()
    }

    /// Rolling ground, so no two neighbours are level, as a patch whose edge
    /// walls hang `step` blocks.
    fn rolling_words(height: &dyn Fn(i32, i32) -> i32, step: u32) -> Vec<u32> {
        dense_patch(&unit_faces(height, step))
    }

    /// Every triangle of a patch that has area, whose middle is strictly
    /// inside the box and that an eye at `eye` would be shown, as its three
    /// corners.
    fn seen_inside_the_box(
        origin: [f32; 2],
        hole: (Vec3, Vec3),
        eye: Vec3,
        hidden: Option<f32>,
        sink: bool,
    ) -> Vec<[Vec3; 3]> {
        let (quad, skirt_depth) = (8.0, 40.0);
        let words = rolling_words(&|i, j| 60 + 9 * ((i * 3 + j * 5) % 7), 8);
        // Ground well above the floor: what an eye inside the box has
        // raised out of sight, unless the old rule's height is asked for.
        let hidden = hidden.unwrap_or_else(|| hidden_height(hole.0, hole.1, eye, 20.0));
        let p = placed_patch(&words, (origin, quad, skirt_depth - 0.0), hole, hidden, sink);
        let strictly_inside =
            |p: Vec3| p.x > hole.0.x && p.x < hole.1.x && p.z > hole.0.z && p.z < hole.1.z;
        let mut seen = Vec::new();
        for t in patch_indices().chunks(3) {
            let [a, b, c] = [p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]];
            // Counter-clockwise is the front (`build_far_pipeline`): this
            // normal points out of the side that is drawn.
            let normal = (b - a).cross(c - a);
            let nothing = normal.length() < 1e-3;
            let faces_away = normal.dot(eye - a) <= 0.0;
            // On the box's side or outside it: not inside. A triangle is
            // inside only if its middle is.
            let middle = (a + b + c) / 3.0;
            if nothing || faces_away || !strictly_inside(middle) {
                continue;
            }
            seen.push([a, b, c]);
        }
        seen
    }

    /// **The curtain of issue #424.** A patch that straddles a side of the
    /// voxels' box, and one that straddles a corner of it, seen by an eye
    /// inside the box -- above the ground or under it: no triangle inside the
    /// box shows the eye its front, risers and walls on the patch's edge
    /// included. The one exception is on the very corner, where the two
    /// sides' skirts meet: triangles within two quads of it and a skirt's
    /// depth of the surface, which are under the voxels' ground like the
    /// skirts themselves.
    #[test]
    fn nothing_inside_the_box_faces_an_eye_inside_it() {
        let hole = (
            Vec3::new(-1024.0, -1064.0, -1024.0),
            Vec3::new(1024.0, 984.0, 1024.0),
        );
        // (the patch's corner, what it straddles)
        let patches = [
            ([896.0, -128.0], "the +x side"),
            ([-1152.0, 0.0], "the -x side"),
            ([256.0, 896.0], "the +z side"),
            ([896.0, 896.0], "the +x +z corner"),
            ([-1152.0, -1152.0], "the -x -z corner"),
        ];
        for eye in [Vec3::new(8.0, 120.0, 8.0), Vec3::new(8.0, -40.0, 8.0)] {
            for (origin, what) in patches {
                let seen = seen_inside_the_box(origin, hole, eye, None, false);
                // Where the skirts meet on the corner, a skirt deep at most.
                let on_the_corner = |t: &[Vec3; 3]| {
                    t.iter().all(|p| {
                        (p.x.abs() - 1024.0).abs() <= 16.0
                            && (p.z.abs() - 1024.0).abs() <= 16.0
                            && p.y > 60.0 - 40.0 - 1.0
                    })
                };
                let wrong: Vec<_> = seen.iter().filter(|t| !on_the_corner(t)).collect();
                assert!(
                    wrong.is_empty(),
                    "eye {eye}, {what}: {} triangles inside the box face the eye, e.g. {:?}",
                    wrong.len(),
                    wrong[0]
                );
                // And the test can tell: sunk to the floor, as before, the
                // same patch shows the eye a wall.
                let before = seen_inside_the_box(origin, hole, eye, Some(hole.0.y - 1.0), true);
                assert!(
                    before.len() > 30,
                    "eye {eye}, {what}: the old rule showed only {} triangles, so this proves nothing",
                    before.len()
                );
            }
        }
    }

    /// **No sheet where the ground crosses the box's floor** (review of #427).
    /// A thousand blocks up, the box's floor cuts through the hills: ground
    /// under the floor is the far terrain's to draw, and stays where it is,
    /// while the vertex next to it, just above the floor, is hidden. Hidden
    /// must then mean *next to it* -- under the floor. Raised above the box
    /// instead, as an eye inside the box otherwise wants, the triangle
    /// between the two is a sheet two thousand blocks tall on the near side
    /// of every hill.
    #[test]
    fn no_sheet_where_the_ground_crosses_the_boxs_floor() {
        let (quad, skirt_depth) = (8.0, 40.0);
        // Steep ground from 60 to 420, so both floors below cut through it.
        let height = |i: i32, j: i32| 60 + 60 * ((i * 3 + j * 5) % 7);
        let words = rolling_words(&height, 8);
        // The tallest triangle of the ground as it is: six steps and a skirt
        // (and the step an edge wall shows).
        let bound = 6.0 * 60.0 + skirt_depth + 8.0 + 3.0 * 4.0 + 1.0;
        for eye_y in [1_100.0, 1_400.0] {
            let eye = Vec3::new(8.0, eye_y, 8.0);
            let hole = (
                Vec3::new(-1024.0, eye_y - 1024.0, -1024.0),
                Vec3::new(1024.0, eye_y + 1024.0, 1024.0),
            );
            // In the middle of the box, across a side of it, and on a corner.
            for origin in [[0.0, 0.0], [896.0, -128.0], [896.0, 896.0]] {
                let lowest = (0..FAR_PATCH_VERTICES)
                    .map(|v| PatchVertex::of(v).y(&words, skirt_depth))
                    .fold(f32::INFINITY, f32::min);
                assert!(
                    lowest < hole.0.y && hole.0.y < 420.0,
                    "the floor does not cut the ground, so this proves nothing"
                );
                let hidden = hidden_height(hole.0, hole.1, eye, lowest);
                let p = placed_patch(&words, (origin, quad, skirt_depth), hole, hidden, false);
                for t in patch_indices().chunks(3) {
                    let [a, b, c] = [p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]];
                    let normal = (b - a).cross(c - a);
                    let faces_the_eye = normal.length() > 1e-3 && normal.dot(eye - a) > 0.0;
                    let tall = a.y.max(b.y).max(c.y) - a.y.min(b.y).min(c.y);
                    assert!(
                        !faces_the_eye || tall <= bound,
                        "eye at y = {eye_y}, patch at {origin:?}: a triangle {tall} blocks \
                         tall faces the eye: {a} {b} {c}"
                    );
                }
            }
        }
    }

    /// Deep inside the box nothing stands but the flat roof or floor the
    /// hidden vertices make: every riser, and every wall on the patch's edge,
    /// is of no area -- its corners are the quads' own, and go where they go
    /// (`nothing_inside_the_box_faces_an_eye_inside_it` for the sides).
    #[test]
    fn a_riser_deep_inside_the_box_has_no_area() {
        let hole = (
            Vec3::new(-1024.0, -1064.0, -1024.0),
            Vec3::new(1024.0, 984.0, 1024.0),
        );
        let words = rolling_words(&|i, j| 60 + 9 * ((i * 3 + j * 5) % 7), 8);
        for eye in [Vec3::new(8.0, 120.0, 8.0), Vec3::new(500.0, 800.0, 8.0)] {
            let hidden = hidden_height(hole.0, hole.1, eye, 20.0);
            // A patch wholly in the box, away from every side.
            let p = placed_patch(&words, ([0.0, 0.0], 8.0, 40.0), hole, hidden, false);
            let mut flat = 0;
            for t in patch_indices().chunks(3) {
                let [a, b, c] = [p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]];
                let cross = (b - a).cross(c - a);
                if cross.length() < 1e-3 {
                    continue;
                }
                assert!(
                    cross.normalize().y.abs() > 0.999,
                    "{a} {b} {c} stands in the middle of the box"
                );
                flat += 1;
            }
            assert!(flat > 1_000, "{flat} flat triangles");
        }
    }

    /// A quad's top is the surface block's, and the first row of quads
    /// inside the box's side folds down into a skirt on it, as it always did.
    #[test]
    fn the_first_row_of_tops_inside_the_box_folds_into_its_skirt() {
        let hole = (
            Vec3::new(-1024.0, -1064.0, -1024.0),
            Vec3::new(1024.0, 984.0, 1024.0),
        );
        // Quads of 8 from x = 896: quad 16 starts on the box's side.
        let patch = ([896.0, -128.0], 8.0, 40.0);
        let eye = Vec3::new(8.0, 120.0, 8.0);
        let hidden = hidden_height(hole.0, hole.1, eye, 20.0);
        let corners = |i: i32, j: i32| {
            [(i, j, 0u8), (i + 1, j, 1), (i, j + 1, 2), (i + 1, j + 1, 3)]
        };
        // Outside, against the side: untouched.
        for (i, j, k) in corners(16, 4) {
            assert_eq!(
                place([i, j], k, 70.0, patch, Some(hole), hidden),
                place([i, j], k, 70.0, patch, None, 0.0)
            );
        }
        // Quad 15 is the first inside. Its corners on the side stay; the two
        // a quad inside drop under them: a wall on the side, from the top
        // down a skirt's depth.
        let p = corners(15, 4).map(|(i, j, k)| place([i, j], k, 70.0, patch, Some(hole), hidden));
        for v in p {
            assert_eq!(v.x, 1024.0, "{p:?}");
        }
        let ys: Vec<f32> = p.iter().map(|v| v.y).collect();
        assert_eq!(ys, vec![30.0, 70.0, 30.0, 70.0]);
        let normal = (p[2] - p[0]).cross(p[1] - p[0]).normalize();
        assert_eq!(normal, Vec3::NEG_X, "the skirt faces into the box");
        // Deeper in: hidden, all four at the hidden height -- and the far end
        // of quad 14, whose near end is the first quad's, with them: it
        // touches no side.
        for (i, j, k) in corners(9, 4).into_iter().chain(corners(14, 4)) {
            if i == 15 {
                continue;
            }
            assert_eq!(
                place([i, j], k, 70.0, patch, Some(hole), hidden).y,
                hidden,
                "grid {i} {j}"
            );
        }
        // The near end of quad 14 is one quad inside as well, but its quad
        // does not touch the side: hidden, not folded.
        let (i, j, k) = (15, 4, 0u8);
        assert_eq!(
            place([i, j], k, 70.0, patch, Some(hole), hidden),
            Vec3::new(1024.0, 30.0, patch.0[1] + 4.0 * 8.0)
        );
        let (i, j, k) = (15, 4, 1u8);
        assert_eq!(
            place([i, j], k, 70.0, patch, Some(hole), hidden).y,
            hidden,
            "the high end of quad 14 is not on the side"
        );
    }

    /// The seam to the voxels stays closed: the first ring inside the box
    /// hangs straight under the vertex on the box's side, a skirt deep --
    /// and only the first, and only from a side this patch holds.
    #[test]
    fn the_first_ring_inside_is_a_skirt_on_the_boxs_side() {
        // 3 quads inside the low x side, far from the others.
        assert_eq!(join([11, 5], 1, [3.0, 40.0], [253.0, 216.0]), Join::Hidden);
        assert_eq!(
            join([9, 5], 1, [1.0, 40.0], [255.0, 216.0]),
            Join::Skirt([8, 5])
        );
        // The high sides step the other way.
        assert_eq!(
            join([9, 5], 0, [255.0, 216.0], [1.0, 40.0]),
            Join::Skirt([10, 5])
        );
        assert_eq!(
            join([9, 5], 0, [40.0, 255.0], [216.0, 1.0]),
            Join::Skirt([9, 6])
        );
        // On a corner, both: straight under the corner vertex.
        assert_eq!(
            join([9, 5], 3, [1.0, 1.0], [255.0, 255.0]),
            Join::Skirt([8, 4])
        );
        // A side one quad beyond this patch's edge is the next patch's to
        // skirt; the side this patch does hold still gets its own.
        assert_eq!(join([0, 5], 1, [1.0, 40.0], [255.0, 216.0]), Join::Hidden);
        assert_eq!(join([32, 5], 0, [255.0, 40.0], [1.0, 216.0]), Join::Hidden);
        assert_eq!(
            join([0, 5], 3, [1.0, 1.0], [255.0, 255.0]),
            Join::Skirt([0, 4])
        );
    }

    /// Hidden means out of the eye's sight, whichever side that is.
    #[test]
    fn hidden_vertices_go_where_the_eye_sees_their_underside() {
        let (min, max) = (
            Vec3::new(-100.0, -50.0, -100.0),
            Vec3::new(100.0, 70.0, 100.0),
        );
        // All the ground above the floor.
        let above = -10.0;
        assert_eq!(
            hidden_height(min, max, Vec3::new(0.0, -20.0, 0.0), above),
            71.0
        );
        assert_eq!(
            hidden_height(min, max, Vec3::new(0.0, 60.0, 99.0), above),
            71.0
        );
        // Above the box, or beside it: under it, as before.
        assert_eq!(
            hidden_height(min, max, Vec3::new(0.0, 120.0, 0.0), above),
            -51.0
        );
        assert_eq!(
            hidden_height(min, max, Vec3::new(150.0, 0.0, 0.0), above),
            -51.0
        );
        // Ground under the floor, or on it: under the box, next to it.
        assert_eq!(
            hidden_height(min, max, Vec3::new(0.0, -20.0, 0.0), -60.0),
            -51.0
        );
        assert_eq!(
            hidden_height(min, max, Vec3::new(0.0, -20.0, 0.0), -50.0),
            -51.0
        );
    }

    /// The shader is the rule above: the same steps, in the same order.
    ///
    /// A pin on its text, which is all that holds the skirt ring on the GPU:
    /// `golden.rs` renders that nothing inside the box is seen, and no image
    /// shows the seam's skirt on its own (a missing one is a line of sky a
    /// few pixels tall, a kilometre away).
    #[test]
    fn the_shader_joins_as_this_module_does() {
        let wgsl = crate::haze::FAR_WGSL;
        for line in [
            "if all(world > params.hole_min.xyz) && all(world < params.hole_max.xyz) {",
            "let lo = (world.xz - params.hole_min.xz) / tile.quad;",
            "let hi = (params.hole_max.xz - world.xz) / tile.quad;",
            "let step = vec2<i32>(round(select(-lo, hi, hi < lo)));",
            "let towards_x = (step.x < 0) == ((corner & 1u) != 0u);",
            "let towards_z = (step.y < 0) == ((corner & 2u) != 0u);",
            "let snap_x = abs(step.x) == 1 && towards_x && ti >= 0 && ti <= QUADS;",
            "let snap_z = abs(step.y) == 1 && towards_z && tj >= 0 && tj <= QUADS;",
            "if snap_x || snap_z {",
            "i = select(i, ti, snap_x);",
            "j = select(j, tj, snap_z);",
            // A snapped corner hangs a skirt under its own height.
            "y - tile.skirt,",
            // `hidden_height`, worked out on the CPU (`FarTerrain::prepare`).
            "world.y = params.hole_min.w;",
            "let sealed = world + outward * (CRACK_SEAL * distance(world, frame.eye.xyz));",
        ] {
            assert!(wgsl.contains(line), "far.wgsl no longer has: {line}");
        }
    }

    #[test]
    fn only_what_the_frustum_sees_is_drawn() {
        let proj = glam::Mat4::perspective_rh(60f32.to_radians(), 1.0, 0.1, 10_000.0);
        let view = glam::Mat4::look_at_rh(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), Vec3::Y);
        let frustum = Frustum::from_view_proj(proj * view);
        let ahead = Aabb {
            min: Vec3::new(-10.0, -10.0, -110.0),
            max: Vec3::new(10.0, 10.0, -90.0),
        };
        let behind = Aabb {
            min: Vec3::new(-10.0, -10.0, 90.0),
            max: Vec3::new(10.0, 10.0, 110.0),
        };
        let boxes = vec![Some(behind), None, Some(ahead), Some(ahead)];
        assert_eq!(visible_slots(&boxes, &frustum), vec![2, 3]);
        // Issue #424: out of sight, none of it is drawn, in view or not.
        assert_eq!(
            slots_to_draw(true, &boxes, &frustum, None, Vec3::ZERO),
            vec![2, 3]
        );
        assert!(slots_to_draw(false, &boxes, &frustum, None, Vec3::ZERO).is_empty());
    }

    /// The ground that decides where hidden vertices go is the ground over
    /// the box, lowest skirt and all -- not ground a kilometre beyond it.
    #[test]
    fn the_lowest_ground_is_looked_for_over_the_box_only() {
        let b = |x: f32, low: f32| {
            Some(Aabb {
                min: Vec3::new(x, low, 0.0),
                max: Vec3::new(x + 10.0, 500.0, 10.0),
            })
        };
        let hole = Aabb {
            min: Vec3::new(-100.0, 40.0, -100.0),
            max: Vec3::new(100.0, 900.0, 100.0),
        };
        //           over it      crossing its side  beyond it     free slot
        let boxes = [b(0.0, 70.0), b(95.0, 55.0), b(300.0, -20.0), None];
        assert_eq!(lowest_over(&boxes, &hole), 55.0);
        assert_eq!(lowest_over(&boxes[2..], &hole), f32::INFINITY);
    }
}

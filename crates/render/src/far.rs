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
use crate::arena::SlabAllocator;
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
    /// The three words `far.wgsl` unpacks: `i0:6 j0:6 i1:6 j1:6 dir:3
    /// border:1`, then `top + 131072 : 18, rise : 14`, then `surface:8
    /// soil:8 open:8`. A height or a step beyond what fits is clamped -- the
    /// terrain's own range is a small part of either (the app asserts it,
    /// `far_streaming`).
    pub fn pack(&self) -> [u32; 3] {
        let top = self.top.clamp(*FAR_TOP_RANGE.start(), *FAR_TOP_RANGE.end());
        let rise = self.rise.min(FAR_MAX_RISE);
        [
            self.i0 as u32
                | (self.j0 as u32) << 6
                | (self.i1 as u32) << 12
                | (self.j1 as u32) << 18
                | (self.dir as u32) << 24
                | (self.border as u32) << 27,
            (top - FAR_TOP_RANGE.start()) as u32 | rise << 18,
            self.mix[0] as u32 | (self.mix[1] as u32) << 8 | (self.open as u32) << 16,
        ]
    }

    /// [`pack`](Self::pack), undone.
    pub fn unpack([w0, w1, w2]: [u32; 3]) -> Self {
        let six = |shift: u32| ((w0 >> shift) & 0x3F) as u8;
        Self {
            i0: six(0),
            j0: six(6),
            i1: six(12),
            j1: six(18),
            dir: FarDir::ALL[((w0 >> 24) & 0x7) as usize],
            border: (w0 >> 27) & 1 != 0,
            top: (w1 & 0x3_FFFF) as i32 + FAR_TOP_RANGE.start(),
            rise: w1 >> 18,
            mix: [(w2 & 0xFF) as u8, ((w2 >> 8) & 0xFF) as u8],
            open: ((w2 >> 16) & 0xFF) as u8,
        }
    }

    /// The corner `far.wgsl` makes of this face for the `k`th of its four
    /// vertices -- the shader's own steps, so a test can hold them against
    /// [`corners`](Self::corners), which is the definition.
    pub fn shader_corner(&self, k: u32) -> FarVertex {
        let wall = self.dir != FarDir::Top;
        // The two wall directions that would wind clockwise run backwards.
        let flip = matches!(self.dir, FarDir::NegX | FarDir::PosZ) as u32;
        let e = (k & 1) ^ flip;
        let u = k >> 1;
        let (di, dj) = ((self.i1 - self.i0) as u32, (self.j1 - self.j0) as u32);
        FarVertex {
            at: [
                (self.i0 as u32 + e * di) as i32,
                (self.j0 as u32 + if wall { e } else { u } * dj) as i32,
            ],
            dir: self.dir,
            corner: (e | u << 1) as u8,
            top: self.top,
            rise: self.rise,
            border: self.border,
        }
    }

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

/// The order a patch's faces are stored in, by direction, so that the
/// directions that can face one eye are next to each other: an eye sees at
/// most one of `PosX`/`NegX` and one of `PosZ`/`NegZ` of a patch it is not
/// over, and the tops. With this order that is one run of faces for two of
/// the four quarters the eye can be in and two runs for the others.
const GROUP_ORDER: [FarDir; 5] = [
    FarDir::PosX,
    FarDir::PosZ,
    FarDir::Top,
    FarDir::NegZ,
    FarDir::NegX,
];

/// Which of a patch's face groups ([`GROUP_ORDER`]) an eye at `eye` can see
/// the front of, for a patch whose box is `b` (both relative to the same
/// anchor): a wall facing `+x` stands somewhere in the box's span of x and is
/// seen only from further along `+x` than itself, so not at all from an eye
/// at or before the box's low side -- and so on for the other three. The tops
/// are always drawn: inside the voxels' box the join folds some of them into
/// walls (`join`), which face wherever the box's side does.
///
/// What the voxel arena does per node (`arena::faces_facing`), and for the
/// same reason: about two walls in five face away from any one eye, and a
/// triangle that is not sent is cheaper than one the GPU culls.
pub fn groups_facing(b: &Aabb, eye: Vec3) -> [bool; 5] {
    GROUP_ORDER.map(|dir| match dir {
        FarDir::Top => true,
        FarDir::PosX => eye.x > b.min.x,
        FarDir::NegX => eye.x < b.max.x,
        FarDir::PosZ => eye.z > b.min.z,
        FarDir::NegZ => eye.z < b.max.z,
    })
}

/// A patch's faces in the order they are stored and drawn in: by direction
/// ([`GROUP_ORDER`]), otherwise as they came.
pub fn stored_order(faces: &[FarFace]) -> Vec<FarFace> {
    let mut out = Vec::with_capacity(faces.len());
    for dir in GROUP_ORDER {
        out.extend(faces.iter().filter(|f| f.dir == dir));
    }
    out
}

/// How many of `faces` look each way, in [`GROUP_ORDER`].
pub fn group_counts(faces: &[FarFace]) -> [u32; 5] {
    GROUP_ORDER.map(|dir| faces.iter().filter(|f| f.dir == dir).count() as u32)
}

/// The runs of consecutive faces to draw, as `(first face, faces)`, for a
/// patch whose groups hold `counts` faces each (in [`GROUP_ORDER`]) and of
/// which `facing` are wanted. Empty groups neither start nor break a run.
pub fn facing_runs(counts: [u32; 5], facing: [bool; 5], mut emit: impl FnMut(u32, u32)) {
    let mut start = 0;
    let mut run: Option<(u32, u32)> = None;
    for (count, wanted) in counts.into_iter().zip(facing) {
        if wanted && count > 0 {
            run = Some(match run {
                Some((first, n)) => (first, n + count),
                None => (start, count),
            });
        } else if !wanted && count > 0 {
            if let Some((first, n)) = run.take() {
                emit(first, n);
            }
        }
        start += count;
    }
    if let Some((first, n)) = run {
        emit(first, n);
    }
}

/// How many faces one draw's instances each hold: the far terrain's runs of
/// faces are sorted into these classes, and each class with anything in it
/// is one instanced draw, every instance padded to the class's size with
/// zero-area faces.
///
/// **Why classes at all.** A run is some of one patch's faces, and runs
/// differ in length; one draw needs one vertex count. A draw per run is
/// hundreds of draws a frame, and draws are what the M3 is bound by
/// (`PHASE1_ARCHITECTURE.md` §2) -- Metal has no multi-draw, so an indirect
/// list would be the same hundreds of draws issued in a loop. Seventeen
/// classes are at most seventeen draws, whatever is in view.
///
/// Each is `sqrt 2` times the last, rounded up, from 16 to the first one that
/// holds a whole patch ([`FAR_MAX_FACES`]), so a run is padded by 41% at
/// most and about a fifth on average. **Chosen, not derived**: a finer
/// ladder pads less and draws more often.
pub const CLASS_FACES: [u32; 17] = [
    16, 23, 32, 46, 64, 91, 128, 182, 256, 363, 512, 725, 1024, 1449, 2048, 2897, 4096,
];

/// The class a run of `faces` faces is drawn in: the smallest that holds it.
pub fn class_of(faces: u32) -> usize {
    CLASS_FACES
        .iter()
        .position(|&room| faces <= room)
        .expect("no run is longer than a patch, and the last class holds one")
}

/// One run of faces to draw, as `far.wgsl` reads it: the patch's slot, the
/// run's first face in the shared buffer, and how many faces it is.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
struct RunGpu {
    slot: u32,
    first: u32,
    count: u32,
}

/// The far terrain on the GPU: every held patch's faces in one storage
/// buffer, and the runs of them drawn this frame.
///
/// There is no vertex buffer. A face is three packed words (12 bytes)
/// ([`FarFace::pack`]); `far.wgsl` makes its four corners from the vertex
/// index and finds its run -- and through it its patch -- by the instance
/// index. Both are sound because every draw here is direct
/// (`scripts/check-architecture.sh`, Rule 5).
pub struct FarTerrain {
    faces: wgpu::Buffer,
    face_slab: SlabAllocator,
    patches: wgpu::Buffer,
    params: wgpu::Buffer,
    indices: wgpu::Buffer,
    /// The runs drawn this frame, a region of [`runs_per_class`] for each
    /// class.
    runs: wgpu::Buffer,
    /// One bind group a class: the same buffers, with that class's region of
    /// `runs`.
    bind_groups: Vec<wgpu::BindGroup>,
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
    /// This frame's runs by class, from the last [`prepare`](Self::prepare).
    by_class: Vec<Vec<RunGpu>>,
    /// How many slots [`prepare`](Self::prepare) found in view.
    drawn: u32,
    /// The triangles of this frame's runs: two a face.
    drawn_triangles: u64,
    /// Whether the far terrain can be seen at all
    /// ([`set_shown`](Self::set_shown)).
    shown: bool,
    /// Whether hidden vertices were last sent above the box rather than
    /// under it ([`hidden_height`]): what `params` tells `far.wgsl`. Under
    /// it until [`prepare`](Self::prepare) says otherwise.
    raised: bool,
}

/// A patch as it was inserted, minus its faces (which the anchor does not
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
    /// Where its faces are in the shared buffer.
    first_face: u32,
    /// How many faces it has of each direction, in [`GROUP_ORDER`] -- the
    /// order they are stored in.
    groups: [u32; 5],
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
            first_face: 0,
            groups: group_counts(patch.faces),
        }
    }

    fn face_len(&self) -> u32 {
        self.groups.iter().sum()
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
    /// absolute (a face carries its own, and a rebase leaves the faces
    /// alone), so the anchor's `y` comes off here, as `far.wgsl` takes it
    /// off every vertex.
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

/// A patch's faces as they go up: grouped by direction ([`GROUP_ORDER`]),
/// three words each.
fn pack_patch(faces: &[FarFace]) -> Vec<[u32; 3]> {
    stored_order(faces).iter().map(FarFace::pack).collect()
}

/// Bytes a face takes: [`FarFace::pack`]'s three words.
const FACE_BYTES: u64 = 12;

/// How many runs one class has room for, for `capacity` patches: every
/// patch's two runs at most ([`GROUP_ORDER`]), rounded up so that a class's
/// region of the buffer starts on a multiple of 256 bytes, as a bound range
/// of a storage buffer must.
fn runs_per_class(capacity: u32) -> u32 {
    (2 * capacity).next_multiple_of(64)
}

impl FarTerrain {
    /// Room for `capacity` patches and `face_capacity` faces, placed against
    /// the render anchor `anchor` until [`rebase`](Self::rebase) moves it.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        (capacity, face_capacity): (u32, u32),
        anchor: Anchor,
    ) -> Self {
        let storage = |label, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(16),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let face_capacity = face_capacity.max(1);
        let faces = storage("far-faces", face_capacity as u64 * FACE_BYTES);
        let patches = storage(
            "far-patches",
            std::mem::size_of::<PatchGpu>() as u64 * capacity as u64,
        );
        let run_bytes = std::mem::size_of::<RunGpu>() as u64;
        let region = runs_per_class(capacity) as u64 * run_bytes;
        let runs = storage("far-runs", region * CLASS_FACES.len() as u64);
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

        let quads = quad_indices();
        let indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("far-quad-indices"),
            size: std::mem::size_of_val(quads.as_slice()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&indices, 0, bytemuck::cast_slice(&quads));

        let layout = far_bind_group_layout(device);
        let bind_groups = (0..CLASS_FACES.len() as u64)
            .map(|class| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                            resource: faces.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &runs,
                                offset: class * region,
                                size: wgpu::BufferSize::new(region),
                            }),
                        },
                    ],
                })
            })
            .collect();
        Self {
            faces,
            face_slab: SlabAllocator::new(face_capacity),
            patches,
            params,
            indices,
            runs,
            bind_groups,
            capacity,
            free: (0..capacity).rev().collect(),
            boxes: vec![None; capacity as usize],
            placed: vec![None; capacity as usize],
            params_world,
            anchor,
            by_class: vec![Vec::new(); CLASS_FACES.len()],
            drawn: 0,
            drawn_triangles: 0,
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
        let faces: usize = patches.iter().map(|p| p.faces.len()).sum();
        let room = (patches.len().max(1) as u32, faces as u32);
        let mut terrain = Self::new(device, queue, room, anchor);
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
    /// (issue #379 decision 2). The faces are untouched -- they are whole
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

    /// Upload a patch. `None` when every slot is taken or its faces do not
    /// fit -- the caller decides what gives way, since it is the one that
    /// knows which patches matter.
    pub fn insert(&mut self, queue: &wgpu::Queue, patch: FarPatch<'_>) -> Option<FarSlot> {
        assert!(
            patch.faces.len() <= FAR_MAX_FACES,
            "a patch of {} faces",
            patch.faces.len()
        );
        let slot = *self.free.last()?;
        let first_face = self.face_slab.alloc(patch.faces.len() as u32)?;
        self.free.pop();
        if !patch.faces.is_empty() {
            queue.write_buffer(
                &self.faces,
                first_face as u64 * FACE_BYTES,
                bytemuck::cast_slice(&pack_patch(patch.faces)),
            );
        }
        let placed = PlacedPatch {
            first_face,
            ..PlacedPatch::of(&patch)
        };
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
            if let Some(placed) = self.placed[slot.0 as usize].take() {
                self.face_slab.free(placed.first_face, placed.face_len());
            }
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
    /// depth buffer before the distant ground it hides is drawn -- and of
    /// each, the faces that can face the eye ([`groups_facing`]). Returns how
    /// many patches. `frustum` and `eye` are anchor-relative, like the boxes
    /// (issue #379 decision 5).
    /// Patches wholly inside the voxels' hole are not drawn at all.
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
        for class in &mut self.by_class {
            class.clear();
        }
        self.drawn_triangles = 0;
        for &slot in &order {
            let (Some(placed), Some(aabb)) =
                (&self.placed[slot as usize], &self.boxes[slot as usize])
            else {
                continue;
            };
            let by_class = &mut self.by_class;
            let triangles = &mut self.drawn_triangles;
            facing_runs(placed.groups, groups_facing(aabb, eye), |first, count| {
                by_class[class_of(count)].push(RunGpu {
                    slot,
                    first: placed.first_face + first,
                    count,
                });
                *triangles += count as u64 * 2;
            });
        }
        let region = runs_per_class(self.capacity) as u64 * std::mem::size_of::<RunGpu>() as u64;
        for (class, runs) in self.by_class.iter().enumerate() {
            if !runs.is_empty() {
                queue.write_buffer(
                    &self.runs,
                    class as u64 * region,
                    bytemuck::cast_slice(runs),
                );
            }
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

    /// Faces held, and room for.
    pub fn face_usage(&self) -> (u32, u32) {
        let held = self.placed.iter().flatten().map(|p| p.face_len()).sum();
        (held, self.face_slab.capacity())
    }

    /// Patches drawn this frame, from the last [`prepare`](Self::prepare).
    pub fn drawn(&self) -> u32 {
        self.drawn
    }

    /// Triangles drawn this frame: the faces of the patches in view that
    /// can face the eye, two each. Not the padding ([`drawn_cost`]).
    ///
    /// [`drawn_cost`]: Self::drawn_cost
    pub fn drawn_triangles(&self) -> u64 {
        self.drawn_triangles
    }

    /// What this frame's far terrain costs to issue: `(draws, vertices)` --
    /// one draw a class with anything in it, and four vertices for every
    /// face of every run's class, padding included ([`CLASS_FACES`]).
    pub fn drawn_cost(&self) -> (u32, u64) {
        issued(&self.by_class.iter().map(Vec::len).collect::<Vec<_>>())
    }

    /// Draw this frame's runs. The pipeline and bind group 0 (the camera)
    /// are the caller's to set.
    pub(crate) fn encode(&self, pass: &mut wgpu::RenderPass<'_>) {
        if self.drawn_triangles == 0 {
            return;
        }
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint16);
        for (class, runs) in self.by_class.iter().enumerate() {
            if runs.is_empty() {
                continue;
            }
            pass.set_bind_group(1, &self.bind_groups[class], &[]);
            pass.draw_indexed(0..CLASS_FACES[class] * 6, 0, 0..runs.len() as u32);
        }
    }
}

/// `(draws, vertices)` for `runs[class]` runs in each class.
pub fn issued(runs: &[usize]) -> (u32, u64) {
    let draws = runs.iter().filter(|&&n| n > 0).count() as u32;
    let vertices = runs
        .iter()
        .zip(CLASS_FACES)
        .map(|(&n, room)| n as u64 * room as u64 * 4)
        .sum();
    (draws, vertices)
}

/// The index list every run is drawn with: [`QUAD_INDICES`] for face after
/// face, as many as the largest class holds. A draw takes a prefix of it.
fn quad_indices() -> Vec<u16> {
    let most = *CLASS_FACES.last().expect("there are classes") as u16;
    (0..most)
        .flat_map(|face| QUAD_INDICES.map(|i| face * 4 + i))
        .collect()
}

/// `far.wgsl`'s `vs_main`, as far as a vertex's position before the crack
/// seal: where the corner `v` of a patch at `origin` with `quad`-wide quads
/// and a skirt `skirt` deep is put, given the voxels' box `hole` (`None`:
/// no box) and the height hidden vertices go to ([`hidden_height`]). In world
/// blocks; the shader does the same with the render anchor taken off
/// everything.
pub fn place(
    v: &FarVertex,
    (origin, quad, skirt): ([f32; 2], f32, f32),
    hole: Option<(Vec3, Vec3)>,
    hidden: f32,
) -> Vec3 {
    let at = |g: [i32; 2]| {
        [
            origin[0] + g[0] as f32 * quad,
            origin[1] + g[1] as f32 * quad,
        ]
    };
    let [x, z] = at(v.at);
    let y = v.y(skirt);
    let mut world = Vec3::new(x, y, z);
    let Some((min, max)) = hole else {
        return world;
    };
    if v.dir != FarDir::Top {
        if wall_in_hole(v, world, quad, (min, max)) {
            world.y = world.y.min(min.y);
        }
    } else if world.cmpgt(min).all() && world.cmplt(max).all() {
        let lo = [(x - min.x) / quad, (z - min.z) / quad];
        let hi = [(max.x - x) / quad, (max.z - z) / quad];
        match join(v.at, lo, hi) {
            Join::Skirt(to) => {
                let [x, z] = at(to);
                world = Vec3::new(x, y - skirt, z);
            }
            Join::Hidden => world.y = hidden,
        }
    }
    world
}

/// Whether the wall whose corner `v` stands at `world` is the voxels' to
/// draw: in their box as a whole, by a point half a quad along it from this
/// corner and by its top. No wall crosses a side of the box
/// (`cubara_world::far::JOIN_GRID`), so that point is on the same side of
/// every one of them for all four corners, and the four agree.
///
/// **Why a wall does not go through [`join`]**, as a top's corners do: a wall
/// with two corners snapped to the box's side and two sent to the hidden
/// height is a vertical sliver a thousand blocks tall, and it faces sideways
/// -- at an eye inside the box, where nothing would cull it. That is the
/// curtain of issue #424 again, a block wide. A wall in the box is cut off at
/// the box's floor instead: all of it above the floor is gone, wherever the
/// eye is, and what stands under the floor -- ground the far terrain still
/// draws -- keeps its wall.
pub fn wall_in_hole(v: &FarVertex, world: Vec3, quad: f32, (min, max): (Vec3, Vec3)) -> bool {
    let inward = if v.corner & 1 != 0 { -0.5 } else { 0.5 } * quad;
    let (x, z) = if v.dir.faces_x() {
        (world.x, world.z + inward)
    } else {
        (world.x + inward, world.z)
    };
    let top = v.top as f32;
    x > min.x && x < max.x && z > min.z && z < max.z && top > min.y && top < max.y
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
/// low sides along x and z and `hi` quads inside its high ones.
///
/// **Why not simply sink everything to the box's floor**, as this did until
/// issue #424: the triangles from the box's side down to the floor are a wall
/// a kilometre from the eye on every side, from the surface to a kilometre
/// below it, and its lit side faces inward. Above ground the voxels hide it.
/// From under the ground nothing does -- the owner's striped curtain. Raised
/// instead, the same triangles face away from an eye inside the box and
/// back-face culling drops them before a single fragment is shaded, which a
/// `discard` would not (`far.wgsl`).
pub fn join(at: [i32; 2], lo: [f32; 2], hi: [f32; 2]) -> Join {
    let quads = FAR_QUADS as i32;
    let mut to = at;
    let mut snapped = false;
    for a in 0..2 {
        // The step to the nearest side along this axis, in quads.
        let step = if hi[a] < lo[a] { hi[a] } else { -lo[a] }.round() as i32;
        let target = at[a] + step;
        if step.abs() == 1 && (0..=quads).contains(&target) {
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

    /// The vertex grid the far terrain was drawn with while it was a smooth
    /// surface: a patch's 33 x 33 corners and a ring around them dropped into
    /// the skirt. Issue #424's tests of [`join`] and [`hidden_height`] below
    /// are written on it and are kept as they were: for a patch of unmerged
    /// tops it is still what a top's corners do.
    const GRID: u32 = FAR_QUADS as u32 + 3;

    /// Two triangles per cell of [`GRID`], wound as [`QUAD_INDICES`] is.
    fn grid_indices() -> Vec<u16> {
        let mut out = Vec::with_capacity(((GRID - 1) * (GRID - 1) * 6) as usize);
        for z in 0..GRID - 1 {
            for x in 0..GRID - 1 {
                let a = (z * GRID + x) as u16;
                let b = a + 1;
                let c = a + GRID as u16;
                let d = c + 1;
                out.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }
        out
    }

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

    /// The shader's `QUADS` and this module's must agree, or the join steps
    /// off the patch.
    #[test]
    fn the_shader_agrees_on_the_patch_size() {
        let wgsl = crate::haze::FAR_WGSL;
        assert!(wgsl.contains(&format!("const QUADS: i32 = {FAR_QUADS};")));
    }

    /// A face survives the three words it travels in, at the ends of every
    /// field as well as in the middle.
    #[test]
    fn a_face_packs_into_three_words_and_back() {
        let mut next = stream(0xFACE);
        for _ in 0..2_000 {
            let face = some_face(&mut next);
            assert_eq!(FarFace::unpack(face.pack()), face);
        }
        let edge = FarFace {
            dir: FarDir::NegZ,
            i0: 0,
            j0: 32,
            i1: 32,
            j1: 32,
            top: *FAR_TOP_RANGE.end(),
            rise: FAR_MAX_RISE,
            border: true,
            mix: [0, 0],
            open: 255,
        };
        assert_eq!(FarFace::unpack(edge.pack()), edge);
        let low = FarFace {
            top: *FAR_TOP_RANGE.start(),
            ..edge
        };
        assert_eq!(FarFace::unpack(low.pack()), low);
        // Beyond what fits: clamped, not wrapped into its neighbour's bits.
        let beyond = FarFace {
            top: 1_000_000,
            rise: 1_000_000,
            ..edge
        };
        assert_eq!(FarFace::unpack(beyond.pack()), edge);
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
            let p = face.corners().map(|v| place(&v, patch, None, 0.0));
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

    /// `far.wgsl` makes a face's corners from the vertex index; that is
    /// [`FarFace::shader_corner`], which must be [`FarFace::corners`] -- and
    /// the shader must still say what `shader_corner` does.
    #[test]
    fn the_shader_makes_the_corners_this_module_does() {
        let mut next = stream(0xC0FFEE);
        for _ in 0..2_000 {
            let face = some_face(&mut next);
            let corners = face.corners();
            for k in 0..4 {
                assert_eq!(
                    face.shader_corner(k),
                    corners[k as usize],
                    "{face:?} corner {k}"
                );
            }
        }
        let wgsl = crate::haze::FAR_WGSL;
        for line in [
            "let nth = vertex / 4u;",
            "let face = faces[run.first + nth];",
            "let k = vertex % 4u;",
            "let flip = u32(dir == DIR_NEG_X || dir == DIR_POS_Z);",
            "let e = (k & 1u) ^ flip;",
            "let u = k >> 1u;",
            "let corner = e | (u << 1u);",
            "var i = i32(i0 + e * di);",
            "var j = i32(j0 + select(u, e, wall) * dj);",
            // `FarVertex::y`.
            "y = top - rise - select(0.0, tile.skirt, border);",
            // `FarVertex::outward`.
            "var outward = vec3<f32>(su, 0.0, sv);",
            "outward = select(vec3<f32>(su, sv, 0.0), vec3<f32>(0.0, sv, su), faces_x);",
        ] {
            assert!(wgsl.contains(line), "far.wgsl no longer has: {line}");
        }
        // `FarDir`'s order is the shader's.
        for (dir, name) in [
            (FarDir::Top, "DIR_TOP"),
            (FarDir::PosX, "DIR_POS_X"),
            (FarDir::NegX, "DIR_NEG_X"),
            (FarDir::PosZ, "DIR_POS_Z"),
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
        assert!(
            wgsl.contains(&normals),
            "FACE_NORMALS is not FarDir's order"
        );
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

    /// Every run fits its class, no class is smaller than the one before,
    /// and the last holds a whole patch.
    #[test]
    fn every_run_has_a_class_that_holds_it() {
        assert!(CLASS_FACES.windows(2).all(|w| w[0] < w[1]));
        assert!(*CLASS_FACES.last().unwrap() as usize >= FAR_MAX_FACES);
        // The index buffer reaches the largest class's last vertex.
        assert!(*CLASS_FACES.last().unwrap() * 4 <= u16::MAX as u32 + 1);
        assert_eq!(quad_indices().len() as u32, CLASS_FACES.last().unwrap() * 6);
        for faces in 1..=FAR_MAX_FACES as u32 {
            let class = class_of(faces);
            assert!(faces <= CLASS_FACES[class], "{faces} in class {class}");
            assert!(
                class == 0 || faces > CLASS_FACES[class - 1],
                "{faces} would fit the class below {class}"
            );
        }
        // Three runs in class 0 and one in class 2: two draws, and every
        // run's whole class in vertices.
        let mut runs = [0usize; 17];
        runs[0] = 3;
        runs[2] = 1;
        assert_eq!(issued(&runs), (2, (3 * 16 + 32) * 4));
    }

    /// **Direction culling never drops a face the eye could see** (issue
    /// #422): over seeded eyes and faces, any face with a corner the eye is
    /// in front of is in a group [`groups_facing`] keeps -- and the runs
    /// [`facing_runs`] makes of the kept groups are exactly their faces.
    #[test]
    fn a_face_that_can_face_the_eye_is_always_drawn() {
        let mut next = stream(0xD1EC);
        let patch = ([64.0, -128.0], 4.0, 9.0);
        let (mut kept, mut dropped) = (0, 0);
        for _ in 0..400 {
            let faces: Vec<FarFace> = (0..40).map(|_| some_face(&mut next)).collect();
            let far_patch = FarPatch {
                origin: [64, -128],
                quad: 4.0,
                faces: &faces,
                skirt: 9.0,
            };
            let placed = PlacedPatch::of(&far_patch);
            let aabb = placed.aabb(anchor::ORIGIN);
            // Eyes around the patch, over it, and inside its box.
            let eye = Vec3::new(
                next(600) as f32 - 200.0,
                next(1200) as f32 - 300.0,
                next(600) as f32 - 400.0,
            );
            let facing = groups_facing(&aabb, eye);
            for face in &faces {
                let group = GROUP_ORDER.iter().position(|d| *d == face.dir).unwrap();
                let seen = face
                    .corners()
                    .iter()
                    .any(|v| face.dir.normal().dot(eye - place(v, patch, None, 0.0)) > 0.0);
                assert!(
                    !seen || facing[group],
                    "eye {eye}: {face:?} faces it and its group is not drawn"
                );
                if facing[group] {
                    kept += 1;
                } else {
                    dropped += 1;
                }
            }
            // The runs: every face of a kept group once, none of a dropped.
            let mut drawn = vec![0u32; faces.len()];
            facing_runs(placed.groups, facing, |first, count| {
                for face in first..first + count {
                    drawn[face as usize] += 1;
                }
            });
            let mut at = 0;
            for (group, &count) in placed.groups.iter().enumerate() {
                for face in at..at + count {
                    assert_eq!(drawn[face as usize], facing[group] as u32, "group {group}");
                }
                at += count;
            }
        }
        // And it does drop some: a rule that kept everything would pass the
        // above and cull nothing.
        assert!(
            dropped * 4 > kept,
            "only {dropped} of {} dropped",
            kept + dropped
        );
    }

    /// Faces go up grouped by direction, so a run is a direction or a few
    /// neighbouring ones.
    #[test]
    fn a_patchs_faces_are_stored_by_direction() {
        let mut next = stream(0x6E0);
        let faces: Vec<FarFace> = (0..200).map(|_| some_face(&mut next)).collect();
        let packed = pack_patch(&faces);
        assert_eq!(packed.len(), faces.len());
        let stored: Vec<FarFace> = packed.into_iter().map(FarFace::unpack).collect();
        let mut at = 0;
        for dir in GROUP_ORDER {
            let of_dir: Vec<FarFace> = faces.iter().copied().filter(|f| f.dir == dir).collect();
            assert_eq!(&stored[at..at + of_dir.len()], of_dir.as_slice(), "{dir:?}");
            at += of_dir.len();
        }
    }

    /// **A wall inside the voxels' box shows an eye inside it nothing**
    /// (issue #424's curtain, for the walls issue #422 adds). A patch of
    /// rolling ground straddling a side of the box, and one straddling a
    /// corner: every wall is in or out as a whole, and of a wall that is in,
    /// nothing is left above the box's floor.
    #[test]
    fn no_wall_stands_inside_the_box() {
        let hole = (
            Vec3::new(-1024.0, -1064.0, -1024.0),
            Vec3::new(1024.0, 984.0, 1024.0),
        );
        let quad = 8.0;
        let height = |i: i32, j: i32| 60 + 9 * ((i * 3 + j * 5) % 7);
        let mut inside = 0;
        for origin in [[896.0, -128.0], [896.0, 896.0], [-1152.0, 0.0]] {
            for line in 0..=FAR_QUADS as u8 {
                for k in 0..FAR_QUADS as u8 {
                    for along_x in [true, false] {
                        let (below, above) = if along_x {
                            (
                                height(line as i32 - 1, k as i32),
                                height(line as i32, k as i32),
                            )
                        } else {
                            (
                                height(k as i32, line as i32 - 1),
                                height(k as i32, line as i32),
                            )
                        };
                        if below == above {
                            continue;
                        }
                        let dir = match (along_x, below > above) {
                            (true, true) => FarDir::PosX,
                            (true, false) => FarDir::NegX,
                            (false, true) => FarDir::PosZ,
                            (false, false) => FarDir::NegZ,
                        };
                        let face = FarFace {
                            dir,
                            i0: if along_x { line } else { k },
                            j0: if along_x { k } else { line },
                            i1: if along_x { line } else { k + 1 },
                            j1: if along_x { k + 1 } else { line },
                            top: below.max(above),
                            rise: (below - above).unsigned_abs(),
                            border: false,
                            mix: [0, 0],
                            open: 255,
                        };
                        let corners = face.corners();
                        let raw = corners.map(|v| place(&v, (origin, quad, 40.0), None, 0.0));
                        let ins =
                            [0, 1, 2, 3].map(|c| wall_in_hole(&corners[c], raw[c], quad, hole));
                        assert!(
                            ins.iter().all(|&b| b == ins[0]),
                            "{face:?} at {origin:?}: its corners disagree, {ins:?}"
                        );
                        let p = corners.map(|v| place(&v, (origin, quad, 40.0), Some(hole), 0.0));
                        let middle = (raw[0] + raw[3]) / 2.0;
                        let strictly = middle.x > hole.0.x
                            && middle.x < hole.1.x
                            && middle.z > hole.0.z
                            && middle.z < hole.1.z;
                        assert_eq!(ins[0], strictly, "{face:?} at {origin:?}");
                        if strictly {
                            inside += 1;
                            assert!(
                                p.iter().all(|v| v.y <= hole.0.y),
                                "{face:?} at {origin:?} still stands in the box: {p:?}"
                            );
                        } else {
                            assert_eq!(p, raw, "a wall outside the box moved");
                        }
                    }
                }
            }
        }
        assert!(
            inside > 500,
            "only {inside} walls were inside, so this proves little"
        );

        // Ground under the box's floor is the far terrain's, wall and all:
        // a wall from above the floor to below it keeps the part below.
        let floor = (Vec3::new(-1024.0, 50.0, -1024.0), hole.1);
        let face = FarFace {
            dir: FarDir::PosX,
            i0: 8,
            j0: 4,
            i1: 8,
            j1: 5,
            top: 70,
            rise: 30,
            border: false,
            mix: [0, 0],
            open: 255,
        };
        let p = face
            .corners()
            .map(|v| place(&v, ([0.0, 0.0], quad, 40.0), Some(floor), 0.0));
        let ys: Vec<f32> = p.iter().map(|v| v.y).collect();
        assert_eq!(ys, vec![40.0, 40.0, 50.0, 50.0]);
    }

    /// A top's corners through [`place`] are [`join`]'s rule, on the faces
    /// the mesher makes against the box's side: the row of single quads
    /// inside it folds down into a skirt on the side, and nothing else of a
    /// top is left inside the box facing an eye inside it.
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
        // Outside, against the side: untouched.
        let outside = unit_top(16, 4, 70).corners();
        for v in outside {
            assert_eq!(
                place(&v, patch, Some(hole), hidden),
                place(&v, patch, None, 0.0)
            );
        }
        // Quad 15 is the first inside: a strip of it, five quads long.
        let strip = FarFace {
            j1: 9,
            ..unit_top(15, 4, 70)
        };
        let p = strip
            .corners()
            .map(|v| place(&v, patch, Some(hole), hidden));
        // Its corners on the side stay; the two a quad inside drop under
        // them: a wall on the side, from the top down a skirt's depth.
        for v in p {
            assert_eq!(v.x, 1024.0, "{p:?}");
        }
        let ys: Vec<f32> = p.iter().map(|v| v.y).collect();
        assert_eq!(ys, vec![30.0, 70.0, 30.0, 70.0]);
        let normal = (p[2] - p[0]).cross(p[1] - p[0]).normalize();
        assert_eq!(normal, Vec3::NEG_X, "the skirt faces into the box");
        // Deeper in: hidden, all four at the hidden height.
        let deep = FarFace {
            i0: 9,
            i1: 13,
            ..strip
        };
        for v in deep.corners() {
            assert_eq!(place(&v, patch, Some(hole), hidden).y, hidden);
        }
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

    /// `far.wgsl`'s `vs_main`, as far as a vertex's position: one patch of
    /// `quad`-wide quads with its corner on `origin`, heights from `height`.
    fn vertex(
        grid: [u32; 2],
        (origin, quad, skirt_depth): ([f32; 2], f32, f32),
        height: &dyn Fn(i32, i32) -> f32,
        hole: (Vec3, Vec3),
        hidden: f32,
        rule: &dyn Fn([i32; 2], [f32; 2], [f32; 2]) -> Join,
    ) -> Vec3 {
        let quads = FAR_QUADS as i32;
        let i = (grid[0] as i32 - 1).clamp(0, quads);
        let j = (grid[1] as i32 - 1).clamp(0, quads);
        let skirt = grid[0] == 0 || grid[0] == GRID - 1 || grid[1] == 0 || grid[1] == GRID - 1;
        let at = |i: i32, j: i32| [origin[0] + i as f32 * quad, origin[1] + j as f32 * quad];
        let [x, z] = at(i, j);
        let y = height(i, j) - if skirt { skirt_depth } else { 0.0 };
        let world = Vec3::new(x, y, z);
        if !(world.cmpgt(hole.0).all() && world.cmplt(hole.1).all()) {
            return world;
        }
        let lo = [(x - hole.0.x) / quad, (z - hole.0.z) / quad];
        let hi = [(hole.1.x - x) / quad, (hole.1.z - z) / quad];
        match rule([i, j], lo, hi) {
            Join::Skirt([i, j]) => {
                let [x, z] = at(i, j);
                Vec3::new(x, height(i, j) - skirt_depth, z)
            }
            Join::Hidden => Vec3::new(x, hidden, z),
        }
    }

    /// What this did until issue #424: every vertex inside the box sunk to
    /// its floor. Kept here so the test below is seen to tell the two apart.
    fn sink_everything(_: [i32; 2], _: [f32; 2], _: [f32; 2]) -> Join {
        Join::Hidden
    }

    /// Every triangle of a patch that has a vertex strictly inside the box
    /// and that an eye at `eye` would be shown, as its three corners.
    fn seen_inside_the_box(
        origin: [f32; 2],
        hole: (Vec3, Vec3),
        eye: Vec3,
        hidden: Option<f32>,
        rule: &dyn Fn([i32; 2], [f32; 2], [f32; 2]) -> Join,
    ) -> Vec<[Vec3; 3]> {
        let quad = 8.0;
        let skirt_depth = 40.0;
        // Rolling ground, so no two neighbours are level.
        let height = |i: i32, j: i32| 60.0 + 9.0 * ((i * 3 + j * 5) % 7) as f32;
        // Ground well above the floor: what an eye inside the box has
        // raised out of sight, unless the old rule's height is asked for.
        let hidden = hidden.unwrap_or_else(|| hidden_height(hole.0, hole.1, eye, 20.0));
        let place = |index: u16| {
            let grid = [index as u32 % GRID, index as u32 / GRID];
            let patch = (origin, quad, skirt_depth);
            vertex(grid, patch, &height, hole, hidden, rule)
        };
        let strictly_inside =
            |p: Vec3| p.x > hole.0.x && p.x < hole.1.x && p.z > hole.0.z && p.z < hole.1.z;
        let mut seen = Vec::new();
        for t in grid_indices().chunks(3) {
            let [a, b, c] = [place(t[0]), place(t[1]), place(t[2])];
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
    /// box shows the eye its front. The one exception is on the very corner,
    /// where the two sides' skirts meet: two triangles within two quads of
    /// it and a skirt's depth of the surface, which is under the voxels'
    /// ground like the skirts themselves.
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
                let seen = seen_inside_the_box(origin, hole, eye, None, &join);
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
                let before =
                    seen_inside_the_box(origin, hole, eye, Some(hole.0.y - 1.0), &sink_everything);
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
        let height = |i: i32, j: i32| 60.0 + 60.0 * ((i * 3 + j * 5) % 7) as f32;
        // The tallest triangle of the ground as it is: six steps and a skirt.
        let bound = 6.0 * 60.0 + skirt_depth + 1.0;
        for eye_y in [1_100.0, 1_400.0] {
            let eye = Vec3::new(8.0, eye_y, 8.0);
            let hole = (
                Vec3::new(-1024.0, eye_y - 1024.0, -1024.0),
                Vec3::new(1024.0, eye_y + 1024.0, 1024.0),
            );
            // In the middle of the box, across a side of it, and on a corner.
            for origin in [[0.0, 0.0], [896.0, -128.0], [896.0, 896.0]] {
                let lowest = (0..=FAR_QUADS as i32)
                    .flat_map(|i| (0..=FAR_QUADS as i32).map(move |j| (i, j)))
                    .map(|(i, j)| height(i, j))
                    .fold(f32::INFINITY, f32::min)
                    - skirt_depth;
                assert!(
                    lowest < hole.0.y && hole.0.y < 420.0,
                    "the floor does not cut the ground, so this proves nothing"
                );
                let hidden = hidden_height(hole.0, hole.1, eye, lowest);
                let place = |index: u16| {
                    let grid = [index as u32 % GRID, index as u32 / GRID];
                    let patch = (origin, quad, skirt_depth);
                    vertex(grid, patch, &height, hole, hidden, &join)
                };
                for t in grid_indices().chunks(3) {
                    let [a, b, c] = [place(t[0]), place(t[1]), place(t[2])];
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

    /// The seam to the voxels stays closed: the first ring inside the box
    /// hangs straight under the vertex on the box's side, a skirt deep --
    /// and only the first, and only from a side this patch holds.
    #[test]
    fn the_first_ring_inside_is_a_skirt_on_the_boxs_side() {
        // 3 quads inside the low x side, far from the others.
        assert_eq!(join([11, 5], [3.0, 40.0], [253.0, 216.0]), Join::Hidden);
        assert_eq!(
            join([9, 5], [1.0, 40.0], [255.0, 216.0]),
            Join::Skirt([8, 5])
        );
        // The high sides step the other way.
        assert_eq!(
            join([9, 5], [255.0, 216.0], [1.0, 40.0]),
            Join::Skirt([10, 5])
        );
        assert_eq!(
            join([9, 5], [40.0, 255.0], [216.0, 1.0]),
            Join::Skirt([9, 6])
        );
        // On a corner, both: straight under the corner vertex.
        assert_eq!(
            join([9, 5], [1.0, 1.0], [255.0, 255.0]),
            Join::Skirt([8, 4])
        );
        // A side one quad beyond this patch's edge is the next patch's to
        // skirt; the side this patch does hold still gets its own.
        assert_eq!(join([0, 5], [1.0, 40.0], [255.0, 216.0]), Join::Hidden);
        assert_eq!(join([32, 5], [255.0, 40.0], [1.0, 216.0]), Join::Hidden);
        assert_eq!(
            join([0, 5], [1.0, 1.0], [255.0, 255.0]),
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
            "let lo = (world.xz - params.hole_min.xz) / tile.quad;",
            "let hi = (params.hole_max.xz - world.xz) / tile.quad;",
            "let step = vec2<i32>(round(select(-lo, hi, hi < lo)));",
            "let snap_x = abs(step.x) == 1 && ti >= 0 && ti <= QUADS;",
            "let snap_z = abs(step.y) == 1 && tj >= 0 && tj <= QUADS;",
            "if snap_x || snap_z {",
            "i = select(i, ti, snap_x);",
            "j = select(j, tj, snap_z);",
            // A snapped corner hangs a skirt under its own face's height.
            "y - tile.skirt,",
            // `hidden_height`, worked out on the CPU (`FarTerrain::prepare`).
            "world.y = params.hole_min.w;",
            // `wall_in_hole`, and what `place` does with it.
            "let inward = select(0.5, -0.5, (corner & 1u) != 0u) * tile.quad;",
            "let middle = world.xz + select(vec2<f32>(inward, 0.0), vec2<f32>(0.0, inward), faces_x);",
            "if all(middle > params.hole_min.xz) && all(middle < params.hole_max.xz)",
            "&& top > params.hole_min.y && top < params.hole_max.y {",
            "world.y = min(world.y, params.hole_min.y);",
            // ...and a wall cut off there is not sealed back open.
            "seal = 0.0;",
            "let sealed = world + outward * (seal * distance(world, frame.eye.xyz));",
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

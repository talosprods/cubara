//! CPU-side mesh data shared between world generation and the renderer.
//!
//! Pure data: this crate knows nothing about the GPU (`ARCHITECTURE.md` Rule 3/4),
//! so the world can be generated and meshed headlessly, with no adapter and no
//! `wgpu`. The matching `wgpu` vertex layout for [`Vertex`] lives with the code
//! that owns pipelines, in `cubara-render`.

/// Which of the six axis-aligned directions a greedy-meshed face points.
///
/// 3 bits is enough to pack because a greedy-meshed voxel face is always
/// exactly one of these -- the shader looks the normal up from this rather
/// than carrying it as vertex data (`docs/PHASE1_ARCHITECTURE.md` §5.2).
#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Face {
    PosX = 0,
    NegX = 1,
    PosY = 2,
    NegY = 3,
    PosZ = 4,
    NegZ = 5,
}

impl Face {
    /// The order [`Mesh::group_by_face`] puts a mesh's quads in, and so the
    /// order of [`Mesh::face_indices`]; a renderer drawing one run per stretch
    /// of directions that can face the camera reads it in the same order.
    ///
    /// No two directions of one axis are neighbours. A camera outside a
    /// node's box sees one direction of each axis, and from all eight corners
    /// two of those three are adjacent here: at most two runs, 1.75 on
    /// average. In `Face` order each axis's pair is adjacent, and the three
    /// seen are never: up to three runs, 2.5 on average.
    pub const DRAW_ORDER: [Face; 6] = [
        Face::PosX,
        Face::PosY,
        Face::PosZ,
        Face::NegY,
        Face::NegZ,
        Face::NegX,
    ];

    /// The face for a mesher axis (0=x, 1=y, 2=z) and sign (+1 or -1).
    pub fn from_axis_sign(axis: usize, sign: i8) -> Self {
        match (axis, sign) {
            (0, 1) => Face::PosX,
            (0, -1) => Face::NegX,
            (1, 1) => Face::PosY,
            (1, -1) => Face::NegY,
            (2, 1) => Face::PosZ,
            (2, -1) => Face::NegZ,
            _ => panic!("axis must be 0..3 and sign must be ±1 (got axis={axis}, sign={sign})"),
        }
    }

    /// The unit outward normal, for callers that need it as a vector (tests;
    /// anything off the GPU path). The shader has its own copy of this table.
    pub fn normal(self) -> [f32; 3] {
        match self {
            Face::PosX => [1.0, 0.0, 0.0],
            Face::NegX => [-1.0, 0.0, 0.0],
            Face::PosY => [0.0, 1.0, 0.0],
            Face::NegY => [0.0, -1.0, 0.0],
            Face::PosZ => [0.0, 0.0, 1.0],
            Face::NegZ => [0.0, 0.0, -1.0],
        }
    }

    fn from_bits(bits: u32) -> Self {
        match bits {
            0 => Face::PosX,
            1 => Face::NegX,
            2 => Face::PosY,
            3 => Face::NegY,
            4 => Face::PosZ,
            5 => Face::NegZ,
            other => panic!("invalid packed face bits: {other}"),
        }
    }
}

const POS_BITS: u32 = 10;
const POS_MASK: u32 = (1 << POS_BITS) - 1;
const AO_MASK: u32 = 0b11;
const TEX_LAYER_BITS: u32 = 12;
const TEX_LAYER_MASK: u32 = (1 << TEX_LAYER_BITS) - 1;
const FACE_BITS: u32 = 3;
const FACE_MASK: u32 = 0b111;
const EXTENT_BITS: u32 = 8;
const EXTENT_MASK: u32 = (1 << EXTENT_BITS) - 1;
const FACE_SHIFT: u32 = TEX_LAYER_BITS;
const U_SHIFT: u32 = TEX_LAYER_BITS + FACE_BITS;
const V_SHIFT: u32 = TEX_LAYER_BITS + FACE_BITS + EXTENT_BITS;
/// Matches `NodeIndexAllocator`'s `MAX_NODES` in `cubara-render`'s arena; 16
/// bits is generous headroom over that cap.
const NODE_INDEX_BITS: u32 = 16;
const NODE_INDEX_MASK: u32 = (1 << NODE_INDEX_BITS) - 1;
/// Which side of its quad a corner is on, along the two sweep axes
/// ([`Vertex::with_corner`]): one bit each, above the node index.
const CORNER_U_BIT: u32 = 1 << NODE_INDEX_BITS;
const CORNER_V_BIT: u32 = 1 << (NODE_INDEX_BITS + 1);
/// log2 of this face's cell width, for scaling its AO gradient
/// ([`Vertex::with_ao_width`]) -- `docs/PROPOSAL_LOD_TRANSITIONS.md` §3 R1's
/// "required alongside it" (issue #327, decision 5). Two bits: a level-3
/// node's horizontal cell is 8 blocks wide (log2 = 3), the widest this ever
/// needs to carry. 0 (the default every vertex starts at) means "AO already
/// spans one block" -- level 0, and every vertex nobody has called
/// `with_ao_width` on, read back unscaled.
const AO_WIDTH_BITS: u32 = 2;
const AO_WIDTH_SHIFT: u32 = NODE_INDEX_BITS + 2;
const AO_WIDTH_MASK: u32 = (1 << AO_WIDTH_BITS) - 1;
/// The widest cell [`Vertex::with_ao_width`] can record: 2 bits, so log2 up
/// to 3 (an 8-block cell, a level-3 node's horizontal width -- the widest R1b
/// ever asks for). A caller computing a wider log2 itself (e.g. `voxel.rs`'s
/// old isotropic `Chunk::build_mesh_lod`, whose downsample factor can reach
/// 16) clamps to this rather than overflowing the field.
pub const MAX_AO_WIDTH_LOG2: u32 = AO_WIDTH_MASK;

/// A single mesh vertex, packed into 12 bytes: node-local lattice position,
/// baked ambient occlusion, texture-array layer, face direction, this
/// vertex's own texel-tile coordinate, and the arena node it belongs to
/// (`docs/PHASE1_ARCHITECTURE.md` §5.2/§5.3).
///
/// ```text
/// word 0:  x:10  y:10  z:10  ao:2
/// word 1:  tex_layer:12  face:3  u:8  v:8  (1 spare)
/// word 2:  node_index:16  corner_u:1  corner_v:1  ao_width_log2:2  (12 spare)
/// ```
///
/// `u`/`v` are *this vertex's* coordinate -- 0 or the greedy quad's extent,
/// depending on which of the four corners it is -- not a value shared by the
/// whole quad. That is what makes the texture tile once per grid cell across
/// a merged face: each corner samples at its own integer tile coordinate.
///
/// Positions are node-local, not world-space; placing a chunk in the world is
/// a GPU-side per-node origin add, keyed by `node_index` (word 2). That index
/// is unknown at mesh-build time -- it is assigned when a mesh lands in the
/// render arena -- so it starts at 0 from [`Vertex::new`] and is patched in
/// later with [`Vertex::with_node_index`].
///
/// `node_index` lives in vertex data rather than `@builtin(instance_index)`
/// because neither `first_instance` (via indirect multi-draw) nor the plain
/// `draw_indexed` instance-range fallback proved reliable across every real
/// backend tested in CI -- see §5.3 for the measured cross-platform failures
/// that led here.
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    packed0: u32,
    packed1: u32,
    packed2: u32,
}

impl Vertex {
    /// `x`/`y`/`z` are node-local lattice coordinates (0..=1023 fits; a full
    /// chunk only ever uses 0..=16). `ao` is 0..=3 (see the voxel crate's
    /// `vertex_ao`). `u`/`v` are this corner's own tile coordinate.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        x: u32,
        y: u32,
        z: u32,
        ao: u32,
        face: Face,
        tex_layer: u32,
        u: u32,
        v: u32,
    ) -> Self {
        debug_assert!(
            x <= POS_MASK && y <= POS_MASK && z <= POS_MASK,
            "position out of range"
        );
        debug_assert!(ao <= AO_MASK, "ao out of range");
        debug_assert!(tex_layer <= TEX_LAYER_MASK, "texture layer out of range");
        debug_assert!(u <= EXTENT_MASK && v <= EXTENT_MASK, "u/v out of range");
        let packed0 = x | (y << POS_BITS) | (z << (2 * POS_BITS)) | (ao << (3 * POS_BITS));
        let packed1 = tex_layer | ((face as u32) << FACE_SHIFT) | (u << U_SHIFT) | (v << V_SHIFT);
        Self {
            packed0,
            packed1,
            packed2: 0,
        }
    }

    /// Returns a copy of this vertex with its arena node index set. Called
    /// once per vertex when a mesh is inserted into the render arena, since
    /// the index isn't known at mesh-build time.
    pub fn with_node_index(self, node_index: u32) -> Self {
        debug_assert!(node_index <= NODE_INDEX_MASK, "node index out of range");
        Self {
            packed2: (self.packed2 & !NODE_INDEX_MASK) | node_index,
            ..self
        }
    }

    /// Returns a copy of this vertex marked as its quad's corner on the
    /// high side of the first sweep axis (`u_max`) and of the second
    /// (`v_max`) -- for a face along axis `d`, the axes `(d + 1) % 3` and
    /// `(d + 2) % 3`. What lets `mesh.wgsl` push each corner a fraction of a
    /// pixel away from its quad, closing the cracks a greedy mesh's
    /// T-junctions leave (the sparkles of block 3.0a).
    pub fn with_corner(self, u_max: bool, v_max: bool) -> Self {
        let bits = if u_max { CORNER_U_BIT } else { 0 } | if v_max { CORNER_V_BIT } else { 0 };
        Self {
            packed2: (self.packed2 & !(CORNER_U_BIT | CORNER_V_BIT)) | bits,
            ..self
        }
    }

    /// The corner [`with_corner`](Self::with_corner) set: `(u_max, v_max)`.
    pub fn corner(self) -> (bool, bool) {
        (
            self.packed2 & CORNER_U_BIT != 0,
            self.packed2 & CORNER_V_BIT != 0,
        )
    }

    /// Returns a copy of this vertex carrying `log2_width` (the log2 of this
    /// face's cell width in blocks) in word 2's spare bits 18-19, for
    /// `mesh.wgsl` to scale its AO gradient by -- without this, a coarse
    /// riser's AO band stays one block wide regardless of how many blocks
    /// its cell actually spans, so coarse rings read darker than their own
    /// blocks would (`docs/PROPOSAL_LOD_TRANSITIONS.md` §3 R1, issue #327
    /// decision 5). Independent of [`with_corner`]/[`with_node_index`]:
    /// setting any of the three keeps the other two.
    pub fn with_ao_width(self, log2_width: u32) -> Self {
        debug_assert!(log2_width <= AO_WIDTH_MASK, "ao width log2 out of range");
        Self {
            packed2: (self.packed2 & !(AO_WIDTH_MASK << AO_WIDTH_SHIFT))
                | (log2_width << AO_WIDTH_SHIFT),
            ..self
        }
    }

    /// The log2 cell width [`with_ao_width`](Self::with_ao_width) set; 0 (one
    /// block) for a vertex nobody has called it on.
    pub fn ao_width_log2(self) -> u32 {
        (self.packed2 >> AO_WIDTH_SHIFT) & AO_WIDTH_MASK
    }

    pub fn node_index(self) -> u32 {
        self.packed2 & NODE_INDEX_MASK
    }

    pub fn x(self) -> u32 {
        self.packed0 & POS_MASK
    }

    pub fn y(self) -> u32 {
        (self.packed0 >> POS_BITS) & POS_MASK
    }

    pub fn z(self) -> u32 {
        (self.packed0 >> (2 * POS_BITS)) & POS_MASK
    }

    pub fn ao(self) -> u32 {
        (self.packed0 >> (3 * POS_BITS)) & AO_MASK
    }

    pub fn tex_layer(self) -> u32 {
        self.packed1 & TEX_LAYER_MASK
    }

    pub fn face(self) -> Face {
        Face::from_bits((self.packed1 >> FACE_SHIFT) & FACE_MASK)
    }

    pub fn u(self) -> u32 {
        (self.packed1 >> U_SHIFT) & EXTENT_MASK
    }

    pub fn v(self) -> u32 {
        (self.packed1 >> V_SHIFT) & EXTENT_MASK
    }
}

/// An indexed triangle mesh ready to be uploaded to the GPU. Vertices are
/// node-local (see [`Vertex`]); world placement is a GPU-side per-node
/// origin, not a CPU translate.
#[derive(Default)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    /// How many of `indices` face each way, in [`Face::DRAW_ORDER`], once
    /// [`group_by_face`](Self::group_by_face) has put them in that order: the
    /// first `face_indices[0]` indices are `DRAW_ORDER[0]` quads, and so on.
    ///
    /// What lets a renderer skip every face pointing away from the camera
    /// without running a vertex of it. All zero for a mesh that was never
    /// grouped, which a renderer must then draw whole.
    pub face_indices: [u32; 6],
}

impl Mesh {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Reorder the quads so all of one [`Face`] direction come together, in
    /// [`Face::DRAW_ORDER`], and record the counts in `face_indices`. Every
    /// quad is six indices into four vertices of one face (what the greedy
    /// mesher emits); the order within a direction is kept.
    pub fn group_by_face(&mut self) {
        debug_assert_eq!(self.indices.len() % 6, 0, "a mesh of whole quads");
        let mut groups: [Vec<u32>; 6] = Default::default();
        for quad in self.indices.chunks(6) {
            let face = self.vertices[quad[0] as usize].face() as usize;
            groups[face].extend_from_slice(quad);
        }
        self.indices.clear();
        for (k, face) in Face::DRAW_ORDER.into_iter().enumerate() {
            let group = &groups[face as usize];
            self.face_indices[k] = group.len() as u32;
            self.indices.extend_from_slice(group);
        }
    }

    /// Whether `face_indices` accounts for every index.
    pub fn is_grouped_by_face(&self) -> bool {
        self.face_indices.iter().map(|&n| n as usize).sum::<usize>() == self.indices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_is_twelve_bytes() {
        assert_eq!(std::mem::size_of::<Vertex>(), 12);
    }

    #[test]
    #[allow(clippy::type_complexity)]
    fn pack_unpack_round_trips_every_field_at_its_limits() {
        let faces = [
            Face::PosX,
            Face::NegX,
            Face::PosY,
            Face::NegY,
            Face::PosZ,
            Face::NegZ,
        ];
        let cases: [(u32, u32, u32, u32, u32, u32, u32, u32); 4] = [
            (0, 0, 0, 0, 0, 0, 0, 0),
            (
                POS_MASK,
                POS_MASK,
                POS_MASK,
                AO_MASK,
                TEX_LAYER_MASK,
                EXTENT_MASK,
                EXTENT_MASK,
                NODE_INDEX_MASK,
            ),
            (17, 512, 999, 2, 4095, 200, 3, 12345),
            (1, 2, 3, 1, 1, 255, 0, 1),
        ];
        for (x, y, z, ao, tex_layer, u, v, node_index) in cases {
            for face in faces {
                let vertex =
                    Vertex::new(x, y, z, ao, face, tex_layer, u, v).with_node_index(node_index);
                assert_eq!(vertex.x(), x, "x for {face:?}");
                assert_eq!(vertex.y(), y, "y for {face:?}");
                assert_eq!(vertex.z(), z, "z for {face:?}");
                assert_eq!(vertex.ao(), ao, "ao for {face:?}");
                assert_eq!(vertex.tex_layer(), tex_layer, "tex_layer for {face:?}");
                assert_eq!(vertex.face(), face, "face round-trip");
                assert_eq!(vertex.u(), u, "u for {face:?}");
                assert_eq!(vertex.v(), v, "v for {face:?}");
                assert_eq!(vertex.node_index(), node_index, "node_index for {face:?}");
            }
        }
    }

    /// The corner bits sit above the node index: setting either keeps the
    /// other, in both orders.
    #[test]
    fn corner_and_node_index_keep_each_other() {
        let v = Vertex::new(1, 2, 3, 1, Face::PosZ, 7, 4, 5);
        let a = v.with_corner(true, false).with_node_index(NODE_INDEX_MASK);
        assert_eq!(
            (a.corner(), a.node_index()),
            ((true, false), NODE_INDEX_MASK)
        );
        let b = v.with_node_index(12345).with_corner(false, true);
        assert_eq!((b.corner(), b.node_index()), ((false, true), 12345));
        assert_eq!(b.with_corner(true, true).corner(), (true, true));
        assert_eq!(v.corner(), (false, false));
    }

    /// `ao_width_log2` sits above the corner bits, above the node index:
    /// setting any one of the three keeps the other two, in every order --
    /// what R1c's AO-width scaling depends on not disturbing the crack-seal
    /// corner bits (#321) or the node index a mesh gets stamped with on
    /// arena insertion.
    #[test]
    fn ao_width_corner_and_node_index_all_keep_each_other() {
        let v = Vertex::new(1, 2, 3, 1, Face::PosZ, 7, 4, 5);
        let a = v
            .with_ao_width(3)
            .with_corner(true, false)
            .with_node_index(NODE_INDEX_MASK);
        assert_eq!(
            (a.ao_width_log2(), a.corner(), a.node_index()),
            (3, (true, false), NODE_INDEX_MASK)
        );
        let b = v
            .with_node_index(12345)
            .with_corner(false, true)
            .with_ao_width(2);
        assert_eq!(
            (b.ao_width_log2(), b.corner(), b.node_index()),
            (2, (false, true), 12345)
        );
        assert_eq!(v.ao_width_log2(), 0, "default is 0 (one block)");
        assert_eq!(b.with_ao_width(0).ao_width_log2(), 0);
    }

    #[test]
    fn fields_do_not_bleed_into_each_other() {
        // Set only one field at a time to a nonzero value; every other field
        // must read back as zero -- proves the bit ranges don't overlap.
        let x_only = Vertex::new(POS_MASK, 0, 0, 0, Face::PosX, 0, 0, 0);
        assert_eq!((x_only.y(), x_only.z(), x_only.ao()), (0, 0, 0));

        let y_only = Vertex::new(0, POS_MASK, 0, 0, Face::PosX, 0, 0, 0);
        assert_eq!((y_only.x(), y_only.z(), y_only.ao()), (0, 0, 0));

        let z_only = Vertex::new(0, 0, POS_MASK, 0, Face::PosX, 0, 0, 0);
        assert_eq!((z_only.x(), z_only.y(), z_only.ao()), (0, 0, 0));

        let ao_only = Vertex::new(0, 0, 0, AO_MASK, Face::PosX, 0, 0, 0);
        assert_eq!((ao_only.x(), ao_only.y(), ao_only.z()), (0, 0, 0));

        let tex_only = Vertex::new(0, 0, 0, 0, Face::PosX, TEX_LAYER_MASK, 0, 0);
        assert_eq!(tex_only.face(), Face::PosX);
        assert_eq!((tex_only.u(), tex_only.v()), (0, 0));

        let u_only = Vertex::new(0, 0, 0, 0, Face::PosX, 0, EXTENT_MASK, 0);
        assert_eq!((u_only.tex_layer(), u_only.v()), (0, 0));

        let v_only = Vertex::new(0, 0, 0, 0, Face::PosX, 0, 0, EXTENT_MASK);
        assert_eq!((v_only.tex_layer(), v_only.u()), (0, 0));

        let node_index_only =
            Vertex::new(0, 0, 0, 0, Face::PosX, 0, 0, 0).with_node_index(NODE_INDEX_MASK);
        assert_eq!(
            (
                node_index_only.x(),
                node_index_only.y(),
                node_index_only.z(),
                node_index_only.ao(),
                node_index_only.tex_layer(),
                node_index_only.u(),
                node_index_only.v(),
            ),
            (0, 0, 0, 0, 0, 0, 0)
        );
        assert_eq!(node_index_only.face(), Face::PosX);
    }

    #[test]
    fn face_from_axis_sign_covers_all_six_directions() {
        assert_eq!(Face::from_axis_sign(0, 1), Face::PosX);
        assert_eq!(Face::from_axis_sign(0, -1), Face::NegX);
        assert_eq!(Face::from_axis_sign(1, 1), Face::PosY);
        assert_eq!(Face::from_axis_sign(1, -1), Face::NegY);
        assert_eq!(Face::from_axis_sign(2, 1), Face::PosZ);
        assert_eq!(Face::from_axis_sign(2, -1), Face::NegZ);
    }
}

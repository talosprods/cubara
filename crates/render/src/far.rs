//! The far terrain's GPU half (`docs/PROPOSAL_FAR_VIEW.md` §3.2): where the
//! patches' heights live on the GPU, which of them are drawn this frame, and
//! the parameters `far.wgsl` reads.
//!
//! Which patches exist and what heights they hold is decided elsewhere
//! (`cubara_world::far`), and handed over as plain numbers -- this crate does
//! not depend on the world crate (Rule 3), so a patch arrives as a
//! [`FarPatch`] and leaves as a [`FarSlot`].

use crate::aggregate::{MaskingTable, Materials, TABLE_SIZE};
use crate::anchor::{self, Anchor};
use crate::culling::{Aabb, Frustum};
use crate::render::far_bind_group_layout;
use glam::Vec3;

/// Quads along one side of a patch. Must match `far.wgsl`'s `QUADS` and
/// `cubara_world::far::PATCH_QUADS`; the app asserts the second.
pub const FAR_QUADS: usize = 32;

/// Heights along one side of a patch.
pub const FAR_VERTS: usize = FAR_QUADS + 1;

/// The vertex grid one patch is drawn with: its own vertices plus a ring
/// around them that `far.wgsl` drops into the skirt.
const GRID: u32 = FAR_VERTS as u32 + 2;

/// One patch, as the renderer needs it: where it is, how coarse, and its
/// heights row by row (`z` outer, `x` inner), [`FAR_VERTS`] a side.
#[derive(Clone, Copy, Debug)]
pub struct FarPatch<'a> {
    /// The lowest `(x, z)` corner, in blocks -- the integer it is
    /// (`cubara_world::far::PatchKey::origin`), so the render anchor is
    /// subtracted from it exactly (issue #379).
    pub origin: [i64; 2],
    /// The width of one quad, in blocks.
    pub quad: f32,
    pub heights: &'a [f32],
    /// How far the skirt drops below the edge, in blocks: deep enough to
    /// cover the gap to a coarser neighbour.
    pub skirt: f32,
}

/// Where a patch was put, to remove it by later.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FarSlot(u32);

/// What `far.wgsl` reads besides the heights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FarParams {
    /// The box the voxel rings draw themselves, `(min, max)` in blocks --
    /// integers (chunk edges), subtracted from exactly like a tile origin.
    /// Nothing of the far terrain is drawn inside it. `None`: no hole.
    pub hole: Option<([i32; 3], [i32; 3])>,
    /// What the ground is made of: the mean colours of its faces
    /// ([`crate::materials::mean_color`]) and the soil's depth, for the
    /// block-aggregate shading (`crate::aggregate`).
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

/// The committed [`MaskingTable`] as the 3D texture `far.wgsl` samples: its
/// first two axes as width and height, its last two stacked as depth
/// (`axis 2 + 16 * axis 3`). That is exactly the table's own byte order, so
/// the bytes go up as they are. Linear filtering interpolates the first three
/// axes; the shader blends along the fourth itself, never across a stack's
/// edge.
fn masking_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::TextureView, wgpu::Sampler) {
    let side = TABLE_SIZE as u32;
    let size = wgpu::Extent3d {
        width: side,
        height: side,
        depth_or_array_layers: side * side,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("far-masking-table"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        MaskingTable::committed().bytes(),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(side),
            rows_per_image: Some(side),
        },
        size,
    );
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("far-masking-sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    (texture.create_view(&Default::default()), sampler)
}

/// The far terrain on the GPU: a fixed number of patch slots, and the list of
/// them drawn this frame.
pub struct FarTerrain {
    heights: wgpu::Buffer,
    patches: wgpu::Buffer,
    draw_slots: wgpu::Buffer,
    params: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
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

/// A patch as it was inserted, minus its heights (which the anchor does not
/// touch): enough to rewrite its `PatchGpu` record and its box.
#[derive(Clone, Copy, Debug)]
struct PlacedPatch {
    origin: [i64; 2],
    quad: f32,
    skirt: f32,
    /// Lowest and highest height.
    lo: f32,
    hi: f32,
}

impl PlacedPatch {
    fn of(patch: &FarPatch<'_>) -> Self {
        let (lo, hi) = min_max(patch.heights);
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
    /// bottom of its skirt to its highest vertex. The heights are absolute
    /// (the `heights` buffer is untouched by a rebase), so the anchor's `y`
    /// comes off here, as `far.wgsl` takes it off every vertex.
    fn aabb(&self, anchor: Anchor) -> Aabb {
        let [ox, oz] = self.origin_relative(anchor);
        let size = self.quad * FAR_QUADS as f32;
        let dy = anchor[1] as f32;
        Aabb {
            min: Vec3::new(ox, self.lo - self.skirt - dy, oz),
            max: Vec3::new(ox + size, self.hi - dy, oz + size),
        }
    }
}

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
        let per_patch = (FAR_VERTS * FAR_VERTS * std::mem::size_of::<f32>()) as u64;
        let heights = storage("far-heights", per_patch * capacity as u64);
        let patches = storage(
            "far-patches",
            std::mem::size_of::<PatchGpu>() as u64 * capacity as u64,
        );
        let draw_slots = storage("far-draw-slots", 4 * capacity as u64);
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
        let (masking, masking_sampler) = masking_texture(device, queue);

        let grid = grid_indices();
        let indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("far-grid-indices"),
            // A multiple of 4, as a buffer write must be; the padding is
            // never indexed.
            size: (std::mem::size_of_val(grid.as_slice()) as u64).next_multiple_of(4),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut bytes = bytemuck::cast_slice::<u16, u8>(&grid).to_vec();
        bytes.resize(bytes.len().next_multiple_of(4), 0);
        queue.write_buffer(&indices, 0, &bytes);

        let layout = far_bind_group_layout(device);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("far-bind-group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: heights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: patches.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: draw_slots.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&masking),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&masking_sampler),
                },
            ],
        });
        Self {
            heights,
            patches,
            draw_slots,
            params,
            indices,
            index_count: grid.len() as u32,
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
    /// (issue #379 decision 2). The heights are untouched -- they stay
    /// absolute, and `far.wgsl` subtracts `Frame.anchor.y` from each; what
    /// is rewritten is one 16-byte record per slot -- all of them, in **one**
    /// `write_buffer` of `capacity * 16` bytes (64 KiB at the window's
    /// `FAR_PATCH_CAPACITY`), which costs less than a frame's own indirect
    /// upload. Not spread over frames: a patch drawn against the old anchor
    /// while the eye is measured from the new one would be 4,096 blocks
    /// off, the very thing the anchor exists to prevent; and not deferred to
    /// each patch's next upload, because a patch at the horizon may not be
    /// re-uploaded for minutes. The bounded single write is the cheapest
    /// form that is also correct on the rebase frame itself. A no-op for
    /// the anchor already in use.
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
        assert_eq!(
            patch.heights.len(),
            FAR_VERTS * FAR_VERTS,
            "a patch's heights"
        );
        let slot = self.free.pop()?;
        let per_patch = (FAR_VERTS * FAR_VERTS * std::mem::size_of::<f32>()) as u64;
        queue.write_buffer(
            &self.heights,
            per_patch * slot as u64,
            bytemuck::cast_slice(patch.heights),
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
    /// many. `frustum` and `eye` are anchor-relative, like the boxes
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
        if !order.is_empty() {
            queue.write_buffer(&self.draw_slots, 0, bytemuck::cast_slice(&order));
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

    /// Triangles drawn this frame, skirts included.
    pub fn drawn_triangles(&self) -> u64 {
        self.drawn as u64 * (self.index_count / 3) as u64
    }

    /// Draw this frame's patches. The pipeline and bind group 0 (the camera)
    /// are the caller's to set.
    pub(crate) fn encode(&self, pass: &mut wgpu::RenderPass<'_>) {
        if self.drawn == 0 {
            return;
        }
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..self.index_count, 0, 0..self.drawn);
    }
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

fn min_max(heights: &[f32]) -> (f32, f32) {
    heights
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &h| {
            (lo.min(h), hi.max(h))
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

/// The index list for one patch's [`GRID`] x [`GRID`] vertices: two triangles
/// per cell, every cell -- including the skirt ring's.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_grid_covers_every_cell_with_two_triangles() {
        let idx = grid_indices();
        let cells = (GRID - 1) * (GRID - 1);
        assert_eq!(idx.len() as u32, cells * 6);
        assert!(idx.iter().all(|&i| (i as u32) < GRID * GRID));
        // Every vertex is used: none of the skirt ring is left out.
        let mut used = vec![false; (GRID * GRID) as usize];
        for &i in &idx {
            used[i as usize] = true;
        }
        assert!(used.iter().all(|u| *u));
    }

    /// The shader's `VERTS`/`GRID` and this module's must agree, or every
    /// patch reads its neighbour's heights.
    #[test]
    fn the_shader_agrees_on_the_patch_size() {
        let wgsl = crate::haze::FAR_WGSL;
        assert!(wgsl.contains(&format!("const QUADS: i32 = {FAR_QUADS};")));
        assert!(wgsl.contains(&format!("const VERTS: u32 = {FAR_VERTS}u;")));
        assert!(wgsl.contains(&format!("const GRID: u32 = {GRID}u;")));
    }

    #[test]
    fn a_patch_box_spans_its_ground_and_its_skirt() {
        let heights: Vec<f32> = (0..FAR_VERTS * FAR_VERTS)
            .map(|i| 10.0 + (i % 7) as f32)
            .collect();
        let b = patch_box(&FarPatch {
            origin: [64, -128],
            quad: 4.0,
            heights: &heights,
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
        let heights: Vec<f32> = (0..FAR_VERTS * FAR_VERTS)
            .map(|i| 10.0 + (i % 7) as f32)
            .collect();
        let shift = 1_048_576i64;
        let here = PlacedPatch::of(&FarPatch {
            origin: [64, -128],
            quad: 4.0,
            heights: &heights,
            skirt: 8.0,
        });
        let there = PlacedPatch::of(&FarPatch {
            origin: [64 + shift, -128 - shift],
            quad: 4.0,
            heights: &heights,
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
        let up: Vec<f32> = heights.iter().map(|h| h + 4096.0).collect();
        let high = PlacedPatch::of(&FarPatch {
            origin: [64 + shift, -128 - shift],
            quad: 4.0,
            heights: &up,
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
            "height(base, i, j) - frame.anchor.y - tile.skirt,",
            // `hidden_height`, worked out on the CPU (`FarTerrain::prepare`).
            "world.y = params.hole_min.w;",
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

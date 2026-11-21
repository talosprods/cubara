// The far terrain (`docs/PROPOSAL_FAR_VIEW.md` §3.2): beyond the voxel rings,
// the land as a height field -- drawn as the blocks it is made of (issue
// #422). Every face is a flat top at a whole number of blocks or a vertical
// wall exactly as tall as the step between two tops (`cubara_world::far::
// mesh`), flat-shaded with the light a voxel face of the same normal gets
// (`face.wgsl`, shared with `mesh.wgsl`).
//
// **Shared corners.** The far pass is bound by the vertices the GPU shades,
// so a quad is four of them: the corners of its top, at its height. The risers
// between neighbouring quads are drawn between those corners (`far.rs`'s
// `patch_indices`, one fixed index list for every patch) and need none of
// their own. What a face is -- a top or which wall -- is read off the screen
// (the normal, from the derivatives of the world position), and whose it is
// from the triangle's first vertex, which names the quad it was drawn from.
//
// There is no vertex buffer. A patch's quads are `SLOT_WORDS` words in a
// storage buffer (`far.rs`'s `dense_patch`); an instance is one patch in
// view, a vertex index one of its `PATCH_VERTICES`.

// Same `Frame` the other pipelines bind (`render.rs`'s `FrameUniform`): one sun,
// one fog, one camera.
struct Frame {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    ambient: vec4<f32>,
    fog_color: vec4<f32>,
    fog: vec4<f32>,
    haze: vec4<f32>,
    // The render anchor (issue #379): every position in this shader, `eye`
    // included, is relative to it. A vertex's height is the exception --
    // absolute, so `vs_main` subtracts `anchor.y` -- and the height haze
    // wants it back.
    anchor: vec4<f32>,
};

// One patch: its lowest corner (relative to the render anchor, issue #379),
// the width of one quad, and how far the walls on its edge hang below the
// step they show, all in blocks.
struct Patch {
    origin: vec2<f32>,
    quad: f32,
    skirt: f32,
};

struct Params {
    // The box the voxel rings draw themselves: nothing here is drawn inside
    // it. Relative to the render anchor, like every position here.
    // `hole_min.w`: the height vertices hidden inside it are sent to.
    hole_min: vec4<f32>,
    hole_max: vec4<f32>,
    // What the ground is made of, in linear light (`aggregate::Materials`):
    // the top of the surface block, its side, the soil and the stone beneath.
    top_color: vec4<f32>,
    // `w`: how many blocks of soil lie between the surface block and stone.
    side_color: vec4<f32>,
    soil_color: vec4<f32>,
    stone_color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var<storage, read> patches: array<Patch>;
@group(1) @binding(1) var<uniform> params: Params;
// Every held patch's quads (`far.rs`'s `dense_patch`), `SLOT_WORDS` a patch.
@group(1) @binding(2) var<storage, read> cells: array<u32>;
// The patches drawn this frame: the slot of each instance.
@group(1) @binding(3) var<storage, read> slots: array<u32>;

const QUADS: i32 = 32;
const QUADS_U: u32 = 32u;
// `far.rs`'s `FAR_CELL_VERTICES`, `FAR_CELL_WORDS`, `FAR_SLOT_WORDS` and where
// in a slot the edge walls' drops start (`EDGE_AT`).
const CELL_VERTICES: u32 = 4096u;
const CELL_WORDS: u32 = 4u;
const SLOT_WORDS: u32 = 4160u;
const EDGE_AT: u32 = 4096u;
// The bit a record carries to say its triangle is a wall on a patch's edge.
const EDGE_BIT: u32 = 0x80000000u;

// `far.rs`'s `FarDir`, in its order.
const DIR_TOP: u32 = 0u;
const DIR_POS_X: u32 = 1u;
const DIR_NEG_X: u32 = 2u;
const DIR_POS_Z: u32 = 3u;
const DIR_NEG_Z: u32 = 4u;
const FACE_NORMALS = array<vec3<f32>, 5>(
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(-1.0, 0.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0),
    vec3<f32>(0.0, 0.0, -1.0),
);

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    // Where it is, for the normal the fragment works out.
    @location(0) world: vec3<f32>,
    // How much of the fog the fragment is in, worked out per vertex and
    // interpolated, as the colour used to be.
    @location(1) fog: f32,
    // The quad the triangle was drawn from: the first word of its record in
    // `cells`, and `EDGE_BIT` for a wall on the patch's edge. The first
    // vertex's, for the whole triangle.
    @location(2) @interpolate(flat, first) record: u32,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex: u32,
    @builtin(instance_index) instance: u32,
) -> VertexOut {
    let slot = slots[instance];
    let base = slot * SLOT_WORDS;
    // Which grid corner this is, whose it is and how high (`far.rs`'s
    // `PatchVertex::of` and `PatchVertex::y`, which a test holds to these
    // lines): the first `CELL_VERTICES` are the corners of the quads' tops,
    // bit 0 of the corner the high end of x and bit 1 of z; then two under
    // every quad along each edge, for the walls that hang there.
    var cell: u32;
    var corner: u32;
    var i: i32;
    var j: i32;
    var edge_wall = false;
    var edge = 0u;
    var along = 0u;
    if vertex < CELL_VERTICES {
        cell = vertex / 4u;
        corner = vertex % 4u;
        i = i32(cell % QUADS_U + (corner & 1u));
        j = i32(cell / QUADS_U + (corner >> 1u));
    } else {
        let n = vertex - CELL_VERTICES;
        edge_wall = true;
        edge = n / 64u;
        along = (n / 2u) % QUADS_U;
        let end = n % 2u;
        if edge == 0u {
            cell = along * QUADS_U;
            corner = end << 1u;
            i = 0;
            j = i32(along + end);
        } else if edge == 1u {
            cell = along * QUADS_U + QUADS_U - 1u;
            corner = 1u | (end << 1u);
            i = QUADS;
            j = i32(along + end);
        } else if edge == 2u {
            cell = along;
            corner = end;
            i = i32(along + end);
            j = 0;
        } else {
            cell = (QUADS_U - 1u) * QUADS_U + along;
            corner = 2u | end;
            i = i32(along + end);
            j = QUADS;
        }
    }
    let record = base + cell * CELL_WORDS;
    let tile = patches[slot];
    // The heights are absolute world heights; every other position here is
    // relative to the render anchor, so the anchor's y comes off here -- the
    // eye, the hole and the voxels are all measured from it (issue #379,
    // review finding B1).
    let top = f32(i32(cells[record] & 0x3FFFFu) - 131072) - frame.anchor.y;
    // A wall on the edge hangs the step it shows below its quad's top, and a
    // skirt below that; its drop is stored plus one, and zero is no wall.
    var y = top;
    if edge_wall {
        let n = edge * QUADS_U + along;
        let drop = (cells[base + EDGE_AT + n / 2u] >> ((n % 2u) * 16u)) & 0xFFFFu;
        if drop > 0u {
            y = top - f32(drop - 1u) - tile.skirt;
        }
    }
    var world = vec3<f32>(
        tile.origin.x + f32(i) * tile.quad,
        y,
        tile.origin.y + f32(j) * tile.quad,
    );
    // Inside the voxels' box, the ground is theirs, and nothing of this is
    // seen there. The box's sides fall on chunk edges and a corner that
    // touches one is a single quad deep (`cubara_world::far::JOIN_GRID`). No
    // fragment is ever discarded (on a tile-based GPU a discard turns off
    // hidden-surface removal for the whole pipeline); it is all done by
    // moving vertices.
    if all(world > params.hole_min.xyz) && all(world < params.hole_max.xyz) {
        // A corner strictly inside the box (`far::join` is this rule, and
        // its tests say why). A riser is drawn between corners, so it
        // follows them.
        // How many quads to the nearest side of the box along each axis, as
        // a step in the grid: negative towards the low side.
        let lo = (world.xz - params.hole_min.xz) / tile.quad;
        let hi = (params.hole_max.xz - world.xz) / tile.quad;
        let step = vec2<i32>(round(select(-lo, hi, hi < lo)));
        let ti = i + step.x;
        let tj = j + step.y;
        // The first ring inside becomes a skirt: it drops straight down from
        // the corner on the box's side next to it, as far as the patch's own
        // skirt does. That closes the seam to the voxels, as the wall on the
        // edge of a patch that ends on the box's side already does.
        // Only along an axis where the corner's own quad touches the side: a
        // quad's far end is hidden (`far.rs`'s `join` says why).
        let towards_x = (step.x < 0) == ((corner & 1u) != 0u);
        let towards_z = (step.y < 0) == ((corner & 2u) != 0u);
        let snap_x = abs(step.x) == 1 && towards_x && ti >= 0 && ti <= QUADS;
        let snap_z = abs(step.y) == 1 && towards_z && tj >= 0 && tj <= QUADS;
        if snap_x || snap_z {
            i = select(i, ti, snap_x);
            j = select(j, tj, snap_z);
            world = vec3<f32>(
                tile.origin.x + f32(i) * tile.quad,
                y - tile.skirt,
                tile.origin.y + f32(j) * tile.quad,
            );
        } else {
            // Everything deeper goes where the eye sees only its underside,
            // which is culled: above the box for an eye inside it with all
            // the ground above the box's floor, and under it otherwise
            // (`far::hidden_height`, which is worked out once a frame on the
            // CPU because it needs to know the ground). Sinking it under the
            // box whatever the eye, as this used to, hung a wall from the
            // surface to the box's floor a kilometre away on every side: the
            // owner's curtain underground (issue #424).
            world.y = params.hole_min.w;
        }
    }
    // Seal the T-junctions between patches of different levels, as
    // `mesh.wgsl` does and by the same amount (`face.wgsl`'s `CRACK_SEAL`):
    // each corner moves out of its quad along x and z. Within a patch the
    // faces share their corners and have no junction to seal.
    let su = select(-1.0, 1.0, (corner & 1u) != 0u);
    let sv = select(-1.0, 1.0, (corner & 2u) != 0u);
    let outward = vec3<f32>(su, 0.0, sv);
    let sealed = world + outward * (CRACK_SEAL * distance(world, frame.eye.xyz));

    var out: VertexOut;
    out.clip_pos = frame.view_proj * vec4<f32>(sealed, 1.0);
    out.world = sealed;
    out.fog = fog_amount(sealed);
    out.record = record | select(0u, EDGE_BIT, edge_wall);
    return out;
}

// ---------------------------------------------------------------------------
// What a face shows. `far.rs`'s `face_colour` and `face_open` are the CPU's
// form of `face_colour` and `face_open` below.
// ---------------------------------------------------------------------------

// **The one place a face's colour is decided** (the seam issue #422 asks
// for): a top wears the top of the surface block, a wall the mix of the
// surface block's side, soil and stone its record carries -- what level
// 0's risers facing its way are made of (issue #423 decision 2).
fn face_colour(dir: u32, surface: f32, soil: f32) -> vec3<f32> {
    if dir == DIR_TOP {
        return params.top_color.rgb;
    }
    return params.side_color.rgb * surface + params.soil_color.rgb * soil
        + params.stone_color.rgb * (1.0 - surface - soil);
}

// How much of its light a face keeps: `mesh.wgsl`'s ambient occlusion,
// `mix(ao_floor, 1, open)`, for a face as open as level 0's are on average
// over the same ground.
fn face_open(open: f32) -> f32 {
    return mix(frame.ambient.z, 1.0, open);
}

// The fog a point is in: distance fog, or the height haze if that is more.
fn fog_amount(p: vec3<f32>) -> f32 {
    let dist = distance(p, frame.eye.xyz);
    let edge = select(0.0, smoothstep(frame.fog.x, frame.fog.y, dist), frame.fog.y > frame.fog.x);
    return max(edge, haze_amount(frame.eye.xyz, p, frame.haze, frame.anchor.y));
}

// Byte `n` (0..12) of a quad's three words of walls: the surface share, the
// soil share and the openness of each of its four walls, in `FarDir`'s order.
fn wall_byte(record: u32, n: u32) -> f32 {
    return f32((cells[record + 1u + n / 4u] >> ((n % 4u) * 8u)) & 0xFFu) / 255.0;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Which way the face looks: a triangle is flat, so its normal is the
    // same all over it, and every face is on a grid -- the longest axis of
    // the cross product of how the position changes across the screen,
    // turned towards the eye.
    var n = cross(dpdx(in.world), dpdy(in.world));
    if dot(n, frame.eye.xyz - in.world) < 0.0 {
        n = -n;
    }
    let a = abs(n);
    var dir = DIR_TOP;
    if a.y < a.x || a.y < a.z {
        if a.x >= a.z {
            dir = select(DIR_NEG_X, DIR_POS_X, n.x > 0.0);
        } else {
            dir = select(DIR_NEG_Z, DIR_POS_Z, n.z > 0.0);
        }
    }
    // Whose it is. The triangle's first vertex is in the quad it was drawn
    // from: a top is that quad's, and so is a riser that faces `+x` or `+z`
    // -- it is drawn from the lower-numbered of the two quads it stands
    // between and faces the lower of them, so one facing `-x` or `-z`
    // belongs to the next quad along. A wall on the patch's edge is its
    // quad's, whichever way it faces.
    var record = in.record & ~EDGE_BIT;
    if (in.record & EDGE_BIT) == 0u {
        if dir == DIR_NEG_X {
            record += CELL_WORDS;
        } else if dir == DIR_NEG_Z {
            record += QUADS_U * CELL_WORDS;
        }
    }
    var colour = params.top_color.rgb;
    var open = f32((cells[record] >> 18u) & 0xFFu) / 255.0;
    if dir != DIR_TOP {
        let byte = 3u * (dir - 1u);
        colour = face_colour(dir, wall_byte(record, byte), wall_byte(record, byte + 1u));
        open = wall_byte(record, byte + 2u);
    }
    // Lit as `mesh.wgsl` lights a block face (`face.wgsl`'s `face_brightness`,
    // with the terms `mesh.wgsl` gets as pipeline constants read from the
    // frame here -- both are `Lighting`'s), without texture or ambient
    // occlusion; then the fog.
    let brightness = face_brightness(
        FACE_NORMALS[dir],
        frame.sun_dir.xyz,
        frame.ambient.x,
        frame.ambient.y,
        frame.sun_color.w,
    );
    let lit = colour * face_open(open) * frame.sun_color.rgb * brightness;
    return vec4<f32>(mix(lit, frame.fog_color.rgb, in.fog), 1.0);
}

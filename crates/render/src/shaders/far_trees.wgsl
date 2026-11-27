// Trees on the far terrain (issue #433 item 1, `far_trees.rs`): one
// instance a tree, drawn as its stand-in -- a few axis-aligned boxes in the
// flat colours the level-3 silhouette has per face direction -- of which
// each box draws the face of each axis that faces the eye. There is no
// vertex buffer of the model's own: a vertex index is a face slot (a box and
// an axis it has faces along, `far_trees.rs`'s `StandIn::slots`) and one of
// its four corners.
//
// Lit as `trees.wgsl` lights a silhouette face (the same sun, ambient and
// `mix(ao_floor, 1, open)`) and fogged as it is, per vertex: a far tree is a
// few pixels, and its faces are flat. It shrinks about the foot of its
// trunk as its size on screen falls from the pixels it is drawn whole at to
// those it is gone at (`fade.wgsl`); the far terrain's tops take over what
// it no longer covers (`far.wgsl`'s `tree_tint`).

struct Frame {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    sun_dir: vec4<f32>,
    // .w is diffuse_weight.
    sun_color: vec4<f32>,
    // .x ambient_low, .y ambient_high, .z ao_floor.
    ambient: vec4<f32>,
    fog_color: vec4<f32>,
    // .x fog_start, .y fog_end.
    fog: vec4<f32>,
    haze: vec4<f32>,
    // The render anchor (issue #379): every position here is relative to it.
    anchor: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;

const MAX_BOXES: u32 = 3u;
// `FarTreeInstance::height`'s flag for a tree of the strip's stand-in
// (`far_trees.rs`'s `STANDIN_TREE`), and how much further along its line of
// sight such a tree is drawn, as a factor on its distance (`far.rs`'s
// `STANDIN_BEHIND`, which `far.wgsl` pushes the stand-in's ground back by).
const STANDIN_TREE: u32 = 0x80000000u;
const STANDIN_BEHIND: f32 = 1.000244140625;

// One box of a stand-in, in blocks from the trunk's base block, and per face
// (+X -X +Y -Y +Z -Z) its colour and openness; `.w` below zero: a face it
// does not draw.
struct Box {
    min: vec4<f32>,
    max: vec4<f32>,
    faces: array<vec4<f32>, 6>,
};

// `far_trees.rs`'s `ParamsGpu`.
struct Params {
    // .x the face slots a tree draws, .y/.z/.w the fade (`fade.wgsl`).
    fade: vec4<f32>,
    // Per face slot, `.x` the box, `.y` the axis.
    slots: array<vec4<u32>, 9>,
    // Per trunk height, `.x` the stand-in's height in blocks, `.y` the radius
    // of a sphere about its middle holding every box of it.
    tall: array<vec4<f32>, 8>,
    // Per trunk height, `MAX_BOXES` boxes.
    boxes: array<Box, 24>,
};

@group(1) @binding(0) var<uniform> params: Params;

struct VsIn {
    @builtin(vertex_index) vertex: u32,
    // Per instance (`far_trees.rs`'s `FarTreeInstance`): the trunk's foot --
    // the middle of its base block's floor -- relative to the anchor, and
    // which trunk height's stand-in it is drawn as.
    @location(0) foot: vec3<f32>,
    @location(1) height: u32,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec3<f32>,
};

const FACE_NORMALS = array<vec3<f32>, 6>(
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(-1.0, 0.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0, -1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0),
    vec3<f32>(0.0, 0.0, -1.0),
);

@vertex
fn vs_main(in: VsIn) -> VsOut {
    // Every corner of a face not drawn is the same point: no area, nothing
    // rasterised (`far.wgsl` pads its runs the same way).
    var out: VsOut;
    out.clip_pos = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    out.color = vec3<f32>(0.0);
    let slot = in.vertex / 4u;
    let corner = in.vertex % 4u;
    if slot >= u32(params.fade.x) {
        return out;
    }
    let b = params.slots[slot].x;
    let axis = params.slots[slot].y;
    let height = in.height & 0xFFu;
    let standin = (in.height & STANDIN_TREE) != 0u;
    let tall = params.tall[height].x;
    let middle = in.foot + vec3<f32>(0.0, 0.5 * tall, 0.0);
    let s = tree_scale(tall, distance(middle, frame.eye.xyz), params.fade);
    if s <= 0.0 {
        return out;
    }
    // A tree off screen costs this and no more (`FarTrees` hands over every
    // tree in reach, wherever the camera looks): the sphere about its middle
    // that holds it (`.y`) is behind the eye, or past a side plane of the
    // frustum. A side plane is the clip-space row `w +- x` (or `y`), whose
    // normal is at most `1 + |row x|` long, so a centre further outside than
    // the radius times that is outside by more than the radius.
    let c = frame.view_proj * vec4<f32>(middle, 1.0);
    let radius = params.tall[height].y;
    let vp = frame.view_proj;
    let kx = radius * (1.0 + length(vec3<f32>(vp[0].x, vp[1].x, vp[2].x)));
    let ky = radius * (1.0 + length(vec3<f32>(vp[0].y, vp[1].y, vp[2].y)));
    if c.w < -radius || abs(c.x) > c.w + kx || abs(c.y) > c.w + ky {
        return out;
    }
    // Read in place, not copied out: `boxes` is a uniform array.
    let bi = height * MAX_BOXES + b;
    // The box, shrunk about the trunk's foot: (0.5, 0, 0.5) in its frame.
    let pivot = vec3<f32>(0.5, 0.0, 0.5);
    let lo = in.foot + (params.boxes[bi].min.xyz - pivot) * s;
    let hi = in.foot + (params.boxes[bi].max.xyz - pivot) * s;
    // Of the box's two faces along `axis`, the one on the eye's side of its
    // middle; when the eye is between that face's plane and the middle (a
    // top seen from under it), it winds away from the eye and is culled.
    // `axis` picked out by a mask, as `high` below is.
    let along = vec3<u32>(0u, 1u, 2u);
    let on_axis = select(vec3<f32>(0.0), vec3<f32>(1.0), along == vec3<u32>(axis));
    let positive = dot(frame.eye.xyz, on_axis) > 0.5 * dot(lo + hi, on_axis);
    let d = axis * 2u + select(1u, 0u, positive);
    let face = params.boxes[bi].faces[d];
    if face.w < 0.0 {
        return out;
    }
    // The corner: bit 0 the high end of the face's first other axis, bit 1
    // of its second, the first flipped on a positive face so both wind
    // counter-clockwise seen from outside with the indices [0, 2, 1, 1, 2, 3]
    // (`far_trees.rs`'s `tree_indices`).
    let u = (axis + 1u) % 3u;
    let v = (axis + 2u) % 3u;
    let e = (corner & 1u) ^ select(0u, 1u, positive);
    let f = corner >> 1u;
    // Which coordinates are the box's high end, as a mask rather than by
    // writing `p[axis]`: a vector written through a varying index is what
    // DX12's shader compiler cannot do ("not natively addressable").
    let high = ((along == vec3<u32>(axis)) & vec3<bool>(positive))
        | ((along == vec3<u32>(u)) & vec3<bool>(e == 1u))
        | ((along == vec3<u32>(v)) & vec3<bool>(f == 1u));
    let p = select(lo, hi, high);
    // A stand-in's tree a hair further along its line of sight: the same
    // pixels, a little deeper, so a level-3 node's silhouette of it drawn
    // already keeps them all.
    let behind = frame.eye.xyz + (p - frame.eye.xyz) * STANDIN_BEHIND;
    out.clip_pos = frame.view_proj * vec4<f32>(select(p, behind, standin), 1.0);

    let n = FACE_NORMALS[d];
    let brightness = face_brightness(
        n,
        frame.sun_dir.xyz,
        frame.ambient.x,
        frame.ambient.y,
        frame.sun_color.w,
    );
    let lit = face.rgb * frame.sun_color.rgb * brightness * mix(frame.ambient.z, 1.0, face.w);
    let dist = distance(p, frame.eye.xyz);
    let edge = select(0.0, smoothstep(frame.fog.x, frame.fog.y, dist), frame.fog.y > frame.fog.x);
    let fog_amount = max(edge, haze_amount(frame.eye.xyz, p, frame.haze, frame.anchor.y));
    out.color = mix(lit, frame.fog_color.rgb, fog_amount);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}

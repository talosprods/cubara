// Trees beyond the first ring (`docs/PROPOSAL_LOD_TRANSITIONS.md` §3 R2,
// issue #332): one instanced draw per model (the real oak, levels 1-2; two
// boxes, level 3 -- decision 3), from `cubara-render`'s own `trees.rs`.
//
// Lit and hazed exactly like `mesh.wgsl` -- same sun, ambient and haze --
// but from the plain `Frame` uniform `figure.wgsl` already reads, not
// `mesh.wgsl`'s pipeline `override` constants (decision 4, "from the same
// uniforms"). `mesh.wgsl` moved to overrides purely for the huge arena
// draw's per-fragment cost on tile-based GPUs (its own doc comment); a tree
// draw is a few hundred instances, so there is nothing to amortise an
// override rebuild against, and reading the uniform directly means a
// lighting change needs no second pipeline rebuild here.
//
// No discard, no alpha (decision 5): every fragment in each model is
// opaque, so hidden-surface removal stays on exactly as the voxels'.
// No AO on the instances (decision 4): ambient/diffuse only.

struct Frame {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    sun_dir: vec4<f32>,
    // .w is diffuse_weight.
    sun_color: vec4<f32>,
    // .x ambient_low, .y ambient_high, .z ao_floor (unused here).
    ambient: vec4<f32>,
    fog_color: vec4<f32>,
    // .x fog_start, .y fog_end, .z time_of_day (unused here).
    fog: vec4<f32>,
    haze: vec4<f32>,
    // The render anchor (issue #379): every position in this shader, `eye`
    // included, is relative to it; only the height haze wants it back.
    anchor: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;

// The same per-node origins `mesh.wgsl` reads (`crates/render/src/arena.rs`):
// an instance stands relative to its node (issue #379 decision 2, so a
// render-anchor rebase rewrites one vec4 per node and no instance), and this
// is where the node stands. Only xyz: a tree position is in blocks already,
// not in the node's lattice units, so `.w` (the lattice scale) is not applied.
@group(1) @binding(0) var<storage, read> node_origins: array<vec4<f32>>;

@group(2) @binding(0) var tree_textures: texture_2d_array<f32>;
@group(2) @binding(1) var tree_sampler: sampler;

struct VsIn {
    // Per-vertex, from the model (`trees.rs::TreeVertex`): local position
    // relative to the trunk's base block, a face normal, this corner's own
    // texel-tile coordinate, the texture layer (negative means "untextured,
    // use `color`"), and a flat colour for the untextured case.
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) layer: f32,
    @location(4) color: vec3<f32>,
    // Per-instance (`trees.rs::TreeInstance`): where the model's local
    // origin stands relative to its node's origin, and that node's slot in
    // `node_origins`.
    @location(5) instance_pos: vec3<f32>,
    @location(6) node_index: u32,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) layer: i32,
    @location(3) @interpolate(flat) color: vec3<f32>,
    @location(4) world_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let world_pos = node_origins[in.node_index].xyz + in.instance_pos + in.position;
    var out: VsOut;
    out.clip_pos = frame.view_proj * vec4<f32>(world_pos, 1.0);
    out.normal = in.normal;
    out.uv = in.uv;
    out.layer = i32(in.layer);
    out.color = in.color;
    out.world_pos = world_pos;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = in.normal;
    let diffuse = max(dot(n, frame.sun_dir.xyz), 0.0) * frame.sun_color.w;
    let ambient = mix(frame.ambient.x, frame.ambient.y, n.y * 0.5 + 0.5);

    // Negative `layer` (the box model) reads `color`; a non-negative one
    // (the oak model) samples the block texture array. `select` evaluates
    // both arguments, so the layer is clamped to a valid index regardless of
    // which branch is actually used -- never an out-of-range read.
    let safe_layer = u32(max(in.layer, 0));
    let sampled = textureSample(tree_textures, tree_sampler, in.uv, safe_layer).rgb;
    let base = select(in.color, sampled, in.layer >= 0);

    let lit = base * frame.sun_color.rgb * (ambient + diffuse);

    // Same fog as the terrain and the figures: radial distance from the eye,
    // fading into `fog_color`, plus the height haze.
    let dist = distance(in.world_pos, frame.eye.xyz);
    let edge = select(0.0, smoothstep(frame.fog.x, frame.fog.y, dist), frame.fog.y > frame.fog.x);
    let fog_amount = max(edge, haze_amount(frame.eye.xyz, in.world_pos, frame.haze, frame.anchor.y));
    let color = mix(lit, frame.fog_color.rgb, fog_amount);
    return vec4<f32>(color, 1.0);
}

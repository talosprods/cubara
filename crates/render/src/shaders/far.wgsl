// The far terrain (`docs/PROPOSAL_FAR_VIEW.md` §3.2): beyond the voxel rings,
// the land as a height field -- drawn as the blocks it is made of (issue
// #422). Every face is a flat top at a whole number of blocks or a vertical
// wall exactly as tall as the step between two tops (`cubara_world::far::
// mesh`), flat-shaded with the light a voxel face of the same normal gets
// (`face.wgsl`, shared with `mesh.wgsl`).
//
// There is no vertex buffer. A face is three packed words (12 bytes) in a storage buffer
// (`far.rs`'s `FarFace::pack`); an instance is one run of a patch's faces --
// those that can face the eye -- and a vertex index is a face of the run and
// one of its four corners. Runs are drawn by size class, one draw a class,
// each run padded to the class's size with zero-area faces: at most 17 draws
// for the whole far terrain, whatever its size.

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
    // What a top shows of the trees not drawn one by one (issue #433 item 1,
    // `far_trees.rs`'s `TintGpu`): `.x` the trunk heights in use, `.yzw` the
    // fade (`fade.wgsl`); per height its density where every column may grow
    // a tree and its stand-in's height; per height and direction (+X -X +Y
    // +Z -Z) the stand-in's area times colour (`.w` the area) and area times
    // colour times openness.
    tree_fade: vec4<f32>,
    // `.x` 1: measured from straight above (`far_trees.rs`'s `TintView`).
    tree_view: vec4<f32>,
    tree_height: array<vec4<f32>, 8>,
    tree_colour: array<array<vec4<f32>, 5>, 8>,
    tree_open: array<array<vec4<f32>, 5>, 8>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var<storage, read> patches: array<Patch>;
@group(1) @binding(1) var<uniform> params: Params;
@group(1) @binding(2) var<storage, read> faces: array<Face>;
@group(1) @binding(3) var<storage, read> runs: array<Run>;

// One face, packed (`far.rs`'s `FarFace::pack`).
struct Face {
    // i0:6 j0:6 i1:6 j1:6 dir:3 border:1
    word0: u32,
    // top + 131072 : 18, rise : 14
    word1: u32,
    // surface:8 soil:8 open:8 trees:8, each in 255ths
    word2: u32,
};

// One run of faces: its patch's slot, its first face, and how many it is
// (and a word of padding: a class's runs start on 256 bytes, `far.rs`'s
// `RUN_ALIGN`).
struct Run {
    slot: u32,
    first: u32,
    count: u32,
    pad: u32,
};

const QUADS: i32 = 32;

// `far.rs`'s `FarDir`, in its order.
const DIR_TOP: u32 = 0u;
const DIR_POS_X: u32 = 1u;
const DIR_NEG_X: u32 = 2u;
const DIR_POS_Z: u32 = 3u;
const FACE_NORMALS = array<vec3<f32>, 5>(
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(-1.0, 0.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0),
    vec3<f32>(0.0, 0.0, -1.0),
);

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    // The lit, fogged colour. The light is the face's, the same at all four
    // corners; the fog is worked out per vertex and interpolated.
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex: u32,
    @builtin(instance_index) instance: u32,
) -> VertexOut {
    let run = runs[instance];
    let nth = vertex / 4u;
    if nth >= run.count {
        // Padding up to the class's size: every corner the same point, so
        // the face has no area and nothing is rasterised.
        var none: VertexOut;
        none.clip_pos = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        none.color = vec3<f32>(0.0);
        return none;
    }
    let face = faces[run.first + nth];
    let word0 = face.word0;
    let word1 = face.word1;
    let word2 = face.word2;
    let dir = (word0 >> 24u) & 0x7u;
    let border = ((word0 >> 27u) & 1u) != 0u;
    let wall = dir != DIR_TOP;
    // Which of the face's four corners (`FarFace::shader_corner` is these
    // lines, and a test holds it against `FarFace::corners`): bit 0 of the
    // corner is the high end of the face's ground axis, bit 1 the high end
    // of the other -- up, for a wall. The two wall directions that would
    // otherwise wind clockwise run backwards.
    let k = vertex % 4u;
    let flip = u32(dir == DIR_NEG_X || dir == DIR_POS_Z);
    let e = (k & 1u) ^ flip;
    let u = k >> 1u;
    let corner = e | (u << 1u);
    let i0 = word0 & 0x3Fu;
    let j0 = (word0 >> 6u) & 0x3Fu;
    let di = ((word0 >> 12u) & 0x3Fu) - i0;
    let dj = ((word0 >> 18u) & 0x3Fu) - j0;
    var i = i32(i0 + e * di);
    var j = i32(j0 + select(u, e, wall) * dj);
    let tile = patches[run.slot];
    // The heights are absolute world heights; every other position here is
    // relative to the render anchor, so the anchor's y comes off here -- the
    // eye, the hole and the voxels are all measured from it (issue #379,
    // review finding B1).
    let top = f32(i32(word1 & 0x3FFFFu) - 131072) - frame.anchor.y;
    let rise = f32(word1 >> 18u);
    // A wall's lower corners hang the step below its top, and on the patch's
    // edge a skirt below that.
    var y = top;
    if wall && (corner & 2u) == 0u {
        y = top - rise - select(0.0, tile.skirt, border);
    }
    var world = vec3<f32>(
        tile.origin.x + f32(i) * tile.quad,
        y,
        tile.origin.y + f32(j) * tile.quad,
    );
    // Inside the voxels' box, the ground is theirs, and nothing of this is
    // seen there. The box's sides fall on chunk edges, no face crosses one of
    // those lines and a top that touches one is a single quad deep
    // (`cubara_world::far::JOIN_GRID`). No fragment is ever discarded (on a
    // tile-based GPU a discard turns off hidden-surface removal for the whole
    // pipeline); it is all done by moving vertices.
    let faces_x = dir == DIR_POS_X || dir == DIR_NEG_X;
    var seal = CRACK_SEAL;
    if wall {
        // A wall is in the box or out of it as a whole (`far::wall_in_hole`):
        // by a point half a quad along it from this corner, which is the same
        // side of every line for all four corners, and by its top. In the box
        // it is cut off at the box's floor: nothing of it is left above, and
        // what the ground shows under the floor is still drawn.
        let inward = select(0.5, -0.5, (corner & 1u) != 0u) * tile.quad;
        let middle = world.xz + select(vec2<f32>(inward, 0.0), vec2<f32>(0.0, inward), faces_x);
        if all(middle > params.hole_min.xz) && all(middle < params.hole_max.xz)
            && top > params.hole_min.y && top < params.hole_max.y {
            world.y = min(world.y, params.hole_min.y);
            // What is left of it, if anything, is under the box. A wall cut
            // down to nothing must stay nothing: sealed, its corners would
            // move apart again, and thousands of walls each a tenth of a
            // pixel tall lie on the box's floor, in sight of an eye
            // underground (seen as three pixels in `golden.rs`'s curtain
            // test).
            seal = 0.0;
        }
    } else if all(world > params.hole_min.xyz) && all(world < params.hole_max.xyz) {
        // A top's corner strictly inside the box (`far::join` is this rule,
        // and its tests say why).
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
        let snap_x = abs(step.x) == 1 && ti >= 0 && ti <= QUADS;
        let snap_z = abs(step.y) == 1 && tj >= 0 && tj <= QUADS;
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
    // Seal the T-junctions a greedy mesh leaves, as `mesh.wgsl` does and by
    // the same amount (`face.wgsl`'s `CRACK_SEAL`): each corner moves out of
    // its face along the face's two in-plane axes. `corner` bit 0: the high
    // end of the face's ground axis (x for a top and a wall facing along z,
    // z for a wall facing along x); bit 1: the high end of the other (z for
    // a top, up for a wall).
    let su = select(-1.0, 1.0, (corner & 1u) != 0u);
    let sv = select(-1.0, 1.0, (corner & 2u) != 0u);
    var outward = vec3<f32>(su, 0.0, sv);
    if wall {
        outward = select(vec3<f32>(su, sv, 0.0), vec3<f32>(0.0, sv, su), faces_x);
    }
    let sealed = world + outward * (seal * distance(world, frame.eye.xyz));

    var out: VertexOut;
    out.clip_pos = frame.view_proj * vec4<f32>(sealed, 1.0);
    // A top shows the trees on it no longer drawn one by one.
    var tint = vec4<f32>(0.0);
    if !wall {
        tint = tree_tint(sealed, f32(word2 >> 24u) / 255.0, tile.quad);
    }
    out.color = shade(sealed, face_colour(dir, word2) * face_open(word2), FACE_NORMALS[dir], tint);
    return out;
}

// The five directions a top can see a tree's faces from: `far_trees.rs`'s
// `TINT_DIRECTIONS`.
const TINT_NORMALS = array<vec3<f32>, 5>(
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(-1.0, 0.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0),
    vec3<f32>(0.0, 0.0, -1.0),
);

// **What a top shows of its trees that are not drawn one by one** (issue
// #433 item 1): `.a` the share of the top's ground they hide from the eye,
// `.rgb` the lit colour they show there -- for a top at `p` whose share of
// ground a tree may grow on is `wooded`.
//
// A far tree shrinks to nothing as its size on screen falls through the
// fade (`fade.wgsl`, `far_trees.wgsl`); what it no longer covers, `1 - s^2`
// of its area, the top shows instead. A stand-in seen along `v` (towards
// the eye) hides its top's area of the ground under it, and its sides'
// area facing the eye over `v.y` -- the ground behind it, a shadow as long
// as the tree is tall over the slope of the line of sight. But only the
// trees standing on a top can hide it, and a top is `quad` deep: a point
// of it is hidden by the trees in front of it on the top, as far as a
// shadow reaches, which over the top's depth is a share `1 - L / 2 quad`
// of the shadows (`L` their length), or `quad / 2 L` of them when they are
// longer than the top. Trees standing at random, `n` a square block, leave
// `exp(-n * hidden)` of the top in sight. Summed over the trunk heights,
// each fading by its own height. Seen straight down it is the share the
// canopies cover from above; seen low over a wide top, more, as a wood is.
// Zero where every tree is drawn whole, and for a top without trees.
fn tree_tint(p: vec3<f32>, wooded: f32, quad: f32) -> vec4<f32> {
    let heights = u32(params.tree_fade.x);
    if heights == 0u || wooded <= 0.0 {
        return vec4<f32>(0.0);
    }
    let to_eye = frame.eye.xyz - p;
    let dist = length(to_eye);
    // The line of sight, or straight up for the share the canopies cover
    // seen from above (a measurement's, `TintView::FromAbove`).
    let v = select(to_eye / max(dist, 1e-3), vec3<f32>(0.0, 1.0, 0.0), params.tree_view.x > 0.5);
    // A top seen from under it, or edge on, shows nothing of what stands on it.
    if v.y <= 1e-4 {
        return vec4<f32>(0.0);
    }
    // How each direction of a stand-in face is lit (`trees.wgsl`'s light),
    // and how much of it faces the eye.
    var lit: array<vec3<f32>, 5>;
    var facing: array<f32, 5>;
    for (var d = 0u; d < 5u; d++) {
        let n = TINT_NORMALS[d];
        lit[d] = frame.sun_color.rgb * face_brightness(
            n,
            frame.sun_dir.xyz,
            frame.ambient.x,
            frame.ambient.y,
            frame.sun_color.w,
        );
        facing[d] = max(dot(n, v), 0.0);
    }
    // How far behind it a tree's side hides the ground, per block of its
    // height: the line of sight's run over its rise.
    let run = length(v.xz) / v.y;
    var hidden = 0.0;
    var colour = vec3<f32>(0.0);
    for (var k = 0u; k < heights; k++) {
        let h = params.tree_height[k];
        let s = tree_scale(h.y, dist, params.tree_fade);
        let missing = 1.0 - s * s;
        if missing <= 0.0 {
            continue;
        }
        // The share of the sides' shadows that falls on the top, `quad`
        // deep, its trees standing anywhere on it.
        let shadow = h.y * run;
        let on_top = select(quad / (2.0 * shadow), 1.0 - shadow / (2.0 * quad), shadow < quad);
        var area = 0.0;
        var seen = vec3<f32>(0.0);
        for (var d = 0u; d < 5u; d++) {
            let c = params.tree_colour[k][d];
            // `mix(ao_floor, 1, open)` is linear in `open`: over the faces
            // of a direction, `ao_floor * sum(area * colour) + (1 - ao_floor)
            // * sum(area * colour * open)`.
            let shaded = mix(params.tree_open[k][d].rgb, c.rgb, frame.ambient.z);
            // The top (`d` 2) hides the ground under it; a side, its
            // shadow's share on the top.
            let w = facing[d] / v.y * select(on_top, 1.0, d == 2u);
            area += c.w * w;
            seen += w * lit[d] * shaded;
        }
        let n = h.x * wooded * missing;
        hidden += n * area;
        colour += n * seen;
    }
    if hidden <= 0.0 {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(colour / hidden, 1.0 - exp(-hidden));
}

// ---------------------------------------------------------------------------
// What a face shows. `far.rs`'s `face_colour` and `face_open` are the CPU's
// form of these two.
// ---------------------------------------------------------------------------

// **The one place a face's colour is decided** (the seam issue #422 asks
// for): the mix of the surface block, soil and stone its third word carries
// -- the surface block's side on a wall, its top on a top. A wall's mix is
// what level 0's risers facing its way are made of (issue #423 decision 2);
// a top's is the surface block, or the floor of a pit level 0 shows from
// above there (`WorldGen::top_layer`).
fn face_colour(dir: u32, word2: u32) -> vec3<f32> {
    let surface = f32(word2 & 0xFFu) / 255.0;
    let soil = f32((word2 >> 8u) & 0xFFu) / 255.0;
    let shown = select(params.side_color.rgb, params.top_color.rgb, dir == DIR_TOP);
    return shown * surface + params.soil_color.rgb * soil
        + params.stone_color.rgb * (1.0 - surface - soil);
}

// How much of its light a face keeps: `mesh.wgsl`'s ambient occlusion,
// `mix(ao_floor, 1, open)`, for a face as open as level 0's are on average
// over the same ground.
fn face_open(word2: u32) -> f32 {
    let open = f32((word2 >> 16u) & 0xFFu) / 255.0;
    return mix(frame.ambient.z, 1.0, open);
}

// A face of colour `colour` and normal `n` at `p`: lit as `mesh.wgsl` lights
// a block face (`face.wgsl`'s `face_brightness`, with the terms `mesh.wgsl`
// gets as pipeline constants read from the frame here -- both are
// `Lighting`'s), without texture or ambient occlusion; then `tint` over it
// (`tree_tint`: the trees on a top not drawn one by one, lit colour and
// share); then the fog.
fn shade(p: vec3<f32>, colour: vec3<f32>, n: vec3<f32>, tint: vec4<f32>) -> vec3<f32> {
    let brightness = face_brightness(
        n,
        frame.sun_dir.xyz,
        frame.ambient.x,
        frame.ambient.y,
        frame.sun_color.w,
    );
    let lit = mix(colour * frame.sun_color.rgb * brightness, tint.rgb, tint.a);

    let dist = distance(p, frame.eye.xyz);
    let edge = select(0.0, smoothstep(frame.fog.x, frame.fog.y, dist), frame.fog.y > frame.fog.x);
    let fog_amount = max(edge, haze_amount(frame.eye.xyz, p, frame.haze, frame.anchor.y));
    return mix(lit, frame.fog_color.rgb, fog_amount);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}

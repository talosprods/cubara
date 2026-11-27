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
    // surface:8 soil:8 open:8, each in 255ths
    word2: u32,
};

// One run of faces: its patch's slot, its first face, how many it is, and
// what its patch is drawn as (`RUN_STANDIN`; sixteen bytes, so a class's
// runs start on 256 bytes, `far.rs`'s `RUN_ALIGN`).
struct Run {
    slot: u32,
    first: u32,
    count: u32,
    flags: u32,
};

const QUADS: i32 = 32;

// `Run::flags`: the run is of a stand-in for voxel nodes not drawn yet
// (`far.rs`'s `FarRole::StandIn` and `RUN_STANDIN`).
const RUN_STANDIN: u32 = 1u;
// How much further along its line of sight a stand-in's corner is drawn,
// as a factor on its distance (`far.rs`'s `STANDIN_BEHIND`).
const STANDIN_BEHIND: f32 = 1.000244140625;

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
    // A stand-in's faces are already cut to the ground the voxel nodes it
    // stands in for will draw, inside the box (`far.rs`'s `FarRole`):
    // nothing of the join below is theirs.
    let standin = (run.flags & RUN_STANDIN) != 0u;
    if standin {
        // Drawn where it stands.
    } else if wall {
        // A wall is in the box or out of it as a whole (`far::wall_in_hole`):
        // by a point half a quad along it from this corner, which is the same
        // side of every line for all four corners, and by its top. In the box
        // it is cut off at the box's floor: nothing of it is left above, and
        // what the ground shows under the floor is still drawn.
        let inward = select(0.5, -0.5, (corner & 1u) != 0u) * tile.quad;
        let middle = world.xz + select(vec2<f32>(inward, 0.0), vec2<f32>(0.0, inward), faces_x);
        if all(middle > params.hole_min.xz) && all(middle < params.hole_max.xz)
            && top > params.hole_min.y && top < params.hole_max.y {
            // The step between the corner quad and the next quad along the
            // z side, where the corner quad is the higher (`far::corner_wall`):
            // a wall facing along x, one quad in from the nearest x side,
            // facing away from it, with an end on a z side.
            let lo_x = (world.x - params.hole_min.x) / tile.quad;
            let hi_x = (params.hole_max.x - world.x) / tile.quad;
            let step_x = i32(round(select(-lo_x, hi_x, hi_x < lo_x)));
            let away = (dir == DIR_POS_X && step_x == -1) || (dir == DIR_NEG_X && step_x == 1);
            let run = f32(dj) * tile.quad;
            let run_lo = select(world.z, world.z - run, e == 1u);
            let on_lo = run_lo == params.hole_min.z;
            let on_hi = run_lo + run == params.hole_max.z;
            let corner_wall = faces_x && away && (on_lo || on_hi);
            if corner_wall {
                // Folded onto the z side over the corner quad's part of it:
                // its far end to the box's corner, at its own height.
                let side_z = select(params.hole_max.z, params.hole_min.z, on_lo);
                if world.z != side_z {
                    world.x = world.x + f32(step_x) * tile.quad;
                    world.z = side_z;
                }
            } else {
                world.y = min(world.y, params.hole_min.y);
                // What is left of it, if anything, is under the box. A wall
                // cut down to nothing must stay nothing: sealed, its corners
                // would move apart again, and thousands of walls each a tenth
                // of a pixel tall lie on the box's floor, in sight of an eye
                // underground (seen as three pixels in `golden.rs`'s curtain
                // test).
                seal = 0.0;
            }
        }
    } else if all(world.xz >= params.hole_min.xz) && all(world.xz <= params.hole_max.xz) {
        // A top's corner in the box or on its sides (`far::join` and
        // `far::corner` are these rules, and their tests say why); every
        // other corner is drawn where it stands, without a step of this.
        // How many quads to the nearest side of the box along each axis, as
        // a step in the grid: negative towards the low side.
        let lo = (world.xz - params.hole_min.xz) / tile.quad;
        let hi = (params.hole_max.xz - world.xz) / tile.quad;
        let step = vec2<i32>(round(select(-lo, hi, hi < lo)));
        // From this corner into its face, and whether that face is in the
        // box: the middle of its quad beside this corner, at a height the
        // box holds.
        let into = vec2<i32>(select(1, -1, e == 1u), select(1, -1, u == 1u));
        let middle = world.xz + 0.5 * tile.quad * vec2<f32>(into);
        let face_in = all(middle > params.hole_min.xz) && all(middle < params.hole_max.xz)
            && y > params.hole_min.y && y < params.hole_max.y;
        if face_in && abs(step.x) == 1 && abs(step.y) <= 1 {
            // At a corner of the box (`far::corner`): the first ring folds
            // onto the x side, and the quad after the corner along the z
            // side stretches its skirt over the corner quad's part of it --
            // to the corner, and a skirt deep but on the side itself.
            let toward_x = into.x == step.x;
            i = i + step.x;
            j = select(j, j + step.y, step.y != 0 && !toward_x);
            let drop = step.y != 0 || toward_x;
            world = vec3<f32>(
                tile.origin.x + f32(i) * tile.quad,
                select(y, y - tile.skirt, drop),
                tile.origin.y + f32(j) * tile.quad,
            );
        } else if all(world > params.hole_min.xyz) && all(world < params.hole_max.xyz) {
            // A corner strictly inside the box (`far::join`).
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
    // A stand-in a hair further along the corner's line of sight: the same
    // pixels, a little deeper, so a voxel node of the same ground drawn
    // already keeps them all.
    let behind = frame.eye.xyz + (sealed - frame.eye.xyz) * STANDIN_BEHIND;

    var out: VertexOut;
    out.clip_pos = frame.view_proj * vec4<f32>(select(sealed, behind, standin), 1.0);
    out.color = shade(sealed, face_colour(dir, word2) * face_open(word2), FACE_NORMALS[dir]);
    return out;
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
// `Lighting`'s), without texture or ambient occlusion; then the fog.
fn shade(p: vec3<f32>, colour: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let brightness = face_brightness(
        n,
        frame.sun_dir.xyz,
        frame.ambient.x,
        frame.ambient.y,
        frame.sun_color.w,
    );
    let lit = colour * frame.sun_color.rgb * brightness;

    let dist = distance(p, frame.eye.xyz);
    let edge = select(0.0, smoothstep(frame.fog.x, frame.fog.y, dist), frame.fog.y > frame.fog.x);
    let fog_amount = max(edge, haze_amount(frame.eye.xyz, p, frame.haze, frame.anchor.y));
    return mix(lit, frame.fog_color.rgb, fog_amount);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}

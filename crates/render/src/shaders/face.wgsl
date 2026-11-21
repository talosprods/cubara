// What a block face is, whichever pipeline draws it: how brightly it is lit,
// and how far its corners are pushed out to seal a greedy mesh's T-junctions.
// Prepended to `mesh.wgsl` (the voxels) and to `far.wgsl` (the far terrain's
// blocks, issue #422), so a far wall and a near wall in the same light are
// the same brightness because they run the same lines, not two copies of
// them (`haze.rs` builds both sources).

// How brightly a face of unit normal `n` is lit, before its colour, the
// sun's colour and any ambient occlusion: hemispheric ambient -- a touch
// brighter facing up (sky) than down (ground) -- plus the sun's diffuse term,
// a clear sun-side / shadow-side split. `sun_dir` is normalized on the CPU
// (`Lighting`'s doc comment); not renormalized here. The CPU's form of this
// is `aggregate::face_light`.
fn face_brightness(
    n: vec3<f32>,
    sun_dir: vec3<f32>,
    ambient_low: f32,
    ambient_high: f32,
    diffuse_weight: f32,
) -> f32 {
    let diffuse = max(dot(n, sun_dir), 0.0) * diffuse_weight;
    let ambient = mix(ambient_low, ambient_high, n.y * 0.5 + 0.5);
    return ambient + diffuse;
}

// How far a corner moves out of its quad, per block of distance from the eye:
// 0.06 of a pixel at 1080p with a 60-degree field of view (a pixel is about
// 0.97 milliradians there). See `mesh.wgsl`'s `vs_main`, and `far.wgsl`'s.
//
// Issue #397: the down flight (`--bench flight --eye 8,450,8 --look
// 0.4,-0.8,0.2`) still confirmed a one-frame, single-pixel hole at 0.05 px
// (the value #321/BENCHMARKS.md footnote 81 picked), in 5 of 7 runs on a
// quiet machine, clustered at one spot (t ~ 11.6-11.9 s, screen x 813-817,
// y 688-706). Traced (`cubara_world::raycast` + `cubara_world::mesh::
// mesh_node`, same seed/registry, no GPU) to a real greedy-mesh T-junction:
// node (level 0, [12, 1, 4])'s top face has a 14-cell-long quad (z 7..8,
// x 2..16) beside a fragmented row (z 8..9, several shorter quads); the
// ray's continuous plane intersection lands within 0.03-0.05 blocks of
// their shared edge -- under a third of a pixel at this ~193-block range,
// i.e. already inside 0.05 px's own margin. 0.06 px (this value) closed it
// in 11 of 11 `--bench flight` runs afterward (6 at this exact value, 5
// more at 0.07 px) with the goldens still well inside tolerance (`terrain`
// 0.31%, `lod_boundary` 0.09%, against 0.5%) -- see the PR for the full
// before/after. A *static* single-frame re-render of the same node from
// the same eye (headless, and the bench's own "finished" pass) never shows
// the gap; only the continuously-streamed live pass does, intermittently --
// consistent with a borderline case landing right at the shader's own
// floating-point margin, not a stable geometric one a CPU test can pin.
// That is also why this fix is a measured margin increase on the existing
// mechanism, not a redesign: nothing here found a different class (a
// same-level node seam, a voxel/far-terrain seam, or a depth-precision
// bug) to fix instead.
//
// Issue #422 (decision 4, "re-evaluated, not carried over blindly"): the far
// terrain's blocks are a greedy mesh too -- merged tops beside unmerged ones,
// walls merged along a row, a fine patch's faces against a coarser
// neighbour's -- so they have the same T-junctions, of the same size in
// pixels: the rounding is in the projection, which does not know which
// pipeline a corner came from. The seal is an angle, so it is the same 0.06 px
// at any distance, and the far terrain takes the same one rather than a
// second number with nothing measured behind it.
const CRACK_SEAL: f32 = 0.00006;

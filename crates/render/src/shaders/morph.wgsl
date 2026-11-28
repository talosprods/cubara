// The distance morph's factor (issue #380's study; CDLOD geomorphing):
// how far, 0 to 1, a vertex of a node at `level` has moved toward its
// parent's height, measured where it was meshed (`at`, relative to the
// render anchor). `cubara_world::morph::morph_factor` over `morph_metric`
// is its CPU twin.
//
// One function for every shader a thing moves with the ground in: the
// terrain (`mesh.wgsl`) and the trees standing on it (`trees.wgsl`, issue
// #433), so a tree and the ground under its trunk move by one factor. Both
// declare `frame` with the morph's `morph_band` and `morph_eye` after
// `anchor` (`render.rs`'s `FrameUniform`); WGSL resolves a module-scope name
// wherever it is declared.
//
// The metric's vertical part counts height once near the eye and `squash`
// times as fast beyond (`cubara_world::morph::squashed_height`):
// `max(v, squash * (v - shift))`, the squash in `morph_eye[0].w` and the
// shift, in blocks, in `morph_eye[1].w` -- so a vertex fewer than
// `NEAR_EXACT` blocks away in a straight line is never in level 0's band.
fn morph_factor(at: vec3<f32>, level: u32) -> f32 {
    let band_pair = frame.morph_band[level >> 1u];
    let band = select(band_pair.xy, band_pair.zw, (level & 1u) == 1u);
    let rel = at - frame.morph_eye[level].xyz;
    let v = abs(rel.y);
    let vertical = max(v, frame.morph_eye[0].w * (v - frame.morph_eye[1].w));
    let metric = max(length(rel.xz), vertical);
    return clamp((metric - band.x) / max(band.y - band.x, 1e-6), 0.0, 1.0);
}

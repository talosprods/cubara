// The far trees' fade by screen size (issue #433 item 1): how large a far
// tree is drawn, 0 to 1 -- `far_trees.rs`'s `tree_scale`, line for line.
// Prepended to `far_trees.wgsl` (the trees) and to `far.wgsl` (the tint of
// the far tops, which shows what the trees no longer drawn covered), so a
// tree shrinks and its ground takes over by one function (`haze.rs` builds
// both sources).
//
// A tree `tall` blocks tall at `dist` blocks stands `tall * fade.w / dist`
// pixels tall (`fade.w`: pixels a block is tall at one block); it is drawn
// whole from `fade.y` pixels up and not at all from `fade.z` down.
fn tree_scale(tall: f32, dist: f32, fade: vec4<f32>) -> f32 {
    let px = tall * fade.w / max(dist, 1e-3);
    return clamp((px - fade.z) / (fade.y - fade.z), 0.0, 1.0);
}

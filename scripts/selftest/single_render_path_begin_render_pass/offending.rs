// CONTRIBUTING.md "one scene, one render path". Trips
// check-single-render-path.sh's grep for `begin_render_pass` outside
// crates/render/src/scene.rs.
fn offending(encoder: &mut wgpu::CommandEncoder) {
    let _ = encoder.begin_render_pass(&Default::default());
}

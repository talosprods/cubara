// CONTRIBUTING.md "one scene, one render path". Trips
// check-single-render-path.sh's grep for `build_pipeline(` outside
// crates/render/src/scene.rs.
fn offending() {
    build_pipeline();
}

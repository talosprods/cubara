// ARCHITECTURE.md Rule 5: one wgpu::Instance, made through
// cubara_render::new_instance. Trips check-architecture.sh's grep for
// `wgpu::Instance::new` outside instance.rs.
fn offending() {
    let _ = wgpu::Instance::new(Default::default());
}

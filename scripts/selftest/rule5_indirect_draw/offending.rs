// ARCHITECTURE.md Rule 5: indirect draws are issued in exactly one file,
// arena.rs. Trips check-architecture.sh's grep for
// `.draw_indirect(`/`.draw_indexed_indirect(`/etc outside arena.rs.
fn offending(pass: &mut wgpu::RenderPass, buf: &wgpu::Buffer) {
    pass.draw_indirect(buf, 0);
}

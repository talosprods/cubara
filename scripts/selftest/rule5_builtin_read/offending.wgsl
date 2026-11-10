// ARCHITECTURE.md Rule 5: an indirectly drawn shader (mesh.wgsl) must not
// read vertex_index/instance_index -- wrong on D3D12 without indirect
// validation. Trips check-architecture.sh's grep on mesh.wgsl for
// `@builtin(vertex_index|instance_index)`.
fn vs_main(@builtin(vertex_index) vi: u32) -> vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

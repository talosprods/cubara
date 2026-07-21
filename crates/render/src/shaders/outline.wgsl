// Wireframe box around the block the player is targeting. Shares the terrain's
// camera uniform (group 0, binding 0) since vertices already arrive in world space.

struct Camera {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

struct VsIn {
    @location(0) position: vec3<f32>,
};

@vertex
fn vs_main(in: VsIn) -> @builtin(position) vec4<f32> {
    return camera.view_proj * vec4<f32>(in.position, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

//! Wireframe box drawn around the block the player is targeting, so edits have
//! clear feedback. Depth-tested against terrain (a closer block occludes it) but
//! drawn a hair outside the block's exact bounds to avoid z-fighting with the
//! coincident terrain face. See issue #52.

use crate::render::DEPTH_FORMAT;

/// One outline-box vertex: world-space position only (solid colour in the shader).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct OutlineVertex {
    pos: [f32; 3],
}

const ATTRS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x3];
/// Outward offset so the box doesn't share a depth value with the block's own face.
const OUTSET: f32 = 0.0015;
/// 12 edges * 2 endpoints.
const MAX_VERTS: usize = 24;

/// The 12 edges of a unit box, as pairs of corner indices (x, y, z bit each 0/1).
const EDGES: [[(usize, usize, usize); 2]; 12] = [
    // bottom face
    [(0, 0, 0), (1, 0, 0)],
    [(1, 0, 0), (1, 0, 1)],
    [(1, 0, 1), (0, 0, 1)],
    [(0, 0, 1), (0, 0, 0)],
    // top face
    [(0, 1, 0), (1, 1, 0)],
    [(1, 1, 0), (1, 1, 1)],
    [(1, 1, 1), (0, 1, 1)],
    [(0, 1, 1), (0, 1, 0)],
    // verticals
    [(0, 0, 0), (0, 1, 0)],
    [(1, 0, 0), (1, 1, 0)],
    [(1, 0, 1), (1, 1, 1)],
    [(0, 0, 1), (0, 1, 1)],
];

/// Draws a box outline around a targeted block. Reuses the terrain's camera bind
/// group (same layout, group 0) since it needs nothing else.
pub struct OutlineRenderer {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    count: u32,
}

impl OutlineRenderer {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        camera_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("outline-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/outline.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("outline-layout"),
            bind_group_layouts: &[camera_bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("outline-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<OutlineVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &ATTRS,
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                // Read the terrain depth (a block in front still occludes the
                // outline) but don't write it — the outline itself shouldn't
                // occlude anything drawn after it.
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outline-vertices"),
            size: (MAX_VERTS * std::mem::size_of::<OutlineVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            vertex_buffer,
            count: 0,
        }
    }

    /// Rebuild the box for the targeted block, or clear it if `None`.
    pub fn set_target(&mut self, queue: &wgpu::Queue, block: Option<[i32; 3]>) {
        let Some(block) = block else {
            self.count = 0;
            return;
        };
        let min = [
            block[0] as f32 - OUTSET,
            block[1] as f32 - OUTSET,
            block[2] as f32 - OUTSET,
        ];
        let max = [
            block[0] as f32 + 1.0 + OUTSET,
            block[1] as f32 + 1.0 + OUTSET,
            block[2] as f32 + 1.0 + OUTSET,
        ];
        let corner = |(ix, iy, iz): (usize, usize, usize)| OutlineVertex {
            pos: [
                if ix == 0 { min[0] } else { max[0] },
                if iy == 0 { min[1] } else { max[1] },
                if iz == 0 { min[2] } else { max[2] },
            ],
        };
        let mut verts = Vec::with_capacity(MAX_VERTS);
        for [a, b] in EDGES {
            verts.push(corner(a));
            verts.push(corner(b));
        }
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&verts));
        self.count = verts.len() as u32;
    }

    /// Draw the outline (no-op if nothing is targeted). Call inside the main pass,
    /// after terrain, with the camera bind group already bound at group 0.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        if self.count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..self.count, 0..1);
    }
}

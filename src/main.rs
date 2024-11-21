use std::{borrow::Cow, collections::HashMap};
use rand::random;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
};

use glam::{vec3, vec4, Vec3, Vec4};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FrameUniforms {
    res:    [u32;2],
    frame:  u32,
    tri_count: u32,
    time:   f32,
    _pad: f32,
}

mod gpu;
use gpu::*;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Tri {
    vertices: [Vec4; 3],
}

impl Tri {
    fn new(p1: Vec3, p2: Vec3, p3: Vec3) -> Tri {
        let c = (p1 + p2 + p3) / 3.0;
        Tri {
            vertices: [vec4(p1.x, p1.y, p1.z, c.x), vec4(p2.x, p2.y, p2.z, c.x), vec4(p3.x, p3.y, p3.z, c.x)],
        }
    }

    fn dummy(c: Vec3, size: f32) -> Tri {
        
        Tri::new(
            c + vec3(random(), random(), random()) * size - size * 0.5,
            c + vec3(random(), random(), random()) * size - size * 0.5,
            c + vec3(random(), random(), random()) * size - size * 0.5,
        )
    }
}

struct Ray {
    origin: Vec3,
    direction: Vec3,
    t: f32,
}

struct Context {
    screen_pipeline:        wgpu::RenderPipeline,
    raytrace_pipeline:      wgpu::ComputePipeline,

    triangles_ssbo:         wgpu::Buffer,
    bvh_ssbo:               wgpu::Buffer,
    screen_ssbo:            wgpu::Buffer,

    rt_data_binding:        wgpu::BindGroup,

    frame_uniforms_binding: wgpu::BindGroup,
    frame_uniforms_buffer:  wgpu::Buffer,
    frame_uniforms:         FrameUniforms,

    resources:        ResourceManager,

    triangles: Vec<Tri>,
}

impl Context {
    fn init(gpu: &Gpu) -> Context {
        let mut triangles: Vec<Tri> = Vec::new();

        for _ in 0..32 {
            triangles.push(Tri::dummy(vec3(random(), random(), random()) * 0.125 + 0.5, 1.0));
        };

        {
            let floor_height = 0.0;
            let s = 100.0;
            triangles[0] = Tri::new(vec3(-s, -s, floor_height), vec3(-s, s, floor_height), vec3(s, -s, floor_height));
            triangles[1] = Tri::new(vec3(-s, s, floor_height), vec3(s, s, floor_height), vec3(s, -s, floor_height));
        }


        let mut resources = ResourceManager::new();

        let u_frame_0 = FrameUniforms {
            frame: 0,
            res: [512, 512],
            tri_count: triangles.len() as u32,
            time: 0.0,
            _pad: 0.0,
        };

        let u_frame_buffer = gpu.new_uniform_buffer(&u_frame_0);

        let u_frame = gpu.new_bind_group()
            .with_buffer(&u_frame_buffer.view_all(), wgpu::ShaderStages::all())
            .finish(&mut resources);

        let buffer_size_mb = 128;

        let triangles_ssbo =    gpu.new_storage_buffer(buffer_size_mb * 1024 * 1024);
        let bvh_ssbo =          gpu.new_storage_buffer(buffer_size_mb * 1024 * 1024);
        let screen_ssbo =       gpu.new_storage_buffer(512 * 512 * 4 * 4);

        let rt_data_bg = gpu.new_bind_group()
            .with_buffer(&triangles_ssbo.view_all(), wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&bvh_ssbo.view_all(),       wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&screen_ssbo.view_all(),    wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .finish(&mut resources);

        // Load the shaders from disk
        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let screen_pipeline_layout = gpu.new_pipeline_layout(
            &resources, &[&u_frame, &rt_data_bg]
        );

        let surface_capabilities = gpu.surface.get_capabilities(&gpu.adapter);
        let surface_format = surface_capabilities.formats[0];

        let screen_pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&screen_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(surface_format.into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let raytrace_pipeline_layout = gpu.new_pipeline_layout(
            &resources, &[&u_frame, &rt_data_bg]
        );

        let raytrace_pipeline = gpu.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("raytrace compute pipeline"),
                module: &shader,
                layout: Some(&raytrace_pipeline_layout),
                entry_point: Some("cs_main"),
                compilation_options: Default::default(),
                cache: None,
            }
        );




        gpu.queue.write_buffer(&triangles_ssbo.raw, 0, bytemuck::cast_slice(triangles.as_slice()));

        Context {
            screen_pipeline,
            frame_uniforms: u_frame_0,
            frame_uniforms_buffer: u_frame_buffer.raw,
            frame_uniforms_binding: u_frame.raw,
            
            raytrace_pipeline,
            screen_ssbo: screen_ssbo.raw,
            bvh_ssbo: bvh_ssbo.raw,
            triangles_ssbo: triangles_ssbo.raw,
            rt_data_binding: rt_data_bg.raw,
            resources,
            triangles
        }
    }
}

fn frame(gpu: &Gpu, ctx: &mut Context) {
    let surface_texture = gpu.surface.get_current_texture().expect("Failed to acquire next swap chain texture");
    let view = surface_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None,
        });

    let rpassdesc = wgpu::RenderPassDescriptor {
        label: None,
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    };


    ctx.frame_uniforms.frame += 1;
    ctx.frame_uniforms.time += 1.0 / 60.0; // hack

    gpu.queue.write_buffer(&ctx.frame_uniforms_buffer, 0, bytemuck::bytes_of(&ctx.frame_uniforms));
    
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cpass.set_pipeline(&ctx.raytrace_pipeline);
        cpass.set_bind_group(0, &ctx.frame_uniforms_binding, &[]);
        cpass.set_bind_group(1, &ctx.rt_data_binding, &[]);
        cpass.dispatch_workgroups(64, 64, 1);
    }

    {
        let mut rpass = encoder.begin_render_pass(&rpassdesc);
        rpass.set_pipeline(&ctx.screen_pipeline);
        rpass.set_bind_group(0, Some(&ctx.frame_uniforms_binding), &[]);
        rpass.set_bind_group(1, Some(&ctx.rt_data_binding), &[]);
        rpass.draw(0..3, 0..1);
    }

    gpu.queue.submit(Some(encoder.finish()));
    surface_texture.present();
}

async fn run() {
    let event_loop = EventLoop::new().unwrap();
    let window = new_window(&event_loop);
    let mut gpu = Gpu::new(&window).await;
    let mut ctx = Context::init(&gpu);
    event_loop.run(
    move |event, target| {

        if let Event::WindowEvent {
            window_id: _,
            event,
        } = event
        {
            match event {
                WindowEvent::Resized(new_size) => {
                    // Reconfigure the surface with the new size
                    gpu.surface_config.width = new_size.width.max(1);
                    gpu.surface_config.height = new_size.height.max(1);
                    gpu.surface_config.width = gpu.surface_config.width.min(4096);
                    gpu.surface_config.height = gpu.surface_config.height.min(4096);

                    gpu.surface.configure(&gpu.device, &gpu.surface_config);
                    // On macos the window needs to be redrawn manually after resizing
                    gpu.window.request_redraw();
                }
                WindowEvent::RedrawRequested => {
                    gpu.window.request_redraw();
                    frame(&gpu, &mut ctx);
                }
                WindowEvent::CloseRequested => target.exit(),
                _ => {}
            };
        }
    })
    .unwrap();
}
pub fn main() {
        {
            #[cfg(not(target_arch = "wasm32"))]
            env_logger::init();
            #[cfg(not(target_arch = "wasm32"))]
            pollster::block_on(run())
        };
        
        {
            #[cfg(target_arch = "wasm32")]
            console_log::init().expect("could not initialize logger");
            #[cfg(target_arch = "wasm32")]
            wasm_bindgen_futures::spawn_local(run());
        };
}
use std::{borrow::Cow};
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FrameUniforms {
    res:    [u32;2],
    frame:  u32,
    time:   f32,
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
}

struct BGBuilder<'a> {
    layout_entries: Vec<wgpu::BindGroupLayoutEntry>,
    entries:        Vec<wgpu::BindGroupEntry<'a>>,
}


// struct BufferView {
//     ty: wgpu::BufferBindingType,
//     dynamic_offset_stride: Option<NonZero<u32>>,
//     offset: u32,
//     size: u32
// }

impl<'a> BGBuilder<'a> {
    fn new() -> BGBuilder<'a> {
        BGBuilder {
            layout_entries: Vec::new(),
            entries:        Vec::new(),
        }
    }

    fn with_buffer(&mut self, buffer: &'a wgpu::Buffer, visibility: wgpu::ShaderStages) -> &mut Self{
        let ty : wgpu::BufferBindingType = match buffer.usage() {
            _ if buffer.usage().contains(wgpu::BufferUsages::UNIFORM)  => wgpu::BufferBindingType::Uniform,
            _ if buffer.usage().contains(wgpu::BufferUsages::STORAGE)  => wgpu::BufferBindingType::Storage { read_only: false },
            _ => panic!("Invalid buffer usage: expected uniform or storage"),
        };

        let layout_entry = wgpu::BindGroupLayoutEntry {
            binding: self.layout_entries.len() as u32,
            count: None,
            visibility,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None }
        };

        self.layout_entries.push(layout_entry);

        let entry = wgpu::BindGroupEntry {
            binding: self.entries.len() as u32,
            resource: wgpu::BindingResource::Buffer(buffer.as_entire_buffer_binding()),
        };
        self.entries.push(entry);
        self
    }


    fn finish(&self, device: &wgpu::Device) -> (wgpu::BindGroup, wgpu::BindGroupLayout) {
        let layout_desc = wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: self.layout_entries.as_slice(),
        };
        let layout = device.create_bind_group_layout(&layout_desc);
        
        let desc = wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: self.entries.as_slice(),
        };

        (device.create_bind_group(&desc), layout)
    }
}



impl Context {
    fn init(
    adapter:    &wgpu::Adapter, 
    device:     &wgpu::Device, 
    queue:      &wgpu::Queue, 
    surface:    &wgpu::Surface) -> Context {

        let u_frame_buffer : wgpu::BufferDescriptor = wgpu::BufferDescriptor{
            label: Some("Frame Uniform Buffer"),
            size: size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        };

        let u_frame_buffer = device.create_buffer(&u_frame_buffer);

        let (u_frame_binding, u_frame_layout) = BGBuilder::new()
            .with_buffer(&u_frame_buffer, wgpu::ShaderStages::all())
            .finish(device);


        let triangles_ssbo : wgpu::BufferDescriptor = wgpu::BufferDescriptor{
            label: Some("Frame Uniform Buffer"),
            size: 127000000,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        };
        let triangles_ssbo = device.create_buffer(&triangles_ssbo);

        let bvh_ssbo : wgpu::BufferDescriptor = wgpu::BufferDescriptor{
            label: Some("Frame Uniform Buffer"),
            size: 127000000,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        };
        let bvh_ssbo = device.create_buffer(&bvh_ssbo);

        let screen_ssbo : wgpu::BufferDescriptor = wgpu::BufferDescriptor{
            label: Some("Frame Uniform Buffer"),
            size: 512 * 512 * 4 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        };
        let screen_ssbo = device.create_buffer(&screen_ssbo);

        let (rt_data_binding, rt_data_layout) = BGBuilder::new()
            .with_buffer(&triangles_ssbo, wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&bvh_ssbo,       wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&screen_ssbo,    wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .finish(device);


        // Load the shaders from disk
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let screen_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&u_frame_layout, &rt_data_layout],
            push_constant_ranges: &[],
        });

        let surface_capabilities = surface.get_capabilities(&adapter);
        let surface_format = surface_capabilities.formats[0];

        let screen_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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

        let raytrace_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("raytrace compute pipeline layout"),
            bind_group_layouts: &[&u_frame_layout, &rt_data_layout],
            push_constant_ranges: &[],
        });

        let raytrace_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("raytrace compute pipeline"),
            module: &shader,
            layout: Some(&raytrace_pipeline_layout),
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });


        Context {
            screen_pipeline,
            frame_uniforms: FrameUniforms { res: [512, 512], frame: 0, time: 0.0 },
            frame_uniforms_buffer: u_frame_buffer,
            frame_uniforms_binding: u_frame_binding,
            
            raytrace_pipeline,
            screen_ssbo,
            bvh_ssbo,
            triangles_ssbo,
            rt_data_binding,
            
        }
    }
}

fn frame(device: &wgpu::Device, queue: &wgpu::Queue, surface: &wgpu::Surface, ctx: &mut Context) {
    let surface_texture = surface.get_current_texture().expect("Failed to acquire next swap chain texture");
    let view = surface_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
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
    ctx.frame_uniforms.time += 1.0 / 16.0;

    queue.write_buffer(&ctx.frame_uniforms_buffer, 0, bytemuck::bytes_of(&ctx.frame_uniforms));
    
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

    queue.submit(Some(encoder.finish()));
    surface_texture.present();
}

async fn run(event_loop: EventLoop<()>, window: Window) {
    let mut size = window.inner_size();
    size.width = size.width.max(1);
    size.height = size.height.max(1);

    size.width = size.width.min(4096);
    size.height = size.height.min(4096);

    let instance = wgpu::Instance::default();

    let surface = instance.create_surface(&window).unwrap();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            // Request an adapter which can render to our surface
            compatible_surface: Some(&surface),
        })
        .await
        .expect("Failed to find an appropriate adapter");


    let device_desc = wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::empty(),
        // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the swapchain.
        required_limits: wgpu::Limits::default()
            .using_resolution(adapter.limits()),
        memory_hints: wgpu::MemoryHints::MemoryUsage,
    };

    // Create the logical device and command queue
    let (device, queue) = adapter.request_device(&device_desc, None)
        .await
        .expect("Failed to create device");

    let mut config = surface
        .get_default_config(&adapter, size.width, size.height)
        .unwrap();

    surface.configure(&device, &config);

    let mut ctx = Context::init(&adapter, &device, &queue, &surface);

    let window = &window;
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
                        config.width = new_size.width.max(1);
                        config.height = new_size.height.max(1);
                        config.width = config.width.min(4096);
                        config.height = config.height.min(4096);

                        surface.configure(&device, &config);
                        // On macos the window needs to be redrawn manually after resizing
                        window.request_redraw();
                    }
                    WindowEvent::RedrawRequested => {
                        window.request_redraw();
                        frame(&device, &queue, &surface, &mut ctx);
                        
                    }
                    WindowEvent::CloseRequested => target.exit(),
                    _ => {}
                };
            }
        })
        .unwrap();
}

pub fn main() {
    let event_loop = EventLoop::new().unwrap();
    #[allow(unused_mut)]
    let mut builder = winit::window::WindowBuilder::new();
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        use wasm_bindgen::JsCast;
        use winit::platform::web::WindowBuilderExtWebSys;
        let canvas = web_sys::window()
            .unwrap()
            .document()
            .unwrap()
            .get_element_by_id("canvas")
            .unwrap()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();
        builder = builder.with_canvas(Some(canvas));
    }
    let window = builder.build(&event_loop).unwrap();

    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        pollster::block_on(run(event_loop, window));
    }
    #[cfg(target_arch = "wasm32")]
    {
        
        console_log::init().expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run(event_loop, window));
    }
}
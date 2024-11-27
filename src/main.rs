use std::{borrow::Cow, collections::{HashMap, HashSet}};
use gltf::Gltf;
use rand::random;
use winit::{
    dpi::PhysicalPosition, event::{DeviceEvent, ElementState, Event, MouseButton, WindowEvent}, event_loop::EventLoop, keyboard::{KeyCode, PhysicalKey}
};

use glam::{vec2, vec3, vec4, Mat3, Mat4, UVec2, Vec2, Vec3, Vec4, Vec4Swizzles};
use web_time::{Instant, SystemTime};
mod input;
use input::*;

// arbitrary and probably not ideal but im not going to keep thinking about it
// right handed



#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FrameUniforms {
    scene: GpuSceneUniform,
    res:    [u32;2],
    frame:  u32,
    time:   f32,
    reject_hist: u32,
    _pad: [u32; 3]

}

#[repr(C)]
#[derive(Default, Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PointLight {
    position: Vec4,
    intensity: Vec4,
}


#[repr(C)]
#[derive(Default, Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DirectionalLight {
    direction: Vec4,
    intensity: Vec4,
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

struct BvhNode {
    // child format:
    // bits 1-0:
    //  00: nothing
    //  01: child is a bvh node
    //  10: child is a triangle
    // bits 32-2: index of triangles or bvh nodes 
    children: [u32; 2],
}
impl BvhNode {
    const EMPTY: u32 = 0b00;
    const NODE: u32 = 0b01;
    const TRI: u32 = 0b10;
    fn new() -> BvhNode {
        BvhNode {
            children: [0, 0]
        }
    }
    fn root(&self) -> bool {
        self.children[0] & 0b11 != Self::NODE && self.children[1] & 0b11 != Self::NODE
    }
}
struct Bvh {
    nodes: Vec<BvhNode>,
    triangles: Vec<Tri>,
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

    scene: FlatScene,
}

struct MatrixStack {
    stack: Vec<Mat4>,
}

impl MatrixStack {
    fn new() -> Self {
        Self {stack: vec![Mat4::IDENTITY]}
    }
    fn top(&mut self) -> &Mat4 {
        self.stack.last().unwrap()
    }
    fn push(&mut self) {
        self.stack.push(self.stack.last().copied().unwrap());
    }
    fn pop(&mut self) {
        if self.stack.len() > 1 {
            self.stack.pop();
        }
    }
    fn apply(&mut self, t: &Mat4) {
        if self.stack.len() == 1 {
            self.push();
        }
        *self.stack.last_mut().unwrap() = self.top().mul_mat4(t);
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSceneUniform {
    point_lights: [PointLight; 12],
    directional_lights: [DirectionalLight; 4],
    camera: GpuCamera,
    tri_count: u32,
    num_point_lights: u32,
    num_directional_lights: u32,
    _pad: u32,
}

#[derive(Default)]
struct FlatScene {
    triangles: Vec<Tri>,
    cameras:   Vec<Camera>,
    point_lights: Vec<PointLight>,
    directional_lights: Vec<DirectionalLight>,
}

impl FlatScene {
    fn add_gltf_bytes(&mut self, bytes: &[u8]) {
        let (document, buffers, _) = gltf::import_slice(include_bytes!("../resources/simple.glb")).unwrap();
        let mut ms = MatrixStack::new();
        for scene in document.scenes(){
            for node in scene.nodes() {
                self.add_gltf_node(&buffers, node, &mut ms);
            }
        }
    }   

    fn add_gltf_node(&mut self, buffers: &Vec<gltf::buffer::Data>, node: gltf::Node, ms: &mut MatrixStack) {
        ms.push();
        ms.apply(&Mat4::from_cols_array_2d(&node.transform().matrix()));
        
        if let Some(camera) = node.camera() {
            self.cameras.push(Camera::from_gltf(camera, ms.top()));
        }
    
        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                if primitive.mode() == gltf::mesh::Mode::Triangles {
                
                    let mut idx = 0;
                    let mut triangle = [Vec3::new(0.0, 0.0, 0.0); 3];
    
                    // TODO: figure out what this lambda that I copied does
                    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
    
                    let positions: Vec<[f32; 3]> = reader.read_positions().unwrap().map(|p| ms.top().mul_vec4(Vec4::from_slice(&[p[0], p[1], p[2], 1.0])).xzy().to_array()).collect();
    
                    if let Some(indices) = reader.read_indices() {
                        // indexed mesh
    
                        let indices = indices.into_u32();
                        for index in indices {
                            let position = positions[index as usize];
                            triangle[idx] = Vec3::from_array(position);
                            idx += 1;
                            if idx > 2 {
                                idx = 0;
                                self.triangles.push(Tri::new(triangle[0], triangle[1], triangle[2]));
                                // println!("loaded triangle at ({}, {}, {})", triangle[0].x, triangle[0].y, triangle[0].z);
                            }
                        }
                    }
                    else {
                        // non-indexed mesh
    
                        for position in positions {
                            triangle[idx] = Vec3::from_array(position);
                            idx += 1;
                            if idx > 2 {
                                idx = 0;
                                self.triangles.push(Tri::new(triangle[0], triangle[1], triangle[2]));
                                // println!("loaded triangle at ({}, {}, {})", triangle[0].x, triangle[0].y, triangle[0].z);
                            }
                        }
                    }
                } else {
                    panic!("Non-triangle primitive");
                }
            }
        }

        for child in node.children() {
            self.add_gltf_node(buffers, child, ms);
        }

        ms.pop();
    }

    fn to_gpu(&self) -> GpuSceneUniform {
        let mut point_lights = [PointLight::default(); 12];
        let mut directional_lights = [DirectionalLight::default(); 4];

        for i in 0..self.point_lights.len().min(point_lights.len()) {
            point_lights[i] = self.point_lights[i];
        }

        for i in 0..self.directional_lights.len().min(directional_lights.len()) {
            directional_lights[i] = self.directional_lights[i];
        }

        GpuSceneUniform {
            _pad: 0,
            camera: self.cameras[0].to_gpu(),
            point_lights,
            directional_lights,
            num_directional_lights: self.directional_lights.len() as u32,
            num_point_lights: self.point_lights.len() as u32,
            tri_count: self.triangles.len() as u32,
        }
    }
}

impl Context {
    fn init(gpu: &Gpu) -> Context {
        let mut scene = FlatScene::default();
        scene.add_gltf_bytes(include_bytes!("../resources/simple_with_camera.glb"));

        if scene.cameras.is_empty() {
            println!("No camera in scene, falling back to default");
            // vec3f(-3.5, -0.5, 0.5), vec3f(1.0, 0.0, 0.0)
            scene.cameras.push(Camera::default());
        }

        let mut resources = ResourceManager::new();

        let u_frame_0 = FrameUniforms {
            scene: scene.to_gpu(),
            frame: 0,
            res: [512, 512],
            time: 0.0,
            reject_hist: 1,
            _pad: [0; 3],
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

        let surface_format = gpu.surface_config.format;

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
                targets: &[Some(surface_format.add_srgb_suffix().into())],
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

        gpu.queue.write_buffer(&triangles_ssbo.raw, 0, bytemuck::cast_slice(scene.triangles.as_slice()));

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
            scene,
        }
    }
}

fn frame(gpu: &Gpu, ctx: &mut Context) {
    let surface_texture = gpu.surface.get_current_texture().expect("Failed to acquire next swap chain texture");

    let mut surface_view_desc = wgpu::TextureViewDescriptor::default();
    surface_view_desc.format =  Some(gpu.surface_config.view_formats.iter().find(|f| f.is_srgb()).copied().unwrap_or(gpu.surface_config.format));
    let view = surface_texture.texture.create_view(&surface_view_desc);

    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None,
        });

    let rpass_desc = wgpu::RenderPassDescriptor {
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

    // ctx.camera.rotate(f32::to_radians(0.0), f32::to_radians(5.0));

    // TODO: fix time and camera situation
    ctx.frame_uniforms.frame += 1;
    ctx.frame_uniforms.time += 1.0 / 60.0; // hack
    ctx.frame_uniforms.scene.camera = ctx.scene.cameras[0].to_gpu();
    ctx.frame_uniforms.reject_hist = ctx.scene.cameras[0].check_moved() as u32;
    
    gpu.queue.write_buffer(&ctx.frame_uniforms_buffer, 0, bytemuck::bytes_of(&ctx.frame_uniforms));
    
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cpass.set_pipeline(&ctx.raytrace_pipeline);
        cpass.set_bind_group(0, &ctx.frame_uniforms_binding, &[]);
        cpass.set_bind_group(1, &ctx.rt_data_binding, &[]);
        cpass.dispatch_workgroups(64, 64, 1);
    }

    {
        let mut rpass = encoder.begin_render_pass(&rpass_desc);
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
    let mut input = InputState {
        keys: HashSet::new(),
        mouse_x: 0.0,
        mouse_y: 0.0,
        scroll: 0.0,
        lmb: false,
        rmb: false,
    };
    let mut this_frame = Instant::now();
    let mut last_frame = Instant::now();
    event_loop.run(
    move |event, target| {
        match event {
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion{ delta, },
                .. // We're not using device_id currently
            } => {
                input.mouse_x += delta.0;
                input.mouse_y += delta.1;
            },
            Event::WindowEvent { window_id, event } => {
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
                        this_frame = Instant::now();
                        gpu.window.request_redraw();
                        let dt = (this_frame - last_frame).as_secs_f32();
                        ctx.scene.cameras[0].update(&mut input, dt);
                        frame(&gpu, &mut ctx);
                        last_frame = this_frame;
                    },
                    WindowEvent::MouseInput { device_id, state, button } => {
                        match button {
                            MouseButton::Left =>  input.lmb = state.is_pressed(),
                            MouseButton::Right => input.rmb = state.is_pressed(),
                            _ => (),
                        }
                    }
                    WindowEvent::MouseWheel { device_id, delta, phase } => {
                        match delta {
                            winit::event::MouseScrollDelta::LineDelta(_, y) => input.scroll += y as f64 / 2.0,
                            winit::event::MouseScrollDelta::PixelDelta(physical_position) => input.scroll += physical_position.y / 128.0,
                        }
                    }
                    WindowEvent::CloseRequested => target.exit(),
                    WindowEvent::KeyboardInput { device_id, event, is_synthetic } => {
                        
                        match event.physical_key {
                            PhysicalKey::Code(code) => {
                                if event.state.is_pressed() {
                                    input.keys.insert(PhysicalKey::Code(code));
                                } else {
                                    input.keys.remove(&PhysicalKey::Code(code));
                                }
                            }
                            _ => ()
                        }
                    },
                    _ => {}
                };
            }

            _ => (),
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
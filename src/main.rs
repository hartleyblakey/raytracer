use std::{borrow::Cow, collections::HashSet, f32::consts::PI, mem::swap, str::FromStr, sync::{Arc, Mutex}};

use pollster::FutureExt;

use js_sys::ArrayBuffer;
use rand::random;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::{spawn_local, JsFuture};
use web_sys::Response;


use winit::{
    dpi::PhysicalPosition, event::{DeviceEvent, ElementState, Event, MouseButton, WindowEvent}, event_loop::EventLoop, keyboard::{KeyCode, PhysicalKey}, window::CursorGrabMode
};

use glam::{uvec2, Mat4};
use web_time::{Instant, SystemTime};

mod input;
use input::*;

mod gpu;
use gpu::*;

mod scene;
use scene::*;

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
    node_count: u32,
    prim_count: u32,
    _pad: u32,

}

struct Context {
    screen_pipeline:            wgpu::RenderPipeline,
    screen_pipeline_layout:     wgpu::PipelineLayout,
    raytrace_pipeline:          wgpu::ComputePipeline,
    raytrace_pipeline_layout:   wgpu::PipelineLayout,

    shader_compiled_timestamp:  SystemTime, 

    shader_module:              wgpu::ShaderModule,

    triangles_ssbo:             Buffer,
    bvh_ssbo:                   Buffer,
    screen_ssbo:                Buffer,
    triangles_ext_ssbo:         Buffer,
    texture_data_ssbo:          Buffer,
    primitive_data_ssbo:        Buffer,

    env_map_texture:            Texture,

    rt_data_binding:            BindGroup,

    frame_uniforms_binding:     BindGroup,
    frame_uniforms_buffer:      Buffer,
    frame_uniforms:             FrameUniforms,

    resources:                  ResourceManager,

    scene:                      Scene,

    should_reupload:            bool,
}

impl Context {

    fn update_resolution(&mut self, gpu: &Gpu) {
        let res = [gpu.surface_config.width, gpu.surface_config.height];
        self.frame_uniforms.res = res;
        println!("x: {}, y: {}", res[0], res[1]);
        self.screen_ssbo = gpu.new_storage_buffer(res[0] as u64 * res[1]  as u64 * 4 * 4);

        self.rt_data_binding = gpu.new_bind_group()
            .with_buffer(&self.triangles_ssbo.view_all(),       wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&self.triangles_ext_ssbo.view_all(),   wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&self.bvh_ssbo.view_all(),             wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&self.screen_ssbo.view_all(),          wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&self.texture_data_ssbo.view_all(),    wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&self.primitive_data_ssbo.view_all(),  wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_texture(&self.env_map_texture,                wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .finish(&mut self.resources);
    }

    fn create_pipelines(
        shader_module: &wgpu::ShaderModule,
        screen_pipeline_layout: &wgpu::PipelineLayout, 
        raytrace_pipeline_layout: &wgpu::PipelineLayout, 
        gpu: &Gpu) -> (wgpu::RenderPipeline, wgpu::ComputePipeline) {

        
        let screen_pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&screen_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(gpu.surface_config.format.add_srgb_suffix().into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let raytrace_pipeline = gpu.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("raytrace compute pipeline"),
                module: &shader_module,
                layout: Some(&raytrace_pipeline_layout),
                entry_point: Some("cs_main"),
                compilation_options: Default::default(),
                cache: None,
            }
        );

        (screen_pipeline, raytrace_pipeline)
    }

    fn check_recompile_shader(&mut self, gpu: &Gpu) -> bool {
    #[cfg(not(target_arch = "wasm32"))] 
    {
        const SHADER_PATH: &str = "src/shader.wgsl";

        let metadata = std::fs::metadata(SHADER_PATH).unwrap();
        let last_write_time = metadata.modified().unwrap();
        
        if last_write_time <= self.shader_compiled_timestamp {
            return false;
        }
        self.shader_compiled_timestamp = std::time::SystemTime::now();

        let shader_module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(std::fs::read_to_string(SHADER_PATH).unwrap().as_str())),
        });

        let compilation_info = shader_module.get_compilation_info().block_on().messages;
        if !compilation_info.is_empty() {
            return false;
        }

        let (screen, rt) = Self::create_pipelines(
            &shader_module, 
            &self.screen_pipeline_layout, 
            &self.raytrace_pipeline_layout, 
            gpu);
        
        self.screen_pipeline = screen;
        self.raytrace_pipeline = rt;
        self.shader_module = shader_module;
        return true;
        
    }
        false
    }

    async fn init<'a>(gpu: &'a Gpu<'a>) -> Context {
        let scene = Scene::from_path("resources/simple.glb", "resources/trail.hdr").await.unwrap();

        println!("Bvh size : {} mb", (scene.bvh_node_data.len() * size_of::<BvhNode>()) / (1000 * 1000));
        let mut resources = ResourceManager::new();

        let u_frame_0 = FrameUniforms {
            scene: scene.to_gpu(),
            frame: 0,
            res: [gpu.surface_config.width, gpu.surface_config.height],
            time: 0.0,
            reject_hist: 1,
            node_count: scene.bvh_node_data.len() as u32,
            prim_count: scene.primitives.len() as u32,
            _pad: 0,
        };

        let u_frame_buffer = gpu.new_uniform_buffer(&u_frame_0);

        let u_frame = gpu.new_bind_group()
            .with_buffer(&u_frame_buffer.view_all(), wgpu::ShaderStages::all())
            .finish(&mut resources);

        // just make everything 128mb for simplicity
        let max_buffer_size_mb = 128;

        let triangles_ssbo =        gpu.new_storage_buffer(max_buffer_size_mb * 1024 * 1024);
        let bvh_ssbo =              gpu.new_storage_buffer(max_buffer_size_mb * 1024 * 1024);
        let triangles_ext_ssbo =    gpu.new_storage_buffer(max_buffer_size_mb * 1024 * 1024);
        let texture_data_ssbo =     gpu.new_storage_buffer(1024 * 1024 * 1024);
        let primitive_data_ssbo =   gpu.new_storage_buffer(max_buffer_size_mb * 1024 * 1024);
        let screen_ssbo =           gpu.new_storage_buffer(u_frame_0.res[0] as u64 * u_frame_0.res[1] as u64 * 4 * 4);

        let hdri_height = f32::sqrt(scene.env_map_data.len() as f32 / 2.0) as u32; // 4 channels
        let env_map_texture = gpu.new_texture(uvec2(2 * hdri_height, hdri_height), wgpu::TextureFormat::Rgba32Float, false);

        let rt_data_bg = gpu.new_bind_group()
            .with_buffer(&triangles_ssbo.view_all(),        wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&triangles_ext_ssbo.view_all(),    wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&bvh_ssbo.view_all(),              wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&screen_ssbo.view_all(),           wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&texture_data_ssbo.view_all(),     wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&primitive_data_ssbo.view_all(),   wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_texture(&env_map_texture,                 wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .finish(&mut resources);

        // fetch shader
        let shader_module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                Cow::Borrowed(
                    std::str::from_utf8(
                        fetch_bytes("src/shader.wgsl").await.unwrap().as_slice()
                    ).expect("Shader is not valid UTF-8")
                )
            ),
        });

        let screen_pipeline_layout = gpu.new_pipeline_layout(
            &resources, &[&u_frame, &rt_data_bg]
        );

        let raytrace_pipeline_layout = gpu.new_pipeline_layout(
            &resources, &[&u_frame, &rt_data_bg]
        );

        let (screen_pipeline, raytrace_pipeline) = Self::create_pipelines(
            &shader_module, 
            &screen_pipeline_layout, 
            &raytrace_pipeline_layout, 
            gpu
        );

        // gpu.queue.write_buffer(&triangles_ssbo.raw,         0, bytemuck::cast_slice(scene.tris.as_slice()));
        // gpu.queue.write_buffer(&triangles_ext_ssbo.raw,     0, bytemuck::cast_slice(scene.tri_exts.as_slice()));
        // gpu.queue.write_buffer(&bvh_ssbo.raw,               0, bytemuck::cast_slice(scene.bvh_node_data.as_slice()));
        // gpu.queue.write_buffer(&texture_data_ssbo.raw,      0, bytemuck::cast_slice(scene.texture_data.as_slice()));
        // gpu.queue.write_buffer(&primitive_data_ssbo.raw,    0, bytemuck::cast_slice(scene.primitives.as_slice()));

        // gpu.queue.write_texture(
        //     env_map_texture.raw.as_image_copy(), 
        //     bytemuck::cast_slice(scene.env_map_data.as_slice()), 
        //     wgpu::ImageDataLayout {
        //         offset: 0,
        //         bytes_per_row: Some(hdri_height * 2 * 4 * 4),
        //         rows_per_image: None,
        //     }, 
        //     wgpu::Extent3d{
        //         width: hdri_height * 2,
        //         height: hdri_height,
        //         depth_or_array_layers: 1,
        //     },
        // );

        let should_reupload = true;

        Context {
            screen_pipeline,
            screen_pipeline_layout,
            shader_module,

            shader_compiled_timestamp: SystemTime::now(),

            frame_uniforms: u_frame_0,
            frame_uniforms_buffer: u_frame_buffer,
            frame_uniforms_binding: u_frame,
            
            raytrace_pipeline,
            raytrace_pipeline_layout,
            screen_ssbo,
            bvh_ssbo,
            triangles_ssbo,
            triangles_ext_ssbo,
            texture_data_ssbo,
            primitive_data_ssbo,

            env_map_texture,

            rt_data_binding: rt_data_bg,

            resources,
            scene,

            should_reupload,
        }
    }

    async fn try_change_scene(&mut self, mesh_path: &str, env_map_path: &str) {
        if let Some(mesh_bytes) = fetch_bytes(mesh_path).await {
            self.try_change_scene_bytes(&mesh_bytes, env_map_path).await
        }
    }

    async fn try_change_scene_bytes(&mut self, mesh_bytes: &[u8], env_map_path: &str) {
        println!("Attempting to change scene");
        if let Some(scene) = Scene::from_bytes(mesh_bytes, env_map_path).await {
            self.scene = scene;
            self.frame_uniforms.scene = self.scene.to_gpu();
            self.frame_uniforms.reject_hist = 1;
            self.frame_uniforms.node_count = self.scene.bvh_node_data.len() as u32;
            self.frame_uniforms.prim_count = self.scene.primitives.len() as u32;

            self.scene.focus_camera(0);

            self.should_reupload = true;
        } else {
            println!("Scene change failed!");
        }
        
    }

    fn upload_scene(&mut self, gpu: &Gpu) {
        println!("Uploading scene to the gpu");
        gpu.queue.write_buffer(&self.triangles_ssbo,         0, bytemuck::cast_slice(self.scene.tris.as_slice()));
        gpu.queue.write_buffer(&self.triangles_ext_ssbo,     0, bytemuck::cast_slice(self.scene.tri_exts.as_slice()));
        gpu.queue.write_buffer(&self.bvh_ssbo,               0, bytemuck::cast_slice(self.scene.bvh_node_data.as_slice()));
        gpu.queue.write_buffer(&self.texture_data_ssbo,      0, bytemuck::cast_slice(self.scene.texture_data.as_slice()));
        gpu.queue.write_buffer(&self.primitive_data_ssbo,    0, bytemuck::cast_slice(self.scene.primitives.as_slice()));

        gpu.queue.write_texture(
            self.env_map_texture.as_image_copy(), 
            bytemuck::cast_slice(self.scene.env_map_data.as_slice()), 
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(self.env_map_texture.height() * 2 * 4 * 4),
                rows_per_image: None,
            }, 
            wgpu::Extent3d{
                width: self.env_map_texture.width(),
                height: self.env_map_texture.height(),
                depth_or_array_layers: 1,
            },
        );

        self.should_reupload = false;

    }
}

fn frame(gpu: &Gpu, ctx: &mut Context, dt: f32) {
    let surface_texture = gpu.surface.get_current_texture().expect("Failed to aquire next surface texture");
    let surface_view = gpu.get_surface_view(&surface_texture);

    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: None,
    });
    
    let rpass_desc = wgpu::RenderPassDescriptor {
        label: None,
        color_attachments: &[Some(surface_view.attachment())],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    };

    ctx.frame_uniforms.frame += 1;
    ctx.frame_uniforms.time += dt; // hack
    ctx.frame_uniforms.scene.camera = ctx.scene.cameras[0].to_gpu();

    if ctx.check_recompile_shader(gpu) || ctx.scene.cameras[0].check_moved() {
        ctx.frame_uniforms.reject_hist = 1;
    }
    
    gpu.queue.write_buffer(&ctx.frame_uniforms_buffer, 0, bytemuck::bytes_of(&ctx.frame_uniforms));
    
    let workgroup_size = [8, 8];
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cpass.set_pipeline(&ctx.raytrace_pipeline);
        cpass.set_bind_group(0, &ctx.frame_uniforms_binding.raw, &[]);
        cpass.set_bind_group(1, &ctx.rt_data_binding.raw, &[]);
        cpass.dispatch_workgroups(
            (ctx.frame_uniforms.res[0] + workgroup_size[0] - 1) / workgroup_size[0],
            (ctx.frame_uniforms.res[1] + workgroup_size[1] - 1) / workgroup_size[1], 
            1
        );
    }

    {
        let mut rpass = encoder.begin_render_pass(&rpass_desc);
        rpass.set_pipeline(&ctx.screen_pipeline);
        rpass.set_bind_group(0, Some(&ctx.frame_uniforms_binding.raw), &[]);
        rpass.set_bind_group(1, Some(&ctx.rt_data_binding.raw), &[]);
        rpass.draw(0..3, 0..1);
    }

    ctx.frame_uniforms.reject_hist = 0;

    gpu.queue.submit(Some(encoder.finish()));
    surface_texture.present();
}

/// Fetch the bytes of a file
/// 
/// # Panics
/// when targetting WASM, panics if the file path is not found
async fn fetch_bytes(path: &str) -> Option<Vec<u8>> {
    #[cfg(not(target_arch = "wasm32"))] 
    {
        if let Ok(bytes) = std::fs::read(path) {
            Some(bytes)
        } else {
            None
        }

    }

    
    #[cfg(target_arch = "wasm32")] 
    {
        let mut web_path = String::from_str("../").unwrap();
        web_path.push_str(path);
        // let opts = web_sys::RequestInit::new();
        // opts.set_method("GET");
        // opts.set_mode(web_sys::RequestMode::Cors);
        // let request = web_sys::Request::new_with_str_and_init(web_path.as_str(), &opts).unwrap();
        let response: Response = JsFuture::from(web_sys::window().unwrap().fetch_with_str(web_path.as_str())).await.unwrap().dyn_into().unwrap();
        let array_buf = JsFuture::from(response.array_buffer().unwrap()).await.unwrap();
        // let response: ArrayBuffer = JsFuture::from(web_sys::window().unwrap().fetch_with_str(path)).await.unwrap().dyn_into().unwrap();
        // assert!(wasm_bindgen::JsCast::is_instance_of::<ArrayBuffer>(&response));
        let typed_arr = js_sys::Uint8Array::new(&array_buf);
        // web_sys::console::log_1(&response);
        // web_sys::console::log(&js_sys::Array::from(&typed_arr));
        Some(typed_arr.to_vec())
    }
}

#[cfg(target_arch = "wasm32")]
pub fn async_spawn<F>(fut: F)
where
    F: std::future::Future<Output = ()> + 'static,
{
    spawn_local(fut);
}

#[cfg(not(target_arch = "wasm32"))]
pub fn async_spawn<F>(fut: F)
where
    F: std::future::Future<Output = ()>  + 'static,
{
    pollster::block_on(fut);
}

async fn run() {
    let event_loop = EventLoop::new().unwrap();

    // default size
    let window = new_window(&event_loop, [512, 512]);

    let mut gpu = Gpu::new(&window).await;

    let mut ctx = Arc::new(Mutex::new(Context::init(&gpu).await));

    let mut input = InputState {
        keys: HashSet::new(),
        mouse_x: 0.0,
        mouse_y: 0.0,
        scroll: 0.0,
        lmb: false,
        rmb: false,
    };

    let mut last_second = Instant::now();
    let mut last_frame  = Instant::now();
    let mut this_frame  = Instant::now();
    let mut frames_in_second: u32 = 0;
    let mut last_cursor_pos = PhysicalPosition::new(0.0, 0.0);

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
                        gpu.surface_config.width  = new_size.width.clamp(1, 4096);
                        gpu.surface_config.height = new_size.height.clamp(1, 4096);
                        
                        gpu.surface.configure(&gpu.device, &gpu.surface_config);

                        if let Ok(mut ctx_guard) = ctx.try_lock(){
                            ctx_guard.update_resolution(&gpu);
                        }
                        
                        // On macos the window needs to be redrawn manually after resizing
                        gpu.window.request_redraw();
                    }

                    WindowEvent::RedrawRequested => {
                        this_frame = Instant::now();
                        frames_in_second += 1;
                        gpu.window.request_redraw();
                        let dt = (this_frame - last_frame).as_secs_f32();

                        

                        if let Ok(mut ctx_guard) = ctx.try_lock() {
                            ctx_guard.scene.cameras[0].update(&mut input, dt);
                            
                            if ctx_guard.should_reupload {
                                ctx_guard.upload_scene(&gpu);
                            }
                            frame(&gpu, &mut ctx_guard, dt);
                        } else {
                            println!("Context is in use!");
                        }

                        if this_frame.duration_since(last_second).as_secs_f32() >= 1.0 {
                            println!("fps: {}", frames_in_second);
                            frames_in_second = 0;
                            last_second = this_frame;
                        }
                        
                        
                        last_frame = this_frame;
                    },
                    WindowEvent::CursorMoved { device_id, position } => if !input.rmb {last_cursor_pos = position},
                    WindowEvent::MouseInput { device_id, state, button } => {
                        match button {
                            MouseButton::Left =>  {
                                if let Ok(mut ctx_guard) = ctx.try_lock(){
                                    input.lmb = state.is_pressed();
                                    ctx_guard.scene.focus_camera(0);
                                }

                            },
                            MouseButton::Right => input.rmb = state.is_pressed(),
                            _ => (),
                        }

                        // hide the curson when moving the camera
                        // and reset it back when released
                        if input.rmb {
                            gpu.window.set_cursor_visible(false);
                            gpu.window.set_cursor_grab(CursorGrabMode::Confined);
                        } else {
                            gpu.window.set_cursor_position(last_cursor_pos);
                            gpu.window.set_cursor_visible(true);
                            gpu.window.set_cursor_grab(CursorGrabMode::None);

                        }

                    }
                    WindowEvent::MouseWheel { device_id, delta, phase } => {
                        // hack: I have no idea how to keep a consistent sensitivity between these
                        //       two units. This works well enough for the devices I tested it on
                        match delta {
                            winit::event::MouseScrollDelta::LineDelta(_, y) => input.scroll += y as f64 / 2.0,
                            winit::event::MouseScrollDelta::PixelDelta(physical_position) => input.scroll += physical_position.y / 128.0,
                        }
                    },
                    WindowEvent::DroppedFile(path) => {
                        if let Some(path_string) = path.to_str() {
                            let path_string = path_string.to_string();
                            let ctx_clone = Arc::clone(&ctx);
                            async_spawn(async move {
                                if let Ok(mut ctx_guard) = ctx_clone.lock() {
                                    ctx_guard.try_change_scene(path_string.as_str(), "resources/trail.hdr").await;
                                }
                            });

                            // Im not sure why, but the window sometimes needs to be manually redrawn here
                            gpu.window.request_redraw();
                        }
                        
                    },
                    WindowEvent::CloseRequested => target.exit(),
                    WindowEvent::KeyboardInput { device_id, event, is_synthetic } => {
                        match event.physical_key {
                            PhysicalKey::Code(code) => {
                                if event.state.is_pressed() {
                                    input.keys.insert(PhysicalKey::Code(code));
                                    if code == KeyCode::KeyO {
                                        let ctx_clone = Arc::clone(&ctx);
                                        async_spawn(async move {
                                            if let Ok(mut ctx_guard) = ctx_clone.lock() {
                                                let file = rfd::AsyncFileDialog::new().set_title("Pick a gltf (or glb) file to render").pick_file().await.unwrap();
                                                ctx_guard.try_change_scene_bytes(&file.read().await, "resources/trail.hdr").await
                                            }
                                        });
                                            
                                    };
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
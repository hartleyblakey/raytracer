



use std::{borrow::Cow, collections::HashSet, sync::{Arc, Mutex}};

use pollster::FutureExt;

#[cfg(target_arch = "wasm32")]
use js_sys::ArrayBuffer;

use wasm_bindgen::{prelude::wasm_bindgen, JsValue};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{JsCast, JsError};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::{spawn_local, JsFuture};


#[cfg(target_arch = "wasm32")]
use web_sys::Response;


use winit::{
    dpi::PhysicalPosition, event::{DeviceEvent, Event, MouseButton, WindowEvent}, event_loop::EventLoop, keyboard::{KeyCode, PhysicalKey}, window::CursorGrabMode
};



use glam::uvec2;
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
    debug_mode: u32,
}

const SHADER_PATH: &'static str = "./src/shader.wgsl";
const DEFAULT_MODEL_PATH: &'static str = "./resources/simple2.glb";
const DEFAULT_ENV_PATH: &'static str = "./resources/trail.hdr";

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

    scene:                      RenderScene,

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

    #[cfg(target_arch = "wasm32")]
        false
    }

    async fn init<'a>(gpu: &'a Gpu<'a>) -> Context {
        let scene = RenderScene::from_path(DEFAULT_MODEL_PATH, DEFAULT_ENV_PATH).await.unwrap();

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
            debug_mode: 0,
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
                        fetch_bytes(SHADER_PATH).await.unwrap().as_slice()
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
        if let Some(scene) = RenderScene::from_bytes(mesh_bytes, env_map_path).await {
            self.scene = scene;
            self.frame_uniforms.scene = self.scene.to_gpu();
            self.frame_uniforms.reject_hist = 1;
            self.frame_uniforms.node_count = self.scene.bvh_node_data.len() as u32;
            self.frame_uniforms.prim_count = self.scene.primitives.len() as u32;

            // self.scene.focus_camera(0);
            // self.scene.cameras[0].update(&mut InputState::default(), 1.0 / 60.0);

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
            wgpu::TexelCopyBufferLayout {
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
    let surface_texture = gpu.surface.get_current_texture().expect("Failed to acquire next surface texture");
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


/// Fetch the bytes of a file. Returns None if an error occurred
/// 
/// # Panics
/// when targeting WASM, panics if the file path is not found
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
        let Ok(js_future) = JsFuture::from(web_sys::window()?.fetch_with_str(path)).await 
            else {return None};

        let Ok(response) = js_future.dyn_into::<Response>()
            else {return None};

        let Ok(array_buf) = response.array_buffer()
            else {return None};

        let Ok(array_buf) = JsFuture::from(array_buf).await 
            else {return None};

        let typed_arr = js_sys::Uint8Array::new(&array_buf);

        Some(typed_arr.to_vec())
    }
}

#[cfg(target_arch = "wasm32")]
pub fn spawn_future<F>(fut: F)
where
    F: std::future::Future<Output = ()> + 'static,
{
    spawn_local(fut);
}

#[cfg(not(target_arch = "wasm32"))]
pub fn spawn_future<F>(fut: F)
where
    F: std::future::Future<Output = ()>  + 'static,
{
    pollster::block_on(fut);
}

#[derive(Debug)]
enum AppError {
    #[cfg(not(target_arch = "wasm32"))]
    PlatformError(winit::error::OsError),

    #[cfg(target_arch = "wasm32")]
    PlatformError(JsValue),

    EventLoopError(winit::error::EventLoopError),
    Text(String)


}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            AppError::PlatformError(err) => write!(f, "Platform error: {:?}", err),
            AppError::EventLoopError(err) => write!(f, "Winit event loop error: {}", err),
            AppError::Text(t) => write!(f, "Text error message: {}", t),
        }
    }
}

impl std::error::Error for AppError {}

#[cfg(target_arch = "wasm32")]
impl From<AppError> for JsValue {
    fn from(error: AppError) -> JsValue {
        use wasm_bindgen::JsError;
        JsError::new(&error.to_string()).into()
    }
}

impl From<winit::error::EventLoopError> for AppError {
    fn from(value: winit::error::EventLoopError) -> Self {
        Self::EventLoopError(value)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl From<winit::error::OsError> for AppError {
    fn from(value: winit::error::OsError) -> Self {
        Self::PlatformError(value)
    }
}

#[cfg(target_arch = "wasm32")]
impl From<JsValue> for AppError {
    fn from(value: JsValue) -> Self {
        Self::PlatformError(value)
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn run_js() -> Result<(), JsValue> {
    console_log::init().expect("could not initialize logger");
    match run().await {
        Ok(_) => Ok(()),
        Err(e) => Err(e.into()),
    }
}

fn update_debug_mode(key_code: KeyCode, numeral: u32, input: &InputState, frame_uniforms: &mut FrameUniforms) {
    if input.keys.contains(&PhysicalKey::Code(key_code)) && frame_uniforms.debug_mode != numeral {
        frame_uniforms.debug_mode = numeral;
        frame_uniforms.reject_hist = 1;
    }
}


async fn run() -> Result<(), AppError> {
    let event_loop = EventLoop::new()?;

    // default size

    #[cfg(target_arch = "wasm32")]
    let window = new_window_in_canvas(&event_loop, "canvas")?;

    #[cfg(not(target_arch = "wasm32"))]
    let window = new_window(&event_loop, [512, 512])?;

    let mut gpu = match Gpu::new(&window).await {
        Some(gpu) => gpu,
        None => return Err(AppError::Text("Failed to create GPU".to_owned()))
    };

    let ctx = Arc::new(Mutex::new(Context::init(&gpu).await));

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
    let mut frames_in_second: u32 = 0;
    let mut last_cursor_pos = PhysicalPosition::new(0.0, 0.0);
    let scale_factor = window.scale_factor();

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
            Event::WindowEvent { window_id: _, event } => {
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
                        let this_frame = Instant::now();
                        frames_in_second += 1;
                        gpu.window.request_redraw();
                        let dt = (this_frame - last_frame).as_secs_f32();

                        

                        if let Ok(mut ctx_guard) = ctx.try_lock() {
                            ctx_guard.scene.cameras[0].update(&mut input, dt);
                            
                            if ctx_guard.should_reupload {
                                ctx_guard.upload_scene(&gpu);
                            }

                            // not the most elegant code in the world
                            // TODO: add a digits array to the InputState struct, if it would
                            // do anything other than move this logic into the key press event
                            update_debug_mode(KeyCode::Digit0, 0, &input, &mut ctx_guard.frame_uniforms);
                            update_debug_mode(KeyCode::Digit1, 1, &input, &mut ctx_guard.frame_uniforms);
                            update_debug_mode(KeyCode::Digit2, 2, &input, &mut ctx_guard.frame_uniforms);
                            update_debug_mode(KeyCode::Digit3, 3, &input, &mut ctx_guard.frame_uniforms);
                            update_debug_mode(KeyCode::Digit4, 4, &input, &mut ctx_guard.frame_uniforms);
                            update_debug_mode(KeyCode::Digit5, 5, &input, &mut ctx_guard.frame_uniforms);
                            update_debug_mode(KeyCode::Digit6, 6, &input, &mut ctx_guard.frame_uniforms);
                            update_debug_mode(KeyCode::Digit7, 7, &input, &mut ctx_guard.frame_uniforms);
                            update_debug_mode(KeyCode::Digit8, 8, &input, &mut ctx_guard.frame_uniforms);
                            update_debug_mode(KeyCode::Digit9, 9, &input, &mut ctx_guard.frame_uniforms);

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
                    WindowEvent::CursorMoved { device_id: _, position } => if !input.rmb {last_cursor_pos = position},
                    WindowEvent::MouseInput { device_id: _, state, button } => {
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
                            gpu.window.set_cursor_grab(CursorGrabMode::Locked)
                                .or_else(|_e| gpu.window.set_cursor_grab(CursorGrabMode::Confined))
                                .or_else(|_e| gpu.window.set_cursor_grab(CursorGrabMode::None))
                                .expect("Failed to set any cursor grab modes");
                        } else {
                            // ignored because it is non-essential
                            let _ = gpu.window.set_cursor_position(last_cursor_pos);
                            gpu.window.set_cursor_visible(true);
                            match gpu.window.set_cursor_grab(CursorGrabMode::None) {
                                Ok(_) => (),
                                Err(e) => panic!("Failed to let go of cursor: {e}"),
                            }

                        }

                    }
                    WindowEvent::MouseWheel { device_id: _, delta, phase: _ } => {
                        // hack: I have no idea how to keep a consistent sensitivity between these
                        //       two units. This works well enough for the devices I tested it on
                        match delta {
                            winit::event::MouseScrollDelta::LineDelta(_, y) => input.scroll += y as f64 / 2.0,
                            winit::event::MouseScrollDelta::PixelDelta(physical_position) => input.scroll += physical_position.y / 128.0 / scale_factor,
                        }
                    },
                    WindowEvent::DroppedFile(path) => {
                        if let Some(path_string) = path.to_str() {
                            let path_string = path_string.to_string();

                            // TODO: accept HDR files
                            let is_mesh = match path.extension() {
                                None => true,
                                Some(os_str) => match os_str.to_str() {
                                    None => false,
                                    Some("gltf") => true,
                                    Some("GLTF") => true,
                                    Some("glb") => true,
                                    Some("GLB") => true,
                                    _ => false,
                                }

                            };
                            let ctx_clone = Arc::clone(&ctx);
                            spawn_future(async move {
                                if let Ok(mut ctx_guard) = ctx_clone.lock() {
                                    if is_mesh {
                                        ctx_guard.try_change_scene(path_string.as_str(), DEFAULT_ENV_PATH).await;
                                    }
                                    
                                }
                            });

                            // Im not sure why, but the window sometimes needs to be manually redrawn here
                            gpu.window.request_redraw();
                        }
                        
                    },
                    WindowEvent::CloseRequested => target.exit(),
                    WindowEvent::KeyboardInput { device_id: _, event, is_synthetic: _ } => {
                        match event.physical_key {
                            PhysicalKey::Code(code) => {
                                if event.state.is_pressed() {
                                    input.keys.insert(PhysicalKey::Code(code));
                                    if code == KeyCode::KeyO {
                                        let ctx_clone = Arc::clone(&ctx);
                                        spawn_future(async move {
                                            if let Ok(mut ctx_guard) = ctx_clone.lock() {
                                                if let Some(file) = rfd::AsyncFileDialog::new().set_title("Pick a gltf (or glb) file to render").pick_file().await {
                                                    ctx_guard.try_change_scene_bytes(&file.read().await, DEFAULT_ENV_PATH).await
                                                }
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

    })?;
    Ok(())
}

pub fn main() {
        #[cfg(not(target_arch = "wasm32"))]
        {
            env_logger::init();
            pollster::block_on(run())
        };
        
}
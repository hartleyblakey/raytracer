use std::{borrow::Cow, sync::{Arc, Mutex}};

#[cfg(not(target_arch = "wasm32"))]
use hb_gpu::winit;
use pollster::FutureExt;

#[cfg(target_arch = "wasm32")]
use js_sys::ArrayBuffer;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{JsCast, JsError};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::{spawn_local, JsFuture};


#[cfg(target_arch = "wasm32")]
use web_sys::Response;


use winit::{
    dpi::PhysicalPosition, event::{DeviceEvent, MouseButton, WindowEvent}, event_loop::EventLoop, keyboard::{KeyCode, PhysicalKey}, window::CursorGrabMode
};

use hb_gpu::{fetch_bytes, glam::{self, Vec4}, new_window, prelude::*, wgpu, winit::{application::ApplicationHandler, window::Window}};

use glam::uvec2;
use web_time::{Instant, SystemTime};

mod input;
use input::*;

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

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuRayState {
    origin_max: Vec4,

    direction_min: Vec4,
   
    throughput_flags: Vec4,

    pixel_medium_depth_pad: Vec4,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuWavefrontQueueParams {

}


const SHADER_PATH: &'static str = "./src/shader.wgsl";
const DEFAULT_MODEL_PATH: &'static str = "./resources/simple2.glb";
const DEFAULT_ENV_PATH: &'static str = "./resources/trail.hdr";

struct Context {
    screen_pipeline:            Option<wgpu::RenderPipeline>,
    screen_pipeline_layout:     wgpu::PipelineLayout,
    raytrace_pipeline:          Option<wgpu::ComputePipeline>,
    raytrace_pipeline_layout:   wgpu::PipelineLayout,

    raygen_pipeline:            Option<wgpu::ComputePipeline>,

    megakernel_shader:          ShaderHandle,
    raygen_shader:              ShaderHandle,
    screen_shader:              ShaderHandle,

    triangles_ssbo:             Buffer,
    bvh_ssbo:                   Buffer,
    screen_ssbo:                Buffer,
    triangles_ext_ssbo:         Buffer,
    texture_data_ssbo:          Buffer,
    primitive_data_ssbo:        Buffer,
    mesh_light_ssbo:            Buffer,
    media_ssbo:                 Buffer,

    ray_queue_ssbo:             Buffer,

    env_map_texture:            Texture,
    env_map_col_cdf:            Texture,
    env_map_pdf:                Texture,
    env_map_rows_cdf:           Buffer,

    rt_data_binding:            BindGroup,

    frame_uniforms_binding:     BindGroup,
    frame_uniforms_buffer:      Buffer,
    frame_uniforms:             FrameUniforms,

    resources:                  ResourceManager,

    scene:                      RenderScene,

    should_reupload:            bool,
}

impl Context {

    fn update_rt_binding(&mut self, gpu: &Gpu) {
        self.rt_data_binding = gpu.new_bind_group()
            .with_buffer(&self.triangles_ssbo.view_all(),        wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&self.triangles_ext_ssbo.view_all(),    wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&self.bvh_ssbo.view_all(),              wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&self.screen_ssbo.view_all(),           wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&self.texture_data_ssbo.view_all(),     wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&self.primitive_data_ssbo.view_all(),   wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&self.env_map_rows_cdf.view_all(),      wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&self.mesh_light_ssbo.view_all(),       wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&self.media_ssbo.view_all(),            wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&self.ray_queue_ssbo.view_all(),        wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_texture(&self.env_map_texture,                 wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_texture(&self.env_map_col_cdf,                 wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_texture(&self.env_map_pdf,                     wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .finish(&mut self.resources);
    }

    fn update_resolution(&mut self, gpu: &Gpu) {
        let res = [gpu.surface_config.width, gpu.surface_config.height];
        self.frame_uniforms.res = res;
        println!("x: {}, y: {}", res[0], res[1]);
        let n = res[0].max(1) as u64 * res[1].max(1) as u64;
        self.screen_ssbo = gpu.new_storage_buffer(n * 4 * 4);
        self.ray_queue_ssbo = gpu.new_storage_buffer(n * size_of::<GpuRayState>() as u64);
        self.update_rt_binding(gpu);
    }

    fn create_pipelines(&mut self, gpu: &Gpu) {
        
        let screen_module = self.resources.get_shader(self.screen_shader).module.as_ref().unwrap();
        
        let screen_pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&self.screen_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &screen_module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &screen_module,
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

        let megakernel_module = self.resources.get_shader(self.megakernel_shader).module.as_ref().unwrap();

        let raytrace_pipeline = gpu.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("raytrace compute pipeline"),
                module: &megakernel_module,
                layout: Some(&self.raytrace_pipeline_layout),
                entry_point: Some("cs_main"),
                compilation_options: Default::default(),
                cache: None,
            }
        );

        let raygen_module = self.resources.get_shader(self.raygen_shader).module.as_ref().unwrap();

        let raygen_pipeline = gpu.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("ray generation compute pipeline"),
                module: &raygen_module,
                layout: Some(&self.raytrace_pipeline_layout),
                entry_point: Some("cs_main"),
                compilation_options: Default::default(),
                cache: None,
            }
        );

        self.screen_pipeline = Some(screen_pipeline);
        self.raytrace_pipeline = Some(raytrace_pipeline);
        self.raygen_pipeline = Some(raygen_pipeline);
    }


    fn check_recompile_shader(&mut self, gpu: &Gpu) -> bool {
    #[cfg(not(target_arch = "wasm32"))] 
    {
        let changed = self.resources.recompile_shaders(gpu);

        if changed.is_empty() {
            return false;
        }

        self.create_pipelines(gpu);
        
        return true;
        
    }

    #[cfg(target_arch = "wasm32")]
        false
    }

    /// Update the wgpu texture to match the data in self.scene
    fn update_env_map_texture(&mut self, gpu: &Gpu) {
        self.env_map_texture = gpu.new_texture(uvec2(2 * self.scene.env_map.height as u32, self.scene.env_map.height as u32), wgpu::TextureFormat::Rgba32Float, false);

        self.env_map_rows_cdf = gpu.new_storage_buffer((self.scene.env_map.height * self.scene.env_map.width * size_of::<f32>()) as u64);

        self.env_map_col_cdf = gpu.new_texture(uvec2(2 * self.scene.env_map.height as u32, self.scene.env_map.height as u32), wgpu::TextureFormat::R32Float, false);
        self.env_map_pdf = gpu.new_texture(uvec2(2 * self.scene.env_map.height as u32, self.scene.env_map.height as u32), wgpu::TextureFormat::R32Float, false);
        self.update_rt_binding(gpu);
    }

    async fn init(gpu: &Gpu) -> Context {
        let scene = RenderScene::from_path(std::path::Path::new(DEFAULT_MODEL_PATH), std::path::Path::new(DEFAULT_ENV_PATH) ).await.unwrap();

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

        // just make everything 1mb for simplicity
        let initial_buffer_size_mb = 1;

        let triangles_ssbo =        gpu.new_storage_buffer(initial_buffer_size_mb * 1024 * 1024);
        let bvh_ssbo =              gpu.new_storage_buffer(initial_buffer_size_mb * 1024 * 1024);
        let triangles_ext_ssbo =    gpu.new_storage_buffer(initial_buffer_size_mb * 1024 * 1024);
        let texture_data_ssbo =     gpu.new_storage_buffer(initial_buffer_size_mb * 1024 * 1024);
        let primitive_data_ssbo =   gpu.new_storage_buffer(initial_buffer_size_mb * 1024 * 1024);
        let screen_ssbo =           gpu.new_storage_buffer(u_frame_0.res[0] as u64 * u_frame_0.res[1] as u64 * 4 * 4);
        let mesh_light_ssbo =       gpu.new_storage_buffer(initial_buffer_size_mb * 1024 * 1024);


        let env_map_rows_cdf = gpu.new_storage_buffer((scene.env_map.height * scene.env_map.width * size_of::<f32>()) as u64);

        let media_ssbo =       gpu.new_storage_buffer(initial_buffer_size_mb * 1024 * 1024);

        let ray_queue_ssbo =   gpu.new_storage_buffer(initial_buffer_size_mb * 1024 * 1024);

        let env_map_texture = gpu.new_texture(uvec2(2 * scene.env_map.height as u32, scene.env_map.height as u32), wgpu::TextureFormat::Rgba32Float, false);
        let env_map_col_cdf = gpu.new_texture(uvec2(2 * scene.env_map.height as u32, scene.env_map.height as u32), wgpu::TextureFormat::R32Float, false);
        let env_map_pdf = gpu.new_texture(uvec2(2 * scene.env_map.height as u32, scene.env_map.height as u32), wgpu::TextureFormat::R32Float, false);

        let rt_data_bg = gpu.new_bind_group()
            .with_buffer(&triangles_ssbo.view_all(),        wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&triangles_ext_ssbo.view_all(),    wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&bvh_ssbo.view_all(),              wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&screen_ssbo.view_all(),           wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&texture_data_ssbo.view_all(),     wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&primitive_data_ssbo.view_all(),   wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&env_map_rows_cdf.view_all(),      wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&mesh_light_ssbo.view_all(),       wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&media_ssbo.view_all(),            wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&ray_queue_ssbo.view_all(),        wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_texture(&env_map_texture,                 wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_texture(&env_map_col_cdf,                 wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_texture(&env_map_pdf,                     wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .finish(&mut resources);

        
        let Some(megakernel) = resources.new_shader(std::path::Path::new(SHADER_PATH), gpu) else {
            panic!("Unable to add shader at {SHADER_PATH}");
        };

        let Some(screen) = resources.new_shader(std::path::Path::new("src/screen.wgsl"), gpu) else {
            panic!("Unable to add screen shader");
        };

        let Some(raygen_shader) = resources.new_shader(std::path::Path::new("src/raygen.wgsl"), gpu) else {
            panic!("Unable to add raygen shader");
        };
        

        let screen_pipeline_layout = gpu.new_pipeline_layout(
            &resources, &[&u_frame, &rt_data_bg]
        );

        let raytrace_pipeline_layout = gpu.new_pipeline_layout(
            &resources, &[&u_frame, &rt_data_bg]
        );

        let should_reupload = true;

        let mut ctx = Context {
            screen_pipeline: None,
            screen_pipeline_layout,
            megakernel_shader: megakernel,
            screen_shader: screen,

            frame_uniforms: u_frame_0,
            frame_uniforms_buffer: u_frame_buffer,
            frame_uniforms_binding: u_frame,
            
            raytrace_pipeline: None,
            raytrace_pipeline_layout,

            raygen_pipeline: None,
            raygen_shader,
            
            screen_ssbo,
            bvh_ssbo,
            triangles_ssbo,
            triangles_ext_ssbo,
            texture_data_ssbo,
            primitive_data_ssbo,
            mesh_light_ssbo,
            media_ssbo,

            ray_queue_ssbo,

            env_map_col_cdf,
            env_map_texture,
            env_map_pdf,
            env_map_rows_cdf,

            rt_data_binding: rt_data_bg,

            resources,
            scene,

            should_reupload,
        };
        ctx.create_pipelines(gpu);
        ctx
    }

    async fn try_change_scene(&mut self, mesh_path: &std::path::Path, env_map_path: &std::path::Path) {
        println!("Attempting to change scene");
        if let Some(scene) = RenderScene::from_path(mesh_path, env_map_path).await {
            self.scene = scene;
            self.frame_uniforms.scene = self.scene.to_gpu();
            self.frame_uniforms.reject_hist = 1;
            self.frame_uniforms.node_count = self.scene.bvh_node_data.len() as u32;
            self.frame_uniforms.prim_count = self.scene.primitives.len() as u32;

            // self.scene.focus_camera(0);
            // self.scene.cameras[0].update(&mut InputState::default(), 1.0 / 60.0);

            self.should_reupload = true;
        } else {
            println!("Scene change failed! RenderScene::from_path returned None");
        }
    }

    async fn try_add_file(&mut self, path: &std::path::Path) {
        let is_mesh = match path.extension() {
            None => false,
            Some(os_str) => match os_str.to_str() {
                None => false,
                Some("gltf") => true,
                Some("GLTF") => true,
                Some("glb") => true,
                Some("GLB") => true,
                _ => false,
            }

        };

        // TODO: Test other HDR image formats
        let is_env = match path.extension() {
            None => false,
            Some(os_str) => match os_str.to_str() {
                None => false,
                Some("hdr") => true,
                _ => false,
            }
        };

        if is_env {
            println!("Trying to add env map");
            self.scene.set_equirectangular_env_map(path).await;
            self.frame_uniforms.reject_hist = 1;
            self.should_reupload = true;
        } else if is_mesh {
            let old_env = self.scene.env_map_path.clone();
            self.try_change_scene(path, &old_env).await
        }
    }
    
    #[cfg(target_arch = "wasm32")] 
    async fn try_change_scene_bytes(&mut self, mesh_bytes: &[u8], env_map_path: &std::path::Path) {
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

        // tack the TLAS onto the back of the BLAS data
        let mut combined_bvh = self.scene.bvh_node_data.clone();
        combined_bvh.append(&mut self.scene.tlas_node_data.clone());
        
        self.bvh_ssbo =             gpu.new_storage_buffer(combined_bvh.len().max(1) as u64 * size_of::<BvhNode>() as u64);
        self.triangles_ssbo =       gpu.new_storage_buffer(self.scene.tris.len().max(1) as u64 * size_of::<Tri>() as u64);
        self.triangles_ext_ssbo =   gpu.new_storage_buffer(self.scene.tri_exts.len().max(1) as u64 * size_of::<GpuTriExt>() as u64);
        self.texture_data_ssbo =    gpu.new_storage_buffer(self.scene.texture_data.len().max(1) as u64 * size_of::<u32>() as u64);
        self.primitive_data_ssbo =  gpu.new_storage_buffer(self.scene.primitives.len().max(1) as u64 * size_of::<GpuPrimitive>() as u64);
        self.env_map_rows_cdf =     gpu.new_storage_buffer(self.scene.env_map.cdf_rows.len().max(1) as u64 * size_of::<f32>() as u64);
        self.mesh_light_ssbo =      gpu.new_storage_buffer(self.scene.mesh_lights.len().max(1) as u64 * size_of::<MeshLight>() as u64);
        self.media_ssbo =           gpu.new_storage_buffer(self.scene.media.len().max(1) as u64 * size_of::<GpuVolume>() as u64);
        self.update_rt_binding(gpu);

        gpu.queue.write_buffer(&self.bvh_ssbo,               0, bytemuck::cast_slice(combined_bvh.as_slice()));
        gpu.queue.write_buffer(&self.triangles_ssbo,         0, bytemuck::cast_slice(self.scene.tris.as_slice()));
        gpu.queue.write_buffer(&self.triangles_ext_ssbo,     0, bytemuck::cast_slice(self.scene.tri_exts.as_slice()));
        gpu.queue.write_buffer(&self.texture_data_ssbo,      0, bytemuck::cast_slice(self.scene.texture_data.as_slice()));
        gpu.queue.write_buffer(&self.primitive_data_ssbo,    0, bytemuck::cast_slice(self.scene.primitives.as_slice()));
        gpu.queue.write_buffer(&self.env_map_rows_cdf,       0, bytemuck::cast_slice(self.scene.env_map.cdf_rows.as_slice()));
        gpu.queue.write_buffer(&self.mesh_light_ssbo,        0, bytemuck::cast_slice(self.scene.mesh_lights.as_slice()));
        gpu.queue.write_buffer(&self.media_ssbo,             0, bytemuck::cast_slice(self.scene.media.as_slice()));
        
        gpu.queue.write_texture(
            self.env_map_texture.as_image_copy(), 
            bytemuck::cast_slice(&self.scene.env_map.data.as_slice()), 
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

        gpu.queue.write_texture(
            self.env_map_col_cdf.as_image_copy(), 
            bytemuck::cast_slice(&self.scene.env_map.cdf_col.as_slice()), 
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.env_map_col_cdf.width() * 4),
                rows_per_image: None,
            }, 
            wgpu::Extent3d{
                width: self.env_map_col_cdf.width(),
                height: self.env_map_col_cdf.height(),
                depth_or_array_layers: 1,
            },
        );

        gpu.queue.write_texture(
            self.env_map_pdf.as_image_copy(), 
            bytemuck::cast_slice(&self.scene.env_map.pdf.as_slice()), 
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.env_map_pdf.width() * 4),
                rows_per_image: None,
            }, 
            wgpu::Extent3d{
                width: self.env_map_pdf.width(),
                height: self.env_map_pdf.height(),
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
    {   // ray gen
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cpass.set_pipeline(&ctx.raygen_pipeline.as_ref().unwrap());
        cpass.set_bind_group(0, &ctx.frame_uniforms_binding.raw, &[]);
        cpass.set_bind_group(1, &ctx.rt_data_binding.raw, &[]);
        cpass.dispatch_workgroups(
            (ctx.frame_uniforms.res[0] + workgroup_size[0] - 1) / workgroup_size[0],
            (ctx.frame_uniforms.res[1] + workgroup_size[1] - 1) / workgroup_size[1], 
            1
        );
    }
    
    {   // megakernel
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cpass.set_pipeline(&ctx.raytrace_pipeline.as_ref().unwrap());
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
        rpass.set_pipeline(&ctx.screen_pipeline.as_ref().unwrap());
        rpass.set_bind_group(0, Some(&ctx.frame_uniforms_binding.raw), &[]);
        rpass.set_bind_group(1, Some(&ctx.rt_data_binding.raw), &[]);
        rpass.draw(0..3, 0..1);
    }

    ctx.frame_uniforms.reject_hist = 0;
    
    gpu.queue.submit(Some(encoder.finish()));
    surface_texture.present();
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


}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            AppError::PlatformError(err) => write!(f, "Platform error: {:?}", err),
            AppError::EventLoopError(err) => write!(f, "Winit event loop error: {}", err),
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

struct App {
    gpu: Gpu,
    ctx: Context,
    frames_in_second: u32,
    last_frame: Instant,
    last_second: Instant,
    input: InputState,
    last_cursor_pos: PhysicalPosition<f64>,
}

#[derive(Clone, Default)]
struct AppShell {
    app: Arc<Mutex<Option<App>>>
}

impl AppShell {
    async fn init(self, window: Arc<Window>) {
        let Some(gpu) = Gpu::new(window).await else {
            panic!("Failed to create gpu");
        };
        let ctx = Context::init(&gpu).await;
        let Ok(mut lock) = self.app.lock() else {
            panic!("Failed to acquire app lock");
        };
        let frames_in_second = 0;
        let last_frame = Instant::now();
        let last_second = Instant::now();
        let input = InputState::default();
        let last_position = PhysicalPosition::<f64>::default();
        *lock = Some(App { gpu, ctx, frames_in_second, last_frame, last_second, input, last_cursor_pos: last_position});
    }

    async fn try_add_file(&self, path: &std::path::Path) {
        let Ok(mut app) = self.app.lock() else {
            return;
        };

        let Some(app) = &mut *app else {
            return;
        };

        app.ctx.try_add_file(path).await;
    }

    async fn open_file(&self) {
        if let Some(file) = rfd::AsyncFileDialog::new().set_title("Pick a gltf (or glb) file to render, or a .hdr if on native").pick_file().await {
            let Ok(mut app) = self.app.lock() else {
                return;
            };

            let Some(app) = &mut *app else {
                return;
            };

            #[cfg(target_arch = "wasm32")]
            {
                let old_env = app.ctx.scene.env_map_path.clone();
                app.ctx.try_change_scene_bytes(file.read().await.as_slice(), &old_env).await;
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                app.ctx.try_add_file(&file.path()).await
            };
        }
    }

}

impl ApplicationHandler for AppShell {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = match new_window(event_loop, [512, 512]) {
            Ok(window) => window,
            Err(e) => panic!("{:?}", e),
        };
        let window = Arc::new(window);
        spawn_future(self.clone().init(window));
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::Resized(new_size) => {
                let Ok(mut app_guard) = self.app.lock() else { return; };
                let Some(app) = &mut *app_guard else { return; };
                let (gpu, ctx, _) = (&mut app.gpu, &mut app.ctx, &mut app.input);
                
                // Reconfigure the surface with the new size
                gpu.surface_config.width  = new_size.width.clamp(1, 4096);
                gpu.surface_config.height = new_size.height.clamp(1, 4096);
                
                gpu.surface.configure(&gpu.device, &gpu.surface_config);

                ctx.update_resolution(&gpu);
                
                // On macos the window needs to be redrawn manually after resizing
                gpu.window.request_redraw();
            }

            WindowEvent::RedrawRequested => {
                let Ok(mut app_guard) = self.app.lock() else { return; };
                let Some(app) = &mut *app_guard else { return; };
                let (gpu, ctx, input) = (&mut app.gpu, &mut app.ctx, &mut app.input);

                let this_frame = Instant::now();
                app.frames_in_second += 1;
                gpu.window.request_redraw();
                let dt = (this_frame - app.last_frame).as_secs_f32();

                ctx.scene.cameras[0].update(input, dt);
                    
                if ctx.should_reupload {
                    ctx.update_env_map_texture(&gpu);
                    ctx.upload_scene(&gpu);
                }

                // not the most elegant code in the world
                // TODO: add a digits array to the InputState struct, if it would
                // do anything other than move this logic into the key press event
                update_debug_mode(KeyCode::Digit0, 0, &input, &mut ctx.frame_uniforms);
                update_debug_mode(KeyCode::Digit1, 1, &input, &mut ctx.frame_uniforms);
                update_debug_mode(KeyCode::Digit2, 2, &input, &mut ctx.frame_uniforms);
                update_debug_mode(KeyCode::Digit3, 3, &input, &mut ctx.frame_uniforms);
                update_debug_mode(KeyCode::Digit4, 4, &input, &mut ctx.frame_uniforms);
                update_debug_mode(KeyCode::Digit5, 5, &input, &mut ctx.frame_uniforms);
                update_debug_mode(KeyCode::Digit6, 6, &input, &mut ctx.frame_uniforms);
                update_debug_mode(KeyCode::Digit7, 7, &input, &mut ctx.frame_uniforms);
                update_debug_mode(KeyCode::Digit8, 8, &input, &mut ctx.frame_uniforms);
                update_debug_mode(KeyCode::Digit9, 9, &input, &mut ctx.frame_uniforms);

                frame(&gpu, ctx, dt);

                if this_frame.duration_since(app.last_second).as_secs_f32() >= 1.0 {
                    println!("fps: {}", app.frames_in_second);
                    app.frames_in_second = 0;
                    app.last_second = this_frame;
                }
                
                
                app.last_frame = this_frame;
            },
            WindowEvent::CursorMoved { device_id: _, position } => {
                let Ok(mut app_guard) = self.app.lock() else { return; };
                let Some(app) = &mut *app_guard else { return; };
                if !app.input.rmb {app.last_cursor_pos = position}
            },
            WindowEvent::MouseInput { device_id: _, state, button } => {
                let Ok(mut app_guard) = self.app.lock() else { return; };
                let Some(app) = &mut *app_guard else { return; };
                let (gpu, ctx, input) = (&mut app.gpu, &mut app.ctx, &mut app.input);

                match button {
                    MouseButton::Left =>  {
                        input.lmb = state.is_pressed();
                        ctx.scene.focus_camera(0);
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
                    let _ = gpu.window.set_cursor_position(app.last_cursor_pos);
                    gpu.window.set_cursor_visible(true);
                    match gpu.window.set_cursor_grab(CursorGrabMode::None) {
                        Ok(_) => (),
                        Err(e) => panic!("Failed to let go of cursor: {e}"),
                    }

                }

            }
            WindowEvent::MouseWheel { device_id: _, delta, phase: _ } => {
                let Ok(mut app_guard) = self.app.lock() else { return; };
                let Some(app) = &mut *app_guard else { return; };
                let (gpu, _, input) = (&mut app.gpu, &mut app.ctx, &mut app.input);

                // hack: I have no idea how to keep a consistent sensitivity between these
                //       two units. This works well enough for the devices I tested it on
                match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => input.scroll += y as f64 / 2.0,
                    winit::event::MouseScrollDelta::PixelDelta(physical_position) => input.scroll += physical_position.y / 128.0 / gpu.window.scale_factor(),
                }
            },
            WindowEvent::DroppedFile(path) => {
                // preempt some errors while the failure path is convenient
                if path.to_str().is_some() {
                    let c = self.clone();
                    
                    spawn_future(async move {
                        c.try_add_file(&path).await
                    });

                    let Ok(mut app_guard) = self.app.lock() else { return; };
                    let Some(app) = &mut *app_guard else { return; };
                    let (gpu, _, _) = (&mut app.gpu, &mut app.ctx, &mut app.input);

                    // Im not sure why, but the window sometimes needs to be manually redrawn here
                    gpu.window.request_redraw();
                }
                
            },
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { device_id: _, event, is_synthetic: _ } => {
                match event.physical_key {
                    PhysicalKey::Code(code) => {
                        if event.state.is_pressed() {
                            if code == KeyCode::KeyO {
                                
                                let c = self.clone();
                                spawn_future(async move {
                                    c.open_file().await;
                                });
                            } else if code == KeyCode::KeyP {
                                #[cfg(not(target_arch = "wasm32"))]
                                {
                                    let Ok(mut app_guard) = self.app.lock() else { return; };
                                    let Some(app) = &mut *app_guard else { return; };

                                    let img = app.ctx.scene.trace_cpu_image(app.ctx.frame_uniforms.scene.camera);
                                    match img.save("screenshots/cpu.png") {
                                        Ok(_) => (),
                                        Err(e) => println!("Failed to save screenshot: \n{e}"),
                                    };
                                }
                            }
                            let Ok(mut app_guard) = self.app.lock() else { return; };
                            let Some(app) = &mut *app_guard else { return; };
                            app.input.keys.insert(PhysicalKey::Code(code));
                        } else {
                            let Ok(mut app_guard) = self.app.lock() else { return; };
                            let Some(app) = &mut *app_guard else { return; };
                            app.input.keys.remove(&PhysicalKey::Code(code));
                        }
                    }
                    _ => ()
                }
            },
            _ => ()
        };
    }

    fn device_event(
            &mut self,
            _event_loop: &winit::event_loop::ActiveEventLoop,
            _device_id: winit::event::DeviceId,
            event: DeviceEvent,
        ) {

        let Ok(mut app) = self.app.lock() else {
            return;
        };

        let Some(app) = &mut *app else {
            return;
        };

        match event {
            DeviceEvent::MouseMotion{ delta, } => {
                app.input.mouse_x += delta.0;
                app.input.mouse_y += delta.1;
            }
            _ => (),
        };


    }
}

async fn run() -> Result<(), AppError> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = AppShell::default();
    event_loop.run_app(&mut app)?;
    Ok(())
}

pub fn main() {
        #[cfg(not(target_arch = "wasm32"))]
        {
            env_logger::init();
            match pollster::block_on(run()) {
                Ok(_) => (),
                Err(e) => panic!("Error running app: {:?}", e)
            }
        };
}
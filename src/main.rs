use std::{borrow::Cow, collections::{HashMap, HashSet}, f32::consts::PI, mem::swap, str::FromStr};
use gltf::Gltf;
use js_sys::ArrayBuffer;
use rand::random;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::Response;
use winit::{
    dpi::PhysicalPosition, event::{DeviceEvent, ElementState, Event, MouseButton, WindowEvent}, event_loop::EventLoop, keyboard::{KeyCode, PhysicalKey}, window::CursorGrabMode
};

use glam::{uvec2, vec2, vec3, vec4, FloatExt, Mat3, Mat4, UVec2, Vec2, Vec3, Vec4, Vec4Swizzles};
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
    node_count: u32,
    _pad: [u32; 2]

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
            vertices: [vec4(p1.x, p1.y, p1.z, c.x), vec4(p2.x, p2.y, p2.z, c.y), vec4(p3.x, p3.y, p3.z, c.z)],
        }
    }

    fn dummy(c: Vec3, size: f32) -> Tri {
        
        Tri::new(
            c + vec3(random(), random(), random()) * size - size * 0.5,
            c + vec3(random(), random(), random()) * size - size * 0.5,
            c + vec3(random(), random(), random()) * size - size * 0.5,
        )
    }

    fn aabb(&self) -> Aabb {
        Aabb::point(self.vertices[0].xyz())
            .with(Aabb::point(self.vertices[1].xyz()))
            .with(Aabb::point(self.vertices[2].xyz()))
    }

    fn centroid(&self) -> Vec3 {
        // (self.vertices[0].xyz() + self.vertices[1].xyz() + self.vertices[2].xyz()) / 3.0
        vec3(self.vertices[0][3], self.vertices[1][3], self.vertices[2][3])
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Aabb {
    // alignment rules
    data: [f32; 6]
}

impl Aabb {
    fn new() -> Self {
        Self {
            data: [0.0; 6]
        }
    }

    fn invalid() -> Self {
        Self {
            data: [f32::MAX, f32::MAX, f32::MAX, f32::MIN, f32::MIN, f32::MIN]
        }
    }

    fn surface(&self) -> f32 {
        let size = self.max() - self.min();
        (size.x * size.y + size.y * size.z + size.z * size.x) * 2.0
    } 

    fn with(&self, other: Self) -> Self {
        Self {
            data:  [self.data[0].min(other.data[0]),
                    self.data[1].min(other.data[1]),
                    self.data[2].min(other.data[2]),
                    self.data[3].max(other.data[3]),
                    self.data[4].max(other.data[4]),
                    self.data[5].max(other.data[5])]
        }
    }

    fn expand(&mut self, other: Self) {
        self.data = self.with(other).data;
    }

    fn point(point: Vec3) -> Self {
        Self {
            data: [point.x - 0.00001, point.y - 0.00001, point.z - 0.00001, point.x + 0.00001, point.y + 0.00001, point.z + 0.00001]
        }
    }

    fn min(&self) -> Vec3 {
        vec3(self.data[0], self.data[1], self.data[2])
    }

    fn max(&self) -> Vec3 {
        vec3(self.data[3], self.data[4], self.data[5])
    }
}

// structure from https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BvhNode {
    aabb: Aabb,

    /// The index of the left child if count is 0. First triangle index otherwise
    first: u32,

    /// the number of triangles in the node
    count: u32,
}


impl BvhNode {
    fn new() -> BvhNode {
        BvhNode {
            first: 0,
            count: 0,
            aabb: Aabb::point(vec3(-100.0, -100.0, -100.0)),
        }
    }

    fn from_tris(first: u32, count: u32, indices: &Vec<u32>, tris: &Vec<Tri>) -> Self {
        let mut new = Self::new();
        new.first = first;
        new.count = count;
        new.update_aabb(indices, tris);
        new
    }

    fn update_aabb(&mut self, indices: &Vec<u32>, tris: &Vec<Tri> ) {
        if self.count != 0 {
            self.aabb = tris[indices[self.first as usize] as usize].aabb();
            for i in self.first..self.first + self.count {
                self.aabb.expand(tris[indices[i as usize] as usize].aabb());
            }
        }
    }
}
struct Bvh<'a> {
    nodes: Vec<BvhNode>,
    tris: &'a Vec<Tri>,
    tri_exts: &'a Vec<TriExt>,
    indices: Vec<u32>,
}

impl<'a> Bvh<'a> {
    fn new(triangles: &'a Vec<Tri>, exts: &'a Vec<TriExt>) -> Self {
        Self {
            nodes: Vec::new(),
            tris: triangles,
            tri_exts: exts,
            indices: (0..triangles.len() as u32).collect(),
        }
    }

    fn build(&mut self) {
        self.nodes.push(BvhNode::from_tris(0, self.tris.len() as u32, &self.indices, &self.tris));
        self.subdivide(self.nodes.len() - 1, -1);
    }

    /// remove the layer of indirection used to build the BVH
    fn flat_triangles(&self) -> (Vec<Tri>, Vec<TriExt>) {
        let mut tris = self.tris.clone();
        let mut exts = self.tri_exts.clone();
        for i in 0..tris.len() {
            tris[i] = self.tris[self.indices[i] as usize];
            exts[i] = self.tri_exts[self.indices[i] as usize];
        }
        (tris, exts)
    }

    fn evaluate_split(&self, node: &BvhNode, axis: usize, split: f32) -> f32 {
        let mut left_aabb = Aabb::invalid();
        let mut right_aabb = Aabb::invalid();
        let mut left_count = 0.0;
        let mut right_count = 0.0;

        for i in (node.first)..(node.first + node.count) {
            let tri = self.tris[self.indices[i as usize] as usize];
            if tri.centroid()[axis] < split {
                left_count += 1.0;
                left_aabb.expand(tri.aabb());
            } else {
                right_count += 1.0;
                right_aabb.expand(tri.aabb());
            }

        }

        let cost = left_count * left_aabb.surface() + right_count * right_aabb.surface();

        if cost > 0.0 {
            cost
        } else {
            f32::MAX
        }
    }

    fn find_best_split(&self, node: &BvhNode) -> (usize, f32) {
        let mut best_axis = 0;
        let mut best_split = 0.0;
        let mut best_cost = f32::MAX;

        for axis in 0..3  as usize {
            for idx in (node.first)..(node.first + node.count) {
                let tri = self.tris[self.indices[idx as usize] as usize];
                let split = tri.centroid()[axis as usize];
                let cost = self.evaluate_split(node, axis, split);
                if cost < best_cost {
                    best_axis = axis;
                    best_cost = cost;
                    best_split = split;
                }
            }
        }

        (best_axis, best_split)
    }

    fn find_split_approx(&self, node: &BvhNode, count: usize) -> (usize, f32) {
        let mut best_axis = 0;
        let mut best_split = 0.0;
        let mut best_cost = f32::MAX;

        for axis in 0..3  as usize {
            for i in 0..count {
                let split = node.aabb.min()[axis] + ((i as f32 + 0.5) / count as f32) * (node.aabb.max()[axis]-node.aabb.min()[axis]);
                let cost = self.evaluate_split(node, axis, split);
                if cost < best_cost {
                    best_axis = axis;
                    best_cost = cost;
                    best_split = split;
                }
            }
        }

        (best_axis, best_split)
    }


    // algorithm from https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
    fn subdivide(&mut self, node_idx: usize, skip: i32) {
        let node = self.nodes[node_idx];
        // println!("first: {}, count: {}, bounds: {} to {}, {} to {}, {} to {}", node.first, node.count, node.aabb.min().x, node.aabb.max().x, node.aabb.min().y, node.aabb.max().y, node.aabb.min().z, node.aabb.max().z);
        if node.count <= 2 {
            return;
        }

        // let mut best_axis = 0;
        // let mut best_size = -1.0;
        // let size = node.aabb.max() - node.aabb.min();
        // for axis in 0..3 {
        //     if skip >= 0 && skip as usize == axis {
        //         continue;
        //     }
        //     if size[axis] >= best_size {
        //         best_axis = axis;
        //         best_size = size[axis];
        //     }
        // }
        // // best_axis = (self.nodes.len() + 1) % 3;
        // let split_pos = node.aabb.min()[best_axis] + size[best_axis] * 0.5;

        let (axis, split) = self.find_split_approx(&node, 16);

        // println!("split: {split_pos}, axis: {best_axis}");
        let mut i = node.first as usize;
        let mut j = (node.first + node.count - 1) as usize;
        while i <= j {
            let first_idx = self.indices[i] as usize;
            
            if self.tris[first_idx].centroid()[axis] < split {
                i += 1;
            } else {
                // swap
                let last_i = self.indices[i];
                self.indices[i] = self.indices[j];
                self.indices[j] = last_i;

                if j == 0 {
                    break;
                }

                j -= 1;
                
            }
        };
        // println!("i: {i}, j: {j}");

        let mut left = BvhNode::new();
        left.first = node.first;
        left.count = i  as u32 - node.first;
        left.update_aabb(&self.indices, &self.tris);

        // dont subdivide empty nodes
        if left.count == 0 || left.count == node.count {
            // if skip < 0  && node.count > 2 {
            //     self.subdivide(node_idx, best_axis as i32);
            // }
            return;
        }
        

        let mut right = BvhNode::new();
        right.first = i as u32;
        right.count = node.count - left.count;
        right.update_aabb(&self.indices, &self.tris);


        // if left.count < right.count / 8 || left.count > right.count * 8 {
        //     if skip < 0 {
        //         self.subdivide(node_idx, best_axis as i32);
        //     }
        //     return;
        // }

        // we no longer hold any triangles
        let children_idx = self.nodes.len();
        self.nodes[node_idx].count = 0;
        self.nodes[node_idx].first = children_idx as u32;

        self.nodes.push(left);
        self.nodes.push(right);

        self.subdivide(children_idx, -1);
        self.subdivide(children_idx + 1, -1);
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

    shader_module:          wgpu::ShaderModule,

    triangles_ssbo:         Buffer,
    bvh_ssbo:               Buffer,
    screen_ssbo:            Buffer,
    triangles_ext_ssbo:     Buffer,
    rt_data_binding:        BindGroup,

    frame_uniforms_binding: BindGroup,
    frame_uniforms_buffer:  Buffer,
    frame_uniforms:         FrameUniforms,

    resources:              ResourceManager,

    scene:                  FlatScene,
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
    fn rotate_y(&mut self, rad: f32) {
        self.apply(&Mat4::from_rotation_y(rad));
    }
    fn translate(&mut self, delta: Vec3) {
        self.apply(&Mat4::from_translation(delta));
    }
    fn scale(&mut self, scale: Vec3) {
        self.apply(&Mat4::from_scale(scale));
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

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuTexcoord {
    offset: u32,
    size: u32,
    pos: Vec2,
}

impl GpuTexcoord {
    fn new(offset: u32, size: UVec2, pos: Vec2) -> Self {
        let size = (size.x << 16) | size.y;
        Self {
            offset,
            size,
            pos
        }
    }

    fn zero() -> Self {
        Self {
            offset: 0,
            size: 0,
            pos: vec2(0.0, 0.0)
        }
    }


    fn size(&self) -> UVec2 {
        uvec2(self.size & 0xFF, self.size >> 16)
    }

    fn index(&self) -> u32 {
        let size = self.size();
        let upos = uvec2((self.pos.x * size.x as f32) as u32, (self.pos.y * size.y as f32) as u32);
        self.offset + upos.x + upos.y * size.x
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuVertexExt {
    tex0: GpuTexcoord,
    normal: Vec2,
    color: u32,
    _pad: f32
}


#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct TriExt {
    vertices: [GpuVertexExt; 3]
}


#[derive(Default)]
struct FlatScene {
    triangles: Vec<Tri>,
    triangles_ext: Vec<TriExt>,
    cameras:   Vec<Camera>,
    point_lights: Vec<PointLight>,
    directional_lights: Vec<DirectionalLight>,
}

impl FlatScene {
    fn add_gltf_bytes(&mut self, transform: &Mat4, bytes: &[u8]) {
        let (document, buffers, _) = gltf::import_slice(bytes).unwrap();
        let mut ms = MatrixStack::new();
        ms.push();
        ms.apply(&transform);
        for scene in document.scenes(){
            for node in scene.nodes() {
                self.add_gltf_node(&buffers, node, &mut ms);
            }
        }
    }   

    fn from_gltf(mat: &Mat4) -> Mat4 {
        Mat4::from_euler(glam::EulerRot::XZY, 0.5 * PI, 0.5 * PI, 0.0).mul_mat4(mat)
    }

    fn from_gltf_vec3(v: Vec3) -> Vec3 {
        vec3(v.z, v.x, v.y)
    }

    fn add_gltf(&mut self, path: &str) {
        let (document, buffers, _) = gltf::import(path).unwrap();
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
                    // TODO: figure out what this lambda that I copied does
                    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
    
                    let positions: Vec<Vec3> = reader.read_positions().unwrap()
                    .map( |p| 
                        Self::from_gltf_vec3(ms.top().transform_point3(Vec3::from_slice(&p)))
                    ).collect();
                    
                    
                    let mut colors = reader.read_colors(0);
                    let mut colors = if colors.is_some() {
                        Some(colors.unwrap().into_rgba_u8().map(|c| *bytemuck::from_bytes::<u32>(&c)))
                    } else {
                        None
                    };
                    
                    if let Some(indices) = reader.read_indices() {
                        // indexed mesh
                        let mut indices = indices.into_u32();
                        while let (Some(a), Some(b), Some(c)) = (indices.next(), indices.next(), indices.next()) {
                            let mut ext = TriExt::default();
                            ext.vertices[0].color = 0x00A0A0FF;
                            ext.vertices[1].color = 0xA000A0FF;
                            ext.vertices[2].color = 0xA0A000FF;
                            if let Some(ref mut colors) = colors {
                                ext.vertices[0].color = colors.next().unwrap_or(0);
                                ext.vertices[1].color = colors.next().unwrap_or(0);
                                ext.vertices[2].color = colors.next().unwrap_or(0);
                            }
                            self.triangles.push(Tri::new(positions[a as usize], positions[b as usize], positions[c as usize]));
                            self.triangles_ext.push(ext);
                        }
                    }
                    else {
                        // non-indexed mesh
                        for p in positions.chunks(3) {
                            let c = [0, 0, 0];
                            let mut ext = TriExt::default();
                            ext.vertices[0].color = c[0];
                            ext.vertices[1].color = c[1];
                            ext.vertices[2].color = c[2];
                            self.triangles.push(Tri::new(p[0], p[1], p[2]));
                            self.triangles_ext.push(ext);
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

    fn update_resolution(&mut self, gpu: &Gpu) {
        let res = [gpu.surface_config.width, gpu.surface_config.height];
        self.frame_uniforms.res = res;
        println!("x: {}, y: {}", res[0], res[1]);
        self.screen_ssbo = gpu.new_storage_buffer(res[0] as u64 * res[1]  as u64 * 4 * 4);

        self.rt_data_binding = gpu.new_bind_group()
            .with_buffer(&self.triangles_ssbo.view_all(), wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&self.triangles_ext_ssbo.view_all(), wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&self.bvh_ssbo.view_all(),       wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&self.screen_ssbo.view_all(),    wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .finish(&mut self.resources);
    }

    fn init(gpu: &Gpu) -> Context {
        let mut scene = FlatScene::default();
        // scene.add_gltf_bytes(&Mat4::from_rotation_x(-90.0_f32.to_radians()), include_bytes!("../resources/suzanne.glb"));
        // scene.add_gltf_bytes(&Mat4::from_translation(vec3(0.0, -3.0, 0.0)), include_bytes!("../resources/simple.glb"));
        let mut ms = MatrixStack::new();
    
        println!("building scene");
        // scene.add_gltf_bytes(ms.top(), include_bytes!("../resources/large/DragonAttenuation.glb"));
 

        // ms.push();
        // ms.translate(vec3(0.0, 0.0, 10.0));
        // ms.rotate_y(180.0f32.to_radians());
        // scene.add_gltf_bytes(ms.top(), include_bytes!("../resources/large/DragonAttenuation.glb"));
        // ms.pop();

        // ms.push();
        // ms.translate(vec3(0.0, 0.0, 5.0));
        // scene.add_gltf_bytes(ms.top(), include_bytes!("../resources/suzanne.glb"));
        // ms.pop();


        scene.add_gltf_bytes(&Mat4::IDENTITY, include_bytes!("../resources/simple_terrain.glb"));

       // scene.add_gltf("resources/large/sponza/Sponza.gltf");
        
        if scene.cameras.is_empty() {
            println!("No camera in scene, falling back to default");
            // vec3f(-3.5, -0.5, 0.5), vec3f(1.0, 0.0, 0.0)
            scene.cameras.push(Camera::default());
        }
        
        println!("Tri count: {}", scene.triangles.len());
        println!("Tri size : {} mb", (scene.triangles.len() * size_of::<Tri>()) / (1000 * 1000));
        println!();
        println!("building bvh");
        let mut bvh = Bvh::new(&scene.triangles, &scene.triangles_ext);
        bvh.build();
        println!("Bvh size : {} mb", (bvh.nodes.len() * size_of::<BvhNode>()) / (1000 * 1000));
        let mut resources = ResourceManager::new();

        let u_frame_0 = FrameUniforms {
            scene: scene.to_gpu(),
            frame: 0,
            res: [gpu.surface_config.width, gpu.surface_config.height],
            time: 0.0,
            reject_hist: 1,
            node_count: bvh.nodes.len() as u32,
            _pad: [0; 2],
        };

        let u_frame_buffer = gpu.new_uniform_buffer(&u_frame_0);

        let u_frame = gpu.new_bind_group()
            .with_buffer(&u_frame_buffer.view_all(), wgpu::ShaderStages::all())
            .finish(&mut resources);

        let buffer_size_mb = 128;

        let triangles_ssbo =        gpu.new_storage_buffer(buffer_size_mb * 1024 * 1024);
        let bvh_ssbo =              gpu.new_storage_buffer(buffer_size_mb * 1024 * 1024);
        let triangles_ext_ssbo =    gpu.new_storage_buffer(buffer_size_mb * 1024 * 1024);
        let screen_ssbo =           gpu.new_storage_buffer(u_frame_0.res[0] as u64 * u_frame_0.res[1] as u64 * 4 * 4);

        let rt_data_bg = gpu.new_bind_group()
            .with_buffer(&triangles_ssbo.view_all(), wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&triangles_ext_ssbo.view_all(), wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&bvh_ssbo.view_all(),       wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .with_buffer(&screen_ssbo.view_all(),    wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .finish(&mut resources);

        // Load the shaders from disk
        let shader_module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
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
                module: &shader_module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
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
                module: &shader_module,
                layout: Some(&raytrace_pipeline_layout),
                entry_point: Some("cs_main"),
                compilation_options: Default::default(),
                cache: None,
            }
        );

        let (flat_tris, flat_exts) = bvh.flat_triangles();
        gpu.queue.write_buffer(&triangles_ssbo.raw, 0, bytemuck::cast_slice(flat_tris.as_slice()));
        gpu.queue.write_buffer(&triangles_ext_ssbo.raw, 0, bytemuck::cast_slice(flat_exts.as_slice()));
        gpu.queue.write_buffer(&bvh_ssbo.raw, 0, bytemuck::cast_slice(bvh.nodes.as_slice()));

        // sanity check for aabb gen
        // let mut last = Aabb::new();
        // for node in &bvh.nodes {
        //     if node.aabb.min() != last.min() && node.aabb.max() != last.max() {
        //         last = node.aabb;
        //         println!("{}, {}, {} to {}, {}, {}", last.data[0], last.data[1], last.data[2], last.data[3], last.data[4], last.data[5]);
        //     }
        // }

        

        Context {
            screen_pipeline,
            shader_module,

            frame_uniforms: u_frame_0,
            frame_uniforms_buffer: u_frame_buffer,
            frame_uniforms_binding: u_frame,
            
            raytrace_pipeline,
            screen_ssbo,
            bvh_ssbo,
            triangles_ssbo,
            triangles_ext_ssbo,
            rt_data_binding: rt_data_bg,

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
    


    gpu.queue.write_buffer(&ctx.frame_uniforms_buffer.raw, 0, bytemuck::bytes_of(&ctx.frame_uniforms));
    
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cpass.set_pipeline(&ctx.raytrace_pipeline);
        cpass.set_bind_group(0, &ctx.frame_uniforms_binding.raw, &[]);
        cpass.set_bind_group(1, &ctx.rt_data_binding.raw, &[]);
        cpass.dispatch_workgroups((ctx.frame_uniforms.res[0] + 7) / 8, (ctx.frame_uniforms.res[1] + 7) / 8, 1);
    }

    {
        let mut rpass = encoder.begin_render_pass(&rpass_desc);
        rpass.set_pipeline(&ctx.screen_pipeline);
        rpass.set_bind_group(0, Some(&ctx.frame_uniforms_binding.raw), &[]);
        rpass.set_bind_group(1, Some(&ctx.rt_data_binding.raw), &[]);
        rpass.draw(0..3, 0..1);
    }

    gpu.queue.submit(Some(encoder.finish()));
    surface_texture.present();
}

async fn fetch_bytes(path: &str) -> Vec<u8> {
    #[cfg(not(target_arch = "wasm32"))] 
    {
        
        std::fs::read(path).unwrap()
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
        typed_arr.to_vec()
    }
}

async fn run() {
    let event_loop = EventLoop::new().unwrap();

    // default size
    let window = new_window(&event_loop, [512, 512]);

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
    let mut last_second = Instant::now();
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
                        gpu.surface_config.width = new_size.width.max(1);
                        gpu.surface_config.height = new_size.height.max(1);
                        gpu.surface_config.width = gpu.surface_config.width.min(4096);
                        gpu.surface_config.height = gpu.surface_config.height.min(4096);
    
                        gpu.surface.configure(&gpu.device, &gpu.surface_config);
                        ctx.update_resolution(&gpu);
                        // On macos the window needs to be redrawn manually after resizing
                        gpu.window.request_redraw();
                    }

                    WindowEvent::RedrawRequested => {
                        this_frame = Instant::now();
                        frames_in_second += 1;
                        gpu.window.request_redraw();
                        let dt = (this_frame - last_frame).as_secs_f32();
                        ctx.scene.cameras[0].update(&mut input, dt);

                        if this_frame.duration_since(last_second).as_secs_f32() >= 1.0 {
                            println!("fps: {}", frames_in_second);
                            frames_in_second = 0;
                            last_second = this_frame;
                        }
                        
                        frame(&gpu, &mut ctx);
                        last_frame = this_frame;
                    },
                    WindowEvent::CursorMoved { device_id, position } => if !input.rmb {last_cursor_pos = position},
                    WindowEvent::MouseInput { device_id, state, button } => {
                        match button {
                            MouseButton::Left =>  input.lmb = state.is_pressed(),
                            MouseButton::Right => input.rmb = state.is_pressed(),
                            _ => (),
                        }

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
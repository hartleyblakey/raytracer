use std::collections::HashMap;

use glam::{uvec2, vec3, vec4, Mat4, UVec2, Vec2, Vec3, Vec4, Vec4Swizzles};
use image::GenericImageView;
use rand::random;

use crate::{fetch_bytes, input::*};

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


pub struct MatrixStack {
    stack: Vec<Mat4>,
}

impl MatrixStack {
    pub fn new() -> Self {
        Self {stack: vec![Mat4::IDENTITY]}
    }
    pub fn top(&mut self) -> &Mat4 {
        self.stack.last().unwrap()
    }
    pub fn push(&mut self) {
        self.stack.push(self.stack.last().copied().unwrap());
    }
    pub fn pop(&mut self) {
        if self.stack.len() > 1 {
            self.stack.pop();
        }
    }
    pub fn rotate_y(&mut self, rad: f32) {
        self.apply(&Mat4::from_rotation_y(rad));
    }
    pub fn translate(&mut self, delta: Vec3) {
        self.apply(&Mat4::from_translation(delta));
    }
    pub fn scale(&mut self, scale: Vec3) {
        self.apply(&Mat4::from_scale(scale));
    }
    pub fn apply(&mut self, t: &Mat4) {
        if self.stack.len() == 1 {
            self.push();
        }
        *self.stack.last_mut().unwrap() = self.top().mul_mat4(t);
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSceneUniform {
    point_lights: [PointLight; 12],
    directional_lights: [DirectionalLight; 4],
    pub camera: GpuCamera,
    pub tri_count: u32,
    pub num_point_lights: u32,
    pub num_directional_lights: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuTexcoord {
    pos: Vec2,
    offset: u32,
    size: u32,
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
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuVertexExt {
    tex0: GpuTexcoord,
    normal: Vec2,
    color: u32,
    _pad: f32
}


#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TriExt {
    vertices: [GpuVertexExt; 3]
}


#[derive(Default)]
pub struct FlatScene {
    pub triangles: Vec<Tri>,
    pub triangles_ext: Vec<TriExt>,
    pub texture_data: Vec<u32>,
    pub texture_map: HashMap<usize, (usize, UVec2)>,
    pub cameras:   Vec<Camera>,
    pub point_lights: Vec<PointLight>,
    pub directional_lights: Vec<DirectionalLight>,
}

impl FlatScene {
    pub fn add_gltf_bytes(&mut self, transform: &Mat4, bytes: &[u8]) {
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

    pub fn from_gltf_vec3(v: Vec3) -> Vec3 {
        vec3(v.z, v.x, v.y)
    }

    pub async fn add_gltf(&mut self, transform: &Mat4, path: &str) {
        self.add_gltf_bytes(transform, fetch_bytes(path).await.as_slice());
    }   

    fn rgba8_to_u32(x: &[u8; 4]) -> u32 {
        let mut r: u32 = 0;
        r |= (x[0] as u32) << 24;
        r |= (x[1] as u32) << 16;
        r |= (x[2] as u32) << 8 ;
        r |= (x[3] as u32) << 0 ;
        r
    }

    fn add_gltf_node(&mut self, buffers: &Vec<gltf::buffer::Data>, node: gltf::Node, ms: &mut MatrixStack) {
        ms.push();
        ms.apply(&Mat4::from_cols_array_2d(&node.transform().matrix()));
        let my_top = from_gltf_mat4(ms.top());
        if let Some(camera) = node.camera() {
            self.cameras.push(Camera::from_gltf(camera, ms.top()));
        }

        if let Some(light) = node.light() {
            match light.kind() {
                gltf::khr_lights_punctual::Kind::Directional => {
                    let dir = my_top.transform_vector3(FORWARD);
                    let d = DirectionalLight { 
                        direction: vec4(dir.x, dir.y, dir.z, 0.0), 
                        intensity: light.intensity() * vec4(light.color()[0], light.color()[1], light.color()[2], 0.0)
                    };
                    self.directional_lights.push(d);
                },
                gltf::khr_lights_punctual::Kind::Point => {
                    let pos = my_top.transform_point3(vec3(0.0, 0.0, 0.0));
                    let p = PointLight {
                        position: vec4(pos.x, pos.y, pos.z, 0.0),
                        intensity: light.intensity() * vec4(light.color()[0], light.color()[1], light.color()[2], 0.0)
                    };
                    self.point_lights.push(p);
                },
                gltf::khr_lights_punctual::Kind::Spot { .. } => (),
            }
        }
        
        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                if primitive.mode() == gltf::mesh::Mode::Triangles {

                    // tell the reader where to find the buffer data
                    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                    
                    // collect transformed vertex positions into a vec of vec3s so we can index them
                    let positions: Vec<Vec3> = reader.read_positions().unwrap()
                    .map( |p| 
                        Self::from_gltf_vec3(ms.top().transform_point3(Vec3::from_slice(&p)))
                    ).collect();

                    let mut base_color_texture_id = 0;
                    let mut base_color_texture_offset = 0;
                    let mut base_color_texture_size = uvec2(0, 0);

                    if let Some(tex) = primitive.material().pbr_metallic_roughness().base_color_texture() {
                        base_color_texture_id = tex.tex_coord();
                        println!("Found base_color_texture");
                        // if we have not already loaded the image
                        if !self.texture_map.contains_key(&tex.texture().index()) {
                            // load the image
                            let image = match tex.texture().source().source() {
                                // image comes buffer view, load the raw bytes
                                gltf::image::Source::View { view, .. } => {
                                    let start = view.offset();
                                    let end = start + view.length();
                                    let image_data = &buffers[view.buffer().index()][start..end];
                                    match image::load_from_memory(image_data) {
                                        Ok(image) => image,
                                        Err(e) => {println!("{e}"); panic!()},
                                    }
                                    
                                },
                                // untested
                                gltf::image::Source::Uri { uri, .. } => {
                                    image::ImageReader::open(uri).unwrap().decode().unwrap()
                                },
                            };

                            let rgba8_image = image.to_rgba8();

                            base_color_texture_offset = self.texture_data.len();
                            base_color_texture_size = uvec2(image.dimensions().0, image.dimensions().1);
                            for pixel in rgba8_image.pixels() {
                                self.texture_data.push(Self::rgba8_to_u32(&pixel.0))
                            }
                            println!("Found texture with offset {base_color_texture_offset}, size {} by {}", base_color_texture_size.x, base_color_texture_size.y);

                            // record that we loaded the image
                            self.texture_map.insert(tex.texture().index(), (base_color_texture_offset, base_color_texture_size));
                        

                        } else {
                            // retrieve the image location from the cache
                            (base_color_texture_offset, base_color_texture_size) = self.texture_map[&tex.texture().index()];
                        }
                    
                    }
                    
                    
                    // collect vertex attributes into vectors so we can index them
                    //  vertex colors
                    let colors = reader.read_colors(0);
                    let colors: Vec<u32> = if colors.is_some() {
                        colors.unwrap().into_rgba_u8().map(|c| Self::rgba8_to_u32(&c)).collect()
                    } else {
                        Vec::new()
                    };
                    //  base color texcoords
                    let texcoords = reader.read_tex_coords(base_color_texture_id);
                    let texcoords: Vec<Vec2> = if texcoords.is_some() {
                        texcoords.unwrap().into_f32().map(|uv| Vec2::from_slice(&uv)).collect()
                    } else {
                        Vec::new()
                    };

                    if let Some(indices) = reader.read_indices() {
                        // indexed mesh
                        let mut indices = indices.into_u32();
                        while let (Some(a), Some(b), Some(c)) = (indices.next(), indices.next(), indices.next()) {
                            let mut ext = TriExt::default();

                            if !colors.is_empty() {
                                ext.vertices[0].color = colors[a as usize];
                                ext.vertices[1].color = colors[b as usize];
                                ext.vertices[2].color = colors[c as usize];
                            }

                            if !texcoords.is_empty() {
                                ext.vertices[0].tex0 = GpuTexcoord::new(base_color_texture_offset as u32, base_color_texture_size, texcoords[a as usize]);
                                ext.vertices[1].tex0 = GpuTexcoord::new(base_color_texture_offset as u32, base_color_texture_size, texcoords[b as usize]);
                                ext.vertices[2].tex0 = GpuTexcoord::new(base_color_texture_offset as u32, base_color_texture_size, texcoords[c as usize]);
                            }

                            self.triangles.push(Tri::new(positions[a as usize], positions[b as usize], positions[c as usize]));
                            self.triangles_ext.push(ext);
                        }
                    }
                    else {
                        // non-indexed mesh (untested)
                        let mut i = 0;
                        for p in positions.chunks(3) {
                        
                            let mut ext = TriExt::default();

                            if !colors.is_empty() {
                                let c  = &colors[i..(i+3)];
                                ext.vertices[0].color = c[0];
                                ext.vertices[1].color = c[1];
                                ext.vertices[2].color = c[2];
                            }

                            if !texcoords.is_empty() {
                                let tc = &texcoords[i..(i+3)];
                                ext.vertices[0].tex0 = GpuTexcoord::new(base_color_texture_offset as u32, base_color_texture_size, tc[i + 0]);
                                ext.vertices[1].tex0 = GpuTexcoord::new(base_color_texture_offset as u32, base_color_texture_size, tc[i + 1]);
                                ext.vertices[2].tex0 = GpuTexcoord::new(base_color_texture_offset as u32, base_color_texture_size, tc[i + 2]);
                            }

                            self.triangles.push(Tri::new(p[0], p[1], p[2]));
                            self.triangles_ext.push(ext);
                            i += 3;
                        }
                    }
                } else {
                    panic!("Non-triangle primitives not supported");
                }
            }
        }

        for child in node.children() {
            self.add_gltf_node(buffers, child, ms);
        }

        ms.pop();
    }

    pub fn to_gpu(&self) -> GpuSceneUniform {
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



#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Tri {
    vertices: [Vec4; 3],
}

impl Tri {
    pub fn new(p1: Vec3, p2: Vec3, p3: Vec3) -> Tri {
        let c = (p1 + p2 + p3) / 3.0;
        Tri {
            vertices: [vec4(p1.x, p1.y, p1.z, c.x), vec4(p2.x, p2.y, p2.z, c.y), vec4(p3.x, p3.y, p3.z, c.z)],
        }
    }

    pub fn dummy(c: Vec3, size: f32) -> Tri {
        
        Tri::new(
            c + vec3(random(), random(), random()) * size - size * 0.5,
            c + vec3(random(), random(), random()) * size - size * 0.5,
            c + vec3(random(), random(), random()) * size - size * 0.5,
        )
    }

    pub fn aabb(&self) -> Aabb {
        Aabb::point(self.vertices[0].xyz())
            .with(Aabb::point(self.vertices[1].xyz()))
            .with(Aabb::point(self.vertices[2].xyz()))
    }

    pub fn centroid(&self) -> Vec3 {
        // (self.vertices[0].xyz() + self.vertices[1].xyz() + self.vertices[2].xyz()) / 3.0
        vec3(self.vertices[0][3], self.vertices[1][3], self.vertices[2][3])
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Aabb {
    // alignment rules
    data: [f32; 6]
}

impl Aabb {
    pub fn new() -> Self {
        Self {
            data: [f32::MAX, f32::MAX, f32::MAX, f32::MIN, f32::MIN, f32::MIN]
        }
    }

    fn surface(&self) -> f32 {
        let size = self.max() - self.min();
        (size.x * size.y + size.y * size.z + size.z * size.x) * 2.0
    } 

    pub fn with(&self, other: Self) -> Self {
        Self {
            data:  [self.data[0].min(other.data[0]),
                    self.data[1].min(other.data[1]),
                    self.data[2].min(other.data[2]),
                    self.data[3].max(other.data[3]),
                    self.data[4].max(other.data[4]),
                    self.data[5].max(other.data[5])]
        }
    }

    pub fn expand(&mut self, other: Self) {
        self.data = self.with(other).data;
    }

    pub fn point(point: Vec3) -> Self {
        Self {
            data: [point.x - 0.00001, point.y - 0.00001, point.z - 0.00001, point.x + 0.00001, point.y + 0.00001, point.z + 0.00001]
        }
    }

    pub fn min(&self) -> Vec3 {
        vec3(self.data[0], self.data[1], self.data[2])
    }

    pub fn max(&self) -> Vec3 {
        vec3(self.data[3], self.data[4], self.data[5])
    }
}

// structure from https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BvhNode {
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
pub struct Bvh<'a> {
    pub nodes: Vec<BvhNode>,
    pub tris: &'a Vec<Tri>,
    pub tri_exts: &'a Vec<TriExt>,
    indices: Vec<u32>,
}

impl<'a> Bvh<'a> {
    pub fn new(triangles: &'a Vec<Tri>, exts: &'a Vec<TriExt>) -> Self {
        Self {
            nodes: Vec::new(),
            tris: triangles,
            tri_exts: exts,
            indices: (0..triangles.len() as u32).collect(),
        }
    }

    pub fn build(&mut self) {
        self.nodes.push(BvhNode::from_tris(0, self.tris.len() as u32, &self.indices, &self.tris));
        self.subdivide(self.nodes.len() - 1);
    }

    /// remove the layer of indirection used to build the BVH
    pub fn flat_triangles(&self) -> (Vec<Tri>, Vec<TriExt>) {
        let mut tris = self.tris.clone();
        let mut exts = self.tri_exts.clone();
        for i in 0..tris.len() {
            tris[i] = self.tris[self.indices[i] as usize];
            exts[i] = self.tri_exts[self.indices[i] as usize];
        }
        (tris, exts)
    }

    fn evaluate_split(&self, node: &BvhNode, axis: usize, split: f32) -> f32 {
        let mut left_aabb = Aabb::new();
        let mut right_aabb = Aabb::new();
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
    fn subdivide(&mut self, node_idx: usize) {
        let node = self.nodes[node_idx];
     
        if node.count <= 2 {
            return;
        }

        let (axis, split) = self.find_split_approx(&node, 16);

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

        let mut left = BvhNode::new();
        left.first = node.first;
        left.count = i  as u32 - node.first;
        left.update_aabb(&self.indices, &self.tris);

        // dont subdivide empty nodes
        if left.count == 0 || left.count == node.count {
            return;
        }

        let mut right = BvhNode::new();
        right.first = i as u32;
        right.count = node.count - left.count;
        right.update_aabb(&self.indices, &self.tris);


        // we no longer hold any triangles
        let children_idx = self.nodes.len();
        self.nodes[node_idx].count = 0;
        self.nodes[node_idx].first = children_idx as u32;

        self.nodes.push(left);
        self.nodes.push(right);

        self.subdivide(children_idx);
        self.subdivide(children_idx + 1);
    }
}
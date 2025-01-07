use std::collections::HashMap;

use glam::{uvec2, vec3, vec4, Mat4, UVec2, Vec2, Vec3, Vec4, Vec4Swizzles};
use image::GenericImageView;
use rand::random;

use crate::{fetch_bytes, input::*};

#[repr(C)]
#[derive(Default, Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PointLight {
    position: Vec4,
    intensity: Vec4,
}


#[repr(C)]
#[derive(Default, Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DirectionalLight {
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
pub struct GpuTextureRef {
    offset: u32,
    size: u32,
}

impl GpuTextureRef {
    fn new(offset: u32, size: UVec2) -> Self {
        let size = (size.x << 16) | size.y;
        Self {
            offset,
            size,
        }
    }

    fn new_literal(val: f32) -> Self {
        Self {
            offset: bytemuck::cast(val),
            size: 0,
        }
    }

    fn size(&self) -> UVec2 {
        uvec2(self.size >> 16, self.size & 0xFFFF)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuVertexExt {
    tex0: Vec2,
    normal: Vec2,
    color: u32,
    _pad: f32
}


#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TriExt {
    vertices: [GpuVertexExt; 3]
}


// pub struct Ray {
//     origin: Vec3,
//     dir: Vec3,
//     idir: Vec3,
// }

// impl Ray {
//     fn default() -> Ray {
//         return Ray {
//             origin: vec3(0.0, 0.0, 0.0),
//             dir: vec3(1.0, f32::MIN, f32::MIN),
//             idir: vec3(1.0, f32::MAX, f32::MAX)
//         }
//     }

//     fn point(&mut self, dir: Vec3) {
//         self.dir = dir;
//         self.idir = vec3(1.0, 1.0, 1.0) / self.dir;
//     }
// }


#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]pub struct Material {
    albedo: GpuTextureRef,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Primitive {
    transform:      Mat4,
    inv_transform:  Mat4,
    material:       Material,
    bvh_idx:        u32,
    _pad:           u32,
}

impl Primitive {
    fn new(transform: &Mat4, material: Material, bvh_idx: u32) -> Self {
        Self {
            transform: *transform,
            inv_transform: transform.as_dmat4().inverse().as_mat4(),
            material,
            bvh_idx,
            _pad: 0,
        }
    }
}

#[derive(Default)]
pub struct Scene {

    /// flat array of primitives that share a material
    pub primitives:         Vec<Primitive>,

    /// map from a mesh's json index to a map of json primitive indices to indices into my primitive array
    /// 
    /// resets on each file load
    pub loaded_meshes:     HashMap<usize, HashMap<usize, usize>>,

    /// global buffer of triangle position data
    pub tris:               Vec<Tri>,
    pub tri_exts:           Vec<TriExt>,

    /// global buffer of rgba8 texture data
    pub texture_data:       Vec<u32>,
    pub texture_map:        HashMap<usize, GpuTextureRef>,

    pub bvh_node_data:      Vec<BvhNode>,

    /// cameras in scene
    pub cameras:            Vec<Camera>,

    pub point_lights:       Vec<PointLight>,
    pub directional_lights: Vec<DirectionalLight>,

    /// rgba32f equirectangular environment map pixel data
    pub env_map_data:       Vec<[f32; 4]>,
}

impl Scene {
    pub async fn add_gltf(&mut self, transform: &Mat4, path: &str) {

        self.add_gltf_bytes(transform, fetch_bytes(path).await.as_slice());
    }

    pub fn add_gltf_bytes(&mut self, transform: &Mat4, bytes: &[u8]) {
        self.loaded_meshes.clear();
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

    fn rgba8_to_u32(x: &[u8; 4]) -> u32 {
        let mut r: u32 = 0;
        r |= (x[0] as u32) << 24;
        r |= (x[1] as u32) << 16;
        r |= (x[2] as u32) << 8 ;
        r |= (x[3] as u32) << 0 ;
        r
    }

    fn add_gltf_texture(&mut self, tex: &gltf::texture::Info, buffers: &Vec<gltf::buffer::Data>) -> GpuTextureRef {
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

            let tex_ref = GpuTextureRef::new(
                self.texture_data.len() as u32, 
                uvec2(image.dimensions().0, image.dimensions().1)
            );
            
            for pixel in rgba8_image.pixels() {
                self.texture_data.push(Self::rgba8_to_u32(&pixel.0))
            }
            println!("Found texture with offset {}, size {} by {}", tex_ref.offset, tex_ref.size().x, tex_ref.size().y);

            // record that we loaded the image
            self.texture_map.insert(tex.texture().index(), tex_ref);
            tex_ref

        } else {
            *self.texture_map.get(&tex.texture().index()).unwrap()
        }
    }
    
    fn add_gltf_node(&mut self, buffers: &Vec<gltf::buffer::Data>, node: gltf::Node, ms: &mut MatrixStack) {
        ms.push();
        ms.apply(&Mat4::from_cols_array_2d(&node.transform().matrix()));
        let node_transform_mine = from_gltf_mat4(ms.top());
        if let Some(camera) = node.camera() {
            self.cameras.push(Camera::from_gltf(camera, ms.top()));
        }

        if let Some(light) = node.light() {
            match light.kind() {
                gltf::khr_lights_punctual::Kind::Directional => {
                    let dir = node_transform_mine.transform_vector3(FORWARD);
                    let d = DirectionalLight { 
                        direction: vec4(dir.x, dir.y, dir.z, 0.0), 
                        intensity: light.intensity() * vec4(light.color()[0], light.color()[1], light.color()[2], 0.0)
                    };
                    self.directional_lights.push(d);
                },
                gltf::khr_lights_punctual::Kind::Point => {
                    let pos = node_transform_mine.transform_point3(vec3(0.0, 0.0, 0.0));
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

            if let Some(loaded_primitives) = self.loaded_meshes.get(&mesh.index()) {
                for primitive in mesh.primitives() {

                    // if the primitive was already loaded, copy it and change the transforms
                    if let Some(&prim_idx) = loaded_primitives.get(&primitive.index()) {
                        let mut new_primitive = self.primitives[prim_idx];
                        new_primitive.transform = node_transform_mine;
                        new_primitive.inv_transform = node_transform_mine.inverse();
                        self.primitives.push(new_primitive);
                        continue;
                    }
                }
            } else {
                let mut loaded_primitives: HashMap<usize, usize> = HashMap::new();

                for primitive in mesh.primitives() {

                    // if the primitive was already loaded, copy it and change the transforms
                    if let Some(&prim_idx) = loaded_primitives.get(&primitive.index()) {
                        println!("Instanced a primitive!");
                        let mut new_primitive = self.primitives[prim_idx];
                        new_primitive.transform = node_transform_mine;
                        new_primitive.inv_transform = node_transform_mine.inverse();
                        self.primitives.push(new_primitive);
                        continue;
                    }

                    if primitive.mode() == gltf::mesh::Mode::Triangles {

                        // tell the reader where to find the buffer data
                        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                        
                        // collect vertex positions into a vec of vec3s so we can index them
                        let positions: Vec<Vec3> = reader.read_positions()
                            .unwrap()
                            .map( |p| Self::from_gltf_vec3(Vec3::from_slice(&p)))
                            .collect();

                        let mut try_load_texture = |opt_tex| {
                            let mut tex_ref = GpuTextureRef::default();
                            let mut texcoord_id = u32::MAX;
                            if let Some(tex) = opt_tex {
                                tex_ref = self.add_gltf_texture(&tex, buffers);
                                texcoord_id = tex.tex_coord();
                            }
                            
                            //  base color texcoords
                            let texcoords = reader.read_tex_coords(texcoord_id);
                            let texcoords: Vec<Vec2> = if texcoords.is_some() {
                                texcoords.unwrap().into_f32().map(|uv| Vec2::from_slice(&uv)).collect()
                            } else {
                                Vec::new()
                            };

                            (tex_ref, texcoords)
                        };

                        let (albedo, albedo_tc) = try_load_texture(primitive.material().pbr_metallic_roughness().base_color_texture());

                        let material = Material {
                            albedo
                        };

                        // collect vertex attributes into vectors so we can index them
                        //  vertex colors
                        let colors = reader.read_colors(0);
                        let colors: Vec<u32> = if colors.is_some() {
                            colors.unwrap().into_rgba_u8().map(|c| Self::rgba8_to_u32(&c)).collect()
                        } else {
                            Vec::new()
                        };

                        let first_new_tri = self.tris.len();
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

                                if !albedo_tc.is_empty() {
                                    ext.vertices[0].tex0 = albedo_tc[a as usize];
                                    ext.vertices[1].tex0 = albedo_tc[b as usize];
                                    ext.vertices[2].tex0 = albedo_tc[c as usize];
                                }

                                self.tris.push(Tri::new(positions[a as usize], positions[b as usize], positions[c as usize]));
                                self.tri_exts.push(ext);
                            }
                        }
                        else {
                            panic!("Only supporting indexed meshes for now");
                        }


                        // build a bvh around the new triangles
                        let mut bvh = Bvh::new(&self.tris.as_slice(), first_new_tri, self.tris.len() - first_new_tri);
                        // re-arrange the new triangles to match the BVH nodes
                        bvh.flatten_triangles(self.tris.as_mut_slice(), self.tri_exts.as_mut_slice());
                        let bvh_root = self.bvh_node_data.len() as u32;
                        self.bvh_node_data.append(&mut bvh.nodes);

                        for node in &mut self.bvh_node_data[bvh_root as usize .. ] {
                            if node.count == 0 {
                                // leaf node
                                node.first += bvh_root;
                            }
                        }

                        // add this primitive to the scene
                        self.primitives.push(
                            Primitive::new(&node_transform_mine, material, bvh_root)
                        );

                        println!("Adding primitive with {} triangles, bvh root at index {}", self.tris.len() - first_new_tri, bvh_root);

                        // mark this primitive as already loaded
                        loaded_primitives.insert(
                            primitive.index(),
                            self.primitives.len() - 1
                        );

                    } else {
                        panic!("Non-triangle primitives not supported");
                    }
                }

                // mark this mesh index as already loaded, keeping a reference to the loaded primitives
                self.loaded_meshes.insert(
                    mesh.index(),
                    loaded_primitives
                );
            }
        }

        for child in node.children() {
            self.add_gltf_node(buffers, child, ms);
        }

        ms.pop();
    }


    pub async fn set_equirectangular_env_map(&mut self, path: &str) {
        let buffer = fetch_bytes(path).await;
        let image = image::load_from_memory(buffer.as_slice()).expect(format!("Expected file at path {path}").as_str());
        let image = image.into_rgba32f();
        self.env_map_data.clear();
        for pixel in image.pixels() {
            self.env_map_data.push(pixel.0);
        }
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
            tri_count: self.tris.len() as u32,
        }
    }

    pub fn closest_hit(&self, ro: Vec3, rd: Vec3) -> Option<f32> {
        let mut closest_t = None;
        for primitive in &self.primitives {
            if let Some(t) = Bvh::closest_hit_unindexed(
                &self.bvh_node_data, 
                primitive.bvh_idx, 
                &self.tris, 
                primitive.inv_transform.transform_point3(ro), 
                primitive.inv_transform.transform_vector3(rd).normalize(),
            ) {
                closest_t = Some(closest_t.unwrap_or(f32::MAX).min(t / primitive.inv_transform.transform_vector3(rd).length()));
            }
            
        }
        if closest_t.is_none() {
            println!("Ray Miss!");
        }
        closest_t
    }
    
}

// #[derive(Default)]
// pub struct FlatScene {
//     pub triangles: Vec<Tri>,
//     pub triangles_ext: Vec<TriExt>,
//     pub texture_data: Vec<u32>,
//     pub texture_map: HashMap<usize, (usize, UVec2)>,
//     pub cameras:   Vec<Camera>,
//     pub point_lights: Vec<PointLight>,
//     pub directional_lights: Vec<DirectionalLight>,
//     pub env_map_data: Vec<[f32; 4]>,
// }

// impl FlatScene {
//     pub fn add_gltf_bytes(&mut self, transform: &Mat4, bytes: &[u8]) {
//         let (document, buffers, _) = gltf::import_slice(bytes).unwrap();
//         let mut ms = MatrixStack::new();
//         ms.push();
//         ms.apply(&transform);
//         for scene in document.scenes(){
//             for node in scene.nodes() {
//                 self.add_gltf_node(&buffers, node, &mut ms);
//             }
//         }
//     }   

//     pub fn from_gltf_vec3(v: Vec3) -> Vec3 {
//         vec3(v.z, v.x, v.y)
//     }

//     pub async fn add_gltf(&mut self, transform: &Mat4, path: &str) {
//         self.add_gltf_bytes(transform, fetch_bytes(path).await.as_slice());
//     }

//     pub async fn set_equirectangular_env_map(&mut self, path: &str) {
//         let buffer = fetch_bytes(path).await;
//         let image = image::load_from_memory(buffer.as_slice()).expect(format!("Expected file at path {path}").as_str());
//         let image = image.into_rgba32f();
//         self.env_map_data.clear();
//         for pixel in image.pixels() {
//             self.env_map_data.push(pixel.0);
//         }
//     }


//     fn rgba8_to_u32(x: &[u8; 4]) -> u32 {
//         let mut r: u32 = 0;
//         r |= (x[0] as u32) << 24;
//         r |= (x[1] as u32) << 16;
//         r |= (x[2] as u32) << 8 ;
//         r |= (x[3] as u32) << 0 ;
//         r
//     }

    

//     fn add_gltf_node(&mut self, buffers: &Vec<gltf::buffer::Data>, node: gltf::Node, ms: &mut MatrixStack) {
//         ms.push();
//         ms.apply(&Mat4::from_cols_array_2d(&node.transform().matrix()));
//         let my_top = from_gltf_mat4(ms.top());
//         if let Some(camera) = node.camera() {
//             self.cameras.push(Camera::from_gltf(camera, ms.top()));
//         }

//         if let Some(light) = node.light() {
//             match light.kind() {
//                 gltf::khr_lights_punctual::Kind::Directional => {
//                     let dir = my_top.transform_vector3(FORWARD);
//                     let d = DirectionalLight { 
//                         direction: vec4(dir.x, dir.y, dir.z, 0.0), 
//                         intensity: light.intensity() * vec4(light.color()[0], light.color()[1], light.color()[2], 0.0)
//                     };
//                     self.directional_lights.push(d);
//                 },
//                 gltf::khr_lights_punctual::Kind::Point => {
//                     let pos = my_top.transform_point3(vec3(0.0, 0.0, 0.0));
//                     let p = PointLight {
//                         position: vec4(pos.x, pos.y, pos.z, 0.0),
//                         intensity: light.intensity() * vec4(light.color()[0], light.color()[1], light.color()[2], 0.0)
//                     };
//                     self.point_lights.push(p);
//                 },
//                 gltf::khr_lights_punctual::Kind::Spot { .. } => (),
//             }
//         }
        
//         if let Some(mesh) = node.mesh() {
//             for primitive in mesh.primitives() {
//                 if primitive.mode() == gltf::mesh::Mode::Triangles {

//                     // tell the reader where to find the buffer data
//                     let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                    
//                     // collect transformed vertex positions into a vec of vec3s so we can index them
//                     let positions: Vec<Vec3> = reader.read_positions().unwrap()
//                     .map( |p| 
//                         Self::from_gltf_vec3(ms.top().transform_point3(Vec3::from_slice(&p)))
//                     ).collect();

//                     let mut base_color_texture_id = 0;
//                     let mut base_color_texture_offset = 0;
//                     let mut base_color_texture_size = uvec2(0, 0);

//                     if let Some(tex) = primitive.material().pbr_metallic_roughness().base_color_texture() {
//                         base_color_texture_id = tex.tex_coord();
//                         println!("Found base_color_texture");
//                         // if we have not already loaded the image
//                         if !self.texture_map.contains_key(&tex.texture().index()) {
//                             // load the image
//                             let image = match tex.texture().source().source() {
//                                 // image comes buffer view, load the raw bytes
//                                 gltf::image::Source::View { view, .. } => {
//                                     let start = view.offset();
//                                     let end = start + view.length();
//                                     let image_data = &buffers[view.buffer().index()][start..end];
//                                     match image::load_from_memory(image_data) {
//                                         Ok(image) => image,
//                                         Err(e) => {println!("{e}"); panic!()},
//                                     }
                                    
//                                 },
//                                 // untested
//                                 gltf::image::Source::Uri { uri, .. } => {
//                                     image::ImageReader::open(uri).unwrap().decode().unwrap()
//                                 },
//                             };

//                             let rgba8_image = image.to_rgba8();

//                             base_color_texture_offset = self.texture_data.len();
//                             base_color_texture_size = uvec2(image.dimensions().0, image.dimensions().1);
//                             for pixel in rgba8_image.pixels() {
//                                 self.texture_data.push(Self::rgba8_to_u32(&pixel.0))
//                             }
//                             println!("Found texture with offset {base_color_texture_offset}, size {} by {}", base_color_texture_size.x, base_color_texture_size.y);

//                             // record that we loaded the image
//                             self.texture_map.insert(tex.texture().index(), (base_color_texture_offset, base_color_texture_size));
                        

//                         } else {
//                             // retrieve the image location from the cache
//                             (base_color_texture_offset, base_color_texture_size) = self.texture_map[&tex.texture().index()];
//                         }
                    
//                     }
                    
                    
//                     // collect vertex attributes into vectors so we can index them
//                     //  vertex colors
//                     let colors = reader.read_colors(0);
//                     let colors: Vec<u32> = if colors.is_some() {
//                         colors.unwrap().into_rgba_u8().map(|c| Self::rgba8_to_u32(&c)).collect()
//                     } else {
//                         Vec::new()
//                     };
//                     //  base color texcoords
//                     let texcoords = reader.read_tex_coords(base_color_texture_id);
//                     let texcoords: Vec<Vec2> = if texcoords.is_some() {
//                         texcoords.unwrap().into_f32().map(|uv| Vec2::from_slice(&uv)).collect()
//                     } else {
//                         Vec::new()
//                     };

//                     if let Some(indices) = reader.read_indices() {
//                         // indexed mesh
//                         let mut indices = indices.into_u32();
//                         while let (Some(a), Some(b), Some(c)) = (indices.next(), indices.next(), indices.next()) {
//                             let mut ext = TriExt::default();

//                             if !colors.is_empty() {
//                                 ext.vertices[0].color = colors[a as usize];
//                                 ext.vertices[1].color = colors[b as usize];
//                                 ext.vertices[2].color = colors[c as usize];
//                             }

//                             if !texcoords.is_empty() {
//                                 ext.vertices[0].tex0 = GpuTextureRef::new(base_color_texture_offset as u32, base_color_texture_size, texcoords[a as usize]);
//                                 ext.vertices[1].tex0 = GpuTextureRef::new(base_color_texture_offset as u32, base_color_texture_size, texcoords[b as usize]);
//                                 ext.vertices[2].tex0 = GpuTextureRef::new(base_color_texture_offset as u32, base_color_texture_size, texcoords[c as usize]);
//                             }

//                             self.triangles.push(Tri::new(positions[a as usize], positions[b as usize], positions[c as usize]));
//                             self.triangles_ext.push(ext);
//                         }
//                     }
//                     else {
//                         // non-indexed mesh (untested)
//                         let mut i = 0;
//                         for p in positions.chunks(3) {
                        
//                             let mut ext = TriExt::default();

//                             if !colors.is_empty() {
//                                 let c  = &colors[i..(i+3)];
//                                 ext.vertices[0].color = c[0];
//                                 ext.vertices[1].color = c[1];
//                                 ext.vertices[2].color = c[2];
//                             }

//                             if !texcoords.is_empty() {
//                                 let tc = &texcoords[i..(i+3)];
//                                 ext.vertices[0].tex0 = GpuTextureRef::new(base_color_texture_offset as u32, base_color_texture_size, tc[i + 0]);
//                                 ext.vertices[1].tex0 = GpuTextureRef::new(base_color_texture_offset as u32, base_color_texture_size, tc[i + 1]);
//                                 ext.vertices[2].tex0 = GpuTextureRef::new(base_color_texture_offset as u32, base_color_texture_size, tc[i + 2]);
//                             }

//                             self.triangles.push(Tri::new(p[0], p[1], p[2]));
//                             self.triangles_ext.push(ext);
//                             i += 3;
//                         }
//                     }
//                 } else {
//                     panic!("Non-triangle primitives not supported");
//                 }
//             }
//         }

//         for child in node.children() {
//             self.add_gltf_node(buffers, child, ms);
//         }

//         ms.pop();
//     }

//     pub fn to_gpu(&self) -> GpuSceneUniform {
//         let mut point_lights = [PointLight::default(); 12];
//         let mut directional_lights = [DirectionalLight::default(); 4];

//         for i in 0..self.point_lights.len().min(point_lights.len()) {
//             point_lights[i] = self.point_lights[i];
//         }

//         for i in 0..self.directional_lights.len().min(directional_lights.len()) {
//             directional_lights[i] = self.directional_lights[i];
//         }

//         GpuSceneUniform {
//             _pad: 0,
//             camera: self.cameras[0].to_gpu(),
//             point_lights,
//             directional_lights,
//             num_directional_lights: self.directional_lights.len() as u32,
//             num_point_lights: self.point_lights.len() as u32,
//             tri_count: self.triangles.len() as u32,
//         }
//     }
// }



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

    pub fn aabb(&self) -> Aabb {
        Aabb::point(self.vertices[0].xyz())
            .with(Aabb::point(self.vertices[1].xyz()))
            .with(Aabb::point(self.vertices[2].xyz()))
    }

    pub fn centroid(&self) -> Vec3 {
        // (self.vertices[0].xyz() + self.vertices[1].xyz() + self.vertices[2].xyz()) / 3.0
        vec3(self.vertices[0][3], self.vertices[1][3], self.vertices[2][3])
    }

    pub fn closest_hit(&self, ro: Vec3, rd: Vec3) -> Option<f32> {
        let edge1 = self.vertices[1].xyz() - self.vertices[0].xyz();
        let edge2 = self.vertices[2].xyz() - self.vertices[0].xyz();
        let h = Vec3::cross( rd, edge2 );
        let a = Vec3::dot( edge1, h );
        if a > -0.000002 && a < 0.000002 {
            return None;
        }// ray parallel to triangle
        let f = 1.0 / a;
        let s = ro - self.vertices[0].xyz();
        let u = f * Vec3::dot( s, h );
        if u < 0.0 || u > 1.0 {
            return None;
        }
        let q = Vec3::cross( s, edge1 );
        let v = f * Vec3::dot( rd, q );
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = f * Vec3::dot( edge2, q );
        if t > 0.000002 {
            return Some(t);
        } else {
            return None;
        }
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

    pub fn closest_hit(&self, ro: Vec3, rd: Vec3) -> Option<f32> {
        let bmin = self.min();
        let bmax = self.max();
    
        if (ro.x > bmin.x && ro.y > bmin.y && ro.z > bmin.z) && (ro.x < bmax.x && ro.y < bmax.y && ro.z < bmax.z) {
            return Some(0.0);
        }
    
        let rmin = (bmin - ro) / rd;
        let rmax = (bmax - ro) / rd;
    
        let tmin = Vec3::min(rmin, rmax);
        let tmax = Vec3::max(rmin, rmax);
    
        let t0 = f32::max(tmin.x, f32::max(tmin.y, tmin.z));
        let t1 = f32::min(tmax.x, f32::min(tmax.y, tmax.z));
    
        if t0 >= t1 || t0 < 0.0 {
            return None;
        }
    
        Some(t0)
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

    fn from_tris(first: u32, count: u32, indices: &Vec<u32>, tris: &[Tri], offset: usize) -> Self {
        let mut new = Self::new();
        new.first = first;
        new.count = count;
        new.update_aabb(indices, tris, offset);
        new
    }

    fn update_aabb(&mut self, indices: &Vec<u32>, tris: &[Tri], offset: usize) {
        if self.count != 0 {
            self.aabb = tris[indices[self.first as usize - offset] as usize].aabb();
            for i in self.first..self.first + self.count {
                self.aabb.expand(tris[indices[i as usize - offset] as usize].aabb());
            }
        }
    }
}

pub struct Bvh {
    pub nodes: Vec<BvhNode>,
    indices: Vec<u32>,
    offset: usize,
    size: usize,
}

impl Bvh {
    pub fn new(tris: &[Tri], offset: usize, size: usize) -> Self {
        let mut res = Self {
            nodes: Vec::new(),
            indices: ( (offset  as u32) .. (offset + size) as u32 ).collect(),
            offset,
            size
        };

        res.nodes.push(BvhNode::from_tris(offset as u32, size as u32, &res.indices, &tris, offset));
        res.subdivide(res.nodes.len() - 1, tris);
        return res;
    }

    /// remove the layer of indirection used to build the BVH
    pub fn flatten_triangles(&mut self, tris: &mut [Tri], tri_exts: &mut [TriExt]) {
        let mut tris_new: Vec<Tri>    = Vec::new();
        let mut exts_new: Vec<TriExt> = Vec::new();

        tris_new.resize(self.size, Tri::new(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)));
        exts_new.resize(self.size, TriExt::default());

        for i in 0..self.size {
            tris_new[i] = tris[self.indices[i] as usize];
            exts_new[i] = tri_exts[self.indices[i] as usize];
        }

        for i in 0..self.size {
            tris[i + self.offset]     = tris_new[i];
            tri_exts[i + self.offset] = exts_new[i];
            self.indices[i] = i as u32;
        }
    }

    fn evaluate_split(&self, tris: &[Tri], node: &BvhNode, axis: usize, split: f32, ) -> f32 {
        let mut left_aabb = Aabb::new();
        let mut right_aabb = Aabb::new();
        let mut left_count = 0.0;
        let mut right_count = 0.0;

        for i in (node.first)..(node.first + node.count) {
            let tri = tris[self.indices[i as usize - self.offset] as usize];
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

    fn find_best_split(&self, tris: &[Tri], node: &BvhNode) -> (usize, f32) {
        let mut best_axis = 0;
        let mut best_split = 0.0;
        let mut best_cost = f32::MAX;

        for axis in 0..3  as usize {
            for idx in (node.first)..(node.first + node.count) {
                let tri = tris[self.indices[idx as usize - self.offset] as usize];
                let split = tri.centroid()[axis as usize];
                let cost = self.evaluate_split(tris, node, axis, split);
                if cost < best_cost {
                    best_axis = axis;
                    best_cost = cost;
                    best_split = split;
                }
            }
        }

        (best_axis, best_split)
    }

    fn find_split_approx(&self, tris: &[Tri], node: &BvhNode,  count: usize) -> (usize, f32) {
        let mut best_axis = 0;
        let mut best_split = 0.0;
        let mut best_cost = f32::MAX;

        for axis in 0..3  as usize {
            for i in 0..count {
                let split = node.aabb.min()[axis] + ((i as f32 + 0.5) / count as f32) * (node.aabb.max()[axis]-node.aabb.min()[axis]);
                let cost = self.evaluate_split(tris, node, axis, split);
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
    fn subdivide(&mut self, node_idx: usize, tris: &[Tri]) {
        let node = self.nodes[node_idx];
     
        if node.count <= 2 {
            return;
        }

        let (axis, split) = if node.count < 64 {
            self.find_best_split(tris, &node) 
        } else {
            self.find_split_approx(tris, &node, 16) 
        };

        let mut i = node.first as usize;
        let mut j = (node.first + node.count - 1) as usize;
        while i <= j {
            let first_idx = self.indices[i - self.offset] as usize;
            
            if tris[first_idx].centroid()[axis] < split {
                i += 1;
            } else {

                self.indices.swap(i - self.offset, j - self.offset);

                if j == 0 {
                    break;
                }

                j -= 1;
                
            }
        };

        let mut left = BvhNode::new();
        left.first = node.first;
        left.count = i  as u32 - node.first;
        left.update_aabb(&self.indices, &tris, self.offset);

        // dont subdivide empty nodes
        if left.count == 0 || left.count == node.count {
            return;
        }

        let mut right = BvhNode::new();
        right.first = i as u32;
        right.count = node.count - left.count;
        right.update_aabb(&self.indices, &tris, self.offset);


        // we no longer hold any triangles
        let children_idx = self.nodes.len();
        self.nodes[node_idx].count = 0;
        self.nodes[node_idx].first = children_idx as u32;

        self.nodes.push(left);
        self.nodes.push(right);

        self.subdivide(children_idx, tris);
        self.subdivide(children_idx + 1, tris);
    }

    pub fn closest_hit(&self, tris: &Vec<Tri>, ro: Vec3, rd: Vec3) -> Option<f32> {
        let mut stack: Vec<u32> = Vec::new();
        stack.push(0);
        let mut best_t = f32::MAX;
        let mut best_i = -1;
        let mut node_count = 0;
        let mut tri_count = 0;
        while !stack.is_empty() {
            node_count += 1;
            let node = self.nodes[stack.pop().unwrap() as usize];

            let aabb_t = node.aabb.closest_hit(ro, rd);
            if aabb_t.is_none() {
                continue;
            }


            if node.count > 0 {
                // leaf node
                tri_count += node.count;
                for i in 0..node.count {
                    if let Some(t) = tris[self.indices[(node.first + i ) as usize - self.offset] as usize].closest_hit(ro, rd) {
                        if t < best_t {
                            best_t = t;
                            best_i = (node.first + i) as i32;
                        }
                    }
                }
            } else {
                // no triangles, internal node - push children onto stack
                stack.push(node.first + 0);
                stack.push(node.first + 1);
            }
        }
        
        if best_i >= 0 {
            Some(best_t)
        } else {
            
            None
        }
    }

    fn closest_hit_unindexed(nodes: &Vec<BvhNode>, root: u32, tris: &Vec<Tri>, ro: Vec3, rd: Vec3) -> Option<f32> {
        let mut stack: Vec<u32> = Vec::new();
        stack.push(root);
        let mut best_t = f32::MAX;
        let mut best_i = -1;
        let mut node_count = 0;
        let mut tri_count = 0;
        while !stack.is_empty() {
            node_count += 1;
            let node = nodes[stack.pop().unwrap() as usize];

            let aabb_t = node.aabb.closest_hit(ro, rd);
            if aabb_t.is_none() {
                continue;
            }


            if node.count > 0 {
                // leaf node
                tri_count += node.count;
                for i in 0..node.count {
                    if let Some(t) = tris[(node.first + i ) as usize].closest_hit(ro, rd) {
                        if t < best_t {
                            best_t = t;
                            best_i = (node.first + i) as i32;
                        }
                    }
                }
            } else {
                // no triangles, internal node - push children onto stack
                stack.push(node.first + 0);
                stack.push(node.first + 1);
            }
        }
        
        if best_i >= 0 {
            Some(best_t)
        } else {
            // println!("Ray Miss! Node checks: {node_count}, Tri checks: {tri_count}");
            None
        }
    }
}
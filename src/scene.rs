use std::{cmp::Ordering, collections::HashMap, f32::consts::PI};

use glam::{uvec2, vec2, vec3, vec4, Mat4, UVec2, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};
use image::GenericImageView;
use rand::random;

use crate::{fetch_bytes, input::*, DEFAULT_ENV_PATH};

#[repr(C)]
#[derive(Default, Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PointLight {
    position: Vec4,
    intensity: Vec4,
}


#[repr(C)]
#[derive(Default, Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuDirectionalLight {
    direction: Vec4,
    intensity: Vec4,
}

type UnitOct32 = u32;

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
    directional_lights: [GpuDirectionalLight; 4],
    pub camera: GpuCamera,
    pub tri_count: u32,
    pub num_point_lights: u32,
    pub num_directional_lights: u32,
    pub tlas_node_count: u32,
}


/// A reference to a span of pixel data on the GPU
/// 
/// Special cased for zero size, in which case offset is a transmuted f32 literal
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

    fn size(&self) -> UVec2 {
        uvec2(self.size >> 16, self.size & 0xFFFF)
    }
}

// TODO: make this dynamic based on loaded mesh
const GPU_TEXCOORD_COUNT: usize = 2;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuVertexExt {
    texcoords:   [Vec2; GPU_TEXCOORD_COUNT],
    normal: u32, // XY components of normalized vector
    tangent: u32,
    color:  u32,
    tangent_sign: f32,
}


#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuTriExt {
    vertices: [GpuVertexExt; 3]
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuVolume {
    absorption:             Vec3,
    ior:                    f32,
}

impl Default for GpuVolume {
    fn default() -> Self {
        Self {
            absorption: Vec3::ZERO,
            ior: 1.5,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMaterial {
    albedo:                 GpuTextureRef,
    emissive:               GpuTextureRef,

    normal:                 GpuTextureRef,
    metallic_roughness:     GpuTextureRef,

    thickness:              GpuTextureRef,
    transmission:           GpuTextureRef,

    albedo_factor:          Vec4,

    emissive_factor:        Vec3,
    normal_scale:           f32,

    albedo_texcoord:        u32,
    emissive_texcoord:      u32,
    normal_texcoord:        u32,
    metal_r_texcoord:       u32,

    thickness_texcoord:     u32,
    transmission_texcoord:  u32,
    thickness_factor:       f32,
    transmission_factor:    f32,
    

    metallic_factor:        f32,
    roughness_factor:       f32,
    id:                     u32,
    alpha_settings:         u32,

    volume:                 GpuVolume,
    
}


impl GpuMaterial {
    
    fn pack_alpha_settings(alpha_mode: gltf::material::AlphaMode, cutoff: f32) -> u32 {
        let mode = match alpha_mode {
            gltf::material::AlphaMode::Opaque => 0,
            gltf::material::AlphaMode::Mask => 1,
            gltf::material::AlphaMode::Blend => 2,
        };
        mode | (((cutoff.clamp(0.0, 1.0) * u16::MAX as f32) as u32) << 16)
    }
}


#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuPrimitive {
    transform:      Mat4,
    inv_transform:  Mat4,
    material:       GpuMaterial,
    bvh_idx:        u32,
    tri_start:      u32,
    tri_count:      u32,
    /// [ 31x unused | has_tangents ]
    flags:          u32,
}

impl GpuPrimitive {
    const TANGENT_FLAG: u32 = 1;

    fn new(transform: &Mat4, material: GpuMaterial, bvh_idx: u32, tri_start: u32, tri_count: u32, has_tangents: bool) -> Self {
        Self {
            transform: *transform,
            inv_transform: transform.as_dmat4().inverse().as_mat4(),
            material,
            bvh_idx,
            tri_start,
            tri_count,
            flags: if has_tangents {Self::TANGENT_FLAG} else {0},

        }
    }
}

fn oct_wrap(v: Vec2) -> Vec2 {
    let scale = v.signum();
    vec2 (
        1.0 - v.y.abs(),
        1.0 - v.x.abs()
    ) * scale
    
}

// https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/
fn pack_vec3_octahedral(v: Vec3) -> Vec2 {
    let mut o = v.xy();
    o /= v.x.abs() + v.y.abs() + v.z.abs();
    if v.z < 0.0 {
        o = oct_wrap(o);
    }
    return o * 0.5 + 0.5
}


fn pack_unit_oct32(v: Vec3) -> UnitOct32 {
    let o = pack_vec3_octahedral(v);
    (((o.x * u16::MAX as f32) as u32) << 16) | (o.y * u16::MAX as f32) as u32
}

pub struct EnvironmentMap {
    /// The width of the texture. Must be 2x height
    pub width: usize,

    /// The height of the texture. Must be half the width
    pub height: usize,

    /// The raw luminance values
    /// 
    /// A 2D array of width x height
    pub data: Vec<[f32; 4]>,

    /// The probability of sampling each pixel
    /// 
    /// luminance * sin(theta), same dimensions as data
    pub pdf: Vec<f32>,

    /// The conditional cumulative probability of sampling each pixel in its row
    /// 
    /// same dimensions as data and the pdf
    pub cdf_col: Vec<f32>,

    /// The marginal cumulative probability of sampling each row of the texture
    /// 
    /// This is a 1D array the length of the height of the texture
    pub cdf_rows: Vec<f32>
}

impl EnvironmentMap {
    pub async fn from_hdr_path(path: &std::path::Path) -> Option<Self> {
        let buffer = fetch_bytes(path.as_os_str().to_str()?).await?;
        let mut data = Vec::new();
        let image = image::load_from_memory(buffer.as_slice()).expect(format!("Expected valid image at path {:?}", path).as_str());
        let image = image.into_rgba32f();
        for pixel in image.pixels() {
            data.push(pixel.0);
        };
        
        let height = image.height() as usize;
        let width = image.width() as usize;

        if width != height * 2 {
            return None;
        }

        Some(Self::new(data, width, height))
    }

    // TODO: replace with analytic integral if this helps
    fn sin_theta(y: usize, height: usize) -> f64 {
        let y = y as f64;
        let h = height as f64;
        let mut sin_theta = 0.0;
        const N: usize = 10;
        for i in 0..N + 1 {
            sin_theta += ((y + i as f64 / N as f64) / h  * core::f64::consts::PI).sin().max(1e-12);
        }
        sin_theta / (N as f64 + 1.0)
    }


    pub fn new(data: Vec<[f32; 4]>, width: usize, height: usize) -> Self {
        let mut pdf = Vec::new();
        let mut pdf_total = 0.0;
        for y in 0..height {
            let sin_theta = Self::sin_theta(y, height);
            for x in 0..width {
                let v = Self::luminance(data[y * width + x]) * sin_theta as f32;

                pdf.push(v);
                pdf_total += v;
            }
        }

        for v in &mut pdf {
            *v /= pdf_total;
        }
        

        let mut cdf_col = Vec::new();

        let mut cdf_rows = Vec::new();
        let mut cdf_rows_total = 0.0;

        for y in 0..height {
            let mut row_prob_sum = 0.0;
            for x in 0..width {
                row_prob_sum += pdf[y * width + x];
                cdf_col.push(row_prob_sum);
            }

            // at this point row_prob_sum is the total pdf within row y, and the maximum value of cdf_col
            if row_prob_sum != 0.0 {
                for x in 0..width {
                    cdf_col[y * width + x] /= row_prob_sum;
                }
            } else {
                // values really should not matter here since it should never get picked
                for x in 0..width {
                    cdf_col[y * width + x] = (x as f32 + 1.0) / width as f32;
                }
            }

            
            cdf_rows_total += row_prob_sum;
            cdf_rows.push(cdf_rows_total);
        }

        for row in 0..height {
            cdf_rows[row] /= cdf_rows_total;
        }

        // ensure a valid cdf in the face of floating point error
        cdf_rows[height - 1] = 1.0;

        // pdf now represents the continuous solid angle PDF of a direction within that pixel
        let mut total_luminance = 0.0;
        pdf.clear();
        for y in 0..height {
            let sin_theta = Self::sin_theta(y, height);
            let solid_angle = (2.0 * core::f64::consts::PI * core::f64::consts::PI / (width * height) as f64) * sin_theta;
            for x in 0..width {
                let v = Self::luminance(data[y * width + x]) as f64;
                pdf.push(v as f32);
                total_luminance += v * solid_angle;
            }
        }
        for v in &mut pdf {
            *v /= total_luminance as f32;
        }



        EnvironmentMap {
            width,
            height,
            data,
            pdf,
            cdf_col,
            cdf_rows
        }
    }

    /// perceived luminance of an rgb pixel. Ignores alpha
    fn luminance(v: [f32; 4]) -> f32 {
        // Rec. 709 linear luminance coefficients
        v[0] * 0.2126 + v[1] * 0.7152 + v[2] * 0.0722
    }

    
    /// Returns the index of the chosen pixel alongside its pdf
    pub fn sample(&self, u: (f32, f32)) -> (UVec2, f32) {

        let row = self.cdf_rows
            .binary_search_by(|p| p.partial_cmp(&u.0).unwrap_or(Ordering::Greater))
            .unwrap_or_else(|i| i)
            .min(self.height - 1);

        let col = self.cdf_col[row * self.width..(row + 1) * self.width]
            .binary_search_by(|p| p.partial_cmp(&u.1).unwrap_or(Ordering::Greater))
            .unwrap_or_else(|i| i)
            .min(self.width - 1);

        (uvec2(col as u32, row as u32), self.pdf[row * self.width + col])
    }

    pub fn test_distribution(&self) -> Vec<[f32; 4]> {

        let mut samples = Vec::with_capacity(self.width * self.height);
        for _ in 0..self.width * self.height {
            samples.push([0.0, 0.0, 0.0, 1.0]);
        }
        let n = 50.0;
        for _ in 0..(self.width * self.height * n as usize) {
            let u = (random(), random());
            let (p, pdf) = self.sample(u);
            let sin_theta = ( (p.y as f32 + 0.5) * PI / self.height as f32 ).sin().max(1e-6);
            let pdf = pdf / ( ( (2.0 * PI * PI) / (self.width * self.height) as f32 ) * sin_theta);
            let s = &mut samples[p.y as usize * self.width + p.x as usize];

            s[0] += (1.0 / pdf) / n;
            s[1] += (1.0 / pdf) / n;
            s[2] += (1.0 / pdf) / n;
        }

        samples
    }
}

impl Default for EnvironmentMap {
    fn default() -> Self {
        let data = vec![[1.0, 0.7, 0.3, 1.0], [0.4, 0.4, 0.1, 1.0]];
        Self::new(data, 2, 1)
    }
}



#[derive(Default)]
pub struct RenderScene {
    // path to the environment map texture
    pub env_map_path:       std::path::PathBuf,

    // path to the GLTF file
    pub gltf_path:          Option<std::path::PathBuf>,

    /// flat array of primitives that share a material
    pub primitives:         Vec<GpuPrimitive>,

    /// bvh over RenderScene::primitives
    pub tlas_node_data: Vec<BvhNode>,

    /// global buffer of triangle position data
    pub tris:               Vec<Tri>,
    pub tri_exts:           Vec<GpuTriExt>,

    /// global buffer of rgba8 texture data
    pub texture_data:       Vec<u32>,
    pub texture_map:        HashMap<usize, GpuTextureRef>,

    pub bvh_node_data:      Vec<BvhNode>,

    

    /// cameras in scene
    pub cameras:            Vec<Camera>,

    pub point_lights:       Vec<PointLight>,
    pub directional_lights: Vec<GpuDirectionalLight>,

    /// rgba32f equirectangular environment map pixel data
    pub env_map:            EnvironmentMap,
}

// https://gamedev.stackexchange.com/questions/169508/octahedral-impostors-octahedral-mapping
fn unpack_unit_octahedral(mut f: Vec2) -> Vec3 {
    f = f * 2.0 - 1.0;
 
    // https://twitter.com/Stubbesaurus/status/937994790553227264
    let mut n = vec3(f.x, f.y, 1.0 - f.x.abs() - f.y.abs());
    let t = (-n.z).clamp(0.0, 1.0);

    n.x -= t * n.x.signum();
    n.y -= t * n.y.signum();
    // n.xy += n.xy >= 0.0 ? -t : t;
    n.normalize()
}

fn unpack_unit_oct32(u_in: u32) -> Vec3 {
    let f = vec2((u_in >> 16) as f32, (u_in & 0xFFFF) as f32) / (0xFFFF as f32);
    unpack_unit_octahedral(f)
}

pub struct PrimitiveGeometry<'a> {
    scene: &'a mut RenderScene,
    primitive_idx: usize,
}

impl<'a> PrimitiveGeometry<'a> {
    pub fn new(scene: &'a mut RenderScene, primitive_idx: usize) -> Self {
        let prim = scene.primitives[primitive_idx];

        Self {
            scene,
            primitive_idx,
        }
    }

    /// convert a local idx into a global idx for RenderScene::tris and friends
    pub fn idx(&self, idx: usize) -> usize {
        self.scene.primitives[self.primitive_idx].tri_start as usize + idx
    }
}


impl<'a> mikktspace::Geometry for PrimitiveGeometry<'a> {
    fn num_faces(&self) -> usize {
        self.scene.primitives[self.primitive_idx].tri_count as usize
    }

    fn num_vertices_of_face(&self, face: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.scene.tris[self.idx(face)].vertices[vert].xyz().to_array()
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        unpack_unit_oct32(self.scene.tri_exts[self.idx(face)].vertices[vert].normal).to_array()
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        let mut tc = self.scene.tri_exts[self.idx(face)].vertices[vert].texcoords[0].to_array();

        // https://github.com/KhronosGroup/glTF-Sample-Models/issues/174
        // https://github.com/KhronosGroup/glTF/issues/2056
        // GLTF (and me i guess) use a texture coordinate system where (0, 0) is the bottom left
        // mikktspace and blender use (0, 0) as the upper left
        // this is the same as flipping the sign of the bitangent for generated meshes, which
        // is the fix most other implementations appear to use
        tc[1] = 1.0 - tc[1];
        tc

    }

    fn set_tangent_encoded(&mut self, tangent: [f32; 4], face: usize, vert: usize) {
        let idx = self.idx(face);
        // TODO: HACK: track which primitives have tangents and generate them on a per-primitive level
        if (self.scene.primitives[self.primitive_idx].flags & GpuPrimitive::TANGENT_FLAG) == 0 {
            self.scene.tri_exts[idx].vertices[vert].tangent = pack_unit_oct32(Vec3::from_array(*tangent.first_chunk().unwrap()));
            self.scene.tri_exts[idx].vertices[vert].tangent_sign = tangent[3];
        }
        // self.scene.primitives[self.primitive_idx].flags |= GpuPrimitive::TANGENT_FLAG;
    }
}


type LoadedMeshCache = HashMap<usize, HashMap<usize, usize>>;

impl RenderScene {

    pub fn add_gltf(&mut self, transform: &Mat4, path: &std::path::Path) -> bool {
        let mut cache = LoadedMeshCache::new();
        
        let (document, buffers, _) = match gltf::import(path) {
            Ok(r) => r,
            Err(_) =>{println!("Failed to import gltf"); return false},
        };

        self.gltf_path = Some(path.to_path_buf());

        let mut ms = MatrixStack::new();
        ms.push();
        ms.apply(&transform);
        if let Some(scene) = document.default_scene() {
            for node in scene.nodes() {
                self.add_gltf_node(&buffers, node, &mut ms, &mut cache);
            }
        } else if let Some(scene) = document.scenes().next() {
            for node in scene.nodes() {
                self.add_gltf_node(&buffers, node, &mut ms, &mut cache);
            }
        }

        self.build_tlas();
        
        true
    }  

    pub fn add_gltf_bytes(&mut self, transform: &Mat4, bytes: &[u8]) -> bool {
        let mut cache = LoadedMeshCache::new();

        let (document, buffers, _) = match gltf::import_slice(bytes) {
            Ok(r) => r,
            Err(_) =>{println!("Failed to import gltf bytes"); return false},
        };



        let mut ms = MatrixStack::new();
        ms.push();
        ms.apply(&transform);
        for scene in document.scenes(){
            for node in scene.nodes() {
                self.add_gltf_node(&buffers, node, &mut ms, &mut cache);
            }
        }

        self.build_tlas();
        
        true
    }   

    
    pub fn from_gltf_vec3(v: Vec3) -> Vec3 {
        // GLTF coordinate system: "glTF defines +Y as up, +Z as forward, and -X as right; the front of a glTF asset faces +Z."

        // from (left, up, forward)
        // to   (forward, left, up)
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

    fn add_gltf_texture(&mut self, tex: &gltf::texture::Texture, buffers: &Vec<gltf::buffer::Data>) -> GpuTextureRef {

        let gltf_dir = self.gltf_path
            .clone()
            .map(|x| 
                x.parent()
                .map(|x| 
                    x.to_path_buf()
                )
            )
            .flatten();

        // if we have not already loaded the image
        if !self.texture_map.contains_key(&tex.index()) {
            // load the image
            let image = match tex.source().source() {
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
                    let path = gltf_dir.map(|x| x.join(uri)).unwrap_or(std::path::Path::new(uri).to_path_buf());

                    image::ImageReader::open(path).unwrap().decode().unwrap()
                    
                    
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
            self.texture_map.insert(tex.index(), tex_ref);
            tex_ref

        } else {
            *self.texture_map.get(&tex.index()).unwrap()
        }
    }
    
    fn add_gltf_node(&mut self, 
        buffers: &Vec<gltf::buffer::Data>, 
        node: gltf::Node, 
        ms: &mut MatrixStack, 
        cache: &mut LoadedMeshCache) 
    {
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
                    let d = GpuDirectionalLight { 
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

            if let Some(loaded_primitives) = cache.get(&mesh.index()) {
                for primitive in mesh.primitives() {

                    // if the primitive was already loaded, copy it and change the transforms
                    if let Some(&prim_idx) = loaded_primitives.get(&primitive.index()) {
                        let mut new_primitive = self.primitives[prim_idx];
                        new_primitive.transform = node_transform_mine;
                        new_primitive.inv_transform = node_transform_mine.as_dmat4().inverse().as_mat4();
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
                        new_primitive.inv_transform = node_transform_mine.as_dmat4().inverse().as_mat4();

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
                        

                        let mut found_texcoords: HashMap<u32, Vec<Vec2>> = HashMap::new();

                        // if the primitive has a texcoord attribute with this id, load it into a vec
                        // if we already loaded the texcoords of that ID, we don't need to do anything 
                        let mut try_load_texcoords = |id| {
                            if found_texcoords.contains_key(id) {
                                return;
                            }

                            let texcoords = reader.read_tex_coords(id.clone());
                            let texcoords = if texcoords.is_some() {
                                texcoords.unwrap().into_f32().map(|uv| Vec2::from_slice(&uv)).collect()
                            } else {
                                Vec::new()
                            };
                            if !texcoords.is_empty() {
                                found_texcoords.insert(id.clone(), texcoords);
                            }
                            
                        };

                        let mut material = GpuMaterial::default();

                        material.alpha_settings = GpuMaterial::pack_alpha_settings(
                            primitive.material().alpha_mode(), 
                            primitive.material().alpha_cutoff().unwrap_or(0.5)
                        );

                        material.albedo_factor = primitive.material().pbr_metallic_roughness().base_color_factor().into();
                        if let Some(albedo_tex) = primitive.material().pbr_metallic_roughness().base_color_texture() {
                            println!("Found base_color_texture");
                            material.albedo = self.add_gltf_texture(&albedo_tex.texture(), buffers);
                            material.albedo_texcoord = albedo_tex.tex_coord();

                            try_load_texcoords(&material.albedo_texcoord);
                        }
                        
                        material.metallic_factor = primitive.material().pbr_metallic_roughness().metallic_factor();
                        material.roughness_factor = primitive.material().pbr_metallic_roughness().roughness_factor();
                        if let Some(metal_r_tex) = primitive.material().pbr_metallic_roughness().metallic_roughness_texture() {
                            println!("Found metallic_roughness_texture");
                            material.metallic_roughness = self.add_gltf_texture(&metal_r_tex.texture(), buffers);
                            material.metal_r_texcoord = metal_r_tex.tex_coord();

                            try_load_texcoords(&material.metal_r_texcoord);
                        }

                        // combine factor and strength into one float, which is how its used anyway
                        material.emissive_factor = primitive.material().emissive_factor().into();
                        material.emissive_factor *= primitive.material().emissive_strength().unwrap_or(1.0);
                        if let Some(emissive_tex) = primitive.material().emissive_texture() {
                            println!("Found emissive_texture with factor {}, {}, {}", material.emissive_factor.x, material.emissive_factor.y, material.emissive_factor.z);
                            material.emissive = self.add_gltf_texture(&emissive_tex.texture(), buffers);
                            material.emissive_texcoord = emissive_tex.tex_coord();
                            
                            try_load_texcoords(&material.emissive_texcoord);
                        }
                        
                        material.normal_scale = primitive.material().normal_texture().map(|t| t.scale()).unwrap_or(1.0);
                        if let Some(normal_tex) = primitive.material().normal_texture() {
                            println!("Found normal_texture");
                            material.normal = self.add_gltf_texture(&normal_tex.texture(), buffers);
                            material.normal_texcoord = normal_tex.tex_coord();

                            try_load_texcoords(&material.normal_texcoord);
                        }

                        material.volume.ior = primitive.material().ior().unwrap_or(1.5);

                        if let Some(volume) = primitive.material().volume() {
                            material.thickness_factor = volume.thickness_factor();
                            
                            material.volume.absorption = -Vec3::from_array(volume.attenuation_color().map(|v| v.ln()))
                                 / volume.attenuation_distance();
                            
                            if let Some(thickness_tex) = volume.thickness_texture() {
                                println!("Found thickness_texture");
                                material.thickness = self.add_gltf_texture(&thickness_tex.texture(), buffers);
                                material.thickness_texcoord = thickness_tex.tex_coord();
                                
                                try_load_texcoords(&material.thickness_texcoord);
                            }

                        }



                        if let Some(transmission) = primitive.material().transmission() {
                            material.transmission_factor = transmission.transmission_factor();
                            
                            if let Some(transmission_tex) = transmission.transmission_texture() {
                                println!("Found transmission_texture");
                                material.transmission = self.add_gltf_texture(&transmission_tex.texture(), buffers);
                                material.transmission_texcoord = transmission_tex.tex_coord();
                                try_load_texcoords(&material.transmission_texcoord);
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

                        let normals = reader.read_normals();
                        let normals: Vec<UnitOct32> = if normals.is_some() {
                            normals.unwrap().map(
                                |c| 
                                pack_unit_oct32(Self::from_gltf_vec3(vec3(c[0], c[1], c[2]).normalize()))
                            ).collect()
                        } else {
                            Vec::new()
                        };


                        let tangents = reader.read_tangents();
                        let tangents: Vec<(UnitOct32, f32)> = if tangents.is_some() {
                            tangents.unwrap().map(
                                |c| 
                                (pack_unit_oct32(Self::from_gltf_vec3(vec3(c[0], c[1], c[2]).normalize())), c[3])
                            ).collect()
                        } else {
                            Vec::new()
                        };

                        let has_tangents = !tangents.is_empty();

                        let first_new_tri = self.tris.len();
                        if let Some(indices) = reader.read_indices() {
                            // indexed mesh
                            let mut indices = indices.into_u32();
                            while let (Some(idx_0), Some(idx_1), Some(idx_2)) = (indices.next(), indices.next(), indices.next()) {
                                let mut ext = GpuTriExt::default();

                                if !colors.is_empty() {
                                    ext.vertices[0].color = colors[idx_0 as usize];
                                    ext.vertices[1].color = colors[idx_1 as usize];
                                    ext.vertices[2].color = colors[idx_2 as usize];
                                }

                                if !normals.is_empty() {
                                    ext.vertices[0].normal = normals[idx_0 as usize];
                                    ext.vertices[1].normal = normals[idx_1 as usize];
                                    ext.vertices[2].normal = normals[idx_2 as usize];
                                }

                                if !tangents.is_empty() {
                                    ext.vertices[0].tangent = tangents[idx_0 as usize].0;
                                    ext.vertices[1].tangent = tangents[idx_1 as usize].0;
                                    ext.vertices[2].tangent = tangents[idx_2 as usize].0;

                                    ext.vertices[0].tangent_sign = tangents[idx_0 as usize].1;
                                    ext.vertices[1].tangent_sign = tangents[idx_1 as usize].1;
                                    ext.vertices[2].tangent_sign = tangents[idx_2 as usize].1;
                                }

                                for i in 0..GPU_TEXCOORD_COUNT {
                                    if let Some(tc) = found_texcoords.get(&(i as u32)) {
                                        ext.vertices[0].texcoords[i] = tc[idx_0 as usize];
                                        ext.vertices[1].texcoords[i] = tc[idx_1 as usize];
                                        ext.vertices[2].texcoords[i] = tc[idx_2 as usize];
                                    }
                                }

                                if found_texcoords.get(&(GPU_TEXCOORD_COUNT as u32)).is_some() {
                                    eprintln!("Model has more texcoords than supported by GPU renderer ({GPU_TEXCOORD_COUNT})");
                                }

                                self.tris.push(Tri::new(positions[idx_0 as usize], positions[idx_1 as usize], positions[idx_2 as usize]));
                                self.tri_exts.push(ext);
                            }
                        }
                        else {
                            panic!("Only supporting indexed meshes for now");
                        }
                        if node_transform_mine.to_scale_rotation_translation().2.length() > 100.0 {
                            println!("Warning: distant geometry is poorly supported ({} units from origin)", node_transform_mine.to_scale_rotation_translation().2.length());
                        }

                        // build a bvh around the new triangles
                        let mut bvh = Bvh::new(&self.tris.as_slice(), first_new_tri, self.tris.len() - first_new_tri);
                        // re-arrange the new triangles to match the BVH nodes
                        bvh.flatten_leaves(self.tris.as_mut_slice(), Some(self.tri_exts.as_mut_slice()));
                        let bvh_root = self.bvh_node_data.len() as u32;
                        self.bvh_node_data.append(&mut bvh.nodes);

                        for node in &mut self.bvh_node_data[bvh_root as usize .. ] {
                            if node.count == 0 {
                                // inner node
                                node.first += bvh_root;
                            }
                        }

                        let gpu_primitive = GpuPrimitive::new(
                            &node_transform_mine, 
                            material, 
                            bvh_root, 
                            first_new_tri as u32, 
                            (self.tris.len() - first_new_tri) as u32, 
                            has_tangents
                        );

                        // add this primitive to the scene
                        self.primitives.push(gpu_primitive);

                        if !has_tangents {
                            mikktspace::generate_tangents(&mut PrimitiveGeometry::new(self, self.primitives.len() - 1));
                        }
                        
                        println!("Adding primitive with {} triangles, bvh root at index {}", self.tris.len() - first_new_tri, bvh_root);

                        // mark this primitive as already loaded
                        loaded_primitives.insert(
                            primitive.index(),
                            self.primitives.len() - 1
                        );

                    } else {
                        println!("Warning: Non-triangle primitives not supported");
                    }
                }

                // mark this mesh index as already loaded, keeping a reference to the loaded primitives
                cache.insert(
                    mesh.index(),
                    loaded_primitives
                );
            }
        }

        for child in node.children() {
            self.add_gltf_node(buffers, child, ms, cache);
        }

        ms.pop();
    }

    /// loads the pixel data from a given equirectangular environment map into the scene
    /// returns true if the file was found, false otherwise
    /// 
    /// panics if the file path is valid but not an image
    pub async fn set_equirectangular_env_map(&mut self, path: &std::path::Path) -> bool {
        if let Some(e) = EnvironmentMap::from_hdr_path(path).await {
            self.env_map = e;
            self.env_map_path = path.to_owned();
            true
        } else {
            println!("Failed to make an EnvironmentMap from path \"{:?}\"", path);
            false
        }
    }

    pub fn build_tlas(&mut self) {
        let mut primitive_leaves: Vec<PrimitiveLeaf> = self.primitives.iter()
            .map(|p| PrimitiveLeaf::new(*p, &self.bvh_node_data, &self.tris))
            .collect();
        let mut tlas = Bvh::new(primitive_leaves.as_slice(), 0, primitive_leaves.len());

        // this is a little ridiculous since there is no actual data, but _ isn't copy.
        tlas.flatten_leaves::<_, u32>(&mut primitive_leaves, None);
        self.primitives.clear();
        for leaf in primitive_leaves {
            self.primitives.push(leaf.primitive)
        }
        println!("TLAS: {} nodes over {} primitives", tlas.nodes.len(), self.primitives.len());

        self.tlas_node_data = tlas.nodes;
    }

    pub fn to_gpu(&self) -> GpuSceneUniform {
        let mut point_lights = [PointLight::default(); 12];
        let mut directional_lights = [GpuDirectionalLight::default(); 4];

        for i in 0..self.point_lights.len().min(point_lights.len()) {
            point_lights[i] = self.point_lights[i];
        }

        for i in 0..self.directional_lights.len().min(directional_lights.len()) {
            directional_lights[i] = self.directional_lights[i];
        }

        GpuSceneUniform {
            tlas_node_count: self.tlas_node_data.len() as u32,
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

    /// Raycast the scene from the camera's center to update its focal length
    /// 
    /// # Examples
    /// ```
    /// if scene.focus_camera(0) {
    ///     // raycast hit - focal length updated
    /// } else {
    ///     // raycast missed - focal length unchanged
    /// }
    /// ```
    pub fn focus_camera(&mut self, camera_id: usize) -> bool {
        if let Some(focus) = self.closest_hit(self.cameras[camera_id].position(), self.cameras[camera_id].forward()) {
            self.cameras[camera_id].focus(focus);
            true
        } else {
            false
        }
    }

    pub async fn from_path(mesh_path: &std::path::Path, env_map_path: &std::path::Path) -> Option<RenderScene> {
        println!("building scene");
        let mut scene = RenderScene::default();
    
        if !scene.add_gltf(&Mat4::IDENTITY, mesh_path) {
            
            return None;
        }
    
        scene.set_equirectangular_env_map(env_map_path).await;
    
        if scene.cameras.is_empty() {
            println!("No camera in scene, falling back to default");
            // vec3f(-3.5, -0.5, 0.5), vec3f(1.0, 0.0, 0.0)
            scene.cameras.push(Camera::default());
        }
        
        println!("Tri count: {}", scene.tris.len());
        println!("Tri size : {} + {} mb", (scene.tris.len() * size_of::<Tri>()) / (1024 * 1024), (scene.tris.len() * size_of::<GpuTriExt>()) / (1024 * 1024));
        println!("Texture data size : {} mb", (scene.texture_data.len() * size_of::<u32>()) / (1024 * 1024));

        println!("Focused camera: {}", scene.focus_camera(0));
        
        Some(scene)
    }
    
    pub async fn from_bytes(mesh_bytes: &[u8], env_map_path: &std::path::Path) -> Option<RenderScene> {
        println!("building scene");
        let mut scene = RenderScene::default();
    
        if !scene.add_gltf_bytes(&Mat4::IDENTITY, mesh_bytes) {
            
            return None;
        }
    
        scene.set_equirectangular_env_map(env_map_path).await;
    
        if scene.cameras.is_empty() {
            println!("No camera in scene, falling back to default");
            // vec3f(-3.5, -0.5, 0.5), vec3f(1.0, 0.0, 0.0)
            scene.cameras.push(Camera::default());
        }
        
        println!("Tri count: {}", scene.tris.len());
        println!("Tri size : {} + {} mb", (scene.tris.len() * size_of::<Tri>()) / (1024 * 1024), (scene.tris.len() * size_of::<GpuTriExt>()) / (1024 * 1024));
        println!("Texture data size : {} mb", (scene.texture_data.len() * size_of::<u32>()) / (1024 * 1024));

        println!("Focused camera: {}", scene.focus_camera(0));
        
        Some(scene)
    }



    
}


#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable, Default)]
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

    pub fn min_max(min: Vec3, max: Vec3) -> Self {
        Self {
            data: [min.x, min.y, min.z, max.x, max.y, max.z]
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
        const EPS: f32 = 0.000001;//0.00001;

        Self {
            data: [point.x - EPS, point.y - EPS, point.z - EPS, point.x + EPS, point.y + EPS, point.z + EPS]
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

impl Default for Aabb {
    fn default() -> Self {
        Self::new()
    }
}

pub trait BvhLeaf : Default + Copy {
    fn aabb(&self) -> Aabb;
    fn closest_hit(&self, ro: Vec3, rd: Vec3) -> Option<f32>;
    fn centroid(&self) -> Vec3 {
        let aabb = self.aabb();
        (aabb.min() + aabb.max()) / 2.0
    }
}

impl BvhLeaf for Tri {
    fn aabb(&self) -> Aabb { self.aabb() }
    
    fn closest_hit(&self, ro: Vec3, rd: Vec3) -> Option<f32> {
        self.closest_hit(ro, rd)
    }

    fn centroid(&self) -> Vec3 {
        self.centroid()
    }
}

#[derive(Clone, Copy, Default)]
struct PrimitiveLeaf {
    primitive: GpuPrimitive,
    aabb: Aabb,
}

fn aabb_over_bvh_node(bvh: &Vec<BvhNode>, tris: &Vec<Tri>, transform: &Mat4, idx: usize) -> Aabb {
    if bvh[idx].count == 0 {
        // interior node
        aabb_over_bvh_node(bvh, tris, transform, bvh[idx].first as usize)
            .with(aabb_over_bvh_node(bvh, tris, transform, bvh[idx].first as usize + 1))
    } else {
        let mut aabb = Aabb::new();
        for i in 0..bvh[idx].count {
            let t = bvh[idx].first as usize + i as usize;
            aabb.expand(Aabb::point(transform.transform_point3(tris[t].vertices[0].xyz())));
            aabb.expand(Aabb::point(transform.transform_point3(tris[t].vertices[1].xyz())));
            aabb.expand(Aabb::point(transform.transform_point3(tris[t].vertices[2].xyz())));
        }
        aabb
    }
}

impl PrimitiveLeaf {
    pub fn new(primitive: GpuPrimitive, bvh: &Vec<BvhNode>, tris: &Vec<Tri>) -> Self {
        let local_aabb = bvh[primitive.bvh_idx as usize].aabb;
        let transform = primitive.transform;
        let aabb = aabb_over_bvh_node(&bvh, &tris, &transform, primitive.bvh_idx as usize);

        // let mut aabb = Aabb::min_max(
        //     transform.transform_point3(min),
        //     transform.transform_point3(max)
        // );
        // aabb.expand(Aabb::point(transform.transform_point3(vec3(min.x, min.y, max.z))));
        // aabb.expand(Aabb::point(transform.transform_point3(vec3(min.x, max.y, min.z))));
        // aabb.expand(Aabb::point(transform.transform_point3(vec3(min.x, max.y, max.z))));
        // aabb.expand(Aabb::point(transform.transform_point3(vec3(max.x, min.y, min.z))));
        // aabb.expand(Aabb::point(transform.transform_point3(vec3(max.x, min.y, max.z))));
        // aabb.expand(Aabb::point(transform.transform_point3(vec3(max.x, max.y, min.z))));

        Self {
            primitive,
            aabb,
        }
    }
}

impl BvhLeaf for PrimitiveLeaf {
    fn aabb(&self) -> Aabb {
        self.aabb
    }

    fn closest_hit(&self, _ro: Vec3, _rd: Vec3) -> Option<f32> {
        unimplemented!()
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

    fn from_leaves<Leaf: BvhLeaf>(first: u32, count: u32, indices: &Vec<u32>, leaves: &[Leaf], offset: usize) -> Self {
        let mut new = Self::new();
        new.first = first;
        new.count = count;
        new.update_aabb(indices, leaves, offset);
        new
    }

    fn update_aabb<Leaf: BvhLeaf>(&mut self, indices: &Vec<u32>, leaves: &[Leaf], offset: usize) {
        if self.count != 0 {
            self.aabb = leaves[indices[self.first as usize - offset] as usize].aabb();
            for i in self.first..self.first + self.count {
                self.aabb.expand(leaves[indices[i as usize - offset] as usize].aabb());
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
    pub fn new<Leaf: BvhLeaf>(leaves: &[Leaf], offset: usize, size: usize) -> Self {
        let mut res = Self {
            nodes: Vec::new(),
            indices: ( (offset  as u32) .. (offset + size) as u32 ).collect(),
            offset,
            size
        };

        res.nodes.push(BvhNode::from_leaves(offset as u32, size as u32, &res.indices, &leaves, offset));
        res.subdivide(res.nodes.len() - 1, leaves);
        return res;
    }

    /// remove the layer of indirection used to build the BVH
    pub fn flatten_leaves<Leaf: BvhLeaf, Ext: Copy + Default>(&mut self, leaves: &mut [Leaf], exts: Option<&mut [Ext]>) {
        let mut leaves_new: Vec<Leaf>    = Vec::new();
        leaves_new.resize(self.size, Leaf::default());
        

        if let Some(exts) = exts {
            let mut exts_new: Vec<Ext> = Vec::new();
            exts_new.resize(self.size, Ext::default());
            for i in 0..self.size {
                leaves_new[i] = leaves[self.indices[i] as usize];
                exts_new[i] = exts[self.indices[i] as usize];
            }

            for i in 0..self.size {
                leaves[i + self.offset] = leaves_new[i];
                exts[i + self.offset] = exts_new[i];
                self.indices[i] = i as u32;
            }
        } else {
            for i in 0..self.size {
                leaves_new[i] = leaves[self.indices[i] as usize];
            }

            for i in 0..self.size {
                leaves[i + self.offset] = leaves_new[i];
                self.indices[i] = i as u32;
            }
        }

    }

    fn evaluate_split<Leaf: BvhLeaf>(&self, leaves: &[Leaf], node: &BvhNode, axis: usize, split: f32, ) -> f32 {
        let mut left_aabb = Aabb::new();
        let mut right_aabb = Aabb::new();
        let mut left_count = 0.0;
        let mut right_count = 0.0;

        for i in (node.first)..(node.first + node.count) {
            let leaf = leaves[self.indices[i as usize - self.offset] as usize];
            if leaf.centroid()[axis] < split {
                left_count += 1.0;
                left_aabb.expand(leaf.aabb());
            } else {
                right_count += 1.0;
                right_aabb.expand(leaf.aabb());
            }

        }

        let cost = left_count * left_aabb.surface() + right_count * right_aabb.surface();

        if cost > 0.0 {
            cost
        } else {
            f32::MAX
        }
    }

    fn find_best_split<Leaf: BvhLeaf>(&self, leaves: &[Leaf], node: &BvhNode) -> (usize, f32) {
        let mut best_axis = 0;
        let mut best_split = 0.0;
        let mut best_cost = f32::MAX;

        for axis in 0..3  as usize {
            for idx in (node.first)..(node.first + node.count) {
                let leaf = leaves[self.indices[idx as usize - self.offset] as usize];
                let split = leaf.centroid()[axis as usize];
                let cost = self.evaluate_split(leaves, node, axis, split);
                if cost < best_cost {
                    best_axis = axis;
                    best_cost = cost;
                    best_split = split;
                }
            }
        }

        (best_axis, best_split)
    }

    fn find_split_approx<Leaf: BvhLeaf>(&self, leaves: &[Leaf], node: &BvhNode,  count: usize) -> (usize, f32) {
        let mut best_axis = 0;
        let mut best_split = 0.0;
        let mut best_cost = f32::MAX;

        for axis in 0..3  as usize {
            for i in 0..count {
                let split = node.aabb.min()[axis] + ((i as f32 + 0.5) / count as f32) * (node.aabb.max()[axis]-node.aabb.min()[axis]);
                let cost = self.evaluate_split(leaves, node, axis, split);
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
    fn subdivide<Leaf: BvhLeaf>(&mut self, node_idx: usize, leaves: &[Leaf]) {
        let node = self.nodes[node_idx];
     
        if node.count <= 2 {
            return;
        }

        let (axis, split) = if node.count < 64 {
            self.find_best_split(leaves, &node) 
        } else {
            self.find_split_approx(leaves, &node, 16) 
        };

        let mut i = node.first as usize;
        let mut j = (node.first + node.count - 1) as usize;
        while i <= j {
            let first_idx = self.indices[i - self.offset] as usize;
            
            if leaves[first_idx].centroid()[axis] < split {
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
        left.update_aabb(&self.indices, &leaves, self.offset);

        // dont subdivide empty nodes
        if left.count == 0 || left.count == node.count {
            return;
        }

        let mut right = BvhNode::new();
        right.first = i as u32;
        right.count = node.count - left.count;
        right.update_aabb(&self.indices, &leaves, self.offset);


        // we no longer hold any triangles
        let children_idx = self.nodes.len();
        self.nodes[node_idx].count = 0;
        self.nodes[node_idx].first = children_idx as u32;

        self.nodes.push(left);
        self.nodes.push(right);

        self.subdivide(children_idx, leaves);
        self.subdivide(children_idx + 1, leaves);
    }

    // preserved for the eventual switch to rendering with indexed triangles on the GPU
    pub fn _closest_hit<Leaf: BvhLeaf>(&self, leaves: &Vec<Leaf>, ro: Vec3, rd: Vec3) -> Option<f32> {
        let mut stack: Vec<u32> = Vec::new();
        stack.push(0);
        let mut best_t = f32::MAX;
        let mut best_i = -1;
        while !stack.is_empty() {
            let node = self.nodes[stack.pop().unwrap() as usize];

            let aabb_t = node.aabb.closest_hit(ro, rd);
            if aabb_t.is_none() {
                continue;
            }


            if node.count > 0 {
                // leaf node
                for i in 0..node.count {
                    if let Some(t) = leaves[self.indices[(node.first + i ) as usize - self.offset] as usize].closest_hit(ro, rd) {
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

    fn closest_hit_unindexed<Leaf: BvhLeaf>(nodes: &Vec<BvhNode>, root: u32, leaves: &Vec<Leaf>, ro: Vec3, rd: Vec3) -> Option<f32> {
        let mut stack: Vec<u32> = Vec::new();
        stack.push(root);
        let mut best_t = f32::MAX;
        let mut best_i = -1;
        while !stack.is_empty() {
            let node = nodes[stack.pop().unwrap() as usize];

            let aabb_t = node.aabb.closest_hit(ro, rd);
            if aabb_t.is_none() {
                continue;
            }


            if node.count > 0 {
                // leaf node
                for i in 0..node.count {
                    if let Some(t) = leaves[(node.first + i ) as usize].closest_hit(ro, rd) {
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
override NUM_TRIANGLES: u32 = 1u;
override NUM_BVH_NODES: u32 = 1u;
override NUM_PRIMITIVES: u32 = 1u;
override NUM_VOLUMES: u32 = 2u;
override NUM_MESH_LIGHTS: u32 = 0u;

@group(0) @binding(0) var<uniform> globals : FrameUniforms;
@group(1) @binding(0) var<storage, read_write> triangles :      array<Tri>;
@group(1) @binding(1) var<storage, read_write> tri_exts :       array<TriExt>;
@group(1) @binding(2) var<storage, read_write> bvh :            array<BvhNode>;
@group(1) @binding(3) var<storage, read_write> screen :         array<vec4f>;
@group(1) @binding(4) var<storage, read_write> texture_data :   array<u32>;
@group(1) @binding(5) var<storage, read_write> primitives :     array<Primitive>;
@group(1) @binding(6) var<storage, read_write> env_map_rows_cdf:array<f32>;
@group(1) @binding(7) var<storage, read_write> mesh_lights:     array<MeshLight>;
@group(1) @binding(8) var<storage, read_write> media:           array<GpuVolume>;
@group(1) @binding(9) var                      env_map:         texture_2d<f32>;
@group(1) @binding(10) var                     env_map_col_cdf: texture_2d<f32>;
@group(1) @binding(11) var                     env_map_pdf:     texture_2d<f32>;


// struct SceneGeometry {
//     triangles: array<Tri, NUM_TRIANGLES>,
//     tri_exts:  array<TriExt, NUM_TRIANGLES>,
//     bvh:       array<BvhNode>
// }


const DEBUG: bool = true;

const pi = 3.141592654;

const FORWARD = vec3f(1.0, 0.0, 0.0);
const UP = vec3f(0.0, 0.0, 1.0);
const RIGHT = vec3f(0.0, -1.0, 0.0);

const NUM_TEXCOORDS = 2;

const EXPOSURE = 1.000;

alias mat4x4f = mat4x4<f32>;



// IQ integer hash 3 https://www.shadertoy.com/view/4tXyWN
fn hash21(in: vec2u) -> u32 {
    var p = in;
    p *= vec2u(73333u, 7777u);
    p ^= (vec2u(3333777777u) >> (p >> vec2u(28u)));
    let n = p.x * p.y;
    return n ^ (n >> 15u);
}

var<private> seed: u32 = 12378231u;
fn rand() -> f32 {
    let old = seed;

    // no basis in anything
    seed = hash21(vec2u(seed, seed ^ 39213742u));

    // uint to 0-1 float from
    // https://www.shadertoy.com/view/4tXyWN and https://iquilezles.org/articles/sfrand/
    return f32(hash21(vec2u(old, seed))) * (1.0 / f32(0xffffffffu));
}

// https://math.stackexchange.com/questions/44689/how-to-find-a-random-axis-or-unit-vector-in-3d
fn rand_sphere() -> vec3f {
    let theta = rand() * 2.0 * pi;
    let z = rand() * 2.0 - 1.0;
    let radius = sqrt(1.0 - z * z);
    return (vec3f(radius * cos(theta), radius * sin(theta), z));
}

fn rand_disk() -> vec2f {
    let theta = rand() * 2.0 * pi;
    let radius = sqrt(rand());
    return radius * vec2f(cos(theta), sin(theta));
}

fn rand_hemisphere(normal: vec3f) -> vec3f {
    var sphere = rand_sphere();
    if (dot(sphere, normal) < 0.0) {
        sphere *= -1.0;
    }
    return sphere;
}

fn rand_color() -> vec3f {
    // https://www.shadertoy.com/view/M3j3RK
    return (0.5 + 0.375 * cos(6.3 * rand() - vec3f(0.0, 2.1, 4.2)));

    // my attempt
    // return 1.0 - pow(vec3f(0.25), normalize(vec3f(rand(), rand(), rand())) + 0.1);
}

fn sign11(x: f32) -> f32 {
    if (x < 0.0) {
        return -1.0;
    } else {
        return 1.0;
    }
}

/// branchlessONB from "Building an Orthonormal Basis, Revisited"
///
/// # Citation
/// Tom Duff, James Burgess, Per Christensen, 
/// Christophe Hery, Andrew Kensler, Max Liani, 
/// and Ryusuke Villemin, Building an Orthonormal Basis, 
/// Revisited, Journal of Computer Graphics Techniques (JCGT), 
/// vol. 6, no. 1, 1-8, 2017
/// Available online http://jcgt.org/published/0006/01/01/
///
fn orthonormal_basis(n: vec3f) -> mat3x3<f32> {
    let sign = sign11(n.z);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let b1 = vec3f(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let b2 = vec3f(b, sign + n.y * n.y * a, -n.y);
    return mat3x3<f32>(b1, b2, n);
}


struct MeshLight {
    prim: i32,
    tri: i32,
    cdf: f32,
    power: f32,
}

struct Camera {
    dir:        vec3f,
    fovy:       f32,
    origin:     vec3f,
    focus:      f32,
    aperture:   f32,
    exposure:   f32,
    bloom:      f32,
    dispersion: f32,
}

struct PointLight {
    position:   vec4f,
    intensity:  vec4f,
}

struct DirectionalLight {
    direction:  vec4f,
    intensity:  vec4f,
}

struct GpuTextureRef {
    offset: u32,
    size: u32,
}

struct GpuVolume {
    absorption: vec3f,
    ior: f32,
    scattering: vec3f,
    g: f32,
}

struct Material {
    albedo:             GpuTextureRef,
    emissive:           GpuTextureRef,

    normal:             GpuTextureRef,
    metallic_roughness: GpuTextureRef,

    thickness:          GpuTextureRef,
    transmission:       GpuTextureRef,

    albedo_factor:      vec4f,

    emissive_factor:    vec3f,
    normal_scale:       f32,

    albedo_texcoord:    u32,
    emissive_texcoord:  u32,
    normal_texcoord:    u32,
    metal_r_texcoord:   u32,

    thickness_texcoord:     u32,
    transmission_texcoord:  u32,
    thickness_factor:       f32,
    transmission_factor:f32,

    metallic_factor:    f32,
    roughness_factor:   f32,
    id:                 u32,
    alpha_settings:     u32,

    /// volume index
    volume:             u32,
    _pad:               array<u32, 3>,
    /// 0: no volume, 1: thin walled, 2: homogeneous
      
}

const DEFAULT_MATERIAL = Material (
    GpuTextureRef (0, 0), // albedo
    GpuTextureRef (0, 0), // emissive

    GpuTextureRef (0, 0), // normal
    GpuTextureRef (0, 0), // metallic_roughness

    GpuTextureRef (0, 0), // thickness
    GpuTextureRef (0, 0), // transmission

    vec4f(1.0, 1.0, 1.0, 1.0), // albedo factor

    vec3f(0.0, 0.0, 0.0),   // emissive factor
    1.0,    // normal_scale

    0, // albedo_texcoord
    0, // emissive_texcoord
    0, // normal_texcoord
    0, // metal_r_texcoord

    0, // thickness_texcoord:     u32,
    0, // transmission_texcoord:  u32,
    0.0, // thickness_factor:       f32,
    0.0, // transmission_factor: f32, 
    

    0.0, // metallic_factor
    0.0, // roughness_factor
    0, // id
    0, // alpha cutoff << 16 | blend mode

    0, // background volume
    array<u32, 3>(0, 0, 0),
);

struct Primitive {
    transform:      mat4x4f,
    inv_transform:  mat4x4f,
    material:       Material,
    bvh_idx:        u32,
    tri_start:      u32,
    tri_count:      u32,
    flags:          u32,
}

struct Scene {
    point_lights:           array<PointLight, 12>,
    directional_lights:     array<DirectionalLight, 4>,
    camera:                 Camera,
    tri_count:              u32,
    num_point_lights:       u32,
    num_directional_lights: u32,
    tlas_node_count:        u32,
    mesh_light_count:       u32,
    pad1:                   u32,
    pad2:                   u32,
    pad3:                   u32,
}

struct FrameUniforms {
    scene:          Scene,
    res:            vec2u,
    frame:          u32,
    time:           f32,
    reject_hist:    u32,
    node_count:     u32,
    prim_count:     u32,
    debug_mode:     u32,
}


struct GpuVertexExt {
    texcoords: array<vec2f, NUM_TEXCOORDS>,
    normal: u32,
    tangent: u32,
    color: u32,
    t_sign: f32,
}

struct ExtSample {
    color: vec4f,

    albedo: vec4f,

    normal: vec3f,
    thickness: f32,

    emissive: vec3f,
    transmission: f32,

    metallic_roughness: vec3f,
    t_sign: f32,

    texcoords: array<vec2f, NUM_TEXCOORDS>,

    vertex_normal: vec3f,

    tangent: vec3f,
}

struct TriExt {
    vertices: array<GpuVertexExt, 3>
}


// MARK: AABB
////////////// aabb //////////////
struct Aabb {
    data: array<f32, 6>
}
fn aabb_min(aabb: Aabb) -> vec3f {
    return vec3f(aabb.data[0], aabb.data[1], aabb.data[2]);
}
fn aabb_max(aabb: Aabb) -> vec3f {
    return vec3f(aabb.data[3], aabb.data[4], aabb.data[5]);
}
fn aabb_mid(aabb: Aabb) -> vec3f {
    return 0.5 * vec3f(aabb.data[3] + aabb.data[0], aabb.data[4] + aabb.data[1], aabb.data[5] + aabb.data[2]);
}


// MARK: BVH Node
////////////// bvh node //////////////
struct BvhNode {
    aabb: Aabb,

    /// The index of the left child if count is 0. First triangle index otherwise
    first: u32,

    /// the number of triangles in the node
    count: u32,
}


// MARK: Tri
////////////// triangle //////////////
struct Tri {
    d0: vec4f,
    d1: vec4f,
    d2: vec4f,
}
fn centroid(tri: Tri) -> vec3f {
    return vec3f(tri.d0.w, tri.d1.w, tri.d2.w);
}

fn tc_size(tc: GpuTextureRef) -> vec2u {
    return vec2u(tc.size >> 16u, tc.size & 0xFFFFu);
}
  
fn sample_texture(tex: GpuTextureRef, tc: vec2f) -> vec4f {
    if tex.size == 0u {   
        return dummy_texture(tc);

    }

    let size = tc_size(tex);

    let sub_pos = fract(fract(tc) * vec2f(size));
    let texel_pos = vec2u(fract(tc) * vec2f(size));

    let x0 = clamp(texel_pos.x, 0u, size.x - 1u);
    let x1 = clamp(texel_pos.x + 1u, 0u, size.x - 1u);
    let y0 = clamp(texel_pos.y, 0u, size.y - 1u);
    let y1 = clamp(texel_pos.y + 1u, 0u, size.y - 1u);

    let ll = unpack_rgba8(texture_data[tex.offset + y0 * size.x + x0]);
    let lr = unpack_rgba8(texture_data[tex.offset + y0 * size.x + x1]);
    let ur = unpack_rgba8(texture_data[tex.offset + y1 * size.x + x1]);
    let ul = unpack_rgba8(texture_data[tex.offset + y1 * size.x + x0]);

    let u = mix(ul, ur, sub_pos.x);
    let l = mix(ll, lr, sub_pos.x);

    return mix(l, u, sub_pos.y);
}


fn unpack_rgba8(x: u32) -> vec4f {
    return vec4f(
        f32((x >> 24u) & 255u) / 255.0,
        f32((x >> 16u) & 255u) / 255.0,
        f32((x >> 8u)  & 255u) / 255.0,
        f32((x >> 0u)  & 255u) / 255.0
    );
}

// https://gamedev.stackexchange.com/questions/169508/octahedral-impostors-octahedral-mapping
fn unpack_unit_octahedral(f_in: vec2f) -> vec3f {
    var f = f_in;
    f = f * 2.0 - 1.0;
 
    // https://twitter.com/Stubbesaurus/status/937994790553227264
    var n = vec3f(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    let t = saturate(-n.z);

    n.x -= t * sign11(n.x);
    n.y -= t * sign11(n.y);
    // n.xy += n.xy >= 0.0 ? -t : t;
    return normalize(n);
}

fn unpack_unit_oct32(u_in: u32) -> vec3f {
    let f = vec2f(f32(u_in >> 16u), f32(u_in & 0xFFFFu)) / f32(0xFFFF);
    return unpack_unit_octahedral(f);
}

fn zeroed_ext_sample() -> ExtSample {
    var s: ExtSample;
    s.color = vec4f(0.0, 0.0, 0.0, 0.0);
    s.normal = vec3f(0.0, 0.0, 0.0);
    s.albedo = vec4f(0.0, 0.0, 0.0, 0.0);
    s.metallic_roughness = vec3f(0.0, 0.0, 0.0);
    s.emissive = vec3f(0.0, 0.0, 0.0);
    s.vertex_normal = vec3f(0.0);
    s.tangent = vec3f(0.0);
    s.t_sign = 0.0;
    s.texcoords[0] = vec2f(0.0);
    s.texcoords[1] = vec2f(0.0);
    s.transmission = 0.0;
    s.thickness = 0.0;
    return s;
}

// red checkerboard for missing textures
fn dummy_texture(uv: vec2f) -> vec4f {
    let scale: f32 = 256.0;
    let checker = f32((u32(uv.x * scale) + u32(uv.y * scale + 1.0)) % 2u);
    var col = mix(vec3f(0.8, 0.3, 0.3), vec3f(0.8, 0.3, 0.3) * 0.5, checker);
    return vec4f(col, 1.0);
}

// MARK: interpolation

/// barycentric interpolation of vertex attributes
fn tri_ext_interpolate(tri: ptr<function, TriExt>, bary: vec3f) -> ExtSample {
    var res = zeroed_ext_sample();

    // cant loop: cannot index into value of type `vec3<f32>`, and no copy-paste macros in wgsl
    res.color += bary.x * unpack_rgba8((*tri).vertices[0].color);
    res.texcoords[0] += bary.x * (*tri).vertices[0].texcoords[0];
    res.texcoords[1] += bary.x * (*tri).vertices[0].texcoords[1];
    res.vertex_normal += bary.x * unpack_unit_oct32((*tri).vertices[0].normal);
    res.tangent += bary.x * unpack_unit_oct32((*tri).vertices[0].tangent);
    res.t_sign += bary.x * (*tri).vertices[0].t_sign;

    res.color += bary.y * unpack_rgba8((*tri).vertices[1].color);
    res.texcoords[0] += bary.y * (*tri).vertices[1].texcoords[0];
    res.texcoords[1] += bary.y * (*tri).vertices[1].texcoords[1];
    res.vertex_normal += bary.y * unpack_unit_oct32((*tri).vertices[1].normal);
    res.tangent += bary.y * unpack_unit_oct32((*tri).vertices[1].tangent);
    res.t_sign += bary.y * (*tri).vertices[1].t_sign;

    res.color += bary.z * unpack_rgba8((*tri).vertices[2].color);
    res.texcoords[0] += bary.z * (*tri).vertices[2].texcoords[0];
    res.texcoords[1] += bary.z * (*tri).vertices[2].texcoords[1];
    res.vertex_normal += bary.z * unpack_unit_oct32((*tri).vertices[2].normal);
    res.tangent += bary.z * unpack_unit_oct32((*tri).vertices[2].tangent);
    res.t_sign += bary.z * (*tri).vertices[2].t_sign;

    res.vertex_normal = normalize(res.vertex_normal);
    res.tangent = normalize(res.tangent);

    return res;
}
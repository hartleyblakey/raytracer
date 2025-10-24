@group(0) @binding(0) var<uniform> globals : FrameUniforms;

@group(1) @binding(0) var<storage, read_write> triangles :      array<Tri>;
@group(1) @binding(1) var<storage, read_write> tri_exts :       array<TriExt>;
@group(1) @binding(2) var<storage, read_write> bvh :            array<BvhNode>;
@group(1) @binding(3) var<storage, read_write> screen :         array<vec4f>;
@group(1) @binding(4) var<storage, read_write> texture_data :   array<u32>;
@group(1) @binding(5) var<storage, read_write> primitives :     array<Primitive>;
@group(1) @binding(6) var<storage, read_write> env_map_rows_cdf:array<f32>;
@group(1) @binding(7) var<storage, read_write> mesh_lights:     array<MeshLight>;
@group(1) @binding(8) var                      env_map:         texture_2d<f32>;
@group(1) @binding(9) var                      env_map_col_cdf: texture_2d<f32>;
@group(1) @binding(10) var                     env_map_pdf:     texture_2d<f32>;

const pi = 3.141592654;

const FORWARD = vec3f(1.0, 0.0, 0.0);
const UP = vec3f(0.0, 0.0, 1.0);
const RIGHT = vec3f(0.0, -1.0, 0.0);

const NUM_TEXCOORDS = 2;

const EXPOSURE = 1.000;

var<private> debug: f32;

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

    volume:             GpuVolume,
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

    GpuVolume (vec3f(0), 1.0), // empty volume
     
);

struct Primitive {
    transform:      mat4x4f,
    inv_transform:  mat4x4f,
    material:       Material,
    bvh_idx:        u32,
    tri_start:      u32,
    tri_count:      u32,
    flags:           u32,
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

fn tc_size(tc: GpuTextureRef) -> vec2u {
    return vec2u(tc.size >> 16u, tc.size & 0xFFFFu);
}
  
fn sample_texture(tex: GpuTextureRef, tc: vec2f) -> vec4f {
    if tex.size == 0 {
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
    let f = vec2f(f32(u_in >> 16u), f32(u_in & 0xFFFF)) / f32(0xFFFF);
    return unpack_unit_octahedral(f);
}




fn zeroed_ext_sample() -> ExtSample {
    var s: ExtSample;
    s.color = vec4f(0.0, 0.0, 0.0, 0.0);
    s.normal = vec3f(0.0, 0.0, 0.0);
    s.albedo = vec4f(0.0, 0.0, 0.0, 0.0);
    s.metallic_roughness = vec3f(0.0, 0.0, 0.0);
    s.emissive = vec3f(0.0, 0.0, 0.0);
    s.vertex_normal = vec3f(0);
    s.tangent = vec3f(0);
    s.t_sign = 0.0;
    s.texcoords[0] = vec2f(0);
    s.texcoords[1] = vec2f(0);
    s.transmission = 0.0;
    s.thickness = 0.0;
    return s;
}

// red checkerboard for missing textures
fn dummy_texture(uv: vec2f) -> vec4f {
    const scale = 256.0;
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

fn trace_transformed_tri(ray: Ray, prim: i32, tri: i32) -> Hit {
    var hit = intersect_full(transform_ray(ray, primitives[prim].inv_transform), tri);
    if hit.idx == -1 {
        return hit;
    }
    hit.t /= length(transform_dir(ray.dir, primitives[prim].inv_transform));
    hit.normal = transform_normal(hit.normal, primitives[prim].inv_transform);
    hit.prim_idx = prim;
    hit.material = primitives[prim].material;
    return hit;
}

fn sample_emission(hit: Hit) -> vec3f {
    var res = zeroed_ext_sample();
    let prim = primitives[hit.prim_idx];
    let ext = tri_exts[hit.idx];
    let tci = prim.material.emissive_texcoord;
    
    var tc = vec2f(0);
    tc += hit.bary.x * ext.vertices[0].texcoords[tci];
    tc += hit.bary.y * ext.vertices[1].texcoords[tci];
    tc += hit.bary.z * ext.vertices[2].texcoords[tci];

    var emissive = hit.material.emissive_factor;
    if hit.material.emissive.size != 0 {
        emissive *= to_linear(sample_texture(hit.material.emissive, tc).rgb);
    }
    return emissive;
}


/// gather material parameters for a hit
fn sample_hit(hit: Hit) -> ExtSample {
    var ext = tri_exts[hit.idx];
    var sample = tri_ext_interpolate(&ext, hit.bary);

    // get smooth_normal into world space, if it exists
    if length(sample.vertex_normal) > 0.001 {
        sample.vertex_normal = transform_normal(normalize(sample.vertex_normal), primitives[hit.prim_idx].inv_transform);
        if dot(sample.vertex_normal, hit.normal) < 0.0 {
            // flip face if vertex normal pointing away from geometric normal
           sample.vertex_normal = -sample.vertex_normal;
           sample.tangent = -sample.tangent;
           sample.t_sign *= -1.0;
        }
    } else {
        
        sample.vertex_normal = hit.normal;
    }

    sample.normal = sample.vertex_normal;
    if (hit.material.normal.size != 0) {
        let normal_tangent = (sample_texture(hit.material.normal, sample.texcoords[hit.material.normal_texcoord]).xyz * 2.0 - 1.0) * hit.material.normal_scale;

        // this is what the guys website says, but it looks a little off to me
        let bt = normalize(sample.t_sign * cross(sample.vertex_normal, sample.tangent));
        // let bt = sample.bi_tangent;

        sample.normal = normalize(
            normal_tangent.x * sample.tangent + normal_tangent.y * bt + normal_tangent.z * sample.vertex_normal
        );
    }

    sample.albedo = hit.material.albedo_factor;
    if (ext.vertices[0].color != 0) { sample.albedo *= sample.color;}
    if hit.material.albedo.size != 0 {
        sample.albedo *= to_linear_4(sample_texture(hit.material.albedo, sample.texcoords[hit.material.albedo_texcoord]));
    }

    sample.emissive = hit.material.emissive_factor;
    if hit.material.emissive.size != 0 {
        sample.emissive *= to_linear(sample_texture(hit.material.emissive, sample.texcoords[hit.material.emissive_texcoord]).rgb);
    }

    sample.transmission = hit.material.transmission_factor;
    if hit.material.transmission.size != 0 {
        sample.transmission *= sample_texture(hit.material.transmission, sample.texcoords[hit.material.transmission_texcoord]).r;
    }

    sample.thickness = hit.material.thickness_factor;
    if hit.material.thickness.size != 0 {
        sample.thickness *= sample_texture(hit.material.thickness, sample.texcoords[hit.material.thickness_texcoord]).r;
    }

    sample.metallic_roughness = vec3f(0.0, hit.material.roughness_factor, hit.material.metallic_factor);
    if hit.material.metallic_roughness.size != 0 {
        sample.metallic_roughness *= sample_texture(hit.material.metallic_roughness, sample.texcoords[hit.material.metal_r_texcoord]).rgb;
    }
    sample.metallic_roughness.g = clamp(sample.metallic_roughness.g, 0.05, 1.0);

    if globals.debug_mode == 9 {
        sample.albedo = vec4(1);
    }
    // sample.metallic_roughness.g = 0.03;

    return sample;
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

fn assert(condition: bool) {
    if !condition {
        debug = -99999999.0;
    }
}

// MARK: Stack
////////////// stack //////////////
struct Stack {
    data: array<u32, 23>,
    size: u32,
}
fn push(stack: ptr<function, Stack>, val: u32) {
    if ((*stack).size < 23) {
        (*stack).data[(*stack).size] = val;
        (*stack).size += 1u;
    } else {
        assert(false);
    }
}
fn pop(stack: ptr<function, Stack>) -> u32 {
    (*stack).size -= 1u;
    return (*stack).data[(*stack).size];
}

// MARK: Hit
////////////// hit //////////////
struct Hit {
    t: f32,
    idx: i32,
    prim_idx: i32,
    material: Material,
    normal: vec3f,
    bary: vec3f,
    backface: bool,
}

// MARK: Ray
////////////// ray //////////////
struct Ray {
    origin: vec3f,
    dir: vec3f,
    idir: vec3f,
}

const TRI_EPS: f32 = 0.00000001;

// https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
// epsilon stolen from https://www.shadertoy.com/view/wlsfRs
fn intersect (ray: Ray, tri: Tri) -> f32 {
    let edge1 = tri.d1.xyz - tri.d0.xyz;
    let edge2 = tri.d2.xyz - tri.d0.xyz;
    let h = cross( ray.dir, edge2 );
    let a = dot( edge1, h );
    if (a > -TRI_EPS && a < TRI_EPS) {
        return -1.0;
    }// ray parallel to triangle
    let f = 1.0 / a;
    let s = ray.origin - tri.d0.xyz;
    let u = f * dot( s, h );
    if (u < 0.0 || u > 1.0) {
        return -1.0;
    }
    let q = cross( s, edge1 );
    let v = f * dot( ray.dir, q );
    if (v < 0.0 || u + v > 1.0) {
        return -1.0;
    }
    let t = f * dot( edge2, q );
    if (t > TRI_EPS) {
        return t;
    } else {
        return -1.0;
    }
}

fn sign11(x: f32) -> f32 {
    if (x < 0.0) {
        return -1.0;
    } else {
        return 1.0;
    }
}

fn hit_default() -> Hit {
    return Hit(0.0, -1, -1, DEFAULT_MATERIAL, vec3f(0.0, 0.0, 1.0), vec3f(0.333, 0.333, 0.333), false);
}

// modified version of intersect() to return more info
//     from https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
fn intersect_full(ray: Ray, idx: i32) -> Hit {
    let tri = triangles[idx];
    var hit = hit_default();

    let edge1 = tri.d1.xyz - tri.d0.xyz;
    let edge2 = tri.d2.xyz - tri.d0.xyz;

    hit.normal = normalize(cross(edge1, edge2));
    if dot(hit.normal, ray.dir) > 0.0 {
        hit.backface = true;
    }
    hit.normal *= -sign11(dot(hit.normal, ray.dir));

    let h = cross( ray.dir, edge2 );
    let a = dot( edge1, h );
    if (a > -TRI_EPS && a < TRI_EPS) {
        return hit;
    }// ray parallel to triangle
    let f = 1.0 / a;
    let s = ray.origin - tri.d0.xyz;
    let u = f * dot( s, h );
    if (u < 0 || u > 1.0) {
        return hit;
    }   // miss?
    let q = cross( s, edge1 );
    let v = f * dot( ray.dir, q );
    if (v < 0 || u + v > 1) {
        return hit;
    }   // miss?
    let t = f * dot( edge2, q );
    if (t <= TRI_EPS) {
        return hit;
    }   // miss?


    hit.idx = idx;
    hit.t = t;
    hit.bary = vec3f((1.0 - u) - v, u, v);
    return hit;
}

// from https://gist.github.com/DomNomNom/46bb1ce47f68d255fd5d
fn intersect_aabb(ray: Ray, aabb: Aabb) -> f32 {

    let bmin = aabb_min(aabb);
    let bmax = aabb_max(aabb);

    let rmin = (bmin - ray.origin) * ray.idir;
    let rmax = (bmax - ray.origin) * ray.idir;

    let tmin = min(rmin, rmax);
    let tmax = max(rmin, rmax);

    let t0 = max(tmin.x, max(tmin.y, tmin.z));
    let t1 = min(tmax.x, min(tmax.y, tmax.z));

    if (t0 > t1 || t1 < 0.0) {
        return 1e30;
    }

    return max(t0, 0.0);
}

// MARK: trace_bvh



fn trace_bvh(ray: Ray, root: u32, t_max: ptr<function, f32>, prim: Primitive) -> i32 {
    var stack: Stack;
    stack.size = 0u;
    var node = bvh[root];
    var best_t = *t_max;
    var best_i: i32 = -1;
    if intersect_aabb(ray, node.aabb) >= best_t {
        return best_i;
    }
    
    while (true) {
        // debug = max(debug, f32(stack.size + 1u));
        // visualize bvh steps
        debug += 0.5;


        if node.count > 0 {

            // if debug > 0.0 {
            //     return i32(node.first);
            // }

            // intersect triangles of node
            for (var i = node.first; i < node.first + node.count; i++) {
                let t = intersect(ray, triangles[i]);
                if t >= 0.0 && t < best_t {
                    if (prim.material.alpha_settings & 3u) != 0u {
                        let hit = intersect_full(ray, i32(i));
                        let ext = tri_exts[i];

                        var texcoord = vec2f(0.0);

                        texcoord += hit.bary.x * ext.vertices[0].texcoords[prim.material.albedo_texcoord];
                        texcoord += hit.bary.y * ext.vertices[1].texcoords[prim.material.albedo_texcoord];
                        texcoord += hit.bary.z * ext.vertices[2].texcoords[prim.material.albedo_texcoord];

                        let alpha = sample_texture(prim.material.albedo, texcoord).a;

                        if (prim.material.alpha_settings & 3u) == 1u {
                            // MASK
                            if alpha < f32(prim.material.alpha_settings >> 16u) / f32(1u << 16u) {
                                continue;
                            }
                        } else {
                            // BLEND
                            if rand() > alpha * alpha {
                                continue;
                            }
                        }
                    }
                    best_i = i32(i);
                    best_t = t;
                }
            }
            if stack.size == 0u {
                break;
            }
            node = bvh[pop(&stack)];
        } else {
            // avoid pushing nodes onto the stack where possible
            // order nodes based on distance

            // try ordering the nodes
            var left  = intersect_aabb(ray, bvh[node.first + 0u].aabb);
            var right = intersect_aabb(ray, bvh[node.first + 1u].aabb);
    
            if (left > best_t) && (right > best_t) {
                if stack.size == 0u {
                    break;
                }
                node = bvh[pop(&stack)];
            } else if (left > best_t) {
                node = bvh[node.first + 1u];
            } else if (right > best_t) {
                node = bvh[node.first + 0u];
            } else if right > left {
                // push(&stack, node.first + 0u);
                // node = bvh[node.first + 1u];

                push(&stack, node.first + 1u);
                node = bvh[node.first + 0u];
            } else {
                // push(&stack, node.first + 1u);
                // node = bvh[node.first + 0u];

                push(&stack, node.first + 0u);
                node = bvh[node.first + 1u];
            } 

        }
    }
    *t_max = best_t;
    return i32(best_i);
}

fn trace_bvh_shadow(ray: Ray, root: u32, t_max: ptr<function, f32>, prim: Primitive) -> bool {
    var stack: Stack;
    stack.size = 0u;
    var node = bvh[root];
    var best_t = *t_max;

    if intersect_aabb(ray, node.aabb) >= best_t {
        return false;
    }
    
    while (true) {
        if node.count > 0 {
            for (var i = node.first; i < node.first + node.count; i++) {
                let t = intersect(ray, triangles[i]);
                if t >= 0.0 && t < best_t {
                    if (prim.material.alpha_settings & 3u) != 0u {
                        let hit = intersect_full(ray, i32(i));
                        let ext = tri_exts[i];

                        var texcoord = vec2f(0.0);

                        texcoord += hit.bary.x * ext.vertices[0].texcoords[prim.material.albedo_texcoord];
                        texcoord += hit.bary.y * ext.vertices[1].texcoords[prim.material.albedo_texcoord];
                        texcoord += hit.bary.z * ext.vertices[2].texcoords[prim.material.albedo_texcoord];

                        let alpha = sample_texture(prim.material.albedo, texcoord).a;

                        if (prim.material.alpha_settings & 3u) == 1u {
                            // MASK
                            if alpha < f32(prim.material.alpha_settings >> 16u) / f32(1u << 16u) {
                                continue;
                            }
                        } else {
                            // BLEND
                            if rand() > alpha * alpha {
                                continue;
                            }
                        }
                    }
                    return true;
                }
            }
            if stack.size == 0u {
                break;
            }
            node = bvh[pop(&stack)];
        } else {
            // avoid pushing nodes onto the stack where possible
            // order nodes based on distance

            // try ordering the nodes
            var left  = intersect_aabb(ray, bvh[node.first + 0u].aabb);
            var right = intersect_aabb(ray, bvh[node.first + 1u].aabb);
    
            if (left > best_t) && (right > best_t) {
                if stack.size == 0u {
                    break;
                }
                node = bvh[pop(&stack)];
            } else if (left > best_t) {
                node = bvh[node.first + 1u];
            } else if (right > best_t) {
                node = bvh[node.first + 0u];
            } else if right > left {
                push(&stack, node.first + 1u);
                node = bvh[node.first + 0u];
            } else {
                push(&stack, node.first + 0u);
                node = bvh[node.first + 1u];
            } 

        }
    }
    return false;
}
fn transform_dir(x: vec3f, t: mat4x4f) -> vec3f {
    return (t * vec4f(x.x, x.y, x.z, 0.0)).xyz;
}

fn transform_pos(x: vec3f, t: mat4x4f) -> vec3f {
    return (t * vec4f(x.x, x.y, x.z, 1.0)).xyz;
}

fn transform_normal(x: vec3f, t_inv: mat4x4f) -> vec3f {
    return normalize(transform_dir(x, transpose(t_inv)));
}

fn transform_ray(x: Ray, it: mat4x4f) -> Ray {
    var r = x;
    r.dir = normalize(transform_dir(r.dir, it));
    r.origin = transform_pos(r.origin, it);
    r.idir = 1.0 / r.dir;
    return r;
}

fn trace(ray: Ray) -> Hit {
    debug = 0.0;
    var stack: Stack;
    stack.size = 0u;
    var node = bvh[globals.node_count];
    var best_t = 99999999.0;
    var closest_tri: i32 = -1;
    var closest_primitive: i32 = -1;
    if intersect_aabb(ray, node.aabb) > best_t {
        return hit_default();
    }
    var tlas_steps = 0.0;
    while (true) {
        // debug = max(debug, f32(stack.size + 1u));
        // visualize bvh steps
        debug += 1.0;

        if node.count > 0 {
            tlas_steps += 1.0;
            // intersect BLAS(s) of node
            for (var i = node.first; i < node.first + node.count; i++) {
                let scale_factor = length(transform_dir(ray.dir, primitives[i].inv_transform));
                let t_ray = transform_ray(ray, primitives[i].inv_transform);
                // debug += 1.0;
                var new_t = best_t * scale_factor;
                let new_tri = trace_bvh(t_ray, primitives[i].bvh_idx, &new_t, primitives[i]);
                

                if new_tri >= 0 {
                    best_t = new_t / scale_factor;
                    closest_tri = new_tri;
                    closest_primitive = i32(i);

                    // var hit = hit_default();
                    // hit.prim_idx = i32(i);
                    // hit.idx = i32(new_tri);
                    // hit.normal = -ray.dir;
                    // return hit;
                }
            }
            if stack.size == 0u {
                break;
            }
            node = bvh[pop(&stack)];
        } else {
            // avoid pushing nodes onto the stack where possible
            // order nodes based on distance

            // TLAS is tacked onto end of bvh:
            let node_first = globals.node_count + node.first;

            // try ordering the nodes
            let left  = intersect_aabb(ray, bvh[node_first + 0u].aabb);
            let right = intersect_aabb(ray, bvh[node_first + 1u].aabb);
    
            if (left > best_t) && (right > best_t) {
                if stack.size == 0u {
                    break;
                }
                node = bvh[pop(&stack)];
            } else if (left > best_t) {
                node = bvh[node_first + 1u];
            } else if (right > best_t) {
                node = bvh[node_first + 0u];
            } else if right < left {
                // push(&stack, node_first + 1u);
                // node = bvh[node_first + 0u];

                push(&stack, node_first + 0u);
                node = bvh[node_first + 1u];

            } else {
                // push(&stack, node_first + 0u);
                // node = bvh[node_first + 1u];

                
                push(&stack, node_first + 1u);
                node = bvh[node_first + 0u];
            }

        }
    }

    let t_ray_final = transform_ray(ray, primitives[closest_primitive].inv_transform);
    var hit = intersect_full(t_ray_final, closest_tri);

    // transform the hit back to world space
    hit.t = best_t;
    hit.normal = transform_normal(hit.normal, primitives[closest_primitive].inv_transform);
    hit.prim_idx = i32(closest_primitive);
    hit.material = primitives[closest_primitive].material;
    return hit;

}

fn trace_shadow(ray: Ray, t: f32) -> bool {
    var stack: Stack;
    stack.size = 0u;
    var node = bvh[globals.node_count];
    var best_t = t;
    if intersect_aabb(ray, node.aabb) > best_t {
        return false;
    }
    while (true) {

        if node.count > 0 {
            // intersect BLAS(s) of node
            for (var i = node.first; i < node.first + node.count; i++) {
                let scale_factor = length(transform_dir(ray.dir, primitives[i].inv_transform));
                let t_ray = transform_ray(ray, primitives[i].inv_transform);
                var new_t = best_t * scale_factor;
                
                if trace_bvh_shadow(t_ray, primitives[i].bvh_idx, &new_t, primitives[i]) {
                    return true;
                }
            }
            if stack.size == 0u {
                break;
            }
            node = bvh[pop(&stack)];
        } else {
            // TLAS is tacked onto end of bvh:
            let node_first = globals.node_count + node.first;

            // try ordering the nodes
            let left  = intersect_aabb(ray, bvh[node_first + 0u].aabb);
            let right = intersect_aabb(ray, bvh[node_first + 1u].aabb);
    
            if (left > best_t) && (right > best_t) {
                if stack.size == 0u {
                    break;
                }
                node = bvh[pop(&stack)];
            } else if (left > best_t) {
                node = bvh[node_first + 1u];
            } else if (right > best_t) {
                node = bvh[node_first + 0u];
            } else if right < left {
                // push(&stack, node_first + 1u);
                // node = bvh[node_first + 0u];

                push(&stack, node_first + 0u);
                node = bvh[node_first + 1u];

            } else {
                // push(&stack, node_first + 0u);
                // node = bvh[node_first + 1u];

                
                push(&stack, node_first + 1u);
                node = bvh[node_first + 0u];
            }

        }
    }
    return false;
}

// IQ integer hash 3 https://www.shadertoy.com/view/4tXyWN
fn hash21(in: vec2u) -> u32 {
    var p = in;
    p *= vec2u(73333, 7777);
    p ^= (vec2u(3333777777) >> (p >> vec2u(28)));
    let n = p.x * p.y;
    return n ^ (n >> 15u);
}

var<private> seed: u32 = 12378231;
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
    return (0.5 + 0.375 * cos(6.3 * rand() - vec3f(0, 2.1, 4.2)));

    // my attempt
    // return 1.0 - pow(vec3f(0.25), normalize(vec3f(rand(), rand(), rand())) + 0.1);
}

fn trace_to_target(ray: Ray, prim: i32, tri: i32) -> Hit {
    var stack: Stack;
    stack.size = 0u;
    var node = bvh[globals.node_count];

    var hit = trace_transformed_tri(ray, prim, tri);
    if hit.idx == -1 {
        return hit;
    }


    if intersect_aabb(ray, node.aabb) > hit.t {
        return hit_default();
    }
    while (true) {
        if node.count > 0 {
            // intersect BLAS(s) of node
            for (var i = node.first; i < node.first + node.count; i++) {
                let scale_factor = length(transform_dir(ray.dir, primitives[i].inv_transform));
                let t_ray = transform_ray(ray, primitives[i].inv_transform);
                var new_t = hit.t * scale_factor;
                let new_tri = trace_bvh(t_ray, primitives[i].bvh_idx, &new_t, primitives[i]);
                
                if new_tri >= 0 {
                    // hit something before target
                    return hit_default();
                }
            }
            if stack.size == 0u {
                break;
            }
            node = bvh[pop(&stack)];
        } else {
            // avoid pushing nodes onto the stack where possible
            // order nodes based on distance

            // TLAS is tacked onto end of bvh:
            let node_first = globals.node_count + node.first;

            // try ordering the nodes
            let left  = intersect_aabb(ray, bvh[node_first + 0u].aabb);
            let right = intersect_aabb(ray, bvh[node_first + 1u].aabb);
    
            if (left > hit.t) && (right > hit.t) {
                if stack.size == 0u {
                    break;
                }
                node = bvh[pop(&stack)];
            } else if (left > hit.t) {
                node = bvh[node_first + 1u];
            } else if (right > hit.t) {
                node = bvh[node_first + 0u];
            } else if right < left {
                // push(&stack, node_first + 1u);
                // node = bvh[node_first + 0u];

                push(&stack, node_first + 0u);
                node = bvh[node_first + 1u];

            } else {
                // push(&stack, node_first + 0u);
                // node = bvh[node_first + 1u];

                
                push(&stack, node_first + 1u);
                node = bvh[node_first + 0u];
            }

        }
    }

    return hit;
}

fn area_to_sa(area: f32, normal: vec3f, p: vec3f, reference: vec3f) -> f32 {
    let d = (p - reference);
    return area * dot(d, d) / abs(dot(normalize(-d), normal));
}

/// uniform sample a point on a triangle - *returns area pdf*
fn sample_triangle_area(prim: i32, tri: i32, pdf: ptr<function, f32>, normal: ptr<function, vec3f>) -> vec3f {
    let e1 = sqrt(rand());
    let e2 = rand();

    let u = 1.0 - e1;
    let v = e2 * e1;

    let transform = primitives[prim].transform;
    let lt = triangles[tri];

    let v0 = transform_pos(lt.d0.xyz, transform);
    let v1 = transform_pos(lt.d1.xyz, transform);
    let v2 = transform_pos(lt.d2.xyz, transform);

    let a = (v1 - v0);
    let b = (v2 - v0);

    *normal = normalize(cross(a, b));

    *pdf = 1.0 / (0.5 * length(cross(a, b)));

    return v0 + a * u + b * v;
}

fn sample_triangle_pdf(prim: i32, tri: i32, point: vec3f, normal: vec3f, reference: vec3f) -> f32 {
    let transform = primitives[prim].transform;
    let lt = triangles[tri];

    let v0 = transform_pos(lt.d0.xyz, transform);
    let v1 = transform_pos(lt.d1.xyz, transform);
    let v2 = transform_pos(lt.d2.xyz, transform);

    let a = (v1 - v0);
    let b = (v2 - v0);
    let area = 0.5 * length(cross(a, b));
    if area == 0.0 {
        return 0.0;
    }
    return area_to_sa(1.0 / area, normal, point, reference);
}

// MARK: - Mesh Lights

fn sample_mesh_light(reference: vec3f, pdf: ptr<function, f32>, prim_idx: ptr<function, i32>, tri_idx: ptr<function, i32>) -> vec3f {

    let dim = globals.scene.mesh_light_count;
    var c = 0;
    {
        var low = 0;
        var high = i32(dim) - 1;
        var cdf = 0.0;
        var cdf_target = rand();

        while high >= low {
            let mid = (low + high) / 2;
            cdf = mesh_lights[mid].cdf;
            if cdf < cdf_target {
                low = mid + 1;
            } else {
                high = mid - 1;
                c = mid;
            }
            
        }
    }
    var tri_pdf: f32;
    var normal: vec3f;
    let p = sample_triangle_area(mesh_lights[c].prim, mesh_lights[c].tri, &tri_pdf, &normal);
    normal *= sign11(dot(normal, reference - p));
    tri_pdf = area_to_sa(tri_pdf, normal, p, reference);
    *pdf = mesh_lights[c].power * tri_pdf;
    *prim_idx = mesh_lights[c].prim;
    *tri_idx = mesh_lights[c].tri;
    return p;
}

// 
fn sample_mesh_light_pdf(prim: i32, tri: i32, point: vec3f, reference: vec3f) -> f32 {
    if globals.scene.mesh_light_count == 0 {
        return 0.0;
    }
    let base = primitives[prim].flags >> 8u;
    if (primitives[prim].flags & (1u << 1u)) == 0 {
        // not am emissive primitive
        return 0.0;
    }

    var tri_pdf: f32;
    var m_normal: vec3f;
    let p = sample_triangle_area(prim, tri, &tri_pdf, &m_normal);
    m_normal *= sign11(dot(m_normal, reference - p));
    tri_pdf = area_to_sa(tri_pdf, m_normal, p, reference);

    let triangle_pdf = mesh_lights[i32(base) + (tri - i32(primitives[prim].tri_start))].power;

    return triangle_pdf * tri_pdf;
}


// MARK: Environment map

fn env_map_to_dir(e: vec2f) -> vec3f {
    var v = vec3f(0);
    v.z = cos(e.y * pi);
    let phi = e.x * 2.0 * pi - pi;
    let sin_theta = sin(e.y * pi);
    v.x = sin_theta * cos(phi);
    v.y = sin_theta * sin(phi);
    return v.yxz;
}

/// Returns the index of the chosen pixel
fn sample_env_map() -> vec3f {
    if globals.debug_mode == 9 {
        return rand_sphere();
    }
    let dim = textureDimensions(env_map);
    var row = 0u;
    {
        var row_low = 0u;
        var row_high = dim.y - 1u;
        var row_mid = (row_low + row_high) / 2;
        var cdf = 0.0;
        var cdf_target = rand();

        while row_high >= row_low {
            
            cdf = env_map_rows_cdf[row_mid];
            if cdf < cdf_target {
                row_low = row_mid + 1;
            } else {
                row_high = row_mid - 1;
                row = row_mid;
            }
            row_mid = (row_low + row_high) / 2;
        }
    }

    var col = 0u;
    {

        var col_low = 0u;
        var col_high = dim.x - 1u;
        var col_mid = (col_low + col_high) / 2;
        var cdf = 0.0;
        var cdf_target = rand();

        while col_high >= col_low {
            
            cdf = textureLoad(env_map_col_cdf, vec2u(col_mid, row), 0).r;
            if cdf < cdf_target {
                col_low = col_mid + 1;
            } else {
                col_high = col_mid - 1;
                col = col_mid;
            }
            col_mid = (col_low + col_high) / 2;
        }
    }
    let res = vec2f(textureDimensions(env_map));

    let uv = vec2f(f32(col) + rand(), f32(row) + rand()) / res;
    
    return env_map_to_dir(uv);
}
fn dir_to_env_map(dir: vec3f) -> vec2f {
    // HACK: For comparison with blender, use their coordinate system for sampling the HDRI
    let d = dir.yxz;
    return vec2f(atan2(d.y, d.x) + pi, acos(d.z)) / vec2f(2.0 * pi, pi);
}

fn sample_env_map_pdf(dir: vec3f) -> f32 {
    if globals.debug_mode == 9 {
        return 1.0 / (4.0 * pi);
    }

    let uv = dir_to_env_map(dir);
    let res = vec2f(textureDimensions(env_map));

    return textureLoad(env_map_pdf, vec2u(uv * res), 0).r;
}

fn evaluate_env_map(dir: vec3f) -> vec4f {
    if globals.debug_mode == 0 {
        // return vec4f(0, 0, 0, 1);
    }
    if globals.debug_mode == 9 {
        return vec4f(0.5, 0.5, 0.5, 1.0);
    }
    let uv = dir_to_env_map(dir);
    return textureLoad(env_map, vec2u(uv * vec2f(textureDimensions(env_map))), 0); 
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

// from https://www.shadertoy.com/view/XtGGzG
fn plasma_quintic( _x: f32 ) -> vec3f {
	let x = saturate( _x );
	let x1 = vec4f( 1.0, x, x * x, x * x * x ); // 1 x x2 x3
	let x2 = x1 * x1.w * x; // x4 x5 x6 x7
	return vec3f(
		dot( x1.xyzw, vec4f(0.063861086, 1.992659096, -1.023901152, -0.490832805 ) ) + dot( x2.xy, vec2f( 1.308442123, -0.914547012 ) ),
		dot( x1.xyzw, vec4f(0.049718590, -0.791144343, 2.892305078, 0.811726816 ) ) + dot( x2.xy, vec2f( -4.686502417, 2.717794514 ) ),
		dot( x1.xyzw, vec4f(0.513275779, 1.580255060, -5.164414457, 4.559573646 ) ) + dot( x2.xy, vec2f( -1.916810682, 0.570638854 ) ) );
}
// from https://www.shadertoy.com/view/XtGGzG
fn magma_quintic( _x: f32 ) -> vec3f {
	let x = saturate( _x );
	let x1 = vec4f( 1.0, x, x * x, x * x * x ); // 1 x x2 x3
	let x2 = x1 * x1.w * x; // x4 x5 x6 x7
	return vec3f(
		dot( x1.xyzw, vec4( -0.023226960, 1.087154378, -0.109964741, 6.333665763 ) ) + dot( x2.xy, vec2( -11.640596589, 5.337625354 ) ),
		dot( x1.xyzw, vec4( 0.010680993, 0.176613780, 1.638227448, -6.743522237 ) ) + dot( x2.xy, vec2( 11.426396979, -5.523236379 ) ),
		dot( x1.xyzw, vec4( -0.008260782, 2.244286052, 3.005587601, -24.279769818 ) ) + dot( x2.xy, vec2( 32.484310068, -12.688259703 ) ) );
}

fn to_linear(srgb: vec3f) -> vec3f {
    // not correct but close enough for now
    return pow(srgb, vec3f(2.2));
}

fn to_linear_4(srgb: vec4f) -> vec4f {
    // not correct but close enough for now
    return pow(srgb, vec4f(2.2));
}

// false color visualizations, x will be clamped from 0..1
fn ramp(x: f32) -> vec3f {
    if x < 0.0 {
        return vec3f(0.0, 0.0, 1.0);
    }
    if x > 1.0 {
        return vec3f(1.0, 1.0, 0.0);
    }
    return to_linear(clamp(magma_quintic(x), vec3f(0.0), vec3f(1.0)));
}

// MARK: camera_ray
fn camera_ray(pixel: vec2u) -> Ray {
    var ray: Ray;

    ray.origin  = globals.scene.camera.origin;
    let forward = globals.scene.camera.dir;
    let fov_factor = (sin(globals.scene.camera.fovy / 2.0) / cos(globals.scene.camera.fovy / 2.0)) * 2.0;

    let unreachable = vec3(0.0, 0.0, 1.0);
    let right = normalize(cross(forward, unreachable));
    let up    = normalize(cross(right,   forward));
    var pixel_pos = ray.origin + forward;


    let aa_pixel = vec2f(pixel) + vec2f(rand(), rand());
    let aspect = f32(globals.res.x) / f32(globals.res.y);

    pixel_pos += right * (aa_pixel.x / f32(globals.res.x) - 0.5) * fov_factor * aspect;
    pixel_pos += up    * (0.5 - aa_pixel.y / f32(globals.res.y)) * fov_factor;
    
    // "bloom"
    // let a = rand() * pi * 2.0;
    // let m = rand();
    // pixel_pos += right * aspect * cos(a) * pow(m, 150.0);
    // pixel_pos += up             * sin(a) * pow(m, 150.0);

    let aperture_radius = globals.scene.camera.aperture;
    ray.dir  = normalize(pixel_pos - ray.origin);

    let aperture = aperture_radius * rand_disk();

    ray.origin += right * aperture.x;
    ray.origin += up * aperture.y;

    pixel_pos +=  ray.dir * (globals.scene.camera.focus - 1.0);
    ray.dir  = normalize(pixel_pos - ray.origin);
    
    ray.idir = 1.0 / ray.dir;

    return ray;
}

fn project_to_hemisphere(dir: vec3f, normal: vec3f) -> vec3f {
    if dot(dir, normal) < 0.0 {
        return normalize(dir - dot(dir, normal) * normal);
    } else {
        return dir;
    }
}

fn sample_lambert(normal: vec3f) -> vec3f {
    // from raytracing in one weekend
    return normalize(normal + rand_sphere());
}

// cosign hemisphere sampling pdf
fn sample_lambert_pdf(wi: vec3f, n: vec3f) -> f32 {
    return max(0.0, dot(wi, n)) / pi;
}

fn evaluate_lambert(to_light: vec3f, normal:  vec3f) -> f32 {
    return max(dot(to_light, normal), 0.0) / pi;
}

// MARK: GGX
// from https://schuttejoe.github.io/post/ggximportancesamplingpart2/
fn evaluate_ggx_smith_masking(o_tangent: vec3f, a2: f32) -> f32 {
    let dotNV = o_tangent.z;
    let denomC = sqrt(a2 + (1.0 - a2) * dotNV * dotNV) + dotNV;

    return 2.0 * dotNV / denomC;
}

// from https://schuttejoe.github.io/post/ggximportancesamplingpart2/
fn ggx_smith_g_fast(i_tangent: vec3f, o_tangent: vec3f, a2: f32) -> f32 {
    let dotNL = i_tangent.z;
    let dotNV = o_tangent.z;

    let denomA = dotNV * sqrt(a2 + (1.0 - a2) * dotNL * dotNL);
    let denomB = dotNL * sqrt(a2 + (1.0 - a2) * dotNV * dotNV);

    return 2.0 * dotNL * dotNV / (denomA + denomB);
}

// https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
// walter GGX paper
fn ggx_smith_g1_general(HoV: f32, NoV: f32, a2: f32) -> f32 {
    let fac = step(0.0, HoV / NoV);
    let tan_theta_squared = 1.0 / (NoV * NoV) - 1.0;
    return fac * (2.0 / (1.0 + sqrt(1.0 + a2 * tan_theta_squared)));
}


fn evaluate_fresnel_dielectric(HoL: f32, ior_i: f32, ior_o: f32) -> f32 {
    let c = abs(HoL);
    let radicand = (ior_o * ior_o) / (ior_i * ior_i) - 1.0 + c * c;
    if radicand < 0.0 {
        return 1.0;
    }
    let g = sqrt( radicand );

    let fac = 0.5 * (((g - c) * (g - c)) / ((g + c) * (g + c)));
    let sqrt_rhs = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
    return fac * (1.0 + sqrt_rhs * sqrt_rhs);
}

fn ggx_smith_g_general(wi: vec3f, wo: vec3f, h: vec3f, n: vec3f, a2: f32) -> f32 {
    return ggx_smith_g1_general(dot(wi, h), dot(wi, n), a2) * ggx_smith_g1_general(dot(wo, h), dot(wo, n), a2);
}

// walter paper, microfacet models for refraction
// reversed convention, im using wo to refer to rays pointing back toward the camera
fn evaluate_ggx_transmission(wi: vec3f, wo: vec3f, n_facing: vec3f, a2: f32, ior_i: f32, ior_o: f32, F: f32) -> vec3f {

    // they expect surface normals point into the medium with the lower index of refraction
    let n = n_facing * sign(ior_i - ior_o);

    var h = -normalize(wi * ior_i + wo * ior_o);

    let NoH = dot(h, n);
    let NoL = dot(n, wi);
    let NoV = dot(n, wo);
    let HoV = dot(wo, h);
    let HoL = dot(wi, h);

    let D = ggx_d(NoH, a2);

    let G = ggx_smith_g_general(wi, wo, h, n, a2);
    
    let denom_sqrt = ior_i * HoL + ior_o * HoV;
    
    if denom_sqrt == 0.0 {
        return vec3f(0.0);
    }
    
    let factor = (abs(HoL) * abs(HoV)) / max(abs(NoL) * abs(NoV), 1e-8);

    return vec3f(factor * (ior_i * ior_i * D * (1.0 - F) * G) / (denom_sqrt * denom_sqrt ));
}

fn evaluate_fresnel_schlick(normal: vec3f, view: vec3f, f0: vec3f) -> vec3f {
    return saturate(f0 + (1.0 - f0) * pow(1.0 - dot(normal, view), 5.0));
}

// one_minus_NoH_squared from
// https://github.com/google/filament/blob/main/shaders/src/surface_brdf.fs
// for floating point precision concerns at low roughness
fn ggx_smith_d_precise(h_tangent: vec3f, a2: f32) -> f32 {
    let NoH = saturate(h_tangent.z);
    let NxH = cross(vec3f(0.0, 0.0, 1.0), h_tangent);
    let one_minus_NoH_squared = dot(NxH, NxH);
    let D_denom_sqr = (NoH * NoH) * a2 + one_minus_NoH_squared;
    return a2 / (D_denom_sqr * D_denom_sqr * pi);
}

// https://jcgt.org/published/0007/04/01/paper.pdf
fn sample_ggx_smith_vndf(view_tangent: vec3f, roughness: f32) -> vec3f {
    let a = roughness * roughness;

    // hemisphere scaled by roughness
    let vh = normalize(vec3f(a, a, 1) * view_tangent);

    let len_2 = vh.x * vh.x + vh.y * vh.y;
    let t1 = select(vec3f(1, 0, 0), vec3f(-vh.y, vh.x, 0) * inverseSqrt(len_2), len_2 > 0);
    let t2 = cross(vh, t1);

    let r = sqrt(rand());
    let phi = 2.0 * pi * rand();

    let c1  = r * cos(phi);
    var c2  = r * sin(phi);
    let s = 0.5 * (1.0 + vh.z);

    c2 = (1.0 - s) * sqrt(1.0 - c1 * c1) + s * c2;

    // re-project back to hemisphere
    let nh = c1 * t1 + c2 * t2 + sqrt(max(0.0, 1.0 - c1 * c1 - c2 * c2)) * vh;

    let ne = normalize(vec3f(a * nh.x, a * nh.y, max(0.0, nh.z)));
    return ne;
}

/// returns the pdf with respect to wi_tangent ( reflect(-wo_tangent, h_tangent) )
fn sample_ggx_smith_vndf_reflection_pdf(wo_tangent: vec3f, h_tangent: vec3f, r: f32) -> f32 {
    let a2 = r * r * r * r;
    
    let D = ggx_smith_d_precise(h_tangent, a2);
    
    if wo_tangent.z <= 0.0 || h_tangent.z <= 0.0 {
        return 0.0;
    }
    let G1 = evaluate_ggx_smith_masking(wo_tangent, a2);
    // let pdf_h = G1 * max(0.0, dot(wo_tangent, h_tangent)) * D / wo_tangent.z;
    // let reflect_jacobian = 1.0 / (4.0 * saturate(dot(wo_tangent, h_tangent)));
    let pdf_h_times_jacobian = G1 * D / (wo_tangent.z * 4.0);

    return pdf_h_times_jacobian;
}

fn invalid_refraction(v: vec3f) -> bool {
    return v.x + v.y + v.z == 0.0;
}

// for refract(incident_ray, normal, eta)
// wo is my incident ray (from the camera to the surface)
fn refraction_jacobian(ior_i_ray: f32, ior_o_ray: f32, i_ray_dot_h: f32, o_ray_dot_h: f32) -> f32 {
    let denom_sqrt = ior_o_ray * o_ray_dot_h + ior_i_ray * i_ray_dot_h;
    let num = ior_o_ray * ior_o_ray * abs(o_ray_dot_h);
    return num / (denom_sqrt * denom_sqrt);
}

/// D_Ve(Ne) = G1(Ve) * max(0, dot(Ve, Ne)) * D(Ne) / Ve.z
fn sample_ggx_smith_vndf_pdf(h: vec3f, n: vec3f, v: vec3f, a2: f32) -> f32 {
    let NoH = dot(n, h);
    let VoH = dot(v, h);
    let VoN = dot(v, n);

    let tan_theta_squared = 1.0 / (NoH * NoH) - 1.0;
    let num = a2 * step(0.0, NoH);
    let d_denom_sqrt = NoH * NoH * (a2 + tan_theta_squared);
    let D = num / (pi * d_denom_sqrt * d_denom_sqrt);

    let G1 = ggx_smith_g1_general(VoH, VoN, a2);

    return G1 * max(0.0, VoH) * D / VoN;
}


fn sample_ndf(a: f32) -> vec3f {
    let e1 = rand();
    let e2 = rand();
    let theta = atan((a * sqrt(e1)) / sqrt(1.0 - e1));
    let phi = 2.0 * pi * e2;
    let st = sin(theta);
    let ct = cos(theta);
    return vec3f(st * cos(phi), st * sin(phi), ct);
}

fn ggx_d(NoH: f32, a2: f32) -> f32 {
    let tan_theta_squared = 1.0 / (NoH * NoH) - 1.0;
    let num = a2 * step(0.0, NoH);
    let d_denom_sqrt = NoH * NoH * (a2 + tan_theta_squared);
    return num / (pi * d_denom_sqrt * d_denom_sqrt);
}

fn sample_ndf_pdf(NoH: f32, a2: f32) -> f32 {
    return ggx_d(NoH, a2) * abs(NoH);
}


/// returns the pdf with respect to the refracted direction wi_tangent
/// wo_tangent is the incident ray from the camera
fn sample_ggx_smith_vndf_refraction_pdf(wi_tangent: vec3f, wo_tangent: vec3f, ior_i: f32, ior_o: f32, roughness: f32) -> f32 {
    let a2 = roughness * roughness * roughness * roughness;

    // 1. Calculate the REQUIRED half-vector for this refraction event.
    var h_required = -normalize(wi_tangent * ior_i + wo_tangent * ior_o) * sign(ior_i - ior_o);

    let n = vec3f(0.0, 0.0, 1.0);

    let NoH = dot(n, h_required);
    let VoH = dot(wo_tangent, h_required);
    let VoN = dot(wo_tangent, n);

    let LoH = dot(wi_tangent, h_required);

    let D = ggx_d(NoH, a2);

    let eta = ior_o / ior_i;
    let denom = pow(LoH + VoH * eta, 2.);
    let jacobian = abs(LoH) / denom;

    let G1 = ggx_smith_g1_general(VoH, VoN, a2);

    let pdf = G1 * max(0.0, VoH) * D * jacobian / VoN;

    return pdf;


}

const DEBUG: bool = true;

/// convert a direction in my coordinate space into the space used by the reference GLTF viewer
fn compare_gltf_space(v: vec3f) -> vec3f {
   return v.xzy * vec3f(-1.0, 1.0, 1.0);
}

fn background_volume() -> GpuVolume {
    return GpuVolume ( vec3f(0), 1.0 );
}


fn sample_metallic_bsdf(wo_tangent: vec3f, h_tangent: vec3f, sample: ExtSample) -> vec3f {
    return reflect(-wo_tangent, h_tangent);
}

fn sample_metallic_bsdf_pdf(wi_tangent: vec3f, wo_tangent: vec3f, sample: ExtSample) -> f32 {
    let h_tangent = normalize(wo_tangent + wi_tangent);
    if h_tangent.z <= 0.0 || wo_tangent.z * wi_tangent.z < 0.0 {
        return 0.0;
    }

    return sample_ggx_smith_vndf_reflection_pdf(wo_tangent, h_tangent, sample.metallic_roughness.g);
}

fn sample_dielectric_bsdf(wo_tangent: vec3f, h_tangent: vec3f, ior_i: f32, ior_o: f32, sample: ExtSample) -> vec3f {
    let fresnel = evaluate_fresnel_dielectric(dot(wo_tangent, h_tangent), ior_o, ior_i);

    if rand() < fresnel {
        return reflect(-wo_tangent, h_tangent);
    } else {
        if rand() < sample.transmission {
            var wi_tangent = refract(-wo_tangent, h_tangent, ior_o / ior_i);
            if all(wi_tangent == vec3f(0)) {
                wi_tangent = reflect(-wo_tangent, h_tangent);
            }
            return wi_tangent;
        } else {
            return sample_lambert(vec3f(0.0, 0.0, 1.0));
        }

    }
}

fn sample_dielectric_bsdf_pdf(wi_tangent: vec3f, wo_tangent: vec3f, ior_i: f32, ior_o: f32, sample: ExtSample) -> f32 {
    let transmitted = wi_tangent.z < 0.0;
    var h_tangent: vec3f;
    
    if transmitted {
        h_tangent = -normalize(wi_tangent * ior_i + wo_tangent * ior_o) * sign(ior_i - ior_o);
    } else {
        h_tangent = normalize(wi_tangent + wo_tangent);
    }

    if h_tangent.z < 0.0 {

        return 0.0;
    }

    let fresnel = evaluate_fresnel_dielectric(dot(wo_tangent, h_tangent), ior_o, ior_i);

    var diffuse_pdf = sample_lambert_pdf(wi_tangent, vec3f(0.0, 0.0, 1.0));
    var transmission_pdf = sample_ggx_smith_vndf_refraction_pdf(wi_tangent, wo_tangent, ior_i, ior_o, sample.metallic_roughness.g);
    var specular_pdf = sample_ggx_smith_vndf_reflection_pdf(wo_tangent, h_tangent, sample.metallic_roughness.g);

    if transmitted {
        diffuse_pdf = 0.0;
        specular_pdf = 0.0;
    } else {
        transmission_pdf = 0.0;
    }


    let pdf = mix(mix(diffuse_pdf, transmission_pdf, sample.transmission), specular_pdf, fresnel);

    return pdf;

}


//MARK: Surface Hit
fn sample_bsdf(wo: vec3f, ior_o: f32, ior_i: f32, sample: ExtSample, lighting: ptr<function, vec3f>, pdf: ptr<function, f32>, bsdf: ptr<function, vec3f>) -> vec3f {
    
    var debug_color = vec3f(0);
    *pdf = 0.0;
    *bsdf = vec3f(0.0);

    let tbn = orthonormal_basis(sample.normal); //* sign(ior_i - ior_o)
    let wo_tangent = normalize(wo * tbn);
    let h_tangent = sample_ggx_smith_vndf(wo_tangent, sample.metallic_roughness.g);

    var metallic_chance = sample.metallic_roughness.b;


    // debug_color = vec3f(normal_tangent * 0.5 + 0.5);
    if globals.debug_mode == 1 {
        debug_color = sample.albedo.rgb;
    }
    if globals.debug_mode == 2 {
        debug_color = compare_gltf_space(sample.normal) * 0.5 + 0.5;
    }
    if globals.debug_mode == 4 {
        debug_color = compare_gltf_space(sample.tangent) * 0.5 + 0.5;
    }
    if globals.debug_mode == 5 {
        debug_color = clamp(vec3f(-sample.t_sign, sample.t_sign, 0.0), vec3f(0), vec3f(1));
    }

    var wi_tangent: vec3f;  
    if rand() < metallic_chance {
        wi_tangent = sample_metallic_bsdf(wo_tangent, h_tangent, sample);
    } else {
        wi_tangent = sample_dielectric_bsdf(wo_tangent, h_tangent, ior_i, ior_o, sample);
    }
     
    var wi = tbn * wi_tangent;

    *pdf = sample_bsdf_pdf(wi, wo, ior_i, ior_o, sample);
    *bsdf = evaluate_bsdf(wi, wo, ior_i, ior_o, sample);

     
    if DEBUG && globals.debug_mode != 0 {
        *lighting = debug_color;
    }



    return wi;
}

// what is the probability the brdf sampling returned wi given the surface properties
// maybe not 100% accurate, hopefully good enough for MIS weighting
fn sample_bsdf_pdf(wi: vec3f, wo: vec3f, ior_i: f32, ior_o: f32, sample: ExtSample) -> f32 {
    if dot(wo, sample.normal) < 0.0 {
        return 0.0;
    }

    let tbn = orthonormal_basis(sample.normal);
    let wi_tangent = wi * tbn;
    let wo_tangent = wo * tbn;

    let metal = sample_metallic_bsdf_pdf(wi_tangent, wo_tangent, sample);
    let dielectric = sample_dielectric_bsdf_pdf(wi_tangent, wo_tangent, ior_i, ior_o, sample);
    
    return mix(dielectric, metal, sample.metallic_roughness.b);
}

/// wo_tangent is the vector pointing toward the viewer
/// wi_tangent is the vector pointing toward the light
fn evaluate_ggx(wo_tangent: vec3f, wi_tangent: vec3f, F: vec3f, a2: f32) -> vec3f {
    let h_tangent = normalize(wi_tangent + wo_tangent);

    let NoH = max(h_tangent.z, 0.0);
    let NoV = max(wo_tangent.z, 0.0);
    let NoL = max(wi_tangent.z, 0.0);
    
    let D = ggx_smith_d_precise(h_tangent, a2);
    let G = ggx_smith_g_fast(wo_tangent, wi_tangent, a2);

    let denom = 4.0 * NoV * NoL;
    if denom <= 0.0 {
        return vec3f(0);
    }

    return (D * F * G) / denom;
}

fn evaluate_metal(wi_tangent: vec3f, wo_tangent: vec3f, sample: ExtSample) -> vec3f {
    let h_tangent = normalize(wi_tangent + wo_tangent);
    if h_tangent.z <= 0.0 || wi_tangent.z * wo_tangent.z <= 0.0 {
        return vec3f(0);
    }
    let fresnel = evaluate_fresnel_schlick(h_tangent, wo_tangent, sample.albedo.rgb);

    let a = sample.metallic_roughness.g * sample.metallic_roughness.g;

    return evaluate_ggx(wo_tangent, wi_tangent, fresnel, a * a);
}

fn evaluate_dielectric(wi_tangent: vec3f, wo_tangent: vec3f, ior_i: f32, ior_o: f32, sample: ExtSample) -> vec3f {
    var h_tangent: vec3f;
    var transmitted = false;

    if wi_tangent.z > 0.0 {
        h_tangent = normalize(wi_tangent + wo_tangent);
    } else {
        transmitted = true;
        h_tangent = -normalize(wi_tangent * ior_i + wo_tangent * ior_o) * sign(ior_i - ior_o);
    }

    let fresnel = evaluate_fresnel_dielectric(dot(wo_tangent, h_tangent), ior_o, ior_i);

    let a = sample.metallic_roughness.g * sample.metallic_roughness.g;
    let a2 = a * a;

    var specular = evaluate_ggx(wo_tangent, wi_tangent, vec3f(fresnel), a2);
    var diffuse = sample.albedo.rgb / pi * (1.0 - fresnel);
    var transmission = sample.albedo.rgb * evaluate_ggx_transmission(wi_tangent, wo_tangent, vec3f(0.0, 0.0, 1.0), a2, ior_i, ior_o, fresnel);

    if transmitted {
        specular = vec3f(0);
        diffuse = vec3f(0);
    } else {
        transmission = vec3f(0);
    }
    
    // all BXDF's contain the fresnel term already
    return mix(diffuse, transmission, sample.transmission) + specular;
}

fn evaluate_bsdf(wi: vec3f, wo: vec3f, ior_i: f32, ior_o: f32, sample: ExtSample) -> vec3f {
    let metallic = sample.metallic_roughness.b;
    let f0 = mix(vec3f(0.04), sample.albedo.rgb, metallic);

    let tbn = orthonormal_basis(sample.normal);
    let wi_tangent = normalize(wi * tbn);
    let wo_tangent = normalize(wo * tbn);

    let dielectric = evaluate_dielectric(wi_tangent, wo_tangent, ior_i, ior_o, sample);
    let metal = evaluate_metal(wi_tangent, wo_tangent, sample);
    return mix(dielectric, metal, metallic);

} 

fn mis_power_heuristic(a: f32, b: f32, a_prob: f32, b_prob: f32) -> f32 {
    let ap = a * a_prob;
    let bp = b * b_prob;
    if (ap <= 0.0) {
        return 0.0;
    } else if (bp <= 0.0) {
        return 1.0;
    }
    
    // return a_prob;
    return ap  / (ap + bp);
    //return (ap * ap) / (ap * ap + bp * bp);
}

fn mis_power_heuristic_3(a: f32, b: f32, c: f32, a_prob: f32, b_prob: f32, c_prob: f32) -> f32 {
    let ap = a * a_prob;
    let bp = b * b_prob;
    let cp = c * c_prob;
    if ap <= 0.0 {
        return 0.0;
    } else if bp <= 0.0 && cp <= 0.0 {
        return 1.0;
    }
    
    // return a_prob;
    return ap  / (ap + bp + cp);
    //return (ap * ap) / (ap * ap + bp * bp + cp * cp);
}

const MESH_CHANCE: f32 = 0.5;

struct LightSample {
    wi: vec3f,
    t_max: f32,
    contrib: vec3f,
    pdf: f32,
}

fn sample_light(reference: vec3f) -> LightSample {
    
    let r = rand();
    var prob = 0.0;
    var pdf = 0.0;
    var wi = vec3f(0,0,1);
    var t_max = 0.0;
    var contrib = vec3f(0);

    // sample NEE shadow ray
    if r < MESH_CHANCE {
        prob = MESH_CHANCE;

        var prim_idx: i32;
        var tri_idx: i32;
        
        let mesh_point = sample_mesh_light(reference, &pdf, &prim_idx, &tri_idx);
        
        wi = normalize(mesh_point - reference);

        var mesh_ray: Ray;
        mesh_ray.origin = reference;
        mesh_ray.dir = wi;
        mesh_ray.idir = 1.0 / mesh_ray.dir;

        let mesh_hit = trace_transformed_tri(mesh_ray, prim_idx, tri_idx);

        t_max = distance(reference, mesh_point) - 0.003;
        contrib = sample_emission(mesh_hit);

    } else if globals.scene.mesh_light_count > 0 {
        prob = 1.0 - MESH_CHANCE;

        wi = normalize(sample_env_map());
        t_max = 99999999999.0;
        contrib = evaluate_env_map(wi).rgb;
        pdf = sample_env_map_pdf(wi);
    }

    return LightSample(wi, t_max, contrib, pdf * prob);
}

fn sample_light_pdf(reference: vec3f, point: vec3f, prim: i32, tri: i32) -> f32 {
    if (tri > 0) {
        // a physical light, can't have been the env sampler
        return sample_mesh_light_pdf(prim, tri, point, reference) * MESH_CHANCE;
    } else {
        // a ray miss light, can't have been the mesh sampler
        return sample_env_map_pdf(normalize(point - reference)) * (1.0 - MESH_CHANCE);
    }
}


@compute
@workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
if (id.x < globals.res.x && id.y < globals.res.y) {

    var lighting   = vec3f(0);
    var throughput = vec3f(1);

    seed = hash21(vec2u(hash21(id.xy), globals.frame));
    // spin the rng to improve the quality of the first samples
    rand(); rand();

    var NEE_PROB = 0.2;
    var RR_PROB = 0.6;

    var ray_volume = background_volume();
    ray_volume.ior = 1.0;
    
    // constant, but no way to select() in const expressions apparently
    var NUM_BOUNCES = 12;
    if DEBUG && globals.debug_mode != 0 {
        if globals.debug_mode != 9 {
            NUM_BOUNCES = 1;
        }
        if globals.debug_mode == 8 {
            NUM_BOUNCES = 3;
        }
    }

    // debug variable
    var right = id.x > globals.res.x / 2;
    right = false;

    if right {
        NEE_PROB = 0.5;
    } else {
        NEE_PROB = 0.5;
    }

    var last_prim = -1;
    var last_tri = -1;

    var ray = camera_ray(id.xy);

    var bsdf_pdf = 1.0;
    var bsdf_mis_weight = 1.0;
    for (var i = 0; i < NUM_BOUNCES; i++) {
        let hit = trace(ray);
        let point = ray.origin + ray.dir * hit.t;
        if hit.idx == -1 {
            let nee_pdf = sample_light_pdf(ray.origin, point, hit.prim_idx, hit.idx);

            if i > 0 {
                bsdf_mis_weight = mis_power_heuristic(bsdf_pdf, nee_pdf, 1.0, 1.0);
            }

            if globals.debug_mode != 8 {
                lighting += throughput * bsdf_mis_weight * evaluate_env_map(ray.dir).rgb;
            }
            
            break;
        }

        let sample = sample_hit(hit);

        var hit_ior = hit.material.volume.ior;
        if hit.backface {
            hit_ior = background_volume().ior;
        }

        if hit_ior == ray_volume.ior {
            hit_ior = 1.6 * ray_volume.ior;
        }

        if globals.debug_mode != 9 {
            throughput *= exp(-ray_volume.absorption * hit.t);
        }

        let nee_pdf = sample_light_pdf(ray.origin, point, hit.prim_idx, hit.idx);
        if i != 0 {
            bsdf_mis_weight = mis_power_heuristic(bsdf_pdf, nee_pdf, 1.0, 1.0);
        }
            
        if nee_pdf != 0.0 {
            lighting += throughput * sample.emissive * bsdf_mis_weight;
        }
        
        let wo = -ray.dir;

        // sample NEE shadow ray
        
        {
            let light = sample_light(ray.origin + ray.dir * hit.t);
            var nee_ray: Ray;
            nee_ray.origin = ray.origin + ray.dir * hit.t;
            nee_ray.origin += light.wi * 0.001;
            nee_ray.dir = light.wi;
            nee_ray.idir = vec3f(1) / nee_ray.dir;

            let nee_ray_bsdf_pdf = sample_bsdf_pdf(light.wi, wo, hit_ior, ray_volume.ior, sample);
            let nee_mis_weight = mis_power_heuristic(light.pdf, nee_ray_bsdf_pdf, 1.0, 1.0);
            let nee_bsdf = evaluate_bsdf(light.wi, wo, hit_ior, ray_volume.ior, sample);
            if !trace_shadow(nee_ray, light.t_max) {
                lighting += throughput * nee_bsdf * max(dot(light.wi, sample.normal), 0.0) * light.contrib * nee_mis_weight / light.pdf;
            }
        }

        RR_PROB = clamp(1.0 - max(throughput.x, max(throughput.y, throughput.z)), 0.0, 0.95);
        if i < 3 {
            RR_PROB = 0.0;
        }
        if rand() < RR_PROB {
            break;
        } else {
            throughput /= (1.0 - RR_PROB);
        }

        if globals.debug_mode == 8 {
            if (i == 2) {
                lighting = vec3f(0);
                if (nee_pdf > 0.0) {
                    lighting = sample.emissive * vec3f(bsdf_mis_weight);
                }
                break;
            }
        }

        // sample BSDF continuation
        var bsdf: vec3f;
        var wi = sample_bsdf(wo, ray_volume.ior, hit_ior, sample, &lighting, &bsdf_pdf, &bsdf);

        if bsdf_pdf > 0.0 {
            throughput /= bsdf_pdf;
        } else {
            throughput *= 0.0;
        }

        throughput *= bsdf * abs(dot(wi, sample.normal));
        
        ray.origin += ray.dir * hit.t;
        ray.dir = wi;
        ray.idir = 1.0 / ray.dir;

        if dot(wi, sample.normal) > 0.0 {
            ray.origin += ray.dir * 0.001;
        } else {
            ray.origin += ray.dir * 0.001;
            if hit.backface {
                ray_volume = background_volume();
            } else {
                ray_volume = hit.material.volume;
            }
        }
        if globals.debug_mode == 6 {
            let backup = seed;
            seed = bitcast<u32>(hit.prim_idx) + 777;
            rand(); rand();
            lighting = rand_color() * evaluate_lambert(wo, sample.normal) * pi;
            seed = backup;
        }

        if globals.debug_mode == 7 {
            let backup = seed;
            seed = bitcast<u32>(hit.idx) + 777;
            rand(); rand();
            lighting = rand_color() * evaluate_lambert(wo, hit.normal) * pi;
            seed = backup;
        }

        last_prim = hit.prim_idx;
        last_tri = hit.idx;
    }

    if DEBUG {
        if globals.debug_mode == 3 {
            lighting = magma_quintic(debug / 256.0);
        } else if globals.debug_mode != 0 && globals.debug_mode != 9 {
            lighting *= 2.0;
        }
        
    }

    if (debug < 0.0) {
        lighting = vec3f(1.0, 1.0, 0.0);
    }



    if (globals.reject_hist > 0) {
        screen[id.x + globals.res.x * id.y] = vec4f(max(lighting, vec3f(0)), 1.0);
    } else {
        // clamp NaNs
        screen[id.x + globals.res.x * id.y] += vec4f(max(lighting, vec3f(0)), 1.0);
    }
}
}


@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    // procedural fullscreen triangle
    let x = f32(i32(i) - 1) * 5.0;
    let y = f32(i32(i & 1u) * 2 - 1) * 5.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

fn tonemap(in: vec3f) -> vec3f {
    // return in;
    return vec3f(pow(1.0 - pow(vec3f(0.25), in), vec3f(1.1)));
}

fn tonemap_pbr_neutral(color_in: vec3f) -> vec3f {
    var color = color_in;
    let startCompression = 0.8 - 0.04;
    let desaturation = 0.15;

    let x = min(color.r, min(color.g, color.b));
    var offset = 0.04;
    if x < 0.08 {
        offset = x - 6.25 * x * x;
    }
    color -= offset;

    let peak = max(color.r, max(color.g, color.b));
    if peak < startCompression {
        return color;
    }

    let d = 1. - startCompression;
    let newPeak = 1. - d * d / (peak + d - startCompression);
    color *= newPeak / peak;

    let g = 1. - 1. / (desaturation * (peak - newPeak) + 1.);
    return mix(color, newPeak * vec3f(1, 1, 1), g);
}

@fragment
fn fs_main(@builtin(position) p: vec4f) -> @location(0) vec4<f32> {
    let id = vec2u(p.xy);
    if (id.x >= globals.res.x || id.y >= globals.res.y) {
        return vec4f(0.5, 0.1, 0.1, 1.0);
    }
    
    let scr = screen[id.x + globals.res.x * id.y];

    let uv = p.xy / vec2f(f32(globals.res.x), f32(globals.res.y));

    // divide total by number of samples
    var col = scr.rgb / scr.a;

    if !(DEBUG && globals.debug_mode != 0) {
        col = tonemap_pbr_neutral(col * globals.scene.camera.exposure);
    }
    
    // col = to_linear(sample_texture(primitives[1].material.albedo, uv).rgb);
    // col = pow(col, vec3f(1.0 / 2.2));
    return vec4f(col, 1.0);
}
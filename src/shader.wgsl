@group(0) @binding(0) var<uniform> globals : FrameUniforms;

@group(1) @binding(0) var<storage, read_write> triangles :      array<Tri>;
@group(1) @binding(1) var<storage, read_write> tri_exts :       array<TriExt>;
@group(1) @binding(2) var<storage, read_write> bvh :            array<BvhNode>;
@group(1) @binding(3) var<storage, read_write> screen :         array<vec4f>;
@group(1) @binding(4) var<storage, read_write> texture_data :   array<u32>;
@group(1) @binding(5) var<storage, read_write> primitives :     array<Primitive>;
@group(1) @binding(6) var                      env_map:         texture_2d<f32>;

const pi = 3.141592654;

const FORWARD = vec3f(1.0, 0.0, 0.0);
const UP = vec3f(0.0, 0.0, 1.0);
const RIGHT = vec3f(0.0, -1.0, 0.0);

const NUM_TEXCOORDS = 2;

// const SUN_DIR = vec3f(0.707106781187, 0.0 , 0.707106781187);
const TO_SUN_VAL = vec3f(0.5, 0.4, 1.49);

const TO_SUN_DIR = TO_SUN_VAL / sqrt(TO_SUN_VAL.x * TO_SUN_VAL.x + TO_SUN_VAL.y * TO_SUN_VAL.y + TO_SUN_VAL.z * TO_SUN_VAL.z);

const SUN_COL = vec3f(1.0, 0.5, 0.3) * 0.0;
const EXPOSURE = 1.0 / 1.0;


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

struct Material {
    albedo:             GpuTextureRef,
    emissive:           GpuTextureRef,
    normal:             GpuTextureRef,
    metallic_roughness: GpuTextureRef,

    albedo_factor:      vec4f,

    emissive_factor:    vec3f,

    normal_scale:       f32,

    albedo_texcoord:    u32,
    emissive_texcoord:  u32,
    normal_texcoord:    u32,
    metal_r_texcoord:   u32,

    metallic_factor:    f32,
    roughness_factor:   f32,
    id:                 u32,
    _pad:               u32,
}

const DEFAULT_MATERIAL = Material (
    GpuTextureRef (0, 0), // albedo
    GpuTextureRef (0, 0), // emissive
    GpuTextureRef (0, 0), // normal
    GpuTextureRef (0, 0), // metallic_roughness

    vec4f(1.0, 1.0, 1.0, 1.0), // albedo factor

    vec3f(0.0, 0.0, 0.0),   // emissive factor

    1.0,    // normal_scale

    0, // albedo_texcoord
    0, // emissive_texcoord
    0, // normal_texcoord
    0, // metal_r_texcoord

    0.0, // metallic_factor
    0.0, // roughness_factor
    0, // id
    0, // padding
);

struct Primitive {
    transform:      mat4x4f,
    inv_transform:  mat4x4f,
    material:       Material,
    bvh_idx:        u32,
    _pad:           u32,
}

struct Scene {
    point_lights:           array<PointLight, 12>,
    directional_lights:     array<DirectionalLight, 4>,
    camera:                 Camera,
    tri_count:              u32,
    num_point_lights:       u32,
    num_directional_lights: u32,
}

struct FrameUniforms {
    scene:          Scene,
    res:            vec2u,
    frame:          u32,
    time:           f32,
    reject_hist:    u32,
    node_count:     u32,
    prim_count:     u32,
}

// struct FrameUniforms {
//     camera: Camera,
//     res:    vec2u,
//     frame:  u32,
//     tri_count: u32,
//     time:   f32,
//     reject_hist: u32,
//     num_point_lights: u32,
//     num_directional_lights: u32,
//     point_lights: array<PointLight, 12>,
//     directional_lights: array<DirectionalLight, 4>,
// }

fn tc_size(tc: GpuTextureRef) -> vec2u {
    return vec2u(tc.size >> 16u, tc.size & 0xFFFFu);
}
  
fn sample_texture(tex: GpuTextureRef, tc: vec2f) -> vec4f {
    if tex.size == 0 {
        return dummy_texture(tc);
    }

    // // visualize texture IDs
    // let backup = seed;
    // seed = tc.size;
    // let col = rand_color();
    // seed = backup;
    // return vec4f(col.r, col.g, col.b, 1.0);

    let size = tc_size(tex);
    let texel_pos = vec2u(fract(tc) * vec2f(size));
    let texel = texture_data[tex.offset + texel_pos.y * size.x + texel_pos.x];
    return unpack_rgba8(texel);
}

struct GpuVertexExt {
    texcoords: array<vec2f, NUM_TEXCOORDS>,
    normal: vec2f,
    color: u32,
    _pad: f32
}

struct ExtSample {
    color: vec4f,
    texcoords: array<vec2f, NUM_TEXCOORDS>,
    normal: vec3f,
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
fn unpack_vec3_octrahedral(f_in: vec2f) -> vec3f {
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


struct TriExt {
    vertices: array<GpuVertexExt, 3>
}

fn zeroed_ext_sample() -> ExtSample {
    var s: ExtSample;
    s.color = vec4f(0.0, 0.0, 0.0, 0.0);
    s.normal = vec3f(0.0, 0.0, 0.0);
    for (var i = 0; i < NUM_TEXCOORDS; i++) {
        s.texcoords[i] = vec2f(0.0, 0.0);
    }
    return s;
}

// red checkerboard for missing textures
fn dummy_texture(uv: vec2f) -> vec4f {



    const scale = 256.0;
    let checker = f32((u32(uv.x * scale) + u32(uv.y * scale + 1.0)) % 2u);
    var col = mix(vec3f(0.8, 0.3, 0.3), vec3f(0.8, 0.3, 0.3) * 0.5, checker);
    // col = vec3f(uv.x, uv.y, 0.0);
    // col = vec3f(checker) * 0.15 + 0.2;
    return vec4f(col.r, col.g, col.b, 1.0);
    // return vec4f(checker, uv.x, uv.y, 1.0);
}

// barycentric interpolation of vertex attributes
fn tri_ext_interpolate(tri: ptr<function, TriExt>, bary: vec3f) -> ExtSample {
    var res = zeroed_ext_sample();

    // cant loop: cannot index into value of type `vec3<f32>`

    res.color += bary.x * unpack_rgba8((*tri).vertices[0].color);
    res.texcoords[0] += bary.x * (*tri).vertices[0].texcoords[0];
    res.texcoords[1] += bary.x * (*tri).vertices[0].texcoords[1];
    res.normal += bary.x * unpack_vec3_octrahedral((*tri).vertices[0].normal);

    res.color += bary.y * unpack_rgba8((*tri).vertices[1].color);
    res.texcoords[0] += bary.y * (*tri).vertices[1].texcoords[0];
    res.texcoords[1] += bary.y * (*tri).vertices[1].texcoords[1];
    res.normal += bary.y * unpack_vec3_octrahedral((*tri).vertices[1].normal);

    res.color += bary.z * unpack_rgba8((*tri).vertices[2].color);
    res.texcoords[0] += bary.z * (*tri).vertices[2].texcoords[0];
    res.texcoords[1] += bary.z * (*tri).vertices[2].texcoords[1];
    res.normal += bary.z * unpack_vec3_octrahedral((*tri).vertices[2].normal);

    res.normal = normalize(res.normal);
   
   //  res.tex0 = vec4f(tc0, 0.0, 1.0);
    return res;
}





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

////////////// bvh node //////////////
struct BvhNode {
    aabb: Aabb,

    /// The index of the left child if count is 0. First triangle index otherwise
    first: u32,

    /// the number of triangles in the node
    count: u32,
}

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


////////////// ray //////////////
struct Ray {
    origin: vec3f,
    dir: vec3f,
    idir: vec3f,
}

////////////// hit //////////////
struct Hit {
    t: f32,
    idx: i32,
    prim_idx: i32,
    material: Material,
    normal: vec3f,
    bary: vec3f,
}

// https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
// epsilon stolen from https://www.shadertoy.com/view/wlsfRs
fn intersect (ray: Ray, tri: Tri) -> f32 {
    let edge1 = tri.d1.xyz - tri.d0.xyz;
    let edge2 = tri.d2.xyz - tri.d0.xyz;
    let h = cross( ray.dir, edge2 );
    let a = dot( edge1, h );
    if (a > -0.000002 && a < 0.000002) {
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
    if (t > 0.000002) {
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

// modified version of intersect() to return more info
//     from https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
fn intersect_full(ray: Ray, idx: i32) -> Hit {
    let tri = triangles[idx];
    var hit = Hit(0.0, -1, -1, DEFAULT_MATERIAL, vec3f(0.0, 0.0, 1.0), vec3f(0.333, 0.333, 0.333));

    let edge1 = tri.d1.xyz - tri.d0.xyz;
    let edge2 = tri.d2.xyz - tri.d0.xyz;

    hit.normal = normalize(cross(edge1, edge2));
    hit.normal *= -sign11(dot(hit.normal, ray.dir));

    let h = cross( ray.dir, edge2 );
    let a = dot( edge1, h );
    if (a > -0.000002 && a < 0.000002) {
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
    if (t <= 0.000002) {
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

    if all(ray.origin >= bmin) && all(ray.origin < bmax) {
        return 0.0;
    }

    let rmin = (bmin - ray.origin) * ray.idir;
    let rmax = (bmax - ray.origin) * ray.idir;

    let tmin = min(rmin, rmax);
    let tmax = max(rmin, rmax);

    let t0 = max(tmin.x, max(tmin.y, tmin.z));
    let t1 = min(tmax.x, min(tmax.y, tmax.z));

    if (t0 > t1 || t0 < 0.0) {
        return -1.0;
    }

    return t0;
}

var<private> debug: f32;

fn trace_bvh(ray: Ray, root: u32, t_max: ptr<function, f32>) -> i32 {
    var stack: Stack;
    stack.size = 0u;
    var node = bvh[root];
    var best_t = *t_max;
    var best_i: i32 = -1;
    if intersect_aabb(ray, node.aabb) < -0.5 {
        return best_i;
    }
    
    while (true) {
        // debug = max(debug, f32(stack.size + 1u));
        // visualize bvh steps
        debug += 1.0;


        if node.count > 0 {
            // intersect triangles of node
            for (var i = node.first; i < node.first + node.count; i++) {
                
                let t = intersect(ray, triangles[i]);
                if t >= 0.0 && t < best_t {
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
            let left  = intersect_aabb(ray, bvh[node.first + 0u].aabb);
            let right = intersect_aabb(ray, bvh[node.first + 1u].aabb);
    
            if (left < -0.5 || left > best_t) && (right < -0.5 || right > best_t) {
                if stack.size == 0u {
                    break;
                }
                node = bvh[pop(&stack)];
            } else if (left < -0.5 || left > best_t) {
                node = bvh[node.first + 1u];
            } else if (right < -0.5 || right > best_t) {
                node = bvh[node.first + 0u];
            } else if left < right {
                push(&stack, node.first + 1u);
                node = bvh[node.first + 0u];
            } else {
                push(&stack, node.first + 0u);
                node = bvh[node.first + 1u];
            }

        }
    }
    *t_max = best_t;
    return i32(best_i);
}

// true if hit, false otherwise
fn trace_bvh_shadow(ray: Ray, root: u32) -> bool {
    var stack: Stack;
    stack.size = 0u;
    var node = bvh[root];
    if intersect_aabb(ray, node.aabb) < -0.5 {
        return false;
    }
    while (true) {
        if node.count > 0 {
            for (var i = node.first; i < node.first + node.count; i++) {
                
                let t = intersect(ray, triangles[i]);
                if t >= 0.0 {
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

            let left  = intersect_aabb(ray, bvh[node.first + 0u].aabb);
            let right = intersect_aabb(ray, bvh[node.first + 1u].aabb);
    
            if left < -0.5 && right < -0.5 {
                if stack.size == 0u {
                    break;
                }
                node = bvh[pop(&stack)];
            } else if left < -0.5 {
                node = bvh[node.first + 1u];
            } else if right < -0.5 {
                node = bvh[node.first + 0u];
            } else if left < right {
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

// just loops over all primitives for now
fn trace(ray: Ray) -> Hit {
    var closest_idx = -1;
    var closest_t   = 99999999.0;
    var closest_primitive = 0u;
    debug = 0.0;
    for (var i = 0u; i < globals.prim_count; i++) {
        let scale_factor = length(transform_dir(ray.dir, primitives[i].inv_transform));
        let t_ray = transform_ray(ray, primitives[i].inv_transform);

        var new_t = closest_t * scale_factor;
        let new_idx = trace_bvh(t_ray, primitives[i].bvh_idx, &new_t);

        if new_idx >= 0 {
            closest_t = new_t / scale_factor;
            closest_idx = new_idx;
            closest_primitive = i;
        }
    }

    let t_ray_final = transform_ray(ray, primitives[closest_primitive].inv_transform);
    var hit = intersect_full(t_ray_final, closest_idx);

    // transform the hit back to world space
    hit.t = closest_t;
    hit.normal = transform_normal(hit.normal, primitives[closest_primitive].inv_transform);
    hit.prim_idx = i32(closest_primitive);
    hit.material = primitives[closest_primitive].material;
    return hit;

}

// true if hit, false otherwise
fn trace_shadow(ray: Ray) -> bool {
    for (var i = 0u; i < globals.prim_count; i++) {
        let t_ray = transform_ray(ray, primitives[i].inv_transform);
        if trace_bvh_shadow(t_ray, primitives[i].bvh_idx) {
            return true;
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

fn sample_env_map(dir: vec3f) -> vec4f {
    let uv = vec2f(atan2(dir.y, dir.x) / (pi * 2.0), asin(-dir.z) / pi) + 0.5;
    return textureLoad(env_map, vec2u(uv * vec2f(textureDimensions(env_map))), 0); 
}

fn sky(dir: vec3f) -> vec3f {
    let horizon = vec3f(1.0 - SUN_COL);
    return sample_env_map(dir).rgb * 1.0;
    // return mix(horizon, SUN_COL,pow(max(dot(dir, TO_SUN_DIR), 0.0), 3.0)) * 1.00;
    // return to_linear(dir * 0.5 + 0.5);
    // return vec3f(1.0);
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
    
    // // "bloom"
    // let a = rand() * pi * 2.0;
    // let m = rand();
    // pixel_pos += right * aspect * cos(a) * pow(m, 150.0);
    // pixel_pos += up             * sin(a) * pow(m, 150.0);
    let aperture_radius = 0.25;
    ray.dir  = normalize(pixel_pos - ray.origin);

    let aperture = aperture_radius * rand_disk();

    ray.origin += right * aperture.x;
    ray.origin += up * aperture.y;

    pixel_pos +=  ray.dir * (globals.scene.camera.focus - 1.0);
    ray.dir  = normalize(pixel_pos - ray.origin);
    
    ray.idir = 1.0 / ray.dir;

    return ray;
}

fn sample_lambert(ray: ptr<function, Ray>, normal: vec3f) {
    // from raytracing in one weekend
    (*ray).dir = normalize(normal + rand_sphere());
    (*ray).idir = vec3f(1.0) / (*ray).dir;
}

fn eval_lambert(to_light: vec3f, normal:  vec3f) -> f32 {
    return max(dot(to_light, normal), 0.0) / pi;
}

fn fresnel_schlick(normal: vec3f, view: vec3f, f0: vec3f) -> vec3f {
    return f0 + (1.0 - f0) * pow(1.0 - dot(normal, view), 5.0);
}

fn handle_surface_hit_brdf(ray: ptr<function, Ray>, hit: Hit, throughput: ptr<function, vec3f>, lighting: ptr<function, vec3f>) {
    rand();

    (*ray).origin += (*ray).dir * hit.t;
    (*ray).origin += hit.normal * 0.001;

    var emissive = vec3f(0);
    var albedo   = vec3f(0);

    // sample extra vertex data
    var ext = tri_exts[hit.idx];
    var sample = tri_ext_interpolate(&ext, hit.bary);

    if length(sample.normal)  > 0.001 {
        // sample.normal = normalize(sample.normal);
        sample.normal = transform_normal(normalize(sample.normal), primitives[hit.prim_idx].inv_transform);
    } else {
        sample.normal = hit.normal;
    }

    
    albedo = hit.material.albedo_factor.rgb;
    if hit.material.albedo.size != 0 {
        albedo *= to_linear(sample_texture(hit.material.albedo, sample.texcoords[hit.material.albedo_texcoord]).rgb);
    }

    emissive = hit.material.emissive_factor;
    if hit.material.emissive.size != 0 {
        emissive *= to_linear(sample_texture(hit.material.emissive, sample.texcoords[hit.material.emissive_texcoord]).rgb);
    }


    

    var metallic_chance = hit.material.metallic_factor;
    var roughness = hit.material.roughness_factor;
    if hit.material.metallic_roughness.size != 0 {
        let metallic_roughness = sample_texture(hit.material.metallic_roughness, sample.texcoords[hit.material.metal_r_texcoord]).gb;
        metallic_chance = metallic_roughness.y;
        roughness *= metallic_roughness.x;
    }

    let metal = rand() < metallic_chance;
    var f0 = vec3f(0.04);
    if metal {
        f0 = albedo;
    }

    let fresnel = fresnel_schlick(sample.normal, -(*ray).dir, f0);
    
    var specular = rand() < fresnel.x;

    *lighting += emissive * *throughput;
    if !specular || metal {
        *throughput *= albedo;
    }

    if specular || metal {
        (*ray).dir = reflect((*ray).dir, sample.normal);
        (*ray).dir = mix((*ray).dir, rand_hemisphere(sample.normal), roughness * roughness);
        (*ray).idir = 1.0 / (*ray).dir;
    } else {
        sample_lambert(ray, sample.normal);
    }
    

}

fn handle_miss(ray: ptr<function, Ray>, hit: Hit, throughput: ptr<function, vec3f>, lighting: ptr<function, vec3f>) {
    *lighting += *throughput * sky((*ray).dir);
}
@compute
@workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
if (id.x < globals.res.x && id.y < globals.res.y) {

    var lighting   = vec3f(0);
    var throughput = vec3f(1);

    // let samples = u32(screen[id.x + globals.res.x * id.y].a);
    seed = hash21(vec2u(hash21(id.xy), globals.frame));
    const SHADOW_PROB = 0.0;
    var ray = camera_ray(id.xy);
    for (var i = 0; i < 8; i++) {
        let hit = trace(ray);
        
        // shade(hit, ray.dir, &throughput, &lighting);

        if (hit.idx == -1) {
            handle_miss(&ray, hit, &throughput, &lighting);
            break;
        } else {
            if rand() < SHADOW_PROB {
                break;
            } else {
                handle_surface_hit_brdf(&ray, hit, &throughput, &lighting); 
            }
        }

        // ray.origin += ray.dir * hit.t + hit.normal * 0.0001;
        // 
        // if rand() < SHADOW_PROB {
            
        //     // sun shadow ray
        //     throughput /= SHADOW_PROB;

        //     ray.dir = normalize(TO_SUN_DIR + rand_sphere() * 0.01);
        //     ray.idir = vec3f(1.0) / ray.dir;
        //     let lambert = eval_lambert(TO_SUN_DIR, hit.normal);
        //     if lambert > 0.0 && !trace_shadow(ray) {
        //         lighting += throughput * SUN_COL * 12.0 * lambert;
        //     }
        //     break;
        // } else {
        //     // diffuse ray
        //     throughput /= (1.0 - SHADOW_PROB);
            


        //     sample_lambert(&ray, hit);
        // }
        
    }

    if (debug < 0.0) {
        lighting = vec3f(1.0, 1.0, 0.0);
    }

    // screen[id.x][id.y] += vec4f(hit.t * 10.0, hit.t, sin(f32(hit.idx) * 137.821) * 0.5 + 0.5, 1.0);

    if (globals.reject_hist == 1) {
        screen[id.x + globals.res.x * id.y] = vec4f(lighting, 1.0);
    } else {
        screen[id.x + globals.res.x * id.y] += vec4f(lighting, 1.0);
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
    col = tonemap_pbr_neutral(col * EXPOSURE);
    
    // col = to_linear(sample_texture(primitives[1].material.albedo, uv).rgb);
    // col = pow(col, vec3f(1.0 / 2.2));
    return vec4f(col, 1.0);
}
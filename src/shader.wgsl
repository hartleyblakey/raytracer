@group(0) @binding(0) var<uniform> globals : FrameUniforms;

@group(1) @binding(0) var<storage, read_write> triangles : array<Tri>;
@group(1) @binding(1) var<storage, read_write> tri_exts : array<TriExt>;
@group(1) @binding(2) var<storage, read_write> bvh : array<BvhNode>;
@group(1) @binding(3) var<storage, read_write> screen : array<vec4f>;

const pi = 3.141592654;
const hemisphere_area = 2.0 * pi;
const sphere_area = 4.0 * pi;

const FORWARD = vec3f(1.0, 0.0, 0.0);
const UP = vec3f(0.0, 0.0, 1.0);
const RIGHT = vec3f(0.0, -1.0, 0.0);

const SUN_DIR = vec3f(0.707106781187, 0.0 , 0.707106781187);
const SUN_COL = vec3f(1.0, 0.7995, 0.5992);

struct Camera {
    dir: vec3f,
    fovy: f32,
    origin: vec3f,
    focus: f32,
}

struct PointLight {
    position: vec4f,
    intensity: vec4f,
}

struct DirectionalLight {
    direction: vec4f,
    intensity: vec4f,
}

struct Scene {
    point_lights: array<PointLight, 12>,
    directional_lights: array<DirectionalLight, 4>,
    camera:   Camera,
    tri_count: u32,
    num_point_lights: u32,
    num_directional_lights: u32,
}

struct FrameUniforms {
    scene: Scene,
    res:    vec2u,
    frame:  u32,
    time:   f32,
    reject_hist: u32,
    node_count: u32,
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

struct GpuTexcoord {
    pos: vec2f,
    offset: u32,
    size: u32,
    
}

struct GpuVertexExt {
    tex0: GpuTexcoord,
    normal: vec2f,
    color: u32,
    _pad: f32
}

struct ExtSample {
    tex0: vec4f,
    color: vec4f,
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

struct TriExt {
    vertices: array<GpuVertexExt, 3>
}

fn dummy_texture(uv: vec2f) -> vec4f {
    let checker = f32((u32(uv.x * 32.0) + u32(uv.y * 32.0 + 1.0)) % 2u);
    let col = mix(vec3f(0.8, 0.3, 0.3), vec3f(0.8, 0.3, 0.3) * 0.5, checker);
    return vec4f(col.x, col.y, col.z, 1.0);
    // return vec4f(checker, uv.x, uv.y, 1.0);
}

fn tri_ext_sample(tri: ptr<function, TriExt>, bary: vec3f) -> ExtSample {
    var res = ExtSample(vec4f(0.0, 0.0, 0.0, 0.0), vec4f(0.0, 0.0, 0.0, 0.0), vec3f(0.0, 0.0, 0.0));

    var tc0 = vec2f(0.0, 0.0);
    res.color += bary.x * unpack_rgba8((*tri).vertices[0].color);
    tc0 += bary.x * (*tri).vertices[0].tex0.pos;

    res.color += bary.y * unpack_rgba8((*tri).vertices[1].color);
    tc0 += bary.y * (*tri).vertices[1].tex0.pos;

    res.color += bary.z * unpack_rgba8((*tri).vertices[2].color);
    tc0 += bary.z * (*tri).vertices[2].tex0.pos;

    res.tex0 = dummy_texture(tc0);
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

fn intersect_full(ray: Ray, idx: i32) -> Hit {
    let tri = triangles[idx];
    var hit = Hit(0.0, -1, vec3f(0.0, 0.0, 1.0), vec3f(0.333, 0.333, 0.333));

    let edge1 = tri.d1.xyz - tri.d0.xyz;
    let edge2 = tri.d2.xyz - tri.d0.xyz;
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

    hit.normal = normalize(cross(edge1, edge2));
    hit.normal *= -sign11(dot(hit.normal, ray.dir));
    hit.idx = idx;
    hit.t = t;
    hit.bary = vec3f((1.0 - u) - v, u, v);
    return hit;
}

// from https://gist.github.com/DomNomNom/46bb1ce47f68d255fd5d
fn intersect_aabb(ray: Ray, aabb: Aabb) -> f32 {

    let bmin = aabb_min(aabb);
    let bmax = aabb_max(aabb);

    if all(ray.origin > bmin) && all(ray.origin < bmax) {
        return 0.0;
    }

    let rmin = (bmin - ray.origin) * ray.idir;
    let rmax = (bmax - ray.origin) * ray.idir;

    let tmin = min(rmin, rmax);
    let tmax = max(rmin, rmax);

    let t0 = max(tmin.x, max(tmin.y, tmin.z));
    let t1 = min(tmax.x, min(tmax.y, tmax.z));

    if (t0 >= t1 || t0 < 0.0) {
        return -1.0;
    }

    return t0;
}

var<private> debug: f32;

fn trace_bvh(ray: Ray) -> i32 {
    var stack: Stack;
    stack.size = 0u;
    var node = bvh[0];
    var best_t = 999999999.0;
    var best_i: i32 = -1;
    if intersect_aabb(ray, node.aabb) < -0.5 {
        return best_i;
    }
    debug = 0.0;
    // var iterations = 0;
    while (true) {
        // debug = max(debug, f32(stack.size + 1u));
        // visualize bvh steps
        debug += 1.0;


        if node.count > 0 {
            // debug += 1.0;
            // debug = max(debug, f32(node.count));
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
            // // push the nodes children onto the stack
            // push(&stack, node.first + 0u);
            // push(&stack, node.first + 1u);

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
    return i32(best_i);
}
fn trace_bvh_shadow(ray: Ray) -> bool {
    var stack: Stack;
    stack.size = 0u;
    var node = bvh[0];
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


fn trace(ray: Ray) -> Hit {
    var closest_idx = -1;
    var closest_t = 999999999.0;

    closest_idx = trace_bvh(ray);
    return intersect_full(ray, closest_idx);

    // // test header
    // var hit: Hit;
    // hit.bary = vec3f(0.33);
    // hit.idx = -1;
    // hit.normal = vec3f(0.0, 0.0, 1.0);
    // hit.t = 99999999.0;
    // debug = 0.0;

    // // test aabb intersection
    // aabb.data = array<f32, 6>(-1.0, -1.0, -1.0, 0.0, 0.0, 0.0);
    // if (intersects_aabb(ray, aabb)) {
    //     hit.t = 1.0;
    //     hit.idx = 0;
    //     debug += 1.0;
    // }
    // return hit;

    // // visualize leaf nodes
    // let start = (globals.frame * 1u) % (globals.node_count - 128u);
    // for (var i = start; i < start + 128u; i++) {
    //     let node = bvh[i];
    //     if (node.count != 0) {
    //         if (intersect_aabb(ray, node.aabb) >= 0.0) {
    //             debug += 1.0;
    //             for (var j = node.first; j < node.first + node.count; j++) {
    //                 let t = intersect(ray, triangles[j]);
    //                 if (t >= 0.0) {
    //                     debug += 1.0;
    //                 }
    //             }
    //         }
    //     }

    // }
    // return hit;

    // for (var i = 0; i < i32(globals.scene.tri_count); i++) {
    //     let t = intersect(ray, triangles[i]);
    //     if (t >= 0.0 && t < hit.t) {
    //         hit.idx = i;
    //         hit.t = t;
    //     }
    // }
    // return hit;
    // return intersect_full(ray, hit.idx);
    
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

fn sky(dir: vec3f) -> vec3f {
    let horizon = vec3f(1.0 - SUN_COL);
    return mix(horizon, SUN_COL,pow(max(dot(dir, SUN_DIR), 0.0), 3.0)) * 1.00;
    // return to_linear(dir * 0.5 + 0.5);
    // return vec3f(1.0);
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

    let a = rand() * pi * 2.0;
    let m = rand();
    let aa_pixel = vec2f(pixel) + vec2f(rand(), rand());
    let aspect = f32(globals.res.x) / f32(globals.res.y);

    pixel_pos += right * (aa_pixel.x / f32(globals.res.x) - 0.5) * fov_factor * aspect;
    pixel_pos += up    * (0.5 - aa_pixel.y / f32(globals.res.y)) * fov_factor;

    pixel_pos += right * aspect * cos(a) * pow(m, 150.0);
    pixel_pos += up             * sin(a) * pow(m, 150.0);

    ray.dir  = normalize(pixel_pos - ray.origin);
    ray.idir = 1.0 / ray.dir;

    return ray;
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

fn ramp(x: f32) -> vec3f {
    if x < 0.0 {
        return vec3f(0.0, 0.0, 1.0);
    }
    if x > 1.0 {
        return vec3f(1.0, 1.0, 0.0);
    }
    return to_linear(clamp(magma_quintic(x), vec3f(0.0), vec3f(1.0)));
}

fn shade (hit: Hit, dir: vec3f, throughput: ptr<function, vec3f>, lighting: ptr<function, vec3f>) {

    let backup = seed;
    seed = u32(hit.idx * 7);
    rand();

    var emissive = vec3f(0);
    var albedo   = vec3f(0.7);

    if (hit.idx == -1) {
        // miss
        emissive = sky(dir);
    } else {
        if (hit.idx % 135 == 2) {
            // emissive = rand_color() * 14.0;
        } else if (hit.idx % 3 == 1) {
            // albedo = rand_color();
        }
        // if hit.normal.z > 0.95 {
        //     emissive = to_linear(rand_color()) * 8.0;
        // }
        // albedo = vec3f(0.7);
        var ext = tri_exts[hit.idx];
        let sample = tri_ext_sample(&ext, hit.bary);
        albedo = sample.tex0.rgb;

        if (min(hit.bary.x, min(hit.bary.y, hit.bary.z)) < 0.02) {
            // albedo = vec3f(0.1);
            // if (hit.idx % 25 == 2) {
            //   emissive = rand_color() * 500.0;
            // }
        }



        // albedo = vec3f(hit.bary, 0.0);
        // if (distance(length(hit.bary), 0.5) > 0.1) {
        //     albedo = rand_color();
        // } else {
        //     // emissive = vec3f(1.0);
        // }
    }

    // // visualize bvh
    // emissive = ramp(debug / 128.0);

    // // visualize normals
    // emissive = to_linear(hit.normal * 0.5 + 0.5);

    // // visualize albedo
    // emissive = albedo * (dot(hit.normal, vec3f(0.0, 0.0, 1.0)) * 0.4 + 0.6);
    
    *lighting += *throughput * emissive;
    *throughput *= albedo;

    seed = backup;
}

fn scatter(ray: ptr<function, Ray>, hit: Hit, throughput: ptr<function, vec3f>) {
    // from raytracing in one weekend
    (*ray).dir = normalize(hit.normal + rand_sphere());
    (*ray).idir = vec3f(1.0) / (*ray).dir;
}

fn lambert(dir: vec3f, normal:  vec3f) -> f32 {
    return max(dot(dir, normal), 0.0) / pi;
}

@compute
@workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
if (id.x < globals.res.x && id.y < globals.res.y) {

    var lighting   = vec3f(0);
    var throughput = vec3f(1);

    // let samples = u32(screen[id.x + globals.res.x * id.y].a);
    seed = hash21(vec2u(hash21(id.xy), globals.frame));

    var ray = camera_ray(id.xy);
    for (var i = 0; i < 5; i++) {
        let hit = trace(ray);
        
        shade(hit, ray.dir, &throughput, &lighting);

        if (hit.idx == -1) {
            break;
        }

        ray.origin += ray.dir * hit.t + hit.normal * 0.0001;
        const SHADOW_PROB = 0.5;
        if rand() < SHADOW_PROB {
            throughput /= SHADOW_PROB;
            // shadow raw
            ray.dir = normalize(SUN_DIR + rand_sphere() * 0.01);
            ray.idir = vec3f(1.0) / ray.dir;
            if !trace_bvh_shadow(ray) {
                lighting += throughput * SUN_COL * 12.0 * lambert(SUN_DIR, hit.normal);
            }
            break;
        } else {
            throughput /= (1.0 - SHADOW_PROB);
            scatter(&ray, hit, &throughput);
        }
        
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
    let x = f32(i32(i) - 1) * 5.0;
    let y = f32(i32(i & 1u) * 2 - 1) * 5.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

fn tonemap(in: vec3f) -> vec3f {
    return vec3f(pow(1.0 - pow(vec3f(0.25), in), vec3f(1.1)));
}

@fragment
fn fs_main(@builtin(position) p: vec4f) -> @location(0) vec4<f32> {
    let id = vec2u(p.xy);
    if (id.x >= globals.res.x || id.y >= globals.res.y) {
        return vec4f(0.5, 0.1, 0.1, 1.0);
    }
    let scr = screen[id.x + globals.res.x * id.y];
    var col = scr.rgb / scr.a;
    col = tonemap(col / 1.5);
    // col = dummy_texture(p.xy / vec2f(globals.res)).rgb;
    return vec4f(col, 1.0);
}
@group(0) @binding(0) var<uniform> globals : FrameUniforms;

@group(1) @binding(0) var<storage, read_write> triangles : array<Tri>;
@group(1) @binding(1) var<storage, read_write> bvh : u32;
@group(1) @binding(2) var<storage, read_write> screen : array<array<vec4f, 512>, 512>;

const pi = 3.141592654;
const hemisphere_area = 2.0 * pi;
const sphere_area = 4.0 * pi;

struct FrameUniforms {
    res:    vec2u,
    frame:  u32,
    tri_count: u32,
    time:   f32,
}

struct Tri {
    d0: vec4f,
    d1: vec4f,
    d2: vec4f,
}

fn centroid(tri: Tri) -> vec3f {
    return vec3f(tri.d0.w, tri.d1.w, tri.d2.w);
}

struct Ray {
    origin: vec3f,
    dir: vec3f,
    idir: vec3f,
}

struct Hit {
    t: f32,
    idx: i32,
}

// https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
fn intersect (ray: Ray, tri: Tri) -> f32 {
    let edge1 = tri.d1.xyz - tri.d0.xyz;
    let edge2 = tri.d2.xyz - tri.d0.xyz;
    let h = cross( ray.dir, edge2 );
    let a = dot( edge1, h );
    if (a > -0.0001f && a < 0.0001f) {
        return -1.0;
    }// ray parallel to triangle
    let f = 1 / a;
    let s = ray.origin - tri.d0.xyz;
    let u = f * dot( s, h );
    if (u < 0 || u > 1) {
        return -1.0;
    }
    let q = cross( s, edge1 );
    let v = f * dot( ray.dir, q );
    if (v < 0 || u + v > 1) {
        return -1.0;
    }
    let t = f * dot( edge2, q );
    if (t > 0.0001f) {
        return t;
    } else {
        return -1.0;
    }
}

fn trace(ray: Ray) -> Hit {

    var closest_hit = Hit(99999.0, -1);

    for (var i = 0; i < i32(globals.tri_count); i++) {
        let t = intersect(ray, triangles[i]);
        if (t >= 0.0 && t < closest_hit.t) {
            closest_hit.idx = i;
            closest_hit.t = t;
        }
    }
    return closest_hit;
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
    let sun = normalize(vec3f(0.0, 0.0, 1.0));
    let col = vec3f(1.0, 0.995, 0.992);
    return col * pow(max(dot(dir, sun), 0.0), 2.0) * 3.5;
    // return vec3f(1.0);
}

fn camera_ray(position: vec3f, forward: vec3f, pixel: vec2u) -> Ray {
    var ray: Ray;
    ray.origin = position;
    let unreachable = vec3(0.0, 0.0, 1.0);
    let right = cross(forward, unreachable);
    let up = cross(right, forward);
    var pixel_pos = position + forward;

    let aa_pixel = vec2f(pixel) + vec2f(rand(), rand());

    pixel_pos += right * (aa_pixel.x / f32(globals.res.x) - 0.5);
    pixel_pos += up * (0.5 - aa_pixel.y / f32(globals.res.y));
    ray.dir = normalize(pixel_pos - position);
    ray.idir = 1.0 / ray.dir;

    return ray;
}


fn shade (hit: Hit, dir: vec3f, throughput: ptr<function, vec3f>, lighting: ptr<function, vec3f>) {


    let backup = seed;
    seed = u32(hit.idx * 7);
    rand();

    var emissive = vec3f(0);
    var albedo   = vec3f(0.5);

    if (hit.idx == -1) {
        // miss
        emissive = sky(dir);
    } else if (hit.idx % 5 == 2) {
        // emissive = rand_color() * 2.0;
    } else if (hit.idx % 3 == 1) {
        albedo = rand_color();
    }
    albedo = rand_color();
    *lighting += *throughput * emissive;
    *throughput *= albedo;

    seed = backup;
}

@compute
@workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {


    var lighting   = vec3f(0);
    var throughput = vec3f(1);

    seed = hash21(vec2u(hash21(id.xy), globals.frame));
    let camera = vec3f(3.0, 3.0, 1.5);
    var ray = camera_ray(camera, normalize(vec3f(0.0, 0.0, 0.5) - camera), id.xy);
    for (var i = 0; i < 4; i++) {
        let hit = trace(ray);
        shade(hit, ray.dir, &throughput, &lighting);
        if (hit.idx == -1) {
            break;
        }
        ray.origin += ray.dir * (hit.t - 0.001);
        ray.dir = rand_sphere();
    }
    // screen[id.x][id.y] += vec4f(hit.t * 10.0, hit.t, sin(f32(hit.idx) * 137.821) * 0.5 + 0.5, 1.0);
    screen[id.x][id.y] += vec4f(lighting, 1.0);
}


@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(i) - 1) * 5.0;
    let y = f32(i32(i & 1u) * 2 - 1) * 5.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

fn tonemap(in: vec3f) -> vec3f {
    return vec3f(1.0 - pow(vec3f(0.25), in));
}

@fragment
fn fs_main(@builtin(position) p: vec4f) -> @location(0) vec4<f32> {
    let up = vec2u(p.xy);
    if (up.x >= 512 || up.y >= 512) {
        return vec4f(0.5, 0.1, 0.1, 1.0);
    }
    let scr = screen[up.x][up.y];
    var col = scr.rgb / scr.a;
    // col = tonemap(col);
    return vec4f(col, 1.0);
}
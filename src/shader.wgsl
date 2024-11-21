@group(0) @binding(0) var<uniform> globals : FrameUniforms;

@group(1) @binding(0) var<storage, read_write> triangles : array<Tri, 32>;
@group(1) @binding(1) var<storage, read_write> bvh : u32;
@group(1) @binding(2) var<storage, read_write> screen : array<array<vec4f, 512>, 512>;

struct FrameUniforms {
    res:    vec2u,
    frame:  u32,
    time:   f32,
}

struct Tri {
    vertex0: vec3f,
    vertex1: vec3f,
    vertex2: vec3f,
    centroid: vec3f,
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
    let edge1 = tri.vertex1 - tri.vertex0;
    let edge2 = tri.vertex2 - tri.vertex0;
    let h = cross( ray.dir, edge2 );
    let a = dot( edge1, h );
    if (a > -0.0001f && a < 0.0001f) {
        return -1.0;
    }// ray parallel to triangle
    let f = 1 / a;
    let s = ray.origin - tri.vertex0;
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

fn trace(_ray: Ray) -> Hit {
    var ray = _ray;
    ray.idir = 1.0 / ray.dir;
    var t = Hit(99999.0, -1);

    var hit_idx = -1;
    for (var i = 0; i < 32; i++) {
        let t2 = intersect(ray, triangles[i]);
        if (t2 >= 0.0 && t2 < t.t) {
            t.idx = i;
            t.t = t2;
        }
    }
    return t;
}

// Hash without Sine https://www.shadertoy.com/view/4djSRW
fn hash13(p: vec3f) -> f32
{
    var p3  = fract(p * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}
fn hash33(p: vec3f) -> vec3f
{
    var p3 = fract(p * vec3f(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yxz+33.33);
    return fract((p3.xxy + p3.yxx)*p3.zyx);
}

var<private> seed: f32 = 123141.0;
fn rand() -> f32 {
    let old = seed;
    seed = hash13(vec3f(seed, sin(seed), fract(seed) * 17.0));
    return hash13(vec3f(old + seed, seed, cos(seed)));
}

fn sky(dir: vec3f) -> vec3f {
    return vec3f(1.0, 0.9, 0.6) * max(dot(dir, vec3(0.0, 0.0, 1.0)), 0.0)
         + vec3f(0.2, 0.4, 0.6) * max(dot(dir, vec3(0.0, 0.0, -1.0)), 0.0);
}

@compute
@workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    var ray: Ray;

    var lighting   = vec3f(0);
    var throughput = vec3f(1);

    seed = hash13(vec3f(globals.time, f32(id.x), f32(id.y)));

    ray.dir = vec3f(1.0, 0.0, 0.0);
    ray.origin = vec3f(0.0, (f32(id.x) + rand()) / 512.0, (f32(id.y) + rand()) / 512.0);

    for (var i = 0; i < 4; i++) {
        let hit = trace(ray);
        if (hit.idx == -1) {
            lighting += sky(ray.dir) * throughput;
            break;
        }
        throughput *= 0.5;
        ray.origin += ray.dir * (hit.t - 0.001);
        ray.dir = normalize(vec3f(rand(), rand(), rand())) - 0.5;
    }

    let c = lighting * 2.0;


    // screen[id.x][id.y] += vec4f(hit.t * 10.0, hit.t, sin(f32(hit.idx) * 137.821) * 0.5 + 0.5, 1.0);
    screen[id.x][id.y] += vec4f(c.r, c.g, c.b, 1.0);
}


@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(i) - 1) * 5.0;
    let y = f32(i32(i & 1u) * 2 - 1) * 5.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) p: vec4f) -> @location(0) vec4<f32> {
    let up = vec2u(p.xy);
    if (up.x >= 512 || up.y >= 512) {
        return vec4f(0.5, 0.1, 0.1, 1.0);
    }
    let scr = screen[up.x][up.y];
    return vec4f(scr.r / scr.a, scr.g / scr.a, scr.b / scr.a, 1.0);
}
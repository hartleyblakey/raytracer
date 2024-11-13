@group(0) @binding(0) var<uniform> globals : FrameUniforms;

@group(1) @binding(0) var<storage, read> triangles : array<Tri, 32>;
@group(1) @binding(1) var<storage, read> bvh : u32;
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

fn trace(ray: Ray) -> f32 {
    var t = 999999.0;
    var hit_idx = -1;
    for (let i = 0; i < arrayLength(triangles); i++) {
        let t2 = intersect(ray, triangles[i]);
        if (t2 < t) {
            hit_idx = i;
            t = t2;
        }
    }
    return t;
}

@compute
@workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    
    screen[id.x][id.y] = vec4f(0.2, 0.5, 0.6, 1.0);
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
    return screen[up.x][up.y];
}
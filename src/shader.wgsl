@group(0) @binding(0) var<uniform> globals : FrameUniforms;

@group(1) @binding(0) var<storage, read_write> triangles : u32;
@group(1) @binding(1) var<storage, read_write> bvh : u32;
@group(1) @binding(2) var<storage, read_write> screen : array<array<vec4f, 512>, 512>;

struct FrameUniforms {
    res:    vec2u,
    frame:  u32,
    time:   f32,
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
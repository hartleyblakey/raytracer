@group(0) @binding(0) var<uniform> globals : FrameUniforms;

struct FrameUniforms {
    res:    vec2u,
    frame:  u32,
    time:   f32,
}

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(i) - 1) * 5.0;
    let y = f32(i32(i & 1u) * 2 - 1) * 5.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) p: vec4f) -> @location(0) vec4<f32> {
    return vec4<f32>(p.x / 500.0, p.y / 500.0, sin(globals.time) * 0.5 + 0.5, 1.0);
}
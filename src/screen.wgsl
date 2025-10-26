#import "common.wgsl"

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
    return mix(color, newPeak * vec3f(1.0), g);
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
    var col = scr.rgb / max(scr.a, 1.0);

    if DEBUG && globals.debug_mode == 8u {
        col = scr.rgb;
        if scr.r > 1.0 {
            col = vec3f(1.0, 0.0, 0.0);
        }
    }

    if !DEBUG || globals.debug_mode == 0u {
        col = tonemap_pbr_neutral(col * globals.scene.camera.exposure);
    }
    
    // col = to_linear(sample_texture(primitives[1].material.albedo, uv).rgb);
    // col = pow(col, vec3f(1.0 / 2.2));
    return vec4f(col, 1.0);
}
#import "common.wgsl"

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


@compute
@workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x > globals.res.x || id.y > globals.res.y) {
        return;
    }

    let lid = id.y * globals.res.x + id.x;

    seed = hash21(vec2u(hash21(id.xy), globals.frame));
    // spin the rng to improve the quality of the first samples
    rand(); rand();

    let ray = camera_ray(id.xy);

    var state: RayState;

    state.direction_min = vec4f(ray.dir, 0.0);
    state.origin_max = vec4f(ray.origin, 1e30);
    state.depth = 0u;
    state.medium = 0u; // background
    state.pixel = lid;
    state.throughput_flags = vec4f(1.0, 1.0, 1.0, bitcast<f32>(0u));
    state.rng_state = seed;

    ray_queue[lid] = state;
}
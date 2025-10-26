@compute
@workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    let lid = id.x;

    if (lid >= atomicLoad(&ray_queue_meta.num_in_rays)) {
        return;
    }

    var ray_state = in_ray_queue[lid];

    var ray_volume = media[ray_state.medium];
    var throughput = ray_state.throughput_flags.rgb;
    seed = ray_state.rng_state;

    var flags = get_flags(ray_state);

    var ray: Ray;
    ray.dir = ray_state.direction_min.xyz;
    ray.origin = ray_state.origin_max.xyz;
    ray.idir = 1.0 / ray.dir;

    var bsdf_mis_weight = 1.0;
    
    let hit = trace(ray);
    let point = ray.origin + ray.dir * hit.t;

    if DEBUG && globals.debug_mode == 3u {
        add_contrib(magma_quintic(debug / 256.0), ray_state.pixel);
        on_kill(ray_state.pixel);
        return;
    }

    if DEBUG && globals.debug_mode == 8u && flags.depth > 0u {
        // add_contrib(vec3f(debug / 2048.0), ray_state.pixel);
        screen[ray_state.pixel] = max(screen[ray_state.pixel], vec4f(vec3f(debug / 512.0), 1.0));
        // on_kill(ray_state.pixel);
        // return;
    }

    // MARK: - Shade
    var lighting   = vec3f(0.0);

    if hit.idx == -1 {
        let nee_pdf = sample_light_pdf(ray.origin, point, hit.prim_idx, hit.idx);

        if flags.depth > 0u {
            bsdf_mis_weight = mis_power_heuristic(ray_state.last_pdf, nee_pdf, 1.0, 1.0);
        }

        lighting += throughput * bsdf_mis_weight * evaluate_env_map(ray.dir).rgb;
        
        if globals.debug_mode != 8u {add_contrib(lighting, ray_state.pixel);}
        screen[ray_state.pixel].w += 1.0;
        return;
    }

    let sample = sample_hit(hit);

    var hit_ior = media[hit.material.volume].ior;
    if hit.backface {
        hit_ior = background_volume().ior;
    }

    if hit_ior == ray_volume.ior {
        hit_ior = 1.6 * ray_volume.ior;
    }

    if globals.debug_mode != 9u {
        throughput *= exp(-ray_volume.absorption * hit.t);
    }

    let nee_pdf = sample_light_pdf(ray.origin, point, hit.prim_idx, hit.idx);

    bsdf_mis_weight = mis_power_heuristic(ray_state.last_pdf, nee_pdf, 1.0, 1.0);
    
    if nee_pdf != 0.0 {
        lighting += throughput * sample.emissive * bsdf_mis_weight;
    }
    
    let wo = -ray.dir;

    // sample NEE shadow ray
    
    var vis_ray_state: VisRayState;
    var cast_vis_ray = true;
    {
        let light = sample_light(ray.origin + ray.dir * hit.t);

        if light.pdf > 0.0 {
            vis_ray_state.origin_max = vec4f(ray.origin + ray.dir * hit.t + light.wi * 0.001, light.t_max);
            vis_ray_state.direction_min = vec4f(light.wi, 0.001);


            let nee_ray_bsdf_pdf = sample_bsdf_pdf(light.wi, wo, hit_ior, ray_volume.ior, sample);
            let nee_mis_weight = mis_power_heuristic(light.pdf, nee_ray_bsdf_pdf, 1.0, 1.0);
            let nee_bsdf = evaluate_bsdf(light.wi, wo, hit_ior, ray_volume.ior, sample);

            vis_ray_state.contrib_pixel = vec4f(
                throughput * nee_bsdf * abs(dot(light.wi, sample.normal)) * light.contrib * nee_mis_weight / light.pdf,
                bitcast<f32>(ray_state.pixel)
            );
        } else {
            cast_vis_ray = false;
        }
    }


    var rr_prob = clamp(1.0 - max(throughput.x, max(throughput.y, throughput.z)), 0.0, 0.95);
    if flags.depth < 2u {
        rr_prob = 0.0;
    } else {
        rr_prob = max(rr_prob, 0.2);
    }
    if rand() < rr_prob {
        on_kill(ray_state.pixel);
        return;
    } else {
        throughput /= (1.0 - rr_prob);
    }

    // sample BSDF continuation
    var bsdf_pdf: f32;
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

    var out_medium = ray_state.medium;
    
    if dot(wi, sample.normal) > 0.0 {
        ray.origin += hit.normal * 0.001;
    } else {
        ray.origin -= hit.normal * 0.001;
        if hit.backface {
            out_medium = 0u;
        } else {
            out_medium = hit.material.volume;
        }
    }
    if globals.debug_mode == 6u {
        let backup = seed;
        seed = bitcast<u32>(hit.prim_idx) + 777u;
        rand(); rand();
        lighting = rand_color() * evaluate_lambert(wo, sample.normal) * pi;
        seed = backup;
    }

    if globals.debug_mode == 7u {
        let backup = seed;
        seed = bitcast<u32>(hit.idx) + 777u;
        rand(); rand();
        lighting = rand_color() * evaluate_lambert(wo, hit.normal) * pi;
        seed = backup;
    }

    if DEBUG && globals.debug_mode == 8u {
        cast_vis_ray = false;
        throughput = vec3f(1.0);
    }
    
    ray_state.direction_min = vec4f(ray.dir, 0.0);
    ray_state.origin_max = vec4f(ray.origin, 1e30);
    ray_state.last_pdf = bsdf_pdf;
    ray_state.medium = out_medium;
    ray_state.rng_state = seed;
    ray_state.throughput_flags = vec4f(throughput, 0.0);

    flags.depth += 1u;
    set_flags(&ray_state, flags);




    {
        let idx = atomicAdd(&ray_queue_meta.num_out_rays, 1u);
        out_ray_queue[idx] = ray_state;
    }

    if cast_vis_ray {
        let idx = atomicAdd(&ray_queue_meta.num_vis_rays, 1u);
        vis_ray_queue[idx] = vis_ray_state;
    }


    add_contrib(lighting, ray_state.pixel);

    if flags.depth >= globals.max_depth {
        on_kill(ray_state.pixel);
    }
}
#import "common.wgsl"

var<private> debug: f32;

// MARK: trace_bvh

fn trace_bvh(ray: Ray, root: u32, t_max: ptr<function, f32>, prim: Primitive) -> i32 {
    var stack: Stack;
    stack.size = 0u;
    var node = bvh[root];
    var best_t = *t_max;
    var best_i: i32 = -1;
    if aabb_close(intersect_aabb(ray, node.aabb)) >= best_t {
        return best_i;
    }
    
    while (true) {
        // debug = max(debug, f32(stack.size + 1u));
        // visualize bvh steps
        debug += 0.5;
        // debug = max(debug, f32(stack.size + 1u));

        // if aabb_close(intersect_aabb(ray, node.aabb)) > best_t {
        //     if stack.size == 0u {
        //         break;
        //     }
        //     node = bvh[pop(&stack)];
        //     continue;
        // }

        if node.count > 0u {

            // if debug > 0.0 {
            //     return i32(node.first);
            // }

            // intersect triangles of node
            for (var i = node.first; i < node.first + node.count; i++) {
                let t = intersect(ray, triangles[i]);
                debug += 0.5;
                if t >= 0.0 && t < best_t {
                    
                    if (prim.material.alpha_settings & 3u) != 0u {
                        let hit = intersect_full(ray, i32(i));
                        var ext = tri_exts[i];

                        var texcoord = vec2f(0.0);

                        texcoord += hit.bary.x * ext.vertices[0].texcoords[prim.material.albedo_texcoord];
                        texcoord += hit.bary.y * ext.vertices[1].texcoords[prim.material.albedo_texcoord];
                        texcoord += hit.bary.z * ext.vertices[2].texcoords[prim.material.albedo_texcoord];

                        let alpha = sample_texture(prim.material.albedo, texcoord).a;

                        if (prim.material.alpha_settings & 3u) == 1u {
                            // MASK
                            if alpha < f32(prim.material.alpha_settings >> 16u) / f32(1u << 16u) {
                                continue;
                            }
                        } else {
                            // BLEND
                            if rand() > alpha * alpha {
                                continue;
                            }
                        }
                    }
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
            let l_h  = intersect_aabb(ray, bvh[node.first + 0u].aabb);
            let r_h  = intersect_aabb(ray, bvh[node.first + 1u].aabb);
            var left  = aabb_close(l_h);
            var right = aabb_close(r_h);
    
            if (left > best_t) && (right > best_t) {
                if stack.size == 0u {
                    break;
                }
                node = bvh[pop(&stack)];
            } else if (left > best_t) {
                node = bvh[node.first + 1u];
            } else if (right > best_t) {
                node = bvh[node.first + 0u];
            } else if right < left {
                push(&stack, node.first + 0u);
                node = bvh[node.first + 1u];
            } else if left < right {
                push(&stack, node.first + 1u);
                node = bvh[node.first + 0u];
            } else if r_h.y < l_h.y {
                push(&stack, node.first + 0u);
                node = bvh[node.first + 1u];
            } else {
                
                push(&stack, node.first + 1u);
                node = bvh[node.first + 0u];
            }

        }
    }
    *t_max = best_t;
    return best_i;
}



fn trace(ray: Ray) -> Hit {
    debug = 0.0;
    var stack: Stack;
    stack.size = 0u;
    var node = bvh[globals.scene.node_count];
    var best_t = 99999999.0;
    var closest_tri: i32 = -1;
    var closest_primitive: i32 = -1;

    if aabb_close(intersect_aabb(ray, node.aabb)) > best_t {
        return hit_default();
    }
    var tlas_steps = 0.0;
    while (true) {
        // visualize bvh steps
        debug += 1.0;

        // if aabb_close(intersect_aabb(ray, node.aabb)) > best_t {
        //     if stack.size == 0u {
        //         break;
        //     }
        //     node = bvh[pop(&stack)];
        //     continue;
        // }

        // debug = max(debug, f32(stack.size + 1u));
        if node.count > 0u {
            tlas_steps += 1.0;
            // intersect BLAS(s) of node
            for (var i = node.first; i < node.first + node.count; i++) {
                let scale_factor = length(transform_dir(ray.dir, primitives[i].inv_transform));
                let t_ray = transform_ray(ray, primitives[i].inv_transform);
                debug += 1.0;
                var new_t = best_t * scale_factor;
                let new_tri = trace_bvh(t_ray, primitives[i].bvh_idx, &new_t, primitives[i]);
                

                if new_tri >= 0 {
                    best_t = new_t / scale_factor;
                    closest_tri = new_tri;
                    closest_primitive = i32(i);

                    // var hit = hit_default();
                    // hit.prim_idx = i32(i);
                    // hit.idx = i32(new_tri);
                    // hit.normal = -ray.dir;
                    // return hit;
                }
            }
            if stack.size == 0u {
                break;
            }
            node = bvh[pop(&stack)];
        } else {
            // avoid pushing nodes onto the stack where possible
            // order nodes based on distance

            // TLAS is tacked onto end of bvh:
            let node_first = globals.scene.node_count + node.first;

            // try ordering the nodes
            let l_h  = intersect_aabb(ray, bvh[node_first + 0u].aabb);
            let r_h = intersect_aabb(ray, bvh[node_first + 1u].aabb);
            let left  = aabb_close(l_h);
            let right = aabb_close(r_h);
    
            if (left > best_t) && (right > best_t) {
                if stack.size == 0u {
                    break;
                }
                node = bvh[pop(&stack)];
            } else if (left > best_t) {
                node = bvh[node_first + 1u];
            } else if (right > best_t) {
                node = bvh[node_first + 0u];
            } else if right < left {
                // push(&stack, node_first + 1u);
                // node = bvh[node_first + 0u];

                push(&stack, node_first + 0u);
                node = bvh[node_first + 1u];

            } else if left < right{

                push(&stack, node_first + 1u);
                node = bvh[node_first + 0u];
            } else if r_h.y < l_h.y {
                push(&stack, node_first + 0u);
                node = bvh[node_first + 1u];
            } else {
                
                push(&stack, node_first + 1u);
                node = bvh[node_first + 0u];
            }

        }
    }

    let t_ray_final = transform_ray(ray, primitives[closest_primitive].inv_transform);
    var hit = intersect_full(t_ray_final, closest_tri);

    // transform the hit back to world space
    hit.t = best_t;
    hit.normal = transform_normal(hit.normal, primitives[closest_primitive].inv_transform);
    hit.prim_idx = closest_primitive;
    hit.material = primitives[closest_primitive].material;
    return hit;

}

@compute
@workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    let lid = id.x;

    if (lid >= atomicLoad(&ray_queue_meta.num_in_rays)) {
        return;
    }

    var ray_state = in_ray_queue[lid];
    seed = ray_state.rng_state;

    var ray: Ray;
    ray.dir = ray_state.direction_min.xyz;
    ray.origin = ray_state.origin_max.xyz;
    ray.idir = 1.0 / ray.dir;
    
    let hit = trace(ray);

    if DEBUG && globals.debug_mode == 3u {
        screen[ray_state.pixel] += vec4f(debug, 0.0, 0.0, 1.0);
        // on_kill(ray_state.pixel);
        return;
    }

    var hit_state: RayHit;
    hit_state.prim = hit.prim_idx;
    hit_state.tri = hit.idx;
    hit_state.t = hit.t;
    hit_state.uv_bf = 0u; // UNIMPLEMENTED
    
    ray_hit_queue[lid] = hit_state;
}
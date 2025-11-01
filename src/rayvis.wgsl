#import "common.wgsl"

fn trace_bvh_shadow(ray: Ray, root: u32, t_max: ptr<function, f32>, prim: Primitive) -> bool {
    var stack: Stack;
    stack.size = 0u;
    var node = bvh[root];
    var best_t = *t_max;

    if intersect_aabb(ray, node.aabb).x >= best_t {
        return false;
    }
    
    while (true) {
        if node.count > 0u {
            for (var i = node.first; i < node.first + node.count; i++) {
                let t = intersect(ray, triangles[i]);
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
                    return true;
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
            var left  = intersect_aabb(ray, bvh[node.first + 0u].aabb).x;
            var right = intersect_aabb(ray, bvh[node.first + 1u].aabb).x;
    
            if (left > best_t) && (right > best_t) {
                if stack.size == 0u {
                    break;
                }
                node = bvh[pop(&stack)];
            } else if (left > best_t) {
                node = bvh[node.first + 1u];
            } else if (right > best_t) {
                node = bvh[node.first + 0u];
            } else if right > left {
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


fn trace_shadow(ray: Ray, t: f32) -> bool {
    var stack: Stack;
    stack.size = 0u;
    var node = bvh[globals.scene.node_count];
    var best_t = min(t, 1e20);
    if intersect_aabb(ray, node.aabb).x > best_t {
        return false;
    }
    while (true) {

        if node.count > 0u {
            // intersect BLAS(s) of node
            for (var i = node.first; i < node.first + node.count; i++) {
                let scale_factor = length(transform_dir(ray.dir, primitives[i].inv_transform));
                let t_ray = transform_ray(ray, primitives[i].inv_transform);
                var new_t = best_t * scale_factor;
                
                if trace_bvh_shadow(t_ray, primitives[i].bvh_idx, &new_t, primitives[i]) {
                    return true;
                }
            }
            if stack.size == 0u {
                break;
            }
            node = bvh[pop(&stack)];
        } else {
            // TLAS is tacked onto end of bvh:
            let node_first = globals.scene.node_count + node.first;

            // try ordering the nodes
            let left  = intersect_aabb(ray, bvh[node_first + 0u].aabb).x;
            let right = intersect_aabb(ray, bvh[node_first + 1u].aabb).x;
    
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

            } else {
                // push(&stack, node_first + 0u);
                // node = bvh[node_first + 1u];

                
                push(&stack, node_first + 1u);
                node = bvh[node_first + 0u];
            }

        }
    }
    return false;
}


@compute
@workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    let lid = id.x;
    if (lid >= atomicLoad(&ray_queue_meta.num_vis_rays)) {
        return;
    }

    

    let state = vis_ray_queue[lid];

    // screen[bitcast<u32>(state.contrib_pixel.w)] = vec4f(1.0, 0.0, 0.0, 1.0);

    var ray: Ray;
    ray.origin = state.origin_max.xyz;
    ray.dir = state.direction_min.xyz;
    ray.idir = 1.0 / ray.dir;

    if !trace_shadow(ray, state.origin_max.w - 0.001) {
        screen[bitcast<u32>(state.contrib_pixel.w)] += vec4f(max(state.contrib_pixel.rgb, vec3f(0.0)), 0.0);
    }
}
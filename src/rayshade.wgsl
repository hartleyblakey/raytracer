#import "common.wgsl"

fn trace_transformed_tri(ray: Ray, prim: i32, tri: i32) -> Hit {
    var hit = intersect_full(transform_ray(ray, primitives[prim].inv_transform), tri);
    if hit.idx == -1 {
        return hit;
    }
    hit.t /= length(transform_dir(ray.dir, primitives[prim].inv_transform));
    hit.normal = transform_normal(hit.normal, primitives[prim].inv_transform);
    hit.prim_idx = prim;
    hit.material = primitives[prim].material;
    return hit;
}

fn sample_emission(hit: Hit) -> vec3f {
    var res = zeroed_ext_sample();
    let prim = primitives[hit.prim_idx];
    var ext = tri_exts[hit.idx];
    let tci = prim.material.emissive_texcoord;
    
    var tc = vec2f();
    tc += hit.bary.x * ext.vertices[0].texcoords[tci];
    tc += hit.bary.y * ext.vertices[1].texcoords[tci];
    tc += hit.bary.z * ext.vertices[2].texcoords[tci];

    var emissive = hit.material.emissive_factor;
    if hit.material.emissive.size != 0u {
        emissive *= to_linear(sample_texture(hit.material.emissive, tc).rgb);
    }
    return emissive;
}


/// gather material parameters for a hit
fn sample_hit(hit: Hit) -> ExtSample {
    var ext = tri_exts[hit.idx];
    var sample = tri_ext_interpolate(&ext, hit.bary);

    // get smooth_normal into world space, if it exists
    if length(sample.vertex_normal) > 0.001 {
        sample.vertex_normal = transform_normal(normalize(sample.vertex_normal), primitives[hit.prim_idx].inv_transform);
        if dot(sample.vertex_normal, hit.normal) < 0.0 {
            // flip face if vertex normal pointing away from geometric normal
           sample.vertex_normal = -sample.vertex_normal;
           sample.tangent = -sample.tangent;
           sample.t_sign *= -1.0;
        }
    } else {
        
        sample.vertex_normal = hit.normal;
    }

    sample.normal = sample.vertex_normal;
    if (hit.material.normal.size != 0u) {
        let normal_tangent = (sample_texture(hit.material.normal, sample.texcoords[hit.material.normal_texcoord]).xyz * 2.0 - 1.0) * hit.material.normal_scale;

        // this is what the guys website says, but it looks a little off to me
        let bt = normalize(sample.t_sign * cross(sample.vertex_normal, sample.tangent));
        // let bt = sample.bi_tangent;
        
        sample.normal = normalize(
            normal_tangent.x * sample.tangent + normal_tangent.y * bt + normal_tangent.z * sample.vertex_normal
        );

        if all(sample.vertex_normal == sample.tangent) {
            sample.normal = sample.vertex_normal;
        }
    }

    sample.albedo = hit.material.albedo_factor;
    if (ext.vertices[0].color != 0u) { sample.albedo *= sample.color;}
    if hit.material.albedo.size != 0u {
        sample.albedo *= to_linear_4(sample_texture(hit.material.albedo, sample.texcoords[hit.material.albedo_texcoord]));
    }

    sample.emissive = hit.material.emissive_factor;
    if hit.material.emissive.size != 0u {
        sample.emissive *= to_linear(sample_texture(hit.material.emissive, sample.texcoords[hit.material.emissive_texcoord]).rgb);
    }

    sample.transmission = hit.material.transmission_factor;
    if hit.material.transmission.size != 0u {
        sample.transmission *= sample_texture(hit.material.transmission, sample.texcoords[hit.material.transmission_texcoord]).r;
    }

    sample.thickness = hit.material.thickness_factor;
    if hit.material.thickness.size != 0u {
        sample.thickness *= sample_texture(hit.material.thickness, sample.texcoords[hit.material.thickness_texcoord]).r;
    }

    sample.metallic_roughness = vec3f(0.0, hit.material.roughness_factor, hit.material.metallic_factor);
    if hit.material.metallic_roughness.size != 0u {
        sample.metallic_roughness *= sample_texture(hit.material.metallic_roughness, sample.texcoords[hit.material.metal_r_texcoord]).rgb;
    }
    sample.metallic_roughness.g = clamp(sample.metallic_roughness.g, 0.05, 1.0);

    if globals.debug_mode == 9u {
        sample.albedo = vec4(1.0);
    }
    // sample.metallic_roughness.g = 0.03;

    return sample;
}



// fn trace_to_target(ray: Ray, prim: i32, tri: i32) -> Hit {
//     var stack: Stack;
//     stack.size = 0u;
//     var node = bvh[globals.scene.node_count];

//     var hit = trace_transformed_tri(ray, prim, tri);
//     if hit.idx == -1 {
//         return hit;
//     }


//     if intersect_aabb(ray, node.aabb) > hit.t {
//         return hit_default();
//     }
//     while (true) {
//         if node.count > 0u {
//             // intersect BLAS(s) of node
//             for (var i = node.first; i < node.first + node.count; i++) {
//                 let scale_factor = length(transform_dir(ray.dir, primitives[i].inv_transform));
//                 let t_ray = transform_ray(ray, primitives[i].inv_transform);
//                 var new_t = hit.t * scale_factor;
//                 let new_tri = trace_bvh(t_ray, primitives[i].bvh_idx, &new_t, primitives[i]);
                
//                 if new_tri >= 0 {
//                     // hit something before target
//                     return hit_default();
//                 }
//             }
//             if stack.size == 0u {
//                 break;
//             }
//             node = bvh[pop(&stack)];
//         } else {
//             // avoid pushing nodes onto the stack where possible
//             // order nodes based on distance

//             // TLAS is tacked onto end of bvh:
//             let node_first = globals.scene.node_count + node.first;

//             // try ordering the nodes
//             let left  = intersect_aabb(ray, bvh[node_first + 0u].aabb);
//             let right = intersect_aabb(ray, bvh[node_first + 1u].aabb);
    
//             if (left > hit.t) && (right > hit.t) {
//                 if stack.size == 0u {
//                     break;
//                 }
//                 node = bvh[pop(&stack)];
//             } else if (left > hit.t) {
//                 node = bvh[node_first + 1u];
//             } else if (right > hit.t) {
//                 node = bvh[node_first + 0u];
//             } else if right < left {
//                 // push(&stack, node_first + 1u);
//                 // node = bvh[node_first + 0u];

//                 push(&stack, node_first + 0u);
//                 node = bvh[node_first + 1u];

//             } else {
//                 // push(&stack, node_first + 0u);
//                 // node = bvh[node_first + 1u];

                
//                 push(&stack, node_first + 1u);
//                 node = bvh[node_first + 0u];
//             }

//         }
//     }

//     return hit;
// }

fn area_to_sa(area: f32, normal: vec3f, p: vec3f, reference: vec3f) -> f32 {
    let d = (p - reference);
    return area * dot(d, d) / abs(dot(normalize(-d), normal));
}

/// uniform sample a point on a triangle - *returns area pdf*
fn sample_triangle_area(prim: i32, tri: i32, pdf: ptr<function, f32>, normal: ptr<function, vec3f>) -> vec3f {
    let e1 = sqrt(rand());
    let e2 = rand();

    let u = 1.0 - e1;
    let v = e2 * e1;

    let transform = primitives[prim].transform;
    let lt = triangles[tri];

    let v0 = transform_pos(lt.d0.xyz, transform);
    let v1 = transform_pos(lt.d1.xyz, transform);
    let v2 = transform_pos(lt.d2.xyz, transform);

    let a = (v1 - v0);
    let b = (v2 - v0);

    *normal = normalize(cross(a, b));

    *pdf = 1.0 / (0.5 * length(cross(a, b)));

    return v0 + a * u + b * v;
}

fn sample_triangle_pdf(prim: i32, tri: i32, point: vec3f, normal: vec3f, reference: vec3f) -> f32 {
    let transform = primitives[prim].transform;
    let lt = triangles[tri];

    let v0 = transform_pos(lt.d0.xyz, transform);
    let v1 = transform_pos(lt.d1.xyz, transform);
    let v2 = transform_pos(lt.d2.xyz, transform);

    let a = (v1 - v0);
    let b = (v2 - v0);
    let area = 0.5 * length(cross(a, b));
    if area == 0.0 {
        return 0.0;
    }
    return area_to_sa(1.0 / area, normal, point, reference);
}

// MARK: - Mesh Lights

fn sample_mesh_light(reference: vec3f, pdf: ptr<function, f32>, prim_idx: ptr<function, i32>, tri_idx: ptr<function, i32>) -> vec3f {

    let dim = globals.scene.mesh_light_count;

    if dim == 0u {
        *pdf = 0.0;
        return vec3f(0.0, 0.0, 1.0);
    }

    var c = 0;
    {
        var low = 0;
        var high = i32(dim) - 1;
        var cdf = 0.0;
        var cdf_target = rand();

        while high >= low {
            let mid = (low + high) / 2;
            cdf = mesh_lights[mid].cdf;
            if cdf < cdf_target {
                low = mid + 1;
            } else {
                high = mid - 1;
                c = mid;
            }
            
        }
    }
    var tri_pdf: f32;
    var normal: vec3f;
    let p = sample_triangle_area(mesh_lights[c].prim, mesh_lights[c].tri, &tri_pdf, &normal);
    normal *= sign11(dot(normal, reference - p));
    tri_pdf = area_to_sa(tri_pdf, normal, p, reference);
    *pdf = mesh_lights[c].power * tri_pdf;
    *prim_idx = mesh_lights[c].prim;
    *tri_idx = mesh_lights[c].tri;
    return p;
}

// 
fn sample_mesh_light_pdf(prim: i32, tri: i32, point: vec3f, reference: vec3f) -> f32 {
    if globals.scene.mesh_light_count == 0u {
        return 0.0;
    }
    let base = primitives[prim].flags >> 8u;
    if (primitives[prim].flags & (1u << 1u)) == 0u {
        // not am emissive primitive
        return 0.0;
    }

    var tri_pdf: f32;
    var m_normal: vec3f;
    let p = sample_triangle_area(prim, tri, &tri_pdf, &m_normal);
    m_normal *= sign11(dot(m_normal, reference - p));
    tri_pdf = area_to_sa(tri_pdf, m_normal, p, reference);

    let triangle_pdf = mesh_lights[i32(base) + (tri - i32(primitives[prim].tri_start))].power;

    return triangle_pdf * tri_pdf;
}


// MARK: Environment map

fn env_map_to_dir(e: vec2f) -> vec3f {
    var v = vec3f(0.0);
    v.z = cos(e.y * pi);
    let phi = e.x * 2.0 * pi - pi;
    let sin_theta = sin(e.y * pi);
    v.x = sin_theta * cos(phi);
    v.y = sin_theta * sin(phi);
    return v.yxz;
}

/// Returns the index of the chosen pixel
fn sample_env_map() -> vec3f {
    if globals.debug_mode == 9u {
        return rand_sphere();
    }
    let dim = textureDimensions(env_map);
    var row = 0u;
    {
        var row_low = 0u;
        var row_high = dim.y - 1u;
        var row_mid = (row_low + row_high) / 2u;
        var cdf = 0.0;
        var cdf_target = rand();

        while row_high >= row_low {
            
            cdf = env_map_rows_cdf[row_mid];
            if cdf < cdf_target {
                row_low = row_mid + 1u;
            } else {
                row_high = row_mid - 1u;
                row = row_mid;
            }
            row_mid = (row_low + row_high) / 2u;
        }
    }

    var col = 0u;
    {

        var col_low = 0u;
        var col_high = dim.x - 1u;
        var col_mid = (col_low + col_high) / 2u;
        var cdf = 0.0;
        var cdf_target = rand();

        while col_high >= col_low {
            
            cdf = textureLoad(env_map_col_cdf, vec2u(col_mid, row), 0).r;
            if cdf < cdf_target {
                col_low = col_mid + 1u;
            } else {
                col_high = col_mid - 1u;
                col = col_mid;
            }
            col_mid = (col_low + col_high) / 2u;
        }
    }
    let res = vec2f(textureDimensions(env_map));

    let uv = vec2f(f32(col) + rand(), f32(row) + rand()) / res;
    
    return env_map_to_dir(uv);
}
fn dir_to_env_map(dir: vec3f) -> vec2f {
    // HACK: For comparison with blender, use their coordinate system for sampling the HDRI
    let d = dir.yxz;
    return vec2f(atan2(d.y, d.x) + pi, acos(d.z)) / vec2f(2.0 * pi, pi);
}

fn sample_env_map_pdf(dir: vec3f) -> f32 {
    if globals.debug_mode == 9u {
        return 1.0 / (4.0 * pi);
    }

    let uv = dir_to_env_map(dir);
    let res = vec2f(textureDimensions(env_map));

    return textureLoad(env_map_pdf, vec2u(uv * res), 0).r;
}

fn evaluate_env_map(dir: vec3f) -> vec4f {
    if globals.debug_mode == 9u {
        return vec4f(0.5, 0.5, 0.5, 1.0);
    }

    let uv = dir_to_env_map(dir);
    return textureLoad(env_map, vec2u(uv * vec2f(textureDimensions(env_map))), 0); 
}


// from https://www.shadertoy.com/view/XtGGzG
fn plasma_quintic( _x: f32 ) -> vec3f {
	let x = saturate( _x );
	let x1 = vec4f( 1.0, x, x * x, x * x * x ); // 1 x x2 x3
	let x2 = x1 * x1.w * x; // x4 x5 x6 x7
	return vec3f(
		dot( x1.xyzw, vec4f(0.063861086, 1.992659096, -1.023901152, -0.490832805 ) ) + dot( x2.xy, vec2f( 1.308442123, -0.914547012 ) ),
		dot( x1.xyzw, vec4f(0.049718590, -0.791144343, 2.892305078, 0.811726816 ) ) + dot( x2.xy, vec2f( -4.686502417, 2.717794514 ) ),
		dot( x1.xyzw, vec4f(0.513275779, 1.580255060, -5.164414457, 4.559573646 ) ) + dot( x2.xy, vec2f( -1.916810682, 0.570638854 ) ) );
}
// from https://www.shadertoy.com/view/XtGGzG
fn magma_quintic( _x: f32 ) -> vec3f {
	let x = saturate( _x );
	let x1 = vec4f( 1.0, x, x * x, x * x * x ); // 1 x x2 x3
	let x2 = x1 * x1.w * x; // x4 x5 x6 x7
	return vec3f(
		dot( x1.xyzw, vec4( -0.023226960, 1.087154378, -0.109964741, 6.333665763 ) ) + dot( x2.xy, vec2( -11.640596589, 5.337625354 ) ),
		dot( x1.xyzw, vec4( 0.010680993, 0.176613780, 1.638227448, -6.743522237 ) ) + dot( x2.xy, vec2( 11.426396979, -5.523236379 ) ),
		dot( x1.xyzw, vec4( -0.008260782, 2.244286052, 3.005587601, -24.279769818 ) ) + dot( x2.xy, vec2( 32.484310068, -12.688259703 ) ) );
}

fn to_linear(srgb: vec3f) -> vec3f {
    // not correct but close enough for now
    return pow(srgb, vec3f(2.2));
}

fn to_linear_4(srgb: vec4f) -> vec4f {
    // not correct but close enough for now
    return pow(srgb, vec4f(2.2));
}

// false color visualizations, x will be clamped from 0..1
fn ramp(x: f32) -> vec3f {
    if x < 0.0 {
        return vec3f(0.0, 0.0, 1.0);
    }
    if x > 1.0 {
        return vec3f(1.0, 1.0, 0.0);
    }
    return to_linear(clamp(magma_quintic(x), vec3f(0.0), vec3f(1.0)));
}


fn project_to_hemisphere(dir: vec3f, normal: vec3f) -> vec3f {
    if dot(dir, normal) < 0.0 {
        return normalize(dir - dot(dir, normal) * normal);
    } else {
        return dir;
    }
}

fn sample_lambert(normal: vec3f) -> vec3f {
    // from raytracing in one weekend
    return normalize(normal + rand_sphere());
}

// cosign hemisphere sampling pdf
fn sample_lambert_pdf(wi: vec3f, n: vec3f) -> f32 {
    return max(0.0, dot(wi, n)) / pi;
}

fn evaluate_lambert(to_light: vec3f, normal:  vec3f) -> f32 {
    return max(dot(to_light, normal), 0.0) / pi;
}

// MARK: GGX
// from https://schuttejoe.github.io/post/ggximportancesamplingpart2/
fn evaluate_ggx_smith_masking(o_tangent: vec3f, a2: f32) -> f32 {
    let dotNV = o_tangent.z;
    let denomC = sqrt(a2 + (1.0 - a2) * dotNV * dotNV) + dotNV;

    return 2.0 * dotNV / denomC;
}

// from https://schuttejoe.github.io/post/ggximportancesamplingpart2/
fn ggx_smith_g_fast(i_tangent: vec3f, o_tangent: vec3f, a2: f32) -> f32 {
    let dotNL = i_tangent.z;
    let dotNV = o_tangent.z;

    let denomA = dotNV * sqrt(a2 + (1.0 - a2) * dotNL * dotNL);
    let denomB = dotNL * sqrt(a2 + (1.0 - a2) * dotNV * dotNV);

    return 2.0 * dotNL * dotNV / (denomA + denomB);
}

// https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
// walter GGX paper
fn ggx_smith_g1_general(HoV: f32, NoV: f32, a2: f32) -> f32 {
    let fac = step(0.0, HoV / NoV);
    let tan_theta_squared = 1.0 / (NoV * NoV) - 1.0;
    return fac * (2.0 / (1.0 + sqrt(1.0 + a2 * tan_theta_squared)));
}


fn evaluate_fresnel_dielectric(HoL: f32, ior_i: f32, ior_o: f32) -> f32 {
    let c = abs(HoL);
    let radicand = (ior_o * ior_o) / (ior_i * ior_i) - 1.0 + c * c;
    if radicand < 0.0 {
        return 1.0;
    }
    let g = sqrt( radicand );

    let fac = 0.5 * (((g - c) * (g - c)) / ((g + c) * (g + c)));
    let sqrt_rhs = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
    return fac * (1.0 + sqrt_rhs * sqrt_rhs);
}

fn ggx_smith_g_general(wi: vec3f, wo: vec3f, h: vec3f, n: vec3f, a2: f32) -> f32 {
    return ggx_smith_g1_general(dot(wi, h), dot(wi, n), a2) * ggx_smith_g1_general(dot(wo, h), dot(wo, n), a2);
}

// walter paper, microfacet models for refraction
// reversed convention, im using wo to refer to rays pointing back toward the camera
fn evaluate_ggx_transmission(wi: vec3f, wo: vec3f, n_facing: vec3f, a2: f32, ior_i: f32, ior_o: f32, F: f32) -> vec3f {

    // they expect surface normals point into the medium with the lower index of refraction
    let n = n_facing * sign(ior_i - ior_o);

    var h = -normalize(wi * ior_i + wo * ior_o);

    let NoH = dot(h, n);
    let NoL = dot(n, wi);
    let NoV = dot(n, wo);
    let HoV = dot(wo, h);
    let HoL = dot(wi, h);

    let D = ggx_d(NoH, a2);

    let G = ggx_smith_g_general(wi, wo, h, n, a2);
    
    let denom_sqrt = ior_i * HoL + ior_o * HoV;
    
    if denom_sqrt == 0.0 {
        return vec3f(0.0);
    }
    
    let factor = (abs(HoL) * abs(HoV)) / max(abs(NoL) * abs(NoV), 1e-8);

    return vec3f(factor * (ior_i * ior_i * D * (1.0 - F) * G) / (denom_sqrt * denom_sqrt ));
}

fn evaluate_fresnel_schlick(normal: vec3f, view: vec3f, f0: vec3f) -> vec3f {
    return saturate(f0 + (1.0 - f0) * pow(1.0 - dot(normal, view), 5.0));
}

// one_minus_NoH_squared from
// https://github.com/google/filament/blob/main/shaders/src/surface_brdf.fs
// for floating point precision concerns at low roughness
fn ggx_smith_d_precise(h_tangent: vec3f, a2: f32) -> f32 {
    let NoH = saturate(h_tangent.z);
    let NxH = cross(vec3f(0.0, 0.0, 1.0), h_tangent);
    let one_minus_NoH_squared = dot(NxH, NxH);
    let D_denom_sqr = (NoH * NoH) * a2 + one_minus_NoH_squared;
    return a2 / (D_denom_sqr * D_denom_sqr * pi);
}

// https://jcgt.org/published/0007/04/01/paper.pdf
fn sample_ggx_smith_vndf(view_tangent: vec3f, roughness: f32) -> vec3f {
    let a = roughness * roughness;

    // hemisphere scaled by roughness
    let vh = normalize(vec3f(a, a, 1.0) * view_tangent);

    let len_2 = vh.x * vh.x + vh.y * vh.y;
    let t1 = select(vec3f(1.0, 0.0, 0.0), vec3f(-vh.y, vh.x, 0.0) * inverseSqrt(len_2), len_2 > 0.0);
    let t2 = cross(vh, t1);

    let r = sqrt(rand());
    let phi = 2.0 * pi * rand();

    let c1  = r * cos(phi);
    var c2  = r * sin(phi);
    let s = 0.5 * (1.0 + vh.z);

    c2 = (1.0 - s) * sqrt(1.0 - c1 * c1) + s * c2;

    // re-project back to hemisphere
    let nh = c1 * t1 + c2 * t2 + sqrt(max(0.0, 1.0 - c1 * c1 - c2 * c2)) * vh;

    let ne = normalize(vec3f(a * nh.x, a * nh.y, max(0.0, nh.z)));
    return ne;
}

/// returns the pdf with respect to wi_tangent ( reflect(-wo_tangent, h_tangent) )
fn sample_ggx_smith_vndf_reflection_pdf(wo_tangent: vec3f, h_tangent: vec3f, r: f32) -> f32 {
    let a2 = r * r * r * r;
    
    let D = ggx_smith_d_precise(h_tangent, a2);
    
    if wo_tangent.z <= 0.0 || h_tangent.z <= 0.0 {
        return 0.0;
    }
    let G1 = evaluate_ggx_smith_masking(wo_tangent, a2);
    // let pdf_h = G1 * max(0.0, dot(wo_tangent, h_tangent)) * D / wo_tangent.z;
    // let reflect_jacobian = 1.0 / (4.0 * saturate(dot(wo_tangent, h_tangent)));
    let pdf_h_times_jacobian = G1 * D / (wo_tangent.z * 4.0);

    return pdf_h_times_jacobian;
}

fn invalid_refraction(v: vec3f) -> bool {
    return v.x + v.y + v.z == 0.0;
}

// for refract(incident_ray, normal, eta)
// wo is my incident ray (from the camera to the surface)
fn refraction_jacobian(ior_i_ray: f32, ior_o_ray: f32, i_ray_dot_h: f32, o_ray_dot_h: f32) -> f32 {
    let denom_sqrt = ior_o_ray * o_ray_dot_h + ior_i_ray * i_ray_dot_h;
    let num = ior_o_ray * ior_o_ray * abs(o_ray_dot_h);
    return num / (denom_sqrt * denom_sqrt);
}

/// D_Ve(Ne) = G1(Ve) * max(0, dot(Ve, Ne)) * D(Ne) / Ve.z
fn sample_ggx_smith_vndf_pdf(h: vec3f, n: vec3f, v: vec3f, a2: f32) -> f32 {
    let NoH = dot(n, h);
    let VoH = dot(v, h);
    let VoN = dot(v, n);

    let tan_theta_squared = 1.0 / (NoH * NoH) - 1.0;
    let num = a2 * step(0.0, NoH);
    let d_denom_sqrt = NoH * NoH * (a2 + tan_theta_squared);
    let D = num / (pi * d_denom_sqrt * d_denom_sqrt);

    let G1 = ggx_smith_g1_general(VoH, VoN, a2);

    return G1 * max(0.0, VoH) * D / VoN;
}


fn sample_ndf(a: f32) -> vec3f {
    let e1 = rand();
    let e2 = rand();
    let theta = atan((a * sqrt(e1)) / sqrt(1.0 - e1));
    let phi = 2.0 * pi * e2;
    let st = sin(theta);
    let ct = cos(theta);
    return vec3f(st * cos(phi), st * sin(phi), ct);
}

fn ggx_d(NoH: f32, a2: f32) -> f32 {
    let tan_theta_squared = 1.0 / (NoH * NoH) - 1.0;
    let num = a2 * step(0.0, NoH);
    let d_denom_sqrt = NoH * NoH * (a2 + tan_theta_squared);
    return num / (pi * d_denom_sqrt * d_denom_sqrt);
}

fn sample_ndf_pdf(NoH: f32, a2: f32) -> f32 {
    return ggx_d(NoH, a2) * abs(NoH);
}


/// returns the pdf with respect to the refracted direction wi_tangent
/// wo_tangent is the incident ray from the camera
fn sample_ggx_smith_vndf_refraction_pdf(wi_tangent: vec3f, wo_tangent: vec3f, ior_i: f32, ior_o: f32, roughness: f32) -> f32 {
    let a2 = roughness * roughness * roughness * roughness;

    var h_required = -normalize(wi_tangent * ior_i + wo_tangent * ior_o) * sign(ior_i - ior_o);

    let n = vec3f(0.0, 0.0, 1.0);

    let NoH = dot(n, h_required);
    let VoH = dot(wo_tangent, h_required);
    let VoN = dot(wo_tangent, n);

    let LoH = dot(wi_tangent, h_required);

    let D = ggx_d(NoH, a2);

    let eta = ior_o / ior_i;
    let denom = pow(LoH + VoH * eta, 2.);
    let jacobian = abs(LoH) / denom;

    let G1 = ggx_smith_g1_general(VoH, VoN, a2);

    let pdf = G1 * max(0.0, VoH) * D * jacobian / VoN;

    return pdf;
}



/// convert a direction in my coordinate space into the space used by the reference GLTF viewer
fn compare_gltf_space(v: vec3f) -> vec3f {
   return v.xzy * vec3f(-1.0, 1.0, 1.0);
}

fn background_volume() -> GpuVolume {
    // return GpuVolume(vec3f(0.0), 1.0, vec3f(0.0), 0.0);
    return media[0];
}

fn sample_metallic_bsdf(wo_tangent: vec3f, h_tangent: vec3f, sample: ExtSample) -> vec3f {
    return reflect(-wo_tangent, h_tangent);
}

fn sample_metallic_bsdf_pdf(wi_tangent: vec3f, wo_tangent: vec3f, sample: ExtSample) -> f32 {
    let h_tangent = normalize(wo_tangent + wi_tangent);
    if h_tangent.z <= 0.0 || wo_tangent.z * wi_tangent.z < 0.0 {
        return 0.0;
    }

    return sample_ggx_smith_vndf_reflection_pdf(wo_tangent, h_tangent, sample.metallic_roughness.g);
}

fn sample_dielectric_bsdf(wo_tangent: vec3f, h_tangent: vec3f, ior_i: f32, ior_o: f32, sample: ExtSample) -> vec3f {
    let fresnel = evaluate_fresnel_dielectric(dot(wo_tangent, h_tangent), ior_o, ior_i);

    if rand() < fresnel {
        return reflect(-wo_tangent, h_tangent);
    } else {
        if rand() < sample.transmission {
            var wi_tangent = refract(-wo_tangent, h_tangent, ior_o / ior_i);
            if all(wi_tangent == vec3f(0.0)) {
                wi_tangent = reflect(-wo_tangent, h_tangent);
            }
            return wi_tangent;
        } else {
            return sample_lambert(vec3f(0.0, 0.0, 1.0));
        }

    }
}

fn sample_dielectric_bsdf_pdf(wi_tangent: vec3f, wo_tangent: vec3f, ior_i: f32, ior_o: f32, sample: ExtSample) -> f32 {
    let transmitted = wi_tangent.z < 0.0;
    var h_tangent: vec3f;
    
    if transmitted {
        h_tangent = -normalize(wi_tangent * ior_i + wo_tangent * ior_o) * sign(ior_i - ior_o);
    } else {
        h_tangent = normalize(wi_tangent + wo_tangent);
    }

    if h_tangent.z < 0.0 {

        return 0.0;
    }

    let fresnel = evaluate_fresnel_dielectric(dot(wo_tangent, h_tangent), ior_o, ior_i);

    var diffuse_pdf = sample_lambert_pdf(wi_tangent, vec3f(0.0, 0.0, 1.0));
    var transmission_pdf = sample_ggx_smith_vndf_refraction_pdf(wi_tangent, wo_tangent, ior_i, ior_o, sample.metallic_roughness.g);
    var specular_pdf = sample_ggx_smith_vndf_reflection_pdf(wo_tangent, h_tangent, sample.metallic_roughness.g);

    if transmitted {
        diffuse_pdf = 0.0;
        specular_pdf = 0.0;
    } else {
        transmission_pdf = 0.0;
    }


    let pdf = mix(mix(diffuse_pdf, transmission_pdf, sample.transmission), specular_pdf, fresnel);

    return pdf;

}


//MARK: Surface Hit
fn sample_bsdf(wo: vec3f, ior_o: f32, ior_i: f32, sample: ExtSample, lighting: ptr<function, vec3f>, pdf: ptr<function, f32>, bsdf: ptr<function, vec3f>) -> vec3f {
    
    var debug_color = vec3f(0.0);
    *pdf = 0.0;
    *bsdf = vec3f(0.0);

    let tbn = orthonormal_basis(sample.normal); //* sign(ior_i - ior_o)
    let wo_tangent = normalize(wo * tbn);
    let h_tangent = sample_ggx_smith_vndf(wo_tangent, sample.metallic_roughness.g);

    var metallic_chance = sample.metallic_roughness.b;


    // debug_color = vec3f(normal_tangent * 0.5 + 0.5);
    if globals.debug_mode == 1u {
        debug_color = sample.albedo.rgb;
    }
    if globals.debug_mode == 2u {
        debug_color = compare_gltf_space(sample.normal) * 0.5 + 0.5;
    }
    if globals.debug_mode == 4u {
        debug_color = compare_gltf_space(sample.tangent) * 0.5 + 0.5;
    }
    if globals.debug_mode == 5u {
        debug_color = clamp(vec3f(-sample.t_sign, sample.t_sign, 0.0), vec3f(0.0), vec3f(1.0));
    }

    var wi_tangent: vec3f;  
    if rand() < metallic_chance {
        wi_tangent = sample_metallic_bsdf(wo_tangent, h_tangent, sample);
    } else {
        wi_tangent = sample_dielectric_bsdf(wo_tangent, h_tangent, ior_i, ior_o, sample);
    }
     
    var wi = tbn * wi_tangent;

    *pdf = sample_bsdf_pdf(wi, wo, ior_i, ior_o, sample);
    *bsdf = evaluate_bsdf(wi, wo, ior_i, ior_o, sample);

     
    if DEBUG && globals.debug_mode != 0u {
        *lighting = debug_color;
    }



    return wi;
}

// what is the probability the brdf sampling returned wi given the surface properties
// maybe not 100% accurate, hopefully good enough for MIS weighting
fn sample_bsdf_pdf(wi: vec3f, wo: vec3f, ior_i: f32, ior_o: f32, sample: ExtSample) -> f32 {
    if dot(wo, sample.normal) < 0.0 {
        return 0.0;
    }

    let tbn = orthonormal_basis(sample.normal);
    let wi_tangent = wi * tbn;
    let wo_tangent = wo * tbn;

    let metal = sample_metallic_bsdf_pdf(wi_tangent, wo_tangent, sample);
    let dielectric = sample_dielectric_bsdf_pdf(wi_tangent, wo_tangent, ior_i, ior_o, sample);
    
    return mix(dielectric, metal, sample.metallic_roughness.b);
}

/// wo_tangent is the vector pointing toward the viewer
/// wi_tangent is the vector pointing toward the light
fn evaluate_ggx(wo_tangent: vec3f, wi_tangent: vec3f, F: vec3f, a2: f32) -> vec3f {
    let h_tangent = normalize(wi_tangent + wo_tangent);

    let NoH = max(h_tangent.z, 0.0);
    let NoV = max(wo_tangent.z, 0.0);
    let NoL = max(wi_tangent.z, 0.0);
    
    let D = ggx_smith_d_precise(h_tangent, a2);
    let G = ggx_smith_g_fast(wo_tangent, wi_tangent, a2);

    let denom = 4.0 * NoV * NoL;
    if denom <= 0.0 {
        return vec3f(0.0);
    }

    return (D * F * G) / denom;
}

fn evaluate_metal(wi_tangent: vec3f, wo_tangent: vec3f, sample: ExtSample) -> vec3f {
    let h_tangent = normalize(wi_tangent + wo_tangent);
    if h_tangent.z <= 0.0 || wi_tangent.z * wo_tangent.z <= 0.0 {
        return vec3f(0.0);
    }
    let fresnel = evaluate_fresnel_schlick(h_tangent, wo_tangent, sample.albedo.rgb);

    let a = sample.metallic_roughness.g * sample.metallic_roughness.g;

    return evaluate_ggx(wo_tangent, wi_tangent, fresnel, a * a);
}

fn evaluate_dielectric(wi_tangent: vec3f, wo_tangent: vec3f, ior_i: f32, ior_o: f32, sample: ExtSample) -> vec3f {
    var h_tangent: vec3f;
    var transmitted = false;

    if wi_tangent.z > 0.0 {
        h_tangent = normalize(wi_tangent + wo_tangent);
    } else {
        transmitted = true;
        h_tangent = -normalize(wi_tangent * ior_i + wo_tangent * ior_o) * sign(ior_i - ior_o);
    }

    let fresnel = evaluate_fresnel_dielectric(dot(wo_tangent, h_tangent), ior_o, ior_i);

    let a = sample.metallic_roughness.g * sample.metallic_roughness.g;
    let a2 = a * a;

    var specular = evaluate_ggx(wo_tangent, wi_tangent, vec3f(fresnel), a2);
    var diffuse = sample.albedo.rgb / pi * (1.0 - fresnel);
    var transmission = sample.albedo.rgb * evaluate_ggx_transmission(wi_tangent, wo_tangent, vec3f(0.0, 0.0, 1.0), a2, ior_i, ior_o, fresnel);

    if transmitted {
        specular = vec3f(0.0);
        diffuse = vec3f(0.0);
    } else {
        transmission = vec3f(0.0);
    }
    
    // all BXDF's contain the fresnel term already
    return mix(diffuse, transmission, sample.transmission) + specular;
}

fn evaluate_bsdf(wi: vec3f, wo: vec3f, ior_i: f32, ior_o: f32, sample: ExtSample) -> vec3f {
    let metallic = sample.metallic_roughness.b;
    let f0 = mix(vec3f(0.04), sample.albedo.rgb, metallic);

    let tbn = orthonormal_basis(sample.normal);
    let wi_tangent = normalize(wi * tbn);
    let wo_tangent = normalize(wo * tbn);

    let dielectric = evaluate_dielectric(wi_tangent, wo_tangent, ior_i, ior_o, sample);
    let metal = evaluate_metal(wi_tangent, wo_tangent, sample);
    return mix(dielectric, metal, metallic);

} 

fn mis_power_heuristic(a: f32, b: f32, a_prob: f32, b_prob: f32) -> f32 {
    let ap = a * a_prob;
    let bp = b * b_prob;
    if (ap <= 0.0) {
        return 0.0;
    } else if (bp <= 0.0) {
        return 1.0;
    }
    
    // return a_prob;
    return ap  / (ap + bp);
    //return (ap * ap) / (ap * ap + bp * bp);
}

fn mis_power_heuristic_3(a: f32, b: f32, c: f32, a_prob: f32, b_prob: f32, c_prob: f32) -> f32 {
    let ap = a * a_prob;
    let bp = b * b_prob;
    let cp = c * c_prob;
    if ap <= 0.0 {
        return 0.0;
    } else if bp <= 0.0 && cp <= 0.0 {
        return 1.0;
    }
    
    // return a_prob;
    return ap  / (ap + bp + cp);
    //return (ap * ap) / (ap * ap + bp * bp + cp * cp);
}

const MESH_CHANCE: f32 = 0.5;

struct LightSample {
    wi: vec3f,
    t_max: f32,
    contrib: vec3f,
    pdf: f32,
}

fn sample_light(reference: vec3f) -> LightSample {
    
    let r = rand();
    var prob = 0.0;
    var pdf = 0.0;
    var wi = vec3f(0.0, 0.0, 1.0);
    var t_max = 0.0;
    var contrib = vec3f(0.0);

    // sample NEE shadow ray
    if globals.scene.mesh_light_count > 0u && r < MESH_CHANCE {
        prob = MESH_CHANCE;

        var prim_idx: i32;
        var tri_idx: i32;
        
        let mesh_point = sample_mesh_light(reference, &pdf, &prim_idx, &tri_idx);
        
        wi = normalize(mesh_point - reference);

        var mesh_ray: Ray;
        mesh_ray.origin = reference;
        mesh_ray.dir = wi;
        mesh_ray.idir = 1.0 / mesh_ray.dir;

        let mesh_hit = trace_transformed_tri(mesh_ray, prim_idx, tri_idx);

        t_max = distance(reference, mesh_point) - 0.003;
        contrib = sample_emission(mesh_hit);

    } else {
        prob = 1.0 - MESH_CHANCE;
        if globals.scene.mesh_light_count == 0u {
            prob = 1.0;
        }
        wi = normalize(sample_env_map());
        t_max = 99999999999.0;
        contrib = evaluate_env_map(wi).rgb;
        pdf = sample_env_map_pdf(wi);
    }

    return LightSample(wi, t_max, contrib, pdf * prob);
}

fn sample_light_pdf(reference: vec3f, point: vec3f, prim: i32, tri: i32) -> f32 {

    if globals.scene.mesh_light_count == 0u {
        return sample_env_map_pdf(normalize(point - reference)); 
    }

    if (tri > 0) {
        // a physical light, can't have been the env sampler
        return sample_mesh_light_pdf(prim, tri, point, reference) * MESH_CHANCE;
    } else {
        // a ray miss light, can't have been the mesh sampler
        return sample_env_map_pdf(normalize(point - reference)) * (1.0 - MESH_CHANCE);
    }
}


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
    
    let hit_state = ray_hit_queue[lid];
    var hit = trace_transformed_tri(ray, hit_state.prim, hit_state.tri);
    let point = ray.origin + ray.dir * hit_state.t;

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

fn add_contrib(lighting: vec3f, pixel: u32) {
    screen[pixel] += vec4f(max(lighting, vec3f(0.0)), 0.0);
    // var lighting = _lighting;
    // if (debug < 0.0) {
    //     lighting = vec3f(1.0, 1.0, 0.0);
    // }

    // if (globals.reject_hist > 0u) {
    //     screen[pixel] = vec4f(max(lighting, vec3f(0.0)), 1.0);
    // } else {
    //     // clamp NaNs
    //     // screen[id.x + globals.res.x * id.y] += vec4f(lighting, 1.0);
    //     screen[pixel] += vec4f(max(lighting, vec3f(0.0)), 1.0);
    // }
}

fn on_kill(pixel: u32) {
    screen[pixel].w += 1.0;
}



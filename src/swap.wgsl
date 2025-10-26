#import "common.wgsl"

@compute
@workgroup_size(1, 1)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    atomicStore(&ray_queue_meta.num_vis_rays, 0u);
    let in = atomicExchange(&ray_queue_meta.num_out_rays, 0u);
    atomicStore(&ray_queue_meta.num_in_rays, in);
}
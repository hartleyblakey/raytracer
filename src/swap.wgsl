#import "common.wgsl"

@compute
@workgroup_size(1, 1)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    atomicStore(&ray_queue_meta.num_vis_rays, 0u);
    let in = atomicExchange(&ray_queue_meta.num_out_rays, 0u);
    atomicStore(&ray_queue_meta.num_in_rays, in);

    let wg_size = 8u * 8u;
    let groups = (in + wg_size - 1u) / wg_size;

    // dispatch a trace kernel for each incoming ray
    ray_queue_meta.in_rays_indirect = IndirectParams(groups, 1u, 1u);

    // just assume all in rays generate a visibility ray. Overly pessimistic but saves the
    // effort of another dispatch
    ray_queue_meta.vis_rays_indirect = IndirectParams(groups, 1u, 1u);

    // ray_queue_meta.out_rays_indirect = IndirectParams(in, 1u, 1u); not actually used atm

}
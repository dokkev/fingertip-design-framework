#include <optix.h>
#include <cuda_runtime.h>

struct LaunchParams
{
    const float3* origins;
    const float3* directions;
    unsigned int* results;
    OptixTraversableHandle handle;
    unsigned int mask;
};

struct HitGroupData
{
    float3 sphere_center;
    float sphere_radius;
};

extern "C" {
__constant__ LaunchParams params;
}

static __forceinline__ __device__ void set_miss_payload()
{
    optixSetPayload_0(0u);
    optixSetPayload_1(__float_as_uint(-1.0f));
    optixSetPayload_2(static_cast<unsigned int>(-1));
    optixSetPayload_3(static_cast<unsigned int>(-1));
    optixSetPayload_4(__float_as_uint(-1.0f));
    optixSetPayload_5(__float_as_uint(-1.0f));
}

extern "C" __global__ void __raygen__trace_closest()
{
    const unsigned int ray_index = optixGetLaunchIndex().x;
    unsigned int hit = 0u;
    unsigned int t = __float_as_uint(-1.0f);
    unsigned int instance_id = static_cast<unsigned int>(-1);
    unsigned int primitive_id = static_cast<unsigned int>(-1);
    unsigned int barycentric_u = __float_as_uint(-1.0f);
    unsigned int barycentric_v = __float_as_uint(-1.0f);

    optixTrace(
        params.handle,
        params.origins[ray_index],
        params.directions[ray_index],
        1.0e-7f,
        1.0e16f,
        0.0f,
        OptixVisibilityMask(params.mask),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0,
        1,
        0,
        hit,
        t,
        instance_id,
        primitive_id,
        barycentric_u,
        barycentric_v);

    unsigned int* result = params.results + 6u * ray_index;
    result[0] = hit;
    result[1] = t;
    result[2] = instance_id;
    result[3] = primitive_id;
    result[4] = barycentric_u;
    result[5] = barycentric_v;
}

extern "C" __global__ void __miss__trace_closest()
{
    set_miss_payload();
}

extern "C" __global__ void __closesthit__triangle()
{
    const float2 barycentrics = optixGetTriangleBarycentrics();
    optixSetPayload_0(1u);
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_2(optixGetInstanceId());
    optixSetPayload_3(optixGetPrimitiveIndex());
    optixSetPayload_4(__float_as_uint(barycentrics.x));
    optixSetPayload_5(__float_as_uint(barycentrics.y));
}

extern "C" __global__ void __closesthit__sphere()
{
    optixSetPayload_0(1u);
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_2(optixGetInstanceId());
    optixSetPayload_3(optixGetPrimitiveIndex());
    optixSetPayload_4(__float_as_uint(-1.0f));
    optixSetPayload_5(__float_as_uint(-1.0f));
}

extern "C" __global__ void __intersection__sphere()
{
    const HitGroupData* data =
        reinterpret_cast<const HitGroupData*>(optixGetSbtDataPointer());
    const float3 origin = optixGetObjectRayOrigin();
    const float3 offset = make_float3(
        origin.x - data->sphere_center.x,
        origin.y - data->sphere_center.y,
        origin.z - data->sphere_center.z);
    const float3 direction = optixGetObjectRayDirection();
    const float a = direction.x * direction.x
        + direction.y * direction.y
        + direction.z * direction.z;
    const float half_b = offset.x * direction.x
        + offset.y * direction.y
        + offset.z * direction.z;
    const float c = offset.x * offset.x
        + offset.y * offset.y
        + offset.z * offset.z
        - data->sphere_radius * data->sphere_radius;
    const float discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0f)
        return;

    const float square_root = sqrtf(discriminant);
    float root = (-half_b - square_root) / a;
    if (root > optixGetRayTmin() && root < optixGetRayTmax())
    {
        if (optixReportIntersection(root, 0u))
            return;
    }

    root = (-half_b + square_root) / a;
    if (root > optixGetRayTmin() && root < optixGetRayTmax())
        optixReportIntersection(root, 0u);
}

#include <optix.h>
#include <optix_device.h>

struct Params {
    OptixTraversableHandle handle;
    const float3* origins;
    const float3* directions;
    float* distances;
    unsigned int* primitives;
    float2* barycentrics;
    unsigned int* hits;
    unsigned int count;
    float tmin;
};

extern "C" {
__constant__ Params params;

extern "C" __global__ void __raygen__transport()
{
    const unsigned int index = optixGetLaunchIndex().x;
    if (index >= params.count) return;
    optixTrace(
        params.handle,
        params.origins[index],
        params.directions[index],
        params.tmin,
        1.0e20f,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0,
        1,
        0
    );
}

extern "C" __global__ void __miss__transport()
{
    const unsigned int index = optixGetLaunchIndex().x;
    params.hits[index] = 0;
    params.distances[index] = 1.0e30f;
    params.primitives[index] = 0xffffffffu;
    params.barycentrics[index] = make_float2(0.0f, 0.0f);
}

extern "C" __global__ void __closesthit__transport()
{
    const unsigned int index = optixGetLaunchIndex().x;
    params.hits[index] = 1;
    params.distances[index] = optixGetRayTmax();
    params.primitives[index] = optixGetPrimitiveIndex();
    params.barycentrics[index] = optixGetTriangleBarycentrics();
}
}

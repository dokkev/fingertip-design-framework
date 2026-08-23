#include <optix.h>
#include <cuda_runtime.h>
#include <OptiXToolkit/ShaderUtil/OptixSelfIntersectionAvoidance.h>

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

static constexpr unsigned int RESULT_WORD_COUNT = 15u;
static constexpr unsigned int NORMAL_X_WORD = 6u;
static constexpr unsigned int SPAWN_FRONT_X_WORD = 9u;
static constexpr unsigned int SPAWN_BACK_X_WORD = 12u;
static constexpr unsigned int NAN_BITS = 0x7fc00000u;

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

static __forceinline__ __device__ void write_world_normal(
    const float3 normal_object)
{
    const float3 normal_world =
        optixTransformNormalFromObjectToWorldSpace(normal_object);
    const float length_squared = normal_world.x * normal_world.x
        + normal_world.y * normal_world.y
        + normal_world.z * normal_world.z;
    unsigned int* result = params.results
        + RESULT_WORD_COUNT * optixGetLaunchIndex().x;
    if (!(length_squared > 0.0f) || !isfinite(length_squared))
    {
        result[NORMAL_X_WORD] = NAN_BITS;
        result[NORMAL_X_WORD + 1u] = NAN_BITS;
        result[NORMAL_X_WORD + 2u] = NAN_BITS;
        return;
    }

    const float inverse_length = rsqrtf(length_squared);
    result[NORMAL_X_WORD] = __float_as_uint(normal_world.x * inverse_length);
    result[NORMAL_X_WORD + 1u] =
        __float_as_uint(normal_world.y * inverse_length);
    result[NORMAL_X_WORD + 2u] =
        __float_as_uint(normal_world.z * inverse_length);
}

static __forceinline__ __device__ void write_spawn_points(
    const float3 front,
    const float3 back)
{
    unsigned int* result = params.results
        + RESULT_WORD_COUNT * optixGetLaunchIndex().x;
    result[SPAWN_FRONT_X_WORD] = __float_as_uint(front.x);
    result[SPAWN_FRONT_X_WORD + 1u] = __float_as_uint(front.y);
    result[SPAWN_FRONT_X_WORD + 2u] = __float_as_uint(front.z);
    result[SPAWN_BACK_X_WORD] = __float_as_uint(back.x);
    result[SPAWN_BACK_X_WORD + 1u] = __float_as_uint(back.y);
    result[SPAWN_BACK_X_WORD + 2u] = __float_as_uint(back.z);
}

extern "C" __global__ void __raygen__trace_closest()
{
    const unsigned int ray_index = optixGetLaunchIndex().x;
    unsigned int* result = params.results + RESULT_WORD_COUNT * ray_index;
    for (unsigned int index = NORMAL_X_WORD; index < RESULT_WORD_COUNT; ++index)
        result[index] = NAN_BITS;

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
        0.0f,
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
    float3 vertices[3];
    optixGetTriangleVertexData(vertices);

    float3 position_object;
    float3 normal_object;
    float offset_object;
    SelfIntersectionAvoidance::getSafeTriangleSpawnOffset(
        position_object,
        normal_object,
        offset_object,
        vertices[0],
        vertices[1],
        vertices[2],
        barycentrics);

    float3 position_world;
    float3 normal_world;
    float offset_world;
    SelfIntersectionAvoidance::transformSafeSpawnOffset(
        position_world,
        normal_world,
        offset_world,
        position_object,
        normal_object,
        offset_object);

    float3 spawn_front_world;
    float3 spawn_back_world;
    SelfIntersectionAvoidance::offsetSpawnPoint(
        spawn_front_world,
        spawn_back_world,
        position_world,
        normal_world,
        offset_world);

    write_world_normal(normal_object);
    write_spawn_points(spawn_front_world, spawn_back_world);

    optixSetPayload_0(1u);
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_2(optixGetInstanceId());
    optixSetPayload_3(optixGetPrimitiveIndex());
    optixSetPayload_4(__float_as_uint(barycentrics.x));
    optixSetPayload_5(__float_as_uint(barycentrics.y));
}

extern "C" __global__ void __closesthit__sphere()
{
    write_world_normal(make_float3(
        __uint_as_float(optixGetAttribute_0()),
        __uint_as_float(optixGetAttribute_1()),
        __uint_as_float(optixGetAttribute_2())));

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
        const float3 normal = make_float3(
            offset.x + root * direction.x,
            offset.y + root * direction.y,
            offset.z + root * direction.z);
        if (optixReportIntersection(
                root,
                0u,
                __float_as_uint(normal.x),
                __float_as_uint(normal.y),
                __float_as_uint(normal.z)))
            return;
    }

    root = (-half_b + square_root) / a;
    if (root > optixGetRayTmin() && root < optixGetRayTmax())
    {
        const float3 normal = make_float3(
            offset.x + root * direction.x,
            offset.y + root * direction.y,
            offset.z + root * direction.z);
        optixReportIntersection(
            root,
            0u,
            __float_as_uint(normal.x),
            __float_as_uint(normal.y),
            __float_as_uint(normal.z));
    }
}

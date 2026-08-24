결론부터
내가 지금 다시 설계하면 OptiX의 역할을 이렇게 넓힐 거야.
\[
\boxed{
\text{OptiX}
=
\text{dynamic GPU scene + hardware ray query backend}
}
\]그리고 LUMO는
\[
\boxed{
\text{LUMO}
=
\text{optical transport semantics + path field}
}
\]만 갖는다.
현재는 silicone, rigid, envelope를 각각 GAS로 만들고 각각 별도로 trace한 뒤 CuPy에서 최소 거리를 고른다. 또 매 deformed state마다 GAS를 전부 새로 BUILD하고, optixLaunch()와 GAS build 뒤에 deviceSynchronize()까지 한다.  
OptiX 관점에선 꽤 많은 걸 손해보고 있는 구조야.
우리가 더 활용할 수 있는 OptiX 기능
Priority	OptiX 기능	LUMO에서 얻는 것
P0	IAS + instances	silicone/object/carrier를 한 scene으로 합쳐 ray query 1회
P0	GAS UPDATE/refit	deformation마다 BVH full rebuild 제거
P0	persistent CUDA streams / async launch	매 ray stage의 강제 GPU sync 제거
P0	instance ID / visibility masks	surface 종류, optical/collision query를 같은 scene에서 분리
P1	OTK self-intersection avoidance	임의 ray_offset 제거, robust secondary-ray spawning
P1	ray flags	collision first-hit query를 RT Core에 최적화
P1	Hit Object / optixTraverse	geometry query와 physics를 더 깔끔하게 분리
P1/P2	SER	branch가 많아졌을 때 divergent GPU work 재정렬
Future	motion geometry / motion transforms	time-dependent collision query
Future	custom primitives / spheres / curves	swept-volume RT collision
Skip for now	CLAS / RTX Mega Geometry	현재 fingertip mesh에는 과함
Skip	opacity micromap, denoiser, textures	우리 문제와 거의 무관


이 중 P0 네 개는 실제로 쓰는 게 맞다고 봐.
1. 제일 큰 것: IAS로 scene을 하나로 만들자
현재는:
ray
 ├─ trace silicone
 ├─ trace rigid
 └─ trace envelope
          ↓
     min(t1,t2,t3)
잖아. 
OptiX 원래 구조는:
             IAS
        ┌─────┼─────┐
        ↓     ↓     ↓
   silicone object carrier
      GAS     GAS    GAS
이야.
그러면
ray
 ↓
OptiX IAS traversal
 ↓
closest hit across entire scene
한 번이면 끝.
OptiX instance에는 이미:
- transform
- instanceId
- SBT offset
- visibility mask
- child GAS handle
가 들어간다. NVIDIA Ray Tracing Documentation
그리고 hit에서:
- 어느 instance인지
- 어느 primitive인지
- distance
- barycentric coordinates
- triangle vertices
까지 직접 조회할 수 있다. NVIDIA Ray Tracing Documentation
즉 결과가:
hit.t
hit.instance_id = SILICONE / OBJECT / CARRIER
hit.primitive_id
hit.barycentric
이면 충분함.
이게 특히 A.6과 완벽하게 맞음
instance == silicone
    → Fresnel / Snell

instance == object
    → absorb

instance == carrier
    → absorb
즉 Appendix의
\[
\text{intersection geometry}
\rightarrow
\text{surface identity}
\rightarrow
\text{boundary condition}
\]을 거의 그대로 코드로 만들 수 있어.
2. Deformed silicone은 GAS를 매번 rebuild할 필요 없음
이건 우리 use case에 거의 정확히 맞는 OptiX 기능이 있었어.
OptiX는 initial GAS를
OPTIX_BUILD_FLAG_ALLOW_UPDATE
로 만들면 이후 vertex positions만 바뀌고 triangle connectivity가 동일한 경우
OPTIX_BUILD_OPERATION_UPDATE
로 acceleration structure를 refit할 수 있어.
NVIDIA 설명도 아주 명확해:
vertex data/AABB가 변한 기존 AS를 update할 수 있고, 일반적으로 rebuild보다 훨씬 빠르다. NVIDIA Ray Tracing Documentation

우리 Newton checkpoint가 정확히:
\[
\mathbf x_i^{(k)}
=
\mathbf X_i+\mathbf u_i^{(k)}
\]이고 connectivity는 그대로니까 정석적인 refit case야.
현재는 매 OptixScene 생성마다 silicone/rigid/envelope GAS를 전부 BUILD해. 
앞으로는:
initialization
    ↓
build silicone GAS(ALLOW_UPDATE)
build carrier GAS(static)
build object GAS(static)
build IAS

checkpoint k
    ↓
update silicone vertex buffer
    ↓
REFIT silicone GAS
    ↓
update IAS
    ↓
trace
가 된다.
단, refit만 영원히 하면 안 됨
NVIDIA도 vertex가 initial build 위치에서 크게 움직이면 BVH quality가 떨어져 traversal이 느려질 수 있다고 경고한다. NVIDIA Developer Forums
그래서:
\[
\boxed{
\text{refit normally, rebuild when beneficial}
}
\]가 맞음.
다만 미리 deformation threshold를 상상해서 만들 필요는 없어.
실제 benchmark에서
- refit time
- traversal time
- full rebuild time
을 비교해서 정책을 정하면 됨.
3. Rigid object는 GAS 자체를 업데이트할 이유가 더 적음
이것도 꽤 좋음.
object가 rigid sphere라면 geometry 자체를 변형시키지 말고:
sphere/object GAS
       ↓
 OptixInstance transform
으로 pose만 바꾸면 돼.
OptixInstance가 3×4 affine transform을 기본 지원한다. NVIDIA Ray Tracing Documentation
즉 object 이동은:
vertex update X
GAS rebuild X

instance transform update O
IAS update O
로 갈 수 있음.
나중에 collision에서도 똑같이 사용할 수 있어.
4. visibility mask는 우리에게 의외로 엄청 유용함
OptiX instance에는 visibilityMask가 있고,
\[
\text{rayMask}\;\&\;\text{instanceMask}=0
\]이면 그 instance를 traversal에서 아예 제외한다. NVIDIA Ray Tracing Documentation
예를 들어:
0x01 silicone
0x02 object
0x04 carrier
0x08 robot
0x10 environment
라고 해놓으면,
optics에서는:
OPTICAL_MASK = 0x01 | 0x02 | 0x04
collision에서는:
COLLISION_MASK = 0x08 | 0x10
같이 쓸 수 있음.
그러면 동일한 GPU scene infrastructure를 optics와 collision이 공유할 수 있어.
이건 장기 architecture에서 꽤 중요함.
5. Collision check용 ray mode도 OptiX가 이미 제공
향후 collision에서는 nearest triangle의 barycentric coordinate 같은 게 아예 필요 없을 때가 많지.
질문은 단순히:
\[
\exists\; t\in[0,L]
\quad
\text{s.t. }
\mathbf o+t\mathbf d
\in\Gamma?
\]일 수 있어.
OptiX에는:
OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
가 이미 있다. NVIDIA SDK도 occlusion ray에 그대로 쓴다. GitHub
즉 공통 backend API를:
traceClosest(...)
traceAny(...)
정도로 만들면 예쁨.
optics
traceClosest
→ t, instance, primitive, barycentric
collision
traceAny
→ bool
그리고 collision query는 첫 hit에서 traversal을 끊을 수 있으니 더 빠르다.
6. Any-hit도 필요 없는 경우 꺼버릴 수 있음
현재 triangle input이:
triangle.flags = [GEOMETRY_FLAG_NONE]
이고 ray도 RAY_FLAG_NONE이야. 
그런데 우리의 normal optical triangle은 alpha-test 같은 any-hit 처리가 필요 없음.
OptiX SDK도 opaque geometry는:
OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
를 적극적으로 사용한다. GitHub
우리 경우 silicone/object/carrier 모두 ordinary solid surfaces니까 기본적으로:
\[
\boxed{\text{any-hit disabled}}
\]로 가도 됨.
작은 최적화지만 거의 공짜.
7. 현재 deviceSynchronize()는 상당히 의심스럽다
이건 꽤 크게 볼 부분이야.
현재 build_gas() 끝에서:
deviceSynchronize()
하고,
모든 launch() 뒤에도
deviceSynchronize()
한다. 
그런데 OptiX 문서상 optixLaunch는 CUDA stream에서 asynchronous하게 실행된다. 동기화가 필요할 때 CUDA stream/event mechanism을 쓰라고 명시돼 있다. NVIDIA Ray Tracing Documentation
지금 구조면 한 transport iteration마다:
launch silicone
SYNC

launch carrier
SYNC

launch envelope
SYNC
였던 셈.
GPU 입장에서는 상당히 답답한 구조일 가능성이 큼.
IAS로 바꾸고:
launch scene
 ↓
same CUDA stream
 ↓
subsequent CuPy kernels
로 ordering을 유지하고,
실제 host에서 값을 읽어야 할 때만 sync
하는 게 맞아.
NVIDIA 쪽도 많은 launch를 같은 stream에 밀어 넣고 마지막에만 synchronize하는 패턴을 사용한다. NVIDIA Developer Forums
이건 실제 profiling하면 큰 차이가 날 가능성이 있다.
8. OTK의 Self-Intersection Avoidance는 가져다 쓰자
이것도 굳이 우리가 만들 필요 없음.
지금은:
ray_offset = max(
    intersection_epsilon,
    8 * float32_eps * scale,
)
같은 custom epsilon을 사용하고 있잖아. 
NVIDIA OptiX Toolkit에는 아예 Self Intersection Avoidance library가 공식으로 있다.
이 라이브러리는 triangle hit의 floating-point error를 고려해서 conservative한 safe spawn point를 계산하고,
- object-space triangle error
- transforms
- world-space error
까지 반영한다. GitHub
공식 API 예시도:
getSafeTriangleSpawnOffset(...)
transformSafeSpawnOffset(...)
offsetSpawnPoint(...)
로 front/back safe point를 만들어준다. GitHub
이건 아주 마음에 들어.
Appendix에도 더 좋아짐
지금:
\(\epsilon_r\) is a manually selected numerical offset

보다는
secondary-ray origins use NVIDIA's conservative floating-point self-intersection avoidance procedure

가 더 강함.
OTK가 BSD-3-Clause라 필요한 header/function을 가져와 adaptation하는 것도 가능하다. NVIDIA Developer
9. optixTraverse + Hit Object도 우리 구조에 꽤 잘 맞음
OptiX의 전통적인:
optixTrace
  ↓
closest-hit shader
뿐 아니라 최신 OptiX에는:
optixTraverse(...)
가 있어.
이건 traversal만 수행하고 closest-hit/miss shader를 바로 호출하지 않고 Hit Object로 결과를 남긴다. NVIDIA Ray Tracing Documentation
그다음 raygen에서:
isHit()
getRayTmax()
getInstanceId()
getPrimitiveIndex()
getTriangleBarycentrics()
getTriangleVertexData()
를 직접 읽을 수 있다. NVIDIA Ray Tracing Documentation
이게 우리의 conceptual API랑 거의 똑같아:
hit = scene.trace_closest(ray)

if hit.instance == SILICONE:
    ...
elif hit.instance == OBJECT:
    ...
그래서 장기적으로는 current closesthit shader가 그냥 output buffers를 채우는 방식보다 Hit Object 기반 query kernel이 더 자연스러울 수 있어.
다만 이건 P0 변경 후에 해도 됨.
10. SER도 우리 경우 실제 후보임
OptiX 9에는 Shader Execution Reordering이 있고, Ada Lovelace부터 hardware support가 있다. NVIDIA Ray Tracing Documentation
이건 특히 ray가 서로 다른 곳을 hit하면서:
ray 1 → silicone → Fresnel
ray 2 → object → absorb
ray 3 → carrier → absorb
ray 4 → silicone → TIR
처럼 branch divergence가 커질 때 GPU threads를 비슷한 작업끼리 다시 묶어준다.
OptiX 가이드에서는 기존
optixTrace()
를
optixTraverse()
optixReorder()
optixInvoke()
패턴으로 바꾸는 것부터 SER 실험을 시작하라고 권장한다. NVIDIA Ray Tracing Documentation
NVIDIA는 실제 ray-tracing workload에서 최대 2× 수준의 향상을 언급하지만, 효과는 workload 의존적이다. NVIDIA Ray Tracing Documentation
하지만 지금 당장 넣지는 말자
먼저:
- IAS
- refit
- sync 제거
가 훨씬 큰 구조적 승리.
그 다음 Nsight에서 divergence가 병목이면 SER.
11. Future collision에서 motion도 재미있음
OptiX 9.1은 vertex motion과 transform motion을 scene에 직접 넣을 수 있고, 각 ray가 rayTime을 갖고 특정 시간의 interpolated geometry와 intersection할 수 있다. NVIDIA Ray Tracing Documentation
즉:
\[
\Gamma(t)
\]를 OptiX에 표현하고
\[
\text{trace}(r,t)
\]가 가능함.
이건 나중에 trajectory collision에서 꽤 쓸 만해.
다만 이것 자체가 continuous collision detection을 보장하는 건 아님.
“time \(t\)에서 collision인가?”

를 빠르게 물을 수 있게 해주는 기능이지,
\[
\exists t\in[t_0,t_1]
\]를 자동으로 풀어주는 CCD solver는 아니야.
그래도 trajectory sampling이나 narrow-phase accelerator로 상당히 유용함.
12. Custom primitives도 collision 쪽에 크게 쓸 수 있음
OptiX는 triangle뿐 아니라 custom primitive를 지원한다.
사용자가 AABB를 주고, 실제 shape-ray intersection을 CUDA intersection program으로 정의할 수 있다. NVIDIA Ray Tracing Documentation
그래서 향후:
- capsule
- swept sphere
- swept link
- trajectory tube
- 특수 convex approximation
등을 custom primitive로 만들 수 있음.
그리고 OptiX 9에는 built-in sphere primitive도 이미 있다. NVIDIA Ray Tracing Documentation
즉 나중에 RT collision 연구까지 생각하면 OptiX backend를 유지하는 선택은 꽤 잘 맞아.
13. RTX Mega Geometry / CLAS는?
재미있긴 해.
OptiX 9부터 Cluster Acceleration Structures가 들어와서 dynamic high-density geometry rebuild를 엄청 빠르게 할 수 있다. NVIDIA는 경우에 따라 acceleration rebuild가 order-of-magnitude 빨라질 수 있다고 설명한다. NVIDIA Developer
특히:
same triangle topology
different vertex positions
를 반복하는 mesh에 cluster template을 재사용할 수 있다.
어... 우리 deformation 문제랑 얼핏 비슷하지 ㅋㅋ.
하지만 현재 LUMO에는 안 쓰는 게 맞아.
CLAS는 수십만~수백만 microtriangle 수준의 dynamic high-density geometry 문제를 겨냥한 기능이라, 현재 fingertip mesh에 넣으면 abstraction complexity가 훨씬 커질 가능성이 높음. NVIDIA도 주요 target을 dynamic high-density/subdivision geometry로 설명한다. NVIDIA Developer
나중에 surface mesh가 엄청 조밀해진다면 다시 보면 됨.
14. 공식 PyOptiX도 이제 꽤 좋아졌음
이것도 이번에 확인하면서 꽤 중요한 변화였어.
NVIDIA가 현재 **공식 otk-pyoptix**를 운영하고 있고, OptiX 9.1 Python bindings를 pip install pyoptix로 설치 가능하게 해놨어. GitHub
우리 pyproject.toml도 이미:
pyoptix
cupy-cuda13x
cuda-python
를 사용한다. 
즉 Python 기반 architecture를 포기할 이유도 딱히 없음.
다만 reproducibility 관점에서는 나중에 production environment 정할 때:
pyoptix==9.1.0
처럼 버전 pin을 고려하는 게 좋아 보여.
OptiX 9.1은 현재 최신 SDK release고 R590+ driver를 요구한다. GitHub
내가 제안하는 새 backend
이 정도가 딱 좋아 보여.
lumo/ray_tracing/
│
├── optix/
│   ├── runtime.py
│   ├── scene.py
│   └── kernels/
│
└── optical_transport/
그리고 public API는 크게 만들 필요 없어.
scene.update_vertices(SILICONE, vertices)
scene.update_transform(OBJECT, T_WO)

hit = scene.trace_closest(
    origins,
    directions,
    mask=OPTICAL_MASK,
)

collision = scene.trace_any(
    origins,
    directions,
    tmax,
    mask=COLLISION_MASK,
)
내부적으로:
static GAS      carrier/object
dynamic GAS     silicone
        \        |        /
              IAS
               |
           RT Cores
이 정도.
OptixScene을 optics-specific class로 만들지 않는 게 핵심이야.
그러면 나중에 collision도 그대로 쓴다.
우선순위
내가 실제 리팩토링한다면 순서는 딱 이래.
Phase 1
Single IAS
+ instance IDs
+ visibility masks
현재 3× traversal 제거.
Phase 2
persistent vertex buffers
+ GAS ALLOW_UPDATE / refit
+ static GAS reuse
매 checkpoint rebuild 제거.
Phase 3
remove unconditional deviceSynchronize()
+ explicit CUDA stream ownership
GPU pipeline 제대로 살리기.
Phase 4
OTK self-intersection avoidance
수동 epsilon 제거.
Phase 5
traceClosest / traceAny
공통 optics/collision API 완성.
그다음에 profile 보고:
Hit Objects / SER
motion geometry
custom collision primitives
로 가는 게 좋아.
꽤 중요한 최종 판단
이번에 찾아보니까 OptiX를 버릴 이유가 오히려 더 없어졌어.
다만 지금까지는:
“Python에서 triangle hit 좀 빨리 찾으려고 OptiX 붙임”

수준으로 사용했다면,
앞으로는:
“deformable robotics geometry를 위한 shared hardware-accelerated ray-query subsystem”

으로 쓰는 게 맞아.
그리고 optical physics는 여전히 우리 Appendix A대로 유지.
\[
\boxed{
\underbrace{\text{OptiX}}_{\text{geometry / acceleration}}
+
\underbrace{\text{LUMO}}_{\text{physical transport semantics}}
}
\]이 경계가 제일 자연스러워 보여.
1. LED는 generalized Lambertian으로 변경
이건 찬성. 지금 cos θ + 80° hard cutoff보다 훨씬 논문에 쓰기 좋고, LED datasheet의 half-power angle도 자연스럽게 들어간다. 현재 LED.emission_half_angle_deg를 half-power semi-angle \(\theta_{1/2}\)로 정의하자. 
Radiant intensity를
\[
I(\theta)=I_0\cos^m\theta,
\qquad
m=-\frac{\ln2}{\ln(\cos\theta_{1/2})}
\]로 두면 된다.
그리고 ray를 equal-weight deterministic quadrature로 쏘려면 normalized solid-angle distribution은
\[
p(\theta,\phi)
=
\frac{m+1}{2\pi}\cos^m\theta,
\qquad
0\le\theta\le\frac{\pi}{2}
\]이고 inverse sampling은
\[
\cos\theta=(1-u)^{1/(m+1)},\qquad
\phi=2\pi v.
\]여기서 \(u,v\)만 지금처럼 Hammersley sequence로 만들면 됨.
Generalized Lambertian model
         ↓
deterministic Hammersley samples
         ↓
equal ray weight P0/N
이렇게 하면 ray count를 바꿔도 같은 continuous emission model을 더 촘촘하게 적분하는 것이 된다. 현재 sampling.py만 국소적으로 바꾸면 되는 방향이야. 
6. Branch cutoff도 ray-count invariant하게 변경
이것도 수정하자.
현재는
\[
w_{\min}=\epsilon P_0
\]인데 앞으로는
\[
\boxed{
w_{\min}
=
\epsilon_b w_{\mathrm{primary}}
=
\epsilon_b\frac{P_0}{N}
}
\]으로 가는 게 논리적으로 맞아.
즉 설정 이름도 가능하면
minimum_branch_weight_fraction
처럼 의미를 명확하게 하는 게 좋아.
중요한 점은 지금 1e-4 값을 그대로 옮기자는 뜻은 아님.
현재 1e-4는 total launched power 기준이어서 256 rays에서는 사실 primary의 2.56% cutoff였어. 새 formulation의 1e-4는 primary의 0.01%라 transport tree가 훨씬 커진다. 
그러니 순서는:
formulation 먼저 수정
→ ε_b를 numerical convergence parameter로 취급
→ branch cutoff / max interactions / segment budget convergence 확인
→ production ε_b 고정
이게 맞다.
7. Virtual envelope가 뭐고, 어떻게 하는 게 논문에 제일 깔끔한가
현재 구조를 단순화하면 ray가 매 step마다 세 물체를 찾고 있어.
1. 실제 deformed silicone
2. 실제 rigid carrier
3. virtual envelope
1, 2는 물리 geometry야.
그런데 3은 물체가 아님.
“이 선을 넘어가면 이제 fingertip optical domain 밖으로 나갔다고 치자”

라는 계산용 경계야.
현재 문제는 이 envelope가 undeformed/reference fingertip 형상을 이용해서 만들어진다는 거야. 
예를 들어 deformation 때문에:
reference boundary
       │
       │      deformed silicone
       │          │
LED ────────────────→ ray
가 되면 ray는 아직 silicone 속에 있는데 reference envelope를 먼저 만날 수 있음.
현재 구현은 그러면 escape로 처리할 수 있다. 이건 물리적으로 이상하지.
내가 추천하는 논문용 방향
reference fingertip-shaped envelope를 없애자.
대신 고정된 observation domain \(\Omega_{\mathrm{obs}}\)를 정의하는 게 훨씬 깨끗해.
이미 production field bounds가:
x = [-16, 16] mm
y = [-31, 4.5] mm
z = [-5.5, 5.5] mm periodic
로 있으니까 이걸 그대로 쓸 수 있어. 
즉:
        fixed observation domain
┌───────────────────────────────┐
│                               │
│       deformed fingertip      │
│            ████               │
│         █████████             │
│                               │
└───────────────────────────────┘
그리고 두 규칙만 지키자.
첫째, 모든 deformed physical geometry가 observation domain 안에 있어야 한다.
\[
\mathcal G_{\mathrm{physical}}
\subset
\Omega_{\mathrm{obs}}.
\]밖으로 나가면 candidate optical failure.
둘째, observation boundary는 air ray에 대해서만 escape boundary다.
silicone ray는 observation boundary를 무시하고 반드시 실제 silicone interface를 통과해야 한다.
이렇게 하면 논문에서도 아주 깔끔해져.
Rays propagate through the physical deformed geometry until they enter air. Airborne rays leaving a fixed observation domain are classified as escaped.

이게 제일 합당해 보여.
그리고 구현도 굳이 OptiX용 envelope GAS가 필요 없을 수 있어.
고정 rectangle이면 ray-box exit distance를 analytical하게 계산할 수 있거든.
OptiX:
  silicone hit
  carrier hit

analytic:
  periodic z
  fixed observation-box exit
나는 이 방향 추천.
8. Periodic z가 무슨 뜻인가
지금 fingertip은 2D cross section을 11 mm 폭으로 extrusion해서 3D로 만든다. 
그런데 ray가:
z = +5.5 mm
를 지나가면 끝에서 사라지는 게 아니라
z = -5.5 mm
에서 다시 나타난다.
즉:
... | cell | cell | cell | cell | ...
        ↑
     우리가 푸는
      11mm cell
로 생각하는 거야.
수학적으로는
\[
(x,y,z_{\max})\sim(x,y,z_{\min}).
\]그래서 실제로는 무한히 반복되는 fingertip strip 하나를 대표 cell로 푸는 것과 비슷해.
왜 그렇게 했냐면
만약 그냥 11 mm에서 잘라버리면 양쪽 end cap에서 빛이 빠져나가고, 그 결과가 우리가 최적화하려는 2D cross-sectional morphology보다 arbitrary extrusion width에 크게 영향을 받을 수 있어.
현재 디자인 변수는 사실상 x-y cross section을 정의하잖아.
그래서 논문이:
cross-sectional contact morphology design

을 중심으로 간다면 periodic z는 꽤 합리적인 modeling assumption이야.
내 추천
유지하자.
다만 논문에서는 11 mm를 실제 physical finger width라고 표현하지 말고:
a periodic 11-mm representative cell in the out-of-plane direction

이라고 명시.
그리고 가능하면 supplementary/validation에서 cell-width sensitivity 정도만 보여주면 더 강해.
작은 코드 문제
현재 periodic 방향 체크에 dimensionless \(d_z\)와 epsilon_mm를 비교하는 부분은 정리하는 게 좋다. 
abs(direction_z) > direction_epsilon
같은 dimensionless tolerance 하나로 분리하면 됨.
이건 큰 physical issue는 아냐.
9. Internal field가 왜 중요하냐
이게 현재 BO objective의 입력이기 때문이야.
Ray tracing을 엄청 잘 만들어도 최종적으로 optimizer가 보는 건 camera image가 아니야.
현재 optimizer는 3D voxel field:
\[
F_{ijk}
\]를 받고 이것을 normalize해서 비교한다. 
그래서 우리가 무엇을 최적화하고 있는지 정확히 말해야 해.
현재 \(F\)의 물리적 의미
직관적으로는:
각 공간 위치를 얼마나 많은 optical energy-carrying ray path가 지나갔는가

야.
예:
low deformation                high deformation

LED -----> ----->              LED -->\
        ----->                           \-->
                                          -->
deformation에 따라 ray path distribution이 바뀌고 그걸 voxelize한다.
즉 우리는:
빛의 spatial transport pattern이 contact에 따라 얼마나 다르게 재배치되는가

를 보고 있는 거야.
왜 논문에서 이 distinction이 중요하냐
이걸 그냥:
optical intensity field

라고 부르면 reviewer 입장에서는:
“그럼 radiative transfer equation을 풀었나?”
“이 값이 camera irradiance인가?”
“각 voxel의 단위가 W/mm²인가?”

를 물을 수 있어.
그런데 아니잖아.
우리는 camera-independent optical transport proxy를 만든 거야.
그래서 오히려 이렇게 주장하는 게 강하다.
We intentionally optimize a camera-independent internal transport representation rather than a specific image formation model.

이러면 morphology와 camera design을 분리할 수 있음.
그리고 나중에 실제 camera signal과 correlation을 보여주면 훨씬 강해져.
morphology
   ↓
internal optical transport proxy
   ↓
camera readout
우리가 최적화하는 건 가운데 단계.
즉 9번은 code bug 이야기가 아니라 논문의 claim boundary를 결정하는 문제야.
10. 패스 👍
11. Objective는 논문 관점에서 바꾸는 게 좋음
현재 D_inter는 location만 다르면 radius/depth가 달라도 비교해. 
나는 논문에는 이보다 matched-condition comparison이 훨씬 깨끗하다고 봐.
Observation을
\[
F(u,r,d)
\]라고 하자.
- \(u\): contact location
- \(r\): object radius
- \(d\): indentation depth
field normalization:
\[
\hat F(u,r,d)
=
\frac{F(u,r,d)}
{\sum_vF_v(u,r,d)}.
\]그리고 TV distance:
\[
D(F_a,F_b)
=
\frac12
\|\hat F_a-\hat F_b\|_1.
\]Contact-location separability
내 추천:
\[
\boxed{
D_{\mathrm{loc}}
=
\min_{r,d}
\;
\min_{u_i\neq u_j}
D
\left(
F(u_i,r,d),
F(u_j,r,d)
\right)
}
\]즉 radius와 indentation은 같게 두고 location만 다르게 한다.
해석:
같은 contact condition에서 위치가 바뀌었을 때, 가장 구분하기 어려운 두 location도 충분히 떨어져 있어야 한다.

아주 명확함.
Radius robustness
\[
\boxed{
D_{\mathrm{rad}}
=
\max_{u,d}
\;
\max_{r_i\neq r_j}
D
\left(
F(u,r_i,d),
F(u,r_j,d)
\right)
}
\]해석:
위치와 indentation이 동일할 때 object radius 변화 때문에 생기는 optical variation은 작았으면 좋겠다.

그럼 최종:
\[
\boxed{
J
=
D_{\mathrm{loc}}
-
\lambda_rD_{\mathrm{rad}}
}
\]가 된다.
이게 현재 formulation의 의도도 살리고 수식도 훨씬 예쁘다.
나는 이 방향으로 코드도 바꾸는 걸 추천.
왜 depth penalty는 안 넣냐
일단 안 넣자.
Depth는 nuisance라기보다 contact progression 자체니까.
즉:
location = 우리가 구분하고 싶은 signal
radius   = nuisance
depth    = trajectory / preload condition
으로 두는 게 paper v1에서는 가장 깔끔하다.
나중에 필요하면 depth invariance를 별도 metric으로 보고.
12. surface u interpolation이 뭐냐
OptiX가 triangle을 hit하면:
triangle 안의 정확히 어디를 맞았는지

를 barycentric coordinate로 알려줘. 현재 kernel도 그 값을 반환하고 있다. 
예를 들어 triangle vertex가:
A: u=0.2
B: u=0.3
C: u=0.3
이고 hit point가:
\[
(\lambda_A,\lambda_B,\lambda_C)
\]라면 material coordinate는 그냥:
\[
\boxed{
u
=
\lambda_Au_A+
\lambda_Bu_B+
\lambda_Cu_C
}
\]면 끝이야.
그런데 현재는 triangle의 첫 edge를 XY plane에 project해서 hit 위치를 추정한다. 
extruded triangle은 이런 식일 수 있잖아.
A (x,y,z0)
|
| z-direction
|
B (x,y,z1)
 \
  \
   C (x2,y2,z1)
A와 B는 XY가 동일함.
그러면 AB를 XY에 projection하면 길이 0이 된다.
그래서 현재 방식은 좀 이상한 approximation.
방향
barycentric interpolation으로 고치면 끝.
그리고 이건 internal BO objective에는 영향 없고:
outgoing_surface_field(u,z)
같은 surface visualization/diagnostic에 주로 영향 준다.
논문에 surface heatmap 넣을 거면 고치는 게 맞아.
13. Object contact optics는 무슨 뜻이냐
지금 sphere는 mechanics에는 존재해.
sphere presses silicone
        ↓
silicone deforms
        ↓
ray tracing
그런데 ray tracing에서 그 sphere 자체는 사실상 안 보인다.
즉 contact patch에서:
silicone
████████
    ● sphere
가 실제로 닿아 있어도, optical simulation은 주로:
silicone → air
interface처럼 취급한다.
현재 production에서는 carrier만 explicit absorber로 넣고 있다. 
실제 물리에서는?
object가 opaque하면:
ray → contact object
        ↓
       absorbed / reflected
가 돼야 하지.
그래서 object contact 자체도 optical signal을 크게 만들 수 있다.
논문 방향은 두 가지가 있음
A. deformation-only study
object optics는 일부러 제거하고 silicone deformation이 optical transport에 미치는 효과만 연구

장점은 morphology mechanism이 깨끗함.
하지만 claim은:
“actual tactile optical sensor response”

가 아니라
“deformation-mediated optical transport proxy”

로 제한해야 해.
B. 실제 sensor model
contact patch의 sphere/object를 absorber 또는 dielectric으로 넣는다.
나는 최종 논문에는 B가 더 좋다고 봐.
다만 복잡하게 object material을 다 모델링하진 말고:
standardized opaque/absorbing indenter

로 두자.
즉 mechanics에서 sphere contact patch가 잡히면 그 부분을:
OBJECT_CONTACT_INTERFACE
로 tagging하고
\[
w_{\text{object}} \rightarrow 0
\]absorber boundary를 쓰는 것.
코드에 이미 ObjectBoundaryOptics("absorber") concept도 있으니까 formulation 자체는 준비돼 있다. 
그러면 실제 실험도 검은/불투명 indenter를 쓰면 simulation assumption이 매우 명확해져.
이걸 추천.
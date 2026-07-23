# LIT Hand — Kratos 2D Hyperelastic Contact Validation

> **문서 목적**  
> LIT Hand의 compliant fingertip pad 해석에 사용할 Kratos 2D FEM 구성의 검증 과정, 실패 사례, 수치 결과, 최종 채택 기준을 보존한다. 이후 실제 pad 모델링, 논문 작성, 코드 재구성 시 이 문서를 기준 기록으로 사용한다.

- 기록일: 2026-07-22
- 대상: 2D plane-strain, nearly incompressible hyperelastic solid, frictionless contact
- 최종 프로젝트 상태: **Mixed solid ADOPT / 실제 zero-clearance fingertip nonlinear solve BLOCKED**

---

## 1. 최종 결론

LIT Hand의 2D baseline 해석에는 다음 구성을 채택한다.

```text
Solid element:
  TotalLagrangianMixedVolumetricStrainElement2D3N

Constitutive law:
  HyperElasticPlaneStrain2DLaw

Poisson's ratio:
  ν = 0.49

Unknowns:
  DISPLACEMENT_X
  DISPLACEMENT_Y
  VOLUMETRIC_STRAIN

Contact:
  ALMContactFrictionless
  AugmentedLagrangianMethodFrictionlessMortarContactCondition

Recommended contact roles:
  MASTER = rigid/kinematically constrained rounded indenter
  SLAVE  = deformable pad surface
```

현재 검증 결과는 다음을 지지한다.

- Mixed finite-strain solid formulation은 **ADOPT**한다.
- `ν = 0.49`는 채택한 mixed formulation에서 안정적으로 사용할 수 있다.
- `ν = 0.499`는 검증 범위에서 불안정했으므로 사용하지 않는다.
- 2D frictionless ALM mortar contact는 localized benchmark에 한해 **conditional ADOPT**한다.
- 접촉 예상 영역은 충분히 세분화하고, medium/fine mesh convergence를 확인해야 한다.
- Coarse mesh의 pointwise contact pressure는 신뢰하지 않는다.
- Phase 4I-D 격리 결과, right-side zero-clearance internal pair 하나만으로도
  첫 step failure가 재현된다. Mixed solid formulation 자체는 계속 ADOPT한다.

중요한 제한은 다음과 같다.

- 이 결과는 2D plane-strain 모델의 수치 검증이다.
- 최종 설계는 3D 모델 또는 실험으로 확인해야 한다.
- 실제 LIT pad의 구조 성능이 검증된 것은 아니다. 지금까지 검증한 것은 해석 기반이다.

---

## 2. 검증 질문

이번 검증은 다음 질문에 답하기 위해 수행했다.

1. Kratos에서 2D hyperelastic finite-strain 해석이 정상적으로 작동하는가?
2. 실리콘과 같은 nearly incompressible material을 `ν = 0.49`로 안정적으로 계산할 수 있는가?
3. Mixed solid와 frictionless ALM contact를 결합할 수 있는가?
4. Localized indentation에서 force, displacement, volumetric strain, contact field가 유한하고 수렴하는가?
5. 이 구성을 실제 LIT fingertip의 deformation-transfer study에 사용할 수 있는가?

---

## 3. Phase 요약

| Phase | 목적 | 원래 판정 | 현재 해석 |
|---|---|---:|---|
| Phase 1 | Kratos 기본 실행, 2D hyperelastic 및 frictionless ALM contact의 개별 baseline | PASS | 기본 기능 확인 |
| Phase 2A | 일반 Q4 및 Q1P0 mixed Q4 후보 평가 | Mixed 후보 FAIL | 일반 요소의 한계와 대체 formulation 필요성 확인 |
| Phase 2B | Mixed volumetric-strain T3 검증 | PASS | 기본 solid formulation으로 ADOPT |
| Phase 3 | Mixed solid + localized ALM contact 결합 | FAIL | Fine mesh에서 active-set cycling 발견 |
| Phase 3R | Master/slave 및 load-step recovery, mesh convergence 재검증 | Formal FAIL | 모든 mesh 수렴; 원래 all-mesh pressure gate만 실패 |
| Phase 4M | 실제 Shapely geometry의 Gmsh T3/Kratos initialization | PASS | 3개 내부 contact runtime contract 확인 |
| Phase 4I | 중앙 rigid-indenter nonlinear solve | FAIL | Trial step 1에서 내부 zero-clearance contact rank deficiency |
| Phase 4I-D | 내부 contact A–E 격리 및 continuous-U recovery | FAIL | Right-only도 실패; continuous U도 35회에서 실패 |
| Phase 4I-E | 우측 mirror/orientation 및 endpoint assembly audit | FAIL | ordering 가설 기각; search 이후 extra pair 확인 |
| Phase 4I-F | Search-pair 인과성 및 contact–bond crosspoint audit | FAIL | invalid-pair ACTIVE는 필요조건 아님; active crosspoint LM row 결손 확인 |
| Phase 4I-G | 제한된 multiplier-space treatment 검증 | FAIL/BLOCKED | 공식 경로 없음; topology-only endpoint LM 제외는 과도함 |
| Phase 4J | No-void external-contact-only baseline | PASS | medium/fine 1.5 mm 완료; 반력 차이 0.319% |
| Project decision | LIT Hand application 적합성 평가 | — | **Mixed solid 및 external baseline 유지 / internal contact 차단** |

`Phase 3R Formal FAIL` 기록은 지우지 않는다. 다만 이는 coarse mesh의 pressure roughness까지 모든 mesh에서 `< 0.5`를 요구한 원래 gate에 따른 판정이다. Solver failure 또는 application blocker를 의미하지 않는다.

---

## 4. Phase 1 — 기본 기능 확인

### 목적

복잡한 fingertip 형상을 만들기 전에 Kratos build와 핵심 해석 기능을 분리해 확인했다.

### 확인된 항목

- Kratos 실행 정상
- 2D Total/Updated Lagrangian hyperelastic baseline 실행 정상
- 2D frictionless ALM contact baseline 실행 정상
- 변위, 반력 및 기본 finite-strain field 추출 가능

### 판정

**PASS**

Phase 1은 contact와 hyperelastic solid를 각각 실행할 수 있음을 보인 단계다. 두 기능의 결합은 Phase 3에서 별도로 검증했다.

---

## 5. Phase 2A — 일반 요소 및 Q1P0 후보

### 시험 결과

| 구성 | 결과 | 해석 |
|---|---|---|
| 일반 Q4, `ν = 0.45` | 실행 가능 | baseline으로 사용 가능 |
| 일반 Q4, `ν = 0.49` | 실행되지만 locking이 큼 | nearly incompressible production 후보로 부적합 |
| Q1P0 mixed Q4 | 실제 계산 실패 | 기본 후보에서 제외 |

### 의미

`ν → 0.5`에서는 일반 displacement formulation이 체적 변형을 과도하게 억제하여 volumetric locking을 일으킬 수 있다. 따라서 `ν = 0.49`를 목표로 할 때는 mixed formulation이 필요하다고 판단했다.

### 판정

Phase 2A의 mixed 후보는 **FAIL**. 다음 fallback으로 mixed volumetric-strain T3를 시험했다.

---

## 6. Phase 2B — Mixed Volumetric-Strain T3

### 채택 구성

```text
Element:
  TotalLagrangianMixedVolumetricStrainElement2D3N

Law:
  HyperElasticPlaneStrain2DLaw

Primary unknowns:
  DISPLACEMENT_X
  DISPLACEMENT_Y
  VOLUMETRIC_STRAIN
```

Structured quadrilateral cell을 두 개의 T3 요소로 분할했다. 공식 Kratos nonlinear patch-test 방식과 같은 topology를 사용했다.

### Runtime contract

| 항목 | 확인 결과 |
|---|---|
| Constitutive law | `HyperElasticPlaneStrain2DLaw` |
| Assembled DOFs | X/Y displacement, `VOLUMETRIC_STRAIN` |
| Additional check DOF | `DISPLACEMENT_Z`; 2D 방정식에는 조립되지 않지만 `SolidElementCheck`가 요구 |
| Required variables | `DISPLACEMENT`, `VOLUMETRIC_STRAIN` |
| Reactions | `REACTION`, `REACTION_STRAIN` |
| Required properties | `YOUNG_MODULUS`, `POISSON_RATIO`, `CONSTITUTIVE_LAW` |
| Benchmark thickness | `THICKNESS = 1 mm` |
| Initial volumetric strain | 모든 node에서 `0.0` |
| Volumetric-strain BC | 구속하지 않고 전 node에서 풀이 |

초기 `VOLUMETRIC_STRAIN`은 `1.0`이 아니라 `0.0`이다. 결과상 이 변수는 reference state로부터의 체적 변형, 즉 이 benchmark에서는 사실상 `J - 1`에 대응했다.

### Boundary conditions

- Bottom 전체: `DISPLACEMENT_Y = 0`
- Bottom-left 한 node: `DISPLACEMENT_X = 0`
- Top: prescribed `DISPLACEMENT_Y`; X 방향 자유
- Lateral boundaries: X 방향 자유
- `VOLUMETRIC_STRAIN`: 전 node에서 자유

일반 TL 대조군도 동일한 triangular mesh와 boundary condition을 사용했다.

### Runtime patch result

| 항목 | 결과 |
|---|---:|
| `Initialize()` | PASS |
| Strategy `Check()` | 0 |
| 첫 1% 압축 | PASS |
| Newton iterations | 3 |
| Reaction | 0.133525 N |
| `det(F)` | 0.99960227 |
| Area ratio | 0.99960227 |
| `VOLUMETRIC_STRAIN` | -3.97729e-4 |
| Finite-value validation | PASS |

### `ν = 0.49` mesh results

| Mesh | Elements | Final reaction | `det(F)` mean | Area ratio | Newton 범위 | Solve time |
|---|---:|---:|---:|---:|---:|---:|
| Coarse | 32 | 7.0576146 N | 0.98055444 | 0.98055444 | 3–4 | 0.020 s |
| Medium | 128 | 7.0576146 N | 0.98055444 | 0.98055444 | 3–4 | 0.086 s |
| Fine | 512 | 7.0576146 N | 0.98055444 | 0.98055444 | 3–5 | 0.540 s |

- Medium/fine reaction 차이: `5.01e-13`
- Coarse/fine reaction 차이: `1.26e-15`
- Mean `VOLUMETRIC_STRAIN`: 약 `-0.01944556`
- 최대 normalized force-curve second difference: `0.002733`
- 30개 load step 모두 finite, smooth, monotonic, 정상 수렴

세 mesh의 반력이 사실상 동일한 이유는 이 문제가 거의 affine deformation이고 P1 T3가 해당 변형을 정확히 표현하는 patch-test 성격이 강하기 때문이다. 이를 모든 비균일 변형 문제에서의 mesh independence로 일반화하면 안 된다.

### Poisson's-ratio sweep

| `ν` | Coarse | Medium | Fine | 판정 |
|---:|---:|---:|---:|---|
| 0.45 | 6.126894 N | 6.126894 N | 6.126894 N | PASS |
| 0.49 | 7.057615 N | 7.057615 N | 7.057615 N | PASS / ADOPT |
| 0.499 | 7.349610 N | FAIL | 7.347945 N | 불안정 / NO ADOPTION |

`ν = 0.499` medium mesh는 30% step에서 `VOLUMETRIC_STRAIN`, displacement, reaction이 non-finite가 되었다. Fine mesh는 통과했지만 Newton iteration이 최대 11회였고 `det(F)` 범위가 `0.8553–1.1056`으로 넓어졌다. 따라서 `ν = 0.499`는 신뢰 가능한 사용 범위로 인정하지 않는다.

### 일반 TL 대조군

- Coarse/medium: 통과
- Fine: 8% 압축에서 displacement와 reaction이 non-finite
- `SolveSolutionStep()`은 `True`를 반환했지만 별도 finite-value validation에서 실패

이는 solver return value만 확인하지 말고 모든 주요 field에 대해 별도의 finite-value validation을 해야 함을 보여준다.

### `det(F)` 계측 방법

현재 build에서 이 요소의 integration-point `DEFORMATION_GRADIENT`가 빈 배열로 반환되었다. Kratos API는 수정하지 않았다.

대신 각 affine T3의 reference/current nodal edge matrix로 `F`와 `det(F)`를 계산하고, 별도의 triangle-area 합으로 계산한 전체 area ratio와 비교했다. 성공한 mixed case에서 area-weighted `det(F)`와 전체 area ratio의 최대 차이는 `0.0`이었다.

두 값은 P1 triangle 기하상 직접 연결되므로, 이 일치는 계측 구현의 consistency check이지 독립적인 물리 검증은 아니다.

### 판정

**PASS / ADOPT**

---

## 7. Phase 3 — Localized Hyperelastic Contact

### 목적

Phase 1의 frictionless ALM contact와 Phase 2B의 mixed finite-strain solid를 localized rounded-indenter benchmark에서 결합했다.

### 공통 구성

- 2D plane-strain hyperelastic block
- Structured T3 mesh
- `TotalLagrangianMixedVolumetricStrainElement2D3N`
- `HyperElasticPlaneStrain2DLaw`
- `ν = 0.49`
- Rounded indenter
- Frictionless ALM mortar contact
- 48개 공통 displacement-controlled load step
- 기본 Kratos ALM penalty/search 값
- `contact_residual_criterion`
- Standard Newton strategy

Rounded indenter에는 완전히 kinematically constrained된 얇은 T3 carrier를 부착했다. 독립 line condition만 사용할 경우 master `NODAL_H`가 `DBL_MAX`로 남았기 때문이다. Carrier의 일반 TL 요소는 characteristic length 제공용이며 deformable block formulation에는 포함하지 않는다.

### 최초 결과

| Mesh | 판정 | 도달 indentation | Reaction | `det(F)` min | 최대 Newton | Solve time |
|---|---|---:|---:|---:|---:|---:|
| Coarse | PASS | 0.500 mm | 0.522363 N | 0.735147 | 3 | 0.132 s |
| Medium | PASS | 0.500 mm | 0.476738 N | 0.709193 | 12 | 0.904 s |
| Fine | FAIL | 0.164167 mm | 0.119033 N | 0.858335 | 35 | 7.419 s |

Fine 값은 마지막 수렴 step 22 기준이다. Step 23에서 active set이 수렴하지 않았다. 실패 iterate도 finite였고 최소 `det(F) = 0.789078 > 0`이었다.

### 관찰

- 모든 mesh에서 load-bearing contact는 step 10, motion `0.129167 mm`에서 시작
- Initial gap `0.12 mm`를 올바르게 bracket
- 기록된 모든 step에서 weighted gap, pressure, displacement, reaction, `VOLUMETRIC_STRAIN` finite
- 실패 iterate를 포함하여 negative `det(F)` 없음
- Coarse/medium force curve는 smooth하고 monotonic
- Fine도 마지막 수렴 step까지 smooth하고 monotonic
- 마지막 공통 수렴 step에서 medium/fine reaction 차이: `4.91%`
- 수렴 상태에서 volumetric checkerboard 및 pressure oscillation 검사 통과
- Fine 실패 iterate에서 pressure roughness가 `0.922 > 0.5`로 증가

### 실패 분류

**Phase 3 전체 판정: FAIL**

실패 원인은 mixed solid formulation의 발산, non-finite material response 또는 element inversion이 아니었다. Fine mesh에서 발생한 **ALM active-set cycling**이었다.

---

## 8. Phase 3R — Contact Recovery

### Runtime master/slave 검증

최초 48-step Fine 설정의 실제 runtime 역할은 다음과 같았다.

| Surface | 실제 역할 | Nodes / Conditions | 좌표 범위 | `NODAL_H` min / mean / max |
|---|---|---:|---|---:|
| Block top | MASTER | 33 / 32 | `x = 0–10`, `y = 5` | 0.15625 / 0.15625 / 0.15625 |
| Rounded indenter | SLAVE | 33 / 32 | `x = 3–7`, `y = 5.12–7.12` | 0.125061 / 0.172044 / 0.5 |

```text
INITIAL_PENALTY = 4.95519918777264
SCALE_FACTOR    = 4.95519918777264
```

LM variable과 DOF는 양쪽 surface node에 할당되지만, 실제 multiplier ownership은 runtime `SLAVE` flag가 결정한다.

Recovery에서는 역할을 다음과 같이 변경했다.

```text
MASTER = rounded indenter
SLAVE  = deformable block top
```

96-step 최종 nonzero LM node는 block top에만 존재했다.

- Coarse: 3개
- Medium: 3개
- Fine: 7개

### Fine step-23 cycling 진단

최초 48-step Fine failure는 단순한 two-state chattering이 아니었다.

- Newton 35회 내 수렴 실패
- Iteration 5부터 시작하는 period-11 multi-state cycle
- 같은 active-set sequence가 iteration 16과 27에서 반복

예시:

| Iteration | ACTIVE slave nodes | Residual norm | Indenter reaction |
|---:|---|---:|---:|
| 12 | 1105, 1107 | 77.7159 | -4.55231 N |
| 13 | 1104–1108 중 5개 | 0.245377 | -0.228933 N |
| 14 | 1105, 1107 | 110.281 | -6.28464 N |

변경 node ID, `WEIGHTED_GAP`, LM pressure, augmented pressure, `NODAL_AREA`, `NODAL_H`, reaction, residual을 Newton iteration별 JSON에 기록했다.

### Recovery 변경 사항

- 모든 mesh에 동일하게 96개 load step 적용
- 최종 motion과 target indentation 유지
- Material, element, geometry, ALM 기본값 및 convergence tolerance 유지
- Master/slave 방향 수정
- Fine mesh만을 위한 별도 penalty tuning은 수행하지 않음
- Maximum Newton iteration을 늘려 cycling을 숨기지 않음

### 96-step 결과

| Mesh | Solve | Final reaction | 전체 최소 `det(F)` | Pressure roughness | 최대 Newton |
|---|---|---:|---:|---:|---:|
| Coarse | PASS | 0.396967 N | 0.692145 | 0.743654 | 3 |
| Medium | PASS | 0.466330 N | 0.767874 | 0.241555 | 3 |
| Fine | PASS | 0.500125 N | 0.854401 | 0.092973 | 3 |

추가 결과:

- 모든 mesh가 `0.5 mm` indentation 도달
- 모든 field finite
- Negative `det(F)` 없음
- 모든 force curve smooth/monotonic
- 모든 step에서 active set 수렴
- Fine cycling 해소
- Medium/fine final reaction 차이: `6.757%` — `< 10%` 기준 통과
- 동일 방향 medium 48/96-step 차이: `0.00497%` — `< 1%` 기준 통과
- 기존 반대 방향 baseline과의 차이: `2.183%`; 순수 step-size 비교로 해석하지 않음

Volumetric checkerboard ratio:

| Mesh | Ratio | 판정 |
|---|---:|---|
| Coarse | 0.00454 | PASS |
| Medium | 0.00206 | PASS |
| Fine | 6.08e-6 | PASS |

Carrier 검증:

- `F = I`
- Strain = 0
- Internal RHS = 0
- 계산 energy = 0
- Inversion 없음
- Block 통계에서 제외
- Mesh별 임의 tuning 없이 동일 생성 규칙 사용

### Formal 판정과 프로젝트 판정

원래 Phase 3R acceptance는 **모든 mesh에서 pressure roughness `< 0.5`**를 요구했다. Coarse mesh의 값이 `0.743654`였으므로 이 기준을 문자 그대로 적용한 판정은 다음과 같다.

```text
Original Phase 3R gate: FORMAL FAIL
```

그러나 이는 solver가 수렴하지 않았다는 뜻이 아니다. Coarse mesh에서는 nonzero pressure node가 3개뿐이므로 매끄러운 contact-pressure profile을 표현하기에 공간 해상도가 부족하다. Pressure roughness는 refinement에 따라 다음처럼 빠르게 감소했다.

```text
0.743654  →  0.241555  →  0.092973
 coarse       medium        fine
```

따라서 LIT Hand application에 맞춘 최종 판단은 다음과 같다.

```text
Mixed solid:                         ADOPT
ν = 0.49:                            ADOPT
2D frictionless ALM mortar contact:  CONDITIONAL ADOPT
Coarse pointwise pressure field:     NO ADOPTION
Actual fingertip modeling:           PROCEED
```

이 결론은 localized block benchmark에 대한 것이다. 이후 Phase 4I에서 실제
zero-clearance 내부 contact topology가 별도의 rank deficiency를 일으켰으므로,
현재 실제 geometry에 대해서는 contact formulation/topology가 blocker다.

---

## 9. 실제 LIT Hand 모델의 사용 규칙

### Mesh

1. 접촉 예상 영역은 최소 benchmark의 medium 수준으로 local refinement한다.
2. 최종 결과는 medium/fine mesh pair로 convergence를 확인한다.
3. Coarse mesh의 nodal peak pressure 또는 pressure smoothness는 설계 판단에 사용하지 않는다.

### 우선 평가 출력

LIT Hand의 핵심 관심 출력은 다음과 같다.

1. Side displacement/bulging profile
2. Reaction force
3. Contact width 및 contact location
4. Deformation-transfer efficiency
5. Maximum strain 및 최소 `det(F)`

Pointwise peak contact pressure는 보조 지표로 취급한다.

### Production acceptance

각 실제 geometry에 대해 다음을 확인한다.

- Medium/fine final reaction 차이 `< 10%`
- 실제 사용하는 medium/fine mesh의 pressure roughness `< 0.5`
- Force–indentation curve가 smooth하고 monotonic
- Active-set cycling 없음
- 모든 주요 field finite
- 모든 solid element에서 `det(F) > 0`
- Volumetric-strain checkerboard 또는 mesh-scale oscillation 없음
- Carrier가 있는 경우 deformation, energy 및 internal force가 numerical zero

### 자동 검증 권장 항목

Kratos strategy의 성공 return만으로 case를 PASS 처리하지 않는다. 각 load step 이후 최소한 다음을 별도로 검사한다.

```text
DISPLACEMENT              finite
REACTION                  finite
VOLUMETRIC_STRAIN         finite
WEIGHTED_GAP              finite
contact pressure          finite
elementwise det(F)        positive and finite
force curve               monotonic/smooth
active-set state          converged
```

---

## 10. 다음 연구 Phase

Solid와 isolated contact baseline 검증은 완료되었지만, 실제 geometry의 첫
nonlinear Trial은 통과하지 못했다. 따라서 위치 sweep 전에 zero-clearance
내부 contact의 조립 문제를 해결해야 한다.

| Phase | 확인할 것 | 핵심 산출물 |
|---|---|---|
| Phase 4 | 실제 pad 외곽, stem/support, solid baseline | 현재 nonlinear contact blocker 해결 필요 |
| Phase 5 | Contact-location 및 indentation/load sweep | 위치·하중별 side-deformation map |
| Phase 6 | Solid, T-core, void/slit, fin-ray류 내부구조 비교 | 구조별 location separability와 load sensitivity |
| Phase 7 | 형상 치수와 material parameter sweep | 최종 geometry 후보와 trade-off |
| Phase 8 | 3D 및 실험 검증 | 2D prediction의 현실 적합성 |

다음 단계의 핵심 출력은 개념적으로 다음과 같다.

\[
\mathbf{y}(x_c,\delta)
=
\begin{bmatrix}
\text{side displacement profile} \\
\text{reaction force} \\
\text{contact width}
\end{bmatrix}
\]

여기서 `x_c`는 contact location, `δ`는 indentation이다.

검증할 연구 질문:

1. 같은 위치를 더 세게 누르면 측면 변형이 monotonic하게 증가하는가?
2. 서로 다른 위치에서 측면 변형 pattern이 충분히 구분되는가?
3. Contact location과 load effect를 서로 식별할 수 있는가?
4. 변형이 camera-visible region까지 충분히 전달되는가?
5. 내부 구조에 과도한 strain, inversion 또는 buckling이 발생하지 않는가?

---

## 11. 재사용 시 주의할 핵심 사실

- `Poisson's ratio`는 `4.9`가 아니라 **`ν = 0.49`**이다.
- `VOLUMETRIC_STRAIN`의 reference initial value는 **`0.0`**이다.
- `DISPLACEMENT_Z` DOF는 2D equation에 조립되지 않지만 runtime element check를 위해 필요하다.
- `ν = 0.499` 결과가 일부 mesh에서 통과했더라도 formulation의 신뢰 범위로 인정하지 않는다.
- Fine active-set cycling은 96-step과 올바른 master/slave 방향에서 해소되었다.
- Phase 3R의 `Formal FAIL`은 coarse pressure roughness gate 때문이며, 현재 실제 application의 blocker가 아니다.
- Contact 역할은 명칭이 아니라 runtime `MASTER`/`SLAVE` flag로 검증한다.
- `det(F)`와 area ratio의 일치는 T3 기하상 연결되어 있으므로 독립 물리 검증으로 과대 해석하지 않는다.
- 2D plane-strain 결과를 최종 현실 예측으로 간주하지 않는다.

---

## 12. 한 줄 상태 요약

> **Mixed T3 hyperelastic solid (`ν = 0.49`)는 유지한다. 그러나 default zero-clearance 내부 접촉을 포함한 실제 fingertip ALM solve는 Phase 4I Trial에서 실패했으므로 contact-location sweep으로 진행하지 않는다.**

---

## 13. Phase 4M — 실제 geometry mesh 및 initialization 상태

`FingertipModel`의 Shapely pad/link polygon과 boundary/contact metadata를
직접 사용하는 Gmsh 4.15.2 adapter를 구현했다. 별도의 ellipse 또는 치수
geometry는 mesher에 복제하지 않았다.

Default zero-clearance geometry의 실행 결과:

| Mesh | Nodes | T3 elements | Minimum angle | Pad area error | Link area error |
|---|---:|---:|---:|---:|---:|
| Medium | 8,391 | 16,164 | 35.906° | 3.18e-15 | 3.69e-16 |
| Fine | 18,961 | 36,964 | 38.828° | 9.40e-15 | 3.51e-15 |

세 zero-clearance contact pair는 pad/stem 양쪽에서 좌표가 일치하지만 node
ID는 분리되어 있다. 별도 U-clearance test에서는 left/right `2.5 mm`, bottom
`3.0 mm` mesh gap을 재현했다.

Kratos 10.3 initialization smoke test에서 확인한 구성:

```text
Pad element:     TotalLagrangianMixedVolumetricStrainElement2D3N
Carrier element: TotalLagrangianElement2D3N
Law:             HyperElasticPlaneStrain2DLaw
Contact:         ALMContactFrictionless
Pad cutouts:     runtime SLAVE
Stem surfaces:   runtime MASTER
```

`StructuralMechanicsAnalysis.Initialize()`, strategy `Check()`, contact flag,
surface normal 및 positive finite `NODAL_H` 검증은 통과했다. Phase 4M은
초기화에서 멈추며 외부 rounded indenter, loading step 또는 nonlinear
indentation solve는 포함하지 않는다.

---

## 14. Phase 4I — 실제 geometry 중앙 indentation

### 구현 범위

Phase 4M 경로를 유지하면서 FEM layer에 원형 rigid indenter fixture와 네 번째
ALM contact pair를 추가했다.

```text
Solid:      TotalLagrangianMixedVolumetricStrainElement2D3N
Law:        HyperElasticPlaneStrain2DLaw
E / nu:     1.0 MPa placeholder / 0.49
Thickness:  1.0 mm
Indenter:   radius 4.0 mm, initial gap 0.0 mm
Loading:    central rigid translation, displacement controlled
Contact:    ALMContactFrictionless, Kratos 10.3 defaults
```

Indenter는 `FingertipModel`의 일부가 아니다. 실제 `pad_outer_arc`와 symmetry
axis의 교점에서 crown point, tangent, outward normal을 계산한 후 FEM fixture를
배치한다. Pad/link geometry의 source of truth는 계속 Shapely model이다.

### Trial 실행

```bash
OMP_NUM_THREADS=1 /home/dk/miniconda3/envs/lit/bin/python -B \
  -m analysis.phase4_indentation_baseline \
  --mesh-level medium --indentation-mm 0.25 --steps 48 --trial
```

실제 실행 결과:

| 항목 | 결과 |
|---|---|
| Preflight initialization | PASS |
| Kratos version | 10.3.0 |
| Fingertip mesh | 8,391 nodes / 16,164 T3 |
| Indenter mesh | 528 nodes / 980 T3 / 24 contact edges |
| Indexed four-pair runtime contract | PASS |
| Nonlinear solve | FAIL, step 1 before a converged history point |
| Process result | Controlled failure (`returncode = 1`) |

Runtime role은 좌표 추론이 아니라 Kratos가 생성한 `ContactSubN`과
`ComputingContactSubN`에서 확인했다.

```text
ContactSub0: PadOuterArc     SLAVE / IndenterContactArc MASTER
ContactSub1: PadCutoutLeft   SLAVE / StemLeft           MASTER
ContactSub2: PadCutoutRight  SLAVE / StemRight          MASTER
ContactSub3: PadCutoutBottom SLAVE / StemBottom         MASTER
```

첫 step 조립에서 Skyline LU가 `Error zero sum`을 기록했다. 이후 builder가
RHS를 0으로 두는 warning을 반복했고 Newton 35회 제한에서 종료됐다.

```text
LUSkylineFactorization::factorize: Error zero sum
ATTENTION: Max iterations exceeded
```

실패 step의 prescribed travel은 `0.00520833 mm`였다. External generated
condition 12개와 각 internal group 39개가 `ACTIVE`였지만 수렴하지 않았고,
reaction component 17,838개 중 13,548개가 non-finite였다. 따라서 이 failed
iterate로 reaction, `det(F)`, penetration 또는 strain 값을 보고하지 않는다.

이는 external pair mapping 실패가 아니다. 별도의 진단용 U-clearance
(`void_width = void_height = 0.2 mm`) 소형 solve는 같은 external fixture,
mixed solid, ALM 설정으로 1 step을 수렴했고 external active condition과
positive `det(F)`를 확인했다. 이 진단은 Phase 4I baseline이 아니며 acceptance
결과로 사용하지 않는다. 이 원래 Trial이 확정한 범위는 default geometry의
three-pair 구성에서 rank deficiency가 재현된다는 사실까지였다. 어느
surface가 충분조건인지와 lower U-corner process 중복의 역할은 아래
Phase 4I-D에서 별도로 격리했다. 내부 shared/fixed endpoint의 scalar LM
ownership은 추가 고정 없이 Kratos process 기본 처리를 유지한다.

### 판정

```text
Phase 4I:                         FAIL
Mixed solid formulation:         ADOPT 유지
Requested 1.5 mm medium/fine:     실행하지 않음 (Trial gate)
Reaction/profile/strain metrics:  사용 가능한 converged Trial step 없음
Next contact-location sweep:      NO
```

현재 결과는 `output/phase4_indentation/trial_medium_0p25/result.json`,
`preflight.json`, `solver.log`에 보존한다. 다음 작업은 parameter sweep이 아니라
default conforming internal contact의 boundary/group/LM formulation을 별도로
해결하는 것이다. Acceptance threshold 완화나 mesh별 penalty tuning으로 이
실패를 PASS 처리하지 않는다.

---

## 15. Phase 4I-D — internal contact topology 격리

### 고정 조건과 실행

Default zero-clearance Shapely geometry, medium mesh, rounded indenter,
mixed T3 solid, `E = 1.0 MPa`, `ν = 0.49`, ALM/search/penalty/tolerance,
Skyline solver 및 `0.25 / 48 = 0.005208333333 mm` 첫 travel을 모든 case에서
동일하게 유지했다. Tangent diagnostic과 nonlinear verdict는 서로 다른
fresh `Kratos.Model`에서 실행했다.

```bash
OMP_NUM_THREADS=1 /home/dk/miniconda3/envs/lit/bin/python -B \
  -m analysis.phase4_internal_contact_diagnostic \
  --mesh-level medium --cases A B C D E --first-step-only
```

### First-step 결과

| Case | Internal contact | 결과 | Newton | Assembled LM DOF | Near-zero row |
|---|---|---:|---:|---:|---:|
| A | none | PASS | 1 | 9 | 0 |
| B | bottom only | PASS | 1 | 30 | 0 |
| C | left/right separate | FAIL | first linear solve에서 native abort | 51 | 2 |
| C-left | left only | PASS | 12 | 30 | 1 |
| C-right | right only | FAIL | 35 | 30 | 1 |
| D | left/right/bottom separate | FAIL | 35 | 70 | 2 |
| E | continuous U | FAIL | 35 | 70 | 2 |

Case C의 native abort 전 Skyline은 `Error zero sum`을 기록했고, 이어서
non-finite contact normal을 보고했다. Child process 격리 때문에 다른 case와
artifact 수집은 계속 진행됐다. D는 기존 Phase 4I Trial의 35-iteration
failure를 재현했다.

Sparse diagnostic은 dense rank/SVD를 사용하지 않았다. 첫 tangent를 CSR로
조립해 row norm, diagonal, graph component 및 diagnostic-only SuperLU
factorization을 조사했다. C/D/E에서 반복된 offender는 다음과 같다.

| Node | Reference coordinate | DOF | D row norm |
|---:|---:|---|---:|
| 2 | `(3.5, 0.0)` | `LAGRANGE_MULTIPLIER_CONTACT_PRESSURE` | `6.35e-17` |
| 5 | `(-3.5, 0.0)` | `LAGRANGE_MULTIPLIER_CONTACT_PRESSURE` | `1.69e-15` |

두 node는 lower U-corner가 아니라 각각 right/left side의 상단 endpoint이며
pad bond와 만나는 위치다. C-right의 node 2 row와 failure, C-left의 node 5
near-zero row와 수렴을 함께 보면, 단순히 near-zero threshold 아래라는
사실만으로 failure를 충분히 설명할 수는 없다. 다만 right-side pair 하나가
failure의 충분조건이라는 실행 증거가 있으므로 opposing-side 동시 조립은
필요조건이 아니다. Right endpoint의 LM assembly, boundary orientation/normal,
zero-clearance activation 중 어느 하나가 단독 원인인지는 아직 확정하지 않는다.

### Continuous U contract와 판정

`PadInternalU`와 `StemInternalU`는 root node/condition을 새로 만들지 않고
left/bottom/right semantic boundary membership을 합쳤다. D와 E를 비교하면:

- 네 lower U-corner의 physical node ID와 incident source condition ID가 동일하다.
- duplicate LineCondition connectivity와 duplicate EquationId는 없다.
- D의 lower pad corner는 두 slave process에 속하지만 scalar LM DOF는 하나다.
- E에서는 lower pad corner의 process registration이 하나로 감소한다.
- external과 internal generated CouplingGeometry는 모두 선언한 source
  slave/master connectivity에만 대응한다.
- E에서도 assembled LM DOF는 70개이고 upper side endpoint의 두 near-zero
  row가 남는다.

따라서 separate pair를 continuous U로 합치는 것은 lower-corner process
registration을 정리하지만 현재 first-step singularity를 해결하지 못한다.
E first-step gate가 실패했으므로 0.25 mm/48-step full Trial은 실행하지 않았다.

```text
Phase 4M initialization:          PASS 유지
Phase 4I-D:                       FAIL
Continuous U recovery:            REJECT
Mixed T3 hyperelastic solid:      ADOPT 유지
Phase 4I:                         still incomplete
0.25 mm continuous-U full Trial: NOT RUN (first-step gate)
1.5 mm medium/fine baseline:      NOT RUN
```

상세 JSON/CSV/log는
`output/phase4_internal_contact_diagnostic/`에 보존한다. 다음 단계는
medium/fine baseline이 아니라 현재 Kratos 2D ALM의 right-side upper endpoint
LM/contact construction을 해결하거나 다른 contact formulation/solver를
검토하는 것이다.

---

## 16. Phase 4I-E — right-side mirror 및 orientation audit

### 고정 조건과 실행

Phase 4I-D의 medium mesh, default zero-clearance geometry, material, mixed T3
solid, external indenter, ALM/search 설정, boundary condition 및
`0.25 / 48 = 0.005208333333 mm` 첫 travel을 그대로 사용했다. 각 case는 별도
process와 fresh `Kratos.Model`에서 실행했다.

```bash
OMP_NUM_THREADS=1 /home/dk/miniconda3/envs/lit/bin/python -B \
  -m analysis.phase4_right_side_audit \
  --mesh-level medium --run-orientation-matrix
```

Source mesh에서 좌우 node와 Line2 condition은 reference-coordinate reflection
`(x, y) -> (-x, y)`에 정확히 대응했다. Shapely material/void topology로
계산한 physical outward normal도
`(nx, ny)_right = (-nx, ny)_left`를 만족했으며, 현재 R00 condition ordering은
네 internal side surface 모두에서 물리적으로 올바르다.

### Orientation matrix

| Case | Right slave | Right master | Physical normal | 결과 |
|---|---|---|---|---|
| R00 | current | current | PASS | FAIL, first step 미수렴 |
| R10 | reversed | current | FAIL | `ExecuteInitialize` zero nodal normal |
| R01 | current | reversed | FAIL | `ExecuteInitialize` zero nodal normal |
| R11 | reversed | reversed | FAIL | `ExecuteInitialize` zero nodal normal |

반전 variant는 volume connectivity, root node/condition 수 또는 undirected
boundary connectivity를 바꾸지 않았지만, 선택한 Line2 ordering normal이
material-to-void 방향과 반대였다. 또한 인접 boundary와 공유하는 endpoint에서
area normal이 상쇄되어 Kratos `NormalCheckProcess`가 zero-norm node를
검출했다. 따라서 단순 right slave/master ordering reversal은 production
수정으로 채택하지 않았다.

### 최초 비대칭과 LM contribution

R00은 process 생성 전, `ExecuteInitialize` 직후까지 좌우 mirror contract를
유지했다. Contact search 후에도 upper endpoint normal은 left
`(+1, 0)`, right `(-1, 0)`이고 두 node 모두 `SLAVE/ACTIVE`, weighted gap
`0.0`이었다. 최초 차이는 contact search의 generated pairing이다.

| 항목 | Left node 5 | Right node 2 |
|---|---:|---:|
| Upper endpoint generated pair | 1 | 2 |
| Valid endpoint projection | 1 | 1 |
| Invalid adjacent-master projection | 0 | 1 |
| First-assembly LM row norm | `1.7164e-15` | `6.3117e-17` |
| Local contributor condition 수 | 1 | 2 |

Left node 5의 condition 737은 대응 master segment에만 투영되고 local LM row
norm은 전체 column 기준 `0.764997`, free column 기준 `1.7164e-15`였다.
Nonlinear solve는 이 endpoint를 비활성화하고 12회 Newton iteration으로
수렴했다.

Right node 2의 condition 699는 올바른 segment에 투영되며 전체/local-free row
norm은 각각 `0.750396`/`6.3117e-17`이었다. 그러나 같은 slave condition에
인접한 아래 master segment를 연결한 condition 700도 추가 생성됐다. 이
projection의 segment parameter는 `2.0`으로 범위 밖이며 local LM row는 전체와
free column 모두 정확히 `0.0`이었다. R00은 기존 C-right 35-iteration 실패를
재현했다.

이 결과는 source geometry, semantic membership, condition normal, nodal normal,
gap sign 또는 runtime slave/master role의 좌우 차이를 배제하고 문제 범위를
right upper endpoint의 contact-search pair generation과 후속 LM
assembly/active-set 처리로 좁힌다. 다만 왜 Kratos search가 right에서만 인접
segment를 추가하는지까지는 확인하지 못했으므로 이를 확정된 library-level
근본 원인이라고 표현하지 않는다.

### 판정

```text
Phase 4I-E:                         FAIL
Orientation hypothesis:            REJECT
Source-level correction:           NONE
A/B/C-left/C-right/C/D/E regression: NOT RUN (fix gate)
D/E 0.25 mm full Trial:            NOT RUN (first-step gate)
Three separate pairs:              REJECT 유지
Continuous U:                      REJECT 유지
Mixed T3 hyperelastic solid:       ADOPT 유지
Phase 4I baseline resume:          NO
1.5 mm medium/fine baseline:       NOT RUN
```

Artifact는 기존 Phase 결과를 덮어쓰지 않고
`output/phase4_right_side_audit/`에 저장했다.

---

## 17. Phase 4I-F — search pair와 upper crosspoint multiplier audit

### 실행 범위

Phase 4I-E와 동일한 medium mesh, `0.25 / 48 mm` 첫 travel, geometry, material,
mixed T3, ALM/search, penalty, tolerance, Newton 35회 및 Skyline solver를
유지했다. Kratos contact strategy의 공개 scheme/builder/criterion API로 첫
nonlinear step 순서를 재현해 각 iteration 전후의 endpoint flag, gap, LM,
local condition row 및 Dirichlet 적용 후 global row를 저장했다.

```bash
OMP_NUM_THREADS=1 PYTHONFAULTHANDLER=1 \
  /home/dk/miniconda3/envs/lit/bin/python -B \
  -m analysis.phase4_search_crosspoint_audit \
  --output-directory output/phase4_search_crosspoint_audit
```

### Search 및 causal matrix 결과

Kratos `Line2D2`의 local domain은 `[-1, 1]`이다. Right extra pair의 기존
segment fraction `t=2.0`은 Kratos local coordinate `xi=2t-1=3.0`에
해당하므로 domain 밖이다. 공개
`ExactMortarIntegrationUtility2D2N.TestGetExactAreaIntegration`으로 계산한
overlap도 `5.3013e-14 mm`로 numerical zero였다. Valid right pair의 overlap은
약 `0.35 mm`였다.

| Case | Invalid condition | 결과 | Newton | 관찰 |
|---|---:|---:|---:|---|
| L00 | 없음 | PASS | 12 | 최종 endpoint inactive |
| F00 | 원래 ACTIVE | FAIL | 35 | Skyline zero-sum, 이후 non-finite |
| F02 | condition만 inactive | FAIL | 35 | F00과 동일한 node ACTIVE sequence와 LM response |
| F01 | 생성 전 제거 | UNAVAILABLE | — | Python에 search `INDEX_MAP` filter hook 없음 |
| F03 | valid generated pair만 유지 | UNAVAILABLE | — | condition만 제거하면 C++ pairing map lifecycle 불일치 |
| symmetric control | extra pair 주입 | NOT RUN | — | pair/map을 원자적으로 추가하는 공개 API 없음 |

F00과 F02에서 right node 2는 active-set check 후
`inactive -> active -> active -> active -> inactive ...`의 동일한 초기
sequence를 보였다. 따라서 extra condition의 `ACTIVE` 상태가 failure의
필요조건이라는 가설은 기각한다. F02는 invalid condition을 container와
search map에 그대로 두고 flag만 비활성화했으므로 “pair의 존재 자체”까지
분리한 실험은 아니다.

### Contact–bond crosspoint assembly

Node 2와 node 5는 각각 `PadCutoutRight/Left`와 `PadBondRight/Left`에 동시에
속하며 X/Y displacement가 모두 고정되어 있다. LM 자체는 free scalar DOF다.
첫 active tangent의 수치는 다음과 같다.

| 항목 | Left node 5 | Right node 2 |
|---|---:|---:|
| Endpoint free X/Y DOF | 0 | 0 |
| Valid condition local all-column norm | `0.764997` | `0.750396` |
| Valid-only free-column norm | `1.7164e-15` | `6.3117e-17` |
| Active LM diagonal | `0.0` | `0.0` |
| Dirichlet 후 global LM row norm | `1.7164e-15` | `6.3117e-17` |
| Invalid condition local norm | — | `0.0` |

즉 직접적인 zero-row mechanism은 extra pair가 아니라 active endpoint LM이
고정된 endpoint primal displacement에 결합하고, 나머지 free primal
column과는 numerical zero 수준으로만 결합한다는 점이다. Active 상태에서는
LM diagonal이 없지만 inactive 상태에서는 condition이 diagonal
stabilization을 제공한다. Right iteration 2의 inactive diagonal은
`3.032058`이었다.

Left도 처음부터 구조적으로 안전한 것은 아니다. Iteration 1의 positive
augmented pressure로 비활성화된 뒤 iteration 4–6, 8, 10에는 다시
활성화됐다. 최종 iteration 12에서는 inactive이고 LM diagonal
`3.091055`가 있는 tangent로 수렴했다. 따라서 left PASS는 crosspoint row가
본질적으로 유효해서가 아니라 final active-set이 endpoint를 inactive
formulation으로 놓은 데 의존한다.

### Library behavior와 application susceptibility

설치 kernel의 exact commit은
`14ee273e97af403622699e797ea5fa356b1a7e60`이다. Source trace에서 확인한
동작은 다음과 같다.

- `BaseContactSearchProcess::SearchUsingKDTree`는 KD-tree/OBB broad-phase
  candidate를 만들고, `CheckCondition`은 identity, normal 및 duplicate
  조건을 확인하지만 endpoint `Line2D2::IsInside` 검사를 하지 않는다.
- `CreateAuxiliaryConditions`/`AddPairing`은 candidate를 coupled condition으로
  만들고 condition을 `ACTIVE`로 설정한다.
- `WEIGHTED_GAP`은 ComputingContact condition의 explicit contribution을
  slave node에 누적한다.
- `ComputeALMFrictionlessActiveSet`은 pair별 exact overlap이 아니라
  `SCALE_FACTOR*LM + INITIAL_PENALTY*WEIGHTED_GAP`의 node-level 부호로
  `ACTIVE`를 판정한다.
- 조사한 공식 search/active-set/mortar-condition 경로에서는 fully
  Dirichlet contact–bond crosspoint LM을 자동 제거하거나 별도로
  stabilization하는 공식 규칙을 찾지 못했다.

따라서 library-level behavior는 out-of-domain numerical-zero-overlap
condition 생성을 허용하는 broad-phase/condition creation 흐름이다.
Application-level susceptibility는 바로 그 slave endpoint를 fully constrained
pad-bond crosspoint로 contact multiplier space에 포함한 것이다. F02 결과상
전자가 직접적인 singular row의 필요조건은 아니다.

### 판정

물리적이고 mesh-independent한 production correction은 검증하지 못했다.
Endpoint 삭제, LM 고정, contact 절단, endpoint 강제 inactive는 채택하지
않았다. Safe F01/F03 경로가 없으므로 pair acceptance correction도 완전한
fix로 주장하지 않는다. 공식 crosspoint treatment가 확인되지 않은 상태에서
application-level LM rule도 임의 도입하지 않았다.

```text
Phase 4I-F:                         FAIL
Invalid-pair ACTIVE causal hypothesis: REJECT
Active contact–bond crosspoint deficiency: CONFIRMED
Production correction:             NONE
A/B/C-left/C-right/C/D/E regression: NOT RUN (correction gate)
D/E 0.25 mm x 48-step Trial:       NOT RUN (first-step gate)
1.5 mm baseline:                   NOT RUN
Mixed T3 hyperelastic solid:       ADOPT 유지
```

상세 lifecycle, pair 비교, crosspoint DOF map, source/API trace 및 격리 로그는
`output/phase4_search_crosspoint_audit/`에 보존한다.

---

## 18. Phase 4I-G — bounded multiplier-space treatment

### Source audit

설치된 Kratos `10.3.0-14ee273e`의 exact commit
`14ee273e97af403622699e797ea5fa356b1a7e60`을 기준으로 다음 경로를
확인했다.

- `AuxiliaryAddDofs`: frictionless ALM이면 scalar pressure LM DOF를 root
  model part 전체 node에 추가한다.
- `ALM_frictionless_mortar_contact_condition.cpp`의 `EquationIdVector`와
  `GetDofList`: slave geometry의 모든 node에서 LM을 요구하며 displacement
  fixity를 검사하지 않는다.
- active LM은 mortar coupling을 사용하고 별도 diagonal이 없으며, inactive
  LM에는 `scale_factor² / penalty` diagonal이 생긴다.
- block builder는 entity assembly 이후 fixed displacement row/column을
  제거한다.
- contact builder의 isolated-node 처리는 incident contact condition을
  기준으로 하며 contact–Dirichlet trace를 검사하지 않는다.

공식 LM omission, static condensation, boundary-trace restriction 또는
crosspoint setting은 확인되지 않았다. 따라서 G1은 `UNAVAILABLE`이다.

### Mirrored/refined minimal algebra

Adopt된 mixed T3 solid, flat rigid master, ACTIVE frictionless ALM contact,
fully fixed endpoint와 인접 free slave node를 가진 fresh patch를 좌우
mirror 및 2/4/8 divisions에서 조립했다.

| Divisions | Left endpoint ID | Right endpoint ID | Left/right post-Dirichlet LM row norm | LM diagonal |
|---:|---:|---:|---:|---:|
| 2 | 3 | 9 | `1.0000e-4` / `1.0000e-4` | 0 / 0 |
| 4 | 5 | 25 | `2.0000e-4` / `2.0000e-4` | 0 / 0 |
| 8 | 9 | 81 | `4.0000e-4` / `4.0000e-4` | 0 / 0 |

모든 case에서 endpoint X/Y는 고정되고 endpoint contact는 ACTIVE였지만,
LM basis가 인접 free slave trace와 결합하여 zero row가 생기지 않았다.
좌우 norm 차이는 최대 `4.44e-16`이며 refinement에 따라 node ID가 바뀌어도
동일했다.

이 결과는 “fully fixed contact endpoint”라는 topology/fixity 조건만으로
독립 LM basis를 제거하면 정상 mortar support도 함께 제거할 수 있음을
보인다. 실제 fingertip의 `6.3117e-17` row는 해당 local geometry/support에서
생긴 cancellation과 결합된 더 좁은 현상이다. Kratos Python lifecycle에서
안전하게 그 basis만 제외/condense할 hook도 없으므로 G2는 production
correction으로 `REJECTED`했다.

```text
G1 official treatment:        UNAVAILABLE
G2 application restriction:  REJECTED AS PRODUCTION FIX
A/B/C-left/C-right/C/D/E:     NOT RUN (candidate gate)
0.25 mm x 48-step Trial:      NOT RUN (candidate gate)
Phase 4I-G:                   FAIL/BLOCKED
```

---

## 19. Phase 4J — no-void external-contact-only baseline

### Preflight

기존 `FingertipParameters()` default를 그대로 사용했다.

- `void_width = void_height = 0`
- `void_geometry is None`, void area `0`
- internal ALM group/process/`ContactSubN`/generated condition 없음
- 유일한 runtime pair:
  `PadOuterArc` slave — `IndenterContactArc` master
- `TotalLagrangianMixedVolumetricStrainElement2D3N`
- `HyperElasticPlaneStrain2DLaw`, `ν = 0.49`

Kratos의 `AuxiliaryAddDofs` 때문에 외부 ALM process 하나만 있어도 scalar LM
DOF object는 root node 전체에 생긴다. 따라서 internal semantic surface의
“LM DOF object가 문자 그대로 0개”는 아니다. 그러나 generated condition
`GetDofList`로 확인한 assembled LM node는 외부 pair에만 속했고 internal
exclusive LM assembly는 모든 J0/J1/J2 case에서 0개였다.

외부 penetration 후처리도 수정했다. 기존 코드는 접촉하지 않는
`PadOuterArc` 전체 node를 작은 indenter에 투영해 최대 `12.5 mm`의 가짜
penetration을 만들었다. 외부 pair는 실제 ACTIVE slave node만 평가하도록
범위를 제한했다. internal pair의 기존 범위는 변경하지 않았다.

### Baseline 결과

| Case | Mesh | Steps | 결과 | Final reaction | min det(F) | Max strain | Max Newton | Solve time |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| J0 | medium | 1 | PASS | `0.0013569 N` | `0.98984` | `0.002493` | 1 | `1.22 s` |
| J1 | medium | 48 | PASS | `0.861926 N` | `0.76282` | `0.17165` | 3 | `114.73 s` |
| J2 | fine | 48 | PASS | `0.864680 N` | `0.69839` | `0.17278` | 3 | `331.65 s` |

Medium/fine final reaction 차이는 `0.003185 = 0.319%`로 10% gate를
통과했다. 두 48-step curve는 monotonic/smooth하고 normalized maximum
second difference는 medium `0.001741`, fine `0.002304`였다. 모든 step에서
finite field, positive det(F), force equilibrium, active-set convergence,
penetration 및 volumetric checkerboard check가 통과했다.

`STRAIN_ENERGY`는 앞 phase에서 accepted runtime output이 아니므로 새
acceptance metric으로 만들지 않았다. 대신 reaction curve trapezoid 적분을
명시적으로 external-work proxy로 저장했다: medium `0.59817 N·mm`, fine
`0.60065 N·mm`.

J1 solve/result/history/profile checkpoint는 정상 완료됐지만 당시 실행 중이던
pre-fix plot writer가 존재하지 않는 `internal_left` key를 요구해 child exit
code 1을 냈다. 수치 solve는 재실행하지 않았고 수정된 writer로 누락 plot을
재생성했다. J0 재실행과 J2는 같은 수정본에서 exit code 0으로 전체
post-processing을 통과했다.

J3는 roadmap에 symbolic `x_c`만 있고 실행 가능한 discrete location list가
없어서 `SKIPPED`했다. J4도 문서화된 no-void mechanics candidate list가
없어 새 design space를 만들지 않고 `SKIPPED`했다.

```text
Phase 4I internal contact: BLOCKED 유지
Phase 4I-G:                FAIL/BLOCKED
Phase 4J:                  PASS
J3 location sweep:         SKIPPED (defined cases 없음)
J4 parameter sweep:        SKIPPED (defined candidates 없음)
```

상세 checkpoint와 result는
`output/phase4_crosspoint_multiplier_treatment/` 및
`output/phase4_no_void_baseline/`에 저장했다.

---

## 20. Phase 4K — contact-to-observation deformation transfer map

Phase 4J solver 설정을 바꾸지 않고 수렴 step 직후 observer만 추가했다.
`PadOuterArc` reference connectivity를 node-ID 독립적으로 정렬하고, 우측
`xi=0→0.25`, 좌측 `xi=1→0.75` 구간에 각각 41개 고정 `eta` sample을
Line2 shape function으로 보간했다. 두 side 모두 reference outward-normal
bulging을 positive primary channel로 사용한다.

Kratos 10.3 source/runtime audit에서 실제 external slave에 저장된
`AUGMENTED_NORMAL_CONTACT_PRESSURE`(non-historical), `NODAL_AREA`
(non-historical), `NORMAL`(historical)을 선택했다. Contact resultant는
global loading direction으로 투영하고 canonical indenter reaction과 2%
closure를 검사한다. Closure 실패 시 centroid/2D length는 공개하지 않고 raw
candidate만 보존한다.

K2 medium/fine은 각각 Phase 4J J1/J2 final reaction과 minimum det(F)를
정확히 재현했다. 모든 location/mesh solve는 48 step에 도달했고 finite
field와 positive det(F)를 유지했다.

| Mesh/location | Solve | Final reaction | Verified centroid/length | Final closure |
|---|---:|---:|---|---:|
| medium `.20` | PASS | `0.132235 N` | `0.173066 / 2.34749 mm` | `0.133%` |
| medium `.35` | PASS | `0.432058 N` | `0.328307 / 2.93384 mm` | `0.0129%` |
| medium `.50` | PASS | `0.861926 N` | `0.500159 / 3.32329 mm` | `0.508%` |
| medium `.65` | PASS | `0.422103 N` | UNVERIFIED | `4.327%` |
| medium `.80` | PASS | `0.130621 N` | UNVERIFIED | `2.286%` |
| fine `.20` | PASS | `0.133677 N` | `0.172961 / 2.13546 mm` | `0.0317%` |
| fine `.50` | PASS | `0.864680 N` | `0.500039 / 3.42512 mm` | `0.0675%` |
| fine `.80` | PASS | `0.129790 N` | UNVERIFIED | `4.702%` |

Spot-check final normal-profile relative L2 difference는 `.20/.50/.80`에서
각각 `0.635% / 0.255% / 0.659%`다. 수치는 양호하지만 사전 정의된
CODTM profile threshold가 없으므로 mesh convergence는 `PROVISIONAL`이다.
Optical/noise model도 없으므로 location separability와 SVD는 descriptive
metric일 뿐 PASS criterion이 아니다.

```text
CODTM extraction pipeline:       PASS
Center reconstruction:           PASS
Medium location map:             PARTIAL
Fine spot checks:                PARTIAL
CODTM mesh convergence:          PROVISIONAL
Contact-distribution closure:    PARTIAL (256/384 records verified)
Mechanical separability:         DESCRIPTIVE ONLY
Geometry optimization:           NOT STARTED
```

상세 정의와 결과는 `mechanical-deformation-transfer-map.md`, artifact는
`output/phase4_mechanical_transfer_map/`에 저장했다.

---

## 21. Phase 4K-Viz — CODTM spatial visualization

Phase 4K canonical artifacts만 read-only로 사용해 10개 static figure를 300
DPI PNG와 vector PDF로 생성했다. NPZ axes, case/side ordering, 31,488 CSV
rows, 384 valid displacement records를 metadata-driven loader로 교차검사했고,
입력 7개 파일의 실행 전/후 SHA-256이 모두 동일했다. Kratos solve와 contact
parameter 변경은 없었다.

Scientific coordinate는 `(side, eta)`로 유지했다. Combined display에서만
`zeta_right=eta-1`, `zeta_left=1-eta`를 사용하며, 서로 다른 `eta=1`
material points 사이에 명시적 unsampled gap을 렌더링했다. Location
interpolation/smoothing, crown endpoint 병합, deformed contact surface 추정은
하지 않았다.

Phase 4K metric은 정확히 재현됐다. `delta=1.5 mm` medium raw distance
off-diagonal range는 `0.265411–0.982068 mm`, normalized shape distance는
`0.552584–1.968494`다. Mirror residual은 `.20/.80`, `.35/.65`, `.50/.50`
pair에서 각각 `0.0670%`, `0.1498%`, `0.0942%`여서 `CONSISTENT`다.
Stored tangent gain과 독립 finite difference의 최대 차이는 `0.0`이다.
Medium/fine profile 결과도 기존 `0.635%/0.255%/0.659%`를 재현했지만
scientific status는 그대로 `PROVISIONAL`이다.

```text
Data ingestion:                 PASS
Coordinate semantics:           PASS
Static visualization pipeline:  PASS
Metric reproduction:            PASS
Mirror symmetry:                CONSISTENT
Medium/fine visualization:      PROVISIONAL
Publication readiness:          READY
Optical observability:          NOT EVALUATED
```

Contact centroid/length는 기존 force-closure mask를 유지해 256/384 record만
verified로 취급했다. Visualization 작업만으로 Phase 4K ledger, descriptor
closure, CODTM mesh-convergence, optical observability를 승격하지 않았다.
상세 설명은 `codtm-visualization.md`, 산출물은
`output/phase4_codtm_visualization/`에 있다.

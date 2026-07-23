# LIT Hand — Kratos 2D Hyperelastic Contact Validation

> **문서 목적**  
> LIT Hand의 compliant fingertip pad 해석에 사용할 Kratos 2D FEM 구성의 검증 과정, 실패 사례, 수치 결과, 최종 채택 기준을 보존한다. 이후 실제 pad 모델링, 논문 작성, 코드 재구성 시 이 문서를 기준 기록으로 사용한다.

- 기록일: 2026-07-22
- 대상: 2D plane-strain, nearly incompressible hyperelastic solid, frictionless contact
- 최종 프로젝트 상태: **실제 LIT fingertip 형상 모델링으로 진행 가능**

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
- 2D frictionless ALM mortar contact는 LIT Hand의 변형 전달 해석에 **conditional ADOPT**한다.
- 접촉 예상 영역은 충분히 세분화하고, medium/fine mesh convergence를 확인해야 한다.
- Coarse mesh의 pointwise contact pressure는 신뢰하지 않는다.
- 현재 solver 또는 contact formulation을 변경해야 할 blocker는 없다.

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
| Project decision | LIT Hand application 적합성 평가 | — | **Conditional PASS / 실제 형상으로 진행** |

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

Penalty mortar 또는 다른 contact solver는 현재 blocker가 아니다. 충분히 refined된 실제 모델에서도 active-set cycling이나 pressure-field 문제가 반복될 경우에만 비교 후보로 검토한다.

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

Solver baseline 검증은 application 진행에 충분하다. 이후에는 실제 구조의 정보 전달 성능을 평가한다.

| Phase | 확인할 것 | 핵심 산출물 |
|---|---|---|
| Phase 4 | 실제 pad 외곽, stem/support, solid baseline | 기본 deformation-transfer response |
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

> **Kratos의 mixed T3 hyperelastic solid (`ν = 0.49`)와 frictionless ALM mortar contact는 LIT Hand의 2D deformation-transfer study에 사용할 수 있다. 접촉부 mesh refinement와 medium/fine convergence 확인을 전제로 실제 fingertip 모델링으로 진행한다.**

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

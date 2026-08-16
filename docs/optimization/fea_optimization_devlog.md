---
title: "물리를 바꾸지 않고 FEA를 21.5배 빠르게 만든 과정"
date: 2026-08-15
tags:
  - robotics
  - FEA
  - Kratos
  - simulation
  - optimization
draft: true
---

# 물리를 바꾸지 않고 FEA를 21.5배 빠르게 만든 과정

로봇 핑거팁 morphology를 최적화하는 과정에서 가장 큰 병목은 결국 FEA였다.

한 morphology를 평가할 때 여러 contact state에 대해 비선형 contact FEA를 풀고, 변형된 geometry를 optical transport 계산으로 넘겨야 한다. 처음에는 이 과정을 그냥 "비싼 물리 시뮬레이션"으로 받아들이고 있었는데, 2D와 3D optical model을 비교하고 더 많은 morphology를 검증하려고 하니 FEA 비용이 갑자기 전체 연구 속도를 제한하기 시작했다.

그래서 질문을 바꿨다.

> 더 빠른 solver를 찾을 수 있을까?

보다 먼저,

> 지금 계산 중인 것들 중 실제로 optimization decision에 필요한 것은 무엇이고, 무엇이 중복이거나 과한가?

를 보기로 했다.

결과적으로 production physics나 constitutive/contact formulation을 바꾸지 않고, 한 morphology의 대표 FEA evaluation 시간을 약 363초에서 16.9초로 줄였다.

최종적으로 측정된 speedup은 21.53배, wall time 감소는 95.36%였다.

---

## 1. 시작점: 한 morphology에 약 6분

기준 protocol은 다음과 같았다.

- medium mesh
- 48 load steps
- full diagnostics
- x = -3 mm, +3 mm
- indentation = 0.5 mm, 1.0 mm
- 각 depth를 독립 history로 계산
- 기존 trusted nonlinear solver 설정 사용

네 개의 loaded state를 계산하는 baseline wall time은 약 362.99초였다.

대략적인 timing breakdown은 다음과 같았다.

| 항목 | 시간 |
| --- | ---: |
| Nonlinear solve | 203.94 s |
| Per-step processing / diagnostics | 158.23 s |
| Total | 362.99 s |

처음 예상과 달리 solver 자체만 느린 것이 아니었다.

전체 시간의 상당 부분이 각 increment에서 수행하는 diagnostics와 post-processing에 쓰이고 있었다. 이 결과 덕분에 solver를 갈아엎기 전에 더 싸고 안전한 최적화 지점들이 있다는 것을 알 수 있었다.

---

## 2. 원칙: FEA를 빠르게 만들되 optimization decision은 보존한다

이번 작업의 목표는 stress field를 가능한 한 정밀하게 재현하는 범용 FEA solver를 만드는 것이 아니었다.

FEA는 morphology optimization pipeline의 한 단계다.

중요한 것은 다음 chain이 유지되는가였다.

morphology  
→ contact deformation  
→ deformed optical geometry  
→ optical transport  
→ contact-state separability  
→ morphology ranking

따라서 fast configuration을 평가할 때 우선적으로 본 값은 다음과 같았다.

- nonlinear convergence
- reaction force
- contact activation topology
- penetration sanity
- deformed semantic boundary geometry
- downstream optical field
- left/right contact-state separability
- morphology ordering

특히 mesh가 바뀌면 node correspondence를 직접 비교하지 않고, 의미가 같은 boundary를 공통 parameter grid로 resampling해서 geometry error를 비교했다.

즉 "FEA field 전체가 똑같은가?"가 아니라 "최적화에 필요한 물리적 결과가 보존되는가?"가 acceptance criterion이었다.

---

## 3. 첫 번째 개선: full diagnostics를 optimization loop에서 빼기

Baseline profiling에서 의외로 큰 병목은 per-step diagnostics였다.

Full diagnostics에서는 validation에 유용한 여러 검사를 매 increment마다 수행한다. 하지만 morphology search의 모든 candidate에서 그 전체 검사가 필요한 것은 아니었다.

그래서 두 경로를 분리했다.

### Validation path

- full diagnostics
- detailed acceptance checks
- reference/finalist validation에 사용

### Optimization path

- minimal diagnostics
- deformation, reaction, contact sanity, downstream optics에 필요한 결과만 유지

같은 physical state에서 full과 minimal을 비교했을 때:

- reaction relative error: 0
- semantic boundary profile error: 0
- optical result: 동일
- end-to-end speedup: 1.52배

즉 physics를 바꾼 것이 아니라, optimization loop에서 필요하지 않은 bookkeeping을 줄인 것이다.

이 단계만으로도 꽤 큰 시간 절감이 나왔다.

---

## 4. 두 번째 개선: mesh를 전체적으로 줄이지 말고 필요한 곳을 남긴다

다음은 mesh였다.

기준 medium mesh와 여러 coarse policy를 비교했다.

Nominal morphology 기준:

| Mesh | Nodes | Elements |
| --- | ---: | ---: |
| Reference medium | 5,968 | 11,364 |
| coarse_b | 4,526 | 8,567 |
| coarse_c | 4,088 | 7,721 |

단순히 element 수만 줄인 것은 아니다.

중요한 contact/optical boundary는 충분히 보존하면서 bulk 쪽 resolution을 낮추는 방식으로 coarse mesh를 만들었다.

coarse_b는 nominal과 candidate49의 네 physical state에서 reference medium과 비교했을 때 설정한 fidelity guardrail을 모두 통과했다.

대표적으로 nominal에서는 outer boundary maximum error가 약 0.02–0.024 mm 수준이었고, candidate49에서는 약 0.012–0.015 mm 수준이었다.

더 중요한 것은 downstream result였다.

Reference medium에서의 left/right separability:

| Morphology | 0.5 mm | 1.0 mm |
| --- | ---: | ---: |
| nominal | 0.07511 | 0.11289 |
| candidate49 | 0.12738 | 0.19863 |

coarse_b:

| Morphology | 0.5 mm | 1.0 mm |
| --- | ---: | ---: |
| nominal | 0.07768 | 0.10245 |
| candidate49 | 0.11717 | 0.19377 |

절대값은 조금 변했지만 candidate49 > nominal이라는 optimization ordering은 유지됐다.

그래서 coarse_b를 optimization용 fast mesh로 선택했다.

Fine mesh는 모든 candidate에 쓰는 reference가 아니라, periodic guardrail과 finalist recheck 용도로 남겼다.

---

## 5. 세 번째 개선: 이미 한 history 안에 있는 state를 다시 풀지 않는다

기존에는 각 위치에서 0.5 mm와 1.0 mm indentation을 독립적으로 풀었다.

즉 한 위치에 대해:

0 → 0.5 mm

그리고 다시:

0 → 1.0 mm

를 각각 계산했다.

하지만 solver는 이미 1.0 mm monotonic history 도중 0.5 mm snapshot을 만들 수 있었다.

따라서:

0 → 0.5 mm snapshot → 1.0 mm

한 history로 두 state를 모두 얻을 수 있다.

이것을 continuation path로 사용하기 전에 independent solve와 직접 비교했다.

두 위치 모두에서:

- reaction relative error: 0
- semantic boundary profile error: 0
- optical field difference delta: 0
- 0.5 mm snapshot mutation check: PASS

즉 이건 approximation도 아니고 새로운 solver trick도 아니다.

기존에 동일한 monotonic trajectory를 두 번 계산하던 중복을 제거한 것이다.

---

## 6. 네 번째 개선: load step 수가 너무 많지 않은가?

기준은 48 steps였다.

하지만 각 increment가 충분히 작다는 이유만으로 48 steps가 optimization에 필요한 최소 resolution이라는 보장은 없었다.

먼저 step count를 줄여 보면서 convergence, Newton iterations, deformation, reaction, contact, optical result를 비교했다.

초기 sweep에서 이미 48 → 12 steps가 매우 유망했다.

예를 들어 representative case에서:

- reference-medium: 약 57.1 s → 17.1 s
- coarse-C: 약 26.7 s → 7.65 s

최종 decision study에서는 coarse_b + minimal diagnostics + continuation을 고정하고 48 / 24 / 12 steps를 nominal과 candidate49에 비교했다.

결과:

| Steps | Gate | Max reaction error | Max boundary error | Max optical delta | Max separability absolute delta |
| --- | --- | ---: | ---: | ---: | ---: |
| 24 | PASS | 0.192% | 0.00282 mm | 0.01680 | 0.00203 |
| 12 | PASS | 0.320% | 0.01724 mm | 0.04077 | 0.01650 |

12 steps는 24 steps보다 당연히 reference에서 더 멀어졌지만, 우리가 요구한 scientific decision gate는 모두 통과했다.

- nonlinear convergence PASS
- snapshot preservation PASS
- contact topology PASS
- penetration sanity PASS
- candidate49 > nominal ordering 유지

추가로 기존 sweep에서 성공했던 candidate_0048을 guardrail morphology로 선택해 48 vs 12 steps를 비교했고, 이것도 PASS했다.

그래서 optimization path에서는 12 steps를 채택했다.

---

## 7. 12 steps가 "48 steps와 동일하다"는 뜻은 아니다

이 부분은 중요하다.

12-step configuration은 high-fidelity replacement가 아니다.

특히 nominal의 1.0 mm separability에서는 48-step 대비 상대 변화가 약 16.1%였다. Absolute delta는 0.01650이었다.

따라서 이 결과를 다음처럼 해석하면 안 된다.

> 12-step FEA는 48-step FEA와 동일하다.

정확한 해석은 다음에 가깝다.

> 현재 morphology optimization decision에 필요한 contact/deformation/optical ordering을 유지하면서 계산량을 크게 줄일 수 있다.

그래서 최종 구조는 fidelity tier로 나뉜다.

### Search

- coarse_b
- 12 steps
- minimal diagnostics
- continuation
- trusted solver

### Periodic / finalist validation

- reference medium 또는 fine mesh
- 48 steps
- 필요한 경우 full diagnostics

빠른 모델과 validation 모델의 역할을 분리한 것이다.

---

## 8. Solver를 바꾸는 것은 별 도움이 되지 않았다

당연히 solver setting도 확인했다.

Tolerance, DOF reform, storage 관련 설정과 alternative linear solver를 비교했다.

AMGCL도 시험했다.

결과적으로 AMGCL은 solve 자체는 가능했지만:

- trusted configuration: 약 57.05 s
- AMGCL: 약 66.94 s
- linear convergence warning 발생

즉 더 느리고, warning까지 있었다.

다른 solver-setting variation들도 의미 있는 speedup을 주지 못했다.

이 결과는 꽤 유용했다.

성능 문제의 핵심이 "더 좋은 linear solver를 찾아야 한다"가 아니라는 것을 확인했기 때문이다.

이후 solver tuning은 중단했다.

---

## 9. 좌우 symmetry는 거의 완벽했지만 사용하지 않았다

Geometry와 loading이 대칭이기 때문에 -3 mm 결과를 mirror해서 +3 mm를 재사용할 가능성도 확인했다.

Measured difference:

- maximum boundary profile error: 0.00080 mm
- reaction difference: 0.034%

수치적으로는 매우 좋았다.

하지만 symmetry reuse는 optimization path에 자동으로 넣지 않았다.

이건 추가적인 assumption을 production path에 넣는 최적화이기 때문이다.

현재 speedup이 충분히 커졌기 때문에, 더 공격적으로 한 history까지 줄이는 것보다 symmetry는 future option으로 남기는 편이 낫다고 판단했다.

---

## 10. Candidate-level multiprocessing

한 candidate의 latency를 줄이는 것과 별개로, 여러 morphology를 동시에 평가하는 throughput도 확인했다.

8 physical core CPU에서:

- 1 process
- 2 processes
- 4 processes

를 비교했고, 4 processes × 2 threads configuration에서 1-process 대비 약 3.60배 throughput을 얻었다.

따라서 Bayesian optimization이나 batch evaluation에서는 candidate-level multiprocessing을 우선 사용한다.

중요한 점은 이 3.60배를 single-candidate 21.53배와 단순 곱해서 "77배 빨라졌다"고 부르지 않는 것이다.

21.53배는 직접 측정한 single-morphology latency speedup이고, 3.60배는 별도로 측정한 multi-candidate throughput effect다.

---

## 11. 최종 결과

마지막에는 개별 speedup을 곱해서 추정하지 않고, complete morphology evaluation을 직접 측정했다.

| Configuration | Wall time |
| --- | ---: |
| Baseline: medium + 48 + full + independent histories | 362.99 s |
| Safe fast: coarse_b + 48 + minimal + continuation | 59.10 s |
| Selected fast: coarse_b + 12 + minimal + continuation | 16.86 s |

Baseline 대비:

- 21.53× speedup
- 95.36% wall-time reduction

결국 약 6분 걸리던 morphology evaluation이 약 17초가 됐다.

---

## 12. 무엇이 실제로 효과가 있었나

이번 최적화를 정리하면 solver 자체보다 workflow와 fidelity allocation이 훨씬 중요했다.

### 큰 효과

1. Full diagnostics를 validation에만 사용
2. Optimization용 coarse mesh 사용
3. 0.5 / 1.0 mm independent history를 continuation으로 합침
4. 48 → 12 load steps
5. Candidate-level multiprocessing

### 효과가 거의 없거나 채택하지 않은 것

- AMGCL
- nonlinear tolerance tweaking
- DOF reform/storage tuning
- automatic symmetry reuse

이 결과를 보고 나면 "FEA optimization"이라는 표현도 조금 다르게 느껴진다.

우리가 한 일은 solver를 마법처럼 빠르게 만든 것이 아니라, 연구 질문에 필요한 fidelity가 어디까지인지 정의하고 그 이상을 매 candidate마다 반복하지 않도록 pipeline을 정리한 것이었다.

---

## 13. Benchmark를 하면서 생긴 의외의 일

이번 작업이 예상보다 오래 걸린 이유 중 하나는 benchmark harness 자체도 검증해야 했기 때문이다.

진행 중 다음과 같은 문제를 발견했다.

- benchmark path에서 ±3 mm fixture location이 잘못 전달되던 문제
- continuation snapshot mutation check가 잘못된 snapshot끼리 비교하던 문제
- symmetry comparison에서 일부 semantic boundary parameterization을 잘못 뒤집던 문제
- 서로 다른 step/history profile artifact가 같은 filename을 사용하던 문제
- failed morphology aggregation이 benchmark summary를 중단시키던 문제

중요한 점은 이런 오류가 발견됐을 때 "이상하지만 결과가 좋아 보이니 계속 간다"가 아니라, 오염된 benchmark result를 폐기하고 영향을 받은 stage만 다시 계산했다는 것이다.

이 과정에서 한 가지 교훈을 얻었다.

> Performance benchmark도 scientific experiment다. Benchmark harness가 틀리면 speedup 숫자가 아무리 예뻐도 의미가 없다.

반대로 metadata나 reporting 문제 때문에 이미 유효한 비싼 계산을 무조건 다시 돌리는 것도 피해야 한다.

---

## 14. 지금의 FEA policy

현재 optimization용 FEA policy는 다음과 같다.

### Optimization path

- coarse_b mesh
- 12 load steps
- minimal diagnostics
- 0 → 0.5 → 1.0 continuation
- existing trusted solver
- symmetry reuse disabled

### Guardrails

- periodic reference-medium recheck
- important finalists는 medium/fine에서 재검증
- high-fidelity 결과가 필요할 때 48 steps / full diagnostics 사용
- morphology ordering과 downstream optical metric이 계속 유지되는지 확인

즉 fast FEA를 새로운 "정답"으로 만든 것이 아니라, search와 validation의 fidelity를 분리했다.

---

## 15. 가장 큰 변화

숫자보다 실제 연구 workflow의 변화가 더 크다.

기존에는 morphology 20개를 추가 검증한다고 하면 FEA만으로도 꽤 부담스러웠다.

Baseline 기준으로는:

20 × 362.99 s ≈ 2시간

이었다.

현재 fast path를 단순 serial로만 계산해도:

20 × 16.86 s ≈ 5.6분

수준이다.

여기에 candidate-level multiprocessing까지 적용하면 FEA는 더 이상 morphology study의 가장 무서운 병목이 아니다.

덕분에 이제 계산 예산을 "FEA를 몇 번 감당할 수 있나?"가 아니라,

- 얼마나 많은 morphology를 2D/3D optics로 비교할 것인가
- 어떤 finalist를 higher-fidelity model로 재검증할 것인가
- physical experiment budget을 어디에 쓸 것인가

같은 연구 질문에 쓸 수 있게 됐다.

---

## Takeaway

이번 작업에서 가장 효과가 컸던 질문은

> 더 빠른 solver가 있나?

가 아니었다.

오히려:

> 이 계산 중에서 매 candidate마다 정말 필요한 것은 무엇인가?

였다.

비선형 contact physics와 production defaults는 그대로 두고,

- diagnostics를 목적에 맞게 분리하고,
- mesh fidelity를 downstream metric으로 검증하고,
- 중복 history를 제거하고,
- load increments를 실제 필요한 수준까지 줄이고,
- parallelism을 candidate level에 배치하자,

한 morphology의 FEA evaluation은 약 363초에서 16.9초로 줄었다.

**21.53× faster, 95.36% less wall time.**

그리고 최종적으로 더 중요한 것은 speedup 자체보다, 그 speedup을 morphology ranking과 scientific guardrail을 유지하면서 얻었다는 점이다.

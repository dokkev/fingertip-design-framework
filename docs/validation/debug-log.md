# LIT Hand FEM Validation — Living Debug Log

> 이 문서는 LIT Hand의 2D hyperelastic-contact validation을 여러 작업 세션에 걸쳐
> 이어가기 위한 **working memory**다. 확정된 수치와 정식 판정은
> [`hyperelastic-contact.md`](hyperelastic-contact.md)를 기준으로
> 하고, 이 문서는 각 단계의 가설, 통제 실험, 증거, 판정, 다음 질문을 연결한다.

- 시작일: 2026-07-23
- 범위: Kratos 10.3, 2D plane strain, nearly incompressible hyperelastic pad,
  frictionless ALM mortar contact
- 현재 단계: **Phase 4J 완료**
- 현재 프로젝트 상태: **Mixed solid ADOPT / external-only baseline PASS /
  actual zero-clearance internal contact BLOCKED**

---

## 0. 세션 재개용 handoff

### 현재까지 확정된 것

- Solid formulation:
  `TotalLagrangianMixedVolumetricStrainElement2D3N`을 유지한다.
- Constitutive law:
  `HyperElasticPlaneStrain2DLaw`를 유지한다.
- Poisson's ratio:
  `ν = 0.49`를 채택하고 `ν = 0.499`는 채택하지 않는다.
- Localized rounded-indenter benchmark에서 2D frictionless ALM contact는
  medium/fine mesh에 대해 conditional ADOPT 상태다.
- 실제 fingertip의 external rounded-indenter contact는 정상이다.
- Actual zero-clearance internal contact에서는 bottom-only가 정상이며,
  left-only는 수렴하고 right-only는 실패한다.
- Right-side source geometry, condition ordering, physical normal,
  nodal normal, semantic membership, slave/master role, initial gap은
  좌우 대칭 계약을 만족한다.
- 최초 좌우 비대칭은 **contact search 후 generated pair set**에서 발생한다.
- Right upper endpoint에만 local segment domain 밖의 adjacent-master pair가 하나 더 생성된다.
- 단순 right-side ordering reversal은 물리적으로 틀리고 zero nodal normal을 만들므로
  orientation 가설은 REJECT다.
- Three separate pairs와 continuous-U 모두 현재 recovery 방법으로는 REJECT다.
- Production boundary/contact source는 아직 수정하지 않았다.

### 현재 blocker

Right upper endpoint `(3.5, 0)`의 slave node 2에서:

1. 올바른 master segment pair가 생성되고,
2. 인접 lower master segment에 대한 extra pair도 생성되며,
3. extra pair의 projection parameter는 `2.0`으로 segment domain 밖이고,
4. 그 condition의 local LM row contribution은 정확히 `0`이며,
5. right endpoint LM은 ACTIVE로 남아 first-step Newton solve가 35회에서 실패한다.

Left upper endpoint `(-3.5, 0)`의 node 5에서는 valid pair 하나만 생성된다.
그 endpoint의 free-column LM coupling도 거의 0이지만 nonlinear iteration 중
endpoint가 비활성화되어 12회에 수렴한다.

### 현재 가장 데이터에 맞는 해석

아래 문장은 **working hypothesis**이며 아직 확정된 root cause가 아니다.

> Right-side search asymmetry가 원래부터 취약했던 upper contact–bond crosspoint
> multiplier를 ACTIVE 상태로 남겨 failure를 드러낸다.

반드시 다음을 구분해야 한다.

- **최초 비대칭 trigger:** right endpoint의 out-of-domain extra pair
- **직접적인 zero-row mechanism 후보:** constrained primal DOF와 결합하지 못하는
  active endpoint LM
- **미확정 사항:** extra pair가 node-level active-set aggregation을 실제로 어떻게
  바꾸는지, Kratos library behavior인지 application-level crosspoint susceptibility인지

### 바로 다음 작업

다음 단계는:

```text
Phase 4I-F: Search-pair causality and endpoint crosspoint audit
```

목표는 search 문제와 contact–bond crosspoint multiplier 문제를 분리하고,
물리적으로 타당하며 mesh-independent한 correction을 검증하는 것이다.

1.5 mm baseline이나 contact-location sweep으로 넘어가지 않는다.

---

## 1. 전체 이야기: Poisson's ratio부터 contact assembly까지

이 작업은 symmetric contact failure에서 시작하지 않았다. 최초 질문은 단순했다.

> 실리콘 pad의 nearly incompressible behavior를 위해 어떤 Poisson's ratio와
> element formulation을 실제 해석에 사용할 수 있는가?

그러나 realistic material value를 입력하는 것만으로는 validated model이 되지 않았다.
검증 범위가 순서대로 다음 층으로 확장됐다.

```text
material parameter
  -> volumetric locking
  -> mixed finite-strain formulation
  -> mesh convergence
  -> localized ALM contact
  -> active-set cycling
  -> actual fingertip geometry
  -> zero-clearance internal-contact topology
  -> left/right search asymmetry
  -> endpoint LM assembly and active-set lifecycle
```

핵심 교훈:

> FEM에서 현실적인 재료값을 넣었다는 사실은 현실적인 모델을 만들었다는 뜻이 아니다.
> Material, element formulation, mesh, contact search, active set, boundary topology,
> constraint space가 함께 일관돼야 한다.

---

## 2. Decision ledger

| 항목 | 판정 | 근거 |
|---|---|---|
| Mixed volumetric-strain T3 solid | **ADOPT** | `ν = 0.49` patch/mesh/sweep 통과 |
| `ν = 0.49` | **ADOPT** | 모든 benchmark mesh에서 finite하고 안정적 |
| `ν = 0.499` | **REJECT** | medium non-finite, fine iteration 및 `det(F)` 변동 증가 |
| General displacement Q4 at `ν = 0.49` | **REJECT** | volumetric locking |
| Q1P0 mixed Q4 candidate | **REJECT** | 실제 계산 실패 |
| 2D frictionless ALM mortar contact | **CONDITIONAL ADOPT** | localized benchmark medium/fine validation |
| Coarse pointwise pressure field | **REJECT** | 공간 해상도 부족, roughness gate 실패 |
| Phase 4M geometry/initialization | **PASS** | mesh/runtime contact contract 검증 |
| Phase 4I actual zero-clearance solve | **FAIL / BLOCKED** | first-step rank deficiency |
| Internal bottom-only contact | **PASS** | first step 1 Newton iteration |
| Internal left-only side contact | **PASS, but structurally suspect** | 12 iterations; endpoint LM deactivation에 의존 가능 |
| Internal right-only side contact | **FAIL** | 35 iterations |
| Right-side orientation hypothesis | **REJECT** | R00 physical; reversal은 zero normal |
| Three separate internal pairs | **REJECT 유지** | right endpoint blocker 유지 |
| Continuous-U recovery | **REJECT 유지** | duplicate는 없지만 blocker 유지 |
| Phase 4I baseline 재개 | **NO** | source-level correction 미검증 |
| 1.5 mm medium/fine baseline | **NOT RUN** | 0.25 mm gate 미통과 |
| Contact-location sweep | **NOT RUN** | actual geometry baseline 미통과 |

---

## 3. Phase chronology

각 Phase는 다음 네 항목으로 기록한다.

- **Hypothesis**
- **Controlled experiment**
- **Evidence**
- **Verdict**

### Phase 1 — 기능 분리 baseline

**Hypothesis**

Kratos build에서 2D hyperelastic solid와 2D frictionless ALM contact를 각각
독립적으로 실행할 수 있다.

**Controlled experiment**

- Hyperelastic solid baseline
- Frictionless ALM contact baseline
- 두 기능을 아직 실제 fingertip geometry에서 결합하지 않음

**Evidence**

- Kratos execution 정상
- Finite-strain displacement와 reaction 추출 정상
- Contact baseline 정상

**Verdict**

`PASS`

Phase 1은 두 기능의 독립 실행 가능성만 확인했다. 결합 안정성은 Phase 3의 질문이다.

---

### Phase 2A — General Q4와 Q1P0 후보

**Hypothesis**

Nearly incompressible silicone를 일반 displacement formulation 또는 Q1P0 mixed Q4로
안정적으로 표현할 수 있다.

**Controlled experiment**

| 구성 | 시험 |
|---|---|
| General Q4 | `ν = 0.45`, `0.49` |
| Q1P0 mixed Q4 | 실제 nonlinear run |

**Evidence**

- General Q4, `ν = 0.45`: 실행 가능
- General Q4, `ν = 0.49`: volumetric locking이 큼
- Q1P0 mixed Q4: 실제 계산 실패

**Verdict**

- General Q4 at `ν = 0.49`: `REJECT`
- Q1P0 mixed Q4: `REJECT`
- 다른 mixed formulation 필요

---

### Phase 2B — Mixed volumetric-strain T3

**Hypothesis**

`TotalLagrangianMixedVolumetricStrainElement2D3N`이 `ν = 0.49`에서 안정적인
nearly incompressible finite-strain baseline을 제공한다.

**Controlled experiment**

- Mixed T3 patch-style compression
- Coarse/medium/fine mesh
- `ν = 0.45`, `0.49`, `0.499` sweep
- General TL triangular control
- Field-level finite validation

**Runtime contract**

```text
Element:              TotalLagrangianMixedVolumetricStrainElement2D3N
Law:                  HyperElasticPlaneStrain2DLaw
Unknowns:             DISPLACEMENT_X, DISPLACEMENT_Y, VOLUMETRIC_STRAIN
Check-only 2D DOF:    DISPLACEMENT_Z
VOLUMETRIC_STRAIN_0:  0.0
Thickness:            1 mm
```

**Key evidence**

Patch result:

| 항목 | 결과 |
|---|---:|
| First 1% compression | PASS |
| Newton iterations | 3 |
| Reaction | `0.133525 N` |
| `det(F)` | `0.99960227` |
| Area ratio | `0.99960227` |
| `VOLUMETRIC_STRAIN` | `-3.97729e-4` |

`ν = 0.49` mesh result:

| Mesh | Elements | Final reaction | Mean `det(F)` | Newton |
|---|---:|---:|---:|---:|
| Coarse | 32 | `7.0576146 N` | `0.98055444` | 3–4 |
| Medium | 128 | `7.0576146 N` | `0.98055444` | 3–4 |
| Fine | 512 | `7.0576146 N` | `0.98055444` | 3–5 |

Poisson sweep:

| `ν` | Coarse | Medium | Fine | 판정 |
|---:|---:|---:|---:|---|
| 0.45 | `6.126894 N` | `6.126894 N` | `6.126894 N` | PASS |
| 0.49 | `7.057615 N` | `7.057615 N` | `7.057615 N` | ADOPT |
| 0.499 | `7.349610 N` | FAIL | `7.347945 N` | REJECT |

`ν = 0.499` medium은 30% step에서 displacement, reaction,
`VOLUMETRIC_STRAIN`이 non-finite가 됐다. Fine은 종료했지만 최대 Newton 11회,
`det(F) = 0.8553–1.1056`으로 신뢰 범위를 넓히지 못했다.

General TL control의 fine mesh도 solver return은 `True`였지만 field가 non-finite였다.
따라서 solver return만으로 PASS 처리하지 않는 규칙을 채택했다.

**Verdict**

- Mixed volumetric-strain T3: `PASS / ADOPT`
- `ν = 0.49`: `ADOPT`
- `ν = 0.499`: `REJECT`

---

### Phase 3 — Localized hyperelastic contact

**Hypothesis**

Phase 2B의 mixed solid와 Phase 1의 frictionless ALM contact를 localized
rounded-indenter benchmark에서 안정적으로 결합할 수 있다.

**Controlled experiment**

- 2D hyperelastic block
- Rounded rigid indenter carrier
- `ν = 0.49`
- Frictionless ALM mortar contact
- Coarse/medium/fine
- 48 displacement-controlled steps
- Default Kratos ALM/search values

**Evidence**

| Mesh | Result | Reached indentation | Reaction | Min `det(F)` | Max Newton |
|---|---|---:|---:|---:|---:|
| Coarse | PASS | `0.500 mm` | `0.522363 N` | `0.735147` | 3 |
| Medium | PASS | `0.500 mm` | `0.476738 N` | `0.709193` | 12 |
| Fine | FAIL | `0.164167 mm` | `0.119033 N` | `0.858335` | 35 |

Fine step 23에서 active set이 수렴하지 않았다. 실패 iterate도 finite였고
negative `det(F)`는 없었다. 실패는 material blow-up이나 element inversion이 아니라
ALM active-set cycling으로 분류했다.

**Verdict**

`FAIL`

---

### Phase 3R — Contact recovery와 acceptance 해석

**Hypothesis**

올바른 deformable/rigid contact role과 충분한 load-step resolution이 fine-mesh
active-set cycling을 제거할 수 있다.

**Controlled experiment**

- Runtime role audit
- `MASTER = rounded indenter`
- `SLAVE = deformable block top`
- 모든 mesh에 동일한 96 steps
- Geometry, material, element, ALM tolerance/penalty 유지
- Fine-only tuning 없음

**Evidence**

| Mesh | Solve | Final reaction | Min `det(F)` | Pressure roughness | Max Newton |
|---|---|---:|---:|---:|---:|
| Coarse | PASS | `0.396967 N` | `0.692145` | `0.743654` | 3 |
| Medium | PASS | `0.466330 N` | `0.767874` | `0.241555` | 3 |
| Fine | PASS | `0.500125 N` | `0.854401` | `0.092973` | 3 |

- 모든 mesh가 `0.5 mm` 도달
- 모든 field finite
- 모든 `det(F) > 0`
- Active-set cycling 제거
- Medium/fine reaction 차이 `6.757% < 10%`
- Medium 48/96-step 차이 `0.00497% < 1%`
- Pressure roughness는 refinement에 따라 `0.743654 -> 0.241555 -> 0.092973`

원래 gate는 모든 mesh에서 pressure roughness `< 0.5`를 요구했다.
Coarse만 이 기준을 실패했다.

**Verdict**

- Original Phase 3R gate: `FORMAL FAIL`
- Mixed solid: `ADOPT`
- 2D frictionless ALM contact: `CONDITIONAL ADOPT`
- Coarse pointwise pressure: `REJECT`
- Actual fingertip modeling: 당시 `PROCEED`

Formal FAIL은 solver failure가 아니라 overly strict all-mesh pressure gate의 결과다.

---

### Phase 4M — Actual geometry mesh와 initialization

**Hypothesis**

Shapely geometry를 단일 source of truth로 유지하면서 Gmsh T3 mesh와 Kratos
internal contact group을 올바르게 생성할 수 있다.

**Controlled experiment**

- Shapely pad/link polygons 직접 사용
- Gmsh 4.15.2 adapter
- Pad와 rigid stem의 topologically separate coincident nodes
- Three internal zero-clearance pairs
- Kratos initialization까지만 실행

**Evidence**

| Mesh | Nodes | T3 elements | Min angle | Pad area error |
|---|---:|---:|---:|---:|
| Medium | 8,391 | 16,164 | `35.906°` | `3.18e-15` |
| Fine | 18,961 | 36,964 | `38.828°` | `9.40e-15` |

- `StructuralMechanicsAnalysis.Initialize()`: PASS
- Strategy `Check()`: PASS
- Pad cutouts: runtime SLAVE
- Stem surfaces: runtime MASTER
- Contact flag, normal, positive finite `NODAL_H`: PASS

**Verdict**

`PASS`

Initialization PASS는 nonlinear indentation solve의 PASS를 뜻하지 않는다.

---

### Phase 4I — Actual geometry central indentation

**Hypothesis**

Phase 4M actual geometry에 external rounded indenter를 추가한 first nonlinear
Trial이 수렴한다.

**Controlled experiment**

```text
Mesh:          medium
Indentation:   0.25 mm
Steps:         48
First travel:  0.005208333333 mm
Internal:      left + right + bottom zero-clearance pairs
External:      rounded indenter
```

**Evidence**

- Preflight initialization: PASS
- Runtime four-pair mapping: PASS
- First linear system: Skyline `Error zero sum`
- Newton 35회에서 실패
- Failed iterate의 reaction/fields는 non-finite이므로 보고하지 않음
- U-clearance diagnostic에서 같은 external fixture와 mixed solid는 수렴

**Verdict**

- Phase 4I: `FAIL`
- Mixed solid: `ADOPT 유지`
- 1.5 mm medium/fine: `NOT RUN`
- Contact-location sweep: `BLOCKED`

---

### Phase 4I-D — Internal-contact topology isolation

**Hypothesis**

Failure가 external contact, 특정 internal side, opposing pairs, lower-corner duplicate
registration, 또는 pair topology 중 어디에서 발생하는지 격리할 수 있다.

**Controlled experiment**

Geometry, material, mesh, solver, ALM settings, first travel을 고정하고
internal-contact configuration만 바꿨다.

| Case | Internal contact |
|---|---|
| A | none |
| B | bottom only |
| C | left + right |
| C-left | left only |
| C-right | right only |
| D | left + right + bottom |
| E | continuous U |

**Evidence**

| Case | Result | Newton | LM DOF | Near-zero LM rows |
|---|---:|---:|---:|---:|
| A | PASS | 1 | 9 | 0 |
| B | PASS | 1 | 30 | 0 |
| C | FAIL | first factorization | 51 | 2 |
| C-left | PASS | 12 | 30 | 1 |
| C-right | FAIL | 35 | 30 | 1 |
| D | FAIL | 35 | 70 | 2 |
| E | FAIL | 35 | 70 | 2 |

Near-zero offender:

| Node | Reference coordinate | DOF | D row norm |
|---:|---:|---|---:|
| 2 | `(3.5, 0)` | `LAGRANGE_MULTIPLIER_CONTACT_PRESSURE` | `6.35e-17` |
| 5 | `(-3.5, 0)` | `LAGRANGE_MULTIPLIER_CONTACT_PRESSURE` | `1.69e-15` |

두 node는 lower U-corner가 아니라 side contact와 pad bond가 만나는 upper endpoint다.

Continuous-U는:

- root node/condition을 복제하지 않음
- lower corner process registration을 줄임
- duplicate connectivity/EquationId를 만들지 않음
- pair purity를 유지함
- 그러나 upper endpoint near-zero row와 failure를 제거하지 못함

**Verdict**

- Phase 4I-D: `FAIL`
- External contact hypothesis: `REJECT`
- Bottom contact hypothesis: `REJECT`
- Opposing sides required hypothesis: `REJECT`
- Lower-corner duplicate as primary cause: `REJECT`
- Continuous-U recovery: `REJECT`
- Right-only side contact: failure의 충분조건

---

### Phase 4I-E — Right-side mirror와 orientation audit

**Hypothesis**

Right-only failure는 right slave/master boundary condition ordering 또는 normal의
좌우 비대칭에서 발생한다.

**Controlled experiment**

Left-only를 numerical oracle로 사용하고 다음 mirror contract를 검사했다.

\[
(x,y)_L \leftrightarrow (-x,y)_R,\qquad
(n_x,n_y)_L \leftrightarrow (-n_x,n_y)_R
\]

Orientation matrix:

| Case | Right slave ordering | Right master ordering |
|---|---|---|
| R00 | current | current |
| R10 | reversed | current |
| R01 | current | reversed |
| R11 | reversed | reversed |

**Evidence**

- Source mesh node/condition mirror mapping: PASS
- Physical normal reflection: PASS
- R00 ordering: physically correct
- Left/right semantic membership: symmetric
- Left/right slave/master flags: symmetric
- Initial weighted gap: both `0.0`
- Post-search endpoint normals:
  - Left node 5: `(1, 0)`
  - Right node 2: `(-1, 0)`

Orientation result:

| Case | Physical ordering | Result |
|---|---|---|
| R00 | PASS | FAIL, Newton 35 |
| R10 | FAIL | Initialize zero-normal, node 174 |
| R01 | FAIL | Initialize zero-normal, node 461 |
| R11 | FAIL | Initialize zero-normal, node 174 |

최초 비대칭은 `after_contact_search`에서 나타났다.

| Endpoint | Generated pairing |
|---|---|
| Left node 5 | Correct master pairing 1개 |
| Right node 2 | Correct pairing 1개 + adjacent lower master에 대한 extra pairing 1개 |

Extra right pairing:

- Projection parameter: `2.0`
- Segment domain 밖
- Local LM row norm: `0.0`
- Free-column norm: `0.0`

Assembly contributor:

| Endpoint | Condition | Full local row norm | Free-column norm |
|---|---:|---:|---:|
| Left node 5 | 737 valid | `0.764997` | `1.7164e-15` |
| Right node 2 | 699 valid | `0.750396` | `6.3117e-17` |
| Right node 2 | 700 invalid | `0.0` | `0.0` |

Left solve는 endpoint를 이후 비활성화하고 Newton 12회에 수렴한다.
Right solve는 extra pair가 존재하며 endpoint가 ACTIVE로 남고 Newton 35회에서 실패한다.

Verification:

- Full pytest: `108 passed in 365.53 s`
- `compileall`: PASS
- `git diff --check`: PASS
- Audit JSON strict parse: 49 files PASS
- Orientation matrix CLI: intentional FAIL verdict, exit code 1

**Verdict**

- Phase 4I-E: `FAIL`
- Orientation hypothesis: `REJECT`
- Confirmed library-level root cause: `NOT YET`
- Production source fix: `NONE`
- First asymmetry: right endpoint search/pair generation
- Remaining scope: pair lifecycle + LM assembly/active-set crosspoint behavior

---

## 4. Current evidence model

### Observed facts

1. R00의 physical ordering과 normals는 올바르다.
2. Left/right upper endpoint는 모두 pad contact boundary와 pad bond가 만나는 crosspoint다.
3. Left valid endpoint LM condition도 Dirichlet/free-column 기준 coupling norm이
   `O(1e-15)`이다.
4. Right valid endpoint LM condition도 같은 의미에서 `O(1e-17)`이다.
5. Right invalid extra condition은 local row contribution이 정확히 0이다.
6. Left endpoint는 nonlinear iteration 중 inactive가 되고 solve가 수렴한다.
7. Right endpoint는 active로 남고 solve가 실패한다.

### Strong inference

Invalid extra pair는 zero row를 직접 더하는 것만으로 문제를 설명하기보다,
node-level gap/ACTIVE aggregation 또는 active-set lifecycle을 바꿔
free primal coupling이 없는 endpoint LM을 active로 남기는 trigger일 가능성이 크다.

### 아직 확인하지 못한 것

- Extra pair가 broad-phase candidate일 뿐인지 실제 active-set logic에 참여하는지
- Pair별 gap가 node `WEIGHTED_GAP`에 어떤 순서/규칙으로 누적되는지
- Invalid condition의 존재와 ACTIVE 상태 중 무엇이 failure를 유발하는지
- Valid endpoint LM row가 Dirichlet elimination 후 near-zero가 되는 정확한 topology
- Kratos가 contact–bond/Dirichlet crosspoint 처리를 공식 지원하는지
- Search traversal, condition insertion order, tie handling 중 무엇이 right-only
  extra candidate를 만드는지

---

## 5. Phase 4I-F plan

### Phase name

```text
Phase 4I-F: Search-pair causality and endpoint crosspoint audit
```

### Goal

Right upper endpoint의 out-of-domain extra pairing과 persistent ACTIVE state 사이의
인과관계를 확인하고, upper contact–bond crosspoint의 multiplier space가
구조적으로 유효한지 좌우 대칭으로 검증한다.

### Fixed conditions

변경하지 않는다.

- Geometry
- Medium mesh
- Material and `ν = 0.49`
- Mixed T3 formulation
- Zero clearance
- ALM penalty/search/tolerance settings
- Newton and linear solver settings
- Step size
- External rounded-indenter contact
- Physical line-condition ordering
- Pad SLAVE / stem MASTER roles

Orientation variants는 더 실행하지 않는다.

### 5.1 Kratos source/API trace

설치된 Kratos 10.3의 실제 source 또는 Python-exposed API에서 확인한다.

- Line2D2 local-coordinate valid domain
- Projection parameter calculation
- Broad-phase search candidate generation
- Exact overlap/projection acceptance
- `ComputingContact` condition creation
- Condition/node `ACTIVE` flag updates
- `WEIGHTED_GAP` reset and accumulation
- One slave node paired with multiple masters의 aggregation rule
- Inactive/zero-overlap condition이 node state에 영향을 주는지
- Contact boundary와 Dirichlet/bond boundary가 만나는 crosspoint 처리

확인한 source file, class, method를 `source_trace.json`에 기록한다.
존재하지 않는 API를 가정하지 않는다.

### 5.2 Iteration lifecycle audit

Left node 5와 right node 2를 다음 시점마다 snapshot한다.

1. Process creation 전
2. `ExecuteInitialize` 직후
3. Contact search 직후
4. `InitializeSolutionStep` 직후
5. 각 Newton `Predict/InitializeNonLinearIteration` 직후
6. Tangent assembly 직후
7. Active-set convergence check 직후
8. `FinalizeNonLinearIteration` 직후
9. Solve 종료 직후

각 snapshot:

- Node `ACTIVE/SLAVE/MASTER`
- LM pressure value, EquationId, fixity
- X/Y displacement EquationId, fixity
- Incident generated contact condition IDs
- Condition별 `ACTIVE`
- Master connectivity
- Projection local coordinate and point
- Local-domain validity
- Pair gap and weighted-gap contribution
- Accumulated node `WEIGHTED_GAP`
- Local/global LM row norm
- Free-column norm
- RHS entry
- 가능한 경우 active-set change reason

핵심 질문:

- Left endpoint는 정확히 어느 iteration에서 왜 inactive가 되는가?
- Right endpoint는 왜 active로 남는가?

### 5.3 Crosspoint structural audit

Node 2와 node 5에 대해:

- Contact boundary membership
- Pad bond boundary membership
- Dirichlet/tied/rigid membership
- X/Y displacement fixity
- 실제 free primal DOF 수
- Contact LM이 coupling하는 local displacement DOFs
- Dirichlet elimination 전 LM row
- Dirichlet elimination 후 LM row
- Valid pair만 남겼을 때 free-column norm
- LM diagonal/stabilization contribution

판정할 것:

- Endpoint LM은 active일 때 항상 zero/near-zero global row가 되는가?
- 이는 extra pair와 무관한 structural crosspoint deficiency인가?
- Left success는 endpoint LM deactivation에 의존하는가?

### 5.4 Diagnostic-only causal matrix

Kratos lifecycle상 안전한 API가 있을 때만 fresh model로 실행한다.

| Case | Right invalid pair | Invalid pair ACTIVE | Purpose |
|---|---|---:|---|
| F00 | keep | original | Baseline reproduction |
| F01 | exact-domain filter로 exclude | none | Pair removal effect |
| F02 | keep | force inactive | Pair existence vs active state |
| F03 | valid pair only | original active-set | Valid crosspoint LM test |

각 case에서:

- First-step convergence and Newton iterations
- Endpoint active history
- LM global/free-column row norm
- Generated/active pair count
- Physical valid contact 유지
- Reaction, finite fields, `det(F)`

Generated contact container를 불완전하게 조작하지 않는다. 안전한 API가 없으면
해당 case는 `UNAVAILABLE`로 기록한다.

### 5.5 Symmetric causality control

안전하게 가능할 때만:

- Left endpoint에 동등한 out-of-domain adjacent candidate를 diagnostic-only로 만들거나
- Condition insertion/ID ordering만 바꾸어 candidate set 변화를 확인한다.

Geometry, connectivity orientation, normals는 바꾸지 않는다.

목적:

- Search asymmetry가 geometry가 아니라 traversal/insertion/tie handling에 의존하는지 확인
- Left에 동일한 state가 생기면 같은 active LM failure가 발생하는지 확인

### 5.6 Production correction candidates

#### Candidate A — Pair acceptance correction

다음이 모두 확인될 때만 고려한다.

- Out-of-domain pair가 node active-set을 잘못 바꿈
- Exact-domain filter 후 right solve 수렴
- Physical valid pair는 ACTIVE 유지
- Endpoint LM row가 정상화
- 좌우 대칭과 mesh refinement independence 유지

Official Kratos setting/API를 먼저 사용한다. 설치 library 직접 patch는 마지막 선택이다.

#### Candidate B — Crosspoint multiplier treatment

다음이 확인될 때 검토한다.

- Endpoint primal displacement DOFs가 모두 constrained
- Active LM row가 Dirichlet elimination 후 항상 zero/near-zero
- Left success가 endpoint deactivation에 의존
- Search pair 정리 후에도 active crosspoint LM이 singular

먼저 official Kratos crosspoint/LM treatment를 확인한다.

아래는 근거 없이 production fix로 채택하지 않는다.

- Endpoint node 삭제
- LM pressure 강제 고정
- Contact condition 일부 절단
- Endpoint `ACTIVE` 강제 해제

### 5.7 Regression and Trial gate

Source-level correction이 검증된 경우에만 first-step regressions:

- A
- B
- C-left
- C-right
- C
- D
- E

필수:

- A/B/C-left 기존 PASS 유지
- C-right와 C PASS
- Invalid out-of-domain active pairing 없음
- Upper endpoint zero/free LM row 없음
- Physical valid internal contact ACTIVE
- Pair purity 유지
- Duplicate connectivity/EquationId 없음

D 또는 E가 first step을 통과한 경우에만:

```text
0.25 mm / 48-step full Trial
```

Full Trial acceptance 전까지 1.5 mm baseline은 실행하지 않는다.

### 5.8 Phase 4I-F PASS condition

모두 만족해야 한다.

- Invalid right extra pair의 source/lifecycle path 확인
- Extra pair와 persistent endpoint ACTIVE 사이의 인과관계 확인
- Valid endpoint LM free-column deficiency 원인 확인
- Search 문제와 crosspoint multiplier 문제 구분
- Physical, mesh-independent source-level correction 검증
- C-right와 C first step PASS
- D 또는 E의 0.25 mm / 48-step Trial PASS

원인을 격리했더라도 correction이 full Trial을 통과하지 못하면 Phase 4I-F는 `FAIL`이다.

---

## 6. Phase 4I-F output contract

새 artifact directory:

```text
output/phase4_search_crosspoint_audit/
    summary.json
    endpoint_lifecycle.csv
    search_pair_comparison.csv
    crosspoint_dof_map.csv
    source_trace.json

    f00_original/
    f01_invalid_removed/
    f02_invalid_inactive/
    f03_valid_only/
    symmetric_control/
    regressions/
    full_trials/
```

기존 Phase artifact를 덮어쓰지 않는다.

Tests:

- Projection local-domain validation
- Out-of-domain pair identification
- Valid endpoint pair preservation
- Invalid-pair filtering의 mesh/ID independence
- Endpoint lifecycle serialization
- Node-level pair aggregation diagnostics
- Dirichlet elimination 후 free-column LM row detection
- Contact–bond crosspoint identification
- Diagnostic variants가 production configuration을 바꾸지 않는지 확인
- Pair purity and duplicate-condition regression

---

## 7. 이번 디버깅에서 금지한 shortcut

다음은 원인 격리 없이 solve만 통과시키므로 사용하지 않는다.

- Penalty/scale factor/tolerance tuning
- Material parameter 변경
- Element formulation 변경
- Artificial clearance 추가
- Step 수 증가로 현재 first-step singularity 숨기기
- Friction 추가
- Right internal contact 비활성화
- Endpoint node 삭제
- Endpoint LM 강제 fix
- Normal의 근거 없는 manual override
- 잘못된 normal로 contact를 열어 가짜 PASS 만들기
- Failed iterate의 non-finite reaction/strain 보고
- Solver `True`만 보고 finite-field validation 생략

---

## 8. 실행 명령 기록

전체 tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 \
/home/dk/miniconda3/envs/lit/bin/python -B -m pytest -q
```

Phase 4I-D:

```bash
OMP_NUM_THREADS=1 \
/home/dk/miniconda3/envs/lit/bin/python -B \
-m analysis.phase4_internal_contact_diagnostic \
--mesh-level medium --cases A B C D E --first-step-only
```

Phase 4I-E:

```bash
OMP_NUM_THREADS=1 \
/home/dk/miniconda3/envs/lit/bin/python -B \
-m analysis.phase4_right_side_audit \
--mesh-level medium --run-orientation-matrix
```

Phase 4I-F 예정:

```bash
OMP_NUM_THREADS=1 \
/home/dk/miniconda3/envs/lit/bin/python -B \
-m analysis.phase4_search_crosspoint_audit \
--mesh-level medium --run-causal-matrix
```

Correction 검증 후에만:

```bash
OMP_NUM_THREADS=1 \
/home/dk/miniconda3/envs/lit/bin/python -B \
-m analysis.phase4_search_crosspoint_audit \
--mesh-level medium --run-regressions
```

First-step gate 이후에만:

```bash
OMP_NUM_THREADS=1 \
/home/dk/miniconda3/envs/lit/bin/python -B \
-m analysis.phase4_search_crosspoint_audit \
--mesh-level medium --run-full-trials \
--indentation-mm 0.25 --steps 48
```

---

## 9. Future update template

새 Phase나 diagnostic run을 마칠 때 이 형식으로 append한다.

```markdown
### Phase X — title

**Date**

YYYY-MM-DD

**Hypothesis**

검증하려는 한 문장 가설.

**Controlled experiment**

- 고정한 것
- 바꾼 것
- 실행 case

**Evidence**

- 수치 결과
- 최초 divergence stage
- artifact path
- test result

**Verdict**

- ADOPT / REJECT / PASS / FAIL / INCONCLUSIVE
- 확인된 사실
- 아직 미확정인 것

**Next gate**

다음 단계로 가기 전에 반드시 통과해야 하는 조건.
```

업데이트 규칙:

1. `PASS`와 `ADOPT`를 구분한다.
2. 이전 Phase의 실패 판정을 소급해서 지우지 않는다.
3. 확인된 사실과 working hypothesis를 분리한다.
4. 첫 비대칭 event와 직접적인 failure mechanism을 분리한다.
5. Failure artifact도 보존한다.
6. Production correction이 없으면 `NONE`이라고 명시한다.
7. Gate를 통과하지 않은 downstream baseline은 실행하지 않는다.

---

## 10. Blog/post framing 메모

이 기록은 추후 technical blog series의 재료가 될 수 있다.

가능한 큰 제목:

- **Poisson's Ratio Was Only the Beginning**
- **Validating Hyperelastic Contact: From Material Locking to Singular Multipliers**
- **How a Silicone Parameter Sweep Became a Contact-Solver Investigation**
- **실리콘의 Poisson's Ratio를 찾다가 Contact Solver까지 뜯어본 이야기**

중심 메시지:

> The first asymmetric event is not necessarily the direct cause of the singularity.

연재 구조 후보:

1. Nearly incompressible silicone와 locking
2. Mixed formulation을 실제로 검증하는 방법
3. Solver success가 field validity를 보장하지 않는 이유
4. Localized contact와 active-set cycling
5. Actual geometry의 zero-clearance contact
6. Symmetric model이 한쪽에서만 실패한 이유
7. Bad pair는 direct cause인가 trigger인가

---

## 11. 2026-07-23 — Phase 4I-F final verdict

**Hypothesis**

Right upper endpoint의 out-of-domain extra search pair가 first-step failure의
필요조건이다.

**Experiment**

- L00 left control
- F00 original right search result
- F02 same generated/search lifecycle with invalid condition flag만 inactive
- first-iteration condition 및 global LM row assembly

**Evidence**

- L00: PASS, 12 Newton iterations, endpoint는 최종 inactive.
- F00: FAIL, 35 iterations.
- F02: FAIL, 35 iterations, F00과 동일한 node-level ACTIVE sequence.
- Right valid-only free-column LM norm: `6.3117e-17`.
- Left valid-only free-column LM norm: `1.7164e-15`.
- Active LM diagonal: `0`; inactive formulation은 LM diagonal을 제공.
- Right extra pair: `xi = 3.0`, exact overlap `5.3013e-14 mm`.
- Invalid condition을 inactive로 두어도 수렴은 회복되지 않음.

**Verdict**

- Invalid-pair ACTIVE causal hypothesis: REJECT.
- Active contact–bond crosspoint deficiency: CONFIRMED.
- Phase 4I-F: FAIL.
- Production correction: NONE.

Artifacts: `output/phase4_search_crosspoint_audit/`.

---

## 12. 2026-07-23 — Phase 4I-G bounded treatment

**Hypothesis**

Supported Kratos crosspoint treatment 또는 topology/fixity 기반 application
multiplier restriction으로 physical contact를 바꾸지 않고 deficient LM
basis를 제거할 수 있다.

**Source experiment**

- Exact Kratos commit:
  `14ee273e97af403622699e797ea5fa356b1a7e60`.
- `AuxiliaryAddDofs`, frictionless ALM condition
  `EquationIdVector`/`GetDofList`, active/inactive algebra,
  block-builder Dirichlet elimination, contact-builder isolated-node handling
  추적.

**Source evidence**

- Kratos는 frictionless ALM pressure LM DOF를 root node 전체에 추가한다.
- Generated slave geometry의 모든 node가 displacement fixity와 무관하게
  pressure LM을 제공한다.
- Supported crosspoint omission, static condensation, boundary-trace
  restriction 또는 Python pre-DOF basis hook을 찾지 못했다.

**Minimal algebra experiment**

- Adopted mixed T3 solid와 flat rigid master.
- Mirrored left/right ACTIVE contact.
- Fully fixed contact endpoint와 adjacent free slave node.
- 2, 4, 8 divisions; case별 fresh `Kratos.Model`.

| Divisions | Left/right endpoint IDs | Left/right post-Dirichlet LM row norm |
|---:|---|---|
| 2 | 3 / 9 | `1.0e-4` / `1.0e-4` |
| 4 | 5 / 25 | `2.0e-4` / `2.0e-4` |
| 8 | 9 / 81 | `4.0e-4` / `4.0e-4` |

모든 contact는 ACTIVE, 모든 LM diagonal은 zero, adjacent X/Y displacement는
free였다. Global zero LM row는 없었고 mirror norm 차이는 최대
`4.44e-16`이었다.

**Verdict**

- G1 official treatment: UNAVAILABLE.
- G2 topology-only restriction: REJECTED AS PRODUCTION FIX.
  Endpoint fixity만으로 deficient basis라고 할 수 없고 unconditional
  exclusion은 healthy adjacent-trace coupling을 제거한다. Kratos에는 더
  좁은 restriction을 완전하게 구현할 safe hook도 없다.
- A/B/C-left/C-right/C/D/E: NOT RUN by correction gate.
- 0.25 mm × 48-step trial: NOT RUN by correction gate.
- Phase 4I-G: FAIL/BLOCKED.

**Fallback**

Physical, supported, mesh/ID-independent correction이 minimal gate를 통과하지
못했다. 따라서 internal contact 조사를 멈추고 Phase 4J를 자동 시작했다.

Artifacts: `output/phase4_crosspoint_multiplier_treatment/`.

---

## 13. 2026-07-23 — Phase 4J no-void external baseline

**Preflight**

- Existing `FingertipParameters()` defaults: zero clearance, no void geometry.
- Internal contact configuration: `none`.
- Runtime contact group: external `PadOuterArc` slave —
  `IndenterContactArc` master only.
- Internal contact process, indexed contact submodel part, generated condition
  없음.
- Adopted mixed T3, `HyperElasticPlaneStrain2DLaw`, `nu = 0.49`.
- External ALM add-DOF policy 때문에 root-wide LM DOF object는 남지만,
  generated condition `GetDofList` 기준 internal-exclusive LM assembly는
  0개.

**Instrumentation correction**

- External geometric penetration이 과거에는 outer-pad slave node 전체를
  local indenter에 투영해 false `12.5 mm` maximum을 만들었다.
- External penetration 범위를 ACTIVE slave node로 제한했다. Internal pair
  동작은 변경하지 않았다.
- CSV/plot output이 external-only contact-group dictionary를 지원하도록
  수정했다.

**Queue result**

| Case | Result | Final reaction | Min det(F) | Max Newton | Solve time |
|---|---:|---:|---:|---:|---:|
| J0 medium, 1 step | PASS | `0.0013569 N` | `0.98984` | 1 | `1.22 s` |
| J1 medium, 1.5 mm/48 | PASS | `0.861926 N` | `0.76282` | 3 | `114.73 s` |
| J2 fine, 1.5 mm/48 | PASS | `0.864680 N` | `0.69839` | 3 | `331.65 s` |
| J3 locations | SKIPPED | — | — | — | no discrete documented list |
| J4 parameters | SKIPPED | — | — | — | no documented candidate list |

Medium/fine final-reaction difference는 `0.319%`다. 두 full curve는 smooth,
monotonic이며 finite field, positive det(F), force equilibrium, penetration,
active-set convergence, volumetric checkerboard check를 모두 통과했다.

J1 numerical output은 정상 checkpoint됐지만 당시 실행 중이던 pre-fix plot
writer가 없는 internal-group key를 요구해 child exit code 1을 냈다. 누락
plot은 corrected writer로 checkpoint에서 재생성했고 48-step solve는
반복하지 않았다. J0 재실행과 J2는 corrected writer에서 exit code 0으로
완료됐다. 실패한 J0 writer attempt 두 개는 별도 directory에 보존했다.

`STRAIN_ENERGY`는 accepted runtime contract에서 unavailable이다. Stored
reaction-work integral medium `0.59817 N·mm`, fine `0.60065 N·mm`는 element
strain energy가 아니라 external-work proxy로 명시했다.

**Verdict**

Phase 4J: PASS.

Artifacts: `output/phase4_no_void_baseline/`.

---

## 14. Current decision ledger

| Item | Decision |
|---|---|
| Mixed volumetric-strain T3 | ADOPT |
| `HyperElasticPlaneStrain2DLaw`, `nu = 0.49` | ADOPT |
| `nu = 0.499` | REJECT |
| General TL nearly-incompressible default | REJECT |
| Q1P0 | REJECT |
| Current internal zero-clearance 2D ALM contact | BLOCKED |
| Search filtering alone | REJECT as complete fix |
| Topology-only fully-fixed endpoint LM exclusion | REJECT as production fix |
| No-void external-contact-only baseline | ADOPT / PASS |
| Contact-location sweep | SKIPPED until discrete cases are defined |

---

## 16. 2026-07-23 — Phase 4K CODTM

**Hypothesis**

검증된 Phase 4J no-void external-contact baseline에 solver 변경 없이
converged-step recorder를 추가하면 `(xi_cmd, delta_n)`에서 fixed material
sidewall signature와 contact descriptor로 가는 재현 가능한 nonlinear
CODTM을 만들 수 있다.

**Observation boundary**

- Semantic source: undeformed `PadOuterArc`.
- Contact coordinate: right top `xi=0`, crown `xi=.5`, left top `xi=1`.
- Right observation: `xi=0→.25`; left: `xi=1→.75`.
- 두 side 모두 `eta=0` bonded endpoint, `eta=1` crownward.
- 각 side 41개 reference-arc Line2 interpolation sample.
- Primary signal: `u(X0) dot n0`, outward positive.

**Contact variable audit**

Kratos `10.3.0-14ee273e` source와 runtime에서 historical pressure LM,
non-historical augmented pressure/NODAL_AREA, historical unit NORMAL을
확인했다. `CONTACT_FORCE`, `CONTACT_PRESSURE`, `NORMAL_CONTACT_STRESS`,
`CONTACT_NORMAL`은 현재 external slave node에 저장되지 않았다.
Augmented pressure/NODAL_AREA resultant를 global loading direction으로
투영하고 canonical indenter reaction과 2% closure를 검사했다.

**Center reproduction**

Medium/fine 모두 48/48 PASS. Final reaction과 minimum det(F)는 기존 J1/J2와
정확히 동일했다. 실제 target coordinate는 두 mesh 모두
`xi=0.5000000000000002`였다.

**Location queue**

모든 logical case는 48 step solve를 완료했다. Existing case gate 기준
medium `.20/.65/.80`, fine `.80`은 FAIL이지만 queue는 계속됐다. `.20`
medium은 step 1 force equilibrium `2.271%` 단일 초과이고, right-side
`.65/.80`은 contact distribution force closure가 불충분했다. Non-finite,
negative det(F), nonlinear failure는 없었다.

| Case | Reaction | Final centroid/length | Closure | min det(F) |
|---|---:|---|---:|---:|
| M `.20` | `0.132235` | `0.173066 / 2.34749` | `0.133%` | `0.927521` |
| M `.35` | `0.432058` | `0.328307 / 2.93384` | `0.0129%` | `0.873200` |
| M `.50` | `0.861926` | `0.500159 / 3.32329` | `0.508%` | `0.762817` |
| M `.65` | `0.422103` | UNVERIFIED | `4.327%` | `0.855006` |
| M `.80` | `0.130621` | UNVERIFIED | `2.286%` | `0.927267` |
| F `.20` | `0.133677` | `0.172961 / 2.13546` | `0.0317%` | `0.909885` |
| F `.50` | `0.864680` | `0.500039 / 3.42512` | `0.0675%` | `0.698391` |
| F `.80` | `0.129790` | UNVERIFIED | `4.702%` | `0.909893` |

**Mechanical map**

At `1.5 mm`, medium normalized gains for
`.20/.35/.50/.65/.80` are
`0.27158/0.33281/0.21803/0.33238/0.27149`. Fixed-indentation signature
distance spans `0.2654–0.9821 mm`. Common-force comparison uses first
loading-path crossing over `0.009019–0.130621 N`, without smoothing. The
mean-centered final signature singular values are
`5.5853, 2.4213, 0.6252, 0.3407, ~0 mm`; no optical noise criterion exists.

**Mesh**

M/F final profile relative L2 differences at `.20/.50/.80` are
`0.635%/0.255%/0.659%`; reaction differences are
`1.078%/0.319%/0.641%`. No predeclared CODTM-specific threshold exists, so
the status is `PROVISIONAL`, not an inferred PASS.

**Verdict / ledger**

| Item | Decision |
|---|---|
| Mixed T3 + `nu=.49` | ADOPT |
| Phase 4J no-void external baseline | ADOPT |
| Internal zero-clearance ALM | BLOCKED |
| CODTM extraction pipeline | PASS |
| Medium location map | PARTIAL |
| Fine spot checks | PARTIAL |
| CODTM mesh convergence | PROVISIONAL |
| Contact-distribution closure | PARTIAL |
| Mechanical separability | DESCRIPTIVE ONLY |

Artifacts: `output/phase4_mechanical_transfer_map/`.

**Exact next action**

Geometry optimization은 시작하지 않는다. Optical forward/noise model과
CODTM-specific mesh threshold를 먼저 선언하고, right-side pressure-resultant
closure의 extraction limitation을 별도 bounded audit 대상으로 결정한다.

## 17. Exact next action

Phase 4I internal ALM tuning을 재개하지 않는다. Adopted mixed solid과 Phase 4J
external baseline을 보존하고, 다음 bounded decision 중 하나를 선택한다.

1. Reviewed discrete external contact-location case list를 정의하고 Phase 4J
   medium baseline에서 실행한다.
2. Internal zero-clearance interface를 다시 연결하기 전에 documented
   contact–Dirichlet-crosspoint multiplier treatment가 있는 contact
   formulation/solver를 선택한다.

Location 값을 임의로 만들거나 이 결정 없이 internal contact를 다시
활성화하지 않는다.

## 18. Phase 4K-Viz — static CODTM visualization

**Scope**

Phase 4K NPZ/CSV/metadata를 post-process했다. Kratos solve, geometry
optimization, optical model, contact retuning은 실행하지 않았다.

**Data/coordinate audit**

- Metadata-declared axes로 arrays를 transpose하며 positional side/case order를
  가정하지 않는다.
- NPZ/CSV는 8 cases × 48 steps × 2 sides × 41 samples = 31,488 rows로
  일치했다.
- Valid displacement field 384/384는 finite다.
- Contact centroid/length verified mask는 256/384이며 displacement validity와
  분리했다.
- Canonical input 7개 SHA-256은 render 전후 동일하다.
- `(side, eta)`가 metric coordinate다. Signed zeta는 plot-only이고 중앙
  unsampled region은 분리·비적분 상태로 유지했다.

**Reproduced metrics**

| Metric | Result |
|---|---:|
| Raw distance at 1.5 mm, off-diagonal | `0.265411–0.982068 mm` |
| Shape distance at 1.5 mm, off-diagonal | `0.552584–1.968494` |
| Distance symmetry / diagonal error | `0 / 0` |
| Mirror relative L2 `.20/.80` | `0.0670%` |
| Mirror relative L2 `.35/.65` | `0.1498%` |
| Mirror relative L2 `.50/.50` | `0.0942%` |
| Stored/independent tangent gain max error | `0.0` |
| M/F profile relative L2 `.20/.50/.80` | `0.635% / 0.255% / 0.659%` |

**Visualization QA**

10 figures have 300 DPI PNG, vector PDF, and corresponding finite/schema-checked
CSV source data. The main atlas uses one zero-centered raw scale across four
indentations, discrete xi rows, and a hatched central gap. Physical deformation
uses equal aspect and 1× displacement. No two sidewall chains are connected.

```text
Phase 4K-Viz pipeline:          PASS
Mirror diagnostic:             CONSISTENT
M/F visualization:             PROVISIONAL
Publication readiness:         READY
Optical observability:         NOT EVALUATED
```

Artifacts: `output/phase4_codtm_visualization/`.
Documentation: `codtm-visualization.md`.

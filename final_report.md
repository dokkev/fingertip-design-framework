# LUMO Production BO 안정화 작업 최종 보고서

- 작성일: 2026-08-21
- 대상 저장소: `/home/dk/workspace/lit_ws`
- 브랜치: `feature/code-cleanup`
- 현재 HEAD: `46f857561b13f82a5a2bbabd3dc99615a2a09867`
- 주요 작업 구간: 2026-08-21 08:34–18:22 CDT, 약 10시간
- 최종 판정: **Production BO readiness BLOCKED**

## 1. Executive summary

이번 작업의 목표는 단순히 Ax를 호출하는 수준을 넘어, 장시간 production Bayesian optimization(BO)을 중단·재개할 수 있고, 잘못된 후보와 인프라 실패를 구분하며, 수치적으로 신뢰할 수 없는 결과가 최적화 데이터로 들어가지 않게 만드는 것이었다.

구현 측면에서는 상당한 진전이 있었다.

- Ax는 여섯 개 physical parameter를 독립적으로 뽑아 대부분 reject하는 대신, normalized latent 좌표에서 `D_geometry_feasible`을 직접 parameterize한다.
- nominal 실패, budget exhaustion, optimizer stall, zero-result가 성공 종료로 위장되지 않는다.
- Ax public JSON state를 포함하는 versioned atomic PRE/POST checkpoint와 explicit resume 경로가 있다.
- evaluator, Ax, registry, artifact가 canonical objective identity를 공유하고 runtime에서 scalar/identifier 일치를 확인한다.
- path-field clipping과 segment-budget termination은 유한한 objective 뒤에 숨지 않고 candidate-local optical failure가 된다.
- strict YAML, source provenance, dirty-worktree policy, registry/checkpoint locking, production budget 정책이 production runner에 연결되었다.
- production-path smoke, 36-state trajectory gate, 세 개 독립 process repeatability, bounded 6D BO는 각각 의미 있는 PASS 결과를 만들었다.

그러나 과학적 production GO는 얻지 못했다.

- 최종 nominal Newton production/reference 비교에서 18개 상태 중 1개가 기존 RMS displacement threshold를 실패했다.
- 해당 상태를 현재 HEAD의 clean detached worktree에서 더 세분화해 비교했지만, iteration과 timestep refinement에 대해 monotonic convergence가 나타나지 않았다.
- 더 엄격한 `480 iterations / 0.0000625 s / 0.003125 mm` reference와 비교해도 RMS 차이가 기준 `0.005 mm`보다 크게 남았다.
- 따라서 현 상태에서 BO가 반환하는 objective variation이나 best candidate를 물리적으로 신뢰할 근거가 부족하다.

결론적으로 **BO orchestration과 failure semantics는 크게 개선되었지만, Newton/contact mechanics의 discretization sensitivity가 해결되지 않아 production optimization을 승인할 수 없다.** 최종 clean-source production BO는 의도적으로 실행하지 않았다.

## 2. 보고 범위와 증거 해석

### 2.1 재구성 기준

이 보고서는 다음 자료를 교차 확인해 작성했다.

- 현재 [INSTRUCTION.md](INSTRUCTION.md)의 Phase A–E 구현 계약과 Production GO checklist
- 현재 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 및 [docs/COMMANDS.md](docs/COMMANDS.md)
- Git commit과 net diff
- `output/` 아래의 fresh `config.json`, `preflight.json`, `summary.json`, checkpoint 및 scientific artifact
- 작업 중 기록된 단계별 unit-test 결과
- 독립 read-only reviewer가 작업 중 전달한 finding과 그 후의 current-code 재검사

정확한 대화형 shell transcript 전체가 별도 로그 파일로 보존된 것은 아니다. 따라서 테스트 결과는 persisted JSON evidence와 대화 중 명시적으로 기록된 수치만 사용했고, 확인할 수 없는 명령을 실행했다고 추정하지 않았다.

### 2.2 Git 범위

BO 관련 net change는 `71bd19f` 이후 현재 HEAD까지 다음 규모다. `lumo2/` 실험 코드와 현재 사용자 소유 uncommitted diff는 집계에서 제외했다.

| 영역 | 파일 수 | 추가 | 삭제 |
| --- | ---: | ---: | ---: |
| `lumo/` | 25 | 1,934 | 95 |
| `validation/` | 7 | 1,970 | 126 |
| `tests/` | 21 | 2,395 | 24 |
| `scripts/` | 4 | 663 | 386 |
| `docs/` | 2 | 118 | 20 |
| `config/` | 1 | 16 | 6 |
| `pyproject.toml` | 1 | 1 | 0 |
| **합계** | **61** | **7,097** | **657** |

주요 commit은 다음과 같다.

| 시각 | Commit | 의미 |
| --- | --- | --- |
| 08:34 | `f1543de` | Phase A–E, YAML/runner, optical correctness, validation harness의 큰 통합 변경 |
| 10:23 | `7eb4f9d` | feasible parameterization 및 Newton diagnostics/contract 보강 |
| 14:58 | `f6b1d7d` | deterministic mechanics, runtime identity, repeatability 및 final validation gate 보강 |
| 15:04 | `46f8575` | `lumo2/` WIP commit; production `lumo/` BO 경로에는 추가 변경 없음 |

현재 worktree에는 별도의 사용자 소유 `lumo2/`/`validation2/` 변경이 남아 있다. 이번 보고서 작성은 이를 수정하거나 정리하지 않았다.

## 3. 구현 변경 상세

### 3.1 Phase A — feasible design space와 Ax campaign semantics

#### Feasible-by-construction parameterization

[lumo/optimization/design_space.py](lumo/optimization/design_space.py)는 현재 `feasible-morphology-v3` parameterization을 소유한다.

- Ax에 노출되는 변수는 `[0, 1]` 범위의 여섯 normalized latent 변수다.
  - `latent_cutout_width`
  - `latent_pad_depth`
  - `latent_pad_split`
  - `latent_stem_width_split`
  - `latent_cutout_depth`
  - `latent_stem_height_split`
- latent proposal은 pad depth/split, stem/void width split, cutout depth, stem/void height split을 순서대로 계산한다.
- fixed LED package와 global silicone wall 조건을 포함하는 빠른 geometry feasibility를 mapping 자체에 반영한다.
- decode 뒤에도 authoritative `validate_physical_parameters()`를 반드시 수행한다.
- evaluator와 registry에는 latent 값이 아닌 실제 `FingertipParameters`가 전달된다.
- latent 값과 decoded physical morphology는 generation audit, trial record, checkpoint에 함께 남는다.
- rare decode failure는 Ax trial을 abandoned 처리한 뒤 bounded resampling하며, evaluator budget을 소비하지 않는다.
- feasibility resampling 고갈은 `feasible_generation_exhausted`로 종료된다.

이 설계는 `D_geometry_feasible >= D_mesh_feasible >= D_mechanics_feasible >= D_objective_success` 경계를 유지한다. Mesh, Newton, OptiX 실패를 억지로 geometry parameterization 안에 숨기지는 않는다.

#### Termination reason과 scientific acceptance 분리

[lumo/optimization/adapters/ax.py](lumo/optimization/adapters/ax.py)에 closed termination contract가 추가·보강되었다.

- `requested_budget_reached`
- `nominal_failed`
- `proposal_budget_exhausted`
- `evaluation_budget_exhausted`
- `optimizer_stalled`
- `feasible_generation_exhausted`

`AxRunResult`는 다음 사실을 분리해 보고한다.

- nominal success 여부
- successful Sobol/initialization count
- successful MBM/search count
- generated success count
- 모든 Ax proposal 수
- 실제 evaluator invocation 수
- feasibility rejection 및 candidate failure 통계
- registry reuse count
- pending trial과 resume reconciliation 상태

[scripts/optimization/run_bo.py](scripts/optimization/run_bo.py)는 이 실행 사실을 다시 campaign acceptance 정책에 대입한다. `COMPLETE` 문자열 하나가 과학적 성공을 뜻하지 않는다.

- smoke PASS: nominal success + successful Sobol 1개 + requested budget 도달
- production PASS: nominal success + successful Sobol 최소 6개 + successful MBM 최소 1개 + requested budget 도달
- controlled scientific/incomplete failure: exit code `3`
- configuration, persistence, dependency, infrastructure failure: exit code `2`
- accepted campaign만 exit code `0`

nominal이 실패하면 Ax의 `get_next_trials()`를 호출하지 않는다. exact cached nominal success는 재사용할 수 있지만 failed placeholder와는 구분된다.

#### Objective identity 통일

[lumo/optimization/objectives.py](lumo/optimization/objectives.py)의 canonical objective는 다음 하나다.

```text
trajectory_separation_margin_fixed_depth_v1
```

다음 경계에서 identity와 scalar를 검증한다.

- evaluator public `objective_identifier`
- Ax experiment objective
- successful evaluation의 nested objective name
- top-level `objective_value`
- registry record
- evaluation/artifact fingerprint
- validation summary

identifier mismatch 또는 nested/top-level scalar mismatch는 과학 결과로 저장하지 않고 invariant failure로 중단한다. 과거 `contact_state_separation@1` 이름은 새 결과에 사용하지 않는다.

### 3.2 Phase B — atomic Ax checkpoint와 exact resume

[lumo/optimization/checkpoint.py](lumo/optimization/checkpoint.py)는 `lumo-ax-checkpoint-v1` schema를 사용한다.

한 checkpoint는 다음으로 구성된다.

```text
<campaign>/
  checkpoint.json
  checkpoints/
    000001/
      ax_client.json
      trials.json
      state.json
```

핵심 보장:

- Ax의 public `save_to_json_file()` / `load_from_json_file()`만 사용한다.
- `PRE_EVALUATION`은 proposal 생성 후 evaluator 호출 전에 저장한다.
- `POST_EVALUATION`은 registry outcome과 Ax complete/abandon 반영 후 저장한다.
- 임시 sibling directory에 세 파일을 완성하고 file/directory `fsync` 후 atomic rename한다.
- 마지막에 작은 current pointer를 `os.replace()`로 교체한다.
- partial temporary directory는 resume 후보가 되지 않는다.
- campaign별 non-blocking POSIX single-writer lock이 있다.
- pointer/state sequence, phase, required fields, Ax JSON 존재를 strict하게 검증한다.

Resume semantics:

- 기존 output 존재만으로 자동 resume하지 않고 `--resume`이 필요하다.
- persisted evaluator contract, objective, parameterization, bounds, protocol, numerical config, device, seed, budget, source policy, Ax version을 검증한다.
- `PRE_EVALUATION` pending candidate는 새 proposal보다 먼저 reconciliation한다.
- registry hit이면 평가를 반복하지 않고 동일 Ax trial에 replay한다.
- registry miss이면 동일 physical candidate를 다시 평가한다.
- decode 불가능한 restored pending trial은 abandoned 처리한다.
- 완료된 POST checkpoint에서 target이 이미 충족되었으면 추가 proposal을 만들지 않는다.
- 임의의 과거 immutable checkpoint directory를 선택하는 rollback은 지원하지 않으며 명시적으로 거부한다. 현재 pointer 또는 campaign root만 resume target이다.

### 3.3 Phase C — optical numerical correctness

#### Path-field clipping 계측

[lumo/ray_tracing/optical_mechanics/path_field.py](lumo/ray_tracing/optical_mechanics/path_field.py)에서 active sample을 한 번 분류하고 동일 mask로 다음 값을 계산한다.

- `processed_sample_count`
- `clipped_sample_count`
- `represented_weighted_path_length_mm`
- `clipped_weighted_path_length_mm`

represented와 clipped path length는 non-negative이며 전체 active weighted path length를 보존한다. 이 정보는 다음 경로로 전달된다.

```text
PathFieldAccumulator
  -> Transport3DResult
  -> unified optical artifact
  -> evaluator state/candidate diagnostics
  -> campaign summary
```

Unified optical artifact schema는 `unified-optix-transport-case-v7`, trajectory evaluation schema는 `lumo3d-trajectory-evaluation-v3`이다.

#### Numerical acceptance contract

[lumo/optimization/optical_contract.py](lumo/optimization/optical_contract.py)의 production hard rule은 다음과 같다.

- segment-budget termination count/weight가 0
- clipped sample count가 0
- clipped weighted path length가 0
- objective pathology가 false

유한한 objective가 계산되었더라도 위 조건을 위반하면 Ax objective로 전달되지 않는다. candidate/state summary에는 termination reason별 count, weight, launched-weight fraction과 clipping/energy evidence가 남는다.

#### Candidate-local exception boundary

- `Transport3DGeometryError`: topology, index, fixed cell, invariant 등 campaign-fatal base error
- `Transport3DCandidateGeometryError`: 특정 deformed candidate의 surface collapse 또는 envelope violation
- `CandidateOpticsError`: simulation boundary에서 candidate subtype만 번역한 오류

Evaluator는 `CandidateOpticsError`만 `optics_failure`로 기록한다. Base geometry error, serialization error, dependency/runtime error, trace invariant는 registry에 candidate outcome으로 넣지 않고 campaign을 abort한다.

Deformed silicone triangle collapse는 candidate-scoped source site에서 좁게 검출하지만, static topology/tag/shape 문제는 계속 fatal이다.

### 3.4 Phase D — production policy, provenance, YAML authority

#### Budget semantics

Budget은 다음 네 값으로 분리되었다.

- `initialization_success_target`
- `search_success_target`
- `maximum_evaluations`
- `maximum_proposals`

`maximum_evaluations`는 nominal을 포함한 실제 evaluator invocation을 센다. Registry replay와 feasibility rejection은 세지 않는다. `maximum_proposals`는 feasibility reject와 duplicate를 포함한 모든 Ax-generated proposal을 센다. Candidate failure가 Sobol/MBM success target을 줄이지 않는다.

Production direct API와 CLI 모두 successful Sobol target을 6 미만으로 낮출 수 없고, MBM target을 최소 1개 요구한다. Smoke만 Sobol 1개, MBM 0개의 축소 protocol을 허용한다.

#### Strict typed YAML

[lumo/config.py](lumo/config.py)가 [config/lumo_execution.yaml](config/lumo_execution.yaml)을 단 한 번 읽어 typed `LumoExecutionConfig`로 변환한다.

- schema version: `2`
- unknown/missing key 거부
- bool/int/real strict type 검증
- numeric string과 non-finite value 거부
- duplicate YAML mapping key 거부
- `cuda:<non-negative index>` device grammar를 config load 시점에 검증
- 오류에 source path와 nested key context 포함

YAML은 다음 numerical execution setting을 소유한다.

- CUDA device
- volume mesh tier/target/minimum quality
- Newton timestep, iteration, load increment, contact coefficient, deterministic mode
- first-contact settings
- optical transport bounds, rays, interactions, segment/grid setting, extrusion depth, epsilon

Raw mapping은 evaluator/simulation으로 전달되지 않는다. Resolved typed values가 evaluation contract fingerprint에 포함되고 YAML path/digest는 provenance로 남는다.

[scripts/optimization/run_bo_ideal.py](scripts/optimization/run_bo_ideal.py)는 더 이상 존재하지 않는 future facade sketch가 아니라, 동일한 `run_bo` engine을 호출하는 canonical human-facing frontend다.

#### Device와 geometry propagation

독립 review 중 발견된 authority gap을 수정했다.

- YAML `cuda:N`이 Newton/Warp뿐 아니라 OptiX runtime device ID까지 전달된다.
- YAML `extrusion_depth_mm`으로 실제 meshed `FingertipSolid`을 만들며, simulation 내부에서 기본 11 mm solid를 다시 만드는 경로를 제거했다.
- E3 persisted-state optical replay에도 동일 extrusion depth를 사용한다.
- `source_epsilon_mm`이 deformed state medium classification과 trace launch에 동일하게 전달된다.
- search/reference `VolumeMeshSettings`가 simulation/evaluator에 주입되고 contract identity를 바꾼다.

#### Source provenance와 mutable-state safety

Campaign config, preflight, every checkpoint, summary에는 다음을 기록한다.

- full Git commit SHA
- tracked dirty 여부와 binary diff SHA-256
- 관련 untracked path와 path+content SHA-256
- deterministic `source_id`
- YAML path/schema/digest
- CUDA/Newton/Warp/CuPy/OptiX runtime identity

Production 기본 정책:

- Git provenance unavailable이면 expensive work 전 fail-fast
- clean worktree 요구
- dirty source는 `--allow-dirty`로만 허용하고 exact diff/content hash 기록
- cross-source registry reuse는 explicit opt-in 필요
- output/registry/checkpoint처럼 실행 중 변하는 경로는 source hash에서 제외
- Git-tracked mutable registry, registry lock, checkpoint lock path 거부
- campaign checkpoint와 shared registry 모두 exclusive advisory lock 사용

External registry는 smoke/validation에서만 허용한다. Production external registry reuse는 evaluator reproducibility가 충분히 확립될 때까지 현재 코드에서 거부한다.

### 3.5 Phase E — scientific validation harness

#### Representative morphology set

[validation/optimization/representative_morphologies.py](validation/optimization/representative_morphologies.py)에 deterministic case catalog를 만들었다.

- nominal
- latent center
- wide cutout edge
- deep cutout edge
- minimum 5 mm wall 인접 feasible edge

각 case는 latent 값, decoded physical morphology, morphology fingerprint, global thickness measure를 기록하고 authoritative feasibility validation을 통과해야 한다.

#### Newton/mesh/optical convergence

[validation/optimization/lumo3d_scientific_convergence.py](validation/optimization/lumo3d_scientific_convergence.py)는 세 결정을 분리한다.

- `execution_status`
- `numerical_acceptance`
- `scientific_convergence`: `PASS`, `FAIL`, `INCONCLUSIVE`

Newton은 기존 threshold를 그대로 사용한다.

- RMS vertex difference `<= 0.005 mm`
- relative maximum displacement difference `<= 3%`

현재 production/reference contract:

| 역할 | VBD iterations | `dt_s` | max load increment |
| --- | ---: | ---: | ---: |
| production | 100 | 0.00025 s | 0.0125 mm |
| reference | 160 | 0.000125 s | 0.00625 mm |

두 설정은 50 mm/s prescribed indentation rate를 유지한다. 이 설정은 diagnostic selection evidence를 갖지만, full representative gate가 실패했으므로 여전히 provisional이다.

Mesh는 search 1.5 mm와 reference 1.0 mm를 비교한다. Reaction-force contract는 현재 solver artifact에 없으므로 force를 contact count나 0으로 꾸미지 않고 `unsupported/null`로 기록한다.

Optical sweep은 production baseline과 다음 family를 one-factor-at-a-time으로 비교한다.

- rays: 256 / 512 / 1024
- max interactions: 6 / 8 / 10
- segment budget: 4096 / 8192 / 16384
- path grid: 32×32×8 / 64×64×16 / 128×128×32

각 family의 production/reference role과 pair ID를 machine-readable하게 기록하며 objective, `D_inter`, `D_radius`의 signed/absolute/relative delta를 계산한다. Mesh/optical sensitivity threshold는 승인된 근거가 없으므로 성공적으로 측정했더라도 `INCONCLUSIVE`로 남긴다.

#### Repeatability와 bounded gate

새 [validation/optimization/lumo3d_repeatability.py](validation/optimization/lumo3d_repeatability.py)는 세 개의 fresh Python/Warp process에서 nominal 18-state 평가를 반복한다. Mechanics artifact digest, state fingerprint, optical field digest, scalar diagnostics, hexadecimal objective가 모두 같아야 PASS한다.

[validation/optimization/lumo6d_test_bo.py](validation/optimization/lumo6d_test_bo.py)는 다음을 gate한다.

- successful Sobol 6개
- successful MBM 4개 목표
- nonzero finite objective variation
- objective pathology 없음
- controlled failure/PASS_WITH_LIMITATION은 CLI exit code `3`
- infrastructure error는 exit code `2`

임의의 “meaningful variation magnitude” threshold는 만들지 않았다. Range가 nonzero인지는 자동 gate하고, magnitude의 과학적 충분성은 `INCONCLUSIVE`로 남긴다.

### 3.6 Newton pose diagnostics와 determinism 보정

Smoke/trajectory 실행 중 실제 solver failure처럼 보였던 두 pose-error failure를 추적한 결과, 원인은 solver가 아니라 진단식의 dtype/unit conversion이었다.

기존 계산은 actual/target을 각각 float32 millimetre로 바꾼 뒤 차를 구해 약 `2.13248e-6 mm`의 인공 오차를 만들 수 있었다. 수정 후에는 solver-native float32 metre 값을 float64로 승격해 먼저 차와 norm을 계산한 뒤 한 번만 `×1000`한다.

- 실제 submitted target과 solver output 차이
- ideal mm pose에서 float32 metre로 submit할 때의 quantization

두 값을 별도 diagnostic으로 분리했고, metric version을 mechanics fingerprint에 포함했다.

또한 Warp `SolverVBD`에 explicit `run_to_run` deterministic mode를 전달하고 runtime/evaluation identity에 포함했다. 동일 GPU architecture에서 반복 실행의 atomic ordering을 고정하지만, 이는 서로 다른 GPU/architecture 간 bitwise portability를 의미하지 않는다.

## 4. 테스트 변경과 확인 결과

### 4.1 추가된 regression coverage

기준 commit 이후 test diff에는 최소 73개의 새 `test_*` 함수가 추가되었다. 주요 보호 범위는 다음과 같다.

- latent boundary/center, encode/decode, fixed LED feasibility, bounded resampling
- failed search retry, proposal/evaluation cap accounting
- nominal fail-fast와 smoke/production exit semantics
- objective identifier/scalar mismatch rejection
- atomic checkpoint, PRE/POST resume, registry replay, corrupt/mismatched state
- completed POST resume의 extra proposal 방지
- infeasible restored pending trial abandon
- registry concurrency lock
- strict YAML complete load, wrong type, nonfinite, duplicate key, device grammar
- source provenance, untracked content hash, mutable output exclusion, dirty policy
- path-field clipping conservation과 all-clipped/nonfinite cases
- segment termination/clipping/pathology numerical acceptance
- optical artifact v7 round-trip과 termination reason fraction
- candidate-only deformed surface collapse와 fatal base error 구분
- mesh/extrusion/device/source epsilon propagation
- representative morphology uniqueness/feasibility
- Newton/mesh/optical comparison의 execution/acceptance/scientific status 분리
- optics failure 뒤 mechanics evidence 보존 및 replay
- repeatability 3-process exact signature
- trajectory 18-state completeness와 direct-path bit-exact equality
- bounded gate zero-variation rejection와 CLI exit code
- prescribed-pose dtype/unit regression과 real-deviation detection

### 4.2 작업 중 기록된 test 결과

다음은 단계별로 실제 보고된 결과다. 서로 다른 시점과 overlapping subset이므로 합산하면 안 된다.

| 단계 | 결과 |
| --- | --- |
| Phase A focused | `24 passed` |
| 당시 전체 unit | `312 passed, 7 warnings` |
| Phase B Ax/checkpoint/runner focused | `35 passed` |
| Phase C focused | `39 passed` |
| Phase C validation artifact focused | `3 passed` |
| 대상 module `py_compile` | PASS로 기록 |
| 단계별 `git diff --check` | PASS로 기록 |

중요한 제한:

- 위 숫자는 각 단계 당시 snapshot의 결과다.
- 이후 Phase D/E와 Newton changes까지 반영한 **현재 HEAD 전체 unit suite를 이 보고서 작성 시 재실행하지 않았다.**
- persisted test log가 없으므로 최종 clean-source test count로 재해석하지 않는다.
- GPU/OptiX/Newton scientific 실행 결과는 다음 절의 artifact를 별도 증거로 사용한다.

## 5. 실행 증거와 시간 순서

### 5.1 주요 timeline

| 시각 | 실행/관찰 | 결과 |
| --- | --- | --- |
| 08:36–08:37 | 첫 preflight + smoke | preflight PASS, generated candidate `invalid_design`, campaign FAIL |
| 08:44–08:51 | smoke v3/seed retry | nominal PASS, generated candidate mechanics failure, campaign FAIL |
| 08:59 | smoke v6 contract | nominal + Sobol 1 성공, PASS |
| 09:04 | nominal/probe trajectory | 12 Newton trajectories, 36 optical states, PASS |
| 09:43 | 첫 five-case convergence | overall FAIL |
| 09:47–10:05 | 첫 bounded 6D BO | 6 Sobol + 4 MBM, process PASS, MBM이 Sobol best를 이기지 못함 |
| 10:06–10:07 | 초기 production-path 시도 | nominal optics failure로 proposal 0개, FAIL |
| 10:12–10:41 | registry-assisted production-path 시도 | algorithmic PASS, 그러나 dirty/old contract/reuse evidence이므로 최종 GO로 무효 |
| 10:46–12:49 | reviewer finding 반영 후 smoke 반복 | 최종 reduced production-path smoke PASS |
| 13:17 | stable trajectory gate | nominal/probe 총 36 states PASS |
| 13:27–13:48 | 세 process repeatability | bit-exact PASS |
| 14:30 | stable nominal Newton convergence | 17/18 accepted, 1 state RMS threshold FAIL |
| 15:02–16:59 | clean `f6b1d7d` bounded 6D BO | 6 Sobol + 4 MBM, process PASS |
| 17:50–18:22 | current-HEAD targeted Newton refinement | 모든 핵심 refinement comparison FAIL |

### 5.2 OptiX/Newton preflight

최종 smoke preflight artifact: [output/optimization/bo_smoke_stable_final_20260821/preflight.json](output/optimization/bo_smoke_stable_final_20260821/preflight.json)

결과: `PASS`

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, compute capability 8.9
- device: `cuda:0`
- CUDA driver/runtime: 13.1 / 13.2 계열 metadata
- CuPy: 14.1.1
- Newton: 1.4.0
- Warp: 1.16.0
- OptiX runtime: 9.1.0
- OptiX binding: 0.1.0
- Gmsh import: PASS
- Newton finite tetra smoke: PASS
- OptiX real GAS/SBT launch: ray 2개 중 hit 1, miss 1, PASS

이 preflight는 dependency import만 확인한 것이 아니라 production OptiX runtime으로 module/GAS/SBT를 만들고 실제 ray launch/copy-back을 검증했다.

### 5.3 Production-path smoke

최종 artifact: [output/optimization/bo_smoke_stable_final_20260821/summary.json](output/optimization/bo_smoke_stable_final_20260821/summary.json)

결과: `PASS`

- Ax termination: `requested_budget_reached`
- nominal success: true
- successful Sobol: 1
- successful MBM: 0, smoke contract상 정상
- generated proposal: 1
- feasibility rejection: 0
- candidate failure: 0
- objective: `trajectory_separation_margin_fixed_depth_v1`
- best reduced-protocol objective: `0.08109032269301196`
- clipping: 0
- segment-budget termination: 0

주의: smoke objective는 2-state reduced protocol이므로 full 18-state objective와 직접 비교하면 안 된다. 또한 source는 `7eb4f9d + hashed dirty diff`로 기록되었으며 clean-source 최종 증거는 아니다.

### 5.4 36-state trajectory validation

Artifact: [output/validation/optimization/lumo3d_trajectory_stable_final_20260821/summary.json](output/validation/optimization/lumo3d_trajectory_stable_final_20260821/summary.json)

결과: `PASS`

- morphology: nominal, probe
- morphology당 6 Newton trajectories × 3 checkpoints = 18 states
- 총 Newton trajectories: 12
- 총 FULL_3D optical states: 36
- duplicate/missing state: 0
- direct production-path equivalence: PASS
- domain check: PASS
- objective pathology: 없음
- nominal objective: `0.023811831827472207`
- probe objective: `0.02332593694551334`

두 morphology 모두 clipping/segment-budget hard failure 없이 complete 18-state objective를 만들었다.

### 5.5 Three-process repeatability

Artifact: [output/validation/optimization/lumo3d_repeatability_stable_final_20260821/summary.json](output/validation/optimization/lumo3d_repeatability_stable_final_20260821/summary.json)

결과: `PASS`

- worker process: 3
- 모든 process exit 0
- 각 run 18-state complete
- unique signature count: 1
- signature: `6393014505a1ea3aa67a4fe729f4118b555779e4dfbdad6ca93554a284a082c7`
- mechanics/optical/scalar/objective: bit-exact

이 결과는 동일 source/config/GPU에서 run-to-run reproducibility가 확보되었음을 뜻한다. 물리적 convergence나 다른 GPU 간 reproducibility를 뜻하지는 않는다.

### 5.6 Representative scientific convergence

#### 첫 full five-case sweep

Artifact: [output/validation/readiness_meta_convergence_PZIRRt/summary.json](output/validation/readiness_meta_convergence_PZIRRt/summary.json)

결과: `FAIL`, elapsed `2684.158 s` 약 44분 44초

- nominal, latent center, wide cutout, deep cutout의 Newton comparison: 모두 FAIL
- minimum-wall edge: mesh failure로 mechanics checkpoint incomplete
- mesh: 네 case 측정 완료이나 threshold 부재로 INCONCLUSIVE, minimum-wall FAIL
- optics rows: 45
  - execution completed: 36
  - candidate failure: 1
  - mechanics baseline incomplete: 8
  - numerical PASS: 35
  - numerical FAIL: 1
  - NOT_RUN: 9
- ray count와 path-field grid 변화에 objective가 크게 민감했지만 승인 threshold가 없어 scientific convergence는 INCONCLUSIVE로 보존

이 sweep은 후속 source 변경 전 diagnostic이다. 결과를 숨기지 않고 failure/harness 개선 근거로 사용했지만 최종 GO 증거로는 사용하지 않는다.

#### 최종 stable nominal Newton gate

Artifact: [output/validation/optimization/lumo3d_scientific_convergence_stable_final_20260821/newton/summary.json](output/validation/optimization/lumo3d_scientific_convergence_stable_final_20260821/newton/summary.json)

결과: `FAIL`

- production/reference 모두 18 mechanics checkpoint complete
- 17/18 state accepted
- 실패 상태: `u=0.5`, radius `5 mm`, depth `1.5 mm`
- RMS vertex difference: `0.015016930676038922 mm`
- RMS threshold: `0.005 mm`
- relative max displacement difference: `0.0059522749523409565`, 약 0.595%, 3% 기준은 PASS
- max displacement-field difference: `0.13005700736428466 mm`
- objective absolute delta: `0.000716154797056015`, 약 2.92%

즉 overall FAIL의 직접 원인은 relative metric이 아니라 RMS displacement field difference다. Mechanics execution 자체와 optics/objective downstream status는 모두 완료되었으므로 infrastructure failure나 optical failure로 오분류할 수 없다.

### 5.7 Current-HEAD targeted Newton refinement

Artifacts:

- [output/validation/diagnostics_20260821/nominal_contact_refinement_canonical_head46f8575/summary.json](output/validation/diagnostics_20260821/nominal_contact_refinement_canonical_head46f8575/summary.json)
- [output/validation/diagnostics_20260821/nominal_contact_refinement_canonical_head46f8575/summary_with_scaled_iterations.json](output/validation/diagnostics_20260821/nominal_contact_refinement_canonical_head46f8575/summary_with_scaled_iterations.json)

이 진단은 current HEAD `46f8575`를 고정한 clean detached worktree에서 canonical production nominal morphology를 사용했다. 조건은 실패한 상태와 동일한 `u=0.5`, radius `5 mm`, depth `1.5 mm`, nominal `void_height=0.25 mm`다.

| Candidate | Reference | RMS diff (mm) | Relative max | 판정 |
| --- | --- | ---: | ---: | --- |
| 160 iter, 0.000125 s, 0.00625 mm | 240 iter, same dt/increment | 0.0161361 | 0.7226% | FAIL: RMS |
| 240 iter, 0.000125 s, 0.00625 mm | 240 iter, half dt/increment | 0.0361979 | 4.0245% | FAIL: both |
| 160 iter, 0.000125 s, 0.00625 mm | 240 iter, half dt/increment | 0.0217314 | 3.3310% | FAIL: both |
| 240 iter, 0.000125 s, 0.00625 mm | 480 iter, half dt/increment | 0.0318764 | 3.4853% | FAIL: both |
| 160 iter, 0.000125 s, 0.00625 mm | 480 iter, half dt/increment | 0.0207525 | 2.7879% | FAIL: RMS |
| 240 iter, 0.0000625 s, 0.003125 mm | 480 iter, same dt/increment | 0.0128736 | 0.5618% | FAIL: RMS |

모든 run에서 carrier contact vertex count는 동일하게 65였고 동일 canonical morphology/condition을 사용했다. `iterations × dt`를 맞춘 240-vs-480 비교도 실패했으므로 단순히 iteration count를 timestep에 비례시켜 늘리면 해결된다는 가설은 지지되지 않았다.

이 결과는 reporting bug가 아니라 실제 solver/contact trajectory의 time/iteration sensitivity가 남아 있음을 보여준다.

### 5.8 Bounded 6D BO

최종 clean-source process evidence: [output/validation/optimization/lumo6d_test_bo_f6b1d7d_process_only_20260821/summary.json](output/validation/optimization/lumo6d_test_bo_f6b1d7d_process_only_20260821/summary.json)

결과: `PASS`, 단 **process-only evidence**

- source commit: clean `f6b1d7d`
- Ax termination: `requested_budget_reached`
- successful Sobol: 6
- successful MBM: 4
- successful observations: 10
- attempted generated trials: 12
- mesh failure: 2
- mechanics failure: 0
- optics failure: 0
- feasibility rejection: 0
- registry reuse: 1
- objective range: `[0.0003184673336473487, 0.06903762651756257]`
- nonzero span: `0.06871915918391522`
- best Sobol: `0.03177270653004637`
- best MBM: `0.06903762651756257`
- MBM beat Sobol: true
- nominal: `0.023811831827472207`
- best-vs-nominal delta: `0.04522579469009036`
- total runtime: `6994.957 s`, 약 1시간 56분 35초

Best physical morphology:

| Parameter | Value (mm) |
| --- | ---: |
| flat pad height | 5.368800108964293 |
| semielliptical pad height | 24.631199891035706 |
| stem height | 8.083412830286528 |
| stem width | 4.079987724779153 |
| void height | 6.285707012417037 |
| void width | 0.9224751181366817 |

이 결과는 Ax가 6D Sobol→MBM 전이를 수행하고, candidate failure를 견디며, nonzero objective variation에서 MBM candidate를 생성할 수 있음을 증명한다. 그러나 mechanics convergence가 실패했으므로 `0.0690376`을 물리적으로 유효한 optimum 또는 production recommendation으로 해석하면 안 된다.

### 5.9 Production BO 시도와 최종 처리

초기 source `f1543de + dirty diff`에서 두 production-path 실행이 있었다.

1. [production_bo/summary.json](output/validation/readiness_20260821_f1543de/production_bo/summary.json)
   - nominal `optics_failure`
   - Ax proposal 0
   - `nominal_failed`
   - campaign FAIL

2. [production_bo_registry/summary.json](output/validation/readiness_20260821_f1543de/production_bo_registry/summary.json)
   - nominal success
   - Sobol success 6
   - MBM success 1
   - mesh failure 5, optics failure 2
   - reused evaluation 7
   - algorithmic campaign status PASS

두 번째 실행은 runner의 budget/MBM process가 동작함을 보여주는 진단 자료다. 하지만 다음 이유로 최종 production GO evidence가 아니다.

- dirty source snapshot
- 후속 parameterization, deterministic mechanics, runtime identity, pose diagnostic 및 validation contract 변경 전 결과
- external registry reuse를 포함
- current production policy는 external registry를 금지
- 이후 Newton convergence가 실패

따라서 **현재 최종 계약과 clean source에서 승인된 production BO는 실행하지 않았다.** Bounded 6D process gate 뒤에 scientific blocker를 확인했으므로 더 큰 production budget을 소비하지 않았다.

## 6. Independent review에서 발견·수정한 주요 문제

작업 중 별도의 read-only reviewer를 반복 사용했다. 다음 finding을 구현에 반영했다.

- completed POST checkpoint resume가 extra Ax proposal을 생성하던 문제
- PRE checkpoint의 infeasible pending trial을 abandoned하지 않던 문제
- bounded gate가 FAIL/PASS_WITH_LIMITATION에도 exit 0을 반환하던 문제
- YAML numeric string coercion 및 duplicate key silent overwrite
- YAML device가 Newton에는 가지만 OptiX device에는 전달되지 않던 문제
- non-11 mm extrusion depth가 mesh/simulation/replay에서 서로 다르게 적용되던 문제
- source epsilon이 medium classification과 launch에서 달랐던 문제
- mutable output/registry가 source provenance를 self-invalidate하던 문제
- shared external registry의 last-writer-wins data loss 가능성
- arbitrary immutable checkpoint directory resume가 실제로 latest pointer를 읽던 문제
- validation provenance unavailable 상태에서도 expensive work를 시작하던 문제
- Newton/mesh comparison이 downstream optics failure를 mechanics failure로 오분류하던 문제
- E3 replay가 candidate geometry exception으로 전체 harness를 abort하던 문제
- termination reason별 launched-weight fraction 누락
- E3 production/reference role 및 sensitivity delta 누락
- baseline numerical failure의 raw diagnostic objective를 replay sensitivity에 사용하지 못하던 문제
- objective pathology failure artifact에서 mesh evidence가 사라지던 문제
- deformed surface collapse가 candidate-local failure로 좁혀지지 않던 문제
- bounded gate가 zero objective variation을 PASS할 수 있던 문제
- production direct API/CLI가 Sobol target 6 미만을 허용하던 문제

사용자가 중단을 요청한 시점에 formal final reviewer report는 완료 전이어서 reviewer를 interrupt했다. 마지막 current-code 재검사에서는 새 BLOCKER가 보이지 않는다는 의견과 함께, Git-tracked lock-file path에 대한 provenance hardening edge가 NON-BLOCKING으로 남아 있었다. 이후 current runner에는 registry lock과 checkpoint lock path의 tracked-mutable 거부가 반영되어 있다.

Independent review는 구현 정확성 검토이며, Newton scientific convergence PASS를 대신하지 않는다.

## 7. 최종 readiness 판정

### 7.1 구현/프로세스 측면

| 항목 | 판정 | 근거 |
| --- | --- | --- |
| Feasible latent proposal | 구현 완료 | parameterization v3, bounded resampling/audit |
| Nominal fail-fast/exit semantics | 구현 완료 | failure smoke와 production nominal failure artifact |
| Ax checkpoint/resume | 구현 완료, unit-level 검증 | public Ax JSON, atomic pointer, reconciliation tests |
| Objective identity | 구현 완료 | canonical v1 identifier와 runtime validation |
| Optical clipping/termination guard | 구현 완료 | artifact v7, zero hard rule, smoke/trajectory evidence |
| YAML authority | 구현 완료 | strict schema v2, typed propagation |
| Provenance/locking | 구현 완료 | source ID, dirty policy, campaign/registry locks |
| Production-path smoke | PASS | nominal + Sobol 1 |
| 36-state trajectory | PASS | nominal/probe 36 states |
| Run-to-run repeatability | PASS | 3 processes, one exact signature |
| Bounded 6D BO process | PASS | 6 Sobol + 4 MBM |

### 7.2 과학적 측면

| 항목 | 판정 | 이유 |
| --- | --- | --- |
| Newton convergence | **FAIL / BLOCKER** | RMS displacement threshold failure와 non-monotonic refinement |
| Mesh convergence | INCONCLUSIVE/부분 FAIL | threshold 없음, minimum-wall case mesh failure, force unsupported |
| Optical convergence | INCONCLUSIVE | ray/grid sensitivity가 크며 승인 threshold 없음 |
| Meaningful objective variation | 측정됨, magnitude INCONCLUSIVE | nonzero variation은 있으나 mechanics가 미수렴 |
| Production optimum 신뢰성 | **미확보** | objective가 solver/discretization sensitivity를 포함할 수 있음 |
| Production BO GO | **BLOCKED** | Section 13 scientific evidence 조건 미충족 |

## 8. 남은 핵심 문제와 재개 시 최소 순서

### P0 — Newton/contact mechanics convergence contract 확립

BO 코드보다 먼저 해결해야 한다.

1. 실패 nominal state를 최소 재현 case로 유지한다.
2. load increment, timestep, VBD iteration을 하나씩 분리해 sweep한다.
3. 단순 `iterations × dt` scaling이 아닌 solver residual/constraint convergence를 측정한다.
4. prescribed-indentation reaction force를 solver-owned contract로 정의한다.
5. contact active set, penetration, strain/energy, displacement field의 안정성을 함께 본다.
6. nominal 한 점이 아니라 representative morphologies에서 동일 production/reference 설정을 검증한다.
7. 기존 `0.005 mm`, `3%` threshold를 결과에 맞춰 완화하지 않는다.

현재 VBD path가 이 quasistatic contact 문제에 적합한지, 혹은 수렴 가능한 다른 solve/integration contract가 필요한지도 별도로 판단해야 한다.

### P1 — Mesh/optical sensitivity의 scientific threshold 승인

- minimum-wall morphology의 mesh failure가 expected domain boundary인지 mesher defect인지 분류
- reaction-force metric이 정의된 뒤 1.5 mm vs 1.0 mm mesh sensitivity 재평가
- ray count와 grid refinement가 objective definition/normalization에 미치는 영향 분석
- objective, `D_inter`, `D_radius`, native field/energy diagnostics를 근거로 threshold 제안
- 결과를 보고 threshold를 사후 조정하지 말고 별도 review/승인

### P2 — 현재 source의 전체 evidence 재생성

Mechanics blocker를 해결한 뒤에만 다음 순서로 fresh output을 생성해야 한다.

1. current clean HEAD focused/full unit tests
2. OptiX/Newton preflight
3. production-path smoke
4. nominal/probe 36-state trajectory validation
5. three-process repeatability
6. five-case Newton/mesh/optical scientific convergence
7. bounded 6D BO 6 Sobol + 4 MBM
8. 모든 gate PASS 후에만 production BO budget 승인

과거 PASS artifact는 회귀 진단에는 유용하지만 source/config가 달라지면 GO 증거로 재사용하지 않는다.

## 9. 보존된 주요 artifact

| Artifact | 크기 | 용도 |
| --- | ---: | --- |
| `output/optimization/bo_smoke_stable_final_20260821` | 2.2 MB | reduced production-path smoke |
| `output/validation/optimization/lumo3d_trajectory_stable_final_20260821` | 14 MB | nominal/probe 36-state trajectory |
| `output/validation/optimization/lumo3d_repeatability_stable_final_20260821` | 16 MB | three-process exact repeatability |
| `output/validation/optimization/lumo3d_scientific_convergence_stable_final_20260821` | 10 MB | final nominal Newton blocker |
| `output/validation/optimization/lumo6d_test_bo_f6b1d7d_process_only_20260821` | 89 MB | clean-source bounded 6D process evidence |
| `output/validation/diagnostics_20260821/nominal_contact_refinement_canonical_head46f8575` | 988 KB | current-HEAD targeted refinement |
| `output/validation/readiness_20260821_f1543de/production_bo_registry` | 156 MB | early algorithmic production-path diagnostic |

Generated output은 삭제하거나 기존 결과 위에 덮어쓰지 않았다. 마지막 진단에 사용한 temporary detached worktree는 작업 중단 시 clean 상태를 확인한 뒤 제거했다.

## 10. 변경하지 않았거나 주장하지 않는 것

- Objective formula와 canonical identity를 PASS를 위해 바꾸지 않았다.
- Newton convergence threshold `0.005 mm` / `3%`를 완화하지 않았다.
- Failed/inconclusive scientific result를 PASS로 바꾸지 않았다.
- Contact count를 reaction force로 가장하지 않았다.
- Finite objective만으로 clipping/termination/pathology를 승인하지 않았다.
- Registry를 Ax generation state의 대체물로 사용하지 않았다.
- Current clean source에서 production BO를 성공했다고 주장하지 않는다.
- Bounded BO best candidate를 최종 물리 설계로 추천하지 않는다.
- 현재 사용자 소유 `lumo2/`와 `validation2/` 변경을 되돌리거나 흡수하지 않았다.

## 11. 최종 결론

이번 작업으로 “실패해도 exit 0인 BO script”는 “budget, source, checkpoint, numerical acceptance, candidate/infrastructure failure를 구분하는 campaign engine”에 가까워졌다. Ax/Sobol/MBM flow와 OptiX/Newton execution도 실제 GPU에서 end-to-end로 동작했고, bounded 6D BO는 clean source에서 requested budget을 달성했다.

하지만 production optimization의 핵심 전제인 mechanics convergence를 확보하지 못했다. 동일 nominal contact state가 iteration/timestep refinement에 안정적으로 수렴하지 않으므로, 현재 objective landscape에는 설계 효과와 solver discretization 효과가 섞여 있을 가능성이 높다. 이 상태에서 더 큰 BO budget을 쓰면 계산은 진행되더라도 최적화 결과의 과학적 의미를 방어하기 어렵다.

따라서 최종 판정은 다음과 같다.

```text
Implementation/process readiness: substantial progress, bounded process demonstrated
Scientific production readiness: BLOCKED
Production BO authorization: NO-GO
Primary blocker: unresolved Newton/contact mechanics convergence
```

재개한다면 더 많은 BO infrastructure를 추가하는 것이 아니라, 실패한 nominal state를 기준으로 mechanics/contact solver contract를 독립적으로 검증하거나 교체하는 것이 가장 짧은 경로다.

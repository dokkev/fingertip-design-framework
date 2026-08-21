

BLOCKER
1. 현재 6D search space에서 유효 후보 생성이 사실상 실패했습니다.
기존 bounded Test BO 결과는 명시적으로 FAIL입니다. nominal만 성공했고, BO 제안 10개 중 geometry reject 9개, mechanics fail 1개, Sobol/MBM 성공은 0개였습니다. 따라서 Ax 모델 업데이트와 objective 비교가 한 번도 발생하지 않았습니다. [reviewer_audit.md (line 8)](/home/dk/workspace/lit_ws/output/validation/optimization/lumo6d_test_bo/reviewer_audit.md:8)
원인은 Ax가 넓은 box bounds와 두 개의 선형 제약만 인식하고, 핵심 silicone-thickness 같은 비선형 제약은 proposal 이후에 reject하기 때문입니다.
- 6D bounds: [design_space.py (line 108)](/home/dk/workspace/lit_ws/lumo/optimization/design_space.py:108)
- Ax 선형 제약: [design_space.py (line 117)](/home/dk/workspace/lit_ws/lumo/optimization/design_space.py:117)
- 사후 비선형 thickness 검증: [design_space.py (line 192)](/home/dk/workspace/lit_ws/lumo/optimization/design_space.py:192)
- production도 동일한 bounds 사용: [run_bo.py (line 123)](/home/dk/workspace/lit_ws/scripts/optimization/run_bo.py:123)
현재 상태로 큰 BO를 시작하면 대부분의 계산 예산을 invalid proposal에 소비할 가능성이 높습니다. 과학적 bounds를 임의로 좁혀 PASS시켜서는 안 되며, 승인된 feasible parameterization, feasibility-aware generation 또는 근거 있는 bounds 변경이 먼저 필요합니다.

2. 캠페인이 과학적으로 실패해도 프로세스가 성공 코드로 종료됩니다.
Ax는 proposal_budget_exhausted나 optimizer stall을 정상 AxRunResult로 반환하지만, production runner는 결과 상태와 무관하게 항상 exit code 0을 반환합니다.
- non-complete 상태 생성: [ax.py (line 658)](/home/dk/workspace/lit_ws/lumo/optimization/adapters/ax.py:658)
- 상태는 summary에만 기록: [run_bo.py (line 474)](/home/dk/workspace/lit_ws/scripts/optimization/run_bo.py:474)
- 무조건 return 0: [run_bo.py (line 543)](/home/dk/workspace/lit_ws/scripts/optimization/run_bo.py:543)
실제 smoke 산출물도 첫 Ax 후보가 invalid_design, 최종 상태가 proposal_budget_exhausted였지만 CLI 관점에서는 성공입니다. [summary.json (line 40)](/home/dk/workspace/lit_ws/output/optimization/bo_smoke_20260820/summary.json:40)
또한 authoritative nominal 평가가 실패해도 proposal loop가 계속됩니다. Production 진입 조건은 최소한 다음을 요구해야 합니다.
- nominal 18-state 평가 성공
- MBM/BO 성공 trial 최소 1개
- 허용된 최종 상태만 exit 0
- stall, proposal exhaustion, zero feasible BO result는 non-zero exit
IMPORTANT


3. Ax model state가 영속화되지 않아 정확한 resume이 불가능합니다.
Production callback은 trials.json만 기록하고 Ax client snapshot은 저장하지 않습니다. [run_bo.py (line 442)](/home/dk/workspace/lit_ws/scripts/optimization/run_bo.py:442)
이는 registry가 Ax model state를 대체하지 않는다는 저장소 아키텍처 계약과 충돌합니다. [ARCHITECTURE.md (line 264)](/home/dk/workspace/lit_ws/docs/ARCHITECTURE.md:264)
반면 bounded validation runner에는 이미 ax_client.json snapshot 저장 구현이 있습니다. [lumo6d_test_bo.py (line 758)](/home/dk/workspace/lit_ws/validation/optimization/lumo6d_test_bo.py:758)
장시간 production BO 전에 snapshot의 원자적 저장과 명시적인 resume/load semantics가 필요합니다. 현재는 중단 후 결과 캐시는 재사용할 수 있어도 동일 캠페인의 생성 모델과 trial history를 정확히 복원할 수 없습니다.

4. 6D validation gate가 production과 다른 objective identity를 사용합니다.
Validation은 contact_state_separation@1을 Ax/registry objective로 등록합니다. [lumo6d_test_bo.py (line 57)](/home/dk/workspace/lit_ws/validation/optimization/lumo6d_test_bo.py:57)
하지만 현재 evaluator가 실제 계산하는 것은 trajectory_separation_margin_fixed_depth_v1입니다. [objectives.py (line 32)](/home/dk/workspace/lit_ws/lumo/optimization/objectives.py:32)
Adapter는 이름을 대조하지 않고 objective_value만 읽기 때문에, 수치는 전달되지만 provenance상 잘못된 objective 이름으로 저장됩니다. Validation gate를 production과 동일한 objective identifier로 통일하고, evaluator/Ax/registry objective ID 일치 테스트가 필요합니다.

5. 현재 worktree에 대한 검증 증거가 없습니다.
기존 smoke 산출물은 evaluator schema v1인데, 현재 코드는 schema v2입니다.
- 기존 산출물: [config.json (line 11)](/home/dk/workspace/lit_ws/output/optimization/bo_smoke_20260820/config.json:11)
- 현재 schema: [evaluator.py (line 80)](/home/dk/workspace/lit_ws/lumo/optimization/evaluator.py:80)
더구나 현재 smoke test는 registry schema가 2라고 단정하지만 production 상수는 3입니다.
- stale assertion: [test_run_bo_smoke.py (line 33)](/home/dk/workspace/lit_ws/tests/smoke/optimization/test_run_bo_smoke.py:33)
- 실제 상수: [evaluation_registry.py (line 18)](/home/dk/workspace/lit_ws/lumo/optimization/evaluation_registry.py:18)
따라서 현재 smoke gate는 정적으로도 실패가 확정되어 있습니다.
NON-BLOCKING / 좋은 부분
- evaluator contract fingerprint에 protocol, objective, mechanics, optics, device 및 fixed inputs가 포함됩니다.
- candidate failure와 infrastructure failure가 구분되어 있습니다.
- 18-state trajectory objective와 artifact 경로가 일관된 production boundary 안에 있습니다.
- 과거 trajectory regression은 nominal/probe 총 36 state를 성공적으로 통과했습니다. 다만 현재 refactor 이전 증거이므로 production readiness 증거로 재사용할 수는 없습니다. [regression_summary.json (line 1)](/home/dk/workspace/lit_ws/output/validation/refactor/pep20_cleanup/regression_summary.json:1)
최소 진입 순서는 다음입니다.
1. exit-status, nominal fail-fast, objective identity, smoke schema 수정
2. Ax snapshot/resume 추가
3. 현재 코드 기준 unit → preflight → 1-trial smoke → 18-state validation 실행
4. bounded 6D Test BO에서 실제 MBM 성공과 objective variation 확보
5. 그 결과가 PASS일 때만 production budget 승인
이번 작업은 read-only review였습니다. 파일은 수정하지 않았고 테스트·시뮬레이션·BO도 재실행하지 않았습니다. git diff --check만 통과했으며, 현재 production BO readiness 판정은 BLOCKED입니다.



Github Repo Review:
1. 확실한 blocker: real BO smoke test가 현재 HEAD에서 stale함
현재 production registry:
REGISTRY_SCHEMA_VERSION = 3
이다. 
그런데 full-stack BO smoke test 마지막은:
assert registry["schema_version"] == 2
다. 
그래서 GPU 환경에서 이 test를 실제로 돌리면 BO path 자체가 성공한 다음 마지막 assertion에서 실패할 가능성이 확정적이야.
코드 bug라기보다 validation drift인데, BO readiness에서는 꽤 중요해. 지금 HEAD에는 visible CI workflow도 없어서 이런 drift가 자동으로 막히지 않는다.
여기는 단순히 literal 2를 REGISTRY_SCHEMA_VERSION으로 바꾸는 게 더 좋겠어.
그리고 이 smoke test는 한 가지 더 약점이 있어. 현재:
assert summary["new_evaluation_count"] >= 1
인데 nominal evaluation 하나만 성공해도 충족할 수 있어. sole Ax proposal이 invalid여도 test가 통과할 여지가 있다. 
production-path smoke라면 가능하면 nominal + 최소 한 Ax-generated candidate 성공까지 보는 게 더 의미 있어.
2. 가장 중요한 scientific issue: energy balance ≠ optical convergence
이게 내가 지금 제일 신경 쓰는 부분이야.
현재 production transport는:
256 rays
max_interactions = 6
maximum_segment_count = 4096
maximum_periodic_wraps = 8
32 × 32 × 8 internal field
를 사용한다. 
그런데 transport에는 실제로 여러 numerical termination channel이 있어:
branch cutoff
max interactions
segment budget
periodic-wrap limit
no-event termination
그리고 이때 잘린 weight는 terminated_weight에 정상적으로 bookkeeping 된다. 
그래서 극단적으로,
launched = 1.0
physical transport = 0.5
numerically terminated = 0.5
여도 energy balance는 완벽하게 1.0일 수 있어.
즉 지금 아주 잘 구현된 energy bookkeeping은:
계산 중 energy가 사라지지 않았음

을 증명하지,
256 rays / 6 interactions / 4096 segments로 충분히 수렴했음

을 증명하지 않아.
문제는 objective가 nonzero field만 있으면 이 결과를 그대로 사용할 수 있다는 거야. objective_pathology=True여도 finite objective 자체는 유지하는 정책이고, 그게 test로도 명시되어 있다.  
그러면 BO가 morphology effect 대신:
이 morphology는 branching이 많아서 maximum_segment_count에 더 빨리 걸림

같은 numerical truncation landscape를 배울 수 있어.
내가 production 전에 반드시 확인할 것
적어도 representative morphology들에서 256 → 512 → 1024 rays, interaction/segment budget 증가에 대해 objective, D_inter, D_radius가 안정적인지 봐야 해.
특히 production 성공 조건으로는 적어도:
segment_budget_termination_count == 0
를 요구하는 걸 상당히 강하게 추천해.
다른 termination weight 허용치는 지금 숫자를 임의로 정하지 말고 convergence sweep을 보고 결정하면 되고.
3. internal path field가 grid 밖 값을 silent drop함
Objective가 쓰는 native internal field에서 이게 꽤 중요한 부분이야.
PathFieldAccumulator는 segment sample을 만든 뒤:
valid &= inside_x & inside_y & inside_z
로 거른 다음 valid sample만 field에 추가한다.
grid 밖 sample의 count도, weighted path length도 기록하지 않는다. 
현재 fixed field bounds는 꽤 넉넉해 보이고 valid search morphology에서 실제 clipping이 0일 가능성이 높아. 하지만 문제는 그 사실을 증명할 방법이 현재 result에 없다는 것이야.
surface escape 쪽은 observation grid 밖 positive event가 나오면 fail-fast하는데, internal field는 조용히 버린다는 비대칭도 있어. 
여기에는:
clipped_sample_count
clipped_weighted_path_length_mm
정도 diagnostic을 추가하는 걸 추천해.
그리고 production BO에서는 ideally 0이어야 함.
이건 당장 physics formulation을 바꾸는 게 아니라 scientific observability를 추가하는 것이라 부담도 작아.
4. candidate-specific optical failure 하나가 campaign 전체를 종료할 수 있음
candidate mechanics 쪽은 잘 되어 있어.
CandidateContactError, CandidateMechanicsError는 candidate failure로 변환되어 다음 trial로 진행한다. 
그런데 optical geometry 단계에는 Transport3DGeometryError 같은 오류가 있고, deformed candidate geometry에 따라 발생할 수 있는 경우가 있다.
현재 evaluator가 명시적으로 candidate failure로 잡아주는 optics exception이 제한적이라 이런 게 위로 올라가면 Ax boundary에서는:
unknown exception
→ current trial abandoned
→ exception re-raise
→ campaign abort
가 된다. 
반대로 Transport3DResultError 같은 실제 implementation/contract bug는 절대 candidate failure로 숨기면 안 된다. 현재 tests도 이 원칙을 잘 지키고 있어. 
그래서 blanket catch가 아니라:
CandidateOpticsError
같이 candidate-dependent geometry/transport invalidity만 따로 분류하는 게 좋아.
이건 unattended BO robustness 측면에서 HIGH라고 봐.
5. 6D BO에 Sobol initialization = 1은 너무 공격적임
현재 production:
INITIALIZATION_TRIALS = 1
이야. 
그런데 같은 repo의 bounded 6D Test BO는 의도적으로:
6 Sobol
4 model-based
를 쓴다. 
나는 production도 최소 6 정도로 맞추는 쪽을 추천해.
특히 지금 search box가 매우 넓어:
flat       0.5 – 29.5
semi       0.5 – 29.5
stem_w     1   – 20
stem_h     1   – 25
void_w     0   – 10
void_h     0   – 25
인데 Ax에 전달되는 analytic constraints는 사실상 두 linear constraint뿐이야. 
실제 fingertip에는 nonlinear geometry/thickness constraints가 훨씬 더 많고, invalid candidate는 Ax에서 failed trial이 될 뿐 feasibility model로 적극 학습되는 구조는 아니다.
현재 box와 linear constraints를 가지고 내가 analytic geometry 조건만 대충 sampling해보면, linear constraints를 통과한 영역 중에서도 상당 부분이 geometry-invalid가 될 수 있는 형태야. 5 mm minimum silicone thickness까지 넣으면 더 빡빡해질 수 있고.
그래서:
1 Sobol + 바로 MBM
은 search efficiency 측면에서 꽤 위험해 보여.
6 Sobol은 과한 숫자도 아니고, 네 own test runner와도 일치해.
6. Newton 쪽은 구현보다 “현재 설정의 evidence”가 남아 있음
여기는 생각보다 좋았어.
repo에 이미 sweep_newton_sphere_parameters.py가 있고:
iterations
load steps
sphere subdivisions
를 reference와 비교하도록 되어 있어.
acceptance도:
RMS vertex difference <= 0.005 mm
relative max displacement difference <= 3%
처럼 명확히 정의되어 있다. 
이건 아주 좋은 방식이야.
다만 production은 현재:
10 VBD iterations
0.05 mm max increment
으로 고정이고 runtime residual 자체를 보는 건 아니다. 
그래서 static code review만으로:
10 iterations가 현재 넓어진 6D search space 전체에서 충분하다

고 certify할 수는 없어.
코드를 다시 설계할 필요는 없고, 이미 만든 sweep을 현재 HEAD에서 돌리는 게 답이야.
가능하면 nominal뿐 아니라 feasible edge/interior morphology 몇 개에서 확인하면 된다.
Mesh도 같은 논리로 search mesh 1.5 mm와 reference mesh 1.0 mm 사이 objective sensitivity를 몇 점에서 보는 게 좋고.
7. config/lumo_execution.yaml은 현재 애매한 상태
파일 첫 줄은:
Expert-owned numerical configuration for the production LUMO BO path

인데 바로 아래는:
future load_lumo_execution_config() loader

라고 적혀 있어. 실제 runner도 YAML을 읽지 않고 Python USER CONFIG를 사용한다.  
그리고 YAML에는 실제 MechanicsContract가 가지는:
soft_contact_mu
rigid_sdf_target_voxel_mm
가 없다. 
현재 BO가 YAML을 사용하지 않으므로 당장 scientific result를 잘못 만드는 bug는 아니야.
하지만 production config가 두 군데 존재하는 것처럼 보이는 건 위험해.
나는 BO 전에 둘 중 하나를 택할 것 같아:
현재는 Python USER CONFIG가 authoritative
→ YAML에 명시적으로 DRAFT / NOT LOADED
또는 loader까지 완성.
지금은 반쯤 연결된 상태라 별로야.
8. provenance는 좋은데 Git SHA도 저장했으면 좋음
현재 evaluation contract가 numerical/scientific input을 상당히 잘 fingerprint한다.
그런데 implementation code는:
LUMO_EXECUTION_CONTRACT =
    "newton-1.4-vbd+full3d-optix-v4"
라는 manually maintained semantic version에 의존한다. 
코드를 바꾼 후 이 string bump를 깜빡하고 외부 registry를 재사용하면 옛 result를 동일 contract로 간주할 여지가 있어.
따라서 campaign config.json에는 최소:
git_commit: 32c05c1...
git_dirty: false
를 남겼으면 좋겠어.
cache invalidation key에 Git SHA를 무조건 넣으라는 뜻은 아니야. 그러면 harmless code change마다 cache가 날아가니까.
provenance에는 반드시 기록, contract invalidation은 지금 semantic execution version 유지해도 충분히 합리적이야.
9. 성능 측면에서는 OptiX runtime을 candidate 간 공유할 수 있음
LumoSimulation은 이미 runtime injection을 지원하고 lazy reuse도 한다. 
그런데 evaluator는 morphology마다 새 LumoSimulation을 만들면서 shared runtime을 전달하지 않아서, candidate마다 OptiX pipeline/runtime을 다시 만든다.
18 optical states within candidate에서는 재사용하니까 끔찍한 건 아닌데, 100 candidate면 runtime setup도 100번.
BO correctness 끝난 다음에는:
Lumo3DTrajectoryEvaluator
    owns one OptixRuntime
          ↓
LumoSimulation.from_fingertip(..., optix_runtime=runtime)
로 바꾸면 꽤 자연스러울 듯.
다만 이건 BO 시작 blocker는 아님.
그래서 실제 GO 조건은?
내가 지금 이 repo를 가지고 long production BO를 시작한다면, 아래 sequence를 다 통과한 뒤 시작할 거야.
1. stale registry-schema smoke test 수정하고 full unit/architecture suite green 확인.
2. candidate-local optical geometry failure 분류 보강.
3. internal-field clipping diagnostic 추가.
4. numerical termination, 특히 segment_budget_termination, 이 production objective를 오염시키지 않는다는 convergence 확인.
5. Newton convergence sweep을 current HEAD + representative morphologies에서 실행.
6. optics 256 vs higher ray-count objective convergence 확인.
7. validation.optimization.lumo3d_trajectory_validation 성공.
8. 6 Sobol + 4 MBM bounded Test BO가 current production evaluator로 정상 완료.
9. production initialization도 최소 6 정도로 두고 fresh registry에서 첫 campaign 시작
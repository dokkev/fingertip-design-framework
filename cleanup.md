lumo/fingertip/ 총평

현재 역할 분리는 좋다.

FingertipParameters       <- 물리 input
 ├─ FingertipGeometry
 ├─ ViscoelasticParameters
 ├─ SiliconeOptics
 └─ LEDParameters
          ↓
      Fingertip
      ├─ Silicone
      ├─ Carrier
      └─ BondingInterface

그리고 이 directory가 Newton/OptiX를 전혀 모르고 순수 physical/analytic domain layer로 남아 있다는 게 가장 좋은 점이야.

우선순위 높은 finding
우선순위	위치	Finding
P0 handoff	docs/geometry.md	현재 코드와 완전히 다른 세대의 설명. 학부생에게 매우 위험
P1	viscoelastic_param.py	ViscoelasticParameters라는 이름이 실제 모델보다 과한 의미를 가짐
P1	optical_param.py	LED parameter 상당수가 실제 tracer에 영향을 안 주는 metadata인데 그게 API에서 잘 안 보임
P1	fingertip.py	Fingertip.__post_init__()가 geometry construction을 너무 한 덩어리로 가지고 있음
P1 handoff	package 전체	pure geometry에 cheap unit-test safety net이 거의 없음
P2	bonding_interface.py	library 코드에서 print("[WARNING]") 사용
P2	geometry/optimization	total_pad_depth <= 30 mm의 ownership을 명확히 해야 함
P3	여러 파일	몇 가지 naming/public API polish
1. geometric_param.py

이 파일은 상당히 좋음.

FingertipGeometry가 frozen dataclass이고, 모든 물리 치수에 _mm suffix가 붙어 있고, mesh resolution 같은 numerical setting은 일부러 배제되어 있어. 이건 학부생 입장에서 정말 좋은 API야.

특히 이런 derived property:

total_pad_depth_mm
cutout_width_mm
cutout_height_mm

가 있어서 downstream에서 식을 복붙하지 않게 한 것도 좋고.

Validation도 읽기 좋음.

_validate_positive_dimensions()
_validate_clearance()
_validate_link_geometry()
_validate_cutout_geometry()

구조도 딱 적당해. 에러 메시지도 required_width, flat_pad_width 같은 실제 값을 보여줘서 디버깅이 쉬워.

여기서 하나 결정할 것

우리가 지금 BO에

h
fp
	​

+h
ep
	​

≤30 mm

constraint를 넣었잖아.

현재 FingertipGeometry에는 이 제한이 없음. 그래서 이것이:

A. 물리적으로 LUMO fingertip이면 무조건 지켜야 하는 constraint

라면 여기 들어가야 하고,

B. 현재 BO campaign에서만 원하는 design envelope

라면 optimization/DesignSpace에 있는 게 맞아.

둘 중 하나를 명확히 해야 돼. 현재 DesignSpace가 linear constraint를 따로 소유하는 구조 자체는 이미 깔끔해.

내 생각엔 지금 상황상 B로 두는 게 더 자연스러워 보여. 31 mm fingertip이 수학적으로 invalid geometry인 건 아니니까.

그러면 README에 딱:

Geometry validity and optimization bounds are different concepts.

라고 써주는 게 좋음.

2. fingertip_param.py

이건 아주 깔끔함.

FingertipParameters(
    geometry=...,
    viscoelastic=...,
    optical=...,
    led=...,
)

하나가 complete physical definition이라는 ownership이 바로 보여.

그리고 LED가 실제 stem 안에 들어가는지 여기서:

led.width_mm <= stem_width_mm
led.height_mm <= stem_height_mm

검증하는 것도 위치가 적절해. geometry 혼자서는 LED를 모르니까 FingertipParameters가 cross-component invariant를 검사하는 게 맞음.

아쉬운 건 viscoelastic 이름

이건 아래에서 더 자세히 말하겠지만:

parameters.viscoelastic

를 학생이 보면 거의 100%:

“아 Maxwell/Prony/stress-relaxation model인가?”

라고 생각할 거야.

실제로는 그렇지 않음.

3. viscoelastic_param.py

여기가 내가 handoff 전에 가장 고치고 싶은 production code야.

현재:

@dataclass(frozen=True)
class ViscoelasticParameters:
    density_kg_m3
    k_mu_pa
    k_lambda_pa
    damping

인데 Newton에서는 이것을 그대로:

density=material.density_kg_m3
k_mu=material.k_mu_pa
k_lambda=material.k_lambda_pa
k_damp=material.damping

로 add_soft_mesh()에 넘김.

문제는 이게 일반적으로 사람들이 생각하는 viscoelastic constitutive model은 아니라는 거야.

우리가 앞에서 stress relaxation 때문에 꽤 오래 헷갈렸던 것도 사실 이 naming이 한몫함 ㅋㅋ.

나는 handoff 전이면 진짜 rename 고려할래.

예를 들어:

SiliconeMechanicalParameters

또는

MechanicalMaterialParameters

그리고:

FingertipParameters.mechanical

정도.

최소한 rename이 부담스럽다면 docstring을 매우 강하게:

"""Newton soft-body material parameters.

This is a near-incompressible elastic model with Newton rate damping.
It is not a hereditary viscoelastic model and contains no Maxwell/Prony
stress-relaxation state.
"""

로 만들어야 돼.

damping도 이름이 아쉬움

다른 값들은:

density_kg_m3
k_mu_pa
k_lambda_pa

인데 혼자:

damping

이야.

차라리:

damping_pa_s

같이 단위까지 드러내는 게 훨씬 낫다.

preset naming도 정리 필요

현재:

SILICONE_VISCOELASTIC
VISCOELASTIC_PRESETS = {"silicone": ...}

인데 ARCHITECTURE는 이 값을 “Dragon Skin 10 NV baseline”이라고 부르고 있어.

근데 우리가 방금 얘기한 것처럼 Solaris에도 representative mechanics로 쓸 생각이면 generic silicone mechanics라고 부르는 게 더 정확해.

예를 들어:

SILICONE_MECHANICS_NOMINAL

정도.

즉 코드와 문서가 material identity에 대해 하나의 말을 하게 해야 함.

4. optical_param.py

여긴 반대로 scientific provenance는 상당히 잘 되어 있어.

Solaris와 Dragon Skin 각각:

refractive index 어디서 왔는지
extinction이 실제 제품 calibration이 아니라 literature prior라는 것
dB/cm → m⁻¹ 변환
LOW/NOMINAL/HIGH 의미

를 코드 바로 옆에서 설명하고 있음. 이런 주석은 유지해야 돼.

학생들이 숫자 보고:

“71.38 m⁻¹는 Dragon Skin datasheet 값인가요?”

하는 사고를 막아줌.

여기서 가장 큰 onboarding trap

LEDParameters에:

dominant_wavelength_nm
peak_wavelength_nm
spectral_half_width_nm
viewing_half_angle_deg

가 있는데 현재 production ray emission에서는 이 값들이 실제 ray distribution을 바꾸지 않음.

LED.emit()은 그냥:

lambertian_emission(
    ...,
    total_power=self.parameters.normalized_power,
)

을 호출해. 즉 emission physics에서 쓰는 LED parameter는 사실상 normalized_power뿐이고, 나머지는 hardware metadata야.

특히 학생이:

viewing_half_angle_deg = 30

으로 바꾸고

“LED angle 바꿨는데 결과 왜 안 변하지?”

할 가능성 매우 높음 ㅋㅋ.

이건 field를 삭제하라는 건 아니고, 주석을:

# Hardware metadata only. Current Lambertian emission does not consume this.
viewing_half_angle_deg: float = 60.0

처럼 명시하는 걸 추천.

wavelength들도 마찬가지.

# Metadata only; current transport is monochromatic and does not sample a spectrum.

이 한 줄들이 학부생한테 엄청 도움됨.

5. fingertip.py

전체적으로 domain modeling 자체는 좋음.

Silicone, Carrier를 parameter input이 아니라 constructed analytic geometry로 분리한 게 특히 좋아.

FingertipGeometry
      ↓
Fingertip
 ├─ Silicone
 └─ Carrier

라는 차이가 코드에서 분명해.

하지만 Fingertip.__post_init__()가 너무 많은 걸 한눈에 보여줌

현재 한 함수 안에서:

derived dimensions 계산
↓
Silicone 생성
↓
Carrier polygon 생성
↓
geometry bonding interface 생성
↓
custom interface clipping
↓
object assignment

을 다 함.

숙련자가 보면 괜찮은데, 학부생이 처음 보면:

“어디까지가 silicone 식이고 어디부터 carrier 식이지?”

가 됨.

여기서는 새 class 필요 없고 private helper 3개 정도면 충분해 보여.

예:

def _build_silicone(geometry) -> Silicone:
    ...

def _build_carrier(geometry, silicone) -> Carrier:
    ...

def _build_geometry_bonding_interface(silicone) -> BondingInterface:
    ...

그러면 __post_init__()는 거의:

geometry = self.parameters.geometry
silicone = _build_silicone(geometry)
carrier = _build_carrier(geometry, silicone)
boundary = _build_bonding_interface(silicone)
...

가 돼.

이건 abstraction 추가라기보다 긴 수식을 이름 붙이는 수준이라 Simple-is-Best에도 맞아.

minimum_silicone_thickness_mm는 알고리즘 설명 한 줄 필요

이 property 자체는 좋은 API야.

하지만 아래쪽 구현은:

sample_count = 4097
...
64 iterations
...
candidate local minima
...
golden-section refinement

이라 갑자기 난이도가 확 뜀.

수학적으로 문제 있어 보이진 않고, SciPy 없이 deterministic하게 전역 최소를 잡으려는 의도도 이해돼.

다만 학생 기준으로는 왜 이렇게 했는지 설명이 없음.

딱 이런 주석이면 충분해:

# The point-to-ellipse distance is not guaranteed to be unimodal.
# First scan the full lower semiellipse for candidate local minima,
# then refine each candidate with bounded golden-section search.
# This avoids a SciPy runtime dependency in the core geometry package.

그리고:

_ELLIPSE_DISTANCE_SAMPLE_COUNT = 4097
_MINIMUM_REFINEMENT_ITERATIONS = 64

로 magic number 이름 붙여주는 것도 좋아.

성능은 지금 GPU simulation에 비하면 별거 아니니까 알고리즘을 바꿀 이유는 없음.

6. bonding_interface.py

책임은 아주 명확함.

BondingInterface(left, right)
bonding_interface.clipped_to(actual_boundary)

끝. 좋은 작은 object야.

print("[WARNING]")는 바꾸는 게 좋음

production library 안에서:

print("[WARNING] ...")

하는 건 handoff 관점에서는 별로야.

학생이 BO 돌리는데 stdout에 갑자기 경고가 섞이고, 나중에 logging 붙이기도 어려움.

Python 기본:

warnings.warn(
    "...",
    UserWarning,
    stacklevel=2,
)

가 더 적절해.

별도 logger framework 만들 필요 전혀 없음.

clipping은 exact geometry 기반임을 알려주면 좋음

Shapely intersection을 그대로 쓰기 때문에 custom BondingInterface가 실제 boundary와 정확하게 coincident해야 해.

학생이 1e-5 mm 정도 삐끗한 polyline을 넣고 “왜 사라졌죠?” 할 수 있음.

지금 당장 tolerance logic을 넣을 필요까진 없고, docstring에:

Requested bond segments are expected to lie on the analytic carrier-silicone boundary.

정도면 충분할 듯.

7. __init__.py

기능적으로 문제 없음.

다만 package를 열면 export가 꽤 많아:

Fingertip
Silicone
Carrier
BondingInterface
FingertipParameters
FingertipGeometry
...
6 optical presets
2 preset dictionaries
...

학생 입장에서는 “뭘 먼저 써야 하지?”가 조금 애매해.

내가 public export를 막 줄이진 않을 것 같고, 대신 README에서:

from lumo.fingertip import Fingertip, FingertipParameters

이 두 개를 normal entry point라고 박아두면 충분함.

Carrier, Silicone, presets는 advanced/downstream use.

그런데 가장 큰 문제는 코드 밖에 있음

docs/geometry.md는 handoff 전에 반드시 고쳐야 함.

현재 문서는 코드와 비교하면:

coordinate를 y라고 설명하는데 현재 코드는 z
parameter 이름이 옛 이름
link_thickness = 3.5 mm라고 하는데 코드 기본값은 10
bond extension 4×2라고 하는데 코드는 5×8
void width 1이라고 하는데 코드는 2
void_height=0 fixed라고 적혀 있음
존재하지 않는 lumo.mechanics_contract.MechanicsContract까지 언급함

등이 섞여 있어.

이건 없는 문서보다 위험함.

학부생에게 넘길 거면 여긴 P0.

테스트도 조금 필요해

현재 validation/fingertip/에는 sampling, visualization, kinematic bond 같은 procedural validation은 꽤 있어.

근데 이 domain layer는 너무 싸고 중요한 코드라서 unit test 몇 개는 있는 게 좋아.

많이 만들 필요 없고 진짜 critical invariant만:

default Fingertip derived coordinates
invalid cutout rejected
invalid LED/stem fit rejected
semiellipse_depth_at_x_mm known values
bond interface default geometry
minimum silicone thickness known nominal result

정도.

특히 학부생이 geometry equation 바꿨다가 이상해지는 걸 바로 잡아줌.

내가 실제로 handoff 전에 고칠 순서

1순위: stale docs/geometry.md rewrite
2순위: ViscoelasticParameters naming/description 정리
3순위: LED metadata-only fields 명확히 표시
4순위: Fingertip.__post_init__ private helper로 가독성 개선
5순위: 작은 geometry unit tests
6순위: print → warnings.warn

우선순위	위치	Finding
P0/P1	mesh validation	현재 mesh sampling validation이 새 BO design envelope를 커버하지 않음
P1	mesh validation	Newton이 실제 쓰는 carrier_collision proxy를 직접 검증하지 않음
P1	fingertip_mesh.py	11 mm / 1 mm mesh contract가 magic default처럼 보임
P1 handoff	FingertipMesh	carrier vs carrier_collision 차이를 object docstring만 보고 이해하기 어려움
P2	private mesh functions	같은 default 값이 여러 private 함수에 중복됨
P2	FingertipMesh.__post_init__	bonded index invariant를 일부 silently normalize하고 완전히 검증하지 않음
P2 handoff	silicone_mesh.py	bond vertex detection 방식과 tolerance의 이유가 설명 부족
P3	carrier/silicone internals	코드 자체는 괜찮지만 geometry 그림/coordinate 설명이 한 곳에 있으면 훨씬 쉬움
1. fingertip_mesh.py

이 파일은 아주 좋음.

실제 학생이 mesh를 만들 때 알아야 할 건 사실상:

fingertip = Fingertip(...)
mesh = make_fingertip_mesh(fingertip)

이게 전부야.

내부적으로:

analytic Fingertip
    ↓
silicone TetMesh
carrier surface Mesh
carrier collision proxy Mesh
bonded vertex indices

를 한 object로 묶는다.

이 facade는 그대로 유지하는 게 맞아.

다만 FingertipMesh 설명을 더 친절하게

현재:

@dataclass(frozen=True)
class FingertipMesh:
    """Newton meshes produced from one analytic fingertip assembly."""

    fingertip
    silicone
    carrier
    carrier_collision
    bonded_vertex_indices

인데 초짜가 보면 바로:

carrier가 두 개인데 왜 두 개지?

함 ㅋㅋ.

실제로는:

silicone
    = deformable volumetric TetMesh
    = Newton mechanics
    = surface geometry는 OptiX에도 사용

carrier
    = complete physical carrier surface
    = visualization / optics

carrier_collision
    = Newton용 closed collision proxy
    = invisible
    = signed SDF query가 안정적으로 되도록 별도로 만든 mesh

잖아. ARCHITECTURE에는 이게 잘 설명돼 있는데 class 자체에는 없음.

나는 docstring에 이 설명을 그대로 짧게 넣겠어.

2. mesh defaults는 한 군데만 source of truth로

현재 public 함수가:

make_fingertip_mesh(
    ...,
    extrusion_depth_mm=11.0,
    element_size_mm=1.0,
)

인데 private 함수들도 각각 다시:

_make_silicone_mesh(... extrusion_depth_mm=11.0, element_size_mm=1.0)
_make_carrier_mesh(... extrusion_depth_mm=11.0)
_make_carrier_collision_mesh(... extrusion_depth_mm=11.0)

로 default를 가지고 있어.

private 함수들은 검색해보면 fingertip_mesh.py에서만 사용되고 있어.

그러면 그냥 private 함수 default를 없애는 게 좋아.

def _make_silicone_mesh(
    ...,
    extrusion_depth_mm: float,
    element_size_mm: float,
):

처럼.

그럼:

mesh resolution/default를 바꾸려면 make_fingertip_mesh() 한 군데만 보면 된다

가 돼.

학생 handoff에서 이런 게 생각보다 엄청 중요함.

그리고 public default에는 차라리:

_DEFAULT_EXTRUSION_DEPTH_MM = 11.0
_DEFAULT_ELEMENT_SIZE_MM = 1.0

이름을 붙여도 좋아.

특히 11 mm는 왜 11인지 현재 code만 읽어서는 전혀 모르겠어. 1 mm는 “mesh element size”라 바로 이해되지만 11 mm는 physical out-of-plane extrusion depth니까 README/geometry docs에서 한 문장 설명이 필요함.

3. silicone_mesh.py

여기도 overall은 잘 짜여 있어.

특히 module 첫 부분에서 frame을:

X: cross-section lateral
Y: extrusion
Z: contact normal

로 설명하고, Gmsh 내부 XY/Z frame을 LUMO XYZ로 변환한다고 명시한 건 굉장히 좋음.

그리고 transformation:

rotated[:, 0] = vertices[:, 0]
rotated[:, 1] = -vertices[:, 2]
rotated[:, 2] = vertices[:, 1]

도 centralized되어 있어서 coordinate conversion이 mesh code 곳곳에 퍼지지 않았어.

이건 유지.

Gmsh lifecycle도 좋음
gmsh.initialize()

try:
    ...
finally:
    gmsh.finalize()

라서 mesh generation에서 exception이 나도 global Gmsh runtime을 정리함.

그리고 gmsh, newton import도 함수 내부에서 해서:

import lumo.mesh

자체는 Gmsh/Newton dependency가 없어도 가능해.

이것도 좋은 구조야.

4. Bonded vertex detection은 맞는데 설명이 더 필요함

현재 bond vertex는 Gmsh semantic tag를 가지고 오는 게 아니라:

points = vertices_mm[:, (0, 2)]

로 XZ projection을 만든 다음 analytic BondingInterface polyline과 거리를 계산해서:

distance <= 1e-5 mm

인 vertex를 bond로 잡고 있어.

이 방식 자체는 합리적이야.

bond interface는:

2D line × extrusion depth

인 surface니까 Y를 무시하고 XZ만 검사하면 extrusion 방향 전체의 surface nodes를 잡을 수 있음.

문제는 학생이 보면:

“왜 Y를 버리지?”

“왜 tolerance가 1e-5 mm지?”

“Gmsh physical group 안 쓰고 왜 이걸 하지?”

할 가능성이 큼.

그래서 _find_bonded_vertex_indices() 위에 한 4줄 정도 explanatory comment 있으면 좋겠어.

예:

# The bonded boundary is an analytic XZ polyline extruded through the
# full Y depth. Therefore a mesh node belongs to the bonded surface when
# its XZ projection lies on either bond polyline; its Y coordinate is
# intentionally irrelevant.

그리고:

_BOND_VERTEX_TOLERANCE_MM = 1.0e-5

가 왜 이 값인지 한 줄.

현재 broad mesh validation이 이 tolerance로 성공해왔다는 걸 근거로 적든, Gmsh coordinate roundoff용이라고 적든.

숫자를 바꿀 필요는 없음. 이유를 남겨야 함.

5. Gmsh cross-section construction은 좋음

_build_cross_section() 흐름도 읽기 괜찮아:

flat rectangle
+
lower half ellipse
+
left/right bond extensions
→ fuse
→ internal cutout subtract

특히 ellipse를 polygon approximation으로 안 만들고 OCC exact ellipse로 만든 점도 좋고.

if radius_x >= radius_y:
    addDisk(rx, ry)
else:
    addDisk(ry, rx, rotated axes)

도 extreme morphology까지 지원하려는 코드라 괜찮아.

다만 이 함수는 geometry를 “다시 정의”하는 게 아니라 Silicone의 derived coordinates를 CAD primitive로 옮기는 adapter라는 걸 docstring에 조금 더 강조하면 좋아.

학생이 나중에 geometry shape 바꾼다고 여기부터 수정하는 사고를 막아야 함.

Shape definition → fingertip/

Gmsh representation → mesh/

이 구분.

6. carrier_mesh.py

이 파일도 구현 자체는 탄탄함.

특히 generic-looking _extrude_closed_polygon()이 실제 responsibility가 명확해.

valid CCW 2D XZ polygon
    ↓
side triangles
+ bottom cap
+ top cap
    ↓
closed Newton Mesh

Shapely triangulation 결과도 그냥 믿지 않고:

sum(triangle.area) ≈ polygon.area

인지 확인하는 것도 좋은 defensive check야.

Triangle winding도 cap마다 의도적으로 뒤집어서 closed outward surface를 만들고 있음.

현재 validation도 carrier에 대해:

triangle degeneracy
every edge exactly twice
directed edge winding
positive signed enclosed volume

까지 검사하고 있으니까, visual carrier mesh 신뢰성은 꽤 괜찮음.

7. 그런데 carrier_collision을 validation에서 안 보고 있음

이게 mesh 쪽에서 내가 가장 중요하게 보는 finding.

현재 Newton이 실제 carrier contact에 쓰는 건 complete carrier가 아니라:

carrier_collision

proxy잖아.

코드에서는 cavity-facing lip/stem surface만 노출하고, 나머지는 carrier interior 안쪽으로 polygon을 닫은 다음 extrusion depth를 2배로 해서 cap을 silicone 밖으로 밀어냄.

이건 상당히 중요한 geometry trick이야.

그런데 fingertip_mesh_sampling.py는:

validate_carrier_surface(mesh.carrier)

까지만 하고 mesh.carrier_collision은 검사하지 않아.

검색해도 collision proxy를 직접 geometry-validation하는 건 거의 production Newton code밖에 안 보여.

나는 handoff 전에 최소한:

validate_carrier_surface(mesh.carrier_collision)

를 기존 mesh sampling에 추가하겠어.

그리고 가능하면:

collision proxy Y caps lie outside silicone Y extent
collision polygon remains inside physical carrier

도 하나 확인.

후자는 production builder에서 이미 carrier_polygon.covers(polygon)으로 검사하고 있으니까 가장 중요한 건 closed/outward triangle surface 검증이야.

이건 테스트 하나 추가할 가치 있음.

8. carrier_collision 구현은 주석 없이 처음 보면 꽤 어렵다

여기:

boundary = (
    cavity left mouth
    stem left top
    stem left bottom
    stem right bottom
    stem right top
    cavity right mouth
    bond top right
    bond top left
)

가 왜 이런 모양인지 처음 보는 학생은 절대 바로 이해 못 함 ㅋㅋ.

현재 주석:

Follow the counter-clockwise carrier boundary through the cavity-facing lip and stem. Close the cross-section through the carrier interior...

가 있긴 해서 방향은 좋아.

여기는 ASCII 그림 하나만 있어도 엄청 좋아질 듯.

예:

physical carrier:

┌──────────────────────┐
│                      │
└───┐              ┌───┘
    │    stem      │
    │              │
    └──────────────┘

collision proxy:
only silicone-facing lip + stem surfaces are physically reachable;
remaining faces close through carrier interior for the SDF.

이걸 코드보다 ARCHITECTURE.md의 mesh section에 넣는 게 더 나을 듯.

9. collision proxy 2× extrusion은 좋은데 magic factor임

현재:

return _extrude_closed_polygon(
    boundary,
    extrusion_depth_mm=2.0 * extrusion_depth_mm,
)

이유는 주석에 잘 적혀 있어.

silicone physical mesh가:

Y ∈ [-D/2, +D/2]

이면 proxy는:

Y ∈ [-D, +D]

가 되니까 proxy cap이 실제 silicone edge에서 D/2 더 바깥에 있게 됨.

이건 맞는 설계야.

다만:

_COLLISION_PROXY_DEPTH_SCALE = 2.0

까지 만들 필요는 없을 것 같고, 현재 주석 정도면 충분.

10. FingertipMesh.__post_init__()는 약간 too forgiving

현재 bonded indices를:

indices = np.asarray(..., dtype=np.int32)
...
indices = np.unique(indices)

함.

즉 input에 duplicate가 있어도 silently 정리함.

또 upper bound:

index < silicone.vertex_count

는 여기선 검사하지 않고 Newton model builder에서 뒤늦게 검사함.

semantic ownership상 나는 FingertipMesh가 자기 bonded indices validity를 완전히 책임지는 게 더 좋다고 봐.

즉:

1D
nonempty
integer
unique
0 <= index < silicone.vertex_count

를 여기서 끝내는 것.

현재 validation도 실제로 이 조건들을 따로 검사하고 있음.

특히 초보자 handoff에서는:

bad FingertipMesh를 만들 수 있지만 Newton 들어갈 때까지 안 터진다

보다:

mesh object 생성 순간 바로 터진다

가 훨씬 좋음.

다만 이건 correctness bug는 아님, invariant ownership cleanup이야.

11. 제일 큰 실질적 문제: mesh sampling validation이 현재 BO space보다 낡음

이건 반드시 손보는 게 좋아.

현재 mesh sampling에서 variation 범위가:

flat_pad_width       25–35
flat_pad_height       3–8
semiellipse_height    6–20
stem_width            7–10
void_width            0–3
void_height           0–3

이고 stem_height는 아예 sample하지 않음.

그런데 현재 새 discrete BO search는:

flat_pad_width       = 30 fixed
flat_pad_height       2–29
semiellipse_height    1–20
stem_width            6–10
stem_height           4–10
void_width            0–4
void_height           0–5

야.

차이가 엄청 커.

특히:

flat height: 8 → 29 mm
ellipse min: 6 → 1 mm
stem_height: fixed → 4–10 mm
void_height: 3 → 5 mm

이제 optimizer가 mesh validation이 한 번도 안 본 morphology들을 적극적으로 만들 수 있어.

BO의 analytic validity gate가 많은 걸 걸러주긴 하지만:

analytic geometry valid ≠ Gmsh meshing robust

라서 이건 별도 문제야.

내가 여기서 딱 하나 추가한다면

기존 fingertip_mesh_sampling.py를 현재 optimization envelope에 맞춰 갱신하겠어.

1000개 필요 없음.

새 design space의 extreme/stress cases + feasible random 50 정도로:

thin flat / tall ellipse
thick flat / shallow ellipse
tall stem
deep void
wide void
small stem width
near minimum silicone thickness

를 포함.

그리고:

silicone tet validity
carrier validity
carrier_collision validity
bond indices

다 PASS하는지.

이건 BO 120 trial보다 훨씬 싼 보험이야.

12. 현재 validation 자체는 꽤 좋음

fingertip_mesh_sampling.py는 validation style이 아주 좋아.

framework 안 만들고 그냥:

sample
→ make Fingertip
→ mesh
→ inspect tets
→ inspect carrier
→ inspect bond indices
→ report

임.

AGENTS의 “procedural/local validation” 철학이 잘 지켜져 있음.

그래서 이 파일은 새 architecture 만들지 말고 coverage만 최신화하면 돼.

13. __init__.py

여긴 거의 완벽해.

from lumo.mesh import FingertipMesh, make_fingertip_mesh

딱 두 개만 public.

학생들은 carrier mesher/Gmsh helpers를 만질 일이 없으면 접근할 이유가 없음.

fingertip/보다 오히려 public surface가 더 깔끔해.

이거 유지.

Mesh handoff 관점 최종 평가

나는 대략:

Architecture       A
Public API         A
Ownership          A-
Readability        B+
Defensive checks   A-
Validation         B   ← current search envelope와 drift
Student onboarding B

정도로 봐.

코드 자체를 뜯어고칠 건 별로 없고, 진짜 필요한 건 “왜 carrier mesh가 두 개냐” 설명 + validation envelope 최신화야.

handoff 전에 mesh에서 내가 실제로 할 변경 순서는:

1. fingertip_mesh_sampling.py를 현재 design envelope로 업데이트하고 carrier_collision validation 추가

2. FingertipMesh docstring에 네 field의 역할을 명확히 설명

3. private mesh functions에서 중복 default 제거

4. bonded index invariant를 FingertipMesh에서 fail-fast하게 완결

5. bond projection/tolerance 설명 주석 추가

이 정도만 하면 돼.

mesh architecture 자체는 건드리지 말자. 지금 analytic Fingertip → one FingertipMesh → Newton + OptiX sharing 구조가 꽤 좋음.

lumo/newton/ 총평

구조는 생각보다 아주 작고 좋음.

lumo/newton/
├── __init__.py
├── model.py
└── indenter.py

public API도:

FingertipNewtonModel
build_fingertip_newton_model()
Indenter

세 개뿐이야.

그리고 ownership도 잘 나뉘어 있음:

mesh/
   ↓
Newton model construction
   ↓
simulation/ owns stepping

즉 newton/은 Newton object를 만드는 곳이지 simulation loop를 돌리는 곳이 아님. 이건 그대로 유지해야 함.

먼저 우선순위 finding
우선순위	위치	Finding
P1 handoff	indenter.py	Indenter.add_mesh()가 현재 repo에서 사실상 사용되지 않는 public API
P1 handoff	model.py	builder를 finalize/consume한다는 ownership이 초보자에게 충분히 강하게 드러나지 않음
P1 handoff	model.py	carrier의 visual mesh / collision proxy / perfect bond 관계가 코드만 보면 어렵다
P1 validation	Newton validation	collision proxy의 SDF/flags/filtering을 직접 검사하는 focused validation이 부족
P2	model.py	SDF numerical settings가 magic numbers
P2	Indenter	직접 Indenter(body_index=-5) 같은 invalid object를 만들 수 있음
P2	contact material	rigid indenter endpoint vs soft endpoint contact parameters ownership이 코드에서는 헷갈릴 수 있음
P3	model.py / runtime.py	_set_body_pose Warp kernel 중복
KEEP	builder.color()	이상해 보여도 VBD 때문에 필요한 upstream requirement
KEEP	manual SDF	full-surface soft contact 때문에 필요한 설계
1. __init__.py

이건 아주 좋음.

from lumo.newton import (
    FingertipNewtonModel,
    Indenter,
    build_fingertip_newton_model,
)

끝.

학생 입장에서도 “Newton에서 제공하는 건 이 세 개구나” 하고 끝낼 수 있어.

손댈 필요 없음.

2. Indenter

현재 object가 아주 단순함.

@dataclass(frozen=True)
class Indenter:
    body_index: int

그리고 두 constructor:

Indenter.add_urdf(...)
Indenter.add_mesh(...)
add_urdf()는 잘 짜여 있음

production에서 실제 쓰는 건 이쪽이야.

DesignStudy도:

builder = newton.ModelBuilder(...)
indenter = Indenter.add_urdf(...)
simulation = LumoSimulation(... builder=builder)

순서로 사용함.

이 흐름은 굉장히 명확해.

builder 생성
↓
external object(indenter) 추가
↓
fingertip 추가
↓
builder finalize
floating=True가 처음엔 이상해 보임
builder.add_urdf(
    ...,
    floating=True,
)

한 뒤:

builder.body_flags[body_start] = KINEMATIC

으로 바꿈.

학생은 여기서 높은 확률로:

“kinematic이면 floating=False 아닌가?”

함 ㅋㅋ.

근데 지금 의도는 맞아.

고정 joint로 world에 박아버리는 게 아니라 world pose를 매 tick 직접 prescribed motion으로 움직이고 싶으니까 free root body를 만들고 SolverVBD에서 kinematic으로 취급하는 구조잖아.

여기 주석 한 줄만 더 강하게 있으면 좋겠어:

# Import a movable free-root body, then mark that body kinematic.
# The simulation prescribes its world pose directly every tick.
3. URDF 초기 pose workaround도 유지해야 함

이 부분:

# Newton stores the free-root pose in joint_q during URDF import, but
# newly created State objects are initialized from builder.body_q.

builder.body_q[body_start] = tf

이건 좋은 주석이야.

이런 건 절대 "cleanup"한다고 지우면 안 되는 주석임.

특히 학부생이:

“이미 add_urdf에 xform 줬는데 이 줄 중복 아닌가요?”

하고 삭제할 만한 부분이라 더더욱.

4. contact material override 처리도 괜찮음

URDF importer 때문에:

previous_ke = builder.default_shape_cfg.ke
previous_kd = builder.default_shape_cfg.kd

try:
    builder.default_shape_cfg.ke = ...
    builder.add_urdf(...)
finally:
    builder.default_shape_cfg.ke = previous_ke

방식으로 처리하고 있어.

이것도 꽤 깔끔함.

특히 finally로 restore해서 다음 object에 material override가 새어나가지 않는 게 좋음.

다만 초보자 관점 설명 추가

이게 왜 필요한지 설명이 없음.

한 줄:

# URDF-imported shapes inherit builder.default_shape_cfg,
# so temporarily override it only for this imported asset.

이면 충분.

5. Indenter.add_mesh()는 나는 제거 후보로 봄

이게 이번 Newton review에서 가장 눈에 띄는 architecture finding.

현재 repo 전체 검색하면:

Indenter.add_mesh(...)

실제 caller가 안 보여. 검색 결과도 docs뿐임.

반면 add_urdf()는 production + validations에서 여러 군데 실제 사용 중임.

우리 원칙이:

concrete second use case 없으면 abstraction/API 만들지 않는다

였잖아.

그래서 실제 사용처가 없다면 add_mesh()는 삭제해도 된다고 봄.

학생들 입장에서는:

URDF indenter?
Mesh indenter?
둘 중 뭘 써야 하지?

라는 선택지만 하나 더 생겨.

현재 production contract가 URDF sphere라면:

Indenter.add_urdf()

하나만 public인 게 훨씬 좋음.

단, 네가 “flat plate validation 같은 데 곧 mesh indenter를 쓸 거다”라는 구체적 use case가 있으면 유지하면 되고.

6. Indenter 자체 validation은 약간 약함

현재 이게 가능해:

Indenter(body_index=-123)

왜냐하면 __post_init__()가 없거든.

물론 runtime에서는:

if indenter.body_index < 0 or >= body_count:
    raise ValueError(...)

로 다시 확인함.

correctness bug는 아님.

그래도 handoff 관점에서는 object가 자기 invariant를 책임지는 게 좋음.

def __post_init__(self):
    if isinstance(self.body_index, bool) or not isinstance(self.body_index, int):
        ...
    if self.body_index < 0:
        ...

정도.

다만 add_mesh()를 없애고 외부에서 constructor 직접 쓸 이유가 전혀 없다면 굳이 이것까지 안 넣어도 됨.

7. model.py의 전체 구조

이 파일은 처음 보면 어렵지만 실제 흐름은 꽤 단순함.

FingertipMesh
    ↓
soft TetMesh 추가
    ↓
bond vertex deactivate
    ↓
carrier collision SDF 생성
    ↓
kinematic carrier body
      ├─ full visible surface
      └─ invisible collision proxy
    ↓
VBD coloring
    ↓
finalize
    ↓
FingertipNewtonModel

이 그림을 ARCHITECTURE에 그대로 넣으면 학생들이 훨씬 빨리 이해할 거야.

8. soft body construction은 깨끗함
builder.add_soft_mesh(
    mesh=fingertip_mesh.silicone,
    density=material.density_kg_m3,
    k_mu=material.k_mu_pa,
    k_lambda=material.k_lambda_pa,
    k_damp=material.damping,
    particle_radius=0.0,
)

여기서 좋은 점은 Newton layer가 material 숫자를 재정의하지 않고:

FingertipParameters
→ material
→ Newton

으로 그대로 소비한다는 것.

이건 아주 좋음.

다만 전 단계에서 말했듯:

material.damping

이름은 향후 damping_pa_s 같은 식으로 바꾸는 게 학생한테 더 명확함.

9. particle_radius=0.0은 설명이 꼭 필요

여기 학생들이 보고:

“particle radius가 0이면 contact 안 되는 거 아닌가요?”

할 가능성 100%.

하지만 현재 contact detection 거리는 runtime의:

CollisionPipeline(
    soft_contact_margin=...
)

가 따로 소유함.

즉:

particle_radius
    ≠ contact detection thickness

soft_contact_margin
    = detection margin

구조.

ARCHITECTURE에는 이미 설명이 있는데, 코드 옆에도 한 줄 있으면 좋음:

# Silicone particles have no physical collision radius.
# Detection tolerance is supplied separately by CollisionPipeline.
particle_radius=0.0

초보자 보호 효과 큼.

10. Perfect bond 구현은 괜찮음

bond node에:

ACTIVE flag 제거

하고, 매 step:

particle_q = transform(local_position)
particle_qd = 0

로 prescribed boundary를 적용함.

이건 현재 Newton에 deformable kinematic target API가 없는 상황에서 꽤 직접적인 구현이야.

실제로 upstream도 moving deformable boundary에는 bespoke particle update workaround가 아직 사용되고 있다고 설명하고 있어서, 이 구현 방향 자체는 이상하지 않아.

그리고 validation도 현재:

carrier kinematic 확인
bonded particles inactive 확인
identity pose 확인
translated pose 확인

까지 하고 있음.

좋음.

11. prepare_step()은 이름이 조금 추상적이지만 역할은 맞음
def prepare_step(state_in, state_out, pose):
    apply_carrier_pose(state_in)
    apply_carrier_pose(state_out)

왜 양쪽 state를 다 건드리냐는 주석도 잘 돼 있음.

Solver swap 이후 stale bond state가 다시 살아나는 걸 방지하려는 거지.

이건 유지.

12. carrier가 mesh 두 개라는 걸 여기서 가장 명확히 해야 함

현재:

carrier_shape
carrier_collision_shape

두 개가 생김.

역할은:

carrier_shape
    visible
    no shape collision
    no particle collision

carrier_collision_shape
    invisible
    SDF
    particle collision enabled
    rigid collision pairs filtered

이거야.

이건 좀 특이한 구조라 student docs에 정말 크게 써놔야 함.

특히:

has_shape_collision=True

인데 바로 아래에서 모든 rigid shape pair를 filter하니까 처음 보면 모순처럼 보임.

왜 has_shape_collision=True냐

full-surface rigid-soft contact 경로에서 SDF participating shape로 잡히려면 shape collision machinery가 provision되어야 하기 때문.

그리고 실제 rigid-vs-rigid contact는:

builder.add_shape_collision_filter_pair(...)

로 다 막음.

현재 주석도 꽤 잘 되어 있음.

Newton upstream도 full-surface rigid-soft contact를 쓸 때 participating mesh에 SDF를 provision해야 한다고 명시하고 있어서 이 설계는 맞음.

13. collision proxy filtering 정책은 docs에 명시할 필요 있음

현재 proxy를 추가하기 전에:

existing_shape_indices = tuple(range(builder.shape_count))

을 저장하고,

for shape_index in existing_shape_indices:
    add_shape_collision_filter_pair(shape_index, carrier_collision_shape)

함.

즉 indenter 같은 기존 rigid shape는 carrier proxy와 rigid-rigid collision하지 않음.

하지만 silicone ↔ indenter contact와 silicone ↔ carrier proxy contact는 유지.

이게 의도적인데 학생이:

“왜 sphere가 carrier를 뚫지?”

라고 생각할 수 있음.

실제로 physical experiment에서는 우리가 관심 있는 게 silicone deformation이라 carrier proxy를 silicone containment 전용으로 쓰는 거잖아.

ARCHITECTURE에:

carrier_collision is a soft-contact-only proxy. It is not a general rigid obstacle.

정도 써두는 게 좋아.

14. SDF 설정은 magic number가 좀 많음

현재:

build_sdf(
    max_resolution=256,
    margin=5.0e-4,
    narrow_band_range=(-1.0e-2, 1.0e-2),
    texture_format="float32",
)

이런 숫자는 학생들이 제일 위험하게 만지는 부분임 ㅋㅋ.

“256이면 비싸니까 64로 내려볼까요?” 같은 거.

현재 mechanics validation contract에 영향을 줄 수 있으니까 이름 붙이는 게 좋아.

예:

_CARRIER_SDF_MAX_RESOLUTION = 256
_CARRIER_SDF_MARGIN_M = 5.0e-4
_CARRIER_SDF_NARROW_BAND_M = (-1.0e-2, 1.0e-2)

그리고:

# Numerical collision representation, not a physical material parameter.

한 줄.

새 config class는 절대 만들 필요 없음.

15. SDF cache reuse도 설명 필요

현재:

if fingertip_mesh.carrier_collision.sdf is None:
    build_sdf(...)

임.

이건 좋은 최적화야.

같은 FingertipMesh를 3 sphere scenario에서 reuse하면 매번 SDF 다시 만드는 비용을 피함.

다만 이것 때문에 FingertipMesh가 frozen dataclass여도 내부 newton.Mesh는 사실 mutable/cache-bearing object임.

학생들한테:

FingertipMesh geometry/topology는 immutable하게 취급하지만 Newton mesh 내부의 derived SDF cache는 lazily populated될 수 있다.

정도 알려주면 좋음.

16. builder.color() 절대 지우면 안 됨

이 부분:

builder.color()
model = builder.finalize(...)

초짜가 보면:

“color? visualization인가?”

하고 지울 가능성 있음 ㅋㅋㅋㅋ.

근데 이 color는 색깔이 아니라 VBD graph coloring임.

Newton SolverVBD는 rigid bodies가 있을 때 ModelBuilder.color() 또는 equivalent coloring을 finalize 전에 요구함. upstream source에도 명시돼 있음.

여기 주석 필수:

# SolverVBD graph coloring, not visualization color.
builder.color()

이 한 줄은 진짜 넣자 ㅋㅋ.

17. builder ownership은 더 크게 알려야 함

build_fingertip_newton_model()은 caller-provided builder를 받아서 finalize까지 해버림.

model = builder.finalize(...)

그러니까 이 함수 호출 후:

builder.add_whatever(...)

하면 안 돼.

현재 docstring에 있긴 하지만 초보자 onboarding에는 더 직접적으로:

"""Add the fingertip and finalize the builder.

All external bodies must be added before calling this function.
The supplied builder must not be modified afterward.
"""

정도 쓰는 게 좋아.

실제 DesignStudy가 올바른 순서의 좋은 예제임:

create builder
↓
add indenter
↓
construct LumoSimulation
    ↓
build_fingertip_newton_model()
    ↓
finalize

README에 이 5줄 예제로 보여주면 됨.

18. FingertipNewtonModel은 역할 좋음

이 object는:

FingertipMesh
Newton Model
body/shape indices
silicone particle range
bond particle indices
bond reference positions

만 보유함.

solver나 collision pipeline을 갖지 않는 점이 좋음.

그건 LumoSimulation이 갖고 있음.

이 boundary는 매우 잘 잡혀 있음.

여기 Solver를 넣지 말자.

19. silicone_vertices()도 API 위치 좋음

Newton global particle buffer에서:

silicone_particle_start
:
silicone_particle_start + count

만 잘라서 FingertipMesh ordering으로 돌려줌.

이 덕분에 evaluator가 Newton internals를 몰라도:

simulation.silicone_vertices()

→ OptiX update

할 수 있음.

이건 아주 좋은 boundary.

20. 다만 bonded-index validation 중복은 나중에 줄일 수 있음

현재 FingertipMesh도 bonded indices를 검증하고, Newton model도 또:

if local_indices.size == 0
if any(index >= vertex_count)

검사함.

지난 mesh 리뷰에서 말했듯 FingertipMesh.__post_init__()가 full invariant를 책임지도록 강화하면, 여기 duplicate defensive checks는 줄여도 돼.

즉:

mesh layer guarantees index validity
Newton trusts FingertipMesh

가 더 깔끔함.

현재는 mesh가 upper-bound를 완전히 보장하지 않으니까 지금 checks는 유지해야 함.

21. _set_body_pose kernel 중복

model.py에:

_set_body_pose()

가 있고 runtime.py에도 똑같은 게 있음.

보통 이런 거 보면 helper로 빼고 싶어지는데...

나는 그냥 둘 것 같아.

5줄짜리 Warp kernel 때문에 lumo/newton/kernels.py 같은 파일 하나 더 만드는 게 더 복잡함.

한쪽은 carrier bond ownership이고 한쪽은 external indenter runtime pose ownership이라 semantic owner도 다르고.

이건 “duplicate지만 acceptable”.

22. contact stiffness ownership이 학생한테 헷갈릴 수 있음

현재 두 레벨이 있음:

Indenter shape
    ke / kd

Newton soft-contact model
    soft_contact_ke / soft_contact_kd

그리고 DesignStudy는 같은 user value를 둘 다에 넣음.

즉:

contact_stiffness_n_m

가 사실 한 곳만 바꾸는 scalar가 아님.

현재 production evaluator에서는 의도적으로 양 endpoint를 동일하게 3e4로 맞추고 있지.

학생이 Indenter.add_urdf(... ke=...)만 직접 호출하면 soft endpoint는 그대로일 수 있음.

그래서 Indenter.add_urdf() docstring에:

This overrides the rigid-shape endpoint only. LumoSimulation.soft_contact_* controls the soft-contact endpoint.

정도 넣어주면 좋음.

23. Newton-specific validation은 하나 더 있으면 좋다

현재 kinematic_bond.py는 좋지만 검사 범위가 bond에 집중돼 있음.

handoff 전에 cheap structural validation 하나만 더 있으면 좋아:

build default FingertipMesh
↓
build Newton model
↓
assert
  carrier body KINEMATIC
  bonded particles inactive
  visible carrier:
      visible=True
      particle collision=False
  collision proxy:
      visible=False
      particle collision=True
      SDF != None
  carrier proxy filtered from indenter rigid shape
↓
PASS

실제 physics simulation까지 길게 돌릴 필요 없음.

이건 critical construction invariant라 unit/smoke에 가까움.

그리고 Indenter.add_urdf() 쪽도:

builder defaults ke/kd before
↓
add indenter with override
↓
builder defaults restored
↓
one kinematic body
↓
requested initial pose preserved

정도.

이런 건 학부생이 Newton API 업데이트하거나 cleanup할 때 바로 잡아줘.

24. Newton upstream과 비교해서 이상한 구현은 없음

current upstream도:

full-surface rigid-soft contact는 SDF provision 필요
CollisionPipeline(... enable_rigid_soft_full_surface_contact=True)
SolverVBD는 coloring 필요
mesh SDF precompute 권장

구조라서, 현재 LUMO의 이상해 보이는 부분들은 대부분 Newton을 잘못 쓴 코드가 아니라 Newton requirement에 맞춘 코드야.

그래서 여기서는 “더 예쁘게 만들기”보다 왜 그런지 문서화가 훨씬 중요함.

내가 handoff 전에 실제로 바꿀 것

내 순서는 이거야:

Indenter.add_mesh() 실제 caller 없으면 삭제
builder.color()에 VBD graph coloring 주석
particle_radius=0 / soft_contact_margin 관계 주석
build_fingertip_newton_model() docstring에 builder finalization ownership 강화
SDF magic values를 named constants로
carrier visual mesh vs soft-contact proxy를 ARCHITECTURE에 ASCII 그림
cheap Newton construction validation 추가
rigid endpoint / soft endpoint contact parameter 차이 docstring에 명시

나머지는 굳이 건드리지 않을래.
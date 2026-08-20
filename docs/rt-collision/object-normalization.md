# Collision-RT Object-Condition Normalization

## Purpose

LUMO uses Collision-RT as a geometric front-end that defines a reproducible contact-onset condition before deformable mechanics is evaluated.

The goal is **not** to force every fingertip morphology to use the same absolute object pose. The correct comparison is:

> a morphology-specific first-contact pose under a common contact-initialization protocol

or equivalently:

> a standardized zero-contact reference defined independently for each morphology

The actual first-contact transforms may differ between morphologies. What is shared is the rule used to compute them.

This separation is important because geometric contact onset can itself depend on fingertip morphology, object geometry, object orientation, and approach direction. Collision-RT therefore separates **geometric contact initialization** from **post-contact mechanics**.

The current LUMO 3D scene is an 11 mm representative cell, corresponding to

one LED pitch. The first-contact query is therefore currently defined over:

```text

11 mm compliant sensing cell

    \+

11 mm representative rigid-carrier slice

    \+

object geometry

```

The compliant TET geometry and the rigid-carrier visualization/geometry slice

use the existing `z = [-5.5, +5.5] mm` extrusion. A future full-width fingertip

can use the same contact-normalization API with a different geometry input;

the current 11 mm cell is the scope of the present query contract.

---

## Variables

Let

- `theta` denote fingertip morphology parameters,

- `M_theta` denote the corresponding undeformed fingertip contact geometry,

- `O` denote a rigid object triangle mesh,

- `R` denote the object's prescribed orientation,

- `T0` denote a non-contacting reference transform,

- `a_hat` denote a unit approach direction,

- `xi = (O, R, T0, a_hat)` denote an object/contact condition.

The object is translated along the prescribed approach direction while its orientation is held fixed during contact initialization.

A convenient scalar definition is:

```text

s_first(theta, xi)

\= inf { s >= 0 : O(T0 + s a\_hat) intersects M\_theta }

and the associated first-contact transform is:

T_first(theta, xi)

\= T0 + s\_first(theta, xi) a\_hat

```

Conceptually,

T_first(theta_1, xi) != T_first(theta_2, xi)

is allowed and may be expected when the external contact geometry changes with morphology.

The invariant is the definition of first contact, not the resulting world pose.

`T_first` must not be treated as an exact floating-point touching pose. Let

`C(T)` be the discrete Collision-RT collision predicate. The implementation

contract is a bracket followed by a refinement:

```text

T_clear:

C(T\_clear) == false

T_hit:

C(T\_hit) == true

T_first:

refined boundary estimate between T\_clear and T\_hit

```

For an approach-direction tolerance `epsilon_a > 0`, acceptance is expressed

by the two-sided bracket:

```text

C(T_first - epsilon_a a_hat) == false

C(T_first + epsilon_a a_hat) == true

```

The contract must not require `C(T_first) == true`. Triangle intersection and

floating-point boundary behavior make that exact predicate an implementation

detail rather than a scientific requirement. The reported bracket width is

part of the provenance of the first-contact estimate.

Newton should not be initialized exactly on this estimated boundary. Define a

separate spawn pose:

```text

T_spawn = T_first - clearance a_hat

```

Here `clearance` is a numerical initialization safeguard that places the

object on the known-clear side of the query. It is not a physical indentation

parameter and must not be folded into `Delta_s` or the optical/mechanics

metrics. The intended transition is:

```text

Collision-RT

-> T\_clear / T\_first

-> T\_spawn

-> Newton initialization

-> T\_first + Delta\_s

```

The Newton path may advance from `T_spawn` to the estimated contact pose and

then apply the post-contact travel protocol. `T_first` remains a geometric

estimate; `T_spawn` exists only to make solver initialization fail-safe.

Collision-RT Role

For each (theta, xi) pair, Collision-RT computes the first-contact translation along a_hat using only rigid/reference geometry.

Collision-RT does not compute deformation, force, penetration response, contact migration, or optical transport.

Its output is a geometry-normalized zero-contact state that can be passed to the post-contact mechanics stage.

morphology theta + object condition xi

              |

              v

         Collision-RT

              |

              v

    T\_clear / T\_first(theta, xi)

              |

              v

         T\_spawn(theta, xi)

              |

              v

         Newton VBD

              |

              v

   deformed fingertip state

              |

              v

         Optical RT

              |

              v

   contact-observation metric

This gives Collision-RT a precise role:

Collision-RT decouples geometric contact initialization from post-contact mechanics, allowing different morphologies to be compared from an equivalent contact-onset condition.

Here, equivalent means equivalent by protocol, not identical in absolute world coordinates.

Why a Fixed World-Frame Pose Is Insufficient

A morphology sweep should avoid ad-hoc initialization such as:

hand-tuned object poses,

morphology-independent object coordinates,

preselected indenter centers that ignore candidate geometry,

hidden initial gaps,

initial interpenetration caused by a changed external surface.

If first contact depends on geometry,

T_first = f(theta, O, R, a_hat)

then using one fixed world-frame transform can introduce a morphology-dependent initial gap or penetration before mechanics even begins.

The comparison would then confound two effects:

the morphology's actual post-contact mechanics,

an arbitrary difference in where the mechanics solve started.

Collision-RT removes this initialization bias by recomputing contact onset under one common geometric rule.

Object-Condition Adjustment

The object condition should be separated into two parts.

Shared sampled condition

The optimization/evaluation protocol samples or prescribes the same high-level condition family across candidate morphologies, for example:

xi_base =

(

object mesh,

object orientation,

lateral placement,

approach direction

)

These quantities define how the object approaches the fingertip.

Morphology-conditioned adjustment

Collision-RT then adjusts only the translation necessary to reach first contact:

xi_base

-> Collision-RT(theta)

-> T_first(theta, xi_base)

The resulting transform is morphology-specific, but the sampled object orientation, placement convention, and approach direction remain governed by the common experiment protocol.

This is the preferred protocol for direct morphology comparison.

Do not allow each morphology to independently search for its own "best" object orientation or lateral placement during the main optimization objective. That would mix morphology quality with morphology-specific scenario selection.

Morphology-specific pose discovery may be useful later for repertoire or robustness studies, but it is a different evaluation question.

Post-Contact Loading Protocol

First-contact normalization defines:

Delta_s = 0

at contact onset.

A second protocol is still required to define how far the object moves after first contact.

The primary LUMO optimization protocol should use a common approach-direction travel:

T_load(theta, xi, Delta_s)

\= T\_first(theta, xi) + Delta\_s a\_hat

where Delta_s is selected from a common displacement family:

Delta_s in {d1, d2, d3, ...}

This is a first-contact-relative displacement, not an absolute world-frame position.

The same Delta_s need not produce the same normal indentation component for all object/morphology conditions.

If the first-contact normal is n_c, then conceptually:

delta_n = Delta_s (a_hat dot n_c)

may differ across cases.

For the main design problem this is acceptable. LUMO is intended to evaluate contact observation under realistic approach geometry, and the geometry-induced loading difference is part of that interaction.

A force-matched protocol may be used later as a secondary validation, but it should not be required for the primary optimization while the fast Newton model is used as a search surrogate rather than a high-fidelity force predictor.

Indentation Depth as a Nuisance Variable

The optimization should generally evaluate more than one post-contact travel.

Rather than optimizing one morphology for one exact indentation depth, use a small common depth family after first contact.

(theta, xi)

-> T_spawn -> T_first

-> Delta_s_1, Delta_s_2, ...

-> Newton states

-> optical observations

This allows the contact-observation objective to reward:

separation between different contact states,

while avoiding excessive sensitivity to modest loading/depth variation within the same contact condition.

Invalid or mechanically pathological depth states should fail explicitly rather than causing an object-specific hidden depth adjustment.

Possible feasibility failures include:

inverted tetrahedra,

rigid-carrier collision,

excessive or nonphysical deformation,

contact leaving the intended sensing surface,

solver/contact-buffer failure.

Large-Scale Object-Condition Sampling

Collision-RT becomes most useful when the object is an arbitrary triangle mesh and the number of candidate conditions is large.

A useful sampling decomposition is:

sample:

object identity

object orientation

lateral placement

approach direction



solve by Collision-RT:

final scalar translation along approach direction

This avoids treating the final approach translation as another brute-force pose dimension.

Each sampled object hypothesis becomes a one-dimensional first-contact query along its prescribed approach direction.

The resulting workload is naturally parallel over:

N_morphology

x N_object

x N_orientation

x N_placement

x N_approach

and is therefore well matched to hardware-accelerated geometric queries.

The first implementation does not need clustering, repertoire discovery, or morphology-specific pose optimization. Those are optional future extensions.

Important Morphology-Dependence Qualification

The phrase morphology-specific first-contact pose is valid only when the candidate morphology changes geometry visible to the approaching object.

If all active design parameters change only internal walls, cavities, stems, or other hidden structure while the external contact surface is identical, then for a fixed xi:

T_first(theta_1, xi) ~= T_first(theta_2, xi)

and Collision-RT does not create a morphology-dependent initialization signal.

In that case its role should be described more narrowly as:

object- and pose-specific first-contact normalization for arbitrary object geometry

It remains valuable for automatic scenario initialization, but the paper must not claim that first-contact onset varies with morphology unless the optimized external geometry actually changes.

This distinction should be checked against the active LUMO design space before finalizing the paper claim.

When the external contact surface is identical across a morphology family, the

first-contact result can be computed once per object condition and reused. In

that case the cache dependency is:

```text

object condition xi

    |

    v

T_first(xi)

    |

    +-- morphology 1

    +-- morphology 2

    +-- morphology 3

```

Internal walls, cavities, stems, and other hidden morphology changes must not

invalidate that cache unless they alter the geometry presented to the

approaching object. If the external contact surface changes, the morphology

remains part of the first-contact key.

The intended solver-neutral result boundary is:

```python

@dataclass(frozen=True)

class ContactCondition:

object\_mesh

orientation

lateral\_placement

approach\_direction



@dataclass(frozen=True)

class FirstContactResult:

clear\_pose

contact\_pose       *# refined T\_first estimate, not an exact-touch predicate*

spawn\_pose         *# T\_first - numerical initialization clearance \* a\_hat*

travel\_to\_contact\_mm

approach\_direction

bracket\_width\_mm



find_first_contact(

fingertip\_contact\_surface,

condition,

) -> FirstContactResult

```

`fingertip_contact_surface` is the current 11 mm compliant sensing cell plus

its representative rigid-carrier slice. The result can then feed Newton as:

```text

T_spawn

-> T\_first

-> T\_first + Delta\_s\_1

-> T\_first + Delta\_s\_2

-> ...

```

This keeps object-condition normalization, solver-safe initialization, and

physical post-contact travel as separate quantities.

Detailed Object-Condition Generation and Contact-Alignment Logic

The first-contact normalization contract above assumes that an object condition
xi already contains an object orientation, lateral placement, and approach
direction. For real object meshes, those quantities should not be generated by
an unconstrained arbitrary 6-DoF search. LUMO should instead construct
physically meaningful candidate contact conditions from the object's actual
surface geometry.

The intended front-end is:

real object triangle mesh
        |
        v
sample candidate surface patches
        |
        v
estimate local normal + curvature / roughness
        |
        v
reject pathological patches
        |
        v
classify usable patches by curvature
        |
        v
construct an object-local contact frame
        |
        v
align that frame to a prescribed fingertip target frame
        |
        v
place object at a guaranteed-clear reference pose T0
        |
        v
Collision-RT along prescribed a_hat
        |
        v
[T_clear, T_hit] bracket
        |
        v
refined T_first
        |
        v
T_spawn
        |
        v
Newton: T_first + Delta_s a_hat

This distinction is important:

surface-patch alignment
    -> generates the contact condition to test

Collision-RT first-contact normalization
    -> determines where contact actually begins for that condition

The alignment stage proposes a meaningful geometric scenario. It does not
declare the final contact location. Collision-RT remains responsible for the
actual geometric onset.

1. Candidate surface-patch sampling

For an object triangle mesh O = (V, F), generate candidate surface samples
over the exposed object surface.

The first implementation should prefer deterministic, geometry-aware sampling
rather than arbitrary vertex selection. Suitable options include:

area-weighted triangle sampling,

Poisson-disk-like surface sampling,

farthest-point sampling over triangle centroids,

deterministic triangle-centroid subsampling for the MVP.

Each sample j defines a candidate point p_o^j and a local geodesic or
Euclidean neighborhood N_j containing nearby surface faces or samples within
a configurable radius r_patch.

The neighborhood radius should be tied to the intended contact scale rather
than the global object size. For the present representative-cell model, the
patch radius should be comparable to the expected local contact footprint and
smaller than the 11 mm longitudinal cell width.

Do not choose candidates from degenerate or zero-area triangles, disconnected
mesh noise, non-manifold neighborhoods when identifiable, or regions that
cannot support a stable local surface estimate.

2. Local normal estimation

For each candidate patch, estimate a robust outward local normal using an
area-weighted average of consistently oriented face normals:

n_bar = normalize(sum_i A_i n_i)

Reject a patch if the accumulated normal magnitude is near zero, the patch
contains strongly conflicting orientations, the result is non-finite, or the
local orientation is otherwise unreliable.

The patch normal should describe the local surface as a whole rather than use
one potentially noisy triangle normal.

3. Curvature / surface-roughness estimation

The MVP does not require exact differential-geometry principal curvature.
A robust normal-variation metric is sufficient for initial filtering:

E_normal =
    sum_i w_i (1 - clamp(n_i dot n_bar, -1, 1))
    ------------------------------------------------
                     sum_i w_i

with area weights w_i = A_i.

Also record a worst-case angular deviation:

theta_max =
    max_i acos(clamp(n_i dot n_bar, -1, 1))

Interpretation:

small E_normal
    -> nearly planar or gently curved patch

moderate E_normal
    -> usable curved surface

large E_normal or theta_max
    -> sharp curvature, edge, corner, faceting, or unstable patch

A later refinement may project the neighborhood into the local tangent frame
and fit a quadratic surface:

z(x, y) ~= 0.5 k_x x^2 + k_xy x y + 0.5 k_y y^2

to estimate approximate principal curvatures kappa_1, kappa_2 and a local
minimum radius R_min. This is not required for the first implementation.

4. Curvature is a filter and stratification variable, not an objective to minimize

Do not always select the flattest available object patch. That would bias the
design toward artificially easy best-case contacts.

Use curvature first to reject pathological contact regions, then stratify valid
conditions into controlled classes:

low curvature
    -> nearly planar contacts

medium curvature
    -> ordinary gently curved surfaces

high-but-valid curvature
    -> smaller-radius / harder contacts

edge-adjacent
    -> optional challenge or held-out class

The primary optimization set should contain a controlled mixture of usable
curvature classes. Final evaluation should deliberately include harder classes.

5. Object-local contact frame

For every accepted object patch, construct a deterministic right-handed frame:

F_o = (p_o, t_o1, t_o2, n_o)

where p_o is the patch center, n_o is the robust outward object normal, and
t_o1, t_o2 span the local tangent plane.

A tangent direction may come from a projected global reference axis, local PCA,
or the dominant in-plane patch direction. If the preferred reference axis is
nearly parallel to n_o, use a documented deterministic fallback axis.

6. Prescribed fingertip target frame

Define target frames on the undeformed compliant sensing surface:

F_f = (p_f, t_f1, t_f2, n_f)

A target may be one canonical sensing location or one element of a prescribed
set of locations along the compliant arc.

The primary morphology objective must not allow each morphology to search for
its own best target location. The target-location family belongs to the common
evaluation protocol.

7. Contact-frame alignment

Construct an object orientation R_align such that:

R_align n_o ~= -n_f

and, when tangent orientation is constrained:

R_align t_o1 ~= t_f1

Rotation around the normal may be fixed deterministically or sampled from a
small prescribed tangent-rotation family. It must not be independently
optimized per morphology in the main objective.

After orientation, apply a lateral translation so the transformed candidate
patch center lies on the prescribed fingertip target line:

R_align p_o + t_lateral ~= p_f + lambda a_hat

for some scalar free-space separation lambda.

Alignment proposes a contact hypothesis only; it must leave an explicit gap and
must not directly place the meshes into contact.

8. Approach direction

For a nominal normal approach, use the fingertip target frame to define the
approach direction with a repository-wide sign convention.

Conceptually, the transformed object patch normal should satisfy:

R_align n_o ~= -a_hat

and positive scalar travel along a_hat must move the object toward the
fingertip.

The exact sign convention should be unit-tested. More general oblique approach
directions can be sampled later.

9. Guaranteed-clear seed pose

Once orientation and lateral alignment are fixed, construct a verified
collision-free reference pose T0.

Do not use a hand-tuned object-specific offset.

A conservative seed may be generated by projecting the fingertip and transformed
object geometry onto the approach axis:

I_f = [min(x_f dot a_hat), max(x_f dot a_hat)]
I_o = [min(x_o dot a_hat), max(x_o dot a_hat)]

Translate the object opposite the approach direction until the projected ranges
are separated by at least seed_clearance_mm.

Then verify using the actual collision predicate:

C(T0) == false

If projected-range separation is still insufficient due to lateral geometry or
concavity, expand the separation deterministically until a verified clear pose
is obtained, subject to an explicit maximum search distance.

Never silently accept an intersecting seed.

10. Collision-RT first-contact bracket

With orientation and lateral placement fixed, vary only:

T(s) = T0 + s a_hat

First establish:

s_clear:
    C(T(s_clear)) == false

s_hit:
    C(T(s_hit)) == true

The bracket may be found by conservative stepping, a projected-distance estimate
followed by local stepping, or a direct RT travel estimate. The correctness
contract is the clear/hit bracket, not the specific stepping strategy.

Then refine deterministically, preferably by bisection:

s_hit - s_clear <= first_contact_tolerance_mm

Record:

T_clear = T(s_clear)
T_hit   = T(s_hit)
T_first = refined boundary estimate

The boundary estimate may use the midpoint while retaining both endpoints as
provenance. Exact touching remains explicitly outside the numerical contract.

11. Representative-cell longitudinal validity gate

The current mechanics domain is only:

z in [-5.5, +5.5] mm

and represents approximately one emitter pitch.

Introduce an explicit z_cell_margin_mm and require the candidate contact center
to lie inside the valid interior:

-5.5 + z_cell_margin_mm
    <= z_contact
    <=
+5.5 - z_cell_margin_mm

A stronger implementation should estimate the candidate patch's longitudinal
footprint and require the usable support to remain inside the representative
cell with the requested margin.

This is a simulation-domain validity gate, not a permanent restriction of the
physical LUMO finger.

12. Rigid-carrier / compliant-surface contact classification

The long-term registration frontend should distinguish contact onset with the
compliant sensing surface from contact onset with the representative rigid
carrier slice.

Under the same pose hypothesis, conceptually compute:

s_first_compliant
s_first_rigid

and classify:

if s_first_compliant < s_first_rigid - tolerance:
    valid tactile-first condition

elif s_first_rigid < s_first_compliant - tolerance:
    rigid-first / non-sensing condition

else:
    mixed or ambiguous onset

The current MVP may initially query only the compliant outer surface. The API
should remain compatible with this later classification and must not assume all
real-object approaches necessarily hit the compliant surface first.

13. Contact-condition identity and caching

A contact-condition identity should include at least:

object mesh identity / hash
candidate patch identity
object aligned orientation
fingertip target frame
tangent rotation
approach direction
external fingertip geometry fingerprint
first-contact query settings

If the external fingertip contact geometry is shared across morphologies, cache
the first-contact result independently of internal morphology.

Do not include internal walls, cavities, or stems in the cache key unless they
alter the externally presented contact geometry.

14. Detailed solver-neutral data contract

A complete eventual neutral contract can look like:

@dataclass(frozen=True)
class SurfacePatchCandidate:
    object_patch_id: int
    center_mm: tuple[float, float, float]
    normal: tuple[float, float, float]
    tangent_1: tuple[float, float, float]
    tangent_2: tuple[float, float, float]
    normal_variation: float
    max_normal_deviation_rad: float
    curvature_class: str
    usable_radius_mm: float


@dataclass(frozen=True)
class FingertipTargetFrame:
    target_id: str
    center_mm: tuple[float, float, float]
    normal: tuple[float, float, float]
    tangent_1: tuple[float, float, float]
    tangent_2: tuple[float, float, float]


@dataclass(frozen=True)
class ContactCondition:
    object_mesh: RigidObjectMesh
    patch: SurfacePatchCandidate
    target: FingertipTargetFrame
    aligned_reference_pose: RigidPose3D
    approach_direction: tuple[float, float, float]
    tangent_rotation_rad: float


@dataclass(frozen=True)
class FirstContactResult:
    condition: ContactCondition
    clear_pose: RigidPose3D
    hit_pose: RigidPose3D
    contact_pose: RigidPose3D
    spawn_pose: RigidPose3D
    travel_to_contact_mm: float
    bracket_width_mm: float
    spawn_clearance_mm: float

The MVP may use fewer classes, but it should preserve the boundaries:

surface analysis
    !=
contact-condition alignment
    !=
first-contact normalization
    !=
Newton mechanics

15. Recommended implementation modules

A clean repository structure is conceptually:

contact/
    surface_patch.py
        candidate sampling
        local normals
        normal-variation / curvature proxy
        curvature classification

    alignment.py
        fingertip target frames
        object patch frame
        frame alignment
        lateral placement
        approach direction

    first_contact.py
        collision predicate abstraction
        verified-clear seed
        clear/hit bracket
        bisection refinement
        T_first / T_spawn result

    types.py
        solver-neutral dataclasses

Follow the repository's existing package conventions if a better location
already exists.

These responsibilities must not live inside
physics/backends/newton_vbd.py, and neutral types must not depend on
OptiX. Collision-RT should later become one backend of the collision-query
interface.

16. Detailed end-to-end pseudocode

def generate_contact_conditions(
    object_mesh,
    fingertip_reference_surface,
    target_frames,
    settings,
):
    patches = sample_surface_patches(
        object_mesh,
        radius_mm=settings.patch_radius_mm,
        spacing_mm=settings.patch_spacing_mm,
    )

    accepted = []

    for patch in patches:
        patch_geometry = analyze_patch(object_mesh, patch)

        if not patch_geometry.valid:
            continue

        if (
            patch_geometry.max_normal_deviation_rad
            > settings.max_patch_normal_deviation_rad
        ):
            continue

        curvature_class = classify_curvature(
            patch_geometry.normal_variation,
            settings.curvature_bins,
        )

        for target in target_frames:
            for tangent_angle in settings.tangent_rotation_family_rad:
                aligned_pose, a_hat = align_patch_to_target(
                    object_mesh,
                    patch_geometry,
                    target,
                    tangent_angle,
                )

                T0 = make_verified_clear_seed(
                    fingertip_reference_surface,
                    object_mesh,
                    aligned_pose,
                    a_hat,
                    clearance_mm=settings.seed_clearance_mm,
                )

                if not representative_cell_valid(
                    patch_geometry,
                    aligned_pose,
                    target,
                    z_margin_mm=settings.z_cell_margin_mm,
                ):
                    continue

                accepted.append(
                    ContactCondition(
                        object_mesh=object_mesh,
                        patch=patch_geometry,
                        target=target,
                        aligned_reference_pose=T0,
                        approach_direction=a_hat,
                        tangent_rotation_rad=tangent_angle,
                    )
                )

    return stratified_sample(
        accepted,
        by="curvature_class",
        budget=settings.condition_budget,
    )

Then normalize one condition:

def normalize_contact(condition, fingertip_surface, settings):
    assert not intersects(
        fingertip_surface,
        condition.object_mesh,
        condition.aligned_reference_pose,
    )

    s_clear, s_hit = establish_first_hit_bracket(
        fingertip_surface,
        condition,
        settings,
    )

    s_clear, s_hit = bisect_first_hit_bracket(
        fingertip_surface,
        condition,
        s_clear,
        s_hit,
        tolerance_mm=settings.first_contact_tolerance_mm,
    )

    s_first = 0.5 * (s_clear + s_hit)

    T_first = pose_at(condition, s_first)

    T_spawn = translate(
        T_first,
        -settings.spawn_clearance_mm * condition.approach_direction,
    )

    assert not intersects(
        fingertip_surface,
        condition.object_mesh,
        T_spawn,
    )

    return FirstContactResult(
        condition=condition,
        clear_pose=pose_at(condition, s_clear),
        hit_pose=pose_at(condition, s_hit),
        contact_pose=T_first,
        spawn_pose=T_spawn,
        travel_to_contact_mm=s_first,
        bracket_width_mm=s_hit - s_clear,
        spawn_clearance_mm=settings.spawn_clearance_mm,
    )

Newton then consumes only the normalized near-contact protocol:

first = normalize_contact(condition, fingertip_surface, settings)

newton.initialize(
    object_pose=first.spawn_pose,
)

for delta_s_mm in post_contact_depth_family_mm:
    target_pose = first.contact_pose.translated(
        delta_s_mm * first.approach_direction
    )

    newton.solve_to(target_pose)

Newton should never replay the full free-space geometric search.

17. Contact-condition validity and failure reasons

Record explicit geometric rejection reasons during development:

degenerate_patch
unstable_patch_normal
curvature_too_high
mesh_nonmanifold_neighborhood
representative_cell_boundary
no_clear_seed
no_first_hit_within_max_travel
rigid_carrier_first
ambiguous_first_contact
collision_backend_failure

Keep these separate from Newton/mechanics failures:

tet_inversion
nonfinite_deformation
solver_failure
mechanically_excessive_depth

18. Optimization/evaluation sampling policy

Whenever the external contact geometry is shared across morphology candidates,
generate and freeze a common condition set before morphology evaluation:

real object set
    |
    v
generate patch candidates
    |
    v
curvature stratification
    |
    v
align to prescribed fingertip target frames
    |
    v
normalize first contact
    |
    v
freeze / cache ContactCondition set
    |
    +--> morphology 1 -> Newton -> Optical RT
    +--> morphology 2 -> Newton -> Optical RT
    +--> morphology 3 -> Newton -> Optical RT

For evaluation, separate held-out object identity from held-out local surface
conditions where possible:

optimization objects / patches
held-out objects
held-out patch classes
harder curvature / edge-adjacent challenge conditions

The purpose of alignment is to generate reproducible, physically meaningful
real-object contact conditions, not to remove contact diversity.

Optimization Contract

The intended eventual optimization pipeline is:

morphology theta

  \+

object condition xi

  |

  v

Collision-RT

  |

  v

T_clear / T_first(theta, xi)

  |

  v

T_spawn

  |

  \+ common Delta\_s family from T\_first

  v

Newton VBD mechanics

  |

  v

FingertipVolumeState

  |

  v

Optical RT / camera-facing observation

  |

  v

J_obs(theta, xi, Delta_s)

with an aggregate design objective of the form:

J(theta)

\= E\_{xi, Delta\_s}

  [J\_obs(theta, xi, Delta\_s)]

with robustness or lower-tail terms added if needed.

The scientific quantity optimized by LUMO is contact observation quality, not Collision-RT performance itself.

Collision-RT is the geometric initialization layer that allows arbitrary real object geometry to enter this design loop reproducibly and at scale.

Scope Boundary

Collision-RT owns

first-contact geometric queries,

object-condition translation normalization,

clear/hit bracketing and refined `T_first` estimates,

numerical `T_spawn` derivation,

optional geometric validity filtering before mechanics.

Newton owns

post-contact deformation,

contact evolution after onset,

compliant mechanics.

Optical RT owns

light transport through the deformed morphology.

Observation model / metric owns

contact-state distinguishability,

robustness to nuisance variation,

final design score.

These responsibilities should remain separate in both code and paper framing.
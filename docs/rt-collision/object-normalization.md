# Collision-RT Object-Condition Normalization

## Purpose

LUMO uses Collision-RT as a geometric front-end that defines a reproducible contact-onset condition before deformable mechanics is evaluated.

The goal is **not** to force every fingertip morphology to use the same absolute object pose. The correct comparison is:

> a morphology-specific first-contact pose under a common contact-initialization protocol

or equivalently:

> a standardized zero-contact reference defined independently for each morphology

The actual first-contact transforms may differ between morphologies. What is shared is the rule used to compute them.

This separation is important because geometric contact onset can itself depend on fingertip morphology, object geometry, object orientation, and approach direction. Collision-RT therefore separates **geometric contact initialization** from **post-contact mechanics**.

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
    = inf { s >= 0 : O(T0 + s a_hat) intersects M_theta }
    and the associated first-contact transform is:

T_first(theta, xi)
    = T0 + s_first(theta, xi) a_hat
```
Conceptually,

T_first(theta_1, xi) != T_first(theta_2, xi)

is allowed and may be expected when the external contact geometry changes with morphology.

The invariant is the definition of first contact, not the resulting world pose.

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
        T_first(theta, xi)
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
    = T_first(theta, xi) + Delta_s a_hat

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
   -> T_first
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

Optimization Contract

The intended eventual optimization pipeline is:

morphology theta
      +
object condition xi
      |
      v
Collision-RT
      |
      v
T_first(theta, xi)
      |
      + common Delta_s family
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
    = E_{xi, Delta_s}
      [J_obs(theta, xi, Delta_s)]

with robustness or lower-tail terms added if needed.

The scientific quantity optimized by LUMO is contact observation quality, not Collision-RT performance itself.

Collision-RT is the geometric initialization layer that allows arbitrary real object geometry to enter this design loop reproducibly and at scale.

Scope Boundary
Collision-RT owns
first-contact geometric queries,
object-condition translation normalization,
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
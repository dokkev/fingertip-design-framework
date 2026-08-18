# Optimized Morphology Parameters

## 1. Purpose

This document defines the production morphology optimization variables and the fixed evaluation protocol for the LIT/LUMO fingertip.

The optimization goal is **not** to enlarge the fingertip or maximize brightness. The goal is to reshape a fixed-size fingertip so that contact produces a strong, robust change in the lateral optical output.

The primary contact metric remains the contact/no-contact lateral state separation defined in `docs/optimization/evaluation.md`.

## 2. Fixed Outer Envelope

To prevent the optimizer from winning simply by increasing sensor size, the overall fingertip envelope is fixed.

### Fixed width

\[
w_{fp}=30\ \mathrm{mm}
\]

```text
flat_pad_width = 30 mm
```

### Fixed distal depth

\[
h_{fp}+h_{ep}=14\ \mathrm{mm}
\]

Thus:

\[
h_{ep}=14-h_{fp}
\]

`flat_pad_height` and `semielliptical_pad_height` are therefore **not independent optimization variables**.

### Fixed basal morphology

```text
void_height = 0 mm
basal_interface = bonded
```

`void_height` is not an optimization variable.

## 3. Optimization Variables

The production optimization is a **4-variable morphology search**:

\[
\boxed{
\mathbf q=[h_{fp},w_s,h_s,w_v]
}
\]

### 3.1 Flat pad height — \(h_{fp}\)

```text
parameter: flat_pad_height
nominal:   5.0 mm
```

Controls the relative amount of straight sidewall versus distal semielliptical morphology.

\[
h_{ep}=14-h_{fp}
\]

Primary effects:

- external contact morphology,
- deformation distribution,
- optical path redistribution near the distal pad.

### 3.2 Stem width — \(w_s\)

```text
parameter: stem_width
nominal:   7.6 mm
```

Primary effects:

- internal structural support,
- compliant material thickness around the stem,
- lateral deformation pathways,
- internal optical transport geometry.

### 3.3 Stem height — \(h_s\)

```text
parameter: stem_height
nominal:   6.0 mm
```

Primary effects:

- deformation pathway,
- local stiffness distribution,
- side-contact engagement,
- optical transport around the internal structure.

### 3.4 Lateral void width — \(w_v\)

```text
parameter: void_width
nominal:   1.0 mm
```

Primary effects:

- lateral contact/adaptation,
- available deformation space,
- internal optical path geometry,
- timing and extent of side-contact engagement.

## 4. Derived / Fixed Parameters

| Parameter | Production treatment |
|---|---|
| `flat_pad_width` | fixed at 30 mm |
| `flat_pad_height` | optimized |
| `semielliptical_pad_height` | derived as `14 - flat_pad_height` |
| `stem_width` | optimized |
| `stem_height` | optimized |
| `void_width` | optimized |
| `void_height` | fixed at 0 mm |
| `link_thickness` | fixed |
| `bond_extension_width` | fixed |
| `bond_extension_height` | fixed |
| `young_modulus_mpa` | fixed |
| `poisson_ratio` | fixed |

Mechanical material properties are not optimization variables.

## 5. Feasibility Constraints

Candidate morphologies must satisfy the existing physical geometry constraints before FEM evaluation.

At minimum:

- valid non-self-intersecting geometry,
- no material overlap,
- connected compliant pad,
- `void_height = 0`,
- bonded basal interface preserved,
- minimum silicone ligament requirement,
- valid mesh generation,
- valid FEA initialization,
- converged contact solve,
- finite physical fields,
- positive deformation Jacobian,
- valid object-contact mapping for optics.

Invalid candidates are treated as **infeasible**, not repaired with penalty weights.

Optimization bounds for the four active variables should be defined separately and must respect these constraints.

## 6. Mechanical Evaluation Protocol

### Indentation depths

\[
d\in\{0.5,1.0,1.5,2.0\}\ \mathrm{mm}
\]

These are **observation states along one monotonic loading trajectory**, not four independent FEM solves.

For each object diameter/location pair:

```text
FEA: 0 -> 2.0 mm
steps: 48

capture:
    0.5 mm
    1.0 mm
    1.5 mm
    2.0 mm
```

For a 2 mm / 48-step trajectory:

```text
0.5 mm -> step 12
1.0 mm -> step 24
1.5 mm -> step 36
2.0 mm -> step 48
```

## 7. Object Curvature Protocol

The frozen production indenter diameters are:

\[
D\in\{6,10,14,20\}\ \mathrm{mm}
\]

equivalently:

\[
R\in\{3,5,7,10\}\ \mathrm{mm}
\]

The set spans localized/high-curvature through broad/low-curvature contact.
The corresponding production optical boundary is the explicit ideal absorber
`IndenterOptics("absorber")`, interpreted as a smooth bulk-black rigid polymer
cylindrical indenter. A bulk-black material is preferred to a painted contact
surface; matte paint or soft optical coatings can change roughness, friction,
compliance, wear, and contact mechanics. The ideal absorber is a modeling
abstraction, not a physical claim of exactly zero reflectance.

## 8. Contact Location Protocol

Because `flat_pad_width` is fixed at 30 mm, contact locations can be defined in normalized half-width coordinates:

\[
\xi=\frac{x}{w_{fp}/2}
\]

Recommended:

\[
\xi\in\{0,0.3,0.6\}
\]

For \(w_{fp}=30\) mm:

\[
x\in\{0,4.5,9.0\}\ \mathrm{mm}
\]

Only one side is required during optimization if geometry and illumination remain symmetric. Opposite-side locations can be used later as a symmetry validation check.

## 9. Evaluation Count Per Morphology

The full grid contains:

\[
4\ \text{diameters}\times3\ \text{locations}\times4\ \text{depths}=48\ \text{contact states}
\]

But mechanics cost is only:

\[
4\ \text{diameters}\times3\ \text{locations}=12\ \text{FEA trajectories}
\]

because the four indentation depths are captured from each 0-to-2-mm trajectory.

Optical evaluation:

```text
1 unloaded reference state
48 loaded contact states
```

per morphology.

## 10. Contact-State Objective

For each contact state:

\[
J_{\mathrm{contact}}(d,D,x)
=
\frac{
\|\phi_L^c-\phi_L^0\|_1
+
\|\phi_R^c-\phi_R^0\|_1
}{
W_{\mathrm{launch}}
}
\]

where:

- \(\phi_L^0,\phi_R^0\): unloaded lateral optical profiles,
- \(\phi_L^c,\phi_R^c\): loaded lateral optical profiles,
- \(W_{\mathrm{launch}}\): common launched optical weight.

No brightness weight, redistribution weight, or empirical multi-objective coefficient is introduced.

## 11. Depth Aggregation

For each diameter/location trajectory:

\[
A(D,x)
=
\frac{1}{d_{\max}}
\int_0^{d_{\max}}
J_{\mathrm{contact}}(d,D,x)\,dd
\]

with:

\[
d_{\max}=2.0\ \mathrm{mm},
\qquad
J_{\mathrm{contact}}(0,D,x)=0
\]

Evaluate using the sampled states:

```text
0.0
0.5
1.0
1.5
2.0 mm
```

with trapezoidal integration.

## 12. Morphology Optimization Score

The proposed robust morphology score is:

\[
\boxed{
J_{\mathrm{morph}}=\min_{D,x}A(D,x)
}
\]

The optimizer maximizes:

\[
\boxed{
\max_{\mathbf q}J_{\mathrm{morph}}
}
\]

with:

\[
\mathbf q=[h_{fp},w_s,h_s,w_v]
\]

Equivalently:

\[
\boxed{
\max_{h_{fp},w_s,h_s,w_v}
\min_{D,x}
\frac{1}{2}
\int_0^2
J_{\mathrm{contact}}(d,D,x)\,dd
}
\]

subject to geometry, mesh, mechanics, and optical feasibility.

Interpretation:

> Within a fixed fingertip envelope, find the morphology that maximizes the weakest contact-induced lateral optical response across representative object curvatures and contact locations, integrated over indentation depth.

## 13. Diagnostics Retained Per Morphology

Retain at least:

```text
48 raw J_contact values
12 depth-AUC values

minimum raw J_contact
mean raw J_contact

minimum depth-AUC
mean depth-AUC
median depth-AUC

limiting diameter
limiting location
minimum raw contact state and depth

reaction force at each depth
contact width / active contact patch
left lateral throughput
right lateral throughput
total side throughput

quadrant transport delta
object absorbed weight
object transmitted weight
object reflected weight
energy balance error
```

These are diagnostics and interpretation aids, not weighted objective terms.

## 14. Production Optimization Contract

```text
DESIGN VARIABLES
    flat_pad_height
    stem_width
    stem_height
    void_width

FIXED / DERIVED
    flat_pad_width = 30 mm
    flat_pad_height + semielliptical_pad_height = 14 mm
    void_height = 0 mm
    basal_interface = bonded

MECHANICS
    48 steps
    0 -> 2.0 mm monotonic indentation
    12 trajectories per morphology

CONTACT STATES
    depths    = 0.5, 1.0, 1.5, 2.0 mm
    diameters = 6, 10, 14, 20 mm
    locations = xi = 0, 0.3, 0.6

OPTICS
    PLANAR_2D OptiX
    one unloaded reference per morphology
    48 loaded optical states

STATE METRIC
    J_contact = lateral contact/no-contact L1 separation / launched weight

TRAJECTORY METRIC
    depth-normalized AUC of J_contact

OPTIMIZATION SCORE
    minimum trajectory AUC across diameter/location

NO
    arbitrary weighted objective
    camera/noise model
    J2 optimization
    state-wise relative normalization
    case-specific contact tuning
```

## 15. Implementation Contract

The production evaluator implements the protocol directly:

```text
diameter + location
        ->
one 0-to-2-mm FEA trajectory
        ->
four captured indentation states
        ->
four PLANAR_2D OptiX evaluations
```

The evaluator performs one unloaded reference trace, twelve 48-step FEM
trajectories, and forty-eight loaded `PLANAR_2D` traces. It aggregates each
trajectory with depth-normalized AUC and maximizes the minimum trajectory AUC.
The old adjacent-state `ScenarioPair` separability objective is not part of
the production search.

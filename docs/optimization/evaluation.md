# Production morphology evaluation

Each candidate morphology is evaluated as one optomechanical experiment. The
unloaded `PLANAR_2D` OptiX trace is computed once. Each diameter/location pair
then receives one monotonic FEM trajectory from 0 to 2 mm using 48 steps; the
converged states at steps 12, 24, 36, and 48 are the four loaded optical
states. The protocol therefore has 12 trajectories and 48 loaded states.

## Fixed protocol

The active morphology vector is

```text
q = (flat_pad_height, stem_width, stem_height, void_width)
```

The fixed envelope and interface condition are:

```text
flat_pad_width = 30 mm
flat_pad_height + semielliptical_pad_height = 14 mm
semielliptical_pad_height = 14 - flat_pad_height
void_height = 0 mm
basal_interface = bonded
internal_contact = sides_separate
```

The loading grid is:

```text
diameters:       6, 10, 14, 20 mm
radii:           3, 5, 7, 10 mm
locations:       x = 0, 4.5, 9.0 mm
depth captures:  0.5, 1.0, 1.5, 2.0 mm
```

The production contact object is an explicit ideal absorbing boundary,
`IndenterOptics("absorber")`, representing a smooth bulk-black rigid polymer
cylindrical probe. This is a transport abstraction rather than a claim of
exactly zero physical reflectance; all physical probes should share the same
material and smooth contact finish.

Locations are one side of the symmetric fingertip. Opposite-side cases are a
separate symmetry check, not additional production optimization variables.

## Contact metric

For each loaded state, let \(\phi_L\) and \(\phi_R\) be the lateral outgoing
profiles integrated over the extrusion coordinate. Profiles use the same
surface-u bins in the unloaded and loaded results. With one shared launch
weight \(W_{launch}\),

\[
J_{contact}(d,D,x) =
\frac{\|\phi_L^{loaded}-\phi_L^{ref}\|_1+
      \|\phi_R^{loaded}-\phi_R^{ref}\|_1}{W_{launch}}.
\]

There is no state-wise percentage normalization, camera term, `J2`, or
empirical weighting. Throughput, quadrant redistribution, object-energy
channels, contact provenance, and reactions are diagnostics only.

## Trajectory aggregation

The depth-robust score for a diameter/location trajectory is

\[
A(D,x)=\frac{1}{2\,mm}\int_0^{2\,mm}J_{contact}(d,D,x)\,\mathrm{d}d.
\]

The implementation inserts \(J(0)=0\) and uses the trapezoid rule at
`[0, 0.5, 1.0, 1.5, 2.0]` mm. The morphology score is

\[
J_{morph}=\min_{D,x}A(D,x).
\]

The search backend maximizes `minimum_auc`; a minimization backend must use the
explicit cost `-minimum_auc`.

All 48 raw state metrics and all 12 AUC values are retained in the neutral
`DesignEvaluation`. `limiting_trajectory` identifies the trajectory with the
minimum AUC; `minimum_raw_contact_state` independently identifies the state
with the minimum raw `J_contact`.

## Exact-state provenance

`fem.solve()` exposes each captured state as `FEACapturedState`. It contains the
exact captured displacement/deformed pad mesh, reaction, indenter pose,
external active-node patch, internal side-contact node provenance, and contact
diagnostics from that converged FEM step. The evaluator passes that state to
`trace_3d()`; it does not rerun FEM independently at each depth.

The transport result exposes `lateral_outgoing_profiles()` as display/metric
post-processing of raw escaped events. It does not alter transport physics or
the arrays used by mechanics.

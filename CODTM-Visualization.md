# CODTM Spatial Visualization

Phase 4K-Viz turns the immutable Phase 4K contact-to-observation deformation
transfer map into static, source-data-backed figures. It is a post-processing
phase: no Kratos solve, geometry optimization, contact retuning, or optical
simulation is performed.

## Data and coordinate contract

The canonical inputs are the NPZ, long-form CSV, metadata, summary,
validation, case summary, and source trace in
`output/phase4_mechanical_transfer_map/`. `map_metadata.json` supplies array
axes, case order, side order, units, and validity semantics. The loader checks
the declared NPZ shapes, all 31,488 CSV samples, finite values on every valid
step, and SHA-256 checksums before and after rendering.

The scientific coordinate is `(side, eta)`, not one continuous crown arc.
Within each independent sidewall material chain, `eta=0` is the bonded endpoint
and `eta=1` is the crownward observation endpoint. The observation intervals
cover only right full-arc `xi≈0→.25` and left full-arc `xi≈1→.75`; their two
`eta=1` endpoints are different material points.

Combined heatmaps use visualization-only coordinates

```text
zeta_right = eta - 1
zeta_left  = 1 - eta
```

with a visible rendered gap between `0-` and `0+`. Metrics remain on
`(side, eta)`. The code never averages the endpoints, draws a line across the
gap, assigns the gap a fictitious integration length, or infers a deformed
contact-facing curve.

## Field and sampling rules

`u_normal` is the raw outward-positive displacement in mm. `G_secant` is the
stored normalized field `u_normal/delta`; `G_tangent` is the stored Phase 4K
finite difference, independently reproduced with centered interior and
one-sided endpoint differences. Neither normalized field is substituted for
the raw field in the main atlas.

The representative indentations are `.25/.50/1.00/1.50 mm`. An exact valid
step is preferred. If it is absent, linear interpolation is allowed only
between two finite, converged steps of the same case. Extrapolation and
interpolation across an invalid step are rejected and every selection/bracket
is recorded in `plot_manifest.json`. Contact location is never interpolated or
smoothed: medium figures contain exactly the five solved
`.20/.35/.50/.65/.80` rows.

## Metrics

Raw location distance is

```text
D_ij = sqrt(sum_side integral_eta (b_i - b_j)^2 d eta).
```

Each side integral is evaluated separately before summation. Shape distance
first divides each signature by the same sidewise integral norm; a zero or
non-finite norm is an error. At `1.5 mm`, the reproduced off-diagonal medium
ranges are `0.265411–0.982068 mm` raw and `0.552584–1.968494` after amplitude
normalization.

Mirror comparison maps `right(eta; xi)` to `left(eta; 1-xi)` and vice versa,
with eta preserved. The `.20/.80`, `.35/.65`, and `.50/.50` relative L2
residuals are `0.0670%`, `0.1498%`, and `0.0942%`; the diagnostic is
`CONSISTENT`. This label is a geometric/numerical visualization diagnostic,
not an optical detection threshold.

Medium/fine profile comparisons at `.20/.50/.80` reproduce relative L2
differences of `0.635%/0.255%/0.659%`, maximum absolute differences of
`0.00549/0.00344/0.00592 mm`, and correlations above `0.99996`. The status
remains `PROVISIONAL`, because Phase 4K did not predeclare a CODTM profile
acceptance threshold.

## Validity semantics

The displacement field uses Phase 4K `valid_mask`: the nonlinear step must be
converged, finite, and have positive det(F). All 384 case/step records meet
that condition, so the sidewall figures include medium `.65/.80`.

Contact centroid and length have a different contract. Only 256 of 384 records
are force-closure verified; unverified values are NaN. Phase 4K-Viz does not
zero-fill, interpolate, mirror-reconstruct, or display those values as
verified. Pressure closure does not invalidate a finite displacement
signature.

## Figures and scientific questions

| Figure | Scientific question |
|---|---|
| `codtm_overview` | What is measured, where are the independent chains, and how do contact locations separate at a glance? |
| `codtm_spatial_atlas` | How does the raw spatial signature grow across indentation and change with commanded contact location? |
| `sidewall_deformation_delta_1p50mm` | What does the measured 1× displacement mean in physical x-y coordinates? |
| `codtm_profiles_delta_1p50mm` | Which eta regions carry the location-dependent outward/inward response on each side? |
| `codtm_secant_gain_atlas` | Which profile changes remain after removing indentation amplitude by an explicit secant normalization? |
| `location_distance_matrices` | Which measured contact-location pairs are closest or farthest in raw mechanical signature? |
| `shape_distance_delta_1p50mm` | Does location encoding remain after amplitude normalization? |
| `mirror_symmetry_delta_1p50mm` | Is the expected side-swapped, eta-preserving response retained? |
| `tangent_transfer_gain` | How does incremental transfer change with indentation without smoothing? |
| `medium_fine_profiles_delta_1p50mm` | How mesh-dependent are the measured profiles at the three common locations? |

Every figure has a numerical CSV source under
`output/phase4_codtm_visualization/figure_data/`, and every PNG has a vector
PDF counterpart. `plot_manifest.json` records inputs, cases, mesh, indentation
selection, variables, units, limits, validity rule, sources, output checksums,
and paths.

## Verdict

```text
Data ingestion:                 PASS
Coordinate semantics:           PASS
Static visualization pipeline:  PASS
Metric reproduction:            PASS
Mirror symmetry:                CONSISTENT
Medium/fine visualization:      PROVISIONAL
Publication readiness:          READY
Optical observability:          NOT EVALUATED
```

The figures establish spatial mechanical encoding in the sampled CODTM. They
do not establish optical observability, sensor resolution, noise robustness,
or an optimized fingertip design.


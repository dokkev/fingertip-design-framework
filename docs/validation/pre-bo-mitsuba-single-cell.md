# Pre-BO Mitsuba single-LED cell validation

## Purpose

This experiment compares nominal morphology with exact pre-BO sweep candidate
49 using one representative longitudinal LED cell. It tests whether the 2D
morphology ranking survives 3D optical propagation through an 11 mm uniform
extrusion. The full 64.8 mm/five-LED configuration is intentionally excluded.

This validation uses an ideal orthographic side-field sampler rather than a
physical camera model. The deformation is two-dimensional and uniformly
extruded along the longitudinal direction; it is not full 3D contact mechanics.

## Protocol

- Git revision: 55e94f3f16459049d850004a24b7395e97c9ad6d
- Exact candidate source: output/validation/optimization/pre_bo_nominal_sweep/inputs/candidate_0049.json
- Nominal parameters: {"arc_resolution": 128, "bond_extension_height": 2.0, "bond_extension_width": 4.0, "flat_pad_height": 5.0, "flat_pad_width": 30.0, "geometry_tolerance": 1e-09, "link_thickness": 3.5, "poisson_ratio": 0.49, "semielliptical_pad_height": 9.0, "stem_height": 6.0, "stem_width": 7.6, "void_height": 0.0, "void_width": 1.0, "young_modulus_mpa": 0.55}
- Candidate 49 parameters: {"arc_resolution": 128, "bond_extension_height": 2.0, "bond_extension_width": 4.0, "flat_pad_height": 3.937175708822906, "flat_pad_width": 30.0, "geometry_tolerance": 1e-09, "link_thickness": 3.5, "poisson_ratio": 0.49, "semielliptical_pad_height": 7.309789158403873, "stem_height": 5.102298432029784, "stem_width": 7.289858109783381, "void_height": 1.2690955214202404, "void_width": 0.6931721470318735, "young_modulus_mpa": 0.55}
- Reduced-order TVs: nominal 0.0751066215, candidate 49 0.1273767967
- Extrusion: 11.0 mm, z bounds [-5.5, 5.5]; point LED at z=0.
- Exact source coordinates: {"candidate49": [[0.0, -4.102298432029784, 0.0]], "nominal": [[0.0, -5.0, 0.0]]}
- Source x=0 and y is the geometric center of each morphology's physical LED package.
- The cell represents the 9 mm package plus 2 mm gap, bounded at pitch midpoints.
- States: unloaded, left shallow (-3.0, 0.5, 4.0), right shallow (3.0, 0.5, 4.0).
- FEM: medium mesh, 48 steps, internal_contact=three_pairs.
- Optical material, LED RGB, and LED power are identical between morphologies.
- Fixed sampler: {"frame_margin_mm": 2.0, "orthographic_scale_mm": 21.5, "position_mm": [17.0, -5.25, 0.0], "projection": "orthographic", "resolution_px": [384, 1024], "target_mm": [0.0, -5.25, 0.0], "union_y_bounds_mm": [-14.0, 3.5], "union_z_bounds_mm": [-5.5, 5.5], "up": [0.0, 0.0, 1.0]}

## Results

| Morphology | Reduced 2D TV | Single-cell Mitsuba TV | Left energy | Right energy | Relative energy difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nominal | 0.0751066215 | 0.0203415651 | 770.47752 | 783.96482 | 0.017505115 |
| Candidate 49 | 0.1273767967 | 0.0169545162 | 706.92191 | 706.79342 | -0.00018175493 |

- 256 spp smoke status: success.
- Final spp: 8192.
- Same-state left-contact noise TV: nominal 0.018911381288943402, candidate 49 0.014043626915892374, maximum 0.018911381288943402.
- Absolute morphology TV gap: 0.0033870489173371543.
- Ranking preserved: False.
- SPP decision: The initial 2048 spp summary is retained at /home/dk/workspace/lit_ws/output/validation/optics/pre_bo_mitsuba_single_cell/initial_2048/summary.json. It measured nominal TV 0.03818167070660283 and candidate 49 TV 0.029884610409729367, with nominal noise 0.0375330769963681 and candidate 49 noise 0.027964609975610316.

## Artifacts

- Main comparison: /home/dk/workspace/lit_ws/output/validation/optics/pre_bo_mitsuba_single_cell/figures/nominal_vs_candidate49_light_fields.png
- Quantitative comparison: /home/dk/workspace/lit_ws/output/validation/optics/pre_bo_mitsuba_single_cell/figures/reduced_vs_mitsuba_tv.png
- Raw fields: /home/dk/workspace/lit_ws/output/validation/optics/pre_bo_mitsuba_single_cell/fields/
- Difference fields: /home/dk/workspace/lit_ws/output/validation/optics/pre_bo_mitsuba_single_cell/differences/

## Interpretation and limitations

The primary claim is ranking preservation between the reduced model and this
single-cell side-field measurement, not equality of absolute TV values. This
test excludes five-LED interaction, PCB end effects, longitudinal illumination
nonuniformity, camera orientation, and full fingertip length. It is not a
camera-performance result or a full-3D-mechanics result. Bayesian optimization
remains on hold until this single-cell result is interpreted.

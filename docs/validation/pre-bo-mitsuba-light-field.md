# Pre-BO Mitsuba 3D light-field validation

## Purpose

This validation compares the current nominal morphology with exact pre-BO
sweep candidate 49 using an extruded 2D deformation and a 3D volumetric
Mitsuba model with the actual five-LED longitudinal layout. Candidate 49 was
selected because it was the best successful point in the pre-BO sweep.

This validation uses an ideal orthographic side-field sampler rather than a
physical camera model. The deformation is two-dimensional and uniformly
extruded along the longitudinal direction.

## Provenance and designs

- Git revision: `55e94f3f16459049d850004a24b7395e97c9ad6d`
- Nominal parameters: `{"arc_resolution": 128, "bond_extension_height": 2.0, "bond_extension_width": 4.0, "flat_pad_height": 5.0, "flat_pad_width": 30.0, "geometry_tolerance": 1e-09, "link_thickness": 3.5, "poisson_ratio": 0.49, "semielliptical_pad_height": 9.0, "stem_height": 6.0, "stem_width": 7.6, "void_height": 0.0, "void_width": 1.0, "young_modulus_mpa": 0.55}`
- Candidate 49 parameters: `{"arc_resolution": 128, "bond_extension_height": 2.0, "bond_extension_width": 4.0, "flat_pad_height": 3.937175708822906, "flat_pad_width": 30.0, "geometry_tolerance": 1e-09, "link_thickness": 3.5, "poisson_ratio": 0.49, "semielliptical_pad_height": 7.309789158403873, "stem_height": 5.102298432029784, "stem_width": 7.289858109783381, "void_height": 1.2690955214202404, "void_width": 0.6931721470318735, "young_modulus_mpa": 0.55}`
- Reduced-order nominal TV: `0.0751066215`
- Reduced-order candidate 49 TV: `0.1273767967`

## Optical and mechanical protocol

- Extrusion length: `64.8 mm`, centered at `z=0`.
- LED z positions: `[-19.9, -8.9, 2.1, 13.1, 24.1]`.
- LED source positions are package centers with `x=0`; the exact per-design
  coordinates are recorded in `summary.json`.
- All five LEDs are on simultaneously with identical default LED RGB and
  relative power. They are separate point sources, not one five-times source.
- Contact states: unloaded, left shallow `(-3.0, 0.5, 4.0)`, and right
  shallow `(3.0, 0.5, 4.0)` in `(location_x_mm, indentation_mm,
  indenter_radius_mm)`.
- FEM: medium mesh, `fem_steps=48`, `internal_contact=three_pairs`.
- Optical material and LED properties are identical between designs.
- Sampler configuration is fixed across both morphologies and all states:
  `{"frame_margin_mm": 2.0, "orthographic_scale_mm": 68.8, "position_mm": [17.0, -5.25, 0.0], "projection": "orthographic", "resolution_px": [384, 1024], "target_mm": [0.0, -5.25, 0.0], "union_y_bounds_mm": [-14.0, 3.5], "union_z_bounds_mm": [-32.4, 32.4], "up": [0.0, 0.0, 1.0]}`

## Results

| Morphology | Reduced 2D TV | Mitsuba side-field TV | Left energy | Right energy | Relative energy difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nominal | 0.0751066215 | 0.0247539256 | 781.53614 | 788.17608 | 0.0084960155 |
| Candidate 49 | 0.1273767967 | 0.0193067481 | 693.64723 | 687.44504 | -0.0089414264 |

- 256 spp smoke renders: `success`.
- The initial 2048 spp comparison produced nominal TV `0.0475354843`,
  candidate-49 TV `0.0369228490`, and same-state noise TV `0.0470995254`.
  Because the morphology gap was below that noise floor, the final comparison
  was rerun at 8192 spp before interpretation; the retained raw artifacts are
  the 8192 spp fields below.
- 8192 spp final renders: `success`.
- Same-state repeated loaded render TV noise floor: `0.02404741468622683`.
- Absolute nominal-versus-candidate Mitsuba TV gap: `0.0054471776`.
- Final render wall times and FEM wall times are recorded in `summary.json`.
- Ranking preserved: **False**.

## Figures and artifacts

- Main comparison: `/home/dk/workspace/lit_ws/output/validation/optics/pre_bo_mitsuba_light_field/figures/nominal_vs_candidate49_light_fields.png`.
- Quantitative comparison: `/home/dk/workspace/lit_ws/output/validation/optics/pre_bo_mitsuba_light_field/figures/reduced_vs_mitsuba_tv.png`.
- Raw final linear RGB and scalar fields: `output/validation/optics/pre_bo_mitsuba_light_field/fields/`.
- Absolute difference fields: `output/validation/optics/pre_bo_mitsuba_light_field/differences/`.

## Interpretation and limitations

The primary claim is ranking preservation between the two morphologies, not
agreement of absolute reduced-order and 3D TV values. The side-field sampler
is an ideal numerical orthographic field sampler, not a calibrated camera or
lens model. The mechanical state is an extruded 2D FEM deformation, not full
3D contact mechanics. No optical calibration or parameter tuning was applied
to candidate 49, and no BO configuration was changed. In this run the
candidate-49 Mitsuba TV is below nominal, and the absolute morphology gap is
smaller than the repeated loaded-state noise floor; the reduced-order ranking
is therefore not supported by this 3D side-field measurement.

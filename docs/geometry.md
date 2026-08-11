# Fingertip geometry

The model uses millimetres, places the link-pad interface at `y = 0`, and uses
negative `y` as the distal direction. `FingertipParameters` separates the
dimensions of four physical components:

| Component | Width | Height | Default |
| --- | --- | --- | --- |
| Rigid stem | `stem_width` (`w_s`) | `stem_height` (`h_s`) | 7.6 × 6 mm |
| Vertical pad | `vertical_pad_width` (`w_vp`) | `vertical_pad_height` (`h_vp`) | 20 × 3 mm |
| Semi-elliptical pad | `semielliptical_pad_width` (`w_sp`) | `semielliptical_pad_height` (`h_sp`) | 20 × 7 mm |
| Void | `void_width` (`w_v`, one side) | `void_height` (`h_v`, below the stem tip) | 0 × 0 mm |

`stem_height` and `vertical_pad_height` are independent. The default geometry
uses `stem_height = 6 mm` and `vertical_pad_height = 3 mm` so this distinction
is visible.

## Mechanical pad material

`FingertipParameters` also owns the compliant pad material values used by the
production FEM path:

| Parameter | Meaning | Default |
| --- | --- | --- |
| `young_modulus_mpa` | pad Young's modulus [MPa] | `1.0` |
| `poisson_ratio` | pad Poisson ratio [-] | `0.49` |

`young_modulus_mpa = 1.0` is the current FEM baseline/default and is not an
experimentally calibrated LIT silicone modulus. `poisson_ratio = 0.49` is the
current validated nearly-incompressible FEM baseline. The production solver
reads these values through the meshed fingertip; the rigid carrier and
indenter remain backend-constrained parts.

## Central derived coordinates

`FingertipParameters` owns the coordinates and dimensions consumed by the
Shapely model, Gmsh adapter, and geometry visualization:

```text
ellipse_start_y = -vertical_pad_height
stem_tip_y      = -stem_height
void_bottom_y   = -(stem_height + void_height)
pad_tip_y       = -(vertical_pad_height + semielliptical_pad_height)

cutout_width    = stem_width + 2*void_width
cutout_height   = stem_height + void_height
total_pad_depth = vertical_pad_height + semielliptical_pad_height
```

## CSG construction

The complete outer envelope is constructed first as the union of the vertical
rectangle

```text
[-w_vp/2, w_vp/2] × [-h_vp, 0]
```

and the lower semi-ellipse

```text
x(theta) = (w_sp/2) cos(theta)
y(theta) = -h_vp - h_sp sin(theta),  theta in [0, pi].
```

The centered cutout is the top-open rectangle

```text
[-cutout_width/2, cutout_width/2] × [-cutout_height, 0].
```

Material and visible clearance then have one explicit definition:

```text
outer_pad    = vertical_rectangle union lower_semiellipse
cutout       = centered_top_open_rectangle
pad_material = outer_pad - cutout
rigid_link   = link_plate union stem
void         = cutout - stem
```

The two component widths must agree within `geometry_tolerance`; unequal
widths would need a transition geometry and are rejected. Values that differ
only within tolerance are snapped to the vertical-pad endpoints during
sampling, so no numerical shoulder is introduced. The full cutout must remain
inside the completed outer envelope. A side-clearance and a bottom-clearance
that are valid separately can therefore be invalid in combination; validation
rejects such a cutout rather than clipping it.

The model exposes the vertical sidewalls as `pad_outer_left` and
`pad_outer_right`, and the curved distal boundary as `pad_outer_arc`. Gmsh
samples and tags these Shapely boundaries rather than rebuilding the geometry.

## Legacy configuration migration

The obsolete `pad_width` and `pad_height` constructor arguments are not
aliases. Migrate an old mapping explicitly:

```python
parameters = FingertipParameters.from_legacy_mapping(
    old_configuration,
    vertical_pad_height=3.0,
)
```

The old `pad_width` maps to both new component widths because width continuity
is required. The old `pad_height` represented the ellipse semi-axis and maps
only to `semielliptical_pad_height`. A value for the new independent
`vertical_pad_height` is mandatory; it cannot be inferred from `stem_height`.
Solver and visualization artifacts generated with the old geometry must be
regenerated because their mesh topology and sampled coordinates describe a
different external envelope.

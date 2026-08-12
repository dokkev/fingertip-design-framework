# Fingertip geometry

The model uses millimetres, places the pad/link interface at `y = 0`, and
uses negative `y` as the distal direction. The canonical fingertip has one
shared full width for the flat pad, the semi-ellipse, and the rigid link.
Compliant bond extensions occupy two lower-corner recesses in the rigid link.

## Canonical geometry parameters

`FingertipParameters` directly owns these geometry dimensions:

| Component | Width | Height | Default |
| --- | --- | --- | --- |
| Flat pad | `flat_pad_width` (`w_fp`) | `flat_pad_height` (`h_fp`) | 20 × 3 mm |
| Semi-ellipse | `w_ep = w_fp` | `semielliptical_pad_height` (`h_ep`) | 20 × 7 mm |
| Rigid link plate | `w_l = w_fp` | `link_thickness` (`h_l`) | 20 × 3.5 mm |
| Bond extension | `bond_extension_width` (`w_cp`) | `bond_extension_height` (`h_cp`) | 4 × 2 mm |
| Rigid stem | `stem_width` (`w_s`) | `stem_height` (`h_s`) | 7.6 × 6 mm |
| Void | `void_width` (`w_v`, one side) | `void_height` (`h_v`, below the stem tip) | 0 × 0 mm |

The mandatory width identity is:

```text
w_l = w_fp = w_ep
```

There is no independent constructor dimension for the rigid-link width or
the semi-ellipse width. `flat_pad_width` is the single source of truth for
all three full widths. `bond_extension_width` is independent and is not
derived from a width difference.

The physical shoulder bolt used for secondary assembly retention is not part
of the parametric geometry, mesh, FEM, optics, or visualization model.

## Derived coordinates

`FingertipParameters` owns the coordinates consumed by the Shapely model,
Gmsh adapter, and geometry visualization:

```text
ellipse_start_y = -flat_pad_height
stem_tip_y      = -stem_height
void_bottom_y   = -(stem_height + void_height)
pad_tip_y       = -(flat_pad_height + semielliptical_pad_height)

cutout_width    = stem_width + 2*void_width
cutout_height   = stem_height + void_height
total_pad_depth = flat_pad_height + semielliptical_pad_height
```

`total_pad_depth` measures only the distal pad depth from `y = 0`; it does not
include the proximal bond-extension height.

## Compliant outer pad

The compliant outer envelope is the union of four pieces:

```text
flat_pad = [-w_fp/2, w_fp/2] × [-h_fp, 0]

semiellipse:
    x(theta) = (w_fp/2) cos(theta)
    y(theta) = -h_fp - h_ep sin(theta), theta in [0, pi]

left_extension  = [-w_fp/2, -w_fp/2 + w_cp] × [0, h_cp]
right_extension = [ w_fp/2 - w_cp, w_fp/2] × [0, h_cp]

outer_pad = flat_pad union semiellipse union left_extension union right_extension
```

The ellipse endpoints exactly meet the lower corners of the flat pad. The
complete compliant pad is one connected valid polygon before the centered
cutout is removed.

## Rigid link with bonding recesses

Construct a full-width rigid plate:

```text
x in [-flat_pad_width/2, +flat_pad_width/2]
y in [0, link_thickness]
```

Remove the two lower outer recesses:

```text
left recess:
    x in [-flat_pad_width/2,
          -flat_pad_width/2 + bond_extension_width]
    y in [0, bond_extension_height]

right recess:
    x in [+flat_pad_width/2 - bond_extension_width,
          +flat_pad_width/2]
    y in [0, bond_extension_height]
```

Then:

```text
link_plate = full_link_plate - left_recess - right_recess
rigid_link = link_plate union stem
```

The compliant bond extensions occupy the two recesses. Rigid and compliant
material do not overlap; they meet only on their bonded boundaries.

## Stem cutout and void

The centered, top-open cutout is:

```text
cutout       = [-cutout_width/2, cutout_width/2]
               × [-(h_s + h_v), 0]
pad_material = outer_pad - cutout
void         = cutout - stem
```

The stem clearance remains meaningful to both mechanics and optics. The
extension surfaces are bonded pad material, not contact surfaces.

## Three-segment bonded interfaces

Each side has one connected, three-segment perfectly bonded interface:

1. horizontal underside bond,
2. vertical recess-wall bond,
3. horizontal recess-top bond.

For the left side:

```python
pad_bond_left = LineString([
    (-cutout_half_width, 0.0),
    (-flat_pad_width / 2 + bond_extension_width, 0.0),
    (-flat_pad_width / 2 + bond_extension_width,
     bond_extension_height),
    (-flat_pad_width / 2,
     bond_extension_height),
])
```

The right side is its exact mirror, represented by the canonical
`pad_bond_right` semantic tag. The three legs remain one public boundary per
side; they are not split into additional semantic names.

The total analytic bonded length is:

```text
flat_pad_width - cutout_width + 2*bond_extension_height
```

The external shell tags remain `pad_bond_left`, `pad_outer_left`,
`pad_outer_arc`, `pad_outer_right`, and `pad_bond_right`. They traverse both
recess interfaces, the outer sidewalls, and the lower semi-ellipse, remaining
open only at the cutout mouth for loaded optical-domain closure.

## Mechanical material

`FingertipParameters` also owns the compliant-pad material values used by the
production FEM path:

| Parameter | Meaning | Default |
| --- | --- | --- |
| `young_modulus_mpa` | pad Young's modulus [MPa] | `1.0` |
| `poisson_ratio` | pad Poisson ratio [-] | `0.49` |

These are the current validated FEM baseline values, not an experimentally
calibrated silicone characterization. The rigid carrier and indenter remain
backend-constrained parts.

## Validation checklist

- [ ] The rigid-link width is not an independent parameter.
- [ ] The semi-ellipse width is not an independent parameter.
- [ ] `flat_pad_width` is the single width source for link, flat pad, and ellipse.
- [ ] `bond_extension_width` is an independent parameter.
- [ ] The rigid link contains two symmetric lower-corner recesses.
- [ ] Compliant extensions occupy those recesses without material overlap.
- [ ] Each pad bond is one connected three-segment line.
- [ ] `bond_extension_height < link_thickness` is enforced.
- [ ] `2*bond_extension_width + cutout_width < flat_pad_width` is enforced.
- [ ] Total bond length is `flat_pad_width - cutout_width + 2*bond_extension_height`.

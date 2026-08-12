# Fingertip geometry

The model uses millimetres, places the pad/link interface at `y = 0`, and
uses negative `y` as the distal direction. The canonical fingertip consists
of a wide compliant flat pad, a lower semi-ellipse, two proximal compliant
bond extensions, and a narrower rigid link with a centered stem and cutout.

## Public dimensions

| Component | Width | Height | Default |
| --- | --- | --- | --- |
| Flat pad | `flat_pad_width` (`w_fp`) | `flat_pad_height` (`h_fp`) | 20 × 3 mm |
| Semi-ellipse | `w_ep = w_fp` | `semielliptical_pad_height` (`h_ep`) | 20 × 7 mm |
| Rigid link plate | `link_width` (`w_l`) | `link_thickness` (`h_l`) | 12 × 3.5 mm |
| Rigid stem | `stem_width` (`w_s`) | `stem_height` (`h_s`) | 7.6 × 6 mm |
| Void | `void_width` (`w_v`, one side) | `void_height` (`h_v`, below the stem tip) | 0 × 0 mm |
| Bond extension | `bond_extension_width` (`w_cp`) | `bond_extension_height` (`h_cp`) | 4 × 2 mm |

The semi-ellipse width is not an independent parameter:

```text
w_ep = w_fp
w_cp = (w_fp - w_l) / 2
```

`flat_pad_width` must be greater than `link_width`, and `bond_extension_height`
must not exceed `link_thickness`. The centered cutout must be narrower than the
rigid link. The physical shoulder bolt used for secondary assembly retention is
not part of these dimensions or of the parametric model.

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

## Canonical construction

The compliant outer envelope is the union of four analytic pieces:

```text
flat_pad = [-w_fp/2, w_fp/2] × [-h_fp, 0]

semiellipse:
    x(theta) = (w_fp/2) cos(theta)
    y(theta) = -h_fp - h_ep sin(theta), theta in [0, pi]

left_extension  = [-w_fp/2, -w_l/2] × [0, h_cp]
right_extension = [ w_l/2,  w_fp/2] × [0, h_cp]

outer_pad = flat_pad union semiellipse union left_extension union right_extension
```

The ellipse endpoints are exactly the lower corners of the flat pad, so the
rectangle and curve have no numerical shoulder. The rigid geometry is:

```text
link_plate = [-w_l/2, w_l/2] × [0, link_thickness]
stem       = [-w_s/2, w_s/2] × [-h_s, 0]
rigid_link = link_plate union stem
```

The centered, top-open cutout is:

```text
cutout       = [-cutout_width/2, cutout_width/2]
               × [-(h_s + h_v), 0]
pad_material = outer_pad - cutout
void         = cutout - stem
```

The stem clearance remains meaningful to both mechanics and optics. The
extension surfaces are bonded pad material, not contact surfaces.

## L-shaped bonded interface

The two yellow interface surfaces in the physical design are represented by
the semantic boundaries `pad_bond_left` and `pad_bond_right`. Each is one
connected L-shaped line: a horizontal bond under the rigid link plus a
vertical bond along its outside wall. The complete interface is always
perfectly bonded to the fixed rigid carrier in the current FEM idealization.

```text
left:  (-cutout_width/2, 0) -> (-w_l/2, 0) -> (-w_l/2, h_cp)
right: ( w_l/2, h_cp) -> ( w_l/2, 0) -> ( cutout_width/2, 0)
```

Each side has length

```text
h_cp + (w_l - cutout_width)/2
```

and the total bonded length is
`2*h_cp + w_l - cutout_width`.

The external shell tags remain `pad_bond_left`, `pad_outer_left`,
`pad_outer_arc`, `pad_outer_right`, and `pad_bond_right`. They traverse both
extension tops and the sidewalls before following the lower semi-ellipse, and
remain open only at the cutout mouth for loaded optical-domain closure.

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

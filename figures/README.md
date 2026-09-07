# LUMO paper figure style

Paper figures use stable display labels and colors even when raw-data keys or
directory names retain older internal identifiers.

## Morphology names

| Internal key | Paper-facing label |
| --- | --- |
| `baseline` | Baseline |
| `flat_opt` | Opt-Flat |
| `angled_opt` | Opt-Curved |

Material labels are `Solaris` and `Dragon Skin`. Combined row labels therefore
read `Solaris Baseline`, `Solaris Opt-Flat`, `Solaris Opt-Curved`, `Dragon Skin
Baseline`, `Dragon Skin Opt-Flat`, and `Dragon Skin Opt-Curved`.

## Morphology identity colors

| Morphology | Color | Role |
| --- | --- | --- |
| Baseline | `#BFC3C7` | neutral light gray |
| Opt-Flat | `#2C758E` | viridis/teal-blue |
| Opt-Curved | `#D97707` | optimization orange |

Morphology marks use `#4C5055` for a neutral edge or border when one is needed.

## Semantic colors

| Meaning | Color |
| --- | --- |
| Rigid carrier | `#59616B` |
| Deformable pad | `#E0D797` |
| LED / optical source | `#009E73` |
| Force / mechanics highlight | `#D62728` |

Morphology identity and semantic colors are separate visual channels. For
example, a morphology bar or marker must not use LED green.

## Policy

- Morphology comparisons always use Baseline = gray, Opt-Flat = teal-blue, and
  Opt-Curved = orange.
- Figure 5, Figure 6, and appendix figures use the same labels and colors.
- Internal data keys and directory names may remain unchanged, but all
  paper-facing labels use the mappings above.
- Shared mappings live in `lumo.visualization.style`; figure code imports those
  constants rather than redefining the palette.

# LUMO paper figure style

This file is the **paper-facing style contract** for figures under `figures/`.
All final manuscript figures should follow these rules unless a panel has a clear
scientific reason to deviate. When an exception is necessary, keep it local and
comment the reason in the figure script.

The executable source of shared dimensions, typography, and colors is
`lumo.visualization.style`. Figure scripts should import those values rather than
redefining them locally.

## 1. Figure dimensions and export

| Item | Required value |
| --- | --- |
| IEEE single-column width | **3.50 in** |
| IEEE double-column width | **7.16 in** |
| PNG resolution | **600 dpi** |
| PDF font embedding | Type 42 (`pdf.fonttype = 42`) |
| Background | white |

- Build figures at their **final manuscript width**. Do not make an oversized
  figure and rely on LaTeX to shrink it substantially.
- Save the final manuscript figure as both **PDF** and **PNG**.
- Prefer vector-native Matplotlib artists for plots, text, arrows, and labels.
- Do not use `bbox_inches="tight"` for final paper figures when it changes the
  requested physical width. Use explicit figure margins and `pad_inches=0`.
- Remove dead space with `GridSpec` geometry and explicit margins rather than by
  shrinking fonts.

## 2. Typography

### Font family

Use **Helvetica Light** as the default paper-figure typeface for prose, axis
labels, ticks, legends, and ordinary annotations. The light weight is an
intentional part of the LUMO visual style; structural elements such as panel
labels may still use bold weight as specified below.

Preferred fallback order:

1. `Helvetica Light`
2. `Helvetica`
3. `Arial`
4. `Liberation Sans`
5. `DejaVu Sans`

- Keep ordinary figure text visually light and consistent across all panels.
- Do not mix Helvetica Light with unrelated display fonts inside a figure.
- If Helvetica Light is unavailable, use the fallback chain above rather than
  changing the figure's typography locally.
- Always inspect the final-size PDF: if a dense annotation becomes illegible,
  increase its size before changing the global font family or weight.

### Font sizes at final figure size

| Element | Size | Weight/style |
| --- | ---: | --- |
| Panel label `(a)`, `(b)`, ... | **9 pt** | Helvetica Light, upright |
| Panel title | **8 pt** | Helvetica Light, upright |
| Axis label | **8 pt** | Helvetica Light, upright |
| Base plot text | **8 pt** | Helvetica Light, upright |
| Tick labels | **7 pt** | Helvetica Light, upright |
| Legend text | **7 pt** | Helvetica Light, upright |
| Material/group header | **7 pt** | bold only when it defines a major group |
| Secondary condition header | **7 pt** | Helvetica Light, upright |
| Annotation / in-panel note | **6.5 pt** | Helvetica Light, upright |
| Absolute minimum paper-facing text | **6 pt** | use only when unavoidable |

These sizes refer to the **final exported figure**, not an enlarged working
preview.

### Bold

Bold is a structural cue, not an emphasis style.

Use bold only for major row/column group headers when needed to distinguish
hierarchy, e.g.
  **Solaris** and **Dragon Skin**.

Do **not** bold:

- panel-title prose;
- axis labels;
- tick labels;
- legend labels;
- data annotations;
- conclusions or keywords inside a plot.

Ordinary prose remains Helvetica Light. When a bold structural element is
needed, use the corresponding Helvetica/Arial bold face rather than trying to
make the light face carry visual emphasis.

The current environment provides only the Helvetica Light face. Therefore the
shared style uses Liberation Sans Bold, an Arial-compatible fallback, only for
bold group headers. Panel labels and ordinary text remain Helvetica Light.

### Italic

- Do not italicize prose labels, titles, legends, or annotations.
- Mathematical variables use normal mathematical italic automatically, e.g.
  `$Q_{\mathrm{recontact}}$`.
- Units, words, and operators inside mathematical labels should remain upright.
- Do not use italics as a generic emphasis device.

## 3. Panel titles and panel labels

Panel titles should identify **what is shown**, not state the conclusion.

Examples:

- `(a) Contact-state variability`
- `(b) Re-contact distinguishability`
- `(c) Spatial vs. scalar decoding`
- `(d) Calibration-set size`

Rules:

- Panel labels are lowercase letters in parentheses: `(a)`, `(b)`, ...
- Panel labels and title text use Helvetica Light; the 9 pt label remains
  distinct from the 8 pt title through size rather than weight.
- Center each panel title within the full width of its owning subfigure. The
  title center must follow the subfigure bounds, not the visible data-axis
  bounds of one nested subplot.
- Keep the panel label at the subfigure's upper-left edge; center only the title
  text. This preserves fast `(a)`, `(b)`, ... scanning while aligning titles
  consistently across heterogeneous panel layouts.
- Align titles to a common vertical baseline within each figure row.
- Use sentence case.
- Prefer **2--5 words**.
- Do not end panel titles with a period.
- Avoid sentence-like claims such as `Decoding improves without increased
  signal magnitude`; move interpretation to the caption or manuscript text.
- Shared column/group headers such as `10 mm`, `30 mm`, `Solaris`, or
  `Dragon Skin` remain independently centered over the groups they describe.

When Matplotlib cannot style the panel label and title independently with
`Axes.set_title`, place the larger panel label and light-weight title as separate
text artists rather than making the entire title bold.

## 4. Axis labels and ticks

Axis labels should be concise **quantity + unit** labels.

Preferred examples:

- `Localization accuracy [%]`
- `Variability change [%]`
- `Optical change [DN]`
- `Contacts / location`
- `$Q_{\mathrm{recontact}}$`

Rules:

- Use square brackets for units.
- Do not write sentence-like axis labels (`Change in ... compared with ...`).
- Put metric definitions and experimental qualifiers in the caption when the
  symbol itself is sufficient on the axis.
- Keep x/y labels light-weight and upright.
- Use shared axis labels for repeated small multiples when possible.
- Suppress repeated tick labels inside a grid if the row/column structure makes
  them redundant.
- Do not rotate tick labels unless horizontal labels genuinely do not fit.

## 5. Group headers and repeated conditions

Use layout to encode experimental structure instead of repeating long labels.

Preferred hierarchy for material/indenter grids:

```text
                 Solaris              Dragon Skin
              10 mm   30 mm         10 mm   30 mm
```

- Major material group headers may be bold.
- Condition headers (`10 mm`, `30 mm`) use Helvetica Light.
- Avoid repeated labels such as `Solaris · 10 mm sphere` on every subplot.
- If `10 mm` and `30 mm` are indenter diameters, define that once in the caption
  or use the diameter symbol when ambiguity is possible.

## 6. Morphology names and identity colors

Paper-facing labels remain stable even when raw-data keys or directory names use
older internal identifiers.

| Internal key | Paper-facing label | Color |
| --- | --- | --- |
| `baseline` | Baseline | `#BFC3C7` |
| `flat_opt` | Opt-Flat | `#2C758E` |
| `angled_opt` | Opt-Curved | `#D97707` |

Material labels are:

- `Solaris`
- `Dragon Skin`

Morphology marks use `#4C5055` for a neutral edge or border when one is needed.

### Morphology color policy

- Morphology comparisons always use **Baseline = gray**, **Opt-Flat =
  teal-blue**, and **Opt-Curved = orange**.
- Figure 5, Figure 6, and appendix figures use these same labels and colors.
- Do not redefine the morphology palette inside individual figure scripts.
- Internal data keys may remain unchanged; all paper-facing labels use the
  mappings from `lumo.visualization.style`.

## 7. Semantic colors

Morphology identity colors and physical/semantic colors are separate visual
channels.

| Meaning | Color |
| --- | --- |
| Rigid carrier | `#59616B` |
| Deformable pad | `#F2F1ED` |
| LED / optical source | `#009E73` |
| Force / mechanics highlight | `#D62728` |

Examples:

- A morphology bar or marker must not use LED green.
- Red is reserved for force/mechanics emphasis or a similarly explicit semantic
  role, not for arbitrary dataset identity.

## 8. Lines, markers, axes, and grids

Default quantitative-plot styling:

| Element | Default |
| --- | --- |
| Data line width | **1.2 pt** |
| Marker size | **4.5 pt** |
| Axis spine width | **0.7 pt** |
| Tick width | **0.7 pt** |
| Tick length | **2.5 pt** |
| Grid width | **0.5 pt** |
| Neutral axis/spine color | approximately `#777777` |
| Light grid color | approximately `#D8D8D8`--`#E3E3E3` |

- For standard quantitative plots, hide the **top and right spines** unless the
  panel type requires a closed frame.
- Use light horizontal grids only when they materially improve quantitative
  reading; do not grid image panels or confusion matrices by default.
- Use the same line width and marker scale across neighboring panels unless a
  distinct visual encoding requires otherwise.
- Error bars and uncertainty bands should be visually secondary to the central
  estimate.
- Avoid excessive marker outlines; use `EDGE_COLOR` only where separation from
  the background is needed.
- In Figure 5's shared table grammar, panel (a) owns the single figure-wide row
  header. Retain a narrow whitespace column between its rotated material label
  and per-row morphology label, and omit those repeated labels from panels (b)
  and (c). All three panels must retain identical row geometry.

## 9. Legends

- Legends use **7 pt Helvetica Light** text and no frame by default.
- Prefer a single shared legend when the same encoding is reused across panels.
- Do not include a legend if row/column headers already identify the condition.
- Place legends in unused data space when possible; avoid adding an entire row
  of whitespace above the figure solely for a legend.
- Keep legend labels short and paper-facing, e.g. `Scalar only` and
  `6-region spatial`.

## 10. Heatmaps and confusion matrices

- Use one shared color scale when panels are quantitatively comparable.
- Use a single shared colorbar per panel group rather than repeating colorbars.
- Cell annotations use Helvetica Light, normally **6.5--7 pt** at final size.
- Hide zero annotations when this improves readability without hiding relevant
  errors.
- If cell values are fractions (`0.8`, `1.0`), do not label the colorbar as
  percent. If values are percentages (`80`, `100`), label them as percent.
- Put `True`/`Predicted` or equivalent axis labels once per matrix group when
  possible.

## 11. Image panels

- Image panels should not show decorative axes, ticks, or frames.
- Use annotations only when they identify a physically relevant quantity such as
  contact position, force direction, or region of interest.
- Keep arrows and semantic overlays consistent with the semantic color palette.
- Do not use large text directly on image data when a row/column header can carry
  the same information.

## 12. Layout and whitespace

- Prefer **shared row/column structure** over repeated labels.
- Use whitespace to separate major groups, not every individual subplot.
- Keep panel-to-panel spacing compact but sufficient to avoid label collisions.
- For double-column multi-panel figures, align panels to a common top/bottom
  grid whenever possible.
- For single-column figures, preserve readable axis text before adding more
  subpanels; do not solve density by shrinking text below the minimum size.
- Use different panel heights when information density differs; not every panel
  needs equal height.

## 13. Matplotlib implementation policy

Final paper figure scripts should use:

```python
from lumo.visualization import DEFAULT_STYLE, publication_context

with publication_context(DEFAULT_STYLE):
    ...
```

and should import shared labels/colors from `lumo.visualization.style`.

Do not call global style packages or themes (`seaborn.set_theme`, arbitrary
`plt.style.use`, etc.) in final figure scripts because they can silently override
the publication style.

Avoid hard-coded font families, typography weights, and morphology colors inside
individual figure scripts. If a local font size is required for a dense
annotation, derive it from the publication hierarchy above and keep it at least
6 pt at final size.

## 14. Final visual check

Before accepting a figure:

1. Export at the exact final IEEE width.
2. Open the PDF at approximately manuscript display size, not only zoomed in.
3. Verify that 7 pt ticks and 6.5 pt annotations remain legible.
4. Check that panel labels are light and only major group headers are bold.
5. Check that prose titles/labels use Helvetica Light, are upright, and are not
   accidentally rendered bold.
6. Confirm that morphology colors and paper-facing names match this file.
7. Remove repeated labels, unnecessary legends, redundant colorbars, and dead
   whitespace.
8. Confirm that the panel title describes the measurement rather than making an
   unsupported conclusion.

## 15. Source and output organization

Every manuscript figure owns one directory named `figN`:

```text
figures/
  README.md
  fig2/
    fig2.py
    fig2.pdf
    fig2.png
  fig3/
    fig3.py
    fig3.pdf
    fig3.png
  fig5/
    fig5.py
    fig5.pdf
    fig5.png
  fig6/
    fig6.py
    fig6.pdf
    fig6.png
```

- Keep composition scripts, panel tools, audit CSVs, and final outputs under
  the directory of the figure that owns them.
- Do not place figure-specific scripts or generated artifacts directly under
  `figures/`, and do not recreate a generic `figures/output/` directory.
- Put non-production visual alternatives under the owning figure's
  `exploration/` directory, with names that distinguish them from the final
  manuscript output.
- Python bytecode and render-review files are temporary. Keep them out of the
  figure tree and remove them before handoff.
- Final Figure 5 uses the canonical `fig5.pdf` and `fig5.png` stems; obsolete
  standalone panel renders do not remain beside the manuscript output.

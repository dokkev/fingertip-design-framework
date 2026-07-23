# LIT Hand Geometry-Aware Scientific Figure Framework

## Purpose and non-goals

This framework renders repeatable scientific figures from FEM artifacts and a
declarative `FigureSpec`. The framework is the product; the transfer-map
comparison and displacement-vector atlas are reference use cases.

It does not run FEM, modify solvers, optimize geometry, invent an optimized
design, infer an unavailable displacement field, simulate optics, or promote a
scientific claim. A transfer map here is a geometry-conditioned
prescribed-indentation response, not a classical frequency- or s-domain
transfer function.

## Architecture

```text
artifact files
    │
    ▼
Data adapter ──► canonical semantic dataclasses
                        │
                        ▼
              scientific transforms
                        │
                        ▼
                  figure builder
                        │
             reusable rendering panels
                        │
                        ▼
        PNG + PDF + source CSV + manifest
```

- `visualization/data.py`: semantic types and repository adapters.
- `visualization/transforms.py`: projection, normalization, interpolation,
  mirror mapping, distances, metrics, and spatial sampling.
- `visualization/rendering.py`: themes, scale policy, reusable panels, layout,
  and deterministic export. Panels render to an axes and never save files.
- `visualization/framework.py`: `FigureSpec`, dataset composition, builders,
  and public API.
- `visualization/__main__.py`: CLI.

Scientific quantities are prepared before rendering. A panel never computes a
normal projection, gain, interpolation, or signature distance.

## Canonical data model

- `MeshData`: explicit node/element IDs, physical coordinates, connectivity,
  spatial dimension, design/mesh IDs, units, and provenance.
- `DisplacementField`: explicit sample identifiers, actual
  `u=[u_x,u_y]`, case/step/design/mesh IDs, represented configuration,
  validity, units, and provenance.
- `ObservationChain`: independent `left` or `right` material chain with eta,
  undeformed coordinates, unit outward normals, and interpolation provenance.
- `ContactCase`: xi, prescribed indentation, reaction, input direction,
  contact point, solve/CODTM validity, descriptor validity, and source.
- `TransferSignature`: side-specific eta, raw `u_normal`, tangential response,
  stored gains, indentation, reaction, validity, units, and provenance.

The adapter converts repository-specific arrays once. Array position is not
used as a substitute for case, side, eta, or component metadata.

## Phase 4K adapter and physical geometry

`load_phase4k_visualization_dataset()` reads the immutable Phase 4K NPZ, CSV,
and JSON artifacts. It checks declared axes/shapes, 31,488 long-form rows,
finite valid fields, and source checksums.

Phase 4K did not persist a complete full-volume nodal displacement snapshot.
The adapter therefore:

1. deterministically reconstructs the undeformed T3 mesh from recorded
   `FingertipModel` parameters and mesh settings;
2. verifies pad/carrier element counts against the Phase 4K result;
3. uses that mesh only as undeformed geometry context;
4. applies deformation and arrows only to stored observation-chain `u_x/u_y`;
5. never interpolates or colors an internal displacement field.

This reconstruction performs no FEM solve.

To support another artifact format, implement an adapter that returns
`VisualizationDataset`. Builders and rendering panels do not change.

## Scientific quantities

Raw outward displacement is

```text
u_normal(side, eta) = n_outward(side, eta)^T u(side, eta).
```

Positive is outward and negative is inward.

| FigureSpec quantity | Definition | Units |
|---|---|---|
| `raw_displacement` | `u_normal` | mm |
| `secant_gain` | `u_normal / delta` | mm/mm |
| `force_compliance` | `u_normal / F_reaction` | mm/N |
| `tangent_gain` | verified stored `du_normal/ddelta` | mm/mm |

Force compliance rejects missing, non-finite, or zero reaction. Stored secant
gain is checked against `u_normal/delta`.

Raw and shape-normalized distances integrate left and right eta chains
separately before summation. The display gap has no integration length.
Shape normalization has a zero-norm guard. Mirror mapping swaps sides while
preserving eta order.

## Physical versus display coordinates

The primary coordinate is `(side, eta)`. On both chains, eta runs from bonded
endpoint `0` toward crownward observation endpoint `1`. Right eta=1 and left
eta=1 are different material points.

Transfer panels may use display-only coordinates:

```text
zeta_right = eta - 1
zeta_left  = 1 - eta
```

The renderer inserts a neutral separator. It never connects, averages,
integrates across, or colors the unsampled contact-facing region.

## FigureSpec schema

The framework accepts strict JSON or JSON syntax in a YAML 1.2 file. JSON
syntax avoids adding a YAML dependency.

Important sections are:

- `datasets`: adapter, design ID, and artifact directory;
- `kind`, `designs`, `mesh`, `contact_locations`, and `indentation_mm`;
- `quantity`;
- explicit interpolation policy;
- shared color-scale policy;
- panel selection and theme;
- independent deformation/arrow scales and deterministic arrow policy;
- PNG/PDF/DPI/output settings.

Reference specs:

- `examples/transfer_map_comparison.yaml`
- `examples/displacement_vector_atlas.yaml`

The transfer reference uses single-design mode because no optimized artifact
exists. It is not presented as an optimization result.

## Python API

```python
from visualization import (
    load_figure_spec,
    load_visualization_dataset,
    render_figure,
)

spec = load_figure_spec("examples/transfer_map_comparison.yaml")
dataset = load_visualization_dataset(spec)
result = render_figure(dataset, spec)
```

For a real comparison, declare two actual data sources:

```json
{
  "datasets": [
    {"adapter": "phase4k", "design_id": "baseline", "input_dir": "path/to/base"},
    {"adapter": "phase4k", "design_id": "optimized", "input_dir": "path/to/optimized"}
  ],
  "figure": {
    "kind": "transfer_map_comparison",
    "designs": ["baseline", "optimized"]
  }
}
```

The remaining fields stay as shown in the reference spec. A single color limit
is computed over both designs.

## CLI

```bash
/home/dk/miniconda3/envs/lit/bin/python -B -m visualization \
  examples/transfer_map_comparison.yaml

/home/dk/miniconda3/envs/lit/bin/python -B -m visualization \
  examples/displacement_vector_atlas.yaml
```

`--output-dir PATH` overrides the spec output. Re-rendering overwrites matching
newer PNG/PDF/CSV/manifest files; unrelated files and source artifacts are not
removed.

## Reusable components and themes

The rendering layer provides `MeshPanel`, `DeformedMeshPanel`,
`DisplacementVectorPanel`, `ContactInputAnnotation`,
`ObservationBoundaryOverlay`, `TransferMapPanel`,
`LocationDistanceMatrixPanel`, `ProfilePanel`, `MetricSummaryPanel`,
`SharedColorbar`, `PanelLabelManager`, `FigureComposer`, and
`FigureExporter`.

`FigureTheme` has journal/blog presets for typography, lines, colormaps,
invalid data, mesh appearance, annotations, DPI, and PDF fonts.
`ScalePolicy` independently owns deformation scale, arrow scale, arrow
threshold, and color limits. Compared panels compute one common symmetric
limit before rendering and record it.

`DisplacementVectorPanel` uses deterministic spatial bins rather than array
stride. Arrows are actual `u_x/u_y`, are not normalized, and use one physical
scale across panels. The contact input arrow has a distinct style.

## Validity and interpolation

Sidewall displacement requires a converged, finite, CODTM-valid state.
Contact descriptor validity is separate and does not exclude valid
displacement.

Descriptor quantities must use the descriptor mask. Unverified values are not
zero-filled, interpolated, mirrored, or shown as verified.

Exact stored states are the default. Delta interpolation must be explicitly
enabled and bracketed by two valid finite states of the same case.
Extrapolation and invalid brackets are rejected. Xi interpolation is disabled.
The vector atlas requires an exact stored `u_xy` state.

## Adding a panel or design

To add a panel:

1. define its semantic input and scientific transform outside Matplotlib;
2. implement an axes renderer returning render metadata;
3. add it to a builder through `FigureComposer`;
4. add deterministic numerical rows through `SourceTable`;
5. test units, coordinates, validity, and scaling with synthetic data.

To add a fingertip design:

1. provide existing artifacts through a compatible adapter;
2. give the dataset a unique `design_id`;
3. add that design to the spec;
4. leave plotting code unchanged.

If a new result format is used, only its adapter is new. Missing optimized
results are never fabricated.

## Outputs and provenance

Every figure produces PNG, vector PDF, numerical source CSVs, and
`plot_manifest.json`. The manifest records:

- serialized spec and source path;
- source artifacts and SHA-256;
- design/mesh/case IDs, xi, and indentation;
- represented variable, normalization, units, and coordinate contract;
- interpolation and validity policies;
- deformation/arrow scales and color limits;
- panel metadata, source-data paths, output paths/hashes;
- framework version and git commit.

CSV, JSON, PNG, and PDF serialization is deterministic for repeated runs in
the same environment. PDF timestamps are suppressed.

## Current limitations

- The reference artifacts contain sidewall displacement, not full-volume nodal
  displacement. Internal deformation vectors are therefore not rendered.
- Only the Phase 4K adapter is implemented.
- No real optimized design is currently available, so comparison mode is
  implemented but the reference output is single-design.
- Metrics are descriptive mechanical quantities. Optical observability and
  optimization benefit are not evaluated.


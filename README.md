# Parameterized Fingertip geometry

This project constructs, validates, and visualizes the two-dimensional geometry of a rigid LIT
Hand link inserted into a compliant silicone fingertip pad. Shapely geometry is kept independent
of meshing or finite-element software.

## Geometry convention

All dimensions are millimeters. The flat link-pad plane is `y = 0`, the compliant pad occupies
`y <= 0`, and the rigid top plate occupies `y >= 0`. The pad is the lower half of an ellipse with
horizontal semi-axis `pad_width / 2` and vertical semi-axis `pad_height`. A centered rigid stem
extends downward into the pad.

The rectangular cutout surrounding the stem is defined by:

```text
cutout_width  = stem_width + 2 * void_width
cutout_height = stem_height + void_height
```

Accordingly, `void_width` (`w_v`) is the clearance on **each side** of the stem and
`void_height` (`h_v`) is the clearance below the stem. The four limiting cases are:

- `w_v = 0`, `h_v = 0`: zero-clearance fit (conforming potential contact)
- `w_v > 0`, `h_v = 0`: side clearance
- `w_v = 0`, `h_v > 0`: bottom clearance
- `w_v > 0`, `h_v > 0`: U-clearance

The link-pad interface consists of the two `y = 0` segments outside the centered cutout and is
**always bonded**. The legacy `bonded` parameter remains only for source compatibility;
`bonded=False` emits a deprecation warning and does not change geometry or boundary semantics.

The stem sides and bottom are never bonded to silicone. They are explicit potential contact
surfaces paired with the corresponding pad cutout boundaries. Each `ContactPair` records its
initial normal gap (`void_width` for left/right and `void_height` for bottom). At zero clearance,
the pad-side and stem-side `BoundarySegment` objects remain distinct semantic tags even though
their Shapely lines are coincident. Pad-side contact tags cover the portions directly facing the
stem; the remaining U-clearance corner walls remain part of the void boundary but are not paired
as initial stem contact surfaces.

`model.boundaries.segments` exposes these stable tags to the FEM mesh adapter:

```text
pad_bond_left       pad_bond_right
pad_cutout_left     pad_cutout_right     pad_cutout_bottom
stem_left           stem_right           stem_bottom
pad_outer_arc
```

## Layout

```text
model/      Parameters, Shapely geometry, and reusable plotting
fem/        Gmsh mesh data/validation and the optional Kratos adapter
analysis/   Geometry, Phase 4M mesh, and Kratos initialization CLIs
tests/      Deterministic geometry, mesh, and optional Kratos tests
output/     Generated figures and JSON diagnostics
```

## Dependencies and running

Python 3.11 or newer is required.

```bash
python -m pip install numpy matplotlib shapely pytest gmsh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m analysis.geometry_sanity_check
python -m analysis.phase4_mesh_sanity_check --levels medium fine
```

The Gmsh command writes `medium_mesh.png`, `fine_mesh.png`, and
`mesh_metrics.json` under `output/phase4_mesh/`. Gmsh is a required mesh
backend; absence of its Python API is reported as a dependency error rather
than silently selecting a different triangulator.

The Kratos initialization smoke test uses the validated Kratos 10.3
applications installed in the selected interpreter:

```bash
/home/dk/miniconda3/envs/lit/bin/python \
  -m analysis.phase4_kratos_smoke_test --mesh-level medium
```

The sanity check writes:

- `output/lit_pad_void_four_cases.png`
- `output/lit_pad_void_parameter_grid.png`

## Phase 4M FEM boundary

The solver-independent `model/` package remains the only geometry owner. The
Gmsh adapter passes the polygon rings from `pad_material_geometry` and
`link_geometry` directly to Gmsh, then classifies generated edges against the
Shapely objects in `boundaries.segments` and `interface_definition`. It does
not reproduce fingertip dimensions or the ellipse equation.

The compliant pad and rigid link/stem are separate mesh domains. This keeps
zero-clearance pad/stem contact nodes topologically distinct even when their
coordinates coincide. The pad-side surfaces are configured as ALM `SLAVE`
and the stem-side surfaces as `MASTER`. `pad_bond_left/right` and all rigid
carrier displacement DOFs are fixed only for the initialization smoke model;
this is not presented as a generic bonded-tie implementation.

Phase 4M stops after `StructuralMechanicsAnalysis.Initialize()`, contact-process
initialization, runtime flag inspection, and `Finalize()`. It does not include
an external rounded indenter, prescribed indentation, or a nonlinear loading
solve.

## Example

```python
from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters
from model.visualize import plot_fingertip

parameters = FingertipParameters(
    pad_width=30.0,
    pad_height=18.0,
    link_thickness=3.5,
    stem_width=7.0,
    stem_height=7.0,
    void_width=2.5,
    void_height=3.0,
)

model = FingertipModel(parameters)
axis = plot_fingertip(model, show_dimensions=True)

left_contact = model.contact_pairs[0]
print(left_contact.initial_normal_gap)
```

External indentation loading and nonlinear solution remain intentionally deferred.

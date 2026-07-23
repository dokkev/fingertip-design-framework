# Parameterized Fingertip geometry

The reusable, declarative visualization pipeline is documented in
[Scientific-Figure-Framework.md](Scientific-Figure-Framework.md). Reference
specs live under `examples/` and render through `python -B -m visualization`.

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

## Phase 4I central indentation

Phase 4I adds the external indenter as an FEM fixture; it does not add it to
`FingertipModel`. This preserves the Shapely fingertip model as the sole owner
of pad/link geometry while allowing the solver layer to change loading tools.
The fixture is a radius-`4 mm`, thickness-`1 mm` circular T3 carrier positioned
from the actual `pad_outer_arc` crown and translated rigidly toward the pad.

The indexed Kratos contact groups are:

```text
ContactSub0: PadOuterArc      (SLAVE) / IndenterContactArc (MASTER)
ContactSub1: PadCutoutLeft    (SLAVE) / StemLeft            (MASTER)
ContactSub2: PadCutoutRight   (SLAVE) / StemRight           (MASTER)
ContactSub3: PadCutoutBottom  (SLAVE) / StemBottom          (MASTER)
```

Lengths and prescribed displacement are in `mm`, material values in `MPa =
N/mm^2`, thickness in `mm`, and the resulting 2D plane-strain reaction in `N`.
The current `E = 1.0 MPa` is a numerical placeholder, not calibrated silicone.
This phase is only the symmetric, central, solid-pad baseline.

The outer profile CSV uses the complete reference `PadOuterArc`, ordered by a
normalized arc coordinate from 0 to 1. It reports global and local normal/
tangential displacement. Contact chord width is the span of active pad nodes
projected onto the crown tangent; contact arc length is the sum of source edges
whose two endpoints are active.

Run the isolated Trial before attempting the baseline:

```bash
OMP_NUM_THREADS=1 /home/dk/miniconda3/envs/lit/bin/python -B \
  -m analysis.phase4_indentation_baseline \
  --mesh-level medium --indentation-mm 0.25 --steps 48 --trial

OMP_NUM_THREADS=1 /home/dk/miniconda3/envs/lit/bin/python -B \
  -m analysis.phase4_indentation_baseline \
  --mesh-levels medium fine --indentation-mm 1.5 --steps 48 --compare
```

The first command currently returns nonzero. Initialization verifies all four
runtime contact groups, but the default zero-clearance internal contacts make
the first nonlinear system rank deficient. Skyline LU reports a zero pivot,
Newton reaches its 35-iteration limit, and the failed iterate contains
non-finite solid fields. The requested 1.5 mm baseline is therefore gated off.
Reproduction logs and JSON diagnostics are retained under
`output/phase4_indentation/trial_medium_0p25/`.

### Phase 4I-D internal-contact isolation

The solver layer now accepts an explicit internal-contact configuration:
`none`, `bottom_only`, `sides_separate`, `three_pairs`, or
`continuous_u`. External pad/indenter contact remains present in every
configuration. The continuous option creates solver-facing
`PadInternalU`/`StemInternalU` aggregate SubModelParts by reusing the
existing left/bottom/right nodes and Line2 conditions; the original semantic
SubModelParts remain available for regional post-processing.

Run the fixed medium-mesh first-step isolation suite with:

```bash
OMP_NUM_THREADS=1 /home/dk/miniconda3/envs/lit/bin/python -B \
  -m analysis.phase4_internal_contact_diagnostic \
  --mesh-level medium --cases A B C D E --first-step-only
```

The executed Phase 4I-D suite found:

```text
A external only:          PASS, 1 iteration
B bottom only:            PASS, 1 iteration
C left + right separate:  FAIL during the first Skyline factorization
C-left only:              PASS, 12 iterations
C-right only:             FAIL, 35 iterations
D three separate pairs:   FAIL, 35 iterations
E one continuous U-pair:  FAIL, 35 iterations
```

Cases C/D/E share near-zero ALM-pressure tangent rows at the upper side
endpoints `(3.5, 0)` and `(-3.5, 0)`; the right-only control also fails,
while the left-only control converges. The four lower U-corners reuse the
same physical node and source-condition IDs in D and E. Continuous U reduces
each lower pad corner from two process registrations to one, creates no
duplicate condition connectivity or EquationId, and preserves pair purity,
but does not remove the upper-endpoint rows or recover the solve.

Therefore Phase 4I-D is `FAIL` and continuous U is `REJECT` as a recovery for
the current Kratos 2D ALM setup. Its gated 0.25 mm/48-step Trial was not run.
Phase 4M initialization remains `PASS`, Phase 4I remains incomplete, and the
internal-contact 1.5 mm medium/fine baseline remains blocked. Phase 4J later
establishes an independent external-only baseline.
Artifacts are separate under
`output/phase4_internal_contact_diagnostic/`.

### Phase 4I-E right-side mirror audit

Run the fixed first-step left/right and orientation matrix with:

```bash
OMP_NUM_THREADS=1 /home/dk/miniconda3/envs/lit/bin/python -B \
  -m analysis.phase4_right_side_audit \
  --mesh-level medium --run-orientation-matrix
```

The executed audit found that the source mesh, Shapely-derived physical
normals, runtime nodal normals, semantic membership, slave/master roles, and
zero initial weighted gap all satisfy the left/right reflection contract.
R00 is already physically oriented. Reversing the right slave, master, or both
produces a non-physical ordering and a zero nodal normal at a shared endpoint
during `ExecuteInitialize`, so no orientation source edit was adopted.

The first left/right asymmetry appears after contact search. Left node 5 has
one valid generated master pairing; right node 2 has that valid pairing plus
an adjacent lower master segment whose endpoint projection is out of range
and whose local LM row contribution is exactly zero. The first assembled
upper-endpoint LM rows remain near zero on both sides, but the left nonlinear
solve deactivates the endpoint and converges while R00 reproduces the
right-only failure.

Phase 4I-E is therefore `FAIL` and the orientation hypothesis is `REJECT`.
The evidence narrows the unresolved issue to right upper-endpoint contact
search/pair generation and subsequent LM assembly/active-set handling; it
does not establish a library-level root cause. No source-level production
fix, regression suite, or D/E full Trial was run. Three-pair and continuous-U
remain rejected for the current contact setup, Phase 4I cannot resume, and
the internal-contact 1.5 mm medium/fine baseline remains blocked. Artifacts are under
`output/phase4_right_side_audit/`.

### Phase 4I-F search/crosspoint audit

Run the fixed first-step causal and multiplier-space audit with:

```bash
OMP_NUM_THREADS=1 PYTHONFAULTHANDLER=1 \
  /home/dk/miniconda3/envs/lit/bin/python -B \
  -m analysis.phase4_search_crosspoint_audit \
  --output-directory output/phase4_search_crosspoint_audit
```

The right extra pair projects to Kratos Line2 local coordinate `xi=3.0`,
outside `[-1, 1]`, and has only `5.3013e-14 mm` exact overlap. F00 reproduces
the 35-iteration failure. F02 retains the generated condition/search
lifecycle but sets only that invalid condition inactive; it still follows the
same endpoint active-set history and fails. The invalid condition's ACTIVE
state is therefore not necessary for failure, while safe generated-pair
removal/insertion experiments are unavailable through the Kratos 10.3 Python
API.

Both upper pad endpoints are also fully constrained pad-bond crosspoints.
Their active LM diagonal is zero, and the valid contact condition alone leaves
free-column norms of `1.7164e-15` (left) and `6.3117e-17` (right) after the
fixed endpoint displacement columns are eliminated. Inactive contact supplies
an LM diagonal; the left control converges at iteration 12 with that endpoint
inactive. This distinguishes the broad-phase extra-pair behavior from the
direct active-crosspoint LM row deficiency.

Phase 4I-F is `FAIL`: no physical, mesh-independent production correction was
validated, so A–E regressions and the gated 0.25 mm/48-step D/E trial were not
run. The mixed T3 solid remains adopted, the current internal ALM contact
configuration remains blocked, and internal-contact 1.5 mm trials remain
deferred. Artifacts
are under `output/phase4_search_crosspoint_audit/`.

### Phase 4I-G and Phase 4J

Run the bounded crosspoint treatment audit and the resumable no-void fallback:

```bash
OMP_NUM_THREADS=1 /home/dk/miniconda3/envs/lit/bin/python -B \
  -m analysis.phase4_crosspoint_multiplier_treatment

OMP_NUM_THREADS=1 PYTHONFAULTHANDLER=1 \
  /home/dk/miniconda3/envs/lit/bin/python -B \
  -m analysis.phase4_no_void_baseline
```

Phase 4I-G is `FAIL/BLOCKED`. Kratos 10.3 provides no official
contact–Dirichlet crosspoint LM omission/condensation setting. In six
mirrored/refined minimal patches, a fully fixed ACTIVE endpoint retained
healthy coupling to its adjacent free slave trace and did not produce a zero
LM row. A topology-only rule that excludes every such endpoint is therefore
not a safe production correction. The gated A–E regressions and full internal
contact trial were not run.

Phase 4J is an independent `PASS` using the existing zero-void default
geometry, mixed T3 solid, `nu = 0.49`, and external pad–indenter ALM pair only.
The 1.5 mm/48-step medium and fine cases both pass with final reactions
`0.861926 N` and `0.864680 N`; their relative difference is `0.319%`.
All fields remain finite, det(F) stays positive, the force curves are smooth
and monotonic, and no internal contact-coupled LM enters the generated
condition DOF lists. The symbolic roadmap contains no executable location list
or documented no-void candidate list, so J3/J4 are recorded as `SKIPPED`
instead of inventing a new sweep.

Artifacts are under `output/phase4_crosspoint_multiplier_treatment/` and
`output/phase4_no_void_baseline/`. The mixed solid and external baseline remain
adopted; the current internal zero-clearance ALM configuration remains blocked.

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

Contact-location sweeps remain deferred until a reviewed discrete location
list is added. The Phase 4J external-only baseline is available for those
future cases; the default internal-contact nonlinear blocker remains separate.

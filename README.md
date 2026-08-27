# LUMO fingertip design

LUMO evaluates parametric optical fingertips with one concrete production
pipeline:

```text
FingertipParameters
  -> full five-LED FingertipMesh
  -> Newton deformation at instantaneous force-threshold crossings
  -> OptiX transport through the deformed silicone
  -> J_contact and J_obs
  -> sequential Ax multi-objective search
```

The production morphology has five variables on a `0.5 mm` lattice:

```text
flat-pad height
semiellipse height
stem width
stem height
void width
```

Flat-pad width is fixed at `30 mm`, void height is fixed at zero, and the
`5.1 x 0.19 mm` LED recess is a hardware feature rather than an optimization
variable. One evaluation uses the sphere diameters, longitudinal contact
locations, and sequential force thresholds configured in `scripts/run_mobo.py`.

## Repository map

- `lumo/fingertip/`: physical parameters and analytic geometry;
- `lumo/mesh/`: discretization of the complete five-LED fingertip;
- `lumo/newton/`: Newton model construction and indenters;
- `lumo/simulation/`: Newton runtime and indentation workflow;
- `lumo/ray_tracing/`: OptiX scene, emission, transport, and observation;
- `lumo/optimization/`: design space, raw evaluator, objectives, and Ax BO;
- `validation/`: procedural scientific and engineering checks;
- `scripts/run_mobo.py`: the explicit production-campaign entry point.

## Environment and checks

Use the `lit` Conda environment:

```bash
conda activate lit
python -m pip install -e ".[mesh,physics,ax,test]"
```

CUDA, OptiX, and the header-only OptiX Toolkit are external dependencies. Set
`OPTIX_INCLUDE_DIR` and `OTK_INCLUDE_DIR` as described in
[docs/COMMANDS.md](docs/COMMANDS.md).

Run the focused repository checks with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n lit \
  pytest -q tests/unit
```

The production BO is expensive and is never started by the focused checks.
Review the settings in `scripts/run_mobo.py` before launching it explicitly.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for ownership and scientific
dataflow, and [docs/COMMANDS.md](docs/COMMANDS.md) for supported commands.

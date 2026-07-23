# Fingertip Design Framework

Parametric 2D fingertip geometry, solver-independent Gmsh meshing, a Kratos
10.3 finite-element backend, and reproducible scientific visualization for
the LIT Hand fingertip.

## Repository map

- `model/`: Shapely geometry, parameters, boundary and contact semantics.
- `mesh/`: Gmsh conversion, discrete topology, quality, and indenter geometry.
- `fem/`: Kratos settings, adapters, contact, solves, and neutral results.
- `visualization/`: artifact adapters, transforms, panels, themes, and export.
- `validation/`: scientific benchmarks and Phase workflows.
- `tests/unit/`: fast deterministic contracts without solver execution.
- `tests/smoke/`: minimal Gmsh, Kratos, and headless-renderer wiring.
- `docs/`: architecture, environment setup, and preserved validation reports.
- `output/`: ignored generated artifacts only.

The enforced dependency and ownership rules are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The fingertip geometry remains
the only source of physical shape and boundary semantics; meshing, Kratos, and
plotting do not reimplement it.

## Environment

Python 3.11 or newer is required. Optional dependency groups are declared in
`pyproject.toml`: `mesh`, `visualization`, `validation`, and `test`. Kratos is
an externally managed dependency and is intentionally not installed by this
project.

Solver-backed work uses:

```text
/home/dk/miniconda3/envs/lit/bin/python
Kratos kernel 10.3.0
```

See [docs/setup/kratos.md](docs/setup/kratos.md) for the runtime contract.

## Main commands

```bash
LIT_PYTHON=/home/dk/miniconda3/envs/lit/bin/python

$LIT_PYTHON -m pytest tests/unit -q
$LIT_PYTHON -m pytest tests/smoke -q -m "not kratos"
$LIT_PYTHON -m pytest tests/smoke -q -m kratos

$LIT_PYTHON -m validation.fingertip.geometry \
  --output-directory output/validation/fingertip/geometry
$LIT_PYTHON -m validation.fingertip.mesh --levels medium fine \
  --output-directory output/validation/fingertip/mesh
$LIT_PYTHON -m validation.benchmarks.volumetric_locking \
  run --output output/validation/benchmarks/volumetric_locking.json
$LIT_PYTHON -m validation.fingertip.transfer_map \
  --output-dir output/validation/fingertip/transfer_map \
  --reference-dir output/validation/fingertip/indentation/no_void

$LIT_PYTHON -m visualization examples/displacement_vector_atlas.yaml \
  --output-dir output/figures/displacement_vector_atlas
```

The complete command index is [docs/COMMANDS.md](docs/COMMANDS.md).
Scientific conclusions and historical debugging evidence are preserved under
`docs/validation/`; figure-system documentation is under
`docs/visualization/`.

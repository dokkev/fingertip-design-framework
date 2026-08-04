# Commands

Activate the Python environment that provides the project dependencies and the
externally managed Kratos installation. Commands below use that environment's
active `python`.

## Development setup

Editable installation is the preferred development setup:

```bash
cd /path/to/lit_ws
python -m pip install -e ".[mesh,visualization,validation,test]"
```

After installation, package and module commands work from any working
directory.

## Tests

```bash
python -m pytest tests/unit -q
python -m pytest tests/smoke -q -m "smoke and not kratos"
python -m pytest tests/smoke -q -m kratos
```

## Validation

```bash
python -m validation.benchmarks.volumetric_locking run \
  --output output/validation/benchmarks/volumetric_locking.json
python -m validation.benchmarks.mixed_volumetric run \
  --output output/validation/benchmarks/mixed_volumetric.json
python -m validation.fingertip.geometry \
  --output-directory output/validation/fingertip/geometry
python -m validation.fingertip.mesh --levels medium fine \
  --output-directory output/validation/fingertip/mesh
python -m validation.fingertip.indentation.no_void
python -m validation.fingertip.transfer_map \
  --output-dir output/validation/fingertip/transfer_map \
  --reference-dir output/validation/fingertip/indentation/no_void
```

## Figures

```bash
python -m visualization examples/transfer_map_comparison.yaml \
  --output-dir output/figures/transfer_map_comparison
python -m visualization examples/displacement_vector_atlas.yaml \
  --output-dir output/figures/displacement_vector_atlas
```

## End-to-end FEM example

From the repository root, run:

```bash
python examples/fem_visualize.py
```

The direct script is also launchable from any working directory:

```bash
python /path/to/lit_ws/examples/fem_visualize.py
```

The example runs or reuses three FEM cases with contact fixed at `x=0 mm` and
indenter radii of `2`, `4`, and `6 mm`. It persists neutral artifacts with
full-pad nodal displacement fields, validates and reloads them through the
visualization adapter, and opens the atlas with Matplotlib. It does not export
PNG, PDF, source-data, or figure-manifest files. The first execution can take
significant time.
Subsequent executions reuse every valid completed case. To deliberately
recompute all cases:

```bash
python -m validation.fingertip.indentation.normal_field_atlas --force
```

`examples/bootstrap.py` makes direct example execution independent of the
current working directory. It cannot affect Python's pre-execution `-m` module
resolution, so module commands still require the editable installation.

## Result-only visualization

To render already-persisted artifacts without starting FEM:

```bash
python -m visualization examples/displacement_vector_atlas.yaml
```

The declarative visualization command preserves the artifact boundary:
`visualization` reads the dataset manifest and never starts the solver.

For a direct geometry-only visualization:

```bash
python /path/to/lit_ws/examples/fingertip_visualize.py
```

All generated validation artifacts and figures are written below `output/`.

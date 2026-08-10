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

Validation figure workflows read persisted artifacts and save directly:

```bash
python -m validation.figures.displacement_atlas \
  --input-dir output/validation/fingertip/indentation/normal_full_field \
  --output output/figures/displacement_vector_atlas/displacement_vector_atlas.png
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

The example constructs one fingertip and mesh, runs `solve()`, traces reference
and loaded optical states, prints the camera-independent proxy metrics, and
opens the loaded transport plot. It does not write artifacts. The solve can
take significant time.

The resumable three-radius scientific atlas remains a validation command:

```bash
python -m validation.fingertip.indentation.normal_field_atlas --force
```

`examples/bootstrap.py` makes direct example execution independent of the
current working directory. It cannot affect Python's pre-execution `-m` module
resolution, so module commands still require the editable installation.

## Result-only visualization

The public `visualization` package contains plotting functions only. Persisted
scientific figures are rendered by validation-specific commands above; they
validate their own artifact manifests and never start the solver.

For a direct geometry-only visualization:

```bash
python /path/to/lit_ws/examples/fingertip_visualize.py
```

For interactive optical previews that do not write artifacts:

```bash
python /path/to/lit_ws/examples/light_transport.py
python /path/to/lit_ws/examples/loaded_light_transport.py
```

The optional Mitsuba camera validator is demonstrated by:

```bash
python /path/to/lit_ws/examples/camera_render.py
```

All generated validation artifacts and figures are written below `output/`.

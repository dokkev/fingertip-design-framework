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

The optional NiceGUI design-space explorer can be installed and launched with:

```bash
python -m pip install -e ".[gui]"
python -m gui.design_space_app
```

For direct development from the application directory, the equivalent command
is `python design_space_app.py`.

It uses the model and existing visualization only; it does not require Kratos,
Gmsh, or an optical transport backend.

The optional Ax adapter can be installed with:

```bash
python -m pip install -e ".[ax]"
```

It has no CLI or GUI entry point in this iteration.

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

# Staged Kratos FEA throughput/fidelity study.  These commands write only to
# output/validation/fem/throughput/ and leave production defaults unchanged.
OMP_NUM_THREADS=1 python -m validation.fem.throughput --stage profile
OMP_NUM_THREADS=1 python -m validation.fem.throughput --stage diagnostics
OMP_NUM_THREADS=1 python -m validation.fem.throughput --stage mesh
OMP_NUM_THREADS=1 python -m validation.fem.throughput --stage full
OMP_NUM_THREADS=1 python -m validation.fem.throughput --stage steps
OMP_NUM_THREADS=1 python -m validation.fem.throughput --stage continuation
OMP_NUM_THREADS=1 python -m validation.fem.throughput --stage symmetry
OMP_NUM_THREADS=1 python -m validation.fem.throughput --stage solver
OMP_NUM_THREADS=1 python -m validation.fem.throughput --stage parallel
# Recompute the report from completed staged artifacts without rerunning FEA.
python -m validation.fem.throughput --stage finalize
```

## Figures

Validation figure workflows read persisted artifacts and save directly:

```bash
python -m validation.figures.displacement_atlas \
  --input-dir output/validation/fingertip/indentation/normal_full_field \
  --output output/figures/displacement_vector_atlas/displacement_vector_atlas.png
```

## Tutorial examples

From the repository root, run:

```bash
python examples/view_fingertip.py
python examples/view_fea.py
python examples/view_light.py
python examples/view_case.py
```

The direct script is also launchable from any working directory:

```bash
python /path/to/lit_ws/examples/view_light.py
```

`view_fingertip.py` teaches the `Fingertip` → `plot_fingertip` flow.
`view_fea.py` compares three indentation diameters with a shared displacement
color scale. `view_light.py` demonstrates the
`Fingertip` → mesh → `solve()` → `trace()` → `evaluate()` flow with shared
reference/loaded transport normalization. `view_case.py` demonstrates the
nominal `Fingertip()` → `FEA2D` → `RayTracing2D` → `FingertipCase.run()`
PLANAR_2D OptiX flow with a 12-step loaded state, a no-contact reference
optical trace, and a 2x2 unloaded-vs-loaded comparison figure. `run_case()`
remains a convenience wrapper for the same flow.
The solver-backed examples can take significant time and do not write artifacts;
`view_case.py` additionally requires the externally managed Kratos and
CUDA/OptiX environment used by the production case path.

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

The optional Mitsuba camera validator is demonstrated by:

```bash
python /path/to/lit_ws/examples/camera_render.py
```

The dependency-light reduced 2D reference transport does not require CUDA or
OptiX. The production `case.run_case()` PLANAR_2D path does require the
externally managed CUDA/OptiX environment described above. The optional
environment doctor reports import, header, and GPU-runtime status without
compiling a kernel:

```bash
python -m optics.optix.doctor
python -m optics.optix.doctor --json
```

`OPTIX_INCLUDE_DIR` should point directly to the directory containing both
`optix.h` and `optix_device.h`. Header resolution precedence is: explicit
`discover_paths(..., optix_include_dir=..., cuda_include_dir=...)` arguments,
`OPTIX_INCLUDE_DIR` / `CUDA_INCLUDE_DIR`, `OptiX_INSTALL_DIR` or `OPTIX_ROOT`
(root and `include`), then the documented conventional CUDA/OptiX locations.

The focused NVIDIA OptiX transport validator uses the externally managed
CUDA/OptiX environment documented in `README.md`:

```bash
OptiX_INSTALL_DIR=/external/optix-dev \
python -m validation.optics.optix_smoke

OptiX_INSTALL_DIR=/external/optix-dev \
python -m validation.optics.transport3d_validation \
  --output output/validation/optics/transport3d
```

The validation command runs the 11 mm single-source cell, the planar
consistency gate, the four authorized medium-mesh 48-step contact states, and
the deterministic 4,096/16,384/65,536-ray convergence check. It writes only
machine-readable generated artifacts below
`output/validation/optics/transport3d/`.

The dependency-light focused contracts can be run without the optional GPU
runtime (the environment here disables unrelated auto-loaded ROS pytest
plugins):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest tests/unit/test_transport3d_contracts.py -q
```

All generated validation artifacts and figures are written below `output/`.

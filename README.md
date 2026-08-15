# Fingertip Design Framework

Parametric 2D fingertip geometry, solver-independent Gmsh meshing, a Kratos
10.3 finite-element backend, and reproducible scientific visualization for
the LIT Hand fingertip.

## Repository map

- `model/`: Shapely geometry, parameters, boundary and contact semantics.
- `mesh/`: Gmsh conversion, discrete topology, quality, and indenter geometry.
- `fem/`: Kratos settings, adapters, contact, solves, and neutral results.
- `visualization/`: thin Matplotlib plotting helpers for model and neutral results.
- `validation/`: scientific benchmarks and Phase workflows.
- `tests/unit/`: fast deterministic contracts without solver execution.
- `tests/smoke/`: minimal Gmsh, Kratos, and headless-renderer wiring.
- `docs/`: architecture, environment setup, and preserved validation reports.
- `output/`: ignored generated artifacts only.

The enforced dependency and ownership rules are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The fingertip geometry remains
the only source of physical shape and boundary semantics; meshing, Kratos, and
plotting do not reimplement it.

The parameter meanings, construction order, derived coordinates, and explicit
legacy migration path are documented in [docs/geometry.md](docs/geometry.md).

## Public workflow

The physical, FEM, and optical API is intentionally shallow:

```python
from fem import solve
from model import Fingertip, FingertipParameters
from optics import evaluate, trace
from visualization import plot_transport

tip = Fingertip(FingertipParameters())
mesh = tip.mesh()

fea = solve(tip, mesh, indentation=1.5)
reference = trace(tip, mesh)
loaded = trace(tip, fea.deformed_mesh)
metrics = evaluate(reference, loaded)
plot_transport(loaded)
```

`FEAResult` contains neutral displacement, reaction, contact, and convergence
data; no Kratos object crosses into optics.
Camera formation is an optional validation path behind
`optics.mitsuba.MitsubaRenderer`; transport density is not labeled as camera
brightness or physical irradiance.

## Examples

The examples follow the framework from design to mechanics to sensing:

1. `view_fingertip.py` defines a `Fingertip` and plots its geometry.
2. `view_fea.py` meshes one design, solves three circular contact cases, and
   compares their displacement fields.
3. `view_light.py` carries one deformation through `trace()` and compares the
   reference and loaded transport with `evaluate()`.

The public flow used by the latter two examples is:

```python
from fem import IndenterSettings, solve
from model import Fingertip, FingertipParameters
from optics import evaluate, trace

tip = Fingertip(FingertipParameters())
mesh = tip.mesh()
fea = solve(
    tip,
    mesh,
    indentation=1.5,
    indenter=IndenterSettings(radius_mm=4.0),
)
reference = trace(tip, mesh)
loaded = trace(tip, fea.deformed_mesh)
metrics = evaluate(reference, loaded)
```

Run them directly from the repository root with the project environment:

```bash
python examples/view_fingertip.py
python examples/view_fea.py
python examples/view_light.py
```

`camera_render.py` remains an optional Mitsuba camera-validation example.

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

### Optional NVIDIA OptiX validation

NVIDIA OptiX is an optional higher-fidelity optical-validation dependency. It
is not required for the normal 2D model, mesh, FEM, or optics workflow. The
currently tested environment is an NVIDIA GPU with an OptiX-compatible driver,
CUDA Toolkit 13.1, OptiX headers 9.1.0, PyOptiX 9.1.0, CuPy for CUDA 13.x,
and `cuda-python`. These are the tested versions, not universal hard
requirements for future OptiX work.

The Conda-first setup used for this smoke path is:

```bash
conda activate lit
conda install -c nvidia/label/cuda-13.1.0 cuda
python -m pip install pyoptix
python -m pip install cupy-cuda13x
python -m pip install cuda-python
```

Install the OptiX 9.1 header SDK outside this repository:

```bash
git clone --branch v9.1.0 https://github.com/NVIDIA/optix-dev.git /desired/external/path/optix-dev
export OptiX_INSTALL_DIR=/desired/external/path/optix-dev
```

Conda CUDA may place headers under
`CONDA_PREFIX/targets/x86_64-linux/include`; the repository resolver checks
that layout automatically. `OPTIX_ROOT` is also accepted as a secondary
OptiX SDK environment variable. Basic environment checks are:

```bash
which nvcc
nvcc --version
python -c "import optix; print(optix.version())"
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount()); print(cp.cuda.Device(0).compute_capability)"
```

Run the repository-only runtime smoke test with:

```bash
python -m validation.optics.optix_smoke
```

The smoke test compiles a tiny repository-owned CUDA/OptiX program with
NVRTC, builds one triangle GAS, and validates deterministic hit and miss rays;
it does not render an image or implement the LUMO 3D transport model.

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

$LIT_PYTHON -m validation.figures.displacement_atlas \
  --input-dir output/validation/fingertip/indentation/normal_full_field \
  --output output/figures/displacement_vector_atlas/displacement_vector_atlas.png
```

The complete command index is [docs/COMMANDS.md](docs/COMMANDS.md).
Scientific conclusions and historical debugging evidence are preserved under
`docs/validation/`; figure-system documentation is under
`docs/visualization/`.

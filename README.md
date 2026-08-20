# Fingertip design framework

LUMO is a parametric fingertip design framework whose production path is:

```text
fingertip morphology
  -> semantic mesh and first contact
  -> Newton/Warp mechanics
  -> deformed geometry
  -> FULL_3D OptiX transport
  -> trajectory objective
  -> Ax morphology search
```

The current morphology space has six active variables:

`flat_pad_height`, `semielliptical_pad_height`, `stem_width`, `stem_height`,
`void_width`, and `void_height`.

The default evaluation protocol uses contact locations `u=(.25,.50,.75)`,
sphere radii `(4,5) mm`, and absolute post-contact depths `(0.5,1.0,1.5) mm`.
That is six continuous Newton trajectories and eighteen FULL_3D optical states
per morphology.

## Repository map

- `finger/`: parametric morphology and optical source/material records;
- `mesh/`: semantic geometry and volume/rigid meshes;
- `contact/`: geometry-derived alignment and first-contact search;
- `physics/`: the single Newton/Warp mechanics implementation;
- `ray_tracing/`: FULL_3D OptiX transport and production runtime;
- `optimization/`: named design space, protocol, objective, registry, and Ax;
- `validation/`: current scientific runners and validation-only references;
- `gui/`: deferred design-space diagnostics.

The old 2D FEM, case, examples, and generic plotting layers are intentionally
not part of this branch.

## Environment

Run repository commands in the `lit` Conda environment:

```bash
conda activate lit
python -m pip install -e ".[mesh,physics,ax,test]"
```

OptiX, CUDA, Warp/Newton, and any reference solver installations remain
externally managed runtime dependencies.

## First checks

```bash
conda run -n lit ./scripts/tools/pytest_lit tests/unit/finger tests/unit/mesh -q
conda run -n lit ./scripts/tools/pytest_lit tests/unit/physics -q
conda run -n lit python scripts/tools/optix_doctor.py --json
conda run -n lit python -m scripts.tools.optix_smoke
```

The doctor diagnoses the environment. The validation smoke command performs a
real OptiX initialization and launch. Both should pass before an unattended
campaign.

## Scientific validation

The current trajectory validation entry point is:

```bash
conda run -n lit python -m validation.optimization.lumo3d_trajectory_validation \
  --output output/validation/lumo3d_trajectory
```

It is validation-only and writes generated artifacts below `output/`.
Do not use it to start a BO campaign; the repository's expensive optimization
commands are intentionally separate from the focused regression path.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for ownership and
[docs/COMMANDS.md](docs/COMMANDS.md) for supported commands.

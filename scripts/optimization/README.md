# Optimization entry points

`run_bo_ideal.py` is the canonical human-facing entry point. It delegates to
the single checkpointed campaign engine in `run_bo.py`, which uses the
production `Lumo3DTrajectoryEvaluator` through the production Ax adapter.
Neither script contains a second evaluator.

The visible code contract owns the nominal `FingertipParameters`, LED,
trajectory protocol, objective, search bounds, and Ax seed. The strict
`config/lumo_execution.yaml` owns device, mesh, Newton/first-contact, and
transport numerical settings. The resolved values, YAML digest, design space,
source snapshot, and explicit budgets are copied into `config.json` and every
checkpoint.

Run the cheap environment and configuration check first:

```bash
conda run -n lit python -m scripts.optimization.run_bo_ideal \
  --preflight --execution-config config/lumo_execution.yaml
```

The preflight checks Gmsh, Newton/Warp, CuPy, OptiX/NVRTC, domain
construction, a tiny real Newton advance, and one deterministic OptiX hit/miss
launch. A failure is an external prerequisite failure; it is not registered as
a candidate result.

For a deliberately small production-path smoke, opt in explicitly:

```bash
conda run -n lit python -m scripts.optimization.run_bo_ideal \
  --smoke \
  --execution-config config/lumo_execution.yaml \
  --output output/optimization/bo_smoke
```

`--smoke` selects the explicit reduced two-state protocol. Without it, the
runner uses the authoritative 18-state production protocol. Smoke requests one
successful Sobol observation after nominal. Production defaults to six
successful Sobol observations and requires explicit `--search-successes >= 1`,
`--max-evaluations`, and `--max-proposals`. Success targets are not reduced by
candidate failures; the two caps independently bound evaluator calls and all
generated proposals.
The runner refuses to overwrite a non-empty output directory and writes
`config.json`, `preflight.json`, `registry.json`, `summary.json`, and a
versioned atomic checkpoint tree. Each checkpoint contains the public Ax JSON
state, the trial audit, and resume contract state. To resume explicitly after
an interruption, pass the campaign directory (or its `checkpoint.json`
pointer) with the same budget and fixed inputs:

```bash
conda run -n lit python -m scripts.optimization.run_bo_ideal \
  --resume output/optimization/bo_smoke \
  --smoke
```

The runner never resumes merely because an output directory already exists.
It accepts only the campaign root or current `checkpoint.json` pointer;
immutable checkpoint directories are audit evidence, not rollback targets.
Production is clean-worktree-only unless `--allow-dirty` is explicit, and
cross-source registry reuse requires `--allow-cross-revision-cache`. Both
choices are part of the exact resume contract.

Optimization contracts and the production evaluator remain owned by
`lumo/optimization/`. Validation-only Test BO and scientific comparison
workflows remain under `validation/optimization/`.

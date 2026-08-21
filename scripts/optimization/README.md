# Optimization entry points

`run_bo.py` is the bounded, concrete campaign entry point. It uses the
production `lumo.optimization.evaluator.Lumo3DTrajectoryEvaluator` through the
production Ax adapter; it is not a second evaluator.

The file has a visible `USER CONFIG` section for the nominal
`FingertipParameters` (kinematic, viscoelastic, and bulk optical groups), the
separate LED source descriptor, trajectory protocol, transport settings,
mechanics contract, objective configuration, search bounds, and Ax seed/budget.
The selected bounds are copied into `config.json` so a campaign can be
reconstructed from its recorded configuration.

Run the cheap environment and configuration check first:

```bash
conda run -n lit python scripts/optimization/run_bo.py --preflight
```

The preflight checks Gmsh, Newton/Warp, CuPy, OptiX/NVRTC, domain
construction, a tiny real Newton advance, and one deterministic OptiX hit/miss
launch. A failure is an external prerequisite failure; it is not registered as
a candidate result.

For a deliberately small production-path smoke, opt in explicitly:

```bash
conda run -n lit python scripts/optimization/run_bo.py \
  --smoke \
  --trials 1 \
  --output output/optimization/bo_smoke
```

`--smoke` selects the explicit reduced two-state protocol. Without it, the
runner uses the authoritative 18-state production protocol. `--trials` is the
number of Ax-generated proposals; the nominal baseline is evaluated separately,
so `--trials 1` performs two candidate evaluations when both succeed. Use a
larger trial count only when the measured campaign is explicitly authorized.
The runner refuses to overwrite a non-empty output directory and writes
`config.json`, `preflight.json`, `registry.json`, `summary.json`, and a
versioned atomic checkpoint tree. Each checkpoint contains the public Ax JSON
state, the trial audit, and resume contract state. To resume explicitly after
an interruption, pass the campaign directory (or its `checkpoint.json`
pointer) with the same budget and fixed inputs:

```bash
conda run -n lit python scripts/optimization/run_bo.py \
  --resume output/optimization/bo_smoke \
  --smoke \
  --trials 1
```

The runner never resumes merely because an output directory already exists.

Optimization contracts and the production evaluator remain owned by
`lumo/optimization/`. Validation-only Test BO and scientific comparison
workflows remain under `validation/optimization/`.

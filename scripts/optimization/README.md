# Optimization entry points

`run_bo.py` is the bounded, concrete campaign entry point. It uses the
production `optimization.evaluator.Lumo3DTrajectoryEvaluator` through the
production Ax adapter; it is not a second evaluator.

The file has a visible `USER CONFIG` section for the nominal
`FingertipParameters`, LED/material values, trajectory protocol, transport
settings, mechanics contract, and Ax seed/budget.

Run the cheap environment and configuration check first:

```bash
conda run -n lit python scripts/optimization/run_bo.py --preflight
```

The preflight checks Gmsh, Newton/Warp, CuPy, OptiX/NVRTC, domain
construction, and one deterministic OptiX hit/miss launch. A failure is an
external prerequisite failure; it is not registered as a candidate result.

For a deliberately small production-path smoke, opt in explicitly:

```bash
conda run -n lit python scripts/optimization/run_bo.py \
  --trials 1 \
  --output output/optimization/bo_smoke
```

Use a larger trial count only when the measured campaign is explicitly
authorized. The runner refuses to overwrite a non-empty output directory and
writes `config.json`, `preflight.json`, `trials.json`, `registry.json`, and
`summary.json` with finite, structured JSON values.

Optimization contracts and the production evaluator remain owned by the
top-level `optimization/` package. Validation-only Test BO and scientific
comparison workflows remain under `validation/optimization/`.

# Architecture

## Dependency direction

```text
model -> mesh -> fem -> neutral result/artifact data
  \        \       \              \
   \        \       +-------------> validation
    \        +---------------------> validation
     +------------------------------> visualization / validation

visualization -> model + mesh + neutral artifacts
validation    -> model + mesh + fem + visualization
tests         -> the package under test
```

Dependencies only flow to the right/down. Production packages never import
`validation` or `tests`. `visualization` has no Kratos import-time dependency.

## Ownership

- `model/` owns parametric Shapely geometry, physical boundary semantics, and
  contact-pair definitions. It owns no Gmsh, Kratos, plotting, or file output.
- `mesh/` owns the deterministic conversion from `FingertipModel` to discrete
  Gmsh topology, mesh settings, quality, and the solver-independent indenter.
- `fem/` is the Kratos backend. It owns model-part assembly, materials,
  contact, constraints, nonlinear indentation, observations, and conversion to
  neutral Python results. It owns no PASS/FAIL policy, report files, or plots.
- `visualization/` owns semantic figure data, transforms, themes, panels,
  plotting, and export. Repository artifact parsers live in
  `visualization/adapters/`.
- `validation/` owns scientific benchmarks, Phase acceptance, subprocess
  isolation, checkpointing, provenance, reports, and generated artifact
  schemas.
- `tests/unit/` owns fast dependency-light contracts. `tests/smoke/` only
  verifies Gmsh, Kratos, or headless-renderer wiring.

## Runtime flow

Validation entrypoints construct immutable model parameters, build a discrete
mesh, configure a fresh Kratos model, run a bounded solve, extract neutral
results, apply scientific acceptance, and write to an explicit output
directory. Visualization reads declared input artifacts and never starts a
solver.

## Failure and artifact policy

Numerical failures stay explicit; non-finite fields, non-convergence, and
invalid contact states are not clamped or hidden. Every repeated solve has a
step/iteration/process timeout boundary. `output/` is only a generated sink.
Reference inputs needed by a clean checkout belong in `tests/fixtures/` or
`validation/reference_data/`.

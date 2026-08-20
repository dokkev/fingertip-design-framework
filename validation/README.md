# Scientific validation

This package consumes the reusable `finger`, `mesh`, `contact`, `physics`, and
`ray_tracing` libraries. It owns current scientific orchestration, regression
provenance, and generated artifacts.

`lumo/` contains the concrete reusable simulation orchestration. `optimization/`
contains the current trajectory evaluator and bounded search workflows.
`physics/` contains Newton validation and correspondence checks.
`reference/kratos3d/` is a read-only loader for persisted nonlinear reference
states; it is not a production solver backend. Shared I/O and process
orchestration live in `common/`.

These runs are not unit tests and may take minutes. Outputs must be directed
under `output/validation/`; they remain untracked.

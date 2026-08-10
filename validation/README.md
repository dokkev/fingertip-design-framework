# Scientific validation

This package consumes the reusable `model`, `mesh`, `fem`, and
`visualization` libraries. It owns scientific benchmarks, Phase acceptance,
checkpoint/resume behavior, provenance, and generated artifacts.

`benchmarks/` contains solver/formulation studies. `fingertip/` contains
geometry, mesh, indentation, internal-contact, and transfer-map validation.
`figures/` contains explicit post-processing workflows for persisted
validation artifacts (`displacement_atlas.py` and `transfer_map.py`); it does
not define a generic figure framework. Shared mechanical I/O and process
orchestration live in `common/`.

These runs are not unit tests and may take minutes. Outputs must be directed
under `output/validation/`; they remain untracked.

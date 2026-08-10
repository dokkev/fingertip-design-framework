# Repository guide

- Preserve dependency direction: `model -> mesh -> fem`; `visualization` consumes
  neutral artifacts; `validation` is the top-level scientific consumer.
- Production packages must not import `validation` or `tests`.
- `model`, `mesh`, and `fem` must not import Matplotlib.
- Kratos is an external environment dependency and must not be added to
  `pyproject.toml`.
- Generated files belong under `output/` and remain untracked.
- Do not add or run tests unless the user explicitly requests them. When tests
  are requested, run only the focused checks needed for the stated contract.
- Do not add or run unit, smoke, integration, FEM, or validation tests unless
  the user explicitly requests them. Do not automatically chain implementation
  into testing or validation; this minimizes token and execution-time usage.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for ownership and
[docs/COMMANDS.md](docs/COMMANDS.md) for supported commands.

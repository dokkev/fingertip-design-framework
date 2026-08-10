# Repository guide

- Preserve dependency direction: `model -> mesh -> fem`; `visualization` consumes
  neutral model/mesh/FEM/optics results; `validation` is the top-level scientific
  consumer.
- Production packages must not import `validation` or `tests`.
- `model`, `mesh`, and `fem` must not import Matplotlib.
- Kratos is an external environment dependency and must not be added to
  `pyproject.toml`.
- Generated files belong under `output/` and remain untracked.
- Keep implementations scoped to the requested task. Do not introduce new
  abstractions, compatibility layers, registries, frameworks, or generalized
  infrastructure unless the current task clearly requires them.
- Backward compatibility with legacy internal APIs is not required unless the
  user explicitly requests it. Prefer migrating repository-owned callers and
  deleting obsolete code over preserving old and new interfaces together.
- Do not add or run unit, smoke, integration, FEM, or validation tests unless
  the user explicitly requests testing or `INSTRUCTION.md` explicitly requires
  them.
- Do not automatically chain implementation into broad testing or validation.
  When testing is explicitly requested, run only the focused checks needed for
  the stated contract unless broader regression testing is specifically
  requested.
- Do not continue into adjacent cleanup, refactoring, validation, or feature
  work after completing the requested scope.

See `docs/ARCHITECTURE.md` for package ownership and dependency rules.
See `docs/COMMANDS.md` for supported commands.

## Current iteration

If `INSTRUCTION.md` exists and is non-empty, read it before making changes.

`INSTRUCTION.md` contains the task-specific scope, implementation requirements,
non-goals, checklist, and completion criteria for the current iteration.
`AGENTS.md` contains stable repository-wide rules.

- Follow only the work requested in `INSTRUCTION.md`; do not expand its scope.
- Inspect the current implementation before editing rather than assuming the
  repository still matches an earlier architecture.
- Do not restore deleted or legacy architecture merely to satisfy stale
  callers, tests, examples, or documentation.
- Do not edit, rewrite, clear, or delete `INSTRUCTION.md` unless the user
  explicitly asks you to.
- If `INSTRUCTION.md` is missing or empty, do not infer a pending implementation
  task from previous work. Follow only the user's current prompt and the stable
  repository rules above.

## Implementation checklist review

If `INSTRUCTION.md` contains an implementation checklist, completing the code
changes is NOT the final step.

After implementation is complete:

1. Do not have the implementation agent certify its own checklist.
2. Use a separate, fresh subagent as an independent reviewer.
3. The reviewer must inspect the current repository state and implementation
   diff directly.
4. The reviewer must evaluate every checklist item independently rather than
   trusting the implementation agent's summary.
5. For each checklist item, report one of:
   - `PASS` — implemented and supported by direct evidence,
   - `FAIL` — missing, incorrect, or contradicted by the implementation,
   - `UNCLEAR` — cannot be verified from the available code or permitted checks.
6. For `PASS` and `FAIL`, cite the relevant file, symbol, behavior, or other
   concrete evidence when practical.
7. The reviewer must also identify:
   - unintended scope expansion,
   - newly introduced unnecessary abstractions,
   - legacy compatibility code that was not requested,
   - duplicated implementations,
   - violations of repository dependency rules,
   - checklist items that were only partially implemented.
8. The reviewer should not modify the implementation during the review.
   Report problems first.
9. Do not run tests merely because a checklist exists. Run tests only when the
   user or `INSTRUCTION.md` explicitly authorizes them.
10. If tests are authorized, the independent reviewer may use the requested
    focused tests as additional evidence.
11. Do not mark or rewrite checklist items inside `INSTRUCTION.md`; report the
    review results separately unless the user explicitly requests the file to
    be updated.
12. If independent subagent review is unavailable, state that clearly instead
    of silently substituting implementation-agent self-review.

The final completion report must distinguish between:

- what the implementation agent changed, and
- what the independent checklist reviewer verified.

Do not claim the iteration is fully complete when any required checklist item
is `FAIL` or `UNCLEAR` without explicitly reporting that status.
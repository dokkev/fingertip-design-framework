# Repository guide

Use the applicable installed skills for reusable engineering workflow.
For any code-related task, use `$code-engineer` and follow its requested mode
(Implement, Review, Test, or Validate) and applicable references.

Do not duplicate general coding or Python-design guidance here.

Read:

- `docs/ARCHITECTURE.md` for package ownership, dependency direction,
  production boundaries, and accepted architecture;
- `docs/COMMANDS.md` for the `lit` environment, supported commands,
  external runtime dependencies, and generated-output locations.

## Current iteration

If `INSTRUCTION.md` exists and is non-empty, read it before making changes.

Instruction precedence is:

1. the user's current explicit request;
2. `INSTRUCTION.md`;
3. this `AGENTS.md`;
4. applicable reusable skill defaults.

`INSTRUCTION.md` defines the current iteration only. Do not edit, clear, or
delete it unless explicitly requested.

Inspect the current repository state before editing. Do not recreate older
architecture from stale tests, documentation, previous iterations, or memory.

Prefer migrating repository-owned callers and deleting obsolete internal APIs
over preserving legacy compatibility unless compatibility is explicitly
required.

## Scientific integrity

Do not change scientific assumptions, solver settings, objective definitions,
design bounds, acceptance thresholds, or physical models merely to make a
validation result pass.

A failed benchmark, rejected approximation, solver failure, lack of
correlation, or worse-performing candidate may be the correct scientific
result.

Keep implementation correctness, scientific validity, reference fidelity, and
preserved execution evidence separate.

Do not repeat expensive scientific computation unless the result itself was
invalidated by a defect affecting the measured quantity. Reporting,
documentation, metadata, formatting, or missing reviewer evidence alone do not
invalidate an otherwise valid computation.

When rerunning is necessary, rerun only the affected stage when practical.

## Existing changes and scope

Treat unrelated user-owned modifications as outside the current task.

Do not revert, overwrite, absorb, format, or clean up unrelated changes merely
to obtain a clean working tree.

Keep work within the requested scope. Do not automatically continue into
adjacent cleanup, refactoring, optimization, validation, or benchmark work.

Stop when the requested implementation or decision question has been answered,
or when a concrete in-scope blocker remains.

## LUMO review workflow

For substantial architecture, scientific-pipeline, evaluation, or optimization
changes, prefer an independent Reviewer before and after implementation.

Before implementation, the Reviewer is read-only and provides:

- MUST DO
- SHOULD DO
- DO NOT DO
- EXIT CRITERIA

The implementation agent then performs the requested work and only the
authorized verification.

After implementation, the Reviewer inspects the current repository state and
diff directly and reports:

- BLOCKER
- IMPORTANT
- NON-BLOCKING

The Reviewer may guide implementation direction but remains read-only unless
the user explicitly requests otherwise.

Do not treat implementation-agent self-review as independent review.

When follow-up review is required after fixes, reuse the same Reviewer when
practical. Do not create repeated fresh reviewers merely to obtain an all-PASS
result.

Subagent use does not expand task scope or authorize tests, validation, or
expensive computation beyond the current request.

## Reporting

Report what was changed, what was actually verified, what was measured or
observed, and what remains failed, blocked, or unverified.

Do not hide negative scientific results or known evidence gaps.
# Repository guide

Preserve dependency direction: model -> mesh -> fem; visualization consumes
neutral model/mesh/FEM/optics results; validation is the top-level scientific
consumer.

Production packages must not import validation or tests.
model, mesh, and fem must not import Matplotlib.
Kratos is an external environment dependency and must not be added to
pyproject.toml.
Generated files belong under output/ and remain untracked.

Keep implementations scoped to the requested task. Do not introduce new
abstractions, compatibility layers, registries, frameworks, or generalized
infrastructure unless the current task clearly requires them.

Prefer task-local functions and explicit data structures for one-off
benchmark, validation, and migration work. Promote such code into production
abstractions only when the production pipeline itself requires that behavior.

Backward compatibility with legacy internal APIs is not required unless the
user explicitly requests it. Prefer migrating repository-owned callers and
deleting obsolete code over preserving old and new interfaces together.

Do not add or run unit, smoke, integration, FEM, or validation tests unless
the user explicitly requests testing or INSTRUCTION.md explicitly requires
them.

Do not automatically chain implementation into broad testing or validation.
When testing is explicitly requested, run only the focused checks needed for
the stated contract unless broader regression testing is specifically
requested.

Do not continue into adjacent cleanup, refactoring, validation, optimization,
or feature work after completing the requested scope.

A negative benchmark or validation result is a valid task outcome. Do not
continue modifying the implementation merely to turn a negative result into
a positive one.

See docs/ARCHITECTURE.md for package ownership and dependency rules.
See docs/COMMANDS.md for supported commands.

## Current iteration

If INSTRUCTION.md exists and is non-empty, read it before making changes.

INSTRUCTION.md contains the task-specific scope, implementation requirements,
non-goals, checklist, completion criteria, and optional task execution controls
for the current iteration.

AGENTS.md contains stable repository-wide rules and default execution behavior.

Follow the user's current explicit request together with INSTRUCTION.md.
Treat INSTRUCTION.md as the baseline for the current iteration, not as
permission to ignore a later explicit user request.

Inspect the current implementation before editing rather than assuming the
repository still matches an earlier architecture.

Do not restore deleted or legacy architecture merely to satisfy stale callers,
tests, examples, or documentation.

Do not edit, rewrite, clear, or delete INSTRUCTION.md unless the user
explicitly asks you to.

If INSTRUCTION.md is missing or empty, do not infer a pending implementation
task from previous work. Follow only the user's current prompt and the stable
repository rules in AGENTS.md.

## Instruction precedence

Within repository-controlled instructions, use this order:

1. The user's current explicit request.
2. INSTRUCTION.md for the current iteration.
3. AGENTS.md stable repository rules and defaults.

A specific task instruction may override a general AGENTS.md execution default
when the override is explicit.

Do not infer an override from ambiguity, omission, stale artifacts, previous
iterations, or old tests.

Stable package ownership, dependency direction, generated-file rules, and other
repository invariants remain in force unless the user explicitly requests a
change to them.

If instructions appear to conflict and the intended override is not explicit,
do not guess. Preserve the narrower current scope and report the ambiguity.

## Task execution controls

INSTRUCTION.md may define the following controls:

- Reviewer subagent: YES | NO
- Complete checklist: YES | NO
- Fix reviewer findings: YES | NO
- Expensive reruns: INVALIDATED_ONLY | ALLOWED

If a control is not explicitly specified, use these defaults:

- Reviewer subagent: NO
- Complete checklist: YES
- Fix reviewer findings: NO
- Expensive reruns: INVALIDATED_ONLY

These controls define how far the agent should autonomously continue.
They do not expand task scope and do not override repository dependency rules.

## Reviewer subagent

If Reviewer subagent is NO:

The implementation agent performs the final checklist review itself.
Do not create or invoke a reviewer subagent solely for final verification.
Report requested checklist items as PASS, FAIL, or UNCLEAR when applicable.

If Reviewer subagent is YES:

Use the designated reviewer subagent or reviewer role for this iteration.
If an appropriate reviewer subagent already exists, reuse it.
If the environment requires creation of a reviewer and none exists, create at
most one reviewer for the iteration.
Do not create a sequence of fresh reviewers.
Invoke the reviewer only after the implementation and pre-review checklist
phase is complete.
The reviewer is read-only and must not modify implementation files,
generated artifacts, or INSTRUCTION.md.
The reviewer must inspect the current repository state and implementation
diff directly rather than trusting the implementation agent's summary.
If reviewer-subagent support is unavailable, state that clearly. Do not
silently substitute another multi-agent workflow.

## Complete checklist

If Complete checklist is YES:

Make a good-faith effort to complete every in-scope implementation checklist
item before final review.
Fix implementation defects discovered during the implementation phase when
they prevent an in-scope checklist item from being completed.
Recheck affected items after such fixes.

Do not interpret checklist completion as a requirement that every item must
ultimately be PASS.

An item may remain FAIL or UNCLEAR when:

- evidence shows the proposed approach does not work,
- the result is legitimately negative,
- an external dependency or environment limitation blocks completion,
- the required evidence cannot be obtained with the permitted checks,
- completing the item would exceed the requested scope.

Report such items rather than expanding the task indefinitely.

If Complete checklist is NO:

Perform one bounded implementation pass over the requested work.
Do not continue working solely to convert incomplete, failed, or unclear
checklist items into PASS.
Report the current state and stop.

## Fix reviewer findings

If Fix reviewer findings is NO:

After the reviewer returns its report, STOP.
Do not modify code or generated artifacts in response to reviewer findings.
Do not rerun benchmarks or tests solely to satisfy reviewer findings.
Do not invoke another reviewer.
Report PASS, FAIL, and UNCLEAR findings to the user as the final task status.

This rule applies even when Complete checklist is YES. Complete checklist governs
the implementation phase before review; it does not authorize a post-review
repair loop.

If Fix reviewer findings is YES:

Address only reviewer FAIL findings that are within the original task scope.
Do not treat UNCLEAR as automatic authorization for speculative
implementation, additional experiments, or broader validation.
Reuse the same reviewer subagent for follow-up verification when possible.
Do not create fresh reviewer agents for each review cycle.
Stop when the in-scope findings are resolved or a concrete blocker remains.
Do not broaden the task merely to obtain an all-PASS review.

## Expensive reruns

If Expensive reruns is INVALIDATED_ONLY:

Do not rerun a completed expensive computation unless a discovered
implementation defect invalidates the scientific or functional result.
Rerun only the affected stage when possible, not the entire study.

Examples that can justify an expensive rerun include:

- wrong geometry,
- wrong load case,
- wrong boundary condition,
- incorrect solver input,
- incorrect physical state,
- incorrect algorithm affecting the measured quantity,
- corrupted or unusable result data,
- a benchmark-harness bug that changes the scientific result.

The following do not justify an expensive rerun by themselves:

- missing metadata,
- missing preserved console output,
- missing reviewer evidence,
- documentation defects,
- formatting defects,
- incomplete timing labels,
- reporting-only errors,
- reviewer preference,
- absence of a saved test log when the underlying result remains valid.

Fix reporting, metadata, documentation, or evidence gaps without repeating
expensive scientific computation whenever possible.

If Expensive reruns is ALLOWED:

Expensive stages may be rerun when required to complete the explicitly
requested task.
Still avoid redundant recomputation and open-ended exploration.
Prefer rerunning only affected stages.

## Checklist review semantics

When a checklist review is required, evaluate each requested checklist item as:

- PASS
- FAIL
- UNCLEAR

Use PASS when the item is implemented and supported by direct evidence.
Use FAIL when the item is missing, incorrect, contradicted by the implementation,
or demonstrably does not satisfy the requested contract.
Use UNCLEAR when the item cannot be verified from the current implementation,
artifacts, and permitted checks.

A FAIL does not automatically mean more implementation work is required.
For exploratory, benchmark, and validation tasks, FAIL may be the scientifically
correct outcome.

For PASS and FAIL, cite the relevant file, symbol, behavior, artifact, or other
concrete evidence when practical.

The review should also identify:

- unintended scope expansion,
- unnecessary new abstractions or generalized infrastructure,
- unrequested legacy compatibility code,
- duplicated implementations,
- repository dependency-rule violations,
- checklist items that were only partially implemented.

Do not mark or rewrite checklist items inside INSTRUCTION.md unless the user
explicitly requests that file to be updated.

Do not run tests merely because a checklist exists. Testing remains governed by
the repository testing rules and explicit task authorization.

## Benchmark and validation tasks

A benchmark or validation task is complete when it provides sufficient
trustworthy evidence to answer the decision question stated in the task.

Do not continue exploring configurations merely because additional
configurations, solvers, fidelities, parameters, or optimizations are available.

Negative results are valid benchmark outcomes.

Examples:

- an alternative solver being slower is a completed result, not a reason to
  tune that solver until it wins;
- a coarse mesh failing fidelity is a completed rejection, not a reason to
  redesign the entire meshing system;
- lack of correlation between two models may be the scientific conclusion,
  not an implementation defect.

Use staged pruning when experiments are expensive.
Reject clearly unsuitable configurations early and do not continue running the
full benchmark matrix for them unless the task explicitly requires it.

Do not optimize the benchmark harness itself beyond what is necessary to obtain
trustworthy measurements.

If a benchmark-harness defect invalidates previously measured results, rerun
only the affected stages whenever possible.

Do not rerun valid expensive results merely to improve presentation,
documentation, metadata completeness, or reviewer confidence.

Once enough evidence exists to answer the task's decision question, stop.

## Scientific fidelity versus implementation correctness

Keep these concepts separate:

- implementation correctness,
- scientific or functional result validity,
- fidelity relative to a reference model,
- preserved evidence that a command was executed.

A missing log or incomplete report does not automatically invalidate a valid
scientific result.

A lower-fidelity method does not need to reproduce every quantity from a
higher-fidelity method if the task explicitly defines which downstream
quantities must be preserved.

For optimization and reduced-order studies, evaluate fidelity using the
task-defined scientific outputs rather than silently promoting unrelated
quantities into acceptance criteria.

Do not loosen acceptance thresholds merely to make a configuration pass.

If the evidence indicates a proposed approximation is unsuitable, report the
negative result and stop pursuing it unless the user explicitly requests
further investigation.

## Evidence and artifacts

Generated benchmark, validation, and analysis artifacts belong under output/
unless the task explicitly specifies another generated location.

Keep generated output untracked.

Distinguish between:

- the actual result,
- the summary of the result,
- preserved execution evidence.

If preserved evidence is incomplete, report the evidence gap.

Do not repeat expensive computation solely to manufacture reviewer evidence
unless the task explicitly requires preserved execution evidence.

Do not silently reconstruct missing scientific values from unrelated artifacts.

Do not overwrite valid historical artifacts merely to make a new result appear
cleaner.

When a discovered bug invalidates generated artifacts, clearly separate or
replace the invalid artifacts so they cannot be mistaken for valid results.

## Existing repository changes

Treat pre-existing user-owned modifications as outside the current task unless
the user explicitly includes them.

Do not revert, overwrite, rewrite, clean up, or absorb unrelated existing
changes merely to produce a clean working tree.

Pre-existing unrelated modifications are not checklist failures.

Report them separately when relevant.

Do not claim unrelated pre-existing changes were introduced by the current
task.

## Scope and stopping rule

Checklist completion does not authorize scope expansion.

Do not turn a bounded implementation task into an open-ended:

- refactor,
- cleanup campaign,
- performance-engineering project,
- benchmark campaign,
- validation campaign,
- architecture redesign,
- compatibility effort,
- test-expansion effort.

Do not pursue adjacent improvements merely because they become apparent during
the task.

Record useful follow-up opportunities in the final report rather than
implementing them unless they are required by the current task.

Stop when:

- the requested implementation is complete to the degree required by the task,
- the requested evidence has been collected,
- the decision question can be answered,
- or a concrete blocker prevents further in-scope progress.

Do not keep working merely to improve the number of PASS items.

## Final completion report

The final report must clearly distinguish:

- what was changed,
- what was verified,
- what was measured or observed,
- what remains FAIL or UNCLEAR,
- what was blocked,
- what was intentionally not pursued because it was outside scope.

If Reviewer subagent is YES, distinguish the implementation agent's work from
the reviewer subagent's findings.

If Reviewer subagent is NO, identify the checklist result as implementation-agent
self-review rather than independent review.

Do not claim an iteration is fully complete when a required checklist item
remains FAIL or UNCLEAR without explicitly reporting that status.

Do not hide negative results, rejected approaches, failed benchmark
configurations, or known evidence gaps.

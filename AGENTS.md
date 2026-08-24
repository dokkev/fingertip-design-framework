# Repository guide

For any code-related task, use `$code-engineer` and follow its requested mode
(Implement, Review, Test, or Validate).

Before editing:

- read `docs/ARCHITECTURE.md`;
- read `docs/COMMANDS.md`;
- read `INSTRUCTION.md` if it exists and is non-empty;
- inspect the current repository state.

Instruction precedence:

1. user's current explicit request;
2. `INSTRUCTION.md`;
3. this `AGENTS.md`;
4. reusable skill defaults.

Do not recreate older architecture from stale tests, documentation, legacy code,
previous iterations, or memory.

## Implementation philosophy

Simple is the Best.

Implement one concrete capability at a time.

Prefer the smallest direct implementation that satisfies the current task.
Do not design for hypothetical future requirements.

Do not introduce a reusable abstraction until a real second use case justifies
it.

A new file, class, configuration object, wrapper, or public API must have a
concrete responsibility required by the current task.

Avoid unnecessary:

- managers, contexts, factories, registries, adapters, and framework layers;
- generic solver, physics, constraint, attachment, simulation, or ray-tracing
  abstractions;
- compatibility layers for repository-internal APIs;
- wrappers around functionality already provided by upstream libraries.

When a clear implementation fits in an existing file, prefer that over creating
another module.

If an additional abstraction or production file appears necessary but was not
implied by the requested architecture, stop and explain why before adding it.

Future work being predictable does not make it part of the current task.

## External libraries

Newton and OptiX are primary implementation dependencies.

Before implementing nontrivial functionality with them:

1. inspect the installed or targeted version;
2. inspect the current public API;
3. read the corresponding official documentation;
4. inspect upstream examples or source when needed.

Prefer upstream functionality over repository-owned reimplementations.

Do not infer current behavior from legacy LUMO code, stale examples, memory, or
a different upstream development branch.

The installed or targeted API is the execution authority.

## Validation

`validation/` is for answering specific engineering or scientific questions.
It is not a reusable library layer.

Keep validation scripts procedural and local by default.

Small local helper functions are fine. Do not create validation classes,
configuration objects, runners, result APIs, or frameworks unless explicitly
required.

Production code must not depend on `validation/`.

A validation should normally:

1. construct the production objects;
2. execute the behavior;
3. measure or assert the relevant property;
4. report the result;
5. stop.

Do not change scientific assumptions, solver settings, objectives, design
bounds, physical models, or acceptance thresholds merely to make validation
pass.

A failed simulation or negative scientific result may be the correct result.

## Scope

Treat unrelated user modifications as outside the current task.

Do not revert, format, clean up, or absorb unrelated changes.

Do not automatically continue into adjacent refactoring, optimization,
validation, benchmarking, or feature work.

Do not run expensive scientific computation unless explicitly authorized.

Prefer migrating repository-owned callers and deleting obsolete internal APIs
over preserving compatibility unless compatibility is explicitly required.

For substantial architecture or scientific-pipeline changes, use an independent
read-only Reviewer when requested or when required by the current instruction.

Stop when the requested task is complete or a concrete in-scope blocker remains.

## Reporting

Report:

- what changed;
- what was actually verified;
- what was measured or observed;
- what remains failed, blocked, or unverified.

Do not hide negative results or evidence gaps.

Update the ARCHITECTURE.md accordinly with changes of the codebase
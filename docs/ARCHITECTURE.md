# Architecture

This document is the navigational map of the current accepted codebase.

A new contributor or agent should be able to read it once and understand:

- what the repository owns;
- the main runtime or data flow;
- where the important code lives;
- which package owns each responsibility;
- which entry points are canonical;
- which dependency directions are allowed;
- which architecture is intentionally absent.

Describe the code that actually exists. When a migration is incomplete,
separate the current structure from the intended target explicitly.

## At a glance

Summarize the repository in one short paragraph.

Show the primary runtime, scientific, or data flow at package level:

```text
input
-> representation
-> processing
-> output
Keep validation, tooling, offline analysis, and other consumers outside the
main production flow when they are not production dependencies.
```
## Code map

Give readers direct starting points for the major concerns in the repository.

If you need to understand...	Start here	Then inspect
primary domain model	path/	relevant implementation
runtime entry point	path/file	owning subsystem
major computation	path/file	helpers or backend
evaluation or validation	path/	relevant production API

Use real paths. Prefer canonical entry points over exhaustive file listings.

Package ownership
Package or path	Owns	Does not own
path/	responsibility, state, public concepts	neighboring responsibilities

Describe ownership, not every implementation detail.

A reader should be able to decide where new code belongs from this table.

Important execution paths

Describe the few execution paths that are important for understanding the
system.

For each path, identify:

the public or readable entry point;
the implementation that owns the behavior;
important conversions or state transitions;
where expensive or optional dependencies enter;
where results leave the subsystem.

Prefer concrete paths and symbols when they are stable architectural landmarks.

Current runtime or scientific contract

Record stable current behavior that materially changes how the codebase should
be understood.

Examples include:

evaluation dimensions or protocol structure;
lifecycle stages;
authoritative configuration;
continuous versus independent execution;
important state or identity semantics.

Keep exact configuration values in their authoritative code or configuration
source when possible. Point to that source instead of creating a second source
of truth.

Public boundaries
Interface	Owner	Consumer	Contract
symbol or artifact	package/	package/	units, ownership, semantics

Document boundaries that help a reader understand coupling between packages.

Failure and artifact semantics

Describe failure classes, persisted artifacts, cache/provenance identity, or
other result semantics when they matter to downstream interpretation.

Separate candidate/domain failures from shared infrastructure failures when
that distinction affects evaluation or orchestration.

Dependency rules

State the important allowed and forbidden dependencies.

Examples:

production packages must not import validation or tests;
low-level domain packages must not depend on execution frameworks;
optional heavy dependencies load only at their execution boundary.

Keep these rules consistent with the package ownership described above.

Intentionally absent architecture

List legacy or tempting structures that are deliberately not part of the
current architecture when recreating them would be a likely mistake.

Do not use this section as repository history. Include only absences that help
prevent incorrect new work.

Current deviations

List verified places where the current implementation does not yet match the
accepted architecture.

Keep each entry concrete and removable. This is not a backlog.

Related documents
AGENTS.md — repository-specific agent behavior
docs/COMMANDS.md — environments, commands, and output locations
# lumo2

`lumo2` is the clean implementation being rebuilt from the ground up.

The existing `lumo/` package is legacy/reference code and remains untouched by
this implementation. Functionality will be rebuilt directory by directory,
with dependencies kept explicit and minimal.

Current ownership is intentionally limited to:

- `util/` — dependency-free shared scalar helpers
- `finger/` — fingertip domain objects
- `mesh/` — mesh data and construction
- `mechanics/` — mechanics behavior
- `ray_tracing/` — ray-tracing behavior

These packages are only skeletons at this stage. No production behavior has
been migrated or implemented yet.

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

The packages are still intentionally small. The initial fingertip parameter
value objects are implemented in `finger/`; no production solver, meshing, or
transport behavior has been migrated yet.

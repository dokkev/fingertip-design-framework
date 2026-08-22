# lumo

`lumo` is the clean implementation being rebuilt from the ground up.

The previous implementation is kept in a separate legacy/reference tree and
is not imported here. Functionality is rebuilt directory by directory, with
dependencies kept explicit and minimal.

Current ownership is intentionally limited to:

- `util/` — dependency-free shared scalar helpers
- `fingertip/` — fingertip parameters and constructed geometry
- `mesh/` — `FingertipMesh` and its silicone/carrier discretization
- `optimization/` — design-space bounds and feasibility constraints
- `mechanics/` — mechanics behavior
- `ray_tracing/` — ray-tracing behavior

The packages are still intentionally small. The initial fingertip parameter
value objects and analytic `Fingertip -> Silicone/Carrier` assembly are
implemented in `fingertip/`; no production solver or transport behavior has
been migrated yet.

# Rigid mesh assets

This directory is reserved for explicit, persistent rigid meshes and their
provenance. `indenters/` is for frozen/reference indenter meshes when a
checked-in asset is needed; `objects/` is for later real-world OBJ, STL, or
PLY-derived objects.

The production primitive meshes (sphere, cylinder, box, and cube) are
generated deterministically in code by `mesh.rigid_object`; they are not
loaded from checked-in OBJ files.

Future real-object preparation should be an explicit command such as
`scripts/prepare_object_mesh.py`. It should record units and scale in
millimetres, triangulate, remove degenerate or unreferenced geometry, validate
or repair winding and watertightness where appropriate, and preserve or
explicitly record origin/recentering transforms. Runtime APIs should receive
an explicit path or object and must not assume the repository root as a data
location. Silent rescaling or recentering of imported objects is not part of
the current API.

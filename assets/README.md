# Rigid mesh assets

This directory is reserved for explicit, persistent rigid meshes and their
provenance. All OBJ, STL, or PLY-derived rigid objects belong under
`objects/`, including indenter assets.

The production primitive meshes (sphere, cylinder, box, and cube) are still
generated deterministically in code by `mesh.rigid.object`. OBJ assets can be
prepared for explicit asset-based runs with
`python scripts/assets/prepare_object_mesh.py --radius-mm 2.0`; they are not loaded
by the default evaluator yet.

The preparation command records millimetre coordinates in the OBJ and the
runtime loader requires an explicit scale. Loading validates the closed,
outward-wound `RigidObjectMesh` contract; it does not silently repair,
recenter, or rescale an asset.

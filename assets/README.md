# Rigid mesh assets

This directory is reserved for explicit, persistent rigid surface meshes and
their provenance. OBJ, STL, or PLY-derived rigid objects belong under
`objects/`.

The production primitive meshes (sphere, cylinder, box, and cube) are still
generated deterministically in code by `mesh.rigid.object`. OBJ assets can be
prepared for explicit asset-based runs with
`python scripts/assets/prepare_object_mesh.py --radius-mm 2.0`; they are not loaded
by the default evaluator yet.

The preparation command records millimetre coordinates in the OBJ. Runtime code
uses `lumo.util.mesh_io.load_mesh()` with an explicit `scale_m_per_unit` to
produce a `newton.Mesh` in metres. URDF files bypass this loader and are passed
directly to `Indenter.add_urdf()`. LUMO-owned URDF resources are installed from
`lumo/assets/objects/` instead of this generated-asset directory.

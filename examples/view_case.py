"""View one nominal explicit-contact FEA -> PLANAR_2D OptiX case."""

from __future__ import annotations

import matplotlib.pyplot as plt

from bootstrap import ensure_repository_root

ensure_repository_root()

from case import ContactState, FEA2D, FingertipCase, RayTracing2D
from mesh.indenter import IndenterSettings
from model import Fingertip
from optics import IndenterOptics
from optics.transport3d import Transport3DSettings
from visualization import plot_case


INDENTATION_MM = 0.5

# Demonstration optical parameter, not a measured indenter material value.
DEMO_INDENTER_OPTICS = IndenterOptics(
    boundary_model="dielectric",
    refractive_index=1.5,
)


def main() -> int:
    tip = Fingertip()
    indenter = IndenterSettings(initial_gap_mm=0.0)
    case = FingertipCase(
        fingertip=tip,
        fea=FEA2D(
            indenter=indenter,
            contact=ContactState(
                location_x_mm=0.0,
                indentation_mm=INDENTATION_MM,
                indenter_radius_mm=indenter.radius_mm,
            ),
        ),
        raytracing=RayTracing2D(
            settings=Transport3DSettings(
                mode="planar",
                ray_count=256,
                max_interactions=8,
                surface_u_bins=64,
                surface_z_bins=16,
                projected_grid_width=96,
                projected_grid_height=96,
                internal_z_bins=8,
                retain_projected_segments=True,
            ),
            indenter_optics=DEMO_INDENTER_OPTICS,
        ),
    )
    case.run()

    assert case.fea.result is not None
    assert case.fea.result.indenter_pose is not None
    assert case.raytracing.raw is not None
    pose = case.fea.result.indenter_pose
    patch_width = 0.0 if pose.contact_patch is None else pose.contact_patch.length
    raw = case.raytracing.raw
    print("FingertipCase summary:")
    print(f"  case_id: {case.case_id}")
    print(f"  reaction_force_n: {case.reaction_force}")
    print(f"  active_contact_nodes: {len(pose.active_contact_node_ids)}")
    print(f"  contact_patch_length_mm: {patch_width:.6g}")
    print(f"  launched_weight: {raw.launched_weight:.6g}")
    print(f"  escaped_weight: {raw.escaped_weight:.6g}")
    print(f"  object_absorbed_weight: {raw.object_absorbed_weight:.6g}")
    print(f"  object_transmitted_weight: {raw.object_transmitted_weight:.6g}")
    print(f"  energy_balance_error: {raw.energy_balance_error:.6g}")

    plot_case(case)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

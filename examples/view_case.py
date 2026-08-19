"""View one nominal FEA -> PLANAR_2D OptiX case with an indenter."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt

from bootstrap import ensure_repository_root

ensure_repository_root()

from case import (
    ContactState,
    FEA2D,
    FingertipCase,
    RayTracing2D,
    load_case,
    save_case,
)
from mesh.indenter import IndenterSettings, pose_from_fixture
from model import Fingertip
from optics import IndenterOptics
from optics.transport3d import Transport3DSettings, trace_3d
from visualization import plot_case_comparison




# Demonstration optical parameter, not a measured indenter material value.
DEMO_INDENTER_OPTICS = IndenterOptics(
    boundary_model="dielectric",
    refractive_index=2.0,
)

DEFAULT_CASE_ARTIFACT = Path("sample/view_case/case.json")


def _configured_case() -> FingertipCase:
    """Build the case configuration without starting FEA or OptiX."""
    tip = Fingertip()
    indenter = IndenterSettings(
        radius_mm=4.0,
        initial_gap_mm=0.0,
    )
    return FingertipCase(
        fingertip=tip,
        fea=FEA2D(
            indenter=indenter,
            steps=12,
            contact=ContactState(
                location_x_mm=0.0,
                indentation_mm=1.0,
                indenter_radius_mm=indenter.radius_mm,
            ),
        ),
        raytracing=RayTracing2D(
            settings=Transport3DSettings(
                mode="planar",
                retain_projected_segments=True,
            ),
            indenter_optics=DEMO_INDENTER_OPTICS,
        ),
    )


def _load_or_run_case(
    configured: FingertipCase,
    artifact: Path,
) -> FingertipCase:
    """Reuse a checked completed case, or compute it when the cache is empty."""
    cache_is_nonempty = (
        artifact.is_file()
        and bool(artifact.read_text(encoding="utf-8").strip())
    )
    if cache_is_nonempty:
        loaded = load_case(artifact)
        if loaded.case_id != configured.case_id:
            raise RuntimeError(
                "cached view_case result does not match the current configuration: "
                f"{artifact}. Empty the artifact before recomputing it."
            )
        # The persisted case also contains an optical result, but this example
        # intentionally uses the artifact only as an FEA cache.  Re-trace
        # loaded optics below so visualization changes are immediately visible.
        loaded.raytracing.raw = None
        loaded.raytracing.summary = None
        print(f"Loaded cached FEA result: {artifact}")
        return loaded

    configured.run()
    save_case(configured, artifact)
    print(f"Saved FEA/OptiX case artifact: {artifact}")
    return configured


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="View the nominal case, reusing a checked completed result when available."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help=f"cached case manifest (default: {DEFAULT_CASE_ARTIFACT})",
    )
    parser.add_argument(
        "--case-artifact",
        type=Path,
        dest="case_artifact",
        help="alternative spelling for the cached case manifest path",
    )
    args = parser.parse_args(argv)
    if args.path is not None and args.case_artifact is not None:
        parser.error("provide either path or --case-artifact, not both")
    artifact = args.case_artifact or args.path or DEFAULT_CASE_ARTIFACT

    case = _load_or_run_case(
        _configured_case(),
        artifact,
    )
    if case.raytracing.raw is None:
        case.trace()

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
    print(f"  object_interface_incident_weight: {raw.object_interface_incident_weight:.6g}")
    print(f"  object_reflected_weight: {raw.object_reflected_weight:.6g}")
    print(f"  escaped_weight: {raw.escaped_weight:.6g}")
    print(f"  object_absorbed_weight: {raw.object_absorbed_weight:.6g}")
    print(f"  object_transmitted_weight: {raw.object_transmitted_weight:.6g}")
    print(f"  energy_balance_error: {raw.energy_balance_error:.6g}")

    unloaded_optics = trace_3d(
        case.fingertip,
        case.fea.result.reference_mesh,
        settings=case.raytracing.settings,
    )
    unloaded_pose = pose_from_fixture(case.indenter_pose.fixture, 0.0)
    plot_case_comparison(
        case,
        unloaded_optics,
        unloaded_pose=unloaded_pose,
        title="Nominal fingertip: unloaded vs loaded",
    )
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

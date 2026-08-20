"""Focused FULL_3D carrier-contact optical-interface validation.

This is a bounded validation runner, not a second transport implementation.
It reuses the Newton contact path, the persisted deformed volume-state adapter,
and the shared OptiX transport core.  All generated files are written below
``output/validation/ray_tracing/carrier_contact_interface`` by default.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from physics import prepare_fingertip_mesh
from mesh import FingertipVolumeState, volume_mesh_settings_for_tier
from mesh.rigid.carrier import make_distal_phalanx_mesh
from mesh.volume.mesh import generate_volume_mesh
from finger import Fingertip, FingertipParameters
from ray_tracing.contracts.objects import CarrierOptics
from ray_tracing.optical_mechanics import (
    Transport3DSettings,
    build_fingertip_volume_state_geometry,
    trace_geometry,
)
from ray_tracing.optical_mechanics.optix_backend import create_runtime
from optimization.optical_artifact import (
    fingerprint_mapping,
    optical_physics_parameters,
    save_case_artifact,
    transport_configuration,
)
from validation.ray_tracing.deformed_state_restore import restore_deformed_optical_state
from validation.physics.multi_location_sphere_contact import (
    DEFAULT_RADIUS_MM,
    SEARCH_MAX_LOAD_INCREMENT_MM,
    SEARCH_SPHERE_SUBDIVISIONS,
    SEARCH_VBD_ITERATIONS,
    run_multi_location_sphere_contact,
)


DEFAULT_OUTPUT = Path("output/validation/ray_tracing/carrier_contact_interface")
DEFAULT_TRAVELS_MM = (0.05, 1.5, 3.0)


def _settings(*, ray_count: int = 256) -> Transport3DSettings:
    return Transport3DSettings(
        ray_count=ray_count,
        max_interactions=6,
        maximum_segment_count=4096,
        maximum_periodic_wraps=8,
        surface_u_bins=32,
        surface_z_bins=16,
        internal_grid_width=32,
        internal_grid_height=32,
        internal_z_bins=8,
        x_bounds_mm=(-16.0, 16.0),
        y_bounds_mm=(-15.0, 4.5),
        terminate_on_periodic_wrap_limit=True,
        terminate_on_no_event=True,
        retain_internal_path_field=True,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _contact_source_ids(case: Any, prepared: Any) -> tuple[int, ...]:
    indices = case.indentation.diagnostics.get("carrier_contact_vertex_indices", ())
    return tuple(
        int(prepared.source_node_ids[index])
        for index in indices
        if 0 <= int(index) < len(prepared.source_node_ids)
    )


def _contact_state(case: Any, morphology_fingerprint: str, source_ids: Iterable[int]) -> dict[str, Any]:
    source_ids = tuple(sorted(int(value) for value in source_ids))
    mapping_tolerance_mm = 0.5 * float(
        case.indentation.diagnostics.get("rigid_sdf_target_voxel_mm", 0.125)
    )
    payload = {
        "morphology_fingerprint": morphology_fingerprint,
        "normalized_location": float(case.normalized_location),
        "radius_mm": DEFAULT_RADIUS_MM,
        "travel_mm": float(case.indentation.diagnostics["post_contact_travel_mm"]),
        "carrier_contact_source_node_ids": source_ids,
        "carrier_mapping_tolerance_mm": mapping_tolerance_mm,
    }
    return {
        "contact_state_fingerprint": fingerprint_mapping(payload),
        "normalized_location": float(case.normalized_location),
        "radius_mm": DEFAULT_RADIUS_MM,
        "post_contact_travel_mm": float(case.indentation.diagnostics["post_contact_travel_mm"]),
        "first_contact_travel_mm": float(case.first_contact.travel_to_contact_mm),
        "carrier_mechanical_contact_active": bool(
            case.indentation.diagnostics.get("carrier_contact_active", False)
        ),
        "carrier_mechanical_contact_count": int(
            case.indentation.diagnostics.get(
                "carrier_interface_contact_count",
                case.indentation.diagnostics.get(
                    "max_void_bottom_carrier_contact_count", 0
                ),
            )
        ),
        "carrier_contact_source_node_ids": list(source_ids),
        "carrier_mapping_tolerance_mm": mapping_tolerance_mm,
    }


@dataclass(frozen=True)
class StateBundle:
    label: str
    travel_mm: float
    case: Any
    state: FingertipVolumeState
    contact_state: dict[str, Any]
    legacy_geometry: Any
    carrier_geometry: Any
    legacy_result: Any
    carrier_result: Any
    legacy_configuration_fingerprint: str
    carrier_configuration_fingerprint: str


def _trace_bundle(
    *,
    tip: Fingertip,
    volume_mesh: Any,
    prepared: Any,
    case: Any,
    label: str,
    travel_mm: float,
    runtime: Any,
    settings: Transport3DSettings,
    candidate_root: Path,
) -> StateBundle:
    source_ids = _contact_source_ids(case, prepared)
    contact_state = _contact_state(case, volume_mesh.morphology_fingerprint, source_ids)
    mapping_tolerance_mm = 0.5 * float(
        case.indentation.diagnostics.get("rigid_sdf_target_voxel_mm", 0.125)
    )
    contact_state["carrier_mapping_tolerance_mm"] = mapping_tolerance_mm
    carrier_mesh = make_distal_phalanx_mesh(volume_mesh.solid)
    restored = restore_deformed_optical_state(
        tip,
        volume_mesh,
        prepared,
        case.mechanics_artifact_path,
        case.mechanics_artifact_sha256,
        carrier_mesh=carrier_mesh,
        carrier_optics=CarrierOptics("absorber"),
        carrier_mapping_tolerance_mm=mapping_tolerance_mm,
        metadata={
            "contact_state_fingerprint": contact_state["contact_state_fingerprint"],
        },
    )
    legacy_geometry = build_fingertip_volume_state_geometry(
        tip,
        restored.state,
        carrier_mesh=carrier_mesh,
        full3d_surface_provenance="actual_deformed_3d_volume_state",
        metadata={
            "contact_state_fingerprint": contact_state["contact_state_fingerprint"],
        },
        carrier_mapping_tolerance_mm=mapping_tolerance_mm,
    )
    material = optical_physics_parameters(tip)
    configuration = transport_configuration(
        settings,
        material=material,
    )
    legacy_configuration = {
        **configuration,
        "carrier_contact_geometry": {"enabled": False},
    }
    carrier_configuration = {
        **configuration,
        "carrier_contact_geometry": {
            "enabled": True,
            "boundary_model": "absorber",
            "mapping_tolerance_mm": mapping_tolerance_mm,
        },
    }
    legacy_configuration_fingerprint = fingerprint_mapping(legacy_configuration)
    carrier_configuration_fingerprint = fingerprint_mapping(carrier_configuration)
    legacy_result = trace_geometry(
        tip, legacy_geometry, settings=settings, runtime=runtime
    )
    carrier_result = trace_geometry(
        tip, restored.geometry, settings=settings, runtime=runtime
    )

    state_root = candidate_root / label
    contract_base = {
        "schema": "carrier-contact-interface-validation-v1",
        "morphology_fingerprint": volume_mesh.morphology_fingerprint,
        "contact_state": contact_state,
        "mechanics_source": str(restored.artifact_path),
        "mechanics_artifact_sha256": restored.artifact_sha256,
        "mapping_method": "exact_semantic_surface_triangle_any_contact_vertex",
        "carrier_boundary_model": "absorber",
        "transport_configuration_fingerprint": legacy_configuration_fingerprint,
    }
    save_case_artifact(
        state_root / "legacy_air.json",
        legacy_result,
        {**contract_base, "interface_semantics": "legacy_silicone_air"},
    )
    save_case_artifact(
        state_root / "carrier_absorber.json",
        carrier_result,
        {
            **contract_base,
            "interface_semantics": "contacted_patch_absorber",
            "transport_configuration_fingerprint": carrier_configuration_fingerprint,
        },
    )
    return StateBundle(
        label=label,
        travel_mm=float(travel_mm),
        case=case,
        state=restored.state,
        contact_state=contact_state,
        legacy_geometry=legacy_geometry,
        carrier_geometry=restored.geometry,
        legacy_result=legacy_result,
        carrier_result=carrier_result,
        legacy_configuration_fingerprint=legacy_configuration_fingerprint,
        carrier_configuration_fingerprint=carrier_configuration_fingerprint,
    )


def _result_summary(bundle: StateBundle) -> dict[str, Any]:
    legacy = bundle.legacy_result
    carrier = bundle.carrier_result
    field_difference = np.asarray(carrier.field) - np.asarray(legacy.field)
    return {
        "label": bundle.label,
        "travel_mm": bundle.travel_mm,
        "mechanics": dict(bundle.case.indentation.diagnostics),
        "contact_state": bundle.contact_state,
        "carrier_optical_contact_triangle_count": int(
            carrier.carrier_contact_triangle_count
        ),
        "legacy_air": {
            "escaped_weight": legacy.escaped_weight,
            "absorbed_weight": legacy.absorbed_weight,
            "energy_balance_error": legacy.energy_balance_error,
        },
        "carrier_absorber": {
            "escaped_weight": carrier.escaped_weight,
            "escaped_transport_fraction": carrier.escaped_weight / max(
                carrier.launched_weight, 1.0e-30
            ),
            "absorbed_weight": carrier.absorbed_weight,
            "carrier_absorbed_weight": carrier.carrier_absorbed_weight,
            "carrier_absorption_fraction": carrier.carrier_absorbed_weight / max(
                carrier.launched_weight, 1.0e-30
            ),
            "energy_balance_error": carrier.energy_balance_error,
        },
        "transport_configuration_fingerprints": {
            "legacy_air": bundle.legacy_configuration_fingerprint,
            "carrier_absorber": bundle.carrier_configuration_fingerprint,
            "changed": bool(
                bundle.legacy_configuration_fingerprint
                != bundle.carrier_configuration_fingerprint
            ),
        },
        "same_geometry_ab": {
            "field_l1": float(np.sum(np.abs(field_difference))),
            "field_l2": float(np.linalg.norm(field_difference)),
            "changed": bool(np.any(np.abs(field_difference) > 1.0e-14)),
        },
    }


def _plot_bundle(bundle: StateBundle, output: Path) -> None:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    fields = [
        np.sum(bundle.legacy_result.field, axis=2).T,
        np.sum(bundle.carrier_result.field, axis=2).T,
        np.sum(bundle.carrier_result.field - bundle.legacy_result.field, axis=2).T,
    ]
    vmax = max(float(np.max(np.abs(fields[0]))), float(np.max(np.abs(fields[1]))), 1.0e-30)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    for axis, field, title in zip(
        axes,
        fields,
        ("legacy silicone-air", "contacted patch absorber", "carrier minus legacy"),
    ):
        limit = vmax if title != "carrier minus legacy" else vmax
        image = axis.imshow(
            field,
            origin="lower",
            cmap="coolwarm" if title == "carrier minus legacy" else "viridis",
            vmin=-limit if title == "carrier minus legacy" else 0.0,
            vmax=limit,
            aspect="auto",
        )
        axis.set_title(title)
        axis.set_xlabel("x bin")
        axis.set_ylabel("y bin")
        fig.colorbar(image, ax=axis, shrink=0.8)
    fig.savefig(output / f"{bundle.label}_optical_ab.png", dpi=180)
    plt.close(fig)


def _plot_energy(records: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [record["label"] for record in records]
    escaped = [record["carrier_absorber"]["escaped_weight"] for record in records]
    silicone = [record["carrier_absorber"]["absorbed_weight"] for record in records]
    carrier = [record["carrier_absorber"]["carrier_absorbed_weight"] for record in records]
    positions = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    axis.bar(positions, escaped, label="escaped")
    axis.bar(positions, silicone, bottom=escaped, label="silicone absorbed")
    axis.bar(
        positions,
        carrier,
        bottom=np.asarray(escaped) + np.asarray(silicone),
        label="carrier absorbed",
    )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("weight")
    axis.set_title("FULL_3D energy accounting")
    axis.legend(frameon=False)
    fig.savefig(output / "energy_accounting.png", dpi=180)
    plt.close(fig)


def run_validation(
    *,
    output: Path = DEFAULT_OUTPUT,
    device: str = "cuda:0",
    void_height_mm: float = 1.0,
    travels_mm: tuple[float, float, float] = DEFAULT_TRAVELS_MM,
    ray_count: int = 256,
) -> dict[str, Any]:
    if len(travels_mm) != 3 or any(float(value) <= 0.0 for value in travels_mm):
        raise ValueError("travels_mm must contain three positive values")
    parameters = FingertipParameters(void_height=float(void_height_mm))
    tip = Fingertip(parameters)
    volume_mesh = generate_volume_mesh(
        tip.solid(),
        volume_mesh_settings_for_tier("search"),
    )
    prepared = prepare_fingertip_mesh(volume_mesh)
    runtime = create_runtime()
    settings = _settings(ray_count=ray_count)
    candidate_root = output / "observations"
    bundles: list[StateBundle] = []
    labels = ("open", "first_contact", "deep_contact")
    for label, travel in zip(labels, travels_mm):
        contact = run_multi_location_sphere_contact(
            parameters=parameters,
            device=device,
            radius_mm=DEFAULT_RADIUS_MM,
            travel_mm=float(travel),
            normalized_locations=(0.50,),
            artifact_dir=output / "mechanics" / "search" / label,
            sphere_subdivisions=SEARCH_SPHERE_SUBDIVISIONS,
            max_load_increment_mm=SEARCH_MAX_LOAD_INCREMENT_MM,
            vbd_iterations=SEARCH_VBD_ITERATIONS,
            carrier_contact=True,
        )
        case = contact.locations[0]
        bundles.append(
            _trace_bundle(
                tip=tip,
                volume_mesh=volume_mesh,
                prepared=prepared,
                case=case,
                label=label,
                travel_mm=float(travel),
                runtime=runtime,
                settings=settings,
                candidate_root=candidate_root,
            )
        )

    records = [_result_summary(bundle) for bundle in bundles]
    open_record = records[0]
    first_record = records[1]
    deep_record = records[2]
    open_on_off = {
        "field_l1": open_record["same_geometry_ab"]["field_l1"],
        "carrier_contact_triangle_count": open_record[
            "carrier_optical_contact_triangle_count"
        ],
        "carrier_absorbed_weight": open_record["carrier_absorber"]["carrier_absorbed_weight"],
        "passed": bool(
            open_record["carrier_optical_contact_triangle_count"] == 0
            and open_record["same_geometry_ab"]["field_l1"] <= 1.0e-12
            and open_record["carrier_absorber"]["carrier_absorbed_weight"] == 0.0
        ),
    }
    first_deep_checks = {
        "first_contact_triangle_count": first_record[
            "carrier_optical_contact_triangle_count"
        ],
        "deep_contact_triangle_count": deep_record[
            "carrier_optical_contact_triangle_count"
        ],
        "first_or_deep_contact_present": bool(
            first_record["carrier_optical_contact_triangle_count"] > 0
            or deep_record["carrier_optical_contact_triangle_count"] > 0
        ),
        "deep_absorption_finite": bool(
            np.isfinite(deep_record["carrier_absorber"]["carrier_absorbed_weight"])
        ),
    }
    _write_json(output / "open_gap_regression.json", open_on_off)
    _write_json(output / "open_first_deep_contact.json", records)
    _write_json(
        output / "optical_ab_same_geometry.json",
        {record["label"]: record["same_geometry_ab"] for record in records},
    )
    _write_json(output / "energy_accounting.json", records)
    for bundle in bundles:
        _plot_bundle(bundle, output / "plots")
    _plot_energy(records, output / "plots")

    summary = {
        "schema": "carrier-contact-interface-validation-v1",
        "parameters": parameters.__dict__,
        "optical_mode": "FULL_3D",
        "carrier_boundary_model": "absorber",
        "mapping_method": "exact_semantic_surface_triangle_any_contact_vertex",
        "search_contract": {
            "sphere_subdivisions": SEARCH_SPHERE_SUBDIVISIONS,
            "max_load_increment_mm": SEARCH_MAX_LOAD_INCREMENT_MM,
            "vbd_iterations": SEARCH_VBD_ITERATIONS,
        },
        "records": records,
        "open_gap_regression": open_on_off,
        "contact_progression": first_deep_checks,
        "pass": bool(open_on_off["passed"] and first_deep_checks["first_or_deep_contact_present"] and first_deep_checks["deep_absorption_finite"]),
    }
    _write_json(output / "config.json", {"parameters": parameters.__dict__, "settings": settings.__dict__})
    _write_json(output / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--void-height", type=float, default=1.0)
    parser.add_argument("--ray-count", type=int, default=256)
    parser.add_argument(
        "--travels",
        type=float,
        nargs=3,
        metavar=("OPEN", "FIRST", "DEEP"),
        default=DEFAULT_TRAVELS_MM,
    )
    args = parser.parse_args(argv)
    summary = run_validation(
        output=args.output,
        device=args.device,
        void_height_mm=args.void_height,
        travels_mm=tuple(args.travels),
        ray_count=args.ray_count,
    )
    print(json.dumps({"status": "PASS" if summary["pass"] else "FAIL", "output": str(args.output)}, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

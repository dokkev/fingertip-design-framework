"""Inventory existing generated 3D FEA reference artifacts without rerunning FEA."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tests.validation.common.io import atomic_write_json, strict_read_json

from .fea3d_reference import FEA3DReferenceError, load_fea3d_reference


@dataclass(frozen=True)
class ArtifactFamily:
    name: str
    root: Path


ARTIFACT_FAMILIES = (
    ArtifactFamily(
        "overnight_force_localized_trend/fea3d",
        Path("output/validation/overnight_force_localized_trend/fea3d"),
    ),
    ArtifactFamily(
        "overnight_24_pair_trend/fea3d",
        Path("output/validation/overnight_24_pair_trend/fea3d"),
    ),
    ArtifactFamily(
        "3d_migration/m5_cases",
        Path("output/validation/3d_migration/m5_cases"),
    ),
    ArtifactFamily(
        "overnight_force_localized_trend/calibration_3d",
        Path("output/validation/overnight_force_localized_trend/calibration_3d"),
    ),
    ArtifactFamily(
        "overnight_force_localized_trend/smoke_3d",
        Path("output/validation/overnight_force_localized_trend/smoke_3d"),
    ),
)


def _nested_case_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = payload.get("payload")
    return nested if isinstance(nested, Mapping) else {}


def _case_field(payload: Mapping[str, Any], key: str) -> Any:
    nested = _nested_case_payload(payload)
    if key in payload:
        return payload[key]
    return nested.get(key)


def _case_force(payload: Mapping[str, Any]) -> float | None:
    direct = _case_field(payload, "force_target_n")
    if direct is not None:
        return float(direct)
    load = _case_field(payload, "load")
    if isinstance(load, Mapping) and load.get("target_force_n") is not None:
        return float(load["target_force_n"])
    force_control = _case_field(payload, "force_control")
    if isinstance(force_control, Mapping) and force_control.get("target_force_n") is not None:
        return float(force_control["target_force_n"])
    return None


def _resolve_native_manifest(
    case_path: Path,
    payload: Mapping[str, Any],
) -> Path | None:
    nested = _nested_case_payload(payload)
    raw = payload.get("native_manifest")
    if raw is None:
        raw = nested.get("native_3d_artifact")
    candidates: list[Path] = []
    if isinstance(raw, str) and raw:
        reference = Path(raw)
        candidates.extend(
            (
                reference,
                Path.cwd() / reference,
                case_path.parent / reference,
                case_path.parent / reference.name,
            )
        )
    candidates.extend(
        sorted(case_path.parent.glob("native_states/*.json"))
        + sorted(case_path.parent.glob("native_3d_states/*.json"))
    )

    morphology = _case_field(payload, "morphology_fingerprint")
    force = _case_force(payload)
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        try:
            manifest = strict_read_json(candidate)
        except (OSError, ValueError):
            continue
        if manifest.get("schema") != "native-3d-fea-state-v1":
            continue
        if morphology is not None and manifest.get("morphology_fingerprint") not in (None, morphology):
            continue
        manifest_force = manifest.get("force_target_n")
        if force is not None and manifest_force is not None and float(manifest_force) != force:
            continue
        return candidate
    return None


def _primary_case_files(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for path in sorted(root.rglob("*.json")):
        try:
            payload = strict_read_json(path)
        except (OSError, ValueError):
            continue
        if payload.get("schema") == "native-3d-fea-state-v1":
            continue
        nested = _nested_case_payload(payload)
        if (
            "native_manifest" in payload
            or "native_3d_artifact" in nested
            or payload.get("schema") == "m5-production-case-v1"
        ):
            files.append(path)
    return tuple(files)


def _record_for_case(family: ArtifactFamily, case_path: Path) -> dict[str, Any]:
    payload = strict_read_json(case_path)
    manifest_path = _resolve_native_manifest(case_path, payload)
    case_id = _case_field(payload, "case_id") or _case_field(payload, "key") or case_path.stem
    status = _case_field(payload, "status") or _case_field(payload, "outcome") or "unknown"
    record: dict[str, Any] = {
        "artifact_family": family.name,
        "case_name": str(case_id),
        "case_artifact_path": str(case_path.resolve()),
        "source_path": str(case_path.resolve()),
        "status": "incomplete" if manifest_path is None else str(status),
        "morphology_fingerprint": _case_field(payload, "morphology_fingerprint"),
        "force_target_n": _case_force(payload),
        "prescribed_travel_mm": None,
        "load_contact_type": None,
        "node_count": None,
        "reference_coordinates_available": False,
        "deformed_coordinates_available": False,
        "tetrahedra_connectivity_available": False,
        "semantic_surface_info_available": False,
        "reaction_force_available": _case_field(payload, "reaction_force_n") is not None
        or isinstance(_case_field(payload, "reaction_diagnostics"), Mapping),
        "direct_node_correspondence_provable": False,
        "native_manifest_path": None,
        "loadable": False,
        "successful_state": False,
        "schema_variants": sorted({str(payload.get("schema", "unknown"))}),
    }
    configuration = _case_field(payload, "configuration")
    if isinstance(configuration, Mapping):
        settings = configuration.get("settings")
        if isinstance(settings, Mapping) and settings.get("indentation_mm") is not None:
            record["prescribed_travel_mm"] = float(settings["indentation_mm"])
    load = _case_field(payload, "load")
    if isinstance(load, Mapping):
        record["load_contact_type"] = load.get("load_type")
        if record["force_target_n"] is None and load.get("target_force_n") is not None:
            record["force_target_n"] = float(load["target_force_n"])

    if manifest_path is None:
        record["error"] = "no native-3d-fea-state-v1 manifest was discoverable"
        return record

    record["native_manifest_path"] = str(manifest_path)
    try:
        reference = load_fea3d_reference(manifest_path, case_metadata=payload)
    except (FEA3DReferenceError, OSError, ValueError) as exception:
        record["status"] = "FAIL"
        record["error"] = str(exception)
        return record

    record.update(
        {
            "status": "PASS" if str(status) in {"PASS", "pass"} else str(status),
            "morphology_fingerprint": reference.morphology_fingerprint or record["morphology_fingerprint"],
            "node_count": reference.node_count,
            "reference_coordinates_available": True,
            "deformed_coordinates_available": True,
            "tetrahedra_connectivity_available": bool(
                reference.provenance.get("tetrahedra_node_ids_available")
            ),
            "semantic_surface_info_available": reference.provenance.get(
                "surface_semantic_tag_count"
            )
            is not None,
            "direct_node_correspondence_provable": reference.direct_node_correspondence_provable,
            "loadable": True,
            "successful_state": str(status) in {"PASS", "pass"},
            "source_path": str(reference.source_path),
            "provenance": dict(reference.provenance),
            "schema_variants": sorted(
                {str(payload.get("schema", "unknown")), str(reference.provenance["artifact_schema"])}
            ),
        }
    )
    manifest_load = reference.load_metadata
    if record["force_target_n"] is None and manifest_load.get("force_target_n") is not None:
        record["force_target_n"] = float(manifest_load["force_target_n"])
    if record["prescribed_travel_mm"] is None and manifest_load.get("total_prescribed_travel_mm") is not None:
        record["prescribed_travel_mm"] = manifest_load["total_prescribed_travel_mm"]
    if record["load_contact_type"] is None:
        record["load_contact_type"] = (
            "localized_load_only" if manifest_load.get("localized_load_only") else "external_contact_or_unknown"
        )
    return record


def build_fea3d_inventory(repo_root: str | Path = ".") -> dict[str, Any]:
    """Scan the five established local artifact families without solver calls."""

    root = Path(repo_root).resolve()
    families: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    for family in ARTIFACT_FAMILIES:
        family_root = root / family.root
        if not family_root.is_dir():
            families.append(
                {
                    "artifact_family": family.name,
                    "root": str(family_root),
                    "discovered_cases": 0,
                    "loadable": 0,
                    "successful_state": 0,
                    "direct_node_correspondence_provable": 0,
                    "unsupported_for_direct_node_comparison": 0,
                    "schema_variants": [],
                    "records": [],
                }
            )
            continue
        records = [_record_for_case(family, path) for path in _primary_case_files(family_root)]
        schemas = sorted({schema for record in records for schema in record.get("schema_variants", [])})
        family_summary = {
            "artifact_family": family.name,
            "root": str(family_root),
            "discovered_cases": len(records),
            "loadable": sum(bool(record["loadable"]) for record in records),
            "successful_state": sum(bool(record["successful_state"]) for record in records),
            "direct_node_correspondence_provable": sum(
                bool(record["direct_node_correspondence_provable"]) for record in records
            ),
            "unsupported_for_direct_node_comparison": sum(
                bool(record["loadable"] and not record["direct_node_correspondence_provable"])
                for record in records
            ),
            "unique_loadable_states": len(
                {
                    record["provenance"]["state_path"]
                    for record in records
                    if record["loadable"] and "provenance" in record
                }
            ),
            "schema_variants": schemas,
            "records": records,
        }
        families.append(family_summary)
        all_records.extend(records)

    return {
        "schema": "physics-fea3d-reference-inventory-v1",
        "artifact_roots_are_generated_evidence": True,
        "kratos_rerun": False,
        "families": families,
        "totals": {
            "discovered_cases": len(all_records),
            "loadable": sum(bool(record["loadable"]) for record in all_records),
            "successful_state": sum(bool(record["successful_state"]) for record in all_records),
            "direct_node_correspondence_provable": sum(
                bool(record["direct_node_correspondence_provable"]) for record in all_records
            ),
            "unsupported_for_direct_node_comparison": sum(
                bool(record["loadable"] and not record["direct_node_correspondence_provable"])
                for record in all_records
            ),
            "unique_loadable_states": len(
                {
                    record["provenance"]["state_path"]
                    for record in all_records
                    if record["loadable"] and "provenance" in record
                }
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/validation/physics/fea3d_reference_inventory.json"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    inventory = build_fea3d_inventory(args.repo_root)
    atomic_write_json(args.output, inventory)
    print(f"PASS: wrote {args.output} ({inventory['totals']['loadable']} loadable states)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ARTIFACT_FAMILIES", "build_fea3d_inventory", "main"]

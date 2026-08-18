"""Deterministic cleanup for the aborted OptiX-infrastructure campaign."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any, Mapping

from validation.common.io import atomic_write_json, strict_read_json


ABORTED_OPTIX_HEADER_FAILURE_SIGNATURE = (
    "Could not find a valid OptiX include directory"
)


@dataclass(frozen=True)
class RegistryCleanupReport:
    """Counts and retained provenance from one deterministic cleanup."""

    before_count: int
    removed_count: int
    after_count: int
    removed_keys: tuple[str, ...]
    retained_statuses: tuple[str, ...]
    retained_campaign_ids: tuple[str, ...]


def _is_proven_infrastructure_failure(
    record: Mapping[str, Any],
    *,
    failure_signature: str,
) -> bool:
    evaluation = record.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return False
    message = str(record.get("failure_message") or "")
    return (
        record.get("phase") != "nominal"
        and record.get("status") == "optics_failure"
        and evaluation.get("fem_trajectories_attempted") == 0
        and evaluation.get("captured_states_attempted") == 0
        and failure_signature in message
        and bool(record.get("registry_key"))
    )


def cleanup_aborted_infrastructure_records(
    registry_path: str | Path,
    checkpoint_path: str | Path,
    *,
    campaign_id: str,
    backup_path: str | Path,
    failure_signature: str = ABORTED_OPTIX_HEADER_FAILURE_SIGNATURE,
) -> RegistryCleanupReport:
    """Remove only checkpoint-proven infrastructure failures from a registry.

    The source checkpoint is the allow-list for removable keys. Registry
    provenance must independently match the campaign and checkpoint artifact,
    so a same-message historical or candidate-specific failure is retained.
    """
    registry_file = Path(registry_path).expanduser().resolve()
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    backup_file = Path(backup_path).expanduser().resolve()
    if not campaign_id:
        raise ValueError("campaign_id must be non-empty")
    if backup_file.exists():
        raise FileExistsError(f"refusing to overwrite registry backup: {backup_file}")

    registry = strict_read_json(registry_file)
    checkpoint = strict_read_json(checkpoint_file)
    raw_registry_records = registry.get("records")
    raw_checkpoint_records = checkpoint.get("records")
    if not isinstance(raw_registry_records, Mapping):
        raise ValueError("registry records must be an object")
    if not isinstance(raw_checkpoint_records, list):
        raise ValueError("checkpoint records must be a list")

    checkpoint_keys = {
        str(record["registry_key"]): record
        for record in raw_checkpoint_records
        if isinstance(record, Mapping)
        and _is_proven_infrastructure_failure(
            record,
            failure_signature=failure_signature,
        )
    }
    removable: set[str] = set()
    for key, payload in raw_registry_records.items():
        if key not in checkpoint_keys or not isinstance(payload, Mapping):
            continue
        if payload.get("first_campaign_id") != campaign_id:
            continue
        artifact_path = payload.get("result_artifact_path")
        if artifact_path is None:
            continue
        if Path(str(artifact_path)).expanduser().resolve() != checkpoint_file:
            continue
        if payload.get("failure_category") != "optics_failure":
            continue
        if failure_signature not in str(payload.get("failure_message") or ""):
            continue
        removable.add(str(key))

    backup_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(registry_file, backup_file)
    retained_records = {
        str(key): value
        for key, value in raw_registry_records.items()
        if str(key) not in removable
    }
    updated = dict(registry)
    updated["records"] = {
        key: retained_records[key] for key in sorted(retained_records)
    }
    atomic_write_json(registry_file, updated)

    retained_values = [
        value for value in retained_records.values() if isinstance(value, Mapping)
    ]
    return RegistryCleanupReport(
        before_count=len(raw_registry_records),
        removed_count=len(removable),
        after_count=len(retained_records),
        removed_keys=tuple(sorted(removable)),
        retained_statuses=tuple(
            sorted({str(value.get("status")) for value in retained_values})
        ),
        retained_campaign_ids=tuple(
            sorted(
                {
                    str(value.get("first_campaign_id"))
                    for value in retained_values
                }
            )
        ),
    )


__all__ = [
    "ABORTED_OPTIX_HEADER_FAILURE_SIGNATURE",
    "RegistryCleanupReport",
    "cleanup_aborted_infrastructure_records",
]

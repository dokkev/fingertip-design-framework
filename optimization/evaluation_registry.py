"""Persistent exact-morphology provenance for production evaluations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from optimization.design_space import OPTIMIZABLE_PARAMETER_NAMES


REGISTRY_SCHEMA_VERSION = 2
SUPPORTED_EVALUATION_STATUSES = frozenset(
    {
        "success",
        "invalid_design",
        "mesh_failure",
        "domain_incompatible",
        "mechanics_failure",
        "fea_failure",
        "optics_failure",
    }
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real number")
    try:
        resolved = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite real number") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved


def canonical_morphology(parameters: Mapping[str, object]) -> dict[str, str]:
    """Return a lossless, deterministic representation of the six fields."""
    expected = set(OPTIMIZABLE_PARAMETER_NAMES)
    supplied = set(parameters)
    if supplied != expected:
        raise ValueError(
            "registry morphology must contain exactly the six production "
            f"parameters; missing={sorted(expected - supplied)!r}, "
            f"unknown={sorted(supplied - expected)!r}"
        )
    return {
        name: _finite_float(name, parameters[name]).hex()
        for name in OPTIMIZABLE_PARAMETER_NAMES
    }


def evaluation_key(
    contract_id: str,
    parameters: Mapping[str, object],
) -> str:
    """Build the exact registry key for one contract and morphology."""
    if not isinstance(contract_id, str) or not contract_id:
        raise ValueError("contract_id must be a non-empty string")
    canonical = canonical_morphology(parameters)
    fields = ";".join(
        f"{name}={canonical[name]}" for name in OPTIMIZABLE_PARAMETER_NAMES
    )
    return f"{contract_id}|{fields}"


@dataclass(frozen=True)
class EvaluationRegistryRecord:
    """One original production evaluation and its lightweight provenance."""

    key: str
    contract_id: str
    morphology: Mapping[str, float]
    canonical_morphology: Mapping[str, str]
    status: str
    first_trial_index: int
    first_campaign_id: str
    result_artifact_path: str | None
    minimum_auc: float | None
    objective_value: float | None
    failure_category: str | None
    failure_message: str | None
    failure_scenario: str | None
    evaluation_wall_time_seconds: float | None
    created_at: str
    duplicate_count: int = 0
    last_duplicate_trial_index: int | None = None
    last_duplicate_campaign_id: str | None = None

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("registry record key must be non-empty")
        if not self.contract_id:
            raise ValueError("registry record contract_id must be non-empty")
        if self.status not in SUPPORTED_EVALUATION_STATUSES:
            raise ValueError(f"unsupported registry status: {self.status!r}")
        if isinstance(self.first_trial_index, bool) or self.first_trial_index < 0:
            raise ValueError("first_trial_index must be nonnegative")
        if not self.first_campaign_id:
            raise ValueError("first_campaign_id must be non-empty")
        if self.minimum_auc is not None:
            minimum_auc = _finite_float("minimum_auc", self.minimum_auc)
            if minimum_auc < 0.0:
                raise ValueError("minimum_auc must be nonnegative")
            object.__setattr__(self, "minimum_auc", minimum_auc)
        if self.objective_value is not None:
            objective_value = _finite_float("objective_value", self.objective_value)
            if objective_value < 0.0:
                raise ValueError("objective_value must be nonnegative")
            object.__setattr__(self, "objective_value", objective_value)
        if self.status == "success" and (
            self.minimum_auc is None and self.objective_value is None
        ):
            raise ValueError(
                "successful registry record requires minimum_auc or objective_value"
            )
        if self.status != "success" and (
            self.minimum_auc is not None or self.objective_value is not None
        ):
            raise ValueError(
                "failed registry record must not carry minimum_auc or objective_value"
            )
        if self.evaluation_wall_time_seconds is not None:
            wall = _finite_float(
                "evaluation_wall_time_seconds",
                self.evaluation_wall_time_seconds,
            )
            if wall < 0.0:
                raise ValueError("evaluation_wall_time_seconds must be nonnegative")
            object.__setattr__(self, "evaluation_wall_time_seconds", wall)
        if isinstance(self.duplicate_count, bool) or self.duplicate_count < 0:
            raise ValueError("duplicate_count must be nonnegative")
        object.__setattr__(self, "morphology", MappingProxyType(dict(self.morphology)))
        object.__setattr__(
            self,
            "canonical_morphology",
            MappingProxyType(dict(self.canonical_morphology)),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "morphology": dict(self.morphology),
            "canonical_morphology": dict(self.canonical_morphology),
            "status": self.status,
            "first_trial_index": self.first_trial_index,
            "first_campaign_id": self.first_campaign_id,
            "result_artifact_path": self.result_artifact_path,
            "minimum_auc": self.minimum_auc,
            "objective_value": self.objective_value,
            "failure_category": self.failure_category,
            "failure_message": self.failure_message,
            "failure_scenario": self.failure_scenario,
            "evaluation_wall_time_seconds": self.evaluation_wall_time_seconds,
            "created_at": self.created_at,
            "duplicate_count": self.duplicate_count,
            "last_duplicate_trial_index": self.last_duplicate_trial_index,
            "last_duplicate_campaign_id": self.last_duplicate_campaign_id,
        }

    @classmethod
    def from_json(
        cls,
        key: str,
        payload: Mapping[str, Any],
    ) -> "EvaluationRegistryRecord":
        return cls(
            key=key,
            contract_id=str(payload["contract_id"]),
            morphology={
                name: _finite_float(name, payload["morphology"][name])
                for name in OPTIMIZABLE_PARAMETER_NAMES
            },
            canonical_morphology={
                name: str(payload["canonical_morphology"][name])
                for name in OPTIMIZABLE_PARAMETER_NAMES
            },
            status=str(payload["status"]),
            first_trial_index=int(payload["first_trial_index"]),
            first_campaign_id=str(payload["first_campaign_id"]),
            result_artifact_path=payload.get("result_artifact_path"),
            minimum_auc=payload.get("minimum_auc"),
            objective_value=payload.get("objective_value", payload.get("minimum_auc")),
            failure_category=payload.get("failure_category"),
            failure_message=payload.get("failure_message"),
            failure_scenario=payload.get("failure_scenario"),
            evaluation_wall_time_seconds=payload.get(
                "evaluation_wall_time_seconds"
            ),
            created_at=str(payload["created_at"]),
            duplicate_count=int(payload.get("duplicate_count", 0)),
            last_duplicate_trial_index=payload.get("last_duplicate_trial_index"),
            last_duplicate_campaign_id=payload.get("last_duplicate_campaign_id"),
        )


class EvaluationRegistry:
    """Small crash-safe JSON index for exact production evaluations."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records = self._load()

    def lookup(
        self,
        contract_id: str,
        parameters: Mapping[str, object],
    ) -> EvaluationRegistryRecord | None:
        """Return the original record for an exact contract/morphology pair."""
        return self._records.get(evaluation_key(contract_id, parameters))

    def records_for_contract(
        self,
        contract_id: str,
    ) -> tuple[EvaluationRegistryRecord, ...]:
        """Return each exact historical result for one contract once."""
        if not isinstance(contract_id, str) or not contract_id:
            raise ValueError("contract_id must be a non-empty string")
        return tuple(
            record
            for _, record in sorted(self._records.items())
            if record.contract_id == contract_id
        )

    def register(
        self,
        contract_id: str,
        parameters: Mapping[str, object],
        *,
        status: str,
        first_trial_index: int,
        first_campaign_id: str,
        result_artifact_path: str | None,
        minimum_auc: float | None,
        failure_category: str | None,
        failure_message: str | None,
        failure_scenario: str | None,
        evaluation_wall_time_seconds: float | None,
        objective_value: float | None = None,
    ) -> EvaluationRegistryRecord:
        """Persist one original result; never overwrite an existing result."""
        key = evaluation_key(contract_id, parameters)
        if key in self._records:
            raise KeyError(f"evaluation is already registered: {key}")
        canonical = canonical_morphology(parameters)
        record = EvaluationRegistryRecord(
            key=key,
            contract_id=contract_id,
            morphology={
                name: _finite_float(name, parameters[name])
                for name in OPTIMIZABLE_PARAMETER_NAMES
            },
            canonical_morphology=canonical,
            status=status,
            first_trial_index=first_trial_index,
            first_campaign_id=first_campaign_id,
            result_artifact_path=result_artifact_path,
            minimum_auc=minimum_auc,
            objective_value=objective_value,
            failure_category=failure_category,
            failure_message=failure_message,
            failure_scenario=failure_scenario,
            evaluation_wall_time_seconds=evaluation_wall_time_seconds,
            created_at=_now(),
        )
        self._records[key] = record
        self._persist()
        return record

    def note_duplicate(
        self,
        record: EvaluationRegistryRecord,
        *,
        trial_index: int,
        campaign_id: str,
    ) -> EvaluationRegistryRecord:
        """Persist a lightweight count for a skipped duplicate proposal."""
        current = self._records.get(record.key)
        if current is None:
            raise KeyError(f"unknown registry record: {record.key}")
        updated = replace(
            current,
            duplicate_count=current.duplicate_count + 1,
            last_duplicate_trial_index=trial_index,
            last_duplicate_campaign_id=campaign_id,
        )
        self._records[record.key] = updated
        self._persist()
        return updated

    def _load(self) -> dict[str, EvaluationRegistryRecord]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(
                self.path.read_text(encoding="utf-8"),
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    ValueError(f"non-standard JSON constant {constant}")
                ),
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid evaluation registry JSON: {self.path}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("evaluation registry root must be an object")
        if payload.get("schema_version") != REGISTRY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported evaluation registry schema: "
                f"{payload.get('schema_version')!r}"
            )
        raw_records = payload.get("records")
        if not isinstance(raw_records, Mapping):
            raise ValueError("evaluation registry records must be an object")
        records: dict[str, EvaluationRegistryRecord] = {}
        for key, value in raw_records.items():
            if not isinstance(value, Mapping):
                raise ValueError(f"evaluation registry record is not an object: {key}")
            record = EvaluationRegistryRecord.from_json(str(key), value)
            expected_key = evaluation_key(record.contract_id, record.morphology)
            if expected_key != record.key:
                raise ValueError(f"evaluation registry key mismatch: {key}")
            records[str(key)] = record
        return records

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.tmp"
        )
        payload = {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "updated_at": _now(),
            "records": {
                key: record.to_json()
                for key, record in sorted(self._records.items())
            },
        }
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            try:
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
            except OSError:
                directory_fd = None
            if directory_fd is not None:
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if temporary.exists():
                temporary.unlink()


__all__ = [
    "EvaluationRegistry",
    "EvaluationRegistryRecord",
    "REGISTRY_SCHEMA_VERSION",
    "SUPPORTED_EVALUATION_STATUSES",
    "canonical_morphology",
    "evaluation_key",
]

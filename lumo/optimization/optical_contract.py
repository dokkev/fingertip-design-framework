"""Typed optical configuration contracts used by optimization boundaries.

This module owns the stable inputs and fingerprints for FULL_3D transport.
Artifact files and JSON schema handling remain in :mod:`optical_artifact`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Integral
from typing import Any, Iterable, Mapping

from lumo.finger.fingertip import Fingertip
from lumo.ray_tracing.optical_mechanics.settings import Transport3DSettings


OPTICAL_NUMERICAL_ACCEPTANCE_CONTRACT_VERSION = (
    "full3d-optical-numerical-acceptance-v1"
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint_mapping(value: Mapping[str, Any]) -> str:
    """Return the stable fingerprint used by optimization contracts."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def optical_physics_parameters(tip: Fingertip) -> dict[str, float]:
    """Return exactly the optical values used by FULL_3D transport."""
    if not isinstance(tip, Fingertip):
        raise TypeError("tip must be a Fingertip")
    return {
        "refractive_index_air": float(tip.optical.refractive_index_air),
        "refractive_index_silicone": float(tip.optical.refractive_index_silicone),
        "absorption_per_mm": float(tip.optical.absorption_per_mm),
        "relative_radiant_power": float(tip.led.relative_radiant_power),
        "emission_half_angle_deg": float(tip.led.emission_half_angle_deg),
    }


def transport_configuration(
    settings: Transport3DSettings,
    *,
    material: Mapping[str, Any],
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize only inputs that can change FULL_3D transport."""
    configuration: dict[str, Any] = {
        "schema": "full3d-transport-configuration-v1",
        "settings": asdict(settings),
        "material": dict(material),
    }
    if source is not None:
        configuration["source"] = dict(source)
    return configuration


def summarize_optical_failure_diagnostics(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate bounded candidate optical-failure evidence for campaign audit."""

    result: dict[str, Any] = {
        "optics_failure_candidate_count": 0,
        "numerical_acceptance_candidate_count": 0,
        "path_field_clipping_candidate_count": 0,
        "segment_budget_termination_candidate_count": 0,
        "objective_pathology_candidate_count": 0,
        "optical_failure_state_count": 0,
        "segment_budget_termination_count": 0,
        "segment_budget_termination_weight": 0.0,
        "clipped_sample_count": 0,
        "represented_weighted_path_length_mm": 0.0,
        "clipped_weighted_path_length_mm": 0.0,
        "cause_type_counts": {},
    }
    cause_type_counts: dict[str, int] = {}
    for record in records:
        if record.get("status") not in ("optics_failure", "optics_failed"):
            continue
        result["optics_failure_candidate_count"] += 1
        scenario = record.get("failure_scenario")
        diagnostics = record.get("failure_diagnostics", {})
        if not isinstance(diagnostics, Mapping):
            diagnostics = {}
        cause_type = diagnostics.get("cause_type")
        if isinstance(cause_type, str) and cause_type:
            cause_type_counts[cause_type] = cause_type_counts.get(cause_type, 0) + 1
        if scenario == "objective_pathology":
            result["objective_pathology_candidate_count"] += 1
        if scenario != "numerical_acceptance":
            continue
        result["numerical_acceptance_candidate_count"] += 1
        summary = diagnostics.get("optical_numerical_summary")
        if not isinstance(summary, Mapping):
            summary = diagnostics.get("optical_numerical_acceptance")
        if not isinstance(summary, Mapping):
            continue
        reasons = summary.get("failure_reasons", ())
        if not isinstance(reasons, (list, tuple, set, frozenset)):
            reasons = ()
        if "path_field_clipping" in reasons:
            result["path_field_clipping_candidate_count"] += 1
        if "segment_budget_termination" in reasons:
            result["segment_budget_termination_candidate_count"] += 1
        result["optical_failure_state_count"] += int(summary.get("failure_count", 1))
        for name in (
            "segment_budget_termination_count",
            "clipped_sample_count",
        ):
            result[name] += int(summary.get(name, 0))
        for name in (
            "segment_budget_termination_weight",
            "represented_weighted_path_length_mm",
            "clipped_weighted_path_length_mm",
        ):
            result[name] += float(summary.get(name, 0.0))
    result["cause_type_counts"] = dict(sorted(cause_type_counts.items()))
    return result


@dataclass(frozen=True)
class OpticalNumericalAcceptanceResult:
    """One state-level decision from the optical numerical contract."""

    accepted: bool
    failure_reasons: tuple[str, ...]
    termination_count: int
    termination_weight: float
    termination_fraction: float
    segment_budget_termination_count: int
    segment_budget_termination_weight: float
    segment_budget_termination_fraction: float
    processed_sample_count: int
    clipped_sample_count: int
    represented_weighted_path_length_mm: float
    clipped_weighted_path_length_mm: float
    objective_pathology: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be a bool")
        if any(
            not isinstance(reason, str) or not reason
            for reason in self.failure_reasons
        ):
            raise ValueError("failure_reasons must contain non-empty strings")
        for name in (
            "termination_count",
            "segment_budget_termination_count",
            "processed_sample_count",
            "clipped_sample_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            object.__setattr__(self, name, int(value))
        if self.clipped_sample_count > self.processed_sample_count:
            raise ValueError("clipped_sample_count cannot exceed processed_sample_count")
        for name in (
            "termination_weight",
            "termination_fraction",
            "segment_budget_termination_weight",
            "segment_budget_termination_fraction",
            "represented_weighted_path_length_mm",
            "clipped_weighted_path_length_mm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if not isinstance(self.objective_pathology, bool):
            raise TypeError("objective_pathology must be a bool")

    def to_dict(self) -> dict[str, Any]:
        """Return state-level acceptance evidence for reports and artifacts."""

        return {
            "accepted": self.accepted,
            "failure_reasons": list(self.failure_reasons),
            "termination_count": self.termination_count,
            "termination_weight": self.termination_weight,
            "termination_fraction": self.termination_fraction,
            "segment_budget_termination_count": (
                self.segment_budget_termination_count
            ),
            "segment_budget_termination_weight": (
                self.segment_budget_termination_weight
            ),
            "segment_budget_termination_fraction": (
                self.segment_budget_termination_fraction
            ),
            "processed_sample_count": self.processed_sample_count,
            "clipped_sample_count": self.clipped_sample_count,
            "represented_weighted_path_length_mm": (
                self.represented_weighted_path_length_mm
            ),
            "clipped_weighted_path_length_mm": (
                self.clipped_weighted_path_length_mm
            ),
            "objective_pathology": self.objective_pathology,
        }


@dataclass(frozen=True)
class OpticalNumericalAcceptanceContract:
    """Hard acceptance rules for production FULL_3D optical states."""

    version: str = OPTICAL_NUMERICAL_ACCEPTANCE_CONTRACT_VERSION
    maximum_segment_budget_termination_count: int = 0
    maximum_clipped_sample_count: int = 0
    maximum_clipped_weighted_path_length_mm: float = 0.0
    reject_objective_pathology: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("optical numerical acceptance version must be non-empty")
        for name in (
            "maximum_segment_budget_termination_count",
            "maximum_clipped_sample_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        value = float(self.maximum_clipped_weighted_path_length_mm)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "maximum_clipped_weighted_path_length_mm must be finite and non-negative"
            )
        object.__setattr__(self, "maximum_clipped_weighted_path_length_mm", value)
        if (
            self.maximum_segment_budget_termination_count != 0
            or self.maximum_clipped_sample_count != 0
            or value != 0.0
        ):
            raise ValueError(
                "Phase-C optical numerical hard rules require zero segment "
                "budget termination and zero path-field clipping"
            )
        if not isinstance(self.reject_objective_pathology, bool):
            raise TypeError("reject_objective_pathology must be a bool")
        if not self.reject_objective_pathology:
            raise ValueError(
                "production optical numerical acceptance always rejects "
                "objective pathology"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned acceptance configuration."""

        return {
            "version": self.version,
            "maximum_segment_budget_termination_count": (
                self.maximum_segment_budget_termination_count
            ),
            "maximum_clipped_sample_count": self.maximum_clipped_sample_count,
            "maximum_clipped_weighted_path_length_mm": (
                self.maximum_clipped_weighted_path_length_mm
            ),
            "reject_objective_pathology": self.reject_objective_pathology,
        }

    def assess(
        self,
        result: Any,
        *,
        objective_pathology: bool = False,
    ) -> OpticalNumericalAcceptanceResult:
        """Assess one transport result without changing its scientific values."""

        launched_weight = float(getattr(result, "launched_weight", 0.0))
        if not math.isfinite(launched_weight) or launched_weight < 0.0:
            raise ValueError("optical result launched_weight must be finite and non-negative")
        raw_segment_count = getattr(result, "segment_budget_termination_count", 0)
        if not isinstance(raw_segment_count, Integral) or isinstance(
            raw_segment_count, bool
        ):
            raise ValueError("optical segment-budget termination count is invalid")
        segment_count = int(raw_segment_count)
        segment_weight = float(
            getattr(result, "segment_budget_termination_weight", 0.0)
        )
        if segment_count < 0 or not math.isfinite(segment_weight) or segment_weight < 0.0:
            raise ValueError("optical segment-budget diagnostics are invalid")
        raw_termination_count = getattr(result, "termination_count", 0)
        if not isinstance(raw_termination_count, Integral) or isinstance(
            raw_termination_count, bool
        ):
            raise ValueError("optical termination count is invalid")
        termination_count = int(raw_termination_count)
        termination_weight = float(getattr(result, "terminated_weight", 0.0))
        if (
            termination_count < 0
            or not math.isfinite(termination_weight)
            or termination_weight < 0.0
        ):
            raise ValueError("optical termination diagnostics are invalid")
        raw_processed_count = getattr(result, "processed_sample_count", 0)
        raw_clipped_count = getattr(result, "clipped_sample_count", 0)
        if any(
            not isinstance(value, Integral) or isinstance(value, bool)
            for value in (raw_processed_count, raw_clipped_count)
        ):
            raise ValueError("optical path-field sample counts are invalid")
        processed_count = int(raw_processed_count)
        clipped_count = int(raw_clipped_count)
        represented_length = float(
            getattr(result, "represented_weighted_path_length_mm", 0.0)
        )
        clipped_length = float(
            getattr(result, "clipped_weighted_path_length_mm", 0.0)
        )
        if (
            processed_count < 0
            or clipped_count < 0
            or clipped_count > processed_count
            or not math.isfinite(represented_length)
            or represented_length < 0.0
            or not math.isfinite(clipped_length)
            or clipped_length < 0.0
        ):
            raise ValueError("optical path-field diagnostics are invalid")
        reasons: list[str] = []
        if (
            segment_count > self.maximum_segment_budget_termination_count
            or segment_weight != 0.0
        ):
            reasons.append("segment_budget_termination")
        if clipped_count > self.maximum_clipped_sample_count:
            reasons.append("path_field_clipping")
        if clipped_length > self.maximum_clipped_weighted_path_length_mm:
            if "path_field_clipping" not in reasons:
                reasons.append("path_field_clipping")
        if objective_pathology and self.reject_objective_pathology:
            reasons.append("objective_pathology")
        return OpticalNumericalAcceptanceResult(
            accepted=not reasons,
            failure_reasons=tuple(reasons),
            termination_count=termination_count,
            termination_weight=termination_weight,
            termination_fraction=(
                termination_weight / max(launched_weight, 1.0e-30)
            ),
            segment_budget_termination_count=segment_count,
            segment_budget_termination_weight=segment_weight,
            segment_budget_termination_fraction=(
                segment_weight / max(launched_weight, 1.0e-30)
            ),
            processed_sample_count=processed_count,
            clipped_sample_count=clipped_count,
            represented_weighted_path_length_mm=represented_length,
            clipped_weighted_path_length_mm=clipped_length,
            objective_pathology=bool(objective_pathology),
        )

    def summarize(self, results: Iterable[Any]) -> dict[str, Any]:
        """Aggregate state diagnostics for a candidate-level report."""

        items = tuple(results)
        assessments = tuple(self.assess(item) for item in items)
        launched_weight = sum(
            float(getattr(item, "launched_weight", 0.0)) for item in items
        )
        terminated_weight = sum(
            float(getattr(item, "terminated_weight", 0.0)) for item in items
        )
        return {
            "contract": self.to_dict(),
            "state_count": len(items),
            "failure_count": sum(not item.accepted for item in assessments),
            "segment_budget_termination_count": sum(
                item.segment_budget_termination_count for item in assessments
            ),
            "segment_budget_termination_weight": sum(
                item.segment_budget_termination_weight for item in assessments
            ),
            "termination_count": sum(item.termination_count for item in assessments),
            "segment_budget_termination_fraction": sum(
                item.segment_budget_termination_weight for item in assessments
            )
            / max(launched_weight, 1.0e-30),
            "terminated_weight": terminated_weight,
            "termination_weight": terminated_weight,
            "terminated_weight_fraction": terminated_weight
            / max(launched_weight, 1.0e-30),
            "termination_fraction": terminated_weight
            / max(launched_weight, 1.0e-30),
            "clipped_sample_count": sum(
                item.clipped_sample_count for item in assessments
            ),
            "represented_weighted_path_length_mm": sum(
                item.represented_weighted_path_length_mm for item in assessments
            ),
            "clipped_weighted_path_length_mm": sum(
                item.clipped_weighted_path_length_mm for item in assessments
            ),
            "failure_reasons": sorted(
                {
                    reason
                    for item in assessments
                    for reason in item.failure_reasons
                }
            ),
        }


DEFAULT_OPTICAL_NUMERICAL_ACCEPTANCE = OpticalNumericalAcceptanceContract()


__all__ = [
    "DEFAULT_OPTICAL_NUMERICAL_ACCEPTANCE",
    "OPTICAL_NUMERICAL_ACCEPTANCE_CONTRACT_VERSION",
    "OpticalNumericalAcceptanceContract",
    "OpticalNumericalAcceptanceResult",
    "fingerprint_mapping",
    "optical_physics_parameters",
    "summarize_optical_failure_diagnostics",
    "transport_configuration",
]

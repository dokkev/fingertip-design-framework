"""Deterministic feasible morphology catalog for scientific convergence work."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Mapping

from lumo.finger import (
    FingertipParameters,
    SiliconeThicknessMeasures,
    fingertip_parameters_fingerprint,
    silicone_thickness_measures,
)
from lumo.optimization.design_space import DesignSpace, LATENT_PARAMETER_NAMES


@dataclass(frozen=True)
class RepresentativeMorphology:
    case_id: str
    latent_values: Mapping[str, float]
    parameters: FingertipParameters
    physical_values: Mapping[str, float]
    morphology_fingerprint: str
    thickness_measures: SiliconeThicknessMeasures

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        object.__setattr__(
            self,
            "latent_values",
            MappingProxyType(dict(self.latent_values)),
        )
        object.__setattr__(
            self,
            "physical_values",
            MappingProxyType(dict(self.physical_values)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "latent_values": dict(self.latent_values),
            "physical_values": dict(self.physical_values),
            "morphology_fingerprint": self.morphology_fingerprint,
            "thickness_measures": asdict(self.thickness_measures),
        }


def representative_morphologies(
    design_space: DesignSpace,
) -> tuple[RepresentativeMorphology, ...]:
    """Build five fixed feasible cases while preserving latent and physical audit."""

    if not isinstance(design_space, DesignSpace):
        raise TypeError("design_space must be a DesignSpace")
    center = {name: 0.5 for name in LATENT_PARAMETER_NAMES}
    definitions: tuple[tuple[str, dict[str, float]], ...] = (
        ("nominal", design_space.encode(design_space.nominal_parameters)),
        ("latent_center", center),
        ("wide_cutout_edge", center | {"latent_cutout_width": 0.98}),
        ("deep_cutout_edge", center | {"latent_cutout_depth": 1.0}),
        (
            "minimum_wall_edge",
            center
            | {
                "latent_cutout_width": 0.98,
                "latent_pad_depth": 0.0,
                "latent_cutout_depth": 1.0,
            },
        ),
    )
    result: list[RepresentativeMorphology] = []
    seen: set[str] = set()
    for case_id, latent in definitions:
        parameters = design_space.decode(latent)
        design_space.validate_physical_parameters(parameters)
        fingerprint = fingertip_parameters_fingerprint(parameters)
        if fingerprint in seen:
            raise ValueError(f"representative morphology is duplicated: {case_id}")
        seen.add(fingerprint)
        result.append(
            RepresentativeMorphology(
                case_id=case_id,
                latent_values=latent,
                parameters=parameters,
                physical_values=design_space.physical_values(parameters),
                morphology_fingerprint=fingerprint,
                thickness_measures=silicone_thickness_measures(parameters),
            )
        )
    by_id = {item.case_id: item for item in result}
    if by_id["wide_cutout_edge"].latent_values["latent_cutout_width"] < 0.95:
        raise ValueError("wide_cutout_edge must remain near the feasible width edge")
    if by_id["deep_cutout_edge"].latent_values["latent_cutout_depth"] != 1.0:
        raise ValueError("deep_cutout_edge must use the cutout-depth boundary")
    minimum_wall = by_id["minimum_wall_edge"].thickness_measures
    if not 5.0 <= minimum_wall.minimum_silicone_thickness_mm <= 5.5:
        raise ValueError(
            "minimum_wall_edge must remain near the 5 mm feasible boundary"
        )
    return tuple(result)


__all__ = ["RepresentativeMorphology", "representative_morphologies"]

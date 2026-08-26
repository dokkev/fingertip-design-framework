"""Fingertip optimization domain types and objective reductions."""

from .design_space import DesignSpace
from .objective import (
    ContactObjective,
    ObservationObjective,
    compute_contact_objective,
    compute_objectives_from_raw,
    compute_observation_objective,
)

__all__ = [
    "DesignSpace",
    "ContactObjective",
    "ObservationObjective",
    "compute_contact_objective",
    "compute_objectives_from_raw",
    "compute_observation_objective",
]

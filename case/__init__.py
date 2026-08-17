"""One complete 2D explicit-contact fingertip simulation case."""

from case.core import (
    CASE_SCHEMA,
    CaseConstructionError,
    FEA2D,
    FingertipCase,
    RayTracing2D,
    case_id_for,
    contact_state_contract,
    run_case,
)
from case.artifact import load_case, save_case
from case.state import ContactState

__all__ = [
    "CASE_SCHEMA",
    "CaseConstructionError",
    "ContactState",
    "FEA2D",
    "FingertipCase",
    "RayTracing2D",
    "case_id_for",
    "contact_state_contract",
    "load_case",
    "run_case",
    "save_case",
]

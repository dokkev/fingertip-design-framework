"""Randomly inspect the LUMO2 fingertip design space."""

from __future__ import annotations

import numpy as np

from lumo.fingertip.fingertip_param import FingertipParameters
from lumo.optimization.design_param_bound import (
    DesignParameterBounds,
    ParameterBound,
)
from lumo.optimization.design_space import (
    DesignSpace,
    LinearConstraint,
)


def sample_candidate(
    space: DesignSpace,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Uniformly sample one candidate from the raw parameter bounds."""

    candidate = {}

    for name, bound in space.parameter_bounds.geometry.items():
        candidate[f"geometry.{name}"] = rng.uniform(
            bound.lower,
            bound.upper,
        )

    for name, bound in space.parameter_bounds.led.items():
        candidate[f"led.{name}"] = rng.uniform(
            bound.lower,
            bound.upper,
        )

    return candidate


def main() -> None:
    rng = np.random.default_rng(0)

    bounds = DesignParameterBounds(
        parameters=FingertipParameters(),
        geometry={
            "flat_pad_width_mm": ParameterBound(25.0, 35.0),
            "flat_pad_height_mm": ParameterBound(3.0, 8.0),
            "semiellipse_height_mm": ParameterBound(6.0, 20.0),
            "stem_width_mm": ParameterBound(7.0, 10.0),
            "void_width_mm": ParameterBound(0.0, 3.0),
            "void_height_mm": ParameterBound(0.0, 3.0),
        },
        led={
            "width_mm": ParameterBound(3.0, 5.0),
        },
    )

    space = DesignSpace(
        parameter_bounds=bounds,
        linear_constraints=(
            LinearConstraint(
                coefficients={
                    "geometry.flat_pad_height_mm": 1.0,
                    "geometry.semiellipse_height_mm": 1.0,
                },
                upper=30.0,
            ),
        ),
        minimum_silicone_thickness_mm=5.0,
    )

    sample_count = 10_000
    feasible = []

    for _ in range(sample_count):
        candidate = sample_candidate(space, rng)

        if space.is_feasible(candidate):
            feasible.append(candidate)

    feasible_count = len(feasible)
    acceptance_rate = feasible_count / sample_count

    print(f"Samples:    {sample_count}")
    print(f"Feasible:   {feasible_count}")
    print(f"Acceptance: {acceptance_rate:.1%}")

    print("\nExample feasible candidates:")

    for candidate in feasible[:5]:
        print()
        for name, value in candidate.items():
            print(f"  {name}: {value:.3f}")


if __name__ == "__main__":
    main()

"""Living reference for the ME-engineer-facing production BO entry point.

This is a target implementation reference, not the current production runner.
It intentionally names APIs that still need to be implemented and has no
``__main__`` block. Keep the user story visible and update this file when an
approved design decision changes.

The intended user knows:

- the parametric fingertip geometry they want to explore;
- the pad's mechanical and bulk optical material properties;
- the LED emission they want to model;
- which geometry parameters to optimize and their bounds;
- the contact protocol, objective, BO strategy, and proposal budget.

The user is not expected to tune mesh, Newton, or OptiX implementation details.
Those values live in the repository-root ``config/lumo_execution.yaml``, are
explained line by line, and are loaded once into a typed execution
configuration. The resolved values still participate in the evaluation
contract and persisted campaign config.

All installable framework APIs live below the ``lumo.*`` namespace. This
workflow assumes the project is installed and never modifies ``sys.path``.

The desired public LED input contains emission properties only. The current
implementation still uses LED package width and height when positioning its
source, so that dependency must be removed deliberately before this reference
can become the production runner.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


DEFAULT_EXECUTION_CONFIG = (
    Path(__file__).resolve().parents[2] / "config" / "lumo_execution.yaml"
)
DEFAULT_OUTPUT_ROOT = Path("output/optimization")
DEFAULT_RUN_NAME: str | None = None  # None asks the campaign to generate a name.


def main(argv: Sequence[str] | None = None) -> int:
    """Read top to bottom like an example of one complete BO campaign."""

    # Target API imports stay local so this reference remains import-safe while
    # the implementation is migrated.
    from lumo.config import load_lumo_execution_config
    from lumo.finger import (
        FingertipParameters,
        LEDEmission,
        OpticalParameters,
        ViscoelasticParameters,
    )
    from lumo.optimization import (
        CampaignOutput,
        DesignSpace,
        DesignVariable,
        Lumo3DTrajectoryEvaluator,
        MaximumCutoutWidth,
        OptimizableParameterName,
        TrajectoryEvaluationProtocol,
        TrajectorySeparationObjective,
        render_run_result,
        run_optimization_campaign,
        run_production_preflight,
    )
    from lumo.optimization.ax import (
        AxBayesianOptimization,
        AxBoSettings,
        AxStrategy,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="validate configuration and runtimes without starting BO",
    )
    mode.add_argument(
        "--trials",
        type=int,
        help="Ax-generated proposal budget; nominal is evaluated separately",
    )
    parser.add_argument(
        "--execution-config",
        type=Path,
        default=DEFAULT_EXECUTION_CONFIG,
        help="expert-owned mesh, mechanics, and transport YAML",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="parent directory containing one directory per campaign",
    )
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help="campaign directory name; generated automatically when omitted",
    )
    args = parser.parse_args(argv)

    if args.trials is not None and args.trials < 1:
        parser.error("--trials must be a positive integer")

    # 1. Define one complete nominal fingertip. Any supported geometry field
    # omitted from the design space below remains fixed at this nominal value.
    nominal_fingertip = FingertipParameters(
        flat_pad_width=30.0,
        flat_pad_height=5.0,
        semielliptical_pad_height=9.0,
        link_thickness=3.5,
        bond_extension_width=4.0,
        bond_extension_height=2.0,
        stem_width=7.6,
        stem_height=6.0,
        void_width=1.0,
        void_height=0.25,
        viscoelastic=ViscoelasticParameters(
            density_kg_m3=1.0e3,
            k_mu_pa=1.0e5,
            k_lambda_pa=1.0e5,
            k_damp=10.0,
        ),
        optical=OpticalParameters(
            refractive_index_air=1.0,
            refractive_index_silicone=1.41,
            absorption_per_mm=0.02,
        ),
    )

    # 2. Describe LED emission only. Package dimensions do not belong in the
    # target physical input contract.
    led = LEDEmission(
        relative_radiant_power=1.0,
        emission_half_angle_deg=80.0,
    )

    # 3. Select the geometry fields to optimize. Membership is the opt-in;
    # there is no separate optimize=True flag. Removing STEM_WIDTH, for
    # example, keeps stem_width=7.6 mm throughout the campaign.
    parameter = OptimizableParameterName
    design_space = DesignSpace(
        nominal=nominal_fingertip,
        variables=(
            DesignVariable(parameter.FLAT_PAD_HEIGHT, lower=0.5, upper=29.5),
            DesignVariable(
                parameter.SEMIELLIPTICAL_PAD_HEIGHT,
                lower=0.5,
                upper=29.5,
            ),
            DesignVariable(parameter.STEM_WIDTH, lower=1.0, upper=20.0),
            DesignVariable(parameter.STEM_HEIGHT, lower=1.0, upper=25.0),
            DesignVariable(parameter.VOID_WIDTH, lower=0.0, upper=10.0),
            DesignVariable(parameter.VOID_HEIGHT, lower=0.0, upper=25.0),
        ),
        search_restrictions=(
            MaximumCutoutWidth(20.0),
        ),
    )

    # 4. Define the physical experiment the morphology must distinguish.
    protocol = TrajectoryEvaluationProtocol(
        contact_locations_u=(0.25, 0.50, 0.75),
        indenter_radii_mm=(4.0, 5.0),
        checkpoint_depths_mm=(0.5, 1.0, 1.5),
        initial_gap_mm=0.25,
    )

    # 5. Select the scientific objective and load expert numerical settings at
    # one typed initialization boundary. Raw YAML does not enter the evaluator.
    objective = TrajectorySeparationObjective(
        radius_penalty_weight=1.0,
    )
    execution = load_lumo_execution_config(args.execution_config)
    evaluator = Lumo3DTrajectoryEvaluator(
        protocol=protocol,
        objective=objective,
        execution=execution,
        led=led,
    )

    if args.preflight:
        result = run_production_preflight(
            design_space=design_space,
            evaluator=evaluator,
        )
    else:
        # 6. Choose the concrete BO backend and generation strategy. The
        # registry remains an internal campaign persistence detail.
        optimizer = AxBayesianOptimization(
            design_space=design_space,
            settings=AxBoSettings(
                strategy=AxStrategy.SOBOL_THEN_MBM,
                proposal_budget=args.trials,
                initialization_trials=1,
                seed=20260820,
                max_consecutive_known_proposals=20,
            ),
        )
        result = run_optimization_campaign(
            optimizer=optimizer,
            evaluator=evaluator,
            output=CampaignOutput(
                root=args.output_root,
                run_name=args.run_name,
            ),
        )

    print(render_run_result(result))
    return 0 if result.succeeded else 2

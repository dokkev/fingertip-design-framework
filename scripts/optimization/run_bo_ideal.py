"""Living reference for the intended production BO entry point.

This module is an executable design specification, not the current production
runner.  It deliberately names target APIs that may not exist yet and is not
wired into ``docs/COMMANDS.md``, tests, or a ``__main__`` block.  Keep it
import-safe while the implementation is migrated, and update it whenever an
approved ownership or lifecycle decision changes.

The intended user-visible flow is:

1. define one complete, explicit nominal fingertip;
2. select the morphology fields that are active in the design space;
3. construct a concrete BO instance from the design space and BO settings;
4. run one campaign with an evaluator and an output location.

Campaign-fixed means "not changed by this design space", not "fundamentally
immutable".  A supported morphology field is active only when it has a
``DesignVariable`` entry.  All omitted morphology fields, representation
settings, viscoelastic parameters, and optical parameters retain their explicit
nominal values for every candidate.

The campaign runtime owns preflight, result persistence, exact-evaluation
history, duplicate detection, and the internal ``EvaluationRegistry``.  None of
those mechanisms appear in the user configuration or optimizer call below.

Output directory contract:

- ``output_root / run_name`` when ``run_name`` is supplied;
- ``output_root / {mode}_{utc_timestamp}_{contract_short_id}`` otherwise;
- an explicit non-empty directory is never overwritten;
- a generated-name collision receives a numeric suffix;
- the run name is operational provenance, not scientific identity.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence


# ---------------------------- OPERATIONAL DEFAULTS ----------------------------

DEFAULT_OUTPUT_ROOT = Path("output/optimization")
DEFAULT_RUN_NAME: str | None = None  # None selects a generated campaign name.


@dataclass(frozen=True)
class IdealRunRequest:
    """CLI-owned operational request; no scientific settings live here."""

    preflight_only: bool
    proposal_budget: int | None
    output_root: Path
    run_name: str | None
    smoke: bool

    def __post_init__(self) -> None:
        if self.preflight_only == (self.proposal_budget is not None):
            raise ValueError("select exactly one of preflight_only or proposal_budget")
        if self.proposal_budget is not None:
            if (
                not isinstance(self.proposal_budget, int)
                or isinstance(self.proposal_budget, bool)
                or self.proposal_budget < 1
            ):
                raise ValueError("proposal_budget must be a positive integer")
        if self.run_name is not None:
            name = self.run_name.strip()
            if not name or name in {".", ".."} or Path(name).name != name:
                raise ValueError("run_name must be one non-empty directory name")
            object.__setattr__(self, "run_name", name)


# ------------------------- COMPLETE NOMINAL FINGERTIP -------------------------


def make_nominal_fingertip() -> FingertipParameters:
    """Return the complete baseline and source for campaign-fixed values."""

    from finger import (
        FingertipParameters,
        OpticalParameters,
        ViscoelasticParameters,
    )

    return FingertipParameters(
        # Geometry and representation: explicit even when equal to defaults.
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
        arc_resolution=128,
        geometry_length_tolerance_mm=1.0e-9,
        geometry_area_tolerance_mm2=1.0e-9,
        # Fixed for this campaign because they are not design variables.
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


# ------------------------------- DESIGN SPACE --------------------------------


def make_design_space(nominal: FingertipParameters) -> DesignSpace:
    """Select active fields; an omitted field stays at its nominal value.

    The target ``DesignVariable`` API has no redundant ``optimize`` flag.
    Membership in ``variables`` is the opt-in.  For example, deleting the
    ``STEM_WIDTH`` entry below fixes ``stem_width`` at the explicit nominal
    value of 7.6 mm for the entire campaign.

    Structured constraints allow the Ax adapter to project an inactive term
    into the right-hand side using its nominal value rather than exposing an Ax
    expression string here.
    """

    from optimization.design_space import (
        ConstraintSource,
        DesignSpace,
        DesignVariable,
        LinearConstraint,
        LinearTerm,
        OptimizableParameterName,
    )

    name = OptimizableParameterName
    return DesignSpace(
        nominal=nominal,
        variables=(
            DesignVariable(name.FLAT_PAD_HEIGHT, lower=0.5, upper=29.5),
            DesignVariable(
                name.SEMIELLIPTICAL_PAD_HEIGHT,
                lower=0.5,
                upper=29.5,
            ),
            DesignVariable(name.STEM_WIDTH, lower=1.0, upper=20.0),
            DesignVariable(name.STEM_HEIGHT, lower=1.0, upper=25.0),
            DesignVariable(name.VOID_WIDTH, lower=0.0, upper=10.0),
            DesignVariable(name.VOID_HEIGHT, lower=0.0, upper=25.0),
        ),
        constraints=(
            LinearConstraint(
                name="maximum_total_pad_depth",
                terms=(
                    LinearTerm(name.FLAT_PAD_HEIGHT, coefficient=1.0),
                    LinearTerm(
                        name.SEMIELLIPTICAL_PAD_HEIGHT,
                        coefficient=1.0,
                    ),
                ),
                upper_bound=30.0,
                source=ConstraintSource.DOMAIN,
            ),
            LinearConstraint(
                name="production_cutout_width_restriction",
                terms=(
                    LinearTerm(name.STEM_WIDTH, coefficient=1.0),
                    LinearTerm(name.VOID_WIDTH, coefficient=2.0),
                ),
                upper_bound=20.0,
                source=ConstraintSource.CAMPAIGN,
            ),
        ),
    )


# ------------------------- EXPLICIT EVALUATION INPUTS -------------------------


def make_production_evaluation_config() -> LumoTrajectoryEvaluationConfig:
    """Return every fixed scientific and numerical production input."""

    from contact import FirstContactSettings
    from finger import LED
    from lumo.mechanics_contract import MechanicsContract
    from mesh.volume.contracts import VolumeMeshSettings
    from optimization.evaluator import LumoTrajectoryEvaluationConfig
    from optimization.objectives import (
        TrajectoryObjectiveConfig,
        TrajectorySeparationObjective,
    )
    from optimization.protocol import TrajectoryEvaluationProtocol
    from ray_tracing.optical_mechanics.settings import Transport3DSettings

    return LumoTrajectoryEvaluationConfig(
        device="cuda:0",
        led=LED(
            width_mm=4.0,
            height_mm=2.0,
            relative_radiant_power=1.0,
            emission_half_angle_deg=80.0,
        ),
        protocol=TrajectoryEvaluationProtocol(
            contact_locations_u=(0.25, 0.50, 0.75),
            indenter_radii_mm=(4.0, 5.0),
            checkpoint_depths_mm=(0.5, 1.0, 1.5),
            initial_gap_mm=0.25,
        ),
        mesh=VolumeMeshSettings(
            tier="search",
            target_size_mm=1.5,
            minimum_quality=0.02,
        ),
        mechanics=MechanicsContract(
            sphere_subdivisions=3,
            max_load_increment_mm=0.05,
            vbd_iterations=10,
            dt_s=1.0e-3,
            soft_contact_margin_mm=0.02,
            soft_contact_ke=1.0e3,
            soft_contact_kd=10.0,
            max_support_displacement_mm=1.0e-9,
            max_final_pose_error_mm=1.0e-6,
            max_carrier_penetration_voxel_fraction=0.5,
            first_contact=FirstContactSettings(
                coarse_step_mm=0.25,
                tolerance_mm=1.0e-3,
                spawn_clearance_mm=0.05,
                max_travel_mm=20.0,
            ),
        ),
        transport=Transport3DSettings(
            ray_count=256,
            max_interactions=6,
            minimum_ray_weight=1.0e-4,
            maximum_segment_count=4096,
            maximum_periodic_wraps=8,
            extrusion_depth_mm=11.0,
            surface_u_bins=32,
            surface_z_bins=16,
            internal_grid_width=32,
            internal_grid_height=32,
            internal_z_bins=8,
            internal_max_samples_per_segment=32,
            x_bounds_mm=(-16.0, 16.0),
            y_bounds_mm=(-31.0, 4.5),
            source_epsilon_mm=1.0e-5,
            intersection_epsilon_mm=1.0e-6,
            energy_balance_tolerance=1.0e-5,
            retain_internal_path_field=True,
            terminate_on_periodic_wrap_limit=True,
            terminate_on_no_event=True,
        ),
        objective=TrajectorySeparationObjective(
            TrajectoryObjectiveConfig(
                radius_penalty_weight=1.0,
            )
        ),
    )


def make_smoke_evaluation_config() -> LumoTrajectoryEvaluationConfig:
    """Return the same evaluator contract with only the explicit smoke protocol."""

    from dataclasses import replace
    from optimization.protocol import TrajectoryEvaluationProtocol

    return replace(
        make_production_evaluation_config(),
        protocol=TrajectoryEvaluationProtocol(
            contact_locations_u=(0.25, 0.75),
            indenter_radii_mm=(5.0,),
            checkpoint_depths_mm=(1.0,),
            initial_gap_mm=0.25,
        ),
    )


# --------------------------- BO INSTANCE AND RUN ------------------------------


def make_optimizer(
    design_space: DesignSpace,
    *,
    proposal_budget: int,
) -> AxBayesianOptimization:
    """Select the concrete backend and its explicit generation strategy."""

    from optimization.adapters.ax import (
        AxBayesianOptimization,
        AxBoSettings,
        AxStrategy,
    )

    return AxBayesianOptimization(
        design_space=design_space,
        settings=AxBoSettings(
            strategy=AxStrategy.SOBOL_THEN_MBM,
            proposal_budget=proposal_budget,
            initialization_trials=1,
            seed=20260820,
            max_consecutive_known_proposals=20,
        ),
    )


def run(request: IdealRunRequest) -> PreflightResult | AxRunResult:
    """Show the complete ideal lifecycle without exposing registry mechanics."""

    from optimization.campaign import (
        CampaignOutput,
        run_optimization_campaign,
        run_production_preflight,
    )
    from optimization.evaluator import Lumo3DTrajectoryEvaluator

    nominal = make_nominal_fingertip()
    design_space = make_design_space(nominal)
    evaluation_config = (
        make_smoke_evaluation_config()
        if request.smoke
        else make_production_evaluation_config()
    )

    if request.preflight_only:
        # Preflight-only is read-only and cannot be combined with a campaign.
        return run_production_preflight(
            design_space=design_space,
            evaluation_config=evaluation_config,
        )

    assert request.proposal_budget is not None
    optimizer = make_optimizer(
        design_space,
        proposal_budget=request.proposal_budget,
    )
    evaluator = Lumo3DTrajectoryEvaluator(config=evaluation_config)

    return run_optimization_campaign(
        optimizer=optimizer,
        evaluator=evaluator,
        output=CampaignOutput(
            root=request.output_root,
            run_name=request.run_name,
        ),
    )


# ------------------------------------ CLI -------------------------------------


def parse_request(argv: Sequence[str] | None = None) -> IdealRunRequest:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="check the selected configuration and runtime without a campaign",
    )
    mode.add_argument(
        "--trials",
        type=int,
        help="Ax-generated proposal budget; nominal is evaluated separately",
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
        help="campaign directory name; generated when omitted",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use the explicit reduced protocol with the production evaluator",
    )
    args = parser.parse_args(argv)
    try:
        return IdealRunRequest(
            preflight_only=bool(args.preflight),
            proposal_budget=args.trials,
            output_root=args.output_root,
            run_name=args.run_name,
            smoke=bool(args.smoke),
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))


def main(argv: Sequence[str] | None = None) -> int:
    """Target CLI shape; intentionally not connected to ``__main__`` yet."""

    from optimization.campaign import render_run_result

    result = run(parse_request(argv))
    print(render_run_result(result))
    return 0 if result.succeeded else 2


# Target-only names are postponed annotations.  They document the desired
# boundaries without importing the optional Ax runtime or unfinished APIs when
# this reference module is imported.
if TYPE_CHECKING:
    from finger import FingertipParameters
    from optimization.adapters.ax import AxBayesianOptimization, AxRunResult
    from optimization.campaign import PreflightResult
    from optimization.design_space import DesignSpace
    from optimization.evaluator import LumoTrajectoryEvaluationConfig

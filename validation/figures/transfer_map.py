"""Artifact tables and figures for Phase 4K transfer-map validation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from validation.fingertip.transfer_map.metrics import (
    FINE_LOCATIONS,
    MEDIUM_LOCATIONS,
    SIDE_NAMES,
    signature,
    tangent_signature,
)

def write_long_csv(rows: Sequence[Mapping[str, Any]], output_root: Path) -> None:
    path = output_root / "codtm_long.csv"
    columns = (
        "case",
        "mesh",
        "step",
        "delta_n",
        "xi_cmd",
        "xi_centroid",
        "F_n",
        "contact_length",
        "side_name",
        "eta",
        "X0_x",
        "X0_y",
        "u_x",
        "u_y",
        "u_normal",
        "u_tangent",
        "deformed_x",
        "deformed_y",
        "min_detF",
        "strain_metric",
        "valid",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)



def write_case_summary(
    loaded: Sequence[
        tuple[Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]]
    ],
    output_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec, result, records in loaded:
        final = records[-1] if records else None
        verified_steps = (
            sum(
                record["contact"]["verification"] == "VERIFIED"
                for record in records
            )
            if records
            else 0
        )
        rows.append(
            {
                "case": spec["case_name"],
                "stage": spec["stage"],
                "mesh": spec["mesh"],
                "xi_cmd": spec["xi_cmd"],
                "solve_status": result.get("solve_status"),
                "case_status": result.get("status"),
                "converged_steps": len(records),
                "final_reaction_n": (
                    final["canonical_normal_reaction_n"]
                    if final is not None
                    else None
                ),
                "final_xi_centroid": (
                    final["contact"]["xi_centroid"]
                    if final is not None
                    else None
                ),
                "final_contact_length_mm": (
                    final["contact"]["contact_length_mm"]
                    if final is not None
                    else None
                ),
                "final_force_closure_error": (
                    final["contact"]["force_closure_relative_error"]
                    if final is not None
                    else None
                ),
                "verified_contact_steps": verified_steps,
                "minimum_det_f": (
                    min(record["minimum_det_f"] for record in records)
                    if records
                    else None
                ),
                "maximum_strain_metric": (
                    max(
                        record["canonical_strain_metric"]["value"]
                        for record in records
                    )
                    if records
                    else None
                ),
                "maximum_nonlinear_iterations": (
                    max(record["nonlinear_iterations"] for record in records)
                    if records
                    else None
                ),
                "solve_wall_clock_seconds": result.get(
                    "solve_wall_clock_seconds"
                ),
                "failure_reason": result.get("failure_reason"),
            }
        )
    path = output_root / "case_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows



def write_plots(
    loaded: Sequence[
        tuple[Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]]
    ],
    metrics: Mapping[str, Any],
    output_root: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_directory = output_root / "plots"
    plot_directory.mkdir(parents=True, exist_ok=True)
    by_key = {
        (str(spec["mesh"]), float(spec["xi_cmd"])): records
        for spec, _, records in loaded
    }
    created: list[str] = []

    def save(name: str) -> None:
        path = plot_directory / name
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        created.append(str(path.relative_to(output_root)))

    for kind in ("force", "length", "detf"):
        plt.figure(figsize=(7.0, 4.5))
        for xi in MEDIUM_LOCATIONS:
            records = by_key[("medium", xi)]
            if not records:
                continue
            delta = [record["delta_n_mm"] for record in records]
            if kind == "force":
                value = [
                    record["canonical_normal_reaction_n"]
                    for record in records
                ]
                ylabel = "Normal reaction (N)"
            elif kind == "length":
                value = [
                    record["contact"]["contact_length_mm"]
                    for record in records
                ]
                ylabel = "Verified active contact length (mm)"
            else:
                value = [record["minimum_det_f"] for record in records]
                ylabel = "Minimum det(F)"
            plt.plot(delta, value, label=fr"$\xi={xi:.2f}$")
        plt.xlabel("Indentation (mm)")
        plt.ylabel(ylabel)
        plt.legend(ncol=2)
        save(
            {
                "force": "force_indentation_by_location.png",
                "length": "contact_length_by_indentation.png",
                "detf": "minimum_detf_by_indentation.png",
            }[kind]
        )

    plt.figure(figsize=(6.0, 4.5))
    commanded = []
    achieved = []
    for xi in MEDIUM_LOCATIONS:
        records = by_key[("medium", xi)]
        if records and records[-1]["contact"]["xi_centroid"] is not None:
            commanded.append(xi)
            achieved.append(records[-1]["contact"]["xi_centroid"])
    plt.plot(commanded, achieved, "o-", label="achieved")
    plt.plot([0.2, 0.8], [0.2, 0.8], "--", label="commanded=achieved")
    plt.xlabel("Commanded xi")
    plt.ylabel("Verified contact centroid xi")
    plt.legend()
    save("achieved_centroid_vs_commanded.png")

    plt.figure(figsize=(8.0, 5.0))
    eta = np.linspace(0.0, 1.0, 41)
    for xi in MEDIUM_LOCATIONS:
        records = by_key[("medium", xi)]
        if not records:
            continue
        for side, linestyle in (("left", "-"), ("right", "--")):
            plt.plot(
                eta,
                [
                    row["u_normal_mm"]
                    for row in records[-1]["observation_sidewalls"][side]
                ],
                linestyle,
                label=f"xi={xi:.2f} {side}",
            )
    plt.xlabel("Observation eta")
    plt.ylabel("Outward-normal displacement (mm)")
    plt.legend(ncol=2, fontsize=8)
    save("sidewall_profiles_1p5mm.png")

    medium_complete = all(
        len(by_key[("medium", xi)]) == 48 for xi in MEDIUM_LOCATIONS
    )
    if medium_complete:
        normal = np.asarray(
            [
                [
                    [
                        row["u_normal_mm"]
                        for row in by_key[("medium", xi)][-1][
                            "observation_sidewalls"
                        ][side]
                    ]
                    for xi in MEDIUM_LOCATIONS
                ]
                for side in SIDE_NAMES
            ]
        )
        tangent = np.asarray(
            [
                tangent_signature(by_key[("medium", xi)])[-1].reshape(
                    2, 41
                )
                for xi in MEDIUM_LOCATIONS
            ]
        ).transpose(1, 0, 2)
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
        for side_index, side in enumerate(SIDE_NAMES):
            image = axes[side_index].imshow(
                normal[side_index].T,
                aspect="auto",
                origin="lower",
                extent=(0.2, 0.8, 0.0, 1.0),
            )
            axes[side_index].set_title(side)
            axes[side_index].set_xlabel("Contact xi")
            axes[side_index].set_ylabel("Observation eta")
            fig.colorbar(image, ax=axes[side_index], label="u_normal (mm)")
        save("codtm_heatmap_1p5mm.png")
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
        for side_index, side in enumerate(SIDE_NAMES):
            image = axes[side_index].imshow(
                tangent[side_index].T,
                aspect="auto",
                origin="lower",
                extent=(0.2, 0.8, 0.0, 1.0),
            )
            axes[side_index].set_title(side)
            axes[side_index].set_xlabel("Contact xi")
            axes[side_index].set_ylabel("Observation eta")
            fig.colorbar(
                image, ax=axes[side_index], label="du_normal/d_delta"
            )
        save("tangent_transfer_gain_heatmap_1p5mm.png")

    final_slice = metrics["representative_slices"]["1.5"]
    if final_slice.get("available"):
        for matrix_key, name, label in (
            (
                "fixed_indentation_distance_matrix_mm",
                "fixed_indentation_distance_matrix.png",
                "Distance (mm)",
            ),
        ):
            plt.figure(figsize=(5.5, 4.8))
            image = plt.imshow(final_slice[matrix_key], origin="lower")
            plt.xticks(range(5), [f"{xi:.2f}" for xi in MEDIUM_LOCATIONS])
            plt.yticks(range(5), [f"{xi:.2f}" for xi in MEDIUM_LOCATIONS])
            plt.xlabel("Contact xi")
            plt.ylabel("Contact xi")
            plt.colorbar(image, label=label)
            save(name)
        plt.figure(figsize=(6.0, 4.5))
        for depth, data in metrics["representative_slices"].items():
            if data.get("available"):
                plt.semilogy(
                    data["signature_singular_values_mm"],
                    "o-",
                    label=f"{depth} mm",
                )
        plt.xlabel("Mode index")
        plt.ylabel("Singular value (mm)")
        plt.legend()
        save("signature_singular_values.png")

    force_metric = metrics["force_conditioned_separability"]
    if force_metric.get("available"):
        plt.figure(figsize=(5.5, 4.8))
        image = plt.imshow(
            force_metric["distance_matrices_mm"][-1], origin="lower"
        )
        plt.xticks(range(5), [f"{xi:.2f}" for xi in MEDIUM_LOCATIONS])
        plt.yticks(range(5), [f"{xi:.2f}" for xi in MEDIUM_LOCATIONS])
        plt.xlabel("Contact xi")
        plt.ylabel("Contact xi")
        plt.colorbar(image, label="Distance at common force (mm)")
        save("force_conditioned_distance_matrix.png")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    for axis, xi in zip(axes, FINE_LOCATIONS):
        for mesh, linestyle in (("medium", "-"), ("fine", "--")):
            records = by_key[(mesh, xi)]
            if records:
                profile = signature(records[-1]).reshape(2, 41)
                axis.plot(eta, profile[0], linestyle, label=f"{mesh} left")
                axis.plot(eta, profile[1], linestyle, label=f"{mesh} right")
        axis.set_title(f"xi={xi:.2f}")
        axis.set_xlabel("eta")
    axes[0].set_ylabel("u_normal (mm)")
    axes[-1].legend(fontsize=7)
    save("medium_fine_profile_comparison.png")
    return created

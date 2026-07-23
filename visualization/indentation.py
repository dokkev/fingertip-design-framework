"""Indentation history, profile, and deformed-mesh plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

def save_history_plots(
    result: Mapping[str, Any],
    plots_directory: Path,
) -> None:
    import matplotlib.pyplot as plt

    history = result.get("history", [])
    if not history:
        return
    plots_directory.mkdir(parents=True, exist_ok=True)
    indentation = [point["achieved_indentation_mm"] for point in history]

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot(
        indentation,
        [point["indenter_normal_reaction_n"] for point in history],
        marker="o",
        markersize=2.5,
    )
    axis.set(xlabel="Indentation [mm]", ylabel="Normal reaction [N]", title="Reaction–indentation")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(plots_directory / "reaction_curve.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot(
        indentation,
        [point["external_contact_width"]["chord_width_mm"] for point in history],
        label="Chord width",
    )
    axis.plot(
        indentation,
        [point["external_contact_width"]["arc_length_mm"] for point in history],
        label="Arc length",
    )
    axis.set(xlabel="Indentation [mm]", ylabel="Contact extent [mm]", title="External contact width")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots_directory / "contact_width.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.0, 4.4))
    for group_name in (
        "external_pad_indenter",
        "internal_left",
        "internal_right",
        "internal_bottom",
    ):
        if group_name not in history[0]["contact_groups"]:
            continue
        axis.plot(
            indentation,
            [point["contact_groups"][group_name]["active_condition_count"] for point in history],
            label=group_name,
        )
    axis.set(xlabel="Indentation [mm]", ylabel="ACTIVE generated conditions", title="Contact groups")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(plots_directory / "contact_groups.png", dpi=180)
    plt.close(figure)

    figure, first_axis = plt.subplots(figsize=(6.8, 4.4))
    second_axis = first_axis.twinx()
    first_axis.plot(
        indentation,
        [point["pad_strain_det_f"]["maximum_principal_green_lagrange_strain"]["value"] for point in history],
        color="#B2182B",
        label="Maximum principal strain",
    )
    second_axis.plot(
        indentation,
        [point["pad_strain_det_f"]["det_f"]["min"] for point in history],
        color="#2166AC",
        label="Minimum det(F)",
    )
    first_axis.set(xlabel="Indentation [mm]", ylabel="Green–Lagrange strain", title="Pad strain and det(F)")
    second_axis.set_ylabel("Minimum det(F)")
    first_axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(plots_directory / "strain_detf.png", dpi=180)
    plt.close(figure)


def save_outer_profile_plot(snapshots: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    if not snapshots:
        return
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True)
    for key, snapshot in sorted(snapshots.items(), key=lambda item: float(item[0])):
        profile = snapshot["profile"]
        coordinate = [record["normalized_arc_coordinate"] for record in profile]
        axes[0].plot(
            coordinate,
            [record["local_normal_displacement_mm"] for record in profile],
            label=f"{float(key):g} mm",
        )
        axes[1].plot(
            coordinate,
            [record["local_tangential_displacement_mm"] for record in profile],
            label=f"{float(key):g} mm",
        )
    axes[0].set_ylabel("Normal displacement [mm]")
    axes[1].set_ylabel("Tangential displacement [mm]")
    axes[1].set_xlabel("Normalized reference PadOuterArc coordinate")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Pad outer-arc displacement profiles")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_deformed_mesh_plot(
    artifacts: Any,
    snapshot: Mapping[str, Any],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection

    mesh = artifacts.mesh
    displacements = snapshot["displacements"]
    figure, axis = plt.subplots(figsize=(8.2, 8.2))

    undeformed_pad = [
        [(mesh.nodes[node_id].x_mm, mesh.nodes[node_id].y_mm) for node_id in element.node_ids]
        for element in mesh.pad_elements
    ]
    deformed_pad = [
        [
            (
                mesh.nodes[node_id].x_mm + displacements[node_id][0],
                mesh.nodes[node_id].y_mm + displacements[node_id][1],
            )
            for node_id in element.node_ids
        ]
        for element in mesh.pad_elements
    ]
    axis.add_collection(
        PolyCollection(undeformed_pad, facecolors="none", edgecolors="#9E9E9E", linewidths=0.08, alpha=0.35)
    )
    axis.add_collection(
        PolyCollection(deformed_pad, facecolors="#9ED7E5", edgecolors="#3B7C8C", linewidths=0.08, alpha=0.62)
    )

    carrier = [
        [
            (
                mesh.nodes[node_id].x_mm + displacements[node_id][0],
                mesh.nodes[node_id].y_mm + displacements[node_id][1],
            )
            for node_id in element.node_ids
        ]
        for element in mesh.carrier_elements
    ]
    axis.add_collection(
        PolyCollection(carrier, facecolors="#747B84", edgecolors="#444A50", linewidths=0.08, alpha=0.8)
    )

    node_map = artifacts.indenter_topology.local_to_global_node_id
    indenter = [
        [
            (
                artifacts.indenter_mesh.nodes[local_id].x_mm + displacements[node_map[local_id]][0],
                artifacts.indenter_mesh.nodes[local_id].y_mm + displacements[node_map[local_id]][1],
            )
            for local_id in element.node_ids
        ]
        for element in artifacts.indenter_mesh.elements
    ]
    axis.add_collection(
        PolyCollection(indenter, facecolors="#D0D0D0", edgecolors="#555555", linewidths=0.1, alpha=0.9)
    )

    active_external = snapshot["active_external_node_ids"]
    if active_external:
        axis.scatter(
            [mesh.nodes[node_id].x_mm + displacements[node_id][0] for node_id in active_external],
            [mesh.nodes[node_id].y_mm + displacements[node_id][1] for node_id in active_external],
            s=12,
            color="#D73027",
            label="ACTIVE external",
            zorder=8,
        )
    internal_colors = {
        "internal_left": "#7B3294",
        "internal_right": "#008837",
        "internal_bottom": "#F46D43",
    }
    for name, node_ids in snapshot["active_internal_node_ids"].items():
        if node_ids:
            axis.scatter(
                [mesh.nodes[node_id].x_mm + displacements[node_id][0] for node_id in node_ids],
                [mesh.nodes[node_id].y_mm + displacements[node_id][1] for node_id in node_ids],
                s=10,
                color=internal_colors[name],
                label=f"ACTIVE {name}",
                zorder=8,
            )
    statistics = snapshot["pad_strain_det_f"]
    strain_point = statistics["maximum_principal_green_lagrange_strain"]["reference_coordinate_mm"]
    det_point = statistics["det_f"]["minimum_reference_coordinate_mm"]
    axis.scatter(*strain_point, marker="*", s=90, color="#B2182B", label="Maximum strain location", zorder=9)
    axis.scatter(*det_point, marker="X", s=55, color="#2166AC", label="Minimum det(F) location", zorder=9)

    all_points = [point for polygon in (*deformed_pad, *carrier, *indenter) for point in polygon]
    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]
    padding = 0.04 * max(max(x_values) - min(x_values), max(y_values) - min(y_values))
    axis.set_xlim(min(x_values) - padding, max(x_values) + padding)
    axis.set_ylim(min(y_values) - padding, max(y_values) + padding)
    axis.set_aspect("equal", adjustable="box")
    axis.set(xlabel="x [mm]", ylabel="y [mm]", title=f"Phase 4I at {snapshot['depth_mm']:g} mm — displacement scale 1×")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=7, frameon=False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


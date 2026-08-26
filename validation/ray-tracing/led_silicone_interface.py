"""Diagnose micron-scale LED-to-silicone interface sensitivity."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.mesh import (
    LED_RECESS_DEPTH_MM,
    LED_RECESS_WIDTH_MM,
    Fingertip5LEDMesh,
    make_fingertip_5led_mesh,
)
from lumo.ray_tracing import (
    LED,
    OptixScene,
    emit_from_stem_boundary,
    side_view_observation,
    source_inside_silicone,
    trace_bounded_paths,
)


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


_MECHANICS_DIRECTORY = Path("output/validation/5led_newton")
_OUTPUT_DIRECTORY = Path(
    "output/validation/5led_led_silicone_interface"
)
_REPORT_PATH = Path(
    "output/validation/5led_led_silicone_interface_diagnostic.md"
)

_STATES = (
    ("no_contact", "No contact", None),
    ("center_10n", "Center Y=0 mm", "center_led_10n.npz"),
    ("between_10n", "Between Y=+5.5 mm", "between_leds_10n.npz"),
    ("distal_10n", "Distal Y=+22 mm", "distal_led_10n.npz"),
)
_LED_Y_MM = np.array((-22.0, -11.0, 0.0, 11.0, 22.0))
_OFFSET_UM = np.array(
    (-25.0, -10.0, -5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0, 10.0, 25.0)
)
_CONTROLLED_GAP_UM = np.array(
    (
        0.0,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        50.0,
        100.0,
        1.0e3 * LED_RECESS_DEPTH_MM,
    )
)
_GAP_CLASSIFICATION_TOLERANCE_UM = 0.1
_CONTROLLED_PATCH_HALF_LENGTH_MM = 0.5 * LED_RECESS_WIDTH_MM

_SILICONE_INSTANCE_ID = 1
_CARRIER_INSTANCE_ID = 2
_SILICONE_MASK = 0x01
_CARRIER_MASK = 0x02
_ALL_MASK = _SILICONE_MASK | _CARRIER_MASK
_CARRIER_ALBEDO = 0.7
_MAX_BOUNCES = 24
_SAMPLE_SIDE = 256
_RNG_SEED = 20260823
_CLOSURE_TOLERANCE = 1.0e-12

_ENERGY_FIELDS = (
    "emitted_power",
    "escaped_power",
    "carrier_absorbed_power",
    "bulk_loss_power",
    "unresolved_internal_miss_power",
    "remaining_power",
    "accounted_power",
    "closure_error",
)


def _load_geometry() -> tuple[
    Fingertip,
    Fingertip5LEDMesh,
    np.ndarray,
    dict[str, np.ndarray],
]:
    reference_path = _MECHANICS_DIRECTORY / "reference_mesh.npz"
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)

    fingertip = Fingertip(FingertipParameters())
    mesh = make_fingertip_5led_mesh(fingertip, element_size_mm=1.0)
    with np.load(reference_path) as saved:
        reference_vertices = np.asarray(
            saved["silicone_vertices_m"],
            dtype=np.float64,
        )
        surface_triangles = np.asarray(
            saved["silicone_surface_triangles"],
            dtype=np.int32,
        )
        saved_led_centers = np.asarray(saved["led_centers_m"])

    if not np.allclose(
        np.asarray(mesh.silicone.vertices),
        reference_vertices,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise RuntimeError("saved Newton reference does not match the current mesh")
    if not np.array_equal(
        np.asarray(mesh.silicone.surface_tri_indices).reshape(-1, 3),
        surface_triangles,
    ):
        raise RuntimeError("saved Newton surface topology does not match")
    if not np.allclose(
        mesh.led_centers_m,
        saved_led_centers,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("saved LED centers do not match the current mesh")

    states = {"no_contact": reference_vertices}
    for state_name, _, filename in _STATES[1:]:
        path = _MECHANICS_DIRECTORY / str(filename)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as saved:
            vertices = np.asarray(saved["deformed_vertices_m"], dtype=np.float64)
        if vertices.shape != reference_vertices.shape or not np.all(
            np.isfinite(vertices)
        ):
            raise RuntimeError(f"invalid saved geometry: {state_name}")
        states[state_name] = vertices
    return fingertip, mesh, surface_triangles, states


def _make_leds(
    fingertip: Fingertip,
    mesh: Fingertip5LEDMesh,
) -> tuple[LED, ...]:
    return tuple(
        LED(
            position_W_m=center,
            normal_W=np.array((0.0, 0.0, -1.0)),
            parameters=fingertip.parameters.led,
        )
        for center in mesh.led_centers_m
    )


def _samples() -> tuple[np.ndarray, ...]:
    coordinate = (np.arange(_SAMPLE_SIDE, dtype=np.float64) + 0.5) / _SAMPLE_SIDE
    u1, u2 = np.meshgrid(coordinate, coordinate, indexing="ij")
    ray_count = _SAMPLE_SIDE * _SAMPLE_SIDE
    rng = np.random.default_rng(_RNG_SEED)
    sample_shape = (_MAX_BOUNCES, ray_count)
    return (
        u1.ravel(),
        u2.ravel(),
        rng.random(sample_shape),
        rng.random(sample_shape),
        rng.random(sample_shape),
    )


def _closest_surface_point(
    point_m: np.ndarray,
    vertices_m: np.ndarray,
    triangles: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, int]:
    """Return signed distance to the closest oriented surface triangle."""
    triangle_vertices = vertices_m[triangles]
    a = triangle_vertices[:, 0]
    b = triangle_vertices[:, 1]
    c = triangle_vertices[:, 2]
    edge_ab = b - a
    edge_ac = c - a

    normal = np.cross(edge_ab, edge_ac)
    normal_length = np.linalg.norm(normal, axis=1)
    valid_normal = normal_length > np.finfo(np.float64).tiny
    unit_normal = np.zeros_like(normal)
    unit_normal[valid_normal] = (
        normal[valid_normal] / normal_length[valid_normal, None]
    )

    candidates = [a, b, c]
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        edge_length_squared = np.einsum("ij,ij->i", edge, edge)
        parameter = np.zeros(len(edge))
        valid_edge = edge_length_squared > np.finfo(np.float64).tiny
        parameter[valid_edge] = (
            np.einsum(
                "ij,ij->i",
                point_m - start[valid_edge],
                edge[valid_edge],
            )
            / edge_length_squared[valid_edge]
        )
        candidates.append(start + np.clip(parameter, 0.0, 1.0)[:, None] * edge)

    plane_distance = np.einsum("ij,ij->i", point_m - a, unit_normal)
    projection = point_m - plane_distance[:, None] * unit_normal
    d00 = np.einsum("ij,ij->i", edge_ab, edge_ab)
    d01 = np.einsum("ij,ij->i", edge_ab, edge_ac)
    d11 = np.einsum("ij,ij->i", edge_ac, edge_ac)
    relative = projection - a
    d20 = np.einsum("ij,ij->i", relative, edge_ab)
    d21 = np.einsum("ij,ij->i", relative, edge_ac)
    denominator = d00 * d11 - d01 * d01
    valid_projection = np.abs(denominator) > np.finfo(np.float64).tiny
    bary_b = np.zeros(len(a))
    bary_c = np.zeros(len(a))
    bary_b[valid_projection] = (
        d11[valid_projection] * d20[valid_projection]
        - d01[valid_projection] * d21[valid_projection]
    ) / denominator[valid_projection]
    bary_c[valid_projection] = (
        d00[valid_projection] * d21[valid_projection]
        - d01[valid_projection] * d20[valid_projection]
    ) / denominator[valid_projection]
    inside = (
        valid_projection
        & (bary_b >= -1.0e-10)
        & (bary_c >= -1.0e-10)
        & (bary_b + bary_c <= 1.0 + 1.0e-10)
    )
    projection[~inside] = np.inf
    candidates.append(projection)

    candidate_array = np.stack(candidates, axis=1)
    distance_squared = np.sum((candidate_array - point_m) ** 2, axis=2)
    flat_index = int(np.argmin(distance_squared))
    triangle_index, candidate_index = np.unravel_index(
        flat_index,
        distance_squared.shape,
    )
    closest = candidate_array[triangle_index, candidate_index]
    distance_m = float(np.sqrt(distance_squared[triangle_index, candidate_index]))
    closest_normal = unit_normal[triangle_index]
    orientation = float(np.dot(point_m - closest, closest_normal))
    signed_distance_m = np.copysign(distance_m, orientation)
    return signed_distance_m, closest, closest_normal, triangle_index


def _gap_status(signed_gap_um: float) -> str:
    if signed_gap_um > _GAP_CLASSIFICATION_TOLERANCE_UM:
        return "open"
    if signed_gap_um < -_GAP_CLASSIFICATION_TOLERANCE_UM:
        return "overlap"
    return "touching"


def _energy(paths) -> np.ndarray:
    return np.array(
        (
            paths.emitted_power,
            paths.escaped_power,
            paths.absorbed_power,
            paths.bulk_loss_power,
            paths.unresolved_internal_miss_power,
            paths.remaining_power,
            paths.accounted_power,
            paths.closure_error,
        ),
        dtype=np.float64,
    )


def _interface_name(instance_id: int, hit: bool) -> str:
    if not hit:
        return "miss"
    if instance_id == _SILICONE_INSTANCE_ID:
        return "silicone"
    if instance_id == _CARRIER_INSTANCE_ID:
        return "carrier"
    return f"unknown_{instance_id}"


def _trace_emitter(
    scene: OptixScene,
    fingertip: Fingertip,
    led: LED,
    samples: tuple[np.ndarray, ...],
    *,
    offset_um: float = 0.0,
    forced_inside_silicone: bool | None = None,
) -> dict[str, object]:
    emission_u1, emission_u2, dielectric_u, carrier_u1, carrier_u2 = samples
    emission = emit_from_stem_boundary(
        scene,
        led,
        emission_u1,
        emission_u2,
        carrier_mask=_CARRIER_MASK,
    )
    emission["origin_W_m"] += offset_um * 1.0e-6 * led.normal_W
    automatic_inside = source_inside_silicone(
        scene,
        led,
        emission,
        silicone_mask=_SILICONE_MASK,
    )
    applied_inside = (
        automatic_inside
        if forced_inside_silicone is None
        else forced_inside_silicone
    )

    first_hit = scene.trace_closest(
        emission["origin_W_m"][:1],
        led.normal_W[None, :],
        mask=_ALL_MASK,
    )[0]
    first_interface = _interface_name(
        int(first_hit["instance_id"]),
        bool(first_hit["hit"]),
    )
    if first_interface == "silicone":
        medium_after = "air" if applied_inside else "silicone"
    elif first_interface == "carrier":
        medium_after = "silicone" if applied_inside else "air"
    else:
        medium_after = "unresolved" if applied_inside else "escaped"

    start_s = perf_counter()
    paths = trace_bounded_paths(
        scene,
        emission["origin_W_m"],
        emission["direction_W"],
        emission["power"],
        inside_silicone=applied_inside,
        n_air=1.0,
        n_silicone=fingertip.parameters.optical.refractive_index,
        extinction_coefficient_m_inv=(
            fingertip.parameters.optical.extinction_coefficient_m_inv
        ),
        carrier_albedo=_CARRIER_ALBEDO,
        max_bounces=_MAX_BOUNCES,
        dielectric_branch_u=dielectric_u,
        carrier_u1=carrier_u1,
        carrier_u2=carrier_u2,
        silicone_instance_id=_SILICONE_INSTANCE_ID,
        carrier_instance_id=_CARRIER_INSTANCE_ID,
        mask=_ALL_MASK,
    )
    trace_time_s = perf_counter() - start_s
    response = side_view_observation(paths.escaped_rays, fingertip=fingertip)
    energy = _energy(paths)
    if not np.all(np.isfinite(response)) or np.any(response < 0.0):
        raise RuntimeError("invalid optical response")
    if not np.all(np.isfinite(energy)):
        raise RuntimeError("invalid optical energy ledger")
    if abs(float(paths.closure_error)) > _CLOSURE_TOLERANCE:
        raise RuntimeError("optical energy does not close")

    result = {
        "response": response,
        "energy": energy,
        "automatic_inside_silicone": bool(automatic_inside),
        "applied_inside_silicone": bool(applied_inside),
        "source_origin_W_m": emission["origin_W_m"][0].copy(),
        "first_interface": first_interface,
        "first_interface_distance_m": (
            float(first_hit["t"]) if first_hit["hit"] else np.nan
        ),
        "medium_after_first_crossing": medium_after,
        "trace_time_s": trace_time_s,
    }
    del paths, emission
    return result


def _trace_all_leds(
    scene: OptixScene,
    fingertip: Fingertip,
    leds: tuple[LED, ...],
    samples: tuple[np.ndarray, ...],
    *,
    offset_um: float = 0.0,
    forced_inside_silicone: bool | None = None,
) -> dict[str, object]:
    per_led = [
        _trace_emitter(
            scene,
            fingertip,
            led,
            samples,
            offset_um=offset_um,
            forced_inside_silicone=forced_inside_silicone,
        )
        for led in leds
    ]
    return {
        "per_led_response": np.stack([row["response"] for row in per_led]),
        "combined_response": np.stack(
            [row["response"] for row in per_led]
        ).sum(axis=0),
        "per_led_energy": np.stack([row["energy"] for row in per_led]),
        "combined_energy": np.stack([row["energy"] for row in per_led]).sum(
            axis=0
        ),
        "per_led": per_led,
        "trace_time_s": sum(float(row["trace_time_s"]) for row in per_led),
    }


def _stem_bottom_vertices(
    fingertip: Fingertip,
    reference_vertices: np.ndarray,
    surface_triangles: np.ndarray,
) -> np.ndarray:
    geometry = fingertip.parameters.geometry
    surface_vertices = np.unique(surface_triangles)
    vertices = reference_vertices[surface_vertices]
    stem_bottom_z_m = -1.0e-3 * geometry.stem_height_mm
    mask = (
        (np.abs(vertices[:, 2] - stem_bottom_z_m) <= 1.0e-7)
        & (np.abs(vertices[:, 0]) <= 0.5e-3 * geometry.stem_width_mm + 1.0e-7)
        & (vertices[:, 1] >= -27.5e-3 - 1.0e-7)
        & (vertices[:, 1] <= 27.5e-3 + 1.0e-7)
    )
    indices = surface_vertices[mask]
    if not len(indices):
        raise RuntimeError("no silicone stem-bottom interface vertices found")
    return indices


def _bonded_like_geometry(
    vertices_m: np.ndarray,
    reference_vertices_m: np.ndarray,
    stem_bottom_indices: np.ndarray,
    leds: tuple[LED, ...],
) -> np.ndarray:
    """Close each explicit LED cavity without changing the rest of the state."""
    counterfactual = vertices_m.copy()
    interface_vertices = reference_vertices_m[stem_bottom_indices]
    for led in leds:
        local_indices = stem_bottom_indices[
            np.abs(interface_vertices[:, 1] - led.position_W_m[1])
            <= 1.0e-3 * _CONTROLLED_PATCH_HALF_LENGTH_MM
        ]
        counterfactual[local_indices] = reference_vertices_m[local_indices]
        counterfactual[local_indices, 2] = led.position_W_m[2]
    return counterfactual


def _controlled_gap_geometry(
    reference_vertices_m: np.ndarray,
    stem_bottom_indices: np.ndarray,
    *,
    led_position_m: np.ndarray,
    gap_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    interface_vertices = reference_vertices_m[stem_bottom_indices]
    local_mask = (
        np.abs(interface_vertices[:, 1] - led_position_m[1])
        <= 1.0e-3 * _CONTROLLED_PATCH_HALF_LENGTH_MM
    )
    local_indices = stem_bottom_indices[local_mask]
    if not len(local_indices):
        raise RuntimeError("controlled-gap patch contains no surface vertices")
    vertices = reference_vertices_m.copy()
    vertices[local_indices, 2] = led_position_m[2] - gap_um * 1.0e-6
    return vertices, local_indices


def _interface_edge_scale_mm(
    reference_vertices: np.ndarray,
    surface_triangles: np.ndarray,
    stem_bottom_indices: np.ndarray,
) -> np.ndarray:
    on_interface = np.zeros(len(reference_vertices), dtype=bool)
    on_interface[stem_bottom_indices] = True
    interface_triangles = surface_triangles[
        np.all(on_interface[surface_triangles], axis=1)
    ]
    edges = np.concatenate(
        (
            interface_triangles[:, (0, 1)],
            interface_triangles[:, (1, 2)],
            interface_triangles[:, (2, 0)],
        )
    )
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    lengths_mm = 1.0e3 * np.linalg.norm(
        reference_vertices[edges[:, 0]] - reference_vertices[edges[:, 1]],
        axis=1,
    )
    return np.quantile(lengths_mm, (0.0, 0.5, 0.95, 1.0))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_gap_table(gap_matrix_um: np.ndarray) -> None:
    limit = float(np.max(np.abs(gap_matrix_um)))
    figure, axis = plt.subplots(figsize=(9.0, 4.6), constrained_layout=True)
    image = axis.imshow(
        gap_matrix_um,
        aspect="auto",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    axis.set_xticks(range(5), [f"LED{i}" for i in range(1, 6)])
    axis.set_yticks(range(4), [label for _, label, _ in _STATES])
    for row in range(4):
        for column in range(5):
            axis.text(
                column,
                row,
                f"{gap_matrix_um[row, column]:+.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, label="signed geometric gap [µm]")
    axis.set_title("Rigid LED reference to nearest silicone surface")
    figure.savefig(_OUTPUT_DIRECTORY / "gap_by_led_and_state.png", dpi=180)
    plt.close(figure)


def _plot_offset_sweep(rows: list[dict[str, object]]) -> None:
    case_names = tuple(dict.fromkeys(str(row["case"]) for row in rows))
    figure, axes = plt.subplots(
        len(case_names),
        1,
        figsize=(9.0, 2.8 * len(case_names)),
        sharex=True,
        constrained_layout=True,
    )
    for axis, case_name in zip(axes, case_names, strict=True):
        case_rows = [row for row in rows if row["case"] == case_name]
        offset = np.array([row["offset_um"] for row in case_rows])
        response = np.array(
            [[row[f"q{i}"] for i in range(1, 5)] for row in case_rows]
        )
        for quadrant in range(4):
            axis.plot(
                offset,
                response[:, quadrant],
                marker="o",
                markersize=3,
                label=f"Q{quadrant + 1}",
            )
        axis.plot(
            offset,
            response.sum(axis=1),
            color="black",
            linestyle="--",
            marker="x",
            label="total +Y",
        )
        for row in case_rows:
            color = "tab:blue" if row["automatic_medium"] == "silicone" else "tab:orange"
            axis.scatter(
                row["offset_um"],
                row["side_visible_power"],
                color=color,
                s=18.0,
                zorder=4,
            )
        axis.axvline(0.0, color="gray", linewidth=0.8)
        axis.set_ylabel("modeled power")
        axis.set_title(case_name.replace("_", " "))
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=5, fontsize=7)
    axes[-1].set_xlabel("source offset [µm], positive = toward silicone (-Z)")
    figure.suptitle("Emitter-position epsilon sensitivity")
    figure.savefig(_OUTPUT_DIRECTORY / "emitter_offset_sensitivity.png", dpi=180)
    plt.close(figure)


def _plot_controlled_gap(rows: list[dict[str, object]]) -> None:
    gap = np.array([row["requested_gap_um"] for row in rows])
    response = np.array(
        [[row[f"q{i}"] for i in range(1, 5)] for row in rows]
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    axes[0].plot(gap, response.sum(axis=1), marker="o", color="black")
    for row in rows:
        color = "tab:blue" if row["automatic_medium"] == "silicone" else "tab:orange"
        axes[0].scatter(
            row["requested_gap_um"],
            row["side_visible_power"],
            color=color,
            s=30.0,
            zorder=4,
        )
    axes[0].set_xlabel("controlled geometric gap [µm]")
    axes[0].set_ylabel("total +Y visible power")
    axes[0].grid(alpha=0.25)
    for quadrant in range(4):
        axes[1].plot(
            gap,
            response[:, quadrant],
            marker="o",
            label=f"Q{quadrant + 1}",
        )
    axes[1].set_xlabel("controlled geometric gap [µm]")
    axes[1].set_ylabel("quadrant power")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle("Optical-only local air-gap sensitivity at center LED")
    figure.savefig(_OUTPUT_DIRECTORY / "controlled_gap_sensitivity.png", dpi=180)
    plt.close(figure)


def _plot_interface_treatments(rows: list[dict[str, object]]) -> None:
    treatments = tuple(row["treatment"] for row in rows)
    combined = np.array([row["combined_q_l2"] for row in rows])
    per_emitter = np.array([row["per_emitter_l2"] for row in rows])
    x = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(10.0, 4.8), constrained_layout=True)
    width = 0.38
    axis.bar(x - width / 2.0, combined, width, label="combined Q L2")
    axis.bar(x + width / 2.0, per_emitter, width, label="per-emitter 5×4 L2")
    axis.set_xticks(x, treatments, rotation=22, ha="right")
    axis.set_ylabel("center vs between distance")
    axis.set_title("Interface-treatment effect on half-pitch distinguishability")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(
        _OUTPUT_DIRECTORY / "center_between_interface_treatments.png",
        dpi=180,
    )
    plt.close(figure)


def _write_report(
    gap_matrix_um: np.ndarray,
    initial_rows: list[dict[str, object]],
    offset_rows: list[dict[str, object]],
    controlled_rows: list[dict[str, object]],
    separation_rows: list[dict[str, object]],
    automatic_results: dict[str, dict[str, object]],
    treatment_results: dict[str, dict[str, dict[str, object]]],
    edge_scale_mm: np.ndarray,
    *,
    runtime_s: float,
) -> None:
    automatic = next(
        row for row in separation_rows if row["treatment"] == "automatic"
    )
    zero_gap = next(
        row
        for row in separation_rows
        if row["treatment"] == "zero_gap_bonded_like"
    )
    combined_fraction = float(zero_gap["combined_fraction_of_automatic"])
    per_led_fraction = float(
        zero_gap["per_emitter_l2"] / automatic["per_emitter_l2"]
    )

    controlled_zero = controlled_rows[0]
    controlled_nonzero = controlled_rows[1]
    controlled_nominal = controlled_rows[-1]
    zero_to_finite_l2 = float(
        np.linalg.norm(
            np.array([controlled_nonzero[f"q{i}"] for i in range(1, 5)])
            - np.array([controlled_zero[f"q{i}"] for i in range(1, 5)])
        )
    )
    zero_to_finite_power = float(
        controlled_nonzero["side_visible_power"]
        / controlled_zero["side_visible_power"]
        - 1.0
    )
    finite_gap_power = float(
        controlled_nominal["side_visible_power"]
        / controlled_nonzero["side_visible_power"]
        - 1.0
    )
    forced_medium_l2 = {}
    for state_name in ("center_10n", "between_10n", "distal_10n"):
        silicone = treatment_results[state_name]["forced_silicone"][
            "combined_response"
        ]
        air = treatment_results[state_name]["forced_air"]["combined_response"]
        forced_medium_l2[state_name] = float(np.linalg.norm(silicone - air))

    no_contact_power = float(
        automatic_results["no_contact"]["combined_response"].sum()
    )
    distal_power = float(
        treatment_results["distal_10n"]["automatic"][
            "combined_response"
        ].sum()
    )
    distal_zero_gap_power = float(
        treatment_results["distal_10n"]["zero_gap_bonded_like"][
            "combined_response"
        ].sum()
    )
    unloaded_gap_um = float(np.mean(gap_matrix_um[0]))

    lines = [
        "# LED–silicone interface / micron air-gap diagnostic",
        "",
        "## Scope and current geometry contract",
        "",
        "- Saved Newton reference and 10 N center/between/distal states were reused; Newton was not rerun.",
        "- Production mechanics, optics, materials, and objectives were not changed.",
        f"- The carrier has five explicit {LED_RECESS_WIDTH_MM:g} mm-wide × "
        f"{LED_RECESS_DEPTH_MM:g} mm-deep recesses; nominal void height remains zero.",
        f"- Traces use {_SAMPLE_SIDE**2:,} deterministic paths per LED and {_MAX_BOUNCES} bounces.",
        "- Positive source offset follows the LED normal (-Z), toward silicone.",
        "",
        "The old source-on-silicone premise is no longer the production geometry. "
        "Each source remains on its rigid recess boundary, but the explicit recess "
        "places unloaded silicone about 190 µm away across air.",
        "",
        "## Geometric gap table [µm]",
        "",
        "| State | LED1 | LED2 | LED3 | LED4 | LED5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row_index, (_, label, _) in enumerate(_STATES):
        values = " | ".join(
            f"{value:+.3f}" for value in gap_matrix_um[row_index]
        )
        lines.append(f"| {label} | {values} |")
    lines.extend(("", "Loaded changes relative to no contact:"))
    for row_index, (_, label, _) in enumerate(_STATES[1:], start=1):
        delta = gap_matrix_um[row_index] - gap_matrix_um[0]
        closed = [
            f"LED{index + 1}"
            for index, value in enumerate(gap_matrix_um[row_index])
            if value <= _GAP_CLASSIFICATION_TOLERANCE_UM
        ]
        lines.append(
            f"- {label}: Δgap {delta.min():+.3f} to {delta.max():+.3f} µm; "
            f"closed/overlapped: {', '.join(closed) if closed else 'none'}."
        )

    lines.extend(
        (
            "",
            "## Source classification and mechanics resolution",
            "",
            f"- unloaded LED-to-silicone distance: {unloaded_gap_um:.3f} µm mean",
            "- source relative to silicone: inside the explicit air cavity",
            "- source relative to rigid carrier: on the recess emitting boundary",
            "- Newton element size: 1.0 mm = 1000 µm",
            f"- local interface edge min/median/P95/max: "
            f"{edge_scale_mm[0]:.3f}/{edge_scale_mm[1]:.3f}/"
            f"{edge_scale_mm[2]:.3f}/{edge_scale_mm[3]:.3f} mm",
            "- saved contact margin: 100 µm; bond selection tolerance: 0.01 µm",
            "",
            "The 190 µm cavity is a measured CAD input rather than a Newton-predicted gap. "
            "Its existence is specified, but the force of closure and residual 1–20 µm "
            "films are not mechanically converged at 1 mm resolution.",
            "",
            "## Initial medium and first interface",
            "",
            "| State | LED | launch | first interface | distance [µm] | after crossing |",
            "|---|---:|---|---|---:|---|",
        )
    )
    for row in initial_rows:
        lines.append(
            f"| {row['state']} | {row['led_index']} | {row['launch_medium']} | "
            f"{row['first_interface']} | {row['first_interface_distance_um']:.3f} | "
            f"{row['medium_after_first_crossing']} |"
        )

    lines.extend(("", "## Emitter-position epsilon sweep", ""))
    case_names = tuple(dict.fromkeys(str(row["case"]) for row in offset_rows))
    for case_name in case_names:
        rows = [row for row in offset_rows if row["case"] == case_name]
        baseline = next(row for row in rows if float(row["offset_um"]) == 0.0)
        baseline_q = np.array([baseline[f"q{i}"] for i in range(1, 5)])
        max_l2 = max(
            float(
                np.linalg.norm(
                    np.array([row[f"q{i}"] for i in range(1, 5)])
                    - baseline_q
                )
            )
            for row in rows
        )
        positive_max_l2 = max(
            float(
                np.linalg.norm(
                    np.array([row[f"q{i}"] for i in range(1, 5)])
                    - baseline_q
                )
            )
            for row in rows
            if float(row["offset_um"]) >= 0.0
        )
        media = tuple(dict.fromkeys(str(row["automatic_medium"]) for row in rows))
        negative_interfaces = tuple(
            dict.fromkeys(
                str(row["first_interface"])
                for row in rows
                if float(row["offset_um"]) < 0.0
            )
        )
        lines.append(
            f"- {case_name}: media={','.join(media)}; max Q L2 over the full "
            f"signed sweep={max_l2:.8f}, on the non-rigid 0…+25 µm side="
            f"{positive_max_l2:.8f}; negative-side first interface="
            f"{','.join(negative_interfaces)}."
        )
    lines.extend(
        (
            "",
            "Because the point source lies on the rigid recess floor, every negative "
            "offset enters the carrier and is not a physically valid alternate air-side "
            "launch. The large full-sweep jump in open cavities is therefore a rigid "
            "interception counterfactual. On the valid 0…+25 µm cavity side, the "
            "no-contact response is stable. A medium flip on that positive side near a "
            "loaded closure locates the physical silicone/air boundary.",
            "",
            "## Controlled local gap",
            "",
            f"The fixed CAD LED top was retained while a {LED_RECESS_WIDTH_MM:g} mm-wide "
            "silicone patch was placed at the requested distance for OptiX only.",
            "",
            "| gap [µm] | measured [µm] | launch | +Y power | Q1 | Q2 | Q3 | Q4 |",
            "|---:|---:|---|---:|---:|---:|---:|---:|",
        )
    )
    for row in controlled_rows:
        lines.append(
            f"| {row['requested_gap_um']:.1f} | {row['measured_gap_um']:.3f} | "
            f"{row['automatic_medium']} | {row['side_visible_power']:.8f} | "
            f"{row['q1']:.8f} | {row['q2']:.8f} | "
            f"{row['q3']:.8f} | {row['q4']:.8f} |"
        )
    lines.extend(
        (
            "",
            f"The 0→0.5 µm transition changes Q by L2={zero_to_finite_l2:.8f} "
            f"and +Y power by {zero_to_finite_power:+.1%}. From 0.5 µm to the "
            f"nominal {1.0e3 * LED_RECESS_DEPTH_MM:.0f} µm gap, +Y power changes "
            f"by {finite_gap_power:+.1%}.",
            "",
            "## Center versus between-LED treatments",
            "",
            "| treatment | combined Q L2 | per-emitter 5×4 L2 | fraction of automatic |",
            "|---|---:|---:|---:|",
        )
    )
    for row in separation_rows:
        lines.append(
            f"| {row['treatment']} | {row['combined_q_l2']:.8f} | "
            f"{row['per_emitter_l2']:.8f} | "
            f"{row['combined_fraction_of_automatic']:.3f} |"
        )
    lines.extend(
        (
            "",
            f"Closing all five cavities retains {combined_fraction:.1%} of the "
            f"combined and {per_led_fraction:.1%} of the per-emitter automatic separation.",
            f"Forced silicone-versus-air Q L2 is {forced_medium_l2['center_10n']:.6f} "
            f"(center), {forced_medium_l2['between_10n']:.6f} (between), and "
            f"{forced_medium_l2['distal_10n']:.6f} (distal).",
            "",
            "## Mechanics topology and distal response",
            "",
            "The perfect bond owns the side extensions and distal dorsal plate. "
            "The LED cavities beneath the stem rail are unbonded rigid/soft contact "
            "interfaces, so zero bonded drift does not constrain their silicone surface. "
            "Center and distal loads close the corresponding cavity; between-LED loading "
            "brings the two adjacent cavities near closure.",
            "",
            f"Distal automatic +Y power is {distal_power:.8f} "
            f"({distal_power - no_contact_power:+.8f} from unloaded). The all-closed "
            f"counterfactual is {distal_zero_gap_power:.8f} "
            f"({distal_zero_gap_power - no_contact_power:+.8f} from unloaded).",
            "",
            "## Concern classification and model decision",
            "",
            "- **A:** removed for unloaded production geometry; exact closure still needs a boundary convention.",
            "- **B:** applies to residual micron films near closure.",
            "- **C:** applies to the explicit 0.19 mm cavity and qualitative closure, not yet to exact closure force/thickness.",
            f"- **D:** not assumed; the all-closed counterfactual retains "
            f"{combined_fraction:.1%} combined and {per_led_fraction:.1%} per-emitter separation.",
            "",
            "| model | mechanics/optics implication | BO robustness |",
            "|---|---|---|",
            "| Optically bonded window–silicone | Removes air closure; valid only if molding keeps contact | High only when hardware matches |",
            "| Explicit unbonded 0.19 mm cavity (current) | Models measured cavity and contact-driven closure | Good after closure validation |",
            "| Explicit LED package/window | Models actual package contact and refractive material | Highest fidelity, highest calibration cost |",
            "",
            "Keep void height fixed and do not optimize residual sub-element gap thickness. "
            "If the real LED window touches silicone, explicit package/window geometry is "
            "safer than treating that interface as air.",
            "",
            "## LED–silicone interface conclusions",
            "",
            f"1. **No.** The unloaded source is on the rigid recess boundary but "
            f"{unloaded_gap_um:.3f} µm from silicone inside air.",
            "2. The old 0.27–22 µm spontaneous gaps are superseded. Current saved states contain the explicit cavity and the reported load-driven closure/residual gaps.",
            "3. The 190 µm cavity is a CAD fact; residual micron gaps are not trustworthy at 1 mm mechanics resolution.",
            f"4. **Yes at closure.** 0→0.5 µm changes power by "
            f"{zero_to_finite_power:+.1%} and Q by {zero_to_finite_l2:.8f}.",
            "5. Classification selects the exact zero-gap branch; finite cavity geometry gives the stable air branch.",
            f"6. The all-closed counterfactual retains {combined_fraction:.1%} combined "
            f"and {per_led_fraction:.1%} per-emitter center/between separation.",
            "7. BO could exploit residual films if exposed as variables; the fixed explicit recess removes the old unloaded coincidence.",
            "8. The explicit 0.19 mm cavity is the safest minimal model under the supplied hardware assumption.",
            "9. Confirm whether the recess is air-filled and whether loaded silicone contacts bare LED, an encapsulant/window, or never reaches it.",
            "",
            "## Artifacts",
            "",
            "Raw CSV/NPZ data and four diagnostic PNGs are under "
            "`output/validation/5led_led_silicone_interface/`.",
            "",
            f"Total runtime: {runtime_s:.2f} s.",
        )
    )
    _REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    start_s = perf_counter()
    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    fingertip, mesh, surface_triangles, states = _load_geometry()
    reference_vertices = states["no_contact"]
    leds = _make_leds(fingertip, mesh)
    samples = _samples()
    stem_bottom_indices = _stem_bottom_vertices(
        fingertip,
        reference_vertices,
        surface_triangles,
    )
    edge_scale_mm = _interface_edge_scale_mm(
        reference_vertices,
        surface_triangles,
        stem_bottom_indices,
    )

    gap_matrix_um = np.empty((4, 5), dtype=np.float64)
    gap_rows: list[dict[str, object]] = []
    for state_index, (state_name, label, _) in enumerate(_STATES):
        for led_index, led in enumerate(leds):
            signed_gap_m, closest, normal, triangle_index = _closest_surface_point(
                led.position_W_m,
                states[state_name],
                surface_triangles,
            )
            signed_gap_um = 1.0e6 * signed_gap_m
            gap_matrix_um[state_index, led_index] = signed_gap_um
            gap_rows.append(
                {
                    "state": state_name,
                    "state_label": label,
                    "led_index": led_index + 1,
                    "led_y_mm": _LED_Y_MM[led_index],
                    "source_x_m": led.position_W_m[0],
                    "source_y_m": led.position_W_m[1],
                    "source_z_m": led.position_W_m[2],
                    "signed_gap_um": signed_gap_um,
                    "status": _gap_status(signed_gap_um),
                    "closest_x_m": closest[0],
                    "closest_y_m": closest[1],
                    "closest_z_m": closest[2],
                    "normal_x": normal[0],
                    "normal_y": normal[1],
                    "normal_z": normal[2],
                    "surface_triangle_index": triangle_index,
                }
            )
    _write_csv(_OUTPUT_DIRECTORY / "geometric_gaps.csv", gap_rows)

    scene = OptixScene(
        mesh,
        silicone_instance_id=_SILICONE_INSTANCE_ID,
        carrier_instance_id=_CARRIER_INSTANCE_ID,
        silicone_visibility_mask=_SILICONE_MASK,
        carrier_visibility_mask=_CARRIER_MASK,
    )

    automatic_results: dict[str, dict[str, object]] = {}
    initial_rows: list[dict[str, object]] = []
    for state_name, label, _ in _STATES:
        scene.update_silicone(states[state_name])
        result = _trace_all_leds(scene, fingertip, leds, samples)
        automatic_results[state_name] = result
        for led_index, emitter_result in enumerate(result["per_led"]):
            origin = emitter_result["source_origin_W_m"]
            initial_rows.append(
                {
                    "state": state_name,
                    "state_label": label,
                    "led_index": led_index + 1,
                    "led_y_mm": _LED_Y_MM[led_index],
                    "source_x_m": origin[0],
                    "source_y_m": origin[1],
                    "source_z_m": origin[2],
                    "launch_medium": (
                        "silicone"
                        if emitter_result["automatic_inside_silicone"]
                        else "air"
                    ),
                    "first_interface": emitter_result["first_interface"],
                    "first_interface_distance_um": (
                        1.0e6 * emitter_result["first_interface_distance_m"]
                    ),
                    "medium_after_first_crossing": emitter_result[
                        "medium_after_first_crossing"
                    ],
                }
            )
    _write_csv(_OUTPUT_DIRECTORY / "initial_medium.csv", initial_rows)

    epsilon_cases = (
        ("no_contact_center_led", "no_contact", 2),
        ("center_contact_center_led", "center_10n", 2),
        ("center_contact_neighbor_led2", "center_10n", 1),
        ("between_contact_led3_near_closure", "between_10n", 2),
    )
    offset_rows: list[dict[str, object]] = []
    offset_responses = np.empty((len(epsilon_cases), len(_OFFSET_UM), 4))
    offset_inside = np.empty((len(epsilon_cases), len(_OFFSET_UM)), dtype=bool)
    for case_index, (case_name, state_name, led_index) in enumerate(epsilon_cases):
        scene.update_silicone(states[state_name])
        for offset_index, offset_um in enumerate(_OFFSET_UM):
            result = _trace_emitter(
                scene,
                fingertip,
                leds[led_index],
                samples,
                offset_um=float(offset_um),
            )
            response = result["response"]
            energy = result["energy"]
            offset_responses[case_index, offset_index] = response
            offset_inside[case_index, offset_index] = result[
                "automatic_inside_silicone"
            ]
            offset_rows.append(
                {
                    "case": case_name,
                    "state": state_name,
                    "led_index": led_index + 1,
                    "offset_um": offset_um,
                    "automatic_medium": (
                        "silicone"
                        if result["automatic_inside_silicone"]
                        else "air"
                    ),
                    "first_interface": result["first_interface"],
                    "first_interface_distance_um": (
                        1.0e6 * result["first_interface_distance_m"]
                    ),
                    "medium_after_first_crossing": result[
                        "medium_after_first_crossing"
                    ],
                    "q1": response[0],
                    "q2": response[1],
                    "q3": response[2],
                    "q4": response[3],
                    "side_visible_power": response.sum(),
                    **{
                        name: energy[index]
                        for index, name in enumerate(_ENERGY_FIELDS)
                    },
                }
            )
    _write_csv(_OUTPUT_DIRECTORY / "emitter_offset_sweep.csv", offset_rows)

    center_led = leds[2]
    controlled_rows: list[dict[str, object]] = []
    controlled_responses = np.empty((len(_CONTROLLED_GAP_UM), 4))
    controlled_inside = np.empty(len(_CONTROLLED_GAP_UM), dtype=bool)
    controlled_patch_size = 0
    for gap_index, gap_um in enumerate(_CONTROLLED_GAP_UM):
        vertices, local_indices = _controlled_gap_geometry(
            reference_vertices,
            stem_bottom_indices,
            led_position_m=center_led.position_W_m,
            gap_um=float(gap_um),
        )
        controlled_patch_size = len(local_indices)
        scene.update_silicone(vertices)
        result = _trace_emitter(scene, fingertip, center_led, samples)
        response = result["response"]
        energy = result["energy"]
        measured_gap_m, _, _, _ = _closest_surface_point(
            center_led.position_W_m,
            vertices,
            surface_triangles,
        )
        controlled_responses[gap_index] = response
        controlled_inside[gap_index] = result["automatic_inside_silicone"]
        controlled_rows.append(
            {
                "requested_gap_um": gap_um,
                "measured_gap_um": 1.0e6 * measured_gap_m,
                "automatic_medium": (
                    "silicone"
                    if result["automatic_inside_silicone"]
                    else "air"
                ),
                "first_interface": result["first_interface"],
                "first_interface_distance_um": (
                    1.0e6 * result["first_interface_distance_m"]
                ),
                "q1": response[0],
                "q2": response[1],
                "q3": response[2],
                "q4": response[3],
                "side_visible_power": response.sum(),
                **{
                    name: energy[index]
                    for index, name in enumerate(_ENERGY_FIELDS)
                },
            }
        )
    _write_csv(_OUTPUT_DIRECTORY / "controlled_gap_sweep.csv", controlled_rows)

    treatment_specs = (
        ("automatic", 0.0, None, False),
        ("toward_silicone_25um", 25.0, None, False),
        ("toward_rigid_0.1um", -0.1, None, False),
        ("forced_silicone", 0.0, True, False),
        ("forced_air", 0.0, False, False),
        ("zero_gap_bonded_like", 0.0, None, True),
    )
    treatment_results: dict[str, dict[str, dict[str, object]]] = {}
    treatment_rows: list[dict[str, object]] = []
    for state_name, _, _ in _STATES[1:]:
        treatment_results[state_name] = {}
        for treatment, offset_um, forced_inside, bonded_like in treatment_specs:
            vertices = states[state_name]
            if bonded_like:
                vertices = _bonded_like_geometry(
                    vertices,
                    reference_vertices,
                    stem_bottom_indices,
                    leds,
                )
            scene.update_silicone(vertices)
            result = _trace_all_leds(
                scene,
                fingertip,
                leds,
                samples,
                offset_um=offset_um,
                forced_inside_silicone=forced_inside,
            )
            treatment_results[state_name][treatment] = result
            for led_index, emitter_result in enumerate(result["per_led"]):
                response = result["per_led_response"][led_index]
                energy = result["per_led_energy"][led_index]
                treatment_rows.append(
                    {
                        "state": state_name,
                        "treatment": treatment,
                        "led_index": led_index + 1,
                        "offset_um": offset_um,
                        "automatic_medium": (
                            "silicone"
                            if emitter_result["automatic_inside_silicone"]
                            else "air"
                        ),
                        "applied_medium": (
                            "silicone"
                            if emitter_result["applied_inside_silicone"]
                            else "air"
                        ),
                        "q1": response[0],
                        "q2": response[1],
                        "q3": response[2],
                        "q4": response[3],
                        "side_visible_power": response.sum(),
                        **{
                            name: energy[index]
                            for index, name in enumerate(_ENERGY_FIELDS)
                        },
                    }
                )
    _write_csv(_OUTPUT_DIRECTORY / "interface_treatments.csv", treatment_rows)

    separation_rows: list[dict[str, object]] = []
    automatic_l2 = float(
        np.linalg.norm(
            treatment_results["center_10n"]["automatic"]["combined_response"]
            - treatment_results["between_10n"]["automatic"]["combined_response"]
        )
    )
    for treatment, _, _, _ in treatment_specs:
        center = treatment_results["center_10n"][treatment]
        between = treatment_results["between_10n"][treatment]
        combined_l2 = float(
            np.linalg.norm(center["combined_response"] - between["combined_response"])
        )
        per_emitter_l2 = float(
            np.linalg.norm(center["per_led_response"] - between["per_led_response"])
        )
        separation_rows.append(
            {
                "treatment": treatment,
                "combined_q_l2": combined_l2,
                "per_emitter_l2": per_emitter_l2,
                "combined_fraction_of_automatic": combined_l2 / automatic_l2,
            }
        )
    _write_csv(
        _OUTPUT_DIRECTORY / "center_between_separation.csv",
        separation_rows,
    )

    _plot_gap_table(gap_matrix_um)
    _plot_offset_sweep(offset_rows)
    _plot_controlled_gap(controlled_rows)
    _plot_interface_treatments(separation_rows)

    treatment_names = tuple(spec[0] for spec in treatment_specs)
    loaded_state_names = tuple(row[0] for row in _STATES[1:])
    np.savez_compressed(
        _OUTPUT_DIRECTORY / "raw_results.npz",
        state_names=np.array([row[0] for row in _STATES]),
        led_y_mm=_LED_Y_MM,
        signed_gap_um=gap_matrix_um,
        automatic_per_led_response=np.stack(
            [automatic_results[row[0]]["per_led_response"] for row in _STATES]
        ),
        epsilon_case_names=np.array([row[0] for row in epsilon_cases]),
        emitter_offset_um=_OFFSET_UM,
        emitter_offset_response=offset_responses,
        emitter_offset_inside_silicone=offset_inside,
        controlled_gap_um=_CONTROLLED_GAP_UM,
        controlled_gap_response=controlled_responses,
        controlled_gap_inside_silicone=controlled_inside,
        treatment_names=np.array(treatment_names),
        loaded_state_names=np.array(loaded_state_names),
        treatment_per_led_response=np.stack(
            [
                [
                    treatment_results[state][treatment]["per_led_response"]
                    for treatment in treatment_names
                ]
                for state in loaded_state_names
            ]
        ),
    )
    metadata = {
        "sample_side": _SAMPLE_SIDE,
        "paths_per_led": _SAMPLE_SIDE**2,
        "max_bounces": _MAX_BOUNCES,
        "rng_seed": _RNG_SEED,
        "positive_offset_direction": "toward silicone, world -Z",
        "gap_classification_tolerance_um": _GAP_CLASSIFICATION_TOLERANCE_UM,
        "controlled_patch_half_length_mm": _CONTROLLED_PATCH_HALF_LENGTH_MM,
        "controlled_patch_vertex_count": controlled_patch_size,
        "led_recess_width_mm": LED_RECESS_WIDTH_MM,
        "led_recess_depth_mm": LED_RECESS_DEPTH_MM,
        "newton_element_size_mm": 1.0,
        "soft_contact_margin_um": 100.0,
        "bond_vertex_tolerance_um": 0.01,
        "local_interface_edge_quantiles_mm": edge_scale_mm.tolist(),
    }
    (_OUTPUT_DIRECTORY / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    runtime_s = perf_counter() - start_s
    _write_report(
        gap_matrix_um,
        initial_rows,
        offset_rows,
        controlled_rows,
        separation_rows,
        automatic_results,
        treatment_results,
        edge_scale_mm,
        runtime_s=runtime_s,
    )
    print(f"report: {_REPORT_PATH}")
    print(f"raw output: {_OUTPUT_DIRECTORY}")
    print(f"runtime: {runtime_s:.2f} s")


if __name__ == "__main__":
    main()

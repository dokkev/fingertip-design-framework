"""Validate full five-LED OptiX transport using saved Newton states."""

from __future__ import annotations

import csv
import json
from itertools import combinations
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
from matplotlib.patches import Rectangle  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402


_MECHANICS_DIRECTORY = Path("output/validation/5led_newton")
_OUTPUT_DIRECTORY = Path("output/validation/5led_optix")
_REPORT_PATH = Path("output/validation/5led_optix_validation.md")

_STATE_SOURCES = (
    ("no_contact", None),
    ("center_10n", "center_led_10n.npz"),
    ("between_10n", "between_leds_10n.npz"),
    ("distal_10n", "distal_led_10n.npz"),
)
_STATE_LABELS = (
    "No contact",
    "Center Y=0 mm",
    "Between Y=+5.5 mm",
    "Distal Y=+22 mm",
)
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

_SILICONE_INSTANCE_ID = 1
_CARRIER_INSTANCE_ID = 2
_SILICONE_MASK = 0x01
_CARRIER_MASK = 0x02
_ALL_MASK = _SILICONE_MASK | _CARRIER_MASK
_CARRIER_ALBEDO = 0.7
_MAX_BOUNCES = 24
_RNG_SEED = 20260823
_PRODUCTION_SAMPLE_SIDE = 256
_CONVERGENCE_SAMPLE_SIDES = (128, 256, 384)
_CLOSURE_TOLERANCE = 1.0e-12


def _load_mesh_and_states() -> tuple[
    Fingertip,
    Fingertip5LEDMesh,
    dict[str, np.ndarray],
    float,
]:
    reference_path = _MECHANICS_DIRECTORY / "reference_mesh.npz"
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"missing saved Newton reference mesh: {reference_path}"
        )

    mesh_start_s = perf_counter()
    fingertip = Fingertip(FingertipParameters())
    mesh = make_fingertip_5led_mesh(fingertip, element_size_mm=1.0)
    mesh_build_time_s = perf_counter() - mesh_start_s

    with np.load(reference_path) as saved:
        saved_vertices = np.asarray(saved["silicone_vertices_m"])
        saved_surface = np.asarray(saved["silicone_surface_triangles"])
        saved_carrier_vertices = np.asarray(saved["carrier_vertices_m"])
        saved_carrier_triangles = np.asarray(saved["carrier_triangles"])
        saved_bonded = np.asarray(saved["bonded_vertex_indices"])
        saved_led_centers = np.asarray(saved["led_centers_m"])

    mesh_vertices = np.asarray(mesh.silicone.vertices)
    mesh_surface = np.asarray(
        mesh.silicone.surface_tri_indices,
        dtype=np.int32,
    ).reshape(-1, 3)
    mesh_carrier_vertices = np.asarray(mesh.carrier.vertices)
    mesh_carrier_triangles = np.asarray(
        mesh.carrier.indices,
        dtype=np.int32,
    ).reshape(-1, 3)
    if not np.allclose(mesh_vertices, saved_vertices, rtol=0.0, atol=1.0e-10):
        raise RuntimeError(
            "saved Newton vertices do not match the current full-finger mesh"
        )
    if not np.array_equal(mesh_surface, saved_surface):
        raise RuntimeError(
            "saved Newton surface topology does not match the current mesh"
        )
    if not np.allclose(
        mesh_carrier_vertices,
        saved_carrier_vertices,
        rtol=0.0,
        atol=1.0e-10,
    ) or not np.array_equal(mesh_carrier_triangles, saved_carrier_triangles):
        raise RuntimeError("saved Newton carrier does not match the current mesh")
    if not np.array_equal(mesh.bonded_vertex_indices, saved_bonded):
        raise RuntimeError("saved Newton bond indices do not match the current mesh")
    if not np.allclose(
        mesh.led_centers_m,
        saved_led_centers,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("saved LED centers do not match the current mesh")

    states = {"no_contact": np.asarray(saved_vertices, dtype=np.float64)}
    for state_name, filename in _STATE_SOURCES[1:]:
        path = _MECHANICS_DIRECTORY / str(filename)
        if not path.is_file():
            raise FileNotFoundError(f"missing saved Newton state: {path}")
        with np.load(path) as saved:
            vertices = np.asarray(saved["deformed_vertices_m"], dtype=np.float64)
        if vertices.shape != mesh_vertices.shape or not np.all(
            np.isfinite(vertices)
        ):
            raise RuntimeError(f"{state_name} has invalid saved silicone vertices")
        states[state_name] = vertices
    return fingertip, mesh, states, mesh_build_time_s


def _make_leds(
    fingertip: Fingertip,
    mesh: Fingertip5LEDMesh,
) -> tuple[LED, ...]:
    centers = np.asarray(mesh.led_centers_m, dtype=np.float64)
    expected_y_m = 1.0e-3 * np.array((-22.0, -11.0, 0.0, 11.0, 22.0))
    if centers.shape != (5, 3) or not np.allclose(
        centers[:, 1],
        expected_y_m,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("full-finger mesh does not define the five LED centers")
    if len(np.unique(centers, axis=0)) != 5:
        raise RuntimeError("LED centers contain a duplicate")
    expected_z_m = 1.0e-3 * (
        -fingertip.parameters.geometry.stem_height_mm
        + LED_RECESS_DEPTH_MM
    )
    if not np.allclose(centers[:, 2], expected_z_m, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("LED centers are not on the carrier recess floor")

    normal_W = np.array((0.0, 0.0, -1.0), dtype=np.float64)
    return tuple(
        LED(
            position_W_m=center,
            normal_W=normal_W,
            parameters=fingertip.parameters.led,
        )
        for center in centers
    )


def _optical_samples(
    sample_side: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coordinate = (np.arange(sample_side, dtype=np.float64) + 0.5) / sample_side
    u1, u2 = np.meshgrid(coordinate, coordinate, indexing="ij")
    ray_count = sample_side * sample_side
    rng = np.random.default_rng(_RNG_SEED)
    shape = (_MAX_BOUNCES, ray_count)
    return (
        u1.ravel(),
        u2.ravel(),
        rng.random(shape),
        rng.random(shape),
        rng.random(shape),
    )


def _energy_row(paths) -> np.ndarray:
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


def _validate_distal_closure(
    scene: OptixScene,
    *,
    reference_state: bool,
) -> tuple[float, np.ndarray]:
    origin = np.array(((4.8e-3, 0.0, -3.0e-3),), dtype=np.float64)
    distal_direction = np.array(((0.0, 1.0, 0.0),), dtype=np.float64)
    proximal_direction = np.array(((0.0, -1.0, 0.0),), dtype=np.float64)
    distal_hit = scene.trace_closest(
        origin,
        distal_direction,
        mask=_SILICONE_MASK,
    )[0]
    proximal_hit = scene.trace_closest(
        origin,
        proximal_direction,
        mask=_SILICONE_MASK,
    )[0]
    if not distal_hit["hit"]:
        raise RuntimeError("longitudinal void ray missed the distal silicone cap")
    if proximal_hit["hit"]:
        raise RuntimeError("proximal longitudinal void opening is unexpectedly closed")
    distance_m = float(distal_hit["t"])
    normal_W = np.asarray(distal_hit["normal_W"], dtype=np.float64)
    if reference_state and not np.isclose(
        distance_m,
        27.5e-3,
        rtol=0.0,
        atol=2.0e-7,
    ):
        raise RuntimeError("distal silicone cap is not at Y=+27.5 mm")
    if not np.all(np.isfinite(normal_W)) or float(
        np.dot(normal_W, distal_direction[0])
    ) >= 0.0:
        raise RuntimeError("distal cap does not expose an air-to-silicone boundary")
    return distance_m, normal_W


def _update_scene(
    scene: OptixScene,
    vertices_m: np.ndarray,
    *,
    reference_state: bool,
) -> tuple[float, float, np.ndarray]:
    update_start_s = perf_counter()
    scene.update_silicone(vertices_m)
    cap_distance_m, cap_normal_W = _validate_distal_closure(
        scene,
        reference_state=reference_state,
    )
    return perf_counter() - update_start_s, cap_distance_m, cap_normal_W


def _trace_state(
    scene: OptixScene,
    fingertip: Fingertip,
    leds: tuple[LED, ...],
    *,
    state_name: str,
    sample_side: int,
    persist_escaped_rays: bool,
) -> dict[str, np.ndarray | float | int | list[bool]]:
    u1, u2, dielectric_u, carrier_u1, carrier_u2 = _optical_samples(sample_side)
    per_led_response = np.empty((len(leds), 4), dtype=np.float64)
    per_led_energy = np.empty((len(leds), len(_ENERGY_FIELDS)), dtype=np.float64)
    per_led_trace_time_s = np.empty(len(leds), dtype=np.float64)
    escaped_ray_counts = np.empty(len(leds), dtype=np.int64)
    remaining_ray_counts = np.empty(len(leds), dtype=np.int64)
    source_hit_distance_m = np.empty(len(leds), dtype=np.float64)
    source_normal_projection = np.empty(len(leds), dtype=np.float64)
    source_inside = []

    for led_index, led in enumerate(leds):
        emission = emit_from_stem_boundary(
            scene,
            led,
            u1,
            u2,
            carrier_mask=_CARRIER_MASK,
        )
        if not np.isclose(
            emission["power"].sum(),
            led.parameters.normalized_power,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(f"LED {led_index + 1} emitted the wrong power")
        source_hit = scene.trace_closest(
            emission["origin_W_m"][:1],
            led.normal_W[None, :],
            mask=_SILICONE_MASK,
        )[0]
        if not source_hit["hit"]:
            raise RuntimeError(f"LED {led_index + 1} normal does not reach silicone")
        source_hit_distance_m[led_index] = float(source_hit["t"])
        source_normal_projection[led_index] = float(
            np.dot(source_hit["normal_W"], led.normal_W)
        )
        inside_silicone = source_inside_silicone(
            scene,
            led,
            emission,
            silicone_mask=_SILICONE_MASK,
        )
        if inside_silicone != (source_normal_projection[led_index] > 0.0):
            raise RuntimeError("source-medium diagnostics disagree")
        source_inside.append(inside_silicone)
        trace_start_s = perf_counter()
        paths = trace_bounded_paths(
            scene,
            emission["origin_W_m"],
            emission["direction_W"],
            emission["power"],
            inside_silicone=inside_silicone,
            n_air=1.0,
            n_silicone=fingertip.parameters.optics.refractive_index,
            extinction_coefficient_m_inv=(
                fingertip.parameters.optics.extinction_coefficient_m_inv
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
        per_led_trace_time_s[led_index] = perf_counter() - trace_start_s
        response = side_view_observation(paths.escaped_rays, fingertip=fingertip)
        energy = _energy_row(paths)
        if not np.all(np.isfinite(response)) or np.any(response < 0.0):
            raise RuntimeError(f"{state_name} LED {led_index + 1} response is invalid")
        if not np.all(np.isfinite(energy[:-1])) or np.any(energy[:-1] < 0.0):
            raise RuntimeError(f"{state_name} LED {led_index + 1} energy is invalid")
        if abs(paths.closure_error) > _CLOSURE_TOLERANCE:
            raise RuntimeError(
                f"{state_name} LED {led_index + 1} energy does not close"
            )
        if response.sum() > paths.escaped_power + _CLOSURE_TOLERANCE:
            raise RuntimeError("side-visible power exceeds total escaped power")
        ray_count = sample_side * sample_side
        if paths.escaped_ray_count > ray_count or paths.remaining_ray_count > ray_count:
            raise RuntimeError("bounded path count exceeded the emitted ray count")

        per_led_response[led_index] = response
        per_led_energy[led_index] = energy
        escaped_ray_counts[led_index] = paths.escaped_ray_count
        remaining_ray_counts[led_index] = paths.remaining_ray_count
        if persist_escaped_rays:
            np.savez_compressed(
                _OUTPUT_DIRECTORY
                / f"escaped_{state_name}_led{led_index + 1}.npz",
                escaped_rays=paths.escaped_rays,
                led_center_W_m=led.position_W_m,
                led_normal_W=led.normal_W,
                sample_side=sample_side,
                max_bounces=_MAX_BOUNCES,
            )
        del paths, emission

    combined_response = per_led_response.sum(axis=0)
    combined_energy = per_led_energy.sum(axis=0)
    return {
        "per_led_response": per_led_response,
        "combined_response": combined_response,
        "per_led_energy": per_led_energy,
        "combined_energy": combined_energy,
        "per_led_trace_time_s": per_led_trace_time_s,
        "trace_time_s": float(per_led_trace_time_s.sum()),
        "escaped_ray_counts": escaped_ray_counts,
        "remaining_ray_counts": remaining_ray_counts,
        "source_inside_silicone": source_inside,
        "source_hit_distance_m": source_hit_distance_m,
        "source_normal_projection": source_normal_projection,
    }


def _write_tables(
    state_names: tuple[str, ...],
    results: dict[str, dict[str, object]],
    pairwise_rows: list[dict[str, float | str]],
    convergence_rows: list[dict[str, float | int | str]],
) -> None:
    with (_OUTPUT_DIRECTORY / "state_summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        fields = (
            "state",
            "q1",
            "q2",
            "q3",
            "q4",
            "side_visible_power",
            *_ENERGY_FIELDS,
            "gas_ias_update_time_s",
            "trace_time_s",
            "total_optical_time_s",
            "distal_cap_hit_y_mm",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for state_name in state_names:
            result = results[state_name]
            response = result["combined_response"]
            energy = result["combined_energy"]
            row: dict[str, float | str] = {
                "state": state_name,
                "q1": response[0],
                "q2": response[1],
                "q3": response[2],
                "q4": response[3],
                "side_visible_power": response.sum(),
                "gas_ias_update_time_s": result["update_time_s"],
                "trace_time_s": result["trace_time_s"],
                "total_optical_time_s": (
                    result["update_time_s"] + result["trace_time_s"]
                ),
                "distal_cap_hit_y_mm": 1.0e3 * result["cap_distance_m"],
            }
            row.update(
                {
                    name: energy[index]
                    for index, name in enumerate(_ENERGY_FIELDS)
                }
            )
            writer.writerow(row)

    with (_OUTPUT_DIRECTORY / "per_emitter_response.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        fields = (
            "state",
            "led_index",
            "led_y_mm",
            "q1",
            "q2",
            "q3",
            "q4",
            "side_visible_power",
            *_ENERGY_FIELDS,
            "trace_time_s",
            "escaped_ray_count",
            "remaining_ray_count",
            "source_inside_silicone",
            "source_first_hit_distance_um",
            "source_normal_projection",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for state_name in state_names:
            result = results[state_name]
            for led_index in range(5):
                response = result["per_led_response"][led_index]
                energy = result["per_led_energy"][led_index]
                row = {
                    "state": state_name,
                    "led_index": led_index + 1,
                    "led_y_mm": (-22.0, -11.0, 0.0, 11.0, 22.0)[led_index],
                    "q1": response[0],
                    "q2": response[1],
                    "q3": response[2],
                    "q4": response[3],
                    "side_visible_power": response.sum(),
                    "trace_time_s": result["per_led_trace_time_s"][led_index],
                    "escaped_ray_count": result["escaped_ray_counts"][led_index],
                    "remaining_ray_count": result["remaining_ray_counts"][led_index],
                    "source_inside_silicone": result[
                        "source_inside_silicone"
                    ][led_index],
                    "source_first_hit_distance_um": (
                        1.0e6 * result["source_hit_distance_m"][led_index]
                    ),
                    "source_normal_projection": result[
                        "source_normal_projection"
                    ][led_index],
                }
                row.update(
                    {
                        name: energy[index]
                        for index, name in enumerate(_ENERGY_FIELDS)
                    }
                )
                writer.writerow(row)

    for filename, rows in (
        ("pairwise_distances.csv", pairwise_rows),
        ("sample_convergence.csv", convergence_rows),
    ):
        with (_OUTPUT_DIRECTORY / filename).open(
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _plot_geometry(mesh: Fingertip5LEDMesh) -> None:
    fingertip = mesh.fingertip
    silicone_vertices_mm = 1.0e3 * np.asarray(mesh.silicone.vertices)
    silicone_triangles = np.asarray(
        mesh.silicone.surface_tri_indices,
        dtype=np.int32,
    ).reshape(-1, 3)
    carrier_vertices_mm = 1.0e3 * np.asarray(mesh.carrier.vertices)
    carrier_triangles = np.asarray(
        mesh.carrier.indices,
        dtype=np.int32,
    ).reshape(-1, 3)
    led_centers_mm = 1.0e3 * mesh.led_centers_m
    geometry = fingertip.parameters.geometry
    silicone = fingertip.silicone
    tip_z_mm = silicone.ellipse_center_z_mm - silicone.ellipse_radius_z_mm

    figure = plt.figure(figsize=(15.0, 6.0), constrained_layout=True)
    layout_axis = figure.add_subplot(1, 2, 1)
    layout_axis.add_patch(
        Rectangle(
            (-27.5, tip_z_mm),
            55.0,
            silicone.cavity_bottom_z_mm - tip_z_mm,
            color="#8ecae6",
            alpha=0.75,
            label="continuous silicone",
        )
    )
    layout_axis.add_patch(
        Rectangle(
            (-27.5, silicone.cavity_bottom_z_mm),
            55.0,
            -silicone.cavity_bottom_z_mm,
            facecolor="#fff3bf",
            edgecolor="#e67700",
            hatch="//",
            label="continuous void",
        )
    )
    layout_axis.add_patch(
        Rectangle(
            (-27.5, 0.0),
            55.0,
            geometry.link_thickness_mm,
            color="#6c757d",
            alpha=0.85,
            label="continuous carrier above side void",
        )
    )
    layout_axis.add_patch(
        Rectangle(
            (27.5, tip_z_mm),
            5.0,
            silicone.bond_top_z_mm - tip_z_mm,
            color="#8ecae6",
            alpha=0.75,
            label="solid distal closure",
        )
    )
    layout_axis.add_patch(
        Rectangle(
            (27.5, silicone.bond_top_z_mm),
            5.0,
            geometry.link_thickness_mm - silicone.bond_top_z_mm,
            color="#6c757d",
            alpha=0.85,
            label="distal dorsal carrier",
        )
    )
    layout_axis.scatter(
        led_centers_mm[:, 1],
        led_centers_mm[:, 2],
        color="#38b000",
        edgecolor="#1b4332",
        s=60.0,
        zorder=4,
    )
    for led_index, center in enumerate(led_centers_mm, start=1):
        layout_axis.annotate(
            f"LED{led_index}",
            (center[1], center[2]),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    layout_axis.annotate(
        "+Y side view",
        xy=(35.0, 7.0),
        xytext=(23.0, 7.0),
        arrowprops={"arrowstyle": "->", "color": "tab:blue"},
        color="tab:blue",
    )
    layout_axis.axvline(27.5, color="tab:red", linestyle="--")
    layout_axis.set_xlim(-30.0, 38.0)
    layout_axis.set_ylim(tip_z_mm - 1.0, geometry.link_thickness_mm + 2.0)
    layout_axis.set_xlabel("Y [mm]")
    layout_axis.set_ylabel("Z [mm]")
    layout_axis.set_title(
        "Longitudinal side-void slice (LED centers overlaid)"
    )
    layout_axis.grid(alpha=0.2)
    layout_axis.legend(fontsize=7, loc="lower left")

    mesh_axis = figure.add_subplot(1, 2, 2, projection="3d")
    mesh_axis.add_collection3d(
        Poly3DCollection(
            silicone_vertices_mm[silicone_triangles],
            facecolor="#8ecae6",
            edgecolor="none",
            alpha=0.12,
        )
    )
    mesh_axis.add_collection3d(
        Poly3DCollection(
            carrier_vertices_mm[carrier_triangles],
            facecolor="#6c757d",
            edgecolor="none",
            alpha=0.82,
        )
    )
    mesh_axis.scatter(
        led_centers_mm[:, 0],
        led_centers_mm[:, 1],
        led_centers_mm[:, 2],
        color="#38b000",
        edgecolor="#1b4332",
        s=35.0,
        depthshade=False,
    )
    mesh_axis.set_xlim(-16.0, 16.0)
    mesh_axis.set_ylim(-30.0, 35.0)
    mesh_axis.set_zlim(
        silicone_vertices_mm[:, 2].min() - 1.0,
        carrier_vertices_mm[:, 2].max() + 1.0,
    )
    mesh_axis.set_box_aspect((32.0, 65.0, 27.0))
    mesh_axis.view_init(elev=21.0, azim=-42.0)
    mesh_axis.set_xlabel("X [mm]")
    mesh_axis.set_ylabel("Y [mm]")
    mesh_axis.set_zlabel("Z [mm]")
    mesh_axis.set_title("Actual OptiX silicone/carrier meshes")
    figure.suptitle("Full five-LED optical geometry")
    figure.savefig(_OUTPUT_DIRECTORY / "optical_geometry.png", dpi=180)
    plt.close(figure)


def _quadrant_image(response: np.ndarray) -> np.ndarray:
    return np.array(
        ((response[1], response[0]), (response[2], response[3])),
        dtype=np.float64,
    )


def _plot_responses(
    state_names: tuple[str, ...],
    results: dict[str, dict[str, object]],
) -> None:
    responses = np.stack(
        [results[state_name]["combined_response"] for state_name in state_names]
    )
    maximum = float(responses.max())
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 8.0), constrained_layout=True)
    for axis, state_name, label, response in zip(
        axes.ravel(),
        state_names,
        _STATE_LABELS,
        responses,
        strict=True,
    ):
        image = axis.imshow(
            _quadrant_image(response),
            vmin=0.0,
            vmax=maximum,
            cmap="viridis",
        )
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    f"{_quadrant_image(response)[row, column]:.5f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )
        axis.set_xticks(())
        axis.set_yticks(())
        axis.set_title(f"{label}\nP(+Y)={response.sum():.6f}")
    figure.colorbar(image, ax=axes, label="combined modeled power")
    figure.suptitle("Five-LED +Y side-view quadrant response")
    figure.savefig(_OUTPUT_DIRECTORY / "combined_responses.png", dpi=180)
    plt.close(figure)

    deltas = responses[1:] - responses[0]
    absolute_maximum = float(np.abs(deltas).max())
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), constrained_layout=True)
    for axis, label, delta in zip(
        axes,
        _STATE_LABELS[1:],
        deltas,
        strict=True,
    ):
        image = axis.imshow(
            _quadrant_image(delta),
            vmin=-absolute_maximum,
            vmax=absolute_maximum,
            cmap="coolwarm",
        )
        axis.set_xticks(())
        axis.set_yticks(())
        axis.set_title(f"{label}\nΔP(+Y)={delta.sum():+.6f}")
    figure.colorbar(image, ax=axes, label="contact − no-contact power")
    figure.suptitle("Contact-induced five-LED response changes")
    figure.savefig(_OUTPUT_DIRECTORY / "contact_response_deltas.png", dpi=180)
    plt.close(figure)

    side_power = np.stack(
        [
            results[state_name]["per_led_response"].sum(axis=1)
            for state_name in state_names
        ]
    )
    figure, axis = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    image = axis.imshow(side_power, aspect="auto", cmap="magma")
    axis.set_xticks(range(5), ("LED1", "LED2", "LED3", "LED4", "LED5"))
    axis.set_yticks(range(4), _STATE_LABELS)
    for row in range(4):
        for column in range(5):
            axis.text(
                column,
                row,
                f"{side_power[row, column]:.4f}",
                ha="center",
                va="center",
                color="white",
                fontsize=7,
            )
    figure.colorbar(image, ax=axis, label="+Y side-visible power")
    axis.set_title("Per-emitter diagnostic contributions")
    figure.savefig(_OUTPUT_DIRECTORY / "per_emitter_side_power.png", dpi=180)
    plt.close(figure)

    center_matrix = results["center_10n"]["per_led_response"]
    between_matrix = results["between_10n"]["per_led_response"]
    difference = between_matrix - center_matrix
    limit = float(np.abs(difference).max())
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.6), constrained_layout=True)
    for axis, title, matrix, cmap, vmin, vmax in (
        (axes[0], "Center", center_matrix, "viridis", 0.0, maximum),
        (axes[1], "Between LEDs", between_matrix, "viridis", 0.0, maximum),
        (axes[2], "Between − center", difference, "coolwarm", -limit, limit),
    ):
        image = axis.imshow(
            matrix,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_xticks(range(4), ("Q1", "Q2", "Q3", "Q4"))
        axis.set_yticks(range(5), ("LED1", "LED2", "LED3", "LED4", "LED5"))
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.suptitle("Half-pitch optical signature: Y=0 versus Y=+5.5 mm")
    figure.savefig(_OUTPUT_DIRECTORY / "center_between_comparison.png", dpi=180)
    plt.close(figure)


def _write_report(
    fingertip: Fingertip,
    mesh: Fingertip5LEDMesh,
    state_names: tuple[str, ...],
    results: dict[str, dict[str, object]],
    pairwise_rows: list[dict[str, float | str]],
    convergence_rows: list[dict[str, float | int | str]],
    *,
    mesh_build_time_s: float,
    scene_build_time_s: float,
    deterministic_max_difference: float,
) -> None:
    optics = fingertip.parameters.optics
    no_contact = results["no_contact"]
    distal = results["distal_10n"]
    center_between = next(
        row
        for row in pairwise_rows
        if row["state_a"] == "center_10n" and row["state_b"] == "between_10n"
    )
    maximum_closure = max(
        abs(float(results[name]["combined_energy"][-1]))
        for name in state_names
    )
    production_convergence = [
        row
        for row in convergence_rows
        if int(row["sample_side"]) != _PRODUCTION_SAMPLE_SIDE
    ]
    maximum_convergence_l2 = max(
        float(row["combined_response_relative_l2"])
        for row in production_convergence
    )
    maximum_side_change = max(
        abs(float(row["side_visible_relative_change"]))
        for row in production_convergence
    )
    total_trace_time_s = sum(float(results[name]["trace_time_s"]) for name in state_names)
    no_contact_gap_um = 1.0e6 * np.asarray(
        no_contact["source_hit_distance_m"],
        dtype=np.float64,
    )
    closed_source_states = {
        name: tuple(
            index + 1
            for index, inside in enumerate(results[name]["source_inside_silicone"])
            if inside
        )
        for name in state_names[1:]
    }
    closure_summary = "; ".join(
        f"{name}: {', '.join(f'LED{index}' for index in indices) or 'none'}"
        for name, indices in closed_source_states.items()
    )

    lines = [
        "# Full 5-LED OptiX optical validation",
        "",
        "## Contract",
        "",
        f"- optical preset: `{optics.name}`, n={optics.refractive_index:g}, "
        f"extinction={optics.extinction_coefficient_m_inv:.6g} 1/m",
        f"- LED centers Y [mm]: `{(1.0e3 * mesh.led_centers_m[:, 1]).tolist()}`",
        f"- hardware stem recess: {LED_RECESS_WIDTH_MM:g} mm along Y × "
        f"{LED_RECESS_DEPTH_MM:g} mm deep at every LED",
        "- LED source: finite package-window emitter, normal `-Z`",
        f"- emitted power: {fingertip.parameters.led.normalized_power:g} modeled unit/LED, "
        f"{5.0 * fingertip.parameters.led.normalized_power:g} total",
        f"- samples: {_PRODUCTION_SAMPLE_SIDE**2:,} paths/LED, "
        f"{5 * _PRODUCTION_SAMPLE_SIDE**2:,} total paths/state",
        f"- bounded transport: {_MAX_BOUNCES} bounces, carrier albedo {_CARRIER_ALBEDO:g}",
        "- observation: current +Y side-view Q1–Q4 response; no camera/image plane exists",
        "- all five LEDs are on simultaneously; linear per-emitter traces are summed "
        "without renormalizing total power",
        "",
        "## Geometry and interface verification",
        "",
        f"- OptiX consumes the same {len(mesh.silicone.vertices)}-vertex silicone and "
        f"{len(mesh.carrier.vertices)}-vertex carrier meshes stored by Newton.",
        "- Saved reference vertices/topology, carrier, bond indices, and LED centers "
        "match the current `Fingertip5LEDMesh` exactly.",
        "- Five unique emitters use exact 11 mm pitch and spawn from their carrier "
        "recess floors; every emitter contributes finite unit power.",
        f"- With nominal `void_height_mm=0`, the unloaded LED-to-silicone "
        f"normal gaps are `{np.array2string(no_contact_gap_um, precision=3)}` µm "
        f"(expected {1.0e3 * LED_RECESS_DEPTH_MM:.1f} µm).",
        f"- The reference longitudinal void ray hits the distal air→silicone wall at "
        f"Y={1.0e3 * no_contact['cap_distance_m']:.6f} mm and exits proximally without a hit.",
        "",
        "## Source-boundary medium diagnostic",
        "",
        "The point sources lie on the recessed carrier floors, 0.19 mm above the "
        "unloaded silicone stem-bottom surface. `air` means that physical cavity "
        "remains open; `silicone` means deformation has closed the cavity and moved "
        "the silicone surface through the source plane.",
        "",
        "| state | LED1 | LED2 | LED3 | LED4 | LED5 | first-hit distance [µm] |",
        "|---|---|---|---|---|---|---|",
    ]
    for state_name, label in zip(state_names, _STATE_LABELS, strict=True):
        result = results[state_name]
        media = [
            "silicone" if inside else "air"
            for inside in result["source_inside_silicone"]
        ]
        distances = 1.0e6 * result["source_hit_distance_m"]
        lines.append(
            f"| {label} | {' | '.join(media)} | "
            f"{np.array2string(distances, precision=2)} |"
        )

    lines.extend(
        (
            "",
            "## Combined raw responses",
            "",
            "| state | Q1 | Q2 | Q3 | Q4 | +Y visible | escaped | carrier absorbed | bulk loss | closure | update [s] | trace [s] | optical total [s] |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for state_name, label in zip(state_names, _STATE_LABELS, strict=True):
        result = results[state_name]
        response = result["combined_response"]
        energy = result["combined_energy"]
        lines.append(
            f"| {label} | {response[0]:.8f} | {response[1]:.8f} | "
            f"{response[2]:.8f} | {response[3]:.8f} | {response.sum():.8f} | "
            f"{energy[1]:.8f} | {energy[2]:.8f} | {energy[3]:.8f} | "
            f"{energy[-1]:+.3e} | {result['update_time_s']:.4f} | "
            f"{result['trace_time_s']:.4f} | "
            f"{result['update_time_s'] + result['trace_time_s']:.4f} |"
        )

    lines.extend(
        (
            "",
            "Raw escaped-ray records for every state × emitter are stored separately. "
            "The four quadrant values are the existing receiver representation, while "
            "the 5×4 per-emitter matrices retain additional diagnostic structure.",
            "",
            "## Contact-induced combined deltas",
            "",
            "| state | ΔQ1 | ΔQ2 | ΔQ3 | ΔQ4 | Δ visible power | relative L2 vs no-contact |",
            "|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for state_name, label in zip(state_names[1:], _STATE_LABELS[1:], strict=True):
        delta = (
            results[state_name]["combined_response"]
            - no_contact["combined_response"]
        )
        relative_l2 = float(
            np.linalg.norm(delta) / np.linalg.norm(no_contact["combined_response"])
        )
        lines.append(
            f"| {label} | {delta[0]:+.8f} | {delta[1]:+.8f} | "
            f"{delta[2]:+.8f} | {delta[3]:+.8f} | {delta.sum():+.8f} | "
            f"{relative_l2:.6f} |"
        )

    lines.extend(
        (
            "",
            "## Pairwise raw-state distances",
            "",
            "Distances use modeled power directly. `per-emitter L2` flattens the "
            "diagnostic 5×4 matrix; no objective or tuned weighting is introduced.",
            "",
            "| state A | state B | combined Q L2 | per-emitter 5×4 L2 | per-emitted-power L2 |",
            "|---|---|---:|---:|---:|",
        )
    )
    for row in pairwise_rows:
        lines.append(
            f"| {row['state_a']} | {row['state_b']} | "
            f"{row['combined_quadrant_l2']:.8f} | "
            f"{row['per_emitter_l2']:.8f} | "
            f"{row['per_emitted_power_l2']:.8f} |"
        )

    lines.extend(
        (
            "",
            "## Sample-count sanity check",
            "",
            "| state | paths/LED | total paths | relative response L2 vs 65,536 | relative +Y power change | escaped-power change |",
            "|---|---:|---:|---:|---:|---:|",
        )
    )
    for row in convergence_rows:
        lines.append(
            f"| {row['state']} | {row['paths_per_led']} | {row['total_paths']} | "
            f"{row['combined_response_relative_l2']:.6f} | "
            f"{row['side_visible_relative_change']:+.6f} | "
            f"{row['escaped_power_relative_change']:+.6f} |"
        )

    lines.extend(
        (
            "",
            "## Full 5-LED optical conclusions",
            "",
            "1. All five LEDs are represented exactly once at the required positions, "
            "with the existing -Z Lambertian point-source convention and one modeled "
            "power unit per LED.",
            "2. Per-emitter matrices show distributed contributions from all five LEDs; "
            "they are diagnostic contributions, not five independent taxels.",
            f"3. Center and between-LED contact differ by combined-Q L2 "
            f"{center_between['combined_quadrant_l2']:.8f} and per-emitter 5×4 L2 "
            f"{center_between['per_emitter_l2']:.8f}. The mechanism distinguishes the "
            "two frozen states, but this is not a localization-accuracy claim.",
            f"4. Distal contact remains finite and energy-conserving; its +Y visible "
            f"power is {distal['combined_response'].sum():.8f} versus "
            f"{no_contact['combined_response'].sum():.8f} unloaded. The distal wall "
            "remains a valid air→silicone interface rather than an open-cavity artifact.",
            "5. Per-emitter response is available without renderer redesign because "
            "transport is linear; summing the five traces gives the simultaneous field.",
            f"6. Across the 16,384/65,536/147,456 paths-per-LED check, maximum relative "
            f"combined-response L2 was {maximum_convergence_l2:.3%} and maximum +Y "
            f"power change was {maximum_side_change:.3%}. See the table before freezing "
            "a later objective's ray budget.",
            f"7. Mesh setup took {mesh_build_time_s:.3f} s and OptiX scene build "
            f"{scene_build_time_s:.3f} s. Production traces for all four states took "
            f"{total_trace_time_s:.3f} s total; individual timings are in the raw table. "
            "Newton remained the larger per-state cost (~14–16 s), excluding one-time "
            "OptiX compilation/build.",
            f"8. Maximum energy-closure error was {maximum_closure:.3e}; deterministic "
            f"rerun maximum response/ledger difference was "
            f"{deterministic_max_difference:.3e}. No ray-tracing or distal-interface "
            "failure was observed.",
            f"9. Explicit 0.19 mm cavity closure by loaded state was `{closure_summary}`. "
            "The unloaded source/interface coincidence has been removed: any initial-medium "
            "switch now occurs only after the geometry-derived air cavity closes.",
            "",
            "## Artifacts",
            "",
            "- `5led_optix/raw_results.npz`: combined and per-emitter responses, energy, timing, and geometry metadata",
            "- `5led_optix/escaped_<state>_led<N>.npz`: raw escaped paths for every primary trace",
            "- `5led_optix/state_summary.csv`: combined response and energy ledger",
            "- `5led_optix/per_emitter_response.csv`: 5×4 diagnostic source contributions",
            "- `5led_optix/pairwise_distances.csv`: raw state distances",
            "- `5led_optix/sample_convergence.csv`: deterministic sample-count comparison",
            "- `5led_optix/optical_geometry.png`: full geometry, sources, closure, and view direction",
            "- `5led_optix/combined_responses.png`: common-scale no-contact/contact response",
            "- `5led_optix/contact_response_deltas.png`: no-contact subtraction",
            "- `5led_optix/per_emitter_side_power.png`: source contribution matrix",
            "- `5led_optix/center_between_comparison.png`: half-pitch diagnostic",
        )
    )
    _REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    fingertip, mesh, states, mesh_build_time_s = _load_mesh_and_states()
    leds = _make_leds(fingertip, mesh)
    state_names = tuple(name for name, _ in _STATE_SOURCES)

    scene_start_s = perf_counter()
    scene = OptixScene(
        mesh,
        silicone_instance_id=_SILICONE_INSTANCE_ID,
        carrier_instance_id=_CARRIER_INSTANCE_ID,
        silicone_visibility_mask=_SILICONE_MASK,
        carrier_visibility_mask=_CARRIER_MASK,
    )
    scene_build_time_s = perf_counter() - scene_start_s

    results: dict[str, dict[str, object]] = {}
    for state_index, state_name in enumerate(state_names):
        update_time_s, cap_distance_m, cap_normal_W = _update_scene(
            scene,
            states[state_name],
            reference_state=state_name == "no_contact",
        )
        print(f"tracing {state_name} with five LEDs", flush=True)
        result = _trace_state(
            scene,
            fingertip,
            leds,
            state_name=state_name,
            sample_side=_PRODUCTION_SAMPLE_SIDE,
            persist_escaped_rays=True,
        )
        result["update_time_s"] = update_time_s
        result["cap_distance_m"] = cap_distance_m
        result["cap_normal_W"] = cap_normal_W
        results[state_name] = result
        print(
            f"  P(+Y)={result['combined_response'].sum():.8f} | "
            f"escaped={result['combined_energy'][1]:.8f} | "
            f"trace={result['trace_time_s']:.3f} s",
            flush=True,
        )

    no_contact_result = results["no_contact"]
    expected_gap_m = 1.0e-3 * LED_RECESS_DEPTH_MM
    if any(no_contact_result["source_inside_silicone"]):
        raise RuntimeError("an unloaded LED is not inside its explicit air cavity")
    if not np.allclose(
        no_contact_result["source_hit_distance_m"],
        expected_gap_m,
        rtol=0.0,
        atol=5.0e-7,
    ):
        raise RuntimeError("unloaded LED-to-silicone gap is not 0.19 mm")

    deterministic_max_difference = 0.0
    for state_name in state_names:
        _update_scene(
            scene,
            states[state_name],
            reference_state=state_name == "no_contact",
        )
        repeated = _trace_state(
            scene,
            fingertip,
            leds,
            state_name=f"{state_name}_repeat",
            sample_side=_PRODUCTION_SAMPLE_SIDE,
            persist_escaped_rays=False,
        )
        deterministic_max_difference = max(
            deterministic_max_difference,
            float(
                np.max(
                    np.abs(
                        repeated["per_led_response"]
                        - results[state_name]["per_led_response"]
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        repeated["per_led_energy"]
                        - results[state_name]["per_led_energy"]
                    )
                )
            ),
        )
        del repeated
    if deterministic_max_difference != 0.0:
        raise RuntimeError("fixed-sample OptiX rerun was not exactly deterministic")

    pairwise_rows: list[dict[str, float | str]] = []
    total_emitted_power = 5.0 * fingertip.parameters.led.normalized_power
    for state_a, state_b in combinations(state_names, 2):
        combined_delta = (
            results[state_b]["combined_response"]
            - results[state_a]["combined_response"]
        )
        per_emitter_delta = (
            results[state_b]["per_led_response"]
            - results[state_a]["per_led_response"]
        )
        pairwise_rows.append(
            {
                "state_a": state_a,
                "state_b": state_b,
                "combined_quadrant_l2": float(np.linalg.norm(combined_delta)),
                "per_emitter_l2": float(np.linalg.norm(per_emitter_delta)),
                "per_emitted_power_l2": float(
                    np.linalg.norm(per_emitter_delta) / total_emitted_power
                ),
            }
        )

    convergence_rows: list[dict[str, float | int | str]] = []
    for state_name in ("no_contact", "center_10n"):
        reference = results[state_name]
        for sample_side in _CONVERGENCE_SAMPLE_SIDES:
            if sample_side == _PRODUCTION_SAMPLE_SIDE:
                current = reference
            else:
                _update_scene(
                    scene,
                    states[state_name],
                    reference_state=state_name == "no_contact",
                )
                current = _trace_state(
                    scene,
                    fingertip,
                    leds,
                    state_name=f"{state_name}_{sample_side}",
                    sample_side=sample_side,
                    persist_escaped_rays=False,
                )
            response_norm = float(np.linalg.norm(reference["combined_response"]))
            side_reference = float(reference["combined_response"].sum())
            escaped_reference = float(reference["combined_energy"][1])
            convergence_rows.append(
                {
                    "state": state_name,
                    "sample_side": sample_side,
                    "paths_per_led": sample_side * sample_side,
                    "total_paths": 5 * sample_side * sample_side,
                    "combined_response_relative_l2": float(
                        np.linalg.norm(
                            current["combined_response"]
                            - reference["combined_response"]
                        )
                        / response_norm
                    ),
                    "side_visible_relative_change": float(
                        (current["combined_response"].sum() - side_reference)
                        / side_reference
                    ),
                    "escaped_power_relative_change": float(
                        (current["combined_energy"][1] - escaped_reference)
                        / escaped_reference
                    ),
                }
            )
            if current is not reference:
                del current

    combined_response = np.stack(
        [results[name]["combined_response"] for name in state_names]
    )
    per_led_response = np.stack(
        [results[name]["per_led_response"] for name in state_names]
    )
    combined_energy = np.stack(
        [results[name]["combined_energy"] for name in state_names]
    )
    per_led_energy = np.stack(
        [results[name]["per_led_energy"] for name in state_names]
    )
    np.savez_compressed(
        _OUTPUT_DIRECTORY / "raw_results.npz",
        state_names=np.asarray(state_names),
        state_labels=np.asarray(_STATE_LABELS),
        led_centers_m=mesh.led_centers_m,
        led_normals_W=np.stack([led.normal_W for led in leds]),
        paths_per_led=_PRODUCTION_SAMPLE_SIDE**2,
        total_paths_per_state=5 * _PRODUCTION_SAMPLE_SIDE**2,
        max_bounces=_MAX_BOUNCES,
        combined_response=combined_response,
        contact_delta_response=combined_response - combined_response[0],
        per_led_response=per_led_response,
        combined_energy=combined_energy,
        per_led_energy=per_led_energy,
        source_inside_silicone=np.stack(
            [results[name]["source_inside_silicone"] for name in state_names]
        ),
        source_first_hit_distances_m=np.stack(
            [results[name]["source_hit_distance_m"] for name in state_names]
        ),
        source_normal_projections=np.stack(
            [results[name]["source_normal_projection"] for name in state_names]
        ),
        energy_fields=np.asarray(_ENERGY_FIELDS),
        update_times_s=np.asarray(
            [results[name]["update_time_s"] for name in state_names]
        ),
        trace_times_s=np.asarray(
            [results[name]["trace_time_s"] for name in state_names]
        ),
        distal_cap_distances_m=np.asarray(
            [results[name]["cap_distance_m"] for name in state_names]
        ),
        distal_cap_normals_W=np.stack(
            [results[name]["cap_normal_W"] for name in state_names]
        ),
    )

    _write_tables(state_names, results, pairwise_rows, convergence_rows)
    _plot_geometry(mesh)
    _plot_responses(state_names, results)
    _write_report(
        fingertip,
        mesh,
        state_names,
        results,
        pairwise_rows,
        convergence_rows,
        mesh_build_time_s=mesh_build_time_s,
        scene_build_time_s=scene_build_time_s,
        deterministic_max_difference=deterministic_max_difference,
    )
    metadata = {
        "optical_preset": fingertip.parameters.optics.name,
        "refractive_index": fingertip.parameters.optics.refractive_index,
        "extinction_coefficient_m_inv": (
            fingertip.parameters.optics.extinction_coefficient_m_inv
        ),
        "led_centers_y_mm": (1.0e3 * mesh.led_centers_m[:, 1]).tolist(),
        "led_normal_W": [0.0, 0.0, -1.0],
        "modeled_power_per_led": fingertip.parameters.led.normalized_power,
        "total_modeled_power": total_emitted_power,
        "paths_per_led": _PRODUCTION_SAMPLE_SIDE**2,
        "total_paths_per_state": 5 * _PRODUCTION_SAMPLE_SIDE**2,
        "max_bounces": _MAX_BOUNCES,
        "rng_seed": _RNG_SEED,
        "carrier_albedo": _CARRIER_ALBEDO,
        "observation": "+Y side-view Q1-Q4",
        "mesh_build_time_s": mesh_build_time_s,
        "scene_build_time_s": scene_build_time_s,
        "deterministic_max_difference": deterministic_max_difference,
    }
    (_OUTPUT_DIRECTORY / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"report: {_REPORT_PATH}", flush=True)
    print(f"raw output: {_OUTPUT_DIRECTORY}", flush=True)


if __name__ == "__main__":
    main()

"""Compare segmentation-free visual contact-location cues on one recorded run.

This read-only exploratory study does not use a fingertip silhouette,
contact-location labels, per-location templates, or a separately recorded
calibration. Methods marked ``auto_reference`` use only unloaded frames already
stored at the beginning of the same acquisition to suppress stationary image
structure.

The current proprioceptive run has no contact-location ground truth. The study
therefore reports rank agreement with a synchronized motor-torque/FT moment-arm
proxy, not absolute localization error in millimetres.
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = (
    REPOSITORY_ROOT / "output" / "experiments" / "proprioceptive_force" / "run_001"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "output"
    / "validation"
    / "segmentation_free_contact_localization"
    / "run_001"
)
METHODS = (
    ("edge_pair_absolute", "Edge pair\n(frame only)", False),
    ("edge_pair_auto_reference", "Edge pair\n(auto reference)", True),
    ("red_marker", "Contactor marker\n(frame only)", False),
    ("green_peak_frame_only", "Green peak\n(frame only)", False),
    ("green_centroid_frame_only", "Green centroid\n(frame only)", False),
    ("green_peak_auto_reference", "Green peak\n(auto reference)", True),
    ("green_centroid_auto_reference", "Green centroid\n(auto reference)", True),
    ("temporal_difference_peak", "Frame difference\n(previous frame)", False),
)
METHOD_COLORS = {
    "edge_pair_absolute": "#8C8C8C",
    "edge_pair_auto_reference": "#2C758E",
    "red_marker": "#D97707",
    "green_peak_frame_only": "#4C9F70",
    "green_centroid_frame_only": "#2E7D5B",
    "green_peak_auto_reference": "#55A868",
    "green_centroid_auto_reference": "#8172B2",
    "temporal_difference_peak": "#C44E52",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--downscale", type=float, default=0.5)
    parser.add_argument("--minimum-force-n", type=float, default=8.0)
    parser.add_argument("--maximum-motor-speed-rad-s", type=float, default=0.15)
    parser.add_argument("--expected-location-count", type=int, default=6)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _nearest_indices(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(reference, query)
    indices = np.clip(indices, 1, len(reference) - 1)
    previous = indices - 1
    use_previous = (
        np.abs(reference[previous] - query) <= np.abs(reference[indices] - query)
    )
    return np.where(use_previous, previous, indices)


def _read_resized_bgr(path: Path, scale: float) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


def _temporal_median(
    run: Path, camera: pd.DataFrame, indices: np.ndarray, scale: float
) -> np.ndarray:
    frames = [
        _read_resized_bgr(run / camera.iloc[index]["filename"], scale)
        for index in indices
    ]
    return np.median(np.asarray(frames), axis=0).astype(np.float32)


def _scene_median(run: Path, camera: pd.DataFrame, scale: float) -> np.ndarray:
    count = min(48, len(camera))
    indices = np.unique(
        np.linspace(0, len(camera) - 1, count).round().astype(int)
    )
    return _temporal_median(run, camera, indices, scale).astype(np.uint8)


def _reference_median(run: Path, camera: pd.DataFrame, scale: float) -> np.ndarray:
    indices = np.flatnonzero(
        camera["capture_kind"].to_numpy() == "unloaded_reference"
    )
    if len(indices) == 0:
        count = max(3, int(np.ceil(0.1 * len(camera))))
        indices = np.argsort(camera["ft_contact_force_N"].to_numpy())[:count]
    return _temporal_median(run, camera, indices, scale)


def _select_regular_vertical_lattice(image: np.ndarray) -> np.ndarray:
    """Return four automatically detected, approximately regular circle centres."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height = gray.shape[0]
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=max(12, int(round(0.045 * height))),
        param1=55,
        param2=13,
        minRadius=max(3, int(round(0.008 * height))),
        maxRadius=max(8, int(round(0.030 * height))),
    )
    if circles is None or len(circles[0]) < 4:
        raise RuntimeError("fewer than four circular reference candidates detected")

    best: tuple[float, np.ndarray] | None = None
    for candidate in combinations(np.asarray(circles[0], dtype=np.float64), 4):
        points = np.asarray(candidate)[:, :2]
        points = points[np.argsort(points[:, 1])]
        spacings = np.diff(points[:, 1])
        mean_spacing = float(np.mean(spacings))
        if mean_spacing < 0.045 * height:
            continue
        fit = np.polyfit(points[:, 1], points[:, 0], 1)
        transverse_rms = float(
            np.sqrt(
                np.mean(
                    (points[:, 0] - np.polyval(fit, points[:, 1])) ** 2
                )
            )
        )
        spacing_cv = float(np.std(spacings) / mean_spacing)
        if spacing_cv > 0.18 or transverse_rms / mean_spacing > 0.20:
            continue
        score = 3.0 * spacing_cv + 2.0 * transverse_rms / mean_spacing
        if best is None or score < best[0]:
            best = (score, points)
    if best is None:
        raise RuntimeError("no regular four-circle longitudinal lattice detected")
    return best[1]


def _green_excess(image: np.ndarray) -> np.ndarray:
    blue, green, red = cv2.split(image.astype(np.float32))
    return green - 0.5 * (red + blue)


def _edge_pair_profile(gray: np.ndarray, x_start: int, x_stop: int) -> np.ndarray:
    gradient = np.abs(
        cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)[
            :, x_start:x_stop
        ]
    )
    height = gray.shape[0]
    score = np.zeros(height, dtype=np.float64)
    for separation in range(2, 9):
        paired = np.sqrt(gradient[:-separation] * gradient[separation:])
        candidate = gaussian_filter1d(
            np.mean(np.clip(paired, 0.0, 40.0), axis=1), sigma=1.0
        )
        score[: len(candidate)] = np.maximum(score[: len(candidate)], candidate)
    return score


def _peak_coordinate(profile: np.ndarray, y_start: int, y_stop: int) -> float:
    window = np.asarray(profile[y_start:y_stop], dtype=np.float64)
    if window.size == 0 or not np.any(np.isfinite(window)):
        return float("nan")
    return float(y_start + np.nanargmax(window))


def _weighted_coordinate(profile: np.ndarray, y_start: int, y_stop: int) -> float:
    window = np.asarray(profile[y_start:y_stop], dtype=np.float64)
    finite = np.isfinite(window)
    if not np.any(finite):
        return float("nan")
    threshold = float(np.percentile(window[finite], 70.0))
    weights = np.maximum(np.where(finite, window, threshold) - threshold, 0.0)
    if float(np.sum(weights)) <= 0.0:
        return float("nan")
    coordinates = np.arange(y_start, y_stop, dtype=np.float64)
    return float(np.sum(coordinates * weights) / np.sum(weights))


def _red_marker_coordinate(
    image: np.ndarray,
    *,
    x_start: int,
    y_start: int,
    y_stop: int,
) -> float:
    blue, green, red = cv2.split(image.astype(np.int16))
    excess = red - (green + blue) // 2
    mask = ((excess > 18) & (red > 35)).astype(np.uint8) * 255
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for index in range(1, count):
        x, _, width, height, area = stats[index]
        center_x, center_y = centroids[index]
        if (
            x >= x_start
            and y_start <= center_y < y_stop
            and 10 <= area <= 800
            and width <= 60
            and height <= 50
        ):
            candidates.append((float(area), float(center_y), float(center_x)))
    if not candidates:
        return float("nan")
    selected = sorted(candidates, reverse=True)[:4]
    return float(
        np.average(
            [candidate[1] for candidate in selected],
            weights=[candidate[0] for candidate in selected],
        )
    )


def _cluster_1d(values: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if len(finite) < count or len(np.unique(finite)) < count:
        raise ValueError("insufficient distinct estimates for location modes")
    centers = np.quantile(finite, np.linspace(0.0, 1.0, count))
    for _ in range(100):
        labels = np.argmin(np.abs(finite[:, None] - centers[None, :]), axis=1)
        updated = np.asarray(
            [
                np.mean(finite[labels == index])
                if np.any(labels == index)
                else centers[index]
                for index in range(count)
            ]
        )
        if np.allclose(updated, centers, atol=1e-8, rtol=0.0):
            break
        centers = updated
    order = np.argsort(centers)
    remap = np.empty(count, dtype=int)
    remap[order] = np.arange(count)
    return centers[order], remap[labels]


def _synchronize_measurements(
    run: Path, camera: pd.DataFrame
) -> tuple[pd.DataFrame, float]:
    motor = pd.read_csv(run / "motor.csv")
    ft = pd.read_csv(run / "ft.csv")
    motor_time = motor["timestamp_ns"].to_numpy(dtype=np.int64)
    ft_time = ft["timestamp_ns"].to_numpy(dtype=np.int64)
    ft_force = np.sqrt(
        ft["fx_N"].to_numpy() ** 2
        + ft["fy_N"].to_numpy() ** 2
        + ft["fz_N"].to_numpy() ** 2
    )
    motor_ft_indices = _nearest_indices(motor_time, ft_time)
    bias_candidates = (
        (ft_force[motor_ft_indices] < 0.8)
        & (np.abs(motor["velocity_rad_s"].to_numpy()) < 0.15)
    )
    if not np.any(bias_candidates):
        raise RuntimeError("no unloaded motor samples available for torque-bias estimate")
    torque_bias = float(np.median(motor.loc[bias_candidates, "torque_Nm"]))

    camera_time = camera["timestamp_ns"].to_numpy(dtype=np.int64)
    motor_indices = _nearest_indices(camera_time, motor_time)
    ft_indices = _nearest_indices(camera_time, ft_time)
    result = camera.copy()
    result["motor_torque_nm"] = motor["torque_Nm"].to_numpy()[motor_indices]
    result["motor_velocity_rad_s"] = motor["velocity_rad_s"].to_numpy()[motor_indices]
    result["ft_fx_n"] = ft["fx_N"].to_numpy()[ft_indices]
    result["ft_fy_n"] = ft["fy_N"].to_numpy()[ft_indices]
    result["ft_fz_n"] = ft["fz_N"].to_numpy()[ft_indices]
    result["ft_force_magnitude_n"] = ft_force[ft_indices]
    result["mechanical_moment_arm_proxy_mm"] = 1000.0 * (
        result["motor_torque_nm"] - torque_bias
    ) / np.maximum(result["ft_force_magnitude_n"], 0.2)
    return result, torque_bias


def _evaluate_algorithms(
    table: pd.DataFrame,
    eligible: np.ndarray,
    expected_location_count: int,
) -> tuple[list[dict[str, object]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    summary = []
    modes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    proxy_all = table.loc[eligible, "mechanical_moment_arm_proxy_mm"].to_numpy()
    for method, _, uses_reference in METHODS:
        estimates_all = table.loc[eligible, f"{method}_y_px"].to_numpy(dtype=float)
        valid = np.isfinite(estimates_all) & np.isfinite(proxy_all)
        estimates = estimates_all[valid]
        proxy = proxy_all[valid]
        if len(estimates) < 3 or len(np.unique(estimates)) < 2:
            rho = float("nan")
            p_value = float("nan")
        else:
            rho, p_value = spearmanr(estimates, proxy)
        centers, labels = _cluster_1d(estimates, expected_location_count)
        residual = np.abs(estimates - centers[labels])
        center_spacings = np.diff(centers)
        median_residual = float(np.median(residual))
        cluster_proxy = np.asarray(
            [np.median(proxy[labels == index]) for index in range(len(centers))]
        )
        cluster_rho, _ = spearmanr(centers, cluster_proxy)
        summary.append(
            {
                "algorithm": method,
                "uses_in_run_unloaded_reference": uses_reference,
                "eligible_frame_count": int(np.sum(eligible)),
                "valid_frame_count": int(np.sum(valid)),
                "coverage_percent": 100.0 * float(np.mean(valid)),
                "spearman_rho_vs_mechanical_proxy": float(rho),
                "absolute_spearman_rho": abs(float(rho)),
                "spearman_p_value": float(p_value),
                "median_location_mode_residual_px": median_residual,
                "minimum_location_mode_spacing_px": float(
                    np.min(center_spacings)
                ),
                "median_location_mode_spacing_px": float(
                    np.median(center_spacings)
                ),
                "minimum_spacing_to_residual_ratio": float(
                    np.min(center_spacings) / max(median_residual, 1e-6)
                ),
                "cluster_median_proxy_monotonic_rho": float(cluster_rho),
            }
        )
        modes[method] = (centers, labels)
    return summary, modes


def _plot_results(
    output: Path,
    table: pd.DataFrame,
    summary: list[dict[str, object]],
    eligible: np.ndarray,
    representative_bgr: np.ndarray,
    lattice: np.ndarray,
    search_bounds: tuple[int, int, int, int],
    scale: float,
    representative_index: int,
    selected_method: str,
) -> None:
    figure = plt.figure(figsize=(13.2, 7.2), constrained_layout=True)
    outer = figure.add_gridspec(2, 2, height_ratios=(0.95, 1.05))

    image_axis = figure.add_subplot(outer[0, 0])
    image_axis.imshow(cv2.cvtColor(representative_bgr, cv2.COLOR_BGR2RGB))
    for x, y in lattice:
        image_axis.add_patch(
            plt.Circle(
                (x, y),
                8.0,
                facecolor="none",
                edgecolor="#00BFC4",
                linewidth=1.2,
            )
        )
    x_start, x_stop, y_start, y_stop = search_bounds
    image_axis.add_patch(
        Rectangle(
            (x_start, y_start),
            x_stop - x_start,
            y_stop - y_start,
            facecolor="none",
            edgecolor="#D97707",
            linewidth=1.0,
            linestyle="--",
        )
    )
    contact_y = float(
        table.loc[representative_index, f"{selected_method}_y_px"]
    )
    image_axis.axhline(contact_y * scale, color="#D62728", linewidth=1.4)
    selected_label = next(
        label.replace("\n", " ")
        for method, label, _ in METHODS
        if method == selected_method
    )
    image_axis.text(
        0.98,
        0.03,
        f"Selected: {selected_label}",
        transform=image_axis.transAxes,
        ha="right",
        va="bottom",
        color="white",
        fontsize=8,
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
    )
    image_axis.set_title("(a) Automatic landmarks and selected visual coordinate")
    image_axis.axis("off")

    summary_axis = figure.add_subplot(outer[0, 1])
    names = [label.replace("\n", " ") for _, label, _ in METHODS]
    correlations = [float(row["absolute_spearman_rho"]) for row in summary]
    coverage = [float(row["coverage_percent"]) for row in summary]
    positions = np.arange(len(names))
    summary_axis.barh(
        positions,
        correlations,
        color=[METHOD_COLORS[method] for method, _, _ in METHODS],
        edgecolor="#4C5055",
        linewidth=0.5,
    )
    summary_axis.set_yticks(positions, names)
    summary_axis.invert_yaxis()
    summary_axis.set_xlim(0.0, 1.0)
    summary_axis.set_xlabel(r"$|\rho|$ vs. mechanical moment-arm proxy")
    summary_axis.set_title("(b) Rank agreement and valid-frame coverage")
    summary_axis.grid(axis="x", color="#DDDDDD", linewidth=0.5)
    for position, (correlation, valid_percent) in enumerate(
        zip(correlations, coverage)
    ):
        summary_axis.text(
            min(correlation + 0.02, 0.93),
            position,
            f"{correlation:.2f} · {valid_percent:.0f}%",
            va="center",
            fontsize=8,
        )

    scatter_grid = outer[1, :].subgridspec(2, 4, hspace=0.16, wspace=0.28)
    proxy = table.loc[eligible, "mechanical_moment_arm_proxy_mm"].to_numpy()
    first_anchor_y = float(lattice[0, 1]) / scale
    anchor_pitch = float(np.median(np.diff(lattice[:, 1]))) / scale
    for index, (method, label, uses_reference) in enumerate(METHODS):
        axis = figure.add_subplot(scatter_grid[index // 4, index % 4])
        estimate = table.loc[eligible, f"{method}_y_px"].to_numpy(dtype=float)
        valid = np.isfinite(estimate) & np.isfinite(proxy)
        relative = (estimate[valid] - first_anchor_y) / anchor_pitch
        axis.scatter(
            relative,
            proxy[valid],
            s=14,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.25,
            alpha=0.78,
        )
        row = summary[index]
        reference_note = "auto reference" if uses_reference else "no unloaded reference"
        axis.set_title(
            f"{label.replace(chr(10), ' ')}\n"
            f"rho={float(row['spearman_rho_vs_mechanical_proxy']):.2f}, "
            f"{reference_note}",
            fontsize=9,
        )
        axis.grid(color="#E3E3E3", linewidth=0.45)
        if index // 4 == 1:
            axis.set_xlabel("Position [reference-pitch units]")
        if index % 4 == 0:
            axis.set_ylabel("Moment-arm proxy [mm]")

    figure.suptitle(
        "Segmentation-free, label-free contact-location cues",
        fontsize=13,
    )
    figure.savefig(output / "segmentation_free_contact_localization.png", dpi=220)
    figure.savefig(output / "segmentation_free_contact_localization.pdf")
    plt.close(figure)


def main() -> None:
    arguments = _arguments()
    run = arguments.run.resolve()
    output = arguments.output.resolve()
    if not 0.1 <= arguments.downscale <= 1.0:
        raise ValueError("--downscale must be within [0.1, 1.0]")
    if arguments.expected_location_count < 2:
        raise ValueError("--expected-location-count must be at least two")

    camera = pd.read_csv(run / "camera_timestamps.csv")
    required = {
        "frame_index",
        "timestamp_ns",
        "filename",
        "capture_kind",
        "ft_contact_force_N",
    }
    if not required.issubset(camera.columns):
        raise ValueError(f"camera table missing {sorted(required - set(camera.columns))}")
    table, torque_bias = _synchronize_measurements(run, camera)

    scene = _scene_median(run, camera, arguments.downscale)
    reference = _reference_median(run, camera, arguments.downscale)
    lattice = _select_regular_vertical_lattice(scene)
    anchor_pitch = float(np.median(np.diff(lattice[:, 1])))
    anchor_fit = np.polyfit(lattice[:, 1], lattice[:, 0], 1)
    anchor_center_y = float(np.mean(lattice[:, 1]))
    anchor_center_x = float(np.polyval(anchor_fit, anchor_center_y))
    height, width = scene.shape[:2]
    y_start = max(0, int(np.floor(lattice[0, 1] - anchor_pitch)))
    y_stop = min(height, int(np.ceil(lattice[-1, 1] + anchor_pitch)))
    edge_x_start = max(0, int(round(anchor_center_x + 0.42 * anchor_pitch)))
    edge_x_stop = min(width, int(round(width - 0.50 * anchor_pitch)))
    optical_x_start = max(0, int(round(anchor_center_x + 0.10 * anchor_pitch)))
    optical_x_stop = min(width, int(round(anchor_center_x + 1.00 * anchor_pitch)))
    marker_x_start = max(0, int(round(anchor_center_x + 4.0 * anchor_pitch)))

    reference_gray = cv2.cvtColor(
        reference.astype(np.uint8), cv2.COLOR_BGR2GRAY
    ).astype(np.float32)
    reference_green = _green_excess(reference)
    reference_edge = _edge_pair_profile(
        reference_gray, edge_x_start, edge_x_stop
    )
    previous_gray: np.ndarray | None = None
    estimates = {method: [] for method, _, _ in METHODS}

    for _, row in table.iterrows():
        image = _read_resized_bgr(run / row["filename"], arguments.downscale)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        green = _green_excess(image)
        edge = _edge_pair_profile(gray, edge_x_start, edge_x_stop)
        estimates["edge_pair_absolute"].append(
            _peak_coordinate(edge, y_start, y_stop) / arguments.downscale
        )
        estimates["edge_pair_auto_reference"].append(
            _peak_coordinate(edge - reference_edge, y_start, y_stop)
            / arguments.downscale
        )
        estimates["red_marker"].append(
            _red_marker_coordinate(
                image,
                x_start=marker_x_start,
                y_start=y_start,
                y_stop=y_stop,
            )
            / arguments.downscale
        )
        absolute_green_profile = gaussian_filter1d(
            np.mean(
                np.maximum(
                    green[:, optical_x_start:optical_x_stop],
                    0.0,
                ),
                axis=1,
            ),
            sigma=3.0,
        )
        estimates["green_peak_frame_only"].append(
            _peak_coordinate(absolute_green_profile, y_start, y_stop)
            / arguments.downscale
        )
        estimates["green_centroid_frame_only"].append(
            _weighted_coordinate(absolute_green_profile, y_start, y_stop)
            / arguments.downscale
        )
        green_profile = gaussian_filter1d(
            np.mean(
                np.abs(
                    green[:, optical_x_start:optical_x_stop]
                    - reference_green[:, optical_x_start:optical_x_stop]
                ),
                axis=1,
            ),
            sigma=3.0,
        )
        estimates["green_peak_auto_reference"].append(
            _peak_coordinate(green_profile, y_start, y_stop) / arguments.downscale
        )
        estimates["green_centroid_auto_reference"].append(
            _weighted_coordinate(green_profile, y_start, y_stop)
            / arguments.downscale
        )
        if previous_gray is None:
            temporal_y = float("nan")
        else:
            temporal_profile = gaussian_filter1d(
                np.mean(np.abs(gray - previous_gray), axis=1), sigma=3.0
            )
            temporal_y = (
                _peak_coordinate(temporal_profile, y_start, y_stop)
                / arguments.downscale
            )
        estimates["temporal_difference_peak"].append(temporal_y)
        previous_gray = gray

    for method, _, _ in METHODS:
        table[f"{method}_y_px"] = estimates[method]
        table[f"{method}_reference_pitch"] = (
            table[f"{method}_y_px"] - lattice[0, 1] / arguments.downscale
        ) / (anchor_pitch / arguments.downscale)

    eligible = (
        (table["capture_kind"].to_numpy() == "contact")
        & (table["ft_force_magnitude_n"].to_numpy() >= arguments.minimum_force_n)
        & (
            np.abs(table["motor_velocity_rad_s"].to_numpy())
            <= arguments.maximum_motor_speed_rad_s
        )
        & ((table["motor_torque_nm"].to_numpy() - torque_bias) > 0.05)
    )
    summary, modes = _evaluate_algorithms(
        table, eligible, arguments.expected_location_count
    )

    # This frame-only geometric cue is the primary zero-calibration candidate.
    # Optical-only and in-run reference methods remain explicit alternatives.
    preferred = "edge_pair_absolute"
    preferred_values = table.loc[
        eligible, f"{preferred}_y_px"
    ].to_numpy(dtype=float)
    preferred_proxy = table.loc[
        eligible, "mechanical_moment_arm_proxy_mm"
    ].to_numpy()
    valid = np.isfinite(preferred_values) & np.isfinite(preferred_proxy)
    centers, labels = modes[preferred]
    cluster_rows = []
    for mode_index, center in enumerate(centers):
        selected = labels == mode_index
        values = preferred_values[valid][selected]
        proxy = preferred_proxy[valid][selected]
        cluster_rows.append(
            {
                "mode_index_distal_to_proximal_image": mode_index,
                "center_y_px": float(center),
                "center_reference_pitch": float(
                    (center - lattice[0, 1] / arguments.downscale)
                    / (anchor_pitch / arguments.downscale)
                ),
                "sample_count": int(np.sum(selected)),
                "within_mode_mad_px": float(
                    np.median(np.abs(values - np.median(values)))
                ),
                "mechanical_proxy_median_mm": float(np.median(proxy)),
                "mechanical_proxy_q25_mm": float(np.percentile(proxy, 25.0)),
                "mechanical_proxy_q75_mm": float(np.percentile(proxy, 75.0)),
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "algorithm_summary.csv", summary)
    _write_csv(output / "preferred_method_location_modes.csv", cluster_rows)
    table.to_csv(output / "per_frame_estimates.csv", index=False)

    eligible_indices = np.flatnonzero(eligible)
    eligible_preferred = table.loc[
        eligible, f"{preferred}_y_px"
    ].to_numpy(dtype=float)
    representative_index = int(
        eligible_indices[
            np.nanargmin(np.abs(eligible_preferred - np.nanmedian(eligible_preferred)))
        ]
    )
    representative = _read_resized_bgr(
        run / table.loc[representative_index, "filename"], arguments.downscale
    )
    _plot_results(
        output,
        table,
        summary,
        eligible,
        representative,
        lattice,
        (edge_x_start, edge_x_stop, y_start, y_stop),
        arguments.downscale,
        representative_index,
        preferred,
    )

    print(f"run: {run}")
    print("location GT: unavailable; reporting mechanical-proxy rank agreement")
    print(f"automatic circle lattice: {lattice.tolist()}")
    print(f"in-run torque bias: {torque_bias:.6f} N m")
    print(f"eligible frames: {int(np.sum(eligible))}/{len(table)}")
    for row in summary:
        print(
            f"{row['algorithm']}: |rho|={row['absolute_spearman_rho']:.3f}, "
            f"coverage={row['coverage_percent']:.1f}%, "
            f"mode residual={row['median_location_mode_residual_px']:.2f} px, "
            f"minimum mode spacing={row['minimum_location_mode_spacing_px']:.2f} px"
        )
    print(output / "segmentation_free_contact_localization.png")
    print(output / "segmentation_free_contact_localization.pdf")


if __name__ == "__main__":
    main()

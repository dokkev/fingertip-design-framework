"""NiceGUI editor and geometry explorer for the current LIT design space."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from math import isfinite
from numbers import Real
from pathlib import Path
import sys
from typing import Mapping

# ``python gui/design_space_app.py`` does not put the repository root on
# ``sys.path``. Reuse the repository's existing bootstrap convention for
# directly executed scripts; the normal module entry point needs no path hack.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from bootstrap import ensure_repository_root

    ensure_repository_root(Path(__file__).resolve().parent)

from nicegui import ui

from model import Fingertip, FingertipParameters, LED, OpticalMaterial
from optimization.design_space import (
    OPTIMIZABLE_PARAMETER_NAMES,
    DesignSpace,
    DesignVariable,
)
from visualization import plot_fingertip

from gui.baseline import current_lit_baseline
from gui.diagnostics import (
    Diagnostic,
    build_led,
    build_optical_material,
    diagnose_design_space,
    diagnose_physical_state,
    diagnose_state,
)


_VARIABLE_LABELS = {
    "flat_pad_width": "Overall width (w_l = w_fp = w_ep)",
    "flat_pad_height": "Flat-pad height (h_fp)",
    "semielliptical_pad_height": "Elliptical-pad height (h_ep)",
    "stem_width": "Stem width (w_s)",
    "stem_height": "Stem height (h_s)",
    "void_width": "Void width (w_v)",
}
_FIXED_GEOMETRY_NAMES = (
    "link_thickness",
    "bond_extension_width",
    "bond_extension_height",
    "void_height",
)
_FIXED_GEOMETRY_LABELS = {
    "link_thickness": "Link thickness (h_l)",
    "bond_extension_width": "Connector-pad width (w_cp)",
    "bond_extension_height": "Connector-pad height (h_cp)",
    "void_height": "Void height (h_v)",
}
_MECHANICAL_NAMES = ("young_modulus_mpa", "poisson_ratio")
_LED_NAMES = (
    "width_mm",
    "height_mm",
    "relative_radiant_power",
    "emission_half_angle_deg",
)
_OPTICAL_NAMES = (
    "refractive_index_air",
    "refractive_index_silicone",
    "absorption_per_mm",
    "scattering_per_mm",
    "anisotropy_g",
)
_GEOMETRY_PRECISION = {
    name: 2 for name in (*OPTIMIZABLE_PARAMETER_NAMES, *_FIXED_GEOMETRY_NAMES)
}
_GEOMETRY_STEP = {
    name: 0.1 for name in (*OPTIMIZABLE_PARAMETER_NAMES, *_FIXED_GEOMETRY_NAMES)
}


@dataclass(frozen=True)
class Preview:
    label: str
    tip: Fingertip | None
    diagnostics: tuple[Diagnostic, ...]

    @property
    def valid(self) -> bool:
        return not any(item.severity == "ERROR" for item in self.diagnostics)


@dataclass(frozen=True)
class Analysis:
    previews: tuple[Preview, Preview, Preview]
    state_diagnostics: tuple[Diagnostic, ...]
    design_diagnostics: tuple[Diagnostic, ...]
    corner_valid: int
    corner_total: int
    active_count: int

    @property
    def physical_diagnostics(self) -> tuple[Diagnostic, ...]:
        return tuple(
            item
            for item in self.state_diagnostics
            if item.source != "OPTICAL"
        )


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    result = float(value)
    return result if isfinite(result) else None


def _initial_state() -> dict[str, object]:
    baseline = current_lit_baseline()
    parameters = asdict(baseline)
    geometry = {
        name: parameters[name]
        for name in (*OPTIMIZABLE_PARAMETER_NAMES, *_FIXED_GEOMETRY_NAMES)
    }
    variables = {
        name: {
            "optimize": False,
            "lower": geometry[name],
            "upper": geometry[name],
        }
        for name in OPTIMIZABLE_PARAMETER_NAMES
    }
    return {
        "geometry": geometry,
        "mechanical": {
            "young_modulus_mpa": parameters["young_modulus_mpa"],
            "poisson_ratio": parameters["poisson_ratio"],
        },
        "led": {**asdict(LED()), "emission_rgb": list(LED().emission_rgb)},
        "optical": asdict(OpticalMaterial()),
        "variables": variables,
    }


def _parameters(
    geometry: Mapping[str, object],
    mechanical: Mapping[str, object],
) -> FingertipParameters:
    defaults = asdict(FingertipParameters())
    defaults.update(geometry)
    defaults.update(mechanical)
    return FingertipParameters(**defaults)


def _tip_for_preview(
    geometry: Mapping[str, object],
    mechanical: Mapping[str, object],
    led_values: Mapping[str, object],
    optical_values: Mapping[str, object],
    optical_valid: bool,
) -> Fingertip | None:
    parameters = _parameters(geometry, mechanical)
    led = build_led(led_values)
    optical = (
        build_optical_material({})
        if not optical_valid
        else build_optical_material(optical_values)
    )
    return Fingertip(parameters, led=led, optical=optical)


def _corner_values(state: Mapping[str, object]) -> tuple[dict[str, float], ...] | None:
    variables = state["variables"]
    assert isinstance(variables, Mapping)
    active = [
        variables[name]
        for name in OPTIMIZABLE_PARAMETER_NAMES
        if bool(variables[name]["optimize"])
    ]
    if not all(
        _number(variable[bound]) is not None
        for variable in active
        for bound in ("lower", "upper")
    ):
        return None
    if not active:
        return ({},)
    active_names = [
        name
        for name in OPTIMIZABLE_PARAMETER_NAMES
        if bool(variables[name]["optimize"])
    ]
    return tuple(
        {
            name: float(value)
            for name, value in zip(active_names, choice, strict=True)
        }
        for choice in product(
            *(
                (
                    float(variables[name]["lower"]),
                    float(variables[name]["upper"]),
                )
                for name in active_names
            )
        )
    )


def _state_with_corner(
    state: Mapping[str, object],
    values: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    geometry = dict(state["geometry"])
    geometry.update(values)
    mechanical = dict(state["mechanical"])
    led = dict(state["led"])
    return geometry, mechanical, led


def _preview(
    label: str,
    state: Mapping[str, object],
    values: Mapping[str, object],
    *,
    optical_valid: bool,
    missing_bound_message: str | None = None,
) -> Preview:
    geometry, mechanical, led = _state_with_corner(state, values)
    diagnostics = list(diagnose_physical_state(geometry, mechanical, led))
    if missing_bound_message is not None:
        diagnostics.append(Diagnostic("ERROR", label, missing_bound_message))
    tip = None
    if not any(item.severity == "ERROR" for item in diagnostics):
        optical = state["optical"]
        assert isinstance(optical, Mapping)
        try:
            tip = _tip_for_preview(
                geometry,
                mechanical,
                led,
                optical,
                optical_valid,
            )
        except Exception as exc:
            diagnostics.append(
                Diagnostic(
                    "ERROR",
                    label,
                    f"{type(exc).__name__}: {exc}\n"
                    "The public Fingertip constructor rejected this preview; "
                    "no automatic repair was applied.",
                )
            )
    return Preview(label, tip, tuple(diagnostics))


def _build_design_space(state: Mapping[str, object]) -> DesignSpace | None:
    try:
        baseline = _parameters(state["geometry"], state["mechanical"])
        variables = tuple(
            DesignVariable(
                name=name,
                optimize=bool(state["variables"][name]["optimize"]),
                lower=state["variables"][name]["lower"],
                upper=state["variables"][name]["upper"],
            )
            for name in OPTIMIZABLE_PARAMETER_NAMES
        )
        return DesignSpace(baseline=baseline, variables=variables)
    except Exception:
        return None


def _analyze(state: Mapping[str, object]) -> Analysis:
    geometry = state["geometry"]
    mechanical = state["mechanical"]
    led = state["led"]
    optical = state["optical"]
    assert isinstance(geometry, Mapping)
    assert isinstance(mechanical, Mapping)
    assert isinstance(led, Mapping)
    assert isinstance(optical, Mapping)

    state_diagnostics = diagnose_state(geometry, mechanical, led, optical)
    variables = state["variables"]
    assert isinstance(variables, Mapping)
    baseline_values = {name: geometry[name] for name in OPTIMIZABLE_PARAMETER_NAMES}
    design_diagnostics = diagnose_design_space(baseline_values, variables)
    active_count = sum(
        bool(variables[name]["optimize"]) for name in OPTIMIZABLE_PARAMETER_NAMES
    )
    optical_valid = not any(
        item.severity == "ERROR" and item.source == "OPTICAL"
        for item in state_diagnostics
    )

    baseline = _preview(
        "BASELINE",
        state,
        {},
        optical_valid=optical_valid,
    )
    design_space = _build_design_space(state)
    bounds = (
        design_space.corner_values()
        if design_space is not None
        else _corner_values(state)
    )
    if bounds is None:
        minimum = _preview(
            "MINIMUM",
            state,
            {},
            optical_valid=optical_valid,
            missing_bound_message="An active Min/Max value is missing or non-finite; "
            "enter numeric bounds before this corner can be previewed.",
        )
        maximum = _preview(
            "MAXIMUM",
            state,
            {},
            optical_valid=optical_valid,
            missing_bound_message="An active Min/Max value is missing or non-finite; "
            "enter numeric bounds before this corner can be previewed.",
        )
        corner_total = 2**active_count
        corner_valid = 0
    else:
        minimum_values = bounds[0]
        maximum_values = bounds[-1]
        minimum = _preview(
            "MINIMUM",
            state,
            minimum_values,
            optical_valid=optical_valid,
        )
        maximum = _preview(
            "MAXIMUM",
            state,
            maximum_values,
            optical_valid=optical_valid,
        )
        corner_results: list[bool] = []
        for corner in bounds:
            corner_geometry, corner_mechanical, corner_led = _state_with_corner(
                state, corner
            )
            diagnostics = diagnose_physical_state(
                corner_geometry,
                corner_mechanical,
                corner_led,
            )
            corner_results.append(
                not any(item.severity == "ERROR" for item in diagnostics)
            )
        corner_total = len(corner_results)
        corner_valid = sum(corner_results)
    return Analysis(
        previews=(minimum, baseline, maximum),
        state_diagnostics=tuple(state_diagnostics),
        design_diagnostics=tuple(design_diagnostics),
        corner_valid=corner_valid,
        corner_total=corner_total,
        active_count=active_count,
    )


def _corner_diagnostics(state: Mapping[str, object]) -> tuple[Diagnostic, ...]:
    variables = state["variables"]
    assert isinstance(variables, Mapping)
    bounds = _corner_values(state)
    if bounds is None:
        return ()
    result: list[Diagnostic] = []
    for corner in bounds:
        geometry, mechanical, led = _state_with_corner(state, corner)
        diagnostics = diagnose_physical_state(geometry, mechanical, led)
        details = tuple(
            item
            for item in diagnostics
            if item.severity in {"ERROR", "INFO"}
        )
        if not any(item.severity == "ERROR" for item in details):
            continue
        labels = []
        for name in OPTIMIZABLE_PARAMETER_NAMES:
            if bool(variables[name]["optimize"]):
                bound = (
                    "MIN"
                    if corner[name] == float(variables[name]["lower"])
                    else "MAX"
                )
                labels.append(f"{name} = {bound}")
        result.append(
            Diagnostic(
                "ERROR",
                "CORNER",
                "\n".join(labels)
                + "\nReason:\n"
                + "\n".join(item.message for item in details),
            )
        )
    return tuple(result)


def _all_console_diagnostics(state: Mapping[str, object], analysis: Analysis) -> tuple[Diagnostic, ...]:
    physical_corner_errors = _corner_diagnostics(state)
    baseline_valid = analysis.previews[1].valid
    optical_valid = not any(
        item.severity == "ERROR" and item.source == "OPTICAL"
        for item in analysis.state_diagnostics
    )
    design_errors = any(item.severity == "ERROR" for item in analysis.design_diagnostics)
    if analysis.active_count == 0:
        status = "PHYSICAL CONFIG VALID" if baseline_valid else "PHYSICAL CONFIG INVALID"
    elif design_errors:
        status = "DESIGN SPACE INCOMPLETE"
    elif analysis.corner_valid != analysis.corner_total:
        status = "DESIGN SPACE HAS INVALID CORNERS"
    else:
        status = "DESIGN SPACE READY"
    summary = Diagnostic(
        "INFO" if baseline_valid else "ERROR",
        "SUMMARY",
        f"{status}; Active optimization variables: {analysis.active_count}; "
        f"Physical corners: {analysis.corner_valid} / {analysis.corner_total} valid; "
        f"Optical settings: {'VALID' if optical_valid else 'INVALID'}.",
    )
    messages: list[Diagnostic] = [summary]
    if baseline_valid:
        messages.append(Diagnostic("INFO", "BASELINE", "Physical fingertip is valid."))
    messages.extend(analysis.design_diagnostics)
    messages.extend(
        item
        for item in analysis.state_diagnostics
        if item.source in {"GEOMETRY", "OPTICAL", "MECHANICAL", "LED"}
    )
    messages.extend(
        item
        for preview in analysis.previews
        for item in preview.diagnostics
        if item.severity == "ERROR"
    )
    messages.extend(physical_corner_errors)
    if analysis.corner_total > 0 and not physical_corner_errors:
        messages.append(
            Diagnostic(
                "INFO",
                "CORNER",
                f"Bound-corner feasibility: {analysis.corner_valid} / "
                f"{analysis.corner_total} valid. This does not prove all "
                "continuous interior designs are feasible.",
            )
        )
    if not optical_valid and baseline_valid:
        messages.append(
            Diagnostic(
                "INFO",
                "VISUALIZATION",
                "Optical settings are invalid; shape-only previews remain available "
                "without ray tracing.",
            )
        )
    return tuple(messages)


def _set_value(state: dict[str, object], section: str, name: str, value: object) -> None:
    section_state = state[section]
    assert isinstance(section_state, dict)
    section_state[name] = value


def _number_input(
    state: dict[str, object],
    section: str,
    name: str,
    *,
    label: str,
    suffix: str | None = None,
    precision: int | None = 2,
    step: float | None = 0.1,
    enabled: bool = True,
):
    values = state[section]
    assert isinstance(values, Mapping)
    control = ui.number(
        label,
        value=values.get(name),
        precision=precision,
        step=step,
        suffix=suffix,
        on_change=lambda event: _set_and_refresh(
            state, section, name, event.value
        ),
    )
    control.set_enabled(enabled)
    return control


def _set_and_refresh(
    state: dict[str, object], section: str, name: str, value: object
) -> None:
    _set_value(state, section, name, value)
    _refresh_state(state)


def _set_variable_and_refresh(
    state: dict[str, object], name: str, field: str, value: object
) -> None:
    variables = state["variables"]
    assert isinstance(variables, dict)
    variables[name][field] = value
    _refresh_state(state)


def _render_geometry_editor(state: dict[str, object]) -> None:
    with ui.card().classes("w-full"):
        ui.label("Geometry Design Space").classes("text-h6")
        ui.label(
            "Min/Max start at baseline. No scientific bound is invented; "
            "Optimize is initially OFF for every variable."
        ).classes("text-caption")
        with ui.row().classes("items-center no-wrap"):
            for heading in ("Parameter", "Optimize", "Baseline", "Min", "Max"):
                ui.label(heading).classes("w-28 text-weight-bold")
        geometry = state["geometry"]
        variables = state["variables"]
        assert isinstance(geometry, Mapping)
        assert isinstance(variables, Mapping)
        for name in OPTIMIZABLE_PARAMETER_NAMES:
            variable = variables[name]
            with ui.row().classes("items-center no-wrap"):
                ui.label(_VARIABLE_LABELS[name]).classes("w-56")
                ui.checkbox(
                    value=bool(variable["optimize"]),
                    on_change=lambda event, name=name: _set_variable_and_refresh(
                        state, name, "optimize", bool(event.value)
                    ),
                )
                _number_input(
                    state,
                    "geometry",
                    name,
                    label="Baseline",
                    suffix="mm",
                    precision=_GEOMETRY_PRECISION[name],
                    step=_GEOMETRY_STEP[name],
                )
                _bound_input(state, name, "lower", bool(variable["optimize"]))
                _bound_input(state, name, "upper", bool(variable["optimize"]))


def _bound_input(
    state: dict[str, object], name: str, bound: str, enabled: bool
) -> None:
    geometry = state["geometry"]
    variables = state["variables"]
    assert isinstance(geometry, Mapping)
    assert isinstance(variables, Mapping)
    baseline = _number(geometry.get(name))
    control = ui.number(
        "",
        value=variables[name][bound],
        min=baseline if bound == "upper" else None,
        max=baseline if bound == "lower" else None,
        suffix="mm",
        precision=_GEOMETRY_PRECISION[name],
        step=_GEOMETRY_STEP[name],
        on_change=lambda event, name=name, bound=bound: _set_variable_and_refresh(
            state, name, bound, event.value
        ),
    )
    control.set_enabled(enabled)


def _render_fixed_geometry(state: dict[str, object]) -> None:
    with ui.expansion("Fixed Geometry", icon="straighten", value=True).classes("w-full"):
        with ui.grid(columns=2).classes("w-full"):
            for name in _FIXED_GEOMETRY_NAMES:
                _number_input(
                    state,
                    "geometry",
                    name,
                    label=_FIXED_GEOMETRY_LABELS[name],
                    suffix="mm",
                    precision=2,
                    step=0.1,
                )
        ui.label(
            "CAD fields link_height, connection_pad_*, and pcb_* are intentionally "
            "not mapped: their physical semantics are not established."
        ).classes("text-caption")


def _render_mechanical(state: dict[str, object]) -> None:
    with ui.expansion(
        "Mechanical Properties", icon="engineering", value=True
    ).classes("w-full"):
        with ui.grid(columns=2).classes("w-full"):
            _number_input(
                state,
                "mechanical",
                "young_modulus_mpa",
                label="Young's modulus",
                suffix="MPa",
                precision=3,
                step=0.01,
            )
            _number_input(
                state,
                "mechanical",
                "poisson_ratio",
                label="Poisson ratio",
                precision=3,
                step=0.01,
            )


def _render_led(state: dict[str, object]) -> None:
    with ui.expansion("LED PCB Dimension", icon="lightbulb", value=True).classes(
        "w-full"
    ):
        with ui.grid(columns=2).classes("w-full"):
            _number_input(state, "led", "width_mm", label="Width", suffix="mm")
            _number_input(state, "led", "height_mm", label="Height", suffix="mm")
            _number_input(
                state,
                "led",
                "relative_radiant_power",
                label="Relative radiant power",
                precision=3,
                step=0.1,
            )
            _number_input(
                state,
                "led",
                "emission_half_angle_deg",
                label="Emission half angle",
                suffix="deg",
                precision=2,
                step=1.0,
            )
        ui.label("Emission RGB").classes("text-subtitle2")
        rgb = state["led"]["emission_rgb"]
        assert isinstance(rgb, list)
        with ui.row():
            for index in range(3):
                control = ui.number(
                    f"RGB {index}",
                    value=rgb[index],
                    precision=3,
                    step=0.05,
                    on_change=lambda event, index=index: _set_rgb_and_refresh(
                        state, index, event.value
                    ),
                )
                control.classes("w-28")


def _set_rgb_and_refresh(state: dict[str, object], index: int, value: object) -> None:
    led = state["led"]
    assert isinstance(led, dict)
    rgb = list(led["emission_rgb"])
    rgb[index] = value
    led["emission_rgb"] = rgb
    _refresh_state(state)


def _render_optical(state: dict[str, object]) -> None:
    with ui.expansion("Optical Properties", icon="blur_on", value=True).classes(
        "w-full"
    ):
        with ui.grid(columns=2).classes("w-full"):
            labels = {
                "refractive_index_air": "Refractive index (air)",
                "refractive_index_silicone": "Refractive index (silicone)",
                "absorption_per_mm": "Absorption / mm",
                "scattering_per_mm": "Scattering / mm",
                "anisotropy_g": "Anisotropy g",
            }
            for name in _OPTICAL_NAMES:
                _number_input(
                    state,
                    "optical",
                    name,
                    label=labels[name],
                    precision=3,
                    step=0.01,
                )
        ui.label("Optical edits validate model settings only; no rays are traced.").classes(
            "text-caption"
        )


_FIXED_DIMENSION_COLOR = "#111111"
_OPTIMIZED_DIMENSION_COLOR = "#0047FF"


def _dimension_color(state: Mapping[str, object], name: str) -> str:
    variables = state["variables"]
    assert isinstance(variables, Mapping)
    if name in variables and bool(variables[name]["optimize"]):
        return _OPTIMIZED_DIMENSION_COLOR
    return _FIXED_DIMENSION_COLOR


def _dimension_label(
    symbol: str,
    _value: float,
    *,
    relation: str | None = None,
) -> str:
    """Return only the variable annotation; values remain in the editor."""
    return relation or symbol


def _horizontal_dimension(
    axis,
    x_start: float,
    x_end: float,
    y: float,
    text: str,
    color: str,
    *,
    text_offset: float,
    guide_y: float | None = None,
) -> None:
    if guide_y is not None:
        axis.plot(
            [x_start, x_start],
            [guide_y, y],
            color=_FIXED_DIMENSION_COLOR,
            linestyle=(0, (4, 4)),
            linewidth=1.0,
            zorder=10,
        )
        axis.plot(
            [x_end, x_end],
            [guide_y, y],
            color=_FIXED_DIMENSION_COLOR,
            linestyle=(0, (4, 4)),
            linewidth=1.0,
            zorder=10,
        )
    if abs(x_end - x_start) <= 1.0e-12:
        axis.text(
            x_start,
            y + text_offset,
            text,
            color=color,
            ha="center",
            va="bottom",
            fontsize=9,
        )
        return
    axis.annotate(
        "",
        xy=(x_end, y),
        xytext=(x_start, y),
        arrowprops={
            "arrowstyle": "<->",
            "color": color,
            "linewidth": 1.8,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
    )
    axis.text(
        (x_start + x_end) / 2.0,
        y + text_offset,
        text,
        color=color,
        ha="center",
        va="bottom",
        fontsize=9,
    )


def _vertical_dimension(
    axis,
    x: float,
    y_start: float,
    y_end: float,
    text: str,
    color: str,
    *,
    text_offset: float,
    guide_x: float | None = None,
) -> None:
    if guide_x is not None:
        axis.plot(
            [guide_x, x],
            [y_start, y_start],
            color=_FIXED_DIMENSION_COLOR,
            linestyle=(0, (4, 4)),
            linewidth=1.0,
            zorder=10,
        )
        axis.plot(
            [guide_x, x],
            [y_end, y_end],
            color=_FIXED_DIMENSION_COLOR,
            linestyle=(0, (4, 4)),
            linewidth=1.0,
            zorder=10,
        )
    if abs(y_end - y_start) <= 1.0e-12:
        axis.text(
            x + text_offset,
            y_start,
            text,
            color=color,
            ha="left",
            va="center",
            fontsize=9,
        )
        return
    axis.annotate(
        "",
        xy=(x, y_end),
        xytext=(x, y_start),
        arrowprops={
            "arrowstyle": "<->",
            "color": color,
            "linewidth": 1.8,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
    )
    axis.text(
        x + text_offset,
        (y_start + y_end) / 2.0,
        text,
        color=color,
        ha="center",
        va="center",
        rotation=90,
        fontsize=9,
    )


def _render_parameter_drawing(
    state: Mapping[str, object],
    analysis: Analysis,
) -> None:
    """Show one annotated baseline cross-section with state-aware dimensions."""
    with ui.card().classes("w-full"):
        ui.label("Geometry parameter map").classes("text-h6")
        ui.label(
            "Black = fixed study parameter; blue = active optimization parameter."
        ).classes("text-caption")
        matplotlib = ui.matplotlib(figsize=(8.5, 7.5))
        figure = matplotlib.figure
        axis = figure.subplots()
        preview = analysis.previews[1]
        if preview.tip is None or not preview.valid:
            axis.text(
                0.5,
                0.5,
                "BASELINE\nINVALID\nSee diagnostics below",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            matplotlib.update()
            return

        tip = preview.tip
        parameters = tip.parameters
        plot_fingertip(
            tip,
            ax=axis,
            show_light_source=False,
            show_interface=False,
            show_contact_boundaries=False,
            show_legend=False,
            show_axes=True,
            title="Baseline cross-section",
        )

        width = parameters.flat_pad_width
        left = -width / 2.0
        right = width / 2.0
        top = parameters.link_thickness
        flat_bottom = -parameters.flat_pad_height
        pad_bottom = parameters.pad_tip_y
        stem_bottom = -parameters.stem_height
        void_bottom = parameters.void_bottom_y
        span = max(width, parameters.total_pad_depth + top, 2.0)
        vertical_offset = 0.025 * span

        _horizontal_dimension(
            axis,
            left,
            right,
            top + 0.18 * span,
            _dimension_label(
                "w_l",
                width,
                relation="w_l = w_fp = w_ep",
            ),
            _dimension_color(state, "flat_pad_width"),
            text_offset=vertical_offset,
            guide_y=top,
        )
        _horizontal_dimension(
            axis,
            left,
            left + parameters.bond_extension_width,
            top + 0.08 * span,
            _dimension_label("w_cp", parameters.bond_extension_width),
            _FIXED_DIMENSION_COLOR,
            text_offset=vertical_offset,
            guide_y=parameters.bond_extension_height,
        )
        _horizontal_dimension(
            axis,
            -parameters.stem_width / 2.0,
            parameters.stem_width / 2.0,
            pad_bottom - 0.10 * span,
            _dimension_label("w_s", parameters.stem_width),
            _dimension_color(state, "stem_width"),
            text_offset=vertical_offset,
            guide_y=stem_bottom,
        )
        _horizontal_dimension(
            axis,
            parameters.stem_width / 2.0,
            parameters.cutout_half_width,
            void_bottom - 0.08 * span,
            _dimension_label("w_v", parameters.void_width),
            _FIXED_DIMENSION_COLOR,
            text_offset=vertical_offset,
            guide_y=void_bottom,
        )

        _vertical_dimension(
            axis,
            left - 0.10 * span,
            0.0,
            flat_bottom,
            _dimension_label("h_fp", parameters.flat_pad_height),
            _dimension_color(state, "flat_pad_height"),
            text_offset=-0.045 * span,
            guide_x=left,
        )
        _vertical_dimension(
            axis,
            left - 0.21 * span,
            flat_bottom,
            pad_bottom,
            _dimension_label("h_ep", parameters.semielliptical_pad_height),
            _dimension_color(state, "semielliptical_pad_height"),
            text_offset=-0.045 * span,
            guide_x=left,
        )
        _vertical_dimension(
            axis,
            right + 0.10 * span,
            0.0,
            top,
            _dimension_label("h_l", parameters.link_thickness),
            _FIXED_DIMENSION_COLOR,
            text_offset=0.045 * span,
            guide_x=right,
        )
        _vertical_dimension(
            axis,
            right + 0.20 * span,
            0.0,
            stem_bottom,
            _dimension_label("h_s", parameters.stem_height),
            _dimension_color(state, "stem_height"),
            text_offset=0.045 * span,
            guide_x=right,
        )
        _vertical_dimension(
            axis,
            right + 0.30 * span,
            stem_bottom,
            void_bottom,
            _dimension_label("h_v", parameters.void_height),
            _FIXED_DIMENSION_COLOR,
            text_offset=0.045 * span,
            guide_x=parameters.cutout_half_width,
        )
        _vertical_dimension(
            axis,
            left - 0.04 * span,
            0.0,
            parameters.bond_extension_height,
            _dimension_label("h_cp", parameters.bond_extension_height),
            _FIXED_DIMENSION_COLOR,
            text_offset=-0.045 * span,
            guide_x=left,
        )

        axis.set_xlim(left - 0.38 * span, right + 0.38 * span)
        axis.set_ylim(pad_bottom - 0.24 * span, top + 0.28 * span)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, color="#D9DDE3", linewidth=0.55, alpha=0.7)
        axis.set_axisbelow(True)
        figure.subplots_adjust(left=0.12, right=0.96, bottom=0.08, top=0.94)
        matplotlib.update()


def _shared_limits(previews: tuple[Preview, Preview, Preview], state: Mapping[str, object]):
    bounds = []
    for preview in previews:
        if preview.tip is None:
            continue
        bounds.append(preview.tip.geometry.raw_material_geometry.bounds)
        bounds.append(preview.tip.led_package_geometry.bounds)
    if bounds:
        min_x = min(item[0] for item in bounds)
        min_y = min(item[1] for item in bounds)
        max_x = max(item[2] for item in bounds)
        max_y = max(item[3] for item in bounds)
    else:
        geometry = state["geometry"]
        assert isinstance(geometry, Mapping)
        width = _number(geometry.get("flat_pad_width")) or 20.0
        flat_height = _number(geometry.get("flat_pad_height")) or 3.0
        ellipse_height = _number(geometry.get("semielliptical_pad_height")) or 7.0
        stem_height = _number(geometry.get("stem_height")) or 6.0
        link_thickness = _number(geometry.get("link_thickness")) or 3.5
        min_x, max_x = -width / 2.0, width / 2.0
        min_y, max_y = -(flat_height + ellipse_height + stem_height), link_thickness
    span = max(max_x - min_x, max_y - min_y, 2.0)
    padding = 0.08 * span
    return (min_x - padding, max_x + padding), (min_y - padding, max_y + padding)


def _render_previews(state: Mapping[str, object], analysis: Analysis) -> None:
    x_limits, y_limits = _shared_limits(analysis.previews, state)
    with ui.card().classes("w-full"):
        ui.label("Geometry comparison").classes("text-h6")
        ui.label("MINIMUM and MAXIMUM are all-lower/all-upper bound corners.").classes(
            "text-caption"
        )
        matplotlib = ui.matplotlib(figsize=(16, 6.5))
        figure = matplotlib.figure
        axes = figure.subplots(1, 3)
        figure.subplots_adjust(left=0.03, right=0.99, bottom=0.22, top=0.90, wspace=0.24)
        for axis, preview in zip(axes, analysis.previews, strict=True):
            if preview.tip is not None and preview.valid:
                plot_fingertip(
                    preview.tip,
                    ax=axis,
                    show_legend=False,
                    title=f"{preview.label} — VALID",
                )
                axis.set_xlim(*x_limits)
                axis.set_ylim(*y_limits)
                axis.set_aspect("equal", adjustable="box")
            else:
                axis.set_xlim(*x_limits)
                axis.set_ylim(*y_limits)
                axis.set_aspect("equal", adjustable="box")
                reason = (
                    preview.diagnostics[0].message.splitlines()[0]
                    if preview.diagnostics
                    else "preview unavailable"
                )
                axis.text(
                    0.5,
                    0.5,
                    f"{preview.label}\nINVALID\n{reason}",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                axis.set_title(f"{preview.label} — INVALID")
        handles = []
        labels = []
        for axis in axes:
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                break
        if handles:
            figure.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.025),
                ncol=min(len(labels), 4),
                fontsize=8,
                frameon=False,
            )
        matplotlib.update()


def _render_console(state: Mapping[str, object], analysis: Analysis) -> None:
    with ui.card().classes("w-full"):
        ui.label("Current diagnostics").classes("text-h6")
        console = ui.log(max_lines=250).classes("w-full h-64")
        for item in _all_console_diagnostics(state, analysis):
            console.push(item.formatted)


def _render_page(state: dict[str, object]) -> None:
    analysis = _analyze(state)
    with ui.row().classes("w-full items-start flex-nowrap gap-4"):
        with ui.column().style("width: 40%; min-width: 0;"):
            _render_geometry_editor(state)
            _render_fixed_geometry(state)
            _render_mechanical(state)
            _render_led(state)
            _render_optical(state)
            ui.button("Reset to LIT baseline", on_click=lambda: _reset_state(state))
        with ui.column().style("width: 60%; min-width: 0;"):
            _render_parameter_drawing(state, analysis)
            _render_previews(state, analysis)
    _render_console(state, analysis)


def _reset_state(state: dict[str, object]) -> None:
    refresh = state.get("_refresh")
    state.clear()
    state.update(_initial_state())
    state["_refresh"] = refresh
    _refresh_state(state)


def _refresh_state(state: Mapping[str, object]) -> None:
    refresh = state.get("_refresh")
    if refresh is not None:
        refresh.refresh()


@ui.page("/")
def design_space_page() -> None:
    """Render the single-page design-space editor."""
    state = _initial_state()

    @ui.refreshable
    def render() -> None:
        _render_page(state)

    state["_refresh"] = render
    render()


def main() -> None:
    """Launch the NiceGUI design-space explorer."""
    ui.run(title="LIT Hand Design Space Explorer")


# NiceGUI's default reload worker imports the entry point as ``__mp_main__``.
# Keep normal imports side-effect free while allowing that worker to start.
if __name__ in {"__main__", "__mp_main__"}:
    main()


__all__ = ["design_space_page", "main"]

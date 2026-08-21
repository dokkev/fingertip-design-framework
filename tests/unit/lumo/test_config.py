from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from lumo.config import LumoExecutionConfigError, load_lumo_execution_config
from lumo.mesh import volume_mesh_settings_for_tier
from lumo.physics.contracts import VBDDeterminismMode


CANONICAL = Path(__file__).resolve().parents[3] / "config" / "lumo_execution.yaml"


def _payload() -> dict:
    return yaml.safe_load(CANONICAL.read_text(encoding="utf-8"))


def test_complete_execution_yaml_resolves_exact_typed_contract() -> None:
    loaded = load_lumo_execution_config(CANONICAL)

    assert loaded.device == "cuda:0"
    assert loaded.volume_mesh == volume_mesh_settings_for_tier("search")
    assert loaded.mechanics.vbd_iterations == 100
    assert loaded.mechanics.deterministic_mode is VBDDeterminismMode.RUN_TO_RUN
    assert loaded.mechanics.max_load_increment_mm == pytest.approx(0.0125)
    assert loaded.mechanics.dt_s == pytest.approx(2.5e-4)
    assert loaded.mechanics.soft_contact_mu == pytest.approx(0.0)
    assert loaded.mechanics.rigid_sdf_target_voxel_mm == pytest.approx(0.125)
    assert loaded.transport.ray_count == 256
    assert loaded.transport.x_bounds_mm == (-16.0, 16.0)
    assert loaded.transport.y_bounds_mm == (-31.0, 4.5)
    assert loaded.source.sha256 == hashlib.sha256(CANONICAL.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload["newton"].pop("soft_contact_mu"), "newton keys mismatch"),
        (
            lambda payload: payload["newton"].__setitem__(
                "deterministic_mode", "not_guaranteed"
            ),
            "mechanics",
        ),
        (lambda payload: payload["transport"].__setitem__("typo", 1), "transport keys mismatch"),
        (lambda payload: payload["transport"].__setitem__("ray_count", True), "transport.ray_count"),
        (
            lambda payload: payload["mesh"].__setitem__("target_size_mm", "1.5"),
            "mesh.target_size_mm",
        ),
        (
            lambda payload: payload["runtime"].__setitem__("device", "cpu"),
            "runtime.device",
        ),
        (
            lambda payload: payload["transport"].__setitem__(
                "extrusion_depth_mm", 12.0
            ),
            "transport.extrusion_depth_mm",
        ),
        (lambda payload: payload["mesh"].__setitem__("target_size_mm", float("nan")), "mesh.target_size_mm"),
    ),
)
def test_execution_yaml_rejects_missing_unknown_wrong_type_and_nonfinite(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    payload = _payload()
    mutation(payload)
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(LumoExecutionConfigError, match=message):
        load_lumo_execution_config(path)


def test_execution_yaml_uses_safe_loader(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.yaml"
    path.write_text("!!python/object/apply:os.system ['false']\n", encoding="utf-8")

    with pytest.raises(LumoExecutionConfigError, match="invalid YAML"):
        load_lumo_execution_config(path)


def test_execution_yaml_rejects_duplicate_mapping_key_with_key_context(
    tmp_path: Path,
) -> None:
    text = CANONICAL.read_text(encoding="utf-8")
    text = text.replace("  ray_count: 256\n", "  ray_count: 256\n  ray_count: 1024\n")
    path = tmp_path / "duplicate.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(
        LumoExecutionConfigError,
        match=r"duplicate key 'ray_count'",
    ):
        load_lumo_execution_config(path)

"""Focused tests for dense observer configuration in the live assembly."""

import numpy as np
import pytest

from experiments.localization import (
    DenseProfileConfig,
    DenseTemplateModel,
    save_dense_template_model,
)
from scripts.live_contact_localization import _dense_config, _load_observer_model


def _model(config: DenseProfileConfig) -> DenseTemplateModel:
    return DenseTemplateModel(
        positions_mm=np.array((0.0, 5.0)),
        templates=np.array(((1.0, 0.0), (0.0, 1.0))),
        canonical_shape=(2, 4),
        feature_config=config,
    )


def test_default_dense_configs_match_validated_material_descriptors() -> None:
    solaris = _dense_config("dense-top10")
    dragon = _dense_config("dense-highpass")

    assert solaris == DenseProfileConfig(
        mode="top10_red",
        transverse_start_fraction=0.15,
        transverse_stop_fraction=0.85,
        longitudinal_smoothing_sigma_px=2.0,
    )
    assert dragon == DenseProfileConfig(
        mode="abs_highpass_red",
        transverse_stop_fraction=0.95,
        transverse_reduction="mean",
    )


def test_loaded_model_feature_config_is_online_source_of_truth(tmp_path) -> None:
    tuned = DenseProfileConfig(
        mode="top10_red",
        transverse_start_fraction=0.2,
        transverse_stop_fraction=0.7,
        longitudinal_smoothing_sigma_px=3.0,
    )
    path = tmp_path / "tuned.npz"
    save_dense_template_model(path, _model(tuned))

    config, model = _load_observer_model(path, _dense_config("dense-top10"))

    assert model is not None
    assert config == tuned
    assert config == model.feature_config


def test_loaded_model_must_match_selected_observer_mode(tmp_path) -> None:
    path = tmp_path / "wrong_mode.npz"
    save_dense_template_model(path, _model(DenseProfileConfig(mode="mean_red")))

    with pytest.raises(ValueError, match="feature mode"):
        _load_observer_model(path, _dense_config("dense-top10"))

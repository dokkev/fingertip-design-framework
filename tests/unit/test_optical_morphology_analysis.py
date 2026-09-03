"""Tests for fixed-frame optical morphology measurements."""

import numpy as np
import pytest

from experiments.optical_morphology_analysis import (
    longitudinal_red_signatures,
    mean_absolute_red_response,
    minimum_pairwise_separation,
    pairwise_signature_distances,
)


def test_response_magnitude_and_longitudinal_signature() -> None:
    differences = np.asarray(
        (
            ((1.0, -1.0), (2.0, -2.0)),
            ((2.0, 2.0), (4.0, 4.0)),
        )
    )

    magnitudes = mean_absolute_red_response(differences)
    signatures = longitudinal_red_signatures(differences)

    assert magnitudes.tolist() == pytest.approx((1.5, 3.0))
    assert np.array_equal(signatures, np.asarray(((0.0, 0.0), (2.0, 4.0))))


def test_pairwise_rms_and_minimum_separation() -> None:
    signatures = np.asarray(((0.0, 0.0), (3.0, 4.0), (3.0, 6.0)))

    distances = pairwise_signature_distances(signatures)
    minimum, pair = minimum_pairwise_separation(distances)

    assert distances[0, 1] == pytest.approx(np.sqrt(12.5))
    assert minimum == pytest.approx(np.sqrt(2.0))
    assert pair == (1, 2)

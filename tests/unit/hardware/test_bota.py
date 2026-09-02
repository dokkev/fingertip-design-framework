from __future__ import annotations

import math
import struct

import pytest

from experiments.hardware import BotaTareOffsets, crc16_x25, decode_bota_payload


def test_crc16_x25_known_check_value() -> None:
    assert crc16_x25(b"123456789") == 0x906E


def test_binary_payload_unpacking_and_tare() -> None:
    payload = struct.pack(
        "<H6fIf",
        7,
        3.0,
        4.0,
        12.0,
        0.1,
        0.2,
        0.2,
        123456,
        31.5,
    )
    sample = decode_bota_payload(
        payload,
        host_time_s=10.25,
        offsets=BotaTareOffsets(1.0, 1.0, 2.0, 0.0, 0.0, 0.0),
    )

    assert sample.host_time_s == 10.25
    assert sample.status == 7
    assert sample.sensor_timestamp == 123456
    assert sample.temperature_c == pytest.approx(31.5)
    assert (sample.fx_n, sample.fy_n, sample.fz_n) == pytest.approx((2.0, 3.0, 10.0))
    assert (sample.mx_nm, sample.my_nm, sample.mz_nm) == pytest.approx((0.1, 0.2, 0.2))


def test_force_torque_magnitudes_and_fz_share() -> None:
    payload = struct.pack(
        "<H6fIf",
        0,
        3.0,
        4.0,
        12.0,
        1.0,
        2.0,
        2.0,
        1,
        20.0,
    )
    sample = decode_bota_payload(payload, host_time_s=1.0)

    assert sample.force_magnitude_n == pytest.approx(13.0)
    assert sample.torque_magnitude_nm == pytest.approx(3.0)
    assert sample.fz_share == pytest.approx(12.0 / 13.0)
    assert math.isfinite(sample.fz_share)

"""Offline packet tests for the CubeMars AK40-10 MIT driver."""

from __future__ import annotations

from collections import deque

import can
import pytest

from experiments.hardware.ak40_10 import (
    AK40_10,
    P_MAX,
    P_MIN,
    T_MAX,
    T_MIN,
    V_MAX,
    V_MIN,
    float_to_uint,
    uint_to_float,
)
from experiments.hardware.can_io import CanIO


class _FakeCanIO:
    def __init__(self, received: list[can.Message] | None = None) -> None:
        self.sent: list[tuple[int, bytes, bool]] = []
        self.received = deque(received or [])

    def send(
        self,
        arbitration_id: int,
        data: bytes,
        *,
        is_extended_id: bool = False,
    ) -> None:
        self.sent.append((arbitration_id, bytes(data), is_extended_id))

    def recv(self, timeout: float | None = None) -> can.Message | None:
        del timeout
        return self.received.popleft() if self.received else None


def _feedback_message(
    *, motor_id: int = 0x13, arbitration_id: int | None = None
) -> can.Message:
    return can.Message(
        arbitration_id=motor_id if arbitration_id is None else arbitration_id,
        is_extended_id=False,
        data=bytes((motor_id, 0x80, 0x00, 0x80, 0x08, 0x00, 65, 3)),
        check=True,
    )


def test_can_io_sends_classic_frame_and_closes_idempotently(monkeypatch) -> None:
    class FakeBus:
        def __init__(self) -> None:
            self.sent: list[can.Message] = []
            self.shutdown_count = 0

        def send(self, message: can.Message) -> None:
            self.sent.append(message)

        def recv(self, timeout: float | None = None) -> None:
            del timeout
            return None

        def shutdown(self) -> None:
            self.shutdown_count += 1

    bus = FakeBus()

    def make_bus(*, interface: str, channel: str) -> FakeBus:
        assert interface == "socketcan"
        assert channel == "vcan-test"
        return bus

    monkeypatch.setattr("experiments.hardware.can_io.can.Bus", make_bus)
    can_io = CanIO("vcan-test")
    can_io.send(0x13, [1, 2, 3])
    can_io.close()
    can_io.close()

    assert len(bus.sent) == 1
    assert bus.sent[0].arbitration_id == 0x13
    assert not bus.sent[0].is_extended_id
    assert bytes(bus.sent[0].data) == b"\x01\x02\x03"
    assert bus.shutdown_count == 1


def test_can_io_rejects_payload_larger_than_classic_can_frame(monkeypatch) -> None:
    class FakeBus:
        def send(self, message: can.Message) -> None:
            raise AssertionError(f"unexpected frame: {message}")

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(
        "experiments.hardware.can_io.can.Bus",
        lambda **_kwargs: FakeBus(),
    )
    can_io = CanIO("vcan-test")

    with pytest.raises(ValueError, match="at most 8 bytes"):
        can_io.send(0x13, bytes(9))
    can_io.close()


def test_mit_conversion_endpoints_and_midpoint() -> None:
    assert float_to_uint(P_MIN, P_MIN, P_MAX, 16) == 0
    assert float_to_uint(P_MAX, P_MIN, P_MAX, 16) == 0xFFFF
    assert float_to_uint(P_MIN - 1.0, P_MIN, P_MAX, 16) == 0
    assert float_to_uint(P_MAX + 1.0, P_MIN, P_MAX, 16) == 0xFFFF
    assert float_to_uint(0.0, P_MIN, P_MAX, 16) == 0x8000
    assert float_to_uint(0.0, V_MIN, V_MAX, 12) == 0x800
    assert float_to_uint(0.0, T_MIN, T_MAX, 12) == 0x800
    assert uint_to_float(0, V_MIN, V_MAX, 12) == V_MIN
    assert uint_to_float(0xFFF, V_MIN, V_MAX, 12) == V_MAX
    assert uint_to_float(0x800, T_MIN, T_MAX, 12) == pytest.approx(0.0012210012)


def test_construction_is_passive_and_commands_use_standard_motor_id() -> None:
    can_io = _FakeCanIO()
    motor = AK40_10(can_io)

    assert can_io.sent == []
    assert motor.motor_id == 13
    motor.enable()
    motor.set_zero()
    motor.set_torque(0.0)
    motor.disable()

    assert can_io.sent == [
        (13, b"\xff\xff\xff\xff\xff\xff\xff\xfc", False),
        (13, b"\xff\xff\xff\xff\xff\xff\xff\xfe", False),
        (13, b"\x80\x00\x80\x00\x00\x00\x08\x00", False),
        (13, b"\xff\xff\xff\xff\xff\xff\xff\xfd", False),
    ]


def test_parse_feedback_decodes_motor_state() -> None:
    motor = AK40_10(_FakeCanIO(), motor_id=0x13)

    state = motor.parse_feedback(_feedback_message())

    assert state.motor_id == 0x13
    assert state.position == pytest.approx(uint_to_float(0x8000, P_MIN, P_MAX, 16))
    assert state.velocity == pytest.approx(uint_to_float(0x800, V_MIN, V_MAX, 12))
    assert state.torque == pytest.approx(uint_to_float(0x800, T_MIN, T_MAX, 12))
    assert state.temperature == 25.0
    assert state.error == 3


def test_read_state_skips_other_can_ids() -> None:
    unrelated = _feedback_message(motor_id=0x22)
    expected = _feedback_message()
    motor = AK40_10(_FakeCanIO([unrelated, expected]), motor_id=0x13)

    state = motor.read_state(timeout=0.1)

    assert state is not None
    assert state.motor_id == 0x13


def test_parse_feedback_rejects_extended_or_mismatched_frames() -> None:
    motor = AK40_10(_FakeCanIO(), motor_id=0x13)
    extended = _feedback_message()
    extended.is_extended_id = True

    with pytest.raises(ValueError, match="standard CAN frame"):
        motor.parse_feedback(extended)
    with pytest.raises(ValueError, match="does not match"):
        motor.parse_feedback(_feedback_message(arbitration_id=0x14))

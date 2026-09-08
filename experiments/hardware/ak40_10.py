"""CubeMars AK40-10 MIT-mode CAN protocol."""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import monotonic

import can

from .can_io import CanIO


P_MIN = -12.5
P_MAX = 12.5
V_MIN = -45.5
V_MAX = 45.5
T_MIN = -5.0
T_MAX = 5.0
KP_MIN = 0.0
KP_MAX = 500.0
KD_MIN = 0.0
KD_MAX = 5.0

_ENTER_CONTROL_MODE = bytes((0xFF,) * 7 + (0xFC,))
_EXIT_CONTROL_MODE = bytes((0xFF,) * 7 + (0xFD,))
_SET_ZERO = bytes((0xFF,) * 7 + (0xFE,))


@dataclass(frozen=True)
class MotorState:
    """One decoded AK40-10 feedback frame in SI units."""

    motor_id: int
    position: float
    velocity: float
    torque: float
    temperature: float
    error: int


def float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    """Clamp and encode a float using the CubeMars MIT command mapping."""

    if not all(math.isfinite(value) for value in (x, x_min, x_max)):
        raise ValueError("conversion values must be finite")
    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min")
    if not isinstance(bits, int) or isinstance(bits, bool) or bits <= 0:
        raise ValueError("bits must be a positive integer")
    clamped = min(max(x, x_min), x_max)
    maximum_integer = (1 << bits) - 1
    value = int((clamped - x_min) * (1 << bits) / (x_max - x_min))
    return min(value, maximum_integer)


def uint_to_float(x_int: int, x_min: float, x_max: float, bits: int) -> float:
    """Map an unsigned protocol field to a floating-point value."""

    if not isinstance(bits, int) or isinstance(bits, bool) or bits <= 0:
        raise ValueError("bits must be a positive integer")
    if not math.isfinite(x_min) or not math.isfinite(x_max) or x_max <= x_min:
        raise ValueError("conversion range must be finite and increasing")
    maximum_integer = (1 << bits) - 1
    if (
        not isinstance(x_int, int)
        or isinstance(x_int, bool)
        or not 0 <= x_int <= maximum_integer
    ):
        raise ValueError(f"x_int must be an unsigned {bits}-bit integer")
    return x_int * (x_max - x_min) / maximum_integer + x_min


class AK40_10:
    """Drive one CubeMars AK40-10 in MIT mode over a shared CAN bus."""

    def __init__(self, can_io: CanIO, motor_id: int = 13) -> None:
        if (
            not isinstance(motor_id, int)
            or isinstance(motor_id, bool)
            or not 0 <= motor_id <= 0xFF
        ):
            raise ValueError("motor_id must be an unsigned 8-bit drive ID")
        self._can_io = can_io
        self.motor_id = motor_id

    def enable(self) -> None:
        """Enable motor control while the driver is in MIT mode."""

        self._send(_ENTER_CONTROL_MODE)

    def disable(self) -> None:
        """Exit motor control mode."""

        self._send(_EXIT_CONTROL_MODE)

    def set_zero(self) -> None:
        """Explicitly set the actuator's current position as zero."""

        self._send(_SET_ZERO)

    def send_command(
        self,
        position: float,
        velocity: float,
        kp: float,
        kd: float,
        torque: float,
    ) -> None:
        """Send one clamped AK40-10 MIT command."""

        p_int = float_to_uint(position, P_MIN, P_MAX, 16)
        v_int = float_to_uint(velocity, V_MIN, V_MAX, 12)
        kp_int = float_to_uint(kp, KP_MIN, KP_MAX, 12)
        kd_int = float_to_uint(kd, KD_MIN, KD_MAX, 12)
        torque_int = float_to_uint(torque, T_MIN, T_MAX, 12)

        self._send(
            bytes(
                (
                    p_int >> 8,
                    p_int & 0xFF,
                    v_int >> 4,
                    ((v_int & 0x0F) << 4) | (kp_int >> 8),
                    kp_int & 0xFF,
                    kd_int >> 4,
                    ((kd_int & 0x0F) << 4) | (torque_int >> 8),
                    torque_int & 0xFF,
                )
            )
        )

    def set_torque(self, torque: float) -> None:
        """Send a feed-forward torque command with all feedback gains zero."""

        self.send_command(0.0, 0.0, 0.0, 0.0, torque)

    def set_position(
        self,
        position: float,
        kp: float,
        kd: float,
        velocity: float = 0.0,
        torque: float = 0.0,
    ) -> None:
        """Send a position command through the same MIT packet path."""

        self.send_command(position, velocity, kp, kd, torque)

    def parse_feedback(self, message: can.Message) -> MotorState:
        """Validate and decode one feedback frame for this actuator."""

        if message.is_extended_id:
            raise ValueError("AK40-10 MIT feedback must use a standard CAN frame")
        if message.is_remote_frame or message.is_error_frame or message.is_fd:
            raise ValueError("AK40-10 MIT feedback must be a classic CAN data frame")
        if message.arbitration_id != self.motor_id:
            raise ValueError(
                f"feedback arbitration ID {message.arbitration_id:#x} does not match "
                f"motor ID {self.motor_id:#x}"
            )
        data = bytes(message.data)
        if message.dlc != 8 or len(data) != 8:
            raise ValueError("AK40-10 MIT feedback must contain exactly 8 bytes")
        if data[0] != self.motor_id:
            raise ValueError(
                f"feedback drive ID {data[0]:#x} does not match motor ID "
                f"{self.motor_id:#x}"
            )

        position_int = (data[1] << 8) | data[2]
        velocity_int = (data[3] << 4) | (data[4] >> 4)
        torque_int = ((data[4] & 0x0F) << 8) | data[5]
        return MotorState(
            motor_id=data[0],
            position=uint_to_float(position_int, P_MIN, P_MAX, 16),
            velocity=uint_to_float(velocity_int, V_MIN, V_MAX, 12),
            torque=uint_to_float(torque_int, T_MIN, T_MAX, 12),
            temperature=float(data[6] - 40),
            error=data[7],
        )

    def read_state(self, timeout: float | None = None) -> MotorState | None:
        """Read until this motor replies or the shared timeout expires."""

        if timeout is not None:
            timeout = float(timeout)
            if not math.isfinite(timeout) or timeout < 0.0:
                raise ValueError("timeout must be finite and nonnegative")
            deadline = monotonic() + timeout
        else:
            deadline = None

        remaining = timeout
        while True:
            message = self._can_io.recv(remaining)
            if message is None:
                return None
            if not message.is_extended_id and message.arbitration_id == self.motor_id:
                return self.parse_feedback(message)
            if deadline is not None:
                remaining = deadline - monotonic()
                if remaining <= 0.0:
                    return None

    def _send(self, data: bytes) -> None:
        self._can_io.send(
            self.motor_id,
            data,
            is_extended_id=False,
        )


__all__ = [
    "AK40_10",
    "KD_MAX",
    "KD_MIN",
    "KP_MAX",
    "KP_MIN",
    "MotorState",
    "P_MAX",
    "P_MIN",
    "T_MAX",
    "T_MIN",
    "V_MAX",
    "V_MIN",
    "float_to_uint",
    "uint_to_float",
]

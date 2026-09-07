"""Safely probe CubeMars AK40-10 feedback over SocketCAN."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from time import monotonic

import can


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from experiments.hardware.ak40_10 import AK40_10, MotorState  # noqa: E402
from experiments.hardware.can_io import CanIO  # noqa: E402


def _motor_id(value: str) -> int:
    try:
        parsed = int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid motor ID {value!r}; use decimal 13 or hexadecimal 0x13"
        ) from exc
    if not 0 <= parsed <= 0xFF:
        raise argparse.ArgumentTypeError("motor ID must be an unsigned 8-bit drive ID")
    return parsed


def _frame_text(message: can.Message) -> str:
    arbitration_id = message.arbitration_id
    data = bytes(message.data)
    frame_kind = "extended" if message.is_extended_id else "standard"
    return (
        f"id=0x{arbitration_id:03X} ({arbitration_id}) "
        f"data={data.hex(' ').upper()} {frame_kind}"
    )


def _wait_for_feedback(
    can_io: CanIO,
    motor: AK40_10,
    timeout_s: float,
) -> MotorState | None:
    deadline = monotonic() + timeout_s
    while True:
        remaining_s = deadline - monotonic()
        if remaining_s <= 0.0:
            return None
        message = can_io.recv(remaining_s)
        if message is None:
            return None

        print(f"  RX {_frame_text(message)}", flush=True)
        if message.is_extended_id or message.arbitration_id != motor.motor_id:
            data = bytes(message.data)
            if len(data) == 8 and data[0] == motor.motor_id:
                print(
                    "  NOTE payload drive ID matches, but arbitration ID does not",
                    flush=True,
                )
            continue

        try:
            return motor.parse_feedback(message)
        except ValueError as exc:
            print(f"  REJECTED target-ID frame: {exc}", flush=True)


def _probe(can_io: CanIO, motor_id: int, timeout_s: float) -> bool:
    motor = AK40_10(can_io, motor_id=motor_id)
    print(f"Probing decimal={motor_id}, hex=0x{motor_id:X}", flush=True)

    enabled = False
    try:
        motor.enable()
        enabled = True
        print("  TX enable  FF FF FF FF FF FF FF FC", flush=True)
        state = _wait_for_feedback(can_io, motor, timeout_s)
        if state is None:
            print(f"  RESULT no valid feedback within {timeout_s:g} s", flush=True)
            return False
        print(
            "  RESULT feedback "
            f"position={state.position:.6f} rad, "
            f"velocity={state.velocity:.6f} rad/s, "
            f"torque={state.torque:.6f} N m, "
            f"temperature={state.temperature:.1f} degC, error={state.error}",
            flush=True,
        )
        return True
    finally:
        if enabled:
            motor.disable()
            print("  TX disable FF FF FF FF FF FF FF FD", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enter MIT mode briefly, inspect raw CAN traffic, and disable each "
            "requested AK40-10. No motion, torque, or zero command is sent."
        )
    )
    parser.add_argument(
        "--motor-id",
        nargs="+",
        required=True,
        type=_motor_id,
        metavar="ID",
        help="one or more IDs; decimal 13 is 0x0D, while 0x13 is decimal 19",
    )
    parser.add_argument("--channel", default="can0", help="SocketCAN channel")
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=1.0,
        help="feedback wait per motor (default: 1.0)",
    )
    args = parser.parse_args()
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0.0:
        parser.error("--timeout-s must be positive")
    return args


def main() -> int:
    args = _parse_args()
    all_responded = True
    with CanIO(args.channel) as can_io:
        for motor_id in args.motor_id:
            all_responded = _probe(can_io, motor_id, args.timeout_s) and all_responded
    return 0 if all_responded else 1


if __name__ == "__main__":
    raise SystemExit(main())

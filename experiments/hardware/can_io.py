"""Small synchronous SocketCAN transport wrapper."""

from __future__ import annotations

import can


class CanIO:
    """Own one python-can SocketCAN bus."""

    def __init__(self, channel: str = "can0") -> None:
        if not isinstance(channel, str) or not channel.strip():
            raise ValueError("channel must be a nonempty string")
        self.channel = channel
        self._bus: can.BusABC | None = can.Bus(
            interface="socketcan",
            channel=channel,
        )

    def send(
        self,
        arbitration_id: int,
        data: bytes | bytearray | list[int],
        *,
        is_extended_id: bool = False,
    ) -> None:
        """Send one classic CAN data frame."""

        payload = bytes(data)
        if len(payload) > 8:
            raise ValueError("classic CAN payload must contain at most 8 bytes")
        message = can.Message(
            arbitration_id=arbitration_id,
            data=payload,
            is_extended_id=is_extended_id,
            check=True,
        )
        self._require_bus().send(message)

    def recv(self, timeout: float | None = None) -> can.Message | None:
        """Receive one frame, returning ``None`` when the timeout expires."""

        return self._require_bus().recv(timeout)

    def close(self) -> None:
        """Close the SocketCAN bus. Repeated calls are harmless."""

        bus = self._bus
        if bus is None:
            return
        self._bus = None
        bus.shutdown()

    def _require_bus(self) -> can.BusABC:
        if self._bus is None:
            raise RuntimeError("CAN bus is closed")
        return self._bus

    def __enter__(self) -> CanIO:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = ["CanIO"]

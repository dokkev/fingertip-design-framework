"""Bota Rokubi serial force/torque acquisition."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import struct
import threading
from time import perf_counter, sleep
from typing import Any

import numpy as np


_PAYLOAD_SIZE = 34
_FRAME_HEADER = b"\xAA"
_DEFAULT_BAUDRATE = 460800
_SERIAL_READ_TIMEOUT_S = 0.1


class BotaSerialError(RuntimeError):
    """Serial transport or Rokubi protocol failure."""


@dataclass(frozen=True)
class BotaTareOffsets:
    """Per-axis calibrated values subtracted after tare."""

    fx_n: float = 0.0
    fy_n: float = 0.0
    fz_n: float = 0.0
    mx_nm: float = 0.0
    my_nm: float = 0.0
    mz_nm: float = 0.0

    def __post_init__(self) -> None:
        if not np.all(np.isfinite(self.as_array())):
            raise ValueError("Bota tare offsets must be finite")

    def as_array(self) -> np.ndarray:
        """Return offsets in Fx, Fy, Fz, Mx, My, Mz order."""

        return np.asarray(
            (self.fx_n, self.fy_n, self.fz_n, self.mx_nm, self.my_nm, self.mz_nm),
            dtype=np.float64,
        )


@dataclass(frozen=True)
class BotaSample:
    """One complete, host-timestamped Rokubi force/torque sample."""

    host_time_s: float
    sensor_timestamp: int
    status: int
    temperature_c: float
    fx_n: float
    fy_n: float
    fz_n: float
    mx_nm: float
    my_nm: float
    mz_nm: float
    force_magnitude_n: float
    torque_magnitude_nm: float
    fz_share: float

    def __post_init__(self) -> None:
        numerical = (
            self.host_time_s,
            self.temperature_c,
            self.fx_n,
            self.fy_n,
            self.fz_n,
            self.mx_nm,
            self.my_nm,
            self.mz_nm,
            self.force_magnitude_n,
            self.torque_magnitude_nm,
            self.fz_share,
        )
        if not np.all(np.isfinite(numerical)):
            raise ValueError("Bota sample values must be finite")
        if self.sensor_timestamp < 0 or self.status < 0:
            raise ValueError("Bota status and sensor timestamp must be nonnegative")
        if self.force_magnitude_n < 0.0 or self.torque_magnitude_nm < 0.0:
            raise ValueError("Bota vector magnitudes must be nonnegative")
        if not 0.0 <= self.fz_share <= 1.0:
            raise ValueError("fz_share must be in [0, 1]")


@dataclass(frozen=True)
class _RawBotaFrame:
    host_time_s: float
    sensor_timestamp: int
    status: int
    temperature_c: float
    axes: tuple[float, float, float, float, float, float]


def crc16_x25(data: bytes) -> int:
    """Return the reflected CRC16-X25 checksum used by Rokubi frames."""

    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
    return (~crc) & 0xFFFF


def decode_bota_payload(
    payload: bytes,
    *,
    host_time_s: float,
    offsets: BotaTareOffsets | None = None,
) -> BotaSample:
    """Decode one validated 34-byte binary Rokubi payload."""

    if len(payload) != _PAYLOAD_SIZE:
        raise ValueError(f"Bota payload must contain {_PAYLOAD_SIZE} bytes")
    if not np.isfinite(host_time_s):
        raise ValueError("host_time_s must be finite")
    unpacked = struct.unpack("<H6fIf", payload)
    status = int(unpacked[0])
    raw_axes = np.asarray(unpacked[1:7], dtype=np.float64)
    corrected = raw_axes - (offsets or BotaTareOffsets()).as_array()
    fx, fy, fz, mx, my, mz = (float(value) for value in corrected)
    force_magnitude = math.sqrt(fx * fx + fy * fy + fz * fz)
    torque_magnitude = math.sqrt(mx * mx + my * my + mz * mz)
    fz_share = 0.0 if force_magnitude <= 1.0e-12 else abs(fz) / force_magnitude
    return BotaSample(
        host_time_s=float(host_time_s),
        sensor_timestamp=int(unpacked[7]),
        status=status,
        temperature_c=float(unpacked[8]),
        fx_n=fx,
        fy_n=fy,
        fz_n=fz,
        mx_nm=mx,
        my_nm=my,
        mz_nm=mz,
        force_magnitude_n=force_magnitude,
        torque_magnitude_nm=torque_magnitude,
        fz_share=min(1.0, fz_share),
    )


def _raw_frame(payload: bytes, host_time_s: float) -> _RawBotaFrame:
    unpacked = struct.unpack("<H6fIf", payload)
    return _RawBotaFrame(
        host_time_s=host_time_s,
        sensor_timestamp=int(unpacked[7]),
        status=int(unpacked[0]),
        temperature_c=float(unpacked[8]),
        axes=tuple(float(value) for value in unpacked[1:7]),
    )


def _sample_from_raw(
    frame: _RawBotaFrame,
    offsets: BotaTareOffsets,
) -> BotaSample:
    payload = struct.pack(
        "<H6fIf",
        frame.status,
        *frame.axes,
        frame.sensor_timestamp,
        frame.temperature_c,
    )
    return decode_bota_payload(
        payload,
        host_time_s=frame.host_time_s,
        offsets=offsets,
    )


class BotaSerialSensor:
    """Own one Rokubi serial connection and background sample reader."""

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        *,
        baudrate: int = _DEFAULT_BAUDRATE,
        history_size: int = 2048,
        startup_timeout_s: float = 10.0,
    ) -> None:
        if not isinstance(port, str) or not port.strip():
            raise ValueError("port must be a nonempty string")
        if not isinstance(baudrate, int) or isinstance(baudrate, bool) or baudrate <= 0:
            raise ValueError("baudrate must be a positive integer")
        if (
            not isinstance(history_size, int)
            or isinstance(history_size, bool)
            or history_size < 2
        ):
            raise ValueError("history_size must be an integer of at least two")
        if not np.isfinite(startup_timeout_s) or startup_timeout_s <= 0.0:
            raise ValueError("startup_timeout_s must be finite and positive")
        self.port = port
        self.baudrate = baudrate
        self.startup_timeout_s = float(startup_timeout_s)
        self._serial: Any | None = None
        self._reader_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._condition = threading.Condition()
        self._history: deque[BotaSample] = deque(maxlen=history_size)
        self._raw_history: deque[tuple[int, _RawBotaFrame]] = deque(
            maxlen=max(history_size, 512)
        )
        self._latest_raw: _RawBotaFrame | None = None
        self._sample_sequence = 0
        self._offsets = BotaTareOffsets()
        self._reader_error: BaseException | None = None
        self._crc_error_count = 0
        self._incomplete_frame_count = 0

    @property
    def is_running(self) -> bool:
        return self._serial is not None

    @property
    def tare_offsets(self) -> BotaTareOffsets:
        with self._condition:
            return self._offsets

    @property
    def crc_error_count(self) -> int:
        with self._condition:
            return self._crc_error_count

    @property
    def incomplete_frame_count(self) -> int:
        with self._condition:
            return self._incomplete_frame_count

    def start(self) -> None:
        """Configure the Rokubi and start its one background reader thread."""

        if self.is_running:
            raise RuntimeError("Bota serial sensor is already running")
        try:
            import serial
        except ImportError as error:
            raise BotaSerialError(
                "pyserial is required for BotaSerialSensor; install the acquisition extra"
            ) from error
        try:
            connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=_SERIAL_READ_TIMEOUT_S,
                write_timeout=1.0,
            )
        except Exception as error:
            raise BotaSerialError(
                f"could not open Bota Rokubi serial port {self.port}: {error}"
            ) from error
        self._serial = connection
        try:
            self._configure_sensor()
        except BaseException:
            connection.close()
            self._serial = None
            raise
        with self._condition:
            self._reader_error = None
            self._history.clear()
            self._raw_history.clear()
            self._latest_raw = None
            self._sample_sequence = 0
            self._offsets = BotaTareOffsets()
            self._crc_error_count = 0
            self._incomplete_frame_count = 0
        self._stop_event.clear()
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="bota_rokubi_reader",
            daemon=True,
        )
        self._reader_thread.start()

    def _wait_for_token(self, token: bytes) -> None:
        if self._serial is None:
            raise RuntimeError("Bota serial sensor is not open")
        deadline = perf_counter() + self.startup_timeout_s
        received = bytearray()
        while token not in received:
            if perf_counter() >= deadline:
                raise BotaSerialError(
                    f"timed out waiting for Rokubi response {token!r}"
                )
            byte = self._serial.read(1)
            if byte:
                received.extend(byte)
                if len(received) > max(512, 4 * len(token)):
                    del received[: len(received) - max(256, 2 * len(token))]

    def _send_and_expect(self, command: bytes, response: bytes) -> None:
        if self._serial is None:
            raise RuntimeError("Bota serial sensor is not open")
        self._serial.write(command)
        self._serial.flush()
        self._wait_for_token(response)

    def _configure_sensor(self) -> None:
        if self._serial is None:
            raise RuntimeError("Bota serial sensor is not open")
        self._wait_for_token(b"App Init")
        sleep(0.5)
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()
        self._send_and_expect(b"C", b"r,0,C,0")
        self._send_and_expect(b"c,0,1,0,4", b"r,0,c,0")
        self._send_and_expect(b"f,256,0,0,1", b"r,0,f,0")
        self._send_and_expect(b"R", b"r,0,R,0")

    def _read_payload(self) -> bytes | None:
        if self._serial is None:
            return None
        while not self._stop_event.is_set():
            if self._serial.read(1) != _FRAME_HEADER:
                continue
            payload = self._serial.read(_PAYLOAD_SIZE)
            checksum_bytes = self._serial.read(2)
            if len(payload) != _PAYLOAD_SIZE or len(checksum_bytes) != 2:
                with self._condition:
                    self._incomplete_frame_count += 1
                continue
            expected = struct.unpack("<H", checksum_bytes)[0]
            if crc16_x25(payload) != expected:
                with self._condition:
                    self._crc_error_count += 1
                continue
            return payload
        return None

    def _reader_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                payload = self._read_payload()
                if payload is None:
                    break
                frame = _raw_frame(payload, perf_counter())
                with self._condition:
                    self._sample_sequence += 1
                    self._latest_raw = frame
                    self._raw_history.append((self._sample_sequence, frame))
                    self._history.append(_sample_from_raw(frame, self._offsets))
                    self._condition.notify_all()
        except BaseException as error:
            if not self._stop_event.is_set():
                with self._condition:
                    self._reader_error = error
                    self._condition.notify_all()

    def _raise_reader_error_locked(self) -> None:
        if self._reader_error is not None:
            raise BotaSerialError(f"Bota reader failed: {self._reader_error}") from self._reader_error

    def latest_sample(self) -> BotaSample | None:
        """Return the newest complete sample, or ``None`` before first data."""

        with self._condition:
            self._raise_reader_error_locked()
            return self._history[-1] if self._history else None

    def nearest_sample(self, host_time_s: float) -> BotaSample | None:
        """Return the buffered sample nearest one host-monotonic time."""

        if not np.isfinite(host_time_s):
            raise ValueError("host_time_s must be finite")
        with self._condition:
            self._raise_reader_error_locked()
            if not self._history:
                return None
            return min(
                self._history,
                key=lambda sample: abs(sample.host_time_s - host_time_s),
            )

    def tare(
        self,
        *,
        skip_samples: int = 50,
        average_samples: int = 200,
        timeout_s: float = 10.0,
    ) -> BotaTareOffsets:
        """Average new raw samples and atomically replace all six tare offsets."""

        for name, value in (
            ("skip_samples", skip_samples),
            ("average_samples", average_samples),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if average_samples < 1:
            raise ValueError("average_samples must be at least one")
        if not np.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        if not self.is_running:
            raise RuntimeError("Bota serial sensor is not running")

        required = skip_samples + average_samples
        collected: list[_RawBotaFrame] = []
        deadline = perf_counter() + timeout_s
        with self._condition:
            cursor = self._sample_sequence
            while len(collected) < required:
                self._raise_reader_error_locked()
                new_frames = [
                    frame for sequence, frame in self._raw_history if sequence > cursor
                ]
                if new_frames:
                    collected.extend(new_frames)
                    cursor = self._sample_sequence
                    continue
                remaining = deadline - perf_counter()
                if remaining <= 0.0:
                    raise BotaSerialError(
                        f"timed out while collecting {required} Rokubi tare samples"
                    )
                self._condition.wait(timeout=remaining)

            axes = np.asarray(
                [frame.axes for frame in collected[skip_samples:required]],
                dtype=np.float64,
            )
            means = np.mean(axes, axis=0)
            self._offsets = BotaTareOffsets(*[float(value) for value in means])
            self._history.clear()
            if self._latest_raw is not None:
                self._history.append(_sample_from_raw(self._latest_raw, self._offsets))
            return self._offsets

    def stop(self) -> None:
        """Stop reading and close the serial device. Repeated calls are safe."""

        self._stop_event.set()
        thread = self._reader_thread
        self._reader_thread = None
        if thread is not None:
            thread.join(timeout=2.0)
        connection = self._serial
        self._serial = None
        if connection is not None:
            connection.close()
        with self._condition:
            self._condition.notify_all()

    def __enter__(self) -> BotaSerialSensor:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


__all__ = [
    "BotaSample",
    "BotaSerialError",
    "BotaSerialSensor",
    "BotaTareOffsets",
    "crc16_x25",
    "decode_bota_payload",
]

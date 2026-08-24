"""Crash-safe versioned checkpoints for an Ax campaign.

This module owns filesystem atomicity only.  Ax state is serialized through
Ax's public JSON API; the checkpoint store never reaches into Ax's private
serialization methods or treats the evaluation registry as an Ax snapshot.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Iterator, Mapping
import uuid

try:  # pragma: no cover - the production environment is POSIX/Linux.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


CHECKPOINT_SCHEMA_VERSION = "lumo-ax-checkpoint-v1"
CHECKPOINT_POINTER_SCHEMA_VERSION = "lumo-ax-checkpoint-pointer-v1"
_CHECKPOINT_PHASES = frozenset({"pre_evaluation", "post_evaluation"})
_REQUIRED_STATE_FIELDS = frozenset(
    {
        "schema_version",
        "sequence",
        "phase",
        "campaign_id",
        "evaluation_contract_id",
        "objective_identifier",
        "design_space",
        "parameterization_version",
        "ax_package_version",
        "seed",
        "budget",
        "counts",
        "pending_trial_index",
        "pending_latent_parameters",
        "pending_physical_parameters",
        "registry_key",
        "source",
        "resume_contract",
    }
)


class CheckpointError(RuntimeError):
    """A checkpoint is missing, corrupt, incompatible, or not writable."""


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"cannot read checkpoint JSON {path}: {exc}") from exc


def _write_json(path: Path, payload: Any) -> None:
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except (OSError, TypeError, ValueError) as exc:
        raise CheckpointError(f"cannot write checkpoint JSON {path}: {exc}") from exc


def _fsync_directory(path: Path) -> None:
    """Best-effort directory fsync for filesystems that expose it."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _fsync_file(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as exc:
        raise CheckpointError(f"cannot open checkpoint file for fsync {path}: {exc}") from exc
    try:
        os.fsync(fd)
    except OSError as exc:
        raise CheckpointError(f"cannot fsync checkpoint file {path}: {exc}") from exc
    finally:
        os.close(fd)


def _safe_checkpoint_directory(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise CheckpointError("checkpoint pointer has no directory")
    candidate = (root / relative).resolve()
    checkpoints_root = (root / "checkpoints").resolve()
    if candidate.parent != checkpoints_root or not candidate.name.isdigit():
        raise CheckpointError(f"invalid checkpoint directory pointer: {relative!r}")
    return candidate


class LoadedCheckpoint:
    """Validated immutable checkpoint files selected by the current pointer."""

    def __init__(
        self,
        *,
        directory: Path,
        pointer: Mapping[str, Any],
        state: Mapping[str, Any],
        trials: list[Mapping[str, Any]],
    ) -> None:
        self.directory = directory
        self.pointer = MappingProxyType(dict(pointer))
        self.state = MappingProxyType(dict(state))
        self.trials = tuple(MappingProxyType(dict(item)) for item in trials)


class CampaignCheckpointStore:
    """Persist one campaign with an atomic current-checkpoint pointer."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.checkpoints_root = self.root / "checkpoints"
        self.pointer_path = self.root / "checkpoint.json"
        self.lock_path = self.root / ".checkpoint.lock"

    @contextmanager
    def writer_lock(self) -> Iterator[None]:
        """Acquire a crash-releasing single-writer campaign lock."""
        self.root.mkdir(parents=True, exist_ok=True)
        if fcntl is None:
            raise CheckpointError("campaign locking requires a POSIX file-lock API")
        fd: int | None = None
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            if fd is not None:
                os.close(fd)
            raise CheckpointError(
                f"campaign checkpoint is already locked: {self.lock_path}"
            ) from exc
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"pid={os.getpid()}\n".encode("ascii"))
            os.fsync(fd)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _next_sequence(self) -> int:
        if not self.pointer_path.exists():
            return 1
        pointer = _read_json(self.pointer_path)
        if not isinstance(pointer, Mapping):
            raise CheckpointError("checkpoint pointer must be a JSON object")
        if pointer.get("schema") != CHECKPOINT_POINTER_SCHEMA_VERSION:
            raise CheckpointError("unsupported checkpoint pointer schema")
        sequence = pointer.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise CheckpointError("checkpoint pointer sequence is invalid")
        return sequence + 1

    def write(
        self,
        *,
        ax_client: Any,
        trials: list[Mapping[str, Any]],
        state: Mapping[str, Any],
    ) -> Path:
        """Write one complete checkpoint and atomically advance the pointer."""
        phase = state.get("phase")
        if phase not in _CHECKPOINT_PHASES:
            raise CheckpointError(f"unsupported checkpoint phase: {phase!r}")
        sequence = self._next_sequence()
        completed_directory = self.checkpoints_root / f"{sequence:06d}"
        self.checkpoints_root.mkdir(parents=True, exist_ok=True)
        if completed_directory.exists():
            raise CheckpointError(
                f"checkpoint sequence already exists: {completed_directory}"
            )

        state_payload = dict(state)
        state_payload.update(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "sequence": sequence,
                "phase": phase,
            }
        )
        missing = _REQUIRED_STATE_FIELDS - set(state_payload)
        if missing:
            raise CheckpointError(
                "checkpoint state is missing required fields: "
                + ", ".join(sorted(missing))
            )

        temporary_directory = Path(
            tempfile.mkdtemp(
                prefix=f".{sequence:06d}-", dir=self.checkpoints_root
            )
        )
        try:
            ax_path = temporary_directory / "ax_client.json"
            save = getattr(ax_client, "save_to_json_file", None)
            if not callable(save):
                raise CheckpointError(
                    "Ax client does not expose public save_to_json_file()"
                )
            try:
                save(str(ax_path))
            except Exception as exc:
                raise CheckpointError(
                    f"cannot save public Ax JSON state: {exc}"
                ) from exc
            if not ax_path.is_file():
                raise CheckpointError("Ax public save API did not create ax_client.json")
            _fsync_file(ax_path)

            _write_json(
                temporary_directory / "trials.json",
                [dict(item) for item in trials],
            )
            _write_json(temporary_directory / "state.json", state_payload)
            _fsync_directory(temporary_directory)
            os.replace(temporary_directory, completed_directory)
            _fsync_directory(self.checkpoints_root)

            pointer_payload = {
                "schema": CHECKPOINT_POINTER_SCHEMA_VERSION,
                "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
                "sequence": sequence,
                "phase": phase,
                "directory": f"checkpoints/{completed_directory.name}",
            }
            pointer_temporary = self.root / (
                f".checkpoint.json.{uuid.uuid4().hex}.tmp"
            )
            try:
                _write_json(pointer_temporary, pointer_payload)
                os.replace(pointer_temporary, self.pointer_path)
                _fsync_directory(self.root)
            finally:
                try:
                    pointer_temporary.unlink()
                except FileNotFoundError:
                    pass
            return completed_directory
        except Exception:
            if temporary_directory.exists():
                for child in temporary_directory.iterdir():
                    child.unlink()
                temporary_directory.rmdir()
            raise

    def load_latest(self) -> LoadedCheckpoint:
        """Load only the directory selected by the current pointer."""
        if not self.pointer_path.is_file():
            raise CheckpointError(f"checkpoint pointer does not exist: {self.pointer_path}")
        pointer = _read_json(self.pointer_path)
        if not isinstance(pointer, Mapping):
            raise CheckpointError("checkpoint pointer must be a JSON object")
        if pointer.get("schema") != CHECKPOINT_POINTER_SCHEMA_VERSION:
            raise CheckpointError("unsupported checkpoint pointer schema")
        if pointer.get("checkpoint_schema") != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointError("unsupported checkpoint schema")
        sequence = pointer.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise CheckpointError("checkpoint pointer sequence is invalid")
        directory = _safe_checkpoint_directory(self.root, pointer.get("directory"))
        if not directory.is_dir():
            raise CheckpointError(f"checkpoint directory does not exist: {directory}")

        state = _read_json(directory / "state.json")
        if not isinstance(state, Mapping):
            raise CheckpointError("checkpoint state must be a JSON object")
        if state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointError("unsupported checkpoint state schema")
        if state.get("sequence") != sequence:
            raise CheckpointError("checkpoint state sequence disagrees with pointer")
        if state.get("phase") not in _CHECKPOINT_PHASES:
            raise CheckpointError("checkpoint state phase is invalid")
        missing = _REQUIRED_STATE_FIELDS - set(state)
        if missing:
            raise CheckpointError(
                "checkpoint state is missing required fields: "
                + ", ".join(sorted(missing))
            )

        trials_payload = _read_json(directory / "trials.json")
        if not isinstance(trials_payload, list):
            raise CheckpointError("checkpoint trials must be a JSON list")
        if any(not isinstance(item, Mapping) for item in trials_payload):
            raise CheckpointError("checkpoint trials must contain JSON objects")
        if not (directory / "ax_client.json").is_file():
            raise CheckpointError("checkpoint is missing ax_client.json")
        return LoadedCheckpoint(
            directory=directory,
            pointer=pointer,
            state=state,
            trials=trials_payload,
        )

    @staticmethod
    def load_ax_client(checkpoint: LoadedCheckpoint) -> Any:
        """Restore Ax through its public JSON file loader."""
        from ax.api.client import Client

        try:
            return Client.load_from_json_file(
                str(checkpoint.directory / "ax_client.json")
            )
        except Exception as exc:
            raise CheckpointError(
                f"cannot restore Ax public JSON state: {exc}"
            ) from exc


__all__ = [
    "CHECKPOINT_POINTER_SCHEMA_VERSION",
    "CHECKPOINT_SCHEMA_VERSION",
    "CampaignCheckpointStore",
    "CheckpointError",
    "LoadedCheckpoint",
]

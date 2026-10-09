from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import tempfile


PAPER2_ROOT = Path(__file__).resolve().parents[2]


def _json_value(value: Any) -> Any:
    """Convert supported values to a deterministic, JSON-safe representation."""
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, set | frozenset):
        converted = [_json_value(item) for item in value]
        return sorted(converted, key=lambda item: canonical_json_bytes(item))
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if hasattr(value, "item"):
        return _json_value(value.item())
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        _json_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def guard_paper2_path(
    path: str | Path,
    *,
    paper2_root: str | Path = PAPER2_ROOT,
    allow_root: bool = False,
) -> Path:
    """Resolve a writable path and reject lexical or symlink escape from paper2/."""
    root = Path(paper2_root).expanduser().resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"output path escapes paper2 root: {resolved}")
    if resolved == root and not allow_root:
        raise ValueError("paper2 root itself is not a writable artifact path")
    return resolved


def _immutable_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to replace immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_canonical_json(
    path: str | Path,
    payload: Any,
    *,
    paper2_root: str | Path = PAPER2_ROOT,
) -> Path:
    output = guard_paper2_path(path, paper2_root=paper2_root)
    _immutable_bytes(output, canonical_json_bytes(payload) + b"\n")
    return output


def write_immutable_bytes(
    path: str | Path,
    payload: bytes,
    *,
    paper2_root: str | Path = PAPER2_ROOT,
) -> Path:
    output = guard_paper2_path(path, paper2_root=paper2_root)
    _immutable_bytes(output, payload)
    return output


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

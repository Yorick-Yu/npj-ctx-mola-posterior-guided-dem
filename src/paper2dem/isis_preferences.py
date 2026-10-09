from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import re
import stat
import tempfile

from .hashing import (
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    read_json,
    write_canonical_json,
    write_immutable_bytes,
)


RECEIPT_SCHEMA = "paper2.isis-spiceql-preferences/v1"
_GROUP = re.compile(
    r"(?ims)^\s*Group\s*=\s*SpiceQL\s*$"
    r"(?P<body>.*?)"
    r"^\s*EndGroup\s*$"
)
_USE = re.compile(
    r'(?im)^(?P<prefix>\s*UseSpiceQL\s*=\s*)(?P<quote>"?)(?P<value>true|false)(?P=quote)\s*$'
)


def _spiceql_value(text: str) -> tuple[re.Match[str], re.Match[str], str]:
    groups = list(_GROUP.finditer(text))
    if len(groups) != 1:
        raise ValueError("ISIS runtime preferences must contain exactly one SpiceQL group")
    group = groups[0]
    uses = list(_USE.finditer(group.group("body")))
    if len(uses) != 1:
        raise ValueError("ISIS SpiceQL group must contain exactly one boolean UseSpiceQL")
    use = uses[0]
    return group, use, use.group("value").lower()


def verify_spiceql_preferences(path: str | Path) -> str:
    preferences = Path(path).resolve()
    if not preferences.is_file():
        raise FileNotFoundError(f"ISIS runtime preferences are missing: {preferences}")
    _, _, value = _spiceql_value(preferences.read_text(encoding="utf-8"))
    if value != "true":
        raise ValueError("ISIS runtime preferences do not enable ALE/SpiceQL")
    return file_sha256(preferences)


def _enable_spiceql(text: str) -> tuple[str, bool]:
    group, use, value = _spiceql_value(text)
    if value == "true":
        return text, False
    body = group.group("body")
    updated_body = body[: use.start("value")] + "true" + body[use.end("value") :]
    updated = text[: group.start("body")] + updated_body + text[group.end("body") :]
    return updated, True


def _atomic_replace(path: Path, payload: bytes) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def configure_isis_spiceql_preferences(
    *,
    paper2_root: str | Path,
    isis_prefix: str | Path,
    template_path: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    root = Path(paper2_root).resolve()
    prefix = guard_paper2_path(isis_prefix, paper2_root=root)
    template = guard_paper2_path(template_path, paper2_root=root)
    receipt = guard_paper2_path(receipt_path, paper2_root=root)
    preferences = guard_paper2_path(prefix / "IsisPreferences", paper2_root=root)
    if not prefix.is_dir() or not (prefix / "conda-meta").is_dir():
        raise FileNotFoundError("isolated ISIS conda prefix is missing")
    if not template.is_file():
        raise FileNotFoundError(f"SpiceQL preference template is missing: {template}")
    verify_spiceql_preferences(template)

    if receipt.exists():
        payload = read_json(receipt)
        if not isinstance(payload, dict) or payload.get("schema") != RECEIPT_SCHEMA:
            raise ValueError("existing ISIS preference receipt schema differs")
        unhashed = dict(payload)
        declared = unhashed.pop("content_sha256", None)
        if declared != canonical_sha256(unhashed):
            raise ValueError("existing ISIS preference receipt content hash differs")
        if payload.get("preferences_relative_path") != preferences.relative_to(root).as_posix():
            raise ValueError("existing ISIS preference receipt path differs")
        if payload.get("template_sha256") != file_sha256(template):
            raise ValueError("existing ISIS preference receipt template differs")
        if payload.get("installed_sha256") != verify_spiceql_preferences(preferences):
            raise ValueError("installed ISIS preferences drifted after configuration")
        backup = guard_paper2_path(payload.get("backup_relative_path", ""), paper2_root=root)
        if not backup.is_file() or payload.get("original_sha256") != file_sha256(backup):
            raise ValueError("ISIS preference receipt original backup differs")
        return payload

    if not preferences.is_file():
        raise FileNotFoundError(f"ISIS conda-prefix preferences are missing: {preferences}")
    original = preferences.read_bytes()
    original_sha256 = file_sha256(preferences)
    original_text = original.decode("utf-8")
    installed_text, modified = _enable_spiceql(original_text)
    backup = guard_paper2_path(
        root / "reports/environment" / f"isis-preferences-original-{original_sha256}.pvl",
        paper2_root=root,
    )
    write_immutable_bytes(backup, original, paper2_root=root)
    if modified:
        _atomic_replace(preferences, installed_text.encode("utf-8"))
    installed_sha256 = verify_spiceql_preferences(preferences)
    payload: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "formal_artifact": False,
        "formal_training_unlocked": False,
        "isis_version": "10.0.0",
        "preferences_relative_path": preferences.relative_to(root).as_posix(),
        "original_sha256": original_sha256,
        "installed_sha256": installed_sha256,
        "backup_relative_path": backup.relative_to(root).as_posix(),
        "template_relative_path": template.relative_to(root).as_posix(),
        "template_sha256": file_sha256(template),
        "modified": modified,
        "use_spiceql": True,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    write_canonical_json(receipt, payload, paper2_root=root)
    return payload

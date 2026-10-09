from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import json
import os
import subprocess

from .hashing import (
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    read_json,
    write_canonical_json,
)


ISIS_ENVIRONMENT_RECEIPT_SCHEMA = "paper2.isis-environment-install/v1"
DEFAULT_CONDA_FORGE_MIRROR = (
    "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge"
)
CONDA_TIMEOUT_SECONDS = 15 * 60


@dataclass(frozen=True)
class FrozenCondaPackage:
    version: str
    build: str
    archive_sha256: str


FROZEN_ISIS_PACKAGES: dict[str, FrozenCondaPackage] = {
    "isis": FrozenCondaPackage(
        "10.0.0",
        "h1f94ec8_1",
        "3b52400e3c13457a167c51822e4dd7a23d74b39087072da49606b88e9018476d",
    ),
    "ale": FrozenCondaPackage(
        "1.2.0",
        "py313h7037e92_1",
        "864e497f76697219603f690b136c46a4913f97cedf4f8e143e01e74dced41465",
    ),
    "csm": FrozenCondaPackage(
        "3.1.0",
        "hb700be7_0",
        "e36853aed5f07fbbb5d9d1c99e21c6cbf4624dc2f5a01bb09a987386f7c0e5f9",
    ),
    "certifi": FrozenCondaPackage(
        "2026.6.17",
        "pyhd8ed1ab_0",
        "6c13620e458ba43278379d0cdacc30c497336bddfda81681fd50d114a65c702f",
    ),
    "usgscsm": FrozenCondaPackage(
        "2.1.0",
        "h71e3a03_3",
        "772376c06c6248e491cea8de0ef9c9efbd1f04e079596240aa8cea83c56336bc",
    ),
}


def _record_path(prefix: Path, name: str, package: FrozenCondaPackage) -> Path:
    return prefix / "conda-meta" / f"{name}-{package.version}-{package.build}.json"


def verify_isis_environment(
    *,
    prefix: Path,
    paper2_root: Path,
    require_usgscsm: bool = True,
) -> dict[str, Any]:
    """Verify exact Conda records plus every installed USGSCSM plugin library."""
    root = paper2_root.resolve()
    prefix = guard_paper2_path(prefix, paper2_root=root)
    packages: dict[str, Any] = {}
    usgscsm_payload: dict[str, Any] | None = None
    for name, package in FROZEN_ISIS_PACKAGES.items():
        record_path = _record_path(prefix, name, package)
        if name == "usgscsm" and not record_path.is_file() and not require_usgscsm:
            unexpected = sorted((prefix / "conda-meta").glob("usgscsm-*.json"))
            if unexpected:
                raise ValueError("an unfrozen USGSCSM package record is installed")
            continue
        if not record_path.is_file():
            raise FileNotFoundError(f"frozen Conda package record is missing: {record_path}")
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        observed_build = str(payload.get("build") or payload.get("build_string") or "")
        if (
            payload.get("name") != name
            or payload.get("version") != package.version
            or observed_build != package.build
            or payload.get("sha256") != package.archive_sha256
        ):
            raise ValueError(f"frozen Conda package record differs for {name}")
        packages[name] = {
            "version": package.version,
            "build": package.build,
            "archive_sha256": package.archive_sha256,
            "record_relative_path": record_path.relative_to(root).as_posix(),
            "record_sha256": file_sha256(record_path),
        }
        if name == "usgscsm":
            usgscsm_payload = payload
    if not require_usgscsm:
        return {"packages": packages, "plugins": []}
    if usgscsm_payload is None:
        raise RuntimeError("USGSCSM package record was not inspected")
    plugin_paths: list[Path] = []
    for raw in usgscsm_payload.get("files", []):
        relative = Path(str(raw))
        if "csmplugins" not in relative.parts or ".so" not in relative.name:
            continue
        plugin = guard_paper2_path(prefix / relative, paper2_root=root)
        if not plugin.is_file() or plugin.stat().st_size <= 0:
            raise ValueError(f"USGSCSM plugin library is missing or empty: {plugin}")
        plugin_paths.append(plugin)
    if not plugin_paths:
        raise ValueError("USGSCSM package contains no installed CSM plugin library")
    plugins = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
            "byte_count": path.stat().st_size,
        }
        for path in sorted(set(plugin_paths))
    ]
    return {"packages": packages, "plugins": plugins}


def parse_conda_json(stdout: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse Conda's first JSON object while preserving evidence of any suffix noise."""
    stripped = stdout.lstrip()
    if not stripped:
        raise ValueError("Conda returned no JSON output")
    payload, end = json.JSONDecoder().raw_decode(stripped)
    if not isinstance(payload, dict):
        raise ValueError("Conda JSON output is not an object")
    trailing = stripped[end:]
    evidence = {
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stdout_byte_count": len(stdout.encode("utf-8")),
        "trailing_nonwhitespace_byte_count": len(trailing.strip().encode("utf-8")),
        "trailing_nonwhitespace_sha256": (
            hashlib.sha256(trailing.strip().encode("utf-8")).hexdigest()
            if trailing.strip()
            else None
        ),
    }
    return payload, evidence


def _action_identity(raw: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(raw.get("name", "")),
        str(raw.get("version", "")),
        str(raw.get("build_string") or raw.get("build") or ""),
    )


def validate_usgscsm_transaction(
    payload: Mapping[str, Any],
    *,
    prefix: Path,
    require_link: bool,
) -> dict[str, Any]:
    if payload.get("success") is not True:
        raise ValueError("Conda transaction did not report success")
    actions = payload.get("actions", {})
    if actions is None:
        actions = {}
    if not isinstance(actions, Mapping):
        raise ValueError("Conda transaction actions are invalid")
    declared_prefix = str(actions.get("PREFIX") or payload.get("prefix") or "")
    if declared_prefix and Path(declared_prefix).resolve() != prefix.resolve():
        raise ValueError("Conda transaction targets a different prefix")
    unlink = list(actions.get("UNLINK", []))
    if unlink:
        raise ValueError("USGSCSM installation would unlink existing packages")
    expected = FROZEN_ISIS_PACKAGES["usgscsm"]
    expected_identity = ("usgscsm", expected.version, expected.build)
    link = [dict(raw) for raw in actions.get("LINK", [])]
    if require_link and [_action_identity(raw) for raw in link] != [expected_identity]:
        raise ValueError("USGSCSM installation is not the sole exact LINK action")
    if not require_link and link and [_action_identity(raw) for raw in link] != [expected_identity]:
        raise ValueError("Conda transaction contains an unexpected LINK action")
    fetch = [dict(raw) for raw in actions.get("FETCH", [])]
    for raw in fetch:
        if _action_identity(raw) != expected_identity:
            raise ValueError("Conda transaction contains an unexpected FETCH action")
        if raw.get("sha256") != expected.archive_sha256:
            raise ValueError("USGSCSM fetched archive hash differs")
    return {
        "link": [list(_action_identity(raw)) for raw in link],
        "fetch": [
            {
                "identity": list(_action_identity(raw)),
                "sha256": raw.get("sha256"),
                "size": raw.get("size"),
                "url": raw.get("url"),
            }
            for raw in fetch
        ],
        "unlink_count": 0,
    }


def _run_conda(command: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    completed = subprocess.run(
        tuple(command),
        check=False,
        text=True,
        capture_output=True,
        timeout=CONDA_TIMEOUT_SECONDS,
        env=dict(os.environ),
    )
    if completed.returncode:
        raise RuntimeError(
            f"Conda command failed with {completed.returncode}: {completed.stderr[-2000:]}"
        )
    payload, output_evidence = parse_conda_json(completed.stdout)
    output_evidence.update(
        {
            "returncode": completed.returncode,
            "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
            "stderr_byte_count": len(completed.stderr.encode("utf-8")),
        }
    )
    return payload, output_evidence


def _verify_existing_receipt(
    receipt: Path,
    *,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    payload = read_json(receipt)
    if payload.get("schema") != ISIS_ENVIRONMENT_RECEIPT_SCHEMA:
        raise ValueError("ISIS environment receipt schema differs")
    declared = payload.get("content_sha256")
    unhashed = dict(payload)
    unhashed.pop("content_sha256", None)
    if declared != canonical_sha256(unhashed):
        raise ValueError("ISIS environment receipt content hash differs")
    if payload.get("environment") != environment:
        raise ValueError("installed ISIS environment differs from its receipt")
    return payload


def install_frozen_usgscsm(
    *,
    paper2_root: Path,
    prefix: Path,
    conda: Path,
    receipt: Path,
    channel: str = DEFAULT_CONDA_FORGE_MIRROR,
) -> dict[str, Any]:
    """Install only the frozen USGSCSM package after a no-UNLINK dry-run."""
    root = paper2_root.resolve()
    prefix = guard_paper2_path(prefix, paper2_root=root)
    receipt = guard_paper2_path(receipt, paper2_root=root)
    if not conda.is_file() or not os.access(conda, os.X_OK):
        raise FileNotFoundError(f"Conda executable is unavailable: {conda}")
    try:
        final_environment = verify_isis_environment(
            prefix=prefix, paper2_root=root, require_usgscsm=True
        )
    except FileNotFoundError:
        final_environment = None
    if receipt.exists():
        if final_environment is None:
            raise ValueError("ISIS environment receipt exists but USGSCSM is absent")
        return _verify_existing_receipt(receipt, environment=final_environment)
    if final_environment is not None:
        payload: dict[str, Any] = {
            "schema": ISIS_ENVIRONMENT_RECEIPT_SCHEMA,
            "state": "complete",
            "formal_artifact": False,
            "formal_training_unlocked": False,
            "installation_performed": False,
            "channel": channel,
            "environment": final_environment,
        }
        payload["content_sha256"] = canonical_sha256(payload)
        write_canonical_json(receipt, payload, paper2_root=root)
        return payload
    verify_isis_environment(prefix=prefix, paper2_root=root, require_usgscsm=False)
    usgscsm = FROZEN_ISIS_PACKAGES["usgscsm"]
    certifi = FROZEN_ISIS_PACKAGES["certifi"]
    base_command = [
        str(conda),
        "install",
        "--json",
        "--freeze-installed",
        "--override-channels",
        "-c",
        channel,
        "-p",
        str(prefix),
        f"usgscsm={usgscsm.version}={usgscsm.build}",
        f"certifi={certifi.version}={certifi.build}",
    ]
    dry_payload, dry_output = _run_conda([*base_command, "--dry-run"])
    dry_actions = validate_usgscsm_transaction(
        dry_payload, prefix=prefix, require_link=True
    )
    install_payload, install_output = _run_conda([*base_command, "--yes"])
    install_actions = validate_usgscsm_transaction(
        install_payload, prefix=prefix, require_link=True
    )
    final_environment = verify_isis_environment(
        prefix=prefix, paper2_root=root, require_usgscsm=True
    )
    payload = {
        "schema": ISIS_ENVIRONMENT_RECEIPT_SCHEMA,
        "state": "complete",
        "formal_artifact": False,
        "formal_training_unlocked": False,
        "installation_performed": True,
        "channel": channel,
        "dry_run": {"actions": dry_actions, "output": dry_output},
        "install": {"actions": install_actions, "output": install_output},
        "environment": final_environment,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    write_canonical_json(receipt, payload, paper2_root=root)
    return payload

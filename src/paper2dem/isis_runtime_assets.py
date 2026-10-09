from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .hashing import file_sha256, guard_paper2_path, read_json, write_canonical_json
from .official_assets import download_official_asset


ISIS_BASE_KERNEL_TEMPLATE_SCHEMA = "paper2.isis-base-kernel-template/v1"
ISIS_BASE_KERNEL_LOCK_SCHEMA = "paper2.isis-base-kernel-lock/v1"
DEFAULT_TEMPLATE = Path("configs/g0/isis-base-kernel-lock.json")
DEFAULT_LOCK = Path("data/assets/isis-base-kernel-lock.json")


def _verify_asset(path: Path, *, expected_bytes: int, expected_sha256: str) -> None:
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise ValueError(f"ISIS base-kernel byte count differs: {path}")
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"ISIS base-kernel SHA-256 differs: {path}")


def prepare_isis_base_kernels(
    *,
    paper2_root: Path,
    template_path: Path = DEFAULT_TEMPLATE,
    output_path: Path = DEFAULT_LOCK,
    download_fn: Callable[..., dict[str, Any]] = download_official_asset,
) -> dict[str, Any]:
    root = paper2_root.resolve()
    template_path = guard_paper2_path(template_path, paper2_root=root)
    output_path = guard_paper2_path(output_path, paper2_root=root)
    template = read_json(template_path)
    if template.get("schema") != ISIS_BASE_KERNEL_TEMPLATE_SCHEMA:
        raise ValueError("ISIS base-kernel template schema differs")
    if template.get("isis_version") != "10.0.0":
        raise ValueError("ISIS base-kernel template version differs")
    raw_assets = template.get("assets")
    if not isinstance(raw_assets, list) or len(raw_assets) != 1:
        raise ValueError("ISIS base-kernel template must contain one frozen asset")
    locked_assets = []
    for raw in raw_assets:
        asset = dict(raw)
        destination = guard_paper2_path(str(asset["relative_path"]), paper2_root=root)
        receipt = root / "data/assets/isis/base-kernel-receipts" / (
            f"{asset['filename']}.receipt.json"
        )
        receipt = guard_paper2_path(receipt, paper2_root=root)
        expected_bytes = int(asset["bytes"])
        expected_sha256 = str(asset["sha256"])
        url = str(asset["url"])
        if destination.is_file() and not receipt.exists():
            _verify_asset(
                destination,
                expected_bytes=expected_bytes,
                expected_sha256=expected_sha256,
            )
            write_canonical_json(
                receipt,
                {
                    "schema": "paper2.official-download-receipt/v1",
                    "url": url,
                    "relative_path": destination.relative_to(root).as_posix(),
                    "bytes": expected_bytes,
                    "sha256": expected_sha256,
                    "hash_source": "existing_bytes_verified_against_frozen_sha256",
                    "download_attempts": 0,
                },
                paper2_root=root,
            )
        if destination.exists() or receipt.exists():
            if not destination.is_file() or not receipt.is_file():
                raise ValueError(f"partial immutable ISIS base-kernel asset: {destination}")
        else:
            download_fn(
                url=url,
                output=destination,
                receipt=receipt,
                paper2_root=root,
                max_bytes=1024 * 1024,
            )
        _verify_asset(
            destination,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
        )
        receipt_payload = read_json(receipt)
        if (
            receipt_payload.get("schema") != "paper2.official-download-receipt/v1"
            or receipt_payload.get("url") != url
            or receipt_payload.get("relative_path")
            != destination.relative_to(root).as_posix()
            or int(receipt_payload.get("bytes", -1)) != expected_bytes
            or receipt_payload.get("sha256") != expected_sha256
        ):
            raise ValueError("ISIS base-kernel download receipt differs")
        asset.update(
            {
                "receipt_relative_path": receipt.relative_to(root).as_posix(),
                "receipt_sha256": file_sha256(receipt),
            }
        )
        locked_assets.append(asset)
    locked = {
        "schema": ISIS_BASE_KERNEL_LOCK_SCHEMA,
        "isis_version": "10.0.0",
        "template_sha256": file_sha256(template_path),
        "assets": locked_assets,
        "downloads_complete": True,
    }
    if output_path.exists():
        if read_json(output_path) != locked:
            raise ValueError("existing immutable ISIS base-kernel lock differs")
    else:
        write_canonical_json(output_path, locked, paper2_root=root)
    return locked


def verify_isis_base_kernel_lock(
    *,
    paper2_root: Path,
    lock_path: Path = DEFAULT_LOCK,
) -> dict[str, Any]:
    root = paper2_root.resolve()
    lock_path = guard_paper2_path(lock_path, paper2_root=root)
    payload = read_json(lock_path)
    if (
        payload.get("schema") != ISIS_BASE_KERNEL_LOCK_SCHEMA
        or payload.get("isis_version") != "10.0.0"
        or payload.get("downloads_complete") is not True
    ):
        raise ValueError("ISIS base-kernel lock contract differs")
    template_path = guard_paper2_path(DEFAULT_TEMPLATE, paper2_root=root)
    if not template_path.is_file() or payload.get("template_sha256") != file_sha256(
        template_path
    ):
        raise ValueError("ISIS base-kernel template hash differs")
    template = read_json(template_path)
    if template.get("schema") != ISIS_BASE_KERNEL_TEMPLATE_SCHEMA:
        raise ValueError("ISIS base-kernel template schema differs")
    assets = payload.get("assets")
    if not isinstance(assets, list) or len(assets) != 1:
        raise ValueError("ISIS base-kernel lock must contain one asset")
    asset = dict(assets[0])
    expected_asset = dict(template["assets"][0])
    if any(asset.get(key) != value for key, value in expected_asset.items()):
        raise ValueError("ISIS base-kernel asset differs from the frozen template")
    path = guard_paper2_path(str(asset.get("relative_path", "")), paper2_root=root)
    _verify_asset(
        path,
        expected_bytes=int(asset.get("bytes", -1)),
        expected_sha256=str(asset.get("sha256", "")),
    )
    receipt = guard_paper2_path(
        str(asset.get("receipt_relative_path", "")), paper2_root=root
    )
    if not receipt.is_file() or asset.get("receipt_sha256") != file_sha256(receipt):
        raise ValueError("ISIS base-kernel receipt hash differs")
    receipt_payload = read_json(receipt)
    if (
        receipt_payload.get("url") != asset.get("url")
        or receipt_payload.get("relative_path") != asset.get("relative_path")
        or receipt_payload.get("sha256") != asset.get("sha256")
    ):
        raise ValueError("ISIS base-kernel receipt binding differs")
    return payload

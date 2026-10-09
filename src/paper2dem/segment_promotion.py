from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import re
import shutil

from .hashing import (
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    is_sha256,
    write_canonical_json,
)


SEGMENT_PREFIX_PROMOTION_PLAN_SCHEMA = "paper2.segment-prefix-promotion-plan/v1"
SEGMENT_PREFIX_PROMOTION_SCHEMA = "paper2.segment-prefix-promotion/v1"


def _pds3_prefix_identity(path: Path, *, expected_filename: str) -> dict[str, Any]:
    with path.open("rb") as stream:
        prefix = stream.read(1024 * 1024)

    def scalar(name: str) -> str:
        match = re.search(
            rb"(?m)^\s*" + re.escape(name.encode("ascii")) + rb"\s*=\s*([^\r\n]+)",
            prefix,
        )
        if match is None:
            raise ValueError(f"partial CTX EDR label lacks {name}")
        return match.group(1).decode("ascii", errors="strict").strip().strip('"')

    if scalar("PDS_VERSION_ID") != "PDS3" or scalar("RECORD_TYPE") != "FIXED_LENGTH":
        raise ValueError("partial CTX EDR is not an attached fixed-length PDS3 product")
    try:
        record_bytes = int(scalar("RECORD_BYTES"))
        file_records = int(scalar("FILE_RECORDS"))
        label_records = int(scalar("LABEL_RECORDS"))
        image_record = int(scalar("^IMAGE"))
    except ValueError as exc:
        raise ValueError("partial CTX EDR record declarations are not integers") from exc
    if min(record_bytes, file_records, label_records) <= 0:
        raise ValueError("partial CTX EDR record declarations must be positive")
    if image_record != label_records + 1:
        raise ValueError("partial CTX EDR image pointer does not follow its label")
    declared_filename = scalar("FILE_NAME")
    if declared_filename.upper() != expected_filename.upper():
        raise ValueError("partial CTX EDR FILE_NAME differs from the destination")
    label_extent = label_records * record_bytes
    if len(prefix) < label_extent:
        raise ValueError("partial CTX EDR ends inside its attached label")
    label = prefix[:label_extent].decode("ascii", errors="strict")
    if re.search(r"(?m)^\s*END\s*$", label) is None:
        raise ValueError("partial CTX EDR label lacks an END marker")
    return {
        "validator": "attached-pds3-prefix-identity/v1",
        "file_name": declared_filename,
        "record_bytes": record_bytes,
        "file_records": file_records,
        "label_records": label_records,
        "image_record": image_record,
        "declared_bytes": record_bytes * file_records,
    }


def _fragment_descriptors(directory: Path, *, asset_name: str) -> list[dict[str, Any]]:
    pattern = re.compile(
        rf"^\.{re.escape(asset_name)}\.range-(\d+)-(\d+)\.part$"
    )
    rows: list[dict[str, Any]] = []
    for path in directory.glob(f".{asset_name}.range-*.part"):
        match = pattern.fullmatch(path.name)
        if match is None or not path.is_file():
            raise ValueError(f"invalid segmented fragment path: {path}")
        start, end = int(match.group(1)), int(match.group(2))
        if end < start:
            raise ValueError(f"invalid segmented fragment interval: {path.name}")
        observed_bytes = path.stat().st_size
        declared_span_bytes = end - start + 1
        if observed_bytes > declared_span_bytes:
            raise ValueError(f"segmented fragment exceeds its interval: {path.name}")
        rows.append(
            {
                "path": path,
                "name": path.name,
                "start": start,
                "end": end,
                "declared_span_bytes": declared_span_bytes,
                "observed_bytes": observed_bytes,
                "sha256": file_sha256(path),
            }
        )
    rows.sort(key=lambda value: (int(value["start"]), int(value["end"])))
    return rows


def promote_contiguous_segment_prefix(
    *,
    paper2_root: str | Path,
    observation_id: str,
    url: str,
    expected_total_bytes: int,
    source_transport_module_sha256: str,
    source_driver_sha256: str,
    exited_writer_pid: int,
) -> dict[str, Any]:
    """Atomically preserve a verified contiguous prefix from interrupted range fragments."""

    if not re.fullmatch(r"[A-Z0-9]+(?:_[A-Z0-9]+)+", observation_id):
        raise ValueError("invalid CTX observation ID")
    if not url.startswith("https://") or not url.endswith(f"/{observation_id}.IMG"):
        raise ValueError("CTX promotion URL does not match the observation")
    if expected_total_bytes <= 0:
        raise ValueError("expected total bytes must be positive")
    if not is_sha256(source_transport_module_sha256):
        raise ValueError("source transport module identity must be a SHA-256 value")
    if not is_sha256(source_driver_sha256):
        raise ValueError("source driver identity must be a SHA-256 value")
    if exited_writer_pid <= 0:
        raise ValueError("exited writer PID must be positive")
    if Path(f"/proc/{exited_writer_pid}").exists():
        raise RuntimeError("segmented writer is still alive; promotion is unsafe")

    root = Path(paper2_root).resolve()
    directory = guard_paper2_path(
        Path("data/assets/ctx-edr") / observation_id, paper2_root=root
    )
    asset_name = f"{observation_id}.IMG"
    destination = guard_paper2_path(directory / asset_name, paper2_root=root)
    receipt = guard_paper2_path(directory / "download-receipt.json", paper2_root=root)
    partial = guard_paper2_path(
        directory / f".{asset_name}.download.part", paper2_root=root
    )
    if destination.exists() or receipt.exists():
        raise FileExistsError("completed CTX EDR or receipt already exists")
    if not partial.is_file() or partial.stat().st_size <= 0:
        raise FileNotFoundError("canonical CTX EDR partial is missing or empty")

    prefix_bytes = partial.stat().st_size
    prefix_sha256 = file_sha256(partial)
    pds3_identity = _pds3_prefix_identity(partial, expected_filename=asset_name)
    if int(pds3_identity["declared_bytes"]) != expected_total_bytes:
        raise ValueError("PDS3 declared bytes differ from the locked remote total")

    fragments = _fragment_descriptors(directory, asset_name=asset_name)
    if not fragments:
        raise FileNotFoundError("no segmented fragments are available for promotion")
    expected_start = prefix_bytes
    for fragment in fragments:
        if int(fragment["start"]) != expected_start:
            raise ValueError("segmented fragment plan does not form an exact partition")
        expected_start = int(fragment["end"]) + 1
    if expected_start != expected_total_bytes:
        raise ValueError("segmented fragment plan does not close the locked remote total")

    included: list[dict[str, Any]] = []
    cursor = prefix_bytes
    for fragment in fragments:
        observed_bytes = int(fragment["observed_bytes"])
        if observed_bytes == 0 or int(fragment["start"]) != cursor:
            break
        included.append(fragment)
        cursor += observed_bytes
        if observed_bytes != int(fragment["declared_span_bytes"]):
            break
    if not included or cursor <= prefix_bytes:
        raise ValueError("segmented fragments do not extend the canonical prefix")

    serializable_fragments = [
        {key: value for key, value in fragment.items() if key != "path"}
        for fragment in fragments
    ]
    plan_core = {
        "schema": SEGMENT_PREFIX_PROMOTION_PLAN_SCHEMA,
        "observation_id": observation_id,
        "url": url,
        "expected_total_bytes": expected_total_bytes,
        "source_transport_module_sha256": source_transport_module_sha256,
        "source_driver_sha256": source_driver_sha256,
        "exited_writer_pid": exited_writer_pid,
        "canonical_prefix": {
            "relative_path": partial.relative_to(root).as_posix(),
            "bytes": prefix_bytes,
            "sha256": prefix_sha256,
        },
        "pds3_identity": pds3_identity,
        "fragment_plan": serializable_fragments,
        "included_fragment_names": [str(value["name"]) for value in included],
        "promoted_prefix_bytes": cursor,
    }
    promotion_id = canonical_sha256(plan_core)
    archive = guard_paper2_path(
        directory / ".segmented-prefix-promotions" / promotion_id,
        paper2_root=root,
    )
    if archive.exists():
        raise FileExistsError(f"segmented prefix promotion archive already exists: {archive}")
    archive.mkdir(parents=True)
    plan = {**plan_core, "promotion_id": promotion_id}
    plan["content_sha256"] = canonical_sha256(plan)
    plan_path = archive / "promotion-plan.json"
    write_canonical_json(plan_path, plan, paper2_root=root)

    assembly = guard_paper2_path(
        directory / f".{asset_name}.promoted-assembly.part", paper2_root=root
    )
    with assembly.open("wb") as target:
        with partial.open("rb") as source:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        for fragment in included:
            with Path(fragment["path"]).open("rb") as source:
                shutil.copyfileobj(source, target, length=1024 * 1024)
        target.flush()
        os.fsync(target.fileno())
    if assembly.stat().st_size != cursor:
        raise ValueError("promoted segment assembly has the wrong byte length")
    promoted_sha256 = file_sha256(assembly)

    archived_prefix = archive / "original-canonical-prefix.part"
    shutil.copy2(partial, archived_prefix)
    if file_sha256(archived_prefix) != prefix_sha256:
        raise ValueError("archived canonical prefix hash differs")
    assembly.replace(partial)

    archived_fragments: list[dict[str, Any]] = []
    for fragment in fragments:
        source = Path(fragment["path"])
        archived = archive / source.name
        source.replace(archived)
        if (
            archived.stat().st_size != int(fragment["observed_bytes"])
            or file_sha256(archived) != fragment["sha256"]
        ):
            raise ValueError("archived segmented fragment differs after promotion")
        archived_fragments.append(
            {
                **{key: value for key, value in fragment.items() if key != "path"},
                "archived_relative_path": archived.relative_to(root).as_posix(),
            }
        )

    result = {
        "schema": SEGMENT_PREFIX_PROMOTION_SCHEMA,
        "promotion_id": promotion_id,
        "plan_relative_path": plan_path.relative_to(root).as_posix(),
        "plan_sha256": file_sha256(plan_path),
        "observation_id": observation_id,
        "url": url,
        "expected_total_bytes": expected_total_bytes,
        "source_transport_module_sha256": source_transport_module_sha256,
        "source_driver_sha256": source_driver_sha256,
        "exited_writer_pid": exited_writer_pid,
        "input_prefix_archive_relative_path": archived_prefix.relative_to(root).as_posix(),
        "input_prefix_bytes": prefix_bytes,
        "input_prefix_sha256": prefix_sha256,
        "archived_fragments": archived_fragments,
        "included_fragment_names": [str(value["name"]) for value in included],
        "output_prefix_relative_path": partial.relative_to(root).as_posix(),
        "output_prefix_bytes": partial.stat().st_size,
        "output_prefix_sha256": file_sha256(partial),
        "expected_output_prefix_sha256": promoted_sha256,
        "pds3_identity": pds3_identity,
    }
    if result["output_prefix_bytes"] != cursor or result["output_prefix_sha256"] != promoted_sha256:
        raise ValueError("published promoted prefix differs from its assembly")
    result["content_sha256"] = canonical_sha256(result)
    success = archive / "_SUCCESS.json"
    write_canonical_json(success, result, paper2_root=root)
    return {**result, "success_relative_path": success.relative_to(root).as_posix()}

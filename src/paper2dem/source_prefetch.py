from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import fcntl
import os
import re
import shutil
import time
import urllib.error
import urllib.request

from .hashing import (
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    is_sha256,
    read_json,
    write_canonical_json,
)
from .official_assets import (
    AssetIntegrityError,
    NetworkRetryExhaustedError,
    _retry_delay,
    _retryable_network_error,
    download_official_asset,
)


SOURCE_PREFETCH_SCHEMA = "paper2.g0.source-prefetch/v1"
SEGMENTED_RANGE_TRANSPORT_SCHEMA = "paper2.segmented-range-transport/v1"
_LEGACY_ASTROGEO_S3_PREFIX = "https://astrogeo-ard.s3-us-west-2.amazonaws.com/"
_CANONICAL_ASTROGEO_S3_PREFIX = "https://astrogeo-ard.s3.us-west-2.amazonaws.com/"


@contextmanager
def source_product_cache_lock(
    *,
    paper2_root: str | Path,
    product_id: str,
    blocking: bool,
    lock_directory: str | Path = "data/source-preparation/prefetch-locks",
):
    """Serialize every writer of one resumable DTM/ORTHO product cache."""

    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(product_id)):
        raise ValueError(f"product_id contains unsafe path characters: {product_id!r}")
    root = Path(paper2_root).resolve()
    locks = guard_paper2_path(lock_directory, paper2_root=root)
    lock_path = guard_paper2_path(
        locks / f"{product_id}.lock", paper2_root=root
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(stream.fileno(), operation)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def source_products_directory_for_assignment(
    *,
    paper2_root: str | Path,
    assignments_file_sha256: str,
    root_assignments_file_sha256: str,
) -> Path:
    """Resolve the immutable source namespace for a root or QA-replacement schedule."""

    if not is_sha256(assignments_file_sha256) or not is_sha256(
        root_assignments_file_sha256
    ):
        raise ValueError("source-prefetch assignment identities must be SHA-256 values")
    relative = (
        Path("data/source-preparation/products")
        if assignments_file_sha256 == root_assignments_file_sha256
        else Path("data/source-preparation/schedules")
        / assignments_file_sha256
        / "products"
    )
    return guard_paper2_path(relative, paper2_root=paper2_root)


@dataclass(frozen=True)
class SourcePrefetchTask:
    group_id: str
    product_id: str
    product_order: int
    assets: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "product_id": self.product_id,
            "product_order": self.product_order,
            "assets": [
                {"name": name, "url": url}
                for name, url in self.assets
            ],
        }


def _http_status(response: Any) -> int | None:
    status = getattr(response, "status", None)
    if status is not None:
        return int(status)
    getcode = getattr(response, "getcode", None)
    return int(getcode()) if callable(getcode) and getcode() is not None else None


def _parse_exact_content_range(
    value: object,
    *,
    expected_start: int,
    expected_end: int,
    expected_total: int | None,
) -> int:
    match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+)", str(value).strip())
    if match is None:
        raise AssetIntegrityError("segmented response lacks an exact Content-Range")
    start, end, total = (int(match.group(index)) for index in (1, 2, 3))
    if start != expected_start or end != expected_end:
        raise AssetIntegrityError(
            "segmented response range differs: "
            f"observed {start}-{end}, expected {expected_start}-{expected_end}"
        )
    if expected_total is not None and total != expected_total:
        raise AssetIntegrityError(
            f"segmented response total differs: observed {total}, expected {expected_total}"
        )
    if end >= total:
        raise AssetIntegrityError("segmented response range exceeds the remote total")
    return total


def _remote_range_identity(
    *,
    url: str,
    opener: Callable[..., Any],
    max_attempts: int,
    sleeper: Callable[[float], Any],
    request_timeout_seconds: float,
) -> dict[str, Any]:
    if max_attempts < 1:
        raise ValueError("segmented identity max_attempts must be positive")
    for attempt in range(1, max_attempts + 1):
        request = urllib.request.Request(
            url,
            headers={"Range": "bytes=0-0", "User-Agent": "paper2dem/0.1"},
        )
        try:
            with opener(request, timeout=request_timeout_seconds) as response:
                if _http_status(response) != 206:
                    raise AssetIntegrityError(
                        "segmented range probe did not return HTTP 206"
                    )
                headers = getattr(response, "headers", {})
                total = _parse_exact_content_range(
                    headers.get("Content-Range"),
                    expected_start=0,
                    expected_end=0,
                    expected_total=None,
                )
                body = response.read()
                if body is None or len(body) != 1:
                    raise AssetIntegrityError(
                        "segmented range probe did not return exactly one byte"
                    )
                return {
                    "remote_total_bytes": total,
                    "etag": headers.get("ETag"),
                    "last_modified": headers.get("Last-Modified"),
                    "probe_attempts": attempt,
                }
        except Exception as exc:
            if not _retryable_network_error(exc) or attempt == max_attempts:
                if _retryable_network_error(exc):
                    raise NetworkRetryExhaustedError(
                        f"segmented official asset identity probe {url}",
                        attempts=attempt,
                        last_error=exc,
                    ) from exc
                raise
            sleeper(
                _retry_delay(
                    attempt,
                    backoff_initial_seconds=1.0,
                    backoff_max_seconds=8.0,
                )
            )
    raise AssertionError("unreachable segmented identity retry loop")


def _segmented_spans(start: int, stop: int, count: int) -> tuple[tuple[int, int], ...]:
    if start < 0 or stop <= start:
        raise ValueError("segmented byte interval must be nonempty and nonnegative")
    actual_count = min(count, stop - start)
    base, extra = divmod(stop - start, actual_count)
    spans: list[tuple[int, int]] = []
    cursor = start
    for index in range(actual_count):
        length = base + (1 if index < extra else 0)
        spans.append((cursor, cursor + length - 1))
        cursor += length
    if cursor != stop:
        raise AssertionError("segmented spans do not close the requested interval")
    return tuple(spans)


def _download_exact_range(
    *,
    url: str,
    path: Path,
    start: int,
    end: int,
    identity: Mapping[str, Any],
    opener: Callable[..., Any],
    max_attempts: int,
    sleeper: Callable[[float], Any],
    request_timeout_seconds: float,
    reuse_existing_fragment: bool = False,
) -> dict[str, Any]:
    expected_bytes = end - start + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise AssetIntegrityError("segmented fragment is not a regular file")
    if reuse_existing_fragment:
        path.touch(exist_ok=True)
    else:
        # General DTM/ORTHO transfers lack a catalog-published content digest,
        # so only their canonical sequential prefix survives process restarts.
        with path.open("wb"):
            pass
    initial_bytes = path.stat().st_size
    if initial_bytes > expected_bytes:
        raise AssetIntegrityError("segmented fragment exceeds its declared span")
    for attempt in range(1, max_attempts + 1):
        observed = path.stat().st_size
        if observed > expected_bytes:
            raise AssetIntegrityError("segmented fragment exceeds its declared span")
        if observed == expected_bytes:
            return {
                "start": start,
                "end": end,
                "bytes": observed,
                "attempts": attempt - 1,
                "initial_bytes": initial_bytes,
            }
        request_start = start + observed
        headers = {
            "Range": f"bytes={request_start}-{end}",
            "User-Agent": "paper2dem/0.1",
        }
        etag = identity.get("etag")
        last_modified = identity.get("last_modified")
        if etag and not str(etag).startswith("W/"):
            headers["If-Match"] = str(etag)
        elif last_modified:
            headers["If-Unmodified-Since"] = str(last_modified)
        request = urllib.request.Request(url, headers=headers)
        try:
            with opener(request, timeout=request_timeout_seconds) as response:
                if _http_status(response) != 206:
                    raise AssetIntegrityError(
                        "segmented asset request did not return HTTP 206"
                    )
                response_headers = getattr(response, "headers", {})
                _parse_exact_content_range(
                    response_headers.get("Content-Range"),
                    expected_start=request_start,
                    expected_end=end,
                    expected_total=int(identity["remote_total_bytes"]),
                )
                response_etag = response_headers.get("ETag")
                response_modified = response_headers.get("Last-Modified")
                if etag and response_etag and str(response_etag) != str(etag):
                    raise AssetIntegrityError("segmented asset ETag changed during transfer")
                if (
                    last_modified
                    and response_modified
                    and str(response_modified) != str(last_modified)
                ):
                    raise AssetIntegrityError(
                        "segmented asset Last-Modified changed during transfer"
                    )
                content_length = response_headers.get("Content-Length")
                requested_bytes = end - request_start + 1
                if content_length is not None and int(content_length) != requested_bytes:
                    raise AssetIntegrityError(
                        "segmented Content-Length differs from requested range"
                    )
                downloaded = 0
                with path.open("ab") as stream:
                    try:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            downloaded += len(chunk)
                            if observed + downloaded > expected_bytes:
                                raise AssetIntegrityError(
                                    "segmented response exceeded the requested range"
                                )
                            stream.write(chunk)
                    finally:
                        stream.flush()
                        os.fsync(stream.fileno())
                if downloaded != requested_bytes or path.stat().st_size != expected_bytes:
                    raise AssetIntegrityError(
                        "segmented response byte count did not close its range"
                    )
                return {
                    "start": start,
                    "end": end,
                    "bytes": expected_bytes,
                    "attempts": attempt,
                    "initial_bytes": initial_bytes,
                }
        except Exception as exc:
            if not _retryable_network_error(exc) or attempt == max_attempts:
                if _retryable_network_error(exc):
                    raise NetworkRetryExhaustedError(
                        f"segmented official asset range {start}-{end}",
                        attempts=attempt,
                        last_error=exc,
                    ) from exc
                raise
            sleeper(
                _retry_delay(
                    attempt,
                    backoff_initial_seconds=1.0,
                    backoff_max_seconds=8.0,
                )
            )
    raise AssertionError("unreachable segmented range retry loop")


def download_official_asset_segmented(
    *,
    url: str,
    output: str | Path,
    receipt: str | Path,
    paper2_root: str | Path,
    max_attempts: int,
    repair_invalid_existing: bool,
    resume_partial: bool,
    validate_attached_pds3: bool = False,
    segment_count: int = 4,
    request_timeout_seconds: float = 120.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
    sleeper: Callable[[float], Any] = time.sleep,
    publisher: DownloadFunction = download_official_asset,
    reuse_existing_fragments: bool = False,
    expected_content_md5: str | None = None,
) -> Mapping[str, Any]:
    """Prefill one official asset with exact concurrent ranges, then publish normally.

    The normal downloader remains the sole publisher of the final file and its
    official receipt. This helper only accelerates the cache-only transport and
    records enough range identity evidence to audit how those bytes arrived.
    """

    if not 2 <= segment_count <= 8:
        raise ValueError("segmented source-prefetch count must be in [2,8]")
    if max_attempts < 1:
        raise ValueError("segmented source-prefetch max_attempts must be positive")
    if request_timeout_seconds <= 0:
        raise ValueError("segmented request timeout must be positive")
    if not resume_partial or not repair_invalid_existing:
        raise ValueError("segmented source-prefetch requires repairable resumable mode")
    if not str(url).lower().startswith("https://"):
        raise ValueError("segmented source-prefetch URL must use HTTPS")
    normalized_content_md5 = (
        str(expected_content_md5).lower() if expected_content_md5 is not None else None
    )
    if reuse_existing_fragments and (
        not validate_attached_pds3
        or normalized_content_md5 is None
        or re.fullmatch(r"[0-9a-f]{32}", normalized_content_md5) is None
    ):
        raise ValueError(
            "cross-process fragment reuse requires attached PDS3 and an exact content MD5"
        )
    root = Path(paper2_root).resolve()
    destination = guard_paper2_path(output, paper2_root=root)
    receipt_path = guard_paper2_path(receipt, paper2_root=root)
    if destination.is_file() and receipt_path.is_file():
        return publisher(
            url=url,
            output=destination,
            receipt=receipt_path,
            paper2_root=root,
            max_attempts=max_attempts,
            validate_attached_pds3=validate_attached_pds3,
            request_timeout_seconds=request_timeout_seconds,
            repair_invalid_existing=True,
            resume_partial=True,
            opener=opener,
            sleeper=sleeper,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = guard_paper2_path(
        destination.parent / f".{destination.name}.download.part", paper2_root=root
    )
    identity = _remote_range_identity(
        url=url,
        opener=opener,
        max_attempts=max_attempts,
        sleeper=sleeper,
        request_timeout_seconds=request_timeout_seconds,
    )
    total = int(identity["remote_total_bytes"])
    if reuse_existing_fragments:
        identity_etag = str(identity.get("etag") or "").strip('"').lower()
        if identity_etag != normalized_content_md5:
            raise AssetIntegrityError(
                "fragment-reuse identity ETag differs from the expected content MD5"
            )
    prefix_bytes = partial.stat().st_size if partial.is_file() else 0
    if prefix_bytes > total:
        raise AssetIntegrityError(
            f"segmented canonical prefix exceeds remote total: {prefix_bytes} > {total}"
        )
    range_audit: tuple[dict[str, Any], ...] = ()
    if prefix_bytes < total:
        spans = _segmented_spans(prefix_bytes, total, segment_count)
        fragment_paths = tuple(
            guard_paper2_path(
                destination.parent / f".{destination.name}.range-{start}-{end}.part",
                paper2_root=root,
            )
            for start, end in spans
        )
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=len(spans)) as executor:
            futures = {
                executor.submit(
                    _download_exact_range,
                    url=url,
                    path=path,
                    start=start,
                    end=end,
                    identity=identity,
                    opener=opener,
                    max_attempts=max_attempts,
                    sleeper=sleeper,
                    request_timeout_seconds=request_timeout_seconds,
                    reuse_existing_fragment=reuse_existing_fragments,
                ): (start, end)
                for (start, end), path in zip(spans, fragment_paths, strict=True)
            }
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda value: int(value["start"]))
        range_audit = tuple(results)
        assembly = guard_paper2_path(
            destination.parent / f".{destination.name}.segmented-assembly.part",
            paper2_root=root,
        )
        with assembly.open("wb") as target:
            if prefix_bytes:
                with partial.open("rb") as source:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
            for fragment in fragment_paths:
                with fragment.open("rb") as source:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        if assembly.stat().st_size != total:
            raise AssetIntegrityError("segmented assembly does not equal the remote total")
        assembly.replace(partial)
        for fragment in destination.parent.glob(f".{destination.name}.range-*.part"):
            fragment.unlink(missing_ok=True)
    assembled_sha256 = file_sha256(partial)
    payload = dict(
        publisher(
            url=url,
            output=destination,
            receipt=receipt_path,
            paper2_root=root,
            max_attempts=max_attempts,
            validate_attached_pds3=validate_attached_pds3,
            request_timeout_seconds=request_timeout_seconds,
            repair_invalid_existing=True,
            resume_partial=True,
            expected_sha256=assembled_sha256,
            preverified_remote_total_bytes=total,
            opener=opener,
            sleeper=sleeper,
        )
    )
    if int(payload.get("bytes", -1)) != total or payload.get("sha256") != file_sha256(
        destination
    ):
        raise AssetIntegrityError("segmented asset did not close through normal publication")
    if reuse_existing_fragments and payload.get("md5") != normalized_content_md5:
        raise AssetIntegrityError(
            "fragment-reused asset differs from the catalog-published content MD5"
        )
    transport_audit = {
        "schema": SEGMENTED_RANGE_TRANSPORT_SCHEMA,
        "url": url,
        "relative_path": destination.relative_to(root).as_posix(),
        "official_receipt_relative_path": receipt_path.relative_to(root).as_posix(),
        "official_receipt_sha256": file_sha256(receipt_path),
        "segment_count": segment_count,
        "prefix_bytes": prefix_bytes,
        "remote_total_bytes": total,
        "etag": identity.get("etag"),
        "last_modified": identity.get("last_modified"),
        "identity_probe_attempts": int(identity["probe_attempts"]),
        "validate_attached_pds3": validate_attached_pds3,
        "reuse_existing_fragments": reuse_existing_fragments,
        "expected_content_md5": normalized_content_md5,
        "request_timeout_seconds": request_timeout_seconds,
        "ranges": list(range_audit),
        "assembled_sha256": assembled_sha256,
    }
    transport_audit["content_sha256"] = canonical_sha256(transport_audit)
    audit_path = destination.with_name(f"{destination.stem}.segmented-range.json")
    write_canonical_json(audit_path, transport_audit, paper2_root=root)
    return payload


def _safe_identifier(value: object, *, name: str) -> str:
    result = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", result):
        raise ValueError(f"{name} contains unsafe path characters: {result!r}")
    return result


def canonical_source_transport_url(url: object) -> str:
    value = str(url)
    if not value.lower().startswith("https://"):
        raise ValueError("source-prefetch URL must use HTTPS")
    if value.startswith(_LEGACY_ASTROGEO_S3_PREFIX):
        return _CANONICAL_ASTROGEO_S3_PREFIX + value[len(_LEGACY_ASTROGEO_S3_PREFIX) :]
    return value


def _task_assets(row: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    raw_assets = row.get("assets")
    if not isinstance(raw_assets, Mapping):
        raise ValueError("source-prefetch catalog row lacks assets")
    dem_key = next(
        (name for name in ("geoid_adjusted_dem", "dtm", "dem") if raw_assets.get(name)),
        None,
    )
    if dem_key is None or not raw_assets.get("orthoimage"):
        raise ValueError("source-prefetch requires a controlled DTM and ORTHO")
    return (
        ("controlled-dtm", canonical_source_transport_url(raw_assets[dem_key])),
        ("coreg-aux-ortho", canonical_source_transport_url(raw_assets["orthoimage"])),
    )


def _asset_cache_complete(cache: Path, *, name: str, url: str) -> bool:
    output = cache / f"{name}.tif"
    receipt = cache / f"{name}.receipt.json"
    if not output.is_file() or not receipt.is_file():
        return False
    try:
        payload = read_json(receipt)
        return bool(
            payload.get("schema") == "paper2.official-download-receipt/v1"
            and payload.get("url") == url
            and int(payload.get("bytes", -1)) == output.stat().st_size
            and int(payload.get("remote_total_bytes", -1)) == output.stat().st_size
            and payload.get("sha256") == file_sha256(output)
            and payload.get("resume_partial") is True
        )
    except (OSError, TypeError, ValueError):
        return False


def _task_cache_complete(task: SourcePrefetchTask, cache_root: Path) -> bool:
    cache = cache_root / task.product_id
    return all(
        _asset_cache_complete(cache, name=name, url=url)
        for name, url in task.assets
    )


def select_tail_source_prefetch_tasks(
    *,
    assignments: Sequence[Mapping[str, Any]],
    catalog_rows: Mapping[str, Mapping[str, Any]],
    products_directory: str | Path,
    source_cache_directory: str | Path,
    maximum_groups: int,
    minimum_assignment_gap: int = 16,
    resume_existing_partials: bool = False,
) -> tuple[SourcePrefetchTask, ...]:
    """Select only the far tail of the formal serial source frontier.

    The active source-preparation process traverses ``assignments`` in order. Selecting the
    farthest incomplete groups prevents this cache-only helper from racing the live product.
    Actual source manifests remain the authority: prefetched bytes never select a product or
    satisfy G0 by themselves.
    """

    if maximum_groups < 1:
        raise ValueError("maximum_groups must be positive")
    if minimum_assignment_gap < 0:
        raise ValueError("minimum_assignment_gap cannot be negative")
    products_root = Path(products_directory)
    cache_root = Path(source_cache_directory)
    incomplete: list[tuple[int, SourcePrefetchTask | None]] = []
    seen_groups: set[str] = set()
    for assignment_index, raw_assignment in enumerate(assignments):
        assignment = dict(raw_assignment)
        group_id = _safe_identifier(assignment.get("group_id", ""), name="group_id")
        if group_id in seen_groups:
            raise ValueError("source-prefetch assignments contain duplicate groups")
        seen_groups.add(group_id)
        products = [
            _safe_identifier(assignment.get("representative_product_id", ""), name="product_id"),
            *(
                _safe_identifier(value, name="product_id")
                for value in assignment.get("fallback_product_ids", [])
            ),
        ]
        pending: SourcePrefetchTask | None = None
        source_ready = False
        for product_order, product_id in enumerate(products):
            row = catalog_rows.get(product_id)
            if row is None:
                raise ValueError(f"source-prefetch catalog lacks product: {product_id}")
            product_directory = products_root / product_id
            manifest = product_directory / "source-prepared-manifest.json"
            failure = product_directory / "source-preparation-failure.json"
            if manifest.exists() and failure.exists():
                raise ValueError(f"source product has both terminal receipts: {product_id}")
            if manifest.is_file():
                source_ready = True
                break
            if failure.is_file():
                continue
            cache_directory = cache_root / product_id
            if (
                any(cache_directory.glob("*.download.part"))
                and not resume_existing_partials
            ):
                # This is either the live serial product or another resumable owner. The
                # prefetcher never opens the same partial bytes concurrently.
                break
            task = SourcePrefetchTask(
                group_id=group_id,
                product_id=product_id,
                product_order=product_order,
                assets=_task_assets(row),
            )
            if not _task_cache_complete(task, cache_root):
                pending = task
            break
        if not source_ready:
            incomplete.append((assignment_index, pending))
    if not incomplete:
        return ()
    frontier_index = incomplete[0][0]
    candidates = [
        task
        for assignment_index, task in incomplete
        if task is not None
        and assignment_index >= frontier_index + minimum_assignment_gap
    ]
    return tuple(candidates[-maximum_groups:])


DownloadFunction = Callable[..., Mapping[str, Any]]


def _prefetch_task(
    task: SourcePrefetchTask,
    *,
    paper2_root: Path,
    source_cache_directory: Path,
    lock_directory: Path,
    max_attempts: int,
    resume_existing_partials: bool,
    downloader: DownloadFunction,
) -> dict[str, Any]:
    cache = guard_paper2_path(
        source_cache_directory / task.product_id, paper2_root=paper2_root
    )
    cache.mkdir(parents=True, exist_ok=True)
    with source_product_cache_lock(
        paper2_root=paper2_root,
        product_id=task.product_id,
        blocking=False,
        lock_directory=lock_directory,
    ) as acquired:
        if not acquired:
            return {**task.to_dict(), "state": "skipped_locked"}
        if any(cache.glob("*.download.part")) and not resume_existing_partials:
            return {**task.to_dict(), "state": "skipped_active_partial"}
        artifacts: list[dict[str, Any]] = []
        try:
            for asset_name, url in task.assets:
                output = cache / f"{asset_name}.tif"
                receipt = cache / f"{asset_name}.receipt.json"
                payload = dict(
                    downloader(
                        url=url,
                        output=output,
                        receipt=receipt,
                        paper2_root=paper2_root,
                        max_attempts=max_attempts,
                        repair_invalid_existing=True,
                        resume_partial=True,
                    )
                )
                artifact = {
                        "name": asset_name,
                        "url": url,
                        "relative_path": output.relative_to(paper2_root).as_posix(),
                        "sha256": file_sha256(output),
                        "bytes": output.stat().st_size,
                        "receipt_relative_path": receipt.relative_to(paper2_root).as_posix(),
                        "receipt_sha256": file_sha256(receipt),
                        "download_receipt_content_sha256": canonical_sha256(payload),
                    }
                segmented_audit = output.with_name(
                    f"{output.stem}.segmented-range.json"
                )
                if segmented_audit.is_file():
                    audit_payload = read_json(segmented_audit)
                    if (
                        audit_payload.get("schema")
                        != SEGMENTED_RANGE_TRANSPORT_SCHEMA
                        or audit_payload.get("url") != url
                        or audit_payload.get("relative_path")
                        != output.relative_to(paper2_root).as_posix()
                        or audit_payload.get("assembled_sha256")
                        != artifact["sha256"]
                        or audit_payload.get("official_receipt_sha256")
                        != artifact["receipt_sha256"]
                        or audit_payload.get("content_sha256")
                        != canonical_sha256(
                            {
                                key: value
                                for key, value in audit_payload.items()
                                if key != "content_sha256"
                            }
                        )
                    ):
                        raise ValueError(
                            f"segmented source-prefetch audit does not verify: {segmented_audit}"
                        )
                    artifact.update(
                        {
                            "segmented_range_audit_relative_path": segmented_audit.relative_to(
                                paper2_root
                            ).as_posix(),
                            "segmented_range_audit_sha256": file_sha256(segmented_audit),
                            "segmented_range_audit_content_sha256": audit_payload[
                                "content_sha256"
                            ],
                        }
                    )
                artifacts.append(artifact)
        except Exception as exc:  # keep resumable bytes and write an auditable batch result
            return {
                **task.to_dict(),
                "state": "incomplete",
                "completed_assets": artifacts,
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }
        return {**task.to_dict(), "state": "complete", "artifacts": artifacts}


def prefetch_source_assets(
    *,
    tasks: Sequence[SourcePrefetchTask],
    paper2_root: str | Path,
    source_cache_directory: str | Path = "data/source-preparation/source-cache",
    lock_directory: str | Path = "data/source-preparation/prefetch-locks",
    workers: int = 2,
    max_attempts: int = 16,
    resume_existing_partials: bool = False,
    downloader: DownloadFunction = download_official_asset,
) -> dict[str, Any]:
    root = Path(paper2_root).resolve()
    cache_root = guard_paper2_path(source_cache_directory, paper2_root=root)
    locks = guard_paper2_path(lock_directory, paper2_root=root)
    if not 1 <= workers <= 4:
        raise ValueError("source-prefetch workers must be in [1,4]")
    if max_attempts < 1:
        raise ValueError("source-prefetch max_attempts must be positive")
    identities = [(task.group_id, task.product_id) for task in tasks]
    if len(identities) != len(set(identities)):
        raise ValueError("source-prefetch tasks must be unique")
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(workers, max(len(tasks), 1))) as executor:
        futures = {
            executor.submit(
                _prefetch_task,
                task,
                paper2_root=root,
                source_cache_directory=cache_root,
                lock_directory=locks,
                max_attempts=max_attempts,
                resume_existing_partials=resume_existing_partials,
                downloader=downloader,
            ): task
            for task in tasks
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda value: (str(value["group_id"]), str(value["product_id"])))
    complete = sum(value["state"] == "complete" for value in results)
    payload: dict[str, Any] = {
        "schema": SOURCE_PREFETCH_SCHEMA,
        "state": "complete" if complete == len(results) else "incomplete",
        "cache_only": True,
        "satisfies_g0": False,
        "reference_values_accessed": False,
        "model_outcomes_accessed": False,
        "worker_count": workers,
        "resume_existing_partials": bool(resume_existing_partials),
        "task_count": len(tasks),
        "complete_task_count": complete,
        "tasks_sha256": canonical_sha256([task.to_dict() for task in tasks]),
        "results": results,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def write_source_prefetch_receipt(
    path: str | Path, payload: Mapping[str, Any], *, paper2_root: str | Path
) -> Path:
    return write_canonical_json(path, payload, paper2_root=paper2_root)

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from . import official_assets as official_assets_module
from .hashing import (
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    read_json,
    write_canonical_json,
)
from .official_assets import (
    AssetIntegrityError,
    CTX_EDR_DATA_SET_ID,
    NetworkRetryExhaustedError,
    download_official_asset,
    file_md5,
    validate_attached_pds3_fixed_length,
)


USGS_CTX_ARCHIVE_PAGE = "https://pdsimage2.wr.usgs.gov/Mars_Reconnaissance_Orbiter/CTX/"
USGS_CTX_CLOUDFRONT_BASE = "https://d32ky7zsovnyu5.cloudfront.net/CTX"
USGS_CTX_S3_BASE = "https://asc-pds-mars-reconnaissance-orbiter.s3.us-west-2.amazonaws.com/CTX"
USGS_CTX_TRANSPORT_NAMESPACE = "s3-us-west-2"
OFFICIAL_CTX_CATALOG_HOSTS = frozenset({"planetarydata.jpl.nasa.gov", "pds-imaging.jpl.nasa.gov"})
MIRROR_EVIDENCE_SCHEMA = "paper2.ctx-edr-official-mirror-evidence/v1"
MIRROR_AUDIT_SCHEMA = "paper2.ctx-edr-official-mirror-transport/v1"


@dataclass(frozen=True)
class CTXMirrorLocation:
    observation_id: str
    volume_id: str
    catalog_url: str
    mirror_url: str
    checksum_url: str
    checksum_entry_path: str


def derive_usgs_ctx_mirror(catalog_url: str, observation_id: str) -> CTXMirrorLocation:
    """Map one locked PDS Imaging CTX URL to the current official USGS archive."""

    requested = observation_id.strip().upper()
    if not re.fullmatch(r"[A-Z0-9]+(?:_[A-Z0-9]+)+", requested):
        raise ValueError(f"invalid CTX observation ID: {observation_id!r}")
    parsed = urllib.parse.urlparse(catalog_url)
    if parsed.scheme.lower() != "https" or (parsed.hostname or "").lower() not in (
        OFFICIAL_CTX_CATALOG_HOSTS
    ):
        raise ValueError("CTX catalog URL is not a frozen official PDS Imaging HTTPS host")
    match = re.search(
        r"/ctx/(?P<volume>mrox_\d{4})/data/(?P<filename>[^/]+\.img)$",
        parsed.path,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError("CTX catalog URL does not contain an exact volume data path")
    volume = match.group("volume").lower()
    filename = match.group("filename")
    if filename.upper() != f"{requested}.IMG":
        raise ValueError("CTX catalog filename differs from the requested observation")
    canonical_filename = f"{requested}.IMG"
    return CTXMirrorLocation(
        observation_id=requested,
        volume_id=volume,
        catalog_url=catalog_url,
        mirror_url=f"{USGS_CTX_S3_BASE}/{volume}/data/{canonical_filename}",
        checksum_url=f"{USGS_CTX_S3_BASE}/{volume}_md5.txt",
        checksum_entry_path=f"{volume}/data/{canonical_filename}",
    )


def parse_volume_md5_manifest(raw: bytes, expected_path: str) -> str:
    """Return the unique archive-published MD5 for an exact relative path."""

    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise AssetIntegrityError("USGS CTX volume checksum manifest is not UTF-8") from exc
    matches: list[str] = []
    for line in text.splitlines():
        match = re.fullmatch(r"\s*([0-9a-fA-F]{32})\s+\*?(.+?)\s*", line)
        if match is not None and match.group(2) == expected_path:
            matches.append(match.group(1).lower())
    if len(matches) != 1:
        raise AssetIntegrityError(
            f"expected one checksum for {expected_path!r}, found {len(matches)}"
        )
    return matches[0]


def probe_exact_range_identity(
    url: str,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
    max_attempts: int = 4,
    request_timeout_seconds: float = 60.0,
    sleeper: Callable[[float], Any] = time.sleep,
) -> dict[str, Any]:
    """Probe one byte and freeze total length plus stable HTTP object identity."""

    if not url.lower().startswith("https://"):
        raise ValueError("mirror probe URL must use HTTPS")
    if max_attempts < 1 or request_timeout_seconds <= 0:
        raise ValueError("mirror probe retry and timeout values must be positive")
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        request = urllib.request.Request(
            url,
            headers={"Range": "bytes=0-0", "User-Agent": "paper2dem/0.1"},
        )
        try:
            with opener(request, timeout=request_timeout_seconds) as response:
                status = getattr(response, "status", None)
                if status is None and callable(getattr(response, "getcode", None)):
                    status = response.getcode()
                if status != 206:
                    raise AssetIntegrityError("official mirror probe did not return HTTP 206")
                headers = getattr(response, "headers", {})
                content_range = str(headers.get("Content-Range", "")).strip()
                match = re.fullmatch(r"bytes\s+0-0/(\d+)", content_range)
                if match is None or int(match.group(1)) <= 1:
                    raise AssetIntegrityError("official mirror probe has an invalid Content-Range")
                body = response.read(2)
                if body is None or len(body) != 1:
                    raise AssetIntegrityError(
                        "official mirror probe did not return exactly one byte"
                    )
                return {
                    "probe_attempts": attempt,
                    "remote_total_bytes": int(match.group(1)),
                    "etag": headers.get("ETag"),
                    "last_modified": headers.get("Last-Modified"),
                    "content_range": content_range,
                }
        except Exception as exc:
            last_error = exc
            if isinstance(exc, urllib.error.HTTPError) and not (
                exc.code in {408, 425, 429} or exc.code >= 500
            ):
                raise
            if attempt == max_attempts:
                break
            sleeper(min(2 ** (attempt - 1), 8))
    assert last_error is not None
    raise NetworkRetryExhaustedError(
        f"official USGS CTX mirror identity probe {url}",
        attempts=max_attempts,
        last_error=last_error,
    ) from last_error


def _normalized_etag(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    if text.startswith("W/"):
        return None
    normalized = text.strip('"').lower()
    return normalized if re.fullmatch(r"[0-9a-f]{32}", normalized) else None


def _closed_mirror_result(
    *,
    root: Path,
    location: CTXMirrorLocation,
    destination: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Revalidate and summarize one already-published official-mirror asset."""

    audit_path = guard_paper2_path(
        Path("data/assets/ctx-usgs-mirror/observations")
        / location.observation_id
        / "transport-audit.json",
        paper2_root=root,
    )
    if not destination.is_file() or not receipt_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError("closed official-mirror CTX triple is incomplete")
    receipt = read_json(receipt_path)
    audit = read_json(audit_path)
    if not isinstance(receipt, Mapping) or not isinstance(audit, Mapping):
        raise ValueError("official-mirror receipt or audit is not an object")
    observed_bytes = destination.stat().st_size
    observed_sha256 = file_sha256(destination)
    observed_md5 = file_md5(destination)
    integrity = validate_attached_pds3_fixed_length(
        destination, expected_filename=destination.name
    )
    evidence = receipt.get("transport_evidence")
    audit_evidence = audit.get("transport_evidence")
    audit_core = {key: value for key, value in audit.items() if key != "content_sha256"}
    if (
        receipt.get("url") != location.catalog_url
        or receipt.get("transport_url") != location.mirror_url
        or receipt.get("relative_path") != destination.relative_to(root).as_posix()
        or receipt.get("bytes") != observed_bytes
        or receipt.get("sha256") != observed_sha256
        or receipt.get("md5") != observed_md5
        or receipt.get("expected_md5") != observed_md5
        or receipt.get("integrity") != integrity
        or not isinstance(evidence, Mapping)
        or evidence != audit_evidence
        or evidence.get("expected_md5") != observed_md5
        or audit.get("location") != asdict(location)
        or audit.get("asset_relative_path") != destination.relative_to(root).as_posix()
        or audit.get("asset_bytes") != observed_bytes
        or audit.get("asset_sha256") != observed_sha256
        or audit.get("asset_md5") != observed_md5
        or audit.get("receipt_relative_path") != receipt_path.relative_to(root).as_posix()
        or audit.get("receipt_sha256") != file_sha256(receipt_path)
        or audit.get("content_sha256") != canonical_sha256(audit_core)
    ):
        raise AssetIntegrityError("existing official-mirror CTX triple does not verify")
    return {
        "observation_id": location.observation_id,
        "status": "complete",
        "catalog_url": location.catalog_url,
        "transport_url": location.mirror_url,
        "asset_relative_path": destination.relative_to(root).as_posix(),
        "bytes": observed_bytes,
        "sha256": observed_sha256,
        "md5": observed_md5,
        "receipt_relative_path": receipt_path.relative_to(root).as_posix(),
        "receipt_sha256": file_sha256(receipt_path),
        "transport_audit_relative_path": audit_path.relative_to(root).as_posix(),
        "transport_audit_sha256": file_sha256(audit_path),
        "transport_audit_content_sha256": str(audit["content_sha256"]),
        "integrity": integrity,
    }


def download_ctx_edr_via_usgs_mirror(
    *,
    observation_id: str,
    catalog_url: str,
    paper2_root: str | Path,
    max_attempts: int = 8,
    request_timeout_seconds: float = 120.0,
    segment_count: int = 1,
    opener: Callable[..., Any] = urllib.request.urlopen,
    sleeper: Callable[[float], Any] = time.sleep,
) -> dict[str, Any]:
    """Resume one CTX EDR through USGS and retain catalog plus transport lineage."""

    root = Path(paper2_root).resolve()
    if not 1 <= segment_count <= 4:
        raise ValueError("official mirror segment count must be in [1,4]")
    location = derive_usgs_ctx_mirror(catalog_url, observation_id)
    directory = guard_paper2_path(
        Path("data/assets/ctx-edr") / location.observation_id,
        paper2_root=root,
    )
    destination = directory / f"{location.observation_id}.IMG"
    receipt_path = directory / "download-receipt.json"
    try:
        return _closed_mirror_result(
            root=root,
            location=location,
            destination=destination,
            receipt_path=receipt_path,
        )
    except (OSError, ValueError, KeyError, TypeError, AssetIntegrityError):
        pass
    checksum_directory = guard_paper2_path(
        Path("data/assets/ctx-usgs-mirror/volume-checksums")
        / USGS_CTX_TRANSPORT_NAMESPACE
        / location.volume_id,
        paper2_root=root,
    )
    checksum_path = checksum_directory / f"{location.volume_id}_md5.txt"
    checksum_receipt_path = checksum_directory / "download-receipt.json"
    checksum_receipt = download_official_asset(
        url=location.checksum_url,
        output=checksum_path,
        receipt=checksum_receipt_path,
        paper2_root=root,
        opener=opener,
        max_attempts=max_attempts,
        request_timeout_seconds=request_timeout_seconds,
        sleeper=sleeper,
    )
    expected_md5 = parse_volume_md5_manifest(
        checksum_path.read_bytes(), location.checksum_entry_path
    )
    identity = probe_exact_range_identity(
        location.mirror_url,
        opener=opener,
        max_attempts=max_attempts,
        request_timeout_seconds=request_timeout_seconds,
        sleeper=sleeper,
    )
    if _normalized_etag(identity.get("etag")) != expected_md5:
        raise AssetIntegrityError(
            "official USGS mirror ETag differs from its published volume checksum"
        )
    evidence = {
        "schema": MIRROR_EVIDENCE_SCHEMA,
        "dataset_id": CTX_EDR_DATA_SET_ID,
        "official_archive_page": USGS_CTX_ARCHIVE_PAGE,
        "official_distribution_base": USGS_CTX_CLOUDFRONT_BASE,
        "official_s3_archive_base": USGS_CTX_S3_BASE,
        "transport_namespace": USGS_CTX_TRANSPORT_NAMESPACE,
        "catalog_url": location.catalog_url,
        "transport_url": location.mirror_url,
        "volume_id": location.volume_id,
        "checksum_url": location.checksum_url,
        "checksum_entry_path": location.checksum_entry_path,
        "expected_md5": expected_md5,
        "checksum_relative_path": checksum_path.relative_to(root).as_posix(),
        "checksum_sha256": file_sha256(checksum_path),
        "checksum_receipt_relative_path": checksum_receipt_path.relative_to(root).as_posix(),
        "checksum_receipt_sha256": file_sha256(checksum_receipt_path),
        "checksum_receipt_content": checksum_receipt,
        "range_identity": identity,
        "segment_count": segment_count,
        "module_sha256": file_sha256(Path(__file__).resolve()),
        "official_assets_module_sha256": file_sha256(
            Path(official_assets_module.__file__).resolve()
        ),
    }
    publication = {
        "url": location.catalog_url,
        "transport_url": location.mirror_url,
        "transport_evidence": evidence,
        "expected_md5": expected_md5,
    }
    common_download = {
        "output": destination,
        "receipt": receipt_path,
        "paper2_root": root,
        "opener": opener,
        "max_attempts": max_attempts,
        "validate_attached_pds3": True,
        "repair_invalid_existing": True,
        "resume_partial": True,
        "request_timeout_seconds": request_timeout_seconds,
        "sleeper": sleeper,
    }
    if segment_count == 1:
        receipt = download_official_asset(
            **publication,
            **common_download,
            preverified_remote_total_bytes=int(identity["remote_total_bytes"]),
        )
    else:
        from .source_prefetch import download_official_asset_segmented

        def catalog_publisher(**kwargs: Any) -> Mapping[str, Any]:
            transport_url = str(kwargs.pop("url"))
            if transport_url != location.mirror_url:
                raise ValueError("segmented publisher received a different mirror URL")
            return download_official_asset(**publication, **kwargs)

        receipt = download_official_asset_segmented(
            url=location.mirror_url,
            segment_count=segment_count,
            publisher=catalog_publisher,
            reuse_existing_fragments=True,
            expected_content_md5=expected_md5,
            **common_download,
        )
    observed_md5 = file_md5(destination)
    if (
        receipt.get("url") != location.catalog_url
        or receipt.get("transport_url") != location.mirror_url
        or receipt.get("expected_md5") != expected_md5
        or receipt.get("md5") != observed_md5
        or observed_md5 != expected_md5
        or receipt.get("transport_evidence") != evidence
    ):
        raise AssetIntegrityError("published official-mirror CTX receipt does not verify")
    audit = {
        "schema": MIRROR_AUDIT_SCHEMA,
        "observation_id": location.observation_id,
        "location": asdict(location),
        "transport_evidence": evidence,
        "asset_relative_path": destination.relative_to(root).as_posix(),
        "asset_bytes": destination.stat().st_size,
        "asset_sha256": file_sha256(destination),
        "asset_md5": observed_md5,
        "receipt_relative_path": receipt_path.relative_to(root).as_posix(),
        "receipt_sha256": file_sha256(receipt_path),
    }
    audit["content_sha256"] = canonical_sha256(audit)
    audit_path = guard_paper2_path(
        Path("data/assets/ctx-usgs-mirror/observations")
        / location.observation_id
        / "transport-audit.json",
        paper2_root=root,
    )
    write_canonical_json(audit_path, audit, paper2_root=root)
    return _closed_mirror_result(
        root=root,
        location=location,
        destination=destination,
        receipt_path=receipt_path,
    )


def download_ctx_edr_resilient(
    *,
    url: str,
    output: str | Path,
    receipt: str | Path,
    paper2_root: str | Path,
    max_attempts: int = 4,
    validate_attached_pds3: bool = True,
    repair_invalid_existing: bool = True,
    resume_partial: bool = True,
    request_timeout_seconds: float = 120.0,
    segment_count: int = 2,
    opener: Callable[..., Any] = urllib.request.urlopen,
    sleeper: Callable[[float], Any] = time.sleep,
) -> Mapping[str, Any]:
    """Reuse a valid catalog receipt offline, otherwise use the official USGS copy.

    The catalog URL remains the immutable product identity.  This routing helper
    changes only transport for a missing or incomplete pair and is intended for
    the serial formal G0 orchestrator on hosts whose JPL route is unavailable.
    """

    if not validate_attached_pds3 or not repair_invalid_existing or not resume_partial:
        raise ValueError("resilient CTX transport requires the complete repair contract")
    root = Path(paper2_root).resolve()
    destination = guard_paper2_path(output, paper2_root=root)
    receipt_path = guard_paper2_path(receipt, paper2_root=root)
    observation_id = destination.stem.upper()
    expected_directory = guard_paper2_path(
        Path("data/assets/ctx-edr") / observation_id, paper2_root=root
    )
    if (
        destination != expected_directory / f"{observation_id}.IMG"
        or receipt_path != expected_directory / "download-receipt.json"
    ):
        raise ValueError("resilient CTX output paths differ from the formal asset layout")

    if destination.is_file() and receipt_path.is_file():

        def offline_only(*_args: Any, **_kwargs: Any) -> Any:
            raise ConnectionError("existing CTX pair is not reusable offline")

        try:
            return download_official_asset(
                url=url,
                output=destination,
                receipt=receipt_path,
                paper2_root=root,
                opener=offline_only,
                max_attempts=1,
                validate_attached_pds3=True,
                repair_invalid_existing=True,
                resume_partial=True,
                request_timeout_seconds=request_timeout_seconds,
                sleeper=lambda _delay: None,
            )
        except NetworkRetryExhaustedError:
            pass

    download_ctx_edr_via_usgs_mirror(
        observation_id=observation_id,
        catalog_url=url,
        paper2_root=root,
        max_attempts=max_attempts,
        request_timeout_seconds=request_timeout_seconds,
        segment_count=segment_count,
        opener=opener,
        sleeper=sleeper,
    )
    published = read_json(receipt_path)
    if (
        not isinstance(published, dict)
        or published.get("url") != url
        or published.get("relative_path") != destination.relative_to(root).as_posix()
        or published.get("sha256") != file_sha256(destination)
        or published.get("integrity", {}).get("observed_bytes") != destination.stat().st_size
    ):
        raise AssetIntegrityError("resilient CTX transport did not publish a valid receipt")
    return published

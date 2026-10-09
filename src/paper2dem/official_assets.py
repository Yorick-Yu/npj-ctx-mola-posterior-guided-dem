from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping
import hashlib
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

from .hashing import (
    file_sha256,
    guard_paper2_path,
    read_json,
    write_canonical_json,
    write_immutable_bytes,
)


CTX_EDR_DATA_SET_ID = "MRO-M-CTX-2-EDR-L0-V1.0"
ODE_PRODUCT_ENDPOINT = "https://oderest.rsl.wustl.edu/live2/"
MOLA_MEGDR_BASE = "https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/meg032"


@dataclass(frozen=True)
class MOLAAsset:
    purpose: str
    product_id: str
    filename: str
    url: str


def _mola_pair(prefix: str, purpose: str) -> tuple[MOLAAsset, MOLAAsset]:
    stem = f"{prefix}90N000FB"
    return tuple(  # type: ignore[return-value]
        MOLAAsset(
            purpose=purpose,
            product_id=stem,
            filename=f"{stem}.{suffix}",
            url=f"{MOLA_MEGDR_BASE}/{stem.lower()}.{suffix.lower()}",
        )
        for suffix in ("IMG", "LBL")
    )


MOLA_32PPD_ASSETS = (
    *_mola_pair("MEGT", "terrain_score_areoid_topography"),
    *_mola_pair("MEGR", "isis_shape_planetary_radius"),
    *_mola_pair("MEGC", "observation_count_qa"),
)


class ODEDiscoveryError(ValueError):
    """A terminal, classifiable ODE response that should not be retried."""

    def __init__(self, message: str, *, status: str, match_count: int | None = None):
        super().__init__(message)
        self.status = status
        self.match_count = match_count


class NetworkRetryExhaustedError(ConnectionError):
    """A transient network operation that still failed after bounded retries."""

    def __init__(self, operation: str, *, attempts: int, last_error: Exception):
        super().__init__(
            f"{operation} failed after {attempts} attempts: "
            f"{type(last_error).__name__}: {last_error}"
        )
        self.operation = operation
        self.attempts = attempts
        self.last_error = last_error


class AssetIntegrityError(ConnectionError):
    """Downloaded bytes cannot be a complete instance of the declared asset."""


def file_md5(path: str | Path) -> str:
    """Return a lowercase MD5 content digest for a published archive checksum.

    SHA-256 remains the Paper2 content identity.  MD5 is computed only where an
    official PDS volume publishes MD5 checksums and is retained as independent
    archive evidence; it is never used as the sole Paper2 lock.
    """

    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_attached_pds3_fixed_length(
    path: str | Path,
    *,
    expected_filename: str | None = None,
    max_label_bytes: int = 1024 * 1024,
) -> dict[str, Any]:
    """Validate an attached-label, fixed-record PDS3 product from its own label.

    CTX EDR products declare both ``RECORD_BYTES`` and ``FILE_RECORDS`` in the
    attached label.  Their product is the exact byte length of the IMG.  This
    catches a failure mode that an HTTP-success flag and a hash of the received
    prefix cannot detect.
    """

    source = Path(path)
    if not source.is_file():
        raise AssetIntegrityError(f"PDS3 asset is missing: {source}")
    if max_label_bytes <= 0:
        raise ValueError("max_label_bytes must be positive")
    with source.open("rb") as stream:
        label_bytes = stream.read(max_label_bytes)

    def scalar(name: str) -> str:
        # Parse only ASCII declarations from the bounded prefix.  A CTX EDR is
        # an attached-label product: binary image samples start immediately
        # after the padded label record and are expected to contain non-ASCII
        # bytes.  Decoding the whole prefix as ASCII would therefore reject a
        # perfectly complete product whenever ``max_label_bytes`` reaches into
        # the image payload.
        encoded_name = re.escape(name.encode("ascii"))
        match = re.search(rb"(?m)^\s*" + encoded_name + rb"\s*=\s*([^\r\n]+)", label_bytes)
        if match is None:
            raise AssetIntegrityError(f"attached PDS3 label lacks {name}")
        try:
            return match.group(1).decode("ascii", errors="strict").strip().strip('"')
        except UnicodeDecodeError as exc:
            raise AssetIntegrityError(f"attached PDS3 {name} value is not ASCII") from exc

    if scalar("PDS_VERSION_ID") != "PDS3":
        raise AssetIntegrityError("asset is not an attached PDS3 product")
    if scalar("RECORD_TYPE") != "FIXED_LENGTH":
        raise AssetIntegrityError("CTX EDR must use fixed-length PDS3 records")
    try:
        record_bytes = int(scalar("RECORD_BYTES"))
        file_records = int(scalar("FILE_RECORDS"))
        label_records = int(scalar("LABEL_RECORDS"))
        image_record = int(scalar("^IMAGE"))
    except ValueError as exc:
        raise AssetIntegrityError("PDS3 record declarations are not integers") from exc
    if record_bytes <= 0 or file_records <= 0 or label_records <= 0:
        raise AssetIntegrityError("PDS3 record declarations must be positive")
    if image_record != label_records + 1:
        raise AssetIntegrityError("PDS3 image pointer does not follow the attached label")
    declared_filename = scalar("FILE_NAME")
    if expected_filename is not None and declared_filename.upper() != expected_filename.upper():
        raise AssetIntegrityError(
            f"PDS3 FILE_NAME differs: {declared_filename!r} != {expected_filename!r}"
        )
    label_extent = label_records * record_bytes
    if label_extent > len(label_bytes):
        raise AssetIntegrityError("download ends inside the attached PDS3 label")
    try:
        label_record = label_bytes[:label_extent].decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise AssetIntegrityError("attached PDS3 label record is not ASCII") from exc
    if re.search(r"(?m)^\s*END\s*$", label_record) is None:
        raise AssetIntegrityError("attached PDS3 label has no END marker")
    declared_bytes = record_bytes * file_records
    observed_bytes = source.stat().st_size
    if observed_bytes != declared_bytes:
        raise AssetIntegrityError(
            f"PDS3 byte length mismatch: observed {observed_bytes}, declared {declared_bytes}"
        )
    return {
        "validator": "attached-pds3-fixed-length/v1",
        "file_name": declared_filename,
        "record_bytes": record_bytes,
        "file_records": file_records,
        "label_records": label_records,
        "image_record": image_record,
        "declared_bytes": declared_bytes,
        "observed_bytes": observed_bytes,
    }


def mola_32ppd_manifest() -> dict[str, Any]:
    return {
        "schema": "paper2.mola-official-assets/v1",
        "dataset_id": "MGS-M-MOLA-5-MEGDR-L3-V1.0",
        "resolution_pixels_per_degree": 32,
        "official_dataset_page": "https://pds-geosciences.wustl.edu/missions/mgs/megdr.html",
        "contracts": {
            "MEGT": "terrain scoring relative to the areoid; never an ISIS shape model",
            "MEGR": "planetary radius; pds2isis+demprep output is the ISIS user shape",
            "MEGC": "observation-count QA; records sparsity/interpolation evidence",
        },
        "assets": [asdict(asset) for asset in MOLA_32PPD_ASSETS],
        "download_hash_policy": (
            "Record SHA256 from downloaded bytes in an immutable receipt; do not use HTTP ETag."
        ),
    }


def build_ode_ctx_query_url(observation_id: str) -> str:
    observation_id = observation_id.strip().upper()
    if not re.fullmatch(r"[A-Z0-9]+(?:_[A-Z0-9]+)+", observation_id):
        raise ValueError(f"invalid CTX observation ID: {observation_id!r}")
    query = urllib.parse.urlencode(
        {
            "target": "mars",
            "ihid": "MRO",
            "iid": "CTX",
            "pt": "EDR",
            "query": "product",
            "results": "fp",
            "output": "JSON",
            "pdsid": observation_id,
        }
    )
    return f"{ODE_PRODUCT_ENDPOINT}?{query}"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def parse_ode_ctx_edr(raw_json: bytes, observation_id: str) -> dict[str, Any]:
    """Validate an ODE response and select one exact attached-label CTX EDR IMG."""
    requested = observation_id.strip().upper()
    try:
        payload = json.loads(raw_json)
        results = payload["ODEResults"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ODEDiscoveryError("invalid ODE JSON response", status="rejected") from exc
    response_status = str(results.get("Status", ""))
    if response_status.lower() != "success":
        normalized_status = response_status.lower()
        status = (
            "missing"
            if "no result" in normalized_status or "not found" in normalized_status
            else "rejected"
        )
        raise ODEDiscoveryError(
            f"ODE query did not succeed: {response_status!r}", status=status, match_count=0
        )
    products = _as_list(results.get("Products", {}).get("Product"))
    exact_products = [
        product
        for product in products
        if isinstance(product, dict)
        and str(product.get("pdsid", "")).upper() == requested
        and str(product.get("Data_Set_Id", "")).upper() == CTX_EDR_DATA_SET_ID
        and str(product.get("ihid", "")).upper() == "MRO"
        and str(product.get("iid", "")).upper() == "CTX"
        and str(product.get("pt", "")).upper() == "EDR"
    ]
    if len(exact_products) != 1:
        raise ODEDiscoveryError(
            f"expected one exact {CTX_EDR_DATA_SET_ID} product for {requested}, "
            f"found {len(exact_products)}",
            status="missing" if not exact_products else "ambiguous",
            match_count=len(exact_products),
        )
    product = exact_products[0]
    expected_filename = f"{requested}.IMG"
    files = _as_list(product.get("Product_files", {}).get("Product_file"))
    image_files = [
        item
        for item in files
        if isinstance(item, dict)
        and str(item.get("Type", "")).lower() == "product"
        and str(item.get("FileName", "")).upper() == expected_filename
        and str(item.get("URL", "")).lower().endswith(".img")
        and str(item.get("URL", "")).lower().startswith("https://")
    ]
    if len(image_files) != 1:
        raise ODEDiscoveryError(
            f"expected one exact Product IMG for {requested}, found {len(image_files)}",
            status="rejected" if not image_files else "ambiguous",
            match_count=len(image_files),
        )
    image = image_files[0]
    label_filename = str(product.get("LabelFileName", ""))
    attached_label = (
        label_filename.upper() == expected_filename
        and "WITH LABEL" in str(image.get("Description", "")).upper()
    )
    if not attached_label:
        raise ODEDiscoveryError(
            "CTX EDR must be an IMG with an attached PDS label",
            status="rejected",
            match_count=1,
        )
    return {
        "schema": "paper2.ode-ctx-edr-discovery/v1",
        "observation_id": requested,
        "dataset_id": CTX_EDR_DATA_SET_ID,
        "pds_volume_id": str(product.get("PDSVolume_Id", "")),
        "filename": expected_filename,
        "url": str(image["URL"]),
        "attached_label": True,
        "declared_kbytes": int(image["KBytes"]) if str(image.get("KBytes", "")).isdigit() else None,
    }


def _read_bounded(stream: BinaryIO, *, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    count = 0
    while True:
        chunk = stream.read(min(1024 * 1024, max_bytes + 1 - count))
        if not chunk:
            break
        chunks.append(chunk)
        count += len(chunk)
        if count > max_bytes:
            raise ValueError(f"response exceeds {max_bytes} byte safety limit")
    return b"".join(chunks)


def _retryable_network_error(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 425, 429} or exc.code >= 500
    return isinstance(
        exc,
        (
            AssetIntegrityError,
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            OSError,
        ),
    )


def _retry_delay(
    attempt: int, *, backoff_initial_seconds: float, backoff_max_seconds: float
) -> float:
    return min(backoff_initial_seconds * (2 ** max(attempt - 1, 0)), backoff_max_seconds)


def _read_cached_discovery(
    *,
    observation_id: str,
    query_url: str,
    raw_path: Path,
    output_path: Path,
    paper2_root: Path,
) -> dict[str, Any]:
    payload = read_json(output_path)
    if not isinstance(payload, dict):
        raise ValueError(f"cached ODE discovery is not a JSON object: {output_path}")
    expected_observation = observation_id.strip().upper()
    if payload.get("schema") != "paper2.ode-ctx-edr-discovery/v1":
        raise ValueError(f"cached ODE discovery schema differs: {output_path}")
    if (
        payload.get("observation_id") != expected_observation
        or payload.get("query_url") != query_url
    ):
        raise ValueError(f"cached ODE discovery identity differs: {output_path}")
    raw = raw_path.read_bytes()
    if payload.get("raw_response_sha256") != hashlib.sha256(raw).hexdigest():
        raise ValueError(f"cached ODE response hash differs: {raw_path}")
    if payload.get("raw_response_path") != raw_path.relative_to(paper2_root).as_posix():
        raise ValueError(f"cached ODE response path differs: {raw_path}")
    parsed = parse_ode_ctx_edr(raw, observation_id)
    for key in ("dataset_id", "pds_volume_id", "filename", "url", "attached_label"):
        if payload.get(key) != parsed.get(key):
            raise ValueError(f"cached ODE discovery {key} differs: {output_path}")
    return payload


def discover_ctx_edr(
    observation_id: str,
    *,
    raw_cache: str | Path,
    output: str | Path,
    paper2_root: str | Path,
    opener: Callable[..., Any] = urllib.request.urlopen,
    max_json_bytes: int = 8 * 1024 * 1024,
    max_attempts: int = 4,
    backoff_initial_seconds: float = 1.0,
    backoff_max_seconds: float = 8.0,
    sleeper: Callable[[float], Any] = time.sleep,
) -> dict[str, Any]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    if backoff_initial_seconds < 0 or backoff_max_seconds < 0:
        raise ValueError("retry backoff values must be nonnegative")
    root = Path(paper2_root).resolve()
    raw_path = guard_paper2_path(raw_cache, paper2_root=root)
    output_path = guard_paper2_path(output, paper2_root=root)
    query_url = build_ode_ctx_query_url(observation_id)
    if output_path.exists():
        if not raw_path.is_file():
            raise ValueError(f"cached ODE discovery lacks its raw response: {output_path}")
        return _read_cached_discovery(
            observation_id=observation_id,
            query_url=query_url,
            raw_path=raw_path,
            output_path=output_path,
            paper2_root=root,
        )

    attempts = 0
    if raw_path.exists():
        if not raw_path.is_file():
            raise ValueError(f"cached ODE response is not a file: {raw_path}")
        raw = raw_path.read_bytes()
    else:
        request = urllib.request.Request(query_url, headers={"User-Agent": "paper2dem/0.1"})
        for attempt in range(1, max_attempts + 1):
            attempts = attempt
            try:
                with opener(request, timeout=60) as response:
                    raw = _read_bounded(response, max_bytes=max_json_bytes)
                break
            except Exception as exc:
                if not _retryable_network_error(exc):
                    raise
                if attempt == max_attempts:
                    raise NetworkRetryExhaustedError(
                        f"ODE discovery for {observation_id.strip().upper()}",
                        attempts=attempt,
                        last_error=exc,
                    ) from exc
                sleeper(
                    _retry_delay(
                        attempt,
                        backoff_initial_seconds=backoff_initial_seconds,
                        backoff_max_seconds=backoff_max_seconds,
                    )
                )
        write_immutable_bytes(raw_path, raw, paper2_root=root)

    discovery = parse_ode_ctx_edr(raw, observation_id)
    discovery.update(
        {
            "query_url": query_url,
            "raw_response_path": raw_path.relative_to(root).as_posix(),
            "raw_response_sha256": hashlib.sha256(raw).hexdigest(),
            "request_attempts": attempts,
        }
    )
    write_canonical_json(output_path, discovery, paper2_root=root)
    return discovery


def download_official_asset(
    *,
    url: str,
    output: str | Path,
    receipt: str | Path,
    paper2_root: str | Path,
    expected_sha256: str | None = None,
    expected_md5: str | None = None,
    transport_url: str | None = None,
    transport_evidence: Mapping[str, Any] | None = None,
    opener: Callable[..., Any] = urllib.request.urlopen,
    max_bytes: int | None = None,
    max_attempts: int = 4,
    validate_attached_pds3: bool = False,
    repair_invalid_existing: bool = False,
    resume_partial: bool = False,
    preverified_remote_total_bytes: int | None = None,
    request_timeout_seconds: float = 120.0,
    backoff_initial_seconds: float = 1.0,
    backoff_max_seconds: float = 8.0,
    sleeper: Callable[[float], Any] = time.sleep,
) -> dict[str, Any]:
    """Stream one official asset and bind its bytes to an immutable SHA256 receipt.

    ``validate_attached_pds3`` is required for CTX EDRs.  It validates the exact
    file length declared inside the attached PDS3 label, independently of HTTP
    headers.  ``repair_invalid_existing`` never overwrites suspect evidence: a
    verified replacement is downloaded first and the old files are then moved
    into adjacent ``.invalid`` directories.  ``resume_partial`` preserves one
    deterministic partial file and uses a validated HTTP Range response after a
    transient disconnect; it is intended for the large CTX EDR products.
    """
    if not url.lower().startswith("https://"):
        raise ValueError("official asset URL must use HTTPS")
    effective_url = transport_url if transport_url is not None else url
    if not effective_url.lower().startswith("https://"):
        raise ValueError("official asset transport URL must use HTTPS")
    if expected_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("expected SHA256 must be 64 lowercase hexadecimal characters")
    if expected_md5 is not None and not re.fullmatch(r"[0-9a-f]{32}", expected_md5):
        raise ValueError("expected MD5 must be 32 lowercase hexadecimal characters")
    frozen_transport_evidence = dict(transport_evidence) if transport_evidence is not None else None
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    if request_timeout_seconds <= 0:
        raise ValueError("official asset request timeout must be positive")
    if preverified_remote_total_bytes is not None and (
        not resume_partial
        or (expected_sha256 is None and expected_md5 is None)
        or preverified_remote_total_bytes <= 0
    ):
        raise ValueError(
            "preverified remote total requires resumable mode, an expected content "
            "checksum, and a positive byte count"
        )
    if backoff_initial_seconds < 0 or backoff_max_seconds < 0:
        raise ValueError("retry backoff values must be nonnegative")
    root = Path(paper2_root).resolve()
    destination = guard_paper2_path(output, paper2_root=root)
    receipt_path = guard_paper2_path(receipt, paper2_root=root)
    existing_error: Exception | None = None
    if destination.exists() or receipt_path.exists():
        try:
            if not destination.is_file() or not receipt_path.is_file():
                raise FileExistsError("partial immutable asset/receipt pair exists")
            payload = read_json(receipt_path)
            observed_sha256 = file_sha256(destination)
            observed_md5 = file_md5(destination) if expected_md5 is not None else None
            expected_relative = destination.relative_to(root).as_posix()
            if (
                not isinstance(payload, dict)
                or payload.get("schema") != "paper2.official-download-receipt/v1"
                or payload.get("url") != url
                or payload.get("relative_path") != expected_relative
                or int(payload.get("bytes", -1)) != destination.stat().st_size
                or payload.get("sha256") != observed_sha256
                or (expected_sha256 is not None and observed_sha256 != expected_sha256)
                or (expected_md5 is not None and observed_md5 != expected_md5)
                or (expected_md5 is not None and payload.get("md5") != observed_md5)
                or (transport_url is not None and payload.get("transport_url") != effective_url)
                or (
                    frozen_transport_evidence is not None
                    and payload.get("transport_evidence") != frozen_transport_evidence
                )
                or (
                    resume_partial
                    and expected_sha256 is None
                    and not validate_attached_pds3
                    and int(payload.get("remote_total_bytes", -1)) != destination.stat().st_size
                )
            ):
                raise ValueError("existing immutable asset/receipt pair does not verify")
            if validate_attached_pds3:
                observed_integrity = validate_attached_pds3_fixed_length(
                    destination, expected_filename=destination.name
                )
                if payload.get("integrity") != observed_integrity:
                    if repair_invalid_existing and payload.get("integrity") is None:
                        # Early Paper2 receipts predated the intrinsic attached-PDS3
                        # fields. When the URL/path/size/SHA256 contract above and the
                        # PDS3 label both verify, preserve the data bytes and archive
                        # only the superseded receipt instead of redownloading them.
                        legacy_sha256 = file_sha256(receipt_path)
                        invalid_directory = receipt_path.parent / ".invalid"
                        invalid_directory.mkdir(parents=True, exist_ok=True)
                        archived = invalid_directory / (
                            f"{receipt_path.name}.{legacy_sha256}.legacy.invalid"
                        )
                        if archived.exists():
                            if not archived.is_file() or file_sha256(archived) != legacy_sha256:
                                raise FileExistsError(
                                    f"legacy receipt archive collision: {archived}"
                                )
                            receipt_path.unlink()
                        else:
                            receipt_path.replace(archived)
                        upgraded = {
                            **payload,
                            "integrity": observed_integrity,
                            "download_performed_for_upgrade": False,
                            "replaced_invalid_paths": [archived.relative_to(root).as_posix()],
                            "receipt_upgrade": {
                                "schema": "paper2.official-download-receipt-upgrade/v1",
                                "reason": "legacy_receipt_missing_attached_pds3_integrity",
                                "legacy_receipt_relative_path": archived.relative_to(
                                    root
                                ).as_posix(),
                                "legacy_receipt_sha256": legacy_sha256,
                                "data_bytes_unchanged": True,
                            },
                        }
                        write_canonical_json(receipt_path, upgraded, paper2_root=root)
                        return upgraded
                    raise ValueError(
                        "existing attached-PDS3 receipt integrity differs from the file"
                    )
            return payload
        except Exception as exc:
            if not repair_invalid_existing:
                raise
            existing_error = exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    resume_offsets: list[int] = []
    invalid_resume_partials: list[dict[str, Any]] = []

    def quarantine_resume_partial(temporary: Path, *, reason: str) -> None:
        """Preserve one content-addressed copy before retrying from byte zero."""
        if not temporary.is_file():
            return
        digest = file_sha256(temporary)
        invalid_directory = temporary.parent / ".invalid-parts"
        invalid_directory.mkdir(parents=True, exist_ok=True)
        archived = invalid_directory / f"{temporary.name}.{digest}.invalid"
        observed_bytes = temporary.stat().st_size
        if archived.exists():
            if not archived.is_file() or file_sha256(archived) != digest:
                raise FileExistsError(f"invalid partial archive collision: {archived}")
            temporary.unlink()
        else:
            temporary.replace(archived)
        invalid_resume_partials.append(
            {
                "relative_path": archived.relative_to(root).as_posix(),
                "bytes": observed_bytes,
                "sha256": digest,
                "reason": reason,
            }
        )

    def publish(
        temporary: Path,
        *,
        attempt: int,
        content_length: int | None,
        remote_total_bytes: int | None,
    ) -> dict[str, Any]:
        integrity: dict[str, Any] | None = None
        if validate_attached_pds3:
            integrity = validate_attached_pds3_fixed_length(
                temporary, expected_filename=destination.name
            )
        byte_count = temporary.stat().st_size
        if (
            resume_partial
            and expected_sha256 is None
            and not validate_attached_pds3
            and remote_total_bytes is None
        ):
            raise AssetIntegrityError(
                "resumable asset without intrinsic hash requires an HTTP total length"
            )
        if remote_total_bytes is not None and byte_count != remote_total_bytes:
            raise AssetIntegrityError(
                "downloaded byte length does not close the HTTP range total: "
                f"observed {byte_count}, remote total {remote_total_bytes}"
            )
        observed_sha256 = file_sha256(temporary)
        if expected_sha256 is not None and observed_sha256 != expected_sha256:
            raise ValueError(
                f"download SHA256 mismatch: observed {observed_sha256}, expected {expected_sha256}"
            )
        observed_md5 = (
            file_md5(temporary) if expected_md5 is not None or transport_url is not None else None
        )
        if expected_md5 is not None and observed_md5 != expected_md5:
            raise AssetIntegrityError(
                f"download MD5 mismatch: observed {observed_md5}, expected {expected_md5}"
            )
        archived_invalid_paths: list[str] = []
        if existing_error is not None:
            for stale in (destination, receipt_path):
                if not stale.exists():
                    continue
                invalid_directory = stale.parent / ".invalid"
                invalid_directory.mkdir(parents=True, exist_ok=True)
                stale_digest = file_sha256(stale) if stale.is_file() else "not-a-file"
                archived = invalid_directory / (
                    f"{stale.name}.{stale_digest}.{time.time_ns()}.invalid"
                )
                stale.replace(archived)
                archived_invalid_paths.append(archived.relative_to(root).as_posix())
        temporary.replace(destination)
        payload = {
            "schema": "paper2.official-download-receipt/v1",
            "url": url,
            "relative_path": destination.relative_to(root).as_posix(),
            "bytes": byte_count,
            "sha256": observed_sha256,
            "hash_source": "downloaded_bytes_not_http_etag",
            "download_attempts": attempt,
            "http_content_length": content_length,
            "remote_total_bytes": remote_total_bytes,
            "integrity": integrity,
            "resume_partial": resume_partial,
            "resume_offsets": resume_offsets,
            "invalid_resume_partials": invalid_resume_partials,
            "replaced_invalid_paths": archived_invalid_paths,
            "request_timeout_seconds": request_timeout_seconds,
        }
        if transport_url is not None:
            payload.update(
                {
                    "transport_url": effective_url,
                    "transport_differs_from_catalog_url": effective_url != url,
                    "md5": observed_md5,
                }
            )
        if expected_md5 is not None:
            payload["expected_md5"] = expected_md5
            payload["md5"] = observed_md5
        if frozen_transport_evidence is not None:
            payload["transport_evidence"] = frozen_transport_evidence
        if preverified_remote_total_bytes is not None:
            payload["preverified_remote_total_bytes"] = preverified_remote_total_bytes
        write_canonical_json(receipt_path, payload, paper2_root=root)
        return payload

    for attempt in range(1, max_attempts + 1):
        if resume_partial:
            temporary = destination.parent / f".{destination.name}.download.part"
            if temporary.exists() and not temporary.is_file():
                raise FileExistsError(f"resume path is not a file: {temporary}")
        else:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
        network_phase = True
        requested_offset = 0
        try:
            if (
                resume_partial
                and temporary.is_file()
                and temporary.stat().st_size > 0
                and (validate_attached_pds3 or expected_sha256 is not None)
            ):
                try:
                    network_phase = False
                    return publish(
                        temporary,
                        attempt=attempt,
                        content_length=None,
                        remote_total_bytes=preverified_remote_total_bytes,
                    )
                except (AssetIntegrityError, ValueError):
                    network_phase = True
            requested_offset = (
                temporary.stat().st_size if resume_partial and temporary.is_file() else 0
            )
            request_headers = {"User-Agent": "paper2dem/0.1"}
            if requested_offset:
                request_headers["Range"] = f"bytes={requested_offset}-"
            request = urllib.request.Request(effective_url, headers=request_headers)
            downloaded_this_attempt = 0
            remote_total_bytes: int | None = None
            with opener(request, timeout=request_timeout_seconds) as response:
                headers = getattr(response, "headers", None)
                content_length_value = (
                    headers.get("Content-Length") if headers is not None else None
                )
                status = getattr(response, "status", None)
                if status is None:
                    getcode = getattr(response, "getcode", None)
                    status = getcode() if callable(getcode) else None
                append = False
                if requested_offset:
                    content_range = headers.get("Content-Range") if headers is not None else None
                    match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+)", str(content_range).strip())
                    if (
                        status == 206
                        and match is not None
                        and int(match.group(1)) == requested_offset
                        and int(match.group(2)) >= requested_offset
                        and int(match.group(2)) < int(match.group(3))
                    ):
                        append = True
                        resume_offsets.append(requested_offset)
                        remote_total_bytes = int(match.group(3))
                        declared_range_bytes = int(match.group(2)) - requested_offset + 1
                        if (
                            content_length_value is not None
                            and str(content_length_value).strip().isdigit()
                            and int(content_length_value) != declared_range_bytes
                        ):
                            raise AssetIntegrityError(
                                "HTTP Content-Length differs from Content-Range span"
                            )
                    elif status == 200:
                        requested_offset = 0
                    else:
                        raise AssetIntegrityError(
                            "server did not honor the requested resumable byte range"
                        )
                with temporary.open("ab" if append else "wb") as target:
                    try:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            downloaded_this_attempt += len(chunk)
                            total_bytes = requested_offset + downloaded_this_attempt
                            if max_bytes is not None and total_bytes > max_bytes:
                                raise ValueError(f"download exceeds {max_bytes} byte safety limit")
                            target.write(chunk)
                    finally:
                        target.flush()
                        os.fsync(target.fileno())
            content_length = (
                int(content_length_value)
                if content_length_value is not None and str(content_length_value).strip().isdigit()
                else None
            )
            if content_length is not None and downloaded_this_attempt != content_length:
                raise AssetIntegrityError(
                    "HTTP response byte length mismatch: "
                    f"observed {downloaded_this_attempt}, declared {content_length}"
                )
            if requested_offset == 0 and content_length is not None:
                remote_total_bytes = content_length
            if remote_total_bytes is not None and temporary.stat().st_size != remote_total_bytes:
                raise AssetIntegrityError(
                    "HTTP range total did not close after download: "
                    f"observed {temporary.stat().st_size}, declared {remote_total_bytes}"
                )
            network_phase = False
            return publish(
                temporary,
                attempt=attempt,
                content_length=content_length,
                remote_total_bytes=remote_total_bytes,
            )
        except Exception as exc:
            if (
                resume_partial
                and requested_offset > 0
                and isinstance(exc, urllib.error.HTTPError)
                and exc.code == 416
            ):
                content_range = exc.headers.get("Content-Range") if exc.headers else None
                total_match = re.fullmatch(r"bytes\s+\*/(\d+)", str(content_range).strip())
                remote_bytes = int(total_match.group(1)) if total_match else None
                if (
                    remote_bytes == requested_offset
                    and expected_sha256 is None
                    and not validate_attached_pds3
                ):
                    network_phase = False
                    return publish(
                        temporary,
                        attempt=attempt,
                        content_length=0,
                        remote_total_bytes=remote_bytes,
                    )
                reason = (
                    "HTTP 416 after the preserved prefix failed intrinsic validation; "
                    f"requested_offset={requested_offset}, remote_bytes={remote_bytes}"
                )
                quarantine_resume_partial(temporary, reason=reason)
                # A 416 at a nonzero offset is not retried with the same Range.
                # The next bounded attempt starts at byte zero, allowing a
                # corrupt same-length prefix to be repaired.
                exc = AssetIntegrityError(reason)
            # Publishing/receipt failures are local and must not enter the network
            # retry loop.  An integrity mismatch after EOF is retryable because a
            # subsequent Range request can complete the preserved prefix.
            if (
                not network_phase and not isinstance(exc, AssetIntegrityError)
            ) or not _retryable_network_error(exc):
                raise
            if attempt == max_attempts:
                raise NetworkRetryExhaustedError(
                    f"official asset download {effective_url}",
                    attempts=attempt,
                    last_error=exc,
                ) from exc
            sleeper(
                _retry_delay(
                    attempt,
                    backoff_initial_seconds=backoff_initial_seconds,
                    backoff_max_seconds=backoff_max_seconds,
                )
            )
        finally:
            if not resume_partial:
                temporary.unlink(missing_ok=True)
    raise AssertionError("unreachable retry loop")

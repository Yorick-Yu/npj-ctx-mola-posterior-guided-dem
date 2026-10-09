from __future__ import annotations

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import math
import os
import re
import tempfile

import numpy as np

from .assets import FORMAL_QUOTAS
from .g0 import (
    SCHEMAS,
    validate_edr_index,
    validate_mola_source_lock,
    validate_terrain_manifest,
)
from .hashing import (
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    is_sha256,
    read_json,
)
from .official_assets import (
    CTX_EDR_DATA_SET_ID,
    NetworkRetryExhaustedError,
    ODEDiscoveryError,
    discover_ctx_edr,
)


TERRAIN_ALGORITHM = "mola-megt-robust-relief-v1"
TERRAIN_QUANTILES = (5.0, 95.0)
TERRAIN_MIN_FINITE_PIXELS = 4
EDR_TERMINAL_STATUSES = frozenset({"resolved", "missing", "ambiguous", "rejected"})
EDR_MAX_WORKERS = 4
FORMAL_ROLE_COUNTS = {role: per_stratum * 12 for role, per_stratum in FORMAL_QUOTAS.items()}


def _read_object_inside(
    path: str | Path, *, paper2_root: str | Path
) -> tuple[Path, dict[str, Any]]:
    resolved = guard_paper2_path(path, paper2_root=paper2_root)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = read_json(resolved)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {resolved}")
    return resolved, payload


def _write_checkpoint(
    path: str | Path, payload: Mapping[str, Any], *, paper2_root: str | Path
) -> Path:
    """Atomically replace a recoverable sidecar while never escaping paper2/."""
    output = guard_paper2_path(path, paper2_root=paper2_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".checkpoint", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _conflict_groups_hash(payload: Mapping[str, Any]) -> str:
    if payload.get("schema") != SCHEMAS["conflicts"]:
        raise ValueError("conflict groups use an unsupported schema")
    unhashed = dict(payload)
    declared = unhashed.pop("content_sha256", None)
    observed = canonical_sha256(unhashed)
    if declared is not None and declared != observed:
        raise ValueError("conflict-groups content_sha256 does not verify")
    return observed


def _finite_bbox(raw: Sequence[Any]) -> tuple[float, float, float, float]:
    bbox = tuple(float(value) for value in raw)
    if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
        raise ValueError("bbox_lonlat must contain four finite values")
    west, south, east, north = bbox
    if west > east or south > north or south < -90.0 or north > 90.0:
        raise ValueError(f"invalid normalized lon/lat bbox: {bbox}")
    if east - west > 360.0:
        raise ValueError(f"longitude span exceeds 360 degrees: {bbox}")
    return bbox


def robust_relief_statistics(
    values: np.ndarray | Sequence[float],
    *,
    lower_percentile: float = TERRAIN_QUANTILES[0],
    upper_percentile: float = TERRAIN_QUANTILES[1],
    min_finite_pixels: int = TERRAIN_MIN_FINITE_PIXELS,
) -> dict[str, float | int]:
    """Return the fixed P95-P05 MEGT relief, robust to isolated extrema and nodata."""
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("terrain percentiles must satisfy 0 <= lower < upper <= 100")
    if min_finite_pixels < 1:
        raise ValueError("min_finite_pixels must be positive")
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size < min_finite_pixels:
        raise ValueError(
            f"MEGT window has {finite.size} finite pixels; requires {min_finite_pixels}"
        )
    lower, upper = np.percentile(finite, [lower_percentile, upper_percentile])
    relief = float(upper - lower)
    if not math.isfinite(relief) or relief < 0:
        raise ValueError("MEGT robust relief is not finite and nonnegative")
    return {
        "terrain_score": relief,
        "lower_quantile_m": float(lower),
        "upper_quantile_m": float(upper),
        "finite_pixel_count": int(finite.size),
        "requested_pixel_count": int(array.size),
        "finite_fraction": float(finite.size / array.size),
    }


def _pds_scalar(raw: str) -> str | int | float:
    value = raw.split("/*", 1)[0].strip()
    value = re.sub(r"\s*<[^>]+>\s*$", "", value)
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1]
    radix = re.fullmatch(r"(\d+)#([0-9A-Fa-f]+)#", value)
    if radix:
        return int(radix.group(2), int(radix.group(1)))
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_pds3_label(path: str | Path) -> dict[str, str | int | float]:
    """Parse scalar fields needed by the frozen detached MOLA MEGT product."""
    fields: dict[str, str | int | float] = {}
    for line in Path(path).read_text(encoding="ascii", errors="strict").splitlines():
        match = re.match(r"^\s*([A-Z][A-Z0-9_^]*)\s*=\s*(.*?)\s*$", line)
        if match:
            fields[match.group(1)] = _pds_scalar(match.group(2))
    return fields


def _signed_constant(value: int | float | str, *, bits: int) -> float:
    numeric = float(value)
    if isinstance(value, int) and value >= 2 ** (bits - 1):
        numeric = float(value - 2**bits)
    return numeric


def _longitude_intervals(west: float, east: float) -> tuple[tuple[float, float], ...]:
    width = east - west
    if width >= 360.0:
        return ((0.0, 360.0),)
    start = west % 360.0
    unwrapped_stop = start + width
    if unwrapped_stop <= 360.0:
        return ((start, unwrapped_stop),)
    return ((start, 360.0), (0.0, unwrapped_stop - 360.0))


class MEGTRaster:
    """Read detached, global 32 pixel/degree MOLA MEGT without loading it in RAM."""

    def __init__(
        self,
        image_path: str | Path,
        label_path: str | Path,
        *,
        pixels_per_degree: int = 32,
    ) -> None:
        self.image_path = Path(image_path)
        self.label_path = Path(label_path)
        fields = parse_pds3_label(self.label_path)
        self.lines = int(fields.get("LINES", 0))
        self.samples = int(fields.get("LINE_SAMPLES", 0))
        self.sample_bits = int(fields.get("SAMPLE_BITS", 0))
        self.sample_type = str(fields.get("SAMPLE_TYPE", "")).upper()
        self.pixels_per_degree = int(pixels_per_degree)
        expected_shape = (180 * self.pixels_per_degree, 360 * self.pixels_per_degree)
        if (self.lines, self.samples) != expected_shape:
            raise ValueError(
                f"MEGT shape {(self.lines, self.samples)} differs from frozen {expected_shape}"
            )
        dtype_by_contract = {
            (16, "MSB_INTEGER"): np.dtype(">i2"),
            (16, "SUN_INTEGER"): np.dtype(">i2"),
            (16, "LSB_INTEGER"): np.dtype("<i2"),
            (16, "PC_INTEGER"): np.dtype("<i2"),
        }
        try:
            self.dtype = dtype_by_contract[(self.sample_bits, self.sample_type)]
        except KeyError as exc:
            raise ValueError(
                f"unsupported MEGT sample contract: {self.sample_bits}/{self.sample_type}"
            ) from exc
        expected_bytes = self.lines * self.samples * self.dtype.itemsize
        if self.image_path.stat().st_size != expected_bytes:
            raise ValueError(
                f"MEGT IMG byte count differs: {self.image_path.stat().st_size} != {expected_bytes}"
            )
        self.scale = float(fields.get("SCALING_FACTOR", 1.0))
        self.offset = float(fields.get("OFFSET", 0.0))
        missing_values = []
        for key in ("MISSING_CONSTANT", "NULL", "INVALID_CONSTANT"):
            if key in fields:
                missing_values.append(_signed_constant(fields[key], bits=self.sample_bits))
        self.missing_values = tuple(missing_values)
        self.valid_minimum = float(fields["VALID_MINIMUM"]) if "VALID_MINIMUM" in fields else None
        self.valid_maximum = float(fields["VALID_MAXIMUM"]) if "VALID_MAXIMUM" in fields else None
        self._array = np.memmap(
            self.image_path, dtype=self.dtype, mode="r", shape=(self.lines, self.samples)
        )
        self.metadata = {
            "lines": self.lines,
            "line_samples": self.samples,
            "sample_bits": self.sample_bits,
            "sample_type": self.sample_type,
            "pixels_per_degree": self.pixels_per_degree,
            "longitude_domain": [0.0, 360.0],
            "latitude_domain": [-90.0, 90.0],
            "row_order": "north_to_south",
        }

    def read_bbox(self, bbox_lonlat: Sequence[float]) -> np.ndarray:
        west, south, east, north = _finite_bbox(bbox_lonlat)
        ppd = self.pixels_per_degree
        row_start = max(0, int(math.floor((90.0 - north) * ppd)))
        row_stop = min(self.lines, int(math.ceil((90.0 - south) * ppd)))
        if row_stop <= row_start:
            raise ValueError(f"MEGT bbox has no latitude pixels: {bbox_lonlat}")
        chunks: list[np.ndarray] = []
        for interval_west, interval_east in _longitude_intervals(west, east):
            column_start = max(0, int(math.floor(interval_west * ppd)))
            column_stop = min(self.samples, int(math.ceil(interval_east * ppd)))
            if column_stop <= column_start:
                continue
            raw = np.asarray(
                self._array[row_start:row_stop, column_start:column_stop], dtype=np.float64
            )
            for missing in self.missing_values:
                raw[raw == missing] = np.nan
            if self.valid_minimum is not None:
                raw[raw < self.valid_minimum] = np.nan
            if self.valid_maximum is not None:
                raw[raw > self.valid_maximum] = np.nan
            chunks.append(raw.reshape(-1) * self.scale + self.offset)
        if not chunks:
            raise ValueError(f"MEGT bbox has no longitude pixels: {bbox_lonlat}")
        return np.concatenate(chunks)


def _mola_megt_product(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not validate_mola_source_lock(payload):
        raise ValueError("MOLA source lock is not fully downloaded and byte-hashed")
    matches = [
        dict(product)
        for product in payload.get("products", [])
        if product.get("product_id") == "MEGT90N000FB" and product.get("purpose") == "terrain_score"
    ]
    if len(matches) != 1:
        raise ValueError("MOLA source lock does not contain one terrain-scoring MEGT product")
    return matches[0]


def _locked_asset_path(
    product: Mapping[str, Any],
    kind: str,
    explicit: str | Path | None,
    *,
    paper2_root: str | Path,
) -> Path:
    candidate = explicit if explicit is not None else product.get(f"{kind}_relative_path")
    if not candidate:
        raise ValueError(f"MOLA source lock lacks {kind}_relative_path")
    return guard_paper2_path(str(candidate), paper2_root=paper2_root)


def _terrain_payload(
    *,
    bindings: Mapping[str, Any],
    raster_contract: Mapping[str, Any],
    entries: Mapping[str, Mapping[str, Any]],
    expected_count: int,
) -> dict[str, Any]:
    ordered = [dict(entries[group_id]) for group_id in sorted(entries)]
    completed = sum(entry.get("status") == "complete" for entry in ordered)
    failed = sum(entry.get("status") == "failed" for entry in ordered)
    return {
        "schema": SCHEMAS["terrain"],
        "state": "complete" if completed == expected_count and not failed else "incomplete",
        **dict(bindings),
        "algorithm": TERRAIN_ALGORITHM,
        "algorithm_parameters": {
            "lower_percentile": TERRAIN_QUANTILES[0],
            "upper_percentile": TERRAIN_QUANTILES[1],
            "score": "upper_quantile_m-minus-lower_quantile_m",
            "minimum_finite_pixels": TERRAIN_MIN_FINITE_PIXELS,
        },
        "raster_contract": dict(raster_contract),
        "expected_group_count": expected_count,
        "complete_group_count": completed,
        "failed_group_count": failed,
        "groups": ordered,
    }


def _verify_completed_terrain_entry(entry: Mapping[str, Any], *, group_id: str) -> None:
    score = float(entry.get("terrain_score", np.nan))
    lower = float(entry.get("lower_quantile_m", np.nan))
    upper = float(entry.get("upper_quantile_m", np.nan))
    if (
        entry.get("status") != "complete"
        or not math.isfinite(score)
        or not math.isfinite(lower)
        or not math.isfinite(upper)
        or not np.isclose(score, upper - lower, rtol=0.0, atol=1e-9)
        or int(entry.get("finite_pixel_count", 0)) < TERRAIN_MIN_FINITE_PIXELS
        or int(entry.get("requested_pixel_count", 0)) < int(entry.get("finite_pixel_count", 0))
    ):
        raise ValueError(f"completed terrain checkpoint entry is invalid: {group_id}")


def build_terrain_manifest(
    *,
    conflict_groups_path: str | Path,
    mola_source_lock_path: str | Path,
    output: str | Path,
    paper2_root: str | Path,
    megt_image_path: str | Path | None = None,
    megt_label_path: str | Path | None = None,
    terrain_reader: Any | None = None,
) -> dict[str, Any]:
    """Build or resume the input-bound, per-group MOLA robust-relief sidecar."""
    root = Path(paper2_root).resolve()
    conflicts_path, conflicts = _read_object_inside(conflict_groups_path, paper2_root=root)
    mola_path, mola = _read_object_inside(mola_source_lock_path, paper2_root=root)
    conflict_hash = _conflict_groups_hash(conflicts)
    product = _mola_megt_product(mola)
    image_path = _locked_asset_path(product, "image", megt_image_path, paper2_root=root)
    label_path = _locked_asset_path(product, "label", megt_label_path, paper2_root=root)
    if not image_path.is_file() or not label_path.is_file():
        raise FileNotFoundError("downloaded MEGT IMG/LBL pair is incomplete")
    image_hash, label_hash = file_sha256(image_path), file_sha256(label_path)
    if image_hash != product.get("image_sha256") or label_hash != product.get("label_sha256"):
        raise ValueError("downloaded MEGT bytes differ from the MOLA source lock")
    bindings = {
        "conflict_groups_sha256": conflict_hash,
        "conflict_groups_file_sha256": file_sha256(conflicts_path),
        "mola_source_lock_sha256": file_sha256(mola_path),
        "megt_image_sha256": image_hash,
        "megt_label_sha256": label_hash,
        "megt_image_relative_path": image_path.relative_to(root).as_posix(),
        "megt_label_relative_path": label_path.relative_to(root).as_posix(),
    }
    bindings["input_bindings_sha256"] = canonical_sha256(bindings)
    reader = terrain_reader if terrain_reader is not None else MEGTRaster(image_path, label_path)
    raster_contract = dict(getattr(reader, "metadata", {}))

    expected_groups: dict[str, tuple[float, float, float, float]] = {}
    for raw in conflicts.get("eligible_groups", []):
        group_id = str(raw.get("group_id", ""))
        latitude = float(raw.get("center_latitude", np.nan))
        if -65.0 <= latitude <= 65.0:
            if not group_id or group_id in expected_groups:
                raise ValueError("eligible in-band group IDs must be nonempty and unique")
            expected_groups[group_id] = _finite_bbox(raw.get("bbox_lonlat", []))
    if not expected_groups:
        raise ValueError("conflict groups contain no eligible in-band groups")

    output_path = guard_paper2_path(output, paper2_root=root)
    entries: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        existing = read_json(output_path)
        if not isinstance(existing, dict) or existing.get("schema") != SCHEMAS["terrain"]:
            raise ValueError("existing terrain checkpoint uses an unsupported schema")
        for key, expected in bindings.items():
            if existing.get(key) != expected:
                raise ValueError(f"existing terrain checkpoint has stale {key}")
        if existing.get("algorithm") != TERRAIN_ALGORITHM:
            raise ValueError("existing terrain checkpoint uses a different algorithm")
        expected_parameters = _terrain_payload(
            bindings=bindings,
            raster_contract=raster_contract,
            entries={},
            expected_count=len(expected_groups),
        )["algorithm_parameters"]
        if existing.get("algorithm_parameters") != expected_parameters:
            raise ValueError("existing terrain checkpoint uses different algorithm parameters")
        if existing.get("raster_contract") != raster_contract:
            raise ValueError("existing terrain checkpoint uses a different raster contract")
        for raw in existing.get("groups", []):
            entry = dict(raw)
            group_id = str(entry.get("group_id", ""))
            if not group_id or group_id in entries or group_id not in expected_groups:
                raise ValueError("existing terrain checkpoint has an unexpected group")
            expected_bbox_hash = canonical_sha256(list(expected_groups[group_id]))
            if entry.get("bbox_sha256") != expected_bbox_hash:
                raise ValueError(f"existing terrain checkpoint has stale bbox for {group_id}")
            if entry.get("status") not in {"complete", "failed"}:
                raise ValueError(f"unsupported terrain checkpoint status for {group_id}")
            if entry.get("status") == "complete":
                _verify_completed_terrain_entry(entry, group_id=group_id)
            entries[group_id] = entry

    _write_checkpoint(
        output_path,
        _terrain_payload(
            bindings=bindings,
            raster_contract=raster_contract,
            entries=entries,
            expected_count=len(expected_groups),
        ),
        paper2_root=root,
    )
    for group_id, bbox in sorted(expected_groups.items()):
        previous = entries.get(group_id, {})
        if previous.get("status") == "complete":
            continue
        attempts = int(previous.get("attempts", 0)) + 1
        base = {
            "group_id": group_id,
            "bbox_lonlat": list(bbox),
            "bbox_sha256": canonical_sha256(list(bbox)),
            "attempts": attempts,
        }
        try:
            statistics = robust_relief_statistics(reader.read_bbox(bbox))
            entries[group_id] = {**base, "status": "complete", **statistics}
        except Exception as exc:
            entries[group_id] = {
                **base,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
            }
        _write_checkpoint(
            output_path,
            _terrain_payload(
                bindings=bindings,
                raster_contract=raster_contract,
                entries=entries,
                expected_count=len(expected_groups),
            ),
            paper2_root=root,
        )
    completed_payload = read_json(output_path)
    if completed_payload.get("state") == "complete":
        validate_terrain_manifest(
            completed_payload,
            conflict_groups_sha256=conflict_hash,
            mola_source_lock_sha256=file_sha256(mola_path),
        )
    return completed_payload


def _fixed_assignments(
    conflicts: Mapping[str, Any],
    assignments_payload: Mapping[str, Any],
    *,
    require_formal_counts: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if assignments_payload.get("schema") not in {
        SCHEMAS["draft"],
        "paper2.g0.role-assignments/v1",
    }:
        raise ValueError("role assignments must be a G0 draft or fixed role-assignment manifest")
    group_by_id = {
        str(group["group_id"]): dict(group) for group in conflicts.get("eligible_groups", [])
    }
    normalized: list[dict[str, Any]] = []
    roles: list[str] = []
    for raw in assignments_payload.get("assignments", []):
        group_id = str(raw.get("group_id", ""))
        role = str(raw.get("role", ""))
        if group_id not in group_by_id or role not in FORMAL_QUOTAS:
            raise ValueError(f"invalid fixed role assignment: {group_id}/{role}")
        group = group_by_id[group_id]
        representative = str(group["representative_product_id"])
        fallbacks = [str(value) for value in group.get("fallback_product_ids", [])]
        if raw.get("representative_product_id") not in {None, representative}:
            raise ValueError(f"assignment representative differs for {group_id}")
        if (
            raw.get("fallback_product_ids") is not None
            and list(raw["fallback_product_ids"]) != fallbacks
        ):
            raise ValueError(f"assignment fallback order differs for {group_id}")
        normalized.append(
            {
                "group_id": group_id,
                "role": role,
                "representative_product_id": representative,
                "fallback_product_ids": fallbacks,
            }
        )
        roles.append(role)
    normalized.sort(key=lambda item: item["group_id"])
    group_ids = [item["group_id"] for item in normalized]
    if not normalized or len(group_ids) != len(set(group_ids)):
        raise ValueError("fixed role assignments must be nonempty and unique by group")
    if require_formal_counts and dict(Counter(roles)) != FORMAL_ROLE_COUNTS:
        raise ValueError(
            f"fixed role assignments have counts {dict(Counter(roles))}; "
            f"expected {FORMAL_ROLE_COUNTS}"
        )

    requests: dict[str, dict[str, Any]] = {}
    request_order: list[str] = []
    for assignment in normalized:
        product_ids = [
            assignment["representative_product_id"],
            *assignment["fallback_product_ids"],
        ]
        for product_order, product_id in enumerate(product_ids):
            observations = [value for value in str(product_id).split("__") if value]
            if len(observations) < 2:
                raise ValueError(f"CTX DTM product lacks an observation pair: {product_id}")
            for observation_order, observation_id in enumerate(observations):
                if observation_id not in requests:
                    request_order.append(observation_id)
                    requests[observation_id] = {
                        "observation_id": observation_id,
                        "sources": [],
                    }
                requests[observation_id]["sources"].append(
                    {
                        "group_id": assignment["group_id"],
                        "role": assignment["role"],
                        "product_id": product_id,
                        "product_order": product_order,
                        "observation_order": observation_order,
                    }
                )
    specifications = []
    for order, observation_id in enumerate(request_order):
        specification = requests[observation_id]
        specification["request_order"] = order
        specification["request_sha256"] = canonical_sha256(
            {key: value for key, value in specification.items() if key != "request_sha256"}
        )
        specifications.append(specification)
    return normalized, specifications


def _edr_payload(
    *,
    bindings: Mapping[str, Any],
    specifications: Sequence[Mapping[str, Any]],
    entries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    ordered_entries = [
        dict(entries[specification["observation_id"]])
        for specification in specifications
        if specification["observation_id"] in entries
    ]
    failed = sum(entry.get("status") == "failed" for entry in ordered_entries)
    terminal = sum(entry.get("status") in EDR_TERMINAL_STATUSES for entry in ordered_entries)
    expected = len(specifications)
    return {
        "schema": SCHEMAS["edr"],
        "state": "complete" if terminal == expected and not failed else "incomplete",
        **dict(bindings),
        "provider": "Washington University ODE REST",
        "dataset_id": CTX_EDR_DATA_SET_ID,
        "requested_observation_count": expected,
        "terminal_observation_count": terminal,
        "failed_observation_count": failed,
        "requests": [dict(value) for value in specifications],
        "entries": ordered_entries,
    }


def _verify_reusable_edr_entry(
    entry: Mapping[str, Any],
    specification: Mapping[str, Any],
    *,
    paper2_root: Path,
) -> None:
    observation_id = str(specification["observation_id"])
    if entry.get("request_sha256") != specification.get("request_sha256"):
        raise ValueError(f"existing EDR entry has stale request binding for {observation_id}")
    status = entry.get("status")
    if status not in EDR_TERMINAL_STATUSES | {"failed"}:
        raise ValueError(f"existing EDR entry has unsupported status for {observation_id}")
    if status == "failed":
        return
    raw_path = guard_paper2_path(str(entry.get("raw_response_path", "")), paper2_root=paper2_root)
    if not raw_path.is_file() or entry.get("ode_response_sha256") != file_sha256(raw_path):
        raise ValueError(f"existing ODE response does not verify for {observation_id}")
    if status == "resolved":
        discovery_path = guard_paper2_path(
            str(entry.get("discovery_path", "")), paper2_root=paper2_root
        )
        if not discovery_path.is_file() or entry.get("discovery_sha256") != file_sha256(
            discovery_path
        ):
            raise ValueError(f"existing ODE discovery does not verify for {observation_id}")
        discovery = read_json(discovery_path)
        if (
            not isinstance(discovery, dict)
            or discovery.get("observation_id") != observation_id
            or discovery.get("url") != entry.get("selected_url")
            or discovery.get("raw_response_sha256") != entry.get("ode_response_sha256")
        ):
            raise ValueError(f"existing ODE discovery content differs for {observation_id}")


def _discover_edr_entry(
    specification: Mapping[str, Any],
    *,
    previous: Mapping[str, Any],
    discovery_directory: Path,
    paper2_root: Path,
    max_attempts: int,
    backoff_initial_seconds: float,
    backoff_max_seconds: float,
    discoverer: Callable[..., Mapping[str, Any]],
    sleeper: Callable[[float], Any] | None,
) -> tuple[str, dict[str, Any]]:
    """Resolve one observation using paths private to that observation.

    The shared EDR checkpoint is deliberately not available to worker threads.  A
    single coordinator serializes the returned entries in frozen request order.
    """
    observation_id = str(specification["observation_id"])
    pipeline_attempt = int(previous.get("pipeline_attempts", 0)) + 1
    observation_root = discovery_directory / "ctx" / observation_id
    raw_cache = observation_root / "ode-response.raw.json"
    discovery_path = observation_root / "discovery.json"
    base = {
        "observation_id": observation_id,
        "request_order": int(specification["request_order"]),
        "request_sha256": specification["request_sha256"],
        "sources": specification["sources"],
        "pipeline_attempts": pipeline_attempt,
    }
    call_kwargs: dict[str, Any] = {
        "raw_cache": raw_cache,
        "output": discovery_path,
        "paper2_root": paper2_root,
        "max_attempts": max_attempts,
        "backoff_initial_seconds": backoff_initial_seconds,
        "backoff_max_seconds": backoff_max_seconds,
    }
    if sleeper is not None:
        call_kwargs["sleeper"] = sleeper
    try:
        discovery = dict(discoverer(observation_id, **call_kwargs))
        entry = {
            **base,
            "status": "resolved",
            "selected_url": str(discovery["url"]),
            "match_count": 1,
            "ode_response_sha256": str(discovery["raw_response_sha256"]),
            "raw_response_path": str(discovery["raw_response_path"]),
            "discovery_path": discovery_path.relative_to(paper2_root).as_posix(),
            "discovery_sha256": file_sha256(discovery_path),
            "http_attempts": int(discovery.get("request_attempts", 0)),
        }
    except ODEDiscoveryError as exc:
        if not raw_cache.is_file():
            entry = {
                **base,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": "terminal ODE classification lacked cached response bytes",
            }
        else:
            entry = {
                **base,
                "status": exc.status,
                "match_count": exc.match_count,
                "ode_response_sha256": file_sha256(raw_cache),
                "raw_response_path": raw_cache.relative_to(paper2_root).as_posix(),
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
            }
    except Exception as exc:
        entry = {
            **base,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:500],
            "http_attempts": (
                exc.attempts if isinstance(exc, NetworkRetryExhaustedError) else None
            ),
        }
    return observation_id, entry


def build_edr_index(
    *,
    conflict_groups_path: str | Path,
    assignments_path: str | Path,
    output: str | Path,
    discovery_root: str | Path,
    paper2_root: str | Path,
    max_attempts: int = 4,
    backoff_initial_seconds: float = 1.0,
    backoff_max_seconds: float = 8.0,
    require_formal_counts: bool = True,
    discoverer: Callable[..., Mapping[str, Any]] = discover_ctx_edr,
    sleeper: Callable[[float], Any] | None = None,
    workers: int = EDR_MAX_WORKERS,
) -> dict[str, Any]:
    """Build or resume the deterministic fixed-role ODE EDR index.

    At most four workers may discover disjoint observation assets concurrently.
    Only this coordinator mutates the atomically replaced shared checkpoint.
    """
    if isinstance(workers, bool) or not isinstance(workers, int):
        raise TypeError("EDR discovery workers must be an integer")
    if not 1 <= workers <= EDR_MAX_WORKERS:
        raise ValueError(f"EDR discovery workers must be between 1 and {EDR_MAX_WORKERS}")
    root = Path(paper2_root).resolve()
    conflicts_path, conflicts = _read_object_inside(conflict_groups_path, paper2_root=root)
    assignments_file, assignments_payload = _read_object_inside(assignments_path, paper2_root=root)
    conflict_hash = _conflict_groups_hash(conflicts)
    assignments, specifications = _fixed_assignments(
        conflicts, assignments_payload, require_formal_counts=require_formal_counts
    )
    bindings = {
        "catalog_sha256": str(conflicts.get("catalog_sha256", "")),
        "conflict_groups_sha256": conflict_hash,
        "conflict_groups_file_sha256": file_sha256(conflicts_path),
        "assignments_file_sha256": file_sha256(assignments_file),
        "role_assignment_sha256": canonical_sha256(assignments),
        "required_observations_sha256": canonical_sha256(specifications),
    }
    if not is_sha256(bindings["catalog_sha256"]):
        raise ValueError("conflict groups do not bind a valid catalog SHA-256")
    bindings["input_bindings_sha256"] = canonical_sha256(bindings)
    discovery_directory = guard_paper2_path(discovery_root, paper2_root=root)
    output_path = guard_paper2_path(output, paper2_root=root)

    entries: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        existing = read_json(output_path)
        if not isinstance(existing, dict) or existing.get("schema") != SCHEMAS["edr"]:
            raise ValueError("existing EDR checkpoint uses an unsupported schema")
        for key, expected in bindings.items():
            if existing.get(key) != expected:
                raise ValueError(f"existing EDR checkpoint has stale {key}")
        specifications_by_id = {
            str(specification["observation_id"]): specification for specification in specifications
        }
        for raw in existing.get("entries", []):
            entry = dict(raw)
            observation_id = str(entry.get("observation_id", ""))
            if (
                not observation_id
                or observation_id in entries
                or observation_id not in specifications_by_id
            ):
                raise ValueError("existing EDR checkpoint has an unexpected observation")
            _verify_reusable_edr_entry(
                entry, specifications_by_id[observation_id], paper2_root=root
            )
            entries[observation_id] = entry

    _write_checkpoint(
        output_path,
        _edr_payload(bindings=bindings, specifications=specifications, entries=entries),
        paper2_root=root,
    )

    pending = [
        (specification, entries.get(str(specification["observation_id"]), {}))
        for specification in specifications
        if entries.get(str(specification["observation_id"]), {}).get("status")
        not in EDR_TERMINAL_STATUSES
    ]

    def commit(result: tuple[str, dict[str, Any]]) -> None:
        observation_id, entry = result
        entries[observation_id] = entry
        _write_checkpoint(
            output_path,
            _edr_payload(bindings=bindings, specifications=specifications, entries=entries),
            paper2_root=root,
        )

    if workers == 1:
        for specification, previous in pending:
            commit(
                _discover_edr_entry(
                    specification,
                    previous=previous,
                    discovery_directory=discovery_directory,
                    paper2_root=root,
                    max_attempts=max_attempts,
                    backoff_initial_seconds=backoff_initial_seconds,
                    backoff_max_seconds=backoff_max_seconds,
                    discoverer=discoverer,
                    sleeper=sleeper,
                )
            )
    else:
        task_iterator = iter(pending)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="paper2-edr") as executor:
            in_flight: dict[Future[tuple[str, dict[str, Any]]], int] = {}

            def submit_next() -> bool:
                try:
                    specification, previous = next(task_iterator)
                except StopIteration:
                    return False
                future = executor.submit(
                    _discover_edr_entry,
                    specification,
                    previous=previous,
                    discovery_directory=discovery_directory,
                    paper2_root=root,
                    max_attempts=max_attempts,
                    backoff_initial_seconds=backoff_initial_seconds,
                    backoff_max_seconds=backoff_max_seconds,
                    discoverer=discoverer,
                    sleeper=sleeper,
                )
                in_flight[future] = int(specification["request_order"])
                return True

            for _ in range(min(workers, len(pending))):
                submit_next()
            while in_flight:
                completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in sorted(completed, key=in_flight.__getitem__):
                    del in_flight[future]
                    commit(future.result())
                    submit_next()
    completed_payload = read_json(output_path)
    if completed_payload.get("state") == "complete":
        validate_edr_index(
            completed_payload,
            catalog_sha256=bindings["catalog_sha256"],
            conflict_groups_sha256=conflict_hash,
        )
    return completed_payload

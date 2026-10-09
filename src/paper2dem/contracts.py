from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse
import json
import re

import numpy as np

from .hashing import (
    PAPER2_ROOT,
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    is_sha256,
    read_json,
    write_canonical_json,
    write_immutable_bytes,
)


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ROLES = frozenset(
    {
        "bridge_train",
        "bridge_val",
        "uncertainty_calibration",
        "paper2_locked_test",
        "sealed_reserve",
        "paper1_bridge_test",
        "locked_external",
        "pilot",
        "candidate_pool",
    }
)
FORBIDDEN_TRUTH_KEY_PARTS = (
    "reference",
    "ground_truth",
    "truth_path",
    "target_dem",
    "target_value",
)


def _require_sha256(name: str, value: str) -> None:
    if not is_sha256(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty")


def _readonly(values: Any, dtype: Any | None = None) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _reject_truth_bearing(value: Any, *, location: str = "payload") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if any(part in normalized for part in FORBIDDEN_TRUTH_KEY_PARTS):
                raise ValueError(f"{location} contains forbidden truth key: {key}")
            _reject_truth_bearing(item, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_truth_bearing(item, location=f"{location}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        if "reference.npz" in lowered or "ground-truth" in lowered or "ground_truth" in lowered:
            raise ValueError(f"{location} contains a forbidden truth locator")


@dataclass(frozen=True)
class AssetFile:
    kind: str
    relative_path: str | None
    sha256: str | None
    byte_count: int | None
    status: str = "present"

    def validate(self) -> None:
        _require_nonempty("asset file kind", self.kind)
        if self.status not in {"present", "missing", "rejected"}:
            raise ValueError(f"unsupported asset file status: {self.status}")
        if self.relative_path is not None:
            relative = Path(self.relative_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("asset paths must be relative and cannot traverse parents")
        if self.status == "present":
            if self.relative_path is None or self.sha256 is None or self.byte_count is None:
                raise ValueError("present assets require path, hash, and byte count")
            _require_sha256("asset file hash", self.sha256)
            if self.byte_count < 0:
                raise ValueError("asset byte count cannot be negative")
        elif self.sha256 is not None:
            _require_sha256("asset file hash", self.sha256)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "kind": self.kind,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AssetFile":
        return cls(
            kind=str(value["kind"]),
            relative_path=None if value.get("relative_path") is None else str(value["relative_path"]),
            sha256=None if value.get("sha256") is None else str(value["sha256"]),
            byte_count=None if value.get("byte_count") is None else int(value["byte_count"]),
            status=str(value.get("status", "present")),
        )


@dataclass(frozen=True)
class AssetQA:
    image_coverage_fraction: float | None = None
    finite_dem_fraction: float | None = None
    alignment_median_pixels: float | None = None
    alignment_p95_pixels: float | None = None
    passed: bool = False
    failure_reasons: tuple[str, ...] = ()

    def validate(self) -> None:
        for name, value in (
            ("image coverage", self.image_coverage_fraction),
            ("finite DEM fraction", self.finite_dem_fraction),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        for name, value in (
            ("alignment median", self.alignment_median_pixels),
            ("alignment p95", self.alignment_p95_pixels),
        ):
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        measured = (
            self.image_coverage_fraction is not None
            and self.finite_dem_fraction is not None
            and self.alignment_median_pixels is not None
            and self.alignment_p95_pixels is not None
        )
        thresholds_pass = measured and (
            self.image_coverage_fraction >= 0.95
            and self.finite_dem_fraction >= 0.99
            and self.alignment_median_pixels <= 1.0
            and self.alignment_p95_pixels <= 2.0
        )
        if self.passed and not thresholds_pass:
            raise ValueError("asset cannot pass QA without satisfying every locked threshold")
        if self.passed and self.failure_reasons:
            raise ValueError("passed QA cannot contain failure reasons")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "image_coverage_fraction": self.image_coverage_fraction,
            "finite_dem_fraction": self.finite_dem_fraction,
            "alignment_median_pixels": self.alignment_median_pixels,
            "alignment_p95_pixels": self.alignment_p95_pixels,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AssetQA":
        def optional_float(name: str) -> float | None:
            return None if value.get(name) is None else float(value[name])

        return cls(
            image_coverage_fraction=optional_float("image_coverage_fraction"),
            finite_dem_fraction=optional_float("finite_dem_fraction"),
            alignment_median_pixels=optional_float("alignment_median_pixels"),
            alignment_p95_pixels=optional_float("alignment_p95_pixels"),
            passed=bool(value.get("passed", False)),
            failure_reasons=tuple(str(item) for item in value.get("failure_reasons", [])),
        )


@dataclass(frozen=True)
class AssetRecord:
    product_id: str
    mission: str
    observation_ids: tuple[str, ...]
    geographic_group_id: str
    role: str
    bbox_lonlat: tuple[float, float, float, float]
    latitude_band: int | None
    terrain_quartile: int | None
    urls: tuple[str, ...] = ()
    files: tuple[AssetFile, ...] = ()
    imagery_lineage: tuple[str, ...] = ()
    dem_lineage: tuple[str, ...] = ()
    prepared_manifest_sha256: str | None = None
    problem_sha256: str | None = None
    seen_by_paper1: bool = False
    qa: AssetQA = field(default_factory=AssetQA)

    def validate(self) -> None:
        _require_nonempty("product_id", self.product_id)
        _require_nonempty("mission", self.mission)
        _require_nonempty("geographic_group_id", self.geographic_group_id)
        if self.role not in ROLES:
            raise ValueError(f"unsupported asset role: {self.role}")
        if not self.observation_ids or len(set(self.observation_ids)) != len(self.observation_ids):
            raise ValueError("observation_ids must be nonempty and unique")
        if len(self.bbox_lonlat) != 4:
            raise ValueError("bbox_lonlat must have four values")
        west, south, east, north = self.bbox_lonlat
        if not all(np.isfinite(value) for value in self.bbox_lonlat):
            raise ValueError("bbox_lonlat must be finite")
        if west > east or south > north:
            raise ValueError("wrapped or inverted bounding boxes must be normalized first")
        if self.latitude_band is not None and self.latitude_band not in {0, 1, 2}:
            raise ValueError("latitude_band must be 0, 1, or 2")
        if self.terrain_quartile is not None and self.terrain_quartile not in {0, 1, 2, 3}:
            raise ValueError("terrain_quartile must be 0, 1, 2, or 3")
        if self.role in {
            "bridge_train",
            "bridge_val",
            "uncertainty_calibration",
            "paper2_locked_test",
            "sealed_reserve",
        } and (self.latitude_band is None or self.terrain_quartile is None):
            raise ValueError("formal CTX roles require locked latitude and terrain strata")
        if self.seen_by_paper1 and self.role not in {
            "bridge_train",
            "paper1_bridge_test",
            "locked_external",
            "pilot",
        }:
            raise ValueError("Paper1-seen products cannot enter new Paper2 validation roles")
        for url in self.urls:
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https", "ftp"} or not parsed.netloc:
                raise ValueError(f"invalid asset URL: {url}")
        if len(set(self.urls)) != len(self.urls):
            raise ValueError("asset URLs must be unique")
        kinds = [asset.kind for asset in self.files]
        if len(kinds) != len(set(kinds)):
            raise ValueError("asset file kinds must be unique per product")
        for asset in self.files:
            asset.validate()
        for name, value in (
            ("prepared manifest", self.prepared_manifest_sha256),
            ("problem", self.problem_sha256),
        ):
            if value is not None:
                _require_sha256(f"{name} hash", value)
        self.qa.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "product_id": self.product_id,
            "mission": self.mission,
            "observation_ids": list(self.observation_ids),
            "geographic_group_id": self.geographic_group_id,
            "role": self.role,
            "bbox_lonlat": list(self.bbox_lonlat),
            "latitude_band": self.latitude_band,
            "terrain_quartile": self.terrain_quartile,
            "urls": list(self.urls),
            "files": [item.to_dict() for item in self.files],
            "imagery_lineage": list(self.imagery_lineage),
            "dem_lineage": list(self.dem_lineage),
            "prepared_manifest_sha256": self.prepared_manifest_sha256,
            "problem_sha256": self.problem_sha256,
            "seen_by_paper1": self.seen_by_paper1,
            "qa": self.qa.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AssetRecord":
        return cls(
            product_id=str(value["product_id"]),
            mission=str(value["mission"]),
            observation_ids=tuple(str(item) for item in value["observation_ids"]),
            geographic_group_id=str(value["geographic_group_id"]),
            role=str(value["role"]),
            bbox_lonlat=tuple(float(item) for item in value["bbox_lonlat"]),
            latitude_band=None if value.get("latitude_band") is None else int(value["latitude_band"]),
            terrain_quartile=None if value.get("terrain_quartile") is None else int(value["terrain_quartile"]),
            urls=tuple(str(item) for item in value.get("urls", [])),
            files=tuple(AssetFile.from_dict(item) for item in value.get("files", [])),
            imagery_lineage=tuple(str(item) for item in value.get("imagery_lineage", [])),
            dem_lineage=tuple(str(item) for item in value.get("dem_lineage", [])),
            prepared_manifest_sha256=value.get("prepared_manifest_sha256"),
            problem_sha256=value.get("problem_sha256"),
            seen_by_paper1=bool(value.get("seen_by_paper1", False)),
            qa=AssetQA.from_dict(dict(value.get("qa", {}))),
        )


@dataclass(frozen=True)
class Paper1SourceLock:
    release_commit: str
    source_lock_sha256: str
    benchmark_lock_sha256: str
    catalog_sha256: str
    prior_sha256: str

    def validate(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", self.release_commit):
            raise ValueError("Paper1 release_commit must be a full lowercase Git commit")
        _require_sha256("Paper1 source-lock hash", self.source_lock_sha256)
        _require_sha256("Paper1 benchmark-lock hash", self.benchmark_lock_sha256)
        _require_sha256("Paper1 catalog hash", self.catalog_sha256)
        _require_sha256("Paper1 prior hash", self.prior_sha256)

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return {
            "release_commit": self.release_commit,
            "source_lock_sha256": self.source_lock_sha256,
            "benchmark_lock_sha256": self.benchmark_lock_sha256,
            "catalog_sha256": self.catalog_sha256,
            "prior_sha256": self.prior_sha256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Paper1SourceLock":
        return cls(**{key: str(value[key]) for key in (
            "release_commit",
            "source_lock_sha256",
            "benchmark_lock_sha256",
            "catalog_sha256",
            "prior_sha256",
        )})


@dataclass(frozen=True)
class AssetLock:
    SCHEMA: ClassVar[str] = "paper2.asset-lock/v1"

    paper1: Paper1SourceLock
    source_catalog_sha256: str
    grouping_buffer_degrees: float
    grouping_algorithm: str
    assignment_seed_sha256: str
    assets: tuple[AssetRecord, ...]

    def validate(self) -> None:
        self.paper1.validate()
        _require_sha256("source catalog hash", self.source_catalog_sha256)
        _require_sha256("assignment seed hash", self.assignment_seed_sha256)
        if not np.isfinite(self.grouping_buffer_degrees) or self.grouping_buffer_degrees < 0:
            raise ValueError("grouping buffer must be finite and nonnegative")
        _require_nonempty("grouping algorithm", self.grouping_algorithm)
        if not self.assets:
            raise ValueError("asset lock cannot be empty")
        product_ids = [asset.product_id for asset in self.assets]
        if len(product_ids) != len(set(product_ids)):
            raise ValueError("asset product IDs must be unique")
        group_roles: dict[str, str] = {}
        group_strata: dict[str, tuple[int | None, int | None]] = {}
        observation_roles: dict[str, str] = {}
        for asset in self.assets:
            asset.validate()
            previous = group_roles.setdefault(asset.geographic_group_id, asset.role)
            if previous != asset.role:
                raise ValueError("a geographic group cannot be split across roles")
            stratum = (asset.latitude_band, asset.terrain_quartile)
            previous_stratum = group_strata.setdefault(asset.geographic_group_id, stratum)
            if previous_stratum != stratum:
                raise ValueError("a geographic group cannot be split across strata")
            for observation_id in asset.observation_ids:
                previous = observation_roles.setdefault(observation_id, asset.role)
                if previous != asset.role:
                    raise ValueError("a raw observation cannot be shared across roles")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.SCHEMA,
            "paper1": self.paper1.to_dict(),
            "source_catalog_sha256": self.source_catalog_sha256,
            "grouping_buffer_degrees": self.grouping_buffer_degrees,
            "grouping_algorithm": self.grouping_algorithm,
            "assignment_seed_sha256": self.assignment_seed_sha256,
            "assets": [asset.to_dict() for asset in self.assets],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def write(self, path: str | Path, *, paper2_root: str | Path = PAPER2_ROOT) -> Path:
        return write_canonical_json(path, self.to_dict(), paper2_root=paper2_root)

    @classmethod
    def read(cls, path: str | Path) -> "AssetLock":
        payload = read_json(path)
        if payload.get("schema") != cls.SCHEMA:
            raise ValueError(f"unsupported asset lock schema: {payload.get('schema')}")
        result = cls(
            paper1=Paper1SourceLock.from_dict(payload["paper1"]),
            source_catalog_sha256=str(payload["source_catalog_sha256"]),
            grouping_buffer_degrees=float(payload["grouping_buffer_degrees"]),
            grouping_algorithm=str(payload["grouping_algorithm"]),
            assignment_seed_sha256=str(payload["assignment_seed_sha256"]),
            assets=tuple(AssetRecord.from_dict(item) for item in payload["assets"]),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class FootprintContract:
    shape: str
    diameter_m: float
    quadrature_points: int
    require_full_support: bool

    def validate(self) -> None:
        if self.shape not in {"point", "disk"}:
            raise ValueError("footprint shape must be point or disk")
        if not np.isfinite(self.diameter_m) or self.diameter_m < 0:
            raise ValueError("footprint diameter must be finite and nonnegative")
        if self.shape == "point" and self.diameter_m != 0:
            raise ValueError("point footprints must have zero diameter")
        if self.quadrature_points <= 0:
            raise ValueError("quadrature_points must be positive")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "shape": self.shape,
            "diameter_m": self.diameter_m,
            "quadrature_points": self.quadrature_points,
            "require_full_support": self.require_full_support,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FootprintContract":
        return cls(
            shape=str(value["shape"]),
            diameter_m=float(value["diameter_m"]),
            quadrature_points=int(value["quadrature_points"]),
            require_full_support=bool(value["require_full_support"]),
        )


@dataclass(frozen=True)
class PlanPoint:
    rank: int
    candidate_id: str
    track_id: str
    x: float
    y: float

    def validate(self) -> None:
        if self.rank <= 0:
            raise ValueError("plan ranks are one-based positive integers")
        _require_nonempty("candidate_id", self.candidate_id)
        _require_nonempty("track_id", self.track_id)
        if not np.isfinite(self.x) or not np.isfinite(self.y):
            raise ValueError("plan coordinates must be finite")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "rank": self.rank,
            "candidate_id": self.candidate_id,
            "track_id": self.track_id,
            "x": self.x,
            "y": self.y,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PlanPoint":
        return cls(
            rank=int(value["rank"]),
            candidate_id=str(value["candidate_id"]),
            track_id=str(value["track_id"]),
            x=float(value["x"]),
            y=float(value["y"]),
        )


@dataclass(frozen=True)
class PlanContract:
    SCHEMA: ClassVar[str] = "paper2.plan/v1"

    tile_id: str
    scenario_id: str
    method: str
    budget: int
    crs_wkt: str
    transform: tuple[float, float, float, float, float, float]
    footprint: FootprintContract
    noise_std_m: float
    candidate_geometry_sha256: str
    problem_sha256: str
    paper1: Paper1SourceLock
    source_selection_sha256: str
    points: tuple[PlanPoint, ...]
    prefix_of_sha256: str | None = None

    def validate(self) -> None:
        _require_nonempty("tile_id", self.tile_id)
        _require_nonempty("scenario_id", self.scenario_id)
        _require_nonempty("method", self.method)
        _require_nonempty("crs_wkt", self.crs_wkt)
        if len(self.transform) != 6 or not all(np.isfinite(item) for item in self.transform):
            raise ValueError("transform must contain six finite affine coefficients")
        if self.budget < 0 or self.budget != len(self.points):
            raise ValueError("budget must equal the non-negative number of plan points")
        if not np.isfinite(self.noise_std_m) or self.noise_std_m < 0:
            raise ValueError("noise_std_m must be finite and nonnegative")
        self.footprint.validate()
        self.paper1.validate()
        for name, value in (
            ("candidate geometry", self.candidate_geometry_sha256),
            ("problem", self.problem_sha256),
            ("source selection", self.source_selection_sha256),
        ):
            _require_sha256(f"{name} hash", value)
        if self.prefix_of_sha256 is not None:
            _require_sha256("prefix parent hash", self.prefix_of_sha256)
        ranks = [point.rank for point in self.points]
        if ranks != list(range(1, self.budget + 1)):
            raise ValueError("plan points must be ordered by contiguous one-based rank")
        identifiers = [point.candidate_id for point in self.points]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("plan candidate IDs must be unique")
        for point in self.points:
            point.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = {
            "schema": self.SCHEMA,
            "tile_id": self.tile_id,
            "scenario_id": self.scenario_id,
            "method": self.method,
            "budget": self.budget,
            "crs_wkt": self.crs_wkt,
            "transform": list(self.transform),
            "footprint": self.footprint.to_dict(),
            "noise_std_m": self.noise_std_m,
            "candidate_geometry_sha256": self.candidate_geometry_sha256,
            "problem_sha256": self.problem_sha256,
            "paper1": self.paper1.to_dict(),
            "source_selection_sha256": self.source_selection_sha256,
            "points": [point.to_dict() for point in self.points],
            "prefix_of_sha256": self.prefix_of_sha256,
        }
        _reject_truth_bearing(payload)
        return payload

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def prefix(self, budget: int) -> "PlanContract":
        if budget < 0 or budget > self.budget:
            raise ValueError("prefix budget must be in [0, Kmax]")
        if budget == self.budget:
            return self
        result = PlanContract(
            tile_id=self.tile_id,
            scenario_id=self.scenario_id,
            method=self.method,
            budget=budget,
            crs_wkt=self.crs_wkt,
            transform=self.transform,
            footprint=self.footprint,
            noise_std_m=self.noise_std_m,
            candidate_geometry_sha256=self.candidate_geometry_sha256,
            problem_sha256=self.problem_sha256,
            paper1=self.paper1,
            source_selection_sha256=self.source_selection_sha256,
            points=self.points[:budget],
            prefix_of_sha256=self.sha256,
        )
        result.validate()
        return result

    def write(self, path: str | Path, *, paper2_root: str | Path = PAPER2_ROOT) -> Path:
        return write_canonical_json(path, self.to_dict(), paper2_root=paper2_root)

    @classmethod
    def read(cls, path: str | Path) -> "PlanContract":
        payload = read_json(path)
        if payload.get("schema") != cls.SCHEMA:
            raise ValueError(f"unsupported plan schema: {payload.get('schema')}")
        _reject_truth_bearing(payload)
        result = cls(
            tile_id=str(payload["tile_id"]),
            scenario_id=str(payload["scenario_id"]),
            method=str(payload["method"]),
            budget=int(payload["budget"]),
            crs_wkt=str(payload["crs_wkt"]),
            transform=tuple(float(item) for item in payload["transform"]),
            footprint=FootprintContract.from_dict(payload["footprint"]),
            noise_std_m=float(payload["noise_std_m"]),
            candidate_geometry_sha256=str(payload["candidate_geometry_sha256"]),
            problem_sha256=str(payload["problem_sha256"]),
            paper1=Paper1SourceLock.from_dict(payload["paper1"]),
            source_selection_sha256=str(payload["source_selection_sha256"]),
            points=tuple(PlanPoint.from_dict(item) for item in payload["points"]),
            prefix_of_sha256=payload.get("prefix_of_sha256"),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class LaserObservations:
    SCHEMA: ClassVar[str] = "paper2.laser-observations/v1"

    tile_id: str
    plan_sha256: str
    replicate: int
    candidate_ids: np.ndarray
    coordinates: np.ndarray
    return_values_m: np.ndarray
    quality_flags: np.ndarray
    noise_repeat_numbers: np.ndarray
    noise_std_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_ids", _readonly(self.candidate_ids, dtype=str))
        object.__setattr__(self, "coordinates", _readonly(self.coordinates, dtype=np.float64))
        object.__setattr__(self, "return_values_m", _readonly(self.return_values_m, dtype=np.float64))
        object.__setattr__(self, "quality_flags", _readonly(self.quality_flags, dtype=np.uint8))
        object.__setattr__(
            self,
            "noise_repeat_numbers",
            _readonly(self.noise_repeat_numbers, dtype=np.int32),
        )

    def validate(self) -> None:
        _require_nonempty("tile_id", self.tile_id)
        _require_sha256("plan hash", self.plan_sha256)
        if self.replicate < 0:
            raise ValueError("replicate must be nonnegative")
        count = self.return_values_m.shape[0]
        if self.return_values_m.ndim != 1:
            raise ValueError("return_values_m must be one-dimensional")
        if self.candidate_ids.shape != (count,):
            raise ValueError("candidate_ids must align with return values")
        if self.coordinates.shape != (count, 2):
            raise ValueError("coordinates must have shape (n, 2)")
        if self.quality_flags.shape != (count,) or self.noise_repeat_numbers.shape != (count,):
            raise ValueError("quality flags and repeats must align with return values")
        if len(set(self.candidate_ids.tolist())) != count:
            raise ValueError("observation candidate IDs must be unique")
        if np.any(self.noise_repeat_numbers != self.replicate):
            raise ValueError("every noise repeat number must equal replicate")
        good = self.quality_flags == 0
        if np.any(~np.isfinite(self.coordinates)):
            raise ValueError("observation coordinates must be finite")
        if np.any(~np.isfinite(self.return_values_m[good])):
            raise ValueError("successful returns must be finite")
        if np.any(np.isfinite(self.return_values_m[~good])):
            raise ValueError("failed returns must be NaN")
        if not np.isfinite(self.noise_std_m) or self.noise_std_m < 0:
            raise ValueError("noise_std_m must be finite and nonnegative")

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "tile_id": self.tile_id,
            "plan_sha256": self.plan_sha256,
            "replicate": self.replicate,
            "noise_std_m": self.noise_std_m,
            "quality_flag_meanings": {"0": "valid", "1": "invalid_support", "2": "nonfinite"},
        }

    def write(self, path: str | Path, *, paper2_root: str | Path = PAPER2_ROOT) -> Path:
        self.validate()
        stream = BytesIO()
        np.savez_compressed(
            stream,
            metadata=np.asarray(canonical_json_bytes(self.metadata).decode("utf-8")),
            candidate_ids=self.candidate_ids,
            coordinates=self.coordinates,
            return_values_m=self.return_values_m,
            quality_flags=self.quality_flags,
            noise_repeat_numbers=self.noise_repeat_numbers,
        )
        return write_immutable_bytes(path, stream.getvalue(), paper2_root=paper2_root)

    @classmethod
    def read(cls, path: str | Path) -> "LaserObservations":
        with np.load(path, allow_pickle=False) as archive:
            forbidden = [name for name in archive.files if "reference" in name.lower() or "truth" in name.lower()]
            if forbidden:
                raise ValueError(f"observation archive contains forbidden fields: {forbidden}")
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("schema") != cls.SCHEMA:
                raise ValueError(f"unsupported observation schema: {metadata.get('schema')}")
            _reject_truth_bearing(metadata)
            result = cls(
                tile_id=str(metadata["tile_id"]),
                plan_sha256=str(metadata["plan_sha256"]),
                replicate=int(metadata["replicate"]),
                noise_std_m=float(metadata["noise_std_m"]),
                candidate_ids=archive["candidate_ids"],
                coordinates=archive["coordinates"],
                return_values_m=archive["return_values_m"],
                quality_flags=archive["quality_flags"],
                noise_repeat_numbers=archive["noise_repeat_numbers"],
            )
        result.validate()
        return result


@dataclass(frozen=True)
class ReconstructionProblem:
    SCHEMA: ClassVar[str] = "paper2.reconstruction-problem/v1"

    tile_id: str
    z0: np.ndarray
    image: np.ndarray
    valid_mask: np.ndarray
    observation_candidate_ids: np.ndarray
    observation_coordinates: np.ndarray
    observation_values_m: np.ndarray
    observation_quality_flags: np.ndarray
    plan_sha256: str
    observations_sha256: str
    frozen_model_hashes: tuple[tuple[str, str], ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "z0", _readonly(self.z0, dtype=np.float32))
        object.__setattr__(self, "image", _readonly(self.image, dtype=np.float32))
        object.__setattr__(self, "valid_mask", _readonly(self.valid_mask, dtype=bool))
        object.__setattr__(
            self,
            "observation_candidate_ids",
            _readonly(self.observation_candidate_ids, dtype=str),
        )
        object.__setattr__(
            self,
            "observation_coordinates",
            _readonly(self.observation_coordinates, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "observation_values_m",
            _readonly(self.observation_values_m, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "observation_quality_flags",
            _readonly(self.observation_quality_flags, dtype=np.uint8),
        )

    def validate(self) -> None:
        _require_nonempty("tile_id", self.tile_id)
        _require_sha256("plan hash", self.plan_sha256)
        _require_sha256("observation hash", self.observations_sha256)
        if self.z0.ndim != 2 or self.valid_mask.shape != self.z0.shape:
            raise ValueError("z0 and valid_mask must be aligned 2-D arrays")
        if self.image.ndim not in {2, 3}:
            raise ValueError("image must be a 2-D or 3-D array")
        image_aligned = self.image.shape[-2:] == self.z0.shape or self.image.shape[:2] == self.z0.shape
        if not image_aligned:
            raise ValueError("image spatial dimensions must align with z0")
        if not np.any(self.valid_mask):
            raise ValueError("reconstruction problem contains no valid DEM pixels")
        if np.any(~np.isfinite(self.z0[self.valid_mask])):
            raise ValueError("z0 must be finite over valid pixels")
        count = self.observation_values_m.shape[0]
        if self.observation_values_m.ndim != 1:
            raise ValueError("observation values must be one-dimensional")
        if self.observation_candidate_ids.shape != (count,):
            raise ValueError("observation IDs must align with values")
        if self.observation_coordinates.shape != (count, 2):
            raise ValueError("observation coordinates must have shape (n, 2)")
        if self.observation_quality_flags.shape != (count,):
            raise ValueError("observation flags must align with values")
        good = self.observation_quality_flags == 0
        if np.any(~np.isfinite(self.observation_values_m[good])):
            raise ValueError("valid laser observations must be finite")
        if not self.frozen_model_hashes:
            raise ValueError("at least one frozen model hash is required")
        names = [name for name, _ in self.frozen_model_hashes]
        if len(names) != len(set(names)):
            raise ValueError("frozen model hash names must be unique")
        for name, digest in self.frozen_model_hashes:
            _require_nonempty("frozen model name", name)
            _require_sha256(f"frozen model {name}", digest)
        _reject_truth_bearing(self.metadata, location="reconstruction metadata")

    @property
    def contract_metadata(self) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "tile_id": self.tile_id,
            "plan_sha256": self.plan_sha256,
            "observations_sha256": self.observations_sha256,
            "frozen_model_hashes": dict(self.frozen_model_hashes),
            "metadata": self.metadata,
        }
        _reject_truth_bearing(payload)
        return payload

    def write(self, path: str | Path, *, paper2_root: str | Path = PAPER2_ROOT) -> Path:
        self.validate()
        stream = BytesIO()
        np.savez_compressed(
            stream,
            contract=np.asarray(canonical_json_bytes(self.contract_metadata).decode("utf-8")),
            z0=self.z0,
            image=self.image,
            valid_mask=self.valid_mask,
            observation_candidate_ids=self.observation_candidate_ids,
            observation_coordinates=self.observation_coordinates,
            observation_values_m=self.observation_values_m,
            observation_quality_flags=self.observation_quality_flags,
        )
        return write_immutable_bytes(path, stream.getvalue(), paper2_root=paper2_root)

    @classmethod
    def read(cls, path: str | Path) -> "ReconstructionProblem":
        with np.load(path, allow_pickle=False) as archive:
            forbidden = [
                name
                for name in archive.files
                if any(part in name.lower() for part in FORBIDDEN_TRUTH_KEY_PARTS)
            ]
            if forbidden:
                raise ValueError(f"reconstruction archive contains forbidden fields: {forbidden}")
            metadata = json.loads(str(archive["contract"].item()))
            if metadata.get("schema") != cls.SCHEMA:
                raise ValueError(f"unsupported reconstruction schema: {metadata.get('schema')}")
            _reject_truth_bearing(metadata)
            result = cls(
                tile_id=str(metadata["tile_id"]),
                z0=archive["z0"],
                image=archive["image"],
                valid_mask=archive["valid_mask"],
                observation_candidate_ids=archive["observation_candidate_ids"],
                observation_coordinates=archive["observation_coordinates"],
                observation_values_m=archive["observation_values_m"],
                observation_quality_flags=archive["observation_quality_flags"],
                plan_sha256=str(metadata["plan_sha256"]),
                observations_sha256=str(metadata["observations_sha256"]),
                frozen_model_hashes=tuple(
                    sorted((str(name), str(digest)) for name, digest in metadata["frozen_model_hashes"].items())
                ),
                metadata=dict(metadata.get("metadata", {})),
            )
        result.validate()
        return result


def reconstruction_from_observations(
    *,
    tile_id: str,
    z0: np.ndarray,
    image: np.ndarray,
    valid_mask: np.ndarray,
    observations: LaserObservations,
    observations_path: str | Path,
    frozen_model_hashes: dict[str, str],
    metadata: dict[str, Any] | None = None,
    paper2_root: str | Path = PAPER2_ROOT,
) -> ReconstructionProblem:
    observations.validate()
    if observations.tile_id != tile_id:
        raise ValueError("observation and reconstruction tile IDs do not match")
    persisted_path = guard_paper2_path(observations_path, paper2_root=paper2_root)
    persisted = LaserObservations.read(persisted_path)
    if (
        persisted.tile_id != observations.tile_id
        or persisted.plan_sha256 != observations.plan_sha256
        or persisted.replicate != observations.replicate
        or not np.array_equal(persisted.candidate_ids, observations.candidate_ids)
        or not np.array_equal(persisted.coordinates, observations.coordinates)
        or not np.array_equal(
            persisted.return_values_m,
            observations.return_values_m,
            equal_nan=True,
        )
        or not np.array_equal(persisted.quality_flags, observations.quality_flags)
    ):
        raise ValueError("observation object does not match its immutable source artifact")
    return ReconstructionProblem(
        tile_id=tile_id,
        z0=z0,
        image=image,
        valid_mask=valid_mask,
        observation_candidate_ids=observations.candidate_ids,
        observation_coordinates=observations.coordinates,
        observation_values_m=observations.return_values_m,
        observation_quality_flags=observations.quality_flags,
        plan_sha256=observations.plan_sha256,
        observations_sha256=file_sha256(persisted_path),
        frozen_model_hashes=tuple(sorted(frozen_model_hashes.items())),
        metadata={} if metadata is None else dict(metadata),
    )

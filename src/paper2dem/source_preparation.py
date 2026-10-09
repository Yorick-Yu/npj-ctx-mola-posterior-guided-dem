from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
import importlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError

import numpy as np

from .contracts import Paper1SourceLock, PlanContract
from .g0_sidecars import _conflict_groups_hash, _fixed_assignments
from .hashing import (
    PAPER2_ROOT,
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    read_json,
    write_canonical_json,
    write_immutable_bytes,
)
from .plan_import import DEFAULT_BUDGETS, import_paper1_kmax_plan, write_plan_family
from .official_assets import NetworkRetryExhaustedError, download_official_asset
from .source_prefetch import source_product_cache_lock


SOURCE_PREPARATION_SCHEMA = "paper2.g0.source-preparation/v1"
SOURCE_PREPARATION_AUDIT_SCHEMA = "paper2.g0.source-preparation-audit/v1"
SOURCE_PREPARATION_FAILURE_SCHEMA = "paper2.g0.source-preparation-failure/v1"
SOURCE_PREPARATION_INDEX_SCHEMA = "paper2.g0.source-preparation-index/v1"
SOURCE_PROTOCOL = "paper1-formal-no-existing-laser-v2-paper2-isolation-v1"
PAPER1_CODE_LOCK_SCHEMA = "paper2.paper1-code-lock/v1"
KMAX_METHOD = "discrete-faopt"
KMAX_BUDGET = 100
SOURCE_CACHE_MINIMUM_FREE_BYTES = 100 * 1024**3
SOURCE_CACHE_MISSING_LENGTH_BOUND_BYTES = 4 * 1024**3

_FORBIDDEN_PROBLEM_FIELDS = frozenset(
    {
        "reference",
        "reference_dem",
        "ground_truth",
        "target_dem",
        "target_values",
        "existing_coordinates",
        "existing_values",
        "existing_noise_std",
        "existing_track_ids",
        "existing_feature_vectors",
        "final_offsets",
        "mola_shots",
    }
)
_FORBIDDEN_PLAN_MARKERS = (
    "reference.npz",
    "/reference",
    "ground_truth",
    "target_value",
    "marginal_gain",
    "cumulative_gain",
    "conditional_density",
    "unit_cost_gain",
    "/home/",
    "/users/",
)
_TERMINAL_G0_PRODUCT_STATUSES = frozenset(
    {"qa_failed", "terminal_unavailable", "source_rejected"}
)
_LEGACY_ASTROGEO_S3_PREFIX = "https://astrogeo-ard.s3-us-west-2.amazonaws.com/"
_CANONICAL_ASTROGEO_S3_PREFIX = "https://astrogeo-ard.s3.us-west-2.amazonaws.com/"


class SourceRejectedError(ValueError):
    """A deterministic source-data defect that authorizes the fixed fallback."""


class SourceCapacityInsufficientError(RuntimeError):
    """A recoverable disk-reserve failure that must not reject a source product."""


def _canonical_transport_url(url: str) -> str:
    """Use the equivalent modern regional endpoint for the locked Astrogeology bucket."""
    value = str(url)
    if value.startswith(_LEGACY_ASTROGEO_S3_PREFIX):
        return _CANONICAL_ASTROGEO_S3_PREFIX + value[len(_LEGACY_ASTROGEO_S3_PREFIX) :]
    return value


def _is_deterministic_source_defect(error: Exception) -> bool:
    message = str(error).lower()
    markers = (
        "has no usable dtm asset",
        "cannot provide crop",
        "requested crop is larger than the raster",
        "cannot fill a raster without valid pixels",
        "valid fraction",
        "gradient anomaly ratio",
        "footprint filtering removed every future-track candidate",
    )
    return any(marker in message for marker in markers)


@dataclass(frozen=True)
class SourcePreparationOptions:
    crop_shape: tuple[int, int] = (512, 512)
    degradation_factor: int = 24
    along_track_spacing_m: float = 300.0
    cross_track_spacing_m: float = 1000.0
    minimum_spacing_m: float = 300.0
    target_count: int = 16384
    minimum_valid_fraction: float = 0.95
    maximum_gradient_anomaly_ratio: float = 50.0
    footprint_shape: str = "disk"
    footprint_diameter_m: float = 100.0
    footprint_quadrature_points: int = 32
    footprint_noise_std_m: float = 1.3
    footprint_require_full_support: bool = True
    seed: int = 0
    budgets: tuple[int, ...] = DEFAULT_BUDGETS

    def validate(self) -> None:
        if self.crop_shape != (512, 512):
            raise ValueError("formal Paper2 source preparation freezes a 512x512 crop")
        if self.degradation_factor != 24:
            raise ValueError("formal Paper2 controlled degradation freezes factor 24")
        if min(
            self.along_track_spacing_m,
            self.cross_track_spacing_m,
            self.minimum_spacing_m,
        ) <= 0:
            raise ValueError("track and minimum spacings must be positive")
        if self.target_count < 1:
            raise ValueError("target_count must be positive")
        if not 0 < self.minimum_valid_fraction <= 1:
            raise ValueError("minimum_valid_fraction must be in (0, 1]")
        if self.maximum_gradient_anomaly_ratio <= 0:
            raise ValueError("maximum_gradient_anomaly_ratio must be positive")
        if self.footprint_shape not in {"point", "disk"}:
            raise ValueError("footprint shape must be point or disk")
        if self.footprint_diameter_m < 0 or self.footprint_noise_std_m < 0:
            raise ValueError("footprint diameter/noise must be nonnegative")
        if self.footprint_shape == "point" and self.footprint_diameter_m != 0:
            raise ValueError("point footprints require zero diameter")
        if self.footprint_quadrature_points < 1:
            raise ValueError("footprint quadrature count must be positive")
        budgets = tuple(sorted(set(int(value) for value in self.budgets)))
        if budgets != DEFAULT_BUDGETS:
            raise ValueError(f"formal Paper2 budgets must be {DEFAULT_BUDGETS}")


@dataclass(frozen=True)
class BuiltSourceArtifacts:
    problem: Path
    reference: Path
    paper1_prepared_manifest: Path
    coreg_aux: Path
    raw_selection: Path
    source_http: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


SourceBuilder = Callable[
    [Mapping[str, Any], Mapping[str, Any], Path, Path, SourcePreparationOptions],
    BuiltSourceArtifacts,
]
HTTPMetadataReader = Callable[[str], Mapping[str, Any]]


def _safe_identifier(value: str, *, name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"{name} contains unsafe path characters: {value!r}")
    return value


def _recoverable_json(path: Path, payload: Mapping[str, Any], *, paper2_root: Path) -> Path:
    output = guard_paper2_path(path, paper2_root=paper2_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".checkpoint", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            from .hashing import canonical_json_bytes

            stream.write(canonical_json_bytes(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _artifact_record(path: Path, *, paper2_root: Path, access_scope: str) -> dict[str, Any]:
    resolved = guard_paper2_path(path, paper2_root=paper2_root)
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"source-preparation artifact is missing/empty: {resolved}")
    return {
        "relative_path": resolved.relative_to(paper2_root).as_posix(),
        "sha256": file_sha256(resolved),
        "byte_count": resolved.stat().st_size,
        "access_scope": access_scope,
    }


def _load_catalog(path: str | Path) -> tuple[Path, dict[str, dict[str, Any]]]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError("Paper2 source catalog must be a JSON list of objects")
    rows: dict[str, dict[str, Any]] = {}
    for raw in payload:
        row = dict(raw)
        product_id = str(row.get("tile_id", ""))
        if not product_id or product_id in rows:
            raise ValueError("catalog tile IDs must be nonempty and unique")
        if row.get("item_id") != product_id:
            raise ValueError(f"catalog item/tile identity differs: {product_id}")
        rows[product_id] = row
    return source, rows


def load_fixed_source_schedule(
    *,
    conflict_groups_path: str | Path,
    assignments_path: str | Path,
    catalog_path: str | Path,
    require_formal_counts: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    conflicts_path = Path(conflict_groups_path)
    assignments_file = Path(assignments_path)
    catalog_file, rows = _load_catalog(catalog_path)
    conflicts = read_json(conflicts_path)
    assignments_payload = read_json(assignments_file)
    assignments, _ = _fixed_assignments(
        conflicts,
        assignments_payload,
        require_formal_counts=require_formal_counts,
    )
    if conflicts.get("catalog_sha256") != file_sha256(catalog_file):
        raise ValueError("conflict groups are not bound to the supplied source catalog")
    for assignment in assignments:
        fixed_products = [
            str(assignment["representative_product_id"]),
            *(str(value) for value in assignment.get("fallback_product_ids", [])),
        ]
        missing = [product_id for product_id in fixed_products if product_id not in rows]
        if missing:
            raise ValueError(
                f"fixed group {assignment['group_id']} has products absent from catalog: {missing}"
            )
    bindings = {
        "conflict_groups_sha256": _conflict_groups_hash(conflicts),
        "conflict_groups_file_sha256": file_sha256(conflicts_path),
        "assignments_file_sha256": file_sha256(assignments_file),
        "role_assignment_sha256": canonical_sha256(assignments),
        "catalog_file_sha256": file_sha256(catalog_file),
    }
    return assignments, rows, bindings


def build_paper1_code_lock(paper1_source_root: str | Path) -> dict[str, Any]:
    root = Path(paper1_source_root).resolve()
    package = root / "laserinfo"
    if not package.is_dir() or not (package / "prepare_v2.py").is_file():
        raise FileNotFoundError("paper1_source_root must contain laserinfo/prepare_v2.py")
    files = [path for path in sorted(package.rglob("*.py")) if path.is_file()]
    if not files:
        raise ValueError("Paper1 source package contains no Python files")
    records = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
            "byte_count": path.stat().st_size,
        }
        for path in files
    ]
    payload = {
        "schema": PAPER1_CODE_LOCK_SCHEMA,
        "source_namespace": "paper1/src",
        "files": records,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def _sanitize_catalog_row(row: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = dict(row)
    assets = dict(row.get("assets", {}))
    dem_name = next((name for name in ("geoid_adjusted_dem", "dtm", "dem") if name in assets), None)
    if dem_name is None:
        raise SourceRejectedError(f"{row.get('tile_id')} has no controlled DTM asset")
    # The Paper1 adapter receives only the DTM. ORTHO is handled by the isolated
    # coreg_aux builder; final_offsets/MOLA shots are never passed to Paper1.
    sanitized["assets"] = {dem_name: _canonical_transport_url(str(assets[dem_name]))}
    sanitized["crop_window"] = None
    return sanitized


def _http_metadata(url: str) -> Mapping[str, Any]:
    request = Request(url, method="HEAD", headers={"User-Agent": "paper2dem-g0-source/1"})
    last_error: Exception | None = None
    for attempt in range(1, 9):
        try:
            with urlopen(request, timeout=120) as response:
                headers = response.headers
                length = headers.get("Content-Length")
                return {
                    "requested_url": url,
                    "resolved_url": response.geturl(),
                    "etag": headers.get("ETag"),
                    "last_modified": headers.get("Last-Modified"),
                    "content_type": headers.get("Content-Type"),
                    "content_length": None if length is None else int(length),
                    "http_status": int(getattr(response, "status", 200)),
                    "metadata_attempts": attempt,
                }
        except HTTPError:
            raise
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < 8:
                time.sleep(min(8.0, float(2 ** (attempt - 1))))
    raise RuntimeError(f"HTTP metadata lookup failed after 8 attempts: {url}") from last_error


def _import_paper1(paper1_source_root: Path) -> dict[str, Any]:
    root = paper1_source_root.resolve()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    modules = {
        name: importlib.import_module(name)
        for name in (
            "laserinfo.acquisition_v2",
            "laserinfo.prepare_v2",
            "laserinfo.raster",
            "laserinfo.runner_v2",
            "laserinfo.schemas",
            "laserinfo.storage",
        )
    }
    for name, module in modules.items():
        module_path = Path(str(module.__file__)).resolve()
        if root != module_path and root not in module_path.parents:
            raise ImportError(f"{name} resolved outside the locked Paper1 source root")
    return modules


def _write_coreg_aux(
    ortho_url: str,
    problem_path: Path,
    output: Path,
) -> None:
    try:
        import rasterio
        from affine import Affine
        from rasterio.warp import Resampling, reproject
    except ImportError as exc:  # pragma: no cover - production optional dependency
        raise RuntimeError("rasterio is required to derive the isolated coreg_aux crop") from exc
    with np.load(problem_path, allow_pickle=False) as archive:
        shape = tuple(int(value) for value in np.asarray(archive["coarse_dem"]).shape)
        transform = tuple(float(value) for value in np.asarray(archive["transform"]))
        crs_wkt = str(archive["crs_wkt"].item())
    destination = np.full(shape, np.nan, dtype=np.float32)
    environment = rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_TIMEOUT="120",
        GDAL_HTTP_MAX_RETRY="5",
        GDAL_HTTP_RETRY_DELAY="2",
    )
    with environment, rasterio.open(ortho_url) as source:
        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=Affine(*transform),
            dst_crs=crs_wkt or source.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
            num_threads=1,
        )
    if not np.any(np.isfinite(destination)):
        raise SourceRejectedError("ORTHO does not overlap the deterministic DTM crop")
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output,
        "w",
        driver="GTiff",
        height=shape[0],
        width=shape[1],
        count=1,
        dtype="float32",
        crs=crs_wkt,
        transform=Affine(*transform),
        nodata=np.nan,
        compress="DEFLATE",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dataset:
        dataset.write(destination, 1)
        dataset.update_tags(
            PAPER2_ROLE="coreg_aux_audit_only",
            COREG_AUX_INFLUENCED_PRIMARY_PIXELS="false",
        )


def _default_source_builder_factory(
    *,
    paper1_source_root: Path,
    http_metadata_reader: HTTPMetadataReader,
    paper2_root: Path,
) -> SourceBuilder:
    root = paper2_root.resolve()
    modules = _import_paper1(paper1_source_root)
    acquisition = modules["laserinfo.acquisition_v2"]
    prepare = modules["laserinfo.prepare_v2"]
    runner = modules["laserinfo.runner_v2"]
    schemas = modules["laserinfo.schemas"]
    storage = modules["laserinfo.storage"]

    def materialize(
        *, product_id: str, name: str, catalog_url: str
    ) -> tuple[Path, Mapping[str, Any], str, Mapping[str, Any]]:
        transport_url = _canonical_transport_url(catalog_url)
        directory = root / "data/source-preparation/source-cache" / product_id
        output = directory / f"{name}.tif"
        receipt_path = directory / f"{name}.receipt.json"
        with source_product_cache_lock(
            paper2_root=root,
            product_id=product_id,
            blocking=True,
        ) as acquired:
            if not acquired:  # pragma: no cover - blocking locks acquire or raise
                raise AssertionError("blocking source-product cache lock was not acquired")
            metadata: dict[str, Any] | None = None
            try:
                metadata = dict(http_metadata_reader(transport_url))
            except (HTTPError, URLError, TimeoutError, OSError, RuntimeError):
                # The canonical downloader below owns retry/rejection semantics.  A
                # failed early HEAD must not become a data-quality decision; use the
                # frozen conservative bound and retain the original GET+confirmation
                # behavior.
                metadata = None
            declared_length = None if metadata is None else metadata.get("content_length")
            declared_length_valid = (
                str(declared_length).isdigit() and int(declared_length) > 0
            )
            estimated_bytes = (
                int(declared_length)
                if declared_length_valid
                else SOURCE_CACHE_MISSING_LENGTH_BOUND_BYTES
            )
            available_bytes = int(shutil.disk_usage(root).free)
            required_available_bytes = (
                estimated_bytes + SOURCE_CACHE_MINIMUM_FREE_BYTES
            )
            if available_bytes < required_available_bytes:
                raise SourceCapacityInsufficientError(
                    "source-cache materialization would violate the frozen 100 GiB "
                    f"reserve: product={product_id} asset={name} "
                    f"available={available_bytes} required={required_available_bytes}"
                )
            capacity_preflight = {
                "estimated_asset_bytes": estimated_bytes,
                "size_basis": (
                    "official_http_content_length"
                    if declared_length_valid
                    else "frozen_missing_content_length_bound"
                ),
                "minimum_free_after_materialization_bytes": (
                    SOURCE_CACHE_MINIMUM_FREE_BYTES
                ),
                "filesystem_available_bytes": available_bytes,
                "required_available_bytes": required_available_bytes,
                "authorized": True,
            }
            for availability_attempt in range(2):
                try:
                    receipt = download_official_asset(
                        url=transport_url,
                        output=output,
                        receipt=receipt_path,
                        paper2_root=root,
                        max_attempts=16,
                        repair_invalid_existing=True,
                        # Controlled DTM/ORTHO GeoTIFFs can be tens to hundreds of MB.
                        # Preserve the canonical .download.part prefix so a transient
                        # proxy/SSH EOF is completed with a verified HTTP Range response.
                        resume_partial=True,
                    )
                    break
                except HTTPError as exc:
                    if exc.code not in {404, 410}:
                        raise
                    try:
                        http_metadata_reader(transport_url)
                    except HTTPError as confirmation:
                        if confirmation.code in {404, 410}:
                            raise SourceRejectedError(
                                f"official {name} asset is unavailable: HTTP {exc.code} "
                                f"confirmed by HTTP {confirmation.code} metadata response"
                            ) from exc
                        raise
                    except (URLError, TimeoutError, OSError) as confirmation:
                        raise NetworkRetryExhaustedError(
                            f"confirm source asset availability {transport_url}",
                            attempts=1,
                            last_error=confirmation,
                        ) from confirmation
                    if availability_attempt == 1:
                        raise NetworkRetryExhaustedError(
                            "download source asset after successful metadata probe "
                            f"{transport_url}",
                            attempts=2,
                            last_error=exc,
                        ) from exc
            else:  # pragma: no cover - the bounded loop either breaks or raises
                raise AssertionError("unreachable source-asset availability loop")
        if metadata is None:
            metadata = dict(http_metadata_reader(str(transport_url)))
        metadata["source_cache_capacity_preflight"] = capacity_preflight
        return output, receipt, transport_url, metadata

    def build(
        row: Mapping[str, Any],
        sanitized_row: Mapping[str, Any],
        staging: Path,
        prior_path: Path,
        options: SourcePreparationOptions,
    ) -> BuiltSourceArtifacts:
        footprint = schemas.FootprintSpec(
            shape=options.footprint_shape,
            diameter_m=options.footprint_diameter_m,
            quadrature_points=options.footprint_quadrature_points,
            noise_std_m=options.footprint_noise_std_m,
            require_full_support=options.footprint_require_full_support,
        )
        paper1_options = prepare.InformationPrepareOptions(
            crop_shape=options.crop_shape,
            degradation_factor=options.degradation_factor,
            along_track_spacing_m=options.along_track_spacing_m,
            cross_track_spacing_m=options.cross_track_spacing_m,
            minimum_spacing_m=options.minimum_spacing_m,
            target_count=options.target_count,
            min_valid_fraction=options.minimum_valid_fraction,
            max_gradient_anomaly_ratio=options.maximum_gradient_anomaly_ratio,
            footprint=footprint,
            stream_raster_assets=True,
            seed=options.seed,
        )
        product_id = _safe_identifier(str(row["tile_id"]), name="product_id")
        assets = dict(row.get("assets", {}))
        shape = tuple(int(value) for value in row.get("shape", ()))
        if len(shape) == 2 and any(
            observed > 0 and observed < required
            for observed, required in zip(shape, options.crop_shape, strict=True)
        ):
            raise SourceRejectedError(
                f"STAC shape {shape} cannot provide frozen crop {options.crop_shape}"
            )
        dem_key = next(
            name for name in ("geoid_adjusted_dem", "dtm", "dem") if name in assets
        )
        catalog_dtm_url = str(assets[dem_key])
        catalog_ortho_url = str(assets.get("orthoimage", ""))
        if not catalog_ortho_url:
            raise SourceRejectedError(f"{row.get('tile_id')} has no ORTHO coreg_aux source")
        local_dtm, dtm_receipt, dtm_transport_url, dtm_metadata = materialize(
            product_id=product_id,
            name="controlled-dtm",
            catalog_url=catalog_dtm_url,
        )
        local_ortho, ortho_receipt, ortho_transport_url, ortho_metadata = materialize(
            product_id=product_id,
            name="coreg-aux-ortho",
            catalog_url=catalog_ortho_url,
        )
        materialized_row = dict(sanitized_row)
        materialized_row["assets"] = {dem_key: str(local_dtm)}
        manifest = schemas.TileManifest.from_dict(materialized_row)
        try:
            problem, reference, _ = prepare.prepare_information_tile(
                manifest,
                staging / "paper1-preparation",
                paper1_options,
            )
        except (ValueError, RuntimeError) as exc:
            if _is_deterministic_source_defect(exc):
                raise SourceRejectedError(str(exc)) from exc
            raise
        prepared_problem = storage.load_information_problem(problem)
        prior = storage.load_prior_model(prior_path)
        missing_prior_features = [
            name for name in prior.feature_names if name not in prepared_problem.feature_names
        ]
        if missing_prior_features:
            center_by_name = dict(zip(prior.feature_names, prior.feature_center, strict=True))
            layers = [
                np.full(
                    prepared_problem.coarse_dem.shape,
                    float(center_by_name[name]),
                    dtype=np.float32,
                )
                for name in missing_prior_features
            ]
            feature_grid = np.concatenate(
                [prepared_problem.feature_grid, np.stack(layers, axis=-1)], axis=-1
            )
            prepared_problem = replace(
                prepared_problem,
                feature_grid=feature_grid,
                feature_names=(*prepared_problem.feature_names, *missing_prior_features),
                metadata={
                    **prepared_problem.metadata,
                    "image_feature_policy": (
                        "paper1-prior-center-imputation-no-image-access"
                    ),
                    "prior_center_imputed_feature_names": missing_prior_features,
                    "image_feature_source_accessed": False,
                    "coreg_aux_feature_values_accessed": False,
                },
            )
            storage.save_information_problem(prepared_problem, problem)
        prepared = problem.with_name("prepared-manifest.json")
        coreg_aux = staging / "coreg-aux.tif"
        _write_coreg_aux(str(local_ortho), problem, coreg_aux)
        selection_directory = staging / "paper1-selection-k100"
        runner.execute_information_selection(
            problem,
            prior_path,
            KMAX_METHOD,
            KMAX_BUDGET,
            selection_directory,
            seed=options.seed,
            options=acquisition.InformationSelectorOptions(
                enforce_min_spacing=True,
                local_swap_rounds=0,
            ),
            covariance_device="cpu",
        )
        source_http: dict[str, Mapping[str, Any]] = {}
        for name, catalog_url, transport_url, local_path, receipt, metadata in (
            (
                "controlled_dtm",
                catalog_dtm_url,
                dtm_transport_url,
                local_dtm,
                dtm_receipt,
                dtm_metadata,
            ),
            (
                "coreg_aux_ortho",
                catalog_ortho_url,
                ortho_transport_url,
                local_ortho,
                ortho_receipt,
                ortho_metadata,
            ),
        ):
            metadata = dict(metadata)
            metadata.update(
                {
                    "catalog_url": catalog_url,
                    "transport_url": transport_url,
                    "transport_endpoint_normalized": transport_url != catalog_url,
                    "materialized_relative_path": local_path.relative_to(root).as_posix(),
                    "materialized_sha256": str(receipt["sha256"]),
                    "materialized_bytes": int(receipt["bytes"]),
                    "materialized_download_attempts": int(receipt["download_attempts"]),
                }
            )
            source_http[name] = metadata
        return BuiltSourceArtifacts(
            problem=problem,
            reference=reference,
            paper1_prepared_manifest=prepared,
            coreg_aux=coreg_aux,
            raw_selection=selection_directory / "selection.json",
            source_http=source_http,
        )

    return build


def _problem_contract(path: Path, *, product_id: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        names = set(archive.files)
        leaked = sorted(names.intersection(_FORBIDDEN_PROBLEM_FIELDS))
        if leaked:
            raise ValueError(f"training problem contains forbidden fields: {leaked}")
        required = {
            "tile_id",
            "scenario_id",
            "coarse_dem",
            "valid_mask",
            "transform",
            "crs_wkt",
            "candidates",
            "candidate_ids",
            "candidate_track_ids",
            "targets",
            "target_weights",
            "feature_grid",
            "feature_names",
            "metadata",
        }
        missing = sorted(required.difference(names))
        if missing:
            raise ValueError(f"training problem lacks required fields: {missing}")
        if str(archive["tile_id"].item()) != product_id:
            raise ValueError("training problem tile differs from the fixed product")
        coarse = np.asarray(archive["coarse_dem"])
        valid = np.asarray(archive["valid_mask"])
        if coarse.shape != (512, 512) or valid.shape != coarse.shape or valid.dtype != np.bool_:
            raise ValueError("training problem must contain aligned 512x512 DEM/mask arrays")
        if not np.all(np.isfinite(coarse[valid])):
            raise ValueError("training problem coarse DEM is nonfinite over valid pixels")
        metadata = json.loads(str(archive["metadata"].item()))
        required_false = (
            "reference_terrain_embedded",
            "existing_laser_geometry_accessed",
            "selection_reference_accessed",
            "crop_uses_final_offsets",
        )
        if metadata.get("preparation_protocol") != "formal-no-existing-laser-v2":
            raise ValueError("training problem was not made by the frozen Paper1 protocol")
        if any(metadata.get(name) is not False for name in required_false):
            raise ValueError("training problem does not prove truth/existing-laser isolation")
        if int(metadata.get("existing_laser_count", -1)) != 0:
            raise ValueError("training problem conditions on existing laser observations")
        feature_names = tuple(str(value) for value in archive["feature_names"])
        image_feature_indices = [
            index
            for index, value in enumerate(feature_names)
            if "ortho" in value.lower() or "image" in value.lower()
        ]
        if image_feature_indices:
            if metadata.get("image_feature_policy") != (
                "paper1-prior-center-imputation-no-image-access"
            ):
                raise ValueError("image-named problem features lack a no-image imputation proof")
            feature_grid = np.asarray(archive["feature_grid"], dtype=np.float64)
            for index in image_feature_indices:
                values = feature_grid[..., index][valid]
                if values.size == 0 or not np.all(values == values[0]):
                    raise ValueError("image-named features contain spatial image/coreg_aux values")
            if metadata.get("image_feature_source_accessed") is not False:
                raise ValueError("problem reports source-image access during feature imputation")
            if metadata.get("coreg_aux_feature_values_accessed") is not False:
                raise ValueError("problem reports coreg_aux feature access")
        candidates = np.asarray(archive["candidates"], dtype=np.float64)
        candidate_ids = np.asarray(archive["candidate_ids"]).astype(str)
        track_ids = np.asarray(archive["candidate_track_ids"]).astype(str)
        if (
            candidates.ndim != 2
            or candidates.shape[1] != 2
            or candidate_ids.shape != (candidates.shape[0],)
            or track_ids.shape != candidate_ids.shape
            or not np.all(np.isfinite(candidates))
        ):
            raise ValueError("candidate geometry arrays are invalid or misaligned")
        geometry = [
            {
                "candidate_id": candidate_ids[index],
                "track_id": track_ids[index],
                "x_hex": float(candidates[index, 0]).hex(),
                "y_hex": float(candidates[index, 1]).hex(),
            }
            for index in range(candidates.shape[0])
        ]
        transform = tuple(float(value) for value in archive["transform"])
        return {
            "shape": [512, 512],
            "transform_hex": [value.hex() for value in transform],
            "crs_wkt_sha256": canonical_sha256(str(archive["crs_wkt"].item())),
            "candidate_count": int(candidates.shape[0]),
            "candidate_geometry_sha256": canonical_sha256(geometry),
            "crop_window": list(metadata["crop_window"]),
            "feature_names": list(feature_names),
            "coreg_aux_influenced_primary_pixels": False,
        }


def _verify_reference(path: Path, *, product_id: str, problem_contract: Mapping[str, Any]) -> None:
    with np.load(path, allow_pickle=False) as archive:
        required = {"tile_id", "dem", "valid_mask", "transform", "crs_wkt"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"evaluator reference lacks fields: {missing}")
        if str(archive["tile_id"].item()) != product_id:
            raise ValueError("evaluator reference tile differs from fixed product")
        dem = np.asarray(archive["dem"])
        mask = np.asarray(archive["valid_mask"])
        if list(dem.shape) != problem_contract["shape"] or mask.shape != dem.shape:
            raise ValueError("evaluator reference does not align with the problem")
        transform = [float(value).hex() for value in archive["transform"]]
        if transform != problem_contract["transform_hex"]:
            raise ValueError("evaluator reference transform differs from the problem")
        if canonical_sha256(str(archive["crs_wkt"].item())) != problem_contract[
            "crs_wkt_sha256"
        ]:
            raise ValueError("evaluator reference CRS differs from the problem")


def _sanitize_selection(
    raw_path: Path,
    output: Path,
    *,
    product_id: str,
    paper2_root: Path,
) -> Path:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if raw.get("tile_id") != product_id or raw.get("method") != KMAX_METHOD:
        raise ValueError("Kmax selection identity/method differs")
    if int(raw.get("budget", -1)) != KMAX_BUDGET:
        raise ValueError("Kmax selection budget must be 100")
    records = raw.get("records")
    if not isinstance(records, list) or len(records) != KMAX_BUDGET:
        raise ValueError("Kmax selection records do not close")
    allowlisted = []
    for expected_rank, source in enumerate(records, start=1):
        record = {
            "rank": int(source["rank"]),
            "candidate_id": str(source["candidate_id"]),
            "track_id": str(source["track_id"]),
            "projected_x": float(source["projected_x"]),
            "projected_y": float(source["projected_y"]),
            "constraints_satisfied": bool(source["constraints_satisfied"]),
        }
        if record["rank"] != expected_rank or not record["constraints_satisfied"]:
            raise ValueError("Kmax selection ranks/constraints are invalid")
        allowlisted.append(record)
    payload = {
        "tile_id": product_id,
        "method": KMAX_METHOD,
        "budget": KMAX_BUDGET,
        "records": allowlisted,
        "metadata": {
            "selection_reference_accessed": False,
            "conditions_on_existing_laser": False,
            "existing_laser_count": 0,
            "selection_source": "Paper1 discrete-FAOPT Kmax allowlist",
        },
    }
    serialized = json.dumps(payload, sort_keys=True).lower()
    if any(marker in serialized for marker in _FORBIDDEN_PLAN_MARKERS):
        raise ValueError("sanitized Kmax selection retains a forbidden locator/value shortcut")
    return write_canonical_json(output, payload, paper2_root=paper2_root)


def _verify_plan_family(
    paths: Sequence[Path],
    budgets: Sequence[int],
    *,
    paper2_root: Path,
) -> dict[str, Any]:
    if len(paths) != len(budgets):
        raise ValueError("plan-family file count differs from budgets")
    plans = [PlanContract.read(path) for path in paths]
    kmax = plans[-1]
    for plan, budget in zip(plans, budgets, strict=True):
        if plan.budget != budget or plan.points != kmax.points[:budget]:
            raise ValueError(f"K={budget} plan is not a strict Kmax prefix")
        serialized = json.dumps(plan.to_dict(), sort_keys=True).lower()
        if any(marker in serialized for marker in _FORBIDDEN_PLAN_MARKERS):
            raise ValueError(f"K={budget} plan contains a forbidden field/locator")
    return {
        str(budget): {
            "relative_path": path.relative_to(paper2_root).as_posix(),
            "sha256": file_sha256(path),
            "plan_semantic_sha256": plan.sha256,
        }
        for budget, path, plan in zip(budgets, paths, plans, strict=True)
    }


def verify_source_preparation_manifest(
    path: str | Path,
    *,
    paper2_root: str | Path = PAPER2_ROOT,
) -> dict[str, Any]:
    root = Path(paper2_root).resolve()
    manifest_path = guard_paper2_path(path, paper2_root=root)
    payload = read_json(manifest_path)
    if payload.get("schema") != SOURCE_PREPARATION_SCHEMA or payload.get("state") != "complete":
        raise ValueError("source preparation manifest is not complete/frozen")
    declared = payload.get("content_sha256")
    unhashed = dict(payload)
    unhashed.pop("content_sha256", None)
    if declared != canonical_sha256(unhashed):
        raise ValueError("source preparation manifest content hash differs")
    if payload.get("coreg_aux_influenced_primary_pixels") is not False:
        raise ValueError("source preparation allows coreg_aux to alter primary image pixels")
    artifacts = dict(payload.get("artifacts", {}))
    required = {
        "training_problem",
        "evaluator_reference",
        "coreg_aux",
        "paper1_prepared_manifest",
        "sanitized_kmax_selection",
        "source_preparation_audit",
    }
    if set(artifacts) != required:
        raise ValueError("source preparation artifact set differs")
    for name, raw in artifacts.items():
        record = dict(raw)
        resolved = guard_paper2_path(str(record.get("relative_path", "")), paper2_root=root)
        if (
            not resolved.is_file()
            or record.get("sha256") != file_sha256(resolved)
            or int(record.get("byte_count", -1)) != resolved.stat().st_size
        ):
            raise ValueError(f"source preparation artifact differs: {name}")
    problem = guard_paper2_path(
        artifacts["training_problem"]["relative_path"], paper2_root=root
    )
    observed_contract = _problem_contract(problem, product_id=str(payload["product_id"]))
    if observed_contract != payload.get("problem_contract"):
        raise ValueError("source preparation problem contract differs")
    reference = guard_paper2_path(
        artifacts["evaluator_reference"]["relative_path"], paper2_root=root
    )
    _verify_reference(reference, product_id=str(payload["product_id"]), problem_contract=observed_contract)
    plan_records = dict(payload.get("plans", {}))
    if set(plan_records) != {str(value) for value in DEFAULT_BUDGETS}:
        raise ValueError("source preparation plan family does not close")
    plan_paths = [
        guard_paper2_path(plan_records[str(budget)]["relative_path"], paper2_root=root)
        for budget in DEFAULT_BUDGETS
    ]
    observed_plans = _verify_plan_family(
        plan_paths,
        DEFAULT_BUDGETS,
        paper2_root=root,
    )
    if observed_plans != plan_records:
        raise ValueError("source preparation plan hashes differ")
    return payload


def prepare_source_product(
    *,
    row: Mapping[str, Any],
    assignment: Mapping[str, Any],
    product_order: int,
    prior_path: str | Path,
    paper1_lock: Paper1SourceLock,
    paper1_source_root: str | Path,
    output_directory: str | Path,
    schedule_bindings: Mapping[str, Any],
    options: SourcePreparationOptions = SourcePreparationOptions(),
    paper2_root: str | Path = PAPER2_ROOT,
    source_builder: SourceBuilder | None = None,
    http_metadata_reader: HTTPMetadataReader = _http_metadata,
) -> dict[str, Any]:
    root = Path(paper2_root).resolve()
    options.validate()
    paper1_lock.validate()
    product_id = _safe_identifier(str(row.get("tile_id", "")), name="product_id")
    if row.get("item_id") != product_id:
        raise ValueError("source row item/tile identity differs")
    fixed_products = [
        str(assignment["representative_product_id"]),
        *(str(value) for value in assignment.get("fallback_product_ids", [])),
    ]
    if product_order < 0 or product_order >= len(fixed_products):
        raise ValueError("product_order is outside the fixed representative/fallback list")
    if fixed_products[product_order] != product_id:
        raise ValueError("source product differs from fixed representative/fallback order")
    prior = guard_paper2_path(prior_path, paper2_root=root)
    if not prior.is_file() or file_sha256(prior) != paper1_lock.prior_sha256:
        raise ValueError("Paper1 prior differs from the frozen imported prior")
    output = guard_paper2_path(output_directory, paper2_root=root)
    manifest_path = output / "source-prepared-manifest.json"
    code_lock = build_paper1_code_lock(paper1_source_root)
    if manifest_path.is_file():
        payload = verify_source_preparation_manifest(manifest_path, paper2_root=root)
        expected_binding = canonical_sha256(
            {
                "schedule": dict(schedule_bindings),
                "assignment": dict(assignment),
                "product_order": product_order,
                "catalog_row_sha256": canonical_sha256(dict(row)),
                "paper1": paper1_lock.to_dict(),
                "options": asdict(options),
            }
        )
        if payload.get("input_bindings_sha256") != expected_binding:
            raise ValueError("existing source preparation uses stale inputs")
        if payload.get("paper1_code_lock_content_sha256") != code_lock["content_sha256"]:
            raise ValueError("existing source preparation uses stale Paper1 source code")
        return payload

    sanitized = _sanitize_catalog_row(row)
    input_payload = {
        "schedule": dict(schedule_bindings),
        "assignment": dict(assignment),
        "product_order": product_order,
        "catalog_row_sha256": canonical_sha256(dict(row)),
        "paper1": paper1_lock.to_dict(),
        "options": asdict(options),
    }
    input_bindings_sha256 = canonical_sha256(input_payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = guard_paper2_path(
        "data/source-preparation/.staging", paper2_root=root
    )
    staging_parent.mkdir(parents=True, exist_ok=True)
    builder = source_builder or _default_source_builder_factory(
        paper1_source_root=Path(paper1_source_root),
        http_metadata_reader=http_metadata_reader,
        paper2_root=root,
    )
    with tempfile.TemporaryDirectory(prefix=f"{product_id}-", dir=staging_parent) as name:
        staging = Path(name)
        built = builder(row, sanitized, staging, prior, options)
        for path in (
            built.problem,
            built.reference,
            built.paper1_prepared_manifest,
            built.coreg_aux,
            built.raw_selection,
        ):
            resolved = Path(path).resolve()
            if staging != resolved and staging not in resolved.parents:
                raise ValueError("source builder returned an artifact outside its Paper2 staging area")
            if not resolved.is_file() or resolved.stat().st_size <= 0:
                raise FileNotFoundError(resolved)
        problem_contract = _problem_contract(built.problem, product_id=product_id)
        _verify_reference(
            built.reference,
            product_id=product_id,
            problem_contract=problem_contract,
        )

        artifacts_directory = output / "artifacts"
        artifacts_directory.mkdir(parents=True, exist_ok=True)
        final_paths = {
            "training_problem": artifacts_directory / "problem-v2.npz",
            "evaluator_reference": artifacts_directory / "reference.npz",
            "coreg_aux": artifacts_directory / "coreg-aux.tif",
            "paper1_prepared_manifest": artifacts_directory / "paper1-prepared-manifest.json",
            "sanitized_kmax_selection": artifacts_directory / "selection-k100-sanitized.json",
        }
        write_immutable_bytes(
            final_paths["training_problem"], built.problem.read_bytes(), paper2_root=root
        )
        write_immutable_bytes(
            final_paths["evaluator_reference"], built.reference.read_bytes(), paper2_root=root
        )
        write_immutable_bytes(final_paths["coreg_aux"], built.coreg_aux.read_bytes(), paper2_root=root)
        prepared_payload = read_json(built.paper1_prepared_manifest)
        write_canonical_json(
            final_paths["paper1_prepared_manifest"], prepared_payload, paper2_root=root
        )
        _sanitize_selection(
            built.raw_selection,
            final_paths["sanitized_kmax_selection"],
            product_id=product_id,
            paper2_root=root,
        )

    plan = import_paper1_kmax_plan(
        final_paths["sanitized_kmax_selection"],
        final_paths["training_problem"],
        paper1_lock=paper1_lock,
        budgets=options.budgets,
    )
    plan_paths = write_plan_family(
        plan,
        output / "plans",
        budgets=options.budgets,
        paper2_root=root,
    )
    plan_records = _verify_plan_family(
        plan_paths,
        options.budgets,
        paper2_root=root,
    )

    source_assets = dict(row.get("assets", {}))
    dem_key = next(
        name for name in ("geoid_adjusted_dem", "dtm", "dem") if name in source_assets
    )
    source_http = {name: dict(value) for name, value in built.source_http.items()}
    audit_payload: dict[str, Any] = {
        "schema": SOURCE_PREPARATION_AUDIT_SCHEMA,
        "state": "complete",
        "product_id": product_id,
        "protocol": SOURCE_PROTOCOL,
        "catalog_row_sha256": canonical_sha256(dict(row)),
        "sanitized_manifest_sha256": canonical_sha256(sanitized),
        "passed_to_paper1_asset_names": sorted(sanitized["assets"]),
        "source_assets": {
            "controlled_dtm": {
                "url": str(source_assets[dem_key]),
                "http": source_http.get("controlled_dtm"),
                "access": "remote_cog_deterministic_window",
                "crop_window": problem_contract["crop_window"],
                "derived_reference_sha256": file_sha256(
                    final_paths["evaluator_reference"]
                ),
            },
            "coreg_aux_ortho": {
                "url": str(source_assets.get("orthoimage", "")),
                "http": source_http.get("coreg_aux_ortho"),
                "access": "remote_raster_reproject_to_frozen_problem_grid",
                "derived_coreg_aux_sha256": file_sha256(final_paths["coreg_aux"]),
                "coreg_aux_influenced_primary_pixels": False,
            },
            "final_offsets": {
                "declared_in_catalog": "final_offsets" in source_assets,
                "url_passed_to_paper1": False,
                "fetched": False,
                "parsed": False,
                "coordinates_accessed": False,
            },
            "mola_shots": {
                "declared_in_catalog": "mola_csv" in source_assets,
                "url_passed_to_paper1": False,
                "fetched": False,
                "parsed": False,
                "coordinates_accessed": False,
            },
        },
        "isolation": {
            "existing_laser_geometry_accessed": False,
            "crop_uses_final_offsets": False,
            "selection_reference_accessed": False,
            "reference_path_exported_to_training_problem": False,
            "coreg_aux_is_primary": False,
            "coreg_aux_influenced_primary_pixels": False,
            "ortho_pixel_values_exported_to_training_problem": False,
            "missing_image_features_may_be_imputed_at_frozen_prior_center": True,
        },
        "problem_contract": problem_contract,
        "paper1_code_lock": code_lock,
        "input_bindings_sha256": input_bindings_sha256,
    }
    audit_payload["content_sha256"] = canonical_sha256(audit_payload)
    audit_path = write_canonical_json(
        output / "source-preparation-audit.json", audit_payload, paper2_root=root
    )
    artifacts = {
        name: _artifact_record(
            path,
            paper2_root=root,
            access_scope="evaluator_only" if name == "evaluator_reference" else "paper2_pipeline",
        )
        for name, path in final_paths.items()
    }
    artifacts["source_preparation_audit"] = _artifact_record(
        audit_path, paper2_root=root, access_scope="audit_only"
    )
    payload: dict[str, Any] = {
        "schema": SOURCE_PREPARATION_SCHEMA,
        "state": "complete",
        "product_id": product_id,
        "geographic_group_id": str(assignment["group_id"]),
        "role": str(assignment["role"]),
        "product_order": product_order,
        "fixed_product_ids": fixed_products,
        "protocol": SOURCE_PROTOCOL,
        "input_bindings_sha256": input_bindings_sha256,
        "paper1": paper1_lock.to_dict(),
        "paper1_code_lock_content_sha256": code_lock["content_sha256"],
        "problem_contract": problem_contract,
        "coreg_aux_is_primary": False,
        "coreg_aux_influenced_primary_pixels": False,
        "reference_access_scope": "independent_simulator_and_evaluator_only",
        "artifacts": artifacts,
        "plans": plan_records,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    write_canonical_json(manifest_path, payload, paper2_root=root)
    return verify_source_preparation_manifest(manifest_path, paper2_root=root)


def _g0_group_attempts(
    processing_index: Mapping[str, Any] | None,
) -> dict[str, list[dict[str, Any]]]:
    if processing_index is None:
        return {}
    groups: dict[str, list[dict[str, Any]]] = {}
    for raw in processing_index.get("groups", []):
        group_id = str(raw.get("group_id", ""))
        if not group_id or group_id in groups:
            raise ValueError("G0 processing groups must have unique nonempty IDs")
        groups[group_id] = [dict(value) for value in raw.get("attempts", [])]
    return groups


def next_fixed_source_product(
    assignment: Mapping[str, Any],
    *,
    g0_attempts: Sequence[Mapping[str, Any]] = (),
    completed_products: set[str] | frozenset[str] = frozenset(),
    rejected_products: set[str] | frozenset[str] = frozenset(),
) -> tuple[int, str] | None:
    products = [
        str(assignment["representative_product_id"]),
        *(str(value) for value in assignment.get("fallback_product_ids", [])),
    ]
    statuses = {str(raw.get("product_id", "")): str(raw.get("status", "")) for raw in g0_attempts}
    for order, product_id in enumerate(products):
        if product_id in rejected_products:
            continue
        status = statuses.get(product_id)
        if status in _TERMINAL_G0_PRODUCT_STATUSES:
            continue
        if product_id in completed_products:
            return None
        return order, product_id
    return None


def source_preparation_contract_sha256(payload: Mapping[str, Any]) -> str:
    """Return the stable contract identity while recoverable group states evolve."""
    required = {
        "conflict_groups_sha256",
        "conflict_groups_file_sha256",
        "assignments_file_sha256",
        "role_assignment_sha256",
        "catalog_file_sha256",
        "source_protocol",
        "paper1",
        "paper1_code_lock_content_sha256",
        "options_sha256",
        "expected_group_count",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"source preparation index lacks contract fields: {missing}")
    return canonical_sha256({key: payload[key] for key in sorted(required)})


def validate_source_preparation_index(
    payload_or_path: Mapping[str, Any] | str | Path,
    *,
    assignments: Sequence[Mapping[str, Any]],
    schedule_bindings: Mapping[str, Any],
    paper2_root: str | Path = PAPER2_ROOT,
) -> dict[str, dict[str, Any]]:
    root = Path(paper2_root).resolve()
    payload = (
        read_json(payload_or_path)
        if isinstance(payload_or_path, (str, Path))
        else dict(payload_or_path)
    )
    if payload.get("schema") != SOURCE_PREPARATION_INDEX_SCHEMA:
        raise ValueError("source preparation index uses an unsupported schema")
    declared = payload.get("content_sha256")
    unhashed = dict(payload)
    unhashed.pop("content_sha256", None)
    if declared != canonical_sha256(unhashed):
        raise ValueError("source preparation index content hash differs")
    for key, value in schedule_bindings.items():
        if payload.get(key) != value:
            raise ValueError(f"source preparation index has stale {key}")
    if payload.get("contract_sha256") != source_preparation_contract_sha256(payload):
        raise ValueError("source preparation index contract hash differs")
    assignment_by_group = {str(value["group_id"]): dict(value) for value in assignments}
    if int(payload.get("expected_group_count", -1)) != len(assignment_by_group):
        raise ValueError("source preparation expected group count differs")
    groups: dict[str, dict[str, Any]] = {}
    for raw in payload.get("groups", []):
        group = dict(raw)
        group_id = str(group.get("group_id", ""))
        if group_id not in assignment_by_group or group_id in groups:
            raise ValueError(f"source preparation group identity differs: {group_id}")
        assignment = assignment_by_group[group_id]
        products = [
            str(assignment["representative_product_id"]),
            *(str(value) for value in assignment.get("fallback_product_ids", [])),
        ]
        if group.get("role") != assignment["role"] or group.get("fixed_product_ids") != products:
            raise ValueError(f"source preparation fixed role/order differs: {group_id}")
        attempts: dict[str, dict[str, Any]] = {}
        for attempt_raw in group.get("attempts", []):
            attempt = dict(attempt_raw)
            product_id = str(attempt.get("product_id", ""))
            if product_id not in products or product_id in attempts:
                raise ValueError(f"source preparation attempts are duplicated/out of plan: {group_id}")
            if int(attempt.get("product_order", -1)) != products.index(product_id):
                raise ValueError(f"source preparation attempt order differs: {product_id}")
            status = str(attempt.get("status", ""))
            if status == "complete":
                manifest = guard_paper2_path(
                    str(attempt.get("manifest_relative_path", "")), paper2_root=root
                )
                if not manifest.is_file() or attempt.get("manifest_sha256") != file_sha256(
                    manifest
                ):
                    raise ValueError(f"source preparation manifest bytes differ: {product_id}")
                prepared = verify_source_preparation_manifest(manifest, paper2_root=root)
                if (
                    prepared.get("product_id") != product_id
                    or prepared.get("geographic_group_id") != group_id
                    or prepared.get("role") != group["role"]
                    or prepared.get("product_order") != products.index(product_id)
                    or attempt.get("manifest_content_sha256") != prepared["content_sha256"]
                ):
                    raise ValueError(f"source preparation manifest identity differs: {product_id}")
            elif status == "source_rejected":
                failure = guard_paper2_path(
                    str(attempt.get("failure_relative_path", "")), paper2_root=root
                )
                if not failure.is_file() or attempt.get("failure_sha256") != file_sha256(failure):
                    raise ValueError(f"source preparation failure bytes differ: {product_id}")
                failed = read_json(failure)
                failed_unhashed = dict(failed)
                failed_declared = failed_unhashed.pop("content_sha256", None)
                if (
                    failed.get("schema") != SOURCE_PREPARATION_FAILURE_SCHEMA
                    or failed.get("product_id") != product_id
                    or failed_declared != canonical_sha256(failed_unhashed)
                ):
                    raise ValueError(f"source preparation failure receipt differs: {product_id}")
            else:
                raise ValueError(f"unsupported source preparation status: {status}")
            attempts[product_id] = attempt
        selected = group.get("source_ready_product_id")
        if selected is not None and (
            selected not in attempts or attempts[selected].get("status") != "complete"
        ):
            raise ValueError(f"source-ready product lacks a complete manifest: {group_id}")
        if group.get("state") == "source_ready" and selected is None:
            raise ValueError(f"source-ready group lacks selected source: {group_id}")
        group["attempts_by_product"] = attempts
        groups[group_id] = group
    if set(groups) != set(assignment_by_group):
        raise ValueError("source preparation group set differs from fixed assignments")
    ready = sum(group.get("state") == "source_ready" for group in groups.values())
    if int(payload.get("source_ready_group_count", -1)) != ready:
        raise ValueError("source preparation ready-group count differs")
    return groups


def source_exhausted_group_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Return groups whose complete fixed product order was rejected by source QA.

    This check deliberately distinguishes a terminal, pre-outcome source exhaustion from an
    ordinary incomplete frontier.  Only the former authorizes a same-stratum group replacement;
    a network error or an unattempted fixed fallback must never change the locked sample.
    """
    exhausted: list[str] = []
    for raw in payload.get("groups", []):
        group = dict(raw)
        if group.get("state") == "source_ready":
            continue
        fixed = [str(value) for value in group.get("fixed_product_ids", [])]
        attempts = {
            str(attempt.get("product_id", "")): str(attempt.get("status", ""))
            for attempt in group.get("attempts", [])
        }
        if fixed and set(attempts) == set(fixed) and all(
            attempts[product_id] == "source_rejected" for product_id in fixed
        ):
            exhausted.append(str(group.get("group_id", "")))
    if any(not value for value in exhausted) or len(exhausted) != len(set(exhausted)):
        raise ValueError("source-exhausted groups must have unique nonempty identities")
    return tuple(sorted(exhausted))


def clone_source_preparations_for_assignment(
    *,
    old_assignments: Sequence[Mapping[str, Any]],
    new_assignments: Sequence[Mapping[str, Any]],
    old_schedule_bindings: Mapping[str, Any],
    new_schedule_bindings: Mapping[str, Any],
    catalog_rows: Mapping[str, Mapping[str, Any]],
    paper1_lock: Paper1SourceLock,
    source_products_directory: str | Path,
    destination_products_directory: str | Path,
    options: SourcePreparationOptions = SourcePreparationOptions(),
    paper2_root: str | Path = PAPER2_ROOT,
) -> dict[str, Any]:
    """Clone unchanged source preparations into a byte-new assignment namespace.

    The historical manifests and failure receipts remain untouched.  Large immutable artifacts
    are hard-linked, while assignment-bound JSON receipts are copied and rebound with an explicit
    revision record.  Changed/replacement groups are intentionally not cloned and must be prepared
    from their own official sources.
    """
    root = Path(paper2_root).resolve()
    source_root = guard_paper2_path(source_products_directory, paper2_root=root)
    destination_root = guard_paper2_path(destination_products_directory, paper2_root=root)
    if source_root == destination_root:
        raise ValueError("source-preparation clone requires a distinct destination namespace")
    old_by_group = {str(value["group_id"]): dict(value) for value in old_assignments}
    new_by_group = {str(value["group_id"]): dict(value) for value in new_assignments}
    if len(old_by_group) != len(old_assignments) or len(new_by_group) != len(new_assignments):
        raise ValueError("source-preparation clone assignments must be unique")
    unchanged = sorted(
        group_id
        for group_id, assignment in new_by_group.items()
        if old_by_group.get(group_id) == assignment
    )
    options.validate()
    paper1_lock.validate()
    cloned_manifests = 0
    cloned_failures = 0
    reused_products = 0
    destination_root.mkdir(parents=True, exist_ok=True)

    def binding(
        *, assignment: Mapping[str, Any], product_order: int, row: Mapping[str, Any],
        schedule: Mapping[str, Any], failure: bool,
    ) -> str:
        value: dict[str, Any] = {
            "schedule": dict(schedule),
            "assignment": dict(assignment),
            "product_order": product_order,
            "catalog_row_sha256": canonical_sha256(dict(row)),
        }
        if not failure:
            value.update({"paper1": paper1_lock.to_dict(), "options": asdict(options)})
        return canonical_sha256(value)

    for group_id in unchanged:
        assignment = new_by_group[group_id]
        products = [
            str(assignment["representative_product_id"]),
            *(str(value) for value in assignment.get("fallback_product_ids", [])),
        ]
        for product_order, product_id in enumerate(products):
            row = catalog_rows.get(product_id)
            if row is None:
                raise ValueError(f"clone catalog lacks fixed product: {product_id}")
            source = source_root / _safe_identifier(product_id, name="product_id")
            destination = destination_root / product_id
            source_manifest = source / "source-prepared-manifest.json"
            source_failure = source / "source-preparation-failure.json"
            if not source_manifest.is_file() and not source_failure.is_file():
                continue
            if source_manifest.is_file() and source_failure.is_file():
                raise ValueError(f"source product has both complete and failed receipts: {product_id}")
            old_input = binding(
                assignment=assignment, product_order=product_order, row=row,
                schedule=old_schedule_bindings, failure=source_failure.is_file(),
            )
            new_input = binding(
                assignment=assignment, product_order=product_order, row=row,
                schedule=new_schedule_bindings, failure=source_failure.is_file(),
            )
            if destination.exists():
                candidate = destination / (
                    "source-prepared-manifest.json"
                    if source_manifest.is_file()
                    else "source-preparation-failure.json"
                )
                payload = (
                    verify_source_preparation_manifest(candidate, paper2_root=root)
                    if source_manifest.is_file()
                    else read_json(candidate)
                )
                if payload.get("input_bindings_sha256") != new_input:
                    raise ValueError(f"existing cloned source has stale bindings: {product_id}")
                reused_products += 1
                continue
            staging_parent = destination_root / ".staging"
            staging_parent.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=f"{product_id}-", dir=staging_parent) as name:
                staging = Path(name) / product_id

                def copy_or_link(src: str, dst: str) -> str:
                    if Path(src).name in {
                        "source-prepared-manifest.json",
                        "source-preparation-audit.json",
                        "source-preparation-failure.json",
                    }:
                        return shutil.copy2(src, dst)
                    os.link(src, dst)
                    return dst

                shutil.copytree(source, staging, copy_function=copy_or_link)
                revision = {
                    "protocol": "assignment-namespace-clone-v1",
                    "old_input_bindings_sha256": old_input,
                    "new_input_bindings_sha256": new_input,
                    "old_assignments_file_sha256": old_schedule_bindings.get(
                        "assignments_file_sha256"
                    ),
                    "new_assignments_file_sha256": new_schedule_bindings.get(
                        "assignments_file_sha256"
                    ),
                    "artifacts_recomputed": False,
                    "model_outcomes_accessed": False,
                    "reference_values_accessed": False,
                }
                if source_failure.is_file():
                    failed = read_json(source_failure)
                    failed_unhashed = dict(failed)
                    failed_declared = failed_unhashed.pop("content_sha256", None)
                    if (
                        failed.get("schema") != SOURCE_PREPARATION_FAILURE_SCHEMA
                        or failed_declared != canonical_sha256(failed_unhashed)
                        or failed.get("input_bindings_sha256") != old_input
                    ):
                        raise ValueError(f"historical source failure cannot be rebound: {product_id}")
                    failed["input_bindings_sha256"] = new_input
                    failed["binding_revisions"] = [
                        *(dict(value) for value in failed.get("binding_revisions", [])),
                        revision,
                    ]
                    failed["content_sha256"] = canonical_sha256(
                        {key: value for key, value in failed.items() if key != "content_sha256"}
                    )
                    _recoverable_json(
                        staging / "source-preparation-failure.json", failed, paper2_root=root
                    )
                    cloned_failures += 1
                else:
                    prepared = verify_source_preparation_manifest(source_manifest, paper2_root=root)
                    if prepared.get("input_bindings_sha256") != old_input:
                        raise ValueError(f"historical source manifest cannot be rebound: {product_id}")
                    audit_record = dict(prepared["artifacts"]["source_preparation_audit"])
                    source_audit = guard_paper2_path(
                        str(audit_record["relative_path"]), paper2_root=root
                    )
                    audit = read_json(source_audit)
                    audit_unhashed = dict(audit)
                    audit_declared = audit_unhashed.pop("content_sha256", None)
                    if (
                        audit_declared != canonical_sha256(audit_unhashed)
                        or audit.get("input_bindings_sha256") != old_input
                    ):
                        raise ValueError(f"historical source audit cannot be rebound: {product_id}")
                    audit["input_bindings_sha256"] = new_input
                    audit["binding_revisions"] = [
                        *(dict(value) for value in audit.get("binding_revisions", [])), revision
                    ]
                    audit["content_sha256"] = canonical_sha256(
                        {key: value for key, value in audit.items() if key != "content_sha256"}
                    )
                    staging_audit = staging / "source-preparation-audit.json"
                    _recoverable_json(staging_audit, audit, paper2_root=root)
                    old_prefix = source.relative_to(root).as_posix() + "/"
                    new_prefix = destination.relative_to(root).as_posix() + "/"
                    for record in prepared["artifacts"].values():
                        relative = str(record["relative_path"])
                        if not relative.startswith(old_prefix):
                            raise ValueError(f"source artifact path escapes product: {product_id}")
                        record["relative_path"] = new_prefix + relative.removeprefix(old_prefix)
                    for record in prepared["plans"].values():
                        relative = str(record["relative_path"])
                        if not relative.startswith(old_prefix):
                            raise ValueError(f"source plan path escapes product: {product_id}")
                        record["relative_path"] = new_prefix + relative.removeprefix(old_prefix)
                    prepared["artifacts"]["source_preparation_audit"] = {
                        "relative_path": (
                            destination / "source-preparation-audit.json"
                        ).relative_to(root).as_posix(),
                        "sha256": file_sha256(staging_audit),
                        "byte_count": staging_audit.stat().st_size,
                        "access_scope": "audit_only",
                    }
                    prepared["input_bindings_sha256"] = new_input
                    prepared["binding_revisions"] = [
                        *(dict(value) for value in prepared.get("binding_revisions", [])), revision
                    ]
                    prepared["content_sha256"] = canonical_sha256(
                        {key: value for key, value in prepared.items() if key != "content_sha256"}
                    )
                    _recoverable_json(
                        staging / "source-prepared-manifest.json", prepared, paper2_root=root
                    )
                    cloned_manifests += 1
                staging.replace(destination)
            if source_manifest.is_file():
                verified = verify_source_preparation_manifest(
                    destination / "source-prepared-manifest.json", paper2_root=root
                )
            else:
                verified = read_json(destination / "source-preparation-failure.json")
            if verified.get("input_bindings_sha256") != new_input:
                raise AssertionError(f"cloned source binding did not close: {product_id}")
    return {
        "schema": "paper2.g0.source-preparation-clone/v1",
        "state": "complete",
        "unchanged_group_count": len(unchanged),
        "cloned_manifest_count": cloned_manifests,
        "cloned_failure_count": cloned_failures,
        "reused_product_count": reused_products,
        "old_schedule_sha256": canonical_sha256(dict(old_schedule_bindings)),
        "new_schedule_sha256": canonical_sha256(dict(new_schedule_bindings)),
        "model_outcomes_accessed": False,
        "reference_values_accessed": False,
    }


def prepare_source_frontier(
    *,
    conflict_groups_path: str | Path,
    assignments_path: str | Path,
    catalog_path: str | Path,
    prior_path: str | Path,
    paper1_lock: Paper1SourceLock,
    paper1_source_root: str | Path,
    output_index: str | Path,
    products_directory: str | Path = "data/source-preparation/products",
    g0_processing_index: str | Path | None = None,
    options: SourcePreparationOptions = SourcePreparationOptions(),
    paper2_root: str | Path = PAPER2_ROOT,
    require_formal_counts: bool = True,
    source_builder: SourceBuilder | None = None,
) -> dict[str, Any]:
    root = Path(paper2_root).resolve()
    assignments, rows, schedule_bindings = load_fixed_source_schedule(
        conflict_groups_path=conflict_groups_path,
        assignments_path=assignments_path,
        catalog_path=catalog_path,
        require_formal_counts=require_formal_counts,
    )
    processing_payload = None
    if g0_processing_index is not None:
        processing_path = guard_paper2_path(g0_processing_index, paper2_root=root)
        if processing_path.is_file():
            processing_payload = read_json(processing_path)
    g0_groups = _g0_group_attempts(processing_payload)
    products_root = guard_paper2_path(products_directory, paper2_root=root)
    products_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for assignment in assignments:
        products = [
            str(assignment["representative_product_id"]),
            *(str(value) for value in assignment.get("fallback_product_ids", [])),
        ]
        completed: set[str] = set()
        rejected: set[str] = set()
        attempts: list[dict[str, Any]] = []
        for order, product_id in enumerate(products):
            directory = products_root / _safe_identifier(product_id, name="product_id")
            manifest = directory / "source-prepared-manifest.json"
            failure = directory / "source-preparation-failure.json"
            if manifest.is_file():
                verified = prepare_source_product(
                    row=rows[product_id],
                    assignment=assignment,
                    product_order=order,
                    prior_path=prior_path,
                    paper1_lock=paper1_lock,
                    paper1_source_root=paper1_source_root,
                    output_directory=directory,
                    schedule_bindings=schedule_bindings,
                    options=options,
                    paper2_root=root,
                    source_builder=source_builder,
                )
                completed.add(product_id)
                attempts.append(
                    {
                        "product_id": product_id,
                        "product_order": order,
                        "status": "complete",
                        "manifest_relative_path": manifest.relative_to(root).as_posix(),
                        "manifest_sha256": file_sha256(manifest),
                        "manifest_content_sha256": verified["content_sha256"],
                    }
                )
            elif failure.is_file():
                failed_payload = read_json(failure)
                failed_unhashed = dict(failed_payload)
                failed_declared = failed_unhashed.pop("content_sha256", None)
                expected_failure_binding = canonical_sha256(
                    {
                        "schedule": schedule_bindings,
                        "assignment": assignment,
                        "product_order": order,
                        "catalog_row_sha256": canonical_sha256(rows[product_id]),
                    }
                )
                if (
                    failed_payload.get("schema") != SOURCE_PREPARATION_FAILURE_SCHEMA
                    or failed_declared != canonical_sha256(failed_unhashed)
                    or failed_payload.get("input_bindings_sha256")
                    != expected_failure_binding
                ):
                    raise ValueError(f"stale source-preparation failure receipt: {product_id}")
                rejected.add(product_id)
                attempts.append(
                    {
                        "product_id": product_id,
                        "product_order": order,
                        "status": "source_rejected",
                        "failure_relative_path": failure.relative_to(root).as_posix(),
                        "failure_sha256": file_sha256(failure),
                    }
                )
        frontier = next_fixed_source_product(
            assignment,
            g0_attempts=g0_groups.get(str(assignment["group_id"]), []),
            completed_products=completed,
            rejected_products=rejected,
        )
        if frontier is not None:
            order, product_id = frontier
            directory = products_root / _safe_identifier(product_id, name="product_id")
            try:
                prepared = prepare_source_product(
                    row=rows[product_id],
                    assignment=assignment,
                    product_order=order,
                    prior_path=prior_path,
                    paper1_lock=paper1_lock,
                    paper1_source_root=paper1_source_root,
                    output_directory=directory,
                    schedule_bindings=schedule_bindings,
                    options=options,
                    paper2_root=root,
                    source_builder=source_builder,
                )
                completed.add(product_id)
                attempts.append(
                    {
                        "product_id": product_id,
                        "product_order": order,
                        "status": "complete",
                        "manifest_relative_path": (
                            directory / "source-prepared-manifest.json"
                        ).relative_to(root).as_posix(),
                        "manifest_sha256": file_sha256(
                            directory / "source-prepared-manifest.json"
                        ),
                        "manifest_content_sha256": prepared["content_sha256"],
                    }
                )
            except SourceRejectedError as exc:
                failure_payload = {
                    "schema": SOURCE_PREPARATION_FAILURE_SCHEMA,
                    "state": "terminal_source_rejected",
                    "product_id": product_id,
                    "geographic_group_id": str(assignment["group_id"]),
                    "product_order": order,
                    "fixed_product_ids": products,
                    "reason_type": type(exc).__name__,
                    "reason": str(exc)[:500],
                    "fallback_authorized": order + 1 < len(products),
                    "input_bindings_sha256": canonical_sha256(
                        {
                            "schedule": schedule_bindings,
                            "assignment": assignment,
                            "product_order": order,
                            "catalog_row_sha256": canonical_sha256(rows[product_id]),
                        }
                    ),
                }
                failure_payload["content_sha256"] = canonical_sha256(failure_payload)
                failure_path = write_canonical_json(
                    directory / "source-preparation-failure.json",
                    failure_payload,
                    paper2_root=root,
                )
                rejected.add(product_id)
                attempts.append(
                    {
                        "product_id": product_id,
                        "product_order": order,
                        "status": "source_rejected",
                        "failure_relative_path": failure_path.relative_to(root).as_posix(),
                        "failure_sha256": file_sha256(failure_path),
                    }
                )
        active = next_fixed_source_product(
            assignment,
            g0_attempts=g0_groups.get(str(assignment["group_id"]), []),
            completed_products=frozenset(),
            rejected_products=rejected,
        )
        selected = None
        if active is not None and active[1] in completed:
            selected = active[1]
        results.append(
            {
                "group_id": str(assignment["group_id"]),
                "role": str(assignment["role"]),
                "fixed_product_ids": products,
                "source_ready_product_id": selected,
                "state": "source_ready" if selected else "incomplete",
                "attempts": sorted(attempts, key=lambda value: int(value["product_order"])),
            }
        )
    state = "frontier_ready" if all(item["state"] == "source_ready" for item in results) else "incomplete"
    payload: dict[str, Any] = {
        "schema": SOURCE_PREPARATION_INDEX_SCHEMA,
        "state": state,
        **schedule_bindings,
        "source_protocol": SOURCE_PROTOCOL,
        "paper1": paper1_lock.to_dict(),
        "paper1_code_lock_content_sha256": build_paper1_code_lock(paper1_source_root)[
            "content_sha256"
        ],
        "options_sha256": canonical_sha256(asdict(options)),
        "expected_group_count": len(assignments),
        "source_ready_group_count": sum(item["state"] == "source_ready" for item in results),
        "groups": sorted(results, key=lambda value: value["group_id"]),
    }
    payload["contract_sha256"] = source_preparation_contract_sha256(payload)
    payload["content_sha256"] = canonical_sha256(payload)
    output = guard_paper2_path(output_index, paper2_root=root)
    _recoverable_json(output, payload, paper2_root=root)
    return payload

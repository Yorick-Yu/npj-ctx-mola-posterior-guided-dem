from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    AssetLock,
    AssetRecord,
    LaserObservations,
    PlanContract,
    reconstruction_from_observations,
)
from .hashing import canonical_sha256, file_sha256, guard_paper2_path, read_json, write_canonical_json
from .observations import simulate_and_write_laser_observations
from .posterior_preparation import (
    FrozenGPPrior,
    GP_PRIOR_SCHEMA,
    merge_training_patch_indices,
    write_gp_posterior,
    write_training_patch_index,
)
from .source_preparation import verify_source_preparation_manifest


TRAINING_STATISTICS_SCHEMA = "paper2.training-statistics/v1"
TRAINING_DATA_PREPARATION_SCHEMA = "paper2.formal-training-data-preparation/v1"
TRAINING_SOURCE_SCHEMA = "paper2.training-source-manifest/v1"
MAXIMUM_CONFIG_SCHEMA = "paper2.g3-maximum-input/v1"
NORMALIZATION_PROTOCOL = "bridge-train-global-median-mad-v1"
GP_PRIOR_PROTOCOL = "bridge-train-residual-matern32-correlation-fit-v1"
CORRELATION_LAGS = (1, 2, 4, 8, 16, 32, 64)


def _present_file(asset: AssetRecord, kind: str, *, root: Path) -> Path:
    matches = [item for item in asset.files if item.kind == kind]
    if len(matches) != 1:
        raise ValueError(f"{asset.product_id} requires exactly one {kind} file")
    record = matches[0]
    if record.status != "present" or record.relative_path is None:
        raise ValueError(f"{asset.product_id} {kind} is not present")
    path = guard_paper2_path(record.relative_path, paper2_root=root)
    if (
        not path.is_file()
        or record.sha256 != file_sha256(path)
        or record.byte_count != path.stat().st_size
    ):
        raise ValueError(f"{asset.product_id} {kind} bytes differ from the asset lock")
    return path


def _controlled_dem_from_problem(
    asset: AssetRecord, *, root: Path
) -> tuple[Path, np.ndarray, np.ndarray]:
    """Load the factor-24 controlled Z0, never the independent MOLA E10 Z0."""

    problem_path = _present_file(asset, "problem", root=root)
    if file_sha256(problem_path) != asset.problem_sha256:
        raise ValueError("controlled source problem differs from the asset lock")
    with np.load(problem_path, allow_pickle=False) as archive:
        required = {"tile_id", "coarse_dem", "valid_mask"}
        if not required.issubset(archive.files):
            raise ValueError("controlled source problem lacks Z0/grid fields")
        tile_id = str(archive["tile_id"].item())
        z0 = np.asarray(archive["coarse_dem"], dtype=np.float32)
        valid = np.asarray(archive["valid_mask"], dtype=bool)
    if (
        tile_id != asset.product_id
        or z0.ndim != 2
        or valid.shape != z0.shape
        or not np.any(valid)
        or np.any(~np.isfinite(z0[valid]))
    ):
        raise ValueError("controlled source problem Z0/grid identity differs")
    return problem_path, z0, valid


def _reference_path(asset: AssetRecord, *, root: Path) -> tuple[Path, str]:
    manifest_path = _present_file(asset, "source_preparation_manifest", root=root)
    manifest = verify_source_preparation_manifest(manifest_path, paper2_root=root)
    if manifest.get("product_id") != asset.product_id or manifest.get("role") != asset.role:
        raise ValueError("source preparation manifest differs from the locked training asset")
    raw = dict(manifest.get("artifacts", {})).get("evaluator_reference")
    if not isinstance(raw, Mapping) or raw.get("access_scope") != "evaluator_only":
        raise ValueError("source preparation lacks one evaluator-only reference")
    reference = guard_paper2_path(str(raw.get("relative_path", "")), paper2_root=root)
    if not reference.is_file() or raw.get("sha256") != file_sha256(reference):
        raise ValueError("training statistics reference differs from its source manifest")
    return reference, file_sha256(manifest_path)


def _load_npy(path: Path, *, name: str) -> np.ndarray:
    if path.suffix != ".npy":
        raise ValueError(f"{name} must use the narrow .npy G0 contract")
    value = np.load(path, allow_pickle=False)
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must contain one NumPy array")
    return value


def _robust_location_scale(values: np.ndarray, *, name: str) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        raise ValueError(f"{name} normalization lacks two finite values")
    center = float(np.median(finite))
    scale = float(1.4826 * np.median(np.abs(finite - center)))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.std(finite, ddof=1))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{name} normalization scale is not positive")
    return center, scale


def _matern32_correlation(distance: np.ndarray, length_scale: float) -> np.ndarray:
    scaled = np.sqrt(3.0) * np.asarray(distance, dtype=np.float64) / length_scale
    return (1.0 + scaled) * np.exp(-scaled)


def _fit_length_scale(records: Sequence[tuple[float, float, int]], *, fallback_m: float) -> float:
    usable = [
        (float(distance), float(correlation), int(weight))
        for distance, correlation, weight in records
        if distance > 0.0 and np.isfinite(correlation) and weight > 0
    ]
    if not usable:
        return float(fallback_m)
    distances = np.asarray([item[0] for item in usable], dtype=np.float64)
    correlations = np.asarray([item[1] for item in usable], dtype=np.float64)
    weights = np.asarray([item[2] for item in usable], dtype=np.float64)
    weights /= np.sum(weights)
    lower = max(float(np.min(distances)) / 4.0, 1.0e-3)
    upper = max(float(np.max(distances)) * 10.0, lower * 2.0)
    candidates = np.geomspace(lower, upper, 512)
    losses = np.asarray(
        [
            np.sum(weights * (_matern32_correlation(distances, scale) - correlations) ** 2)
            for scale in candidates
        ]
    )
    return float(candidates[int(np.argmin(losses))])


def _axis_correlation_records(
    residual: np.ndarray,
    valid: np.ndarray,
    transform: Sequence[float],
) -> tuple[list[tuple[float, float, int]], list[tuple[float, float, int]]]:
    centered = np.asarray(residual, dtype=np.float64).copy()
    centered[~valid] = np.nan
    centered -= float(np.nanmean(centered))
    a, b, _, d, e, _ = (float(value) for value in transform)
    x_pixel_m = float(np.hypot(a, d))
    y_pixel_m = float(np.hypot(b, e))
    if min(x_pixel_m, y_pixel_m) <= 0.0:
        raise ValueError("training reference transform has a zero pixel scale")
    results: list[list[tuple[float, float, int]]] = [[], []]
    for axis, pixel_m in ((1, x_pixel_m), (0, y_pixel_m)):
        dimension = centered.shape[axis]
        for lag in CORRELATION_LAGS:
            if lag >= dimension:
                continue
            first_slice = [slice(None), slice(None)]
            second_slice = [slice(None), slice(None)]
            first_slice[axis] = slice(0, dimension - lag)
            second_slice[axis] = slice(lag, dimension)
            first = centered[tuple(first_slice)]
            second = centered[tuple(second_slice)]
            pair = np.isfinite(first) & np.isfinite(second)
            count = int(np.sum(pair))
            if count < 32:
                continue
            first_values = first[pair]
            second_values = second[pair]
            denominator = float(
                np.sqrt(np.sum(first_values**2) * np.sum(second_values**2))
            )
            if denominator <= 0.0:
                continue
            correlation = float(np.sum(first_values * second_values) / denominator)
            results[axis].append((lag * pixel_m, correlation, count))
    return results[0], results[1]


def derive_training_statistics(
    *,
    paper2_root: str | Path,
    asset_lock_path: str | Path,
    normalization_output: str | Path,
    gp_prior_output: str | Path,
    receipt_output: str | Path,
    require_formal_counts: bool = True,
) -> dict[str, Any]:
    """Freeze train-only normalization and a residual GP prior before validation/test use."""

    root = Path(paper2_root).resolve()
    lock_path = guard_paper2_path(asset_lock_path, paper2_root=root)
    asset_lock = AssetLock.read(lock_path)
    training_assets = sorted(
        (asset for asset in asset_lock.assets if asset.role == "bridge_train"),
        key=lambda asset: asset.product_id,
    )
    expected_count = 48 if require_formal_counts else len(training_assets)
    if not training_assets or len(training_assets) != expected_count:
        raise ValueError(
            f"training statistics require {expected_count} bridge_train assets; "
            f"found {len(training_assets)}"
        )

    elevation_values: list[np.ndarray] = []
    image_values: list[np.ndarray] = []
    residual_values: list[np.ndarray] = []
    x_correlations: list[tuple[float, float, int]] = []
    y_correlations: list[tuple[float, float, int]] = []
    tile_records: list[dict[str, Any]] = []
    pixel_scales: list[float] = []
    for asset in training_assets:
        if not asset.qa.passed:
            raise ValueError(f"bridge_train asset did not pass G0 QA: {asset.product_id}")
        image_path = _present_file(asset, "image", root=root)
        controlled_problem_path, controlled_z0, controlled_mask = (
            _controlled_dem_from_problem(asset, root=root)
        )
        reference_path, source_manifest_sha256 = _reference_path(asset, root=root)
        image = np.asarray(_load_npy(image_path, name="image"), dtype=np.float64)
        z0 = np.asarray(controlled_z0, dtype=np.float64)
        dem_mask = np.asarray(controlled_mask, dtype=bool)
        with np.load(reference_path, allow_pickle=False) as archive:
            required = {"tile_id", "dem", "valid_mask", "transform", "crs_wkt"}
            if not required.issubset(archive.files):
                raise ValueError("bridge_train evaluator reference has an incomplete grid contract")
            tile_id = str(archive["tile_id"].item())
            target = np.asarray(archive["dem"], dtype=np.float64)
            target_mask = np.asarray(archive["valid_mask"], dtype=bool)
            transform = tuple(float(value) for value in archive["transform"])
            crs_wkt = str(archive["crs_wkt"].item())
        if tile_id != asset.product_id:
            raise ValueError("bridge_train reference tile differs from the asset lock")
        if image.shape != z0.shape or dem_mask.shape != z0.shape or target.shape != z0.shape:
            raise ValueError("bridge_train statistics arrays do not share the G0 grid")
        valid = dem_mask & target_mask & np.isfinite(z0) & np.isfinite(target)
        image_valid = dem_mask & np.isfinite(image)
        if not np.any(valid) or not np.any(image_valid):
            raise ValueError("bridge_train tile has no finite statistics support")
        residual = target - z0
        elevation_values.append(target[valid])
        image_values.append(image[image_valid])
        residual_values.append(residual[valid])
        x_records, y_records = _axis_correlation_records(residual, valid, transform)
        x_correlations.extend(x_records)
        y_correlations.extend(y_records)
        a, b, _, d, e, _ = transform
        pixel_scales.extend((float(np.hypot(a, d)), float(np.hypot(b, e))))
        tile_records.append(
            {
                "tile_id": asset.product_id,
                "geographic_group_id": asset.geographic_group_id,
                "source_preparation_manifest_sha256": source_manifest_sha256,
                "reference_sha256": file_sha256(reference_path),
                "image_sha256": file_sha256(image_path),
                "controlled_problem_sha256": file_sha256(controlled_problem_path),
                "z0_source": "paper1-controlled-degradation-factor-24",
                "crs_wkt_sha256": canonical_sha256(crs_wkt),
                "transform_hex": [float(value).hex() for value in transform],
                "elevation_sample_count": int(np.sum(valid)),
                "image_sample_count": int(np.sum(image_valid)),
            }
        )

    elevations = np.concatenate(elevation_values)
    images = np.concatenate(image_values)
    residuals = np.concatenate(residual_values)
    elevation_center, elevation_scale = _robust_location_scale(
        elevations, name="elevation"
    )
    image_center, image_scale = _robust_location_scale(images, name="image")
    residual_center, residual_scale = _robust_location_scale(residuals, name="residual")
    fallback_scale = float(np.median(np.asarray(pixel_scales))) * 8.0
    length_x = _fit_length_scale(x_correlations, fallback_m=fallback_scale)
    length_y = _fit_length_scale(y_correlations, fallback_m=fallback_scale)
    normalization = {
        "elevation_center": elevation_center,
        "elevation_scale": elevation_scale,
        "image_center": image_center,
        "image_scale": image_scale,
    }
    prior_payload = {
        "schema": GP_PRIOR_SCHEMA,
        "kernel": "matern32",
        "feature_space": "world_xy_pixel_centers",
        "variance": float(residual_scale**2),
        "length_scale_xy_m": [length_x, length_y],
        "nugget_variance": 0.1,
        "block_size": 4096,
        "epsilon": 1.0e-10,
    }
    FrozenGPPrior.from_mapping(prior_payload)
    normalization_path = write_canonical_json(
        normalization_output, normalization, paper2_root=root
    )
    prior_path = write_canonical_json(gp_prior_output, prior_payload, paper2_root=root)
    receipt: dict[str, Any] = {
        "schema": TRAINING_STATISTICS_SCHEMA,
        "state": "complete",
        "reference_access_scope": "bridge_train_truth_authorized_statistics_only",
        "validation_or_test_reference_accessed": False,
        "asset_lock_sha256": file_sha256(lock_path),
        "bridge_train_tile_count": len(training_assets),
        "normalization_protocol": NORMALIZATION_PROTOCOL,
        "gp_prior_protocol": GP_PRIOR_PROTOCOL,
        "normalization_sha256": file_sha256(normalization_path),
        "gp_prior_sha256": file_sha256(prior_path),
        "residual_center_m": residual_center,
        "residual_robust_scale_m": residual_scale,
        "correlation_lags_pixels": list(CORRELATION_LAGS),
        "correlation_record_counts": {
            "x": len(x_correlations),
            "y": len(y_correlations),
        },
        "tiles": tile_records,
    }
    receipt["content_sha256"] = canonical_sha256(receipt)
    receipt_path = write_canonical_json(receipt_output, receipt, paper2_root=root)
    return {
        **receipt,
        "normalization_relative_path": normalization_path.relative_to(root).as_posix(),
        "gp_prior_relative_path": prior_path.relative_to(root).as_posix(),
        "receipt_relative_path": receipt_path.relative_to(root).as_posix(),
        "receipt_file_sha256": file_sha256(receipt_path),
    }


def validate_training_statistics_receipt(
    path: str | Path, *, paper2_root: str | Path
) -> dict[str, Any]:
    root = Path(paper2_root).resolve()
    receipt_path = guard_paper2_path(path, paper2_root=root)
    payload = read_json(receipt_path)
    if payload.get("schema") != TRAINING_STATISTICS_SCHEMA or payload.get("state") != "complete":
        raise ValueError("training statistics receipt is not complete")
    declared = payload.get("content_sha256")
    unhashed = dict(payload)
    unhashed.pop("content_sha256", None)
    if declared != canonical_sha256(unhashed):
        raise ValueError("training statistics receipt content hash differs")
    if payload.get("validation_or_test_reference_accessed") is not False:
        raise ValueError("training statistics accessed validation/test truth")
    return payload


def _source_plan_and_reference(
    asset: AssetRecord, *, root: Path, budget: int
) -> tuple[Path, Path, dict[str, Any]]:
    manifest_path = _present_file(asset, "source_preparation_manifest", root=root)
    manifest = verify_source_preparation_manifest(manifest_path, paper2_root=root)
    if manifest.get("product_id") != asset.product_id or manifest.get("role") != asset.role:
        raise ValueError("source preparation identity differs from the training asset")
    plan_record = dict(manifest.get("plans", {})).get(str(budget))
    if not isinstance(plan_record, Mapping):
        raise ValueError(f"source preparation lacks the K={budget} plan")
    plan_path = guard_paper2_path(
        str(plan_record.get("relative_path", "")), paper2_root=root
    )
    if not plan_path.is_file() or plan_record.get("sha256") != file_sha256(plan_path):
        raise ValueError("source preparation plan bytes differ")
    reference_record = dict(manifest.get("artifacts", {})).get("evaluator_reference")
    if (
        not isinstance(reference_record, Mapping)
        or reference_record.get("access_scope") != "evaluator_only"
    ):
        raise ValueError("source preparation lacks an evaluator-only reference")
    reference_path = guard_paper2_path(
        str(reference_record.get("relative_path", "")), paper2_root=root
    )
    if (
        not reference_path.is_file()
        or reference_record.get("sha256") != file_sha256(reference_path)
    ):
        raise ValueError("source preparation reference bytes differ")
    return plan_path, reference_path, manifest


def prepare_formal_training_data(
    *,
    paper2_root: str | Path,
    asset_lock_path: str | Path,
    training_statistics_receipt_path: str | Path,
    normalization_path: str | Path,
    gp_prior_path: str | Path,
    output_root: str | Path = "data/formal-training",
    seed: int = 20260721,
    require_formal_counts: bool = True,
) -> dict[str, Any]:
    """Build the complete real K=100 bridge-train/validation input closure.

    This is a truth-authorized preparation process.  Its reconstruction problems,
    patches, merged indices, and returned receipt contain hashes but no reference path.
    The separate training-source manifest is the only verifier-only path-bearing artifact.
    """

    root = Path(paper2_root).resolve()
    asset_path = guard_paper2_path(asset_lock_path, paper2_root=root)
    asset_lock = AssetLock.read(asset_path)
    statistics_path = guard_paper2_path(
        training_statistics_receipt_path, paper2_root=root
    )
    statistics = validate_training_statistics_receipt(
        statistics_path, paper2_root=root
    )
    if statistics.get("asset_lock_sha256") != file_sha256(asset_path):
        raise ValueError("training statistics were derived from another asset lock")
    normalization_file = guard_paper2_path(normalization_path, paper2_root=root)
    prior_file = guard_paper2_path(gp_prior_path, paper2_root=root)
    if (
        statistics.get("normalization_sha256") != file_sha256(normalization_file)
        or statistics.get("gp_prior_sha256") != file_sha256(prior_file)
    ):
        raise ValueError("training statistics output bytes differ from the receipt")
    normalization = read_json(normalization_file)
    if set(normalization) != {
        "elevation_center",
        "elevation_scale",
        "image_center",
        "image_scale",
    }:
        raise ValueError("formal normalization fields differ")
    prior = FrozenGPPrior.from_mapping(read_json(prior_file))
    preparation_root = guard_paper2_path(
        output_root, paper2_root=root, allow_root=True
    )
    if preparation_root == root:
        raise ValueError("formal training preparation requires an output below paper2 root")

    assets = sorted(
        (
            asset
            for asset in asset_lock.assets
            if asset.role in {"bridge_train", "bridge_val"}
        ),
        key=lambda asset: (asset.role, asset.product_id),
    )
    counts = Counter(asset.role for asset in assets)
    expected = {"bridge_train": 48, "bridge_val": 12}
    if require_formal_counts and dict(counts) != expected:
        raise ValueError(f"formal training roles differ from {expected}: {dict(counts)}")
    if not assets:
        raise ValueError("formal training preparation has no bridge assets")

    asset_lock_sha256 = file_sha256(asset_path)
    normalization_sha256 = file_sha256(normalization_file)
    prior_sha256 = file_sha256(prior_file)
    tile_indices: dict[str, list[tuple[dict[str, Any], str]]] = {
        "bridge_train": [],
        "bridge_val": [],
    }
    source_records: list[dict[str, Any]] = []
    maximum_records: list[dict[str, Any]] = []
    receipt_tiles: list[dict[str, Any]] = []
    for asset in assets:
        if not asset.qa.passed or asset.prepared_manifest_sha256 is None:
            raise ValueError(f"formal training asset is not QA/prepared: {asset.product_id}")
        plan_path, reference_path, source_manifest = _source_plan_and_reference(
            asset, root=root, budget=100
        )
        plan = PlanContract.read(plan_path)
        if (
            plan.tile_id != asset.product_id
            or plan.budget != 100
            or plan.problem_sha256 != asset.problem_sha256
        ):
            raise ValueError("formal K=100 plan differs from the locked G0 asset")
        image_path = _present_file(asset, "image", root=root)
        _, controlled_z0, controlled_mask = _controlled_dem_from_problem(
            asset, root=root
        )
        image = np.asarray(_load_npy(image_path, name="image"), dtype=np.float32)
        z0 = np.asarray(controlled_z0, dtype=np.float32)
        valid_mask = np.asarray(controlled_mask, dtype=bool)
        if image.shape != z0.shape or valid_mask.shape != z0.shape:
            raise ValueError("formal CTX image/controlled-Z0/mask arrays do not align")

        tile_root = preparation_root / asset.role / asset.product_id
        observations_path = tile_root / "observations/laser-observations.npz"
        simulate_and_write_laser_observations(
            plan,
            reference_path,
            observations_path,
            replicate=0,
            seed=seed,
            paper2_root=root,
        )
        observations = LaserObservations.read(observations_path)
        problem = reconstruction_from_observations(
            tile_id=asset.product_id,
            z0=z0,
            image=image,
            valid_mask=valid_mask,
            observations=observations,
            observations_path=observations_path,
            frozen_model_hashes={
                "asset_lock": asset_lock_sha256,
                "gp_prior": prior_sha256,
                "normalization": normalization_sha256,
            },
            metadata={
                "prepared_manifest_sha256": asset.prepared_manifest_sha256,
                "z0_source": "paper1-controlled-degradation-factor-24",
                "degradation_factor": 24,
            },
            paper2_root=root,
        )
        problem_path = problem.write(
            tile_root / "reconstruction-problem.npz", paper2_root=root
        )
        posterior_path = tile_root / "gp/gp-posterior.npz"
        gp_manifest_path = tile_root / "gp/gp-posterior-manifest.json"
        write_gp_posterior(
            problem=problem,
            plan=plan,
            prior=prior,
            problem_sha256=file_sha256(problem_path),
            plan_sha256=file_sha256(plan_path),
            prior_sha256=prior_sha256,
            posterior_output=posterior_path,
            manifest_output=gp_manifest_path,
            paper2_root=root,
        )
        with np.load(reference_path, allow_pickle=False) as archive:
            target = np.asarray(archive["dem"], dtype=np.float64)
            target_mask = np.asarray(archive["valid_mask"], dtype=bool)
        with np.load(posterior_path, allow_pickle=False) as archive:
            zf = np.asarray(archive["zf"], dtype=np.float64)
            omega = np.asarray(archive["omega"], dtype=np.float64)
        source_hashes = {
            "asset_lock_sha256": asset_lock_sha256,
            "problem_sha256": file_sha256(problem_path),
            "plan_sha256": file_sha256(plan_path),
            "gp_manifest_sha256": file_sha256(gp_manifest_path),
            "gp_posterior_sha256": file_sha256(posterior_path),
            "target_label_sha256": file_sha256(reference_path),
            "normalization_sha256": normalization_sha256,
        }
        index_path = tile_root / "training-index.json"
        index_payload = write_training_patch_index(
            problem=problem,
            plan=plan,
            target=target,
            target_valid_mask=target_mask,
            zf=zf,
            omega=omega,
            normalization=normalization,
            role=asset.role,
            geographic_group_id=asset.geographic_group_id,
            source_hashes=source_hashes,
            output_directory=tile_root / "patches",
            index_output=index_path,
            paper2_root=root,
        )
        tile_indices[asset.role].append((index_payload, file_sha256(index_path)))
        relative_sources = {
            "problem": problem_path.relative_to(root).as_posix(),
            "plan": plan_path.relative_to(root).as_posix(),
            "gp_manifest": gp_manifest_path.relative_to(root).as_posix(),
            "gp_posterior": posterior_path.relative_to(root).as_posix(),
            "target_label": reference_path.relative_to(root).as_posix(),
            "normalization": normalization_file.relative_to(root).as_posix(),
        }
        source_records.append(
            {
                "tile_id": asset.product_id,
                "role": asset.role,
                "geographic_group_id": asset.geographic_group_id,
                "target_dem_key": "dem",
                "target_mask_key": "valid_mask",
                **relative_sources,
                **source_hashes,
            }
        )
        maximum_records.append(
            {
                "tile_id": asset.product_id,
                "plan": plan_path.relative_to(root).as_posix(),
                "plan_sha256": file_sha256(plan_path),
                "problem": problem_path.relative_to(root).as_posix(),
                "problem_sha256": file_sha256(problem_path),
            }
        )
        receipt_tiles.append(
            {
                "tile_id": asset.product_id,
                "role": asset.role,
                "geographic_group_id": asset.geographic_group_id,
                "source_preparation_manifest_sha256": str(
                    source_manifest["content_sha256"]
                ),
                "plan_sha256": file_sha256(plan_path),
                "observations_sha256": file_sha256(observations_path),
                "problem_sha256": file_sha256(problem_path),
                "gp_manifest_sha256": file_sha256(gp_manifest_path),
                "gp_posterior_sha256": file_sha256(posterior_path),
                "training_index_sha256": file_sha256(index_path),
            }
        )

    merged_paths: dict[str, Path] = {}
    merged_payloads: dict[str, dict[str, Any]] = {}
    for role in ("bridge_train", "bridge_val"):
        if not tile_indices[role]:
            raise ValueError(f"formal training preparation has no {role} components")
        path = preparation_root / f"{role}-index.json"
        payload = merge_training_patch_indices(
            components=tile_indices[role], output=path, paper2_root=root
        )
        merged_paths[role] = path
        merged_payloads[role] = payload
    train_path = merged_paths["bridge_train"]
    val_path = merged_paths["bridge_val"]
    source_payload = {
        "schema": TRAINING_SOURCE_SCHEMA,
        "training_index_sha256": file_sha256(train_path),
        "validation_index_sha256": file_sha256(val_path),
        "asset_lock_sha256": asset_lock_sha256,
        "tiles": sorted(source_records, key=lambda item: str(item["tile_id"])),
    }
    source_manifest_path = write_canonical_json(
        preparation_root / "training-source-manifest.json",
        source_payload,
        paper2_root=root,
    )
    maximum_payload = {
        "schema": MAXIMUM_CONFIG_SCHEMA,
        "patch_size": 256,
        "image_count": 1,
        "laser_budget": 100,
        "actual_imagery": True,
        "full_training_validation_sampling_loop": True,
        "training_index_sha256": file_sha256(train_path),
        "validation_index_sha256": file_sha256(val_path),
        "asset_lock_sha256": asset_lock_sha256,
        "validation_sampling": {"path_count": 32, "step_count": 40},
        "tiles": sorted(maximum_records, key=lambda item: str(item["tile_id"])),
    }
    maximum_path = write_canonical_json(
        preparation_root / "g3-maximum-input.json",
        maximum_payload,
        paper2_root=root,
    )
    receipt: dict[str, Any] = {
        "schema": TRAINING_DATA_PREPARATION_SCHEMA,
        "state": "complete",
        "asset_lock_sha256": asset_lock_sha256,
        "training_statistics_receipt_sha256": file_sha256(statistics_path),
        "normalization_sha256": normalization_sha256,
        "gp_prior_sha256": prior_sha256,
        "observation_seed": int(seed),
        "observation_replicate": 0,
        "bridge_train_index_sha256": file_sha256(train_path),
        "bridge_val_index_sha256": file_sha256(val_path),
        "training_source_manifest_sha256": file_sha256(source_manifest_path),
        "g3_maximum_input_sha256": file_sha256(maximum_path),
        "tile_counts": dict(sorted(counts.items())),
        "reference_paths_exported_to_training_workers": False,
        "tiles": sorted(receipt_tiles, key=lambda item: str(item["tile_id"])),
    }
    receipt["content_sha256"] = canonical_sha256(receipt)
    receipt_path = write_canonical_json(
        preparation_root / "preparation-receipt.json", receipt, paper2_root=root
    )
    return {
        **receipt,
        "bridge_train_index": train_path.relative_to(root).as_posix(),
        "bridge_val_index": val_path.relative_to(root).as_posix(),
        "training_source_manifest": source_manifest_path.relative_to(root).as_posix(),
        "g3_maximum_input": maximum_path.relative_to(root).as_posix(),
        "preparation_receipt": receipt_path.relative_to(root).as_posix(),
        "preparation_receipt_file_sha256": file_sha256(receipt_path),
        "sample_counts": {
            role: len(merged_payloads[role]["samples"])
            for role in ("bridge_train", "bridge_val")
        },
    }

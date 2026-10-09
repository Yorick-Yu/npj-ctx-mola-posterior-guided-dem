"""Frozen GP posterior fields and leakage-safe neural training patches."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage

from .contracts import (
    AssetLock,
    AssetRecord,
    LaserObservations,
    PlanContract,
    ReconstructionProblem,
)
from .footprints import build_footprint_operator_with_metadata
from .gp import Matern32Kernel, footprint_gp_posterior
from .hashing import (
    bytes_sha256,
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    is_sha256,
    write_canonical_json,
    write_immutable_bytes,
)
from .models import method_spec
from .objectives import shifted_omega_field


GP_PRIOR_SCHEMA = "paper2.gp-prior/v1"
GP_POSTERIOR_SCHEMA = "paper2.gp-posterior-manifest/v1"
TRAINING_INDEX_SCHEMA = "paper2.training-index/v1"
TRAINING_ROLES = frozenset({"bridge_train", "bridge_val"})
PATCH_SIZE = 256
PATCH_OVERLAP = 0.25
REQUIRED_TRAINING_SOURCE_HASHES = frozenset(
    {
        "asset_lock_sha256",
        "problem_sha256",
        "plan_sha256",
        "gp_manifest_sha256",
        "gp_posterior_sha256",
        "target_label_sha256",
        "normalization_sha256",
    }
)
_SAMPLE_CONTRACT_KEYS = (
    "relative_path",
    "sha256",
    "tile_id",
    "geographic_group_id",
    "role",
    "row_start",
    "column_start",
)


def training_index_sample_set_sha256(
    *,
    role: str,
    asset_lock_sha256: str,
    tiles: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
) -> str:
    return canonical_sha256(
        {
            "schema": TRAINING_INDEX_SCHEMA,
            "role": role,
            "asset_lock_sha256": asset_lock_sha256,
            "patch_size": PATCH_SIZE,
            "overlap_fraction": PATCH_OVERLAP,
            "stride": int(PATCH_SIZE * (1.0 - PATCH_OVERLAP)),
            "tiles": [dict(tile) for tile in tiles],
            "samples": [
                {key: sample[key] for key in _SAMPLE_CONTRACT_KEYS}
                for sample in samples
            ],
        }
    )


@dataclass(frozen=True)
class FrozenGPPrior:
    variance: float
    length_scale_xy_m: tuple[float, float]
    nugget_variance: float
    block_size: int = 4096
    epsilon: float = 1.0e-10
    feature_space: str = "world_xy_pixel_centers"
    kernel: str = "matern32"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FrozenGPPrior":
        if value.get("schema") != GP_PRIOR_SCHEMA:
            raise ValueError(f"GP prior schema must be {GP_PRIOR_SCHEMA}")
        raw_scale = np.asarray(value.get("length_scale_xy_m"), dtype=np.float64)
        if raw_scale.ndim == 0:
            raw_scale = np.repeat(raw_scale, 2)
        if raw_scale.shape != (2,):
            raise ValueError("length_scale_xy_m must be a scalar or two values")
        result = cls(
            variance=float(value["variance"]),
            length_scale_xy_m=(float(raw_scale[0]), float(raw_scale[1])),
            nugget_variance=float(value.get("nugget_variance", 0.0)),
            block_size=int(value.get("block_size", 4096)),
            epsilon=float(value.get("epsilon", 1.0e-10)),
            feature_space=str(value.get("feature_space", "")),
            kernel=str(value.get("kernel", "")),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.kernel != "matern32":
            raise ValueError("the frozen GP kernel must be matern32")
        if self.feature_space != "world_xy_pixel_centers":
            raise ValueError("the frozen GP feature space must be world_xy_pixel_centers")
        if not np.isfinite(self.variance) or self.variance <= 0.0:
            raise ValueError("GP variance must be finite and positive")
        if not np.all(np.isfinite(self.length_scale_xy_m)) or any(
            value <= 0.0 for value in self.length_scale_xy_m
        ):
            raise ValueError("GP XY length scales must be finite and positive")
        if not np.isfinite(self.nugget_variance) or self.nugget_variance < 0.0:
            raise ValueError("GP nugget variance must be finite and nonnegative")
        if self.block_size <= 0:
            raise ValueError("GP block size must be positive")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("GP epsilon must be finite and positive")


def world_xy_features(
    grid_shape: Sequence[int], transform: Sequence[float]
) -> np.ndarray:
    """Return affine world coordinates at raster pixel centres."""

    if len(grid_shape) != 2 or len(transform) != 6:
        raise ValueError("grid shape and affine transform have invalid dimensions")
    rows, columns = (int(grid_shape[0]), int(grid_shape[1]))
    if rows <= 0 or columns <= 0:
        raise ValueError("grid dimensions must be positive")
    a, b, c, d, e, f = (float(item) for item in transform)
    if not np.all(np.isfinite((a, b, c, d, e, f))):
        raise ValueError("affine transform must be finite")
    row_grid, column_grid = np.indices((rows, columns), dtype=np.float64)
    column_center = column_grid + 0.5
    row_center = row_grid + 0.5
    x = a * column_center + b * row_center + c
    y = d * column_center + e * row_center + f
    return np.column_stack((x.ravel(), y.ravel()))


def _nearest_fill(
    value: np.ndarray,
    support: np.ndarray,
    *,
    fallback: float | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    valid = np.asarray(support, dtype=bool)
    if array.shape != valid.shape:
        raise ValueError("fill support must align with its array")
    valid &= np.isfinite(array)
    if np.all(valid):
        return np.array(array, copy=True)
    if not np.any(valid):
        if fallback is None or not np.isfinite(fallback):
            raise ValueError("array has no finite supported samples")
        return np.full(array.shape, float(fallback), dtype=np.float64)
    nearest = ndimage.distance_transform_edt(
        ~valid,
        return_distances=False,
        return_indices=True,
    )
    return np.asarray(array[tuple(nearest)], dtype=np.float64)


def _image_plane(image: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(image, dtype=np.float64)
    if value.ndim == 2 and value.shape == grid_shape:
        return value
    if value.ndim == 3 and value.shape[-2:] == grid_shape:
        return value[0]
    if value.ndim == 3 and value.shape[:2] == grid_shape:
        return value[..., 0]
    raise ValueError("image does not contain a plane aligned with the DEM")


def _laser_residual_grid(
    problem: ReconstructionProblem,
    plan: PlanContract,
    filled_z0: np.ndarray,
) -> np.ndarray:
    grid = np.zeros(problem.z0.shape, dtype=np.float64)
    a, b, c, d, e, f = plan.transform
    inverse = np.linalg.inv(np.asarray(((a, b), (d, e)), dtype=np.float64))
    for coordinate, value, flag in zip(
        problem.observation_coordinates,
        problem.observation_values_m,
        problem.observation_quality_flags,
        strict=True,
    ):
        if flag != 0:
            continue
        column_row = inverse @ (coordinate - np.asarray((c, f), dtype=np.float64)) - 0.5
        column, row = np.rint(column_row).astype(int)
        if 0 <= row < grid.shape[0] and 0 <= column < grid.shape[1]:
            grid[row, column] = float(value - filled_z0[row, column])
    return grid


def build_condition_planes(
    problem: ReconstructionProblem,
    plan: PlanContract,
    zf: np.ndarray,
    omega: np.ndarray,
    *,
    method: str,
    normalization: Mapping[str, float],
) -> np.ndarray:
    """Build the same finite six-plane condition tensor for training and inference."""

    spec = method_spec(method)
    required = {"elevation_center", "elevation_scale", "image_center", "image_scale"}
    if set(normalization) != required:
        raise ValueError(f"normalization must contain exactly {sorted(required)}")
    elevation_center = float(normalization["elevation_center"])
    elevation_scale = float(normalization["elevation_scale"])
    image_center = float(normalization["image_center"])
    image_scale = float(normalization["image_scale"])
    if not np.all(np.isfinite((elevation_center, elevation_scale, image_center, image_scale))):
        raise ValueError("normalization values must be finite")
    if elevation_scale <= 0.0 or image_scale <= 0.0:
        raise ValueError("normalization scales must be positive")
    shape = problem.z0.shape
    fused = np.asarray(zf, dtype=np.float64)
    uncertainty = np.asarray(omega, dtype=np.float64)
    if fused.shape != shape or uncertainty.shape != shape:
        raise ValueError("zf and omega must align with the reconstruction grid")
    if not np.all(np.isfinite(uncertainty)) or np.any((uncertainty < 0.0) | (uncertainty > 1.0)):
        raise ValueError("omega must be finite and lie in [0,1]")
    dem_support = problem.valid_mask & np.isfinite(problem.z0)
    z0_filled = _nearest_fill(problem.z0, dem_support, fallback=elevation_center)
    zf_filled = _nearest_fill(fused, problem.valid_mask & np.isfinite(fused), fallback=elevation_center)
    image = _image_plane(problem.image, shape)
    image_support = problem.valid_mask & np.isfinite(image)
    image_filled = _nearest_fill(image, image_support, fallback=image_center)
    laser = _laser_residual_grid(problem, plan, z0_filled)
    planes = np.stack(
        (
            (z0_filled - elevation_center) / elevation_scale,
            (image_filled - image_center) / image_scale
            if spec.uses_image
            else np.zeros(shape, dtype=np.float64),
            image_support.astype(np.float64)
            if spec.uses_image
            else np.zeros(shape, dtype=np.float64),
            (zf_filled - elevation_center) / elevation_scale
            if spec.uses_laser
            else np.zeros(shape, dtype=np.float64),
            uncertainty
            if spec.uses_uncertainty_condition
            else np.zeros(shape, dtype=np.float64),
            laser / elevation_scale
            if spec.uses_laser
            else np.zeros(shape, dtype=np.float64),
        )
    ).astype(np.float32)
    if not np.all(np.isfinite(planes)):
        raise RuntimeError("condition preparation produced non-finite values")
    return planes


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return canonical_sha256(
        {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "bytes_sha256": bytes_sha256(array.tobytes()),
        }
    )


def write_gp_posterior(
    *,
    problem: ReconstructionProblem,
    plan: PlanContract,
    prior: FrozenGPPrior,
    problem_sha256: str,
    plan_sha256: str,
    prior_sha256: str,
    posterior_output: str | Path,
    manifest_output: str | Path,
    paper2_root: str | Path,
) -> dict[str, Any]:
    """Assimilate all valid footprints once over a complete frozen tile."""

    prior.validate()
    if problem.tile_id != plan.tile_id or problem.plan_sha256 != plan.sha256:
        raise ValueError("GP problem and plan do not share one frozen identity")
    expected_ids = np.asarray([point.candidate_id for point in plan.points], dtype=str)
    expected_coordinates = np.asarray(
        [(point.x, point.y) for point in plan.points],
        dtype=np.float64,
    ).reshape((-1, 2))
    if not np.array_equal(problem.observation_candidate_ids, expected_ids):
        raise ValueError("GP problem observation IDs differ from the frozen plan")
    if not np.array_equal(problem.observation_coordinates, expected_coordinates):
        raise ValueError("GP problem observation coordinates differ from the frozen plan")
    for name, digest in (
        ("problem", problem_sha256),
        ("plan", plan_sha256),
        ("GP prior", prior_sha256),
    ):
        if not is_sha256(digest):
            raise ValueError(f"{name} hash must be a lowercase SHA-256")
    frozen_hashes = dict(problem.frozen_model_hashes)
    if prior_sha256 not in frozen_hashes.values():
        raise ValueError("GP prior file is not one of the problem's frozen model hashes")
    coarse = _nearest_fill(problem.z0, problem.valid_mask & np.isfinite(problem.z0))
    good = problem.observation_quality_flags == 0
    if np.any(good) and plan.noise_std_m <= 0.0:
        raise ValueError("positive GP observation noise is required when returns are present")
    footprint = build_footprint_operator_with_metadata(
        problem.observation_coordinates[good],
        grid_shape=problem.z0.shape,
        transform=plan.transform,
        diameter_m=plan.footprint.diameter_m,
        quadrature_points=plan.footprint.quadrature_points,
        valid_mask=problem.valid_mask,
        require_full_support=plan.footprint.require_full_support,
        drop_invalid=False,
    )
    maximum_support = 4 * plan.footprint.quadrature_points
    row_support = np.diff(footprint.operator.indptr)
    if row_support.size and int(np.max(row_support)) > maximum_support:
        raise RuntimeError("footprint support exceeds the locked quadrature bound")
    retained = problem.observation_values_m[good][footprint.kept_indices]
    grid_features = world_xy_features(problem.z0.shape, plan.transform)
    posterior = footprint_gp_posterior(
        coarse,
        grid_features,
        footprint.operator,
        retained,
        noise_covariance=plan.noise_std_m**2,
        kernel=Matern32Kernel(
            variance=prior.variance,
            length_scale=prior.length_scale_xy_m,
        ),
        nugget_variance=prior.nugget_variance,
        block_size=prior.block_size,
        epsilon=prior.epsilon,
        retain_cross_covariance=False,
    )
    zf = np.asarray(posterior.corrected_dem, dtype=np.float64)
    omega = np.asarray(posterior.relative_uncertainty, dtype=np.float64)
    variance = np.asarray(posterior.variance, dtype=np.float64)
    zf[~problem.valid_mask] = coarse[~problem.valid_mask]
    omega[~problem.valid_mask] = 1.0
    variance[~problem.valid_mask] = prior.variance
    posterior_path = write_immutable_bytes(
        posterior_output,
        _npz_bytes(
            zf=zf.astype(np.float32),
            omega=omega.astype(np.float32),
            variance=variance.astype(np.float32),
        ),
        paper2_root=paper2_root,
    )
    root = Path(paper2_root).resolve()
    payload: dict[str, Any] = {
        "schema": GP_POSTERIOR_SCHEMA,
        "tile_id": problem.tile_id,
        "problem_sha256": problem_sha256,
        "plan_sha256": plan_sha256,
        "observations_sha256": problem.observations_sha256,
        "prepared_manifest_sha256": problem.metadata.get("prepared_manifest_sha256"),
        "gp_prior_sha256": prior_sha256,
        "gp_prior_contract_sha256": canonical_sha256(
            {
                "kernel": prior.kernel,
                "feature_space": prior.feature_space,
                "variance": prior.variance,
                "length_scale_xy_m": list(prior.length_scale_xy_m),
                "nugget_variance": prior.nugget_variance,
                "block_size": prior.block_size,
                "epsilon": prior.epsilon,
            }
        ),
        "grid_features": {
            "definition": prior.feature_space,
            "order": "C_row_major",
            "shape": list(grid_features.shape),
            "sha256": _array_sha256(grid_features),
        },
        "footprint": plan.footprint.to_dict(),
        "input_valid_returns": int(np.sum(good)),
        "assimilated_returns": int(footprint.operator.shape[0]),
        "assimilated_candidate_ids": problem.observation_candidate_ids[good].tolist(),
        "footprint_operator_sha256": canonical_sha256(
            {
                "shape": list(footprint.operator.shape),
                "indptr": footprint.operator.indptr.tolist(),
                "indices": footprint.operator.indices.tolist(),
                "data": footprint.operator.data.tolist(),
            }
        ),
        "maximum_row_support": int(np.max(row_support)) if row_support.size else 0,
        "cholesky_jitter": posterior.cholesky_jitter,
        "outputs": {
            "posterior": {
                "relative_path": posterior_path.relative_to(root).as_posix(),
                "sha256": file_sha256(posterior_path),
                "keys": ["zf", "omega", "variance"],
            },
        },
        "omega_range": [float(np.min(omega)), float(np.max(omega))],
        "posterior_variance_range": [
            float(np.min(variance)),
            float(np.max(variance)),
        ],
        "invalid_z0_fill": {
            "algorithm": "nearest-valid-pixel-euclidean-index-v1",
            "invalid_pixel_count": int(np.sum(~(problem.valid_mask & np.isfinite(problem.z0)))),
            "source_support_sha256": _array_sha256(
                problem.valid_mask & np.isfinite(problem.z0)
            ),
            "filled_z0_sha256": _array_sha256(coarse),
            "metrics_mask": "problem.valid_mask",
        },
        "numerical_precision": "float64",
        "reference_dem_accessed": False,
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    manifest_path = write_canonical_json(
        manifest_output,
        payload,
        paper2_root=paper2_root,
    )
    return {**payload, "manifest_file_sha256": file_sha256(manifest_path)}


def verify_gp_observations(
    problem: ReconstructionProblem,
    plan: PlanContract,
    observations: LaserObservations,
    *,
    observations_file_sha256: str,
) -> None:
    """Prove that the GP inputs are the immutable simulated-observation artifact."""

    observations.validate()
    if observations_file_sha256 != problem.observations_sha256:
        raise ValueError("GP observation file hash differs from the reconstruction problem")
    if observations.tile_id != problem.tile_id or observations.tile_id != plan.tile_id:
        raise ValueError("GP observation tile identity mismatch")
    if observations.plan_sha256 != plan.sha256 or problem.plan_sha256 != plan.sha256:
        raise ValueError("GP observation plan identity mismatch")
    if observations.noise_std_m != plan.noise_std_m:
        raise ValueError("GP observation noise differs from the frozen plan")
    comparisons = (
        (observations.candidate_ids, problem.observation_candidate_ids, "candidate IDs"),
        (observations.coordinates, problem.observation_coordinates, "coordinates"),
        (observations.quality_flags, problem.observation_quality_flags, "quality flags"),
    )
    for left, right, name in comparisons:
        if not np.array_equal(left, right):
            raise ValueError(f"GP observation {name} differ from the reconstruction problem")
    if not np.array_equal(
        observations.return_values_m,
        problem.observation_values_m,
        equal_nan=True,
    ):
        raise ValueError("GP observation returns differ from the reconstruction problem")


def verify_gp_posterior_manifest(
    payload: Mapping[str, Any],
    *,
    manifest_file_sha256: str,
    problem_sha256: str,
    plan_sha256: str,
    posterior_sha256: str,
) -> None:
    if payload.get("schema") != GP_POSTERIOR_SCHEMA:
        raise ValueError(f"GP posterior schema must be {GP_POSTERIOR_SCHEMA}")
    declared = dict(payload)
    declared_hash = str(declared.pop("manifest_sha256", ""))
    if declared_hash != canonical_sha256(declared):
        raise ValueError("GP posterior manifest content hash mismatch")
    if not is_sha256(manifest_file_sha256):
        raise ValueError("GP posterior manifest file hash is invalid")
    if payload.get("problem_sha256") != problem_sha256 or payload.get(
        "plan_sha256"
    ) != plan_sha256:
        raise ValueError("GP posterior manifest binds a different problem or plan")
    outputs = payload.get("outputs", {})
    if not isinstance(outputs, Mapping):
        raise ValueError("GP posterior manifest outputs are missing")
    record = outputs.get("posterior")
    if not isinstance(record, Mapping) or record.get("sha256") != posterior_sha256:
        raise ValueError("GP posterior manifest binds a different posterior artifact")
    if record.get("keys") != ["zf", "omega", "variance"]:
        raise ValueError("GP posterior artifact must contain zf, omega, and variance")


def locked_training_asset(
    asset_lock: AssetLock,
    *,
    problem: ReconstructionProblem,
    plan: PlanContract,
) -> AssetRecord:
    """Resolve one tile's immutable role before any training reference is opened."""

    asset_lock.validate()
    matches = [asset for asset in asset_lock.assets if asset.product_id == problem.tile_id]
    if len(matches) != 1:
        raise ValueError("training tile must match exactly one asset-lock product")
    asset = matches[0]
    if asset.role not in TRAINING_ROLES:
        raise PermissionError(
            f"reference labels are forbidden for role {asset.role}; only bridge_train/bridge_val are allowed"
        )
    if not asset.qa.passed:
        raise ValueError("training tile has not passed frozen asset QA")
    if problem.tile_id != plan.tile_id or problem.plan_sha256 != plan.sha256:
        raise ValueError("training problem and plan do not share one frozen identity")
    prepared_hash = problem.metadata.get("prepared_manifest_sha256")
    if asset.prepared_manifest_sha256 is not None and prepared_hash != asset.prepared_manifest_sha256:
        raise ValueError("training problem differs from the asset's prepared manifest")
    if asset.problem_sha256 is not None and plan.problem_sha256 != asset.problem_sha256:
        raise ValueError("training plan differs from the asset's frozen source problem")
    return asset


def sliding_window_starts(length: int, *, patch_size: int = PATCH_SIZE) -> tuple[int, ...]:
    if patch_size != PATCH_SIZE:
        raise ValueError(f"formal training patch size is fixed at {PATCH_SIZE}")
    if length < patch_size:
        raise ValueError("tile dimension is smaller than the formal patch size")
    stride = int(round(patch_size * (1.0 - PATCH_OVERLAP)))
    last = length - patch_size
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return tuple(starts)


def write_training_patch_index(
    *,
    problem: ReconstructionProblem,
    plan: PlanContract,
    target: np.ndarray,
    target_valid_mask: np.ndarray,
    zf: np.ndarray,
    omega: np.ndarray,
    normalization: Mapping[str, float],
    role: str,
    geographic_group_id: str,
    source_hashes: Mapping[str, str],
    output_directory: str | Path,
    index_output: str | Path,
    paper2_root: str | Path,
) -> dict[str, Any]:
    """Write immutable, path-free patches and their content-addressed index."""

    if role not in TRAINING_ROLES:
        raise PermissionError("training patches may only be prepared for bridge_train/bridge_val")
    if not geographic_group_id:
        raise ValueError("geographic_group_id cannot be empty")
    for name, digest in source_hashes.items():
        if not is_sha256(str(digest)):
            raise ValueError(f"source hash {name} must be a lowercase SHA-256")
    missing_hashes = sorted(REQUIRED_TRAINING_SOURCE_HASHES.difference(source_hashes))
    if missing_hashes:
        raise ValueError(f"training patch sources lack hashes: {missing_hashes}")
    shape = problem.z0.shape
    truth = np.asarray(target, dtype=np.float64)
    truth_mask = np.asarray(target_valid_mask, dtype=bool)
    fused = np.asarray(zf, dtype=np.float64)
    uncertainty = np.asarray(omega, dtype=np.float64)
    if truth.shape != shape or truth_mask.shape != shape:
        raise ValueError("training target and target mask must align with the problem")
    if fused.shape != shape or uncertainty.shape != shape:
        raise ValueError("training zf and omega must align with the problem")
    train_valid = (
        problem.valid_mask
        & truth_mask
        & np.isfinite(truth)
        & np.isfinite(problem.z0)
        & np.isfinite(fused)
        & np.isfinite(uncertainty)
    )
    if not np.any(train_valid):
        raise ValueError("training tile contains no valid target pixels")
    elevation_center = float(normalization["elevation_center"])
    elevation_scale = float(normalization["elevation_scale"])
    if not np.isfinite(elevation_center) or not np.isfinite(elevation_scale) or elevation_scale <= 0:
        raise ValueError("training elevation normalization is invalid")
    target_filled = _nearest_fill(truth, train_valid, fallback=elevation_center)
    zf_filled = _nearest_fill(
        fused,
        problem.valid_mask & np.isfinite(fused),
        fallback=elevation_center,
    )
    condition = build_condition_planes(
        problem,
        plan,
        fused,
        uncertainty,
        method="OT-SB-full",
        normalization=normalization,
    )
    shifted = shifted_omega_field(
        uncertainty,
        tile_hash=str(source_hashes["problem_sha256"]),
    )
    target_normalized = ((target_filled - elevation_center) / elevation_scale).astype(np.float32)
    zf_normalized = ((zf_filled - elevation_center) / elevation_scale).astype(np.float32)
    output_root = guard_paper2_path(
        output_directory,
        paper2_root=paper2_root,
        allow_root=True,
    )
    root = Path(paper2_root).resolve()
    if output_root == root:
        raise ValueError("training patches require an output directory below paper2 root")
    row_starts = sliding_window_starts(shape[0])
    column_starts = sliding_window_starts(shape[1])
    samples: list[dict[str, Any]] = []
    expected_windows = {
        (row, column) for row in row_starts for column in column_starts
    }
    for row in row_starts:
        for column in column_starts:
            rows = slice(row, row + PATCH_SIZE)
            columns = slice(column, column + PATCH_SIZE)
            patch_valid = train_valid[rows, columns]
            if not np.any(patch_valid):
                raise ValueError(
                    f"formal training window ({row},{column}) has no valid target pixels"
                )
            patch_path = output_root / f"patch-r{row:06d}-c{column:06d}.npz"
            # There are deliberately no path-, reference-, or truth-bearing metadata
            # fields in a patch. The label is named ``target`` for the training API.
            patch_bytes = _npz_bytes(
                tile_id=np.asarray(problem.tile_id),
                geographic_group_id=np.asarray(geographic_group_id),
                role=np.asarray(role),
                target=target_normalized[rows, columns],
                zf=zf_normalized[rows, columns],
                omega=uncertainty[rows, columns].astype(np.float32),
                shifted_omega=shifted[rows, columns].astype(np.float32),
                condition=condition[:, rows, columns],
                valid_mask=patch_valid,
            )
            written = write_immutable_bytes(
                patch_path,
                patch_bytes,
                paper2_root=paper2_root,
            )
            samples.append(
                {
                    "relative_path": written.relative_to(root).as_posix(),
                    "sha256": file_sha256(written),
                    "tile_id": problem.tile_id,
                    "geographic_group_id": geographic_group_id,
                    "role": role,
                    "row_start": row,
                    "column_start": column,
                    "valid_fraction": float(np.mean(patch_valid)),
                }
            )
    if not samples:
        raise ValueError("no nonempty training patches were produced")
    payload: dict[str, Any] = {
        "schema": TRAINING_INDEX_SCHEMA,
        "role": role,
        "asset_lock_sha256": str(source_hashes["asset_lock_sha256"]),
        "tile_id": problem.tile_id,
        "geographic_group_id": geographic_group_id,
        "patch_size": PATCH_SIZE,
        "overlap_fraction": PATCH_OVERLAP,
        "stride": int(PATCH_SIZE * (1.0 - PATCH_OVERLAP)),
        "tiles": [
            {
                "tile_id": problem.tile_id,
                "geographic_group_id": geographic_group_id,
                "shape": list(shape),
            }
        ],
        "source_hashes": dict(sorted((str(key), str(value)) for key, value in source_hashes.items())),
        "fill_algorithms": {
            "target": "nearest-valid-pixel-euclidean-index-v1",
            "zf": "nearest-valid-pixel-euclidean-index-v1",
            "metrics_mask": "target_valid_mask_and_problem_valid_mask",
        },
        "samples": samples,
    }
    observed_windows = {
        (int(sample["row_start"]), int(sample["column_start"])) for sample in samples
    }
    if observed_windows != expected_windows or len(samples) != len(expected_windows):
        raise AssertionError("training patch window set is not exactly closed")
    payload["sample_set_sha256"] = training_index_sample_set_sha256(
        role=role,
        asset_lock_sha256=payload["asset_lock_sha256"],
        tiles=payload["tiles"],
        samples=samples,
    )
    index_path = write_canonical_json(index_output, payload, paper2_root=paper2_root)
    return {**payload, "index_file_sha256": file_sha256(index_path)}


def merge_training_patch_indices(
    *,
    components: Sequence[tuple[Mapping[str, Any], str]],
    output: str | Path,
    paper2_root: str | Path,
) -> dict[str, Any]:
    """Merge independently prepared tile indices into one immutable role index."""

    if not components:
        raise ValueError("at least one training index is required for merging")
    roles = {str(payload.get("role", "")) for payload, _ in components}
    asset_hashes = {
        str(payload.get("asset_lock_sha256", "")) for payload, _ in components
    }
    if len(roles) != 1 or next(iter(roles)) not in TRAINING_ROLES:
        raise ValueError("merged training indices must share one training role")
    if len(asset_hashes) != 1 or not is_sha256(next(iter(asset_hashes))):
        raise ValueError("merged training indices must share one asset-lock hash")
    role = next(iter(roles))
    asset_hash = next(iter(asset_hashes))
    tiles: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    source_hashes_by_tile: dict[str, Any] = {}
    component_records: list[dict[str, str]] = []
    seen_tiles: set[str] = set()
    seen_paths: set[str] = set()
    seen_windows: set[tuple[str, int, int]] = set()
    for payload, component_sha256 in components:
        if payload.get("schema") != TRAINING_INDEX_SCHEMA:
            raise ValueError("training index component has the wrong schema")
        if (
            payload.get("patch_size") != PATCH_SIZE
            or payload.get("overlap_fraction") != PATCH_OVERLAP
            or payload.get("stride") != int(PATCH_SIZE * (1.0 - PATCH_OVERLAP))
        ):
            raise ValueError("training index component has a different patch protocol")
        raw_tiles = payload.get("tiles")
        raw_samples = payload.get("samples")
        if not isinstance(raw_tiles, list) or not isinstance(raw_samples, list):
            raise ValueError("training index component lacks tiles or samples")
        expected_hash = training_index_sample_set_sha256(
            role=role,
            asset_lock_sha256=asset_hash,
            tiles=raw_tiles,
            samples=raw_samples,
        )
        if payload.get("sample_set_sha256") != expected_hash:
            raise ValueError("training index component sample-set hash mismatch")
        if not is_sha256(component_sha256):
            raise ValueError("training index component file hash is invalid")
        component_records.append(
            {
                "file_sha256": component_sha256,
                "sample_set_sha256": expected_hash,
            }
        )
        component_tile_ids: list[str] = []
        for raw_tile in raw_tiles:
            tile = dict(raw_tile)
            tile_id = str(tile.get("tile_id", ""))
            if not tile_id or tile_id in seen_tiles:
                raise ValueError(f"duplicate or empty tile in merged index: {tile_id}")
            seen_tiles.add(tile_id)
            component_tile_ids.append(tile_id)
            tiles.append(tile)
        source_hashes = payload.get("source_hashes")
        if source_hashes is not None:
            if len(component_tile_ids) != 1 or not isinstance(source_hashes, Mapping):
                raise ValueError("per-tile source hashes require a single-tile component")
            converted_sources = {str(name): str(digest) for name, digest in source_hashes.items()}
            if REQUIRED_TRAINING_SOURCE_HASHES.difference(converted_sources) or any(
                not is_sha256(digest) for digest in converted_sources.values()
            ):
                raise ValueError("training index component has incomplete source hashes")
            source_hashes_by_tile[component_tile_ids[0]] = converted_sources
        elif isinstance(payload.get("source_hashes_by_tile"), Mapping):
            for tile_id, hashes in payload["source_hashes_by_tile"].items():
                converted_sources = {
                    str(name): str(digest) for name, digest in dict(hashes).items()
                }
                if REQUIRED_TRAINING_SOURCE_HASHES.difference(converted_sources) or any(
                    not is_sha256(digest) for digest in converted_sources.values()
                ):
                    raise ValueError("training index component has incomplete source hashes")
                source_hashes_by_tile[str(tile_id)] = converted_sources
        else:
            raise ValueError("training index component lacks source hashes")
        for raw_sample in raw_samples:
            sample = dict(raw_sample)
            path = str(sample.get("relative_path", ""))
            window = (
                str(sample.get("tile_id", "")),
                int(sample.get("row_start", -1)),
                int(sample.get("column_start", -1)),
            )
            if not path or path in seen_paths or window in seen_windows:
                raise ValueError("merged training index contains a duplicate path or window")
            seen_paths.add(path)
            seen_windows.add(window)
            samples.append(sample)
    tiles.sort(key=lambda value: str(value["tile_id"]))
    samples.sort(
        key=lambda value: (
            str(value["tile_id"]),
            int(value["row_start"]),
            int(value["column_start"]),
        )
    )
    component_records.sort(key=lambda value: value["file_sha256"])
    payload = {
        "schema": TRAINING_INDEX_SCHEMA,
        "role": role,
        "asset_lock_sha256": asset_hash,
        "patch_size": PATCH_SIZE,
        "overlap_fraction": PATCH_OVERLAP,
        "stride": int(PATCH_SIZE * (1.0 - PATCH_OVERLAP)),
        "tiles": tiles,
        "source_hashes_by_tile": dict(sorted(source_hashes_by_tile.items())),
        "components": component_records,
        "samples": samples,
    }
    if set(source_hashes_by_tile) != seen_tiles:
        raise ValueError("merged source hashes do not close over the tile set")
    payload["sample_set_sha256"] = training_index_sample_set_sha256(
        role=role,
        asset_lock_sha256=asset_hash,
        tiles=tiles,
        samples=samples,
    )
    output_path = write_canonical_json(output, payload, paper2_root=paper2_root)
    return {**payload, "index_file_sha256": file_sha256(output_path)}


def verify_training_patch_derivation(
    *,
    problem: ReconstructionProblem,
    plan: PlanContract,
    target: np.ndarray,
    target_valid_mask: np.ndarray,
    zf: np.ndarray,
    omega: np.ndarray,
    normalization: Mapping[str, float],
    role: str,
    geographic_group_id: str,
    index_payload: Mapping[str, Any],
    paper2_root: str | Path,
) -> dict[str, Any]:
    """Re-derive every indexed patch from its frozen sources and compare arrays.

    This is the formal G2/G3 evidence bridge.  Index hashes alone prove byte
    stability, but not that a caller did not fabricate a hash-consistent NPZ.
    Recomputing the same deterministic preparation expressions closes that gap.
    """

    if role not in TRAINING_ROLES:
        raise PermissionError("patch derivation verification accepts training roles only")
    if index_payload.get("schema") != TRAINING_INDEX_SCHEMA or index_payload.get(
        "role"
    ) != role:
        raise ValueError("training index role/schema differs from the derivation request")
    shape = problem.z0.shape
    truth = np.asarray(target, dtype=np.float64)
    truth_mask = np.asarray(target_valid_mask, dtype=bool)
    fused = np.asarray(zf, dtype=np.float64)
    uncertainty = np.asarray(omega, dtype=np.float64)
    if any(value.shape != shape for value in (truth, truth_mask, fused, uncertainty)):
        raise ValueError("patch derivation sources do not align with the problem grid")
    train_valid = (
        problem.valid_mask
        & truth_mask
        & np.isfinite(truth)
        & np.isfinite(problem.z0)
        & np.isfinite(fused)
        & np.isfinite(uncertainty)
    )
    if not np.any(train_valid):
        raise ValueError("patch derivation sources contain no valid training pixels")
    elevation_center = float(normalization["elevation_center"])
    elevation_scale = float(normalization["elevation_scale"])
    if not np.isfinite(elevation_center) or not np.isfinite(elevation_scale) or elevation_scale <= 0:
        raise ValueError("patch derivation normalization is invalid")
    target_filled = _nearest_fill(truth, train_valid, fallback=elevation_center)
    zf_filled = _nearest_fill(
        fused,
        problem.valid_mask & np.isfinite(fused),
        fallback=elevation_center,
    )
    condition = build_condition_planes(
        problem,
        plan,
        fused,
        uncertainty,
        method="OT-SB-full",
        normalization=normalization,
    )
    problem_hash = None
    raw_sources = index_payload.get("source_hashes_by_tile")
    if isinstance(raw_sources, Mapping):
        tile_sources = raw_sources.get(problem.tile_id)
        if isinstance(tile_sources, Mapping):
            problem_hash = tile_sources.get("problem_sha256")
    elif index_payload.get("tile_id") == problem.tile_id:
        tile_sources = index_payload.get("source_hashes")
        if isinstance(tile_sources, Mapping):
            problem_hash = tile_sources.get("problem_sha256")
    if not is_sha256(str(problem_hash or "")):
        raise ValueError("training index lacks the tile's problem hash")
    shifted = shifted_omega_field(uncertainty, tile_hash=str(problem_hash))
    target_normalized = ((target_filled - elevation_center) / elevation_scale).astype(np.float32)
    zf_normalized = ((zf_filled - elevation_center) / elevation_scale).astype(np.float32)
    tile_contracts = [
        item for item in index_payload.get("tiles", []) if item.get("tile_id") == problem.tile_id
    ]
    if len(tile_contracts) != 1 or tile_contracts[0].get("geographic_group_id") != (
        geographic_group_id
    ) or tile_contracts[0].get("shape") != list(shape):
        raise ValueError("training index tile contract differs from the re-derived tile")
    samples = [
        item for item in index_payload.get("samples", []) if item.get("tile_id") == problem.tile_id
    ]
    sample_by_window = {
        (int(item["row_start"]), int(item["column_start"])): item for item in samples
    }
    expected_windows = {
        (row, column)
        for row in sliding_window_starts(shape[0])
        for column in sliding_window_starts(shape[1])
    }
    if set(sample_by_window) != expected_windows or len(samples) != len(expected_windows):
        raise ValueError("training index windows differ from the deterministic patch layout")
    root = Path(paper2_root).resolve()
    verified: list[dict[str, Any]] = []
    for row, column in sorted(expected_windows):
        record = sample_by_window[(row, column)]
        if record.get("role") != role or record.get("geographic_group_id") != geographic_group_id:
            raise ValueError("training patch role/group differs from its frozen tile")
        path = guard_paper2_path(str(record["relative_path"]), paper2_root=root)
        if not path.is_file() or file_sha256(path) != record.get("sha256"):
            raise ValueError("training patch bytes differ from the index")
        rows = slice(row, row + PATCH_SIZE)
        columns = slice(column, column + PATCH_SIZE)
        expected_arrays = {
            "target": target_normalized[rows, columns],
            "zf": zf_normalized[rows, columns],
            "omega": uncertainty[rows, columns].astype(np.float32),
            "shifted_omega": shifted[rows, columns].astype(np.float32),
            "condition": condition[:, rows, columns],
            "valid_mask": train_valid[rows, columns],
        }
        with np.load(path, allow_pickle=False) as archive:
            expected_fields = {
                "tile_id",
                "geographic_group_id",
                "role",
                *expected_arrays,
            }
            if set(archive.files) != expected_fields:
                raise ValueError("training patch fields differ from deterministic preparation")
            identities = {
                "tile_id": problem.tile_id,
                "geographic_group_id": geographic_group_id,
                "role": role,
            }
            for name, expected in identities.items():
                value = np.asarray(archive[name])
                if value.ndim != 0 or str(value.item()) != expected:
                    raise ValueError(f"training patch {name} differs from its source")
            for name, expected in expected_arrays.items():
                observed = np.asarray(archive[name])
                if observed.dtype != expected.dtype or not np.array_equal(observed, expected):
                    raise ValueError(f"training patch {name} was not derived from its locked source")
        verified.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": str(record["sha256"]),
                "row_start": row,
                "column_start": column,
            }
        )
    return {
        "tile_id": problem.tile_id,
        "geographic_group_id": geographic_group_id,
        "role": role,
        "shape": list(shape),
        "sample_count": len(verified),
        "sample_derivation_sha256": canonical_sha256(verified),
    }

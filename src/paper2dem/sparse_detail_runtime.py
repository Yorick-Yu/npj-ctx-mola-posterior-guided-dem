from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from paper2dem.sparse_sampling import (
    deterministic_random_order,
    greedy_a_optimal_order,
    greedy_mutual_information_order,
    max_prior_variance_order,
    uniform_farthest_order,
)


DETAIL_BUDGETS = (5, 10, 20, 40, 80)
DETAIL_SELECTORS = (
    "random-00",
    "random-01",
    "random-02",
    "uniform-farthest",
    "max-prior-variance",
    "greedy-mi-point",
    "terrain-faopt",
)
SELECTOR_FAMILIES = (
    "random",
    "uniform-farthest",
    "max-prior-variance",
    "greedy-mi-point",
    "terrain-faopt",
)
PRIOR_LENGTHSCALE_PX = 96.0
PRIOR_NOISE_M = 8.0


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    return int.from_bytes(
        hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big"
    )


def sample_raster(field: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    return map_coordinates(field, [rows, columns], order=1, mode="nearest")


@dataclass
class SparseDetailCase:
    tile_id: str
    region: str
    role: str
    path: Path
    arrays: dict[str, np.ndarray]


def load_case(path: str | Path, *, tile_id: str, region: str, role: str) -> SparseDetailCase:
    case_path = Path(path)
    required = (
        "image",
        "z0",
        "dem",
        "valid",
        "row",
        "column",
        "elevation",
        "orbit",
        "assimilation",
        "evaluation",
    )
    with np.load(case_path, allow_pickle=False) as archive:
        missing = sorted(set(required) - set(archive.files))
        if missing:
            raise ValueError(f"{tile_id} case is missing arrays: {missing}")
        arrays = {key: np.asarray(archive[key]) for key in required}
    shape = arrays["z0"].shape
    if shape != (504, 504) or arrays["image"].shape != shape or arrays["dem"].shape != shape:
        raise ValueError(f"{tile_id} has unsupported raster shape {shape}")
    if arrays["valid"].shape != shape:
        raise ValueError(f"{tile_id} valid mask differs from raster shape")
    return SparseDetailCase(str(tile_id), str(region), str(role), case_path, arrays)


def _filled(field: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if not np.any(mask):
        raise ValueError("terrain prior requires finite valid pixels")
    return np.where(mask, values, float(np.median(values[mask])))


def _robust_image(image: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = _filled(image, valid)
    selected = values[np.asarray(valid, dtype=bool)]
    low, high = np.quantile(selected, [0.02, 0.98])
    return np.clip((values - low) / max(float(high - low), 1.0e-6), 0.0, 1.0)


def terrain_prior_features(
    image: np.ndarray, z0: np.ndarray, valid: np.ndarray, *, pixel_size_m: float = 20.0
) -> np.ndarray:
    """Observable-only terrain descriptors used by the train-frozen selector prior."""

    mask = np.asarray(valid, dtype=bool)
    terrain = _filled(z0, mask)
    normalized_image = _robust_image(image, mask)
    image_small = gaussian_filter(normalized_image, 2.0, mode="nearest")
    image_large = gaussian_filter(normalized_image, 8.0, mode="nearest")
    image_y, image_x = np.gradient(image_small)
    terrain_small = gaussian_filter(terrain, 2.0, mode="nearest")
    terrain_large = gaussian_filter(terrain, 8.0, mode="nearest")
    terrain_y, terrain_x = np.gradient(terrain_small, pixel_size_m)
    local_relief = np.sqrt(
        np.maximum(
            gaussian_filter((terrain - terrain_large) ** 2, 8.0, mode="nearest"),
            0.0,
        )
    )
    return np.stack(
        (
            np.hypot(terrain_x, terrain_y),
            np.abs(terrain - terrain_small),
            local_relief,
            np.hypot(image_x, image_y),
            np.abs(image_small - image_large),
            np.abs(gaussian_filter(image_small, 1.0) - gaussian_filter(image_small, 4.0)),
        ),
        axis=-1,
    ).astype(np.float32)


def terrain_prior_target(case: SparseDetailCase) -> np.ndarray:
    valid = np.asarray(case.arrays["valid"], dtype=bool)
    residual = _filled(case.arrays["dem"] - case.arrays["z0"], valid)
    residual -= float(np.median(residual[valid]))
    local = residual - gaussian_filter(residual, 8.0, mode="nearest")
    return np.abs(local).astype(np.float32)


def fit_terrain_prior(
    cases: Sequence[SparseDetailCase],
    *,
    samples_per_tile: int = 2048,
    ridge: float = 0.1,
    seed: int = 20260810,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("terrain prior requires training cases")
    feature_rows = []
    targets = []
    sample_counts = {}
    for case in cases:
        valid = np.asarray(case.arrays["valid"], dtype=bool)
        features = terrain_prior_features(
            case.arrays["image"], case.arrays["z0"], valid
        )
        target = terrain_prior_target(case)
        available = np.flatnonzero(valid & np.all(np.isfinite(features), axis=-1))
        count = min(int(samples_per_tile), int(available.size))
        generator = np.random.default_rng(stable_seed(seed, case.tile_id, "prior"))
        chosen = generator.choice(available, size=count, replace=False)
        feature_rows.append(features.reshape(-1, features.shape[-1])[chosen])
        targets.append(np.log1p(target.ravel()[chosen]))
        sample_counts[case.tile_id] = count
    x = np.concatenate(feature_rows).astype(np.float64)
    y = np.concatenate(targets).astype(np.float64)
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1.0e-6] = 1.0
    normalized = (x - mean) / scale
    design = np.column_stack((np.ones(len(normalized)), normalized))
    penalty = np.eye(design.shape[1]) * float(ridge)
    penalty[0, 0] = 0.0
    coefficient = np.linalg.solve(
        design.T @ design + penalty, design.T @ y
    )
    predicted = design @ coefficient
    payload = {
        "schema": "mars-sparse-detail-terrain-prior/v1",
        "state": "frozen_from_development_training_only",
        "seed": int(seed),
        "ridge": float(ridge),
        "samples_per_tile": int(samples_per_tile),
        "training_tile_count": len(cases),
        "sample_count": int(len(y)),
        "feature_names": [
            "coarse_slope",
            "coarse_small_scale_relief",
            "coarse_large_scale_relief",
            "image_gradient",
            "image_local_contrast",
            "image_difference_of_scales",
        ],
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coefficient": coefficient.tolist(),
        "target_clip_m": float(np.quantile(np.expm1(y), 0.995)),
        "training_rmse_log1p": float(np.sqrt(np.mean((predicted - y) ** 2))),
        "training_correlation": float(np.corrcoef(predicted, y)[0, 1]),
        "sample_count_by_tile": sample_counts,
        "permitted_selector_inputs": "registered image and coarse DEM only",
        "validation_or_external_targets_used": False,
        "lengthscale_px": PRIOR_LENGTHSCALE_PX,
        "noise_m": PRIOR_NOISE_M,
        "finite_footprint_sigma_px": 1.0,
    }
    payload["content_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def apply_terrain_prior(case: SparseDetailCase, prior: Mapping[str, Any]) -> np.ndarray:
    features = terrain_prior_features(
        case.arrays["image"], case.arrays["z0"], case.arrays["valid"]
    ).astype(np.float64)
    mean = np.asarray(prior["feature_mean"], dtype=np.float64)
    scale = np.asarray(prior["feature_scale"], dtype=np.float64)
    coefficient = np.asarray(prior["coefficient"], dtype=np.float64)
    normalized = (features - mean) / scale
    design = np.concatenate(
        (np.ones((*normalized.shape[:-1], 1), dtype=np.float64), normalized), axis=-1
    )
    predicted = np.expm1(np.tensordot(design, coefficient, axes=([-1], [0])))
    return np.clip(predicted, 0.0, float(prior["target_clip_m"])).astype(np.float32)


def _candidate_ids(case: SparseDetailCase, indices: np.ndarray) -> list[str]:
    return [
        f"{case.tile_id}|{int(index)}|{int(case.arrays['orbit'][index])}"
        for index in indices
    ]


def _target_grid(valid: np.ndarray, maximum: int = 1024) -> np.ndarray:
    rows, columns = np.nonzero(valid)
    if rows.size <= maximum:
        return np.column_stack((rows, columns)).astype(np.float64)
    positions = np.linspace(0, rows.size - 1, maximum, dtype=np.int64)
    return np.column_stack((rows[positions], columns[positions])).astype(np.float64)


def build_selector_orders(
    case: SparseDetailCase,
    prior: Mapping[str, Any],
    *,
    maximum_budget: int = 80,
) -> dict[str, np.ndarray]:
    context_indices = np.flatnonzero(case.arrays["assimilation"]).astype(np.int64)
    if context_indices.size < min(5, maximum_budget):
        raise ValueError(f"{case.tile_id} has too few assimilation candidates")
    maximum = min(int(maximum_budget), int(context_indices.size))
    coordinates = np.column_stack(
        (case.arrays["row"][context_indices], case.arrays["column"][context_indices])
    ).astype(np.float64)
    identifiers = _candidate_ids(case, context_indices)
    prior_raster = apply_terrain_prior(case, prior)
    candidate_information = sample_raster(
        prior_raster, coordinates[:, 0], coordinates[:, 1]
    )
    amplitude = 5.0 + candidate_information.astype(np.float64)
    lengthscale = float(prior["lengthscale_px"])
    noise_variance = float(prior["noise_m"]) ** 2
    squared = np.sum(
        (coordinates[:, None, :] - coordinates[None, :, :]) ** 2, axis=2
    )
    candidate_covariance = np.outer(amplitude, amplitude) * np.exp(
        -0.5 * squared / lengthscale**2
    )
    candidate_covariance.flat[:: len(candidate_covariance) + 1] += 1.0e-8
    target_coordinates = _target_grid(np.asarray(case.arrays["valid"], dtype=bool))
    target_information = sample_raster(
        prior_raster, target_coordinates[:, 0], target_coordinates[:, 1]
    ).astype(np.float64)
    target_amplitude = 5.0 + target_information
    target_squared = np.sum(
        (target_coordinates[:, None, :] - coordinates[None, :, :]) ** 2, axis=2
    )
    target_cross = np.outer(target_amplitude, amplitude) * np.exp(
        -0.5 * target_squared / lengthscale**2
    )
    target_weights = np.maximum(target_information, 0.05)

    relative_orders = {
        f"random-{replicate:02d}": deterministic_random_order(
            identifiers,
            stream=f"mars-sparse-detail-v30|{case.tile_id}|random-{replicate:02d}",
            maximum_budget=maximum,
        )
        for replicate in range(3)
    }
    relative_orders["uniform-farthest"] = uniform_farthest_order(
        coordinates, identifiers, maximum_budget=maximum
    )
    relative_orders["max-prior-variance"] = max_prior_variance_order(
        np.diag(candidate_covariance), identifiers, maximum_budget=maximum
    )
    relative_orders["greedy-mi-point"] = greedy_mutual_information_order(
        candidate_covariance,
        identifiers,
        noise_variance=noise_variance,
        maximum_budget=maximum,
    )
    relative_orders["terrain-faopt"] = greedy_a_optimal_order(
        target_cross,
        candidate_covariance,
        identifiers,
        noise_variance=noise_variance,
        maximum_budget=maximum,
        target_weights=target_weights,
    )
    output = {
        selector: context_indices[np.asarray(order, dtype=np.int64)]
        for selector, order in relative_orders.items()
    }
    for selector, order in output.items():
        if len(order) != maximum or len(set(map(int, order))) != maximum:
            raise ValueError(f"{case.tile_id} {selector} order is not a unique prefix")
    output["prior_information_raster"] = prior_raster
    return output


def selector_for_family(
    family: str, *, rng: np.random.Generator
) -> str:
    if family == "random":
        return f"random-{int(rng.integers(3)):02d}"
    if family not in DETAIL_SELECTORS:
        raise ValueError(f"unknown selector family: {family}")
    return family


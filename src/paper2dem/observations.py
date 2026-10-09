from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from .contracts import LaserObservations, PlanContract
from .footprints import build_footprint_operator_with_metadata, project_footprints
from .hashing import PAPER2_ROOT, canonical_sha256


def _nearest_valid_fill(dem: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    usable = valid_mask & np.isfinite(dem)
    if not np.any(usable):
        raise ValueError("sealed reference contains no finite valid pixels")
    if np.all(usable):
        return np.asarray(dem, dtype=np.float64)
    nearest = ndimage.distance_transform_edt(
        ~usable,
        return_distances=False,
        return_indices=True,
    )
    return np.asarray(dem, dtype=np.float64)[tuple(nearest)]


def _paired_noise(
    *,
    seed: int,
    replicate: int,
    tile_id: str,
    candidate_id: str,
    standard_deviation: float,
) -> float:
    if standard_deviation == 0:
        return 0.0
    digest = canonical_sha256(
        {
            "protocol": "paper2-candidate-paired-noise-v1",
            "seed": int(seed),
            "replicate": int(replicate),
            "tile_id": tile_id,
            "candidate_id": candidate_id,
        }
    )
    generator = np.random.Generator(np.random.PCG64(int(digest[:16], 16)))
    return float(generator.normal(0.0, standard_deviation))


def simulate_laser_observations(
    plan: PlanContract | str | Path,
    reference_path: str | Path,
    *,
    replicate: int,
    seed: int,
) -> LaserObservations:
    """Sealed reference reader; its return object contains no truth path or clean values."""
    if not isinstance(plan, PlanContract):
        plan = PlanContract.read(plan)
    plan.validate()
    if replicate < 0:
        raise ValueError("replicate must be nonnegative")
    with np.load(reference_path, allow_pickle=False) as archive:
        required = {"dem", "valid_mask", "transform"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"reference archive is missing fields: {missing}")
        if "tile_id" in archive.files and str(archive["tile_id"].item()) != plan.tile_id:
            raise ValueError("plan and sealed reference tile IDs do not match")
        if "crs_wkt" in archive.files and str(archive["crs_wkt"].item()) != plan.crs_wkt:
            raise ValueError("plan and sealed reference CRS do not match")
        dem = np.asarray(archive["dem"], dtype=np.float64)
        valid_mask = np.asarray(archive["valid_mask"], dtype=bool)
        transform = tuple(float(item) for item in archive["transform"])
    if dem.ndim != 2 or valid_mask.shape != dem.shape:
        raise ValueError("sealed reference DEM and mask must be aligned 2-D arrays")
    if transform != plan.transform:
        raise ValueError("plan and sealed reference affine transforms do not match")
    centers = np.asarray([[point.x, point.y] for point in plan.points], dtype=np.float64).reshape(-1, 2)
    support_mask = valid_mask & np.isfinite(dem)
    footprint = build_footprint_operator_with_metadata(
        centers,
        grid_shape=dem.shape,
        transform=transform,
        diameter_m=0.0 if plan.footprint.shape == "point" else plan.footprint.diameter_m,
        quadrature_points=plan.footprint.quadrature_points,
        valid_mask=support_mask,
        require_full_support=plan.footprint.require_full_support,
        drop_invalid=True,
    )
    clean_values = project_footprints(footprint.operator, _nearest_valid_fill(dem, support_mask))
    values = np.full(plan.budget, np.nan, dtype=np.float64)
    flags = np.full(
        plan.budget,
        1 if plan.footprint.require_full_support else 2,
        dtype=np.uint8,
    )
    for row, index in enumerate(footprint.kept_indices.tolist()):
        point = plan.points[index]
        clean_average = float(clean_values[row])
        if not np.isfinite(clean_average):
            flags[index] = 2
            continue
        flags[index] = 0
        values[index] = clean_average + _paired_noise(
            seed=seed,
            replicate=replicate,
            tile_id=plan.tile_id,
            candidate_id=point.candidate_id,
            standard_deviation=plan.noise_std_m,
        )
    result = LaserObservations(
        tile_id=plan.tile_id,
        plan_sha256=plan.sha256,
        replicate=replicate,
        candidate_ids=np.asarray([point.candidate_id for point in plan.points]),
        coordinates=centers,
        return_values_m=values,
        quality_flags=flags,
        noise_repeat_numbers=np.full(plan.budget, replicate, dtype=np.int32),
        noise_std_m=plan.noise_std_m,
    )
    result.validate()
    return result


def simulate_and_write_laser_observations(
    plan: PlanContract | str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    *,
    replicate: int,
    seed: int,
    paper2_root: str | Path = PAPER2_ROOT,
) -> Path:
    observations = simulate_laser_observations(
        plan,
        reference_path,
        replicate=replicate,
        seed=seed,
    )
    return observations.write(output_path, paper2_root=paper2_root)

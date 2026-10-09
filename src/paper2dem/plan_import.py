from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np

from .contracts import (
    FootprintContract,
    Paper1SourceLock,
    PlanContract,
    PlanPoint,
)
from .hashing import PAPER2_ROOT, canonical_sha256, file_sha256, guard_paper2_path


DEFAULT_BUDGETS = (0, 5, 10, 20, 50, 100)


def _load_selection(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"tile_id", "method", "budget", "records"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"selection is missing required fields: {missing}")
    records = payload["records"]
    if not isinstance(records, list) or len(records) != int(payload["budget"]):
        raise ValueError("selection records must be a list aligned with budget")
    if payload.get("metadata", {}).get("selection_reference_accessed") is True:
        raise ValueError("Paper1 selection reports reference access and cannot be imported")
    return payload


def _record_identity(record: dict[str, Any]) -> tuple[Any, ...]:
    required = (
        "rank",
        "candidate_id",
        "track_id",
        "projected_x",
        "projected_y",
        "constraints_satisfied",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"selection record is missing fields: {missing}")
    return (
        int(record["rank"]),
        str(record["candidate_id"]),
        str(record["track_id"]),
        float(record["projected_x"]),
        float(record["projected_y"]),
        bool(record["constraints_satisfied"]),
    )


def validate_strict_prefix(
    kmax_selection: dict[str, Any] | str | Path,
    prefix_selection: dict[str, Any] | str | Path,
) -> None:
    full = _load_selection(kmax_selection) if isinstance(kmax_selection, (str, Path)) else kmax_selection
    prefix = _load_selection(prefix_selection) if isinstance(prefix_selection, (str, Path)) else prefix_selection
    if prefix["tile_id"] != full["tile_id"] or prefix["method"] != full["method"]:
        raise ValueError("prefix tile and method must match Kmax")
    prefix_budget = int(prefix["budget"])
    if prefix_budget <= 0 or prefix_budget >= int(full["budget"]):
        raise ValueError("strict prefix budget must be positive and smaller than Kmax")
    expected = [_record_identity(record) for record in full["records"][:prefix_budget]]
    observed = [_record_identity(record) for record in prefix["records"]]
    if observed != expected:
        raise ValueError(f"K={prefix_budget} selection is not a strict prefix of Kmax")


def _problem_fields(
    path: str | Path,
    *,
    footprint_override: FootprintContract | None = None,
    noise_std_m_override: float | None = None,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        base_required = {
            "tile_id",
            "scenario_id",
            "transform",
            "crs_wkt",
            "candidates",
            "candidate_ids",
            "candidate_track_ids",
        }
        measurement_fields = {
            "footprint_shape",
            "footprint_diameter_m",
            "footprint_quadrature_points",
            "footprint_noise_std_m",
            "footprint_require_full_support",
        }
        missing = sorted(base_required.difference(archive.files))
        if missing:
            raise ValueError(f"Paper1 problem is missing required fields: {missing}")
        present_measurement_fields = measurement_fields.intersection(archive.files)
        if present_measurement_fields and present_measurement_fields != measurement_fields:
            missing_measurement = sorted(measurement_fields.difference(archive.files))
            raise ValueError(
                f"Paper1 problem has a partial measurement contract: {missing_measurement}"
            )
        if present_measurement_fields:
            if footprint_override is not None or noise_std_m_override is not None:
                raise ValueError("measurement overrides are forbidden when the problem is complete")
            footprint = FootprintContract(
                shape=str(archive["footprint_shape"].item()),
                diameter_m=float(archive["footprint_diameter_m"].item()),
                quadrature_points=int(archive["footprint_quadrature_points"].item()),
                require_full_support=bool(archive["footprint_require_full_support"].item()),
            )
            noise_std_m = float(archive["footprint_noise_std_m"].item())
        else:
            if footprint_override is None or noise_std_m_override is None:
                raise ValueError(
                    "Paper1 problem lacks a measurement contract; explicit footprint and noise "
                    "overrides are required"
                )
            footprint_override.validate()
            if not np.isfinite(noise_std_m_override) or noise_std_m_override < 0:
                raise ValueError("noise_std_m_override must be finite and nonnegative")
            footprint = footprint_override
            noise_std_m = float(noise_std_m_override)
        candidates = np.asarray(archive["candidates"], dtype=np.float64)
        candidate_ids = np.asarray(archive["candidate_ids"]).astype(str)
        track_ids = np.asarray(archive["candidate_track_ids"]).astype(str)
        if candidates.ndim != 2 or candidates.shape[1] != 2:
            raise ValueError("Paper1 candidates must have shape (n, 2)")
        if candidate_ids.shape != (candidates.shape[0],) or track_ids.shape != candidate_ids.shape:
            raise ValueError("Paper1 candidate identities must align with geometry")
        geometry = [
            {
                "candidate_id": str(candidate_ids[index]),
                "track_id": str(track_ids[index]),
                "x_hex": float(candidates[index, 0]).hex(),
                "y_hex": float(candidates[index, 1]).hex(),
            }
            for index in range(candidates.shape[0])
        ]
        return {
            "tile_id": str(archive["tile_id"].item()),
            "scenario_id": str(archive["scenario_id"].item()),
            "transform": tuple(float(item) for item in archive["transform"]),
            "crs_wkt": str(archive["crs_wkt"].item()),
            "candidates": candidates,
            "candidate_ids": candidate_ids,
            "track_ids": track_ids,
            "candidate_geometry_sha256": canonical_sha256(geometry),
            "footprint": footprint,
            "noise_std_m": noise_std_m,
        }


def load_candidate_problem(
    path: str | Path,
    *,
    footprint_override: FootprintContract | None = None,
    noise_std_m_override: float | None = None,
) -> dict[str, Any]:
    """Load the truth-free candidate and measurement fields used by plan contracts."""

    return _problem_fields(
        path,
        footprint_override=footprint_override,
        noise_std_m_override=noise_std_m_override,
    )


def import_paper1_kmax_plan(
    selection_path: str | Path,
    problem_path: str | Path,
    *,
    paper1_lock: Paper1SourceLock,
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    prefix_selection_paths: dict[int, str | Path] | None = None,
    footprint_override: FootprintContract | None = None,
    noise_std_m_override: float | None = None,
) -> PlanContract:
    """Rebuild an allowlisted truth-free Kmax plan from Paper1 artifacts."""
    selection = _load_selection(selection_path)
    problem = load_candidate_problem(
        problem_path,
        footprint_override=footprint_override,
        noise_std_m_override=noise_std_m_override,
    )
    budgets = tuple(sorted(set(int(value) for value in budgets)))
    if not budgets or budgets[0] < 0:
        raise ValueError("budgets must contain non-negative integers")
    if int(selection["budget"]) != budgets[-1]:
        raise ValueError("the imported selection must be the requested Kmax plan")
    if selection["tile_id"] != problem["tile_id"]:
        raise ValueError("selection and problem tile IDs do not match")
    prefix_selection_paths = {} if prefix_selection_paths is None else prefix_selection_paths
    for budget, path in prefix_selection_paths.items():
        if int(budget) == 0 or int(budget) not in budgets[:-1]:
            raise ValueError(f"unexpected prefix budget: {budget}")
        validate_strict_prefix(selection, _load_selection(path))
    missing_prefixes = set(budgets[:-1]).difference({0}).difference(
        int(item) for item in prefix_selection_paths
    )
    if prefix_selection_paths and missing_prefixes:
        raise ValueError(f"prefix audit is incomplete for budgets: {sorted(missing_prefixes)}")
    candidate_index = {
        str(candidate_id): index for index, candidate_id in enumerate(problem["candidate_ids"])
    }
    points: list[PlanPoint] = []
    for expected_rank, source in enumerate(selection["records"], start=1):
        rank, candidate_id, track_id, x, y, constraints_satisfied = _record_identity(source)
        if rank != expected_rank:
            raise ValueError("selection ranks must be ordered and contiguous")
        if not constraints_satisfied:
            raise ValueError(f"selection rank {rank} violates a locked constraint")
        if candidate_id not in candidate_index:
            raise ValueError(f"selection contains unknown candidate: {candidate_id}")
        index = candidate_index[candidate_id]
        expected_track = str(problem["track_ids"][index])
        if track_id != expected_track:
            raise ValueError(f"candidate {candidate_id} has a mismatched track ID")
        expected_xy = problem["candidates"][index]
        if not np.allclose([x, y], expected_xy, rtol=0.0, atol=1e-7):
            raise ValueError(f"candidate {candidate_id} coordinates do not match problem geometry")
        points.append(
            PlanPoint(
                rank=rank,
                candidate_id=candidate_id,
                track_id=track_id,
                x=x,
                y=y,
            )
        )
    plan = PlanContract(
        tile_id=problem["tile_id"],
        scenario_id=problem["scenario_id"],
        method=str(selection["method"]),
        budget=int(selection["budget"]),
        crs_wkt=problem["crs_wkt"],
        transform=problem["transform"],
        footprint=problem["footprint"],
        noise_std_m=problem["noise_std_m"],
        candidate_geometry_sha256=problem["candidate_geometry_sha256"],
        problem_sha256=file_sha256(problem_path),
        paper1=paper1_lock,
        source_selection_sha256=file_sha256(selection_path),
        points=tuple(points),
    )
    plan.validate()
    return plan


def write_plan_family(
    kmax_plan: PlanContract,
    output_directory: str | Path,
    *,
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    paper2_root: str | Path = PAPER2_ROOT,
) -> tuple[Path, ...]:
    kmax_plan.validate()
    output = guard_paper2_path(output_directory, paper2_root=paper2_root, allow_root=True)
    if output == Path(paper2_root).resolve():
        raise ValueError("plan family must be written below the paper2 root")
    budgets = tuple(sorted(set(int(item) for item in budgets)))
    if not budgets or budgets[-1] != kmax_plan.budget:
        raise ValueError("plan family budgets must end at Kmax")
    written: list[Path] = []
    for budget in budgets:
        plan = kmax_plan.prefix(budget)
        written.append(
            plan.write(output / f"k{budget:03d}" / "plan.json", paper2_root=paper2_root)
        )
    return tuple(written)

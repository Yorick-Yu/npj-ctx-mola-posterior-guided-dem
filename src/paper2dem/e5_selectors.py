from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence
import json

import numpy as np

from .contracts import PlanContract, PlanPoint
from .hashing import (
    PAPER2_ROOT,
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    is_sha256,
    write_canonical_json,
    write_immutable_bytes,
)
from .plan_import import load_candidate_problem, write_plan_family


E5_BUDGETS = (5, 20, 100)
E5_RANDOM_INSTANCES = tuple(f"random-{index:02d}" for index in range(10))
E5_SELECTOR_INSTANCES = (*E5_RANDOM_INSTANCES, "uniform", "max-variance", "FAOPT")
E5_RECONSTRUCTORS = ("gp-footprint", "Iso-SB-cond", "OT-SB-full")
E5_SHARED_SEED_LABELS = 3
E5_NOISE_REPLICATES = 5
E5_BRIDGE_TILES = 12
E5_EXPECTED_CELLS = (
    len(E5_SELECTOR_INSTANCES)
    * len(E5_RECONSTRUCTORS)
    * E5_SHARED_SEED_LABELS
    * len(E5_BUDGETS)
    * E5_BRIDGE_TILES
    * E5_NOISE_REPLICATES
)

if E5_EXPECTED_CELLS != 21_060:
    raise AssertionError("the preregistered E5 expected-matrix must contain 21,060 cells")


def _readonly(values: Any, *, dtype: Any) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _guard_input(path: str | Path, root: Path) -> Path:
    resolved = guard_paper2_path(path, paper2_root=root)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _manifest_content_sha256(payload: Mapping[str, Any]) -> str:
    unhashed = dict(payload)
    unhashed.pop("content_sha256", None)
    return canonical_sha256(unhashed)


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    location: str,
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected.difference(observed))
        extra = sorted(observed.difference(expected))
        raise ValueError(f"{location} keys differ; missing={missing}, extra={extra}")


@dataclass(frozen=True)
class E5PriorVariance:
    """Truth-free, candidate-aligned prior uncertainty used by max-variance."""

    SCHEMA: ClassVar[str] = "paper2.e5-prior-variance/v1"

    tile_id: str
    problem_sha256: str
    candidate_geometry_sha256: str
    source_model_sha256: str
    candidate_ids: np.ndarray
    prior_variance: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_ids", _readonly(self.candidate_ids, dtype=str))
        object.__setattr__(
            self,
            "prior_variance",
            _readonly(self.prior_variance, dtype=np.float64),
        )

    @property
    def candidate_ids_sha256(self) -> str:
        return canonical_sha256(self.candidate_ids.tolist())

    @property
    def prior_variance_sha256(self) -> str:
        return canonical_sha256([float(value).hex() for value in self.prior_variance])

    def validate(self) -> None:
        if not self.tile_id:
            raise ValueError("E5 prior-variance tile_id cannot be empty")
        for name, value in (
            ("problem", self.problem_sha256),
            ("candidate geometry", self.candidate_geometry_sha256),
            ("source model", self.source_model_sha256),
        ):
            if not is_sha256(value):
                raise ValueError(f"E5 prior-variance {name} hash is invalid")
        if self.candidate_ids.ndim != 1:
            raise ValueError("E5 prior-variance candidate_ids must be one-dimensional")
        if self.prior_variance.shape != self.candidate_ids.shape:
            raise ValueError("E5 prior variance must align one-to-one with candidate_ids")
        if len(self.candidate_ids) < E5_BUDGETS[-1]:
            raise ValueError("E5 prior variance must contain at least 100 candidates")
        if len(set(self.candidate_ids.tolist())) != len(self.candidate_ids):
            raise ValueError("E5 prior-variance candidate_ids must be unique")
        if not np.all(np.isfinite(self.prior_variance)) or np.any(self.prior_variance < 0.0):
            raise ValueError("E5 prior variance must be finite and nonnegative")

    @property
    def metadata(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.SCHEMA,
            "tile_id": self.tile_id,
            "problem_sha256": self.problem_sha256,
            "candidate_geometry_sha256": self.candidate_geometry_sha256,
            "source_model_sha256": self.source_model_sha256,
            "candidate_ids_sha256": self.candidate_ids_sha256,
            "prior_variance_sha256": self.prior_variance_sha256,
            "score_semantics": "frozen-prior-predictive-variance-at-candidate",
            "reference_accessed": False,
        }

    def write(
        self,
        path: str | Path,
        *,
        paper2_root: str | Path = PAPER2_ROOT,
    ) -> Path:
        stream = BytesIO()
        np.savez_compressed(
            stream,
            metadata=np.asarray(canonical_json_bytes(self.metadata).decode("utf-8")),
            candidate_ids=self.candidate_ids,
            prior_variance=self.prior_variance,
        )
        return write_immutable_bytes(path, stream.getvalue(), paper2_root=paper2_root)

    @classmethod
    def read(cls, path: str | Path) -> E5PriorVariance:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"metadata", "candidate_ids", "prior_variance"}:
                raise ValueError("E5 prior-variance archive contains unexpected fields")
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("schema") != cls.SCHEMA:
                raise ValueError(
                    f"unsupported E5 prior-variance schema: {metadata.get('schema')}"
                )
            _require_exact_keys(
                metadata,
                {
                    "schema",
                    "tile_id",
                    "problem_sha256",
                    "candidate_geometry_sha256",
                    "source_model_sha256",
                    "candidate_ids_sha256",
                    "prior_variance_sha256",
                    "score_semantics",
                    "reference_accessed",
                },
                location="E5 prior-variance metadata",
            )
            if metadata.get("reference_accessed") is not False:
                raise ValueError("E5 selector scores must be explicitly reference-free")
            result = cls(
                tile_id=str(metadata["tile_id"]),
                problem_sha256=str(metadata["problem_sha256"]),
                candidate_geometry_sha256=str(metadata["candidate_geometry_sha256"]),
                source_model_sha256=str(metadata["source_model_sha256"]),
                candidate_ids=archive["candidate_ids"],
                prior_variance=archive["prior_variance"],
            )
        result.validate()
        if metadata.get("candidate_ids_sha256") != result.candidate_ids_sha256:
            raise ValueError("E5 prior-variance candidate ID hash differs")
        if metadata.get("prior_variance_sha256") != result.prior_variance_sha256:
            raise ValueError("E5 prior-variance score hash differs")
        if metadata.get("score_semantics") != "frozen-prior-predictive-variance-at-candidate":
            raise ValueError("E5 prior-variance score semantics differ")
        return result


def write_e5_prior_variance_contract(
    *,
    problem_path: str | Path,
    prior_variance: Sequence[float] | np.ndarray,
    source_model_sha256: str,
    output_path: str | Path,
    paper2_root: str | Path = PAPER2_ROOT,
) -> Path:
    """Bind a frozen-prior score vector to one Paper2 candidate geometry."""

    root = Path(paper2_root).resolve()
    problem_path = _guard_input(problem_path, root)
    problem = load_candidate_problem(problem_path)
    contract = E5PriorVariance(
        tile_id=str(problem["tile_id"]),
        problem_sha256=file_sha256(problem_path),
        candidate_geometry_sha256=str(problem["candidate_geometry_sha256"]),
        source_model_sha256=source_model_sha256,
        candidate_ids=np.asarray(problem["candidate_ids"]).astype(str),
        prior_variance=np.asarray(prior_variance, dtype=np.float64),
    )
    return contract.write(output_path, paper2_root=root)


def _validate_problem(problem: Mapping[str, Any], *, problem_sha256: str) -> None:
    candidates = np.asarray(problem["candidates"], dtype=np.float64)
    candidate_ids = np.asarray(problem["candidate_ids"]).astype(str)
    track_ids = np.asarray(problem["track_ids"]).astype(str)
    if candidates.ndim != 2 or candidates.shape[1] != 2:
        raise ValueError("E5 candidate geometry must have shape (n, 2)")
    if candidates.shape[0] < E5_BUDGETS[-1]:
        raise ValueError("E5 requires at least 100 candidate points per bridge tile")
    if not np.all(np.isfinite(candidates)):
        raise ValueError("E5 candidate coordinates must be finite")
    if len(set(candidate_ids.tolist())) != len(candidate_ids):
        raise ValueError("E5 candidate IDs must be unique")
    if track_ids.shape != candidate_ids.shape:
        raise ValueError("E5 candidate track IDs must align with candidate IDs")
    if not is_sha256(problem_sha256):
        raise ValueError("E5 problem hash is invalid")


def _random_order(
    problem: Mapping[str, Any],
    *,
    selector_instance: str,
    master_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if selector_instance not in E5_RANDOM_INSTANCES:
        raise ValueError(f"unsupported E5 random selector instance: {selector_instance}")
    if master_seed < 0:
        raise ValueError("E5 random master seed must be nonnegative")
    candidate_ids = np.asarray(problem["candidate_ids"]).astype(str)
    stream = {
        "schema": "paper2.e5-random-stream/v1",
        "tile_id": str(problem["tile_id"]),
        "candidate_geometry_sha256": str(problem["candidate_geometry_sha256"]),
        "selector_instance": selector_instance,
        "master_seed": int(master_seed),
        "algorithm": "sha256-ranked-permutation",
    }
    stream_sha256 = canonical_sha256(stream)
    keyed = [
        (
            canonical_sha256(
                {
                    "stream_sha256": stream_sha256,
                    "candidate_id": str(candidate_id),
                }
            ),
            str(candidate_id),
            index,
        )
        for index, candidate_id in enumerate(candidate_ids)
    ]
    order = np.asarray([record[2] for record in sorted(keyed)], dtype=np.int64)
    return order, {**stream, "stream_sha256": stream_sha256}


def _uniform_farthest_order(problem: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    coordinates = np.asarray(problem["candidates"], dtype=np.float64)
    candidate_ids = np.asarray(problem["candidate_ids"]).astype(str)
    start_domain = {
        "schema": "paper2.e5-uniform-farthest/v1",
        "tile_id": str(problem["tile_id"]),
        "candidate_geometry_sha256": str(problem["candidate_geometry_sha256"]),
        "start_rule": "minimum-sha256-candidate-key",
    }
    start_sha256 = canonical_sha256(start_domain)
    start = min(
        range(len(candidate_ids)),
        key=lambda index: (
            canonical_sha256(
                {
                    "start_sha256": start_sha256,
                    "candidate_id": str(candidate_ids[index]),
                }
            ),
            str(candidate_ids[index]),
        ),
    )
    # One scalar normalization avoids overflow without changing Euclidean
    # distance ordering in the common projected CRS.
    span = float(np.max(np.ptp(coordinates, axis=0)))
    normalized = coordinates / (span if span > 0.0 else 1.0)
    selected = np.zeros(len(candidate_ids), dtype=bool)
    order: list[int] = [start]
    selected[start] = True
    min_distance = np.sum((normalized - normalized[start]) ** 2, axis=1)
    while len(order) < E5_BUDGETS[-1]:
        available = np.flatnonzero(~selected)
        best_distance = float(np.max(min_distance[available]))
        tolerance = max(1.0, abs(best_distance)) * 1e-14
        tied = available[np.abs(min_distance[available] - best_distance) <= tolerance]
        chosen = min(tied.tolist(), key=lambda index: str(candidate_ids[index]))
        order.append(chosen)
        selected[chosen] = True
        distance = np.sum((normalized - normalized[chosen]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, distance)
    provenance = {
        **start_domain,
        "start_sha256": start_sha256,
        "algorithm": "deterministic-maximin-euclidean",
        "tie_break": "candidate-id-ascending",
        "distance_normalization": "single-common-coordinate-span",
    }
    return np.asarray(order, dtype=np.int64), provenance


def _max_variance_order(
    problem: Mapping[str, Any],
    prior: E5PriorVariance,
    *,
    prior_file_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    candidate_ids = np.asarray(problem["candidate_ids"]).astype(str)
    order = np.asarray(
        sorted(
            range(len(candidate_ids)),
            key=lambda index: (-float(prior.prior_variance[index]), str(candidate_ids[index])),
        ),
        dtype=np.int64,
    )
    provenance = {
        "schema": "paper2.e5-max-prior-variance/v1",
        "tile_id": str(problem["tile_id"]),
        "candidate_geometry_sha256": str(problem["candidate_geometry_sha256"]),
        "prior_variance_file_sha256": prior_file_sha256,
        "prior_variance_sha256": prior.prior_variance_sha256,
        "algorithm": "descending-prior-variance",
        "tie_break": "candidate-id-ascending",
    }
    return order, provenance


def _validate_faopt(
    faopt: PlanContract,
    problem: Mapping[str, Any],
    *,
    problem_sha256: str,
) -> None:
    faopt.validate()
    if faopt.method.casefold() not in {"discrete-faopt", "faopt"}:
        raise ValueError("E5 FAOPT source plan must be a discrete-FAOPT plan")
    if faopt.budget != E5_BUDGETS[-1]:
        raise ValueError("E5 FAOPT source plan must be the Kmax=100 contract")
    if faopt.tile_id != problem["tile_id"] or faopt.scenario_id != problem["scenario_id"]:
        raise ValueError("E5 FAOPT source plan differs from the candidate problem identity")
    if faopt.problem_sha256 != problem_sha256:
        raise ValueError("E5 FAOPT source plan problem hash differs")
    if faopt.candidate_geometry_sha256 != problem["candidate_geometry_sha256"]:
        raise ValueError("E5 FAOPT source plan candidate geometry differs")
    if faopt.crs_wkt != problem["crs_wkt"] or not np.allclose(
        faopt.transform,
        problem["transform"],
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("E5 FAOPT source plan raster geometry differs")
    candidate_index = {
        str(candidate_id): index
        for index, candidate_id in enumerate(np.asarray(problem["candidate_ids"]).astype(str))
    }
    track_ids = np.asarray(problem["track_ids"]).astype(str)
    coordinates = np.asarray(problem["candidates"], dtype=np.float64)
    for point in faopt.points:
        if point.candidate_id not in candidate_index:
            raise ValueError("E5 FAOPT source plan contains an unknown candidate")
        index = candidate_index[point.candidate_id]
        if point.track_id != track_ids[index] or not np.allclose(
            (point.x, point.y), coordinates[index], rtol=0.0, atol=1e-7
        ):
            raise ValueError("E5 FAOPT source plan point differs from candidate geometry")


def _generated_plan(
    *,
    faopt: PlanContract,
    problem: Mapping[str, Any],
    problem_sha256: str,
    method: str,
    order: np.ndarray,
    source_selection_sha256: str,
) -> PlanContract:
    candidate_count = len(problem["candidate_ids"])
    if order.ndim != 1 or len(order) < E5_BUDGETS[-1]:
        raise ValueError("E5 selector order must contain at least 100 candidates")
    if (
        len(set(order.tolist())) != len(order)
        or int(np.min(order)) < 0
        or int(np.max(order)) >= candidate_count
    ):
        raise ValueError("E5 selector order contains duplicate or unknown candidate indices")
    candidates = np.asarray(problem["candidates"], dtype=np.float64)
    candidate_ids = np.asarray(problem["candidate_ids"]).astype(str)
    track_ids = np.asarray(problem["track_ids"]).astype(str)
    points = tuple(
        PlanPoint(
            rank=rank,
            candidate_id=str(candidate_ids[index]),
            track_id=str(track_ids[index]),
            x=float(candidates[index, 0]),
            y=float(candidates[index, 1]),
        )
        for rank, index in enumerate(order[: E5_BUDGETS[-1]], start=1)
    )
    plan = PlanContract(
        tile_id=str(problem["tile_id"]),
        scenario_id=str(problem["scenario_id"]),
        method=method,
        budget=E5_BUDGETS[-1],
        crs_wkt=str(problem["crs_wkt"]),
        transform=tuple(float(value) for value in problem["transform"]),
        footprint=faopt.footprint,
        noise_std_m=faopt.noise_std_m,
        candidate_geometry_sha256=str(problem["candidate_geometry_sha256"]),
        problem_sha256=problem_sha256,
        paper1=faopt.paper1,
        source_selection_sha256=source_selection_sha256,
        points=points,
    )
    plan.validate()
    return plan


def _selector_plans(
    *,
    problem: Mapping[str, Any],
    problem_sha256: str,
    faopt: PlanContract,
    prior: E5PriorVariance,
    prior_file_sha256: str,
    master_seed: int,
) -> dict[str, tuple[PlanContract, dict[str, Any]]]:
    plans: dict[str, tuple[PlanContract, dict[str, Any]]] = {}
    for selector_instance in E5_RANDOM_INSTANCES:
        order, provenance = _random_order(
            problem,
            selector_instance=selector_instance,
            master_seed=master_seed,
        )
        source_hash = canonical_sha256(provenance)
        plans[selector_instance] = (
            _generated_plan(
                faopt=faopt,
                problem=problem,
                problem_sha256=problem_sha256,
                method="random",
                order=order,
                source_selection_sha256=source_hash,
            ),
            provenance,
        )
    order, provenance = _uniform_farthest_order(problem)
    plans["uniform"] = (
        _generated_plan(
            faopt=faopt,
            problem=problem,
            problem_sha256=problem_sha256,
            method="uniform-farthest",
            order=order,
            source_selection_sha256=canonical_sha256(provenance),
        ),
        provenance,
    )
    order, provenance = _max_variance_order(
        problem,
        prior,
        prior_file_sha256=prior_file_sha256,
    )
    plans["max-variance"] = (
        _generated_plan(
            faopt=faopt,
            problem=problem,
            problem_sha256=problem_sha256,
            method="max-prior-variance",
            order=order,
            source_selection_sha256=canonical_sha256(provenance),
        ),
        provenance,
    )
    plans["FAOPT"] = (
        faopt,
        {
            "schema": "paper2.e5-faopt-reuse/v1",
            "source_plan_sha256": faopt.sha256,
            "source_selection_sha256": faopt.source_selection_sha256,
            "algorithm": "reuse-imported-paper1-discrete-faopt",
        },
    )
    if tuple(plans) != E5_SELECTOR_INSTANCES:
        raise AssertionError("E5 selector family ordering differs from the preregistration")
    random_orders = {
        tuple(point.candidate_id for point in plans[name][0].points)
        for name in E5_RANDOM_INSTANCES
    }
    if len(random_orders) != len(E5_RANDOM_INSTANCES):
        raise ValueError("the ten E5 random streams did not produce independent Kmax orders")
    return plans


def build_e5_selector_bundle(
    *,
    problem_path: str | Path,
    faopt_kmax_plan_path: str | Path,
    prior_variance_path: str | Path,
    output_directory: str | Path,
    master_seed: int = 20260721,
    paper2_root: str | Path = PAPER2_ROOT,
) -> Path:
    """Generate and lock the complete 13-selector E5 plan family for one tile."""

    root = Path(paper2_root).resolve()
    problem_path = _guard_input(problem_path, root)
    faopt_path = _guard_input(faopt_kmax_plan_path, root)
    prior_path = _guard_input(prior_variance_path, root)
    output = guard_paper2_path(output_directory, paper2_root=root, allow_root=True)
    if output == root:
        raise ValueError("E5 selector bundle must be written below the paper2 root")
    problem = load_candidate_problem(problem_path)
    problem_hash = file_sha256(problem_path)
    _validate_problem(problem, problem_sha256=problem_hash)
    faopt = PlanContract.read(faopt_path)
    _validate_faopt(faopt, problem, problem_sha256=problem_hash)
    prior = E5PriorVariance.read(prior_path)
    if (
        prior.tile_id != problem["tile_id"]
        or prior.problem_sha256 != problem_hash
        or prior.candidate_geometry_sha256 != problem["candidate_geometry_sha256"]
        or not np.array_equal(prior.candidate_ids, problem["candidate_ids"])
    ):
        raise ValueError("E5 prior-variance contract differs from the candidate problem")
    prior_hash = file_sha256(prior_path)
    selector_plans = _selector_plans(
        problem=problem,
        problem_sha256=problem_hash,
        faopt=faopt,
        prior=prior,
        prior_file_sha256=prior_hash,
        master_seed=master_seed,
    )
    families = []
    for selector_instance, (plan, provenance) in selector_plans.items():
        paths = write_plan_family(
            plan,
            output / selector_instance,
            budgets=E5_BUDGETS,
            paper2_root=root,
        )
        records = []
        for budget, path in zip(E5_BUDGETS, paths, strict=True):
            contract = PlanContract.read(path)
            records.append(
                {
                    "budget": budget,
                    "relative_path": _relative_path(path, root),
                    "file_sha256": file_sha256(path),
                    "contract_sha256": contract.sha256,
                    "prefix_of_sha256": contract.prefix_of_sha256,
                }
            )
        families.append(
            {
                "selector_instance": selector_instance,
                "plan_method": plan.method,
                "source_selection_sha256": plan.source_selection_sha256,
                "kmax_order_sha256": canonical_sha256(
                    [point.candidate_id for point in plan.points]
                ),
                "provenance": provenance,
                "plans": records,
            }
        )
    payload: dict[str, Any] = {
        "schema": "paper2.e5-selector-bundle/v1",
        "tile_id": str(problem["tile_id"]),
        "scenario_id": str(problem["scenario_id"]),
        "budgets": list(E5_BUDGETS),
        "selector_instances": list(E5_SELECTOR_INSTANCES),
        "random_master_seed": int(master_seed),
        "bindings": {
            "problem_relative_path": _relative_path(problem_path, root),
            "problem_sha256": problem_hash,
            "candidate_geometry_sha256": str(problem["candidate_geometry_sha256"]),
            "faopt_kmax_plan_relative_path": _relative_path(faopt_path, root),
            "faopt_kmax_plan_file_sha256": file_sha256(faopt_path),
            "faopt_kmax_plan_contract_sha256": faopt.sha256,
            "prior_variance_relative_path": _relative_path(prior_path, root),
            "prior_variance_file_sha256": prior_hash,
            "prior_variance_sha256": prior.prior_variance_sha256,
        },
        "families": families,
        "reference_accessed": False,
    }
    payload["content_sha256"] = _manifest_content_sha256(payload)
    manifest_path = write_canonical_json(
        output / "manifest.json",
        payload,
        paper2_root=root,
    )
    audit_e5_selector_bundle(manifest_path, paper2_root=root)
    return manifest_path


def audit_e5_selector_bundle(
    manifest_path: str | Path,
    *,
    paper2_root: str | Path = PAPER2_ROOT,
) -> dict[str, Any]:
    """Recompute every selector and verify all plan bytes and strict prefixes."""

    root = Path(paper2_root).resolve()
    manifest_path = _guard_input(manifest_path, root)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "paper2.e5-selector-bundle/v1":
        raise ValueError("unsupported E5 selector-bundle schema")
    _require_exact_keys(
        payload,
        {
            "schema",
            "tile_id",
            "scenario_id",
            "budgets",
            "selector_instances",
            "random_master_seed",
            "bindings",
            "families",
            "reference_accessed",
            "content_sha256",
        },
        location="E5 selector-bundle manifest",
    )
    if payload.get("reference_accessed") is not False:
        raise ValueError("E5 selector bundle must be explicitly reference-free")
    if payload.get("content_sha256") != _manifest_content_sha256(payload):
        raise ValueError("E5 selector-bundle content hash differs")
    if tuple(int(value) for value in payload.get("budgets", [])) != E5_BUDGETS:
        raise ValueError("E5 selector-bundle budgets differ")
    if tuple(payload.get("selector_instances", [])) != E5_SELECTOR_INSTANCES:
        raise ValueError("E5 selector-bundle instances differ")
    bindings = dict(payload.get("bindings", {}))
    _require_exact_keys(
        bindings,
        {
            "problem_relative_path",
            "problem_sha256",
            "candidate_geometry_sha256",
            "faopt_kmax_plan_relative_path",
            "faopt_kmax_plan_file_sha256",
            "faopt_kmax_plan_contract_sha256",
            "prior_variance_relative_path",
            "prior_variance_file_sha256",
            "prior_variance_sha256",
        },
        location="E5 selector-bundle bindings",
    )
    problem_path = _guard_input(str(bindings["problem_relative_path"]), root)
    faopt_path = _guard_input(str(bindings["faopt_kmax_plan_relative_path"]), root)
    prior_path = _guard_input(str(bindings["prior_variance_relative_path"]), root)
    if file_sha256(problem_path) != bindings.get("problem_sha256"):
        raise ValueError("E5 selector-bundle problem bytes differ")
    if file_sha256(faopt_path) != bindings.get("faopt_kmax_plan_file_sha256"):
        raise ValueError("E5 selector-bundle FAOPT source bytes differ")
    if file_sha256(prior_path) != bindings.get("prior_variance_file_sha256"):
        raise ValueError("E5 selector-bundle prior-variance bytes differ")
    problem = load_candidate_problem(problem_path)
    problem_hash = file_sha256(problem_path)
    _validate_problem(problem, problem_sha256=problem_hash)
    if (
        payload.get("tile_id") != problem["tile_id"]
        or payload.get("scenario_id") != problem["scenario_id"]
        or bindings.get("candidate_geometry_sha256")
        != problem["candidate_geometry_sha256"]
    ):
        raise ValueError("E5 selector-bundle problem identity differs")
    faopt = PlanContract.read(faopt_path)
    _validate_faopt(faopt, problem, problem_sha256=problem_hash)
    if faopt.sha256 != bindings.get("faopt_kmax_plan_contract_sha256"):
        raise ValueError("E5 selector-bundle FAOPT contract hash differs")
    prior = E5PriorVariance.read(prior_path)
    if (
        prior.tile_id != problem["tile_id"]
        or prior.problem_sha256 != problem_hash
        or prior.candidate_geometry_sha256 != problem["candidate_geometry_sha256"]
        or prior.prior_variance_sha256 != bindings.get("prior_variance_sha256")
        or not np.array_equal(prior.candidate_ids, problem["candidate_ids"])
    ):
        raise ValueError("E5 selector-bundle prior-variance binding differs")
    expected = _selector_plans(
        problem=problem,
        problem_sha256=problem_hash,
        faopt=faopt,
        prior=prior,
        prior_file_sha256=file_sha256(prior_path),
        master_seed=int(payload["random_master_seed"]),
    )
    families = list(payload.get("families", []))
    if [family.get("selector_instance") for family in families] != list(
        E5_SELECTOR_INSTANCES
    ):
        raise ValueError("E5 selector-bundle family closure differs")
    plan_file_count = 0
    expected_plan_paths: set[Path] = set()
    for family in families:
        _require_exact_keys(
            family,
            {
                "selector_instance",
                "plan_method",
                "source_selection_sha256",
                "kmax_order_sha256",
                "provenance",
                "plans",
            },
            location="E5 selector family",
        )
        selector_instance = str(family["selector_instance"])
        expected_plan, expected_provenance = expected[selector_instance]
        if family.get("plan_method") != expected_plan.method:
            raise ValueError(f"E5 {selector_instance} plan method differs")
        if family.get("source_selection_sha256") != expected_plan.source_selection_sha256:
            raise ValueError(f"E5 {selector_instance} selection provenance hash differs")
        if family.get("provenance") != expected_provenance:
            raise ValueError(f"E5 {selector_instance} provenance differs")
        if family.get("kmax_order_sha256") != canonical_sha256(
            [point.candidate_id for point in expected_plan.points]
        ):
            raise ValueError(f"E5 {selector_instance} Kmax order differs")
        records = list(family.get("plans", []))
        if [int(record.get("budget", -1)) for record in records] != list(E5_BUDGETS):
            raise ValueError(f"E5 {selector_instance} plan-budget closure differs")
        for record in records:
            _require_exact_keys(
                record,
                {
                    "budget",
                    "relative_path",
                    "file_sha256",
                    "contract_sha256",
                    "prefix_of_sha256",
                },
                location=f"E5 {selector_instance} plan record",
            )
            budget = int(record["budget"])
            path = _guard_input(str(record["relative_path"]), root)
            expected_path = (
                manifest_path.parent
                / selector_instance
                / f"k{budget:03d}"
                / "plan.json"
            ).resolve()
            if path != expected_path:
                raise ValueError(f"E5 {selector_instance} K={budget} plan path differs")
            expected_plan_paths.add(path)
            if file_sha256(path) != record.get("file_sha256"):
                raise ValueError(f"E5 {selector_instance} K={budget} plan bytes differ")
            observed = PlanContract.read(path)
            expected_prefix = expected_plan.prefix(budget)
            if observed.to_dict() != expected_prefix.to_dict():
                raise ValueError(f"E5 {selector_instance} K={budget} plan contract differs")
            if observed.sha256 != record.get("contract_sha256"):
                raise ValueError(f"E5 {selector_instance} K={budget} contract hash differs")
            if observed.prefix_of_sha256 != record.get("prefix_of_sha256"):
                raise ValueError(f"E5 {selector_instance} K={budget} prefix hash differs")
            plan_file_count += 1
    if plan_file_count != len(E5_SELECTOR_INSTANCES) * len(E5_BUDGETS):
        raise AssertionError("E5 selector-bundle plan count differs")
    observed_plan_paths = {
        path.resolve() for path in manifest_path.parent.glob("*/k*/plan.json")
    }
    if observed_plan_paths != expected_plan_paths:
        raise ValueError("E5 selector-bundle filesystem plan closure differs")
    return {
        "schema": "paper2.e5-selector-audit/v1",
        "manifest_sha256": file_sha256(manifest_path),
        "tile_id": str(problem["tile_id"]),
        "selector_instances": len(E5_SELECTOR_INSTANCES),
        "plan_files": plan_file_count,
        "random_instances": len(E5_RANDOM_INSTANCES),
        "budgets": list(E5_BUDGETS),
        "reference_accessed": False,
        "passed": True,
    }


@lru_cache(maxsize=None)
def _audited_e5_plan_index(
    manifest_path_text: str,
    paper2_root_text: str,
    manifest_file_sha256: str,
) -> dict[tuple[str, int], tuple[str, str]]:
    manifest_path = Path(manifest_path_text)
    root = Path(paper2_root_text)
    if file_sha256(manifest_path) != manifest_file_sha256:
        raise ValueError("E5 selector manifest changed before plan binding")
    audit_e5_selector_bundle(manifest_path, paper2_root=root)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    index: dict[tuple[str, int], tuple[str, str]] = {}
    for family in payload["families"]:
        selector = str(family["selector_instance"])
        for record in family["plans"]:
            key = (selector, int(record["budget"]))
            if key in index:
                raise ValueError("E5 selector manifest contains a duplicate plan key")
            index[key] = (str(record["relative_path"]), str(record["file_sha256"]))
    return index


def validate_e5_plan_binding(
    *,
    manifest_path: str | Path,
    selector_instance: str,
    budget: int,
    plan_path: str | Path,
    paper2_root: str | Path = PAPER2_ROOT,
) -> str:
    """Bind one E5 matrix cell to its exact audited selector plan family."""

    root = Path(paper2_root).resolve()
    manifest_path = _guard_input(manifest_path, root)
    plan_path = _guard_input(plan_path, root)
    if selector_instance not in E5_SELECTOR_INSTANCES:
        raise ValueError(f"unknown E5 selector instance: {selector_instance}")
    if budget not in E5_BUDGETS:
        raise ValueError(f"unknown E5 selector budget: {budget}")
    manifest_hash = file_sha256(manifest_path)
    index = _audited_e5_plan_index(
        str(manifest_path),
        str(root),
        manifest_hash,
    )
    expected_relative_path, expected_hash = index[(selector_instance, budget)]
    expected_path = guard_paper2_path(expected_relative_path, paper2_root=root)
    if plan_path != expected_path or file_sha256(plan_path) != expected_hash:
        raise ValueError("E5 matrix cell plan differs from its audited selector family")
    return manifest_hash

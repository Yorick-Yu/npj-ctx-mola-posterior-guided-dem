"""Truth-free E5 max-variance scoring and formal selector input preparation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from .audit_footprints import write_audit_footprint_plan
from .contracts import LaserObservations, PlanContract, ReconstructionProblem, reconstruction_from_observations
from .e5_selectors import (
    E5_BUDGETS,
    E5_SELECTOR_INSTANCES,
    audit_e5_selector_bundle,
    build_e5_selector_bundle,
    write_e5_prior_variance_contract,
)
from .hashing import (
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    is_sha256,
    read_json,
    write_canonical_json,
)
from .matrix import InferenceManifest
from .observations import simulate_and_write_laser_observations
from .plan_import import load_candidate_problem
from .posterior_preparation import FrozenGPPrior, write_gp_posterior
from .sealed_reference import read_sealed_reference_lock
from .training import FORMAL_SEEDS


E5_PRIOR_SOURCE_SCHEMA = "paper2.e5-prior-source/v1"
E5_PREDICTION_INDEX_SCHEMA = "paper2.e5-prior-prediction-index/v1"
E5_INPUT_INDEX_SCHEMA = "paper2.e5-formal-input-index/v1"
E5_REPLICATES = tuple(range(5))


def _content_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )


def _existing(path: str | Path, *, root: Path) -> Path:
    result = guard_paper2_path(path, paper2_root=root)
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def _relative(path: Path, *, root: Path) -> str:
    return _existing(path, root=root).relative_to(root).as_posix()


def _artifact(path: Path, *, root: Path) -> dict[str, Any]:
    path = _existing(path, root=root)
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "byte_count": path.stat().st_size,
    }


def _verified_artifact(record: Mapping[str, Any], *, root: Path) -> Path:
    relative = Path(str(record.get("relative_path", "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("E5 artifact path is invalid")
    path = _existing(root / relative, root=root)
    if (
        file_sha256(path) != record.get("sha256")
        or path.stat().st_size != int(record.get("byte_count", -1))
    ):
        raise ValueError(f"E5 artifact bytes differ: {relative}")
    return path


def _sample_grid(
    values: np.ndarray,
    coordinates: np.ndarray,
    transform: Sequence[float],
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    xy = np.asarray(coordinates, dtype=np.float64)
    if array.ndim != 2 or xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("E5 score raster/candidate coordinates are malformed")
    a, b, c, d, e, f = (float(value) for value in transform)
    basis = np.asarray(((a, b), (d, e)), dtype=np.float64)
    if abs(float(np.linalg.det(basis))) <= 1.0e-12:
        raise ValueError("E5 candidate transform is singular")
    column_row = (np.linalg.inv(basis) @ (xy - np.asarray((c, f))).T).T - 0.5
    columns = column_row[:, 0]
    rows = column_row[:, 1]
    tolerance = 1.0e-6
    if (
        np.any(rows < -tolerance)
        or np.any(columns < -tolerance)
        or np.any(rows > array.shape[0] - 1 + tolerance)
        or np.any(columns > array.shape[1] - 1 + tolerance)
    ):
        raise ValueError("E5 candidate coordinates leave the prediction grid")
    return ndimage.map_coordinates(
        array,
        np.vstack((rows, columns)),
        order=1,
        mode="nearest",
        prefilter=False,
    )


def build_e5_prior_variance_from_predictions(
    *,
    problem_path: str | Path,
    prediction_paths: Sequence[str | Path],
    inference_manifest_paths: Sequence[str | Path],
    output_path: str | Path,
    source_manifest_path: str | Path,
    paper2_root: str | Path,
) -> tuple[Path, Path]:
    """Apply total variance to three frozen K=0 OT-SB-full predictions."""

    root = Path(paper2_root).resolve()
    candidate_path = _existing(problem_path, root=root)
    predictions = tuple(_existing(path, root=root) for path in prediction_paths)
    manifests = tuple(
        _existing(path, root=root) for path in inference_manifest_paths
    )
    if len(predictions) != len(FORMAL_SEEDS) or len(manifests) != len(FORMAL_SEEDS):
        raise ValueError("E5 max-variance scoring requires exactly three formal seed paths")
    problem = load_candidate_problem(candidate_path)
    tile_id = str(problem["tile_id"])
    expected_sentinel = canonical_sha256({"calibration_manifest": "not-applied"})
    means: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    source_records: list[dict[str, Any]] = []
    observed_seeds: set[int] = set()
    checkpoint_hashes: set[str] = set()
    for prediction_path, manifest_path in zip(predictions, manifests, strict=True):
        manifest = InferenceManifest.read(manifest_path)
        key = manifest.key
        if (
            manifest.status != "complete"
            or key.experiment != "E5-prior"
            or key.tile_id != tile_id
            or key.method != "OT-SB-full"
            or key.model_identity != "OT-SB-full"
            or key.budget != 0
            or key.noise_replicate != 0
            or key.condition != "max-variance-score"
            or key.training_seed not in FORMAL_SEEDS
            or key.training_seed in observed_seeds
            or key.path_count < 2
            or key.step_count <= 0
        ):
            raise ValueError("E5 prior inference identity differs from the frozen protocol")
        artifacts = {artifact.name: artifact for artifact in manifest.artifacts}
        prediction_artifact = artifacts.get("prediction")
        if (
            prediction_artifact is None
            or root / prediction_artifact.relative_path != prediction_path
            or prediction_artifact.sha256 != file_sha256(prediction_path)
        ):
            raise ValueError("E5 prior prediction differs from its inference manifest")
        trace = dict(manifest.trace_hashes)
        checkpoint_hash = str(trace.get("checkpoint_sha256", ""))
        calibration_hash = str(trace.get("calibration_manifest_sha256", ""))
        if (
            not is_sha256(checkpoint_hash)
            or not is_sha256(calibration_hash)
            or calibration_hash == expected_sentinel
        ):
            raise ValueError("E5 prior prediction lacks checkpoint/calibration provenance")
        with np.load(prediction_path, allow_pickle=False) as archive:
            if set(archive.files) != {"mean", "variance"}:
                raise ValueError("E5 prior prediction archive keys differ")
            mean = np.asarray(archive["mean"], dtype=np.float64)
            variance = np.asarray(archive["variance"], dtype=np.float64)
        if (
            mean.ndim != 2
            or variance.shape != mean.shape
            or not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(variance))
            or np.any(variance < -1.0e-8)
        ):
            raise ValueError("E5 prior prediction mean/variance is invalid")
        means.append(mean)
        variances.append(np.maximum(variance, 0.0))
        observed_seeds.add(key.training_seed)
        checkpoint_hashes.add(checkpoint_hash)
        source_records.append(
            {
                "training_seed": key.training_seed,
                "path_count": key.path_count,
                "step_count": key.step_count,
                "prediction": _artifact(prediction_path, root=root),
                "inference_manifest": _artifact(manifest_path, root=root),
                "checkpoint_sha256": checkpoint_hash,
                "calibration_manifest_sha256": calibration_hash,
                "footprint_calibration_manifest_sha256": str(
                    trace.get("footprint_calibration_manifest_sha256", "")
                ),
                "code_sha256": str(trace.get("code_sha256", "")),
                "environment_sha256": str(trace.get("environment_sha256", "")),
            }
        )
    if observed_seeds != set(FORMAL_SEEDS) or len(checkpoint_hashes) != len(FORMAL_SEEDS):
        raise ValueError("E5 prior predictions do not cover three distinct formal seeds")
    shapes = {value.shape for value in (*means, *variances)}
    if len(shapes) != 1:
        raise ValueError("E5 prior prediction grids differ across seeds")
    coordinates = np.asarray(problem["candidates"], dtype=np.float64)
    sampled_means = np.stack(
        [_sample_grid(value, coordinates, problem["transform"]) for value in means]
    )
    sampled_variances = np.stack(
        [_sample_grid(value, coordinates, problem["transform"]) for value in variances]
    )
    total_variance = np.mean(sampled_variances, axis=0) + np.var(
        sampled_means, axis=0, ddof=0
    )
    if not np.all(np.isfinite(total_variance)) or np.any(total_variance < 0.0):
        raise RuntimeError("E5 total predictive variance is invalid")
    source_records.sort(key=lambda value: int(value["training_seed"]))
    source_model_sha256 = canonical_sha256(
        {
            "schema": E5_PRIOR_SOURCE_SCHEMA,
            "aggregation": "mean-calibrated-within-seed-plus-between-seed-mean-variance",
            "sampling": "bilinear-at-candidate-world-coordinate",
            "sources": source_records,
        }
    )
    prior_path = write_e5_prior_variance_contract(
        problem_path=candidate_path,
        prior_variance=total_variance,
        source_model_sha256=source_model_sha256,
        output_path=output_path,
        paper2_root=root,
    )
    payload: dict[str, Any] = {
        "schema": E5_PRIOR_SOURCE_SCHEMA,
        "status": "complete",
        "tile_id": tile_id,
        "reference_accessed": False,
        "candidate_problem": _artifact(candidate_path, root=root),
        "prior_variance": _artifact(prior_path, root=root),
        "source_model_sha256": source_model_sha256,
        "aggregation": "law-of-total-variance-over-three-frozen-training-seeds",
        "candidate_sampling": "bilinear-world-to-grid",
        "sources": source_records,
    }
    payload["content_sha256"] = _content_sha256(payload)
    manifest_path = write_canonical_json(
        source_manifest_path, payload, paper2_root=root
    )
    return prior_path, manifest_path


def _paper1_maps(
    payload: Mapping[str, Any], *, root: Path
) -> tuple[dict[tuple[str, int, int], dict[str, Any]], dict[str, dict[str, Any]]]:
    cells: dict[tuple[str, int, int], dict[str, Any]] = {}
    for raw in payload.get("cells", ()):
        if raw.get("role") != "paper1_bridge_test":
            continue
        cell = dict(raw)
        key = (
            str(cell["tile_id"]),
            int(cell["budget"]),
            int(cell["noise_replicate"]),
        )
        if key in cells:
            raise ValueError("E5 Paper1 input index contains duplicate cells")
        for name in ("problem", "plan", "observations", "gp_posterior", "gp_manifest"):
            path = _existing(root / str(cell[name]), root=root)
            if file_sha256(path) != cell[f"{name}_sha256"]:
                raise ValueError(f"E5 Paper1 base {name} bytes differ")
        cells[key] = cell
    tiles: dict[str, dict[str, Any]] = {}
    for raw in payload.get("tiles", ()):
        if raw.get("role") != "paper1_bridge_test":
            continue
        tile = dict(raw)
        tile_id = str(tile["tile_id"])
        if tile_id in tiles:
            raise ValueError("E5 Paper1 input index contains duplicate tiles")
        candidate = _existing(root / str(tile["candidate_problem"]), root=root)
        if file_sha256(candidate) != tile.get("candidate_problem_sha256"):
            raise ValueError("E5 candidate problem bytes differ")
        tiles[tile_id] = tile
    return cells, tiles


def _prediction_maps(
    payload: Mapping[str, Any], *, root: Path
) -> dict[str, list[dict[str, Any]]]:
    if (
        payload.get("schema") != E5_PREDICTION_INDEX_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("reference_accessed") is not False
        or payload.get("content_sha256") != _content_sha256(payload)
    ):
        raise ValueError("E5 prior prediction index identity/hash/boundary differs")
    result: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, int]] = set()
    for raw in payload.get("predictions", ()):
        record = dict(raw)
        key = (str(record.get("tile_id", "")), int(record.get("training_seed", -1)))
        if key in seen or key[1] not in FORMAL_SEEDS:
            raise ValueError("E5 prior prediction index identities are not unique")
        for name in ("prediction", "inference_manifest"):
            artifact = record.get(name)
            if not isinstance(artifact, Mapping):
                raise ValueError("E5 prior prediction artifact is malformed")
            _verified_artifact(artifact, root=root)
        result.setdefault(key[0], []).append(record)
        seen.add(key)
    if any({int(value["training_seed"]) for value in records} != set(FORMAL_SEEDS) for records in result.values()):
        raise ValueError("E5 prior prediction index lacks a three-seed tile")
    return result


def prepare_e5_formal_inputs(
    *,
    paper2_root: str | Path,
    paper1_input_index_path: str | Path,
    prediction_index_path: str | Path,
    role_manifest_path: str | Path,
    asset_lock_path: str | Path,
    sealed_reference_lock_path: str | Path,
    paper1_import_manifest_path: str | Path,
    gp_prior_path: str | Path,
    output_root: str | Path,
    observation_seed: int = 20260721,
    selector_master_seed: int = 20260721,
) -> dict[str, Any]:
    """Build the 13-selector E5 plans and 2,340 shared physical input cells."""

    root = Path(paper2_root).resolve()
    paper1_path = _existing(paper1_input_index_path, root=root)
    prediction_path = _existing(prediction_index_path, root=root)
    role_path = _existing(role_manifest_path, root=root)
    asset_path = _existing(asset_lock_path, root=root)
    sealed_path = _existing(sealed_reference_lock_path, root=root)
    import_path = _existing(paper1_import_manifest_path, root=root)
    prior_path = _existing(gp_prior_path, root=root)
    output = guard_paper2_path(output_root, paper2_root=root, allow_root=True)
    if output == root:
        raise ValueError("E5 output must be below the Paper2 root")
    paper1 = read_json(paper1_path)
    roles = read_json(role_path)
    if (
        not isinstance(paper1, Mapping)
        or paper1.get("schema") != "paper2.paper1-formal-inference-input-index/v1"
        or paper1.get("status") != "complete"
        or paper1.get("truth_paths_exported_to_inference_workers") is not False
        or paper1.get("asset_lock_sha256") != file_sha256(asset_path)
        or paper1.get("role_manifest_sha256") != file_sha256(role_path)
        or paper1.get("paper1_import_manifest_sha256") != file_sha256(import_path)
        or paper1.get("gp_prior_sha256") != file_sha256(prior_path)
    ):
        raise ValueError("E5 Paper1 formal input index locks differ")
    tile_ids = tuple(
        str(value)
        for value in roles.get("roles", {}).get("paper1_bridge_test", ())
    )
    if len(tile_ids) != 12 or len(set(tile_ids)) != 12:
        raise ValueError("formal E5 requires exactly 12 Paper1 bridge tiles")
    base_cells, base_tiles = _paper1_maps(paper1, root=root)
    if set(base_tiles) != set(tile_ids):
        raise ValueError("E5 Paper1 tile index differs from the role manifest")
    required_base = {
        (tile_id, budget, replicate)
        for tile_id in tile_ids
        for budget in (0, 100)
        for replicate in E5_REPLICATES
        if budget != 0 or replicate == 0
    }
    if not required_base.issubset(base_cells):
        raise ValueError("E5 Paper1 base cells are incomplete")
    prediction_payload = read_json(prediction_path)
    predictions_by_tile = _prediction_maps(prediction_payload, root=root)
    if set(predictions_by_tile) != set(tile_ids):
        raise ValueError("E5 prior prediction tile set differs from the bridge role")
    if prediction_payload.get("paper1_input_index_sha256") != file_sha256(paper1_path):
        raise ValueError("E5 prior predictions bind a different Paper1 input index")
    sealed = read_sealed_reference_lock(
        sealed_path, paper2_root=root, verify_files=True
    )
    simulator_by_tile = {
        str(entry["tile_id"]): dict(entry)
        for entry in sealed["entries"]
        if str(entry["tile_id"]) in tile_ids
    }
    if set(simulator_by_tile) != set(tile_ids):
        raise ValueError("E5 sealed simulator lock lacks Paper1 bridge tiles")
    prior = FrozenGPPrior.from_mapping(read_json(prior_path))

    all_cells: list[dict[str, Any]] = []
    tile_records: list[dict[str, Any]] = []
    for tile_id in sorted(tile_ids):
        tile = base_tiles[tile_id]
        candidate_problem = _existing(root / str(tile["candidate_problem"]), root=root)
        base_cell = base_cells[(tile_id, 100, 0)]
        base_problem = ReconstructionProblem.read(
            _existing(root / str(base_cell["problem"]), root=root)
        )
        faopt_path = _existing(root / str(base_cell["plan"]), root=root)
        simulator_path = _existing(
            root / str(simulator_by_tile[tile_id]["reference_relative_path"]),
            root=root,
        )
        source_records = sorted(
            predictions_by_tile[tile_id], key=lambda value: int(value["training_seed"])
        )
        tile_root = output / tile_id
        score_path, score_manifest_path = build_e5_prior_variance_from_predictions(
            problem_path=candidate_problem,
            prediction_paths=[
                _verified_artifact(record["prediction"], root=root)
                for record in source_records
            ],
            inference_manifest_paths=[
                _verified_artifact(record["inference_manifest"], root=root)
                for record in source_records
            ],
            output_path=tile_root / "prior/prior-variance.npz",
            source_manifest_path=tile_root / "prior/prior-source-manifest.json",
            paper2_root=root,
        )
        selector_manifest = build_e5_selector_bundle(
            problem_path=candidate_problem,
            faopt_kmax_plan_path=faopt_path,
            prior_variance_path=score_path,
            output_directory=tile_root / "selectors",
            master_seed=selector_master_seed,
            paper2_root=root,
        )
        selector_audit = audit_e5_selector_bundle(
            selector_manifest, paper2_root=root
        )
        selector_payload = read_json(selector_manifest)
        plan_paths = {
            (str(family["selector_instance"]), int(record["budget"])): _existing(
                root / str(record["relative_path"]), root=root
            )
            for family in selector_payload["families"]
            for record in family["plans"]
        }
        audit_records: list[dict[str, Any]] = []
        for selector in E5_SELECTOR_INSTANCES:
            selector_cells: list[dict[str, Any]] = []
            for budget in E5_BUDGETS:
                plan_path = plan_paths[(selector, budget)]
                plan = PlanContract.read(plan_path)
                for replicate in E5_REPLICATES:
                    cell_root = tile_root / "physical" / selector / (
                        f"k{budget:03d}/r{replicate:02d}"
                    )
                    observations_path = cell_root / "laser-observations.npz"
                    simulate_and_write_laser_observations(
                        plan,
                        simulator_path,
                        observations_path,
                        replicate=replicate,
                        seed=observation_seed,
                        paper2_root=root,
                    )
                    observations = LaserObservations.read(observations_path)
                    frozen_hashes = dict(base_problem.frozen_model_hashes)
                    frozen_hashes.update(
                        {
                            "e5_selector_manifest": file_sha256(selector_manifest),
                            "e5_prior_source": file_sha256(score_manifest_path),
                        }
                    )
                    metadata = dict(base_problem.metadata)
                    metadata.update(
                        {
                            "selector_instance": selector,
                            "e5_base_problem_sha256": file_sha256(
                                _existing(root / str(base_cell["problem"]), root=root)
                            ),
                            "e5_prior_source_sha256": file_sha256(
                                score_manifest_path
                            ),
                            "z0_source": "paper1-controlled-degradation-factor-24",
                            "degradation_factor": 24,
                        }
                    )
                    problem = reconstruction_from_observations(
                        tile_id=tile_id,
                        z0=base_problem.z0,
                        image=base_problem.image,
                        valid_mask=base_problem.valid_mask,
                        observations=observations,
                        observations_path=observations_path,
                        frozen_model_hashes=frozen_hashes,
                        metadata=metadata,
                        paper2_root=root,
                    )
                    problem_path = problem.write(
                        cell_root / "reconstruction-problem.npz", paper2_root=root
                    )
                    posterior_path = cell_root / "gp-posterior.npz"
                    gp_manifest_path = cell_root / "gp-posterior-manifest.json"
                    write_gp_posterior(
                        problem=problem,
                        plan=plan,
                        prior=prior,
                        problem_sha256=file_sha256(problem_path),
                        plan_sha256=file_sha256(plan_path),
                        prior_sha256=file_sha256(prior_path),
                        posterior_output=posterior_path,
                        manifest_output=gp_manifest_path,
                        paper2_root=root,
                    )
                    record = {
                        "tile_id": tile_id,
                        "role": "paper1_bridge_test",
                        "selector_instance": selector,
                        "budget": budget,
                        "noise_replicate": replicate,
                        "problem": _artifact(problem_path, root=root),
                        "plan": _artifact(plan_path, root=root),
                        "observations": _artifact(observations_path, root=root),
                        "gp_posterior": _artifact(posterior_path, root=root),
                        "gp_manifest": _artifact(gp_manifest_path, root=root),
                    }
                    selector_cells.append(record)
                    all_cells.append(record)
            audit_source = next(
                value
                for value in selector_cells
                if value["budget"] == 100 and value["noise_replicate"] == 0
            )
            audit_path = tile_root / "physical" / selector / "audit-footprint-plan.json"
            write_audit_footprint_plan(
                audit_path,
                problem=ReconstructionProblem.read(
                    root / str(audit_source["problem"]["relative_path"])
                ),
                selection_plan=PlanContract.read(plan_paths[(selector, 100)]),
                selection_plan_path=plan_paths[(selector, 100)],
                asset_lock_path=asset_path,
                paper2_root=root,
            )
            audit_records.append(
                {
                    "selector_instance": selector,
                    "audit_footprint_plan": _artifact(audit_path, root=root),
                }
            )
        tile_records.append(
            {
                "tile_id": tile_id,
                "role": "paper1_bridge_test",
                "candidate_problem": _artifact(candidate_problem, root=root),
                "prepared_manifest": _artifact(
                    root / str(tile["prepared_manifest"]), root=root
                ),
                "paper1_import_manifest": _artifact(import_path, root=root),
                "prior_variance": _artifact(score_path, root=root),
                "prior_source_manifest": _artifact(score_manifest_path, root=root),
                "selector_manifest": _artifact(selector_manifest, root=root),
                "selector_audit": selector_audit,
                "audit_plans": audit_records,
            }
        )

    expected_cells = (
        len(tile_ids)
        * len(E5_SELECTOR_INSTANCES)
        * len(E5_BUDGETS)
        * len(E5_REPLICATES)
    )
    if len(all_cells) != expected_cells:
        raise AssertionError("E5 physical input closure differs from the preregistration")
    payload: dict[str, Any] = {
        "schema": E5_INPUT_INDEX_SCHEMA,
        "status": "complete",
        "role": "paper1_bridge_test",
        "truth_paths_exported_to_inference_workers": False,
        "reference_accessed_for_selector_scores": False,
        "asset_lock_sha256": file_sha256(asset_path),
        "role_manifest_sha256": file_sha256(role_path),
        "paper1_import_manifest_sha256": file_sha256(import_path),
        "sealed_simulator_lock_sha256": file_sha256(sealed_path),
        "paper1_input_index_sha256": file_sha256(paper1_path),
        "prediction_index_sha256": file_sha256(prediction_path),
        "gp_prior_relative_path": _relative(prior_path, root=root),
        "gp_prior_sha256": file_sha256(prior_path),
        "normalization_relative_path": str(paper1["normalization_relative_path"]),
        "normalization_sha256": str(paper1["normalization_sha256"]),
        "selector_instances": list(E5_SELECTOR_INSTANCES),
        "budgets": list(E5_BUDGETS),
        "replicates": list(E5_REPLICATES),
        "tile_ids": sorted(tile_ids),
        "tiles": sorted(tile_records, key=lambda value: value["tile_id"]),
        "cells": sorted(
            all_cells,
            key=lambda value: (
                value["tile_id"],
                value["selector_instance"],
                value["budget"],
                value["noise_replicate"],
            ),
        ),
    }
    payload["content_sha256"] = _content_sha256(payload)
    index_path = write_canonical_json(
        output / "formal-inference-input-index.json", payload, paper2_root=root
    )
    return {
        **payload,
        "index_relative_path": _relative(index_path, root=root),
        "index_file_sha256": file_sha256(index_path),
    }

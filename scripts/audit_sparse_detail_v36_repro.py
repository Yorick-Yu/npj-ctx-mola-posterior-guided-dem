#!/usr/bin/env python3
"""Read-only reproducibility audit for the frozen sparse-detail v36 result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import npj_real_pipeline_v3 as base
import run_frozen_external_v36 as external
from paper2dem.sparse_detail_runtime import (
    DETAIL_SELECTORS,
    PRIOR_LENGTHSCALE_PX,
    PRIOR_NOISE_M,
    file_sha256,
)


EXPECTED_FREEZE_SHA256 = (
    "4af8a1e2bdc3751bac7fb08a9f639213320c95576759b4722b9fb246e0671968"
)
EXPECTED_RESULT_SHA256 = (
    "fe60c600ad29070adddc258df96229044034f19fe3f27a4a48cc7c7e44a04c77"
)
EXPECTED_SEEDS = (20260808, 20260809, 20260810)
ALLOWED_INFERENCE_KEYS = {
    "image",
    "z0",
    "valid",
    "candidate_global_index",
    "candidate_row",
    "candidate_column",
    "candidate_elevation",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def require(checks: dict[str, bool], name: str, condition: bool) -> None:
    checks[name] = bool(condition)


def replay_one_cell(
    project: Path,
    frozen: dict[str, Any],
    inference_root: Path,
    plan_root: Path,
    prediction_root: Path,
    device_name: str,
) -> dict[str, Any]:
    torch, _, _ = base.torch_components()
    seed = EXPECTED_SEEDS[0]
    checkpoint_record = next(
        row for row in frozen["checkpoints"] if int(row["seed"]) == seed
    )
    checkpoint_path = Path(checkpoint_record["path"])
    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = checkpoint["architecture"]
    model = base.make_model(
        base_channels=int(architecture["base_channels"]),
        steps=int(architecture["transport_steps"]),
        minimum_mobility=float(architecture["minimum_mobility"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    cases = external.load_inference_cases(inference_root)
    plans = external.load_plans(plan_root)
    tile_id = sorted(cases)[0]
    selector = "terrain-faopt"
    isolated_case = cases[tile_id]
    current = external.model_case(isolated_case)
    context_global = plans[tile_id][selector][: external.PRIMARY_BUDGET]
    context = external.remap(isolated_case, context_global)
    fields = base.build_gp_fields(
        current,
        context,
        lengthscale_px=PRIOR_LENGTHSCALE_PX,
        noise_m=PRIOR_NOISE_M,
    )
    replay = base.predict_from_fields(
        model, current, *fields, context, device
    ).astype(np.float32)

    archived_path = (
        prediction_root
        / f"seed-{seed}"
        / selector
        / tile_id
        / "prediction.npz"
    )
    with np.load(archived_path, allow_pickle=False) as archive:
        archived = np.asarray(archive["mean"], dtype=np.float32)
        archived_context = np.asarray(
            archive["context_global_indices"], dtype=np.int64
        )
    difference = np.asarray(replay, dtype=np.float64) - np.asarray(
        archived, dtype=np.float64
    )
    return {
        "seed": seed,
        "tile_id": tile_id,
        "selector": selector,
        "target_isolated_case": str(inference_root / f"{tile_id}.npz"),
        "archived_prediction": str(archived_path),
        "context_indices_exact": bool(
            np.array_equal(context_global, archived_context)
        ),
        "max_abs_difference_m": float(np.max(np.abs(difference))),
        "rms_difference_m": float(np.sqrt(np.mean(np.square(difference)))),
        "tolerance_m": 1.0e-4,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project = args.project_root.resolve()
    run = args.run_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"reproducibility audit already exists: {output}")

    freeze_path = run / "frozen-candidate-evaluator.json"
    result_path = run / "external-one-shot-result.json"
    external_contract_path = run / "external-evaluation-contract.json"
    full_case_root = project / "data/sparse-detail-external-cases-v36"
    inference_root = project / "data/sparse-detail-external-inference-v36"
    plan_root = run / "external-selector-plans"
    prediction_root = run / "external-predictions"
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    require(
        checks,
        "frozen_candidate_file_sha256",
        file_sha256(freeze_path) == EXPECTED_FREEZE_SHA256,
    )
    require(
        checks,
        "external_result_file_sha256",
        file_sha256(result_path) == EXPECTED_RESULT_SHA256,
    )
    frozen = read_json(freeze_path)
    unhashed = dict(frozen)
    declared_content_hash = unhashed.pop("content_sha256", None)
    require(
        checks,
        "frozen_candidate_content_sha256",
        declared_content_hash == canonical_hash(unhashed),
    )
    require(
        checks,
        "candidate_frozen_before_external",
        frozen.get("state") == "frozen_before_external_materialization"
        and frozen.get("external_targets_read_before_freeze") is False,
    )
    require(
        checks,
        "development_passed_before_external",
        frozen.get("development_result", {}).get("state") == "passed"
        and all(frozen.get("development_result", {}).get("checks", {}).values()),
    )

    binding_paths = {
        "protocol_sha256": project
        / "runs/sparse-detail-v30-preparation-20260810-v2/third-round-protocol.json",
        "validation_matrix_sha256": project
        / "runs/sparse-detail-v32-lowpass/validation-matrix.json",
        "calibration_sha256": project
        / "runs/sparse-detail-v30-formal/calibration/detail-thresholds.json",
        "gates_sha256": run / "frozen-development-gates-v1.json",
        "training_prior_sha256": project
        / "runs/sparse-detail-v30-formal/terrain-prior.json",
        "mechanism_contract_sha256": run / "projection-contract.json",
        "external_contract_sha256": external_contract_path,
        "passed_development_result_sha256": run / "development-diagnostic.json",
    }
    binding_status = {
        key: path.is_file() and file_sha256(path) == frozen["bindings"][key]
        for key, path in binding_paths.items()
    }
    require(checks, "all_frozen_data_and_contract_bindings", all(binding_status.values()))
    details["binding_status"] = binding_status

    checkpoint_status = []
    for record in frozen["checkpoints"]:
        checkpoint = Path(record["path"])
        manifest = Path(record["training_manifest"])
        manifest_payload = read_json(manifest)
        ok = (
            checkpoint.is_file()
            and manifest.is_file()
            and file_sha256(checkpoint) == record["sha256"]
            and file_sha256(manifest) == record["training_manifest_sha256"]
            and manifest_payload.get("state") == "complete"
            and int(manifest_payload.get("completed_updates", -1)) == 1000
            and manifest_payload.get("network_precision") == "fp32"
            and manifest_payload.get("validation_targets_read") is False
            and manifest_payload.get("external_targets_read") is False
        )
        checkpoint_status.append({"seed": int(record["seed"]), "passed": ok})
    require(
        checks,
        "three_fp32_training_runs_bound_and_target_isolated",
        len(checkpoint_status) == 3 and all(row["passed"] for row in checkpoint_status),
    )
    details["checkpoint_status"] = checkpoint_status

    code_status = [
        {
            "path": record["path"],
            "passed": Path(record["path"]).is_file()
            and file_sha256(Path(record["path"])) == record["sha256"],
        }
        for record in frozen["code"]
    ]
    require(checks, "all_frozen_code_hashes", all(row["passed"] for row in code_status))
    details["frozen_code_file_count"] = len(code_status)

    full_manifest_path = full_case_root / "cases-manifest.json"
    inference_manifest_path = inference_root / "cases-manifest.json"
    plan_manifest_path = plan_root / "plans-manifest.json"
    full_manifest = read_json(full_manifest_path)
    inference_manifest = read_json(inference_manifest_path)
    plan_manifest = read_json(plan_manifest_path)
    result = read_json(result_path)
    require(
        checks,
        "external_materialization_after_freeze",
        freeze_path.stat().st_mtime_ns < full_manifest_path.stat().st_mtime_ns
        and freeze_path.stat().st_mtime_ns < inference_manifest_path.stat().st_mtime_ns
        and freeze_path.stat().st_mtime_ns < plan_manifest_path.stat().st_mtime_ns,
    )
    require(
        checks,
        "six_external_cases_and_manifest_binding",
        full_manifest.get("case_count") == 6
        and len(full_manifest.get("cases", [])) == 6
        and result["bindings"]["external_case_manifest_sha256"]
        == file_sha256(full_manifest_path),
    )
    require(
        checks,
        "inference_manifest_target_isolated",
        inference_manifest.get("case_count") == 6
        and inference_manifest.get("reference_dem_exported") is False
        and inference_manifest.get("heldout_mola_exported") is False
        and inference_manifest.get("evaluation_mask_exported") is False,
    )
    isolated_files_ok = True
    for record in inference_manifest["cases"]:
        path = inference_root / record["case"]
        if file_sha256(path) != record["sha256"]:
            isolated_files_ok = False
            continue
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != ALLOWED_INFERENCE_KEYS:
                isolated_files_ok = False
    require(checks, "all_inference_archives_contain_only_allowed_keys", isolated_files_ok)
    require(
        checks,
        "selector_plans_training_only",
        plan_manifest.get("case_count") == 6
        and plan_manifest.get("validation_or_external_targets_used_by_selector") is False
        and plan_manifest.get("prior_sha256")
        == frozen["bindings"]["training_prior_sha256"],
    )
    plan_files_ok = True
    for record in plan_manifest["plans"]:
        path = plan_root / record["plan"]
        if file_sha256(path) != record["sha256"]:
            plan_files_ok = False
            continue
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(DETAIL_SELECTORS):
                plan_files_ok = False
            for selector in DETAIL_SELECTORS:
                order = np.asarray(archive[selector], dtype=np.int64)
                if order.size < 20 or np.unique(order[:20]).size != 20:
                    plan_files_ok = False
    require(checks, "all_selector_plan_hashes_and_k20_prefixes", plan_files_ok)

    prediction_count = 0
    prediction_files_ok = True
    metadata_ok = True
    for seed in EXPECTED_SEEDS:
        receipt_path = prediction_root / f"seed-{seed}" / "receipt.json"
        receipt = read_json(receipt_path)
        receipt_ok = (
            receipt.get("state") == "complete"
            and receipt.get("seed") == seed
            and receipt.get("prediction_count") == 48
            and len(receipt.get("predictions", [])) == 48
            and receipt.get("frozen_candidate_sha256") == EXPECTED_FREEZE_SHA256
            and receipt.get("reference_dem_read") is False
            and receipt.get("heldout_mola_read") is False
        )
        prediction_files_ok &= receipt_ok
        checkpoint_hash = next(
            row["sha256"] for row in frozen["checkpoints"] if int(row["seed"]) == seed
        )
        for record in receipt.get("predictions", []):
            prediction_count += 1
            path = project / record["prediction"]
            if not path.is_file() or file_sha256(path) != record["sha256"]:
                prediction_files_ok = False
                continue
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {
                    "mean",
                    "context_global_indices",
                    "metadata_json",
                }:
                    prediction_files_ok = False
                mean = np.asarray(archive["mean"])
                prediction_files_ok &= bool(np.isfinite(mean).all())
                metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
                metadata_ok &= (
                    metadata.get("seed") == seed
                    and metadata.get("checkpoint_sha256") == checkpoint_hash
                    and metadata.get("reference_dem_read") is False
                    and metadata.get("heldout_mola_read") is False
                )
    require(
        checks,
        "three_receipts_and_144_prediction_hashes",
        prediction_count == 144 and prediction_files_ok,
    )
    require(checks, "all_prediction_metadata_target_isolated", metadata_ok)

    require(
        checks,
        "one_shot_result_bindings",
        result.get("one_shot") is True
        and result.get("no_post_external_tuning_permitted") is True
        and result.get("external_case_count") == 6
        and result["bindings"]["frozen_candidate_sha256"] == EXPECTED_FREEZE_SHA256
        and result["bindings"]["calibration_sha256"]
        == frozen["bindings"]["calibration_sha256"]
        and result["bindings"]["gates_sha256"] == frozen["bindings"]["gates_sha256"]
        and result["bindings"]["external_contract_sha256"]
        == frozen["bindings"]["external_contract_sha256"]
        and result["bindings"]["script_sha256"]
        == next(
            row["sha256"]
            for row in frozen["code"]
            if Path(row["path"]).name == "run_frozen_external_v36.py"
        ),
    )
    require(
        checks,
        "external_result_all_preregistered_checks_passed",
        result.get("state") == "passed"
        and result.get("result", {}).get("state") == "passed"
        and len(result.get("result", {}).get("checks", {})) == 8
        and all(result.get("result", {}).get("checks", {}).values()),
    )
    rc_path = project / "logs/sparse-detail-v36-external-one-shot-evaluation.rc"
    log_path = project / "logs/sparse-detail-v36-external-one-shot-evaluation.log"
    require(
        checks,
        "one_shot_process_closed_cleanly",
        rc_path.read_text().strip() == "0"
        and sum(
            line.startswith("EXTERNAL_EVAL ")
            for line in log_path.read_text().splitlines()
        )
        == 126,
    )

    replay = replay_one_cell(
        project,
        frozen,
        inference_root,
        plan_root,
        prediction_root,
        args.device,
    )
    require(
        checks,
        "target_isolated_prediction_replay",
        replay["context_indices_exact"]
        and replay["max_abs_difference_m"] <= replay["tolerance_m"],
    )
    details["replay"] = replay
    details["external_metrics"] = {
        key: result["result"][key]
        for key in (
            "detail_reduction",
            "false_detail_reduction",
            "heldout_mola_reduction",
            "lowpass_ratio",
            "detail_improvement_tile_bootstrap_ci95",
            "tile_win_fraction",
            "complex_tile_count",
            "complex_mean_detail_reduction",
            "complex_tile_win_fraction",
        )
    }

    passed = all(checks.values())
    payload = {
        "schema": "mars-sparse-detail-v36-reproducibility-audit/v1",
        "state": "passed" if passed else "failed",
        "read_only": True,
        "external_metrics_recomputed": False,
        "external_targets_used_for_replay": False,
        "expected_frozen_candidate_sha256": EXPECTED_FREEZE_SHA256,
        "expected_external_result_sha256": EXPECTED_RESULT_SHA256,
        "audit_script_sha256": file_sha256(Path(__file__).resolve()),
        "checks": checks,
        "details": details,
    }
    atomic_json(output, payload)
    print(json.dumps({"state": payload["state"], "checks": checks}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prepare, infer, and evaluate the frozen v36 candidate on external terrain."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import diagnose_sparse_detail_snapshots_v32 as diagnostic
import evaluate_lowpass_projection_v34 as v34
import evaluate_local_laser_projection_v35 as v35
import evaluate_sparse_detail_v30 as evaluator
import npj_real_pipeline_v3 as base
from paper2dem.detail_metrics import DetailThresholdCalibration
from paper2dem.sparse_detail_runtime import (
    DETAIL_SELECTORS,
    PRIOR_LENGTHSCALE_PX,
    PRIOR_NOISE_M,
    SparseDetailCase,
    apply_terrain_prior,
    build_selector_orders,
    file_sha256,
)


MODEL_SEEDS = (20260808, 20260809, 20260810)
PRIMARY_BUDGET = 20
SOURCE_ROOT = Path("data/controlled_ctx_prospective_pool-v4")
MOLA_ROOT = Path("data/real_mola_prospective_pool-v4")
METRICS = diagnostic.METRICS


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def verified_freeze(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("state") != "frozen_before_external_materialization":
        raise PermissionError("v36 candidate is not frozen for external evaluation")
    unhashed = dict(payload)
    declared = unhashed.pop("content_sha256", None)
    actual = hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if declared != actual:
        raise ValueError("frozen candidate content hash differs")
    return payload


def command_prepare(args: argparse.Namespace) -> int:
    project = args.project_root.resolve()
    freeze_path = args.frozen_candidate.resolve()
    frozen = verified_freeze(freeze_path)
    protocol_path = args.protocol.resolve()
    protocol = read_json(protocol_path)
    if frozen["bindings"]["protocol_sha256"] != file_sha256(protocol_path):
        raise ValueError("external protocol differs from frozen candidate")
    case_root = args.case_output.resolve()
    inference_root = args.inference_output.resolve()
    plan_root = args.plan_output.resolve()
    for root in (case_root, inference_root, plan_root):
        if root.exists():
            raise FileExistsError(f"external output already exists: {root}")
        root.mkdir(parents=True)
    prior_path = args.prior.resolve()
    prior = read_json(prior_path)
    if prior.get("validation_or_external_targets_used") is not False:
        raise PermissionError("selector prior is not training-only")

    case_receipts = []
    inference_receipts = []
    plan_receipts = []
    records = [record for record in protocol["tiles"] if record["role"] == "external"]
    if len(records) != 6:
        raise ValueError("frozen protocol external count differs from six")
    for number, record in enumerate(records, start=1):
        tile_id = str(record["tile_id"])
        source_path = project / SOURCE_ROOT / "tiles" / tile_id / "source.npz"
        mola_path = project / MOLA_ROOT / "tiles" / tile_id / "mola_points.npz"
        if file_sha256(source_path) != record["source_sha256"]:
            raise ValueError(f"external source hash differs: {tile_id}")
        if file_sha256(mola_path) != record["mola_points_sha256"]:
            raise ValueError(f"external MOLA hash differs: {tile_id}")
        with np.load(source_path, allow_pickle=False) as archive:
            source = {key: np.asarray(archive[key]) for key in archive.files}
        with np.load(mola_path, allow_pickle=False) as archive:
            row = np.asarray(archive["row"], dtype=np.float64)
            column = np.asarray(archive["column"], dtype=np.float64)
            elevation = np.asarray(archive["elevation"], dtype=np.float64)
            orbit = np.asarray(archive["orbit"], dtype=np.int64)
            quality = np.asarray(archive["quality_valid"], dtype=bool)
        keep, screening = base.robust_point_filter(
            source["z0"], row, column, elevation
        )
        keep &= quality
        assimilation = keep & np.isin(orbit, record["assimilation_orbits"])
        evaluation = keep & np.isin(orbit, record["evaluation_orbits"])
        if np.count_nonzero(assimilation) < PRIMARY_BUDGET:
            raise ValueError(f"external assimilation points are insufficient: {tile_id}")
        if np.count_nonzero(evaluation) < 20:
            raise ValueError(f"external evaluation points are insufficient: {tile_id}")
        full = {
            "image": np.asarray(source["image"], dtype=np.float32),
            "z0": np.asarray(source["z0"], dtype=np.float32),
            "dem": np.asarray(source["dem"], dtype=np.float32),
            "valid": np.asarray(source["valid_mask"], dtype=bool),
            "row": row.astype(np.float32),
            "column": column.astype(np.float32),
            "elevation": elevation.astype(np.float32),
            "orbit": orbit,
            "assimilation": assimilation,
            "evaluation": evaluation,
            "filter_report": np.asarray(json.dumps(screening)),
        }
        case_path = case_root / f"{tile_id}.npz"
        atomic_npz(case_path, **full)
        candidate_global = np.flatnonzero(assimilation).astype(np.int64)
        selector_case = SparseDetailCase(
            tile_id,
            str(record["region"]),
            "external-selector",
            case_path,
            {
                "image": full["image"],
                "z0": full["z0"],
                "valid": full["valid"],
                "row": full["row"],
                "column": full["column"],
                "orbit": full["orbit"],
                "assimilation": full["assimilation"],
            },
        )
        orders = build_selector_orders(
            selector_case, prior, maximum_budget=max(PRIMARY_BUDGET, 80)
        )
        plan_path = plan_root / f"{tile_id}.npz"
        atomic_npz(plan_path, **orders)
        inference_path = inference_root / f"{tile_id}.npz"
        atomic_npz(
            inference_path,
            image=full["image"],
            z0=full["z0"],
            valid=full["valid"],
            candidate_global_index=candidate_global,
            candidate_row=full["row"][candidate_global],
            candidate_column=full["column"][candidate_global],
            candidate_elevation=full["elevation"][candidate_global],
        )
        case_receipts.append(
            {
                "tile_id": tile_id,
                "region": record["region"],
                "role": "external",
                "case": case_path.name,
                "sha256": file_sha256(case_path),
                "assimilation_point_count": int(np.count_nonzero(assimilation)),
                "evaluation_point_count": int(np.count_nonzero(evaluation)),
                "complexity": record["complexity"],
            }
        )
        inference_receipts.append(
            {
                "tile_id": tile_id,
                "region": record["region"],
                "case": inference_path.name,
                "sha256": file_sha256(inference_path),
                "candidate_count": int(candidate_global.size),
            }
        )
        plan_receipts.append(
            {
                "tile_id": tile_id,
                "region": record["region"],
                "role": "external",
                "plan": plan_path.name,
                "sha256": file_sha256(plan_path),
                "maximum_selected_count": int(len(orders[DETAIL_SELECTORS[0]])),
            }
        )
        print(f"EXTERNAL_PREP {number}/{len(records)} {tile_id}", flush=True)

    common = {
        "protocol_sha256": file_sha256(protocol_path),
        "frozen_candidate_sha256": file_sha256(freeze_path),
        "case_count": len(records),
        "external_files_accessed": True,
    }
    atomic_json(
        case_root / "cases-manifest.json",
        {
            "schema": "mars-sparse-detail-external-cases/v1",
            "state": "materialized_after_candidate_freeze",
            **common,
            "cases": case_receipts,
        },
    )
    atomic_json(
        inference_root / "cases-manifest.json",
        {
            "schema": "mars-sparse-detail-external-inference-cases/v1",
            "state": "target_isolated_after_candidate_freeze",
            **common,
            "reference_dem_exported": False,
            "heldout_mola_exported": False,
            "evaluation_mask_exported": False,
            "cases": inference_receipts,
        },
    )
    atomic_json(
        plan_root / "plans-manifest.json",
        {
            "schema": "mars-sparse-detail-external-selector-plans/v1",
            "state": "generated_after_candidate_freeze_from_training_prior",
            **common,
            "prior_sha256": file_sha256(prior_path),
            "selectors": list(DETAIL_SELECTORS),
            "budgets": [PRIMARY_BUDGET],
            "validation_or_external_targets_used_by_selector": False,
            "plans": plan_receipts,
        },
    )
    return 0


def load_inference_cases(root: Path) -> dict[str, base.LoadedCase]:
    manifest = read_json(root / "cases-manifest.json")
    output = {}
    for record in manifest["cases"]:
        path = root / record["case"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"external inference case hash differs: {record['tile_id']}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
        output[record["tile_id"]] = base.LoadedCase(
            record["tile_id"], record["region"], "external-inference", arrays
        )
    return output


def load_plans(root: Path) -> dict[str, dict[str, np.ndarray]]:
    manifest = read_json(root / "plans-manifest.json")
    output = {}
    for record in manifest["plans"]:
        path = root / record["plan"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"external plan hash differs: {record['tile_id']}")
        with np.load(path, allow_pickle=False) as archive:
            output[record["tile_id"]] = {
                selector: np.asarray(archive[selector], dtype=np.int64)
                for selector in DETAIL_SELECTORS
            }
    return output


def model_case(case: base.LoadedCase) -> base.LoadedCase:
    return base.LoadedCase(
        case.tile_id,
        case.region,
        case.role,
        {
            "image": case.arrays["image"],
            "z0": case.arrays["z0"],
            "valid": case.arrays["valid"],
            "row": case.arrays["candidate_row"],
            "column": case.arrays["candidate_column"],
            "elevation": case.arrays["candidate_elevation"],
        },
    )


def remap(case: base.LoadedCase, global_indices: np.ndarray) -> np.ndarray:
    lookup = {
        int(value): index
        for index, value in enumerate(case.arrays["candidate_global_index"])
    }
    return np.asarray([lookup[int(value)] for value in global_indices], dtype=np.int64)


def command_infer(args: argparse.Namespace) -> int:
    torch, _, _ = base.torch_components()
    seed = int(args.seed)
    if seed not in MODEL_SEEDS:
        raise ValueError("unsupported external inference seed")
    project = args.project_root.resolve()
    frozen = verified_freeze(args.frozen_candidate.resolve())
    checkpoint_record = next(
        record for record in frozen["checkpoints"] if int(record["seed"]) == seed
    )
    checkpoint_path = Path(checkpoint_record["path"])
    if file_sha256(checkpoint_path) != checkpoint_record["sha256"]:
        raise ValueError("frozen checkpoint hash differs")
    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = checkpoint["architecture"]
    model = base.make_model(
        base_channels=int(architecture["base_channels"]),
        steps=int(architecture["transport_steps"]),
        minimum_mobility=float(architecture["minimum_mobility"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    cases = load_inference_cases(args.inference_cases.resolve())
    plans = load_plans(args.plans.resolve())
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"external seed output exists: {output}")
    receipts = []
    for tile_id in sorted(cases):
        case = cases[tile_id]
        current = model_case(case)
        shape = current.arrays["z0"].shape
        fields = (
            np.asarray(current.arrays["z0"], dtype=np.float32).copy(),
            np.full(shape, base.HEIGHT_SCALE_M, dtype=np.float32),
            np.full(shape, 400.0, dtype=np.float32),
        )
        tasks = [("shared-k000", np.empty(0, dtype=np.int64), fields)]
        for selector in DETAIL_SELECTORS:
            context = remap(case, plans[tile_id][selector][:PRIMARY_BUDGET])
            tasks.append(
                (
                    selector,
                    context,
                    base.build_gp_fields(
                        current,
                        context,
                        lengthscale_px=PRIOR_LENGTHSCALE_PX,
                        noise_m=PRIOR_NOISE_M,
                    ),
                )
            )
        for selector, context, gp_fields in tasks:
            prediction = base.predict_from_fields(
                model, current, *gp_fields, context, device
            ).astype(np.float32)
            global_context = (
                np.empty(0, dtype=np.int64)
                if selector == "shared-k000"
                else plans[tile_id][selector][:PRIMARY_BUDGET]
            )
            path = output / selector / tile_id / "prediction.npz"
            atomic_npz(
                path,
                mean=prediction,
                context_global_indices=global_context,
                metadata_json=np.asarray(
                    json.dumps(
                        {
                            "schema": "mars-sparse-detail-external-prediction/v1",
                            "tile_id": tile_id,
                            "seed": seed,
                            "selector": selector,
                            "budget": 0 if selector == "shared-k000" else PRIMARY_BUDGET,
                            "checkpoint_sha256": checkpoint_record["sha256"],
                            "reference_dem_read": False,
                            "heldout_mola_read": False,
                        },
                        sort_keys=True,
                    )
                ),
            )
            receipts.append(
                {
                    "tile_id": tile_id,
                    "seed": seed,
                    "selector": selector,
                    "prediction": str(path.relative_to(project)),
                    "sha256": file_sha256(path),
                }
            )
            print(f"EXTERNAL_INFER {seed} {tile_id} {selector}", flush=True)
    atomic_json(
        output / "receipt.json",
        {
            "schema": "mars-sparse-detail-external-inference-receipt/v1",
            "state": "complete",
            "seed": seed,
            "prediction_count": len(receipts),
            "frozen_candidate_sha256": file_sha256(args.frozen_candidate.resolve()),
            "reference_dem_read": False,
            "heldout_mola_read": False,
            "predictions": receipts,
        },
    )
    return 0


def load_full_cases(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    manifest = read_json(root / "cases-manifest.json")
    output = {}
    for record in manifest["cases"]:
        path = root / record["case"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"external case hash differs: {record['tile_id']}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
        output[record["tile_id"]] = {
            "arrays": arrays,
            "pixel_size_m": 20.0,
            "region": record["region"],
            "complexity": record["complexity"],
        }
    return output, manifest


def aggregate_external(
    rows: list[dict[str, Any]],
    gates: dict[str, float],
    contract: dict[str, Any],
    cases: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    family_rows = []
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family = "random" if row["selector"].startswith("random-") else row["selector"]
        groups[(row["tile_id"], row["seed"], family)].append(row)
    for (tile_id, seed, family), members in sorted(groups.items()):
        family_rows.append(
            {
                "tile_id": tile_id,
                "seed": seed,
                "family": family,
                **{metric: diagnostic.mean(members, metric) for metric in METRICS},
            }
        )
    tile_rows = []
    groups = defaultdict(list)
    for row in family_rows:
        groups[(row["tile_id"], row["family"])].append(row)
    for (tile_id, family), members in sorted(groups.items()):
        tile_rows.append(
            {
                "tile_id": tile_id,
                "family": family,
                **{metric: diagnostic.mean(members, metric) for metric in METRICS},
            }
        )
    baseline_rows = [row for row in tile_rows if row["family"] == "shared-k000"]
    candidate_rows = [row for row in tile_rows if row["family"] != "shared-k000"]
    if len(baseline_rows) != 6 or len(candidate_rows) != 30:
        raise ValueError("external family closure differs from 6 plus 30")
    baseline = {metric: diagnostic.mean(baseline_rows, metric) for metric in METRICS}
    candidate = {metric: diagnostic.mean(candidate_rows, metric) for metric in METRICS}
    detail_reduction = diagnostic.reduction(
        baseline["mean_detail_rmse_m"], candidate["mean_detail_rmse_m"]
    )
    false_reduction = diagnostic.reduction(
        baseline["mean_false_detail_fraction_valid"],
        candidate["mean_false_detail_fraction_valid"],
    )
    mola_reduction = diagnostic.reduction(
        baseline["heldout_mola_rmse_m"], candidate["heldout_mola_rmse_m"]
    )
    lowpass_ratio = candidate["lowpass_rmse_m"] / baseline["lowpass_rmse_m"]
    per_tile = []
    for base_row in baseline_rows:
        members = [row for row in candidate_rows if row["tile_id"] == base_row["tile_id"]]
        improvement = diagnostic.reduction(
            base_row["mean_detail_rmse_m"],
            diagnostic.mean(members, "mean_detail_rmse_m"),
        )
        complexity = cases[base_row["tile_id"]]["complexity"]
        complex_terrain = bool(
            float(complexity["p90_slope_deg"]) >= 10.0
            or float(complexity["elevation_range_m"]) >= 350.0
        )
        per_tile.append(
            {
                "tile_id": base_row["tile_id"],
                "detail_improvement": improvement,
                "complex_terrain": complex_terrain,
                "complexity": complexity,
            }
        )
    values = np.asarray([row["detail_improvement"] for row in per_tile])
    interval = diagnostic.bootstrap(values)
    wins = float(np.mean(values > 0))
    complex_values = np.asarray(
        [row["detail_improvement"] for row in per_tile if row["complex_terrain"]]
    )
    complex_checks = contract["complex_terrain_checks"]
    checks = {
        "detail_reduction": detail_reduction
        >= gates["k20_mean_detail_rmse_reduction_vs_shared_k0_min"],
        "false_detail_reduction": false_reduction
        >= gates["k20_false_detail_fraction_reduction_vs_shared_k0_min"],
        "heldout_mola_reduction": mola_reduction
        >= gates["k20_heldout_mola_rmse_reduction_vs_shared_k0_min"],
        "lowpass_noninferiority": lowpass_ratio
        <= gates["k20_lowpass_rmse_ratio_vs_shared_k0_max"],
        "detail_bootstrap_lower_bound": interval[0]
        > gates["detail_improvement_ci95_lower_bound_min"],
        "tile_win_fraction": wins >= gates["tile_win_fraction_min"],
        "complex_mean_detail_reduction": float(np.mean(complex_values))
        >= complex_checks["mean_detail_rmse_reduction_vs_shared_k0_min"],
        "complex_tile_win_fraction": float(np.mean(complex_values > 0))
        >= complex_checks["tile_detail_win_fraction_min"],
    }
    return {
        "state": "passed" if all(checks.values()) else "failed",
        "baseline_k0": baseline,
        "candidate_k20": candidate,
        "detail_reduction": detail_reduction,
        "false_detail_reduction": false_reduction,
        "heldout_mola_reduction": mola_reduction,
        "lowpass_ratio": lowpass_ratio,
        "detail_improvement_tile_bootstrap_ci95": interval,
        "tile_win_fraction": wins,
        "complex_tile_count": int(complex_values.size),
        "complex_mean_detail_reduction": float(np.mean(complex_values)),
        "complex_tile_win_fraction": float(np.mean(complex_values > 0)),
        "per_tile": per_tile,
        "checks": checks,
    }


def command_evaluate(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"one-shot external result exists: {output}")
    project = args.project_root.resolve()
    freeze_path = args.frozen_candidate.resolve()
    frozen = verified_freeze(freeze_path)
    cases, case_manifest = load_full_cases(args.external_cases.resolve())
    calibration = DetailThresholdCalibration.from_dict(
        read_json(args.calibration.resolve())
    )
    gates = read_json(args.gates.resolve())["development_gates"]
    contract = read_json(args.external_contract.resolve())
    prediction_root = args.predictions.resolve()
    rows = []
    mechanism_rows = []
    v35.LOCAL_NOISE_M = 0.0
    for seed in MODEL_SEEDS:
        receipt = read_json(prediction_root / f"seed-{seed}" / "receipt.json")
        if receipt.get("prediction_count") != 48:
            raise ValueError(f"external inference closure differs for seed {seed}")
        by_key = {
            (record["tile_id"], record["selector"]): record
            for record in receipt["predictions"]
        }
        for tile_id in sorted(cases):
            k0_record = by_key[(tile_id, "shared-k000")]
            with np.load(project / k0_record["prediction"], allow_pickle=False) as archive:
                k0 = np.asarray(archive["mean"], dtype=np.float64)
            rows.append(
                {
                    "tile_id": tile_id,
                    "seed": seed,
                    "selector": "shared-k000",
                    **diagnostic.evaluate_prediction(k0, cases[tile_id], calibration),
                }
            )
            for selector in DETAIL_SELECTORS:
                record = by_key[(tile_id, selector)]
                path = project / record["prediction"]
                if file_sha256(path) != record["sha256"]:
                    raise ValueError("external prediction hash differs")
                with np.load(path, allow_pickle=False) as archive:
                    sparse = np.asarray(archive["mean"], dtype=np.float64)
                    context = np.asarray(
                        archive["context_global_indices"], dtype=np.int64
                    )
                arrays = cases[tile_id]["arrays"]
                projected, projection = v34.project_lowpass_shape(
                    sparse,
                    k0,
                    np.asarray(arrays["valid"], dtype=bool),
                    float(cases[tile_id]["pixel_size_m"]),
                )
                corrected, local = v35.impose_local_laser_consistency(
                    projected, context, arrays
                )
                rows.append(
                    {
                        "tile_id": tile_id,
                        "seed": seed,
                        "selector": selector,
                        **diagnostic.evaluate_prediction(
                            corrected, cases[tile_id], calibration
                        ),
                    }
                )
                mechanism_rows.append({**projection, **local})
                print(f"EXTERNAL_EVAL {seed} {tile_id} {selector}", flush=True)
    result = aggregate_external(rows, gates, contract, cases)
    payload = {
        "schema": "mars-sparse-detail-v36-external-result/v1",
        "state": result["state"],
        "one_shot": True,
        "result": result,
        "mechanism_summary": {
            key: float(np.mean([row[key] for row in mechanism_rows]))
            for key in (
                "projection_rms_m",
                "context_rmse_before_m",
                "context_rmse_after_m",
                "local_correction_rms_m",
            )
        },
        "bindings": {
            "frozen_candidate_sha256": file_sha256(freeze_path),
            "external_case_manifest_sha256": file_sha256(
                args.external_cases.resolve() / "cases-manifest.json"
            ),
            "calibration_sha256": file_sha256(args.calibration.resolve()),
            "gates_sha256": file_sha256(args.gates.resolve()),
            "external_contract_sha256": file_sha256(args.external_contract.resolve()),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "external_files_accessed": True,
        "external_case_count": case_manifest["case_count"],
        "no_post_external_tuning_permitted": True,
    }
    atomic_json(output, payload)
    print(json.dumps({"state": result["state"], "checks": result["checks"]}))
    return 0 if result["state"] == "passed" else 2


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--project-root", type=Path, required=True)
    prepare.add_argument("--frozen-candidate", type=Path, required=True)
    prepare.add_argument("--protocol", type=Path, required=True)
    prepare.add_argument("--prior", type=Path, required=True)
    prepare.add_argument("--case-output", type=Path, required=True)
    prepare.add_argument("--inference-output", type=Path, required=True)
    prepare.add_argument("--plan-output", type=Path, required=True)
    prepare.set_defaults(function=command_prepare)
    infer = commands.add_parser("infer-seed")
    infer.add_argument("--project-root", type=Path, required=True)
    infer.add_argument("--frozen-candidate", type=Path, required=True)
    infer.add_argument("--inference-cases", type=Path, required=True)
    infer.add_argument("--plans", type=Path, required=True)
    infer.add_argument("--seed", type=int, required=True)
    infer.add_argument("--device", default="cuda:0")
    infer.add_argument("--output", type=Path, required=True)
    infer.set_defaults(function=command_infer)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--project-root", type=Path, required=True)
    evaluate.add_argument("--frozen-candidate", type=Path, required=True)
    evaluate.add_argument("--external-cases", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--calibration", type=Path, required=True)
    evaluate.add_argument("--gates", type=Path, required=True)
    evaluate.add_argument("--external-contract", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.set_defaults(function=command_evaluate)
    return root


def main() -> int:
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())

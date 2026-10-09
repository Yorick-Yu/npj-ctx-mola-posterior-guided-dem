#!/usr/bin/env python3
"""K0/K20 validation diagnostic for the predefined v32 snapshots."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_sparse_detail_v30 as evaluator
import npj_real_pipeline_v3 as base
import run_sparse_detail_inference_v30 as inference
from paper2dem.detail_metrics import DetailThresholdCalibration, multiscale_detail_metrics
from paper2dem.sparse_detail_runtime import (
    DETAIL_SELECTORS,
    PRIOR_LENGTHSCALE_PX,
    PRIOR_NOISE_M,
    file_sha256,
)


MODEL_SEEDS = (20260808, 20260809, 20260810)
METRICS = (
    "lowpass_rmse_m",
    "mean_detail_rmse_m",
    "mean_false_detail_fraction_valid",
    "heldout_mola_rmse_m",
)


def mean(rows: list[dict[str, Any]], metric: str) -> float:
    return float(np.mean([float(row[metric]) for row in rows]))


def reduction(baseline: float, candidate: float) -> float:
    return float((baseline - candidate) / baseline)


def bootstrap(values: np.ndarray, replicates: int = 20000) -> list[float]:
    generator = np.random.default_rng(20260811)
    indices = generator.integers(0, len(values), size=(replicates, len(values)))
    estimates = np.mean(values[indices], axis=1)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def evaluate_prediction(
    prediction: np.ndarray,
    evaluation_case: dict[str, Any],
    calibration: DetailThresholdCalibration,
) -> dict[str, float]:
    arrays = evaluation_case["arrays"]
    valid = np.asarray(arrays["valid"], dtype=bool)
    reference = np.asarray(arrays["dem"], dtype=np.float64)
    baseline = np.asarray(arrays["z0"], dtype=np.float64)
    aligned_prediction, _ = evaluator._aligned(prediction, reference, valid)
    aligned_baseline, _ = evaluator._aligned(baseline, reference, valid)
    detail = multiscale_detail_metrics(
        aligned_prediction,
        reference,
        valid,
        pixel_size_m=float(evaluation_case["pixel_size_m"]),
        calibration=calibration,
        baseline=aligned_baseline,
    )
    heldout = np.flatnonzero(np.asarray(arrays["evaluation"], dtype=bool))
    heldout_error = base.sample_raster(
        prediction, arrays["row"][heldout], arrays["column"][heldout]
    ) - arrays["elevation"][heldout]
    return {
        "lowpass_rmse_m": float(detail["lowpass_rmse_m"]),
        "mean_detail_rmse_m": float(detail["mean_detail_rmse_m"]),
        "mean_false_detail_fraction_valid": float(
            detail["mean_false_detail_fraction_valid"]
        ),
        "heldout_mola_rmse_m": evaluator._rmse(heldout_error),
    }


def summarize(rows: list[dict[str, Any]], gates: dict[str, Any]) -> dict[str, Any]:
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
                **{metric: mean(members, metric) for metric in METRICS},
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
                **{metric: mean(members, metric) for metric in METRICS},
            }
        )

    baseline_rows = [row for row in tile_rows if row["family"] == "shared-k000"]
    candidate_rows = [row for row in tile_rows if row["family"] != "shared-k000"]
    if len(baseline_rows) != 4 or len(candidate_rows) != 20:
        raise ValueError("snapshot K0/K20 family closure differs")
    baseline = {metric: mean(baseline_rows, metric) for metric in METRICS}
    candidate = {metric: mean(candidate_rows, metric) for metric in METRICS}
    detail_reduction = reduction(
        baseline["mean_detail_rmse_m"], candidate["mean_detail_rmse_m"]
    )
    false_reduction = reduction(
        baseline["mean_false_detail_fraction_valid"],
        candidate["mean_false_detail_fraction_valid"],
    )
    mola_reduction = reduction(
        baseline["heldout_mola_rmse_m"], candidate["heldout_mola_rmse_m"]
    )
    lowpass_ratio = candidate["lowpass_rmse_m"] / baseline["lowpass_rmse_m"]
    tile_improvements = []
    for base_row in baseline_rows:
        members = [
            row
            for row in candidate_rows
            if row["tile_id"] == base_row["tile_id"]
        ]
        tile_improvements.append(
            reduction(
                base_row["mean_detail_rmse_m"],
                mean(members, "mean_detail_rmse_m"),
            )
        )
    tile_values = np.asarray(tile_improvements, dtype=np.float64)
    interval = bootstrap(tile_values)
    wins = float(np.mean(tile_values > 0))
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
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--validation-cases", type=Path, required=True)
    parser.add_argument("--inference-cases", type=Path, required=True)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--update", type=int, action="append", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite snapshot diagnostic: {output}")
    project = args.project_root.resolve()
    protocol = evaluator.read_json(args.protocol.resolve())
    cases = evaluator._load_evaluation_cases(
        project, args.validation_cases.resolve(), protocol
    )
    inference_cases = inference._load_inference_cases(args.inference_cases.resolve())
    plans = inference._load_plans(args.plans.resolve())
    calibration = DetailThresholdCalibration.from_dict(
        evaluator.read_json(args.calibration.resolve())
    )
    gate_contract = evaluator.read_json(args.gates.resolve())
    thresholds = gate_contract["development_gates"]
    torch, _, _ = base.torch_components()
    device = torch.device(args.device)

    fields: dict[tuple[str, str], tuple[Any, tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = {}
    for tile_id, case in inference_cases.items():
        model_case = inference._model_case(case)
        fields[(tile_id, "shared-k000")] = (
            model_case,
            inference._empty_fields(model_case),
            np.empty(0, dtype=np.int64),
        )
        for selector in DETAIL_SELECTORS:
            global_context = plans[tile_id][selector][:20]
            local_context = inference._remap_context(case, global_context)
            fields[(tile_id, selector)] = (
                model_case,
                base.build_gp_fields(
                    model_case,
                    local_context,
                    lengthscale_px=PRIOR_LENGTHSCALE_PX,
                    noise_m=PRIOR_NOISE_M,
                ),
                local_context,
            )

    results = {}
    checkpoint_bindings = []
    for update in args.update:
        rows = []
        for seed in MODEL_SEEDS:
            checkpoint_path = (
                args.training_root.resolve()
                / f"seed-{seed}"
                / f"snapshot-{update}.pt"
            )
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            architecture = checkpoint["architecture"]
            model = base.make_model(
                base_channels=int(architecture["base_channels"]),
                steps=int(architecture["transport_steps"]),
                minimum_mobility=float(architecture["minimum_mobility"]),
            ).to(device)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
            checkpoint_bindings.append(
                {
                    "update": update,
                    "seed": seed,
                    "path": str(checkpoint_path),
                    "sha256": file_sha256(checkpoint_path),
                }
            )
            for tile_id in sorted(inference_cases):
                for selector in ("shared-k000", *DETAIL_SELECTORS):
                    model_case, gp_fields, context = fields[(tile_id, selector)]
                    prediction = base.predict_from_fields(
                        model, model_case, *gp_fields, context, device
                    ).astype(np.float64)
                    metrics = evaluate_prediction(
                        prediction, cases[tile_id], calibration
                    )
                    rows.append(
                        {
                            "tile_id": tile_id,
                            "seed": seed,
                            "selector": selector,
                            **metrics,
                        }
                    )
                    print(
                        f"SNAPSHOT {update} SEED {seed} {tile_id} {selector}",
                        flush=True,
                    )
            del model
            torch.cuda.empty_cache()
        results[str(update)] = summarize(rows, thresholds)

    payload = {
        "schema": "mars-sparse-detail-snapshot-diagnostic/v1",
        "state": "complete",
        "diagnostic_only": True,
        "formal_external_evidence": False,
        "updates": sorted(args.update),
        "results": results,
        "checkpoint_bindings": checkpoint_bindings,
        "bindings": {
            "protocol_sha256": file_sha256(args.protocol.resolve()),
            "calibration_sha256": file_sha256(args.calibration.resolve()),
            "gates_sha256": file_sha256(args.gates.resolve()),
            "plan_manifest_sha256": file_sha256(
                args.plans.resolve() / "plans-manifest.json"
            ),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "external_files_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    print(json.dumps({key: value["state"] for key, value in results.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

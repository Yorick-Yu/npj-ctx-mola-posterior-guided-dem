#!/usr/bin/env python3
"""Calibrate and evaluate the frozen third-round sparse-detail matrix."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import npj_real_pipeline_v3 as base
from paper2dem.detail_metrics import (
    DEFAULT_DETAIL_SCALES_M,
    DEFAULT_LOWPASS_SCALE_M,
    DetailThresholdCalibration,
    calibrate_detail_thresholds,
    multiscale_detail_metrics,
)
from paper2dem.sparse_detail_runtime import DETAIL_BUDGETS, DETAIL_SELECTORS, file_sha256


MODEL_SEEDS = (20260808, 20260809, 20260810)
METRICS = (
    "effective_laser_count",
    "aligned_global_rmse_m",
    "lowpass_rmse_m",
    "mean_detail_rmse_m",
    "mean_detail_rmse_improvement_fraction",
    "mean_detail_support_f1",
    "mean_false_detail_fraction_valid",
    "mean_false_detail_energy_fraction",
    "heldout_mola_rmse_m",
    "laser_consistency_rmse_m",
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError(f"inconsistent CSV fields: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _calibration_fields(project: Path, index: dict[str, Any]) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    for sample in index["samples"]:
        path = project / sample["case_relative_path"]
        if file_sha256(path) != sample["case_sha256"]:
            raise ValueError(f"training calibration case hash differs: {sample['tile_id']}")
        with np.load(path, allow_pickle=False) as archive:
            yield (
                np.asarray(archive["dem"], dtype=np.float64),
                np.asarray(archive["valid"], dtype=bool),
            )


def command_calibrate(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    if output.exists():
        existing = read_json(output)
        if existing.get("source") != "expanded_development_training_references_only":
            raise FileExistsError(f"another calibration already exists: {output}")
        print(json.dumps({"state": "complete", "reused": True}))
        return 0
    project = args.project_root.resolve()
    index_path = args.training_index.resolve()
    metric_path = args.metric_contract.resolve()
    index = read_json(index_path)
    contract = read_json(metric_path)
    scales = tuple(float(value) for value in contract["detail_scales_m"])
    calibration = calibrate_detail_thresholds(
        _calibration_fields(project, index),
        pixel_size_m=float(args.nominal_pixel_size_m),
        scales_m=scales,
        quantile=float(contract["detail_threshold_quantile"]),
        maximum_samples_per_field=int(args.maximum_samples_per_tile),
        seed=int(args.seed),
    )
    payload = {
        **calibration.to_dict(),
        "state": "complete_before_formal_inference",
        "source": "expanded_development_training_references_only",
        "training_index_sha256": file_sha256(index_path),
        "metric_contract_sha256": file_sha256(metric_path),
        "training_tile_count": int(index["training_tile_count"]),
        "validation_targets_read": False,
        "external_files_accessed": False,
    }
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "state": payload["state"],
                "thresholds_m": payload["thresholds_m"],
                "sample_count_by_scale": payload["sample_count_by_scale"],
            }
        )
    )
    return 0


def _physical_pixel_size(transform: np.ndarray) -> float:
    values = tuple(float(value) for value in transform)
    x = math.hypot(values[0], values[3])
    y = math.hypot(values[1], values[4])
    if not np.isfinite(x + y) or min(x, y) <= 0:
        raise ValueError("invalid source transform")
    return float(math.sqrt(x * y))


def _load_evaluation_cases(
    project: Path, case_root: Path, protocol: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    manifest = read_json(case_root / "cases-manifest.json")
    protocol_records = {
        str(record["tile_id"]): record
        for record in protocol["tiles"]
        if record["role"] == "validation"
    }
    output = {}
    for record in manifest["cases"]:
        tile_id = str(record["tile_id"])
        if tile_id not in protocol_records or record["role"] != "validation":
            raise PermissionError(f"non-validation tile entered evaluation: {tile_id}")
        path = case_root / record["case"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"evaluation case hash differs: {tile_id}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
        source_path = project / "data" / protocol_records[tile_id]["source_relative_path"]
        if file_sha256(source_path) != protocol_records[tile_id]["source_sha256"]:
            raise ValueError(f"validation source hash differs: {tile_id}")
        with np.load(source_path, allow_pickle=False) as source:
            pixel_size = _physical_pixel_size(np.asarray(source["transform"]))
        output[tile_id] = {
            "region": record["region"],
            "arrays": arrays,
            "pixel_size_m": pixel_size,
        }
    if set(output) != set(protocol_records):
        raise ValueError("evaluation cases and frozen validation protocol do not close")
    return output


def _prediction_records(
    project: Path,
    matrix: dict[str, Any],
    receipt_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    receipt_records = {}
    receipts = []
    for seed in MODEL_SEEDS:
        path = receipt_root / f"seed-{seed}-receipt.json"
        receipt = read_json(path)
        if not (
            receipt.get("state") == "complete"
            and int(receipt.get("seed", -1)) == seed
            and receipt.get("reference_dem_read") is False
            and receipt.get("heldout_mola_read") is False
            and receipt.get("external_files_accessed") is False
            and int(receipt.get("matrix_prediction_count", -1)) == 140
            and int(receipt.get("shared_k0_prediction_count", -1)) == 4
        ):
            raise ValueError(f"inference receipt is invalid for seed {seed}")
        receipts.append({"path": path, "sha256": file_sha256(path)})
        for record in receipt["predictions"]:
            cell_id = str(record["cell_id"])
            if cell_id in receipt_records:
                raise ValueError(f"duplicate receipt cell: {cell_id}")
            prediction = project / record["prediction_relative_path"]
            if file_sha256(prediction) != record["prediction_sha256"]:
                raise ValueError(f"prediction hash differs: {cell_id}")
            receipt_records[cell_id] = record
    matrix_records = {str(cell["cell_id"]): cell for cell in matrix["cells"]}
    for cell in matrix["shared_k0_cells"]:
        seed = int(cell["model_seed"])
        cell_id = f"{cell['tile_id']}__shared-k000__s{seed}"
        matrix_records[cell_id] = {
            **cell,
            "cell_id": cell_id,
            "selector_instance": "shared-k000",
            "selector_family": "shared-k000",
            "budget": 0,
        }
    if set(matrix_records) != set(receipt_records) or len(matrix_records) != 432:
        raise ValueError("matrix and inference receipt closure differs")
    return receipts, {
        cell_id: {**matrix_records[cell_id], **receipt_records[cell_id]}
        for cell_id in sorted(matrix_records)
    }


def _aligned(values: np.ndarray, reference: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, float]:
    offset = float(np.median((values - reference)[valid]))
    return values - offset, offset


def _rmse(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("RMSE requires finite samples")
    return float(np.sqrt(np.mean(finite**2)))


def _family(selector: str) -> str:
    return "random" if selector.startswith("random-") else selector


def _evaluate_cell(
    project: Path,
    cell: dict[str, Any],
    case: dict[str, Any],
    calibration: DetailThresholdCalibration,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = project / cell["prediction_relative_path"]
    with np.load(path, allow_pickle=False) as archive:
        prediction = np.asarray(archive["mean"], dtype=np.float64)
        context = np.asarray(archive["context_global_indices"], dtype=np.int64)
        metadata = json.loads(str(archive["metadata_json"].item()))
    arrays = case["arrays"]
    valid = np.asarray(arrays["valid"], dtype=bool)
    reference = np.asarray(arrays["dem"], dtype=np.float64)
    baseline = np.asarray(arrays["z0"], dtype=np.float64)
    if prediction.shape != reference.shape or not np.all(np.isfinite(prediction[valid])):
        raise FloatingPointError(f"invalid prediction array: {cell['cell_id']}")
    if metadata["cell_id"] != cell["cell_id"]:
        raise ValueError(f"prediction metadata differs: {cell['cell_id']}")
    if len(context) > int(cell["budget"]) or (int(cell["budget"]) == 0) != (len(context) == 0):
        raise ValueError(f"selected count exceeds or contradicts budget: {cell['cell_id']}")
    aligned_prediction, vertical_offset = _aligned(prediction, reference, valid)
    aligned_baseline, baseline_offset = _aligned(baseline, reference, valid)
    detail = multiscale_detail_metrics(
        aligned_prediction,
        reference,
        valid,
        pixel_size_m=float(case["pixel_size_m"]),
        calibration=calibration,
        baseline=aligned_baseline,
        lowpass_scale_m=DEFAULT_LOWPASS_SCALE_M,
    )
    evaluation = np.flatnonzero(np.asarray(arrays["evaluation"], dtype=bool))
    heldout_error = base.sample_raster(
        prediction, arrays["row"][evaluation], arrays["column"][evaluation]
    ) - arrays["elevation"][evaluation]
    if context.size:
        selected_error = base.sample_raster(
            prediction, arrays["row"][context], arrays["column"][context]
        ) - arrays["elevation"][context]
        laser_rmse = _rmse(selected_error)
    else:
        laser_rmse = None
    row = {
        "cell_id": cell["cell_id"],
        "tile_id": cell["tile_id"],
        "region": case["region"],
        "selector_instance": cell["selector_instance"],
        "selector_family": _family(cell["selector_instance"]),
        "budget": int(cell["budget"]),
        "model_seed": int(cell["model_seed"]),
        "selected_count": int(context.size),
        "effective_laser_count": int(context.size),
        "candidate_pool_saturated": bool(context.size < int(cell["budget"])),
        "pixel_size_m": float(case["pixel_size_m"]),
        "reference_vertical_offset_removed_m": vertical_offset,
        "baseline_vertical_offset_removed_m": baseline_offset,
        "aligned_global_rmse_m": _rmse((aligned_prediction - reference)[valid]),
        "lowpass_rmse_m": detail["lowpass_rmse_m"],
        "baseline_lowpass_rmse_m": detail["baseline_lowpass_rmse_m"],
        "mean_detail_rmse_m": detail["mean_detail_rmse_m"],
        "mean_detail_rmse_improvement_fraction": detail[
            "mean_detail_rmse_improvement_fraction"
        ],
        "mean_detail_support_f1": detail["mean_detail_support_f1"],
        "mean_false_detail_fraction_valid": detail[
            "mean_false_detail_fraction_valid"
        ],
        "mean_false_detail_energy_fraction": detail[
            "mean_false_detail_energy_fraction"
        ],
        "heldout_mola_rmse_m": _rmse(heldout_error),
        "heldout_mola_count": int(evaluation.size),
        "laser_consistency_rmse_m": laser_rmse,
        "prediction_sha256": cell["prediction_sha256"],
        "prediction_relative_path": cell["prediction_relative_path"],
    }
    scale_rows = []
    for scale in detail["scales"]:
        scale_rows.append(
            {
                "cell_id": row["cell_id"],
                "tile_id": row["tile_id"],
                "region": row["region"],
                "selector_instance": row["selector_instance"],
                "selector_family": row["selector_family"],
                "budget": row["budget"],
                "model_seed": row["model_seed"],
                **scale,
            }
        )
    return row, scale_rows


def _mean(values: Iterable[Any]) -> float | None:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else None


def _aggregate(
    rows: list[dict[str, Any]], keys: tuple[str, ...], *, include_std: bool = False
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for group, members in sorted(groups.items()):
        item = {key: value for key, value in zip(keys, group, strict=True)}
        item["member_count"] = len(members)
        for metric in METRICS:
            value = _mean(member[metric] for member in members)
            item[metric] = value
            if include_std:
                finite = np.asarray(
                    [member[metric] for member in members if member[metric] is not None],
                    dtype=np.float64,
                )
                finite = finite[np.isfinite(finite)]
                item[f"{metric}_std"] = (
                    float(np.std(finite, ddof=1)) if finite.size > 1 else None
                )
        output.append(item)
    return output


def _curve_rows(tile_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["tile_id"], row["selector_family"], int(row["budget"])): row
        for row in tile_rows
    }
    tiles = sorted({row["tile_id"] for row in tile_rows})
    families = ["random", "uniform-farthest", "max-prior-variance", "greedy-mi-point", "terrain-faopt"]
    output = []
    budgets = np.asarray((0, *DETAIL_BUDGETS), dtype=np.float64)
    x = budgets / budgets[-1]
    for tile in tiles:
        k0 = lookup[(tile, "shared-k000", 0)]
        for family in families:
            rows = [k0] + [lookup[(tile, family, budget)] for budget in DETAIL_BUDGETS]
            detail = np.asarray([row["mean_detail_rmse_m"] for row in rows])
            false_detail = np.asarray(
                [row["mean_false_detail_fraction_valid"] for row in rows]
            )
            support = np.asarray([row["mean_detail_support_f1"] for row in rows])
            lowpass = np.asarray([row["lowpass_rmse_m"] for row in rows])
            mola = np.asarray([row["heldout_mola_rmse_m"] for row in rows])
            detail_gain = (detail[0] - detail) / max(detail[0], 1.0e-12)
            target = detail_gain[-1] * 0.9
            reached = [int(k) for k, value in zip(budgets[1:], detail_gain[1:]) if value >= target]
            output.append(
                {
                    "tile_id": tile,
                    "selector_family": family,
                    "detail_retention_auc": float(np.trapezoid(detail_gain, x)),
                    "detail_support_f1_auc": float(np.trapezoid(support, x)),
                    "false_detail_fraction_auc": float(np.trapezoid(false_detail, x)),
                    "lowpass_rmse_auc_m": float(np.trapezoid(lowpass, x)),
                    "heldout_mola_rmse_auc_m": float(np.trapezoid(mola, x)),
                    "budget_to_90pct_k80_detail_gain": min(reached) if reached else None,
                }
            )
    return output


def _bootstrap_ci(values: np.ndarray, seed: int, replicates: int = 20000) -> tuple[float, float]:
    if values.ndim != 1 or values.size < 2:
        raise ValueError("tile bootstrap requires at least two values")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.size, size=(replicates, values.size))
    means = np.mean(values[indices], axis=1)
    return tuple(map(float, np.quantile(means, (0.025, 0.975))))


def _comparisons(tile_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {
        (row["tile_id"], row["selector_family"], int(row["budget"])): row
        for row in tile_rows
    }
    families = ("uniform-farthest", "max-prior-variance", "greedy-mi-point", "terrain-faopt")
    orientations = {
        "detail_rmse_reduction_m": ("mean_detail_rmse_m", -1.0),
        "detail_support_f1_gain": ("mean_detail_support_f1", 1.0),
        "false_detail_fraction_reduction": ("mean_false_detail_fraction_valid", -1.0),
        "false_detail_energy_reduction": ("mean_false_detail_energy_fraction", -1.0),
        "lowpass_rmse_reduction_m": ("lowpass_rmse_m", -1.0),
        "heldout_mola_rmse_reduction_m": ("heldout_mola_rmse_m", -1.0),
        "laser_consistency_rmse_reduction_m": ("laser_consistency_rmse_m", -1.0),
    }
    tile_output = []
    for tile in sorted({row["tile_id"] for row in tile_rows}):
        for budget in DETAIL_BUDGETS:
            random_row = lookup[(tile, "random", budget)]
            for family in families:
                candidate = lookup[(tile, family, budget)]
                item = {"tile_id": tile, "selector_family": family, "budget": budget}
                for output_name, (metric, sign) in orientations.items():
                    left = candidate[metric]
                    right = random_row[metric]
                    item[output_name] = (
                        None if left is None or right is None else sign * (left - right)
                    )
                tile_output.append(item)
    aggregate = []
    for family in families:
        for budget in DETAIL_BUDGETS:
            rows = [
                row
                for row in tile_output
                if row["selector_family"] == family and row["budget"] == budget
            ]
            item = {"selector_family": family, "budget": budget, "tile_count": len(rows)}
            for index, output_name in enumerate(orientations):
                values = np.asarray(
                    [row[output_name] for row in rows if row[output_name] is not None],
                    dtype=np.float64,
                )
                item[f"{output_name}_mean"] = float(np.mean(values)) if values.size else None
                if values.size >= 2:
                    low, high = _bootstrap_ci(
                        values,
                        seed=20260810 + budget * 31 + index * 101 + sum(map(ord, family)),
                    )
                    item[f"{output_name}_ci_low"] = low
                    item[f"{output_name}_ci_high"] = high
                else:
                    item[f"{output_name}_ci_low"] = None
                    item[f"{output_name}_ci_high"] = None
            aggregate.append(item)
    return tile_output, aggregate


def command_evaluate(args: argparse.Namespace) -> int:
    project = args.project_root.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"formal evaluation output is nonempty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    matrix_path = args.matrix.resolve()
    protocol_path = args.protocol.resolve()
    calibration_path = args.detail_calibration.resolve()
    frozen_path = args.frozen_candidate.resolve()
    matrix = read_json(matrix_path)
    protocol = read_json(protocol_path)
    frozen = read_json(frozen_path)
    if not (
        frozen.get("state") == "frozen_before_inference"
        and frozen.get("matrix_sha256") == file_sha256(matrix_path)
        and frozen.get("protocol_sha256") == file_sha256(protocol_path)
    ):
        raise PermissionError("formal candidate bindings differ")
    unhashed = dict(frozen)
    declared = unhashed.pop("content_sha256", None)
    if declared != hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest():
        raise ValueError("frozen candidate content hash differs")
    for record in frozen["code"]:
        if file_sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"frozen code hash differs: {record['path']}")
    calibration_payload = read_json(calibration_path)
    if frozen["detail_calibration_sha256"] != file_sha256(calibration_path):
        raise ValueError("frozen detail calibration hash differs")
    calibration = DetailThresholdCalibration.from_dict(calibration_payload)
    cases = _load_evaluation_cases(
        project, args.validation_cases.resolve(), protocol
    )
    receipts, records = _prediction_records(
        project, matrix, args.receipt_root.resolve()
    )
    plan_root = args.plans.resolve()
    plan_manifest = read_json(plan_root / "plans-manifest.json")
    plans = {}
    for record in plan_manifest["plans"]:
        if record["role"] != "validation":
            continue
        with np.load(plan_root / record["plan"], allow_pickle=False) as archive:
            plans[record["tile_id"]] = {
                selector: np.asarray(archive[selector], dtype=np.int64)
                for selector in DETAIL_SELECTORS
            }
    cell_rows = []
    scale_rows = []
    nested_prefix_passed = True
    for number, cell_id in enumerate(sorted(records), start=1):
        cell = records[cell_id]
        row, scales = _evaluate_cell(
            project, cell, cases[cell["tile_id"]], calibration
        )
        if row["budget"] > 0:
            with np.load(project / row["prediction_relative_path"], allow_pickle=False) as archive:
                context = np.asarray(archive["context_global_indices"], dtype=np.int64)
            expected = plans[row["tile_id"]][row["selector_instance"]][
                : row["budget"]
            ]
            nested_prefix_passed &= np.array_equal(context, expected)
        cell_rows.append(row)
        scale_rows.extend(scales)
        print(f"METRIC {number}/432 {cell_id}", flush=True)
    if not nested_prefix_passed:
        raise ValueError("prediction context differs from a frozen nested selector prefix")
    finite_failures = []
    for row in cell_rows:
        for metric in METRICS:
            value = row[metric]
            if metric == "laser_consistency_rmse_m" and row["budget"] == 0:
                continue
            if value is None or not np.isfinite(float(value)):
                finite_failures.append({"cell_id": row["cell_id"], "metric": metric})
    if finite_failures:
        raise FloatingPointError(f"formal metrics are nonfinite: {finite_failures[:3]}")
    saturated_cells = [
        row["cell_id"] for row in cell_rows if row["candidate_pool_saturated"]
    ]
    instance_rows = _aggregate(
        cell_rows,
        ("tile_id", "region", "selector_family", "budget", "model_seed"),
    )
    tile_rows = _aggregate(
        instance_rows,
        ("tile_id", "region", "selector_family", "budget"),
        include_std=True,
    )
    aggregate_rows = _aggregate(
        tile_rows, ("selector_family", "budget"), include_std=True
    )
    curve_rows = _curve_rows(tile_rows)
    curve_aggregate = []
    curve_metrics = (
        "detail_retention_auc",
        "detail_support_f1_auc",
        "false_detail_fraction_auc",
        "lowpass_rmse_auc_m",
        "heldout_mola_rmse_auc_m",
        "budget_to_90pct_k80_detail_gain",
    )
    for family in sorted({row["selector_family"] for row in curve_rows}):
        members = [row for row in curve_rows if row["selector_family"] == family]
        item = {"selector_family": family, "tile_count": len(members)}
        for metric in curve_metrics:
            values = np.asarray(
                [row[metric] for row in members if row[metric] is not None], dtype=np.float64
            )
            item[f"{metric}_mean"] = float(np.mean(values)) if values.size else None
            item[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if values.size > 1 else None
            )
        curve_aggregate.append(item)
    comparison_tiles, comparison_aggregate = _comparisons(tile_rows)
    outputs = {
        "cell-metrics.csv": cell_rows,
        "scale-metrics.csv": scale_rows,
        "instance-aggregated-metrics.csv": instance_rows,
        "tile-family-budget-metrics.csv": tile_rows,
        "aggregate-family-budget-metrics.csv": aggregate_rows,
        "tile-budget-curves.csv": curve_rows,
        "aggregate-budget-curves.csv": curve_aggregate,
        "tile-selector-vs-random.csv": comparison_tiles,
        "aggregate-selector-vs-random.csv": comparison_aggregate,
    }
    for name, rows in outputs.items():
        write_csv(output / name, rows)
    best_by_budget = []
    for budget in DETAIL_BUDGETS:
        rows = [
            row
            for row in aggregate_rows
            if int(row["budget"]) == budget and row["selector_family"] != "shared-k000"
        ]
        best_by_budget.append(
            {
                "budget": budget,
                "lowest_detail_rmse_family": min(
                    rows, key=lambda row: row["mean_detail_rmse_m"]
                )["selector_family"],
                "lowest_false_detail_family": min(
                    rows, key=lambda row: row["mean_false_detail_fraction_valid"]
                )["selector_family"],
                "lowest_heldout_mola_rmse_family": min(
                    rows, key=lambda row: row["heldout_mola_rmse_m"]
                )["selector_family"],
            }
        )
    audit = {
        "schema": "mars-sparse-detail-formal-audit/v1",
        "state": "complete",
        "matrix_cell_count_expected": 420,
        "matrix_cell_count_observed": sum(row["budget"] > 0 for row in cell_rows),
        "shared_k0_count_expected": 12,
        "shared_k0_count_observed": sum(row["budget"] == 0 for row in cell_rows),
        "prediction_count": len(cell_rows),
        "all_metrics_finite": not finite_failures,
        "nested_selector_prefixes_match": nested_prefix_passed,
        "budget_semantics": "maximum permitted measurements; use all candidates when fewer are available",
        "candidate_pool_saturated_cell_count": len(saturated_cells),
        "candidate_pool_saturated_cells": saturated_cells,
        "random_instances_averaged_before_tile_aggregation": True,
        "model_seeds_averaged_before_tile_aggregation": True,
        "tile_equal_aggregation": True,
        "tile_cluster_bootstrap_replicates": 20000,
        "reference_alignment": (
            "one valid-mask median vertical offset removed for reference-terrain metrics; "
            "heldout and selected MOLA metrics remain in the absolute measurement frame"
        ),
        "global_gp_superiority_evaluated": False,
        "external_files_accessed": False,
        "validation_tile_count": len(cases),
        "selector_instances": list(DETAIL_SELECTORS),
        "selector_families": [
            "random",
            "uniform-farthest",
            "max-prior-variance",
            "greedy-mi-point",
            "terrain-faopt",
        ],
        "budgets": list(DETAIL_BUDGETS),
        "model_seeds": list(MODEL_SEEDS),
        "bindings": {
            "matrix_sha256": file_sha256(matrix_path),
            "protocol_sha256": file_sha256(protocol_path),
            "frozen_candidate_sha256": file_sha256(frozen_path),
            "detail_calibration_sha256": file_sha256(calibration_path),
            "plan_manifest_sha256": file_sha256(plan_root / "plans-manifest.json"),
            "inference_receipts": [
                {"path": item["path"].name, "sha256": item["sha256"]}
                for item in receipts
            ],
        },
    }
    atomic_json(output / "formal-audit.json", audit)
    summary = {
        "schema": "mars-sparse-detail-formal-results/v1",
        "state": "complete",
        "primary_claim": read_json(args.metric_contract.resolve())["primary_claim"],
        "interpretation_policy": (
            "raw metrics and paired tile-cluster intervals are reported; no unregistered "
            "pass threshold or global-GP superiority requirement is introduced"
        ),
        "counts": {
            "validation_tiles": len(cases),
            "matrix_predictions": 420,
            "shared_k0_predictions": 12,
            "metric_cells": len(cell_rows),
        },
        "best_family_by_budget": best_by_budget,
        "outputs": {},
    }
    for path in sorted(output.glob("*.csv")):
        summary["outputs"][path.name] = {
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
    summary["outputs"]["formal-audit.json"] = {
        "sha256": file_sha256(output / "formal-audit.json"),
        "bytes": (output / "formal-audit.json").stat().st_size,
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--project-root", type=Path, required=True)
    calibrate.add_argument("--training-index", type=Path, required=True)
    calibrate.add_argument("--metric-contract", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--nominal-pixel-size-m", type=float, default=20.0)
    calibrate.add_argument("--maximum-samples-per-tile", type=int, default=4096)
    calibrate.add_argument("--seed", type=int, default=20260810)
    calibrate.set_defaults(function=command_calibrate)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--project-root", type=Path, required=True)
    evaluate.add_argument("--matrix", type=Path, required=True)
    evaluate.add_argument("--protocol", type=Path, required=True)
    evaluate.add_argument("--metric-contract", type=Path, required=True)
    evaluate.add_argument("--detail-calibration", type=Path, required=True)
    evaluate.add_argument("--frozen-candidate", type=Path, required=True)
    evaluate.add_argument("--validation-cases", type=Path, required=True)
    evaluate.add_argument("--plans", type=Path, required=True)
    evaluate.add_argument("--receipt-root", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.set_defaults(function=command_evaluate)
    return root


def main() -> int:
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())

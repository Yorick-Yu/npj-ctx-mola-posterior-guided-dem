#!/usr/bin/env python3
"""Evaluate K0 low-pass anchoring followed by local laser consistency."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import diagnose_sparse_detail_snapshots_v32 as diagnostic
import evaluate_lowpass_projection_v34 as v34
import evaluate_sparse_detail_v30 as evaluator
import npj_real_pipeline_v3 as base
from paper2dem.detail_metrics import DetailThresholdCalibration
from paper2dem.sparse_detail_runtime import file_sha256


MODEL_SEEDS = (20260808, 20260809, 20260810)
PRIMARY_BUDGET = 20
LOCAL_LENGTHSCALE_PX = 16.0
LOCAL_NOISE_M = 8.0


def read_prediction_and_context(
    project: Path, relative_path: str
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(project / relative_path, allow_pickle=False) as archive:
        return (
            np.asarray(archive["mean"], dtype=np.float64),
            np.asarray(archive["context_global_indices"], dtype=np.int64),
        )


def impose_local_laser_consistency(
    prediction: np.ndarray,
    context: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, float]]:
    rows = np.asarray(arrays["row"][context], dtype=np.float64)
    columns = np.asarray(arrays["column"][context], dtype=np.float64)
    observed = np.asarray(arrays["elevation"][context], dtype=np.float64)
    before = observed - base.sample_raster(prediction, rows, columns)
    grid_row, grid_column = np.indices(prediction.shape, dtype=np.float32)
    correction, _ = base.gp_predict(
        rows,
        columns,
        before,
        grid_row.ravel(),
        grid_column.ravel(),
        lengthscale_px=LOCAL_LENGTHSCALE_PX,
        noise_m=LOCAL_NOISE_M,
    )
    correction = correction.reshape(prediction.shape)
    output = prediction + correction
    after = observed - base.sample_raster(output, rows, columns)
    return output, {
        "context_rmse_before_m": float(np.sqrt(np.mean(before**2))),
        "context_rmse_after_m": float(np.sqrt(np.mean(after**2))),
        "local_correction_rms_m": float(np.sqrt(np.mean(correction**2))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--validation-cases", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v35 diagnostic: {output}")
    project = args.project_root.resolve()
    protocol = evaluator.read_json(args.protocol.resolve())
    cases = evaluator._load_evaluation_cases(  # noqa: SLF001
        project, args.validation_cases.resolve(), protocol
    )
    calibration = DetailThresholdCalibration.from_dict(
        evaluator.read_json(args.calibration.resolve())
    )
    thresholds = evaluator.read_json(args.gates.resolve())["development_gates"]
    matrix = evaluator.read_json(args.matrix.resolve())
    k0_records = {
        (str(record["tile_id"]), int(record["model_seed"])): record
        for record in matrix["shared_k0_cells"]
    }
    primary_records = [
        record for record in matrix["cells"] if int(record["budget"]) == PRIMARY_BUDGET
    ]
    if len(primary_records) != len(cases) * len(MODEL_SEEDS) * 7:
        raise ValueError("v35 K20 matrix closure differs")

    rows: list[dict[str, Any]] = []
    mechanism_rows = []
    for tile_id in sorted(cases):
        for seed in MODEL_SEEDS:
            k0, _ = read_prediction_and_context(
                project, k0_records[(tile_id, seed)]["prediction_relative_path"]
            )
            rows.append(
                {
                    "tile_id": tile_id,
                    "seed": seed,
                    "selector": "shared-k000",
                    **diagnostic.evaluate_prediction(k0, cases[tile_id], calibration),
                }
            )

    for number, record in enumerate(primary_records, start=1):
        tile_id = str(record["tile_id"])
        seed = int(record["model_seed"])
        sparse, context = read_prediction_and_context(
            project, record["prediction_relative_path"]
        )
        k0, _ = read_prediction_and_context(
            project, k0_records[(tile_id, seed)]["prediction_relative_path"]
        )
        valid = np.asarray(cases[tile_id]["arrays"]["valid"], dtype=bool)
        projected, projection = v34.project_lowpass_shape(
            sparse, k0, valid, float(cases[tile_id]["pixel_size_m"])
        )
        corrected, local = impose_local_laser_consistency(
            projected, context, cases[tile_id]["arrays"]
        )
        rows.append(
            {
                "tile_id": tile_id,
                "seed": seed,
                "selector": str(record["selector_instance"]),
                **diagnostic.evaluate_prediction(
                    corrected, cases[tile_id], calibration
                ),
            }
        )
        mechanism_rows.append({**projection, **local})
        print(f"LOCAL {number}/{len(primary_records)} {record['cell_id']}", flush=True)

    result = diagnostic.summarize(rows, thresholds)
    summary_keys = (
        "projection_rms_m",
        "context_rmse_before_m",
        "context_rmse_after_m",
        "local_correction_rms_m",
    )
    payload = {
        "schema": "mars-sparse-detail-local-laser-projection/v1",
        "state": result["state"],
        "diagnostic_only": True,
        "formal_external_evidence": False,
        "mechanism": {
            "primary_budget": PRIMARY_BUDGET,
            "lowpass_projection_scale_m": v34.LOWPASS_SCALE_M,
            "local_laser_lengthscale_px": LOCAL_LENGTHSCALE_PX,
            "local_laser_lengthscale_m": LOCAL_LENGTHSCALE_PX * 20.0,
            "local_laser_noise_m": LOCAL_NOISE_M,
            "datum_offset_preserved": True,
        },
        "result": result,
        "mechanism_summary": {
            key: float(np.mean([row[key] for row in mechanism_rows]))
            for key in summary_keys
        },
        "bindings": {
            "matrix_sha256": file_sha256(args.matrix.resolve()),
            "protocol_sha256": file_sha256(args.protocol.resolve()),
            "calibration_sha256": file_sha256(args.calibration.resolve()),
            "gates_sha256": file_sha256(args.gates.resolve()),
            "contract_sha256": file_sha256(args.contract.resolve()),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "v34_script_sha256": hashlib.sha256(Path(v34.__file__).read_bytes()).hexdigest(),
        },
        "validation_targets_read": True,
        "external_files_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"state": result["state"], "checks": result["checks"]}))
    return 0 if result["state"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

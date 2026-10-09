#!/usr/bin/env python3
"""Evaluate a fixed K0-anchored low-pass projection on open validation data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import diagnose_sparse_detail_snapshots_v32 as diagnostic
import evaluate_sparse_detail_v30 as evaluator
from paper2dem.detail_metrics import (
    DetailThresholdCalibration,
    masked_gaussian_smooth,
)
from paper2dem.sparse_detail_runtime import file_sha256


MODEL_SEEDS = (20260808, 20260809, 20260810)
PRIMARY_BUDGET = 20
LOWPASS_SCALE_M = 320.0


def read_prediction(project: Path, relative_path: str) -> np.ndarray:
    path = project / relative_path
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(archive["mean"], dtype=np.float64)


def project_lowpass_shape(
    sparse: np.ndarray,
    k0: np.ndarray,
    valid: np.ndarray,
    pixel_size_m: float,
) -> tuple[np.ndarray, dict[str, float]]:
    difference = sparse - k0
    datum_offset = float(np.median(difference[valid]))
    centered = difference - datum_offset
    low_shape, support = masked_gaussian_smooth(
        centered,
        valid,
        sigma_pixels=LOWPASS_SCALE_M / pixel_size_m,
    )
    correction = np.where(np.isfinite(low_shape), low_shape, 0.0)
    projected = sparse - correction
    return projected, {
        "preserved_datum_offset_m": datum_offset,
        "projection_rms_m": float(np.sqrt(np.mean(correction[support] ** 2))),
        "projection_support_fraction": float(np.mean(support)),
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
        raise FileExistsError(f"refusing to overwrite v34 diagnostic: {output}")
    project = args.project_root.resolve()
    protocol = evaluator.read_json(args.protocol.resolve())
    cases = evaluator._load_evaluation_cases(  # noqa: SLF001
        project, args.validation_cases.resolve(), protocol
    )
    calibration = DetailThresholdCalibration.from_dict(
        evaluator.read_json(args.calibration.resolve())
    )
    gate_contract = evaluator.read_json(args.gates.resolve())
    thresholds = gate_contract["development_gates"]
    matrix = evaluator.read_json(args.matrix.resolve())

    k0_records = {
        (str(record["tile_id"]), int(record["model_seed"])): record
        for record in matrix["shared_k0_cells"]
    }
    primary_records = [
        record for record in matrix["cells"] if int(record["budget"]) == PRIMARY_BUDGET
    ]
    expected = len(cases) * len(MODEL_SEEDS) * 7
    if len(primary_records) != expected:
        raise ValueError(f"K20 matrix closure differs: {len(primary_records)} != {expected}")

    rows: list[dict[str, Any]] = []
    projection_rows = []
    for tile_id in sorted(cases):
        for seed in MODEL_SEEDS:
            record = k0_records[(tile_id, seed)]
            prediction = read_prediction(project, record["prediction_relative_path"])
            metrics = diagnostic.evaluate_prediction(
                prediction, cases[tile_id], calibration
            )
            rows.append(
                {
                    "tile_id": tile_id,
                    "seed": seed,
                    "selector": "shared-k000",
                    **metrics,
                }
            )

    for number, record in enumerate(primary_records, start=1):
        tile_id = str(record["tile_id"])
        seed = int(record["model_seed"])
        sparse = read_prediction(project, record["prediction_relative_path"])
        k0 = read_prediction(
            project, k0_records[(tile_id, seed)]["prediction_relative_path"]
        )
        projected, projection = project_lowpass_shape(
            sparse,
            k0,
            np.asarray(cases[tile_id]["arrays"]["valid"], dtype=bool),
            float(cases[tile_id]["pixel_size_m"]),
        )
        metrics = diagnostic.evaluate_prediction(projected, cases[tile_id], calibration)
        rows.append(
            {
                "tile_id": tile_id,
                "seed": seed,
                "selector": str(record["selector_instance"]),
                **metrics,
            }
        )
        projection_rows.append(
            {
                "tile_id": tile_id,
                "seed": seed,
                "selector": str(record["selector_instance"]),
                **projection,
            }
        )
        print(f"PROJECT {number}/{len(primary_records)} {record['cell_id']}", flush=True)

    result = diagnostic.summarize(rows, thresholds)
    payload = {
        "schema": "mars-sparse-detail-k0-lowpass-projection/v1",
        "state": result["state"],
        "diagnostic_only": True,
        "formal_external_evidence": False,
        "mechanism": {
            "primary_budget": PRIMARY_BUDGET,
            "lowpass_scale_m": LOWPASS_SCALE_M,
            "projection_strength": 1.0,
            "datum_offset_preserved": True,
            "definition": "K20 minus the masked 320 m Gaussian low-pass shape of K20 minus same-seed K0 after removing its valid-pixel median",
        },
        "result": result,
        "projection_summary": {
            key: float(np.mean([row[key] for row in projection_rows]))
            for key in (
                "preserved_datum_offset_m",
                "projection_rms_m",
                "projection_support_fraction",
            )
        },
        "bindings": {
            "matrix_sha256": file_sha256(args.matrix.resolve()),
            "protocol_sha256": file_sha256(args.protocol.resolve()),
            "calibration_sha256": file_sha256(args.calibration.resolve()),
            "gates_sha256": file_sha256(args.gates.resolve()),
            "contract_sha256": file_sha256(args.contract.resolve()),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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

#!/usr/bin/env python3
"""Freeze and execute the third-round sparse-detail inference matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

import npj_real_pipeline_v3 as base
from paper2dem.sparse_detail_runtime import (
    DETAIL_BUDGETS,
    DETAIL_SELECTORS,
    PRIOR_LENGTHSCALE_PX,
    PRIOR_NOISE_M,
    file_sha256,
)


MODEL_SEEDS = (20260808, 20260809, 20260810)
CHECKPOINT_UPDATES = 2000


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def command_sanitize(args: argparse.Namespace) -> int:
    source_root = args.validation_cases.resolve()
    plan_root = args.plans.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"sanitized validation output is nonempty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_manifest_path = source_root / "cases-manifest.json"
    plan_manifest_path = plan_root / "plans-manifest.json"
    source_manifest = read_json(source_manifest_path)
    plan_manifest = read_json(plan_manifest_path)
    plans_by_tile = {
        str(record["tile_id"]): record
        for record in plan_manifest["plans"]
        if record["role"] == "validation"
    }
    receipts = []
    for record in source_manifest["cases"]:
        tile_id = str(record["tile_id"])
        source_path = source_root / record["case"]
        if file_sha256(source_path) != record["sha256"]:
            raise ValueError(f"validation case hash differs: {tile_id}")
        plan_record = plans_by_tile[tile_id]
        plan_path = plan_root / plan_record["plan"]
        if file_sha256(plan_path) != plan_record["sha256"]:
            raise ValueError(f"validation plan hash differs: {tile_id}")
        with np.load(source_path, allow_pickle=False) as archive:
            assimilation = np.asarray(archive["assimilation"], dtype=bool)
            candidate_global = np.flatnonzero(assimilation).astype(np.int64)
            # This trusted segregation step exports only assimilable measurements.
            payload = {
                "image": np.asarray(archive["image"], dtype=np.float32),
                "z0": np.asarray(archive["z0"], dtype=np.float32),
                "valid": np.asarray(archive["valid"], dtype=bool),
                "candidate_global_index": candidate_global,
                "candidate_row": np.asarray(archive["row"], dtype=np.float32)[
                    candidate_global
                ],
                "candidate_column": np.asarray(archive["column"], dtype=np.float32)[
                    candidate_global
                ],
                "candidate_elevation": np.asarray(
                    archive["elevation"], dtype=np.float32
                )[candidate_global],
            }
        with np.load(plan_path, allow_pickle=False) as plans:
            allowed = set(map(int, candidate_global))
            for selector in DETAIL_SELECTORS:
                order = np.asarray(plans[selector], dtype=np.int64)
                if not set(map(int, order)).issubset(allowed):
                    raise ValueError(f"{tile_id} {selector} exports a held-out point")
        output_path = output / f"{tile_id}.npz"
        atomic_npz(output_path, **payload)
        receipts.append(
            {
                "tile_id": tile_id,
                "region": record["region"],
                "case": output_path.name,
                "sha256": file_sha256(output_path),
                "candidate_count": int(candidate_global.size),
            }
        )
    if set(plans_by_tile) != {record["tile_id"] for record in receipts}:
        raise ValueError("sanitized cases and validation plans do not close")
    manifest = {
        "schema": "mars-sparse-detail-inference-cases/v1",
        "state": "sanitized_before_model_inference",
        "source_case_manifest_sha256": file_sha256(source_manifest_path),
        "plan_manifest_sha256": file_sha256(plan_manifest_path),
        "case_count": len(receipts),
        "exported_arrays": [
            "image",
            "z0",
            "valid",
            "candidate_global_index",
            "candidate_row",
            "candidate_column",
            "candidate_elevation",
        ],
        "reference_dem_exported": False,
        "heldout_mola_exported": False,
        "evaluation_mask_exported": False,
        "external_files_accessed": False,
        "cases": receipts,
    }
    atomic_json(output / "cases-manifest.json", manifest)
    print(json.dumps({"state": manifest["state"], "case_count": len(receipts)}))
    return 0


def _training_record(training_root: Path, seed: int) -> dict[str, Any]:
    directory = training_root / f"seed-{seed}"
    manifest_path = directory / "training-manifest.json"
    checkpoint_path = directory / "final.pt"
    manifest = read_json(manifest_path)
    if not (
        manifest.get("state") == "complete"
        and int(manifest.get("seed", -1)) == seed
        and int(manifest.get("completed_updates", -1)) == CHECKPOINT_UPDATES
        and manifest.get("network_precision") == "fp32"
        and manifest.get("validation_targets_read") is False
        and manifest.get("external_targets_read") is False
        and manifest.get("final_checkpoint_sha256") == file_sha256(checkpoint_path)
    ):
        raise ValueError(f"training seed {seed} does not satisfy the frozen contract")
    return {
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "training_manifest": str(manifest_path),
        "training_manifest_sha256": file_sha256(manifest_path),
        "completed_updates": CHECKPOINT_UPDATES,
        "network_precision": "fp32",
    }


def command_freeze(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    if output.exists():
        existing = read_json(output)
        unhashed = dict(existing)
        declared = unhashed.pop("content_sha256", None)
        if (
            existing.get("state") != "frozen_before_inference"
            or declared != canonical_sha256(unhashed)
        ):
            raise FileExistsError(f"candidate freeze already exists in another state: {output}")
        print(json.dumps({"state": existing["state"], "reused": True}))
        return 0
    matrix_path = args.matrix.resolve()
    protocol_path = args.protocol.resolve()
    metric_path = args.metric_contract.resolve()
    plan_manifest_path = args.plans.resolve() / "plans-manifest.json"
    sanitized_manifest_path = args.inference_cases.resolve() / "cases-manifest.json"
    calibration_path = args.detail_calibration.resolve()
    matrix = read_json(matrix_path)
    sanitized = read_json(sanitized_manifest_path)
    calibration = read_json(calibration_path)
    if tuple(map(int, matrix["model_seeds"])) != MODEL_SEEDS:
        raise ValueError("matrix model seeds differ from the third-round contract")
    if matrix["matrix_cell_count"] != 420 or matrix["shared_k0_cell_count"] != 12:
        raise ValueError("matrix closure differs from 420 plus 12 shared K0")
    if not (
        sanitized.get("reference_dem_exported") is False
        and sanitized.get("heldout_mola_exported") is False
        and sanitized.get("external_files_accessed") is False
    ):
        raise PermissionError("inference cases are not target-isolated")
    if sanitized.get("plan_manifest_sha256") != file_sha256(plan_manifest_path):
        raise ValueError("sanitized inference cases bind another selector plan")
    if calibration.get("source") != "expanded_development_training_references_only":
        raise PermissionError("detail threshold calibration is not train-only")
    if calibration.get("metric_contract_sha256") != file_sha256(metric_path):
        raise ValueError("detail calibration binds another metric contract")
    checkpoints = [
        _training_record(args.training_root.resolve(), seed) for seed in MODEL_SEEDS
    ]
    code_paths = [path.resolve() for path in args.code]
    payload = {
        "schema": "mars-sparse-detail-frozen-candidate/v1",
        "state": "frozen_before_inference",
        "matrix_sha256": file_sha256(matrix_path),
        "protocol_sha256": file_sha256(protocol_path),
        "metric_contract_sha256": file_sha256(metric_path),
        "plan_manifest_sha256": file_sha256(plan_manifest_path),
        "inference_case_manifest_sha256": file_sha256(sanitized_manifest_path),
        "detail_calibration_sha256": file_sha256(calibration_path),
        "checkpoints": checkpoints,
        "code": [
            {"path": str(path), "sha256": file_sha256(path)} for path in code_paths
        ],
        "selection_or_checkpoint_tuned_on_validation_metrics": False,
        "external_files_accessed": False,
        "network_precision": "fp32",
        "matrix_cell_count": 420,
        "shared_k0_cell_count": 12,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    atomic_json(output, payload)
    print(json.dumps({"state": payload["state"], "content_sha256": payload["content_sha256"]}))
    return 0


def _load_inference_cases(root: Path) -> dict[str, base.LoadedCase]:
    manifest = read_json(root / "cases-manifest.json")
    output = {}
    for record in manifest["cases"]:
        path = root / record["case"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"sanitized inference case hash differs: {record['tile_id']}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
        output[record["tile_id"]] = base.LoadedCase(
            record["tile_id"], record["region"], "validation-inference", arrays
        )
    return output


def _load_plans(root: Path) -> dict[str, dict[str, np.ndarray]]:
    manifest = read_json(root / "plans-manifest.json")
    output = {}
    for record in manifest["plans"]:
        if record["role"] != "validation":
            continue
        path = root / record["plan"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"selector plan hash differs: {record['tile_id']}")
        with np.load(path, allow_pickle=False) as archive:
            output[record["tile_id"]] = {
                selector: np.asarray(archive[selector], dtype=np.int64)
                for selector in DETAIL_SELECTORS
            }
    return output


def _remap_context(case: base.LoadedCase, global_indices: np.ndarray) -> np.ndarray:
    candidates = np.asarray(case.arrays["candidate_global_index"], dtype=np.int64)
    lookup = {int(value): index for index, value in enumerate(candidates)}
    try:
        return np.asarray([lookup[int(value)] for value in global_indices], dtype=np.int64)
    except KeyError as error:
        raise ValueError("selector plan references a non-assimilation point") from error


def _model_case(case: base.LoadedCase) -> base.LoadedCase:
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


def _empty_fields(case: base.LoadedCase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = case.arrays["z0"].shape
    return (
        np.asarray(case.arrays["z0"], dtype=np.float32).copy(),
        np.full(shape, base.HEIGHT_SCALE_M, dtype=np.float32),
        np.full(shape, 400.0, dtype=np.float32),
    )


def _validate_existing_prediction(path: Path, expected_id: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        mean = np.asarray(archive["mean"])
        metadata = json.loads(str(archive["metadata_json"].item()))
    if metadata["cell_id"] != expected_id or mean.shape != (504, 504):
        raise ValueError(f"existing prediction differs from requested cell: {path}")
    if not np.all(np.isfinite(mean)):
        raise FloatingPointError(f"existing prediction is nonfinite: {path}")
    return metadata


def command_infer(args: argparse.Namespace) -> int:
    torch, _, _ = base.torch_components()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    seed = int(args.seed)
    if seed not in MODEL_SEEDS:
        raise ValueError(f"unsupported formal model seed: {seed}")
    project = args.project_root.resolve()
    matrix_path = args.matrix.resolve()
    matrix = read_json(matrix_path)
    frozen = read_json(args.frozen_candidate.resolve())
    if frozen.get("state") != "frozen_before_inference":
        raise PermissionError("candidate is not frozen")
    unhashed = dict(frozen)
    declared = unhashed.pop("content_sha256", None)
    if declared != canonical_sha256(unhashed):
        raise ValueError("frozen candidate content hash differs")
    for record in frozen["code"]:
        if file_sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"frozen code hash differs: {record['path']}")
    if frozen["matrix_sha256"] != file_sha256(matrix_path):
        raise ValueError("frozen matrix hash differs")
    checkpoint_record = next(
        record for record in frozen["checkpoints"] if int(record["seed"]) == seed
    )
    checkpoint_path = Path(checkpoint_record["checkpoint"])
    if file_sha256(checkpoint_path) != checkpoint_record["checkpoint_sha256"]:
        raise ValueError("frozen checkpoint hash differs")
    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    architecture = checkpoint["architecture"]
    model = base.make_model(
        base_channels=int(architecture["base_channels"]),
        steps=int(architecture["transport_steps"]),
        minimum_mobility=float(architecture["minimum_mobility"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    cases = _load_inference_cases(args.inference_cases.resolve())
    plans = _load_plans(args.plans.resolve())
    if set(cases) != set(plans):
        raise ValueError("sanitized inference cases and plans do not close")
    cells = [cell for cell in matrix["cells"] if int(cell["model_seed"]) == seed]
    shared = [
        cell for cell in matrix["shared_k0_cells"] if int(cell["model_seed"]) == seed
    ]
    if len(cells) != 140 or len(shared) != 4:
        raise ValueError("per-seed inference closure differs from 140 plus 4 shared K0")
    receipt_root = args.receipt_root.resolve()
    progress_path = receipt_root / f"seed-{seed}-progress.json"
    started = time.time()
    receipts = []

    def write_prediction(
        *, cell_id: str, tile_id: str, selector: str, budget: int, output_path: Path
    ) -> None:
        case = cases[tile_id]
        model_case = _model_case(case)
        if budget == 0:
            global_context = np.empty(0, dtype=np.int64)
            local_context = np.empty(0, dtype=np.int64)
            gp, gp_std, distance = _empty_fields(model_case)
        else:
            global_context = plans[tile_id][selector][:budget]
            local_context = _remap_context(case, global_context)
            gp, gp_std, distance = base.build_gp_fields(
                model_case,
                local_context,
                lengthscale_px=PRIOR_LENGTHSCALE_PX,
                noise_m=PRIOR_NOISE_M,
            )
        if output_path.exists():
            metadata = _validate_existing_prediction(output_path, cell_id)
        else:
            prediction = base.predict_from_fields(
                model, model_case, gp, gp_std, distance, local_context, device
            ).astype(np.float32)
            valid = np.asarray(model_case.arrays["valid"], dtype=bool)
            if prediction.shape != (504, 504) or not np.all(np.isfinite(prediction[valid])):
                raise FloatingPointError(f"nonfinite prediction for {cell_id}")
            metadata = {
                "schema": "mars-sparse-detail-prediction/v1",
                "cell_id": cell_id,
                "tile_id": tile_id,
                "selector_instance": selector,
                "budget": int(budget),
                "model_seed": seed,
                "selected_count": int(local_context.size),
                "network_precision": "fp32",
                "checkpoint_sha256": checkpoint_record["checkpoint_sha256"],
                "reference_dem_read": False,
                "heldout_mola_read": False,
                "external_files_accessed": False,
            }
            atomic_npz(
                output_path,
                mean=prediction,
                context_global_indices=global_context,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        receipts.append(
            {
                **metadata,
                "prediction_relative_path": output_path.relative_to(project).as_posix(),
                "prediction_sha256": file_sha256(output_path),
            }
        )
        elapsed = time.time() - started
        atomic_json(
            progress_path,
            {
                "state": "inference",
                "seed": seed,
                "completed": len(receipts),
                "expected": 144,
                "elapsed_seconds": elapsed,
                "eta_seconds": elapsed / len(receipts) * (144 - len(receipts)),
                "last_cell_id": cell_id,
                "network_precision": "fp32",
            },
        )
        print(f"INFER {seed} {len(receipts)}/144 {cell_id}", flush=True)

    for cell in shared:
        cell_id = f"{cell['tile_id']}__shared-k000__s{seed}"
        write_prediction(
            cell_id=cell_id,
            tile_id=cell["tile_id"],
            selector="shared-k000",
            budget=0,
            output_path=project / cell["prediction_relative_path"],
        )
    for cell in cells:
        write_prediction(
            cell_id=cell["cell_id"],
            tile_id=cell["tile_id"],
            selector=cell["selector_instance"],
            budget=int(cell["budget"]),
            output_path=project / cell["prediction_relative_path"],
        )
    receipt = {
        "schema": "mars-sparse-detail-inference-receipt/v1",
        "state": "complete",
        "seed": seed,
        "matrix_sha256": file_sha256(matrix_path),
        "frozen_candidate_sha256": file_sha256(args.frozen_candidate.resolve()),
        "checkpoint_sha256": checkpoint_record["checkpoint_sha256"],
        "network_precision": "fp32",
        "matrix_prediction_count": len(cells),
        "shared_k0_prediction_count": len(shared),
        "reference_dem_read": False,
        "heldout_mola_read": False,
        "external_files_accessed": False,
        "elapsed_seconds": time.time() - started,
        "predictions": receipts,
    }
    receipt_path = receipt_root / f"seed-{seed}-receipt.json"
    atomic_json(receipt_path, receipt)
    atomic_json(
        progress_path,
        {
            "state": "complete",
            "seed": seed,
            "completed": 144,
            "expected": 144,
            "elapsed_seconds": receipt["elapsed_seconds"],
            "network_precision": "fp32",
        },
    )
    return 0


def command_smoke(args: argparse.Namespace) -> int:
    torch, _, _ = base.torch_components()
    device = torch.device(args.device)
    cases = _load_inference_cases(args.inference_cases.resolve())
    plans = _load_plans(args.plans.resolve())
    tile_id = sorted(cases)[0]
    case = cases[tile_id]
    model_case = _model_case(case)
    checkpoint = torch.load(args.checkpoint.resolve(), map_location=device)
    architecture = checkpoint["architecture"]
    model = base.make_model(
        base_channels=int(architecture["base_channels"]),
        steps=int(architecture["transport_steps"]),
        minimum_mobility=float(architecture["minimum_mobility"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    global_context = plans[tile_id]["uniform-farthest"][:5]
    local_context = _remap_context(case, global_context)
    fields = base.build_gp_fields(
        model_case,
        local_context,
        lengthscale_px=PRIOR_LENGTHSCALE_PX,
        noise_m=PRIOR_NOISE_M,
    )
    prediction = base.predict_from_fields(
        model, model_case, *fields, local_context, device
    )
    if prediction.shape != (504, 504) or not np.all(np.isfinite(prediction)):
        raise FloatingPointError("inference smoke prediction is invalid")
    atomic_npz(args.output.resolve(), mean=prediction.astype(np.float32))
    print(json.dumps({"state": "complete", "tile_id": tile_id, "shape": list(prediction.shape)}))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    sanitize = commands.add_parser("sanitize-validation")
    sanitize.add_argument("--validation-cases", type=Path, required=True)
    sanitize.add_argument("--plans", type=Path, required=True)
    sanitize.add_argument("--output", type=Path, required=True)
    sanitize.set_defaults(function=command_sanitize)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--matrix", type=Path, required=True)
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--metric-contract", type=Path, required=True)
    freeze.add_argument("--plans", type=Path, required=True)
    freeze.add_argument("--inference-cases", type=Path, required=True)
    freeze.add_argument("--detail-calibration", type=Path, required=True)
    freeze.add_argument("--training-root", type=Path, required=True)
    freeze.add_argument("--code", type=Path, action="append", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.set_defaults(function=command_freeze)
    infer = commands.add_parser("infer-seed")
    infer.add_argument("--project-root", type=Path, required=True)
    infer.add_argument("--matrix", type=Path, required=True)
    infer.add_argument("--frozen-candidate", type=Path, required=True)
    infer.add_argument("--inference-cases", type=Path, required=True)
    infer.add_argument("--plans", type=Path, required=True)
    infer.add_argument("--receipt-root", type=Path, required=True)
    infer.add_argument("--seed", type=int, required=True)
    infer.add_argument("--device", default="cuda:0")
    infer.set_defaults(function=command_infer)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--inference-cases", type=Path, required=True)
    smoke.add_argument("--plans", type=Path, required=True)
    smoke.add_argument("--checkpoint", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--device", default="cuda:0")
    smoke.set_defaults(function=command_smoke)
    return root


def main() -> int:
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())

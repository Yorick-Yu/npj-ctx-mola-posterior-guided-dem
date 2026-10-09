#!/usr/bin/env python3
"""Freeze the passed v36 candidate and one-shot external evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from paper2dem.sparse_detail_runtime import file_sha256


MODEL_SEEDS = (20260808, 20260809, 20260810)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def canonical(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--development-result", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--mechanism-contract", type=Path, required=True)
    parser.add_argument("--external-contract", type=Path, required=True)
    parser.add_argument("--failed-development", type=Path, action="append", default=[])
    parser.add_argument("--code", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"frozen v36 candidate already exists: {output}")
    project = args.project_root.resolve()
    development_path = args.development_result.resolve()
    development = read_json(development_path)
    if development.get("state") != "passed":
        raise ValueError("v36 development result did not pass")
    if not all(development["result"]["checks"].values()):
        raise ValueError("v36 development checks are not all true")
    if development.get("external_files_accessed") is not False:
        raise PermissionError("v36 development result reports external access")

    checkpoints = []
    training_root = args.training_root.resolve()
    for seed in MODEL_SEEDS:
        directory = training_root / f"seed-{seed}"
        checkpoint = directory / "final.pt"
        manifest_path = directory / "training-manifest.json"
        manifest = read_json(manifest_path)
        if not (
            manifest.get("state") == "complete"
            and int(manifest.get("seed", -1)) == seed
            and int(manifest.get("completed_updates", -1)) == 1000
            and manifest.get("network_precision") == "fp32"
            and manifest.get("validation_targets_read") is False
            and manifest.get("external_targets_read") is False
            and manifest.get("final_checkpoint_sha256") == file_sha256(checkpoint)
        ):
            raise ValueError(f"v32 seed {seed} does not satisfy the frozen parent contract")
        checkpoints.append(
            {
                "seed": seed,
                "path": str(checkpoint),
                "sha256": file_sha256(checkpoint),
                "training_manifest": str(manifest_path),
                "training_manifest_sha256": file_sha256(manifest_path),
                "completed_updates": 1000,
                "network_precision": "fp32",
            }
        )

    bindings = {
        "protocol_sha256": file_sha256(args.protocol.resolve()),
        "validation_matrix_sha256": file_sha256(args.matrix.resolve()),
        "calibration_sha256": file_sha256(args.calibration.resolve()),
        "gates_sha256": file_sha256(args.gates.resolve()),
        "training_prior_sha256": file_sha256(args.prior.resolve()),
        "mechanism_contract_sha256": file_sha256(
            args.mechanism_contract.resolve()
        ),
        "external_contract_sha256": file_sha256(args.external_contract.resolve()),
        "passed_development_result_sha256": file_sha256(development_path),
    }
    failed = []
    for path in args.failed_development:
        resolved = path.resolve()
        payload = read_json(resolved)
        if payload.get("state") == "passed":
            raise ValueError(f"failed-development artifact unexpectedly passed: {resolved}")
        failed.append({"path": str(resolved), "sha256": file_sha256(resolved)})
    code = []
    for path in args.code:
        resolved = path.resolve()
        code.append({"path": str(resolved), "sha256": file_sha256(resolved)})
    payload = {
        "schema": "mars-sparse-detail-v36-frozen-candidate/v1",
        "state": "frozen_before_external_materialization",
        "project_root": str(project),
        "model": "v32 three-seed endpoint transport ensemble",
        "mechanism": "datum-preserving K0 320 m low-pass anchoring followed by exact K20 footprint interpolation with a fixed 320 m local kernel",
        "primary_budget": 20,
        "checkpoints": checkpoints,
        "bindings": bindings,
        "code": code,
        "preserved_failed_development": failed,
        "development_result": development["result"],
        "external_evaluation_policy": "one shot with unchanged gates plus the frozen complex-terrain audit",
        "validation_targets_used_for_final_selection": True,
        "external_targets_read_before_freeze": False,
        "no_post_external_tuning_permitted": True,
    }
    payload["content_sha256"] = canonical(payload)
    atomic_json(output, payload)
    print(json.dumps({"state": payload["state"], "content_sha256": payload["content_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

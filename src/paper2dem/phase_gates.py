from __future__ import annotations

from pathlib import Path
from typing import Any

from .hashing import (
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    read_json,
    write_canonical_json,
)
from .matrix import (
    EvaluationManifest,
    ExpectedMatrix,
    InferenceManifest,
    MatrixClosureError,
    audit_expected_matrix,
)


G6_DELAYED_EXPERIMENTS = frozenset({"E3b", "E5", "E9"})


def build_g6_unlock(
    *,
    expected_matrix_path: str | Path,
    analysis_lock_path: str | Path,
    g5_unlock_path: str | Path,
    inference_manifest_paths: list[str | Path],
    evaluation_manifest_paths: list[str | Path],
    output: str | Path,
    paper2_root: str | Path,
) -> Path:
    root = Path(paper2_root).resolve()
    expected_path = guard_paper2_path(expected_matrix_path, paper2_root=root)
    analysis_path = guard_paper2_path(analysis_lock_path, paper2_root=root)
    g5_path = guard_paper2_path(g5_unlock_path, paper2_root=root)
    expected = ExpectedMatrix.read(expected_path)
    if expected.analysis_lock_sha256 != file_sha256(analysis_path):
        raise ValueError("G6 expected matrix and analysis lock differ")
    g5 = read_json(g5_path)
    g5_semantic = canonical_sha256(
        {key: value for key, value in g5.items() if key != "g5_unlock_sha256"}
    )
    if (
        g5.get("schema") != "paper2.g5-unlock/v1"
        or g5.get("status") != "complete"
        or g5.get("formal_test_unlocked") is not True
        or g5.get("g5_unlock_sha256") != g5_semantic
        or g5.get("expected_matrix_sha256") != file_sha256(expected_path)
        or g5.get("analysis_lock_sha256") != file_sha256(analysis_path)
    ):
        raise ValueError("G6 requires the matching completed G5 unlock")
    e3_expected = ExpectedMatrix(
        analysis_lock_sha256=expected.analysis_lock_sha256,
        cells=tuple(cell for cell in expected.cells if cell.key.experiment == "E3"),
    )
    if not e3_expected.cells:
        raise ValueError("G6 expected matrix contains no E3 cells")
    inferences = [
        InferenceManifest.read(guard_paper2_path(path, paper2_root=root))
        for path in inference_manifest_paths
    ]
    evaluations = [
        EvaluationManifest.read(guard_paper2_path(path, paper2_root=root))
        for path in evaluation_manifest_paths
    ]
    e3_inferences = [value for value in inferences if value.key.experiment == "E3"]
    e3_evaluations = [value for value in evaluations if value.key.experiment == "E3"]
    audit = audit_expected_matrix(
        e3_expected,
        e3_inferences,
        e3_evaluations,
        paper2_root=root,
    )
    if not audit.ok:
        raise MatrixClosureError(audit)
    audit_path = write_canonical_json(
        Path(output).with_name("G6-E3-matrix-audit.json"),
        audit.to_dict(),
        paper2_root=root,
    )
    payload: dict[str, Any] = {
        "schema": "paper2.g6-unlock/v1",
        "status": "complete",
        "bridge_external_unlocked": True,
        "delayed_experiments": sorted(G6_DELAYED_EXPERIMENTS),
        "g5_unlock_sha256": file_sha256(g5_path),
        "expected_matrix_sha256": file_sha256(expected_path),
        "analysis_lock_sha256": file_sha256(analysis_path),
        "e3_expected_cells": len(e3_expected.cells),
        "e3_matrix_audit_sha256": file_sha256(audit_path),
    }
    payload["g6_unlock_sha256"] = canonical_sha256(payload)
    return write_canonical_json(output, payload, paper2_root=root)


def validate_g6_unlock(
    path: str | Path,
    *,
    g5_unlock_path: str | Path,
    expected_matrix_path: str | Path,
    analysis_lock_path: str | Path,
    paper2_root: str | Path,
    inference_manifest_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
    evaluation_manifest_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> None:
    root = Path(paper2_root).resolve()
    unlock_path = guard_paper2_path(path, paper2_root=root)
    g5_path = guard_paper2_path(g5_unlock_path, paper2_root=root)
    expected_path = guard_paper2_path(expected_matrix_path, paper2_root=root)
    analysis_path = guard_paper2_path(analysis_lock_path, paper2_root=root)
    payload = read_json(unlock_path)
    semantic = canonical_sha256(
        {key: value for key, value in payload.items() if key != "g6_unlock_sha256"}
    )
    if payload.get("g6_unlock_sha256") != semantic:
        raise ValueError("G6 unlock semantic hash differs")
    if (
        payload.get("schema") != "paper2.g6-unlock/v1"
        or payload.get("status") != "complete"
        or payload.get("bridge_external_unlocked") is not True
        or set(payload.get("delayed_experiments", [])) != G6_DELAYED_EXPERIMENTS
        or payload.get("g5_unlock_sha256") != file_sha256(g5_path)
        or payload.get("expected_matrix_sha256") != file_sha256(expected_path)
        or payload.get("analysis_lock_sha256") != file_sha256(analysis_path)
    ):
        raise ValueError("G6 unlock is missing, incomplete, or bound to different G5 inputs")
    audit_path = unlock_path.with_name("G6-E3-matrix-audit.json")
    if not audit_path.is_file() or file_sha256(audit_path) != payload.get(
        "e3_matrix_audit_sha256"
    ):
        raise ValueError("G6 E3 matrix audit bytes differ")
    audit = read_json(audit_path)
    if audit.get("ok") is not True or int(audit.get("expected_cells", 0)) != int(
        payload.get("e3_expected_cells", -1)
    ):
        raise ValueError("G6 E3 matrix audit is not closed")
    supplied = (inference_manifest_paths is not None, evaluation_manifest_paths is not None)
    if any(supplied) and not all(supplied):
        raise ValueError("strict G6 validation requires both inference and evaluation manifests")
    if all(supplied):
        expected = ExpectedMatrix.read(expected_path)
        e3_expected = ExpectedMatrix(
            analysis_lock_sha256=expected.analysis_lock_sha256,
            cells=tuple(cell for cell in expected.cells if cell.key.experiment == "E3"),
        )
        if not e3_expected.cells:
            raise ValueError("strict G6 validation found no E3 cells")
        inferences = [
            InferenceManifest.read(guard_paper2_path(item, paper2_root=root))
            for item in inference_manifest_paths or ()
        ]
        evaluations = [
            EvaluationManifest.read(guard_paper2_path(item, paper2_root=root))
            for item in evaluation_manifest_paths or ()
        ]
        recomputed = audit_expected_matrix(
            e3_expected,
            [item for item in inferences if item.key.experiment == "E3"],
            [item for item in evaluations if item.key.experiment == "E3"],
            paper2_root=root,
        )
        if not recomputed.ok or recomputed.to_dict() != audit:
            raise ValueError("G6 E3 audit is not reproducible from the final formal manifests")

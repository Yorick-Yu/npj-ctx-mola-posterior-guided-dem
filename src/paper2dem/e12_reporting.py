from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import binascii
import math
import re
import struct
import zlib

import numpy as np

from .contracts import AssetLock
from .e10_mola import array_sha256
from .hashing import (
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    read_json,
    write_canonical_json,
    write_immutable_bytes,
)
from .matrix import (
    EvaluationManifest,
    ExpectedCell,
    ExpectedMatrix,
    InferenceManifest,
    MatrixKey,
    audit_expected_matrix,
)
from .sealed_reference import read_sealed_reference_lock


E12_SELECTION_SCHEMA = "paper2.e12-selection-lock/v1"
E12_TILE_RESULTS_SCHEMA = "paper2.e12-tile-results/v1"
E12_ERROR_MAPS_SCHEMA = "paper2.e12-error-maps/v1"
E12_VISUAL_METHODS = ("coarse", "gp-footprint", "det-fusion", "Iso-SB-cond", "OT-SB-full")
E12_FIXED_ABSOLUTE_ERROR_LIMIT_M = 100.0
E12_PALETTE = "paper2-viridis-5/v1"
E12_PALETTE_RGB = (
    (68, 1, 84),
    (59, 82, 139),
    (33, 145, 140),
    (94, 201, 98),
    (253, 231, 37),
)


def _relative_record(path: Path, *, root: Path) -> dict[str, Any]:
    guarded = guard_paper2_path(path, paper2_root=root)
    if not guarded.is_file() or guarded.stat().st_size <= 0:
        raise FileNotFoundError(guarded)
    return {
        "relative_path": guarded.relative_to(root).as_posix(),
        "sha256": file_sha256(guarded),
        "size_bytes": guarded.stat().st_size,
    }


def _verify_record(record: Mapping[str, Any], *, root: Path) -> Path:
    path = guard_paper2_path(str(record.get("relative_path", "")), paper2_root=root)
    if (
        not path.is_file()
        or file_sha256(path) != record.get("sha256")
        or path.stat().st_size != int(record.get("size_bytes", -1))
    ):
        raise ValueError(f"E12 bound artifact bytes differ: {path}")
    return path


def _safe_component(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.=-]+", value):
        raise ValueError(f"unsafe E12 output component: {value!r}")
    return value


def _visual_cell(cells: Sequence[ExpectedCell], *, experiment: str, tile_id: str, method: str) -> ExpectedCell:
    candidates = [
        cell
        for cell in cells
        if cell.key.experiment == experiment
        and cell.key.tile_id == tile_id
        and cell.key.method == method
        and cell.key.condition == "main"
        and cell.key.noise_replicate == 0
    ]
    if not candidates:
        raise ValueError(f"E12 lacks a visual cell for {experiment}/{tile_id}/{method}")
    budget = 0 if method == "coarse" else max(cell.key.budget for cell in candidates)
    candidates = [cell for cell in candidates if cell.key.budget == budget]
    minimum_seed = min(cell.key.training_seed for cell in candidates)
    candidates = [cell for cell in candidates if cell.key.training_seed == minimum_seed]
    if len(candidates) != 1:
        raise ValueError(f"E12 visual-cell rule is not unique for {experiment}/{tile_id}/{method}")
    return candidates[0]


def build_e12_selection_lock(
    *,
    expected_matrix_path: str | Path,
    analysis_lock_path: str | Path,
    role_manifest_path: str | Path,
    asset_lock_path: str | Path,
    paper2_root: str | Path,
    formal: bool = True,
) -> dict[str, Any]:
    """Freeze identity-only visual choices without opening any result artifact."""
    root = Path(paper2_root).resolve()
    expected_path = guard_paper2_path(expected_matrix_path, paper2_root=root)
    analysis_path = guard_paper2_path(analysis_lock_path, paper2_root=root)
    role_path = guard_paper2_path(role_manifest_path, paper2_root=root)
    asset_path = guard_paper2_path(asset_lock_path, paper2_root=root)
    expected = ExpectedMatrix.read(expected_path)
    if expected.analysis_lock_sha256 != file_sha256(analysis_path):
        raise ValueError("E12 selection expected matrix binds a different analysis lock")
    roles = read_json(role_path)
    if not isinstance(roles, Mapping) or roles.get("schema") != "paper2.role-manifest/v1":
        raise ValueError("E12 selection requires the formal role manifest")
    if roles.get("asset_lock_sha256") != file_sha256(asset_path):
        raise ValueError("E12 role manifest binds a different asset lock")
    asset_lock = AssetLock.read(asset_path)
    locked_tiles = tuple(
        sorted(str(value) for value in roles.get("roles", {}).get("paper2_locked_test", []))
    )
    external_tiles = tuple(
        sorted(str(value) for value in roles.get("roles", {}).get("locked_external", []))
    )
    if formal and len(locked_tiles) not in {24, 30, 36}:
        raise ValueError("formal E12 requires the locked CTX role after the reserve rule")
    if not locked_tiles or len(locked_tiles) != len(set(locked_tiles)):
        raise ValueError("E12 locked CTX tile IDs are empty or duplicated")
    assets = {asset.product_id: asset for asset in asset_lock.assets}
    if not set(locked_tiles).issubset(assets):
        raise ValueError("E12 locked CTX tile is missing from the asset lock")
    for tile_id in locked_tiles:
        asset = assets[tile_id]
        if asset.role not in {"paper2_locked_test", "sealed_reserve"} or (
            asset.role == "sealed_reserve" and not roles.get("power_report_sha256")
        ) or asset.terrain_quartile is None:
            raise ValueError("E12 locked CTX asset role/terrain stratum differs")

    def choose(quartile: int) -> str:
        candidates = [
            tile_id for tile_id in locked_tiles if assets[tile_id].terrain_quartile == quartile
        ]
        if not candidates:
            raise ValueError(f"E12 has no locked CTX in terrain quartile {quartile}")
        return min(
            candidates,
            key=lambda tile_id: canonical_sha256(
                {
                    "protocol": "paper2-e12-preoutcome-tile-choice/v1",
                    "role_manifest_sha256": file_sha256(role_path),
                    "terrain_quartile": quartile,
                    "tile_id": tile_id,
                }
            ),
        )

    preselected = [
        {
            "label": "coarse-easy-preselected",
            "tile_id": choose(0),
            "terrain_quartile": 0,
            "selection_time": "before-results",
        },
        {
            "label": "coarse-hard-preselected",
            "tile_id": choose(3),
            "terrain_quartile": 3,
            "selection_time": "before-results",
        },
    ]
    visual_cells: list[dict[str, Any]] = []
    for experiment, tiles in (("E3", locked_tiles), ("E9", external_tiles)):
        for tile_id in tiles:
            for method in E12_VISUAL_METHODS:
                cell = _visual_cell(
                    expected.cells, experiment=experiment, tile_id=tile_id, method=method
                )
                visual_cells.append(
                    {
                        "key_sha256": cell.key.identifier,
                        "key": cell.key.to_dict(),
                    }
                )
    payload: dict[str, Any] = {
        "schema": E12_SELECTION_SCHEMA,
        "status": "locked",
        "formal": formal,
        "test_results_inspected": False,
        "selection_rule": "terrain-quartile-extremes-then-tile-id-hash/v1",
        "expected_matrix": _relative_record(expected_path, root=root),
        "analysis_lock": _relative_record(analysis_path, root=root),
        "role_manifest": _relative_record(role_path, root=root),
        "asset_lock": _relative_record(asset_path, root=root),
        "locked_ctx_tiles": list(locked_tiles),
        "locked_external_tiles": list(external_tiles),
        "preselected_tiles": preselected,
        "postoutcome_order_statistics": ["best", "median", "worst"],
        "visual_methods": list(E12_VISUAL_METHODS),
        "visual_cells": sorted(visual_cells, key=lambda value: value["key_sha256"]),
        "error_map_contract": {
            "quantity": "absolute_elevation_error_m",
            "minimum_m": 0.0,
            "maximum_m": E12_FIXED_ABSOLUTE_ERROR_LIMIT_M,
            "clipping": "clip-to-fixed-range",
            "palette": E12_PALETTE,
            "scale_selected_from_results": False,
        },
    }
    payload["selection_lock_sha256"] = canonical_sha256(payload)
    return payload


def write_e12_selection_lock(path: str | Path, **kwargs: Any) -> Path:
    return write_canonical_json(
        path,
        build_e12_selection_lock(**kwargs),
        paper2_root=kwargs["paper2_root"],
    )


def read_e12_selection_lock(
    path: str | Path, *, paper2_root: str | Path, verify_sources: bool = True
) -> dict[str, Any]:
    root = Path(paper2_root).resolve()
    lock_path = guard_paper2_path(path, paper2_root=root)
    payload = read_json(lock_path)
    if not isinstance(payload, Mapping):
        raise ValueError("E12 selection lock must be a JSON object")
    if (
        payload.get("schema") != E12_SELECTION_SCHEMA
        or payload.get("status") != "locked"
        or payload.get("test_results_inspected") is not False
        or payload.get("selection_lock_sha256")
        != canonical_sha256(
            {key: value for key, value in payload.items() if key != "selection_lock_sha256"}
        )
    ):
        raise ValueError("E12 selection lock identity/state/hash differs")
    contract = payload.get("error_map_contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("maximum_m") != E12_FIXED_ABSOLUTE_ERROR_LIMIT_M
        or contract.get("palette") != E12_PALETTE
        or contract.get("scale_selected_from_results") is not False
    ):
        raise ValueError("E12 selection lock error-map scale is not preregistered")
    preselected = payload.get("preselected_tiles")
    if (
        not isinstance(preselected, list)
        or len(preselected) != 2
        or {int(record["terrain_quartile"]) for record in preselected} != {0, 3}
    ):
        raise ValueError("E12 selection lock does not preselect easy/hard terrain")
    if verify_sources:
        for name in ("expected_matrix", "analysis_lock", "role_manifest", "asset_lock"):
            record = payload.get(name)
            if not isinstance(record, Mapping):
                raise ValueError(f"E12 selection lock lacks {name}")
            _verify_record(record, root=root)
    return dict(payload)


def _artifact_path(
    manifest: InferenceManifest | EvaluationManifest, name: str, *, root: Path
) -> Path:
    matches = [artifact for artifact in manifest.artifacts if artifact.name == name]
    if len(matches) != 1:
        raise ValueError(f"E12 requires exactly one {name} artifact")
    artifact = matches[0]
    path = guard_paper2_path(artifact.relative_path, paper2_root=root)
    if not path.is_file() or file_sha256(path) != artifact.sha256:
        raise ValueError(f"E12 {name} artifact bytes differ")
    return path


def _evaluation_payload(manifest: EvaluationManifest, *, root: Path) -> dict[str, Any]:
    path = _artifact_path(manifest, "metrics", root=root)
    payload = read_json(path)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "paper2.evaluation/v1"
        or payload.get("key") != manifest.key.to_dict()
        or payload.get("inference_manifest_sha256") != manifest.inference_manifest_sha256
        or payload.get("sealed_reference_sha256") != manifest.sealed_reference_sha256
    ):
        raise ValueError("E12 evaluation artifact identity/hash bindings differ")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("E12 evaluation artifact lacks metrics")
    rmse = float(metrics.get("rmse", float("nan")))
    if not math.isfinite(rmse) or rmse < 0:
        raise ValueError("E12 evaluation RMSE is invalid")
    return dict(payload)


def _paired_effect_rows(
    expected: ExpectedMatrix,
    evaluation_by_id: Mapping[str, EvaluationManifest],
    *,
    locked_tiles: Sequence[str],
    root: Path,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for tile_id in locked_tiles:
        cells = [
            cell
            for cell in expected.cells
            if cell.key.experiment == "E3"
            and cell.key.tile_id == tile_id
            and cell.key.method in {"Iso-SB-cond", "OT-SB-full"}
            and cell.key.condition == "main"
        ]
        if not cells:
            raise ValueError(f"E12 lacks E3 full/iso cells for {tile_id}")
        max_budget = max(cell.key.budget for cell in cells)
        cells = [cell for cell in cells if cell.key.budget == max_budget]
        paired: dict[tuple[Any, ...], dict[str, float]] = {}
        for cell in cells:
            key = cell.key
            pair_key = (
                key.noise_replicate,
                key.training_seed,
                key.path_seed_set,
                key.path_count,
                key.step_count,
                key.dc_mode,
            )
            evaluation = evaluation_by_id.get(key.identifier)
            if evaluation is None:
                raise ValueError("E12 paired cell lacks its evaluation")
            rmse = float(_evaluation_payload(evaluation, root=root)["metrics"]["rmse"])
            method_values = paired.setdefault(pair_key, {})
            if key.method in method_values:
                raise ValueError("E12 paired effect has a duplicate method cell")
            method_values[key.method] = rmse
        if not paired or any(set(values) != {"Iso-SB-cond", "OT-SB-full"} for values in paired.values()):
            raise ValueError("E12 full/iso paired RMSE units do not close")
        deltas = np.asarray(
            [
                values["Iso-SB-cond"] - values["OT-SB-full"]
                for _, values in sorted(paired.items(), key=lambda value: str(value[0]))
            ],
            dtype=np.float64,
        )
        mean_delta = float(np.mean(deltas))
        results.append(
            {
                "tile_id": tile_id,
                "budget": max_budget,
                "paired_unit_count": int(deltas.size),
                "effect_definition": "rmse_iso_minus_rmse_full_m",
                "mean_paired_effect_m": mean_delta,
                "median_paired_effect_m": float(np.median(deltas)),
                "full_better_pair_fraction": float(np.mean(deltas > 0)),
                "benefit_class": (
                    "positive_full_better" if mean_delta > 0 else "nonpositive_full_better"
                ),
            }
        )
    return sorted(results, key=lambda value: value["tile_id"])


def build_e12_tile_results(
    *,
    expected_matrix_path: str | Path,
    analysis_lock_path: str | Path,
    role_manifest_path: str | Path,
    selection_lock_path: str | Path,
    inference_manifest_paths: Sequence[str | Path],
    evaluation_manifest_paths: Sequence[str | Path],
    paper2_root: str | Path,
    formal: bool = True,
) -> dict[str, Any]:
    root = Path(paper2_root).resolve()
    expected_path = guard_paper2_path(expected_matrix_path, paper2_root=root)
    analysis_path = guard_paper2_path(analysis_lock_path, paper2_root=root)
    role_path = guard_paper2_path(role_manifest_path, paper2_root=root)
    selection_path = guard_paper2_path(selection_lock_path, paper2_root=root)
    expected = ExpectedMatrix.read(expected_path)
    if expected.analysis_lock_sha256 != file_sha256(analysis_path):
        raise ValueError("E12 expected matrix binds a different analysis lock")
    selection = read_e12_selection_lock(selection_path, paper2_root=root)
    if (
        selection["expected_matrix"]["sha256"] != file_sha256(expected_path)
        or selection["analysis_lock"]["sha256"] != file_sha256(analysis_path)
        or selection["role_manifest"]["sha256"] != file_sha256(role_path)
    ):
        raise ValueError("E12 selection lock differs from reporting inputs")
    roles = read_json(role_path)
    locked_tiles = tuple(
        sorted(str(value) for value in roles.get("roles", {}).get("paper2_locked_test", []))
    )
    if formal and len(locked_tiles) not in {24, 30, 36}:
        raise ValueError("formal E12 tile table has the wrong locked CTX count")
    inference_paths = [
        guard_paper2_path(path, paper2_root=root) for path in inference_manifest_paths
    ]
    evaluation_paths = [
        guard_paper2_path(path, paper2_root=root) for path in evaluation_manifest_paths
    ]
    inferences = [InferenceManifest.read(path) for path in inference_paths]
    evaluations = [EvaluationManifest.read(path) for path in evaluation_paths]
    audit = audit_expected_matrix(expected, inferences, evaluations, paper2_root=root)
    if not audit.ok:
        raise ValueError("E12 cannot report an open expected matrix")
    evaluation_by_id = {manifest.key.identifier: manifest for manifest in evaluations}
    tile_rows = _paired_effect_rows(
        expected, evaluation_by_id, locked_tiles=locked_tiles, root=root
    )
    ordered = sorted(tile_rows, key=lambda value: (value["mean_paired_effect_m"], value["tile_id"]))
    order_statistics = {
        "worst": ordered[0]["tile_id"],
        "median": ordered[(len(ordered) - 1) // 2]["tile_id"],
        "best": ordered[-1]["tile_id"],
    }
    positive = sum(row["benefit_class"] == "positive_full_better" for row in tile_rows)
    payload: dict[str, Any] = {
        "schema": E12_TILE_RESULTS_SCHEMA,
        "status": "complete",
        "formal": formal,
        "positive_and_negative_tiles_retained": True,
        "expected_matrix": _relative_record(expected_path, root=root),
        "analysis_lock": _relative_record(analysis_path, root=root),
        "role_manifest": _relative_record(role_path, root=root),
        "selection_lock": _relative_record(selection_path, root=root),
        "matrix_audit": audit.to_dict(),
        "matrix_audit_sha256": canonical_sha256(audit.to_dict()),
        "source_manifests": {
            "inference": [_relative_record(path, root=root) for path in sorted(inference_paths)],
            "evaluation": [_relative_record(path, root=root) for path in sorted(evaluation_paths)],
        },
        "effect_scope": "E3-maximum-budget-all-paired-noise-and-training-seeds",
        "tile_count": len(tile_rows),
        "positive_tile_count": positive,
        "nonpositive_tile_count": len(tile_rows) - positive,
        "order_statistic_tiles": order_statistics,
        "order_statistic_selection_time": "after-results-labeled-descriptive-only",
        "tiles": tile_rows,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def write_e12_tile_results(path: str | Path, **kwargs: Any) -> Path:
    return write_canonical_json(
        path,
        build_e12_tile_results(**kwargs),
        paper2_root=kwargs["paper2_root"],
    )


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)


def _absolute_error_rgba(error: np.ndarray, valid: np.ndarray) -> np.ndarray:
    normalized = np.clip(
        np.asarray(error, dtype=np.float64) / E12_FIXED_ABSOLUTE_ERROR_LIMIT_M, 0.0, 1.0
    )
    position = normalized * (len(E12_PALETTE_RGB) - 1)
    left = np.floor(position).astype(np.int64)
    right = np.minimum(left + 1, len(E12_PALETTE_RGB) - 1)
    fraction = position - left
    palette = np.asarray(E12_PALETTE_RGB, dtype=np.float64)
    rgb = palette[left] * (1.0 - fraction[..., None]) + palette[right] * fraction[..., None]
    rgba = np.zeros((*error.shape, 4), dtype=np.uint8)
    rgba[..., :3] = np.rint(rgb).astype(np.uint8)
    rgba[..., 3] = np.where(valid, 255, 0).astype(np.uint8)
    return rgba


def _encode_rgba_png(rgba: np.ndarray) -> bytes:
    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
    if rgba.ndim != 3 or rgba.shape[2] != 4 or min(rgba.shape[:2]) < 1:
        raise ValueError("E12 PNG input must be a nonempty RGBA array")
    height, width = rgba.shape[:2]
    scanlines = b"".join(b"\x00" + rgba[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + _png_chunk(b"IEND", b"")
    )


def build_e12_error_maps(
    *,
    expected_matrix_path: str | Path,
    analysis_lock_path: str | Path,
    role_manifest_path: str | Path,
    selection_lock_path: str | Path,
    tile_results_path: str | Path,
    sealed_reference_lock_path: str | Path,
    inference_manifest_paths: Sequence[str | Path],
    evaluation_manifest_paths: Sequence[str | Path],
    output_directory: str | Path,
    paper2_root: str | Path,
    formal: bool = True,
) -> dict[str, Any]:
    root = Path(paper2_root).resolve()
    expected_path = guard_paper2_path(expected_matrix_path, paper2_root=root)
    analysis_path = guard_paper2_path(analysis_lock_path, paper2_root=root)
    role_path = guard_paper2_path(role_manifest_path, paper2_root=root)
    selection_path = guard_paper2_path(selection_lock_path, paper2_root=root)
    results_path = guard_paper2_path(tile_results_path, paper2_root=root)
    reference_lock_path = guard_paper2_path(sealed_reference_lock_path, paper2_root=root)
    output = guard_paper2_path(output_directory, paper2_root=root)
    selection = read_e12_selection_lock(selection_path, paper2_root=root)
    results = read_e12_tile_results(results_path, paper2_root=root)
    if formal and (selection.get("formal") is not True or results.get("formal") is not True):
        raise ValueError("formal E12 maps require formal selection/table artifacts")
    expected = ExpectedMatrix.read(expected_path)
    inferences = [
        InferenceManifest.read(guard_paper2_path(path, paper2_root=root))
        for path in inference_manifest_paths
    ]
    evaluations = [
        EvaluationManifest.read(guard_paper2_path(path, paper2_root=root))
        for path in evaluation_manifest_paths
    ]
    audit = audit_expected_matrix(expected, inferences, evaluations, paper2_root=root)
    if not audit.ok or canonical_sha256(audit.to_dict()) != results.get("matrix_audit_sha256"):
        raise ValueError("E12 map matrix audit differs from the tile table")
    inference_by_id = {manifest.key.identifier: manifest for manifest in inferences}
    evaluation_by_id = {manifest.key.identifier: manifest for manifest in evaluations}
    reference_lock = read_sealed_reference_lock(
        reference_lock_path, paper2_root=root, verify_files=True
    )
    references = {str(entry["tile_id"]): entry for entry in reference_lock["entries"]}
    visual_cells = selection.get("visual_cells")
    if not isinstance(visual_cells, list) or not visual_cells:
        raise ValueError("E12 selection lock has no visual cells")
    preselected = {
        str(record["tile_id"]): str(record["label"])
        for record in selection["preselected_tiles"]
    }
    order_statistics = {
        str(tile_id): str(label) for label, tile_id in results["order_statistic_tiles"].items()
    }
    map_records: list[dict[str, Any]] = []
    mapped_tiles: set[str] = set()
    for selected in visual_cells:
        key = MatrixKey.from_dict(dict(selected["key"]))
        if key.identifier != selected.get("key_sha256"):
            raise ValueError("E12 visual-cell key hash does not reproduce")
        inference = inference_by_id.get(key.identifier)
        evaluation = evaluation_by_id.get(key.identifier)
        reference_record = references.get(key.tile_id)
        if inference is None or evaluation is None or reference_record is None:
            raise ValueError("E12 visual cell lacks inference/evaluation/reference")
        prediction_path = _artifact_path(inference, "prediction", root=root)
        if evaluation.sealed_reference_sha256 != reference_record["reference_sha256"]:
            raise ValueError("E12 visual evaluation uses a different sealed reference")
        reference_path = guard_paper2_path(
            str(reference_record["reference_relative_path"]), paper2_root=root
        )
        with np.load(prediction_path, allow_pickle=False) as archive:
            if "mean" not in archive.files:
                raise ValueError("E12 prediction lacks mean")
            prediction = np.asarray(archive["mean"], dtype=np.float64)
        with np.load(reference_path, allow_pickle=False) as archive:
            reference = np.asarray(archive["dem"], dtype=np.float64)
            valid = np.asarray(archive["valid_mask"], dtype=bool)
        if (
            prediction.shape != reference.shape
            or valid.shape != reference.shape
            or array_sha256(reference) != reference_record["dem_array_sha256"]
            or array_sha256(valid) != reference_record["valid_mask_array_sha256"]
        ):
            raise ValueError("E12 prediction/reference arrays differ from the sealed grid")
        finite = valid & np.isfinite(reference) & np.isfinite(prediction)
        if not np.any(finite):
            raise ValueError("E12 error map has no finite evaluated pixels")
        absolute_error = np.abs(prediction - reference)
        absolute_error[~finite] = 0.0
        rgba = _absolute_error_rgba(absolute_error, finite)
        destination = output / _safe_component(key.experiment) / _safe_component(
            key.tile_id
        ) / f"{_safe_component(key.method)}-absolute-error.png"
        write_immutable_bytes(
            destination, _encode_rgba_png(rgba), paper2_root=root
        )
        tags = []
        if key.tile_id in preselected:
            tags.append(preselected[key.tile_id])
        if key.tile_id in order_statistics:
            tags.append(f"postoutcome-{order_statistics[key.tile_id]}-order-statistic")
        map_records.append(
            {
                "key_sha256": key.identifier,
                "key": key.to_dict(),
                "prediction_sha256": file_sha256(prediction_path),
                "sealed_reference_sha256": file_sha256(reference_path),
                "valid_pixel_count": int(np.sum(finite)),
                "clipped_pixel_count": int(
                    np.sum(finite & (absolute_error > E12_FIXED_ABSOLUTE_ERROR_LIMIT_M))
                ),
                "absolute_error_mean_m": float(np.mean(absolute_error[finite])),
                "absolute_error_p95_m": float(np.percentile(absolute_error[finite], 95)),
                "map": _relative_record(destination, root=root),
                "feature_tags": sorted(tags),
            }
        )
        mapped_tiles.add(key.tile_id)
    expected_tiles = set(selection["locked_ctx_tiles"]) | set(
        selection["locked_external_tiles"]
    )
    if mapped_tiles != expected_tiles:
        raise ValueError("E12 supplementary error maps do not cover every locked tile")
    if len(map_records) != len(expected_tiles) * len(E12_VISUAL_METHODS):
        raise ValueError("E12 supplementary map method cross does not close")
    payload: dict[str, Any] = {
        "schema": E12_ERROR_MAPS_SCHEMA,
        "status": "complete",
        "formal": formal,
        "uniform_template": True,
        "fixed_scale_outcome_independent": True,
        "complete_supplement_tile_coverage": True,
        "expected_matrix": _relative_record(expected_path, root=root),
        "analysis_lock": _relative_record(analysis_path, root=root),
        "role_manifest": _relative_record(role_path, root=root),
        "selection_lock": _relative_record(selection_path, root=root),
        "tile_results": _relative_record(results_path, root=root),
        "sealed_reference_lock": _relative_record(reference_lock_path, root=root),
        "matrix_audit_sha256": canonical_sha256(audit.to_dict()),
        "template": {
            "quantity": "absolute_elevation_error_m",
            "color_minimum_m": 0.0,
            "color_maximum_m": E12_FIXED_ABSOLUTE_ERROR_LIMIT_M,
            "palette": E12_PALETTE,
            "rgba_invalid_alpha": 0,
            "png_bit_depth": 8,
            "png_color_type": "rgba",
        },
        "tile_count": len(mapped_tiles),
        "method_count": len(E12_VISUAL_METHODS),
        "map_count": len(map_records),
        "maps": sorted(map_records, key=lambda value: value["key_sha256"]),
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def write_e12_error_maps(path: str | Path, **kwargs: Any) -> Path:
    return write_canonical_json(
        path,
        build_e12_error_maps(**kwargs),
        paper2_root=kwargs["paper2_root"],
    )


def read_e12_tile_results(
    path: str | Path, *, paper2_root: str | Path, verify_sources: bool = True
) -> dict[str, Any]:
    root = Path(paper2_root).resolve()
    result_path = guard_paper2_path(path, paper2_root=root)
    payload = read_json(result_path)
    if not isinstance(payload, Mapping):
        raise ValueError("E12 tile results must be a JSON object")
    if (
        payload.get("schema") != E12_TILE_RESULTS_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("positive_and_negative_tiles_retained") is not True
        or payload.get("content_sha256")
        != canonical_sha256(
            {key: value for key, value in payload.items() if key != "content_sha256"}
        )
    ):
        raise ValueError("E12 tile-results identity/state/hash differs")
    tiles = payload.get("tiles")
    if not isinstance(tiles, list) or len(tiles) != int(payload.get("tile_count", -1)):
        raise ValueError("E12 tile-results count does not close")
    tile_ids = [str(record.get("tile_id", "")) for record in tiles]
    if not all(tile_ids) or len(tile_ids) != len(set(tile_ids)):
        raise ValueError("E12 tile-results IDs are empty or duplicated")
    positive = sum(record.get("benefit_class") == "positive_full_better" for record in tiles)
    if (
        positive != int(payload.get("positive_tile_count", -1))
        or len(tiles) - positive != int(payload.get("nonpositive_tile_count", -1))
    ):
        raise ValueError("E12 positive/nonpositive tile counts do not reproduce")
    audit = payload.get("matrix_audit")
    if not isinstance(audit, Mapping) or audit.get("ok") is not True or payload.get(
        "matrix_audit_sha256"
    ) != canonical_sha256(audit):
        raise ValueError("E12 tile results do not bind a closed matrix audit")
    if verify_sources:
        for name in ("expected_matrix", "analysis_lock", "role_manifest", "selection_lock"):
            record = payload.get(name)
            if not isinstance(record, Mapping):
                raise ValueError(f"E12 tile results lack {name}")
            _verify_record(record, root=root)
        sources = payload.get("source_manifests")
        if not isinstance(sources, Mapping):
            raise ValueError("E12 tile-results sources are malformed")
        for records in sources.values():
            if not isinstance(records, list):
                raise ValueError("E12 tile-results source list is malformed")
            for record in records:
                _verify_record(record, root=root)
    return dict(payload)


def read_e12_error_maps(
    path: str | Path, *, paper2_root: str | Path, verify_sources: bool = True
) -> dict[str, Any]:
    root = Path(paper2_root).resolve()
    manifest_path = guard_paper2_path(path, paper2_root=root)
    payload = read_json(manifest_path)
    if not isinstance(payload, Mapping):
        raise ValueError("E12 error-map manifest must be a JSON object")
    if (
        payload.get("schema") != E12_ERROR_MAPS_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("uniform_template") is not True
        or payload.get("fixed_scale_outcome_independent") is not True
        or payload.get("complete_supplement_tile_coverage") is not True
        or payload.get("content_sha256")
        != canonical_sha256(
            {key: value for key, value in payload.items() if key != "content_sha256"}
        )
    ):
        raise ValueError("E12 error-map identity/state/template/hash differs")
    template = payload.get("template")
    if (
        not isinstance(template, Mapping)
        or template.get("color_minimum_m") != 0.0
        or template.get("color_maximum_m") != E12_FIXED_ABSOLUTE_ERROR_LIMIT_M
        or template.get("palette") != E12_PALETTE
    ):
        raise ValueError("E12 error-map color scale differs")
    maps = payload.get("maps")
    if (
        not isinstance(maps, list)
        or len(maps) != int(payload.get("map_count", -1))
        or len(maps)
        != int(payload.get("tile_count", -1)) * int(payload.get("method_count", -1))
    ):
        raise ValueError("E12 error-map cross does not close")
    if verify_sources:
        for name in (
            "expected_matrix",
            "analysis_lock",
            "role_manifest",
            "selection_lock",
            "tile_results",
            "sealed_reference_lock",
        ):
            record = payload.get(name)
            if not isinstance(record, Mapping):
                raise ValueError(f"E12 error-map manifest lacks {name}")
            _verify_record(record, root=root)
        for record in maps:
            map_path = _verify_record(record["map"], root=root)
            if not map_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("E12 error-map artifact is not a PNG")
    return dict(payload)

"""Resolve every preregistered matrix cell to immutable pre-test inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .calibration import read_calibration_manifest
from .e10_preparation import E10_INPUT_INDEX_SCHEMA
from .e5_preparation import E5_INPUT_INDEX_SCHEMA
from .e8_stress import E8_INPUT_INDEX_SCHEMA
from .g4 import read_g4_freeze
from .g5 import validate_calibration_set, validate_checkpoint_set
from .hashing import (
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    read_json,
    write_canonical_json,
)
from .matrix_spec import formal_matrix_cells
from .models import canonical_model_identity, method_spec
from .paper1_inference_preparation import PAPER1_FORMAL_INPUT_INDEX_SCHEMA
from .inference_preparation import FORMAL_INPUT_INDEX_SCHEMA


CELL_BINDING_SCHEMA = "paper2.cell-input-bindings/v1"


def _content_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )


def _existing(path: str | Path, *, root: Path) -> Path:
    result = guard_paper2_path(path, paper2_root=root)
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def _binding_path(path: Path, *, root: Path) -> str:
    return _existing(path, root=root).relative_to(root).as_posix()


def _resolve_flat(
    record: Mapping[str, Any],
    name: str,
    *,
    root: Path,
) -> Path:
    path = _existing(root / str(record.get(name, "")), root=root)
    expected = str(record.get(f"{name}_sha256", ""))
    if file_sha256(path) != expected:
        raise ValueError(f"cell binding source {name} bytes differ")
    return path


def _resolve_artifact(
    record: Mapping[str, Any],
    name: str,
    *,
    root: Path,
) -> Path:
    artifact = record.get(name)
    if not isinstance(artifact, Mapping):
        raise ValueError(f"cell binding source {name} artifact is malformed")
    relative = Path(str(artifact.get("relative_path", "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"cell binding source {name} path is invalid")
    path = _existing(root / relative, root=root)
    if (
        file_sha256(path) != artifact.get("sha256")
        or path.stat().st_size != int(artifact.get("byte_count", -1))
    ):
        raise ValueError(f"cell binding source {name} artifact bytes differ")
    return path


def _read_index(path: Path, *, schema: str) -> dict[str, Any]:
    payload = read_json(path)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != schema
        or payload.get("status") != "complete"
        or payload.get("content_sha256") != _content_sha256(payload)
    ):
        raise ValueError(f"formal input index identity/status/hash differs: {path}")
    return dict(payload)


def _flat_maps(
    payload: Mapping[str, Any],
    *,
    root: Path,
) -> tuple[dict[tuple[str, int, int], dict[str, Any]], dict[str, dict[str, Any]]]:
    cells: dict[tuple[str, int, int], dict[str, Any]] = {}
    for raw in payload.get("cells", ()):
        record = dict(raw)
        key = (
            str(record.get("tile_id", "")),
            int(record.get("budget", -1)),
            int(record.get("noise_replicate", -1)),
        )
        if key in cells:
            raise ValueError(f"formal input index contains duplicate cell {key}")
        for name in ("problem", "plan", "observations", "gp_posterior", "gp_manifest"):
            _resolve_flat(record, name, root=root)
        cells[key] = record
    tiles: dict[str, dict[str, Any]] = {}
    for raw in payload.get("tiles", ()):
        record = dict(raw)
        tile_id = str(record.get("tile_id", ""))
        if not tile_id or tile_id in tiles:
            raise ValueError("formal input tile records are empty or duplicate")
        tiles[tile_id] = record
    return cells, tiles


def _merge_standard_indexes(
    paths: Sequence[Path], *, root: Path
) -> tuple[
    dict[tuple[str, int, int], dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    cells: dict[tuple[str, int, int], dict[str, Any]] = {}
    tiles: dict[str, dict[str, Any]] = {}
    common: dict[str, Any] | None = None
    common_fields = (
        "asset_lock_sha256",
        "role_manifest_sha256",
        "normalization_relative_path",
        "normalization_sha256",
        "gp_prior_relative_path",
        "gp_prior_sha256",
    )
    for path in paths:
        payload = _read_index(path, schema=FORMAL_INPUT_INDEX_SCHEMA)
        observed = {name: payload.get(name) for name in common_fields}
        if common is None:
            common = observed
        elif common != observed:
            raise ValueError("standard formal input indexes do not share one contract")
        current_cells, current_tiles = _flat_maps(payload, root=root)
        for key, record in current_cells.items():
            previous = cells.setdefault(key, record)
            if previous != record:
                raise ValueError(f"standard formal input indexes conflict for {key}")
        for tile_id, record in current_tiles.items():
            previous = tiles.get(tile_id)
            if previous is None:
                tiles[tile_id] = record
                continue
            stable = (
                "role",
                "asset_role",
                "geographic_group_id",
                "prepared_manifest",
                "prepared_manifest_sha256",
                "source_preparation_manifest",
                "source_preparation_manifest_sha256",
                "ctx_input_lineage_manifest",
                "ctx_input_lineage_manifest_sha256",
            )
            if any(previous.get(name) != record.get(name) for name in stable):
                raise ValueError(f"standard formal tile metadata conflicts: {tile_id}")
            if previous.get("audit_footprint_plan") is None and record.get(
                "audit_footprint_plan"
            ) is not None:
                tiles[tile_id] = record
    if common is None:
        raise ValueError("at least one standard formal input index is required")
    return cells, tiles, common


def _nested_cell_maps(
    payload: Mapping[str, Any],
    *,
    root: Path,
    key_fields: Sequence[str],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for raw in payload.get("cells", ()):
        record = dict(raw)
        key = tuple(record.get(name) for name in key_fields)
        if key in result:
            raise ValueError(f"nested formal input index contains duplicate cell {key}")
        for name in ("problem", "plan", "observations", "gp_posterior", "gp_manifest"):
            _resolve_artifact(record, name, root=root)
        result[key] = record
    return result


def _standard_tile_paths(tile: Mapping[str, Any], *, root: Path) -> dict[str, Path]:
    return {
        "prepared_manifest": _resolve_flat(tile, "prepared_manifest", root=root),
        "source_preparation_manifest": _resolve_flat(
            tile, "source_preparation_manifest", root=root
        ),
        "ctx_input_lineage_manifest": _resolve_flat(
            tile, "ctx_input_lineage_manifest", root=root
        ),
        "audit_footprint_plan": _resolve_flat(
            tile, "audit_footprint_plan", root=root
        ),
    }


def _paper1_tile_paths(tile: Mapping[str, Any], *, root: Path) -> dict[str, Path]:
    return {
        "prepared_manifest": _resolve_flat(tile, "prepared_manifest", root=root),
        "paper1_import_manifest": _resolve_flat(
            tile, "paper1_import_manifest", root=root
        ),
        "audit_footprint_plan": _resolve_flat(
            tile, "audit_footprint_plan", root=root
        ),
    }


def _checkpoint_calibration_maps(
    *,
    root: Path,
    checkpoint_root: Path,
    calibration_root: Path,
    asset_lock_path: Path,
    role_manifest_path: Path,
    sealed_reference_lock_path: Path,
    calibration_tile_ids: Sequence[str],
) -> tuple[
    dict[tuple[str, int], tuple[Path, Path]],
    dict[tuple[str, int, str], Path],
]:
    checkpoint_manifests = sorted(
        checkpoint_root.glob("**/checkpoint-manifest.json")
    )
    checkpoint_contracts = validate_checkpoint_set(checkpoint_manifests)
    checkpoints: dict[tuple[str, int], tuple[Path, Path]] = {}
    for manifest_path in checkpoint_manifests:
        payload = read_json(manifest_path)
        key = (str(payload["model_identity"]), int(payload["seed"]))
        checkpoint = _existing(manifest_path.parent / "best.pt", root=root)
        if file_sha256(checkpoint) != payload.get("checkpoint_sha256"):
            raise ValueError("cell binding checkpoint bytes differ")
        checkpoints[key] = (checkpoint, manifest_path)
    if set(checkpoints) != set(checkpoint_contracts):
        raise AssertionError("cell binding checkpoint path map differs")
    calibration_manifests = sorted(
        calibration_root.glob("**/calibration-manifest.json")
    )
    calibration_contracts = validate_calibration_set(
        calibration_manifests,
        checkpoint_contracts,
        asset_lock_sha256=file_sha256(asset_lock_path),
        role_manifest_sha256=file_sha256(role_manifest_path),
        sealed_reference_lock_sha256=file_sha256(sealed_reference_lock_path),
        calibration_tile_ids=calibration_tile_ids,
    )
    calibrations: dict[tuple[str, int, str], Path] = {}
    for path in calibration_manifests:
        calibrator, payload = read_calibration_manifest(path)
        key = (
            canonical_model_identity(str(payload["method"])),
            int(payload["training_seed"]),
            calibrator.target,
        )
        if key in calibrations:
            raise ValueError(f"cell binding calibration is duplicate: {key}")
        calibrations[key] = path
    if set(calibrations) != set(calibration_contracts):
        raise AssertionError("cell binding calibration path map differs")
    return checkpoints, calibrations


def build_cell_input_bindings(
    *,
    paper2_root: str | Path,
    role_manifest_path: str | Path,
    asset_lock_path: str | Path,
    environment_lock_path: str | Path,
    sealed_reference_lock_path: str | Path,
    g4_freeze_path: str | Path,
    main_index_paths: Sequence[str | Path],
    bridge_val_index_path: str | Path,
    paper1_index_path: str | Path,
    e5_index_path: str | Path,
    e8_index_path: str | Path,
    e10_index_path: str | Path,
    checkpoint_root: str | Path,
    calibration_root: str | Path,
    output_path: str | Path,
) -> Path:
    """Build the exact path map consumed by the pre-test trace-lock audit."""

    root = Path(paper2_root).resolve()
    role_path = _existing(role_manifest_path, root=root)
    asset_path = _existing(asset_lock_path, root=root)
    environment_path = _existing(environment_lock_path, root=root)
    sealed_path = _existing(sealed_reference_lock_path, root=root)
    freeze_path = _existing(g4_freeze_path, root=root)
    roles = read_json(role_path)
    if roles.get("schema") != "paper2.role-manifest/v1":
        raise ValueError("cell bindings require a formal role manifest")
    freeze = read_g4_freeze(freeze_path, paper2_root=root)
    if (
        freeze.get("asset_lock_sha256") != file_sha256(asset_path)
        or freeze.get("role_manifest_sha256") != file_sha256(role_path)
    ):
        raise ValueError("cell bindings G4 freeze differs from data roles")
    main_paths = tuple(_existing(path, root=root) for path in main_index_paths)
    main_cells, main_tiles, main_common = _merge_standard_indexes(
        main_paths, root=root
    )
    bridge_path = _existing(bridge_val_index_path, root=root)
    bridge_payload = _read_index(bridge_path, schema=FORMAL_INPUT_INDEX_SCHEMA)
    bridge_cells, bridge_tiles = _flat_maps(bridge_payload, root=root)
    paper1_path = _existing(paper1_index_path, root=root)
    paper1_payload = _read_index(
        paper1_path, schema=PAPER1_FORMAL_INPUT_INDEX_SCHEMA
    )
    paper1_cells, paper1_tiles = _flat_maps(paper1_payload, root=root)
    e10_path = _existing(e10_index_path, root=root)
    e10_payload = _read_index(e10_path, schema=E10_INPUT_INDEX_SCHEMA)
    e10_cells, e10_tiles = _flat_maps(e10_payload, root=root)
    e5_path = _existing(e5_index_path, root=root)
    e5_payload = _read_index(e5_path, schema=E5_INPUT_INDEX_SCHEMA)
    e5_cells = _nested_cell_maps(
        e5_payload,
        root=root,
        key_fields=("tile_id", "selector_instance", "budget", "noise_replicate"),
    )
    e5_tiles = {str(value["tile_id"]): dict(value) for value in e5_payload["tiles"]}
    if len(e5_tiles) != len(e5_payload["tiles"]):
        raise ValueError("E5 binding tile records are duplicate")
    e8_path = _existing(e8_index_path, root=root)
    e8_payload = _read_index(e8_path, schema=E8_INPUT_INDEX_SCHEMA)
    e8_cells = _nested_cell_maps(
        e8_payload,
        root=root,
        key_fields=("tile_id", "condition", "budget", "noise_replicate"),
    )
    e8_manifests = {
        (str(value["tile_id"]), str(value["condition"])): dict(value)
        for value in e8_payload["manifests"]
    }
    if len(e8_manifests) != len(e8_payload["manifests"]):
        raise ValueError("E8 binding manifest records are duplicate")

    index_payloads = (
        ("main", main_common),
        ("bridge_val", bridge_payload),
        ("paper1", paper1_payload),
        ("E5", e5_payload),
        ("E8", e8_payload),
        ("E10", e10_payload),
    )
    for label, payload in index_payloads:
        if payload.get("asset_lock_sha256") != file_sha256(asset_path):
            raise ValueError(f"{label} input index binds a different asset lock")
        if payload.get("role_manifest_sha256") != file_sha256(role_path):
            raise ValueError(f"{label} input index binds a different role manifest")
    normalization_paths = {
        _existing(root / str(payload["normalization_relative_path"]), root=root)
        for _, payload in index_payloads
    }
    if len(normalization_paths) != 1:
        raise ValueError("formal input indexes do not share one normalization file")
    normalization_path = next(iter(normalization_paths))
    if any(
        payload.get("normalization_sha256") != file_sha256(normalization_path)
        for _, payload in index_payloads
    ):
        raise ValueError("formal input indexes bind different normalization bytes")

    checkpoint_dir = guard_paper2_path(
        checkpoint_root, paper2_root=root, allow_root=True
    )
    calibration_dir = guard_paper2_path(
        calibration_root, paper2_root=root, allow_root=True
    )
    checkpoints, calibrations = _checkpoint_calibration_maps(
        root=root,
        checkpoint_root=checkpoint_dir,
        calibration_root=calibration_dir,
        asset_lock_path=asset_path,
        role_manifest_path=role_path,
        sealed_reference_lock_path=sealed_path,
        calibration_tile_ids=roles["roles"]["uncertainty_calibration"],
    )
    paper1_import_path = _existing(
        root / str(paper1_payload["tiles"][0]["paper1_import_manifest"]), root=root
    )
    settings = {
        "g_min": float(freeze["g_min"]),
        "alpha": float(freeze["alpha"]),
        "ddpm_cosine_offset": 0.008,
        "ddpm_eta": 1.0,
        "tau_dc": float(freeze["tau_dc"]),
        "dc_delta": float(freeze["dc_delta"]),
        "lambda_dc": float(freeze["dc_lambda"]),
    }

    bindings: dict[str, dict[str, Any]] = {}
    formal_path_count = int(freeze["path_count"])
    formal_step_count = int(freeze["step_count"])
    for expected in formal_matrix_cells(
        roles["roles"],
        formal_path_count=formal_path_count,
        formal_step_count=formal_step_count,
    ):
        key = expected.key
        experiment = key.experiment
        selector: str | None = None
        if experiment in {"E3", "E4", "E6"}:
            cell = main_cells[(key.tile_id, key.budget, key.noise_replicate)]
            paths = {
                name: _resolve_flat(cell, name, root=root)
                for name in ("problem", "plan", "observations", "gp_posterior")
            }
            tile_paths = _standard_tile_paths(main_tiles[key.tile_id], root=root)
        elif experiment == "E7":
            cell = bridge_cells[(key.tile_id, key.budget, key.noise_replicate)]
            paths = {
                name: _resolve_flat(cell, name, root=root)
                for name in ("problem", "plan", "observations", "gp_posterior")
            }
            tile_paths = _standard_tile_paths(bridge_tiles[key.tile_id], root=root)
        elif experiment == "E8":
            cell = e8_cells[
                (key.tile_id, key.condition, key.budget, key.noise_replicate)
            ]
            paths = {
                name: _resolve_artifact(cell, name, root=root)
                for name in ("problem", "plan", "observations", "gp_posterior")
            }
            manifest_record = e8_manifests[(key.tile_id, key.condition)]
            tile_paths = {
                name: _resolve_artifact(manifest_record, name, root=root)
                for name in (
                    "prepared_manifest",
                    "source_preparation_manifest",
                    "ctx_input_lineage_manifest",
                    "audit_footprint_plan",
                )
            }
            tile_paths["e8_stress_manifest"] = _resolve_artifact(
                manifest_record, "manifest", root=root
            )
        elif experiment in {"E3b", "E9"}:
            cell = paper1_cells[(key.tile_id, key.budget, key.noise_replicate)]
            paths = {
                name: _resolve_flat(cell, name, root=root)
                for name in ("problem", "plan", "observations", "gp_posterior")
            }
            tile_paths = _paper1_tile_paths(paper1_tiles[key.tile_id], root=root)
        elif experiment == "E5":
            prefix = "selector="
            if not key.condition.startswith(prefix):
                raise ValueError("E5 matrix condition lacks a selector instance")
            selector = key.condition[len(prefix) :]
            cell = e5_cells[
                (key.tile_id, selector, key.budget, key.noise_replicate)
            ]
            paths = {
                name: _resolve_artifact(cell, name, root=root)
                for name in ("problem", "plan", "observations", "gp_posterior")
            }
            tile_record = e5_tiles[key.tile_id]
            audit_records = {
                str(value["selector_instance"]): dict(value)
                for value in tile_record["audit_plans"]
            }
            tile_paths = {
                "prepared_manifest": _resolve_artifact(
                    tile_record, "prepared_manifest", root=root
                ),
                "paper1_import_manifest": _resolve_artifact(
                    tile_record, "paper1_import_manifest", root=root
                ),
                "selector_manifest": _resolve_artifact(
                    tile_record, "selector_manifest", root=root
                ),
                "audit_footprint_plan": _resolve_artifact(
                    audit_records[selector], "audit_footprint_plan", root=root
                ),
            }
        elif experiment == "E10":
            cell = e10_cells[(key.tile_id, key.budget, key.noise_replicate)]
            paths = {
                name: _resolve_flat(cell, name, root=root)
                for name in ("problem", "plan", "observations", "gp_posterior")
            }
            tile_paths = _standard_tile_paths(e10_tiles[key.tile_id], root=root)
            tile_paths["e10_mola_problem_manifest"] = _resolve_flat(
                cell, "e10_mola_problem_manifest", root=root
            )
        else:  # pragma: no cover - matrix specification must remain exhaustive
            raise ValueError(f"unsupported formal experiment binding: {experiment}")

        binding: dict[str, Any] = {
            "problem": _binding_path(paths["problem"], root=root),
            "plan": _binding_path(paths["plan"], root=root),
            "observations": _binding_path(paths["observations"], root=root),
            "prepared_manifest": _binding_path(
                tile_paths["prepared_manifest"], root=root
            ),
            "zf": _binding_path(paths["gp_posterior"], root=root),
            "omega": _binding_path(paths["gp_posterior"], root=root),
            "normalization": _binding_path(normalization_path, root=root),
            "asset_lock": _binding_path(asset_path, root=root),
            "environment_lock": _binding_path(environment_path, root=root),
            "audit_footprint_plan": _binding_path(
                tile_paths["audit_footprint_plan"], root=root
            ),
            "settings": settings,
        }
        for optional in (
            "source_preparation_manifest",
            "ctx_input_lineage_manifest",
            "paper1_import_manifest",
            "selector_manifest",
            "e8_stress_manifest",
            "e10_mola_problem_manifest",
        ):
            if optional in tile_paths:
                binding[optional] = _binding_path(tile_paths[optional], root=root)
        spec = method_spec(key.method)
        identity = canonical_model_identity(key.method)
        if spec.learned:
            checkpoint, checkpoint_manifest = checkpoints[(identity, key.training_seed)]
            binding["checkpoint"] = _binding_path(checkpoint, root=root)
            binding["checkpoint_manifest"] = _binding_path(
                checkpoint_manifest, root=root
            )
        if spec.generative:
            binding["calibration_manifest"] = _binding_path(
                calibrations[(identity, key.training_seed, "pixel")], root=root
            )
            binding["footprint_calibration_manifest"] = _binding_path(
                calibrations[(identity, key.training_seed, "footprint")], root=root
            )
        identifier = key.identifier
        if identifier in bindings:
            raise AssertionError("formal cell binding identifiers are duplicate")
        bindings[identifier] = binding

    payload: dict[str, Any] = {
        "schema": CELL_BINDING_SCHEMA,
        "status": "complete",
        "role_manifest_sha256": file_sha256(role_path),
        "asset_lock_sha256": file_sha256(asset_path),
        "environment_lock_sha256": file_sha256(environment_path),
        "sealed_reference_lock_sha256": file_sha256(sealed_path),
        "g4_freeze_sha256": file_sha256(freeze_path),
        "formal_path_count": formal_path_count,
        "formal_step_count": formal_step_count,
        "normalization_sha256": file_sha256(normalization_path),
        "paper1_import_manifest_sha256": file_sha256(paper1_import_path),
        "source_indexes": {
            "main": [file_sha256(path) for path in main_paths],
            "bridge_val": file_sha256(bridge_path),
            "paper1": file_sha256(paper1_path),
            "E5": file_sha256(e5_path),
            "E8": file_sha256(e8_path),
            "E10": file_sha256(e10_path),
        },
        "cell_count": len(bindings),
        "cells": bindings,
    }
    payload["content_sha256"] = _content_sha256(payload)
    return write_canonical_json(output_path, payload, paper2_root=root)

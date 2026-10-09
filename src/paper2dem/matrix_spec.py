from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .e5_selectors import E5_BUDGETS, E5_RECONSTRUCTORS, E5_SELECTOR_INSTANCES
from .e7_prefix import E7_PATH_COUNTS, E7_STEP_COUNTS, e7_condition
from .hashing import (
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    read_json,
    write_canonical_json,
)
from .matrix import ExpectedCell, ExpectedMatrix, MatrixKey
from .incremental_anchor import ANCHORED_METHOD, IncrementalAnchorPolicy
from .models import canonical_model_identity, method_spec
from .training import FORMAL_SEEDS


TRACE_FIELDS = (
    "problem_sha256",
    "plan_sha256",
    "observations_sha256",
    "checkpoint_sha256",
    "checkpoint_manifest_sha256",
    "normalization_sha256",
    "zf_sha256",
    "omega_sha256",
    "prepared_manifest_sha256",
    "frozen_models_sha256",
    "calibration_manifest_sha256",
    "footprint_calibration_manifest_sha256",
    "audit_footprint_plan_sha256",
    "mechanism_parameters_sha256",
    "dc_parameters_sha256",
    "asset_lock_sha256",
    "analysis_lock_sha256",
    "code_sha256",
    "environment_sha256",
)
CTX_SOURCE_TRACE_FIELDS = (
    *TRACE_FIELDS,
    "ctx_input_lineage_manifest_sha256",
    "source_preparation_manifest_sha256",
)
PAPER1_IMPORT_TRACE_FIELDS = (*TRACE_FIELDS, "paper1_import_manifest_sha256")
E5_TRACE_FIELDS = (*PAPER1_IMPORT_TRACE_FIELDS, "selector_manifest_sha256")
E6_TRACE_FIELDS = (*CTX_SOURCE_TRACE_FIELDS, "uncorrected_path_sha256")
E8_TRACE_FIELDS = (*CTX_SOURCE_TRACE_FIELDS, "e8_stress_manifest_sha256")
E10_TRACE_FIELDS = (*CTX_SOURCE_TRACE_FIELDS, "e10_mola_problem_manifest_sha256")
DYNAMIC_TRACE_FIELDS = frozenset({"analysis_lock_sha256", "uncorrected_path_sha256"})


def _seeds(method: str) -> tuple[int, ...]:
    return FORMAL_SEEDS if method_spec(method).learned else (0,)


def _replicates(method: str, budget: int, standard: int, extended: int | None = None) -> range:
    if budget == 0 or method in {"coarse", "det-image"}:
        return range(1)
    return range(extended if extended is not None else standard)


def _cell(
    experiment: str,
    tile: str,
    method: str,
    budget: int,
    replicate: int,
    seed: int,
    *,
    condition: str = "main",
    path_count: int = 16,
    step_count: int = 20,
    dc_mode: str = "D1",
) -> ExpectedCell:
    spec = method_spec(method)
    return ExpectedCell(
        key=MatrixKey(
            experiment=experiment,
            tile_id=tile,
            method=method,
            budget=budget,
            noise_replicate=replicate,
            training_seed=seed,
            condition=condition,
            model_identity=canonical_model_identity(method),
            path_seed_set=f"paired-{experiment}-{replicate}",
            path_count=path_count if spec.generative else 1,
            step_count=step_count if spec.generative else 0,
            dc_mode=dc_mode if spec.terminal_dc else "D0",
        ),
        required_trace_hashes=(
            E5_TRACE_FIELDS
            if experiment == "E5"
            else E6_TRACE_FIELDS
            if experiment == "E6"
            else E8_TRACE_FIELDS
            if experiment == "E8"
            else E10_TRACE_FIELDS
            if experiment == "E10"
            else CTX_SOURCE_TRACE_FIELDS
            if experiment in {"E3", "E4", "E7", "E8"}
            else PAPER1_IMPORT_TRACE_FIELDS
            if experiment in {"E3b", "E9"}
            else TRACE_FIELDS
        ),
    )


def _main_cells(
    experiment: str,
    tiles: Sequence[str],
    methods: Sequence[str],
    budgets: Sequence[int],
    *,
    standard_replicates: int,
    formal_path_count: int,
    formal_step_count: int,
    extended_methods: set[str] | None = None,
    extended_budgets: set[int] | None = None,
    extended_replicates: int = 20,
) -> list[ExpectedCell]:
    extended_methods = set() if extended_methods is None else extended_methods
    extended_budgets = set() if extended_budgets is None else extended_budgets
    cells = []
    for tile in tiles:
        for method in methods:
            for seed in _seeds(method):
                for budget in budgets:
                    if method in {"coarse", "det-image"} and budget != 0:
                        continue
                    extended = (
                        extended_replicates
                        if method in extended_methods and budget in extended_budgets
                        else None
                    )
                    for replicate in _replicates(
                        method, budget, standard_replicates, extended
                    ):
                        cells.append(
                            _cell(
                                experiment,
                                tile,
                                method,
                                budget,
                                replicate,
                                seed,
                                path_count=formal_path_count,
                                step_count=formal_step_count,
                            )
                        )
    return cells


def formal_matrix_cells(
    role_tiles: Mapping[str, Sequence[str]],
    *,
    formal_path_count: int = 16,
    formal_step_count: int = 20,
) -> tuple[ExpectedCell, ...]:
    if formal_path_count not in E7_PATH_COUNTS:
        raise ValueError("formal path count differs from the preregistered M grid")
    if formal_step_count not in E7_STEP_COUNTS:
        raise ValueError("formal step count differs from the preregistered N grid")
    required_roles = {
        "paper1_bridge_test": 12,
        "bridge_val": 12,
        "locked_external": 6,
    }
    for role, expected in required_roles.items():
        values = role_tiles.get(role, ())
        if len(values) != expected or len(values) != len(set(values)):
            raise ValueError(f"role {role} must contain exactly {expected} unique tile IDs")
    locked_test_count = len(role_tiles.get("paper2_locked_test", ()))
    if locked_test_count not in {24, 30, 36}:
        raise ValueError(
            "role paper2_locked_test must contain 24, 30, or 36 unique tile IDs"
        )
    if locked_test_count != len(set(role_tiles["paper2_locked_test"])):
        raise ValueError("role paper2_locked_test contains duplicate tile IDs")
    cells: list[ExpectedCell] = []
    main_methods = (
        "coarse",
        "gp-footprint",
        "dem-sr-dc",
        "det-image",
        "det-fusion",
        "Iso-SB",
        "Iso-SB-cond",
        "DDPM-DC",
        "OT-SB-full",
    )
    cells.extend(
        _main_cells(
            "E3",
            role_tiles["paper2_locked_test"],
            main_methods,
            (0, 5, 10, 20, 50, 100),
            standard_replicates=5,
            formal_path_count=formal_path_count,
            formal_step_count=formal_step_count,
            extended_methods={"Iso-SB-cond", "OT-SB-full"},
            extended_budgets={20, 100},
        )
    )
    cells.extend(
        _main_cells(
            "E3b",
            role_tiles["paper1_bridge_test"],
            ("coarse", "gp-footprint", "det-fusion", "Iso-SB-cond", "OT-SB-full"),
            (0, 5, 10, 20, 50, 100),
            standard_replicates=5,
            formal_path_count=formal_path_count,
            formal_step_count=formal_step_count,
        )
    )
    for tile in role_tiles["paper2_locked_test"]:
        for method in ("F00", "F10", "F01", "F11", "spatial-shift-M"):
            for seed in _seeds(method):
                for budget in (20, 100):
                    for replicate in range(20):
                        cells.append(
                            _cell(
                                "E4",
                                tile,
                                method,
                                budget,
                                replicate,
                                seed,
                                path_count=formal_path_count,
                                step_count=formal_step_count,
                            )
                        )
    for tile in role_tiles["paper1_bridge_test"]:
        for plan_instance in E5_SELECTOR_INSTANCES:
            for method in E5_RECONSTRUCTORS:
                # E5 is a fully crossed selector x reconstructor experiment.  The
                # three shared formal seed labels are retained for every
                # reconstructor, including deterministic GP, so the expected
                # matrix has the preregistered balanced 13 x 3 x 3 design.  For
                # GP the seed labels are paired analysis identities rather than
                # learned checkpoints.
                for seed in FORMAL_SEEDS:
                    for budget in E5_BUDGETS:
                        for replicate in range(5):
                            cells.append(
                                _cell(
                                    "E5",
                                    tile,
                                    method,
                                    budget,
                                    replicate,
                                    seed,
                                    condition=f"selector={plan_instance}",
                                    path_count=formal_path_count,
                                    step_count=formal_step_count,
                                )
                            )
    for tile in role_tiles["paper2_locked_test"]:
        for seed in FORMAL_SEEDS:
            for budget in (20, 100):
                for replicate in range(20):
                    for mode in ("D0", "D1", "D3"):
                        cells.append(
                            _cell(
                                "E6",
                                tile,
                                "OT-SB-full",
                                budget,
                                replicate,
                                seed,
                                condition=f"dc={mode}",
                                path_count=formal_path_count,
                                step_count=formal_step_count,
                                dc_mode=mode,
                            )
                        )
    for tile in role_tiles["bridge_val"]:
        for method in ("Iso-SB-cond", "OT-SB-full"):
            for seed in FORMAL_SEEDS:
                for budget in (20, 100):
                    for replicate in range(5):
                        # N denotes a complete reverse solve at that discretization;
                        # an N=10/20 cell is never a truncated N=40 trajectory.  M
                        # cells share one ordered M=32 source stack for the same N.
                        for step_count in E7_STEP_COUNTS:
                            for path_count in E7_PATH_COUNTS:
                                cells.append(
                                    _cell(
                                        "E7",
                                        tile,
                                        method,
                                        budget,
                                        replicate,
                                        seed,
                                        condition=e7_condition(path_count, step_count),
                                        path_count=path_count,
                                        step_count=step_count,
                                    )
                                )
    stress_conditions = (
        "degradation-scale=0.5",
        "degradation-scale=2.0",
        "noise-scale=0.5",
        "noise-scale=2.0",
        "footprint=point",
        "footprint-diameter-m=50",
        "footprint-diameter-m=200",
        "track-spacing-scale=0.5",
        "track-spacing-scale=2.0",
        "image=missing",
        "stitch-window=uniform",
        "stitch-window=tukey-0.5",
    )
    for tile in role_tiles["bridge_val"]:
        for condition in stress_conditions:
            for method in ("gp-footprint", "Iso-SB-cond", "OT-SB-full"):
                for seed in _seeds(method):
                    for budget in (20, 100):
                        for replicate in range(5):
                            cells.append(
                                _cell(
                                    "E8",
                                    tile,
                                    method,
                                    budget,
                                    replicate,
                                    seed,
                                    condition=condition,
                                    path_count=formal_path_count,
                                    step_count=formal_step_count,
                                )
                            )
    cells.extend(
        _main_cells(
            "E9",
            role_tiles["locked_external"],
            ("coarse", "gp-footprint", "det-fusion", "Iso-SB-cond", "OT-SB-full"),
            (0, 5, 10, 20),
            standard_replicates=5,
            formal_path_count=formal_path_count,
            formal_step_count=formal_step_count,
        )
    )
    cells.extend(
        _main_cells(
            "E10",
            role_tiles["paper2_locked_test"],
            ("coarse", "gp-footprint", "det-fusion", "Iso-SB-cond", "OT-SB-full"),
            (0, 20, 100),
            standard_replicates=5,
            formal_path_count=formal_path_count,
            formal_step_count=formal_step_count,
        )
    )
    identifiers = [cell.key.identifier for cell in cells]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("formal matrix specification contains duplicate physical keys")
    return tuple(sorted(cells, key=lambda cell: cell.key.identifier))


def matrix_spec_payload(
    role_tiles: Mapping[str, Sequence[str]],
    *,
    formal_path_count: int = 16,
    formal_step_count: int = 20,
    locked_trace_hashes: Mapping[str, Mapping[str, str]] | None = None,
    require_complete_trace_lock: bool = False,
    sealed_reference_hashes: Mapping[str, Mapping[str, str]] | None = None,
    require_complete_reference_lock: bool = False,
) -> dict[str, Any]:
    cells = formal_matrix_cells(
        role_tiles,
        formal_path_count=formal_path_count,
        formal_step_count=formal_step_count,
    )
    if locked_trace_hashes is not None or require_complete_trace_lock:
        locked_trace_hashes = {} if locked_trace_hashes is None else locked_trace_hashes
        locked_cells = []
        for cell in cells:
            values = dict(locked_trace_hashes.get(cell.key.identifier, {}))
            unknown = set(values).difference(cell.required_trace_hashes)
            if unknown:
                raise ValueError(
                    f"trace lock for {cell.key.identifier} contains unknown fields: {sorted(unknown)}"
                )
            # analysis_lock_sha256 is bound by ExpectedMatrix itself and is
            # necessarily unavailable while the matrix spec is being hashed.
            lockable = set(cell.required_trace_hashes).difference(DYNAMIC_TRACE_FIELDS)
            if require_complete_trace_lock and set(values) != lockable:
                missing = sorted(lockable.difference(values))
                raise ValueError(
                    f"trace lock for {cell.key.identifier} is incomplete: {missing}"
                )
            locked_cells.append(
                replace(cell, locked_trace_hashes=tuple(sorted(values.items())))
            )
        cells = tuple(locked_cells)
    if sealed_reference_hashes is not None or require_complete_reference_lock:
        sealed_reference_hashes = (
            {} if sealed_reference_hashes is None else sealed_reference_hashes
        )
        expected_tiles = {cell.key.tile_id for cell in cells}
        if require_complete_reference_lock and not expected_tiles.issubset(
            sealed_reference_hashes
        ):
            raise ValueError(
                "sealed reference lock lacks formal-matrix tiles: "
                f"missing={sorted(expected_tiles - set(sealed_reference_hashes))}"
            )
        reference_cells = []
        for cell in cells:
            reference = sealed_reference_hashes.get(cell.key.tile_id)
            if reference is None:
                reference_cells.append(cell)
                continue
            if set(reference) != {
                "sealed_reference_sha256",
                "sealed_reference_grid_sha256",
            }:
                raise ValueError(
                    f"sealed reference fields differ for {cell.key.tile_id}"
                )
            reference_cells.append(
                replace(
                    cell,
                    sealed_reference_sha256=str(reference["sealed_reference_sha256"]),
                    sealed_reference_grid_sha256=str(
                        reference["sealed_reference_grid_sha256"]
                    ),
                )
            )
        cells = tuple(reference_cells)
    anchor_policy = IncrementalAnchorPolicy()
    payload = {
        "schema": "paper2.expected-matrix-spec/v1",
        "formal_path_count": formal_path_count,
        "formal_step_count": formal_step_count,
        "cells": [cell.to_dict() for cell in cells],
        "derived_only_experiments": ["E11", "E12"],
        "derived_analysis_methods": {
            ANCHORED_METHOD: {
                "materialization": "analysis-only-from-registered-source-cells",
                "creates_physical_cell": False,
                "creates_learned_identity": False,
                "policy": anchor_policy.to_dict(),
                "policy_sha256": anchor_policy.sha256,
            }
        },
        "sealed_reference_tile_count": len(
            {cell.key.tile_id for cell in cells if cell.sealed_reference_sha256 is not None}
        ),
        "note": "E8 is a validation-only exploratory matrix and cannot support confirmatory claims",
    }
    payload["matrix_spec_sha256"] = canonical_sha256(payload)
    return payload


def write_matrix_spec(
    path: str | Path,
    role_tiles: Mapping[str, Sequence[str]],
    *,
    paper2_root: str | Path,
    formal_path_count: int = 16,
    formal_step_count: int = 20,
    locked_trace_hashes: Mapping[str, Mapping[str, str]] | None = None,
    require_complete_trace_lock: bool = False,
    sealed_reference_hashes: Mapping[str, Mapping[str, str]] | None = None,
    require_complete_reference_lock: bool = False,
) -> Path:
    return write_canonical_json(
        path,
        matrix_spec_payload(
            role_tiles,
            formal_path_count=formal_path_count,
            formal_step_count=formal_step_count,
            locked_trace_hashes=locked_trace_hashes,
            require_complete_trace_lock=require_complete_trace_lock,
            sealed_reference_hashes=sealed_reference_hashes,
            require_complete_reference_lock=require_complete_reference_lock,
        ),
        paper2_root=paper2_root,
    )


def expected_matrix_from_spec(
    matrix_spec_path: str | Path,
    analysis_lock_path: str | Path,
) -> ExpectedMatrix:
    import json

    payload = json.loads(Path(matrix_spec_path).read_text(encoding="utf-8"))
    cells = tuple(ExpectedCell.from_dict(value) for value in payload["cells"])
    return ExpectedMatrix(analysis_lock_sha256=file_sha256(analysis_lock_path), cells=cells)


def validate_formal_matrix_derivation(
    *,
    matrix_spec_path: str | Path,
    expected_matrix_path: str | Path,
    analysis_lock_path: str | Path,
    role_tiles: Mapping[str, Sequence[str]],
    paper2_root: str | Path,
) -> tuple[dict[str, Any], ExpectedMatrix]:
    """Verify that the final matrix is the exact frozen formal design.

    This deliberately does more than compare two file hashes.  It rebuilds the
    preregistered physical keys from the frozen role manifest, checks every
    pre-test trace/reference lock, and then proves that ``expected-matrix.json``
    is the canonical materialization of that locked specification.
    """

    root = Path(paper2_root).resolve()
    spec_path = guard_paper2_path(matrix_spec_path, paper2_root=root)
    expected_path = guard_paper2_path(expected_matrix_path, paper2_root=root)
    analysis_path = guard_paper2_path(analysis_lock_path, paper2_root=root)
    for path in (spec_path, expected_path, analysis_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    payload = read_json(spec_path)
    if not isinstance(payload, dict) or payload.get("schema") != "paper2.expected-matrix-spec/v1":
        raise ValueError("formal matrix spec schema differs")
    declared = payload.get("matrix_spec_sha256")
    unhashed = dict(payload)
    unhashed.pop("matrix_spec_sha256", None)
    if declared != canonical_sha256(unhashed):
        raise ValueError("formal matrix spec semantic hash differs")
    if payload.get("derived_only_experiments") != ["E11", "E12"]:
        raise ValueError("formal matrix spec derived-only experiment contract differs")
    anchor_policy = IncrementalAnchorPolicy()
    expected_derived_methods = {
        ANCHORED_METHOD: {
            "materialization": "analysis-only-from-registered-source-cells",
            "creates_physical_cell": False,
            "creates_learned_identity": False,
            "policy": anchor_policy.to_dict(),
            "policy_sha256": anchor_policy.sha256,
        }
    }
    if payload.get("derived_analysis_methods") != expected_derived_methods:
        raise ValueError("formal matrix spec derived analysis-method contract differs")

    raw_cells = payload.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise ValueError("formal matrix spec contains no physical cells")
    locked_cells = tuple(ExpectedCell.from_dict(dict(value)) for value in raw_cells)
    try:
        formal_path_count = int(payload["formal_path_count"])
        formal_step_count = int(payload["formal_step_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("formal matrix spec lacks the frozen M/N selection") from exc
    analysis = read_json(analysis_path)
    if (
        int(analysis.get("path_count", -1)) != formal_path_count
        or int(analysis.get("step_count", -1)) != formal_step_count
    ):
        raise ValueError("formal matrix M/N differs from the analysis lock")
    template_cells = formal_matrix_cells(
        role_tiles,
        formal_path_count=formal_path_count,
        formal_step_count=formal_step_count,
    )
    locked_by_id = {cell.key.identifier: cell for cell in locked_cells}
    template_by_id = {cell.key.identifier: cell for cell in template_cells}
    if len(locked_by_id) != len(locked_cells):
        raise ValueError("formal matrix spec contains duplicate physical keys")
    if set(locked_by_id) != set(template_by_id):
        raise ValueError("formal matrix spec does not exactly match the preregistered role design")
    for identifier, template in template_by_id.items():
        locked = locked_by_id[identifier]
        locked.validate()
        if (
            locked.key != template.key
            or locked.required_trace_hashes != template.required_trace_hashes
        ):
            raise ValueError(f"formal matrix cell contract differs: {identifier}")
        required_pretest = set(locked.required_trace_hashes).difference(
            DYNAMIC_TRACE_FIELDS
        )
        if set(dict(locked.locked_trace_hashes)) != required_pretest:
            raise ValueError(f"formal matrix cell lacks complete pre-test trace locks: {identifier}")
        if (
            locked.sealed_reference_sha256 is None
            or locked.sealed_reference_grid_sha256 is None
        ):
            raise ValueError(f"formal matrix cell lacks its sealed reference lock: {identifier}")

    expected = ExpectedMatrix.read(expected_path)
    derived = ExpectedMatrix(
        analysis_lock_sha256=file_sha256(analysis_path),
        cells=locked_cells,
    )
    if expected.to_dict() != derived.to_dict():
        raise ValueError("expected matrix is not the canonical materialization of the formal spec")
    return dict(payload), expected

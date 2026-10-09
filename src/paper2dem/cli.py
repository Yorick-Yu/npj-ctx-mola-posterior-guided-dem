from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping
import argparse
import gc
import json
import os
import subprocess
import sys
import time

import numpy as np

from .audit_footprints import (
    AuditFootprintObservations,
    read_audit_footprint_plan,
    read_audit_path_projections,
    simulate_audit_footprint_observations,
    validate_audit_plan_binding,
    write_audit_path_projections,
)
from .calibration_fit import CALIBRATION_FIT_MODE_SHA256
from .catalog_audit import write_catalog_prelock_audit
from .bridge import mobility_from_uncertainty
from .contracts import (
    AssetLock,
    LaserObservations,
    Paper1SourceLock,
    PlanContract,
    ReconstructionProblem,
    reconstruction_from_observations,
)
from .evaluation import (
    audit_footprint_metrics,
    dem_metrics,
    footprint_consistency_metrics,
    gaussian_probabilistic_metrics,
)
from .e5_selectors import validate_e5_plan_binding
from .e7_prefix import (
    build_e7_source_payload,
    e7_condition,
    prefix_statistics,
    validate_e7_key,
)
from .footprints import build_footprint_operator_with_metadata
from .g0 import build_g0_bundle, write_g0_bundle
from .hashing import (
    PAPER2_ROOT,
    canonical_sha256,
    file_sha256 as _uncached_file_sha256,
    guard_paper2_path,
    write_canonical_json,
    write_immutable_bytes,
)
from .matrix import (
    ArtifactDigest,
    EvaluationManifest,
    InferenceManifest,
    MatrixKey,
    finalize_expected_matrix,
)
from .models import TORCH_AVAILABLE, UNetConfig, build_model, canonical_model_identity, method_spec
from .observations import simulate_and_write_laser_observations
from .official_assets import (
    discover_ctx_edr,
    download_official_asset,
    mola_32ppd_manifest,
)
from .plan_import import import_paper1_kmax_plan, write_plan_family
from .preparation import CTXPipelineConfig, run_ctx_isis
from .qa_replacement import (
    ROOT_ASSIGNMENT_DRAFT_FILE_SHA256,
    validate_formal_assignment_parent,
)
from .posterior_preparation import (
    PATCH_OVERLAP,
    PATCH_SIZE,
    FrozenGPPrior,
    build_condition_planes,
    locked_training_asset,
    merge_training_patch_indices,
    sliding_window_starts,
    training_index_sample_set_sha256,
    verify_gp_observations,
    verify_gp_posterior_manifest,
    write_gp_posterior,
    write_training_patch_index,
)
from .provenance import dc_parameters_sha256, mechanism_parameters_sha256


FROZEN_LOCKS = {
    "release_commit": "d1e8140d866f697d13ee3100305f248ee5287ae3",
    "source_lock_sha256": "f463debf22aa42f4ab49e36aa6b1dcf8c0231ee569a3b717df01981947179fca",
    "benchmark_lock_sha256": "8bea15c610c27d645347be90865fe0a626b2c3861324feeba2e57520d94033be",
    "catalog_sha256": "f2348ba38b297d4b2d26b513e9645dd8edd9bbc0cdc4a8f30725f75dbc0f244f",
    "prior_sha256": "eaf349a0e2e388e11077f9028065a3af71fad666c85e8e4ea56031c805d2d0bf",
}

# Formal matrix workers call ``main`` repeatedly in one long-lived process.  Keep
# exactly one checkpoint resident per device so a 88k-cell matrix does not reload
# a 256-patch U-Net for every cell, while also preventing multi-checkpoint GPU
# accumulation.
_INFERENCE_MODEL_CACHE: dict[str, tuple[str, Any]] = {}
_FORMAL_EXECUTION_CONTEXT: dict[str, Any] | None = None
FileSignature = tuple[int, int, int, int, int]


def _file_signature(path: Path) -> FileSignature:
    status = path.stat()
    return (
        int(status.st_dev),
        int(status.st_ino),
        int(status.st_size),
        int(status.st_mtime_ns),
        int(status.st_ctime_ns),
    )


@lru_cache(maxsize=16_384)
def _cached_file_sha256(
    path_text: str, signature: FileSignature
) -> str:
    del signature
    return _uncached_file_sha256(Path(path_text))


def file_sha256(path: str | Path) -> str:
    """Hash immutable worker inputs once while detecting atomic replacements."""

    resolved = Path(path).resolve()
    return _cached_file_sha256(str(resolved), _file_signature(resolved))


def _load_inference_model(checkpoint_path: Path, checkpoint_hash: str, device: Any) -> Any:
    import torch

    cache_key = str(device)
    cached = _INFERENCE_MODEL_CACHE.get(cache_key)
    if cached is not None and cached[0] == checkpoint_hash:
        return cached[1]
    if cached is not None:
        _INFERENCE_MODEL_CACHE.pop(cache_key, None)
        del cached
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    model = build_model(UNetConfig()).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)["model"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    model.eval()
    _INFERENCE_MODEL_CACHE[cache_key] = (checkpoint_hash, model)
    return model


def register_formal_execution_context(
    *,
    paper2_root: str | Path,
    expected_matrix_path: str | Path,
    matrix_spec_path: str | Path,
    analysis_lock_path: str | Path,
    g5_unlock_path: str | Path,
    g4_freeze_path: str | Path,
    role_manifest_path: str | Path,
    asset_lock_path: str | Path,
    environment_lock_path: str | Path,
    sealed_reference_lock_path: str | Path,
    expected_by_id: Mapping[str, Any],
    sealed_references: Mapping[str, Mapping[str, Any]],
    source_code_sha256: str,
) -> None:
    """Register the already validated worker-level G5 contract.

    This is deliberately an in-process optimization.  Direct CLI invocations
    still execute the complete G5 validation path, while a formal worker avoids
    reparsing an 88k-cell matrix for every individual inference cell.
    """

    global _FORMAL_EXECUTION_CONTEXT
    root = Path(paper2_root).resolve()
    raw_paths = {
        "expected_matrix": expected_matrix_path,
        "matrix_spec": matrix_spec_path,
        "analysis_lock": analysis_lock_path,
        "g5_unlock": g5_unlock_path,
        "g4_freeze": g4_freeze_path,
        "role_manifest": role_manifest_path,
        "asset_lock": asset_lock_path,
        "environment_lock": environment_lock_path,
        "sealed_reference_lock": sealed_reference_lock_path,
    }
    paths = {
        name: _paper2_input(path, root) for name, path in raw_paths.items()
    }
    current = _registered_formal_context(root=root, paths=paths)
    if (
        current is not None
        and current.get("expected_by_id") is expected_by_id
        and current.get("sealed_references_source") is sealed_references
        and current.get("source_code_sha256") == source_code_sha256
    ):
        return
    if _FORMAL_EXECUTION_CONTEXT is not None and current is None:
        raise RuntimeError(
            "formal execution contract changed during the registered worker lifetime"
        )
    sealed_references_sha256 = canonical_sha256(
        {
            str(tile_id): dict(entry)
            for tile_id, entry in sorted(sealed_references.items())
        }
    )
    if (
        current is not None
        and current.get("expected_by_id") is expected_by_id
        and current.get("sealed_references_sha256") == sealed_references_sha256
        and current.get("source_code_sha256") == source_code_sha256
    ):
        return
    if not expected_by_id:
        raise ValueError("registered formal execution context has no expected cells")
    if not sealed_references or any(
        not tile_id or not isinstance(entry, Mapping)
        for tile_id, entry in sealed_references.items()
    ):
        raise ValueError("registered formal execution context has invalid sealed references")
    if len(source_code_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in source_code_sha256
    ):
        raise ValueError("registered formal source digest is not SHA-256")
    _FORMAL_EXECUTION_CONTEXT = {
        "root": root,
        "paths": {
            name: (path, _file_signature(path)) for name, path in paths.items()
        },
        "formal_lock_hashes": {
            "asset_lock_sha256": file_sha256(paths["asset_lock"]),
            "analysis_lock_sha256": file_sha256(paths["analysis_lock"]),
            "environment_sha256": file_sha256(paths["environment_lock"]),
        },
        "unlock": _json(paths["g5_unlock"]),
        "role_manifest": _json(paths["role_manifest"]),
        "expected_by_id": expected_by_id,
        "sealed_references": {
            str(tile_id): dict(entry)
            for tile_id, entry in sealed_references.items()
        },
        "sealed_references_source": sealed_references,
        "sealed_references_sha256": sealed_references_sha256,
        "source_code_sha256": source_code_sha256,
    }


def _registered_formal_context(
    *, root: Path, paths: Mapping[str, Path]
) -> Mapping[str, Any] | None:
    context = _FORMAL_EXECUTION_CONTEXT
    if context is None or context.get("root") != root:
        return None
    registered_paths = context.get("paths", {})
    for name, path in paths.items():
        registered = registered_paths.get(name)
        if registered != (path, _file_signature(path)):
            return None
    return context


def _registered_formal_reference(
    *, root: Path, reference_lock_path: Path, tile_id: str
) -> Mapping[str, Any] | None:
    context = _registered_formal_context(
        root=root, paths={"sealed_reference_lock": reference_lock_path}
    )
    if context is None:
        return None
    references = context.get("sealed_references")
    if not isinstance(references, Mapping):
        raise RuntimeError("registered formal context lost its sealed-reference map")
    entry = references.get(tile_id)
    if not isinstance(entry, Mapping):
        raise ValueError(f"registered sealed-reference map lacks tile {tile_id}")
    return entry


def _json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _paper2_input(path: str | Path, root: Path) -> Path:
    result = guard_paper2_path(path, paper2_root=root)
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def _array(path: str | Path, *, key: str | None = None) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key is None:
                if len(archive.files) != 1:
                    raise ValueError(f"NPZ requires --key; available fields: {archive.files}")
                key = archive.files[0]
            return np.array(archive[key], copy=True)
    raise ValueError(f"unsupported array file: {path}")


@lru_cache(maxsize=8)
def _tree_sha256(root: Path) -> str:
    # A Python worker cannot hot-reload a changed source tree safely.  Bind the
    # loaded tree once; formal execution additionally checks this digest against
    # every cell's G5 trace lock before the worker starts.
    records = []
    for path in sorted(root.rglob("*.py")):
        records.append((path.relative_to(root).as_posix(), file_sha256(path)))
    return canonical_sha256(records)


def _cuda_device_identity(device: Any) -> tuple[int, str, str | None]:
    """Return an observed CUDA identity; never synthesize a UUID."""

    import torch

    index = int(device.index if device.index is not None else torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(index)
    name = str(properties.name)
    raw_uuid = getattr(properties, "uuid", None)
    uuid = None if raw_uuid is None else str(raw_uuid).strip()
    if not uuid:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in completed.stdout.splitlines():
                observed_index, observed_uuid = (part.strip() for part in line.split(",", 1))
                if int(observed_index) == index and observed_uuid:
                    uuid = observed_uuid
                    break
        except (FileNotFoundError, subprocess.SubprocessError, ValueError):
            uuid = None
    return index, name, uuid


def _valid_mask_area_contract(
    transform: tuple[float, float, float, float, float, float],
    valid_mask: np.ndarray,
) -> tuple[float, float, int]:
    a, b, _, d, e, _ = (float(value) for value in transform)
    pixel_area_m2 = abs(a * e - b * d)
    valid_pixel_count = int(np.count_nonzero(np.asarray(valid_mask, dtype=bool)))
    area_km2 = pixel_area_m2 * valid_pixel_count / 1.0e6
    if not np.isfinite(area_km2) or area_km2 <= 0:
        raise ValueError("inference valid-mask area is non-positive")
    return pixel_area_m2, area_km2, valid_pixel_count


def _path_seed_contract(
    experiment: str,
    noise_replicate: int,
    *,
    requested_seed: int | None,
    requested_set: str | None,
) -> tuple[str, int]:
    if noise_replicate < 0:
        raise ValueError("noise replicate must be nonnegative")
    if requested_seed is not None and requested_seed < 0:
        raise ValueError("path seed must be nonnegative")
    if requested_seed is not None:
        label = requested_set or f"seed-{requested_seed}"
        return label, requested_seed
    label = requested_set or f"paired-{experiment}-{noise_replicate}"
    seed = int(canonical_sha256({"path_seed_set": label})[:16], 16) % (2**63 - 1)
    return label, seed


def _patch_path_seed(base_seed: int, path_index: int, location: tuple[slice, slice]) -> int:
    rows, columns = location
    payload = {
        "base_seed": base_seed,
        "path_index": path_index,
        "row": [rows.start, rows.stop, rows.step],
        "column": [columns.start, columns.stop, columns.step],
    }
    return int(canonical_sha256(payload)[:16], 16) % (2**63 - 1)


def command_audit_assets(args: argparse.Namespace) -> int:
    root = Path(args.paper2_root).resolve()
    if args.ctx_observation_id:
        if not args.ode_cache_json or not args.output:
            raise ValueError("ODE discovery requires --ode-cache-json and --output")
        payload = discover_ctx_edr(
            args.ctx_observation_id,
            raw_cache=args.ode_cache_json,
            output=args.output,
            paper2_root=root,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    if args.write_mola_manifest:
        if not args.output:
            raise ValueError("MOLA manifest generation requires --output")
        output = write_canonical_json(args.output, mola_32ppd_manifest(), paper2_root=root)
        print(json.dumps({"output": str(output), "sha256": file_sha256(output)}, indent=2))
        return 0
    if args.download_url:
        if not args.download_output or not args.download_receipt:
            raise ValueError("asset download requires --download-output and --download-receipt")
        payload = download_official_asset(
            url=args.download_url,
            output=args.download_output,
            receipt=args.download_receipt,
            expected_sha256=args.expected_sha256,
            paper2_root=root,
        )
        print(json.dumps(payload, indent=2))
        return 0
    if not args.catalog or not args.benchmark_lock or not args.output:
        raise ValueError("catalog audit requires --catalog, --benchmark-lock, and --output")
    catalog_source_lock = None
    expected_catalog_sha256 = FROZEN_LOCKS["catalog_sha256"]
    seed_material = FROZEN_LOCKS["catalog_sha256"]
    if args.catalog_source_lock:
        catalog_source_lock = _paper2_input(args.catalog_source_lock, root)
        source_lock_payload = _json(catalog_source_lock)
        if not isinstance(source_lock_payload, dict):
            raise ValueError("catalog source lock must be a JSON object")
        expected_catalog_sha256 = str(source_lock_payload.get("catalog_sha256", ""))
        seed_material = canonical_sha256(
            {
                "catalog_source_lock_file_sha256": file_sha256(catalog_source_lock),
                "benchmark_lock_sha256": FROZEN_LOCKS["benchmark_lock_sha256"],
            }
        )
    output = write_catalog_prelock_audit(
        args.output,
        args.catalog,
        args.benchmark_lock,
        expected_catalog_sha256=expected_catalog_sha256,
        expected_benchmark_sha256=FROZEN_LOCKS["benchmark_lock_sha256"],
        seed_material=seed_material,
        paper2_root=root,
        catalog_source_lock_path=catalog_source_lock,
    )
    payload = _json(output)
    if not args.conflict_groups or not args.draft_asset_lock:
        raise ValueError(
            "catalog audit requires --conflict-groups and --draft-asset-lock sidecar outputs"
        )
    sidecar_inputs = {
        name: None if value is None else _paper2_input(value, root)
        for name, value in {
            "processed_lock_path": args.processed_lock,
            "edr_index_path": args.edr_index,
            "mola_source_lock_path": args.mola_source_lock,
            "terrain_manifest_path": args.terrain_manifest,
            "paper1_seen_lock_path": args.paper1_seen_lock,
        }.items()
    }
    assignment_override = None
    assignment_parent_file_sha256 = None
    if (args.assignment_parent is None) != (
        args.expected_assignment_parent_file_sha256 is None
    ):
        raise ValueError(
            "assignment parent path and expected file SHA256 must be supplied together"
        )
    if args.assignment_parent is not None:
        assignment_path = _paper2_input(args.assignment_parent, root)
        assignment_payload = validate_formal_assignment_parent(
            assignment_path,
            expected_file_sha256=args.expected_assignment_parent_file_sha256,
            root_assignment_file_sha256=ROOT_ASSIGNMENT_DRAFT_FILE_SHA256,
        )
        assignment_override = assignment_payload.get("assignments")
        if not isinstance(assignment_override, list):
            raise ValueError("validated assignment parent lacks an assignment list")
        assignment_parent_file_sha256 = file_sha256(assignment_path)
    draft, formal, conflicts = build_g0_bundle(
        payload,
        paper1_lock=FROZEN_LOCKS,
        paper2_root=root,
        assignment_override=assignment_override,
        assignment_parent_file_sha256=assignment_parent_file_sha256,
        **sidecar_inputs,
    )
    outputs = write_g0_bundle(
        draft=draft,
        conflicts=conflicts,
        formal=formal,
        draft_path=args.draft_asset_lock,
        conflicts_path=args.conflict_groups,
        formal_path=args.formal_asset_lock,
        paper2_root=root,
    )
    result = {
        "catalog_audit": payload,
        "g0_state": draft["state"],
        "formal_test_unlocked": formal is not None,
        "outputs": outputs,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if formal is not None or not args.require_ready else 2


def command_import_plans(args: argparse.Namespace) -> int:
    root = Path(args.paper2_root).resolve()
    prefix_paths = {int(item.split("=", 1)[0]): item.split("=", 1)[1] for item in args.prefix}
    lock = Paper1SourceLock(**FROZEN_LOCKS)
    budgets = tuple(int(value) for value in args.budgets.split(","))
    plan = import_paper1_kmax_plan(
        args.selection,
        args.problem,
        paper1_lock=lock,
        budgets=budgets,
        prefix_selection_paths=prefix_paths,
    )
    paths = write_plan_family(plan, args.output, budgets=budgets, paper2_root=root)
    print(json.dumps({"tile_id": plan.tile_id, "plans": [str(path) for path in paths]}, indent=2))
    return 0


def command_simulate(args: argparse.Namespace) -> int:
    root = Path(args.paper2_root).resolve()
    plan = _paper2_input(args.plan, root)
    sealed_reference = _paper2_input(args.sealed_reference, root)
    output = simulate_and_write_laser_observations(
        plan,
        sealed_reference,
        args.output,
        replicate=args.replicate,
        seed=args.seed,
        paper2_root=root,
    )
    observation = LaserObservations.read(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": file_sha256(output),
                "valid_returns": int(np.sum(observation.quality_flags == 0)),
                "total_returns": len(observation.return_values_m),
            },
            indent=2,
        )
    )
    return 0


def command_prepare(args: argparse.Namespace) -> int:
    root = Path(args.paper2_root).resolve()
    active_modes = sum(
        bool(value)
        for value in (
            args.ctx_edr,
            args.gp_problem,
            args.patch_problem,
            args.merge_training_index,
        )
    )
    if active_modes > 1:
        raise ValueError(
            "prepare-data accepts only one of CTX, GP, training-patch, or index-merge mode"
        )
    if args.merge_training_index:
        if not args.training_index_output:
            raise ValueError("training-index merge requires --training-index-output")
        components = []
        for requested in args.merge_training_index:
            path = _paper2_input(requested, root)
            payload = _json(path)
            role = str(payload.get("role", ""))
            _IndexedBatches(path, root=root, device=None, role=role)
            components.append((payload, file_sha256(path)))
        payload = merge_training_patch_indices(
            components=components,
            output=args.training_index_output,
            paper2_root=root,
        )
        print(json.dumps(payload, indent=2))
        return 0
    if args.ctx_edr:
        required = {
            "ctx_output_directory": args.ctx_output_directory,
            "isis_prefix": args.isis_prefix,
            "isis_data": args.isis_data,
            "spice_cache": args.spice_cache,
            "conda_cache": args.conda_cache,
            "mola_radius_label": args.mola_radius_label,
            "mola_radius_raw_cube": args.mola_radius_raw_cube,
            "mola_radius_prepared_cube": args.mola_radius_prepared_cube,
            "map_template": args.map_template,
            "isis_preferences": args.isis_preferences,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"CTX ISIS preparation missing options: {', '.join(missing)}")
        config = CTXPipelineConfig(
            paper2_root=root,
            isis_prefix=guard_paper2_path(args.isis_prefix, paper2_root=root),
            isis_data=guard_paper2_path(args.isis_data, paper2_root=root),
            spice_cache=guard_paper2_path(args.spice_cache, paper2_root=root),
            conda_cache=guard_paper2_path(args.conda_cache, paper2_root=root),
            mola_radius_label=guard_paper2_path(args.mola_radius_label, paper2_root=root),
            mola_radius_raw_cube=guard_paper2_path(args.mola_radius_raw_cube, paper2_root=root),
            mola_radius_prepared_cube=guard_paper2_path(
                args.mola_radius_prepared_cube, paper2_root=root
            ),
            map_template=guard_paper2_path(args.map_template, paper2_root=root),
            isis_preferences=guard_paper2_path(args.isis_preferences, paper2_root=root),
        )
        payload = run_ctx_isis(
            edr=_paper2_input(args.ctx_edr, root),
            output_directory=guard_paper2_path(args.ctx_output_directory, paper2_root=root),
            config=config,
            dry_run=args.dry_run,
            spatial_summing=args.spatial_summing,
        )
        if args.isis_run_manifest:
            write_canonical_json(args.isis_run_manifest, payload, paper2_root=root)
        print(json.dumps(payload, indent=2))
        return 0
    if args.gp_problem:
        required = {
            "gp_plan": args.gp_plan,
            "gp_observations": args.gp_observations,
            "gp_prior": args.gp_prior,
            "gp_posterior_output": args.gp_posterior_output,
            "gp_manifest_output": args.gp_manifest_output,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"GP preparation missing options: {', '.join(missing)}")
        problem_path = _paper2_input(args.gp_problem, root)
        plan_path = _paper2_input(args.gp_plan, root)
        observations_path = _paper2_input(args.gp_observations, root)
        prior_path = _paper2_input(args.gp_prior, root)
        problem = _cached_reconstruction_problem(
            str(problem_path.resolve()), _file_signature(problem_path)
        )
        plan = _cached_plan_contract(
            str(plan_path.resolve()), _file_signature(plan_path)
        )
        observations = LaserObservations.read(observations_path)
        verify_gp_observations(
            problem,
            plan,
            observations,
            observations_file_sha256=file_sha256(observations_path),
        )
        prior = FrozenGPPrior.from_mapping(_json(prior_path))
        payload = write_gp_posterior(
            problem=problem,
            plan=plan,
            prior=prior,
            problem_sha256=file_sha256(problem_path),
            plan_sha256=file_sha256(plan_path),
            prior_sha256=file_sha256(prior_path),
            posterior_output=args.gp_posterior_output,
            manifest_output=args.gp_manifest_output,
            paper2_root=root,
        )
        print(json.dumps(payload, indent=2))
        return 0
    if args.patch_problem:
        required = {
            "patch_plan": args.patch_plan,
            "patch_gp_manifest": args.patch_gp_manifest,
            "patch_gp_posterior": args.patch_gp_posterior,
            "patch_asset_lock": args.patch_asset_lock,
            "patch_reference": args.patch_reference,
            "patch_normalization": args.patch_normalization,
            "patch_output_directory": args.patch_output_directory,
            "training_index_output": args.training_index_output,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"training-patch preparation missing options: {', '.join(missing)}")
        problem_path = _paper2_input(args.patch_problem, root)
        plan_path = _paper2_input(args.patch_plan, root)
        asset_lock_path = _paper2_input(args.patch_asset_lock, root)
        problem = _cached_reconstruction_problem(
            str(problem_path.resolve()), _file_signature(problem_path)
        )
        plan = _cached_plan_contract(
            str(plan_path.resolve()), _file_signature(plan_path)
        )
        # Resolve the locked role before opening the reference-label artifact.
        asset = locked_training_asset(
            AssetLock.read(asset_lock_path),
            problem=problem,
            plan=plan,
        )
        posterior_path = _paper2_input(args.patch_gp_posterior, root)
        gp_manifest_path = _paper2_input(args.patch_gp_manifest, root)
        posterior_hash = file_sha256(posterior_path)
        problem_hash = file_sha256(problem_path)
        plan_hash = file_sha256(plan_path)
        verify_gp_posterior_manifest(
            _json(gp_manifest_path),
            manifest_file_sha256=file_sha256(gp_manifest_path),
            problem_sha256=problem_hash,
            plan_sha256=plan_hash,
            posterior_sha256=posterior_hash,
        )
        normalization_path = _paper2_input(args.patch_normalization, root)
        normalization = _normalization(normalization_path)
        # This is deliberately below the locked-role check above.
        target_path = _paper2_input(args.patch_reference, root)
        with np.load(target_path, allow_pickle=False) as archive:
            required_arrays = {
                args.patch_reference_dem_key,
                args.patch_reference_mask_key,
                "transform",
                "tile_id",
                "crs_wkt",
            }
            missing_arrays = sorted(required_arrays.difference(archive.files))
            if missing_arrays:
                raise ValueError(f"training reference lacks arrays: {missing_arrays}")
            target = np.asarray(archive[args.patch_reference_dem_key], dtype=np.float64)
            target_mask = np.asarray(archive[args.patch_reference_mask_key], dtype=bool)
            target_transform = tuple(float(item) for item in archive["transform"])
            target_tile_id = str(archive["tile_id"].item())
            target_crs_wkt = str(archive["crs_wkt"].item())
        if target_tile_id != problem.tile_id or target_tile_id != plan.tile_id:
            raise ValueError("training reference tile differs from the frozen problem/plan")
        if target_transform != plan.transform:
            raise ValueError("training reference transform differs from the frozen plan")
        if target_crs_wkt != plan.crs_wkt:
            raise ValueError("training reference CRS differs from the frozen plan")
        gp_manifest = _json(gp_manifest_path)
        source_hashes = {
            "asset_lock_sha256": file_sha256(asset_lock_path),
            "problem_sha256": problem_hash,
            "plan_sha256": plan_hash,
            "gp_manifest_sha256": file_sha256(gp_manifest_path),
            "gp_prior_sha256": str(gp_manifest["gp_prior_sha256"]),
            "gp_posterior_sha256": posterior_hash,
            "target_label_sha256": file_sha256(target_path),
            "normalization_sha256": file_sha256(normalization_path),
        }
        payload = write_training_patch_index(
            problem=problem,
            plan=plan,
            target=target,
            target_valid_mask=target_mask,
            zf=np.asarray(_array(posterior_path, key="zf"), dtype=np.float64),
            omega=np.asarray(_array(posterior_path, key="omega"), dtype=np.float64),
            normalization=normalization,
            role=asset.role,
            geographic_group_id=asset.geographic_group_id,
            source_hashes=source_hashes,
            output_directory=args.patch_output_directory,
            index_output=args.training_index_output,
            paper2_root=root,
        )
        print(json.dumps(payload, indent=2))
        return 0
    required = {
        "tile_id": args.tile_id,
        "z0": args.z0,
        "image": args.image,
        "mask": args.mask,
        "observations": args.observations,
        "models": args.models,
        "prepared_manifest_sha256": args.prepared_manifest_sha256,
        "output": args.output,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"reconstruction preparation missing options: {', '.join(missing)}")
    z0_path = _paper2_input(args.z0, root)
    image_path = _paper2_input(args.image, root)
    mask_path = _paper2_input(args.mask, root)
    observations_path = _paper2_input(args.observations, root)
    observations = LaserObservations.read(observations_path)
    hashes = {
        str(name): str(digest) for name, digest in _json(_paper2_input(args.models, root)).items()
    }
    problem = reconstruction_from_observations(
        tile_id=args.tile_id,
        z0=_array(z0_path, key=args.z0_key),
        image=_array(image_path, key=args.image_key),
        valid_mask=_array(mask_path, key=args.mask_key).astype(bool),
        observations=observations,
        observations_path=observations_path,
        frozen_model_hashes=hashes,
        metadata={"prepared_manifest_sha256": args.prepared_manifest_sha256},
        paper2_root=root,
    )
    output = problem.write(args.output, paper2_root=root)
    print(json.dumps({"output": str(output), "sha256": file_sha256(output)}, indent=2))
    return 0


@dataclass(frozen=True)
class _IndexedSample:
    relative_path: str
    sha256: str
    tile_id: str
    geographic_group_id: str
    role: str
    row_start: int
    column_start: int
    path: Path
    file_signature: FileSignature

    @property
    def contract(self) -> dict[str, str]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "tile_id": self.tile_id,
            "geographic_group_id": self.geographic_group_id,
            "role": self.role,
            "row_start": self.row_start,
            "column_start": self.column_start,
        }


def _archive_scalar_string(archive: Any, name: str) -> str:
    value = np.asarray(archive[name])
    if value.ndim != 0:
        raise ValueError(f"training patch {name} must be a scalar string")
    result = str(value.item())
    if not result:
        raise ValueError(f"training patch {name} cannot be empty")
    return result


@lru_cache(maxsize=64)
def _cached_reconstruction_problem(
    path_text: str, signature: FileSignature
) -> ReconstructionProblem:
    path = Path(path_text)
    if _file_signature(path) != signature:
        raise RuntimeError("reconstruction problem changed before cached load")
    return ReconstructionProblem.read(path)


@lru_cache(maxsize=512)
def _cached_plan_contract(
    path_text: str, signature: FileSignature
) -> PlanContract:
    path = Path(path_text)
    if _file_signature(path) != signature:
        raise RuntimeError("measurement plan changed before cached load")
    return PlanContract.read(path)


@lru_cache(maxsize=128)
def _cached_audit_footprint_plan(
    path_text: str, signature: FileSignature
) -> Any:
    path = Path(path_text)
    if _file_signature(path) != signature:
        raise RuntimeError("audit footprint plan changed before cached load")
    return read_audit_footprint_plan(path)


@lru_cache(maxsize=256)
def _cached_measurement_footprint(
    problem_path_text: str,
    problem_signature: FileSignature,
    plan_path_text: str,
    plan_signature: FileSignature,
    drop_invalid: bool,
) -> Any:
    problem = _cached_reconstruction_problem(problem_path_text, problem_signature)
    plan = _cached_plan_contract(plan_path_text, plan_signature)
    good = problem.observation_quality_flags == 0
    return build_footprint_operator_with_metadata(
        problem.observation_coordinates[good],
        grid_shape=problem.z0.shape,
        transform=plan.transform,
        diameter_m=plan.footprint.diameter_m,
        quadrature_points=plan.footprint.quadrature_points,
        valid_mask=problem.valid_mask,
        require_full_support=plan.footprint.require_full_support,
        drop_invalid=drop_invalid,
    )


@lru_cache(maxsize=128)
def _cached_audit_binding(
    audit_path_text: str,
    audit_signature: FileSignature,
    problem_path_text: str,
    problem_signature: FileSignature,
    plan_path_text: str,
    plan_signature: FileSignature,
) -> tuple[Any, Any]:
    audit_plan = _cached_audit_footprint_plan(audit_path_text, audit_signature)
    problem = _cached_reconstruction_problem(problem_path_text, problem_signature)
    plan = _cached_plan_contract(plan_path_text, plan_signature)
    operator = validate_audit_plan_binding(
        audit_plan, problem=problem, measurement_plan=plan
    )
    return audit_plan, operator


@lru_cache(maxsize=32)
def _cached_sealed_reference_arrays(
    path_text: str, signature: FileSignature
) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path_text)
    if _file_signature(path) != signature:
        raise RuntimeError("sealed reference changed before cached evaluation load")
    with np.load(path, allow_pickle=False) as archive:
        reference = np.array(archive["dem"], dtype=np.float64, copy=True)
        valid = np.array(archive["valid_mask"], dtype=bool, copy=True)
    reference.setflags(write=False)
    valid.setflags(write=False)
    return reference, valid


@lru_cache(maxsize=64)
def _cached_condition_bundle(
    zf_path_text: str,
    zf_signature: FileSignature,
    zf_key: str,
    omega_path_text: str,
    omega_signature: FileSignature,
    omega_key: str,
    variance_key: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    zf_path = Path(zf_path_text)
    omega_path = Path(omega_path_text)
    if (
        _file_signature(zf_path) != zf_signature
        or _file_signature(omega_path) != omega_signature
    ):
        raise RuntimeError("GP condition bundle changed before cached load")
    variance: np.ndarray | None = None
    if zf_path == omega_path and zf_path.suffix == ".npz":
        with np.load(zf_path, allow_pickle=False) as archive:
            zf = np.array(archive[zf_key], dtype=np.float64, order="C", copy=True)
            omega = np.array(
                archive[omega_key], dtype=np.float64, order="C", copy=True
            )
            if variance_key is not None and variance_key in archive.files:
                variance = np.array(
                    archive[variance_key], dtype=np.float64, order="C", copy=True
                )
    else:
        zf = np.array(
            _array(zf_path, key=zf_key), dtype=np.float64, order="C", copy=True
        )
        omega = np.array(
            _array(omega_path, key=omega_key),
            dtype=np.float64,
            order="C",
            copy=True,
        )
        if variance_key is not None and zf_path.suffix == ".npz":
            with np.load(zf_path, allow_pickle=False) as archive:
                if variance_key in archive.files:
                    variance = np.array(
                        archive[variance_key],
                        dtype=np.float64,
                        order="C",
                        copy=True,
                    )
    if (
        _file_signature(zf_path) != zf_signature
        or _file_signature(omega_path) != omega_signature
    ):
        raise RuntimeError("GP condition bundle changed during cached load")
    zf.setflags(write=False)
    omega.setflags(write=False)
    if variance is not None:
        variance.setflags(write=False)
    return zf, omega, variance


@lru_cache(maxsize=16)
def _cached_normalization(
    path_text: str, signature: FileSignature
) -> Mapping[str, float]:
    path = Path(path_text)
    if _file_signature(path) != signature:
        raise RuntimeError("normalization contract changed before cached load")
    value = _normalization(path)
    if _file_signature(path) != signature:
        raise RuntimeError("normalization contract changed during cached load")
    return MappingProxyType(value)


@lru_cache(maxsize=64)
def _cached_calibration_manifest(
    path_text: str, signature: FileSignature
) -> tuple[Any, Mapping[str, Any]]:
    from .calibration import read_calibration_manifest

    path = Path(path_text)
    if _file_signature(path) != signature:
        raise RuntimeError("calibration manifest changed before cached load")
    calibrator, payload = read_calibration_manifest(path)
    if _file_signature(path) != signature:
        raise RuntimeError("calibration manifest changed during cached load")
    return calibrator, MappingProxyType(dict(payload))


@lru_cache(maxsize=64)
def _cached_checkpoint_manifest(
    path_text: str, signature: FileSignature
) -> Mapping[str, Any]:
    path = Path(path_text)
    if _file_signature(path) != signature:
        raise RuntimeError("checkpoint manifest changed before cached load")
    payload = _json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint manifest must be a JSON object")
    if _file_signature(path) != signature:
        raise RuntimeError("checkpoint manifest changed during cached load")
    return MappingProxyType(dict(payload))


class _IndexedBatches:
    SCHEMA = "paper2.training-index/v1"

    def __init__(
        self,
        index_path: Path,
        *,
        root: Path,
        device: Any,
        role: str,
        rank: int = 0,
        world_size: int = 1,
        shard: bool = False,
    ) -> None:
        from .training import deterministic_rank_indices

        payload = _json(index_path)
        if payload.get("schema") != self.SCHEMA:
            raise ValueError(f"training index schema must be {self.SCHEMA}")
        if payload.get("role") != role:
            raise ValueError(f"index role must be {role}")
        if payload.get("patch_size") != PATCH_SIZE:
            raise ValueError(f"training index patch_size must be {PATCH_SIZE}")
        if payload.get("overlap_fraction") != PATCH_OVERLAP:
            raise ValueError(f"training index overlap_fraction must be {PATCH_OVERLAP}")
        expected_stride = int(PATCH_SIZE * (1.0 - PATCH_OVERLAP))
        if payload.get("stride") != expected_stride:
            raise ValueError(f"training index stride must be {expected_stride}")
        asset_lock_sha256 = str(payload.get("asset_lock_sha256", ""))
        if len(asset_lock_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in asset_lock_sha256
        ):
            raise ValueError("training index must bind a lowercase asset_lock_sha256")
        raw_samples = payload.get("samples")
        if not isinstance(raw_samples, list) or not raw_samples:
            raise ValueError("training index has no samples")
        cache_indices = (
            set(
                deterministic_rank_indices(
                    len(raw_samples), rank=rank, world_size=world_size
                )
            )
            if device is not None and shard
            else (set(range(len(raw_samples))) if device is not None else set())
        )
        cached_arrays: dict[str, dict[str, np.ndarray]] = {}
        raw_tiles = payload.get("tiles")
        if not isinstance(raw_tiles, list) or not raw_tiles:
            raise ValueError("training index has no tile window contracts")
        tile_contracts: dict[str, tuple[str, tuple[int, int]]] = {}
        expected_windows: dict[str, set[tuple[int, int]]] = {}
        for raw_tile in raw_tiles:
            if not isinstance(raw_tile, dict):
                raise ValueError("training index tile contracts must be objects")
            tile_id = str(raw_tile.get("tile_id", ""))
            group_id = str(raw_tile.get("geographic_group_id", ""))
            raw_shape = raw_tile.get("shape")
            if (
                not tile_id
                or not group_id
                or not isinstance(raw_shape, list)
                or len(raw_shape) != 2
            ):
                raise ValueError("training index tile contract is incomplete")
            shape = (int(raw_shape[0]), int(raw_shape[1]))
            if tile_id in tile_contracts:
                raise ValueError(f"duplicate training tile contract: {tile_id}")
            tile_contracts[tile_id] = (group_id, shape)
            expected_windows[tile_id] = {
                (row, column)
                for row in sliding_window_starts(shape[0])
                for column in sliding_window_starts(shape[1])
            }
        samples: list[_IndexedSample] = []
        seen_paths: set[str] = set()
        tile_groups: dict[str, str] = {}
        observed_windows: dict[str, set[tuple[int, int]]] = {
            tile_id: set() for tile_id in tile_contracts
        }
        required_arrays = {"target", "zf", "omega", "condition", "valid_mask"}
        required_identity = {"tile_id", "geographic_group_id", "role"}
        for index, item in enumerate(raw_samples):
            if not isinstance(item, dict):
                raise ValueError(f"training index sample {index} must be an object")
            missing = sorted(
                {
                    "relative_path",
                    "sha256",
                    "tile_id",
                    "geographic_group_id",
                    "role",
                    "row_start",
                    "column_start",
                }.difference(item)
            )
            if missing:
                raise ValueError(f"training index sample {index} lacks fields: {missing}")
            relative_path = str(item["relative_path"])
            lexical = Path(relative_path)
            if lexical.is_absolute() or ".." in lexical.parts or lexical == Path("."):
                raise ValueError(
                    "training sample paths must be relative and cannot traverse parents"
                )
            if relative_path in seen_paths:
                raise ValueError(f"duplicate training sample path: {relative_path}")
            seen_paths.add(relative_path)
            sample_role = str(item["role"])
            tile_id = str(item["tile_id"])
            group_id = str(item["geographic_group_id"])
            if sample_role != role:
                raise ValueError(f"training sample {relative_path} role must be {role}")
            if not tile_id or not group_id:
                raise ValueError("training samples require nonempty tile and geographic group IDs")
            if tile_id not in tile_contracts:
                raise ValueError(f"training sample references undeclared tile {tile_id}")
            if tile_contracts[tile_id][0] != group_id:
                raise ValueError(f"training sample {tile_id} differs from its locked group")
            row_start = int(item["row_start"])
            column_start = int(item["column_start"])
            window = (row_start, column_start)
            if window not in expected_windows[tile_id]:
                raise ValueError(f"training sample {tile_id} has an unexpected window origin")
            if window in observed_windows[tile_id]:
                raise ValueError(f"training sample {tile_id} duplicates window {window}")
            observed_windows[tile_id].add(window)
            previous_group = tile_groups.setdefault(tile_id, group_id)
            if previous_group != group_id:
                raise ValueError(f"tile {tile_id} is assigned to multiple geographic groups")
            expected_sha256 = str(item["sha256"])
            if len(expected_sha256) != 64 or any(
                character not in "0123456789abcdef" for character in expected_sha256
            ):
                raise ValueError(f"training sample {relative_path} has an invalid SHA-256")
            path = _paper2_input(relative_path, root)
            if file_sha256(path) != expected_sha256:
                raise ValueError(f"training sample hash mismatch: {relative_path}")
            with np.load(path, allow_pickle=False) as archive:
                forbidden = [name for name in archive.files if "path" in name.lower()]
                if forbidden:
                    raise ValueError(f"training patch contains path-bearing fields: {forbidden}")
                missing_arrays = sorted(required_arrays.difference(archive.files))
                missing_identity = sorted(required_identity.difference(archive.files))
                if missing_arrays or missing_identity:
                    raise ValueError(
                        f"training patch {relative_path} lacks arrays={missing_arrays}, "
                        f"identity={missing_identity}"
                    )
                observed_identity = {
                    name: _archive_scalar_string(archive, name) for name in required_identity
                }
                expected_identity = {
                    "tile_id": tile_id,
                    "geographic_group_id": group_id,
                    "role": sample_role,
                }
                if observed_identity != expected_identity:
                    raise ValueError(f"training patch identity mismatch: {relative_path}")
                condition = np.asarray(archive["condition"])
                if condition.ndim != 3 or condition.shape[0] != 6:
                    raise ValueError("training condition must have shape (6,H,W)")
                spatial_shape = condition.shape[-2:]
                if spatial_shape != (PATCH_SIZE, PATCH_SIZE):
                    raise ValueError(f"training patches must be {PATCH_SIZE}x{PATCH_SIZE}")
                if not np.all(np.isfinite(condition)):
                    raise ValueError("training condition must be finite")
                for name in ("target", "zf", "omega", "valid_mask"):
                    value = np.asarray(archive[name])
                    if value.ndim not in {2, 3} or value.shape[-2:] != spatial_shape:
                        raise ValueError(f"training patch {relative_path} has misaligned {name}")
                    if value.ndim == 3 and value.shape[0] != 1:
                        raise ValueError(f"training patch {name} must have one channel")
                    if name == "valid_mask":
                        if value.dtype != np.bool_:
                            raise ValueError("training valid_mask must be boolean")
                    elif not np.all(np.isfinite(value)):
                        raise ValueError(f"training patch {name} must be finite")
                    if name == "omega" and np.any((value < 0.0) | (value > 1.0)):
                        raise ValueError("training omega must lie in [0,1]")
                if "shifted_omega" in archive.files:
                    shifted = np.asarray(archive["shifted_omega"])
                    if shifted.ndim not in {2, 3} or shifted.shape[-2:] != spatial_shape:
                        raise ValueError("shifted_omega must align with the training patch")
                    if shifted.ndim == 3 and shifted.shape[0] != 1:
                        raise ValueError("shifted_omega must have one channel")
                    if not np.all(np.isfinite(shifted)) or np.any(
                        (shifted < 0.0) | (shifted > 1.0)
                    ):
                        raise ValueError("shifted_omega must be finite and lie in [0,1]")
                if index in cache_indices:
                    names = ["target", "zf", "omega", "condition", "valid_mask"]
                    if "shifted_omega" in archive.files:
                        names.append("shifted_omega")
                    cached: dict[str, np.ndarray] = {}
                    for name in names:
                        value = np.asarray(archive[name])
                        if name != "condition" and value.ndim == 2:
                            value = value[None, ...]
                        cached[name] = np.ascontiguousarray(value, dtype=np.float32)
                    cached_arrays[relative_path] = cached
            samples.append(
                _IndexedSample(
                    relative_path=relative_path,
                    sha256=expected_sha256,
                    tile_id=tile_id,
                    geographic_group_id=group_id,
                    role=sample_role,
                    row_start=row_start,
                    column_start=column_start,
                    path=path,
                    file_signature=_file_signature(path),
                )
            )
        incomplete_tiles = [
            tile_id
            for tile_id in tile_contracts
            if observed_windows[tile_id] != expected_windows[tile_id]
        ]
        if incomplete_tiles:
            raise ValueError(f"training index has missing windows for tiles: {incomplete_tiles}")
        self.asset_lock_sha256 = asset_lock_sha256
        self.all_samples = tuple(samples)
        self.sample_set_sha256 = training_index_sample_set_sha256(
            role=role,
            asset_lock_sha256=asset_lock_sha256,
            tiles=raw_tiles,
            samples=[sample.contract for sample in self.all_samples],
        )
        if payload.get("sample_set_sha256") != self.sample_set_sha256:
            raise ValueError("training index sample_set_sha256 mismatch")
        if shard:
            indices = deterministic_rank_indices(len(samples), rank=rank, world_size=world_size)
            samples = [samples[index] for index in indices]
        self.samples = tuple(samples)
        self.paths = tuple(sample.path for sample in self.samples)
        self.device = device
        self._cached_arrays = {
            sample.relative_path: cached_arrays[sample.relative_path]
            for sample in self.samples
            if sample.relative_path in cached_arrays
        }
        self.cached_sample_count = len(self._cached_arrays)
        self.cached_bytes = sum(
            value.nbytes
            for arrays in self._cached_arrays.values()
            for value in arrays.values()
        )
        self.cache_protocol = (
            "verified-cpu-resident-rank-shard/v1"
            if device is not None
            else "disabled-index-validation-only/v1"
        )

    def _load_batch(self, sample: _IndexedSample):
        import torch

        path = sample.path
        if _file_signature(path) != sample.file_signature:
            raise RuntimeError(f"training sample changed after index verification: {path}")
        cached = self._cached_arrays.get(sample.relative_path)
        if cached is not None:
            return {
                name: torch.as_tensor(value[None, ...], device=self.device)
                for name, value in cached.items()
            }
        with np.load(path, allow_pickle=False) as archive:
            forbidden = [name for name in archive.files if "path" in name.lower()]
            if forbidden:
                raise ValueError(f"training patch contains path-bearing fields: {forbidden}")
            batch = {}
            observed_identity = {
                name: _archive_scalar_string(archive, name)
                for name in ("tile_id", "geographic_group_id", "role")
            }
            if observed_identity != {
                "tile_id": sample.tile_id,
                "geographic_group_id": sample.geographic_group_id,
                "role": sample.role,
            }:
                raise RuntimeError(
                    f"training sample identity changed after verification: {path}"
                )
            names = ["target", "zf", "omega", "condition", "valid_mask"]
            if "shifted_omega" in archive.files:
                names.append("shifted_omega")
            for name in names:
                value = np.asarray(archive[name])
                if name == "condition":
                    if value.ndim != 3 or value.shape[0] != 6:
                        raise ValueError("training condition must have shape (6,H,W)")
                elif value.ndim == 2:
                    value = value[None, ...]
                batch[name] = torch.as_tensor(
                    value[None, ...], device=self.device, dtype=torch.float32
                )
            return batch

    def __iter__(self):
        for sample in self.samples:
            yield self._load_batch(sample)

    def iter_from(self, offset: int):
        """Yield the exact infinite cyclic sequence beginning at a verified offset."""

        position = int(offset)
        if position < 0 or position != offset or not self.samples:
            raise ValueError("training index offset must be a nonnegative integer")
        index = position % len(self.samples)
        while True:
            yield self._load_batch(self.samples[index])
            index = (index + 1) % len(self.samples)


class _BudgetBalancedBatches:
    """Round-robin immutable v1 indices so every laser budget is seen equally."""

    SCHEMA = "paper2.multibudget-training-index/v1"

    def __init__(
        self,
        index_path: Path,
        *,
        root: Path,
        device: Any,
        role: str,
        rank: int = 0,
        world_size: int = 1,
        shard: bool = False,
    ) -> None:
        payload = _json(index_path)
        if payload.get("schema") != self.SCHEMA:
            raise ValueError(f"multibudget index schema must be {self.SCHEMA}")
        unhashed = dict(payload)
        declared_content = unhashed.pop("content_sha256", None)
        if declared_content != canonical_sha256(unhashed):
            raise ValueError("multibudget index content hash differs")
        if payload.get("role") != role:
            raise ValueError(f"multibudget index role must be {role}")
        budgets = tuple(int(value) for value in payload.get("budgets", []))
        if (
            budgets != tuple(sorted(set(budgets)))
            or budgets != (0, 20, 100)
            or payload.get("sampling_protocol")
            != "deterministic_round_robin_equal_budget/v1"
        ):
            raise ValueError("multibudget index must freeze K={0,20,100} round-robin sampling")
        raw_components = payload.get("components")
        if not isinstance(raw_components, list) or len(raw_components) != len(budgets):
            raise ValueError("multibudget index lacks one component per budget")
        expected_asset_hash = str(payload.get("asset_lock_sha256", ""))
        components: list[tuple[int, _IndexedBatches]] = []
        for budget, raw in zip(budgets, raw_components, strict=True):
            if not isinstance(raw, Mapping) or int(raw.get("budget", -1)) != budget:
                raise ValueError("multibudget components are not in frozen budget order")
            relative_path = str(raw.get("relative_path", ""))
            lexical = Path(relative_path)
            if lexical.is_absolute() or ".." in lexical.parts or lexical == Path("."):
                raise ValueError("multibudget component path is not a safe relative path")
            component_path = _paper2_input(relative_path, root)
            if file_sha256(component_path) != raw.get("file_sha256"):
                raise ValueError("multibudget component file hash differs")
            component = _IndexedBatches(
                component_path,
                root=root,
                device=device,
                role=role,
                rank=rank,
                world_size=world_size,
                shard=shard,
            )
            if (
                component.asset_lock_sha256 != expected_asset_hash
                or component.sample_set_sha256 != raw.get("sample_set_sha256")
                or len(component.all_samples) != int(raw.get("sample_count", -1))
            ):
                raise ValueError("multibudget component contract differs")
            components.append((budget, component))
        if len({component.asset_lock_sha256 for _, component in components}) != 1:
            raise ValueError("multibudget components bind different asset locks")
        self.budgets = budgets
        self.components = tuple(components)
        self.asset_lock_sha256 = expected_asset_hash
        self.sample_set_sha256 = str(declared_content)
        self.all_samples = tuple(
            sample
            for _, component in self.components
            for sample in component.all_samples
        )
        self.samples = tuple(
            sample for _, component in self.components for sample in component.samples
        )
        self.paths = tuple(sample.path for sample in self.samples)
        self.device = device
        self.cached_sample_count = sum(
            component.cached_sample_count for _, component in self.components
        )
        self.cached_bytes = sum(component.cached_bytes for _, component in self.components)
        self.cache_protocol = "verified-balanced-multibudget-components/v1"

    def _budget_batch(self, budget: int, component: _IndexedBatches, index: int):
        import torch

        batch = dict(component._load_batch(component.samples[index]))
        batch["laser_budget"] = torch.as_tensor(
            [budget], device=self.device, dtype=torch.int64
        )
        return batch

    def __iter__(self):
        maximum = max(len(component.samples) for _, component in self.components)
        for index in range(maximum):
            for budget, component in self.components:
                if index < len(component.samples):
                    yield self._budget_batch(budget, component, index)

    def iter_from(self, offset: int):
        position = int(offset)
        if position < 0 or position != offset:
            raise ValueError("multibudget index offset must be a nonnegative integer")
        count = len(self.components)
        while True:
            component_index = position % count
            cycle_index = position // count
            budget, component = self.components[component_index]
            sample_index = cycle_index % len(component.samples)
            yield self._budget_batch(budget, component, sample_index)
            position += 1


def _training_batches(
    index_path: Path,
    *,
    root: Path,
    device: Any,
    role: str,
    rank: int = 0,
    world_size: int = 1,
    shard: bool = False,
):
    schema = _json(index_path).get("schema")
    batch_type = _BudgetBalancedBatches if schema == _BudgetBalancedBatches.SCHEMA else _IndexedBatches
    return batch_type(
        index_path,
        root=root,
        device=device,
        role=role,
        rank=rank,
        world_size=world_size,
        shard=shard,
    )


def _validation_metric(
    model: Any,
    batches: _IndexedBatches,
    method: str,
    *,
    g_min: float,
    alpha: float,
    ddpm_cosine_offset: float = 0.008,
    bridge_steps: int = 1,
    validation_seed: int = 0,
) -> float:
    import torch

    from .objectives import (
        _cosine_alpha_bar,
        masked_condition,
        reverse_bridge_path,
    )

    if bridge_steps <= 0:
        raise ValueError("validation bridge_steps must be positive")
    spec = method_spec(method)
    values = []
    model.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(batches):
            zf = batch["zf"]
            target = batch["target"]
            valid = batch["valid_mask"]
            condition = masked_condition(batch["condition"], method)
            baseline = zf if spec.uses_laser else condition[:, :1]
            if method == "DDPM-DC":
                time_value = torch.full((zf.shape[0],), 0.5, device=zf.device, dtype=zf.dtype)
                alpha_bar = _cosine_alpha_bar(time_value, cosine_offset=ddpm_cosine_offset)[
                    :, None, None, None
                ]
                clean_residual = target - zf
                # Fixed, index-derived validation noise makes checkpoint selection
                # reproducible and independent of the training RNG stream.
                index = torch.arange(
                    clean_residual.numel(), device=zf.device, dtype=zf.dtype
                ).reshape_as(clean_residual)
                epsilon = torch.sin(index * 0.017 + 0.31)
                noisy = (
                    torch.sqrt(alpha_bar) * clean_residual + torch.sqrt(1.0 - alpha_bar) * epsilon
                )
                epsilon_hat = model(noisy, condition, time_value)
                residual_hat = (noisy - torch.sqrt(1.0 - alpha_bar) * epsilon_hat) / torch.sqrt(
                    alpha_bar
                ).clamp_min(1.0e-6)
                prediction = zf + residual_hat
            elif spec.generative:
                stream = torch.Generator(device=zf.device)
                stream.manual_seed(
                    int(validation_seed) + 1_000_003 * batch_index
                )
                prediction = reverse_bridge_path(
                    model,
                    zf=baseline,
                    condition=condition,
                    omega=batch["omega"],
                    method=method,
                    g_min=g_min,
                    alpha=alpha,
                    steps=bridge_steps,
                    generator=stream,
                    shifted_omega=batch.get("shifted_omega"),
                )
            else:
                residual = model(
                    baseline,
                    condition,
                    torch.ones(zf.shape[0], device=zf.device),
                )
                prediction = baseline + residual
            numerator = (((prediction - target) ** 2) * valid).sum()
            denominator = (((baseline - target) ** 2) * valid).sum().clamp_min(1.0e-12)
            values.append(float((numerator / denominator).cpu()))
    model.train()
    return float(np.mean(values))


def command_train(args: argparse.Namespace) -> int:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is unavailable in this runtime")
    import torch

    from .objectives import bridge_training_loss, ddpm_training_loss, deterministic_training_loss
    from .training import (
        TrainingConfig,
        configure_determinism,
        load_checkpoint_model_state,
        train_model,
    )

    root = Path(args.paper2_root).resolve()
    train_index = _paper2_input(args.training_index, root)
    val_index = _paper2_input(args.validation_index, root)
    environment_lock = _paper2_input(args.environment_lock, root)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    config = TrainingConfig(
        method=args.method,
        seed=args.seed,
        updates=args.updates,
        validation_interval=args.validation_interval,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        learning_rate_schedule=args.learning_rate_schedule,
        warmup_updates=args.warmup_updates,
        cosine_decay_updates=args.cosine_decay_updates,
        minimum_learning_rate_ratio=args.minimum_learning_rate_ratio,
        network_precision=args.network_precision,
        data_loader_workers=0,
        g_min=args.g_min,
        alpha=args.alpha,
        endpoint_weight=args.endpoint_weight,
        gradient_weight=args.gradient_weight,
        bridge_t_min=args.bridge_t_min,
        gradient_clip_norm=args.gradient_clip_norm,
        amp_forward_replay=args.amp_forward_replay,
        ddpm_schedule="cosine",
        ddpm_cosine_offset=args.ddpm_cosine_offset,
    )
    config.validate(formal=args.formal, world_size=world_size)
    formal_code_hash: str | None = None
    resume_checkpoint = (
        None if args.resume_checkpoint is None else _paper2_input(args.resume_checkpoint, root)
    )
    resume_manifest = (
        None
        if args.resume_manifest is None
        else _json(_paper2_input(args.resume_manifest, root))
    )
    if (resume_checkpoint is None) != (resume_manifest is None):
        raise ValueError("resume checkpoint and manifest must be supplied together")
    initial_checkpoint = (
        None
        if args.initial_checkpoint is None
        else _paper2_input(args.initial_checkpoint, root)
    )
    initial_manifest_path = (
        None
        if args.initial_manifest is None
        else _paper2_input(args.initial_manifest, root)
    )
    if (initial_checkpoint is None) != (initial_manifest_path is None):
        raise ValueError("initial checkpoint and manifest must be supplied together")
    if initial_checkpoint is not None and resume_checkpoint is not None:
        raise ValueError("warm-start initialization and exact resume are mutually exclusive")
    if args.formal and initial_checkpoint is not None:
        raise ValueError("formal training cannot warm-start from an external checkpoint")
    initialization = None
    initial_model_state = None
    if initial_checkpoint is not None and initial_manifest_path is not None:
        initial_manifest = _json(initial_manifest_path)
        source_training = initial_manifest.get("training_config")
        if not isinstance(source_training, Mapping) or source_training.get("method") != args.method:
            raise ValueError("initial checkpoint method differs from the requested method")
        if initial_manifest.get("status") != "complete":
            raise ValueError("initial checkpoint manifest must be complete")
        if file_sha256(initial_checkpoint) != initial_manifest.get("checkpoint_sha256"):
            raise ValueError("initial checkpoint bytes differ from their manifest")
        initial_state = torch.load(
            initial_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        initial_model_state = initial_state.get("model")
        if not isinstance(initial_model_state, Mapping) or not initial_model_state:
            raise ValueError("initial checkpoint lacks model state")
        initialization = {
            "protocol": "warm-start-model-weights-only/v1",
            "checkpoint_sha256": file_sha256(initial_checkpoint),
            "manifest_sha256": file_sha256(initial_manifest_path),
            "source_method": args.method,
            "source_selected_update": int(initial_manifest["selected_update"]),
            "source_validation_metric": float(initial_manifest["validation_metric"]),
            "optimizer_state_reused": False,
        }
    if args.formal:
        if not args.asset_lock or not args.role_manifest or not args.g2_g3_report:
            raise ValueError(
                "formal training requires --asset-lock, --role-manifest, and --g2-g3-report"
            )
        from .training_gate_runtime import (
            _filestore_process_group_kwargs,
            paper2_training_code_sha256,
            validate_formal_training_gate_report,
        )

        validate_formal_training_gate_report(
            paper2_root=root,
            report_path=_paper2_input(args.g2_g3_report, root),
            training_index=train_index,
            validation_index=val_index,
            environment_lock=environment_lock,
            asset_lock=_paper2_input(args.asset_lock, root),
            role_manifest=_paper2_input(args.role_manifest, root),
            # G2/G3 deliberately stresses the maximum two-GPU configuration.
            # A later blind real-data receipt may select one GPU per formal model.
            expected_world_size=2,
        )
        formal_code_hash = paper2_training_code_sha256(root)
        if args.formal_training_config:
            from .formal_training import (
                read_formal_training_config,
                training_config_for_identity,
            )

            frozen_training = read_formal_training_config(
                _paper2_input(args.formal_training_config, root), paper2_root=root
            )
            expected_training = training_config_for_identity(
                frozen_training, method=args.method, seed=args.seed
            )
            if config != expected_training or world_size != int(
                frozen_training["world_size"]
            ):
                raise ValueError("formal training invocation differs from its frozen config")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        process_group_kwargs: dict[str, Any] = {}
        if args.formal or os.environ.get("PAPER2_DDP_INIT_METHOD"):
            from .training_gate_runtime import _filestore_process_group_kwargs

            process_group_kwargs = _filestore_process_group_kwargs(
                paper2_root=root, rank=rank, world_size=world_size
            )
            if not process_group_kwargs and args.formal:
                raise ValueError("formal DDP training requires an explicit Paper2 FileStore")
        torch.distributed.init_process_group(
            "nccl",
            device_id=torch.device("cuda", local_rank),
            **process_group_kwargs,
        )
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    # Every DDP rank must construct the same initial weights. ``train_model``
    # switches to a rank-derived stochastic stream only after DDP broadcasts them.
    configure_determinism(config.seed)
    model_config = UNetConfig()
    model = build_model(model_config).to(device)
    if initial_model_state is not None:
        load_checkpoint_model_state(model, initial_model_state)
        del initial_model_state
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    train_batches = _training_batches(
        train_index,
        root=root,
        device=device,
        role="bridge_train",
        rank=rank,
        world_size=world_size,
        shard=world_size > 1,
    )
    val_batches = _training_batches(val_index, root=root, device=device, role="bridge_val")
    if args.formal and (
        isinstance(train_batches, _BudgetBalancedBatches)
        or isinstance(val_batches, _BudgetBalancedBatches)
    ):
        raise ValueError("formal training remains frozen to the K=100 v1 indices")
    if train_batches.asset_lock_sha256 != val_batches.asset_lock_sha256:
        raise ValueError("training and validation indices must bind the same asset-lock")
    if args.formal:
        from .assets import audit_asset_lock_geography
        from .contracts import AssetLock

        asset_path = _paper2_input(args.asset_lock, root)
        role_path = _paper2_input(args.role_manifest, root)
        if file_sha256(asset_path) != train_batches.asset_lock_sha256:
            raise ValueError("training indices bind a different formal asset-lock")
        asset_contract = AssetLock.read(asset_path)
        geography = audit_asset_lock_geography(asset_contract)
        if not geography.ok:
            raise ValueError("formal training asset-lock fails geography/QA audit")
        role_payload = _json(role_path)
        if role_payload.get("schema") != "paper2.role-manifest/v1":
            raise ValueError("formal training role manifest schema is invalid")
        asset_membership = {
            (asset.product_id, asset.geographic_group_id, asset.role)
            for asset in asset_contract.assets
        }
        for indexed, expected_role in (
            (train_batches, "bridge_train"),
            (val_batches, "bridge_val"),
        ):
            role_tiles = set(role_payload.get("roles", {}).get(expected_role, []))
            role_groups = set(role_payload.get("geographic_groups", {}).get(expected_role, []))
            for sample in indexed.all_samples:
                if (
                    sample.tile_id,
                    sample.geographic_group_id,
                    expected_role,
                ) not in asset_membership:
                    raise ValueError("training patch is not authorized by the formal asset-lock")
                if (
                    sample.tile_id not in role_tiles
                    or sample.geographic_group_id not in role_groups
                ):
                    raise ValueError("training patch is not authorized by the role manifest")
    spec = method_spec(args.method)
    replay_telemetry: dict[str, Any] | None = None
    if spec.generative and config.amp_forward_replay:
        replay_telemetry = {
            "protocol": "same-sample-amp-forward-replay/v1",
            "enabled": True,
            "accumulation_steps": config.accumulation_steps(world_size),
            "objective_calls": 0,
            "attempt_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "first_success_objective_call": None,
            "first_success_update": None,
        }

    def loss(model_value, batch):
        if args.method == "DDPM-DC":
            return ddpm_training_loss(
                model_value,
                batch,
                gradient_weight=config.gradient_weight,
                cosine_offset=config.ddpm_cosine_offset,
            )
        if spec.generative:
            return bridge_training_loss(
                model_value,
                batch,
                method=args.method,
                g_min=config.g_min,
                alpha=config.alpha,
                endpoint_weight=config.endpoint_weight,
                gradient_weight=config.gradient_weight,
                t_min=config.bridge_t_min,
                amp_forward_replay=config.amp_forward_replay,
                replay_telemetry=replay_telemetry,
            )
        return deterministic_training_loss(
            model_value, batch, method=args.method, gradient_weight=config.gradient_weight
        )

    manifest = train_model(
        model=model,
        train_batches=train_batches,
        validation=lambda model_value, _: _validation_metric(
            model_value,
            val_batches,
            args.method,
            g_min=config.g_min,
            alpha=config.alpha,
            ddpm_cosine_offset=config.ddpm_cosine_offset,
            bridge_steps=args.validation_steps,
            validation_seed=args.validation_seed,
        ),
        loss_function=loss,
        output=guard_paper2_path(args.output, paper2_root=root, allow_root=True),
        config=config,
        model_config=model_config,
        split_hashes={
            "bridge_train": train_batches.sample_set_sha256,
            "bridge_train_index": file_sha256(train_index),
            "bridge_val": val_batches.sample_set_sha256,
            "bridge_val_index": file_sha256(val_index),
        },
        code_hash=(
            formal_code_hash
            if formal_code_hash is not None
            # Non-formal pilots may run from an immutable Paper2 code snapshot
            # while the live G0 process retains the root source tree.  Bind the
            # checkpoint to the source tree that this interpreter actually
            # imported, which is identical to ``root/src`` in the normal case.
            else _tree_sha256(Path(__file__).resolve().parents[1])
        ),
        environment_hash=file_sha256(environment_lock),
        formal=args.formal,
        stop_update=args.stop_update,
        resume_checkpoint=resume_checkpoint,
        resume_manifest=resume_manifest,
        preserve_validation_history_on_resume=args.formal,
        objective_telemetry=replay_telemetry,
        initialization=initialization,
    )
    if world_size > 1:
        torch.distributed.destroy_process_group()
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(manifest, indent=2))
    return 0


def _normalization(path: Path) -> dict[str, float]:
    value = {key: float(item) for key, item in _json(path).items()}
    required = {"elevation_center", "elevation_scale", "image_center", "image_scale"}
    if set(value) != required or value["elevation_scale"] <= 0 or value["image_scale"] <= 0:
        raise ValueError(
            f"normalization must contain exactly {sorted(required)} with positive scales"
        )
    return value


def _condition_planes(
    problem: ReconstructionProblem,
    plan: PlanContract,
    zf: np.ndarray,
    omega: np.ndarray,
    method: str,
    normalization: Mapping[str, float],
) -> np.ndarray:
    return build_condition_planes(
        problem,
        plan,
        zf,
        omega,
        method=method,
        normalization=normalization,
    )


def command_infer(args: argparse.Namespace) -> int:
    root = Path(args.paper2_root).resolve()
    problem_path = _paper2_input(args.problem, root)
    plan_path = _paper2_input(args.plan, root)
    problem = _cached_reconstruction_problem(
        str(problem_path.resolve()), _file_signature(problem_path)
    )
    plan = _cached_plan_contract(str(plan_path.resolve()), _file_signature(plan_path))
    if problem.tile_id != plan.tile_id or problem.plan_sha256 != plan.sha256:
        raise ValueError("problem and plan do not share the same frozen identity")
    window_kind = "hann"
    if args.experiment == "E8":
        if args.condition.startswith("stitch-window="):
            window_kind = args.condition.split("=", 1)[1]
        elif problem.metadata.get("experimental_condition") != args.condition:
            raise ValueError(
                "E8 physical stress cells require a condition-specific reconstruction problem"
            )
    zf_path = _paper2_input(args.zf, root)
    omega_path = _paper2_input(args.omega, root)
    normalization_path = _paper2_input(args.normalization, root)
    spec = method_spec(args.method)
    zf, omega, gp_variance = _cached_condition_bundle(
        str(zf_path.resolve()),
        _file_signature(zf_path),
        args.zf_key,
        str(omega_path.resolve()),
        _file_signature(omega_path),
        args.omega_key,
        args.gp_variance_key if args.method == "gp-footprint" else None,
    )
    if zf.shape != problem.z0.shape or omega.shape != problem.z0.shape:
        raise ValueError("zf and omega must align with the reconstruction grid")
    norm = _cached_normalization(
        str(normalization_path.resolve()), _file_signature(normalization_path)
    )
    path_seed_set, path_seed = _path_seed_contract(
        args.experiment,
        args.noise_replicate,
        requested_seed=args.path_seed,
        requested_set=args.path_seed_set,
    )
    matrix_key = MatrixKey(
        experiment=args.experiment,
        tile_id=problem.tile_id,
        method=args.method,
        budget=plan.budget,
        noise_replicate=args.noise_replicate,
        training_seed=args.training_seed,
        condition=args.condition,
        model_identity=canonical_model_identity(args.method),
        path_seed_set=path_seed_set,
        path_count=args.paths if spec.generative else 1,
        step_count=args.steps if spec.generative else 0,
        dc_mode=args.dc_mode if spec.terminal_dc else "D0",
    )
    if args.calibration_fit_mode:
        if args.formal:
            raise ValueError("calibration-fit mode precedes G4/G5 and cannot be formal test inference")
        if not spec.generative or args.paths < 2:
            raise ValueError("calibration-fit mode requires at least two generative complete paths")
        if args.calibration_manifest or args.footprint_calibration_manifest:
            raise ValueError("calibration-fit mode must precede both calibration manifests")
        if not args.role_manifest or not args.asset_lock or not args.checkpoint_manifest:
            raise ValueError(
                "calibration-fit mode requires role, asset-lock, and checkpoint manifests"
            )
        calibration_roles = _json(_paper2_input(args.role_manifest, root))
        if (
            calibration_roles.get("schema") != "paper2.role-manifest/v1"
            or problem.tile_id
            not in calibration_roles.get("roles", {}).get("uncertainty_calibration", [])
        ):
            raise ValueError("calibration-fit inference is restricted to uncertainty_calibration")
    audit_plan_path: Path | None = None
    audit_plan = None
    audit_operator = None
    if args.audit_footprint_plan:
        audit_plan_path = _paper2_input(args.audit_footprint_plan, root)
        audit_plan, audit_operator = _cached_audit_binding(
            str(audit_plan_path.resolve()),
            _file_signature(audit_plan_path),
            str(problem_path.resolve()),
            _file_signature(problem_path),
            str(plan_path.resolve()),
            _file_signature(plan_path),
        )
    elif args.formal or args.calibration_fit_mode:
        raise ValueError("formal and calibration-fit inference require --audit-footprint-plan")
    selector_manifest_hash: str | None = None
    if args.experiment == "E5":
        if not args.selector_manifest:
            raise ValueError("E5 inference requires --selector-manifest")
        selector_prefix = "selector="
        if not args.condition.startswith(selector_prefix):
            raise ValueError("E5 condition must identify one selector instance")
        selector_manifest_hash = validate_e5_plan_binding(
            manifest_path=_paper2_input(args.selector_manifest, root),
            selector_instance=args.condition[len(selector_prefix) :],
            budget=plan.budget,
            plan_path=plan_path,
            paper2_root=root,
        )
    elif args.selector_manifest:
        raise ValueError("--selector-manifest is valid only for E5 inference")
    if args.experiment == "E7":
        validate_e7_key(matrix_key, source=True)
        if not spec.generative:
            raise ValueError("E7 M/N convergence is defined only for generative methods")
        if args.condition != e7_condition(args.paths, args.steps):
            raise ValueError("E7 condition, --paths, and --steps must encode the same cell")
    start = time.monotonic()
    checkpoint_hash: str | None = None
    checkpoint_manifest_hash = canonical_sha256({"checkpoint_manifest": "not-applicable"})
    calibration_manifest_hash = canonical_sha256({"calibration_manifest": "not-applied"})
    footprint_calibration_manifest_hash = canonical_sha256(
        {"footprint_calibration_manifest": "not-applied"}
    )
    pixel_calibrator = None
    model = None
    device = None
    e6_uncorrected_stack: np.ndarray | None = None
    e6_uncorrected_path: Path | None = None
    final_path_stack: np.ndarray | None = None
    if spec.learned:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is unavailable in this runtime")
        import torch

        checkpoint_path = _paper2_input(args.checkpoint, root)
        checkpoint_hash = file_sha256(checkpoint_path)
        if args.checkpoint_manifest:
            checkpoint_manifest_path = _paper2_input(args.checkpoint_manifest, root)
            checkpoint_contract = _cached_checkpoint_manifest(
                str(checkpoint_manifest_path.resolve()),
                _file_signature(checkpoint_manifest_path),
            )
            expected_identity = canonical_model_identity(args.method)
            if checkpoint_contract.get("status") != "complete":
                raise ValueError("checkpoint manifest is not complete")
            if checkpoint_contract.get("model_identity") != expected_identity:
                raise ValueError("checkpoint model identity differs from inference method")
            if int(checkpoint_contract.get("seed", -1)) != args.training_seed:
                raise ValueError("checkpoint training seed differs from matrix key")
            if checkpoint_contract.get("checkpoint_sha256") != checkpoint_hash:
                raise ValueError("checkpoint file differs from checkpoint manifest")
            checkpoint_manifest_hash = file_sha256(checkpoint_manifest_path)
        elif args.formal:
            raise ValueError("formal learned inference requires --checkpoint-manifest")
        device = torch.device(
            args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = _load_inference_model(checkpoint_path, checkpoint_hash, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
    if args.calibration_manifest:
        calibration_path = _paper2_input(args.calibration_manifest, root)
        pixel_calibrator, calibration_contract = _cached_calibration_manifest(
            str(calibration_path.resolve()), _file_signature(calibration_path)
        )
        if pixel_calibrator.target != "pixel":
            raise ValueError("inference requires a pixel variance calibration manifest")
        # F11 is not a separately fitted model/calibrator: the registered
        # ablation reuses the OT-SB-full identity and changes only inference fields.
        if calibration_contract.get("method") != canonical_model_identity(args.method):
            raise ValueError("calibration method differs from inference method")
        if spec.learned and calibration_contract.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError("calibration manifest binds a different checkpoint")
        calibration_manifest_hash = file_sha256(calibration_path)
    elif args.formal and spec.generative:
        raise ValueError("formal generative inference requires --calibration-manifest")
    if args.footprint_calibration_manifest:
        footprint_calibration_path = _paper2_input(
            args.footprint_calibration_manifest, root
        )
        footprint_calibrator_for_trace, footprint_calibration_contract = (
            _cached_calibration_manifest(
                str(footprint_calibration_path.resolve()),
                _file_signature(footprint_calibration_path),
            )
        )
        if footprint_calibrator_for_trace.target != "footprint":
            raise ValueError("inference trace requires a footprint calibration manifest")
        if canonical_model_identity(
            str(footprint_calibration_contract.get("method"))
        ) != canonical_model_identity(args.method):
            raise ValueError("footprint calibration method differs from inference")
        if spec.learned and footprint_calibration_contract.get(
            "checkpoint_sha256"
        ) != checkpoint_hash:
            raise ValueError("footprint calibration binds a different checkpoint")
        footprint_calibration_manifest_hash = file_sha256(
            footprint_calibration_path
        )
    elif args.formal and spec.generative:
        raise ValueError(
            "formal generative inference requires --footprint-calibration-manifest"
        )
    formal_lock_hashes: dict[str, str] = {}
    formal_input_lineage_hashes: dict[str, str] = {}
    formal_code_sha256: str | None = None
    for label, requested in (
        ("asset_lock_sha256", args.asset_lock),
        ("analysis_lock_sha256", args.analysis_lock),
        ("environment_sha256", args.environment_lock),
    ):
        if requested:
            formal_lock_hashes[label] = file_sha256(_paper2_input(requested, root))
        elif args.formal:
            raise ValueError(
                f"formal inference requires --{label.removesuffix('_sha256').replace('_', '-')}"
            )
        else:
            formal_lock_hashes[label] = canonical_sha256({label: "pilot-unlocked"})
    if args.formal:
        required_formal = {
            "g5_unlock": args.g5_unlock,
            "g4_freeze": args.g4_freeze,
            "role_manifest": args.role_manifest,
            "expected_matrix": args.expected_matrix,
            "matrix_spec": args.matrix_spec,
        }
        missing_formal = [name for name, value in required_formal.items() if not value]
        if missing_formal:
            raise ValueError(f"formal inference lacks G5 inputs: {missing_formal}")
        unlock_path = _paper2_input(args.g5_unlock, root)
        role_manifest_path = _paper2_input(args.role_manifest, root)
        expected_matrix_path = _paper2_input(args.expected_matrix, root)
        formal_paths = {
            "expected_matrix": expected_matrix_path,
            "matrix_spec": _paper2_input(args.matrix_spec, root),
            "analysis_lock": _paper2_input(args.analysis_lock, root),
            "g5_unlock": unlock_path,
            "g4_freeze": _paper2_input(args.g4_freeze, root),
            "role_manifest": role_manifest_path,
            "asset_lock": _paper2_input(args.asset_lock, root),
            "environment_lock": _paper2_input(args.environment_lock, root),
        }
        registered = _registered_formal_context(root=root, paths=formal_paths)
        if registered is not None:
            if registered.get("formal_lock_hashes") != formal_lock_hashes:
                raise ValueError("registered formal worker lock hashes differ")
            unlock = registered["unlock"]
            role_manifest = registered["role_manifest"]
            expected_by_id = registered["expected_by_id"]
            formal_code_sha256 = str(registered["source_code_sha256"])
        else:
            from .analysis import read_analysis_lock
            from .assets import audit_asset_lock_geography
            from .contracts import AssetLock
            from .g5 import validate_g5_unlock
            from .matrix import ExpectedMatrix

            validate_g5_unlock(
                unlock_path,
                matrix_spec_path=formal_paths["matrix_spec"],
                expected_matrix_path=expected_matrix_path,
                analysis_lock_path=formal_paths["analysis_lock"],
                role_manifest_path=role_manifest_path,
                asset_lock_path=formal_paths["asset_lock"],
                g4_freeze_path=formal_paths["g4_freeze"],
                paper2_root=root,
            )
            unlock = _json(unlock_path)
            read_analysis_lock(formal_paths["analysis_lock"])
            asset_contract = AssetLock.read(formal_paths["asset_lock"])
            asset_audit = audit_asset_lock_geography(asset_contract)
            if not asset_audit.ok:
                raise ValueError("formal inference asset lock fails geography/QA audit")
            role_manifest = _json(role_manifest_path)
            expected_contract = ExpectedMatrix.read(expected_matrix_path)
            if (
                expected_contract.analysis_lock_sha256
                != formal_lock_hashes["analysis_lock_sha256"]
            ):
                raise ValueError("expected matrix does not bind the selected analysis lock")
            expected_by_id = {
                cell.key.identifier: cell for cell in expected_contract.cells
            }
            formal_code_sha256 = _tree_sha256(root / "src")
        if (
            unlock.get("schema") != "paper2.g5-unlock/v1"
            or unlock.get("status") != "complete"
            or unlock.get("formal_test_unlocked") is not True
        ):
            raise ValueError("formal inference requires a completed G5 unlock")
        if unlock.get("asset_lock_sha256") != formal_lock_hashes["asset_lock_sha256"]:
            raise ValueError("G5 unlock asset lock differs from inference")
        if unlock.get("analysis_lock_sha256") != formal_lock_hashes["analysis_lock_sha256"]:
            raise ValueError("G5 unlock analysis lock differs from inference")
        if unlock.get("role_manifest_sha256") != file_sha256(role_manifest_path):
            raise ValueError("G5 unlock role manifest differs from inference")
        if unlock.get("expected_matrix_sha256") != file_sha256(expected_matrix_path):
            raise ValueError("G5 unlock expected matrix differs from inference")
        expected_role = {
            "E3": "paper2_locked_test",
            "E3b": "paper1_bridge_test",
            "E4": "paper2_locked_test",
            "E5": "paper1_bridge_test",
            "E6": "paper2_locked_test",
            "E7": "bridge_val",
            "E8": "bridge_val",
            "E9": "locked_external",
            "E10": "paper2_locked_test",
        }.get(args.experiment)
        if expected_role is None or problem.tile_id not in role_manifest.get("roles", {}).get(
            expected_role, []
        ):
            raise ValueError("tile is not authorized for this formal experiment role")
        expected_cell = expected_by_id.get(matrix_key.identifier)
        if expected_cell is None:
            raise ValueError("inference key is not present in the G5 expected matrix")
        locked_trace_hashes = dict(expected_cell.locked_trace_hashes)
        if not args.prepared_manifest:
            raise ValueError("formal inference requires --prepared-manifest")
        prepared_manifest_path = _paper2_input(args.prepared_manifest, root)
        from .input_lineage import (
            CTX_SOURCE_EXPERIMENTS,
            PAPER1_IMPORT_EXPERIMENTS,
            validate_ctx_reconstruction_lineage,
            validate_paper1_import_lineage,
        )

        if args.experiment in CTX_SOURCE_EXPERIMENTS:
            if not args.ctx_input_lineage_manifest:
                raise ValueError(
                    "formal CTX inference requires --ctx-input-lineage-manifest"
                )
            if args.experiment == "E10" and not args.e10_mola_problem_manifest:
                raise ValueError(
                    "formal E10 inference requires --e10-mola-problem-manifest"
                )
            formal_input_lineage_hashes = validate_ctx_reconstruction_lineage(
                experiment=args.experiment,
                problem_path=problem_path,
                plan_path=plan_path,
                prepared_manifest_path=prepared_manifest_path,
                ctx_input_lineage_path=_paper2_input(
                    args.ctx_input_lineage_manifest, root
                ),
                asset_lock_path=_paper2_input(args.asset_lock, root),
                paper2_root=root,
                e10_manifest_path=(
                    _paper2_input(args.e10_mola_problem_manifest, root)
                    if args.experiment == "E10"
                    else None
                ),
            )
            if args.experiment == "E8":
                if not args.e8_stress_manifest:
                    raise ValueError(
                        "formal E8 inference requires --e8-stress-manifest"
                    )
                from .e8_stress import validate_e8_stress_binding

                formal_input_lineage_hashes["e8_stress_manifest_sha256"] = (
                    validate_e8_stress_binding(
                        manifest_path=_paper2_input(args.e8_stress_manifest, root),
                        condition=args.condition,
                        budget=plan.budget,
                        noise_replicate=args.noise_replicate,
                        problem_path=problem_path,
                        plan_path=plan_path,
                        observations_path=None,
                        paper2_root=root,
                    )
                )
        elif args.experiment in PAPER1_IMPORT_EXPERIMENTS:
            if not args.paper1_import_manifest:
                raise ValueError(
                    "formal Paper1 bridge/external inference requires --paper1-import-manifest"
                )
            import_path = _paper2_input(args.paper1_import_manifest, root)
            if role_manifest.get("paper1_import_manifest_sha256") != file_sha256(
                import_path
            ):
                raise ValueError(
                    "formal Paper1 import manifest differs from the role manifest"
                )
            formal_input_lineage_hashes = validate_paper1_import_lineage(
                experiment=args.experiment,
                problem_path=problem_path,
                plan_path=plan_path,
                prepared_manifest_path=prepared_manifest_path,
                paper1_import_manifest_path=import_path,
                asset_lock_path=_paper2_input(args.asset_lock, root),
                paper2_root=root,
            )
        else:  # pragma: no cover - formal matrix spec is exhaustive
            raise ValueError(
                f"formal inference lacks an input-lineage policy for {args.experiment}"
            )
        observed_precompute_trace = {
            "problem_sha256": file_sha256(problem_path),
            "plan_sha256": file_sha256(plan_path),
            "observations_sha256": problem.observations_sha256,
            "prepared_manifest_sha256": file_sha256(prepared_manifest_path),
            **formal_input_lineage_hashes,
        }
        for field, observed in observed_precompute_trace.items():
            if locked_trace_hashes.get(field) != observed:
                raise ValueError(
                    f"formal inference {field} differs from the G5 trace lock"
                )
        locked_audit_hash = locked_trace_hashes.get(
            "audit_footprint_plan_sha256"
        )
        if (
            audit_plan_path is None
            or audit_plan is None
            or locked_audit_hash != file_sha256(audit_plan_path)
            or audit_plan.asset_lock_sha256 != formal_lock_hashes["asset_lock_sha256"]
        ):
            raise ValueError("audit footprint plan differs from the G5 trace/asset lock")
        if args.experiment == "E5" and locked_trace_hashes.get(
            "selector_manifest_sha256"
        ) != selector_manifest_hash:
            raise ValueError("E5 selector manifest differs from the G5 trace lock")
        from .phase_gates import G6_DELAYED_EXPERIMENTS, validate_g6_unlock

        if args.experiment in G6_DELAYED_EXPERIMENTS:
            if not args.g6_unlock:
                raise ValueError(
                    f"formal {args.experiment} inference is locked until G6 closes E3"
                )
            validate_g6_unlock(
                _paper2_input(args.g6_unlock, root),
                g5_unlock_path=unlock_path,
                expected_matrix_path=expected_matrix_path,
                analysis_lock_path=_paper2_input(args.analysis_lock, root),
                paper2_root=root,
            )
    condition = _condition_planes(problem, plan, zf, omega, args.method, norm)
    from .inference import infer_path_ensemble_batched, infer_paths

    if args.method == "coarse":
        mean = np.asarray(problem.z0, dtype=np.float64)
        variance = np.zeros_like(mean)
    elif args.method == "gp-footprint":
        mean = zf
        if gp_variance is None:
            if args.formal:
                raise ValueError(
                    "formal gp-footprint inference requires posterior variance in the Zf bundle"
                )
            variance = np.zeros_like(mean)
        else:
            variance = gp_variance
        if (
            variance.shape != mean.shape
            or np.any(variance < 0)
            or not np.all(np.isfinite(variance))
        ):
            raise ValueError("GP posterior variance must align with Zf and be finite/non-negative")
    else:
        import torch

        from .objectives import reverse_bridge_path, reverse_ddpm_path, shifted_omega_field

        elevation_center = norm["elevation_center"]
        elevation_scale = norm["elevation_scale"]
        use_inference_amp = args.network_precision == "fp16" and device.type == "cuda"
        generator = torch.Generator(device=device)
        shifted_omega = (
            shifted_omega_field(omega, tile_hash=file_sha256(problem_path))
            if args.method == "spatial-shift-M"
            else None
        )
        endpoint = zf if spec.uses_laser else problem.z0

        def predictor(location, path_index):
            rows, columns = location
            generator.manual_seed(_patch_path_seed(path_seed, path_index, location))
            endpoint_patch = torch.as_tensor(
                ((endpoint[rows, columns] - elevation_center) / elevation_scale)[None, None],
                device=device,
                dtype=torch.float32,
            )
            condition_patch = torch.as_tensor(
                condition[:, rows, columns][None], device=device, dtype=torch.float32
            )
            omega_patch = torch.as_tensor(
                omega[rows, columns][None, None], device=device, dtype=torch.float32
            )
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_inference_amp,
                ),
            ):
                if args.method == "DDPM-DC":
                    prediction = reverse_ddpm_path(
                        model,
                        zf=endpoint_patch,
                        condition=condition_patch,
                        steps=args.steps,
                        generator=generator,
                        cosine_offset=args.ddpm_cosine_offset,
                        eta=args.ddpm_eta,
                    )
                elif spec.generative:
                    prediction = reverse_bridge_path(
                        model,
                        zf=endpoint_patch,
                        condition=condition_patch,
                        omega=omega_patch,
                        method=args.method,
                        g_min=args.g_min,
                        alpha=args.alpha,
                        steps=args.steps,
                        generator=generator,
                        shifted_omega=(
                            torch.as_tensor(
                                shifted_omega[rows, columns][None, None],
                                device=device,
                                dtype=torch.float32,
                            )
                            if shifted_omega is not None
                            else None
                        ),
                    )
                else:
                    residual = model(
                        endpoint_patch,
                        condition_patch,
                        torch.zeros(1, device=device),
                    )
                    prediction = endpoint_patch + residual
            return prediction.float().cpu().numpy()[0, 0] * elevation_scale + elevation_center

        def batched_predictor(location, path_indices):
            rows, columns = location
            batch_size = len(path_indices)
            if batch_size <= 0:
                raise ValueError("batched inference received no path indices")
            endpoint_patch = torch.as_tensor(
                ((endpoint[rows, columns] - elevation_center) / elevation_scale)[
                    None, None
                ],
                device=device,
                dtype=torch.float32,
            ).expand(batch_size, -1, -1, -1).contiguous()
            condition_patch = torch.as_tensor(
                condition[:, rows, columns][None],
                device=device,
                dtype=torch.float32,
            ).expand(batch_size, -1, -1, -1).contiguous()
            omega_patch = torch.as_tensor(
                omega[rows, columns][None, None],
                device=device,
                dtype=torch.float32,
            ).expand(batch_size, -1, -1, -1).contiguous()
            generators = []
            for path_index in path_indices:
                stream = torch.Generator(device=device)
                stream.manual_seed(_patch_path_seed(path_seed, path_index, location))
                generators.append(stream)
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_inference_amp,
                ),
            ):
                if args.method == "DDPM-DC":
                    prediction = reverse_ddpm_path(
                        model,
                        zf=endpoint_patch,
                        condition=condition_patch,
                        steps=args.steps,
                        generator=tuple(generators),
                        cosine_offset=args.ddpm_cosine_offset,
                        eta=args.ddpm_eta,
                    )
                elif spec.generative:
                    prediction = reverse_bridge_path(
                        model,
                        zf=endpoint_patch,
                        condition=condition_patch,
                        omega=omega_patch,
                        method=args.method,
                        g_min=args.g_min,
                        alpha=args.alpha,
                        steps=args.steps,
                        generator=tuple(generators),
                        shifted_omega=(
                            torch.as_tensor(
                                shifted_omega[rows, columns][None, None],
                                device=device,
                                dtype=torch.float32,
                            )
                            .expand(batch_size, -1, -1, -1)
                            .contiguous()
                            if shifted_omega is not None
                            else None
                        ),
                    )
                else:  # pragma: no cover - only generative paths use this callback
                    raise AssertionError("batched predictor is reserved for generative methods")
            return prediction.float().cpu().numpy()[:, 0] * elevation_scale + elevation_center

        def generate_ensemble(*, terminal_dc, retain_path_stack):
            return infer_path_ensemble_batched(
                tile_shape=problem.z0.shape,
                path_count=args.paths,
                path_batch_size=min(args.path_batch_size, args.paths),
                patch_predictor=batched_predictor,
                terminal_dc=terminal_dc,
                patch_size=256,
                overlap=0.25,
                window_kind=window_kind,
                retain_path_stack=retain_path_stack,
            )

        good = problem.observation_quality_flags == 0
        operator_metadata = _cached_measurement_footprint(
            str(problem_path.resolve()),
            _file_signature(problem_path),
            str(plan_path.resolve()),
            _file_signature(plan_path),
            not args.formal,
        )
        retained_values = problem.observation_values_m[good][operator_metadata.kept_indices]

        def dc(tile):
            if not spec.terminal_dc:
                return tile
            from .dc import terminal_data_consistency

            point = None
            if args.dc_mode == "D3":
                point = build_footprint_operator_with_metadata(
                    problem.observation_coordinates[good][operator_metadata.kept_indices],
                    grid_shape=problem.z0.shape,
                    transform=plan.transform,
                    diameter_m=0.0,
                    quadrature_points=1,
                    valid_mask=problem.valid_mask,
                    require_full_support=True,
                    drop_invalid=False,
                ).operator
            return terminal_data_consistency(
                tile,
                operator_metadata.operator,
                retained_values,
                noise_covariance=plan.noise_std_m**2,
                mobility=mobility_from_uncertainty(
                    omega,
                    g_min=args.g_min,
                    alpha=args.alpha,
                ).ravel(),
                tau_dc=args.tau_dc,
                delta=args.dc_delta,
                lambda_dc=args.lambda_dc,
                mode=args.dc_mode,
                point_operator=point,
            ).corrected

        if args.experiment == "E6":
            if not spec.generative:
                raise ValueError("E6 shared uncorrected paths require a generative method")
            if args.dc_mode == "D0":
                e6_uncorrected_path = guard_paper2_path(
                    args.e6_uncorrected_path_stack
                    or Path(args.output).with_name("e6-uncorrected-path-stack.npz"),
                    paper2_root=root,
                )
                ensemble = generate_ensemble(
                    terminal_dc=lambda tile: tile,
                    retain_path_stack=True,
                )
                if ensemble.path_stack is None:
                    raise AssertionError("E6 D0 did not retain its uncorrected paths")
                e6_uncorrected_stack = np.asarray(
                    ensemble.path_stack, dtype=np.float32
                )
                final_path_stack = e6_uncorrected_stack
                mean, variance = prefix_statistics(e6_uncorrected_stack, args.paths)
            else:
                if not args.e6_uncorrected_path_stack:
                    raise ValueError(
                        "E6 D1/D3 require the D0 --e6-uncorrected-path-stack artifact"
                    )
                e6_uncorrected_path = _paper2_input(
                    args.e6_uncorrected_path_stack, root
                )
                with np.load(e6_uncorrected_path, allow_pickle=False) as archive:
                    if "paths" not in archive.files:
                        raise ValueError("E6 uncorrected path artifact lacks paths")
                    e6_uncorrected_stack = np.asarray(
                        archive["paths"], dtype=np.float32
                    )
                if (
                    e6_uncorrected_stack.shape
                    != (args.paths, *problem.z0.shape)
                    or not np.all(np.isfinite(e6_uncorrected_stack))
                ):
                    raise ValueError("E6 uncorrected path stack shape/values differ")
                corrected_stack = np.stack(
                    [dc(path) for path in e6_uncorrected_stack], axis=0
                ).astype(np.float32)
                final_path_stack = corrected_stack
                mean, variance = prefix_statistics(corrected_stack, args.paths)
        elif args.experiment == "E7":
            ensemble = generate_ensemble(
                terminal_dc=dc,
                retain_path_stack=True,
            )
            if ensemble.path_stack is None:
                raise AssertionError("E7 source inference did not retain its path stack")
            final_path_stack = np.asarray(ensemble.path_stack, dtype=np.float32)
            mean, variance = prefix_statistics(ensemble.path_stack, args.paths)
        else:
            if spec.generative:
                ensemble = generate_ensemble(
                    terminal_dc=dc,
                    retain_path_stack=True,
                )
                if ensemble.path_stack is None:
                    raise AssertionError("generative inference did not retain complete paths")
                final_path_stack = np.asarray(ensemble.path_stack, dtype=np.float32)
                mean, variance = prefix_statistics(final_path_stack, args.paths)
            else:
                ensemble = None
                mean, variance = infer_paths(
                    tile_shape=problem.z0.shape,
                    path_count=1,
                    patch_predictor=predictor,
                    terminal_dc=dc,
                    patch_size=256,
                    overlap=0.25,
                    window_kind=window_kind,
                )
    if final_path_stack is None:
        final_path_stack = np.asarray(mean, dtype=np.float32)[None, ...]
    if pixel_calibrator is not None:
        variance = pixel_calibrator.apply(variance)
    output = guard_paper2_path(args.output, paper2_root=root)
    stream = __import__("io").BytesIO()
    np.savez_compressed(stream, mean=mean.astype(np.float32), variance=variance.astype(np.float32))
    write_immutable_bytes(output, stream.getvalue(), paper2_root=root)
    extra_artifacts: list[ArtifactDigest] = []
    if audit_plan_path is not None and audit_plan is not None and audit_operator is not None:
        audit_projection_path = output.with_name("audit-footprint-path-projections.npz")
        write_audit_path_projections(
            audit_projection_path,
            audit_plan_path=audit_plan_path,
            audit_plan=audit_plan,
            operator=audit_operator,
            full_dem_paths=final_path_stack,
            matrix_key_identifier=matrix_key.identifier,
            generative=spec.generative,
            paper2_root=root,
        )
        extra_artifacts.append(
            ArtifactDigest(
                "audit_footprint_projection",
                audit_projection_path.relative_to(root).as_posix(),
                file_sha256(audit_projection_path),
            )
        )
    uncorrected_path_hash: str | None = None
    if args.experiment == "E6":
        if e6_uncorrected_stack is None or e6_uncorrected_path is None:
            raise AssertionError("E6 lost its shared uncorrected path contract")
        if args.dc_mode == "D0":
            stack_stream = __import__("io").BytesIO()
            np.savez_compressed(stack_stream, paths=e6_uncorrected_stack)
            write_immutable_bytes(
                e6_uncorrected_path, stack_stream.getvalue(), paper2_root=root
            )
        uncorrected_path_hash = file_sha256(e6_uncorrected_path)
        extra_artifacts.append(
            ArtifactDigest(
                "e6_uncorrected_path_stack",
                e6_uncorrected_path.relative_to(root).as_posix(),
                uncorrected_path_hash,
            )
        )
    if args.experiment == "E7":
        if ensemble is None or ensemble.path_stack is None:
            raise AssertionError("E7 source path stack was lost before persistence")
        stack_path = output.with_name("e7-path-stack.npz")
        stack_stream = __import__("io").BytesIO()
        np.savez_compressed(stack_stream, paths=np.asarray(ensemble.path_stack, dtype=np.float32))
        write_immutable_bytes(stack_path, stack_stream.getvalue(), paper2_root=root)
        calibration_scale = float(pixel_calibrator.scale) if pixel_calibrator else 1.0
        calibration_floor = float(pixel_calibrator.floor_variance) if pixel_calibrator else 0.0
        source_payload = build_e7_source_payload(
            key=matrix_key,
            path_seed=path_seed,
            path_stack=np.asarray(ensemble.path_stack, dtype=np.float32),
            path_stack_sha256=file_sha256(stack_path),
            prediction_sha256=file_sha256(output),
            calibration_manifest_sha256=calibration_manifest_hash,
            calibration_scale=calibration_scale,
            calibration_floor_variance=calibration_floor,
            mechanism_parameters={
                "g_min": args.g_min,
                "alpha": args.alpha,
                "ddpm_cosine_offset": args.ddpm_cosine_offset,
                "ddpm_eta": args.ddpm_eta,
                "generative": spec.generative,
            },
        )
        source_path = output.with_name("e7-prefix-source.json")
        write_canonical_json(source_path, source_payload, paper2_root=root)
        extra_artifacts.extend(
            (
                ArtifactDigest(
                    "e7_path_stack",
                    stack_path.relative_to(root).as_posix(),
                    file_sha256(stack_path),
                ),
                ArtifactDigest(
                    "e7_prefix_source",
                    source_path.relative_to(root).as_posix(),
                    file_sha256(source_path),
                ),
            )
        )
    runtime_path = output.with_name("runtime.json")
    wall = time.monotonic() - start
    nominal_nfe = int(args.paths * args.steps if spec.generative else int(spec.learned))
    executed_nfe = (
        0
        if args.experiment == "E6" and args.dc_mode in {"D1", "D3"}
        else nominal_nfe
    )
    pixel_area_m2, area_km2, valid_pixel_count = _valid_mask_area_contract(
        plan.transform, problem.valid_mask
    )
    runtime = {
        "schema": "paper2.inference-runtime/v1",
        "status": "complete",
        "wall_seconds": wall,
        "area_km2": area_km2,
        "area_basis": "affine_pixel_determinant_times_reconstruction_valid_mask_count",
        "valid_pixel_count": valid_pixel_count,
        "pixel_area_m2": pixel_area_m2,
        "seconds_per_km2": wall / area_km2,
        "nfe": executed_nfe,
        "shared_source_nfe": nominal_nfe if args.experiment == "E6" else 0,
        "path_batch_size": min(args.path_batch_size, args.paths) if spec.generative else 1,
        "reused_uncorrected_path": args.experiment == "E6" and args.dc_mode in {"D1", "D3"},
        "peak_gpu_memory_bytes": None,
        "peak_gpu_memory_allocated_bytes": None,
        "peak_gpu_memory_reserved_bytes": None,
        "device_index": None,
        "device_name": None,
        "device_uuid": None,
    }
    if TORCH_AVAILABLE and device is not None and device.type == "cuda":
        import torch

        allocated = int(torch.cuda.max_memory_allocated(device))
        runtime["peak_gpu_memory_bytes"] = allocated
        runtime["peak_gpu_memory_allocated_bytes"] = allocated
        runtime["peak_gpu_memory_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(device)
        )
        device_index, device_name, device_uuid = _cuda_device_identity(device)
        runtime["device_index"] = device_index
        runtime["device_name"] = device_name
        runtime["device_uuid"] = device_uuid
    write_canonical_json(runtime_path, runtime, paper2_root=root)
    manifest = InferenceManifest(
        key=matrix_key,
        status="complete",
        trace_hashes=tuple(
            sorted(
                {
                    "problem_sha256": file_sha256(problem_path),
                    "plan_sha256": file_sha256(plan_path),
                    "observations_sha256": problem.observations_sha256,
                    "checkpoint_sha256": checkpoint_hash
                    or canonical_sha256({"method": args.method}),
                    "normalization_sha256": file_sha256(normalization_path),
                    "zf_sha256": file_sha256(zf_path),
                    "omega_sha256": file_sha256(omega_path),
                    "prepared_manifest_sha256": str(
                        problem.metadata.get(
                            "prepared_manifest_sha256",
                            canonical_sha256({"prepared_manifest": "missing"}),
                        )
                    ),
                    "frozen_models_sha256": canonical_sha256(dict(problem.frozen_model_hashes)),
                    "checkpoint_manifest_sha256": checkpoint_manifest_hash,
                    "calibration_manifest_sha256": calibration_manifest_hash,
                    "footprint_calibration_manifest_sha256": footprint_calibration_manifest_hash,
                    **(
                        {"audit_footprint_plan_sha256": file_sha256(audit_plan_path)}
                        if audit_plan_path is not None
                        else {}
                    ),
                    **(
                        {"calibration_fit_mode_sha256": CALIBRATION_FIT_MODE_SHA256}
                        if args.calibration_fit_mode
                        else {}
                    ),
                    "mechanism_parameters_sha256": mechanism_parameters_sha256(
                        g_min=args.g_min,
                        alpha=args.alpha,
                        ddpm_cosine_offset=args.ddpm_cosine_offset,
                        ddpm_eta=args.ddpm_eta,
                        steps=args.steps,
                        paths=args.paths,
                        generative=spec.generative,
                    ),
                    "dc_parameters_sha256": dc_parameters_sha256(
                        mode=args.dc_mode,
                        terminal_dc=spec.terminal_dc,
                        tau_dc=args.tau_dc,
                        delta=args.dc_delta,
                        lambda_dc=args.lambda_dc,
                        footprint_diameter_m=plan.footprint.diameter_m,
                        quadrature_points=plan.footprint.quadrature_points,
                    ),
                    **formal_lock_hashes,
                    **formal_input_lineage_hashes,
                    **(
                        {"selector_manifest_sha256": selector_manifest_hash}
                        if selector_manifest_hash is not None
                        else {}
                    ),
                    **(
                        {"uncorrected_path_sha256": uncorrected_path_hash}
                        if uncorrected_path_hash is not None
                        else {}
                    ),
                    "code_sha256": formal_code_sha256 or _tree_sha256(root / "src"),
                }.items()
            )
        ),
        artifacts=(
            ArtifactDigest("prediction", output.relative_to(root).as_posix(), file_sha256(output)),
            ArtifactDigest(
                "runtime", runtime_path.relative_to(root).as_posix(), file_sha256(runtime_path)
            ),
            *extra_artifacts,
        ),
    )
    manifest_path = manifest.write(output.with_name("inference-manifest.json"), paper2_root=root)
    print(json.dumps({"prediction": str(output), "manifest": str(manifest_path)}, indent=2))
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    root = Path(args.paper2_root).resolve()
    prediction = _paper2_input(args.prediction, root)
    sealed_reference = _paper2_input(args.sealed_reference, root)
    inference_manifest_path = _paper2_input(args.inference_manifest, root)
    inference_manifest = InferenceManifest.read(inference_manifest_path)
    evaluation_spec = method_spec(inference_manifest.key.method)
    audit_plan_path: Path | None = None
    audit_observations_path: Path | None = None
    if args.audit_footprint_plan and args.audit_footprint_observations:
        audit_plan_path = _paper2_input(args.audit_footprint_plan, root)
        audit_observations_path = _paper2_input(args.audit_footprint_observations, root)
    elif args.formal:
        raise ValueError(
            "formal evaluation requires audit footprint plan and held-out observations"
        )
    elif args.audit_footprint_plan or args.audit_footprint_observations:
        raise ValueError("audit footprint plan and observations must be supplied together")
    footprint_calibrator = None
    footprint_calibration_hash = canonical_sha256({"footprint_calibration_manifest": "not-applied"})
    if args.footprint_calibration_manifest:
        footprint_calibration_path = _paper2_input(args.footprint_calibration_manifest, root)
        footprint_calibrator, footprint_calibration = _cached_calibration_manifest(
            str(footprint_calibration_path.resolve()),
            _file_signature(footprint_calibration_path),
        )
        if footprint_calibrator.target != "footprint":
            raise ValueError("evaluation requires a footprint calibration manifest")
        if canonical_model_identity(
            str(footprint_calibration.get("method"))
        ) != canonical_model_identity(inference_manifest.key.method):
            raise ValueError("footprint calibration method differs from inference")
        if evaluation_spec.learned and footprint_calibration.get("checkpoint_sha256") != dict(
            inference_manifest.trace_hashes
        ).get("checkpoint_sha256"):
            raise ValueError("footprint calibration differs from inference checkpoint")
        footprint_calibration_hash = file_sha256(footprint_calibration_path)
    elif evaluation_spec.generative:
        raise ValueError("generative evaluation requires --footprint-calibration-manifest")
    problem_path = _paper2_input(args.problem, root)
    plan_path = _paper2_input(args.plan, root)
    problem = _cached_reconstruction_problem(
        str(problem_path.resolve()), _file_signature(problem_path)
    )
    plan = _cached_plan_contract(str(plan_path.resolve()), _file_signature(plan_path))
    if problem.tile_id != inference_manifest.key.tile_id or problem.tile_id != plan.tile_id:
        raise ValueError("evaluation problem/plan tile differs from inference key")
    if problem.plan_sha256 != plan.sha256:
        raise ValueError("evaluation problem and plan hashes do not match")
    from .sealed_reference import read_sealed_reference_lock, reference_grid_sha256

    sealed_grid_sha256 = reference_grid_sha256(
        tile_id=problem.tile_id,
        shape=problem.z0.shape,
        crs_wkt=plan.crs_wkt,
        transform=plan.transform,
    )
    reference_lock_file_sha256 = canonical_sha256(
        {"sealed_reference_lock": "pilot-unlocked"}
    )
    if args.reference_lock:
        reference_lock_path = _paper2_input(args.reference_lock, root)
        locked_reference = _registered_formal_reference(
            root=root,
            reference_lock_path=reference_lock_path,
            tile_id=problem.tile_id,
        )
        if locked_reference is None:
            reference_lock = read_sealed_reference_lock(
                reference_lock_path, paper2_root=root, verify_files=True
            )
            matches = [
                entry
                for entry in reference_lock["entries"]
                if str(entry["tile_id"]) == problem.tile_id
            ]
            if len(matches) != 1:
                raise ValueError("sealed reference lock lacks one exact inference tile")
            locked_reference = matches[0]
        if (
            locked_reference["reference_relative_path"]
            != sealed_reference.relative_to(root).as_posix()
            or locked_reference["reference_sha256"] != file_sha256(sealed_reference)
            or locked_reference["grid_sha256"] != sealed_grid_sha256
            or locked_reference["problem_sha256"] != plan.problem_sha256
        ):
            raise ValueError("sealed reference file/grid/problem differs from its prelock")
        reference_lock_file_sha256 = file_sha256(reference_lock_path)
    elif args.formal:
        raise ValueError("formal evaluation requires --reference-lock")
    inference_trace = dict(inference_manifest.trace_hashes)
    if inference_trace.get("footprint_calibration_manifest_sha256") != footprint_calibration_hash:
        raise ValueError("evaluation footprint calibration differs from inference/G5 trace")
    if inference_trace.get("problem_sha256") != file_sha256(problem_path):
        raise ValueError("evaluation problem differs from inference manifest")
    if inference_trace.get("plan_sha256") != file_sha256(plan_path):
        raise ValueError("evaluation plan differs from inference manifest")
    prediction_artifacts = {artifact.name: artifact for artifact in inference_manifest.artifacts}
    if "prediction" not in prediction_artifacts:
        raise ValueError("inference manifest has no prediction artifact")
    locked_prediction = prediction_artifacts["prediction"]
    if file_sha256(prediction) != locked_prediction.sha256:
        raise ValueError("prediction differs from the bound inference artifact")
    with np.load(prediction, allow_pickle=False) as archive:
        estimate = np.asarray(archive["mean"], dtype=np.float64)
        variance = (
            np.asarray(archive["variance"], dtype=np.float64)
            if "variance" in archive.files
            else None
        )
    reference, valid = _cached_sealed_reference_arrays(
        str(sealed_reference.resolve()), _file_signature(sealed_reference)
    )
    if reference.shape != problem.z0.shape or valid.shape != problem.z0.shape:
        raise ValueError("sealed reference grid shape differs from reconstruction problem")
    metrics: dict[str, Any] = dem_metrics(
        estimate, reference, valid, pixel_size_m=args.pixel_size_m
    )
    if variance is not None and np.all(variance[valid] > 0):
        metrics["probabilistic"] = gaussian_probabilistic_metrics(
            estimate,
            np.sqrt(variance),
            reference,
            valid,
        )
    good = problem.observation_quality_flags == 0
    footprint = _cached_measurement_footprint(
        str(problem_path.resolve()),
        _file_signature(problem_path),
        str(plan_path.resolve()),
        _file_signature(plan_path),
        False,
    )
    retained = problem.observation_values_m[good][footprint.kept_indices]
    metrics["footprint_consistency"] = footprint_consistency_metrics(
        estimate,
        footprint.operator,
        retained,
        noise_variance=plan.noise_std_m**2,
    )
    audit_hashes: dict[str, str] = {}
    if audit_plan_path is not None and audit_observations_path is not None:
        audit_plan, audit_operator = _cached_audit_binding(
            str(audit_plan_path.resolve()),
            _file_signature(audit_plan_path),
            str(problem_path.resolve()),
            _file_signature(problem_path),
            str(plan_path.resolve()),
            _file_signature(plan_path),
        )
        if inference_trace.get("audit_footprint_plan_sha256") != file_sha256(
            audit_plan_path
        ):
            raise ValueError("evaluation audit plan differs from inference trace")
        projection_artifacts = [
            artifact
            for artifact in inference_manifest.artifacts
            if artifact.name == "audit_footprint_projection"
        ]
        if len(projection_artifacts) != 1:
            raise ValueError("inference lacks one exact audit footprint projection artifact")
        projection_artifact = projection_artifacts[0]
        projection_path = _paper2_input(root / projection_artifact.relative_path, root)
        if file_sha256(projection_path) != projection_artifact.sha256:
            raise ValueError("audit footprint projection artifact bytes differ")
        projection_metadata, _, audit_mean, audit_variance = read_audit_path_projections(
            projection_path
        )
        if (
            projection_metadata.get("matrix_key_identifier")
            != inference_manifest.key.identifier
            or projection_metadata.get("audit_plan_file_sha256")
            != file_sha256(audit_plan_path)
        ):
            raise ValueError("audit path projection identity differs")
        audit_observations = AuditFootprintObservations.read(audit_observations_path)
        if (
            audit_observations.tile_id != problem.tile_id
            or audit_observations.replicate != inference_manifest.key.noise_replicate
            or audit_observations.audit_plan_file_sha256 != file_sha256(audit_plan_path)
            or audit_observations.audit_plan_contract_sha256 != audit_plan.sha256
            or audit_observations.noise_std_m != plan.noise_std_m
            or audit_observations.audit_ids.tolist()
            != [point.audit_id for point in audit_plan.points]
        ):
            raise ValueError("held-out audit observations differ from inference/plan pairing")
        expected_audit_observations = simulate_audit_footprint_observations(
            audit_plan_path=audit_plan_path,
            reference_path=sealed_reference,
            replicate=inference_manifest.key.noise_replicate,
            noise_std_m=plan.noise_std_m,
        )
        if not np.array_equal(
            audit_observations.return_values_m,
            expected_audit_observations.return_values_m,
        ):
            raise ValueError(
                "held-out audit returns do not reproduce from the prelocked noise protocol"
            )
        supported = np.unique(audit_operator.indices)
        if not np.all(valid.ravel()[supported]) or not np.all(
            np.isfinite(reference.ravel()[supported])
        ):
            raise ValueError("sealed reference lacks full support for a prelocked audit footprint")
        clean_audit = np.asarray(
            audit_operator @ np.where(valid, reference, 0.0).ravel(), dtype=np.float64
        )
        metrics["audit_footprints"] = audit_footprint_metrics(
            predicted_mean=audit_mean,
            path_sample_variance=audit_variance,
            clean_reference=clean_audit,
            heldout_returns=audit_observations.return_values_m,
            instrument_noise_variance=audit_observations.noise_std_m**2,
            calibration_scale=(
                footprint_calibrator.scale if footprint_calibrator is not None else None
            ),
            calibration_floor_variance=(
                footprint_calibrator.floor_variance
                if footprint_calibrator is not None
                else 0.0
            ),
        )
        audit_hashes = {
            "audit_footprint_plan_sha256": file_sha256(audit_plan_path),
            "audit_footprint_observations_sha256": file_sha256(
                audit_observations_path
            ),
            "audit_footprint_projection_sha256": file_sha256(projection_path),
        }
    result = {
        "schema": "paper2.evaluation/v1",
        "key": inference_manifest.key.to_dict(),
        "inference_manifest_sha256": inference_manifest.sha256,
        "prediction_sha256": file_sha256(prediction),
        "sealed_reference_sha256": file_sha256(sealed_reference),
        "sealed_reference_grid_sha256": sealed_grid_sha256,
        "sealed_reference_lock_sha256": reference_lock_file_sha256,
        "problem_sha256": file_sha256(problem_path),
        "plan_sha256": file_sha256(plan_path),
        "evaluation_code_sha256": _tree_sha256(root / "src"),
        "footprint_calibration_manifest_sha256": footprint_calibration_hash,
        **audit_hashes,
        "metrics": metrics,
    }
    output = write_canonical_json(args.output, result, paper2_root=root)
    manifest = EvaluationManifest(
        key=inference_manifest.key,
        status="complete",
        inference_manifest_sha256=inference_manifest.sha256,
        sealed_reference_sha256=file_sha256(sealed_reference),
        sealed_reference_grid_sha256=sealed_grid_sha256,
        artifacts=(
            ArtifactDigest("metrics", output.relative_to(root).as_posix(), file_sha256(output)),
        ),
    )
    manifest_path = manifest.write(output.with_name("evaluation-manifest.json"), paper2_root=root)
    print(
        json.dumps(
            {"output": str(output), "manifest": str(manifest_path), **result},
            indent=2,
        )
    )
    return 0


def command_finalize(args: argparse.Namespace) -> int:
    root = Path(args.paper2_root).resolve()
    manifests = sorted(root.glob(args.manifest_glob))
    evaluations = sorted(root.glob(args.evaluation_manifest_glob))
    success = finalize_expected_matrix(
        args.expected_matrix,
        args.analysis_lock,
        manifests,
        args.output,
        evaluation_manifest_paths=evaluations,
        g7_manifest_path=args.g7_manifest,
        matrix_spec_path=args.matrix_spec,
        g5_unlock_path=args.g5_unlock,
        g6_unlock_path=args.g6_unlock,
        role_manifest_path=args.role_manifest,
        asset_lock_path=args.asset_lock,
        g4_freeze_path=args.g4_freeze,
        paper2_root=root,
    )
    print(json.dumps(_json(success), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paper2dem")
    parser.add_argument("--paper2-root", default=str(PAPER2_ROOT))
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit-assets")
    audit.add_argument("--catalog")
    audit.add_argument("--catalog-source-lock")
    audit.add_argument("--benchmark-lock")
    audit.add_argument("--output")
    audit.add_argument("--require-ready", action="store_true")
    audit.add_argument("--processed-lock")
    audit.add_argument("--edr-index")
    audit.add_argument("--mola-source-lock")
    audit.add_argument("--terrain-manifest")
    audit.add_argument("--paper1-seen-lock")
    audit.add_argument("--conflict-groups")
    audit.add_argument("--draft-asset-lock")
    audit.add_argument("--formal-asset-lock")
    audit.add_argument("--assignment-parent")
    audit.add_argument("--expected-assignment-parent-file-sha256")
    audit.add_argument("--ctx-observation-id")
    audit.add_argument("--ode-cache-json")
    audit.add_argument("--write-mola-manifest", action="store_true")
    audit.add_argument("--download-url")
    audit.add_argument("--download-output")
    audit.add_argument("--download-receipt")
    audit.add_argument("--expected-sha256")
    audit.set_defaults(function=command_audit_assets)

    plans = commands.add_parser("import-paper1-plans")
    plans.add_argument("--selection", required=True)
    plans.add_argument("--problem", required=True)
    plans.add_argument("--prefix", action="append", default=[], help="K=/path/selection.json")
    plans.add_argument("--budgets", default="0,5,10,20,50,100")
    plans.add_argument("--output", required=True)
    plans.set_defaults(function=command_import_plans)

    prepare = commands.add_parser("prepare-data")
    prepare.add_argument("--tile-id")
    prepare.add_argument("--z0")
    prepare.add_argument("--z0-key")
    prepare.add_argument("--image")
    prepare.add_argument("--image-key")
    prepare.add_argument("--mask")
    prepare.add_argument("--mask-key")
    prepare.add_argument("--observations")
    prepare.add_argument("--models")
    prepare.add_argument("--prepared-manifest-sha256")
    prepare.add_argument("--output")
    prepare.add_argument("--ctx-edr")
    prepare.add_argument("--ctx-output-directory")
    prepare.add_argument("--isis-prefix")
    prepare.add_argument("--isis-data")
    prepare.add_argument("--spice-cache")
    prepare.add_argument("--conda-cache")
    prepare.add_argument("--mola-radius-label")
    prepare.add_argument("--mola-radius-raw-cube")
    prepare.add_argument("--mola-radius-prepared-cube")
    prepare.add_argument("--map-template")
    prepare.add_argument("--isis-preferences", default=".envs/isis/IsisPreferences")
    prepare.add_argument("--spatial-summing", type=int)
    prepare.add_argument("--dry-run", action="store_true")
    prepare.add_argument("--isis-run-manifest")
    prepare.add_argument("--gp-problem")
    prepare.add_argument("--gp-plan")
    prepare.add_argument("--gp-observations")
    prepare.add_argument("--gp-prior")
    prepare.add_argument("--gp-posterior-output")
    prepare.add_argument("--gp-manifest-output")
    prepare.add_argument("--patch-problem")
    prepare.add_argument("--patch-plan")
    prepare.add_argument("--patch-gp-manifest")
    prepare.add_argument("--patch-gp-posterior")
    prepare.add_argument("--patch-asset-lock")
    prepare.add_argument("--patch-reference")
    prepare.add_argument("--patch-reference-dem-key", default="dem")
    prepare.add_argument("--patch-reference-mask-key", default="valid_mask")
    prepare.add_argument("--patch-normalization")
    prepare.add_argument("--patch-output-directory")
    prepare.add_argument("--training-index-output")
    prepare.add_argument("--merge-training-index", action="append", default=[])
    prepare.set_defaults(function=command_prepare)

    simulate = commands.add_parser("simulate-observations")
    simulate.add_argument("--plan", required=True)
    simulate.add_argument("--sealed-reference", required=True)
    simulate.add_argument("--replicate", type=int, required=True)
    simulate.add_argument("--seed", type=int, default=20260721)
    simulate.add_argument("--output", required=True)
    simulate.set_defaults(function=command_simulate)

    train = commands.add_parser("train")
    train.add_argument("--training-index", required=True)
    train.add_argument("--validation-index", required=True)
    train.add_argument("--environment-lock", required=True)
    train.add_argument("--asset-lock")
    train.add_argument("--role-manifest")
    train.add_argument("--g2-g3-report")
    train.add_argument("--formal-training-config")
    train.add_argument("--method", required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--updates", type=int, default=120000)
    train.add_argument("--validation-interval", type=int, default=2000)
    train.add_argument("--learning-rate", type=float, default=2.0e-4)
    train.add_argument("--weight-decay", type=float, default=1.0e-2)
    train.add_argument(
        "--learning-rate-schedule",
        choices=("constant", "warmup_cosine"),
        default="constant",
    )
    train.add_argument("--warmup-updates", type=int, default=0)
    train.add_argument("--cosine-decay-updates", type=int)
    train.add_argument("--minimum-learning-rate-ratio", type=float, default=0.0)
    train.add_argument("--network-precision", choices=("fp16", "fp32"), default="fp16")
    train.add_argument("--g-min", type=float, default=0.05)
    train.add_argument("--alpha", type=float, default=1.0)
    train.add_argument("--endpoint-weight", type=float, default=0.1)
    train.add_argument("--gradient-weight", type=float, default=0.1)
    train.add_argument("--bridge-t-min", type=float, default=1.0e-3)
    train.add_argument("--gradient-clip-norm", type=float)
    train.add_argument("--amp-forward-replay", action="store_true")
    train.add_argument("--ddpm-cosine-offset", type=float, default=0.008)
    train.add_argument("--validation-steps", type=int, default=1)
    train.add_argument("--validation-seed", type=int, default=0)
    train.add_argument("--formal", action="store_true")
    train.add_argument("--stop-update", type=int)
    train.add_argument("--resume-checkpoint")
    train.add_argument("--resume-manifest")
    train.add_argument("--initial-checkpoint")
    train.add_argument("--initial-manifest")
    train.add_argument("--output", required=True)
    train.set_defaults(function=command_train)

    infer = commands.add_parser("infer")
    infer.add_argument("--problem", required=True)
    infer.add_argument("--plan", required=True)
    infer.add_argument("--zf", required=True)
    infer.add_argument("--zf-key")
    infer.add_argument("--gp-variance-key", default="variance")
    infer.add_argument("--omega", required=True)
    infer.add_argument("--omega-key")
    infer.add_argument("--normalization", required=True)
    infer.add_argument("--prepared-manifest")
    infer.add_argument("--ctx-input-lineage-manifest")
    infer.add_argument("--paper1-import-manifest")
    infer.add_argument("--e10-mola-problem-manifest")
    infer.add_argument("--e8-stress-manifest")
    infer.add_argument("--checkpoint")
    infer.add_argument("--checkpoint-manifest")
    infer.add_argument("--calibration-manifest")
    infer.add_argument("--footprint-calibration-manifest")
    infer.add_argument("--audit-footprint-plan")
    infer.add_argument("--calibration-fit-mode", action="store_true")
    infer.add_argument("--asset-lock")
    infer.add_argument("--analysis-lock")
    infer.add_argument("--environment-lock")
    infer.add_argument("--g5-unlock")
    infer.add_argument("--g4-freeze")
    infer.add_argument("--g6-unlock")
    infer.add_argument("--role-manifest")
    infer.add_argument("--expected-matrix")
    infer.add_argument("--matrix-spec")
    infer.add_argument("--selector-manifest")
    infer.add_argument("--method", required=True)
    infer.add_argument("--experiment", required=True)
    infer.add_argument("--condition", default="main")
    infer.add_argument("--noise-replicate", type=int, required=True)
    infer.add_argument("--training-seed", type=int, default=0)
    infer.add_argument("--path-seed", type=int)
    infer.add_argument("--path-seed-set")
    infer.add_argument("--paths", type=int, default=16)
    infer.add_argument("--steps", type=int, default=20)
    infer.add_argument("--path-batch-size", type=int, default=32)
    infer.add_argument("--g-min", type=float, default=0.05)
    infer.add_argument("--alpha", type=float, default=1.0)
    infer.add_argument("--ddpm-cosine-offset", type=float, default=0.008)
    infer.add_argument("--ddpm-eta", type=float, default=1.0)
    infer.add_argument("--dc-mode", choices=("D0", "D1", "D3"), default="D1")
    infer.add_argument("--tau-dc", type=float, default=1.0)
    infer.add_argument("--dc-delta", type=float, default=1.0e-4)
    infer.add_argument("--lambda-dc", type=float, default=1.0)
    infer.add_argument("--e6-uncorrected-path-stack")
    infer.add_argument("--device")
    infer.add_argument("--network-precision", choices=("fp16", "fp32"), default="fp16")
    infer.add_argument("--formal", action="store_true")
    infer.add_argument("--output", required=True)
    infer.set_defaults(function=command_infer)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--prediction", required=True)
    evaluate.add_argument("--inference-manifest", required=True)
    evaluate.add_argument("--problem", required=True)
    evaluate.add_argument("--plan", required=True)
    evaluate.add_argument("--footprint-calibration-manifest")
    evaluate.add_argument("--audit-footprint-plan")
    evaluate.add_argument("--audit-footprint-observations")
    evaluate.add_argument("--sealed-reference", required=True)
    evaluate.add_argument("--reference-lock")
    evaluate.add_argument("--formal", action="store_true")
    evaluate.add_argument("--pixel-size-m", type=float, required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(function=command_evaluate)

    finalize = commands.add_parser("final-audit")
    finalize.add_argument("--expected-matrix", required=True)
    finalize.add_argument("--analysis-lock", required=True)
    finalize.add_argument("--manifest-glob", default="runs/**/inference-manifest.json")
    finalize.add_argument("--evaluation-manifest-glob", default="runs/**/evaluation-manifest.json")
    finalize.add_argument("--g7-manifest", required=True)
    finalize.add_argument("--matrix-spec", required=True)
    finalize.add_argument("--g5-unlock", required=True)
    finalize.add_argument("--g6-unlock", required=True)
    finalize.add_argument("--role-manifest", required=True)
    finalize.add_argument("--asset-lock", required=True)
    finalize.add_argument("--g4-freeze", required=True)
    finalize.add_argument("--output", required=True)
    finalize.set_defaults(function=command_finalize)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(arguments.function(arguments))
    except Exception as exc:
        print(f"paper2dem: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

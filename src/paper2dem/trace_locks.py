from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from .audit_footprints import read_audit_footprint_plan, validate_audit_plan_binding
from .calibration import read_calibration_manifest
from .contracts import PlanContract, ReconstructionProblem
from .e5_selectors import validate_e5_plan_binding
from .e8_stress import validate_e8_stress_binding
from .hashing import canonical_sha256, file_sha256, guard_paper2_path
from .input_lineage import (
    CTX_SOURCE_EXPERIMENTS,
    PAPER1_IMPORT_EXPERIMENTS,
    validate_ctx_reconstruction_lineage,
    validate_paper1_import_lineage,
)
from .matrix import ExpectedCell
from .models import canonical_model_identity, method_spec
from .provenance import dc_parameters_sha256, mechanism_parameters_sha256


def _file(binding: Mapping[str, Any], name: str, root: Path) -> Path:
    value = binding.get(name)
    if not value:
        raise ValueError(f"trace binding lacks {name}")
    path = guard_paper2_path(str(value), paper2_root=root)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


@lru_cache(maxsize=None)
def _hash(path_text: str) -> str:
    return file_sha256(Path(path_text))


@lru_cache(maxsize=16_384)
def _trace_input_facts(
    problem_path_text: str,
    plan_path_text: str,
    observations_path_text: str,
    prepared_path_text: str,
    audit_plan_path_text: str,
    asset_path_text: str,
) -> dict[str, Any]:
    problem_path = Path(problem_path_text)
    plan_path = Path(plan_path_text)
    observations_path = Path(observations_path_text)
    prepared_path = Path(prepared_path_text)
    audit_plan_path = Path(audit_plan_path_text)
    asset_path = Path(asset_path_text)
    problem = ReconstructionProblem.read(problem_path)
    plan = PlanContract.read(plan_path)
    audit_plan = read_audit_footprint_plan(audit_plan_path)
    validate_audit_plan_binding(audit_plan, problem=problem, measurement_plan=plan)
    if audit_plan.asset_lock_sha256 != _hash(str(asset_path)):
        raise ValueError("trace audit footprint plan differs from the asset lock")
    if problem.plan_sha256 != plan.sha256:
        raise ValueError("trace problem does not bind the selected plan")
    if problem.observations_sha256 != _hash(str(observations_path)):
        raise ValueError("trace observations differ from reconstruction problem")
    if problem.metadata.get("prepared_manifest_sha256") != _hash(str(prepared_path)):
        raise ValueError("trace prepared manifest differs from reconstruction problem")
    return {
        "tile_id": problem.tile_id,
        "budget": plan.budget,
        "problem_plan_sha256": problem.plan_sha256,
        "plan_contract_sha256": plan.sha256,
        "frozen_models_sha256": canonical_sha256(dict(problem.frozen_model_hashes)),
        "footprint_diameter_m": plan.footprint.diameter_m,
        "footprint_quadrature_points": plan.footprint.quadrature_points,
    }


@lru_cache(maxsize=16_384)
def _ctx_lineage_hashes(
    experiment: str,
    problem_path_text: str,
    plan_path_text: str,
    prepared_path_text: str,
    lineage_path_text: str,
    asset_path_text: str,
    e10_manifest_path_text: str,
    root_text: str,
) -> dict[str, str]:
    return validate_ctx_reconstruction_lineage(
        experiment=experiment,
        problem_path=problem_path_text,
        plan_path=plan_path_text,
        prepared_manifest_path=prepared_path_text,
        ctx_input_lineage_path=lineage_path_text,
        asset_lock_path=asset_path_text,
        paper2_root=root_text,
        e10_manifest_path=e10_manifest_path_text or None,
    )


@lru_cache(maxsize=8_192)
def _paper1_lineage_hashes(
    experiment: str,
    problem_path_text: str,
    plan_path_text: str,
    prepared_path_text: str,
    import_path_text: str,
    asset_path_text: str,
    root_text: str,
) -> dict[str, str]:
    return validate_paper1_import_lineage(
        experiment=experiment,
        problem_path=problem_path_text,
        plan_path=plan_path_text,
        prepared_manifest_path=prepared_path_text,
        paper1_import_manifest_path=import_path_text,
        asset_lock_path=asset_path_text,
        paper2_root=root_text,
    )


@lru_cache(maxsize=256)
def _calibration(path_text: str) -> tuple[Any, Mapping[str, Any]]:
    return read_calibration_manifest(Path(path_text))


@lru_cache(maxsize=64)
def _json_manifest(path_text: str) -> Mapping[str, Any]:
    return __import__("json").loads(Path(path_text).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _code_hash(root_text: str) -> str:
    root = Path(root_text)
    return canonical_sha256(
        [
            (path.relative_to(root / "src").as_posix(), _hash(str(path)))
            for path in sorted((root / "src").rglob("*.py"))
        ]
    )


def build_cell_trace_lock(
    cell: ExpectedCell,
    binding: Mapping[str, Any],
    *,
    paper2_root: str | Path,
) -> dict[str, str]:
    root = Path(paper2_root).resolve()
    key = cell.key
    spec = method_spec(key.method)
    problem_path = _file(binding, "problem", root)
    plan_path = _file(binding, "plan", root)
    observations_path = _file(binding, "observations", root)
    prepared_path = _file(binding, "prepared_manifest", root)
    zf_path = _file(binding, "zf", root)
    omega_path = _file(binding, "omega", root)
    normalization_path = _file(binding, "normalization", root)
    asset_path = _file(binding, "asset_lock", root)
    environment_path = _file(binding, "environment_lock", root)
    audit_plan_path = _file(binding, "audit_footprint_plan", root)
    facts = _trace_input_facts(
        str(problem_path),
        str(plan_path),
        str(observations_path),
        str(prepared_path),
        str(audit_plan_path),
        str(asset_path),
    )
    if facts["tile_id"] != key.tile_id or facts["budget"] != key.budget:
        raise ValueError("trace problem/plan does not match the expected matrix key")

    input_lineage_hashes: dict[str, str]
    if key.experiment in CTX_SOURCE_EXPERIMENTS:
        source_preparation_path = _file(
            binding, "source_preparation_manifest", root
        )
        lineage_path = _file(binding, "ctx_input_lineage_manifest", root)
        e10_path = (
            _file(binding, "e10_mola_problem_manifest", root)
            if key.experiment == "E10"
            else None
        )
        input_lineage_hashes = _ctx_lineage_hashes(
            key.experiment,
            str(problem_path),
            str(plan_path),
            str(prepared_path),
            str(lineage_path),
            str(asset_path),
            "" if e10_path is None else str(e10_path),
            str(root),
        )
        if input_lineage_hashes["source_preparation_manifest_sha256"] != _hash(
            str(source_preparation_path)
        ):
            raise ValueError(
                "trace source-preparation bytes differ from the truth-free input lineage"
            )
    elif key.experiment in PAPER1_IMPORT_EXPERIMENTS:
        import_path = _file(binding, "paper1_import_manifest", root)
        input_lineage_hashes = _paper1_lineage_hashes(
            key.experiment,
            str(problem_path),
            str(plan_path),
            str(prepared_path),
            str(import_path),
            str(asset_path),
            str(root),
        )
    else:  # pragma: no cover - formal matrix spec must remain exhaustive
        raise ValueError(f"formal trace lacks an input-lineage policy for {key.experiment}")

    checkpoint_hash = canonical_sha256({"method": key.method})
    checkpoint_manifest_hash = canonical_sha256({"checkpoint_manifest": "not-applicable"})
    if spec.learned:
        checkpoint_path = _file(binding, "checkpoint", root)
        checkpoint_manifest_path = _file(binding, "checkpoint_manifest", root)
        checkpoint_hash = _hash(str(checkpoint_path))
        checkpoint_manifest = _json_manifest(str(checkpoint_manifest_path))
        if checkpoint_manifest.get("status") != "complete":
            raise ValueError("trace checkpoint manifest is not complete")
        if checkpoint_manifest.get("model_identity") != canonical_model_identity(key.method):
            raise ValueError("trace checkpoint model identity differs from matrix key")
        if int(checkpoint_manifest.get("seed", -1)) != key.training_seed:
            raise ValueError("trace checkpoint seed differs from matrix key")
        if checkpoint_manifest.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError("trace checkpoint file differs from its manifest")
        checkpoint_manifest_hash = _hash(str(checkpoint_manifest_path))

    calibration_hash = canonical_sha256({"calibration_manifest": "not-applied"})
    footprint_calibration_hash = canonical_sha256(
        {"footprint_calibration_manifest": "not-applied"}
    )
    if spec.generative:
        calibration_path = _file(binding, "calibration_manifest", root)
        calibrator, calibration = _calibration(str(calibration_path))
        if calibrator.target != "pixel":
            raise ValueError("inference trace requires the pixel calibration manifest")
        if canonical_model_identity(str(calibration.get("method"))) != canonical_model_identity(
            key.method
        ):
            raise ValueError("trace calibration identity differs from matrix key")
        if calibration.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError("trace calibration differs from checkpoint")
        calibration_hash = _hash(str(calibration_path))
        footprint_calibration_path = _file(
            binding, "footprint_calibration_manifest", root
        )
        footprint_calibrator, footprint_calibration = _calibration(
            str(footprint_calibration_path)
        )
        if footprint_calibrator.target != "footprint":
            raise ValueError("inference trace requires the footprint calibration manifest")
        if canonical_model_identity(
            str(footprint_calibration.get("method"))
        ) != canonical_model_identity(key.method):
            raise ValueError("trace footprint calibration identity differs from matrix key")
        if footprint_calibration.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError("trace footprint calibration differs from checkpoint")
        footprint_calibration_hash = _hash(str(footprint_calibration_path))

    settings = binding.get("settings")
    if not isinstance(settings, Mapping):
        raise ValueError("trace binding lacks a settings object")
    trace = {
        "problem_sha256": _hash(str(problem_path)),
        "plan_sha256": _hash(str(plan_path)),
        "observations_sha256": _hash(str(observations_path)),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_manifest_sha256": checkpoint_manifest_hash,
        "normalization_sha256": _hash(str(normalization_path)),
        "zf_sha256": _hash(str(zf_path)),
        "omega_sha256": _hash(str(omega_path)),
        "prepared_manifest_sha256": _hash(str(prepared_path)),
        "frozen_models_sha256": str(facts["frozen_models_sha256"]),
        "calibration_manifest_sha256": calibration_hash,
        "footprint_calibration_manifest_sha256": footprint_calibration_hash,
        "audit_footprint_plan_sha256": _hash(str(audit_plan_path)),
        "mechanism_parameters_sha256": mechanism_parameters_sha256(
            g_min=float(settings["g_min"]),
            alpha=float(settings["alpha"]),
            ddpm_cosine_offset=float(settings["ddpm_cosine_offset"]),
            ddpm_eta=float(settings["ddpm_eta"]),
            steps=key.step_count,
            paths=key.path_count,
            generative=spec.generative,
        ),
        "dc_parameters_sha256": dc_parameters_sha256(
            mode=key.dc_mode,
            terminal_dc=spec.terminal_dc,
            tau_dc=float(settings["tau_dc"]),
            delta=float(settings["dc_delta"]),
            lambda_dc=float(settings["lambda_dc"]),
            footprint_diameter_m=float(facts["footprint_diameter_m"]),
            quadrature_points=int(facts["footprint_quadrature_points"]),
        ),
        "asset_lock_sha256": _hash(str(asset_path)),
        "code_sha256": _code_hash(str(root)),
        "environment_sha256": _hash(str(environment_path)),
        **input_lineage_hashes,
    }
    if key.experiment == "E5":
        prefix = "selector="
        if not key.condition.startswith(prefix):
            raise ValueError("E5 matrix condition does not identify a selector instance")
        selector = key.condition[len(prefix) :]
        trace["selector_manifest_sha256"] = validate_e5_plan_binding(
            manifest_path=_file(binding, "selector_manifest", root),
            selector_instance=selector,
            budget=key.budget,
            plan_path=plan_path,
            paper2_root=root,
        )
    if key.experiment == "E8":
        trace["e8_stress_manifest_sha256"] = validate_e8_stress_binding(
            manifest_path=_file(binding, "e8_stress_manifest", root),
            condition=key.condition,
            budget=key.budget,
            noise_replicate=key.noise_replicate,
            problem_path=problem_path,
            plan_path=plan_path,
            observations_path=observations_path,
            paper2_root=root,
        )
    lockable = set(cell.required_trace_hashes).difference(
        {"analysis_lock_sha256", "uncorrected_path_sha256"}
    )
    if set(trace) != lockable:
        raise ValueError(
            f"trace implementation/spec fields differ: missing={sorted(lockable - set(trace))}, "
            f"extra={sorted(set(trace) - lockable)}"
        )
    return trace

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import math

from .hashing import (
    canonical_sha256,
    file_sha256,
    guard_paper2_path,
    is_sha256,
    read_json,
    write_canonical_json,
)


EVIDENCE_NAMES = {
    "asset_lock",
    "role_manifest",
    "formal_training_config",
    "training_gate_report",
    "e7_validation_report",
    "validation_curve_freeze",
    "dc_validation_report",
    "power_report",
}
CALIBRATION_SELECTION_RULE = "minimum_nll_scale_floor_on_calibration_role"


def _strict_training_gate(path: Path, *, root: Path) -> Mapping[str, Any]:
    gate = read_json(path)
    evidence = gate.get("evidence") if isinstance(gate, Mapping) else None
    runtime = evidence.get("runtime_inputs") if isinstance(evidence, Mapping) else None
    if not isinstance(runtime, Mapping):
        raise ValueError("G4 training-gate report lacks its complete runtime input map")
    from .training_gate_runtime import validate_formal_training_gate_report

    return validate_formal_training_gate_report(
        paper2_root=root,
        report_path=path,
        training_index=root / str(runtime.get("training_index", "")),
        validation_index=root / str(runtime.get("validation_index", "")),
        environment_lock=root / str(runtime.get("environment_lock", "")),
        asset_lock=root / str(runtime.get("asset_lock", "")),
        role_manifest=root / str(runtime.get("role_manifest", "")),
        expected_world_size=2,
    )


def _strict_formal_training_config(path: Path, *, root: Path) -> Mapping[str, Any]:
    from .formal_training import read_formal_training_config

    return read_formal_training_config(path, paper2_root=root)


def _relative_record(path: Path, *, root: Path) -> dict[str, str]:
    guarded = guard_paper2_path(path, paper2_root=root)
    if not guarded.is_file():
        raise FileNotFoundError(guarded)
    return {
        "relative_path": guarded.relative_to(root).as_posix(),
        "sha256": file_sha256(guarded),
    }


def _manifest_set(
    paths: Sequence[Path], *, root: Path
) -> tuple[list[dict[str, str]], str]:
    records = [_relative_record(path, root=root) for path in sorted(paths)]
    if not records:
        raise ValueError("G4 requires the complete calibration manifest set")
    relative_paths = [record["relative_path"] for record in records]
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("G4 calibration manifest paths contain duplicates")
    return records, canonical_sha256(records)


def build_g4_freeze(
    *,
    paper2_root: str | Path,
    evidence_paths: Mapping[str, Path],
    calibration_manifest_paths: Sequence[Path],
    selections: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(paper2_root).resolve()
    if set(evidence_paths) != EVIDENCE_NAMES:
        raise ValueError(
            f"G4 evidence must be exactly {sorted(EVIDENCE_NAMES)}"
        )
    evidence = {
        name: _relative_record(Path(path), root=root)
        for name, path in sorted(evidence_paths.items())
    }
    gate = read_json(root / evidence["training_gate_report"]["relative_path"])
    if (
        gate.get("schema") != "paper2.training-gate-report/v1"
        or gate.get("gate") != "G2/G3"
        or gate.get("status") != "complete"
        or gate.get("formal_gate") is not True
        or gate.get("gates", {}).get("G2", {}).get("status") != "PASS"
        or gate.get("gates", {}).get("G3", {}).get("status") != "PASS"
    ):
        raise ValueError("G4 requires a completed real-data G2/G3 training gate")
    _strict_training_gate(
        root / evidence["training_gate_report"]["relative_path"], root=root
    )
    e7 = read_json(root / evidence["e7_validation_report"]["relative_path"])
    if (
        e7.get("schema") != "paper2.e7-validation-report/v1"
        or e7.get("status") != "complete"
        or e7.get("role") != "bridge_val"
        or e7.get("test_results_inspected") is not False
    ):
        raise ValueError("G4 requires a blind completed E7 validation report")
    curve = read_json(root / evidence["validation_curve_freeze"]["relative_path"])
    if (
        curve.get("schema") != "paper2.validation-curve-freeze/v1"
        or curve.get("status") != "complete"
        or curve.get("role") != "bridge_val"
        or curve.get("test_results_inspected") is not False
    ):
        raise ValueError("G4 requires a blind completed validation-curve freeze")
    dc = read_json(root / evidence["dc_validation_report"]["relative_path"])
    if (
        dc.get("schema") != "paper2.dc-validation-report/v1"
        or dc.get("status") != "complete"
        or dc.get("role") != "bridge_val"
        or dc.get("test_results_inspected") is not False
    ):
        raise ValueError("G4 requires a blind completed DC validation report")
    power = read_json(root / evidence["power_report"]["relative_path"])
    if (
        power.get("schema") != "paper2.power-analysis/v1"
        or power.get("status") != "complete"
        or power.get("test_results_inspected") is not False
    ):
        raise ValueError("G4 requires a blind completed power report")
    calibration_records, calibration_set_sha256 = _manifest_set(
        calibration_manifest_paths, root=root
    )
    required_selections = {
        "g_min",
        "alpha",
        "tau",
        "tau_dc",
        "dc_delta",
        "dc_lambda",
        "dc_noninferiority_margin",
        "path_count",
        "step_count",
        "test_group_count",
        "power",
        "calibration_selection_rule",
    }
    if set(selections) != required_selections:
        raise ValueError(
            "G4 selections differ: "
            f"missing={sorted(required_selections - set(selections))}, "
            f"extra={sorted(set(selections) - required_selections)}"
        )
    payload: dict[str, Any] = {
        "schema": "paper2.g4-freeze/v1",
        "status": "complete",
        "test_results_inspected": False,
        "selection_role": "bridge_val",
        "calibration_role": "uncertainty_calibration",
        "evidence": evidence,
        "asset_lock_sha256": evidence["asset_lock"]["sha256"],
        "role_manifest_sha256": evidence["role_manifest"]["sha256"],
        "training_config_sha256": evidence["formal_training_config"]["sha256"],
        "training_gate_report_sha256": evidence["training_gate_report"]["sha256"],
        "e7_validation_sha256": evidence["e7_validation_report"]["sha256"],
        "validation_curve_freeze_sha256": evidence["validation_curve_freeze"]["sha256"],
        "dc_validation_report_sha256": evidence["dc_validation_report"]["sha256"],
        "power_report_sha256": evidence["power_report"]["sha256"],
        "calibration_manifests": calibration_records,
        "calibration_manifest_set_sha256": calibration_set_sha256,
        **dict(selections),
    }
    if int(payload["test_group_count"]) != int(power.get("test_group_count", -1)):
        raise ValueError("G4 test_group_count differs from the blind power report")
    if float(payload["power"]) != float(power.get("power", float("nan"))):
        raise ValueError("G4 power differs from the blind power report")
    if int(payload["path_count"]) != int(e7.get("selected_path_count", -1)) or int(
        payload["step_count"]
    ) != int(e7.get("selected_step_count", -1)):
        raise ValueError("G4 M/N selections differ from the E7 validation report")
    training = _strict_formal_training_config(
        root / evidence["formal_training_config"]["relative_path"],
        root=root,
    )
    for name in ("g_min", "alpha"):
        if float(payload[name]) != float(training.get(name, float("nan"))):
            raise ValueError(f"G4 {name} differs from the formal training selection")
    if float(payload["tau"]) != float(curve.get("selected_tau", float("nan"))):
        raise ValueError("G4 tau differs from the blind validation-curve freeze")
    for selection_name, evidence_name in (
        ("tau_dc", "selected_tau_dc"),
        ("dc_delta", "selected_delta"),
        ("dc_lambda", "selected_lambda"),
        ("dc_noninferiority_margin", "selected_noninferiority_margin"),
    ):
        if float(payload[selection_name]) != float(dc.get(evidence_name, float("nan"))):
            raise ValueError(f"G4 {selection_name} differs from the DC validation report")
    validate_g4_payload(payload)
    payload["g4_freeze_sha256"] = canonical_sha256(payload)
    return payload


def validate_g4_payload(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("schema") != "paper2.g4-freeze/v1"
        or payload.get("status") != "complete"
        or payload.get("test_results_inspected") is not False
        or payload.get("selection_role") != "bridge_val"
        or payload.get("calibration_role") != "uncertainty_calibration"
    ):
        raise ValueError("G4 freeze identity, state, roles, or blind flag differs")
    numeric = {
        name: float(payload.get(name, float("nan")))
        for name in (
            "g_min",
            "alpha",
            "tau",
            "tau_dc",
            "dc_delta",
            "dc_lambda",
            "dc_noninferiority_margin",
            "power",
        )
    }
    if not all(math.isfinite(value) for value in numeric.values()):
        raise ValueError("G4 numeric selections must be finite")
    if not 0.0 < numeric["g_min"] <= 1.0 or numeric["alpha"] <= 0.0:
        raise ValueError("G4 g_min/alpha selections are invalid")
    if numeric["tau"] not in {0.9, 0.8, 0.7, 0.6}:
        raise ValueError("G4 tau differs from its preregistered grid")
    if numeric["tau_dc"] <= 0.0 or numeric["dc_delta"] <= 0.0 or numeric["dc_lambda"] < 0.0:
        raise ValueError("G4 DC selections are invalid")
    if numeric["dc_noninferiority_margin"] < 0.0 or numeric["power"] < 0.8:
        raise ValueError("G4 noninferiority/power selections are invalid")
    if int(payload.get("path_count", -1)) not in {2, 4, 8, 16, 32} or int(
        payload.get("step_count", -1)
    ) not in {10, 20, 40}:
        raise ValueError("G4 M/N selections differ from the E7 grid")
    if int(payload.get("test_group_count", -1)) not in {24, 30, 36}:
        raise ValueError("G4 test count differs from the reserve expansion rule")
    if payload.get("calibration_selection_rule") != CALIBRATION_SELECTION_RULE:
        raise ValueError("G4 calibration selection rule differs")
    for name in (
        "asset_lock_sha256",
        "role_manifest_sha256",
        "training_config_sha256",
        "training_gate_report_sha256",
        "e7_validation_sha256",
        "validation_curve_freeze_sha256",
        "dc_validation_report_sha256",
        "power_report_sha256",
        "calibration_manifest_set_sha256",
    ):
        if not is_sha256(str(payload.get(name, ""))):
            raise ValueError(f"G4 {name} is invalid")
    evidence = payload.get("evidence")
    if isinstance(evidence, Mapping):
        for field, evidence_name in (
            ("asset_lock_sha256", "asset_lock"),
            ("role_manifest_sha256", "role_manifest"),
            ("training_config_sha256", "formal_training_config"),
            ("training_gate_report_sha256", "training_gate_report"),
            ("e7_validation_sha256", "e7_validation_report"),
            ("validation_curve_freeze_sha256", "validation_curve_freeze"),
            ("dc_validation_report_sha256", "dc_validation_report"),
            ("power_report_sha256", "power_report"),
        ):
            record = evidence.get(evidence_name)
            if not isinstance(record, Mapping) or record.get("sha256") != payload.get(field):
                raise ValueError(f"G4 top-level {field} differs from evidence")


def read_g4_freeze(
    path: str | Path, *, paper2_root: str | Path
) -> Mapping[str, Any]:
    root = Path(paper2_root).resolve()
    freeze_path = guard_paper2_path(path, paper2_root=root)
    payload = read_json(freeze_path)
    if not isinstance(payload, Mapping):
        raise ValueError("G4 freeze must be a JSON object")
    expected = canonical_sha256(
        {key: value for key, value in payload.items() if key != "g4_freeze_sha256"}
    )
    if payload.get("g4_freeze_sha256") != expected:
        raise ValueError("G4 freeze semantic hash mismatch")
    validate_g4_payload(payload)
    evidence = payload.get("evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != EVIDENCE_NAMES:
        raise ValueError("G4 evidence set differs")
    evidence_paths: dict[str, Path] = {}
    for name, record in evidence.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"G4 evidence record is malformed: {name}")
        evidence_path = guard_paper2_path(
            root / str(record.get("relative_path", "")), paper2_root=root
        )
        if not evidence_path.is_file() or file_sha256(evidence_path) != record.get("sha256"):
            raise ValueError(f"G4 evidence bytes differ: {name}")
        evidence_paths[str(name)] = evidence_path
    gate = read_json(evidence_paths["training_gate_report"])
    if (
        gate.get("schema") != "paper2.training-gate-report/v1"
        or gate.get("gate") != "G2/G3"
        or gate.get("status") != "complete"
        or gate.get("formal_gate") is not True
        or gate.get("gates", {}).get("G2", {}).get("status") != "PASS"
        or gate.get("gates", {}).get("G3", {}).get("status") != "PASS"
    ):
        raise ValueError("G4 training-gate evidence is not eligible")
    _strict_training_gate(evidence_paths["training_gate_report"], root=root)
    e7 = read_json(evidence_paths["e7_validation_report"])
    if (
        e7.get("schema") != "paper2.e7-validation-report/v1"
        or e7.get("status") != "complete"
        or e7.get("role") != "bridge_val"
        or e7.get("test_results_inspected") is not False
        or int(e7.get("selected_path_count", -1)) != int(payload["path_count"])
        or int(e7.get("selected_step_count", -1)) != int(payload["step_count"])
    ):
        raise ValueError("G4 E7 evidence is not eligible or differs")
    curve = read_json(evidence_paths["validation_curve_freeze"])
    if (
        curve.get("schema") != "paper2.validation-curve-freeze/v1"
        or curve.get("status") != "complete"
        or curve.get("role") != "bridge_val"
        or curve.get("test_results_inspected") is not False
        or float(curve.get("selected_tau", float("nan"))) != float(payload["tau"])
    ):
        raise ValueError("G4 validation-curve evidence is not eligible or differs")
    dc = read_json(evidence_paths["dc_validation_report"])
    if (
        dc.get("schema") != "paper2.dc-validation-report/v1"
        or dc.get("status") != "complete"
        or dc.get("role") != "bridge_val"
        or dc.get("test_results_inspected") is not False
    ):
        raise ValueError("G4 DC validation evidence is not eligible")
    for selection_name, evidence_name in (
        ("tau_dc", "selected_tau_dc"),
        ("dc_delta", "selected_delta"),
        ("dc_lambda", "selected_lambda"),
        ("dc_noninferiority_margin", "selected_noninferiority_margin"),
    ):
        if float(dc.get(evidence_name, float("nan"))) != float(payload[selection_name]):
            raise ValueError(f"G4 DC evidence differs: {selection_name}")
    training = _strict_formal_training_config(
        evidence_paths["formal_training_config"], root=root
    )
    for name in ("g_min", "alpha"):
        if float(training.get(name, float("nan"))) != float(payload[name]):
            raise ValueError(f"G4 formal training evidence differs: {name}")
    power = read_json(evidence_paths["power_report"])
    if (
        power.get("schema") != "paper2.power-analysis/v1"
        or power.get("status") != "complete"
        or power.get("test_results_inspected") is not False
        or int(power.get("test_group_count", -1)) != int(payload["test_group_count"])
        or float(power.get("power", float("nan"))) != float(payload["power"])
    ):
        raise ValueError("G4 power evidence is not eligible or differs")
    records = payload.get("calibration_manifests")
    if not isinstance(records, list) or not records:
        raise ValueError("G4 calibration manifest records are missing")
    observed_records = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("G4 calibration record is malformed")
        calibration_path = guard_paper2_path(
            root / str(record.get("relative_path", "")), paper2_root=root
        )
        if not calibration_path.is_file() or file_sha256(calibration_path) != record.get(
            "sha256"
        ):
            raise ValueError("G4 calibration manifest bytes differ")
        observed_records.append(dict(record))
    if canonical_sha256(observed_records) != payload.get("calibration_manifest_set_sha256"):
        raise ValueError("G4 calibration manifest set hash differs")
    return payload


def write_g4_freeze(
    path: str | Path,
    *,
    paper2_root: str | Path,
    evidence_paths: Mapping[str, Path],
    calibration_manifest_paths: Sequence[Path],
    selections: Mapping[str, Any],
) -> Path:
    payload = build_g4_freeze(
        paper2_root=paper2_root,
        evidence_paths=evidence_paths,
        calibration_manifest_paths=calibration_manifest_paths,
        selections=selections,
    )
    return write_canonical_json(path, payload, paper2_root=paper2_root)

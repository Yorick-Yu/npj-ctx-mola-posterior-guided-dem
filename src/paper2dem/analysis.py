from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
import json

from .hashing import canonical_sha256, write_canonical_json
from .incremental_anchor import ANCHORED_METHOD, ISO_METHOD, IncrementalAnchorPolicy


@dataclass(frozen=True)
class AnalysisLockConfig:
    expected_matrix_spec_sha256: str
    asset_lock_sha256: str
    checkpoint_manifest_set_sha256: str
    calibration_manifest_set_sha256: str
    statistical_code_sha256: str
    power_report_sha256: str
    expected_trace_locks_sha256: str
    sealed_reference_lock_sha256: str
    g4_freeze_sha256: str
    training_config_sha256: str
    training_gate_report_sha256: str
    e7_validation_sha256: str
    tau: float
    g_min: float
    alpha: float
    tau_dc: float
    dc_delta: float
    dc_lambda: float
    dc_noninferiority_margin: float
    path_count: int
    step_count: int
    test_group_count: int
    power: float
    calibration_selection_rule: str
    bootstrap_replicates: int = 10_000

    def validate(self) -> None:
        for name in (
            "expected_matrix_spec_sha256",
            "asset_lock_sha256",
            "checkpoint_manifest_set_sha256",
            "calibration_manifest_set_sha256",
            "statistical_code_sha256",
            "power_report_sha256",
            "expected_trace_locks_sha256",
            "sealed_reference_lock_sha256",
            "g4_freeze_sha256",
            "training_config_sha256",
            "training_gate_report_sha256",
            "e7_validation_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if self.tau not in {0.9, 0.8, 0.7, 0.6}:
            raise ValueError("tau must be selected from the preregistered grid")
        if not 0.0 < self.g_min <= 1.0:
            raise ValueError("g_min must lie in (0,1]")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if self.tau_dc <= 0.0 or self.dc_delta <= 0.0 or self.dc_lambda < 0.0:
            raise ValueError("DC tau/delta must be positive and lambda must be nonnegative")
        if self.dc_noninferiority_margin < 0:
            raise ValueError("DC noninferiority margin must be non-negative")
        if self.path_count not in {2, 4, 8, 16, 32}:
            raise ValueError("path_count must come from the E7 prefix grid")
        if self.step_count not in {10, 20, 40}:
            raise ValueError("step_count must come from the E7 prefix grid")
        if self.test_group_count not in {24, 30, 36}:
            raise ValueError("test group count must follow the 6-group reserve rule")
        if self.power < 0.80:
            raise ValueError("confirmatory analysis requires at least 80% locked power")
        if self.bootstrap_replicates != 10_000:
            raise ValueError("formal tile bootstrap requires exactly 10000 replicates")
        if self.calibration_selection_rule != "minimum_nll_scale_floor_on_calibration_role":
            raise ValueError("calibration selection rule differs from the frozen protocol")


def build_analysis_lock(config: AnalysisLockConfig) -> dict[str, Any]:
    config.validate()
    anchor_policy = IncrementalAnchorPolicy()
    bootstrap_seed = int(
        canonical_sha256(
            {
                "protocol": "paper2-tile-cluster-bootstrap-v1",
                "expected_matrix_spec_sha256": config.expected_matrix_spec_sha256,
                "statistical_code_sha256": config.statistical_code_sha256,
            }
        )[:16],
        16,
    )
    payload: dict[str, Any] = {
        "schema": "paper2.analysis-lock/v1",
        **asdict(config),
        "primary_comparison": [ANCHORED_METHOD, ISO_METHOD],
        "incremental_anchor_policy": anchor_policy.to_dict(),
        "incremental_anchor_policy_sha256": anchor_policy.sha256,
        "primary_endpoints": {
            "A": {"direction": "lower", "one_sided_alpha": 0.05},
            "V": {"direction": "higher", "one_sided_alpha": 0.05},
            "joint_rule": "intersection-union",
        },
        "laser_saving": {
            "one_sided_alpha": 0.05,
            "practical_fraction": 0.10,
            "practical_points": 5,
            "extrapolation": False,
        },
        "outer_statistical_unit": "geographic_tile_group",
        "bootstrap_seed": bootstrap_seed,
        "holm_families": {
            "ot_attribution": [
                "Delta_M",
                "Delta_G",
                "Delta_MG",
                "D_M_given_G1",
                "D_placebo",
            ],
            "k_specific_error": [0, 5, 10, 20, 50, 100],
            "uncertainty": ["coverage", "crps", "nll", "ause"],
        },
        "failure_policy": {
            "no_favorable_seed_reruns": True,
            "no_test_tile_replacement": True,
            "retain_negative_tiles": True,
            "main_claim_requires_A_and_V": True,
        },
    }
    payload["analysis_lock_sha256"] = canonical_sha256(payload)
    return payload


def write_analysis_lock(
    path: str | Path, config: AnalysisLockConfig, *, paper2_root: str | Path
) -> Path:
    return write_canonical_json(
        path, build_analysis_lock(config), paper2_root=paper2_root
    )


def read_analysis_lock(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = canonical_sha256(
        {key: value for key, value in payload.items() if key != "analysis_lock_sha256"}
    )
    if payload.get("analysis_lock_sha256") != expected:
        raise ValueError("analysis-lock hash mismatch")
    return payload

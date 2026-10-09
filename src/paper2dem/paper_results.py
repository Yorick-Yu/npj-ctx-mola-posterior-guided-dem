"""Locked scientific summaries for E3--E10 after exact matrix closure."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import math

import numpy as np

from .confirmatory import e4_mechanism_analysis, primary_e3_analysis
from .evaluation import curve_areas, laser_saving, paired_tile_bootstrap
from .hashing import canonical_sha256, file_sha256, guard_paper2_path, read_json, write_canonical_json
from .incremental_anchor import (
    ANCHORED_METHOD,
    ISO_METHOD,
    OT_METHOD,
    IncrementalAnchorPolicy,
    anchored_curve,
)
from .matrix import EvaluationManifest, ExpectedMatrix, InferenceManifest, audit_expected_matrix


PAPER_RESULTS_SCHEMA = "paper2.paper-results/v1"


def _key(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("key")
    if not isinstance(value, Mapping):
        raise ValueError("paper result record lacks a matrix key")
    return value


def _metric(record: Mapping[str, Any], name: str) -> float:
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("paper result record lacks metrics")
    result = float(metrics.get(name, float("nan")))
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"paper result metric {name} must be finite and nonnegative")
    return result


def _bootstrap(values: Sequence[float], *, replicates: int, seed: int) -> dict[str, float]:
    return dict(paired_tile_bootstrap(values, replicates=replicates, seed=seed))


def bridge_curve_analysis(
    records: Sequence[Mapping[str, Any]],
    *,
    tau: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    budgets = (0, 5, 10, 20, 50, 100)
    policy = IncrementalAnchorPolicy()
    values: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for record in records:
        key = _key(record)
        if key.get("experiment") != "E3b" or key.get("method") not in {
            "coarse",
            ISO_METHOD,
            OT_METHOD,
        }:
            continue
        values[(str(key["tile_id"]), str(key["method"]), int(key["budget"]))].append(
            _metric(record, "mse")
        )
    tiles = sorted({key[0] for key in values})
    if len(tiles) != 12:
        raise ValueError("E3b bridge analysis requires exactly 12 Paper1 test tiles")
    rows = []
    for tile in tiles:
        coarse = values.get((tile, "coarse", 0), [])
        if len(coarse) != 1:
            raise ValueError(f"E3b tile {tile} lacks one coarse baseline")
        baseline = coarse[0]
        curves = {}
        areas = {}
        for method in (ISO_METHOD, OT_METHOD):
            curve = []
            for budget in budgets:
                samples = values.get((tile, method, budget), [])
                if not samples:
                    raise ValueError(f"E3b tile {tile}/{method}/K={budget} is missing")
                curve.append(float(np.mean(samples)))
            curves[method] = curve
            areas[method] = curve_areas(budgets, curve, coarse_mse=baseline)
        curves[ANCHORED_METHOD] = anchored_curve(
            budgets=budgets,
            iso_curve=curves[ISO_METHOD],
            ot_curve=curves[OT_METHOD],
            policy=policy,
        ).tolist()
        areas[ANCHORED_METHOD] = curve_areas(
            budgets, curves[ANCHORED_METHOD], coarse_mse=baseline
        )
        saving = laser_saving(
            budgets,
            np.asarray(curves[ANCHORED_METHOD]) / baseline,
            np.asarray(curves[ISO_METHOD]) / baseline,
            tau,
        )
        rows.append(
            {
                "tile_id": tile,
                "A_improvement": areas[ISO_METHOD]["A"]
                - areas[ANCHORED_METHOD]["A"],
                "V_improvement": areas[ANCHORED_METHOD]["V"]
                - areas[ISO_METHOD]["V"],
                "curves": curves,
                "areas": areas,
                "saving": saving,
            }
        )
    reached = [row for row in rows if row["saving"]["saving_estimable"]]
    return {
        "schema": "paper2.e3b-bridge-analysis/v1",
        "role": "Paper1 bridge; supportive, not the Paper2 confirmatory test",
        "primary_comparison": [ANCHORED_METHOD, ISO_METHOD],
        "incremental_anchor_policy": policy.to_dict(),
        "incremental_anchor_policy_sha256": policy.sha256,
        "tile_count": len(rows),
        "A_improvement": _bootstrap(
            [row["A_improvement"] for row in rows],
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "V_improvement": _bootstrap(
            [row["V_improvement"] for row in rows],
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 1,
        ),
        "saving_estimable_tiles": len(reached),
        "saving_fraction": (
            None
            if len(reached) < 2
            else _bootstrap(
                [float(row["saving"]["saving_fraction"]) for row in reached],
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + 2,
            )
        ),
        "tiles": rows,
    }


def e5_selector_analysis(
    records: Sequence[Mapping[str, Any]], *, bootstrap_replicates: int, bootstrap_seed: int
) -> dict[str, Any]:
    random_names = tuple(f"random-{index:02d}" for index in range(10))
    named = ("uniform", "max-variance", "FAOPT")
    values: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    for record in records:
        key = _key(record)
        if key.get("experiment") != "E5":
            continue
        condition = str(key["condition"])
        if not condition.startswith("selector="):
            raise ValueError("E5 result lacks its selector condition")
        selector = condition.split("=", 1)[1]
        values[
            (str(key["tile_id"]), str(key["method"]), int(key["budget"]), selector)
        ].append(_metric(record, "mse"))
    tiles = sorted({key[0] for key in values})
    if len(tiles) != 12:
        raise ValueError("E5 selector analysis requires exactly 12 bridge tiles")
    rows = []
    summaries = []
    methods = ("gp-footprint", "Iso-SB-cond", "OT-SB-full")
    for method in methods:
        for budget in (5, 20, 100):
            contrasts: dict[str, list[float]] = {name: [] for name in named}
            for tile in tiles:
                random = []
                for selector in random_names:
                    samples = values.get((tile, method, budget, selector), [])
                    if not samples:
                        raise ValueError(f"E5 random cell is missing: {tile}/{method}/{budget}")
                    random.append(float(np.mean(samples)))
                random_mean = float(np.mean(random))
                for selector in named:
                    samples = values.get((tile, method, budget, selector), [])
                    if not samples:
                        raise ValueError(f"E5 named selector is missing: {selector}")
                    selector_mean = float(np.mean(samples))
                    contrast = random_mean - selector_mean
                    contrasts[selector].append(contrast)
                    rows.append(
                        {
                            "tile_id": tile,
                            "method": method,
                            "budget": budget,
                            "selector": selector,
                            "random_ensemble_mse": random_mean,
                            "selector_mse": selector_mean,
                            "mse_improvement_over_random": contrast,
                        }
                    )
            for offset, selector in enumerate(named):
                summaries.append(
                    {
                        "method": method,
                        "budget": budget,
                        "selector": selector,
                        "mse_improvement_over_random": _bootstrap(
                            contrasts[selector],
                            replicates=bootstrap_replicates,
                            seed=bootstrap_seed
                            + 100 * methods.index(method)
                            + 10 * (5, 20, 100).index(budget)
                            + offset,
                        ),
                    }
                )
    return {
        "schema": "paper2.e5-selector-analysis/v1",
        "role": "supportive Paper1 bridge experiment",
        "tile_weighting": "equal after averaging seeds and noise repeats",
        "random_reference": "mean of ten independently locked random selectors",
        "summaries": summaries,
        "tiles": rows,
    }


def e6_dc_analysis(
    records: Sequence[Mapping[str, Any]], *, bootstrap_replicates: int, bootstrap_seed: int
) -> dict[str, Any]:
    values: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for record in records:
        key = _key(record)
        if key.get("experiment") == "E6":
            values[(str(key["tile_id"]), int(key["budget"]), str(key["dc_mode"]))].append(
                _metric(record, "mse")
            )
    tiles = sorted({key[0] for key in values})
    if len(tiles) not in {24, 30, 36}:
        raise ValueError("E6 DC analysis has the wrong locked-test tile count")
    summaries = []
    rows = []
    for budget_index, budget in enumerate((20, 100)):
        correct_over_none = []
        correct_over_point = []
        for tile in tiles:
            means = {}
            for mode in ("D0", "D1", "D3"):
                samples = values.get((tile, budget, mode), [])
                if not samples:
                    raise ValueError(f"E6 cell is missing: {tile}/{budget}/{mode}")
                means[mode] = float(np.mean(samples))
            d0_d1 = means["D0"] - means["D1"]
            d3_d1 = means["D3"] - means["D1"]
            correct_over_none.append(d0_d1)
            correct_over_point.append(d3_d1)
            rows.append(
                {
                    "tile_id": tile,
                    "budget": budget,
                    "mean_mse": means,
                    "correct_footprint_dc_improvement_over_no_dc": d0_d1,
                    "correct_footprint_dc_improvement_over_wrong_point_dc": d3_d1,
                }
            )
        summaries.append(
            {
                "budget": budget,
                "correct_over_no_dc": _bootstrap(
                    correct_over_none,
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed + 2 * budget_index,
                ),
                "correct_over_wrong_point_dc": _bootstrap(
                    correct_over_point,
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed + 2 * budget_index + 1,
                ),
            }
        )
    return {
        "schema": "paper2.e6-dc-analysis/v1",
        "shared_uncorrected_paths_required": True,
        "positive_contrast_means_correct_footprint_DC_has_lower_MSE": True,
        "summaries": summaries,
        "tiles": rows,
    }


def e7_convergence_analysis(
    records: Sequence[Mapping[str, Any]], *, frozen_path_count: int, frozen_step_count: int
) -> dict[str, Any]:
    values: dict[tuple[str, int, int, int], list[tuple[float, float, float]]] = defaultdict(list)
    for record in records:
        key = _key(record)
        if key.get("experiment") != "E7":
            continue
        metrics = record["metrics"]
        probabilistic = metrics.get("probabilistic")
        if not isinstance(probabilistic, Mapping):
            raise ValueError("E7 generative result lacks calibrated probabilistic metrics")
        values[
            (
                str(key["method"]),
                int(key["budget"]),
                int(key["path_count"]),
                int(key["step_count"]),
            )
        ].append(
            (
                _metric(record, "rmse"),
                float(probabilistic["coverage_90"]),
                float(probabilistic["coverage_95"]),
            )
        )
    rows = []
    for key, samples in sorted(values.items()):
        array = np.asarray(samples, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise ValueError("E7 convergence values are non-finite")
        method, budget, paths, steps = key
        rows.append(
            {
                "method": method,
                "budget": budget,
                "path_count": paths,
                "step_count": steps,
                "cell_count": len(samples),
                "mean_rmse": float(np.mean(array[:, 0])),
                "mean_coverage_90": float(np.mean(array[:, 1])),
                "mean_coverage_95": float(np.mean(array[:, 2])),
                "is_G4_selected_setting": paths == frozen_path_count
                and steps == frozen_step_count,
            }
        )
    if len(rows) != 2 * 2 * 5 * 3:
        raise ValueError("E7 convergence summary does not close the 2x2x5x3 grid")
    return {
        "schema": "paper2.e7-convergence-analysis/v1",
        "role": "validation-only numerical convergence evidence",
        "frozen_path_count": frozen_path_count,
        "frozen_step_count": frozen_step_count,
        "rows": rows,
    }


def e8_stress_analysis(
    records: Sequence[Mapping[str, Any]], *, bootstrap_replicates: int, bootstrap_seed: int
) -> dict[str, Any]:
    values: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    for record in records:
        key = _key(record)
        if key.get("experiment") == "E8":
            values[
                (
                    str(key["tile_id"]),
                    str(key["condition"]),
                    int(key["budget"]),
                    str(key["method"]),
                )
            ].append(_metric(record, "mse"))
    tiles = sorted({key[0] for key in values})
    conditions = sorted({key[1] for key in values})
    if len(tiles) != 12 or len(conditions) != 12:
        raise ValueError("E8 stress analysis requires 12 validation tiles and conditions")
    rows = []
    for condition_index, condition in enumerate(conditions):
        for budget_index, budget in enumerate((20, 100)):
            contrasts = []
            method_means = defaultdict(list)
            for tile in tiles:
                tile_means = {}
                for method in ("gp-footprint", "Iso-SB-cond", "OT-SB-full"):
                    samples = values.get((tile, condition, budget, method), [])
                    if not samples:
                        raise ValueError(f"E8 stress cell is missing: {condition}/{method}")
                    tile_means[method] = float(np.mean(samples))
                    method_means[method].append(tile_means[method])
                contrasts.append(tile_means["Iso-SB-cond"] - tile_means["OT-SB-full"])
            rows.append(
                {
                    "condition": condition,
                    "budget": budget,
                    "tile_equal_mean_mse": {
                        method: float(np.mean(method_means[method]))
                        for method in ("gp-footprint", "Iso-SB-cond", "OT-SB-full")
                    },
                    "OT_full_improvement_over_Iso": _bootstrap(
                        contrasts,
                        replicates=bootstrap_replicates,
                        seed=bootstrap_seed + 2 * condition_index + budget_index,
                    ),
                }
            )
    return {
        "schema": "paper2.e8-stress-analysis/v1",
        "role": "validation-only exploratory stress test",
        "confirmatory_claim_allowed": False,
        "rows": rows,
    }


def e9_external_analysis(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for record in records:
        key = _key(record)
        if key.get("experiment") == "E9":
            values[(str(key["tile_id"]), str(key["method"]), int(key["budget"]))].append(
                _metric(record, "rmse")
            )
    tiles = sorted({key[0] for key in values})
    if len(tiles) != 6:
        raise ValueError("E9 external analysis requires exactly six HiRISE tiles")
    rows = []
    for tile in tiles:
        for method in ("coarse", "gp-footprint", "det-fusion", "Iso-SB-cond", "OT-SB-full"):
            budgets = (0,) if method == "coarse" else (0, 5, 10, 20)
            for budget in budgets:
                samples = values.get((tile, method, budget), [])
                if not samples:
                    raise ValueError(f"E9 external cell is missing: {tile}/{method}/{budget}")
                rows.append(
                    {
                        "tile_id": tile,
                        "method": method,
                        "budget": budget,
                        "mean_rmse": float(np.mean(samples)),
                        "cell_count": len(samples),
                    }
                )
    return {
        "schema": "paper2.e9-external-analysis/v1",
        "role": "zero-shot external validation; tile-level descriptive reporting",
        "fine_tuning": False,
        "tile_count": len(tiles),
        "rows": rows,
    }


def e10_mola_analysis(
    records: Sequence[Mapping[str, Any]], *, bootstrap_replicates: int, bootstrap_seed: int
) -> dict[str, Any]:
    values: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for record in records:
        key = _key(record)
        if key.get("experiment") in {"E3", "E10"} and int(key["budget"]) in {0, 20, 100}:
            method = str(key["method"])
            if method in {"coarse", "gp-footprint", "det-fusion", "Iso-SB-cond", "OT-SB-full"}:
                values[
                    (str(key["experiment"]), str(key["tile_id"]), method, int(key["budget"]))
                ].append(_metric(record, "mse"))
    tiles = sorted({key[1] for key in values if key[0] == "E10"})
    if len(tiles) not in {24, 30, 36}:
        raise ValueError("E10 MOLA analysis has the wrong locked-test tile count")
    summaries = []
    rows = []
    methods = ("coarse", "gp-footprint", "det-fusion", "Iso-SB-cond", "OT-SB-full")
    for method_index, method in enumerate(methods):
        budgets = (0,) if method == "coarse" else (0, 20, 100)
        for budget_index, budget in enumerate(budgets):
            penalties = []
            for tile in tiles:
                e3 = values.get(("E3", tile, method, budget), [])
                e10 = values.get(("E10", tile, method, budget), [])
                if not e3 or not e10:
                    raise ValueError(f"E10/E3 paired cell is missing: {tile}/{method}/{budget}")
                e3_mean = float(np.mean(e3))
                e10_mean = float(np.mean(e10))
                penalty = e10_mean - e3_mean
                penalties.append(penalty)
                rows.append(
                    {
                        "tile_id": tile,
                        "method": method,
                        "budget": budget,
                        "controlled_Z0_mse": e3_mean,
                        "independent_MOLA_Z0_mse": e10_mean,
                        "MOLA_mse_penalty": penalty,
                    }
                )
            summaries.append(
                {
                    "method": method,
                    "budget": budget,
                    "MOLA_mse_penalty": _bootstrap(
                        penalties,
                        replicates=bootstrap_replicates,
                        seed=bootstrap_seed + 10 * method_index + budget_index,
                    ),
                }
            )
    return {
        "schema": "paper2.e10-mola-analysis/v1",
        "role": "independent coarse-DEM robustness",
        "positive_penalty_means_MOLA_is_harder": True,
        "summaries": summaries,
        "tiles": rows,
    }


def _evaluation_record(path: Path, manifest: EvaluationManifest, *, root: Path) -> dict[str, Any]:
    metrics = [artifact for artifact in manifest.artifacts if artifact.name == "metrics"]
    if len(metrics) != 1:
        raise ValueError("paper results require one metrics artifact per evaluation")
    metrics_path = guard_paper2_path(root / metrics[0].relative_path, paper2_root=root)
    if not metrics_path.is_file() or file_sha256(metrics_path) != metrics[0].sha256:
        raise ValueError("paper result metrics artifact bytes differ")
    payload = read_json(metrics_path)
    if (
        payload.get("schema") != "paper2.evaluation/v1"
        or payload.get("key") != manifest.key.to_dict()
        or payload.get("inference_manifest_sha256") != manifest.inference_manifest_sha256
    ):
        raise ValueError(f"paper result evaluation payload differs: {path}")
    return dict(payload)


def build_paper_results(
    *,
    expected_matrix_path: str | Path,
    analysis_lock_path: str | Path,
    inference_manifest_paths: Sequence[str | Path],
    evaluation_manifest_paths: Sequence[str | Path],
    paper2_root: str | Path,
) -> dict[str, Any]:
    root = Path(paper2_root).resolve()
    expected_path = guard_paper2_path(expected_matrix_path, paper2_root=root)
    analysis_path = guard_paper2_path(analysis_lock_path, paper2_root=root)
    expected = ExpectedMatrix.read(expected_path)
    if expected.analysis_lock_sha256 != file_sha256(analysis_path):
        raise ValueError("paper results expected matrix differs from the analysis lock")
    inference_paths = [guard_paper2_path(path, paper2_root=root) for path in inference_manifest_paths]
    evaluation_paths = [
        guard_paper2_path(path, paper2_root=root) for path in evaluation_manifest_paths
    ]
    inferences = [InferenceManifest.read(path) for path in inference_paths]
    evaluations = [EvaluationManifest.read(path) for path in evaluation_paths]
    audit = audit_expected_matrix(expected, inferences, evaluations, paper2_root=root)
    if not audit.ok:
        raise ValueError("paper results require the exactly closed physical matrix")
    records = [
        _evaluation_record(path, manifest, root=root)
        for path, manifest in zip(evaluation_paths, evaluations, strict=True)
    ]
    analysis = read_json(analysis_path)
    replicates = int(analysis["bootstrap_replicates"])
    seed = int(analysis["bootstrap_seed"])
    primary = primary_e3_analysis(
        records,
        tau=float(analysis["tau"]),
        bootstrap_replicates=replicates,
        bootstrap_seed=seed,
    )
    mechanism = e4_mechanism_analysis(
        records,
        bootstrap_replicates=replicates,
        bootstrap_seed=seed + 100,
    )
    if primary["claim_pass"] and mechanism["ot_geometry_attribution_pass"]:
        claim_policy = "primary efficiency claim and OT-geometry attribution both supported"
    elif primary["claim_pass"]:
        claim_policy = "primary efficiency claim supported; OT attribution downgraded"
    else:
        claim_policy = "retain all results and downgrade the primary claim to exploratory"
    payload: dict[str, Any] = {
        "schema": PAPER_RESULTS_SCHEMA,
        "status": "complete",
        "expected_matrix": {
            "relative_path": expected_path.relative_to(root).as_posix(),
            "sha256": file_sha256(expected_path),
        },
        "analysis_lock": {
            "relative_path": analysis_path.relative_to(root).as_posix(),
            "sha256": file_sha256(analysis_path),
        },
        "matrix_audit": audit.to_dict(),
        "matrix_audit_sha256": canonical_sha256(audit.to_dict()),
        "inference_manifest_set_sha256": canonical_sha256(
            sorted(file_sha256(path) for path in inference_paths)
        ),
        "evaluation_manifest_set_sha256": canonical_sha256(
            sorted(file_sha256(path) for path in evaluation_paths)
        ),
        "claim_policy": claim_policy,
        "E3_primary": primary,
        "E4_OT_mechanism": mechanism,
        "E3b_Paper1_bridge": bridge_curve_analysis(
            records,
            tau=float(analysis["tau"]),
            bootstrap_replicates=replicates,
            bootstrap_seed=seed + 200,
        ),
        "E5_selector_efficiency": e5_selector_analysis(
            records, bootstrap_replicates=replicates, bootstrap_seed=seed + 300
        ),
        "E6_data_consistency": e6_dc_analysis(
            records, bootstrap_replicates=replicates, bootstrap_seed=seed + 400
        ),
        "E7_numerical_convergence": e7_convergence_analysis(
            records,
            frozen_path_count=int(analysis["path_count"]),
            frozen_step_count=int(analysis["step_count"]),
        ),
        "E8_stress": e8_stress_analysis(
            records, bootstrap_replicates=replicates, bootstrap_seed=seed + 500
        ),
        "E9_HiRISE_external": e9_external_analysis(records),
        "E10_independent_MOLA": e10_mola_analysis(
            records, bootstrap_replicates=replicates, bootstrap_seed=seed + 600
        ),
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def write_paper_results(path: str | Path, **kwargs: Any) -> Path:
    return write_canonical_json(
        path,
        build_paper_results(**kwargs),
        paper2_root=kwargs["paper2_root"],
    )


def read_paper_results(
    path: str | Path, *, paper2_root: str | Path, verify_sources: bool = True
) -> dict[str, Any]:
    root = Path(paper2_root).resolve()
    result_path = guard_paper2_path(path, paper2_root=root)
    payload = read_json(result_path)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != PAPER_RESULTS_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("content_sha256")
        != canonical_sha256(
            {name: value for name, value in payload.items() if name != "content_sha256"}
        )
    ):
        raise ValueError("paper results identity/status/content hash differs")
    audit = payload.get("matrix_audit")
    if (
        not isinstance(audit, Mapping)
        or audit.get("ok") is not True
        or payload.get("matrix_audit_sha256") != canonical_sha256(audit)
    ):
        raise ValueError("paper results do not bind a closed matrix audit")
    for name in ("expected_matrix", "analysis_lock"):
        record = payload.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"paper results lack {name}")
        source = guard_paper2_path(root / str(record.get("relative_path", "")), paper2_root=root)
        if not source.is_file() or file_sha256(source) != record.get("sha256"):
            raise ValueError(f"paper results {name} bytes differ")
    if verify_sources:
        inference_paths = sorted(root.glob("runs/formal/**/inference-manifest.json"))
        evaluation_paths = sorted(root.glob("runs/formal/**/evaluation-manifest.json"))
        if payload.get("inference_manifest_set_sha256") != canonical_sha256(
            sorted(file_sha256(source) for source in inference_paths)
        ):
            raise ValueError("paper results inference-manifest set differs")
        if payload.get("evaluation_manifest_set_sha256") != canonical_sha256(
            sorted(file_sha256(source) for source in evaluation_paths)
        ):
            raise ValueError("paper results evaluation-manifest set differs")
        expected = ExpectedMatrix.read(
            root / str(payload["expected_matrix"]["relative_path"])
        )
        if (
            int(audit.get("expected_cells", -1)) != len(expected.cells)
            or int(audit.get("observed_manifests", -1)) != len(expected.cells)
            or int(audit.get("observed_evaluations", -1)) != len(expected.cells)
        ):
            raise ValueError("paper results matrix-audit counts differ")
    return dict(payload)

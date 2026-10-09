from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any


THIRD_ROUND_BUDGETS = (5, 10, 20, 40, 80)
THIRD_ROUND_MODEL_SEEDS = (20260808, 20260809, 20260810)
THIRD_ROUND_SELECTORS = (
    "random-00",
    "random-01",
    "random-02",
    "uniform-farthest",
    "max-prior-variance",
    "greedy-mi-point",
    "terrain-faopt",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def stable_number(*parts: object) -> int:
    return int.from_bytes(
        hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big"
    )


def _orbit_counts(record: Mapping[str, Any]) -> dict[int, int]:
    return {int(item["orbit"]): int(item["point_count"]) for item in record["orbits"]}


def partition_fresh_validation_orbits(
    record: Mapping[str, Any], *, seed: str
) -> dict[str, Any] | None:
    counts = _orbit_counts(record)
    if len(counts) < 4:
        return None
    for salt in range(10_000):
        evaluation = sorted(
            orbit
            for orbit in counts
            if stable_number(seed, record["tile_id"], salt, orbit) % 10 < 4
        )
        assimilation = sorted(orbit for orbit in counts if orbit not in evaluation)
        assimilation_count = sum(counts[orbit] for orbit in assimilation)
        evaluation_count = sum(counts[orbit] for orbit in evaluation)
        if (
            len(assimilation) >= 2
            and len(evaluation) >= 2
            and assimilation_count >= 20
            and evaluation_count >= 20
        ):
            return {
                "assimilation_orbits": assimilation,
                "evaluation_orbits": evaluation,
                "assimilation_point_count": assimilation_count,
                "evaluation_point_count": evaluation_count,
                "partition_salt": salt,
            }
    return None


def build_third_round_protocol(
    parent_protocol: Mapping[str, Any],
    fresh_qa: Mapping[str, Any],
    *,
    parent_protocol_file_sha256: str,
    fresh_qa_file_sha256: str,
    seed: str = "mars-sparse-detail-v30-20260810",
    validation_target: int = 4,
) -> dict[str, Any]:
    """Reclassify all opened evidence as development and reserve unseen validation."""

    parent_records = [deepcopy(record) for record in parent_protocol["tiles"]]
    used_ids = {str(record["tile_id"]) for record in parent_records}
    external_orbits = {
        int(orbit)
        for record in parent_records
        if record["role"] == "external"
        for key in ("assimilation_orbits", "evaluation_orbits")
        for orbit in record.get(key, [])
    }
    candidates = []
    rejected = []
    for record in sorted(fresh_qa["tiles"], key=lambda value: str(value["tile_id"])):
        tile_id = str(record["tile_id"])
        if tile_id in used_ids or not bool(record.get("qa_v2_pass")):
            continue
        counts = _orbit_counts(record)
        overlaps = sorted(set(counts) & external_orbits)
        if overlaps:
            rejected.append(
                {
                    "tile_id": tile_id,
                    "reason": "orbit_overlap_with_sealed_external",
                    "orbit_count": len(overlaps),
                }
            )
            continue
        partition = partition_fresh_validation_orbits(record, seed=seed)
        if partition is None:
            rejected.append(
                {"tile_id": tile_id, "reason": "insufficient_partitionable_orbits_or_points"}
            )
            continue
        candidates.append((record, partition))

    # Prefer regional and terrain-complexity diversity before deterministic rank.
    region_frequency = Counter(str(record["region"]) for record, _ in candidates)
    candidates.sort(
        key=lambda item: (
            region_frequency[str(item[0]["region"])],
            stable_number(seed, "fresh-validation", item[0]["tile_id"]),
        )
    )
    selected: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    selected_orbits: set[int] = set()
    for record, partition in candidates:
        orbits = set(_orbit_counts(record))
        if orbits & selected_orbits:
            rejected.append(
                {"tile_id": record["tile_id"], "reason": "orbit_overlap_with_selected_validation"}
            )
            continue
        selected.append((record, partition))
        selected_orbits.update(orbits)
        if len(selected) == validation_target:
            break

    validation_orbits = {
        orbit for record, _ in selected for orbit in _orbit_counts(record)
    }
    output_records = []
    reclassified = Counter()
    excluded_development = []
    for record in parent_records:
        original_role = str(record["role"])
        if original_role in {"train", "validation", "locked_test"}:
            record_orbits = {
                int(orbit)
                for key in ("assimilation_orbits", "evaluation_orbits")
                for orbit in record.get(key, [])
            }
            overlap = sorted(record_orbits & validation_orbits)
            if overlap:
                record["role"] = "excluded_development_orbit_overlap"
                record["third_round_provenance"] = {
                    "parent_role": original_role,
                    "evidence_status": (
                        "excluded from third-round training to preserve fresh validation "
                        "orbit independence"
                    ),
                    "overlapping_validation_orbits": overlap,
                }
                excluded_development.append(
                    {
                        "tile_id": record["tile_id"],
                        "parent_role": original_role,
                        "overlapping_validation_orbits": overlap,
                    }
                )
            else:
                record["role"] = "development_train"
                record["third_round_provenance"] = {
                    "parent_role": original_role,
                    "evidence_status": (
                        "historically opened; development use only; not confirmatory evidence"
                    ),
                }
                reclassified[original_role] += 1
        elif original_role == "external":
            record["third_round_provenance"] = {
                "parent_role": "external",
                "evidence_status": "sealed and forbidden before final candidate freeze",
            }
        else:
            raise ValueError(f"unsupported parent role: {original_role}")
        output_records.append(record)

    for record, partition in selected:
        output_records.append(
            {
                "tile_id": record["tile_id"],
                "region": record["region"],
                "role": "validation",
                "qa": record["qa"],
                "complexity": record["complexity"],
                "source_relative_path": record["source_relative_path"],
                "source_sha256": record["source_sha256"],
                "mola_points_relative_path": record["mola_points_relative_path"],
                "mola_points_sha256": record["mola_points_sha256"],
                **partition,
                "third_round_provenance": {
                    "parent_role": "prospective_unallocated",
                    "evidence_status": "fresh validation allocated before third-round training",
                },
            }
        )

    counts = Counter(record["role"] for record in output_records)
    payload: dict[str, Any] = {
        "schema": "mars-sparse-detail-protocol/v1",
        "state": "prepared-before-third-round-training",
        "seed": seed,
        "parent_protocol_file_sha256": parent_protocol_file_sha256,
        "fresh_qa_file_sha256": fresh_qa_file_sha256,
        "role_counts": dict(sorted(counts.items())),
        "reclassified_parent_roles": dict(sorted(reclassified.items())),
        "validation_target": validation_target,
        "validation_selected": len(selected),
        "validation_shortfall": max(0, validation_target - len(selected)),
        "fresh_validation_rejections": rejected,
        "excluded_development_orbit_overlap": excluded_development,
        "external_policy": (
            "external records are copied as metadata only and remain forbidden until the "
            "selector, reconstructor, evaluator, and all hashes are frozen"
        ),
        "tiles": sorted(output_records, key=lambda value: (value["role"], value["tile_id"])),
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def build_expanded_training_index(
    protocol: Mapping[str, Any],
    case_manifests: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    protocol_file_sha256: str,
) -> dict[str, Any]:
    cases: dict[str, dict[str, Any]] = {}
    for root, manifest in case_manifests:
        for record in manifest["cases"]:
            tile_id = str(record["tile_id"])
            if tile_id in cases:
                raise ValueError(f"duplicate materialized case for {tile_id}")
            cases[tile_id] = {
                **record,
                "case_relative_path": f"{root.rstrip('/')}/{record['case']}",
            }
    training_records = [
        record for record in protocol["tiles"] if record["role"] == "development_train"
    ]
    missing = sorted(str(record["tile_id"]) for record in training_records if record["tile_id"] not in cases)
    if missing:
        raise ValueError(f"expanded training cases are not materialized: {missing[:5]}")
    samples = []
    for record in sorted(training_records, key=lambda value: str(value["tile_id"])):
        case = cases[str(record["tile_id"])]
        samples.append(
            {
                "tile_id": record["tile_id"],
                "region": record["region"],
                "parent_role": record["third_round_provenance"]["parent_role"],
                "case_relative_path": case["case_relative_path"],
                "case_sha256": case["sha256"],
                "assimilation_point_count": case["assimilation_point_count"],
                "evaluation_point_count": case["evaluation_point_count"],
                "permitted_use": "third_round_development_training_only",
            }
        )
    payload: dict[str, Any] = {
        "schema": "mars-sparse-detail-training-index/v1",
        "state": "frozen-before-third-round-training",
        "protocol_file_sha256": protocol_file_sha256,
        "training_tile_count": len(samples),
        "added_from_opened_parent_validation": sum(
            sample["parent_role"] == "validation" for sample in samples
        ),
        "added_from_opened_parent_locked_test": sum(
            sample["parent_role"] == "locked_test" for sample in samples
        ),
        "selector_instances": list(THIRD_ROUND_SELECTORS),
        "budgets": list(THIRD_ROUND_BUDGETS),
        "sampling_policy": (
            "balance selector family and laser budget within each epoch; random instances "
            "are averaged as one family; no validation or external target enters training"
        ),
        "samples": samples,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def build_selection_budget_matrix(
    protocol: Mapping[str, Any],
    *,
    protocol_file_sha256: str,
    output_root: str = "runs/sparse-detail-v30",
) -> dict[str, Any]:
    validation = sorted(
        (record for record in protocol["tiles"] if record["role"] == "validation"),
        key=lambda value: str(value["tile_id"]),
    )
    cells = []
    for record in validation:
        for selector in THIRD_ROUND_SELECTORS:
            family = "random" if selector.startswith("random-") else selector
            for budget in THIRD_ROUND_BUDGETS:
                for model_seed in THIRD_ROUND_MODEL_SEEDS:
                    cell_id = (
                        f"{record['tile_id']}__{selector}__k{budget:03d}__s{model_seed}"
                    )
                    cells.append(
                        {
                            "cell_id": cell_id,
                            "tile_id": record["tile_id"],
                            "region": record["region"],
                            "selector_instance": selector,
                            "selector_family": family,
                            "budget": budget,
                            "model_seed": model_seed,
                            "prediction_relative_path": (
                                f"{output_root}/predictions/{selector}/k{budget:03d}/"
                                f"seed-{model_seed}/{record['tile_id']}/prediction.npz"
                            ),
                            "status": "awaiting_case_materialization_and_inference",
                        }
                    )
    shared_k0 = [
        {
            "tile_id": record["tile_id"],
            "model_seed": seed,
            "prediction_relative_path": (
                f"{output_root}/predictions/shared-k000/seed-{seed}/"
                f"{record['tile_id']}/prediction.npz"
            ),
            "status": "awaiting_case_materialization_and_inference",
        }
        for record in validation
        for seed in THIRD_ROUND_MODEL_SEEDS
    ]
    payload: dict[str, Any] = {
        "schema": "mars-sparse-detail-inference-matrix/v1",
        "state": "frozen-before-inference",
        "protocol_file_sha256": protocol_file_sha256,
        "validation_tile_count": len(validation),
        "selectors": list(THIRD_ROUND_SELECTORS),
        "selector_family_count": 5,
        "budgets": list(THIRD_ROUND_BUDGETS),
        "model_seeds": list(THIRD_ROUND_MODEL_SEEDS),
        "matrix_cell_count": len(cells),
        "shared_k0_cell_count": len(shared_k0),
        "reconstructor_policy": (
            "one frozen selector-agnostic three-seed reconstructor; only selected laser "
            "locations and budget vary across matrix cells"
        ),
        "random_aggregation": (
            "average the three random selector instances within tile and budget before "
            "tile-level inference"
        ),
        "cells": cells,
        "shared_k0_cells": shared_k0,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def selector_contract() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "mars-sparse-detail-selector-contract/v1",
        "state": "frozen-before-plan-generation",
        "maximum_budget": max(THIRD_ROUND_BUDGETS),
        "nested_prefixes_required": True,
        "selectors": [
            {
                "id": "random-00..02",
                "algorithm": "SHA256-ranked candidate permutation with frozen streams",
            },
            {
                "id": "uniform-farthest",
                "algorithm": "deterministic Euclidean maximin ordering",
            },
            {
                "id": "max-prior-variance",
                "algorithm": "descending train-frozen terrain-prior marginal variance",
            },
            {
                "id": "greedy-mi-point",
                "algorithm": "greedy conditional point variance under the frozen prior",
            },
            {
                "id": "terrain-faopt",
                "algorithm": (
                    "Paper1-style greedy integrated posterior-variance reduction with "
                    "finite-footprint covariance and a train-frozen terrain prior"
                ),
                "naming_note": (
                    "new-domain re-estimation of the Paper1 mechanism; not a reused coordinate "
                    "sequence and not described as byte-identical Paper1 Discrete-FAOPT"
                ),
            },
        ],
        "permitted_inputs": [
            "coarse_dem",
            "registered_orbital_image",
            "candidate_coordinates",
            "candidate_track_ids",
            "finite_footprint_geometry",
            "train_frozen_prior",
        ],
        "forbidden_inputs": [
            "reference_dem",
            "heldout_mola_elevations",
            "validation_metrics",
            "external_metrics",
            "landing_safety_labels",
        ],
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload

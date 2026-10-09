#!/usr/bin/env python3
"""Third-round selector-by-budget training preparation and fixed-model training."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

import npj_real_pipeline_v3 as base
from paper2dem.sparse_detail_runtime import (
    DETAIL_BUDGETS,
    DETAIL_SELECTORS,
    PRIOR_LENGTHSCALE_PX,
    PRIOR_NOISE_M,
    SELECTOR_FAMILIES,
    SparseDetailCase,
    build_selector_orders,
    file_sha256,
    fit_terrain_prior,
    load_case,
    selector_for_family,
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_training_cases(project: Path, index_path: Path) -> list[SparseDetailCase]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    output = []
    for record in index["samples"]:
        path = project / record["case_relative_path"]
        if file_sha256(path) != record["case_sha256"]:
            raise ValueError(f"training case hash differs: {record['tile_id']}")
        output.append(
            load_case(
                path,
                tile_id=record["tile_id"],
                region=record["region"],
                role="train",
            )
        )
    return output


def load_validation_cases(case_root: Path) -> list[SparseDetailCase]:
    manifest = json.loads((case_root / "cases-manifest.json").read_text(encoding="utf-8"))
    output = []
    for record in manifest["cases"]:
        path = case_root / record["case"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"validation case hash differs: {record['tile_id']}")
        output.append(
            load_case(
                path,
                tile_id=record["tile_id"],
                region=record["region"],
                role="validation",
            )
        )
    return output


def command_materialize_validation(args: argparse.Namespace) -> int:
    project = args.project_root.resolve()
    protocol_path = args.protocol.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"validation case output is nonempty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    records = [record for record in protocol["tiles"] if record["role"] == "validation"]
    receipts = []
    for record in records:
        source_path = project / "data" / record["source_relative_path"]
        mola_path = (
            project
            / "data/real_mola_risk_independent_combined-v1"
            / record["mola_points_relative_path"]
        )
        if file_sha256(source_path) != record["source_sha256"]:
            raise ValueError(f"source hash differs: {record['tile_id']}")
        if file_sha256(mola_path) != record["mola_points_sha256"]:
            raise ValueError(f"MOLA hash differs: {record['tile_id']}")
        with np.load(source_path, allow_pickle=False) as source:
            source_arrays = {key: np.asarray(source[key]) for key in source.files}
        with np.load(mola_path, allow_pickle=False) as points:
            row = np.asarray(points["row"], dtype=np.float64)
            column = np.asarray(points["column"], dtype=np.float64)
            elevation = np.asarray(points["elevation"], dtype=np.float64)
            orbit = np.asarray(points["orbit"], dtype=np.int64)
            quality_valid = np.asarray(points["quality_valid"], dtype=bool)
        keep, screening = base.robust_point_filter(
            source_arrays["z0"], row, column, elevation
        )
        keep &= quality_valid
        assimilation = keep & np.isin(orbit, record["assimilation_orbits"])
        evaluation = keep & np.isin(orbit, record["evaluation_orbits"])
        if np.count_nonzero(assimilation) < 20 or np.count_nonzero(evaluation) < 20:
            raise ValueError(f"screened validation points are insufficient: {record['tile_id']}")
        payload = {
            "image": np.asarray(source_arrays["image"], dtype=np.float32),
            "z0": np.asarray(source_arrays["z0"], dtype=np.float32),
            "dem": np.asarray(source_arrays["dem"], dtype=np.float32),
            "valid": np.asarray(source_arrays["valid_mask"], dtype=bool),
            "row": row.astype(np.float32),
            "column": column.astype(np.float32),
            "elevation": elevation.astype(np.float32),
            "orbit": orbit,
            "assimilation": assimilation,
            "evaluation": evaluation,
            "filter_report": np.asarray(json.dumps(screening)),
        }
        case_path = output / f"{record['tile_id']}.npz"
        np.savez_compressed(case_path, **payload)
        receipts.append(
            {
                "tile_id": record["tile_id"],
                "region": record["region"],
                "role": "validation",
                "case": case_path.name,
                "sha256": file_sha256(case_path),
                "assimilation_point_count": int(np.count_nonzero(assimilation)),
                "evaluation_point_count": int(np.count_nonzero(evaluation)),
                "screening": screening,
            }
        )
        print(f"MATERIALIZED {record['tile_id']}", flush=True)
    manifest = {
        "schema": "mars-sparse-detail-validation-cases/v1",
        "state": "materialized_before_training_without_metric_evaluation",
        "protocol_sha256": file_sha256(protocol_path),
        "case_count": len(receipts),
        "external_records_read": 0,
        "external_files_accessed": False,
        "formal_metrics_computed": False,
        "cases": receipts,
    }
    atomic_json(output / "cases-manifest.json", manifest)
    return 0


def command_fit_prior(args: argparse.Namespace) -> int:
    project = args.project_root.resolve()
    index_path = args.training_index.resolve()
    cases = load_training_cases(project, index_path)
    prior = fit_terrain_prior(
        cases,
        samples_per_tile=args.samples_per_tile,
        ridge=args.ridge,
        seed=args.seed,
    )
    prior["training_index_sha256"] = file_sha256(index_path)
    prior["training_case_file_hashes_verified"] = True
    prior["validation_cases_loaded"] = False
    prior["external_cases_loaded"] = False
    prior["content_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in prior.items() if key != "content_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    atomic_json(args.output.resolve(), prior)
    print(
        json.dumps(
            {
                "state": prior["state"],
                "training_tile_count": prior["training_tile_count"],
                "sample_count": prior["sample_count"],
                "training_correlation": prior["training_correlation"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def command_generate_plans(args: argparse.Namespace) -> int:
    project = args.project_root.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"selector plan output is nonempty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    prior_path = args.prior.resolve()
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    cases = load_training_cases(project, args.training_index.resolve())
    cases.extend(load_validation_cases(args.validation_cases.resolve()))
    receipts = []
    for number, case in enumerate(cases, start=1):
        orders = build_selector_orders(case, prior, maximum_budget=max(DETAIL_BUDGETS))
        prior_raster = orders.pop("prior_information_raster")
        plan_path = output / f"{case.tile_id}.npz"
        np.savez_compressed(plan_path, **orders)
        receipts.append(
            {
                "tile_id": case.tile_id,
                "region": case.region,
                "role": case.role,
                "plan": plan_path.name,
                "sha256": file_sha256(plan_path),
                "candidate_count": int(np.count_nonzero(case.arrays["assimilation"])),
                "maximum_selected_count": int(len(orders[DETAIL_SELECTORS[0]])),
                "prior_information_mean": float(
                    np.mean(prior_raster[case.arrays["valid"]])
                ),
                "prior_information_p95": float(
                    np.quantile(prior_raster[case.arrays["valid"]], 0.95)
                ),
            }
        )
        print(f"PLAN {number}/{len(cases)} {case.role} {case.tile_id}", flush=True)
    manifest = {
        "schema": "mars-sparse-detail-selector-plans/v1",
        "state": "frozen_before_network_training",
        "prior_sha256": file_sha256(prior_path),
        "training_index_sha256": file_sha256(args.training_index.resolve()),
        "validation_case_manifest_sha256": file_sha256(
            args.validation_cases.resolve() / "cases-manifest.json"
        ),
        "selectors": list(DETAIL_SELECTORS),
        "budgets": list(DETAIL_BUDGETS),
        "nested_prefixes": True,
        "plan_count": len(receipts),
        "training_plan_count": sum(record["role"] == "train" for record in receipts),
        "validation_plan_count": sum(
            record["role"] == "validation" for record in receipts
        ),
        "validation_or_external_targets_used_by_selector": False,
        "plans": receipts,
    }
    atomic_json(output / "plans-manifest.json", manifest)
    return 0


def load_plans(plan_root: Path, roles: set[str]) -> dict[str, dict[str, np.ndarray]]:
    manifest = json.loads((plan_root / "plans-manifest.json").read_text(encoding="utf-8"))
    output = {}
    for record in manifest["plans"]:
        if record["role"] not in roles:
            continue
        path = plan_root / record["plan"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"selector plan hash differs: {record['tile_id']}")
        with np.load(path, allow_pickle=False) as archive:
            output[record["tile_id"]] = {
                selector: np.asarray(archive[selector], dtype=np.int64)
                for selector in DETAIL_SELECTORS
            }
    return output


class FieldCache:
    def __init__(self, maximum: int) -> None:
        self.maximum = int(maximum)
        self.values: OrderedDict[
            tuple[str, str, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]
        ] = OrderedDict()

    def get(
        self,
        case: Any,
        plans: dict[str, dict[str, np.ndarray]],
        selector: str,
        budget: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        key = (case.tile_id, selector, int(budget))
        if key in self.values:
            self.values.move_to_end(key)
            return self.values[key]
        order = plans[case.tile_id][selector]
        context = order[: min(int(budget), len(order))]
        gp, gp_std, distance = base.build_gp_fields(
            case,
            context,
            lengthscale_px=PRIOR_LENGTHSCALE_PX,
            noise_m=PRIOR_NOISE_M,
        )
        rows = case.arrays["row"][context]
        columns = case.arrays["column"][context]
        dem_at_context = base.sample_raster(case.arrays["dem"], rows, columns)
        datum_shift = float(
            np.median(case.arrays["elevation"][context] - dem_at_context)
        )
        value = (gp, gp_std, distance, context, datum_shift)
        self.values[key] = value
        self.values.move_to_end(key)
        while len(self.values) > self.maximum:
            self.values.popitem(last=False)
        return value


def point_maps(
    case: Any, context: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_map = np.zeros((504, 504), dtype=np.float32)
    target_count = np.zeros((504, 504), dtype=np.float32)
    context_map = np.zeros((504, 504), dtype=np.float32)
    context_count = np.zeros((504, 504), dtype=np.float32)
    for indices, value_map, count_map in (
        (np.flatnonzero(case.arrays["evaluation"]), target_map, target_count),
        (context, context_map, context_count),
    ):
        rows = np.clip(np.rint(case.arrays["row"][indices]).astype(int), 0, 503)
        columns = np.clip(np.rint(case.arrays["column"][indices]).astype(int), 0, 503)
        np.add.at(value_map, (rows, columns), case.arrays["elevation"][indices])
        np.add.at(count_map, (rows, columns), 1.0)
    target_mask = target_count > 0
    context_mask = context_count > 0
    target_map[target_mask] /= target_count[target_mask]
    context_map[context_mask] /= context_count[context_mask]
    return target_map, target_mask, context_map, context_mask


def crop_sample(
    case: Any,
    fields: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float],
    crop_size: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    gp, gp_std, distance, context, datum_shift = fields
    condition = base.condition_array_from_fields(case, gp, gp_std, distance)
    target_map, target_mask, context_map, context_mask = point_maps(case, context)
    target_indices = np.flatnonzero(target_mask)
    selected = int(rng.choice(target_indices))
    target_row, target_column = np.unravel_index(selected, target_mask.shape)
    top = int(
        np.clip(
            target_row - rng.integers(crop_size // 4, 3 * crop_size // 4),
            0,
            504 - crop_size,
        )
    )
    left = int(
        np.clip(
            target_column - rng.integers(crop_size // 4, 3 * crop_size // 4),
            0,
            504 - crop_size,
        )
    )
    selection = np.s_[top : top + crop_size, left : left + crop_size]
    output = {
        "condition": condition[:, selection[0], selection[1]],
        "gp": gp[selection],
        "dense_target": (case.arrays["dem"] + datum_shift)[selection],
        "valid": case.arrays["valid"][selection],
        "target_map": target_map[selection],
        "target_mask": target_mask[selection],
        "context_map": context_map[selection],
        "context_mask": context_mask[selection],
    }
    if rng.random() < 0.5:
        output = {key: np.flip(value, axis=-1).copy() for key, value in output.items()}
    if rng.random() < 0.5:
        output = {key: np.flip(value, axis=-2).copy() for key, value in output.items()}
    return output


def command_train(args: argparse.Namespace) -> int:
    torch, _, functional = base.torch_components()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    project = args.project_root.resolve()
    runtime_cases = load_training_cases(project, args.training_index.resolve())
    cases = [
        base.LoadedCase(case.tile_id, case.region, "train", case.arrays)
        for case in runtime_cases
    ]
    plans = load_plans(args.plans.resolve(), {"train"})
    if set(plans) != {case.tile_id for case in cases}:
        raise ValueError("training cases and selector plans do not close")
    by_region: dict[str, list[Any]] = {}
    for case in cases:
        by_region.setdefault(case.region, []).append(case)
    regions = sorted(by_region)
    model = base.make_model(
        base_channels=args.base_channels,
        steps=args.transport_steps,
        minimum_mobility=args.minimum_mobility,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.updates, eta_min=args.learning_rate * 0.02
    )
    rng = np.random.default_rng(seed)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"training output is nonempty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cache = FieldCache(args.field_cache_size)
    started = time.time()
    family_counts = {family: 0 for family in SELECTOR_FAMILIES}
    selector_counts = {selector: 0 for selector in DETAIL_SELECTORS}
    budget_counts = {budget: 0 for budget in DETAIL_BUDGETS}
    model.train()
    for update in range(1, args.updates + 1):
        samples = []
        for _ in range(args.batch_size):
            region = regions[int(rng.integers(len(regions)))]
            members = by_region[region]
            case = members[int(rng.integers(len(members)))]
            family = SELECTOR_FAMILIES[int(rng.integers(len(SELECTOR_FAMILIES)))]
            selector = selector_for_family(family, rng=rng)
            budget = DETAIL_BUDGETS[int(rng.integers(len(DETAIL_BUDGETS)))]
            family_counts[family] += 1
            selector_counts[selector] += 1
            budget_counts[budget] += 1
            fields = cache.get(case, plans, selector, budget)
            samples.append(crop_sample(case, fields, args.crop_size, rng))

        def stack(name: str, dtype: Any = torch.float32) -> Any:
            return torch.from_numpy(np.stack([sample[name] for sample in samples])).to(
                device=device, dtype=dtype
            )

        condition = stack("condition")
        gp = stack("gp")
        dense_target = stack("dense_target")
        valid = stack("valid", torch.bool)
        target_map = stack("target_map")
        target_mask = stack("target_mask", torch.bool)
        context_map = stack("context_map")
        context_mask = stack("context_mask", torch.bool)
        optimizer.zero_grad(set_to_none=True)
        residual, kinetic = model(condition)
        prediction = gp[:, None] + residual * base.HEIGHT_SCALE_M
        dense_error = (prediction[:, 0] - dense_target) / base.HEIGHT_SCALE_M
        dense_loss = functional.smooth_l1_loss(
            dense_error[valid], torch.zeros_like(dense_error[valid])
        )
        sparse_error = (prediction[:, 0] - target_map) / base.HEIGHT_SCALE_M
        sparse_loss = (
            functional.smooth_l1_loss(
                sparse_error[target_mask], torch.zeros_like(sparse_error[target_mask])
            )
            if torch.any(target_mask)
            else torch.zeros((), device=device)
        )
        context_error = (prediction[:, 0] - context_map) / base.HEIGHT_SCALE_M
        context_loss = (
            functional.smooth_l1_loss(
                context_error[context_mask], torch.zeros_like(context_error[context_mask])
            )
            if torch.any(context_mask)
            else torch.zeros((), device=device)
        )
        pred_dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
        pred_dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
        target_dx = dense_target[:, None, :, 1:] - dense_target[:, None, :, :-1]
        target_dy = dense_target[:, None, 1:, :] - dense_target[:, None, :-1, :]
        gradient_loss = functional.smooth_l1_loss(
            pred_dx / 10.0, target_dx / 10.0
        ) + functional.smooth_l1_loss(pred_dy / 10.0, target_dy / 10.0)
        loss = (
            dense_loss
            + 3.0 * sparse_loss
            + context_loss
            + 0.25 * gradient_loss
            + 0.005 * kinetic
        )
        if not torch.isfinite(loss):
            failure = {
                "state": "failed_nonfinite_loss",
                "seed": seed,
                "update": update,
                "dense_loss": float(dense_loss.detach().cpu()),
                "sparse_loss": float(sparse_loss.detach().cpu()),
                "context_loss": float(context_loss.detach().cpu()),
                "gradient_loss": float(gradient_loss.detach().cpu()),
                "kinetic_loss": float(kinetic.detach().cpu()),
            }
            atomic_json(output / "failure.json", failure)
            raise FloatingPointError(f"nonfinite training loss at update {update}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if update % args.log_interval == 0 or update == 1:
            elapsed = time.time() - started
            progress = {
                "state": "training",
                "seed": seed,
                "update": update,
                "updates": args.updates,
                "loss": float(loss.detach().cpu()),
                "dense_loss": float(dense_loss.detach().cpu()),
                "sparse_loss": float(sparse_loss.detach().cpu()),
                "context_loss": float(context_loss.detach().cpu()),
                "gradient_loss": float(gradient_loss.detach().cpu()),
                "kinetic_loss": float(kinetic.detach().cpu()),
                "learning_rate": scheduler.get_last_lr()[0],
                "updates_per_second": update / elapsed,
                "eta_seconds": (args.updates - update) / max(update / elapsed, 1.0e-9),
                "field_cache_entries": len(cache.values),
                "family_counts": family_counts,
                "selector_counts": selector_counts,
                "budget_counts": budget_counts,
                "network_precision": "fp32",
            }
            atomic_json(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
        if update in set(args.snapshot_update):
            torch.save(
                {
                    "model": model.state_dict(),
                    "seed": seed,
                    "update": update,
                    "architecture": {
                        "base_channels": args.base_channels,
                        "transport_steps": args.transport_steps,
                        "minimum_mobility": args.minimum_mobility,
                    },
                    "training_contract": {
                        "selector_family_balanced": True,
                        "budgets": list(DETAIL_BUDGETS),
                        "network_precision": "fp32",
                    },
                },
                output / f"snapshot-{update}.pt",
            )
    final_path = output / "final.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "seed": seed,
            "update": args.updates,
            "architecture": {
                "base_channels": args.base_channels,
                "transport_steps": args.transport_steps,
                "minimum_mobility": args.minimum_mobility,
            },
            "training_contract": {
                "selector_family_balanced": True,
                "budgets": list(DETAIL_BUDGETS),
                "network_precision": "fp32",
                "validation_targets_read": False,
                "external_targets_read": False,
            },
        },
        final_path,
    )
    manifest = {
        "schema": "mars-sparse-detail-training/v1",
        "state": "complete",
        "seed": seed,
        "completed_updates": args.updates,
        "training_tile_count": len(cases),
        "network_precision": "fp32",
        "selector_family_counts": family_counts,
        "selector_instance_counts": selector_counts,
        "budget_counts": budget_counts,
        "final_checkpoint_sha256": file_sha256(final_path),
        "training_index_sha256": file_sha256(args.training_index.resolve()),
        "plan_manifest_sha256": file_sha256(
            args.plans.resolve() / "plans-manifest.json"
        ),
        "validation_targets_read": False,
        "external_targets_read": False,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(output / "training-manifest.json", manifest)
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    materialize = commands.add_parser("materialize-validation")
    materialize.add_argument("--project-root", type=Path, required=True)
    materialize.add_argument("--protocol", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    materialize.set_defaults(function=command_materialize_validation)
    prior = commands.add_parser("fit-prior")
    prior.add_argument("--project-root", type=Path, required=True)
    prior.add_argument("--training-index", type=Path, required=True)
    prior.add_argument("--output", type=Path, required=True)
    prior.add_argument("--samples-per-tile", type=int, default=2048)
    prior.add_argument("--ridge", type=float, default=0.1)
    prior.add_argument("--seed", type=int, default=20260810)
    prior.set_defaults(function=command_fit_prior)
    plans = commands.add_parser("generate-plans")
    plans.add_argument("--project-root", type=Path, required=True)
    plans.add_argument("--training-index", type=Path, required=True)
    plans.add_argument("--validation-cases", type=Path, required=True)
    plans.add_argument("--prior", type=Path, required=True)
    plans.add_argument("--output", type=Path, required=True)
    plans.set_defaults(function=command_generate_plans)
    train = commands.add_parser("train")
    train.add_argument("--project-root", type=Path, required=True)
    train.add_argument("--training-index", type=Path, required=True)
    train.add_argument("--plans", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--updates", type=int, default=2000)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--crop-size", type=int, default=320)
    train.add_argument("--base-channels", type=int, default=32)
    train.add_argument("--transport-steps", type=int, default=3)
    train.add_argument("--minimum-mobility", type=float, default=0.20)
    train.add_argument("--learning-rate", type=float, default=2.0e-4)
    train.add_argument("--field-cache-size", type=int, default=64)
    train.add_argument("--log-interval", type=int, default=25)
    train.add_argument("--snapshot-update", type=int, action="append", default=[])
    train.set_defaults(function=command_train)
    return root


def main() -> int:
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Frequency-balanced fine-tuning for sparse-detail DEM reconstruction."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

import npj_real_pipeline_v3 as base
import run_sparse_detail_v30 as v30
import run_sparse_detail_v31 as v31


SCHEMA = "mars-sparse-detail-lowpass-training/v1"
PIXEL_SIZE_M = 20.0
DETAIL_SCALES_M = (40.0, 80.0, 160.0)
LOWPASS_SCALE_M = 320.0


def _gaussian_kernel(torch: Any, sigma: float, device: Any, dtype: Any) -> Any:
    radius = max(1, int(math.ceil(4.0 * sigma)))
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (coordinates / sigma) ** 2)
    return kernel / torch.sum(kernel)


def _gaussian_blur(torch: Any, functional: Any, field: Any, sigma: float) -> Any:
    kernel = _gaussian_kernel(torch, sigma, field.device, field.dtype)
    radius = (kernel.numel() - 1) // 2
    horizontal = kernel.view(1, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1)
    padded = functional.pad(field, (radius, radius, 0, 0), mode="reflect")
    blurred = functional.conv2d(padded, horizontal)
    padded = functional.pad(blurred, (0, 0, radius, radius), mode="reflect")
    return functional.conv2d(padded, vertical)


def _masked_gaussian_blur(
    torch: Any,
    functional: Any,
    field: Any,
    valid: Any,
    sigma: float,
) -> tuple[Any, Any]:
    weights = valid[:, None].to(dtype=field.dtype)
    numerator = _gaussian_blur(
        torch, functional, torch.where(weights > 0, field, torch.zeros_like(field)), sigma
    )
    denominator = _gaussian_blur(torch, functional, weights, sigma)
    smoothed = numerator / torch.clamp(denominator, min=1.0e-6)
    support = valid[:, None] & (denominator >= 0.995)
    return smoothed, support


def _masked_downsample(functional: Any, field: Any, valid: Any, factor: int) -> tuple[Any, Any]:
    if factor == 1:
        return field, valid
    weights = valid[:, None].to(dtype=field.dtype)
    pooled_weights = functional.avg_pool2d(weights, factor, stride=factor)
    pooled_values = functional.avg_pool2d(
        field * weights, factor, stride=factor
    )
    downsampled = pooled_values / pooled_weights.clamp_min(1.0e-6)
    return downsampled, pooled_weights[:, 0] >= 0.995


def _frequency_losses(
    torch: Any,
    functional: Any,
    prediction: Any,
    target: Any,
    valid: Any,
) -> tuple[Any, Any]:
    low_factor = int(LOWPASS_SCALE_M / 40.0)
    prediction_low_grid, valid_low_grid = _masked_downsample(
        functional, prediction, valid, low_factor
    )
    target_low_grid, target_valid_low_grid = _masked_downsample(
        functional, target, valid, low_factor
    )
    predicted_low, predicted_support = _masked_gaussian_blur(
        torch, functional, prediction_low_grid, valid_low_grid, 2.0
    )
    target_low, target_support = _masked_gaussian_blur(
        torch, functional, target_low_grid, target_valid_low_grid, 2.0
    )
    low_support = predicted_support & target_support
    lowpass_loss = functional.smooth_l1_loss(
        (predicted_low[low_support] - target_low[low_support]) / base.HEIGHT_SCALE_M,
        torch.zeros_like(predicted_low[low_support]),
    )

    detail_terms = []
    for scale_m in DETAIL_SCALES_M:
        factor = int(scale_m / 40.0)
        prediction_grid, valid_grid = _masked_downsample(
            functional, prediction, valid, factor
        )
        target_grid, target_valid_grid = _masked_downsample(
            functional, target, valid, factor
        )
        predicted_smooth, predicted_support = _masked_gaussian_blur(
            torch, functional, prediction_grid, valid_grid, 2.0
        )
        target_smooth, target_support = _masked_gaussian_blur(
            torch, functional, target_grid, target_valid_grid, 2.0
        )
        support = predicted_support & target_support
        predicted_detail = prediction_grid - predicted_smooth
        target_detail = target_grid - target_smooth
        detail_terms.append(
            functional.smooth_l1_loss(
                (predicted_detail[support] - target_detail[support]) / 10.0,
                torch.zeros_like(predicted_detail[support]),
            )
        )
    detail_loss = torch.stack(detail_terms).mean()
    return lowpass_loss, detail_loss


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
    runtime_cases = v30.load_training_cases(project, args.training_index.resolve())
    cases = [
        base.LoadedCase(case.tile_id, case.region, "train", case.arrays)
        for case in runtime_cases
    ]
    plans = v30.load_plans(args.plans.resolve(), {"train"})
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
    initial_path = args.initial_checkpoint.resolve()
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    architecture = initial.get("architecture", {})
    expected_architecture = {
        "base_channels": args.base_channels,
        "transport_steps": args.transport_steps,
        "minimum_mobility": args.minimum_mobility,
    }
    if architecture != expected_architecture:
        raise ValueError(
            f"initial checkpoint architecture differs: {architecture} != {expected_architecture}"
        )
    model.load_state_dict(initial["model"], strict=True)

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
    cache = v31.TorchFieldCache(args.field_cache_size, args.device)
    started = time.time()
    family_counts = {family: 0 for family in v30.SELECTOR_FAMILIES}
    selector_counts = {selector: 0 for selector in v30.DETAIL_SELECTORS}
    budget_counts = {budget: 0 for budget in v30.DETAIL_BUDGETS}
    snapshot_updates = set(args.snapshot_update)
    model.train()

    for update in range(1, args.updates + 1):
        samples = []
        for _ in range(args.batch_size):
            region = regions[int(rng.integers(len(regions)))]
            members = by_region[region]
            case = members[int(rng.integers(len(members)))]
            family = v30.SELECTOR_FAMILIES[
                int(rng.integers(len(v30.SELECTOR_FAMILIES)))
            ]
            selector = v30.selector_for_family(family, rng=rng)
            budget = v30.DETAIL_BUDGETS[
                int(rng.integers(len(v30.DETAIL_BUDGETS)))
            ]
            family_counts[family] += 1
            selector_counts[selector] += 1
            budget_counts[budget] += 1
            fields = cache.get(case, plans, selector, budget)
            samples.append(v30.crop_sample(case, fields, args.crop_size, rng))

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
        target = dense_target[:, None]
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
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        gradient_loss = functional.smooth_l1_loss(
            pred_dx / 10.0, target_dx / 10.0
        ) + functional.smooth_l1_loss(pred_dy / 10.0, target_dy / 10.0)
        lowpass_loss, detail_loss = _frequency_losses(
            torch, functional, prediction, target, valid
        )
        loss = (
            dense_loss
            + 3.0 * sparse_loss
            + context_loss
            + 0.25 * gradient_loss
            + args.lowpass_weight * lowpass_loss
            + args.detail_weight * detail_loss
            + 0.005 * kinetic
        )
        if not torch.isfinite(loss):
            failure = {
                "state": "failed_nonfinite_loss",
                "seed": seed,
                "update": update,
                "loss": float(loss.detach().cpu()),
                "lowpass_loss": float(lowpass_loss.detach().cpu()),
                "detail_loss": float(detail_loss.detach().cpu()),
            }
            v30.atomic_json(output / "failure.json", failure)
            raise FloatingPointError(f"nonfinite training loss at update {update}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if update % args.log_interval == 0 or update == 1:
            elapsed = time.time() - started
            progress = {
                "state": "training",
                "schema": SCHEMA,
                "seed": seed,
                "update": update,
                "updates": args.updates,
                "loss": float(loss.detach().cpu()),
                "dense_loss": float(dense_loss.detach().cpu()),
                "sparse_loss": float(sparse_loss.detach().cpu()),
                "context_loss": float(context_loss.detach().cpu()),
                "gradient_loss": float(gradient_loss.detach().cpu()),
                "lowpass_loss": float(lowpass_loss.detach().cpu()),
                "detail_loss": float(detail_loss.detach().cpu()),
                "kinetic_loss": float(kinetic.detach().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "learning_rate": scheduler.get_last_lr()[0],
                "updates_per_second": update / elapsed,
                "eta_seconds": (args.updates - update)
                / max(update / elapsed, 1.0e-9),
                "field_cache_entries": len(cache.values),
                "family_counts": family_counts,
                "selector_counts": selector_counts,
                "budget_counts": budget_counts,
                "network_precision": "fp32",
            }
            v30.atomic_json(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)

        if update in snapshot_updates:
            torch.save(
                {
                    "model": model.state_dict(),
                    "seed": seed,
                    "update": update,
                    "architecture": expected_architecture,
                    "training_contract": {
                        "schema": SCHEMA,
                        "initialized_from": str(initial_path),
                        "initialized_from_sha256": v30.file_sha256(initial_path),
                        "selector_family_balanced": True,
                        "budgets": list(v30.DETAIL_BUDGETS),
                        "detail_scales_m": list(DETAIL_SCALES_M),
                        "lowpass_scale_m": LOWPASS_SCALE_M,
                        "lowpass_weight": args.lowpass_weight,
                        "detail_weight": args.detail_weight,
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
            "architecture": expected_architecture,
            "training_contract": {
                "schema": SCHEMA,
                "initialized_from": str(initial_path),
                "initialized_from_sha256": v30.file_sha256(initial_path),
                "selector_family_balanced": True,
                "budgets": list(v30.DETAIL_BUDGETS),
                "detail_scales_m": list(DETAIL_SCALES_M),
                "lowpass_scale_m": LOWPASS_SCALE_M,
                "lowpass_weight": args.lowpass_weight,
                "detail_weight": args.detail_weight,
                "network_precision": "fp32",
                "validation_targets_read": False,
                "external_targets_read": False,
            },
        },
        final_path,
    )
    manifest = {
        "schema": SCHEMA,
        "state": "complete",
        "seed": seed,
        "completed_updates": args.updates,
        "training_tile_count": len(cases),
        "network_precision": "fp32",
        "initial_checkpoint": str(initial_path),
        "initial_checkpoint_sha256": v30.file_sha256(initial_path),
        "selector_family_counts": family_counts,
        "selector_instance_counts": selector_counts,
        "budget_counts": budget_counts,
        "detail_scales_m": list(DETAIL_SCALES_M),
        "lowpass_scale_m": LOWPASS_SCALE_M,
        "lowpass_weight": args.lowpass_weight,
        "detail_weight": args.detail_weight,
        "final_checkpoint_sha256": v30.file_sha256(final_path),
        "training_index_sha256": v30.file_sha256(args.training_index.resolve()),
        "plan_manifest_sha256": v30.file_sha256(
            args.plans.resolve() / "plans-manifest.json"
        ),
        "field_backend": v31.FIELD_BACKEND,
        "field_backend_equivalence_receipt": str(
            args.field_equivalence_receipt.resolve()
        ),
        "field_backend_equivalence_receipt_sha256": v30.file_sha256(
            args.field_equivalence_receipt.resolve()
        ),
        "validation_targets_read": False,
        "external_targets_read": False,
        "elapsed_seconds": time.time() - started,
    }
    v30.atomic_json(output / "training-manifest.json", manifest)
    return 0


def parser() -> argparse.ArgumentParser:
    root = v30.parser()
    choices = root._subparsers._group_actions[0].choices  # noqa: SLF001
    train = choices["train"]
    train.add_argument("--field-equivalence-receipt", type=Path, required=True)
    train.add_argument("--initial-checkpoint", type=Path, required=True)
    train.add_argument("--lowpass-weight", type=float, default=2.0)
    train.add_argument("--detail-weight", type=float, default=0.5)
    train.set_defaults(function=command_train)
    return root


def main() -> int:
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())

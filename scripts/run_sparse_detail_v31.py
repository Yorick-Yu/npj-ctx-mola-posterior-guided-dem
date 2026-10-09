#!/usr/bin/env python3
"""Numerically equivalent GPU-field backend for third-round training."""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

import npj_real_pipeline_v3 as base
import run_sparse_detail_v30 as v30
import run_sparse_detail_inference_v30 as inference
from paper2dem.sparse_detail_runtime import PRIOR_LENGTHSCALE_PX, PRIOR_NOISE_M


FIELD_BACKEND = "torch-gpu-float64-equivalent-v1"


def torch_gp_fields(
    case: Any,
    context_indices: np.ndarray,
    *,
    device: Any,
    lengthscale_px: float = PRIOR_LENGTHSCALE_PX,
    noise_m: float = PRIOR_NOISE_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the frozen GP equations in float64 on the assigned GPU."""

    torch, _, _ = base.torch_components()
    arrays = case.arrays
    rows = np.asarray(arrays["row"][context_indices], dtype=np.float64)
    columns = np.asarray(arrays["column"][context_indices], dtype=np.float64)
    residual = np.asarray(arrays["elevation"][context_indices], dtype=np.float64) - base.sample_raster(
        arrays["z0"], rows, columns
    )
    if residual.size == 0:
        shape = arrays["z0"].shape
        return (
            np.asarray(arrays["z0"], dtype=np.float32).copy(),
            np.full(shape, base.HEIGHT_SCALE_M, dtype=np.float32),
            np.full(shape, 400.0, dtype=np.float32),
        )
    center = float(np.median(residual))
    centered = residual - center
    amplitude = max(5.0, float(1.4826 * np.median(np.abs(centered))))
    x = torch.as_tensor(
        np.column_stack((rows, columns)), dtype=torch.float64, device=device
    )
    centered_tensor = torch.as_tensor(centered, dtype=torch.float64, device=device)
    squared = torch.sum((x[:, None, :] - x[None, :, :]) ** 2, dim=2)
    kernel = amplitude**2 * torch.exp(-0.5 * squared / lengthscale_px**2)
    kernel.diagonal().add_(noise_m**2 + 1.0e-5)
    cholesky = torch.linalg.cholesky(kernel)
    alpha = torch.cholesky_solve(centered_tensor[:, None], cholesky)[:, 0]
    grid_row, grid_column = np.indices(arrays["z0"].shape, dtype=np.float32)
    queries = np.column_stack((grid_row.ravel(), grid_column.ravel()))
    mean = np.empty(len(queries), dtype=np.float64)
    standard_deviation = np.empty(len(queries), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(queries), 32768):
            stop = min(start + 32768, len(queries))
            query = torch.as_tensor(
                queries[start:stop], dtype=torch.float64, device=device
            )
            cross_squared = torch.sum((query[:, None, :] - x[None, :, :]) ** 2, dim=2)
            cross = amplitude**2 * torch.exp(
                -0.5 * cross_squared / lengthscale_px**2
            )
            mean[start:stop] = (center + cross @ alpha).cpu().numpy()
            projection = torch.linalg.solve_triangular(
                cholesky, cross.T, upper=False
            )
            variance = amplitude**2 - torch.sum(projection**2, dim=0)
            standard_deviation[start:stop] = torch.sqrt(
                torch.clamp(variance, min=noise_m**2)
            ).cpu().numpy()
    mask = np.zeros(arrays["z0"].shape, dtype=bool)
    mask[
        np.clip(np.rint(rows).astype(int), 0, mask.shape[0] - 1),
        np.clip(np.rint(columns).astype(int), 0, mask.shape[1] - 1),
    ] = True
    return (
        (arrays["z0"] + mean.reshape(arrays["z0"].shape)).astype(np.float32),
        standard_deviation.reshape(arrays["z0"].shape).astype(np.float32),
        base.distance_transform_edt(~mask).astype(np.float32),
    )


class TorchFieldCache:
    def __init__(self, maximum: int, device: str) -> None:
        self.maximum = int(maximum)
        self.device = base.torch_components()[0].device(device)
        self.values: OrderedDict[
            tuple[str, str, int],
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float],
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
        gp, gp_std, distance = torch_gp_fields(case, context, device=self.device)
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


def command_verify(args: Any) -> int:
    torch, _, _ = base.torch_components()
    device = torch.device(args.device)
    project = args.project_root.resolve()
    cases = inference._load_inference_cases(args.inference_cases.resolve())
    plans = inference._load_plans(args.plans.resolve())
    selectors = ("uniform-farthest", "terrain-faopt")
    budgets = (5, 20, 80)
    rows = []
    started = time.time()
    for tile_id in sorted(cases):
        case = cases[tile_id]
        model_case = inference._model_case(case)
        for selector in selectors:
            for budget in budgets:
                global_context = plans[case.tile_id][selector][:budget]
                context = inference._remap_context(case, global_context)
                cpu = base.build_gp_fields(
                    model_case,
                    context,
                    lengthscale_px=PRIOR_LENGTHSCALE_PX,
                    noise_m=PRIOR_NOISE_M,
                )
                gpu = torch_gp_fields(model_case, context, device=device)
                rows.append(
                    {
                        "tile_id": case.tile_id,
                        "selector": selector,
                        "requested_budget": budget,
                        "selected_count": int(len(context)),
                        "mean_max_abs_m": float(np.max(np.abs(cpu[0] - gpu[0]))),
                        "std_max_abs_m": float(np.max(np.abs(cpu[1] - gpu[1]))),
                        "distance_max_abs_px": float(np.max(np.abs(cpu[2] - gpu[2]))),
                    }
                )
                print(f"VERIFY {case.tile_id} {selector} K{budget}", flush=True)
    passed = all(
        row["mean_max_abs_m"] == 0.0
        and row["std_max_abs_m"] == 0.0
        and row["distance_max_abs_px"] == 0.0
        for row in rows
    )
    payload = {
        "schema": "mars-sparse-detail-gpu-field-equivalence/v1",
        "state": "passed" if passed else "failed",
        "field_backend": FIELD_BACKEND,
        "project_root": str(project),
        "comparison_count": len(rows),
        "network_or_targets_trained": False,
        "reference_dem_read": False,
        "heldout_mola_read": False,
        "external_files_accessed": False,
        "elapsed_seconds": time.time() - started,
        "comparisons": rows,
    }
    v30.atomic_json(args.output.resolve(), payload)
    print(json.dumps({"state": payload["state"], "comparison_count": len(rows)}))
    return 0 if passed else 2


def command_train(args: Any) -> int:
    v30.FieldCache = lambda maximum: TorchFieldCache(maximum, args.device)
    result = v30.command_train(args)
    manifest_path = args.output.resolve() / "training-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["field_backend"] = FIELD_BACKEND
    manifest["field_arithmetic"] = "float64 GP equations followed by frozen float32 fields"
    manifest["field_backend_equivalence_receipt"] = str(
        args.field_equivalence_receipt.resolve()
    )
    manifest["field_backend_equivalence_receipt_sha256"] = v30.file_sha256(
        args.field_equivalence_receipt.resolve()
    )
    v30.atomic_json(manifest_path, manifest)
    return int(result)


def parser() -> Any:
    root = v30.parser()
    verify = root._subparsers._group_actions[0].choices  # noqa: SLF001
    train = verify["train"]
    train.add_argument("--field-equivalence-receipt", type=Path, required=True)
    train.set_defaults(function=command_train)
    equivalence = root._subparsers._group_actions[0].add_parser(  # noqa: SLF001
        "verify-gpu-fields"
    )
    equivalence.add_argument("--project-root", type=Path, required=True)
    equivalence.add_argument("--inference-cases", type=Path, required=True)
    equivalence.add_argument("--plans", type=Path, required=True)
    equivalence.add_argument("--device", default="cuda:0")
    equivalence.add_argument("--output", type=Path, required=True)
    equivalence.set_defaults(function=command_verify)
    return root


def main() -> int:
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())

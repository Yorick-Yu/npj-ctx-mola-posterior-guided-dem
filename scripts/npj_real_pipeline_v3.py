#!/usr/bin/env python3
"""Illumination-invariant real MOLA and CTX pipeline for the npj evidence gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates


BUDGETS = (5, 10, 16, 18, 20, 40, 80)
PRIMARY_BUDGET = 20
GROSS_ERROR_FLOOR_M = 200.0
GROSS_ERROR_MAD_MULTIPLIER = 8.0
HEIGHT_SCALE_M = 50.0
CONDITION_CHANNELS = 10
TERMINAL_LENGTHSCALE_PX = float(os.environ.get("NPJ_TERMINAL_LENGTHSCALE_PX", "48"))
DISABLE_TERMINAL = os.environ.get("NPJ_DISABLE_TERMINAL", "0") == "1"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_raster(array: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    return map_coordinates(array, [rows, columns], order=1, mode="nearest")


def robust_point_filter(
    z0: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    elevation: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    residual = elevation - sample_raster(z0, rows, columns)
    center = float(np.median(residual))
    mad = float(1.4826 * np.median(np.abs(residual - center)))
    threshold = max(GROSS_ERROR_FLOOR_M, GROSS_ERROR_MAD_MULTIPLIER * max(mad, 1.0))
    keep = np.isfinite(residual) & (np.abs(residual - center) <= threshold)
    return keep, {
        "residual_median_m": center,
        "residual_mad_m": mad,
        "gross_error_threshold_m": threshold,
        "input_count": int(residual.size),
        "retained_count": int(np.count_nonzero(keep)),
    }


def nested_spatial_order(rows: np.ndarray, columns: np.ndarray, seed: int) -> np.ndarray:
    if rows.size == 0:
        return np.empty(0, dtype=np.int64)
    points = np.column_stack((rows, columns)).astype(np.float64)
    rng = np.random.default_rng(seed)
    centroid = np.mean(points, axis=0)
    first_candidates = np.flatnonzero(
        np.isclose(np.sum((points - centroid) ** 2, axis=1), np.max(np.sum((points - centroid) ** 2, axis=1)))
    )
    first = int(rng.choice(first_candidates))
    selected = [first]
    minimum_distance = np.sum((points - points[first]) ** 2, axis=1)
    minimum_distance[first] = -1.0
    while len(selected) < len(points):
        maximum = float(np.max(minimum_distance))
        candidates = np.flatnonzero(np.isclose(minimum_distance, maximum))
        index = int(rng.choice(candidates))
        selected.append(index)
        minimum_distance = np.minimum(
            minimum_distance, np.sum((points - points[index]) ** 2, axis=1)
        )
        minimum_distance[np.asarray(selected, dtype=np.int64)] = -1.0
    return np.asarray(selected, dtype=np.int64)


def gp_predict(
    context_row: np.ndarray,
    context_column: np.ndarray,
    context_residual: np.ndarray,
    query_row: np.ndarray,
    query_column: np.ndarray,
    *,
    lengthscale_px: float,
    noise_m: float,
    return_std: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    if context_residual.size == 0:
        mean = np.zeros_like(query_row, dtype=np.float64)
        return mean, np.full_like(mean, HEIGHT_SCALE_M) if return_std else None
    center = float(np.median(context_residual))
    centered = np.asarray(context_residual, dtype=np.float64) - center
    amplitude = max(5.0, float(1.4826 * np.median(np.abs(centered))))
    x = np.column_stack((context_row, context_column)).astype(np.float64)
    q = np.column_stack((query_row, query_column)).astype(np.float64)
    squared = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)
    kernel = amplitude**2 * np.exp(-0.5 * squared / lengthscale_px**2)
    kernel.flat[:: len(x) + 1] += noise_m**2 + 1.0e-5
    try:
        chol = np.linalg.cholesky(kernel)
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, centered))
    except np.linalg.LinAlgError:
        alpha = np.linalg.solve(kernel + np.eye(len(x)) * 1.0e-3, centered)
        chol = None
    output = np.empty(len(q), dtype=np.float64)
    std = np.empty(len(q), dtype=np.float64) if return_std else None
    for start in range(0, len(q), 32768):
        stop = min(start + 32768, len(q))
        cross_squared = np.sum((q[start:stop, None, :] - x[None, :, :]) ** 2, axis=2)
        cross = amplitude**2 * np.exp(-0.5 * cross_squared / lengthscale_px**2)
        output[start:stop] = center + cross @ alpha
        if std is not None:
            if chol is None:
                variance = np.full(stop - start, amplitude**2)
            else:
                projection = np.linalg.solve(chol, cross.T)
                variance = amplitude**2 - np.sum(projection**2, axis=0)
            std[start:stop] = np.sqrt(np.maximum(variance, noise_m**2))
    return output, std


def load_raw_case(
    record: dict[str, Any], data_root: Path, mola_root: Path
) -> dict[str, np.ndarray]:
    source_path = data_root / "tiles" / record["tile_id"] / "source.npz"
    point_path = mola_root / record["mola_points_relative_path"]
    with np.load(source_path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    with np.load(point_path, allow_pickle=False) as points:
        rows = np.asarray(points["row"], dtype=np.float64)
        columns = np.asarray(points["column"], dtype=np.float64)
        elevation = np.asarray(points["elevation"], dtype=np.float64)
        orbit = np.asarray(points["orbit"], dtype=np.int64)
        quality_valid = np.asarray(points["quality_valid"], dtype=bool)
    keep, filter_report = robust_point_filter(
        arrays["z0"], rows, columns, elevation
    )
    keep &= quality_valid
    assimilation = keep & np.isin(orbit, record["assimilation_orbits"])
    evaluation = keep & np.isin(orbit, record["evaluation_orbits"])
    if np.count_nonzero(assimilation) < 5 or np.count_nonzero(evaluation) < 5:
        raise ValueError(f"{record['tile_id']} has too few screened points")
    return {
        "image": np.asarray(arrays["image"], dtype=np.float32),
        "z0": np.asarray(arrays["z0"], dtype=np.float32),
        "dem": np.asarray(arrays["dem"], dtype=np.float32),
        "valid": np.asarray(arrays["valid_mask"], dtype=bool),
        "row": rows.astype(np.float32),
        "column": columns.astype(np.float32),
        "elevation": elevation.astype(np.float32),
        "orbit": orbit,
        "assimilation": assimilation,
        "evaluation": evaluation,
        "filter_report": np.asarray(json.dumps(filter_report)),
    }


def evaluate_gp_parameters(
    records: list[dict[str, Any]],
    data_root: Path,
    mola_root: Path,
    lengthscale_px: float,
    noise_m: float,
) -> dict[str, float]:
    tile_squared_errors = []
    wins = 0
    for record in records:
        case = load_raw_case(record, data_root, mola_root)
        context = case["assimilation"]
        evaluation = case["evaluation"]
        order = nested_spatial_order(
            case["row"][context], case["column"][context],
            int(hashlib.sha256(record["tile_id"].encode()).hexdigest()[:8], 16),
        )
        use = order[: min(PRIMARY_BUDGET, len(order))]
        cr = case["row"][context][use]
        cc = case["column"][context][use]
        base_context = sample_raster(case["z0"], cr, cc)
        residual = case["elevation"][context][use] - base_context
        correction, _ = gp_predict(
            cr, cc, residual, case["row"][evaluation], case["column"][evaluation],
            lengthscale_px=lengthscale_px, noise_m=noise_m,
        )
        baseline = sample_raster(
            case["z0"], case["row"][evaluation], case["column"][evaluation]
        )
        target = case["elevation"][evaluation]
        coarse_rmse = float(np.sqrt(np.mean((baseline - target) ** 2)))
        gp_rmse = float(np.sqrt(np.mean((baseline + correction - target) ** 2)))
        tile_squared_errors.append(float(np.mean((baseline + correction - target) ** 2)))
        wins += gp_rmse < coarse_rmse
    return {
        "pooled_rmse_m": float(np.sqrt(np.mean(tile_squared_errors))),
        "tile_win_fraction": wins / len(records),
    }


def command_prepare(args: argparse.Namespace) -> int:
    protocol = json.loads(args.protocol.read_text())
    data_root = args.data_root.resolve()
    mola_root = args.mola_root.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"case directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    validation_records = [r for r in protocol["tiles"] if r["role"] == "validation"]
    search = []
    for lengthscale in (12.0, 24.0, 48.0, 96.0):
        for noise in (1.0, 3.0, 8.0):
            metrics = evaluate_gp_parameters(
                validation_records, data_root, mola_root, lengthscale, noise
            )
            search.append({"lengthscale_px": lengthscale, "noise_m": noise, **metrics})
    selected = min(search, key=lambda value: value["pooled_rmse_m"])
    grid_row, grid_column = np.indices((504, 504), dtype=np.float32)
    flat_row = grid_row.ravel()
    flat_column = grid_column.ravel()
    receipts = []
    for record in protocol["tiles"]:
        case = load_raw_case(record, data_root, mola_root)
        context = case["assimilation"]
        order = nested_spatial_order(
            case["row"][context], case["column"][context],
            int(hashlib.sha256(record["tile_id"].encode()).hexdigest()[:8], 16),
        )
        payload: dict[str, np.ndarray] = {
            key: value for key, value in case.items() if key != "filter_report"
        }
        payload["filter_report"] = case["filter_report"]
        payload["assimilation_order"] = order.astype(np.int64)
        for budget in BUDGETS:
            use = order[: min(budget, len(order))]
            cr = case["row"][context][use]
            cc = case["column"][context][use]
            context_elevation = case["elevation"][context][use]
            residual = context_elevation - sample_raster(case["z0"], cr, cc)
            correction, std = gp_predict(
                cr, cc, residual, flat_row, flat_column,
                lengthscale_px=float(selected["lengthscale_px"]),
                noise_m=float(selected["noise_m"]), return_std=True,
            )
            mask = np.zeros((504, 504), dtype=bool)
            mask[
                np.clip(np.rint(cr).astype(int), 0, 503),
                np.clip(np.rint(cc).astype(int), 0, 503),
            ] = True
            payload[f"gp_k{budget}"] = (
                case["z0"] + correction.reshape(504, 504)
            ).astype(np.float32)
            payload[f"gp_std_k{budget}"] = std.reshape(504, 504).astype(np.float32)
            payload[f"distance_k{budget}"] = distance_transform_edt(~mask).astype(np.float32)
            payload[f"context_indices_k{budget}"] = np.flatnonzero(context)[use].astype(np.int64)
            dem_at_context = sample_raster(case["dem"], cr, cc)
            payload[f"datum_shift_k{budget}"] = np.asarray(
                np.median(context_elevation - dem_at_context), dtype=np.float32
            )
        case_path = output / f"{record['tile_id']}.npz"
        np.savez_compressed(case_path, **payload)
        receipts.append(
            {
                "tile_id": record["tile_id"],
                "region": record["region"],
                "role": record["role"],
                "case": case_path.name,
                "sha256": sha256(case_path),
                "screening": json.loads(str(case["filter_report"].item())),
                "assimilation_point_count": int(np.count_nonzero(case["assimilation"])),
                "evaluation_point_count": int(np.count_nonzero(case["evaluation"])),
            }
        )
        print(f"CASE {record['role']} {record['tile_id']}", flush=True)
    manifest = {
        "schema": "mars-npj-real-cases/v1",
        "protocol_sha256": sha256(args.protocol),
        "data_root": str(data_root),
        "mola_root": str(mola_root),
        "budgets": list(BUDGETS),
        "primary_budget": PRIMARY_BUDGET,
        "screening": {
            "floor_m": GROSS_ERROR_FLOOR_M,
            "mad_multiplier": GROSS_ERROR_MAD_MULTIPLIER,
        },
        "gp_validation_search": search,
        "gp_selected": selected,
        "cases": receipts,
    }
    atomic_json(output / "cases-manifest.json", manifest)
    print(json.dumps({"gp_selected": selected, "case_count": len(receipts)}, indent=2))
    return 0


@dataclass
class LoadedCase:
    tile_id: str
    region: str
    role: str
    arrays: dict[str, np.ndarray]


def load_cases(case_root: Path, roles: set[str]) -> list[LoadedCase]:
    manifest = json.loads((case_root / "cases-manifest.json").read_text())
    output = []
    for record in manifest["cases"]:
        if record["role"] not in roles:
            continue
        with np.load(case_root / record["case"], allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        output.append(LoadedCase(record["tile_id"], record["region"], record["role"], arrays))
    return output


def image_normalize(image: np.ndarray) -> np.ndarray:
    low, high = np.quantile(image[np.isfinite(image)], (0.02, 0.98))
    return np.clip((image - low) / max(high - low, 1.0e-6), 0.0, 1.0).astype(np.float32)


def illumination_invariant_features(image: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return local, scale-normalized structure channels with weak gain sensitivity."""
    image = image.astype(np.float32)
    mean_small = gaussian_filter(image, sigma=2.0)
    mean_large = gaussian_filter(image, sigma=8.0)
    variance_small = gaussian_filter((image - mean_small) ** 2, sigma=2.0)
    variance_large = gaussian_filter((image - mean_large) ** 2, sigma=8.0)
    contrast_small = (image - mean_small) / np.sqrt(np.maximum(variance_small, 1.0e-4))
    contrast_large = (image - mean_large) / np.sqrt(np.maximum(variance_large, 1.0e-4))
    gradient_y, gradient_x = np.gradient(mean_small)
    gradient = np.hypot(gradient_x, gradient_y)
    gradient_scale = float(np.quantile(gradient[valid], 0.95)) if np.any(valid) else 1.0
    difference_of_scales = (mean_small - mean_large) / np.sqrt(
        np.maximum(variance_large, 1.0e-4)
    )
    return (
        np.clip(contrast_small / 3.0, -1.0, 1.0).astype(np.float32),
        np.clip(contrast_large / 3.0, -1.0, 1.0).astype(np.float32),
        np.clip(gradient / max(gradient_scale, 1.0e-4), 0.0, 2.0).astype(np.float32),
        np.clip(difference_of_scales / 3.0, -1.0, 1.0).astype(np.float32),
    )


def condition_array_from_fields(
    case: LoadedCase,
    gp: np.ndarray,
    gp_std: np.ndarray,
    distance: np.ndarray,
) -> np.ndarray:
    arrays = case.arrays
    z0 = arrays["z0"].astype(np.float32)
    gp = gp.astype(np.float32)
    image = image_normalize(arrays["image"])
    invariant = illumination_invariant_features(image, arrays["valid"])
    center = float(np.median(z0[arrays["valid"]]))
    gy, gx = np.gradient(z0, 20.0)
    slope = np.hypot(gx, gy).astype(np.float32)
    condition = np.stack(
        (
            image * 2.0 - 1.0,
            np.clip((z0 - center) / 250.0, -4.0, 4.0),
            np.clip((gp - z0) / HEIGHT_SCALE_M, -6.0, 6.0),
            np.clip(gp_std / HEIGHT_SCALE_M, 0.0, 4.0),
            np.clip(distance / 100.0, 0.0, 4.0),
            np.clip(slope / 0.5, 0.0, 4.0),
            *invariant,
        ),
        axis=0,
    ).astype(np.float32)
    return condition


def condition_array(case: LoadedCase, budget: int) -> tuple[np.ndarray, np.ndarray]:
    arrays = case.arrays
    gp = arrays[f"gp_k{budget}"].astype(np.float32)
    condition = condition_array_from_fields(
        case,
        gp,
        arrays[f"gp_std_k{budget}"],
        arrays[f"distance_k{budget}"],
    )
    return condition, gp


def point_maps(case: LoadedCase, budget: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = case.arrays
    target_map = np.zeros((504, 504), dtype=np.float32)
    target_count = np.zeros((504, 504), dtype=np.float32)
    context_map = np.zeros((504, 504), dtype=np.float32)
    context_count = np.zeros((504, 504), dtype=np.float32)
    for indices, value_map, count_map in (
        (np.flatnonzero(arrays["evaluation"]), target_map, target_count),
        (arrays[f"context_indices_k{budget}"], context_map, context_count),
    ):
        rr = np.clip(np.rint(arrays["row"][indices]).astype(int), 0, 503)
        cc = np.clip(np.rint(arrays["column"][indices]).astype(int), 0, 503)
        np.add.at(value_map, (rr, cc), arrays["elevation"][indices])
        np.add.at(count_map, (rr, cc), 1.0)
    target_mask = target_count > 0
    context_mask = context_count > 0
    target_map[target_mask] /= target_count[target_mask]
    context_map[context_mask] /= context_count[context_mask]
    return target_map, target_mask, context_map, context_mask


def torch_components() -> tuple[Any, Any, Any]:
    import torch
    from torch import nn
    from torch.nn import functional as functional
    return torch, nn, functional


def make_model(base_channels: int = 32, steps: int = 3, minimum_mobility: float = 0.20) -> Any:
    torch, nn, functional = torch_components()

    class Block(nn.Module):
        def __init__(self, input_channels: int, output_channels: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, padding=1),
                nn.GroupNorm(8, output_channels),
                nn.SiLU(),
                nn.Conv2d(output_channels, output_channels, 3, padding=1),
                nn.GroupNorm(8, output_channels),
                nn.SiLU(),
            )

        def forward(self, value: Any) -> Any:
            return self.net(value)

    class VelocityUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            b = base_channels
            self.e1 = Block(CONDITION_CHANNELS + 2, b)
            self.e2 = Block(b, 2 * b)
            self.e3 = Block(2 * b, 4 * b)
            self.middle = Block(4 * b, 4 * b)
            self.d2 = Block(6 * b, 2 * b)
            self.d1 = Block(3 * b, b)
            self.out = nn.Conv2d(b, 1, 3, padding=1)

        def forward(self, value: Any) -> Any:
            e1 = self.e1(value)
            e2 = self.e2(functional.avg_pool2d(e1, 2))
            e3 = self.e3(functional.avg_pool2d(e2, 2))
            middle = self.middle(e3)
            d2 = functional.interpolate(middle, size=e2.shape[-2:], mode="bilinear", align_corners=False)
            d2 = self.d2(torch.cat((d2, e2), dim=1))
            d1 = functional.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
            d1 = self.d1(torch.cat((d1, e1), dim=1))
            return self.out(d1)

    class TransportRefiner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.velocity = VelocityUNet()
            self.steps = steps

        def forward(self, condition: Any) -> tuple[Any, Any]:
            state = torch.zeros_like(condition[:, :1])
            kinetic = torch.zeros((), device=condition.device)
            mobility = minimum_mobility + (1.0 - minimum_mobility) * torch.clamp(
                condition[:, 3:4] / 2.0, 0.0, 1.0
            )
            for step in range(self.steps):
                time_channel = torch.full_like(state, (step + 0.5) / self.steps)
                velocity = self.velocity(torch.cat((condition, state, time_channel), dim=1))
                state = state + mobility * velocity / self.steps
                kinetic = kinetic + torch.mean(velocity**2) / self.steps
            return state, kinetic

    return TransportRefiner()


def crop_training_sample(
    case: LoadedCase, budget: int, crop_size: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    condition, gp = condition_array(case, budget)
    target_map, target_mask, context_map, context_mask = point_maps(case, budget)
    target_indices = np.flatnonzero(target_mask)
    if target_indices.size:
        selected = int(rng.choice(target_indices))
        target_row, target_column = np.unravel_index(selected, target_mask.shape)
    else:
        target_row = int(rng.integers(0, 504))
        target_column = int(rng.integers(0, 504))
    top = int(np.clip(target_row - rng.integers(crop_size // 4, 3 * crop_size // 4), 0, 504 - crop_size))
    left = int(np.clip(target_column - rng.integers(crop_size // 4, 3 * crop_size // 4), 0, 504 - crop_size))
    selection = np.s_[top : top + crop_size, left : left + crop_size]
    datum_shift = float(case.arrays[f"datum_shift_k{budget}"])
    dense_target = case.arrays["dem"].astype(np.float32) + datum_shift
    output = {
        "condition": condition[:, selection[0], selection[1]],
        "gp": gp[selection],
        "dense_target": dense_target[selection],
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


def predict_from_fields(
    model: Any,
    case: LoadedCase,
    gp: np.ndarray,
    gp_std: np.ndarray,
    distance: np.ndarray,
    context_indices: np.ndarray,
    device: Any,
) -> np.ndarray:
    torch, _, _ = torch_components()
    condition = condition_array_from_fields(case, gp, gp_std, distance)
    with torch.inference_mode():
        tensor = torch.from_numpy(condition[None]).to(device)
        residual, _ = model(tensor)
        correction = residual[0, 0].float().cpu().numpy() * HEIGHT_SCALE_M
    prediction = gp + correction
    if DISABLE_TERMINAL:
        return prediction
    context_rows = case.arrays["row"][context_indices]
    context_columns = case.arrays["column"][context_indices]
    remaining = case.arrays["elevation"][context_indices] - sample_raster(
        prediction, context_rows, context_columns
    )
    grid_row, grid_column = np.indices(prediction.shape, dtype=np.float32)
    terminal, _ = gp_predict(
        context_rows,
        context_columns,
        remaining,
        grid_row.ravel(),
        grid_column.ravel(),
        lengthscale_px=TERMINAL_LENGTHSCALE_PX,
        noise_m=8.0,
    )
    return prediction + terminal.reshape(prediction.shape).astype(np.float32)


def predict_case(model: Any, case: LoadedCase, budget: int, device: Any) -> np.ndarray:
    arrays = case.arrays
    return predict_from_fields(
        model,
        case,
        arrays[f"gp_k{budget}"],
        arrays[f"gp_std_k{budget}"],
        arrays[f"distance_k{budget}"],
        arrays[f"context_indices_k{budget}"],
        device,
    )


def build_gp_fields(
    case: LoadedCase,
    context_indices: np.ndarray,
    lengthscale_px: float = 48.0,
    noise_m: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = case.arrays
    rows = arrays["row"][context_indices]
    columns = arrays["column"][context_indices]
    residual = arrays["elevation"][context_indices] - sample_raster(
        arrays["z0"], rows, columns
    )
    grid_row, grid_column = np.indices(arrays["z0"].shape, dtype=np.float32)
    correction, std = gp_predict(
        rows,
        columns,
        residual,
        grid_row.ravel(),
        grid_column.ravel(),
        lengthscale_px=lengthscale_px,
        noise_m=noise_m,
        return_std=True,
    )
    mask = np.zeros(arrays["z0"].shape, dtype=bool)
    mask[
        np.clip(np.rint(rows).astype(int), 0, mask.shape[0] - 1),
        np.clip(np.rint(columns).astype(int), 0, mask.shape[1] - 1),
    ] = True
    gp = arrays["z0"] + correction.reshape(arrays["z0"].shape)
    return (
        gp.astype(np.float32),
        std.reshape(arrays["z0"].shape).astype(np.float32),
        distance_transform_edt(~mask).astype(np.float32),
    )


def crossfit_gate_alpha(
    model: Any,
    case: LoadedCase,
    budget: int,
    device: Any,
    folds: int = 4,
    shrinkage: float = 1.0,
) -> tuple[float, dict[str, Any]]:
    context = np.asarray(case.arrays[f"context_indices_k{budget}"], dtype=np.int64)
    if len(context) < max(8, 2 * folds):
        return 0.0, {"reason": "insufficient_context", "point_count": int(len(context))}
    numerator = 0.0
    denominator = 0.0
    fold_rows = []
    calibration_points = []
    for fold in range(folds):
        calibration = context[np.arange(len(context)) % folds == fold]
        fitting = context[np.arange(len(context)) % folds != fold]
        gp, gp_std, distance = build_gp_fields(case, fitting)
        prediction = predict_from_fields(
            model, case, gp, gp_std, distance, fitting, device
        )
        rows = case.arrays["row"][calibration]
        columns = case.arrays["column"][calibration]
        target = case.arrays["elevation"][calibration]
        baseline = sample_raster(gp, rows, columns)
        proposed = sample_raster(prediction, rows, columns)
        direction = proposed - baseline
        point_std = sample_raster(gp_std, rows, columns)
        sorted_std = np.sort(gp_std[case.arrays["valid"]].ravel())
        std_quantile = np.searchsorted(sorted_std, point_std, side="right") / max(
            len(sorted_std), 1
        )
        numerator += float(np.dot(direction, target - baseline))
        denominator += float(np.dot(direction, direction))
        for point_direction, point_error, point_quantile in zip(
            direction, target - baseline, std_quantile
        ):
            calibration_points.append(
                {
                    "direction_m": float(point_direction),
                    "target_residual_m": float(point_error),
                    "std_quantile": float(point_quantile),
                }
            )
        fold_rows.append(
            {
                "fold": fold,
                "fit_count": int(len(fitting)),
                "calibration_count": int(len(calibration)),
                "gp_rmse_m": float(np.sqrt(np.mean((baseline - target) ** 2))),
                "model_rmse_m": float(np.sqrt(np.mean((proposed - target) ** 2))),
            }
        )
    raw_alpha = 0.0 if denominator <= 1.0e-9 else numerator / denominator
    alpha = float(np.clip(raw_alpha * shrinkage, 0.0, 1.0))
    return alpha, {
        "raw_alpha": float(raw_alpha),
        "alpha": alpha,
        "numerator": numerator,
        "denominator": denominator,
        "folds": fold_rows,
        "calibration_points": calibration_points,
    }


def weighted_prediction_from_fields(
    models: list[Any],
    weights: np.ndarray,
    case: LoadedCase,
    gp: np.ndarray,
    gp_std: np.ndarray,
    distance: np.ndarray,
    context_indices: np.ndarray,
    device: Any,
) -> np.ndarray:
    predictions = [
        predict_from_fields(
            model, case, gp, gp_std, distance, context_indices, device
        )
        for model in models
    ]
    return np.sum(
        np.stack(predictions) * weights[:, None, None], axis=0
    ).astype(np.float32)


def crossfit_ensemble_alpha(
    models: list[Any],
    weights: np.ndarray,
    case: LoadedCase,
    budget: int,
    device: Any,
    folds: int,
) -> tuple[float, list[dict[str, Any]]]:
    context = np.asarray(case.arrays[f"context_indices_k{budget}"], dtype=np.int64)
    numerator = 0.0
    denominator = 0.0
    fold_rows = []
    calibration_points = []
    for fold in range(folds):
        calibration = context[np.arange(len(context)) % folds == fold]
        fitting = context[np.arange(len(context)) % folds != fold]
        gp, gp_std, distance = build_gp_fields(case, fitting)
        prediction = weighted_prediction_from_fields(
            models, weights, case, gp, gp_std, distance, fitting, device
        )
        rows = case.arrays["row"][calibration]
        columns = case.arrays["column"][calibration]
        target = case.arrays["elevation"][calibration]
        baseline = sample_raster(gp, rows, columns)
        proposed = sample_raster(prediction, rows, columns)
        direction = proposed - baseline
        point_std = sample_raster(gp_std, rows, columns)
        sorted_std = np.sort(gp_std[case.arrays["valid"]].ravel())
        std_quantile = np.searchsorted(sorted_std, point_std, side="right") / max(
            len(sorted_std), 1
        )
        numerator += float(np.dot(direction, target - baseline))
        denominator += float(np.dot(direction, direction))
        for point_direction, point_error, point_quantile in zip(
            direction, target - baseline, std_quantile
        ):
            calibration_points.append(
                {
                    "direction_m": float(point_direction),
                    "target_residual_m": float(point_error),
                    "std_quantile": float(point_quantile),
                }
            )
        fold_rows.append(
            {
                "fold": fold,
                "gp_rmse_m": float(np.sqrt(np.mean((baseline - target) ** 2))),
                "model_rmse_m": float(np.sqrt(np.mean((proposed - target) ** 2))),
            }
        )
    raw_alpha = 0.0 if denominator <= 1.0e-9 else numerator / denominator
    return float(raw_alpha), {"folds": fold_rows, "calibration_points": calibration_points}


def binned_crossfit_alphas(
    calibration: dict[str, Any], quantile: float, shrinkage: float
) -> tuple[float, float]:
    points = calibration.get("calibration_points", [])
    output = []
    for high in (False, True):
        selected = [
            point
            for point in points
            if (point["std_quantile"] >= quantile) == high
        ]
        numerator = sum(
            point["direction_m"] * point["target_residual_m"] for point in selected
        )
        denominator = sum(point["direction_m"] ** 2 for point in selected)
        raw = 0.0 if denominator <= 1.0e-9 else numerator / denominator
        output.append(float(np.clip(raw * shrinkage, 0.0, 1.0)))
    return output[0], output[1]


def real_mola_metrics(prediction: np.ndarray, case: LoadedCase) -> dict[str, float]:
    index = case.arrays["evaluation"]
    predicted = sample_raster(
        prediction, case.arrays["row"][index], case.arrays["column"][index]
    )
    target = case.arrays["elevation"][index]
    error = predicted - target
    return {
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "mae_m": float(np.mean(np.abs(error))),
        "bias_m": float(np.mean(error)),
        "point_count": int(target.size),
    }


def validate(model: Any, cases: list[LoadedCase], device: Any, budget: int = 20) -> dict[str, Any]:
    rows = []
    for case in cases:
        prediction = predict_case(model, case, budget, device)
        proposed = real_mola_metrics(prediction, case)
        gp = real_mola_metrics(case.arrays[f"gp_k{budget}"], case)
        rows.append(
            {
                "tile_id": case.tile_id,
                "proposed_rmse_m": proposed["rmse_m"],
                "gp_rmse_m": gp["rmse_m"],
            }
        )
    proposed_rmse = float(np.sqrt(np.mean([row["proposed_rmse_m"] ** 2 for row in rows])))
    gp_rmse = float(np.sqrt(np.mean([row["gp_rmse_m"] ** 2 for row in rows])))
    return {
        "proposed_rmse_m": proposed_rmse,
        "gp_rmse_m": gp_rmse,
        "relative_gain": (gp_rmse - proposed_rmse) / gp_rmse,
        "tile_win_fraction": float(np.mean([row["proposed_rmse_m"] < row["gp_rmse_m"] for row in rows])),
        "tiles": rows,
    }


def command_train(args: argparse.Namespace) -> int:
    torch, _, functional = torch_components()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    train_cases = load_cases(args.cases, {"train"})
    validation_cases = load_cases(args.cases, {"validation"})
    if args.training_region:
        requested_regions = set(args.training_region)
        train_cases = [case for case in train_cases if case.region in requested_regions]
        missing_regions = requested_regions - {case.region for case in train_cases}
        if missing_regions:
            raise ValueError(f"requested training regions are unavailable: {sorted(missing_regions)}")
    train_by_region: dict[str, list[LoadedCase]] = {}
    for case in train_cases:
        train_by_region.setdefault(case.region, []).append(case)
    train_regions = sorted(train_by_region)
    model = make_model(
        args.base_channels, args.transport_steps, args.minimum_mobility
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.scheduler_updates or args.updates,
        eta_min=args.learning_rate * 0.02,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    rng = np.random.default_rng(seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    best_gain = -math.inf
    best_update = 0
    history = []
    started = time.time()
    model.train()
    for update in range(1, args.updates + 1):
        samples = []
        for _ in range(args.batch_size):
            if args.region_balanced:
                region = train_regions[int(rng.integers(len(train_regions)))]
                members = train_by_region[region]
                case = members[int(rng.integers(len(members)))]
            else:
                case = train_cases[int(rng.integers(len(train_cases)))]
            training_budgets = args.training_budget or list(BUDGETS[:-1])
            budget = int(rng.choice(np.asarray(training_budgets)))
            samples.append(crop_training_sample(case, budget, args.crop_size, rng))
        def stack(name: str, dtype: Any = torch.float32) -> Any:
            return torch.from_numpy(np.stack([sample[name] for sample in samples])).to(device=device, dtype=dtype)
        condition = stack("condition")
        gp = stack("gp")
        dense_target = stack("dense_target")
        valid = stack("valid", torch.bool)
        target_map = stack("target_map")
        target_mask = stack("target_mask", torch.bool)
        context_map = stack("context_map")
        context_mask = stack("context_mask", torch.bool)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp):
            residual, kinetic = model(condition)
            prediction = gp[:, None] + residual * HEIGHT_SCALE_M
            dense_error = (prediction[:, 0] - dense_target) / HEIGHT_SCALE_M
            dense_loss = functional.smooth_l1_loss(dense_error[valid], torch.zeros_like(dense_error[valid]))
            sparse_error = (prediction[:, 0] - target_map) / HEIGHT_SCALE_M
            sparse_loss = functional.smooth_l1_loss(
                sparse_error[target_mask], torch.zeros_like(sparse_error[target_mask])
            ) if torch.any(target_mask) else torch.zeros((), device=device)
            context_error = (prediction[:, 0] - context_map) / HEIGHT_SCALE_M
            context_loss = functional.smooth_l1_loss(
                context_error[context_mask], torch.zeros_like(context_error[context_mask])
            ) if torch.any(context_mask) else torch.zeros((), device=device)
            pred_dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
            pred_dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
            target_dx = dense_target[:, None, :, 1:] - dense_target[:, None, :, :-1]
            target_dy = dense_target[:, None, 1:, :] - dense_target[:, None, :-1, :]
            gradient_loss = (
                functional.smooth_l1_loss(pred_dx / 10.0, target_dx / 10.0)
                + functional.smooth_l1_loss(pred_dy / 10.0, target_dy / 10.0)
            )
            trust_scale = torch.clamp(condition[:, 3:4], min=0.10)
            trust_loss = torch.mean((residual / trust_scale) ** 2)
            loss = (
                args.dense_weight * dense_loss
                + args.sparse_weight * sparse_loss
                + args.context_weight * context_loss
                + args.gradient_weight * gradient_loss
                + args.kinetic_weight * kinetic
                + args.trust_weight * trust_loss
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if update % args.log_interval == 0:
            elapsed = time.time() - started
            progress = {
                "update": update,
                "loss": float(loss.detach().cpu()),
                "dense_loss": float(dense_loss.detach().cpu()),
                "sparse_loss": float(sparse_loss.detach().cpu()),
                "context_loss": float(context_loss.detach().cpu()),
                "gradient_loss": float(gradient_loss.detach().cpu()),
                "trust_loss": float(trust_loss.detach().cpu()),
                "learning_rate": scheduler.get_last_lr()[0],
                "updates_per_second": update / elapsed,
                "eta_seconds": (args.updates - update) / max(update / elapsed, 1.0e-9),
                "best_validation_gain": best_gain,
                "best_update": best_update,
            }
            atomic_json(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
        if update % args.validation_interval == 0 or update == args.updates:
            model.eval()
            metrics = validate(
                model, validation_cases, device, budget=args.validation_budget
            )
            metrics["update"] = update
            history.append(metrics)
            print("VALIDATION " + json.dumps(metrics), flush=True)
            if metrics["relative_gain"] > best_gain:
                best_gain = float(metrics["relative_gain"])
                best_update = update
                torch.save(
                    {
                        "model": model.state_dict(),
                        "seed": seed,
                        "args": {
                            key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()
                            if key not in {"function", "command"}
                        },
                        "validation": metrics,
                    },
                    output / "best.pt",
                )
            atomic_json(output / "validation-history.json", {"history": history})
            model.train()
        if update in set(args.snapshot_update):
            torch.save(
                {
                    "model": model.state_dict(),
                    "seed": seed,
                    "update": update,
                    "args": {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                        if key not in {"function", "command"}
                    },
                },
                output / f"snapshot-{update}.pt",
            )
    manifest = {
        "state": "complete",
        "seed": seed,
        "completed_updates": args.updates,
        "best_update": best_update,
        "best_validation_gain": best_gain,
        "best_checkpoint_sha256": sha256(output / "best.pt"),
        "history": history,
    }
    atomic_json(output / "training-manifest.json", manifest)
    return 0


def terrain_safety(dem: np.ndarray) -> np.ndarray:
    gradient_y, gradient_x = np.gradient(dem.astype(np.float64), 20.0)
    slope = np.degrees(np.arctan(np.hypot(gradient_x, gradient_y)))
    roughness = np.abs(dem - gaussian_filter(dem, sigma=2.0))
    return (slope < 15.0) & (roughness < 2.0)


def safety_metrics(prediction: np.ndarray, reference: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    predicted_safe = terrain_safety(prediction) & valid
    reference_safe = terrain_safety(reference) & valid
    tp = np.count_nonzero(predicted_safe & reference_safe)
    fp = np.count_nonzero(predicted_safe & ~reference_safe & valid)
    fn = np.count_nonzero(~predicted_safe & reference_safe & valid)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    false_safe = fp / max(np.count_nonzero(predicted_safe), 1)
    return {"safe_f1": float(f1), "false_safe_rate": float(false_safe)}


def bootstrap_summary(rows: list[dict[str, Any]], repeats: int = 10000, seed: int = 20260808) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    proposed = np.asarray([row["proposed_rmse_m"] for row in rows])
    gp = np.asarray([row["gp_rmse_m"] for row in rows])
    gains = []
    win_fractions = []
    for _ in range(repeats):
        index = rng.integers(0, len(rows), len(rows))
        p = float(np.sqrt(np.mean(proposed[index] ** 2)))
        g = float(np.sqrt(np.mean(gp[index] ** 2)))
        gains.append((g - p) / g)
        win_fractions.append(float(np.mean(proposed[index] < gp[index])))
    return {
        "relative_rmse_gain": float((np.sqrt(np.mean(gp**2)) - np.sqrt(np.mean(proposed**2))) / np.sqrt(np.mean(gp**2))),
        "relative_gain_ci95": [float(value) for value in np.quantile(gains, (0.025, 0.975))],
        "tile_win_fraction": float(np.mean(proposed < gp)),
        "tile_win_fraction_ci95": [float(value) for value in np.quantile(win_fractions, (0.025, 0.975))],
    }


def load_ensemble(checkpoints: list[Path], device: Any) -> list[Any]:
    torch, _, _ = torch_components()
    models = []
    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        args = payload["args"]
        model = make_model(
            int(args["base_channels"]),
            int(args["transport_steps"]),
            float(args.get("minimum_mobility", 0.20)),
        )
        model.load_state_dict(payload["model"])
        model.to(device).eval()
        models.append(model)
    return models


def command_validate_checkpoint(args: argparse.Namespace) -> int:
    torch, _, _ = torch_components()
    device = torch.device(args.device)
    model = load_ensemble([args.checkpoint], device)[0]
    cases = load_cases(args.cases, {"validation"})
    metrics = validate(model, cases, device, budget=args.budget)
    print(json.dumps(metrics, indent=2))
    return 0


def command_validate_crossfit(args: argparse.Namespace) -> int:
    torch, _, _ = torch_components()
    device = torch.device(args.device)
    model = load_ensemble([args.checkpoint], device)[0]
    cases = load_cases(args.cases, {"validation"})
    stored = []
    for case in cases:
        full_prediction = predict_case(model, case, args.budget, device)
        _, calibration = crossfit_gate_alpha(
            model,
            case,
            args.budget,
            device,
            folds=args.folds,
            shrinkage=1.0,
        )
        stored.append((case, full_prediction, calibration))
    candidates = []
    quantiles: list[float | None] = [None] + list(args.spatial_quantile)
    for spatial_quantile in quantiles:
        for shrinkage in args.shrinkage:
            rows = []
            for case, full_prediction, calibration in stored:
                gp = case.arrays[f"gp_k{args.budget}"]
                if spatial_quantile is None:
                    alpha = float(
                        np.clip(calibration.get("raw_alpha", 0.0) * shrinkage, 0.0, 1.0)
                    )
                    alpha_low = alpha_high = alpha
                    prediction = gp + alpha * (full_prediction - gp)
                else:
                    alpha_low, alpha_high = binned_crossfit_alphas(
                        calibration, spatial_quantile, shrinkage
                    )
                    std = case.arrays[f"gp_std_k{args.budget}"]
                    threshold = float(
                        np.quantile(std[case.arrays["valid"]], spatial_quantile)
                    )
                    alpha_map = np.where(std >= threshold, alpha_high, alpha_low)
                    prediction = gp + alpha_map * (full_prediction - gp)
                    alpha = float(np.mean(alpha_map[case.arrays["valid"]]))
                proposed = real_mola_metrics(prediction, case)
                baseline = real_mola_metrics(gp, case)
                rows.append(
                    {
                        "tile_id": case.tile_id,
                        "alpha": alpha,
                        "alpha_low": alpha_low,
                        "alpha_high": alpha_high,
                        "proposed_rmse_m": proposed["rmse_m"],
                        "gp_rmse_m": baseline["rmse_m"],
                        "calibration": calibration,
                    }
                )
            proposed_rmse = float(
                np.sqrt(np.mean([row["proposed_rmse_m"] ** 2 for row in rows]))
            )
            gp_rmse = float(np.sqrt(np.mean([row["gp_rmse_m"] ** 2 for row in rows])))
            candidates.append(
                {
                    "spatial_quantile": spatial_quantile,
                    "shrinkage": float(shrinkage),
                    "relative_gain": (gp_rmse - proposed_rmse) / gp_rmse,
                    "tile_win_fraction": float(
                        np.mean([row["proposed_rmse_m"] < row["gp_rmse_m"] for row in rows])
                    ),
                    "proposed_rmse_m": proposed_rmse,
                    "gp_rmse_m": gp_rmse,
                    "rows": rows,
                }
            )
    payload = {
        "schema": "mars-npj-crossfit-gate-validation/v1",
        "checkpoint_sha256": sha256(args.checkpoint),
        "budget": args.budget,
        "folds": args.folds,
        "candidates": candidates,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


def command_validate_crossfit_ensemble(args: argparse.Namespace) -> int:
    torch, _, _ = torch_components()
    device = torch.device(args.device)
    models = load_ensemble(args.checkpoint, device)
    weights = np.asarray(args.weight, dtype=np.float64)
    if len(weights) != len(models) or np.any(weights < 0) or np.sum(weights) <= 0:
        raise ValueError("one non-negative --weight is required per checkpoint")
    weights /= np.sum(weights)
    cases = load_cases(args.cases, {"validation"})
    stored = []
    for case in cases:
        arrays = case.arrays
        full_prediction = weighted_prediction_from_fields(
            models,
            weights,
            case,
            arrays[f"gp_k{args.budget}"],
            arrays[f"gp_std_k{args.budget}"],
            arrays[f"distance_k{args.budget}"],
            arrays[f"context_indices_k{args.budget}"],
            device,
        )
        raw_alpha, calibration = crossfit_ensemble_alpha(
            models, weights, case, args.budget, device, args.folds
        )
        stored.append((case, full_prediction, raw_alpha, calibration))
    candidates = []
    for spatial_quantile in [None] + list(args.spatial_quantile):
        for shrinkage in args.shrinkage:
            rows = []
            for case, full_prediction, raw_alpha, calibration in stored:
                gp = case.arrays[f"gp_k{args.budget}"]
                if spatial_quantile is None:
                    alpha_low = alpha_high = float(
                        np.clip(raw_alpha * shrinkage, 0.0, 1.0)
                    )
                    prediction = gp + alpha_low * (full_prediction - gp)
                else:
                    alpha_low, alpha_high = binned_crossfit_alphas(
                        calibration, spatial_quantile, shrinkage
                    )
                    std = case.arrays[f"gp_std_k{args.budget}"]
                    threshold = float(
                        np.quantile(std[case.arrays["valid"]], spatial_quantile)
                    )
                    alpha_map = np.where(std >= threshold, alpha_high, alpha_low)
                    prediction = gp + alpha_map * (full_prediction - gp)
                proposed = real_mola_metrics(prediction, case)
                baseline = real_mola_metrics(gp, case)
                rows.append(
                    {
                        "tile_id": case.tile_id,
                        "raw_alpha": raw_alpha,
                        "alpha_low": alpha_low,
                        "alpha_high": alpha_high,
                        "proposed_rmse_m": proposed["rmse_m"],
                        "gp_rmse_m": baseline["rmse_m"],
                        "calibration": calibration,
                    }
                )
            proposed_rmse = float(
                np.sqrt(np.mean([row["proposed_rmse_m"] ** 2 for row in rows]))
            )
            gp_rmse = float(np.sqrt(np.mean([row["gp_rmse_m"] ** 2 for row in rows])))
            bootstrap = bootstrap_summary(rows)
            candidates.append(
                {
                    "spatial_quantile": spatial_quantile,
                    "shrinkage": shrinkage,
                    "relative_gain": (gp_rmse - proposed_rmse) / gp_rmse,
                    "tile_win_fraction": float(
                        np.mean([row["proposed_rmse_m"] < row["gp_rmse_m"] for row in rows])
                    ),
                    "bootstrap": bootstrap,
                    "rows": rows,
                }
            )
    payload = {
        "schema": "mars-npj-crossfit-ensemble-validation/v1",
        "weights": weights.tolist(),
        "folds": args.folds,
        "candidates": candidates,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


def gate_features(case: LoadedCase, prediction: np.ndarray, budget: int) -> np.ndarray:
    arrays = case.arrays
    gp = arrays[f"gp_k{budget}"]
    context = arrays[f"context_indices_k{budget}"]
    context_residual = arrays["elevation"][context] - sample_raster(
        arrays["z0"], arrays["row"][context], arrays["column"][context]
    )
    gradient_y, gradient_x = np.gradient(arrays["z0"].astype(np.float64), 20.0)
    correction = prediction - gp
    return np.asarray(
        [
            float(np.mean(arrays[f"gp_std_k{budget}"])) / HEIGHT_SCALE_M,
            float(np.median(arrays[f"distance_k{budget}"])) / 100.0,
            float(np.std(image_normalize(arrays["image"]))),
            float(np.mean(np.hypot(gradient_x, gradient_y))),
            float(1.4826 * np.median(np.abs(context_residual - np.median(context_residual)))) / HEIGHT_SCALE_M,
            float(np.sqrt(np.mean(correction**2))) / HEIGHT_SCALE_M,
            math.log1p(len(context)) / 5.0,
        ],
        dtype=np.float64,
    )


def optimal_gate_alpha(
    case: LoadedCase, prediction: np.ndarray, budget: int
) -> tuple[float, float, float]:
    arrays = case.arrays
    evaluation = arrays["evaluation"]
    rows = arrays["row"][evaluation]
    columns = arrays["column"][evaluation]
    target = arrays["elevation"][evaluation]
    gp = sample_raster(arrays[f"gp_k{budget}"], rows, columns)
    proposed = sample_raster(prediction, rows, columns)
    direction = proposed - gp
    denominator = float(np.dot(direction, direction))
    alpha = 0.0 if denominator <= 1.0e-9 else float(
        np.clip(np.dot(direction, target - gp) / denominator, 0.0, 1.25)
    )
    blended = gp + alpha * direction
    return (
        alpha,
        float(np.sqrt(np.mean((blended - target) ** 2))),
        float(np.sqrt(np.mean((gp - target) ** 2))),
    )


def fit_ridge_gate(features: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale[scale < 1.0e-6] = 1.0
    normalized = (features - mean) / scale
    design = np.column_stack((np.ones(len(normalized)), normalized))
    candidates = []
    for ridge in (0.01, 0.1, 1.0, 10.0, 100.0):
        predictions = []
        for held_out in range(len(design)):
            keep = np.arange(len(design)) != held_out
            penalty = np.eye(design.shape[1]) * ridge
            penalty[0, 0] = 0.0
            coefficient = np.linalg.solve(
                design[keep].T @ design[keep] + penalty,
                design[keep].T @ targets[keep],
            )
            predictions.append(float(np.clip(design[held_out] @ coefficient, 0.0, 1.25)))
        candidates.append(
            {
                "ridge": ridge,
                "loocv_mse": float(np.mean((np.asarray(predictions) - targets) ** 2)),
            }
        )
    selected = min(candidates, key=lambda value: value["loocv_mse"])
    penalty = np.eye(design.shape[1]) * float(selected["ridge"])
    penalty[0, 0] = 0.0
    coefficient = np.linalg.solve(
        design.T @ design + penalty, design.T @ targets
    )
    return {
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coefficient": coefficient.tolist(),
        "ridge_search": candidates,
        "selected_ridge": selected["ridge"],
    }


def apply_ridge_gate(calibration: dict[str, Any], features: np.ndarray) -> float:
    normalized = (
        features - np.asarray(calibration["feature_mean"])
    ) / np.asarray(calibration["feature_scale"])
    design = np.concatenate(([1.0], normalized))
    return float(np.clip(design @ np.asarray(calibration["coefficient"]), 0.0, 1.25))


def command_fit_gate(args: argparse.Namespace) -> int:
    torch, _, _ = torch_components()
    device = torch.device(args.device)
    model = load_ensemble([args.checkpoint], device)[0]
    train_cases = load_cases(args.cases, {"train"})
    validation_cases = load_cases(args.cases, {"validation"})
    training_rows = []
    for case in train_cases:
        prediction = predict_case(model, case, args.budget, device)
        alpha, rmse, gp_rmse = optimal_gate_alpha(case, prediction, args.budget)
        training_rows.append(
            {
                "tile_id": case.tile_id,
                "features": gate_features(case, prediction, args.budget).tolist(),
                "optimal_alpha": alpha,
                "optimal_rmse_m": rmse,
                "gp_rmse_m": gp_rmse,
            }
        )
    calibration = fit_ridge_gate(
        np.asarray([row["features"] for row in training_rows]),
        np.asarray([row["optimal_alpha"] for row in training_rows]),
    )
    validation_rows = []
    for case in validation_cases:
        prediction = predict_case(model, case, args.budget, device)
        features = gate_features(case, prediction, args.budget)
        alpha = apply_ridge_gate(calibration, features)
        arrays = case.arrays
        evaluation = arrays["evaluation"]
        rows = arrays["row"][evaluation]
        columns = arrays["column"][evaluation]
        target = arrays["elevation"][evaluation]
        gp = sample_raster(arrays[f"gp_k{args.budget}"], rows, columns)
        proposed = sample_raster(prediction, rows, columns)
        blended = gp + alpha * (proposed - gp)
        validation_rows.append(
            {
                "tile_id": case.tile_id,
                "alpha": alpha,
                "proposed_rmse_m": float(np.sqrt(np.mean((blended - target) ** 2))),
                "gp_rmse_m": float(np.sqrt(np.mean((gp - target) ** 2))),
            }
        )
    proposed_rmse = float(
        np.sqrt(np.mean([row["proposed_rmse_m"] ** 2 for row in validation_rows]))
    )
    gp_rmse = float(np.sqrt(np.mean([row["gp_rmse_m"] ** 2 for row in validation_rows])))
    payload = {
        "schema": "mars-npj-conservative-gate/v1",
        "checkpoint_sha256": sha256(args.checkpoint),
        "budget": args.budget,
        "calibration": calibration,
        "training_rows": training_rows,
        "validation": {
            "relative_gain": (gp_rmse - proposed_rmse) / gp_rmse,
            "tile_win_fraction": float(
                np.mean([row["proposed_rmse_m"] < row["gp_rmse_m"] for row in validation_rows])
            ),
            "proposed_rmse_m": proposed_rmse,
            "gp_rmse_m": gp_rmse,
            "rows": validation_rows,
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


def command_blend_checkpoints(args: argparse.Namespace) -> int:
    torch, _, _ = torch_components()
    device = torch.device(args.device)
    models = load_ensemble(args.checkpoint, device)
    if len(models) != 2:
        raise ValueError("blend-checkpoints requires exactly two checkpoints")
    cases = load_cases(args.cases, {"validation"})
    stored = []
    for case in cases:
        predictions = [predict_case(model, case, args.budget, device) for model in models]
        stored.append((case, predictions))
    candidates = []
    for weight in np.linspace(0.0, 1.0, 21):
        rows = []
        for case, predictions in stored:
            prediction = weight * predictions[0] + (1.0 - weight) * predictions[1]
            proposed = real_mola_metrics(prediction, case)
            gp = real_mola_metrics(case.arrays[f"gp_k{args.budget}"], case)
            rows.append(
                {
                    "tile_id": case.tile_id,
                    "proposed_rmse_m": proposed["rmse_m"],
                    "gp_rmse_m": gp["rmse_m"],
                }
            )
        proposed_rmse = float(np.sqrt(np.mean([row["proposed_rmse_m"] ** 2 for row in rows])))
        gp_rmse = float(np.sqrt(np.mean([row["gp_rmse_m"] ** 2 for row in rows])))
        candidates.append(
            {
                "first_checkpoint_weight": float(weight),
                "relative_gain": (gp_rmse - proposed_rmse) / gp_rmse,
                "tile_win_fraction": float(
                    np.mean([row["proposed_rmse_m"] < row["gp_rmse_m"] for row in rows])
                ),
                "rows": rows,
            }
        )
    payload = {
        "schema": "mars-npj-validation-blend/v1",
        "checkpoints": [str(path) for path in args.checkpoint],
        "candidates": candidates,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    torch, _, _ = torch_components()
    device = torch.device(args.device)
    models = load_ensemble(args.checkpoint, device)
    roles = {"locked_test"} if not args.include_external else {"locked_test", "external"}
    cases = load_cases(args.cases, roles)
    rows = []
    for case in cases:
        for budget in BUDGETS:
            predictions = [predict_case(model, case, budget, device) for model in models]
            prediction = np.mean(predictions, axis=0)
            proposed = real_mola_metrics(prediction, case)
            gp = real_mola_metrics(case.arrays[f"gp_k{budget}"], case)
            datum_shift = float(case.arrays[f"datum_shift_k{budget}"])
            reference = case.arrays["dem"] + datum_shift
            proposed_safety = safety_metrics(prediction, reference, case.arrays["valid"])
            gp_safety = safety_metrics(case.arrays[f"gp_k{budget}"], reference, case.arrays["valid"])
            rows.append(
                {
                    "tile_id": case.tile_id,
                    "region": case.region,
                    "role": case.role,
                    "budget": budget,
                    "proposed_rmse_m": proposed["rmse_m"],
                    "gp_rmse_m": gp["rmse_m"],
                    "proposed_safe_f1": proposed_safety["safe_f1"],
                    "gp_safe_f1": gp_safety["safe_f1"],
                    "proposed_false_safe_rate": proposed_safety["false_safe_rate"],
                    "gp_false_safe_rate": gp_safety["false_safe_rate"],
                }
            )
        print(f"EVALUATED {case.role} {case.tile_id}", flush=True)
    primary = [row for row in rows if row["budget"] == PRIMARY_BUDGET and row["role"] == "locked_test"]
    bootstrap = bootstrap_summary(primary)
    proposed_f1 = float(np.mean([row["proposed_safe_f1"] for row in primary]))
    gp_f1 = float(np.mean([row["gp_safe_f1"] for row in primary]))
    proposed_false = float(np.mean([row["proposed_false_safe_rate"] for row in primary]))
    gp_false = float(np.mean([row["gp_false_safe_rate"] for row in primary]))
    by_budget: dict[int, dict[str, float]] = {}
    for budget in BUDGETS:
        group = [row for row in rows if row["budget"] == budget and row["role"] == "locked_test"]
        by_budget[budget] = {
            "proposed_rmse_m": float(np.sqrt(np.mean([row["proposed_rmse_m"] ** 2 for row in group]))),
            "gp_rmse_m": float(np.sqrt(np.mean([row["gp_rmse_m"] ** 2 for row in group]))),
        }
    gp_target = by_budget[PRIMARY_BUDGET]["gp_rmse_m"]
    qualifying = [budget for budget in BUDGETS if by_budget[budget]["proposed_rmse_m"] <= gp_target]
    required_budget = min(qualifying) if qualifying else None
    saving = None if required_budget is None else 1.0 - required_budget / PRIMARY_BUDGET
    summary = {
        "schema": "mars-npj-real-evaluation/v1",
        "locked_test_primary": bootstrap,
        "laser_saving_relative_to_gp_k20": saving,
        "required_proposed_budget": required_budget,
        "downstream": {
            "safe_f1_gain_points": proposed_f1 - gp_f1,
            "false_safe_relative_reduction": (gp_false - proposed_false) / max(gp_false, 1.0e-9),
            "proposed_safe_f1": proposed_f1,
            "gp_safe_f1": gp_f1,
        },
        "budget_curves": by_budget,
        "goal_pass": {
            "real_mola": bootstrap["relative_rmse_gain"] >= 0.05
            and bootstrap["tile_win_fraction"] >= 0.70
            and bootstrap["relative_gain_ci95"][0] > 0.0,
            "laser_saving": saving is not None and saving >= 0.20,
            "downstream": (proposed_f1 - gp_f1) >= 0.05
            or (gp_false - proposed_false) / max(gp_false, 1.0e-9) >= 0.15,
        },
        "rows": rows,
    }
    summary["goal_pass"]["all"] = all(summary["goal_pass"].values())
    atomic_json(args.output, summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["goal_pass"]["all"] else 3


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    subparsers = root.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--protocol", type=Path, required=True)
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--mola-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.set_defaults(function=command_prepare)
    train = subparsers.add_parser("train")
    train.add_argument("--cases", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--seed", type=int, default=20260808)
    train.add_argument("--updates", type=int, default=5000)
    train.add_argument("--scheduler-updates", type=int, default=None)
    train.add_argument("--validation-interval", type=int, default=250)
    train.add_argument("--log-interval", type=int, default=25)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--crop-size", type=int, default=320)
    train.add_argument("--base-channels", type=int, default=32)
    train.add_argument("--transport-steps", type=int, default=3)
    train.add_argument("--learning-rate", type=float, default=2.0e-4)
    train.add_argument("--dense-weight", type=float, default=1.0)
    train.add_argument("--sparse-weight", type=float, default=3.0)
    train.add_argument("--context-weight", type=float, default=1.0)
    train.add_argument("--gradient-weight", type=float, default=0.25)
    train.add_argument("--kinetic-weight", type=float, default=0.005)
    train.add_argument("--trust-weight", type=float, default=0.0)
    train.add_argument("--minimum-mobility", type=float, default=0.20)
    train.add_argument("--gradient-clip", type=float, default=1.0)
    train.add_argument("--amp", action="store_true")
    train.add_argument("--snapshot-update", type=int, action="append", default=[])
    train.add_argument("--training-budget", type=int, action="append", default=[])
    train.add_argument("--validation-budget", type=int, default=20)
    train.add_argument("--region-balanced", action="store_true")
    train.add_argument("--training-region", action="append", default=[])
    train.set_defaults(function=command_train)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--cases", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, action="append", required=True)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--include-external", action="store_true")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.set_defaults(function=command_evaluate)
    validate_checkpoint = subparsers.add_parser("validate-checkpoint")
    validate_checkpoint.add_argument("--cases", type=Path, required=True)
    validate_checkpoint.add_argument("--checkpoint", type=Path, required=True)
    validate_checkpoint.add_argument("--device", default="cpu")
    validate_checkpoint.add_argument("--budget", type=int, default=20)
    validate_checkpoint.set_defaults(function=command_validate_checkpoint)
    crossfit = subparsers.add_parser("validate-crossfit")
    crossfit.add_argument("--cases", type=Path, required=True)
    crossfit.add_argument("--checkpoint", type=Path, required=True)
    crossfit.add_argument("--device", default="cpu")
    crossfit.add_argument("--budget", type=int, default=20)
    crossfit.add_argument("--folds", type=int, default=4)
    crossfit.add_argument(
        "--shrinkage", type=float, action="append", default=[]
    )
    crossfit.add_argument(
        "--spatial-quantile", type=float, action="append", default=[]
    )
    crossfit.add_argument("--output", type=Path, required=True)
    crossfit.set_defaults(function=command_validate_crossfit)
    crossfit_ensemble = subparsers.add_parser("validate-crossfit-ensemble")
    crossfit_ensemble.add_argument("--cases", type=Path, required=True)
    crossfit_ensemble.add_argument(
        "--checkpoint", type=Path, action="append", required=True
    )
    crossfit_ensemble.add_argument("--weight", type=float, action="append", required=True)
    crossfit_ensemble.add_argument("--device", default="cpu")
    crossfit_ensemble.add_argument("--budget", type=int, default=20)
    crossfit_ensemble.add_argument("--folds", type=int, default=4)
    crossfit_ensemble.add_argument(
        "--shrinkage", type=float, action="append", required=True
    )
    crossfit_ensemble.add_argument(
        "--spatial-quantile", type=float, action="append", default=[]
    )
    crossfit_ensemble.add_argument("--output", type=Path, required=True)
    crossfit_ensemble.set_defaults(function=command_validate_crossfit_ensemble)
    fit_gate = subparsers.add_parser("fit-gate")
    fit_gate.add_argument("--cases", type=Path, required=True)
    fit_gate.add_argument("--checkpoint", type=Path, required=True)
    fit_gate.add_argument("--device", default="cpu")
    fit_gate.add_argument("--budget", type=int, default=20)
    fit_gate.add_argument("--output", type=Path, required=True)
    fit_gate.set_defaults(function=command_fit_gate)
    blend = subparsers.add_parser("blend-checkpoints")
    blend.add_argument("--cases", type=Path, required=True)
    blend.add_argument("--checkpoint", type=Path, action="append", required=True)
    blend.add_argument("--device", default="cpu")
    blend.add_argument("--budget", type=int, default=20)
    blend.add_argument("--output", type=Path, required=True)
    blend.set_defaults(function=command_blend_checkpoints)
    return root


def main() -> int:
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())

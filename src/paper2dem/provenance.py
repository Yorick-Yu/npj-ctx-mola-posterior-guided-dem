from __future__ import annotations

from typing import Any

from .hashing import canonical_sha256


def mechanism_parameters_sha256(
    *,
    g_min: float,
    alpha: float,
    ddpm_cosine_offset: float,
    ddpm_eta: float,
    steps: int,
    paths: int,
    generative: bool,
) -> str:
    return canonical_sha256(
        {
            "g_min": float(g_min),
            "alpha": float(alpha),
            "ddpm_cosine_offset": float(ddpm_cosine_offset),
            "ddpm_eta": float(ddpm_eta),
            "steps": int(steps) if generative else 0,
            "paths": int(paths) if generative else 1,
        }
    )


def dc_parameters_sha256(
    *,
    mode: str,
    terminal_dc: bool,
    tau_dc: float,
    delta: float,
    lambda_dc: float,
    footprint_diameter_m: float,
    quadrature_points: int,
) -> str:
    return canonical_sha256(
        {
            "mode": str(mode) if terminal_dc else "D0",
            "tau_dc": float(tau_dc),
            "delta": float(delta),
            "lambda": float(lambda_dc),
            "footprint_diameter_m": float(footprint_diameter_m),
            "quadrature_points": int(quadrature_points),
        }
    )


def not_applicable_sha256(name: str, value: Any) -> str:
    return canonical_sha256({name: value})

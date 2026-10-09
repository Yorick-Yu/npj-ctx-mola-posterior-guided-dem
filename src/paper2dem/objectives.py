from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .models import TORCH_AVAILABLE, method_spec


@dataclass(frozen=True)
class MechanismFields:
    condition_omega: np.ndarray
    mobility: np.ndarray
    gate: np.ndarray


def mechanism_fields(
    method: str,
    omega: np.ndarray,
    *,
    g_min: float,
    alpha: float,
    shifted_omega: np.ndarray | None = None,
) -> MechanismFields:
    spec = method_spec(method)
    uncertainty = np.clip(np.asarray(omega, dtype=np.float64), 0.0, 1.0)
    if not 0 < g_min <= 1 or alpha <= 0:
        raise ValueError("g_min and alpha must satisfy 0<g_min<=1, alpha>0")
    real = g_min + (1.0 - g_min) * uncertainty**alpha
    trace_match = np.full_like(real, float(np.mean(real)))
    if method in {"Iso-SB", "Iso-SB-cond"}:
        mobility = np.ones_like(real)
        gate = np.ones_like(real)
    elif method == "spatial-shift-M":
        if shifted_omega is None:
            raise ValueError("spatial-shift-M requires a frozen shifted omega field")
        shifted = np.clip(np.asarray(shifted_omega, dtype=np.float64), 0.0, 1.0)
        if shifted.shape != uncertainty.shape:
            raise ValueError("shifted omega must share shape")
        mobility = g_min + (1.0 - g_min) * shifted**alpha
        gate = real
    else:
        mobility = real if spec.uses_spatial_mobility else trace_match
        gate = real if spec.uses_spatial_gate else trace_match
    condition = uncertainty if spec.uses_uncertainty_condition else np.ones_like(uncertainty)
    return MechanismFields(condition, mobility, gate)


def shifted_omega_field(omega: np.ndarray, *, tile_hash: str) -> np.ndarray:
    value = np.asarray(omega)
    if value.ndim != 2 or len(tile_hash) < 16:
        raise ValueError("omega must be 2-D and tile hash must be available")
    row_shift = 1 + int(tile_hash[:8], 16) % max(value.shape[0] - 1, 1)
    column_shift = 1 + int(tile_hash[8:16], 16) % max(value.shape[1] - 1, 1)
    return np.roll(value, shift=(row_shift, column_shift), axis=(0, 1))


def _require_torch() -> Any:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for bridge training/inference")
    import torch

    return torch


def masked_condition(condition: Any, method: str) -> Any:
    """Apply the registered method's information boundary to six condition planes."""

    torch = _require_torch()
    spec = method_spec(method)
    if condition.ndim != 4 or condition.shape[1] != 6:
        raise ValueError("condition must have shape (B,6,H,W)")
    keep = torch.tensor(
        [
            True,
            spec.uses_image,
            spec.uses_image,
            spec.uses_laser,
            spec.uses_uncertainty_condition,
            spec.uses_laser,
        ],
        device=condition.device,
        dtype=condition.dtype,
    ).reshape(1, 6, 1, 1)
    return condition * keep


def torch_mechanism_fields(
    method: str,
    omega: Any,
    *,
    g_min: float,
    alpha: float,
    shifted_omega: Any | None = None,
) -> tuple[Any, Any]:
    """Return torch mobility/gate fields with the same E4 semantics as ``mechanism_fields``."""

    _require_torch()
    spec = method_spec(method)
    if not 0 < g_min <= 1 or alpha <= 0:
        raise ValueError("g_min and alpha must satisfy 0<g_min<=1, alpha>0")
    real_g = g_min + (1.0 - g_min) * omega.clamp(0, 1).pow(alpha)
    mean_g = real_g.flatten(1).mean(dim=1)[:, None, None, None]
    if method in {"Iso-SB", "Iso-SB-cond"}:
        mobility = omega.new_ones(omega.shape)
        gate = omega.new_ones(omega.shape)
    elif method == "spatial-shift-M":
        if shifted_omega is None:
            raise ValueError("spatial-shift-M requires shifted_omega in every batch/path")
        if shifted_omega.shape != omega.shape:
            raise ValueError("shifted_omega must share omega shape")
        mobility = g_min + (1.0 - g_min) * shifted_omega.clamp(0, 1).pow(alpha)
        gate = real_g
    else:
        mobility = real_g if spec.uses_spatial_mobility else mean_g.expand_as(real_g)
        gate = real_g if spec.uses_spatial_gate else mean_g.expand_as(real_g)
    return mobility, gate


def bridge_training_loss(
    model: Any,
    batch: Mapping[str, Any],
    *,
    method: str,
    g_min: float,
    alpha: float,
    endpoint_weight: float,
    gradient_weight: float,
    t_min: float = 1.0e-3,
    amp_forward_replay: bool = False,
    replay_telemetry: MutableMapping[str, Any] | None = None,
) -> Any:
    """White-noise bridge objective plus endpoint elevation/slope auxiliaries.

    When explicitly enabled, ``amp_forward_replay`` handles a bounded failure
    mode in which a finite FP16 activation overflows inside the network.  The
    replay uses the already-drawn bridge time/noise and the exact same model
    inputs under FP32; it never skips a batch or draws a replacement sample.
    """

    torch = _require_torch()
    x0 = batch["target"]
    xT = batch["zf"]
    omega = batch["omega"].detach()
    valid = batch["valid_mask"].to(dtype=x0.dtype)
    condition = masked_condition(batch["condition"], method)
    objective_call = None
    if replay_telemetry is not None:
        replay_telemetry["objective_calls"] = int(
            replay_telemetry.get("objective_calls", 0)
        ) + 1
        objective_call = int(replay_telemetry["objective_calls"])
    batch_size = x0.shape[0]
    t = torch.rand(batch_size, device=x0.device) * (1.0 - 2.0 * t_min) + t_min
    B_t = t[:, None, None, None]
    B_bar = 1.0 - B_t
    shifted_omega = batch.get("shifted_omega")
    mobility, gate = torch_mechanism_fields(
        method,
        omega,
        g_min=g_min,
        alpha=alpha,
        shifted_omega=shifted_omega,
    )
    epsilon = torch.randn_like(x0)
    bridge_std = torch.sqrt(B_t * B_bar * mobility)
    x_t = B_bar * x0 + B_t * xT + bridge_std * epsilon
    residual = model(x_t, condition, t)

    def loss_from_residual(residual_value: Any) -> Any:
        # The network remains on the registered autocast/FP16 path, but the
        # bridge normalization and reductions must be FP32.  Near an endpoint
        # the factor (1-t)/sqrt(t(1-t)M) can make a perfectly finite FP16
        # residual exceed 256; squaring that value in FP16 overflows even though
        # the true loss is finite.  Algebraically, epsilon_hat-epsilon equals
        # the normalized endpoint error below, so this also avoids a
        # cancellation-prone subtract/reconstruct path.
        loss_dtype = torch.float32
        x0_loss = x0.to(dtype=loss_dtype)
        xT_loss = xT.to(dtype=loss_dtype)
        valid_loss = valid.to(dtype=loss_dtype)
        residual_loss = residual_value.to(dtype=loss_dtype)
        gate_loss = gate.to(dtype=loss_dtype)
        mobility_loss = mobility.to(dtype=loss_dtype)
        B_t_loss = B_t.to(dtype=loss_dtype)
        B_bar_loss = B_bar.to(dtype=loss_dtype)
        bridge_std_loss = torch.sqrt(B_t_loss * B_bar_loss * mobility_loss)
        x0_hat = xT_loss + torch.sqrt(gate_loss) * residual_loss
        normalized_endpoint_error = (
            B_bar_loss * (x0_loss - x0_hat) / bridge_std_loss.clamp_min(1.0e-6)
        )
        denominator = valid_loss.sum().clamp_min(1.0)
        bridge_loss = ((normalized_endpoint_error**2) * valid_loss).sum() / denominator
        endpoint_l1 = (torch.abs(x0_hat - x0_loss) * valid_loss).sum() / denominator
        dx_hat = x0_hat[..., :, 1:] - x0_hat[..., :, :-1]
        dx = x0_loss[..., :, 1:] - x0_loss[..., :, :-1]
        dy_hat = x0_hat[..., 1:, :] - x0_hat[..., :-1, :]
        dy = x0_loss[..., 1:, :] - x0_loss[..., :-1, :]
        valid_x = valid_loss[..., :, 1:] * valid_loss[..., :, :-1]
        valid_y = valid_loss[..., 1:, :] * valid_loss[..., :-1, :]
        gradient_loss = (
            (torch.abs(dx_hat - dx) * valid_x).sum()
            + (torch.abs(dy_hat - dy) * valid_y).sum()
        ) / (valid_x.sum() + valid_y.sum()).clamp_min(1.0)
        return bridge_loss + endpoint_weight * (
            endpoint_l1 + gradient_weight * gradient_loss
        )

    total_loss = loss_from_residual(residual)
    if bool(torch.isfinite(total_loss)) or not amp_forward_replay:
        return total_loss

    if replay_telemetry is not None:
        replay_telemetry["attempt_count"] = int(
            replay_telemetry.get("attempt_count", 0)
        ) + 1
    # Nested autocast-disable overrides the outer training autocast context.
    # No RNG-consuming operation occurs between the two forwards.
    with torch.autocast(device_type=x_t.device.type, enabled=False):
        replay_residual = model(x_t.float(), condition.float(), t.float())
        replay_loss = loss_from_residual(replay_residual)
    if bool(torch.isfinite(replay_loss)):
        if replay_telemetry is not None:
            replay_telemetry["success_count"] = int(
                replay_telemetry.get("success_count", 0)
            ) + 1
            if replay_telemetry.get("first_success_objective_call") is None:
                replay_telemetry["first_success_objective_call"] = objective_call
                accumulation = int(replay_telemetry.get("accumulation_steps", 1))
                if objective_call is not None and accumulation > 0:
                    replay_telemetry["first_success_update"] = (
                        objective_call - 1
                    ) // accumulation + 1
        return replay_loss
    if replay_telemetry is not None:
        replay_telemetry["failure_count"] = int(
            replay_telemetry.get("failure_count", 0)
        ) + 1
    return replay_loss


def deterministic_training_loss(
    model: Any,
    batch: Mapping[str, Any],
    *,
    method: str,
    gradient_weight: float,
) -> Any:
    torch = _require_torch()
    spec = method_spec(method)
    target = batch["target"]
    zf = batch["zf"]
    valid = batch["valid_mask"].to(dtype=target.dtype)
    condition = masked_condition(batch["condition"], method)
    baseline = zf if spec.uses_laser else condition[:, :1]
    prediction = baseline + model(
        baseline,
        condition,
        torch.zeros(target.shape[0], device=target.device),
    )
    denominator = valid.sum().clamp_min(1.0)
    mse = (((prediction - target) ** 2) * valid).sum() / denominator
    dx_prediction = prediction[..., :, 1:] - prediction[..., :, :-1]
    dx_target = target[..., :, 1:] - target[..., :, :-1]
    dy_prediction = prediction[..., 1:, :] - prediction[..., :-1, :]
    dy_target = target[..., 1:, :] - target[..., :-1, :]
    valid_x = valid[..., :, 1:] * valid[..., :, :-1]
    valid_y = valid[..., 1:, :] * valid[..., :-1, :]
    gradient = (
        (torch.abs(dx_prediction - dx_target) * valid_x).sum()
        + (torch.abs(dy_prediction - dy_target) * valid_y).sum()
    ) / (valid_x.sum() + valid_y.sum()).clamp_min(1.0)
    return mse + gradient_weight * gradient


def _cosine_alpha_bar(time: Any, *, cosine_offset: float = 0.008) -> Any:
    """Continuous cosine cumulative-noise schedule used by the DDPM control.

    ``time`` is in [0, 1].  Normalising by the value at zero makes the clean
    endpoint exact, which is important for the K=0/endpoint contract.
    """

    torch = _require_torch()
    if not 0.0 <= cosine_offset < 1.0:
        raise ValueError("cosine_offset must be in [0,1)")
    angle = (time + cosine_offset) / (1.0 + cosine_offset) * (torch.pi / 2.0)
    initial = cosine_offset / (1.0 + cosine_offset) * (torch.pi / 2.0)
    return (torch.cos(angle) ** 2 / torch.cos(torch.as_tensor(initial, device=time.device)) ** 2).clamp(
        min=1.0e-6, max=1.0
    )


def ddpm_training_loss(
    model: Any,
    batch: Mapping[str, Any],
    *,
    method: str = "DDPM-DC",
    gradient_weight: float = 0.0,
    cosine_offset: float = 0.008,
    t_min: float = 1.0e-4,
) -> Any:
    """Conditional DDPM epsilon objective on the residual DEM ``target-zf``.

    This is deliberately separate from the Schrödinger-bridge objective: the
    forward marginal starts from the target residual and approaches white
    Gaussian noise, and neither OT mobility nor endpoint gate is used.
    """

    torch = _require_torch()
    if method != "DDPM-DC":
        raise ValueError("ddpm_training_loss is reserved for DDPM-DC")
    target_residual = batch["target"] - batch["zf"]
    valid = batch["valid_mask"].to(dtype=target_residual.dtype)
    condition = masked_condition(batch["condition"], method)
    batch_size = target_residual.shape[0]
    time = torch.rand(batch_size, device=target_residual.device) * (1.0 - t_min) + t_min
    alpha_bar = _cosine_alpha_bar(time, cosine_offset=cosine_offset)[:, None, None, None]
    epsilon = torch.randn_like(target_residual)
    noisy = torch.sqrt(alpha_bar) * target_residual + torch.sqrt(1.0 - alpha_bar) * epsilon
    epsilon_hat = model(noisy, condition, time)
    denominator = valid.sum().clamp_min(1.0)
    epsilon_loss = (((epsilon_hat - epsilon) ** 2) * valid).sum() / denominator
    if gradient_weight == 0:
        return epsilon_loss
    residual_hat = (noisy - torch.sqrt(1.0 - alpha_bar) * epsilon_hat) / torch.sqrt(
        alpha_bar
    ).clamp_min(1.0e-6)
    dx_hat = residual_hat[..., :, 1:] - residual_hat[..., :, :-1]
    dx = target_residual[..., :, 1:] - target_residual[..., :, :-1]
    dy_hat = residual_hat[..., 1:, :] - residual_hat[..., :-1, :]
    dy = target_residual[..., 1:, :] - target_residual[..., :-1, :]
    valid_x = valid[..., :, 1:] * valid[..., :, :-1]
    valid_y = valid[..., 1:, :] * valid[..., :-1, :]
    gradient_loss = (
        (torch.abs(dx_hat - dx) * valid_x).sum()
        + (torch.abs(dy_hat - dy) * valid_y).sum()
    ) / (valid_x.sum() + valid_y.sum()).clamp_min(1.0)
    return epsilon_loss + gradient_weight * gradient_loss


def _randn_with_generator_contract(
    shape: Any, *, device: Any, dtype: Any, generator: Any
) -> Any:
    """Draw one independent stream per batch row when generators are supplied."""

    torch = _require_torch()
    if isinstance(generator, Sequence):
        if len(shape) < 1 or len(generator) != int(shape[0]) or not generator:
            raise ValueError("batched reverse sampling requires one generator per row")
        sample_shape = (1, *tuple(int(value) for value in shape[1:]))
        return torch.cat(
            [
                torch.randn(
                    sample_shape,
                    device=device,
                    dtype=dtype,
                    generator=stream,
                )
                for stream in generator
            ],
            dim=0,
        )
    return torch.randn(shape, device=device, dtype=dtype, generator=generator)


def reverse_bridge_path(
    model: Any,
    *,
    zf: Any,
    condition: Any,
    omega: Any,
    method: str,
    g_min: float,
    alpha: float,
    steps: int,
    generator: Any,
    shifted_omega: Any | None = None,
) -> Any:
    """Sample the registered analytic reverse bridge kernel on one patch."""

    torch = _require_torch()
    if steps <= 0:
        raise ValueError("steps must be positive")
    mobility, gate = torch_mechanism_fields(
        method,
        omega,
        g_min=g_min,
        alpha=alpha,
        shifted_omega=shifted_omega,
    )
    condition = masked_condition(condition, method)
    state = zf.clone()
    for index in range(steps, 0, -1):
        t_value = index / steps
        s_value = (index - 1) / steps
        t = torch.full((state.shape[0],), t_value, device=state.device, dtype=state.dtype)
        residual = model(state, condition, t)
        x0_hat = zf + torch.sqrt(gate) * residual
        mean = x0_hat + (s_value / t_value) * (state - x0_hat)
        if index > 1:
            variance = s_value * (t_value - s_value) / t_value
            noise = _randn_with_generator_contract(
                state.shape,
                device=state.device,
                dtype=state.dtype,
                generator=generator,
            )
            state = mean + torch.sqrt(torch.as_tensor(variance, device=state.device) * mobility) * noise
        else:
            state = mean
    return state


def reverse_ddpm_path(
    model: Any,
    *,
    zf: Any,
    condition: Any,
    steps: int,
    generator: Any,
    cosine_offset: float = 0.008,
    eta: float = 1.0,
) -> Any:
    """Sample the conditional DDPM residual with a locked reduced-step schedule.

    The update is the variance-preserving ancestral/DDIM form. ``eta=1`` is the
    stochastic DDPM control used by the formal protocol; returning ``zf`` plus
    the sampled residual keeps the endpoint definition distinct from SB.
    """

    torch = _require_torch()
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be in [0,1]")
    condition = masked_condition(condition, "DDPM-DC")
    residual = _randn_with_generator_contract(
        zf.shape,
        device=zf.device,
        dtype=zf.dtype,
        generator=generator,
    )
    for index in range(steps, 0, -1):
        t_value = index / steps
        s_value = (index - 1) / steps
        time = torch.full(
            (residual.shape[0],), t_value, device=residual.device, dtype=residual.dtype
        )
        next_time = torch.full_like(time, s_value)
        alpha_t = _cosine_alpha_bar(time, cosine_offset=cosine_offset)[:, None, None, None]
        alpha_s = _cosine_alpha_bar(next_time, cosine_offset=cosine_offset)[
            :, None, None, None
        ]
        epsilon_hat = model(residual, condition, time)
        clean = (residual - torch.sqrt(1.0 - alpha_t) * epsilon_hat) / torch.sqrt(
            alpha_t
        ).clamp_min(1.0e-6)
        sigma_squared = (
            eta**2
            * (1.0 - alpha_s)
            / (1.0 - alpha_t).clamp_min(1.0e-8)
            * (1.0 - alpha_t / alpha_s.clamp_min(1.0e-8))
        ).clamp_min(0.0)
        direction = torch.sqrt((1.0 - alpha_s - sigma_squared).clamp_min(0.0)) * epsilon_hat
        mean = torch.sqrt(alpha_s) * clean + direction
        if index > 1 and eta > 0:
            noise = _randn_with_generator_contract(
                residual.shape,
                device=residual.device,
                dtype=residual.dtype,
                generator=generator,
            )
            residual = mean + torch.sqrt(sigma_squared) * noise
        else:
            residual = mean
    return zf + residual

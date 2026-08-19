"""Shared-memory gradient synchronization for two Windows CUDA workers."""

from __future__ import annotations

import torch

from reliable_objective import reliable_site_objective_from_statistics


_rank: int | None = None
_world_size: int | None = None
_gradient_buffer: torch.Tensor | None = None
_metric_buffer: torch.Tensor | None = None
_correlation_buffer: torch.Tensor | None = None
_barrier = None


def configure(
    rank, world_size, gradient_buffer, metric_buffer, barrier, correlation_buffer=None
) -> None:
    global _rank, _world_size, _gradient_buffer, _metric_buffer, _barrier, _correlation_buffer
    _rank = int(rank)
    _world_size = int(world_size)
    _gradient_buffer = gradient_buffer
    _metric_buffer = metric_buffer
    _correlation_buffer = correlation_buffer
    _barrier = barrier


def enabled() -> bool:
    return _rank is not None


def wait() -> None:
    if enabled():
        _barrier.wait()


def synchronize_gradients(parameters) -> None:
    if not enabled():
        return
    parameters = list(parameters)
    required = sum(parameter.numel() for parameter in parameters)
    if required > _gradient_buffer.shape[1]:
        raise RuntimeError(
            f"shared gradient capacity {int(_gradient_buffer.shape[1]):,} is below {required:,}"
        )
    row = _gradient_buffer[_rank]
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        target = row[offset : offset + count]
        if parameter.grad is None:
            target.zero_()
        else:
            target.copy_(parameter.grad.detach().reshape(-1), non_blocking=False)
        offset += count
    _barrier.wait()
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        averaged = _gradient_buffer[:, offset : offset + count].mean(dim=0)
        if parameter.grad is None:
            parameter.grad = torch.empty_like(parameter)
        parameter.grad.copy_(averaged.reshape_as(parameter), non_blocking=False)
        offset += count
    _barrier.wait()


def mean(total: float, count: float) -> float:
    if not enabled():
        return total / max(count, 1.0)
    _metric_buffer[_rank, 0] = total
    _metric_buffer[_rank, 1] = count
    _barrier.wait()
    combined = _metric_buffer.sum(dim=0)
    result = float(combined[0] / combined[1].clamp_min(1.0))
    _barrier.wait()
    return result


def globalize_site_statistics(local_statistics: torch.Tensor) -> torch.Tensor:
    """Sum one correlation-window statistics tensor across Windows workers."""
    if not enabled() or _correlation_buffer is None:
        raise RuntimeError("global site statistics were not configured")
    local = torch.as_tensor(local_statistics, dtype=torch.float64, device="cpu")
    if local.ndim != 2 or local.shape[0] != 7:
        raise ValueError("site statistics must have shape [7, sites]")
    width = local.shape[1]
    if width > _correlation_buffer.shape[2]:
        raise RuntimeError("site chunk exceeds the shared correlation buffer")
    _correlation_buffer[_rank, :, :width].copy_(local)
    _barrier.wait()
    total = _correlation_buffer[:, :, :width].sum(0).clone()
    _barrier.wait()
    return total


def window_reliable_site_objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    reliability: torch.Tensor,
    global_statistics: torch.Tensor,
    minimum_observations: int,
    huber_weight: float,
):
    return reliable_site_objective_from_statistics(
        prediction, target, mask, reliability, global_statistics,
        minimum_pearson_observations=minimum_observations,
        huber_weight=huber_weight, gradient_scale=float(_world_size),
    )


def global_site_pearson_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    minimum_observations: int,
) -> torch.Tensor:
    """Differentiate the exact global-batch Pearson loss on each local shard."""
    if not enabled() or _correlation_buffer is None:
        raise RuntimeError("global Pearson statistics were not configured")
    width = prediction.shape[1]
    if width > _correlation_buffer.shape[2]:
        raise RuntimeError("site chunk exceeds the shared correlation buffer")
    value = prediction.float()
    target_value = target.float()
    weight = mask.float()
    local = torch.stack(
        [
            weight.sum(dim=0),
            (value * weight).sum(dim=0),
            (target_value * weight).sum(dim=0),
            (value.square() * weight).sum(dim=0),
            (target_value.square() * weight).sum(dim=0),
            (value * target_value * weight).sum(dim=0),
        ],
        dim=0,
    )
    _correlation_buffer[_rank, :6, :width].copy_(
        local.detach().to(device="cpu", dtype=_correlation_buffer.dtype)
    )
    _barrier.wait()
    total_detached = _correlation_buffer[:, :6, :width].sum(dim=0).to(
        device=value.device, dtype=value.dtype
    )
    local_detached = _correlation_buffer[_rank, :6, :width].to(
        device=value.device, dtype=value.dtype
    )
    total = local + (total_detached - local_detached)
    count = total[0].clamp_min(1.0)
    sum_x, sum_y, sum_x2, sum_y2, sum_xy = total[1:]
    covariance = sum_xy - sum_x * sum_y / count
    variance_x = (sum_x2 - sum_x.square() / count).clamp_min(1.0e-8)
    variance_y = (sum_y2 - sum_y.square() / count).clamp_min(1.0e-8)
    correlation = covariance / (variance_x.sqrt() * variance_y.sqrt()).clamp_min(1.0e-8)
    valid = (count >= int(minimum_observations)) & torch.isfinite(correlation)
    loss = value.sum() * 0.0 if not bool(valid.any()) else 1.0 - correlation[valid].mean()
    _barrier.wait()
    # Shared-memory gradient synchronization averages workers.  Multiplying
    # here restores the sum of the two local derivatives of one global loss.
    return loss * float(_world_size)


def global_reliable_site_objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    reliability: torch.Tensor,
    minimum_observations: int,
    huber_weight: float,
):
    """Differentiate one exact global site-equal objective across all shards."""
    if not enabled() or _correlation_buffer is None:
        raise RuntimeError("global correlation statistics were not configured")
    width = prediction.shape[1]
    value, truth, observed = prediction.float(), target.float(), mask.float()
    absolute = (value - truth).abs()
    point = torch.where(absolute < 1, 0.5 * absolute.square(), absolute - 0.5) * observed
    local = torch.stack(
        [observed.sum(0), (value * observed).sum(0), (truth * observed).sum(0),
         (value.square() * observed).sum(0), (truth.square() * observed).sum(0),
         (value * truth * observed).sum(0), point.sum(0)], dim=0,
    )
    _correlation_buffer[_rank, :, :width].copy_(local.detach().cpu().double())
    _barrier.wait()
    total_detached = _correlation_buffer[:, :, :width].sum(0).to(value.device, value.dtype)
    local_detached = _correlation_buffer[_rank, :, :width].to(value.device, value.dtype)
    total = local + (total_detached - local_detached)
    count = total[0].clamp_min(1)
    sum_x, sum_y, sum_x2, sum_y2, sum_xy = total[1:6]
    covariance = sum_xy - sum_x * sum_y / count
    variance_x = (sum_x2 - sum_x.square() / count).clamp_min(1e-8)
    variance_y = (sum_y2 - sum_y.square() / count).clamp_min(1e-8)
    correlation = covariance / (variance_x.sqrt() * variance_y.sqrt()).clamp_min(1e-8)
    weight = (reliability.to(value.dtype) > 0).to(value.dtype)
    eligible = (count >= int(minimum_observations)) & torch.isfinite(correlation) & (weight > 0)
    pearson = value.sum() * 0 if not bool(eligible.any()) else (
        ((1 - correlation[eligible]) * weight[eligible]).sum() / weight[eligible].sum().clamp_min(1e-8)
    )
    site_huber = total[6] / count
    low = (~eligible) & (total[0] > 0)
    high = eligible
    fallback = site_huber[low].mean() if bool(low.any()) else value.sum() * 0
    weak = site_huber[high].mean() if bool(high.any()) else value.sum() * 0
    _barrier.wait()
    # Every worker differentiates its shard contribution to the same global
    # objective.  Gradient synchronization averages workers, so restore the sum.
    loss = (pearson + fallback + float(huber_weight) * weak) * float(_world_size)
    return loss, {
        "pearson_loss": pearson,
        "fallback_huber": fallback,
        "weak_huber": weak,
        "pearson_site_count": eligible.sum().to(value.dtype),
        "fallback_site_count": low.sum().to(value.dtype),
    }

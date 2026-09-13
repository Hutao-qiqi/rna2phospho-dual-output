"""Linear-memory masked correlation objectives for RNA-to-protein training.

All statistics are evaluated across samples, independently for each protein.
Inputs may use mixed precision; reductions and rank calculations always use
``float32``. Missing targets are excluded through ``mask`` and each eligible
protein contributes equally to the correlation objectives.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F


ValueLoss = Literal["huber", "mse"]
ValueReduction = Literal["per_observation", "per_target"]


def _validate_inputs(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if prediction.ndim != 2:
        raise ValueError("prediction, target and mask must be two-dimensional")
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target and mask must have identical shapes")
    if mask.dtype != torch.bool:
        raise TypeError("mask must have boolean dtype")
    if prediction.device != target.device or prediction.device != mask.device:
        raise ValueError("prediction, target and mask must be on the same device")


def _zero_with_gradient(prediction: torch.Tensor) -> torch.Tensor:
    return prediction.float().sum() * 0.0


def _pad_rows(tensor: torch.Tensor, rows: int, value: float | bool) -> torch.Tensor:
    if tensor.shape[0] == rows:
        return tensor
    padding = tensor.new_full((rows - tensor.shape[0], tensor.shape[1]), value)
    return torch.cat((tensor, padding), dim=0)


def _distributed_rows(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather the current forward batch while retaining prediction gradients.

    The function supports unequal local row counts by padding before the
    collective. It must be called by every rank in the process group. With
    DDP, each rank evaluates the same global loss; the differentiable gather
    and DDP gradient averaging then produce the global-batch gradient.
    """

    if not enabled or not dist.is_available() or not dist.is_initialized():
        return prediction, target, mask
    world_size = dist.get_world_size()
    if world_size == 1:
        return prediction, target, mask

    local_size = torch.tensor(
        [prediction.shape[0]], device=prediction.device, dtype=torch.long
    )
    gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(gathered_sizes, local_size)
    sizes = [int(value.item()) for value in gathered_sizes]
    max_rows = max(sizes)

    prediction_pad = _pad_rows(prediction, max_rows, 0.0)
    target_pad = _pad_rows(target, max_rows, 0.0)
    mask_pad = _pad_rows(mask, max_rows, False)

    try:
        from torch.distributed.nn.functional import all_gather as grad_all_gather
    except ImportError as error:  # pragma: no cover - torch>=2.2 provides it
        raise RuntimeError(
            "differentiable distributed gathering requires torch>=2.2"
        ) from error

    prediction_parts = grad_all_gather(prediction_pad)
    target_parts = [torch.empty_like(target_pad) for _ in range(world_size)]
    mask_parts = [torch.empty_like(mask_pad) for _ in range(world_size)]
    dist.all_gather(target_parts, target_pad)
    dist.all_gather(mask_parts, mask_pad)

    prediction_all = torch.cat(
        [part[:size] for part, size in zip(prediction_parts, sizes)], dim=0
    )
    target_all = torch.cat(
        [part[:size] for part, size in zip(target_parts, sizes)], dim=0
    )
    mask_all = torch.cat(
        [part[:size] for part, size in zip(mask_parts, sizes)], dim=0
    )
    return prediction_all, target_all, mask_all


def _float_inputs(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    gather_distributed: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_inputs(prediction, target, mask)
    prediction32 = prediction.float()
    target32 = target.float()
    prediction32, target32, mask = _distributed_rows(
        prediction32, target32, mask, gather_distributed
    )
    valid = mask & torch.isfinite(prediction32) & torch.isfinite(target32)
    prediction32 = torch.where(valid, prediction32, torch.zeros_like(prediction32))
    target32 = torch.where(valid, target32, torch.zeros_like(target32))
    return prediction32, target32, valid


def _select_columns(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    max_targets: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if max_targets is None or prediction.shape[1] <= max_targets:
        return prediction, target, valid
    if max_targets <= 0:
        raise ValueError("max_targets must be positive or None")
    counts = valid.sum(dim=0)
    indices = torch.topk(counts, k=max_targets, largest=True, sorted=False).indices
    return (
        prediction.index_select(1, indices),
        target.index_select(1, indices),
        valid.index_select(1, indices),
    )


def masked_value_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    kind: ValueLoss = "huber",
    reduction: ValueReduction = "per_observation",
    huber_delta: float = 1.0,
    gather_distributed: bool = False,
) -> torch.Tensor:
    """Huber or MSE over observed finite entries, evaluated in float32.

    ``per_observation`` gives every observed sample-protein pair equal weight.
    ``per_target`` first averages within each protein and then gives every
    protein with at least one finite observation equal weight.
    """

    if kind not in {"huber", "mse"}:
        raise ValueError("kind must be 'huber' or 'mse'")
    if huber_delta <= 0:
        raise ValueError("huber_delta must be positive")
    if reduction not in {"per_observation", "per_target"}:
        raise ValueError("reduction must be 'per_observation' or 'per_target'")
    prediction32, target32, valid = _float_inputs(
        prediction, target, mask, gather_distributed=gather_distributed
    )
    return _masked_value_loss_float(
        prediction32,
        target32,
        valid,
        kind=kind,
        reduction=reduction,
        huber_delta=huber_delta,
    )


def _masked_value_loss_float(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    kind: ValueLoss,
    reduction: ValueReduction,
    huber_delta: float,
) -> torch.Tensor:
    if kind == "huber":
        elementwise = F.huber_loss(
            prediction, target, delta=huber_delta, reduction="none"
        )
    else:
        elementwise = (prediction - target).square()
    valid32 = valid.float()
    if reduction == "per_observation":
        return (elementwise * valid32).sum() / valid32.sum().clamp_min(1.0)
    if reduction != "per_target":
        raise ValueError("reduction must be 'per_observation' or 'per_target'")

    count_per_target = valid32.sum(dim=0)
    eligible = count_per_target > 0
    mean_per_target = (elementwise * valid32).sum(dim=0) / count_per_target.clamp_min(1.0)
    return (
        mean_per_target * eligible.to(dtype=mean_per_target.dtype)
    ).sum() / eligible.sum().to(dtype=torch.float32).clamp_min(1.0)


def _column_correlation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    min_observations: int,
    variance_epsilon: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if min_observations < 2:
        raise ValueError("min_observations must be at least 2")
    if variance_epsilon <= 0:
        raise ValueError("variance_epsilon must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    loss_sum = _zero_with_gradient(prediction)
    eligible_total = torch.zeros((), device=prediction.device, dtype=torch.long)
    for start in range(0, prediction.shape[1], chunk_size):
        stop = min(start + chunk_size, prediction.shape[1])
        pred = prediction[:, start:stop]
        truth = target[:, start:stop]
        observed = valid[:, start:stop]
        observed32 = observed.float()
        count = observed32.sum(dim=0)
        denominator_count = count.clamp_min(1.0)
        pred_mean = (pred * observed32).sum(dim=0) / denominator_count
        truth_mean = (truth * observed32).sum(dim=0) / denominator_count
        pred_centered = torch.where(observed, pred - pred_mean, 0.0)
        truth_centered = torch.where(observed, truth - truth_mean, 0.0)
        pred_ss = pred_centered.square().sum(dim=0)
        truth_ss = truth_centered.square().sum(dim=0)
        covariance = (pred_centered * truth_centered).sum(dim=0)
        eligible = (
            (count >= min_observations)
            & (pred_ss > variance_epsilon)
            & (truth_ss > variance_epsilon)
        )
        correlation = covariance / torch.sqrt(
            pred_ss.clamp_min(variance_epsilon)
            * truth_ss.clamp_min(variance_epsilon)
        )
        correlation = correlation.clamp(-1.0, 1.0)
        loss_sum = loss_sum + torch.where(
            eligible, 1.0 - correlation, torch.zeros_like(correlation)
        ).sum()
        eligible_total = eligible_total + eligible.sum()
    loss = loss_sum / eligible_total.to(dtype=torch.float32).clamp_min(1.0)
    return loss, eligible_total


def masked_pearson_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    min_observations: int = 4,
    variance_epsilon: float = 1e-8,
    chunk_size: int = 2048,
    max_targets: int | None = None,
    gather_distributed: bool = False,
) -> torch.Tensor:
    """Mean ``1 - Pearson r`` across eligible protein columns.

    Constant columns and columns with too few observations contribute zero.
    Memory use is linear in batch size and target count.
    """

    prediction32, target32, valid = _float_inputs(
        prediction, target, mask, gather_distributed=gather_distributed
    )
    return _masked_pearson_loss_float(
        prediction32,
        target32,
        valid,
        min_observations=min_observations,
        variance_epsilon=variance_epsilon,
        chunk_size=chunk_size,
        max_targets=max_targets,
    )


def _masked_pearson_loss_float(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    min_observations: int,
    variance_epsilon: float,
    chunk_size: int,
    max_targets: int | None,
) -> torch.Tensor:
    prediction32, target32, valid = _select_columns(
        prediction, target, valid, max_targets
    )
    loss, _ = _column_correlation_loss(
        prediction32,
        target32,
        valid,
        min_observations=min_observations,
        variance_epsilon=variance_epsilon,
        chunk_size=chunk_size,
    )
    return loss


def _masked_midranks(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Exact masked midranks with O(samples * targets) memory."""

    n_samples = values.shape[0]
    if n_samples == 0:
        return values.clone()
    detached = values.detach()
    sortable = torch.where(valid, detached, torch.full_like(detached, torch.inf))
    sorted_values, order = torch.sort(sortable, dim=0, stable=True)
    sorted_valid = torch.gather(valid, 0, order)
    positions = torch.arange(
        n_samples, device=values.device, dtype=torch.long
    ).unsqueeze(1).expand_as(order)

    previous_differs = torch.ones_like(sorted_valid)
    if n_samples > 1:
        previous_differs[1:] = (
            ~sorted_valid[:-1]
            | (sorted_values[1:] != sorted_values[:-1])
        )
    group_start_marker = torch.where(
        sorted_valid & previous_differs, positions, torch.zeros_like(positions)
    )
    group_start = torch.cummax(group_start_marker, dim=0).values

    next_differs = torch.ones_like(sorted_valid)
    if n_samples > 1:
        next_differs[:-1] = (
            ~sorted_valid[1:]
            | (sorted_values[:-1] != sorted_values[1:])
        )
    group_end_marker = torch.where(
        sorted_valid & next_differs,
        positions,
        torch.full_like(positions, n_samples),
    )
    group_end = torch.flip(
        torch.cummin(torch.flip(group_end_marker, dims=(0,)), dim=0).values,
        dims=(0,),
    )
    sorted_rank = 0.5 * (group_start.float() + group_end.float())
    rank = torch.zeros_like(values, dtype=torch.float32)
    rank.scatter_(0, order, sorted_rank)
    count = valid.sum(dim=0).float().clamp_min(1.0)
    rank = (rank + 0.5) / count
    return torch.where(valid, rank, torch.zeros_like(rank))


def masked_spearman_surrogate_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    min_observations: int = 4,
    variance_epsilon: float = 1e-8,
    chunk_size: int = 2048,
    max_targets: int | None = 2048,
    gather_distributed: bool = False,
) -> torch.Tensor:
    """Straight-through Spearman surrogate without pairwise sample tensors.

    Forward values are exact masked midranks for predictions and targets.
    The prediction rank uses an identity straight-through gradient. Sorting is
    O(B log B) per protein and memory remains O(BP), where B is batch size and
    P is the number of selected proteins.
    """

    prediction32, target32, valid = _float_inputs(
        prediction, target, mask, gather_distributed=gather_distributed
    )
    return _masked_spearman_loss_float(
        prediction32,
        target32,
        valid,
        min_observations=min_observations,
        variance_epsilon=variance_epsilon,
        chunk_size=chunk_size,
        max_targets=max_targets,
    )


def _masked_spearman_loss_float(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    min_observations: int,
    variance_epsilon: float,
    chunk_size: int,
    max_targets: int | None,
) -> torch.Tensor:
    prediction32, target32, valid = _select_columns(
        prediction, target, valid, max_targets
    )
    prediction_rank = _masked_midranks(prediction32, valid)
    target_rank = _masked_midranks(target32, valid)
    prediction_rank_ste = prediction32 + (prediction_rank - prediction32).detach()
    loss, _ = _column_correlation_loss(
        prediction_rank_ste,
        target_rank,
        valid,
        min_observations=min_observations,
        variance_epsilon=variance_epsilon,
        chunk_size=chunk_size,
    )
    return loss


def correlation_guided_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    value_kind: ValueLoss = "huber",
    value_reduction: ValueReduction = "per_observation",
    value_weight: float = 1.0,
    pearson_weight: float = 0.10,
    spearman_weight: float = 0.05,
    huber_delta: float = 1.0,
    min_observations: int = 4,
    variance_epsilon: float = 1e-8,
    chunk_size: int = 2048,
    pearson_max_targets: int | None = None,
    spearman_max_targets: int | None = 2048,
    gather_distributed: bool = False,
) -> dict[str, torch.Tensor]:
    """Mix an observed-value objective with correlation-oriented objectives."""

    weights = (value_weight, pearson_weight, spearman_weight)
    if any(weight < 0 for weight in weights):
        raise ValueError("loss weights must be non-negative")
    if sum(weights) <= 0:
        raise ValueError("at least one loss weight must be positive")

    if value_kind not in {"huber", "mse"}:
        raise ValueError("value_kind must be 'huber' or 'mse'")
    if value_reduction not in {"per_observation", "per_target"}:
        raise ValueError(
            "value_reduction must be 'per_observation' or 'per_target'"
        )
    if huber_delta <= 0:
        raise ValueError("huber_delta must be positive")
    prediction32, target32, valid = _float_inputs(
        prediction, target, mask, gather_distributed=gather_distributed
    )
    value = _masked_value_loss_float(
        prediction32,
        target32,
        valid,
        kind=value_kind,
        reduction=value_reduction,
        huber_delta=huber_delta,
    )
    pearson = _masked_pearson_loss_float(
        prediction32,
        target32,
        valid,
        min_observations=min_observations,
        variance_epsilon=variance_epsilon,
        chunk_size=chunk_size,
        max_targets=pearson_max_targets,
    )
    spearman = _masked_spearman_loss_float(
        prediction32,
        target32,
        valid,
        min_observations=min_observations,
        variance_epsilon=variance_epsilon,
        chunk_size=chunk_size,
        max_targets=spearman_max_targets,
    )
    total = (
        value_weight * value
        + pearson_weight * pearson
        + spearman_weight * spearman
    )
    return {
        "loss": total,
        "value": value,
        "pearson": pearson,
        "spearman": spearman,
    }


__all__ = [
    "correlation_guided_loss",
    "masked_pearson_loss",
    "masked_spearman_surrogate_loss",
    "masked_value_loss",
]

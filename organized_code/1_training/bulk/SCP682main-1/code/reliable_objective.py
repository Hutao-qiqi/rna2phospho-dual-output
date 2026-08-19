"""Coverage-aware correlation objective with a deterministic Huber fallback."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def site_objective_statistics(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Seven sufficient statistics for Pearson and Huber, one column per site."""
    value, truth, observed = prediction.float(), target.float(), mask.float()
    absolute = (value - truth).abs()
    point = torch.where(absolute < 1, 0.5 * absolute.square(), absolute - 0.5) * observed
    return torch.stack([
        observed.sum(0), (value * observed).sum(0), (truth * observed).sum(0),
        (value.square() * observed).sum(0), (truth.square() * observed).sum(0),
        (value * truth * observed).sum(0), point.sum(0),
    ])


def reliable_site_objective_from_statistics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    reliability: torch.Tensor,
    global_statistics: torch.Tensor,
    *,
    minimum_pearson_observations: int = 16,
    huber_weight: float = 0.05,
    gradient_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Use detached window statistics while differentiating one physical batch."""
    local = site_objective_statistics(prediction, target, mask)
    global_value = torch.as_tensor(
        global_statistics, dtype=local.dtype, device=local.device
    )
    if global_value.shape != local.shape:
        raise ValueError("global site statistics differ from the current site chunk")
    total = local + (global_value - local.detach())
    raw_count = total[0]
    count = raw_count.clamp_min(1)
    sum_x, sum_y, sum_x2, sum_y2, sum_xy = total[1:6]
    covariance = sum_xy - sum_x * sum_y / count
    raw_variance_x = sum_x2 - sum_x.square() / count
    raw_variance_y = sum_y2 - sum_y.square() / count
    variance_x = raw_variance_x.clamp_min(1e-8)
    variance_y = raw_variance_y.clamp_min(1e-8)
    denominator = (variance_x.sqrt() * variance_y.sqrt()).clamp_min(1e-8)
    correlation = covariance / denominator
    valid_reliability = torch.as_tensor(
        reliability, dtype=local.dtype, device=local.device
    ) > 0
    eligible = (
        (raw_count >= int(minimum_pearson_observations))
        & torch.isfinite(correlation) & (raw_variance_x > 1e-8)
        & (raw_variance_y > 1e-8) & valid_reliability
    )
    zero = prediction.sum() * 0
    pearson = zero if not bool(eligible.any()) else (1 - correlation[eligible]).mean()
    site_huber = total[6] / count
    observed_site = (raw_count > 0) & torch.isfinite(site_huber)
    low = (~eligible) & observed_site
    high = eligible & observed_site
    fallback = site_huber[low].mean() if bool(low.any()) else zero
    weak = site_huber[high].mean() if bool(high.any()) else zero
    unscaled = pearson + fallback + float(huber_weight) * weak
    return unscaled * float(gradient_scale), {
        "pearson_loss": pearson.detach(), "fallback_huber": fallback.detach(),
        "weak_huber": weak.detach(), "pearson_site_count": eligible.sum().to(local.dtype),
        "fallback_site_count": low.sum().to(local.dtype),
    }


def site_pearson_loss_from_statistics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    reliability: torch.Tensor,
    global_statistics: torch.Tensor,
    *,
    minimum_observations: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiate a Pearson-only site loss from fixed window statistics."""
    local = site_objective_statistics(prediction, target, mask)
    global_value = torch.as_tensor(
        global_statistics, dtype=local.dtype, device=local.device
    )
    if global_value.shape != local.shape:
        raise ValueError("global site statistics differ from the current site chunk")
    total = local + (global_value - local.detach())
    raw_count = total[0]
    count = raw_count.clamp_min(1)
    sum_x, sum_y, sum_x2, sum_y2, sum_xy = total[1:6]
    covariance = sum_xy - sum_x * sum_y / count
    variance_x = sum_x2 - sum_x.square() / count
    variance_y = sum_y2 - sum_y.square() / count
    correlation = covariance / (
        variance_x.clamp_min(1e-8).sqrt() * variance_y.clamp_min(1e-8).sqrt()
    ).clamp_min(1e-8)
    reliable = torch.as_tensor(
        reliability, dtype=local.dtype, device=local.device
    ) > 0
    eligible = (
        (raw_count >= int(minimum_observations))
        & (variance_x > 1e-8) & (variance_y > 1e-8)
        & torch.isfinite(correlation) & reliable
    )
    zero = prediction.sum() * 0
    loss = zero if not bool(eligible.any()) else (1 - correlation[eligible]).mean()
    return loss, eligible.sum().to(local.dtype)


def reliable_site_objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    reliability: torch.Tensor,
    *,
    minimum_pearson_observations: int = 16,
    huber_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("prediction, target and mask must have identical shapes")
    reliability = torch.as_tensor(reliability, dtype=prediction.dtype, device=prediction.device)
    if reliability.ndim != 1 or reliability.numel() != prediction.shape[1]:
        raise ValueError("reliability must contain one value per site")
    mask = mask.bool()
    count = mask.sum(0)
    # Model selection gives every evaluable site one vote.  Coverage therefore
    # gates invalid sites but does not multiply their influence.
    weight = (reliability > 0).to(prediction.dtype)
    enough = count >= int(minimum_pearson_observations)
    count_float = count.clamp_min(1).to(prediction.dtype)
    pred_mean = (prediction * mask).sum(0) / count_float
    target_mean = (target * mask).sum(0) / count_float
    pred_centered = (prediction - pred_mean) * mask
    target_centered = (target - target_mean) * mask
    numerator = (pred_centered * target_centered).sum(0)
    denominator = pred_centered.square().sum(0).sqrt() * target_centered.square().sum(0).sqrt()
    correlation = numerator / denominator.clamp_min(1.0e-8)
    pearson_sites = enough & torch.isfinite(correlation) & (denominator > 1.0e-8) & (weight > 0)
    if bool(pearson_sites.any()):
        pearson = ((1.0 - correlation[pearson_sites]) * weight[pearson_sites]).sum()
        pearson = pearson / weight[pearson_sites].sum().clamp_min(1.0e-8)
    else:
        pearson = prediction.sum() * 0.0

    point = F.huber_loss(prediction, target, reduction="none", delta=1.0) * mask
    site_huber = point.sum(0) / count_float
    huber_sites = (count > 0) & torch.isfinite(site_huber)
    huber = site_huber[huber_sites].mean() if bool(huber_sites.any()) else prediction.sum() * 0.0
    # Low-coverage sites receive only Huber; higher-coverage sites receive weak Huber.
    fallback = (~pearson_sites) & huber_sites
    fallback_huber = (
        site_huber[fallback].mean() if bool(fallback.any()) else prediction.sum() * 0.0
    )
    high_huber = (
        site_huber[pearson_sites & huber_sites].mean()
        if bool((pearson_sites & huber_sites).any()) else prediction.sum() * 0.0
    )
    total = pearson + fallback_huber + float(huber_weight) * high_huber
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("coverage-aware objective produced a non-finite value")
    return total, {
        "pearson_loss": pearson,
        "fallback_huber": fallback_huber,
        "weak_huber": high_huber,
        "pearson_site_count": pearson_sites.sum().to(prediction.dtype),
        "fallback_site_count": fallback.sum().to(prediction.dtype),
        "mean_valid_observations": count.to(prediction.dtype).mean(),
        "all_site_huber": huber,
    }

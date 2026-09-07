"""Masked objectives for protein and phosphosite translation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    mask = mask.to(dtype=prediction.dtype)
    loss = F.huber_loss(prediction, target, delta=delta, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def masked_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error restricted to observed target positions."""

    observed = mask.to(dtype=prediction.dtype)
    squared_error = (prediction - target).square()
    return (squared_error * observed).sum() / observed.sum().clamp_min(1.0)


def masked_pairwise_rank_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 0.20,
    max_targets: int = 2048,
) -> torch.Tensor:
    """Logistic rank loss across samples for each observed target."""

    if prediction.shape[1] > max_targets:
        observed = mask.sum(dim=0)
        eligible = torch.nonzero(observed >= 2, as_tuple=False).flatten()
        if eligible.numel() > max_targets:
            perm = torch.randperm(eligible.numel(), device=prediction.device)[:max_targets]
            eligible = eligible.index_select(0, perm)
        prediction = prediction.index_select(1, eligible)
        target = target.index_select(1, eligible)
        mask = mask.index_select(1, eligible)

    pred_diff = prediction[:, None, :] - prediction[None, :, :]
    target_diff = target[:, None, :] - target[None, :, :]
    pair_mask = mask[:, None, :].bool() & mask[None, :, :].bool()
    pair_mask &= target_diff.abs() > 1e-6
    if not pair_mask.any():
        return prediction.sum() * 0.0
    direction = target_diff.sign()
    loss = F.softplus(-direction * pred_diff / temperature)
    return loss[pair_mask].mean()


def joint_translation_loss(
    output: dict[str, torch.Tensor],
    protein_target: torch.Tensor,
    protein_mask: torch.Tensor,
    site_target: torch.Tensor | None = None,
    site_mask: torch.Tensor | None = None,
    protein_weight: float = 0.25,
    protein_rank_weight: float = 0.05,
    site_rank_weight: float = 0.10,
    residual_target_weight: float = 0.50,
    residual_l2_weight: float = 1e-3,
) -> dict[str, torch.Tensor]:
    protein_huber = masked_huber(output["protein"], protein_target, protein_mask)
    protein_rank = masked_pairwise_rank_loss(
        output["protein"], protein_target, protein_mask
    )
    protein_loss = protein_huber + protein_rank_weight * protein_rank

    if site_target is None or site_mask is None or "phosphosite" not in output:
        total = protein_loss
        zero = total.detach() * 0.0
        return {
            "loss": total,
            "protein_huber": protein_huber,
            "protein_rank": protein_rank,
            "site_huber": zero,
            "site_rank": zero,
            "residual_huber": zero,
            "residual_l2": zero,
        }

    site_huber = masked_huber(output["phosphosite"], site_target, site_mask)
    site_rank = masked_pairwise_rank_loss(
        output["phosphosite"], site_target, site_mask
    )
    residual_target = site_target - output["phosphosite_anchor"].detach()
    residual_huber = masked_huber(
        output["phosphosite_residual"], residual_target, site_mask
    )
    residual_l2 = output["phosphosite_residual"].square().mean()
    total = (
        site_huber
        + site_rank_weight * site_rank
        + protein_weight * protein_loss
        + residual_target_weight * residual_huber
        + residual_l2_weight * residual_l2
    )
    return {
        "loss": total,
        "protein_huber": protein_huber,
        "protein_rank": protein_rank,
        "site_huber": site_huber,
        "site_rank": site_rank,
        "residual_huber": residual_huber,
        "residual_l2": residual_l2,
    }


def protein_translation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    rank_weight: float = 0.05,
    rank_temperature: float = 0.20,
    rank_max_targets: int = 2048,
) -> dict[str, torch.Tensor]:
    """Protein-only objective evaluated solely at observed target positions."""

    huber = masked_huber(prediction, target, mask)
    rank = masked_pairwise_rank_loss(
        prediction,
        target,
        mask,
        temperature=rank_temperature,
        max_targets=rank_max_targets,
    )
    total = huber + rank_weight * rank
    return {"loss": total, "protein_huber": huber, "protein_rank": rank}

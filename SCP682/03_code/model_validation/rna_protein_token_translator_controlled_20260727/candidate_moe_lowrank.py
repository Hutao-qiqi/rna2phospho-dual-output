"""RNA-gated mixture of low-rank residual experts for protein prediction."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

try:
    from .candidate_lowrank_cognate import LowRankAxisTranslator
    from .model import RNAEncoder
    from .protein_model import ProteinModelConfig
except ImportError:
    from candidate_lowrank_cognate import LowRankAxisTranslator
    from model import RNAEncoder
    from protein_model import ProteinModelConfig


class LowRankResidualExpert(nn.Module):
    """Translate shared RNA states into one expert's protein residual vector."""

    def __init__(self, config: ProteinModelConfig, rank: int) -> None:
        super().__init__()
        self.translator = LowRankAxisTranslator(
            n_genes=config.n_genes,
            n_proteins=config.n_proteins,
            rank=rank,
            dropout=config.dropout,
        )
        self.output_norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, 1)

    def forward(self, rna_hidden: Tensor, protein_identity: Tensor) -> Tensor:
        translated = self.translator(rna_hidden)
        return self.output_head(
            self.output_norm(translated + protein_identity)
        ).squeeze(-1)


class RNAStateGate(nn.Module):
    """Compute sample-specific expert weights from that sample's RNA state."""

    def __init__(self, d_model: int, n_experts: int, hidden: int) -> None:
        super().__init__()
        if n_experts < 2:
            raise ValueError("n_experts must be at least 2")
        if hidden < 1:
            raise ValueError("gate hidden dimension must be positive")
        self.network = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_experts),
        )
        # Begin from an equal mixture so no expert receives an arbitrary prior.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, rna_hidden: Tensor, valid_mask: Tensor) -> Tensor:
        if rna_hidden.ndim != 3:
            raise ValueError("rna_hidden must be three-dimensional")
        if valid_mask.shape != rna_hidden.shape[:2]:
            raise ValueError("valid_mask must match the RNA token axis")
        weights = valid_mask.unsqueeze(-1).to(rna_hidden.dtype)
        pooled = (rna_hidden * weights).sum(dim=1)
        pooled = pooled / weights.sum(dim=1).clamp_min(1.0)
        return torch.softmax(self.network(pooled), dim=-1)


class MoELowRankProteinTranslator(nn.Module):
    """Predict a frozen cognate anchor plus an RNA-gated expert mixture.

    For sample ``i`` and protein ``j`` the model computes

    ``prediction_ij = anchor_ij + sum_k gate_ik * residual_ikj``.

    The gate sees only the current sample's encoded RNA tokens. Study labels,
    cancer labels and protein observations are absent from the forward path.
    """

    def __init__(
        self,
        config: ProteinModelConfig,
        parent_gene_index: Tensor | Sequence[int],
        anchor_intercept: Tensor | Sequence[float],
        anchor_slope: Tensor | Sequence[float],
        *,
        rank: int = 256,
        n_experts: int = 4,
        gate_hidden: int = 64,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        parent_index = torch.as_tensor(parent_gene_index, dtype=torch.long)
        intercept = torch.as_tensor(anchor_intercept, dtype=torch.float32)
        slope = torch.as_tensor(anchor_slope, dtype=torch.float32)
        expected = (config.n_proteins,)
        if parent_index.shape != expected:
            raise ValueError("parent_gene_index must contain one entry per protein")
        if intercept.shape != expected or slope.shape != expected:
            raise ValueError("anchor coefficients must contain one entry per protein")
        if (parent_index < -1).any() or (parent_index >= config.n_genes).any():
            raise ValueError("parent_gene_index contains an invalid RNA index")
        if not torch.isfinite(intercept).all() or not torch.isfinite(slope).all():
            raise ValueError("anchor coefficients must be finite")

        self.config = config
        self.rank = int(rank)
        self.n_experts = int(n_experts)
        self.rna_encoder = RNAEncoder(config)
        self.protein_identity = nn.Embedding(config.n_proteins, config.d_model)
        self.experts = nn.ModuleList(
            LowRankResidualExpert(config, rank=self.rank)
            for _ in range(self.n_experts)
        )
        self.gate = RNAStateGate(
            d_model=config.d_model,
            n_experts=self.n_experts,
            hidden=gate_hidden,
        )
        self.register_buffer("parent_gene_index", parent_index, persistent=True)
        self.register_buffer("anchor_intercept", intercept, persistent=True)
        self.register_buffer("anchor_slope", slope, persistent=True)
        self.register_buffer(
            "protein_indices",
            torch.arange(config.n_proteins, dtype=torch.long),
            persistent=False,
        )

    def linear_anchor(self, expression: Tensor) -> Tensor:
        if expression.ndim != 2 or expression.shape[1] != self.config.n_genes:
            raise ValueError(
                f"expression must have shape (batch, {self.config.n_genes})"
            )
        safe_index = self.parent_gene_index.clamp_min(0)
        cognate_expression = expression.index_select(1, safe_index)
        anchor = self.anchor_intercept + self.anchor_slope * cognate_expression
        return torch.where(
            self.parent_gene_index.ge(0).view(1, -1),
            anchor,
            torch.zeros_like(anchor),
        )

    def forward(
        self,
        expression: Tensor,
        rna_valid_mask: Tensor | None = None,
        return_hidden: bool = False,
    ) -> dict[str, Tensor]:
        rna_hidden, effective_mask = self.rna_encoder(expression, rna_valid_mask)
        mixture_weights = self.gate(rna_hidden, effective_mask)
        protein_identity = self.protein_identity(self.protein_indices).unsqueeze(0)

        mixed_residual = expression.new_zeros(
            (expression.shape[0], self.config.n_proteins)
        )
        expert_outputs: list[Tensor] = []
        for expert_index, expert in enumerate(self.experts):
            expert_residual = expert(rna_hidden, protein_identity)
            mixed_residual = mixed_residual + (
                mixture_weights[:, expert_index : expert_index + 1]
                * expert_residual
            )
            if return_hidden:
                expert_outputs.append(expert_residual)

        anchor = self.linear_anchor(expression)
        output: dict[str, Tensor] = {"protein": anchor + mixed_residual}
        if return_hidden:
            output.update(
                {
                    "rna_hidden": rna_hidden,
                    "rna_effective_mask": effective_mask,
                    "linear_anchor": anchor,
                    "protein_residual": mixed_residual,
                    "mixture_weights": mixture_weights,
                    "expert_residuals": torch.stack(expert_outputs, dim=1),
                }
            )
        return output

    def fix_projection_matrices_(self) -> "MoELowRankProteinTranslator":
        self.rna_encoder.encoder.fix_projection_matrices_()
        return self

    def parameter_count(self, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )

    def parameter_count_by_component(self) -> dict[str, int]:
        components: dict[str, nn.Module] = {
            "rna_encoder": self.rna_encoder,
            "protein_identity": self.protein_identity,
            "gate": self.gate,
        }
        counts = {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in components.items()
        }
        counts.update(
            {
                f"expert_{index}": sum(
                    parameter.numel() for parameter in expert.parameters()
                )
                for index, expert in enumerate(self.experts)
            }
        )
        return counts


__all__ = [
    "LowRankResidualExpert",
    "MoELowRankProteinTranslator",
    "RNAStateGate",
]

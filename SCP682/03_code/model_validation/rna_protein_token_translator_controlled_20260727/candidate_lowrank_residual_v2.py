"""Second-round low-rank RNA-to-protein translator.

The model keeps the complete gene-token encoder and the cognate-RNA hidden
shortcut from the first low-rank candidate. It adds two controlled changes:

1. The single rank-R axis map is split into several low-rank branches whose
   ranks sum to R. Protein-specific softmax weights combine the branches.
2. A protein-specific linear cognate-RNA residual is added after the protein
   decoder. The effective direct RNA coefficient is therefore inspectable.

No protein-by-gene parameter matrix is created. For G genes, P proteins,
scale ranks (r_1, ..., r_K), and R = sum_k r_k, the multi-scale translator has

    (G + P + 1) * R + P * (K + 1)

parameters. The direct cognate residual adds 3P parameters.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

try:
    from .model import ProteinDecoder, RNAEncoder
    from .protein_model import ProteinModelConfig
except ImportError:
    from model import ProteinDecoder, RNAEncoder
    from protein_model import ProteinModelConfig


def _validate_parent_index(
    parent_gene_index: Tensor | Sequence[int],
    n_genes: int,
    n_proteins: int,
) -> Tensor:
    parent_index = torch.as_tensor(parent_gene_index, dtype=torch.long)
    if parent_index.ndim != 1 or parent_index.numel() != n_proteins:
        raise ValueError("parent_gene_index must contain one entry per protein")
    if (parent_index < -1).any() or (parent_index >= n_genes).any():
        raise ValueError("parent_gene_index entries must be -1 or valid gene indices")
    return parent_index


class LowRankAxisBranch(nn.Module):
    """One nonlinear low-rank map from the complete gene axis to proteins."""

    def __init__(
        self,
        n_genes: int,
        n_proteins: int,
        rank: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if n_genes < 1 or n_proteins < 1 or rank < 1:
            raise ValueError("n_genes, n_proteins and rank must be positive")
        self.n_genes = n_genes
        self.n_proteins = n_proteins
        self.rank = rank
        self.gene_to_rank = nn.Linear(n_genes, rank)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.rank_to_protein = nn.Linear(rank, n_proteins, bias=False)

    def forward(self, rna_hidden: Tensor) -> Tensor:
        if rna_hidden.ndim != 3 or rna_hidden.shape[1] != self.n_genes:
            raise ValueError(
                f"rna_hidden must have shape (batch, {self.n_genes}, dim)"
            )
        position_last = rna_hidden.transpose(1, 2).contiguous()
        latent = self.dropout(self.activation(self.gene_to_rank(position_last)))
        translated = self.rank_to_protein(latent)
        return translated.transpose(1, 2).contiguous()


class MultiScaleLowRankAxisTranslator(nn.Module):
    """Combine several rank-constrained axis maps with per-protein weights.

    The branch ranks form a fixed total rank budget. The scale weights are
    shared across samples and hidden channels, so each protein has an explicit
    and directly exportable preference over coarse and fine RNA summaries.
    """

    def __init__(
        self,
        n_genes: int,
        n_proteins: int,
        ranks: Sequence[int] = (64, 128, 320),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        ranks = tuple(int(rank) for rank in ranks)
        if not ranks or any(rank < 1 for rank in ranks):
            raise ValueError("ranks must contain positive integers")
        if n_genes < 1 or n_proteins < 1:
            raise ValueError("n_genes and n_proteins must be positive")

        self.n_genes = n_genes
        self.n_proteins = n_proteins
        self.ranks = ranks
        self.total_rank = sum(ranks)
        self.branches = nn.ModuleList(
            LowRankAxisBranch(n_genes, n_proteins, rank, dropout)
            for rank in ranks
        )
        self.protein_scale_logits = nn.Parameter(
            torch.zeros(n_proteins, len(ranks))
        )
        self.protein_bias = nn.Parameter(torch.zeros(n_proteins))

    def scale_weights(self) -> Tensor:
        return torch.softmax(self.protein_scale_logits, dim=-1)

    def forward(
        self,
        rna_hidden: Tensor,
        return_components: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        branch_outputs = [branch(rna_hidden) for branch in self.branches]
        weights = self.scale_weights()
        translated = torch.zeros_like(branch_outputs[0])
        for scale_index, branch_output in enumerate(branch_outputs):
            translated = translated + (
                branch_output * weights[:, scale_index].view(1, -1, 1)
            )
        translated = translated + self.protein_bias.view(1, -1, 1)

        if return_components:
            return translated, weights, torch.stack(branch_outputs, dim=1)
        return translated

    def expected_parameter_count(self) -> int:
        n_scales = len(self.ranks)
        return (
            (self.n_genes + self.n_proteins + 1) * self.total_rank
            + self.n_proteins * (n_scales + 1)
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class ProteinSpecificCognateResidual(nn.Module):
    """Add an inspectable protein-specific linear cognate-RNA correction.

    For protein j with mapped cognate RNA x_j, the correction is

        delta_j = sigmoid(gamma_j) * (a_j + beta_j * x_j).

    Unmapped or masked cognate RNA values always contribute zero.
    """

    def __init__(
        self,
        n_genes: int,
        n_proteins: int,
        parent_gene_index: Tensor | Sequence[int],
        initial_gate: float = 0.20,
    ) -> None:
        super().__init__()
        if not 0.0 < initial_gate < 1.0:
            raise ValueError("initial_gate must lie strictly between zero and one")
        parent_index = _validate_parent_index(
            parent_gene_index,
            n_genes=n_genes,
            n_proteins=n_proteins,
        )
        self.n_genes = n_genes
        self.n_proteins = n_proteins
        self.cognate_intercept = nn.Parameter(torch.zeros(n_proteins))
        self.cognate_slope = nn.Parameter(torch.ones(n_proteins))
        gate_logit = math.log(initial_gate / (1.0 - initial_gate))
        self.cognate_gate_logit = nn.Parameter(
            torch.full((n_proteins,), gate_logit)
        )
        self.register_buffer("parent_gene_index", parent_index, persistent=True)
        self.register_buffer(
            "has_cognate",
            parent_index.ge(0),
            persistent=True,
        )

    def gather_expression(
        self,
        expression: Tensor,
        rna_effective_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if expression.ndim != 2 or expression.shape[1] != self.n_genes:
            raise ValueError(
                f"expression must have shape (batch, {self.n_genes})"
            )
        if rna_effective_mask.shape != expression.shape:
            raise ValueError("rna_effective_mask must match expression")
        safe_index = self.parent_gene_index.clamp_min(0)
        cognate_expression = expression.index_select(1, safe_index)
        cognate_valid = (
            rna_effective_mask.index_select(1, safe_index)
            & self.has_cognate.unsqueeze(0)
        )
        cognate_expression = cognate_expression * cognate_valid.to(expression.dtype)
        return cognate_expression, cognate_valid

    def forward(
        self,
        expression: Tensor,
        rna_effective_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        cognate_expression, cognate_valid = self.gather_expression(
            expression,
            rna_effective_mask,
        )
        gate = torch.sigmoid(self.cognate_gate_logit).unsqueeze(0)
        gate = gate * cognate_valid.to(expression.dtype)
        linear_term = (
            self.cognate_intercept.unsqueeze(0)
            + self.cognate_slope.unsqueeze(0) * cognate_expression
        )
        residual = gate * linear_term
        effective_slope = (
            torch.sigmoid(self.cognate_gate_logit) * self.cognate_slope
        )
        effective_slope = effective_slope * self.has_cognate.to(expression.dtype)
        return residual, gate, cognate_expression, effective_slope

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class LowRankResidualProteinTranslatorV2(nn.Module):
    """Multi-scale low-rank translator with protected cognate-RNA residual."""

    def __init__(
        self,
        config: ProteinModelConfig,
        parent_gene_index: Tensor | Sequence[int],
        ranks: Sequence[int] = (64, 128, 320),
        initial_cognate_gate: float = 0.20,
    ) -> None:
        super().__init__()
        if config.n_genes < 1 or config.n_proteins < 1:
            raise ValueError("n_genes and n_proteins must be positive")
        parent_index = _validate_parent_index(
            parent_gene_index,
            n_genes=config.n_genes,
            n_proteins=config.n_proteins,
        )

        self.config = config
        self.ranks = tuple(int(rank) for rank in ranks)
        self.rna_encoder = RNAEncoder(config)
        self.global_translator = MultiScaleLowRankAxisTranslator(
            n_genes=config.n_genes,
            n_proteins=config.n_proteins,
            ranks=self.ranks,
            dropout=config.dropout,
        )
        self.cognate_projection = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
        )
        self.fusion_gate = nn.Linear(2 * config.d_model, config.d_model)
        self.fusion_norm = nn.LayerNorm(config.d_model)
        self.protein_decoder = ProteinDecoder(config)
        self.cognate_residual = ProteinSpecificCognateResidual(
            n_genes=config.n_genes,
            n_proteins=config.n_proteins,
            parent_gene_index=parent_index,
            initial_gate=initial_cognate_gate,
        )

        self.register_buffer("parent_gene_index", parent_index, persistent=True)
        self.register_buffer(
            "has_cognate",
            parent_index.ge(0),
            persistent=True,
        )

    def encode_rna(
        self,
        expression: Tensor,
        rna_valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        return self.rna_encoder(expression, rna_valid_mask)

    def _gather_cognate_hidden(self, rna_hidden: Tensor) -> Tensor:
        safe_index = self.parent_gene_index.clamp_min(0)
        cognate_hidden = rna_hidden.index_select(1, safe_index)
        cognate_hidden = self.cognate_projection(cognate_hidden)
        return cognate_hidden * self.has_cognate.view(1, -1, 1).to(
            cognate_hidden.dtype
        )

    def fuse_translations(
        self,
        rna_hidden: Tensor,
        return_components: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
        scale_outputs: Tensor | None = None
        if return_components:
            global_hidden, scale_weights, scale_outputs = self.global_translator(
                rna_hidden,
                return_components=True,
            )
        else:
            global_hidden = self.global_translator(rna_hidden)
            scale_weights = self.global_translator.scale_weights()

        cognate_hidden = self._gather_cognate_hidden(rna_hidden)
        hidden_gate = torch.sigmoid(
            self.fusion_gate(torch.cat((global_hidden, cognate_hidden), dim=-1))
        )
        hidden_gate = torch.where(
            self.has_cognate.view(1, -1, 1),
            hidden_gate,
            torch.ones_like(hidden_gate),
        )
        fused = self.fusion_norm(
            hidden_gate * global_hidden + (1.0 - hidden_gate) * cognate_hidden
        )
        return fused, hidden_gate, global_hidden, cognate_hidden, scale_outputs

    def forward(
        self,
        expression: Tensor,
        rna_valid_mask: Tensor | None = None,
        return_hidden: bool = False,
    ) -> dict[str, Tensor]:
        rna_hidden, effective_mask = self.encode_rna(expression, rna_valid_mask)
        (
            fused,
            hidden_gate,
            global_hidden,
            cognate_hidden,
            scale_outputs,
        ) = self.fuse_translations(rna_hidden, return_components=return_hidden)
        decoder_prediction, protein_hidden = self.protein_decoder(fused)
        (
            direct_residual,
            direct_gate,
            cognate_expression,
            effective_cognate_slope,
        ) = self.cognate_residual(expression, effective_mask)
        protein_prediction = decoder_prediction + direct_residual

        output: dict[str, Tensor] = {"protein": protein_prediction}
        if return_hidden:
            if scale_outputs is None:
                raise RuntimeError("scale components were not collected")
            output.update(
                {
                    "rna_hidden": rna_hidden,
                    "rna_effective_mask": effective_mask,
                    "global_hidden": global_hidden,
                    "cognate_hidden": cognate_hidden,
                    "hidden_fusion_gate": hidden_gate,
                    "scale_weights": self.global_translator.scale_weights(),
                    "scale_outputs": scale_outputs,
                    "protein_hidden": protein_hidden,
                    "decoder_prediction": decoder_prediction,
                    "cognate_expression": cognate_expression,
                    "cognate_direct_gate": direct_gate,
                    "cognate_direct_residual": direct_residual,
                    "effective_cognate_slope": effective_cognate_slope,
                }
            )
        return output

    def fix_projection_matrices_(self) -> "LowRankResidualProteinTranslatorV2":
        self.rna_encoder.encoder.fix_projection_matrices_()
        self.protein_decoder.decoder.fix_projection_matrices_()
        return self

    def parameter_count(self, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )

    def parameter_count_by_component(self) -> dict[str, int]:
        components = {
            "rna_encoder": self.rna_encoder,
            "global_translator": self.global_translator,
            "cognate_projection": self.cognate_projection,
            "fusion_gate": self.fusion_gate,
            "fusion_norm": self.fusion_norm,
            "protein_decoder": self.protein_decoder,
            "cognate_residual": self.cognate_residual,
        }
        return {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in components.items()
        }


__all__ = [
    "LowRankAxisBranch",
    "LowRankResidualProteinTranslatorV2",
    "MultiScaleLowRankAxisTranslator",
    "ProteinSpecificCognateResidual",
]

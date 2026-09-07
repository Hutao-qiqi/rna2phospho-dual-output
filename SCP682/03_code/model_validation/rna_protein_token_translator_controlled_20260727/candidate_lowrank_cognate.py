"""R1: low-rank global translation with a cognate-RNA gated shortcut."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

try:
    from .model import ProteinDecoder, RNAEncoder
    from .protein_model import ProteinModelConfig
except ImportError:
    from model import ProteinDecoder, RNAEncoder
    from protein_model import ProteinModelConfig


class LowRankAxisTranslator(nn.Module):
    """Map the complete gene axis to the protein axis through a small rank."""

    def __init__(
        self,
        n_genes: int,
        n_proteins: int,
        rank: int = 512,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if n_genes < 1 or n_proteins < 1 or rank < 1:
            raise ValueError("n_genes, n_proteins and rank must be positive")
        self.n_genes = n_genes
        self.n_proteins = n_proteins
        self.rank = rank
        self.layers = nn.Sequential(
            nn.Linear(n_genes, rank),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rank, n_proteins),
        )

    def forward(self, rna_hidden: Tensor) -> Tensor:
        if rna_hidden.ndim != 3 or rna_hidden.shape[1] != self.n_genes:
            raise ValueError(
                f"rna_hidden must have shape (batch, {self.n_genes}, dim)"
            )
        translated = self.layers(rna_hidden.transpose(1, 2).contiguous())
        return translated.transpose(1, 2).contiguous()

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class LowRankCognateProteinTranslator(nn.Module):
    """Encode RNA and fuse global translation with cognate-gene states.

    ``parent_gene_index[j]`` gives the RNA-axis index of protein ``j``.
    A value of ``-1`` denotes a protein without a mapped cognate RNA. Such
    proteins use the global low-rank branch only.
    """

    def __init__(
        self,
        config: ProteinModelConfig,
        parent_gene_index: Tensor | Sequence[int],
        rank: int = 512,
    ) -> None:
        super().__init__()
        if config.n_genes < 1 or config.n_proteins < 1:
            raise ValueError("n_genes and n_proteins must be positive")

        parent_index = torch.as_tensor(parent_gene_index, dtype=torch.long)
        if parent_index.ndim != 1 or parent_index.numel() != config.n_proteins:
            raise ValueError(
                "parent_gene_index must contain one entry for every protein"
            )
        if (parent_index < -1).any() or (parent_index >= config.n_genes).any():
            raise ValueError("parent_gene_index values must be -1 or valid gene indices")

        self.config = config
        self.rank = rank
        self.rna_encoder = RNAEncoder(config)
        self.global_translator = LowRankAxisTranslator(
            n_genes=config.n_genes,
            n_proteins=config.n_proteins,
            rank=rank,
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

        self.register_buffer("parent_gene_index", parent_index, persistent=True)
        self.register_buffer(
            "has_cognate",
            parent_index.ge(0),
            persistent=True,
        )

    def _gather_cognate(self, rna_hidden: Tensor) -> Tensor:
        safe_index = self.parent_gene_index.clamp_min(0)
        cognate = rna_hidden.index_select(dim=1, index=safe_index)
        cognate = self.cognate_projection(cognate)
        return cognate * self.has_cognate.view(1, -1, 1).to(cognate.dtype)

    def encode_rna(
        self,
        expression: Tensor,
        rna_valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        return self.rna_encoder(expression, rna_valid_mask)

    def fuse_translations(
        self,
        rna_hidden: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        global_hidden = self.global_translator(rna_hidden)
        cognate_hidden = self._gather_cognate(rna_hidden)
        gate = torch.sigmoid(
            self.fusion_gate(torch.cat((global_hidden, cognate_hidden), dim=-1))
        )

        # Unmapped proteins must use the low-rank global state exclusively.
        availability = self.has_cognate.view(1, -1, 1)
        gate = torch.where(availability, gate, torch.ones_like(gate))
        fused = self.fusion_norm(gate * global_hidden + (1.0 - gate) * cognate_hidden)
        return fused, gate, global_hidden, cognate_hidden

    def forward(
        self,
        expression: Tensor,
        rna_valid_mask: Tensor | None = None,
        return_hidden: bool = False,
    ) -> dict[str, Tensor]:
        rna_hidden, effective_mask = self.encode_rna(expression, rna_valid_mask)
        fused, gate, global_hidden, cognate_hidden = self.fuse_translations(rna_hidden)
        protein_prediction, protein_hidden = self.protein_decoder(fused)

        output: dict[str, Tensor] = {"protein": protein_prediction}
        if return_hidden:
            output.update(
                {
                    "rna_hidden": rna_hidden,
                    "rna_effective_mask": effective_mask,
                    "global_hidden": global_hidden,
                    "cognate_hidden": cognate_hidden,
                    "fusion_gate": gate,
                    "protein_hidden": protein_hidden,
                }
            )
        return output

    def fix_projection_matrices_(self) -> "LowRankCognateProteinTranslator":
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
        }
        return {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in components.items()
        }


__all__ = ["LowRankAxisTranslator", "LowRankCognateProteinTranslator"]

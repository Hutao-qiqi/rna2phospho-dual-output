"""Protein-only Performer translator from fixed-order bulk RNA tokens."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

try:
    from .model import AxisMLPTranslator, ProteinDecoder, RNAEncoder
except ImportError:
    from model import AxisMLPTranslator, ProteinDecoder, RNAEncoder


@dataclass(frozen=True)
class ProteinModelConfig:
    """Architecture of the RNA-to-total-protein model.

    The RNA and protein axes are complete fixed vocabularies. The translator
    acts on the position axis independently for every hidden channel, matching
    the public scTranslator construction without its 1,000-protein cap.
    """

    n_genes: int
    n_proteins: int
    d_model: int = 128
    encoder_depth: int = 2
    decoder_depth: int = 2
    n_heads: int = 8
    dim_head: int = 64
    translator_depth: int = 2
    translator_hidden: int = 0
    ff_mult: int = 4
    dropout: float = 0.10
    gene_mask_probability: float = 0.10
    nb_features: int | None = None
    feature_redraw_interval: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RNAProteinPerformerTranslator(nn.Module):
    """Encode all RNA genes, translate the axis and decode all proteins."""

    def __init__(self, config: ProteinModelConfig) -> None:
        super().__init__()
        if config.n_genes < 1 or config.n_proteins < 1:
            raise ValueError("n_genes and n_proteins must be positive")
        self.config = config
        self.rna_encoder = RNAEncoder(config)
        self.cross_modal_translator = AxisMLPTranslator(
            config.n_genes,
            config.n_proteins,
            config.translator_depth,
            config.translator_hidden,
            config.dropout,
        )
        self.protein_decoder = ProteinDecoder(config)

    def encode_rna(
        self,
        expression: Tensor,
        rna_valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        return self.rna_encoder(expression, rna_valid_mask)

    def decode_protein(self, rna_hidden: Tensor) -> tuple[Tensor, Tensor]:
        translated = self.cross_modal_translator(rna_hidden)
        return self.protein_decoder(translated)

    def forward(
        self,
        expression: Tensor,
        rna_valid_mask: Tensor | None = None,
        return_hidden: bool = False,
    ) -> dict[str, Tensor]:
        rna_hidden, effective_mask = self.encode_rna(expression, rna_valid_mask)
        protein_prediction, protein_hidden = self.decode_protein(rna_hidden)
        output: dict[str, Tensor] = {"protein": protein_prediction}
        if return_hidden:
            output.update(
                {
                    "rna_hidden": rna_hidden,
                    "rna_effective_mask": effective_mask,
                    "protein_hidden": protein_hidden,
                }
            )
        return output

    def fix_projection_matrices_(self) -> "RNAProteinPerformerTranslator":
        self.rna_encoder.encoder.fix_projection_matrices_()
        self.protein_decoder.decoder.fix_projection_matrices_()
        return self

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

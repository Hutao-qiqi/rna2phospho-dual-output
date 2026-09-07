"""TCGA-CPTAC hybrid direct total-protein translator.

The model keeps the local STRING and global RNA state readers, adds a
controlled scTranslator-style position-axis map, and propagates shared RNA
module states back to every protein. Total protein is predicted directly;
there is no cognate linear anchor and no total-protein residual head.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn

try:
    from .candidate_multiscale_protein_query import (
        ExactLocalGeneAttention,
        LocalGenePrior,
    )
    from .performer_core import PerformerCrossAttention, PerformerEncoder
except ImportError:
    from candidate_multiscale_protein_query import (
        ExactLocalGeneAttention,
        LocalGenePrior,
    )
    from performer_core import PerformerCrossAttention, PerformerEncoder


CPTAC_PLATFORM = 0
TCPA_PLATFORM = 1


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("softplus target must be positive")
    return math.log(math.expm1(value))


@dataclass(frozen=True)
class HybridProteinConfig:
    n_genes: int
    n_proteins: int
    n_cancers: int
    n_platforms: int = 2
    d_model: int = 128
    encoder_depth: int = 2
    n_heads: int = 4
    dim_head: int = 32
    ff_mult: int = 2
    dropout: float = 0.10
    gene_mask_probability: float = 0.0
    nb_features: int | None = 128
    feature_redraw_interval: int | None = None
    n_global_states: int = 128
    map_hidden: int = 512
    map_states: int = 256
    n_module_states: int = 64
    module_depth: int = 2
    max_local_genes: int = 64
    protein_chunk: int = 512
    branch_dropout: float = 0.05
    initial_map_gate: float = 0.10
    initial_local_gate: float = 0.45
    initial_global_gate: float = 0.45
    use_direct_cognate: bool = False
    initial_cognate_gate: float = 0.0
    use_direct_local_linear: bool = False
    max_direct_local_genes: int = 1
    initial_local_linear_mix: float = 0.0
    use_direct_global_linear: bool = False
    direct_global_components: int = 1
    initial_global_linear_mix: float = 0.0
    use_direct_supervised_linear: bool = False
    max_direct_supervised_genes: int = 1
    initial_supervised_linear_mix: float = 0.0
    joint_direct_value_fusion: bool = False
    initial_joint_neural_weight: float = 0.0

    def __post_init__(self) -> None:
        integer_fields = (
            self.n_genes,
            self.n_proteins,
            self.n_cancers,
            self.n_platforms,
            self.d_model,
            self.encoder_depth,
            self.n_heads,
            self.dim_head,
            self.ff_mult,
            self.n_global_states,
            self.map_hidden,
            self.map_states,
            self.n_module_states,
            self.module_depth,
            self.max_local_genes,
            self.max_direct_local_genes,
            self.direct_global_components,
            self.max_direct_supervised_genes,
            self.protein_chunk,
        )
        if min(integer_fields) < 1:
            raise ValueError("all architecture dimensions must be positive")
        if not 0.0 <= self.branch_dropout < 1.0:
            raise ValueError("branch_dropout must be in [0, 1)")
        gates = (
            self.initial_map_gate,
            self.initial_local_gate,
            self.initial_global_gate,
        )
        if self.use_direct_cognate:
            gates = (*gates, self.initial_cognate_gate)
        elif self.initial_cognate_gate != 0.0:
            raise ValueError("initial_cognate_gate requires use_direct_cognate")
        if min(gates) <= 0 or not math.isclose(sum(gates), 1.0, abs_tol=1e-6):
            raise ValueError("initial branch gates must be positive and sum to one")
        if self.use_direct_local_linear:
            if not 0.0 < self.initial_local_linear_mix < 1.0:
                raise ValueError("initial local linear mix must lie in (0, 1)")
        elif self.initial_local_linear_mix != 0.0:
            raise ValueError(
                "initial_local_linear_mix requires use_direct_local_linear"
            )
        if self.use_direct_global_linear:
            if not 0.0 < self.initial_global_linear_mix < 1.0:
                raise ValueError("initial global linear mix must lie in (0, 1)")
        elif self.initial_global_linear_mix != 0.0:
            raise ValueError(
                "initial_global_linear_mix requires use_direct_global_linear"
            )
        if self.use_direct_supervised_linear:
            if not 0.0 < self.initial_supervised_linear_mix < 1.0:
                raise ValueError("initial supervised linear mix must lie in (0, 1)")
        elif self.initial_supervised_linear_mix != 0.0:
            raise ValueError(
                "initial_supervised_linear_mix requires use_direct_supervised_linear"
            )
        direct_branches = sum(
            (
                self.use_direct_local_linear,
                self.use_direct_global_linear,
                self.use_direct_supervised_linear,
            )
        )
        if self.joint_direct_value_fusion:
            if direct_branches < 1:
                raise ValueError("joint direct value fusion requires a direct branch")
            if not 0.0 < self.initial_joint_neural_weight < 1.0:
                raise ValueError("initial joint neural weight must lie in (0, 1)")
        elif self.initial_joint_neural_weight != 0.0:
            raise ValueError(
                "initial_joint_neural_weight requires joint_direct_value_fusion"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConditionedRNAEncoder(nn.Module):
    """RNA value and identity tokens conditioned on platform and cancer."""

    def __init__(self, config: HybridProteinConfig) -> None:
        super().__init__()
        self.config = config
        self.gene_identity = nn.Embedding(config.n_genes, config.d_model)
        self.expression_projection = nn.Linear(1, config.d_model)
        self.platform_identity = nn.Embedding(config.n_platforms, config.d_model)
        self.cancer_identity = nn.Embedding(config.n_cancers, config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.encoder = PerformerEncoder(
            dim=config.d_model,
            depth=config.encoder_depth,
            heads=config.n_heads,
            dim_head=config.dim_head,
            dropout=config.dropout,
            ff_mult=config.ff_mult,
            nb_features=config.nb_features,
            feature_redraw_interval=config.feature_redraw_interval,
        )
        self.register_buffer(
            "gene_indices",
            torch.arange(config.n_genes, dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        expression: Tensor,
        platform_index: Tensor,
        cancer_index: Tensor,
        valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if expression.ndim != 2 or expression.shape[1] != self.config.n_genes:
            raise ValueError("expression has an incompatible shape")
        batch = expression.shape[0]
        if platform_index.shape != (batch,) or cancer_index.shape != (batch,):
            raise ValueError("platform and cancer indices require one value per sample")
        if valid_mask is None:
            effective = torch.ones_like(expression, dtype=torch.bool)
        else:
            if valid_mask.shape != expression.shape:
                raise ValueError("valid_mask must match expression")
            effective = valid_mask.bool()
        if self.training and self.config.gene_mask_probability > 0:
            keep = torch.rand_like(expression) >= self.config.gene_mask_probability
            effective = effective & keep
            empty = ~effective.any(dim=1)
            if empty.any():
                effective = effective.clone()
                effective[empty] = valid_mask[empty] if valid_mask is not None else True

        tokens = self.gene_identity(self.gene_indices).unsqueeze(0)
        tokens = tokens + self.expression_projection(expression.unsqueeze(-1))
        tokens = self.input_norm(tokens)
        condition = self.platform_identity(platform_index)
        condition = condition + self.cancer_identity(cancer_index)
        tokens = self.dropout(tokens + condition.unsqueeze(1))
        hidden = self.encoder(tokens, mask=effective)
        # A masked gene identity can remain non-zero after attention. The
        # position-axis map and every downstream reader receive explicit zeros.
        hidden = hidden * effective.unsqueeze(-1).to(hidden.dtype)
        return hidden, effective


class ControlledPositionMap(nn.Module):
    """Map the complete RNA position axis to 256 supplementary context states."""

    def __init__(self, config: HybridProteinConfig) -> None:
        super().__init__()
        self.n_genes = config.n_genes
        self.layers = nn.Sequential(
            nn.Linear(config.n_genes, config.map_hidden),
            nn.Dropout(config.dropout),
            nn.ReLU(),
            nn.Linear(config.map_hidden, config.map_states),
        )

    def forward(self, masked_rna_hidden: Tensor) -> Tensor:
        if masked_rna_hidden.ndim != 3 or masked_rna_hidden.shape[1] != self.n_genes:
            raise ValueError("masked RNA hidden state has an incompatible shape")
        mapped = self.layers(masked_rna_hidden.transpose(1, 2).contiguous())
        return mapped.transpose(1, 2).contiguous()


def _attention_kwargs(config: HybridProteinConfig) -> dict[str, Any]:
    return {
        "dim": config.d_model,
        "heads": config.n_heads,
        "dim_head": config.dim_head,
        "dropout": config.dropout,
        "nb_features": config.nb_features,
        "feature_redraw_interval": config.feature_redraw_interval,
    }


class TCGACPTACHybridProteinTranslator(nn.Module):
    """Direct multi-context total-protein predictor with platform calibration."""

    def __init__(
        self,
        config: HybridProteinConfig,
        local_prior: LocalGenePrior,
        *,
        direct_local_prior: LocalGenePrior | None = None,
        direct_supervised_prior: LocalGenePrior | None = None,
        tcpa_calibrated_protein_mask: Tensor | Sequence[bool] | None = None,
    ) -> None:
        super().__init__()
        if local_prior.n_targets != config.n_proteins:
            raise ValueError("local prior must contain one row per protein")
        if local_prior.max_local_genes != config.max_local_genes:
            raise ValueError("local prior width differs from config")
        if config.use_direct_local_linear:
            if direct_local_prior is None:
                raise ValueError("direct local linear branch requires its local prior")
            if direct_local_prior.n_targets != config.n_proteins:
                raise ValueError("direct local prior must contain one row per protein")
            if direct_local_prior.max_local_genes != config.max_direct_local_genes:
                raise ValueError("direct local prior width differs from config")
        elif direct_local_prior is not None:
            raise ValueError("direct local prior requires direct local linear branch")
        if config.use_direct_supervised_linear:
            if direct_supervised_prior is None:
                raise ValueError(
                    "direct supervised linear branch requires its local prior"
                )
            if direct_supervised_prior.n_targets != config.n_proteins:
                raise ValueError(
                    "direct supervised prior must contain one row per protein"
                )
            if (
                direct_supervised_prior.max_local_genes
                != config.max_direct_supervised_genes
            ):
                raise ValueError("direct supervised prior width differs from config")
        elif direct_supervised_prior is not None:
            raise ValueError(
                "direct supervised prior requires direct supervised linear branch"
            )
        self.config = config
        self.local_prior_metadata = dict(local_prior.metadata)
        self.direct_local_prior_metadata = (
            dict(direct_local_prior.metadata) if direct_local_prior is not None else None
        )
        self.direct_supervised_prior_metadata = (
            dict(direct_supervised_prior.metadata)
            if direct_supervised_prior is not None
            else None
        )
        attention = _attention_kwargs(config)
        self.rna_encoder = ConditionedRNAEncoder(config)
        self.position_map = ControlledPositionMap(config)
        self.protein_identity = nn.Embedding(config.n_proteins, config.d_model)
        self.local_reader = ExactLocalGeneAttention(
            config.d_model,
            config.n_heads,
            config.dim_head,
            config.dropout,
            local_prior,
        )

        self.global_queries = nn.Parameter(
            torch.empty(config.n_global_states, config.d_model)
        )
        self.global_reader = PerformerCrossAttention(**attention)
        self.global_encoder = PerformerEncoder(
            dim=config.d_model,
            depth=1,
            heads=config.n_heads,
            dim_head=config.dim_head,
            dropout=config.dropout,
            ff_mult=config.ff_mult,
            nb_features=config.nb_features,
            feature_redraw_interval=config.feature_redraw_interval,
        )
        self.map_reader = PerformerCrossAttention(**attention)
        self.protein_global_reader = PerformerCrossAttention(**attention)

        self.map_projection = nn.Linear(config.d_model, config.d_model)
        self.local_projection = nn.Linear(config.d_model, config.d_model)
        self.global_projection = nn.Linear(config.d_model, config.d_model)
        if config.use_direct_cognate:
            self.cognate_projection = nn.Sequential(
                nn.Linear(1, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.d_model),
            )
            cognate_index = torch.full((config.n_proteins,), -1, dtype=torch.long)
            prior_index = torch.as_tensor(local_prior.gene_index, dtype=torch.long)
            prior_mask = torch.as_tensor(local_prior.gene_mask, dtype=torch.bool)
            prior_relation = torch.as_tensor(local_prior.relation_index, dtype=torch.long)
            cognate_entries = prior_mask & (prior_relation == 0)
            for protein in range(config.n_proteins):
                positions = torch.nonzero(cognate_entries[protein], as_tuple=False).flatten()
                if positions.numel():
                    cognate_index[protein] = prior_index[protein, int(positions[0])]
            self.register_buffer("cognate_gene_index", cognate_index, persistent=True)
        else:
            self.cognate_projection = None
            self.register_buffer(
                "cognate_gene_index",
                torch.full((config.n_proteins,), -1, dtype=torch.long),
                persistent=False,
            )
        gate_branches = 4 if config.use_direct_cognate else 3
        gate_inputs = 5 if config.use_direct_cognate else 4
        self.gate = nn.Sequential(
            nn.LayerNorm(config.d_model * gate_inputs),
            nn.Linear(config.d_model * gate_inputs, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, gate_branches),
        )
        final_gate = self.gate[-1]
        assert isinstance(final_gate, nn.Linear)
        nn.init.zeros_(final_gate.weight)
        with torch.no_grad():
            final_gate.bias.copy_(
                torch.log(
                    torch.tensor(
                        [
                            config.initial_map_gate,
                            config.initial_local_gate,
                            config.initial_global_gate,
                        ]
                        + (
                            [config.initial_cognate_gate]
                            if config.use_direct_cognate
                            else []
                        )
                    )
                )
            )
        self.fusion_norm = nn.LayerNorm(config.d_model)

        self.module_queries = nn.Parameter(
            torch.empty(config.n_module_states, config.d_model)
        )
        self.module_reader = PerformerCrossAttention(**attention)
        self.module_encoder = PerformerEncoder(
            dim=config.d_model,
            depth=config.module_depth,
            heads=config.n_heads,
            dim_head=config.dim_head,
            dropout=config.dropout,
            ff_mult=config.ff_mult,
            nb_features=config.nb_features,
            feature_redraw_interval=config.feature_redraw_interval,
        )
        self.protein_module_reader = PerformerCrossAttention(**attention)
        self.output_norm = nn.LayerNorm(config.d_model)
        self.shared_output = nn.Linear(config.d_model, 1)
        if config.use_direct_local_linear:
            assert direct_local_prior is not None
            self.register_buffer(
                "direct_local_gene_index",
                torch.as_tensor(direct_local_prior.gene_index, dtype=torch.long),
                persistent=True,
            )
            self.register_buffer(
                "direct_local_gene_mask",
                torch.as_tensor(direct_local_prior.gene_mask, dtype=torch.bool),
                persistent=True,
            )
            self.direct_local_weight = nn.Parameter(
                torch.zeros(config.n_proteins, config.max_direct_local_genes)
            )
            self.direct_local_bias = nn.Parameter(torch.zeros(config.n_proteins))
            initial_logit = math.log(
                config.initial_local_linear_mix
                / (1.0 - config.initial_local_linear_mix)
            )
            self.direct_local_mix_logit = nn.Parameter(
                torch.full((config.n_proteins,), initial_logit)
            )
        else:
            self.register_buffer(
                "direct_local_gene_index",
                torch.full((config.n_proteins, 1), -1, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "direct_local_gene_mask",
                torch.zeros((config.n_proteins, 1), dtype=torch.bool),
                persistent=False,
            )
            self.register_parameter("direct_local_weight", None)
            self.register_parameter("direct_local_bias", None)
            self.register_parameter("direct_local_mix_logit", None)
        if config.use_direct_global_linear:
            self.register_buffer(
                "direct_global_projection",
                torch.zeros(config.n_genes, config.direct_global_components),
                persistent=True,
            )
            self.direct_global_weight = nn.Parameter(
                torch.zeros(config.n_proteins, config.direct_global_components)
            )
            self.direct_global_bias = nn.Parameter(torch.zeros(config.n_proteins))
            initial_global_logit = math.log(
                config.initial_global_linear_mix
                / (1.0 - config.initial_global_linear_mix)
            )
            self.direct_global_mix_logit = nn.Parameter(
                torch.full((config.n_proteins,), initial_global_logit)
            )
        else:
            self.register_buffer(
                "direct_global_projection",
                torch.zeros(config.n_genes, 1),
                persistent=False,
            )
            self.register_parameter("direct_global_weight", None)
            self.register_parameter("direct_global_bias", None)
            self.register_parameter("direct_global_mix_logit", None)
        if config.use_direct_supervised_linear:
            assert direct_supervised_prior is not None
            self.register_buffer(
                "direct_supervised_gene_index",
                torch.as_tensor(direct_supervised_prior.gene_index, dtype=torch.long),
                persistent=True,
            )
            self.register_buffer(
                "direct_supervised_gene_mask",
                torch.as_tensor(direct_supervised_prior.gene_mask, dtype=torch.bool),
                persistent=True,
            )
            self.direct_supervised_weight = nn.Parameter(
                torch.zeros(config.n_proteins, config.max_direct_supervised_genes)
            )
            self.direct_supervised_bias = nn.Parameter(
                torch.zeros(config.n_proteins)
            )
            initial_supervised_logit = math.log(
                config.initial_supervised_linear_mix
                / (1.0 - config.initial_supervised_linear_mix)
            )
            self.direct_supervised_mix_logit = nn.Parameter(
                torch.full((config.n_proteins,), initial_supervised_logit)
            )
        else:
            self.register_buffer(
                "direct_supervised_gene_index",
                torch.full((config.n_proteins, 1), -1, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "direct_supervised_gene_mask",
                torch.zeros((config.n_proteins, 1), dtype=torch.bool),
                persistent=False,
            )
            self.register_parameter("direct_supervised_weight", None)
            self.register_parameter("direct_supervised_bias", None)
            self.register_parameter("direct_supervised_mix_logit", None)

        if config.joint_direct_value_fusion:
            sequential_weights = [1.0]
            for enabled, mix in (
                (config.use_direct_local_linear, config.initial_local_linear_mix),
                (config.use_direct_global_linear, config.initial_global_linear_mix),
                (
                    config.use_direct_supervised_linear,
                    config.initial_supervised_linear_mix,
                ),
            ):
                if not enabled:
                    continue
                sequential_weights = [
                    value * (1.0 - mix) for value in sequential_weights
                ] + [mix]
            old_direct_total = 1.0 - sequential_weights[0]
            if old_direct_total <= 0:
                raise ValueError("joint direct value fusion has no direct mass")
            direct_scale = (
                (1.0 - config.initial_joint_neural_weight) / old_direct_total
            )
            joint_weights = [config.initial_joint_neural_weight] + [
                value * direct_scale for value in sequential_weights[1:]
            ]
            self.direct_joint_mix_logits = nn.Parameter(
                torch.log(torch.tensor(joint_weights, dtype=torch.float32))
                .unsqueeze(0)
                .expand(config.n_proteins, -1)
                .clone()
            )
        else:
            self.register_parameter("direct_joint_mix_logits", None)

        nn.init.normal_(self.global_queries, std=config.d_model**-0.5)
        nn.init.normal_(self.module_queries, std=config.d_model**-0.5)
        if tcpa_calibrated_protein_mask is None:
            calibration_mask = torch.zeros(config.n_proteins, dtype=torch.bool)
        else:
            calibration_mask = torch.as_tensor(
                tcpa_calibrated_protein_mask, dtype=torch.bool
            )
            if calibration_mask.shape != (config.n_proteins,):
                raise ValueError("TCPA calibration mask must match protein vocabulary")
        self.tcpa_scale_raw = nn.Parameter(
            torch.full(
                (config.n_proteins,),
                _inverse_softplus(1.0),
                dtype=torch.float32,
            )
        )
        self.tcpa_bias = nn.Parameter(torch.zeros(config.n_proteins))
        self.register_buffer(
            "tcpa_calibrated_protein_mask", calibration_mask, persistent=True
        )
        self.register_buffer(
            "protein_indices",
            torch.arange(config.n_proteins, dtype=torch.long),
            persistent=False,
        )

    def _branch_dropout_mask(
        self,
        logits: Tensor,
        local_valid: Tensor,
        cognate_valid: Tensor | None = None,
    ) -> Tensor:
        available = torch.ones_like(logits, dtype=torch.bool)
        available[..., 1] = local_valid
        if self.config.use_direct_cognate:
            if cognate_valid is None:
                raise ValueError("cognate validity is required")
            available[..., 3] = cognate_valid
        if not self.training or self.config.branch_dropout == 0:
            return available
        keep = torch.rand_like(logits) >= self.config.branch_dropout
        keep &= available
        empty = ~keep.any(dim=-1)
        if empty.any():
            keep[..., 2] |= empty  # global context is always available
        return keep

    def _shared_states(
        self, rna_hidden: Tensor, rna_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch = rna_hidden.shape[0]
        mapped = self.position_map(rna_hidden)
        global_query = self.global_queries.unsqueeze(0).expand(batch, -1, -1)
        global_states = global_query + self.global_reader(
            global_query, rna_hidden, context_mask=rna_mask
        )
        global_states = self.global_encoder(global_states)
        module_query = self.module_queries.unsqueeze(0).expand(batch, -1, -1)
        module_states = module_query + self.module_reader(
            module_query, rna_hidden, context_mask=rna_mask
        )
        module_states = self.module_encoder(module_states)
        return mapped, global_states, module_states

    def _decode_chunk(
        self,
        selected: Tensor,
        expression: Tensor,
        rna_hidden: Tensor,
        rna_mask: Tensor,
        mapped_states: Tensor,
        global_states: Tensor,
        module_states: Tensor,
        direct_global_features: Tensor | None,
    ) -> dict[str, Tensor]:
        batch = rna_hidden.shape[0]
        identity = self.protein_identity(selected).unsqueeze(0).expand(batch, -1, -1)
        map_context = self.map_reader(identity, mapped_states)
        local_context, local_valid = self.local_reader(
            identity, rna_hidden, rna_mask, selected
        )
        global_context = self.protein_global_reader(identity, global_states)
        contexts = [identity, map_context, local_context, global_context]
        cognate_context: Tensor | None = None
        cognate_valid: Tensor | None = None
        if self.config.use_direct_cognate:
            selected_cognate = self.cognate_gene_index.index_select(0, selected)
            cognate_valid = (selected_cognate >= 0).unsqueeze(0).expand(batch, -1)
            safe_cognate = selected_cognate.clamp_min(0)
            cognate_value = expression.index_select(1, safe_cognate).unsqueeze(-1)
            assert self.cognate_projection is not None
            cognate_context = self.cognate_projection(cognate_value)
            cognate_context = cognate_context * cognate_valid.unsqueeze(-1)
            contexts.append(cognate_context)
        logits = self.gate(torch.cat(contexts, dim=-1))
        available = self._branch_dropout_mask(logits, local_valid, cognate_valid)
        logits = logits.masked_fill(~available, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
        fused = (
            identity
            + weights[..., 0:1] * self.map_projection(map_context)
            + weights[..., 1:2] * self.local_projection(local_context)
            + weights[..., 2:3] * self.global_projection(global_context)
        )
        if cognate_context is not None:
            fused = fused + weights[..., 3:4] * cognate_context
        hidden = self.fusion_norm(fused)
        hidden = hidden + self.protein_module_reader(hidden, module_states)
        main_value = self.shared_output(self.output_norm(hidden)).squeeze(-1)
        joint_values = [main_value]
        result = {
            "main_value": main_value,
            "gate": weights,
            "local_valid": local_valid,
        }
        if self.config.use_direct_local_linear:
            direct_index = self.direct_local_gene_index.index_select(0, selected)
            direct_mask = self.direct_local_gene_mask.index_select(0, selected)
            safe_index = direct_index.clamp_min(0)
            direct_expression = expression.index_select(
                1, safe_index.reshape(-1)
            ).reshape(batch, selected.numel(), self.config.max_direct_local_genes)
            assert self.direct_local_weight is not None
            assert self.direct_local_bias is not None
            assert self.direct_local_mix_logit is not None
            direct_weight = self.direct_local_weight.index_select(0, selected)
            direct_value = (
                direct_expression
                * direct_weight.unsqueeze(0)
                * direct_mask.unsqueeze(0)
            ).sum(dim=-1)
            direct_value = direct_value + self.direct_local_bias.index_select(0, selected)
            direct_mix = torch.sigmoid(
                self.direct_local_mix_logit.index_select(0, selected)
            ).unsqueeze(0)
            if self.config.joint_direct_value_fusion:
                joint_values.append(direct_value)
            else:
                main_value = (
                    (1.0 - direct_mix) * main_value + direct_mix * direct_value
                )
            result.update(
                {
                    "main_value": main_value,
                    "direct_local_value": direct_value,
                    "direct_local_mix": direct_mix.expand(batch, -1),
                }
            )
        if self.config.use_direct_global_linear:
            if direct_global_features is None:
                raise ValueError("direct global features are required")
            assert self.direct_global_weight is not None
            assert self.direct_global_bias is not None
            assert self.direct_global_mix_logit is not None
            global_weight = self.direct_global_weight.index_select(0, selected)
            direct_global_value = direct_global_features @ global_weight.T
            direct_global_value = direct_global_value + self.direct_global_bias.index_select(
                0, selected
            )
            direct_global_mix = torch.sigmoid(
                self.direct_global_mix_logit.index_select(0, selected)
            ).unsqueeze(0)
            if self.config.joint_direct_value_fusion:
                joint_values.append(direct_global_value)
            else:
                main_value = (
                    (1.0 - direct_global_mix) * main_value
                    + direct_global_mix * direct_global_value
                )
            result.update(
                {
                    "main_value": main_value,
                    "direct_global_value": direct_global_value,
                    "direct_global_mix": direct_global_mix.expand(batch, -1),
                }
            )
        if self.config.use_direct_supervised_linear:
            supervised_index = self.direct_supervised_gene_index.index_select(
                0, selected
            )
            supervised_mask = self.direct_supervised_gene_mask.index_select(
                0, selected
            )
            supervised_expression = expression.index_select(
                1, supervised_index.clamp_min(0).reshape(-1)
            ).reshape(
                batch,
                selected.numel(),
                self.config.max_direct_supervised_genes,
            )
            assert self.direct_supervised_weight is not None
            assert self.direct_supervised_bias is not None
            assert self.direct_supervised_mix_logit is not None
            supervised_weight = self.direct_supervised_weight.index_select(0, selected)
            direct_supervised_value = (
                supervised_expression
                * supervised_weight.unsqueeze(0)
                * supervised_mask.unsqueeze(0)
            ).sum(dim=-1)
            direct_supervised_value = (
                direct_supervised_value
                + self.direct_supervised_bias.index_select(0, selected)
            )
            direct_supervised_mix = torch.sigmoid(
                self.direct_supervised_mix_logit.index_select(0, selected)
            ).unsqueeze(0)
            if self.config.joint_direct_value_fusion:
                joint_values.append(direct_supervised_value)
            else:
                main_value = (
                    (1.0 - direct_supervised_mix) * main_value
                    + direct_supervised_mix * direct_supervised_value
                )
            result.update(
                {
                    "main_value": main_value,
                    "direct_supervised_value": direct_supervised_value,
                    "direct_supervised_mix": direct_supervised_mix.expand(batch, -1),
                }
            )
        if self.config.joint_direct_value_fusion:
            assert self.direct_joint_mix_logits is not None
            joint_logits = self.direct_joint_mix_logits.index_select(0, selected)
            joint_weights = torch.softmax(joint_logits, dim=-1)
            stacked_values = torch.stack(joint_values, dim=-1)
            if stacked_values.shape[-1] != joint_weights.shape[-1]:
                raise RuntimeError("joint direct value branch count differs")
            main_value = (
                stacked_values * joint_weights.unsqueeze(0)
            ).sum(dim=-1)
            result.update(
                {
                    "main_value": main_value,
                    "direct_joint_weights": joint_weights.unsqueeze(0).expand(
                        batch, -1, -1
                    ),
                }
            )
        return result

    def _apply_platform_calibration(
        self,
        main_value: Tensor,
        platform_index: Tensor,
        selected: Tensor,
    ) -> Tensor:
        is_tcpa = (platform_index == TCPA_PLATFORM).unsqueeze(-1)
        calibrated = self.tcpa_calibrated_protein_mask.index_select(0, selected)
        use_calibration = is_tcpa & calibrated.unsqueeze(0)
        scale = torch.nn.functional.softplus(
            self.tcpa_scale_raw.index_select(0, selected)
        ).unsqueeze(0)
        bias = self.tcpa_bias.index_select(0, selected).unsqueeze(0)
        tcpa_value = scale * main_value + bias
        return torch.where(use_calibration, tcpa_value, main_value)

    def forward(
        self,
        expression: Tensor,
        platform_index: Tensor | None = None,
        cancer_index: Tensor | None = None,
        rna_valid_mask: Tensor | None = None,
        protein_indices: Tensor | None = None,
        return_hidden: bool = False,
    ) -> dict[str, Tensor]:
        batch = expression.shape[0]
        device = expression.device
        if platform_index is None:
            platform_index = torch.full(
                (batch,), CPTAC_PLATFORM, dtype=torch.long, device=device
            )
        if cancer_index is None:
            cancer_index = torch.zeros(batch, dtype=torch.long, device=device)
        rna_hidden, rna_mask = self.rna_encoder(
            expression, platform_index.long(), cancer_index.long(), rna_valid_mask
        )
        mapped, global_states, module_states = self._shared_states(rna_hidden, rna_mask)
        direct_global_features = (
            expression @ self.direct_global_projection
            if self.config.use_direct_global_linear
            else None
        )
        selected_all = (
            self.protein_indices if protein_indices is None else protein_indices.long()
        )
        values: list[Tensor] = []
        main_values: list[Tensor] = []
        gates: list[Tensor] = []
        local_valid_parts: list[Tensor] = []
        direct_local_values: list[Tensor] = []
        direct_local_mixes: list[Tensor] = []
        direct_global_values: list[Tensor] = []
        direct_global_mixes: list[Tensor] = []
        direct_supervised_values: list[Tensor] = []
        direct_supervised_mixes: list[Tensor] = []
        direct_joint_weights: list[Tensor] = []
        for start in range(0, selected_all.numel(), self.config.protein_chunk):
            selected = selected_all[start : start + self.config.protein_chunk]
            decoded = self._decode_chunk(
                selected,
                expression,
                rna_hidden,
                rna_mask,
                mapped,
                global_states,
                module_states,
                direct_global_features,
            )
            values.append(
                self._apply_platform_calibration(
                    decoded["main_value"], platform_index, selected
                )
            )
            main_values.append(decoded["main_value"])
            gates.append(decoded["gate"])
            local_valid_parts.append(decoded["local_valid"])
            if self.config.use_direct_local_linear:
                direct_local_values.append(decoded["direct_local_value"])
                direct_local_mixes.append(decoded["direct_local_mix"])
            if self.config.use_direct_global_linear:
                direct_global_values.append(decoded["direct_global_value"])
                direct_global_mixes.append(decoded["direct_global_mix"])
            if self.config.use_direct_supervised_linear:
                direct_supervised_values.append(decoded["direct_supervised_value"])
                direct_supervised_mixes.append(decoded["direct_supervised_mix"])
            if self.config.joint_direct_value_fusion:
                direct_joint_weights.append(decoded["direct_joint_weights"])
        output: dict[str, Tensor] = {"protein": torch.cat(values, dim=1)}
        if return_hidden:
            gate = torch.cat(gates, dim=1)
            output.update(
                {
                    "shared_main_value": torch.cat(main_values, dim=1),
                    "branch_gate": gate,
                    "map_gate_penalty": gate[..., 0].square().mean(),
                    "local_valid": torch.cat(local_valid_parts, dim=1),
                    "rna_hidden": rna_hidden,
                    "rna_effective_mask": rna_mask,
                    "mapped_states": mapped,
                    "global_states": global_states,
                    "module_states": module_states,
                    "protein_indices": selected_all,
                }
            )
            if self.config.use_direct_local_linear:
                output["direct_local_value"] = torch.cat(direct_local_values, dim=1)
                output["direct_local_mix"] = torch.cat(direct_local_mixes, dim=1)
            if self.config.use_direct_global_linear:
                output["direct_global_value"] = torch.cat(
                    direct_global_values, dim=1
                )
                output["direct_global_mix"] = torch.cat(direct_global_mixes, dim=1)
            if self.config.use_direct_supervised_linear:
                output["direct_supervised_value"] = torch.cat(
                    direct_supervised_values, dim=1
                )
                output["direct_supervised_mix"] = torch.cat(
                    direct_supervised_mixes, dim=1
                )
            if self.config.joint_direct_value_fusion:
                output["direct_joint_weights"] = torch.cat(
                    direct_joint_weights, dim=1
                )
        return output

    def fix_projection_matrices_(self) -> "TCGACPTACHybridProteinTranslator":
        self.rna_encoder.encoder.fix_projection_matrices_()
        for module in (
            self.global_reader,
            self.map_reader,
            self.protein_global_reader,
            self.module_reader,
            self.protein_module_reader,
        ):
            module.fix_projection_matrices_()
        self.global_encoder.fix_projection_matrices_()
        self.module_encoder.fix_projection_matrices_()
        return self

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def architecture_contract(self) -> dict[str, Any]:
        return {
            "direct_total_protein_prediction": True,
            "linear_anchor": False,
            "total_protein_residual": False,
            "local_string_context": True,
            "global_rna_states": self.config.n_global_states,
            "controlled_position_map": [
                self.config.n_genes,
                self.config.map_hidden,
                self.config.map_states,
            ],
            "protein_conditioned_map_read": True,
            "softmax_three_branch_gate": not self.config.use_direct_cognate,
            "softmax_context_gate_branches": (
                4 if self.config.use_direct_cognate else 3
            ),
            "direct_cognate_expression_branch": self.config.use_direct_cognate,
            "direct_local_linear_branch": self.config.use_direct_local_linear,
            "direct_local_linear_genes": (
                self.config.max_direct_local_genes
                if self.config.use_direct_local_linear
                else 0
            ),
            "direct_local_linear_is_convex_value_mix": True,
            "direct_global_linear_branch": self.config.use_direct_global_linear,
            "direct_global_components": (
                self.config.direct_global_components
                if self.config.use_direct_global_linear
                else 0
            ),
            "direct_global_linear_is_convex_value_mix": True,
            "direct_supervised_linear_branch": (
                self.config.use_direct_supervised_linear
            ),
            "direct_supervised_linear_genes": (
                self.config.max_direct_supervised_genes
                if self.config.use_direct_supervised_linear
                else 0
            ),
            "direct_supervised_linear_is_final_convex_value_mix": True,
            "joint_direct_value_fusion": self.config.joint_direct_value_fusion,
            "initial_joint_neural_weight": (
                self.config.initial_joint_neural_weight
                if self.config.joint_direct_value_fusion
                else 0.0
            ),
            "initial_gate": [
                self.config.initial_map_gate,
                self.config.initial_local_gate,
                self.config.initial_global_gate,
            ],
            "branch_dropout": self.config.branch_dropout,
            "rna_sourced_module_states": self.config.n_module_states,
            "protein_chunk_is_computation_only": True,
            "cptac_reference_output": True,
            "tcpa_positive_affine_calibration": True,
        }


__all__ = [
    "CPTAC_PLATFORM",
    "TCPA_PLATFORM",
    "ConditionedRNAEncoder",
    "ControlledPositionMap",
    "HybridProteinConfig",
    "TCGACPTACHybridProteinTranslator",
]

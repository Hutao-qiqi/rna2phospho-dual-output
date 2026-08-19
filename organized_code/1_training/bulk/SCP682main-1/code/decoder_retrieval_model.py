"""Latent-transformer biology operator for cross-patient phosphosite prediction.

The model maps one patient's deployable RNA and predicted-protein measurements
to phosphoproteome latent tokens and then queries those tokens with fixed
biological site coordinates. It contains no ridge floor, graph propagation,
training-label retrieval, or chunk-local site self-attention. Consequently,
decoding a site alone or inside any <=1,000-site chunk is mathematically
identical in evaluation mode.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


CONFIGURATIONS = {
    "neural_operator": (False, False),
    "parent_direct": (True, False),
    "parent_direct_separated": (True, True),
}


@dataclass(frozen=True)
class DecoderRetrievalConfig:
    n_rna: int
    n_proteins: int
    n_sites: int
    n_kinases: int
    n_cancers: int = 0
    hidden: int = 128
    heads: int = 8
    decoder_layers: int = 2
    hgt_layers: int = 0
    global_tokens: int = 1
    maximum_kinases_per_site: int = 128
    site_output_rank: int = 128
    dropout: float = 0.1
    cancer_adapter_rank: int = 8
    esm2_dimension: int = 0
    esm2_fusion: str = "residue_attention"
    esm3_dimension: int = 0
    modality_tokens: int = 32
    phospho_latent_tokens: int = 16
    transformer_layers: int = 2

    def validate(self) -> None:
        if min(self.n_rna, self.n_proteins, self.n_sites, self.n_kinases) < 1:
            raise ValueError("all vocabulary sizes must be positive")
        if self.hidden < 8 or self.site_output_rank < 2:
            raise ValueError("hidden and operator rank are too small")
        if self.hidden % self.heads:
            raise ValueError("hidden must be divisible by heads")
        if self.maximum_kinases_per_site < 1:
            raise ValueError("maximum_kinases_per_site must be positive")
        if self.n_cancers < 0 or self.cancer_adapter_rank < 1:
            raise ValueError("cancer adapter dimensions are invalid")
        if self.esm2_dimension < 0:
            raise ValueError("esm2_dimension cannot be negative")
        if self.esm3_dimension < 0:
            raise ValueError("esm3_dimension cannot be negative")
        if min(
            self.modality_tokens,
            self.phospho_latent_tokens,
            self.transformer_layers,
        ) < 1:
            raise ValueError("latent-transformer dimensions must be positive")
        if self.esm2_fusion not in {
            "residue_attention",
            "residue_residual",
            "center_gated",
            "local_window_attention",
            "local_window_pair",
        }:
            raise ValueError("unknown ESM-2 fusion mode")


class ResidualMLP(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(self.norm(value))


class DecoderRetrievalModel(nn.Module):
    """Patient-branch/site-coordinate operator with masked biological metadata."""

    def __init__(
        self,
        config: DecoderRetrievalConfig,
        *,
        configuration: str = "neural_operator",
        parent_protein_index: torch.Tensor,
        parent_protein_mask: torch.Tensor,
        kinase_rna_index: torch.Tensor,
        kinase_rna_mask: torch.Tensor,
        kinase_protein_index: torch.Tensor,
        kinase_protein_mask: torch.Tensor,
        parent_rna_index: torch.Tensor | None = None,
        parent_rna_mask: torch.Tensor | None = None,
        site_residue_index: torch.Tensor | None = None,
        site_position: torch.Tensor | None = None,
        site_esm2_residue_embedding: torch.Tensor | None = None,
        site_esm2_residue_mask: torch.Tensor | None = None,
        site_esm2_local_embedding: torch.Tensor | None = None,
        site_esm2_local_mask: torch.Tensor | None = None,
        site_esm3_residue_embedding: torch.Tensor | None = None,
        site_esm3_residue_mask: torch.Tensor | None = None,
        site_output_scale: torch.Tensor | None = None,
        site_output_shift: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        if configuration not in CONFIGURATIONS:
            raise ValueError(f"unknown configuration: {configuration}")
        self.config = config
        self.configuration = configuration
        self.use_parent_direct, self.use_separated_readout = CONFIGURATIONS[
            configuration
        ]
        if site_output_scale is None:
            site_output_scale = torch.ones(config.n_sites, dtype=torch.float32)
        if site_output_shift is None:
            site_output_shift = torch.zeros(config.n_sites, dtype=torch.float32)
        site_output_scale = torch.as_tensor(site_output_scale, dtype=torch.float32)
        site_output_shift = torch.as_tensor(site_output_shift, dtype=torch.float32)
        if site_output_scale.shape != (config.n_sites,) or site_output_shift.shape != (
            config.n_sites,
        ):
            raise ValueError("site output coordinate transform must cover every site")
        self.register_buffer("site_output_scale", site_output_scale)
        self.register_buffer("site_output_shift", site_output_shift)

        parent_protein_index = torch.as_tensor(parent_protein_index, dtype=torch.long)
        parent_protein_mask = torch.as_tensor(parent_protein_mask, dtype=torch.bool)
        if parent_protein_index.shape != (config.n_sites,):
            raise ValueError("parent protein mapping must cover every site")
        if parent_protein_mask.shape != parent_protein_index.shape:
            raise ValueError("parent protein mask differs from its index")
        if parent_rna_index is None:
            parent_rna_index = torch.zeros(config.n_sites, dtype=torch.long)
        if parent_rna_mask is None:
            parent_rna_mask = torch.zeros(config.n_sites, dtype=torch.bool)
        if site_residue_index is None:
            site_residue_index = torch.full((config.n_sites,), 3, dtype=torch.long)
        if site_position is None:
            site_position = torch.zeros(config.n_sites, dtype=torch.float32)

        buffers = {
            "parent_protein_index": (parent_protein_index, torch.long),
            "parent_protein_mask": (parent_protein_mask, torch.bool),
            "parent_rna_index": (parent_rna_index, torch.long),
            "parent_rna_mask": (parent_rna_mask, torch.bool),
            "kinase_rna_index": (kinase_rna_index, torch.long),
            "kinase_rna_mask": (kinase_rna_mask, torch.bool),
            "kinase_protein_index": (kinase_protein_index, torch.long),
            "kinase_protein_mask": (kinase_protein_mask, torch.bool),
            "site_residue_index": (site_residue_index, torch.long),
            "site_position": (site_position, torch.float32),
        }
        for name, (value, dtype) in buffers.items():
            self.register_buffer(name, torch.as_tensor(value, dtype=dtype))

        h = config.hidden
        rank = config.site_output_rank

        # Patient branch. Each high-dimensional modality is projected to a
        # small token set before nonlinear attention, retaining multiple RNA
        # and protein directions instead of one early bottleneck vector.
        tokens = config.modality_tokens
        self.rna_token_projection = nn.Sequential(
            nn.LayerNorm(config.n_rna),
            nn.Linear(config.n_rna, tokens),
        )
        self.protein_token_projection = nn.Sequential(
            nn.LayerNorm(config.n_proteins),
            nn.Linear(config.n_proteins, tokens),
        )
        self.reliability_token_projection = nn.Sequential(
            nn.LayerNorm(config.n_proteins),
            nn.Linear(config.n_proteins, tokens),
        )
        self.token_value_projection = nn.Sequential(
            nn.Linear(1, h),
            nn.GELU(),
            nn.Linear(h, h),
        )
        self.modality_identity = nn.Embedding(3, h)
        self.modality_slot_identity = nn.Embedding(tokens, h)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=config.heads,
            dim_feedforward=h * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.modality_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_layers,
            norm=nn.LayerNorm(h),
        )
        self.phospho_latent_identity = nn.Parameter(
            torch.empty(config.phospho_latent_tokens, h)
        )
        self.phospho_cross_attention = nn.MultiheadAttention(
            h,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        latent_layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=config.heads,
            dim_feedforward=h * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.phospho_latent_transformer = nn.TransformerEncoder(
            latent_layer,
            num_layers=config.transformer_layers,
            norm=nn.LayerNorm(h),
        )
        self.patient_pool_norm = nn.LayerNorm(h)
        self.branch_coefficients = nn.Linear(h, rank)
        self.site_latent_query = nn.Linear(h, h, bias=False)
        self.site_latent_key = nn.Linear(h, h, bias=False)
        self.site_latent_value = nn.Linear(h, h, bias=False)
        self.site_latent_output = nn.Linear(h, h, bias=False)

        # Site coordinate network.  Site identity is supplemented by parent,
        # residue position, and the sparse kinase set.  No phosphosite label is
        # consumed by this network.
        self.site_identity = nn.Embedding(config.n_sites, h)
        self.protein_identity = nn.Embedding(config.n_proteins, h)
        self.parent_identity_projection = nn.Linear(h, h, bias=False)
        self.residue_identity = nn.Embedding(4, h)  # S, T, Y, unknown
        self.position_projection = nn.Sequential(
            nn.Linear(1, h),
            nn.GELU(),
            nn.Linear(h, h),
        )
        self.kinase_identity = nn.Embedding(config.n_kinases, h)
        self.kinase_query = nn.Linear(h, h, bias=False)
        self.kinase_key = nn.Linear(h, h, bias=False)
        if config.esm2_dimension:
            if site_esm2_residue_embedding is None or site_esm2_residue_mask is None:
                raise ValueError("enabled ESM-2 prior requires residue embeddings and mask")
            site_esm2_residue_embedding = torch.as_tensor(
                site_esm2_residue_embedding, dtype=torch.float16
            )
            site_esm2_residue_mask = torch.as_tensor(
                site_esm2_residue_mask, dtype=torch.bool
            )
            if site_esm2_residue_embedding.ndim != 3:
                raise ValueError("ESM-2 residue embeddings must be a three-dimensional tensor")
            if site_esm2_residue_embedding.shape[0] != config.n_sites:
                raise ValueError("ESM-2 residue embeddings must cover every site")
            if site_esm2_residue_embedding.shape[2] != config.esm2_dimension:
                raise ValueError("ESM-2 embedding dimension differs from configuration")
            if site_esm2_residue_mask.shape != site_esm2_residue_embedding.shape[:2]:
                raise ValueError("ESM-2 residue mask differs from embeddings")
            # The frozen prior is supplied from the audited NPZ for every run.
            # Keeping it in every checkpoint adds about 140 MB and dominates
            # per-epoch checkpoint time without preserving trainable state.
            self.register_buffer(
                "site_esm2_residue_embedding",
                site_esm2_residue_embedding,
                persistent=False,
            )
            self.register_buffer(
                "site_esm2_residue_mask",
                site_esm2_residue_mask,
                persistent=False,
            )
            self.esm2_input_norm = nn.LayerNorm(config.esm2_dimension)
            self.esm2_residue_projection = nn.Linear(config.esm2_dimension, h)
            self.esm2_value = nn.Linear(h, h, bias=False)
            if config.esm2_fusion in {
                "local_window_attention",
                "local_window_pair",
            }:
                if site_esm2_local_embedding is None or site_esm2_local_mask is None:
                    raise ValueError("local ESM-2 fusion requires local embeddings and mask")
                site_esm2_local_embedding = torch.as_tensor(
                    site_esm2_local_embedding, dtype=torch.float16
                )
                site_esm2_local_mask = torch.as_tensor(
                    site_esm2_local_mask, dtype=torch.bool
                )
                if site_esm2_local_embedding.ndim != 4:
                    raise ValueError("local ESM-2 embeddings must be four-dimensional")
                if site_esm2_local_embedding.shape[0] != config.n_sites:
                    raise ValueError("local ESM-2 embeddings must cover every site")
                if site_esm2_local_embedding.shape[-1] != config.esm2_dimension:
                    raise ValueError("local ESM-2 dimension differs from configuration")
                if site_esm2_local_mask.shape != site_esm2_local_embedding.shape[:-1]:
                    raise ValueError("local ESM-2 mask differs from embeddings")
                self.register_buffer(
                    "site_esm2_local_embedding",
                    site_esm2_local_embedding,
                    persistent=False,
                )
                self.register_buffer(
                    "site_esm2_local_mask",
                    site_esm2_local_mask,
                    persistent=False,
                )
                local_slots = int(site_esm2_local_embedding.shape[1])
                local_width = int(site_esm2_local_embedding.shape[2])
                self.esm2_local_position = nn.Embedding(local_width, h)
                self.esm2_local_slot = nn.Embedding(local_slots, h)
                self.esm2_query = nn.Linear(h, h, bias=False)
                self.esm2_key = nn.Linear(h, h, bias=False)
                self.esm2_pair = (
                    nn.Linear(h, h, bias=False)
                    if config.esm2_fusion == "local_window_pair"
                    else None
                )
                self.esm2_gate = None
                self.esm2_center_norm = None
            else:
                self.register_buffer(
                    "site_esm2_local_embedding",
                    torch.zeros((config.n_sites, 1, 0, config.esm2_dimension), dtype=torch.float16),
                )
                self.register_buffer(
                    "site_esm2_local_mask",
                    torch.zeros((config.n_sites, 1, 0), dtype=torch.bool),
                )
                self.esm2_local_position = None
                self.esm2_local_slot = None
                self.esm2_pair = None
            if config.esm2_fusion in {"residue_attention", "residue_residual"}:
                self.esm2_query = nn.Linear(h, h, bias=False)
                self.esm2_key = nn.Linear(h, h, bias=False)
                self.esm2_gate = None
                self.esm2_center_norm = None
            elif config.esm2_fusion == "center_gated":
                self.esm2_query = None
                self.esm2_key = None
                self.esm2_center_norm = nn.LayerNorm(h)
                self.esm2_gate = nn.Sequential(
                    nn.Linear(h * 2 + 1, h),
                    nn.GELU(),
                    nn.Linear(h, 1),
                )
        else:
            self.register_buffer(
                "site_esm2_residue_embedding",
                torch.zeros((config.n_sites, 1, 0), dtype=torch.float16),
                persistent=False,
            )
            self.register_buffer(
                "site_esm2_residue_mask",
                torch.zeros((config.n_sites, 1), dtype=torch.bool),
                persistent=False,
            )
            self.register_buffer(
                "site_esm2_local_embedding",
                torch.zeros((config.n_sites, 1, 0, 0), dtype=torch.float16),
                persistent=False,
            )
            self.register_buffer(
                "site_esm2_local_mask",
                torch.zeros((config.n_sites, 1, 0), dtype=torch.bool),
                persistent=False,
            )
            self.esm2_input_norm = None
            self.esm2_residue_projection = None
            self.esm2_query = None
            self.esm2_key = None
            self.esm2_value = None
            self.esm2_gate = None
            self.esm2_center_norm = None
            self.esm2_local_position = None
            self.esm2_local_slot = None
            self.esm2_pair = None
        if config.esm3_dimension:
            if site_esm3_residue_embedding is None or site_esm3_residue_mask is None:
                raise ValueError("enabled ESM-3 prior requires residue embeddings and mask")
            site_esm3_residue_embedding = torch.as_tensor(
                site_esm3_residue_embedding, dtype=torch.float16
            )
            site_esm3_residue_mask = torch.as_tensor(
                site_esm3_residue_mask, dtype=torch.bool
            )
            if site_esm3_residue_embedding.ndim != 3:
                raise ValueError("ESM-3 residue embeddings must be three-dimensional")
            if site_esm3_residue_embedding.shape[0] != config.n_sites:
                raise ValueError("ESM-3 residue embeddings must cover every site")
            if site_esm3_residue_embedding.shape[2] != config.esm3_dimension:
                raise ValueError("ESM-3 embedding dimension differs from configuration")
            if site_esm3_residue_mask.shape != site_esm3_residue_embedding.shape[:2]:
                raise ValueError("ESM-3 residue mask differs from embeddings")
            self.register_buffer(
                "site_esm3_residue_embedding",
                site_esm3_residue_embedding,
                persistent=False,
            )
            self.register_buffer(
                "site_esm3_residue_mask", site_esm3_residue_mask, persistent=False
            )
            self.esm3_input_norm = nn.LayerNorm(config.esm3_dimension)
            self.esm3_residue_projection = nn.Linear(config.esm3_dimension, h)
            self.esm3_value = nn.Linear(h, h, bias=False)
        else:
            self.register_buffer(
                "site_esm3_residue_embedding",
                torch.zeros((config.n_sites, 1, 0), dtype=torch.float16),
                persistent=False,
            )
            self.register_buffer(
                "site_esm3_residue_mask",
                torch.zeros((config.n_sites, 1), dtype=torch.bool),
                persistent=False,
            )
            self.esm3_input_norm = None
            self.esm3_residue_projection = None
            self.esm3_value = None
        self.site_trunk = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, h * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h * 2, h),
        )
        self.site_basis = nn.Linear(h, rank, bias=False)

        # Patient-conditioned local coordinates preserve parent and kinase
        # information without graph averaging between output sites.
        self.parent_value_projection = nn.Sequential(
            nn.Linear(3, h),
            nn.GELU(),
            nn.Linear(h, h),
        )
        self.kinase_value_projection = nn.Sequential(
            nn.Linear(3, h),
            nn.GELU(),
            nn.Linear(h, h),
        )
        self.local_output = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, 1),
        )
        self.separated_local_output = (
            nn.Sequential(
                nn.LayerNorm(h * 3),
                nn.Linear(h * 3, h),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(h, 1),
            )
            if self.use_separated_readout
            else None
        )
        self.parent_direct_coefficients = (
            nn.Embedding(config.n_sites, 3) if self.use_parent_direct else None
        )
        self.site_bias = nn.Embedding(config.n_sites, 1)

        if config.n_cancers:
            self.cancer_identity = nn.Embedding(config.n_cancers, h)
            self.cancer_adapter_down = nn.Parameter(
                torch.empty(config.n_cancers, h, config.cancer_adapter_rank)
            )
            self.cancer_adapter_up = nn.Parameter(
                torch.zeros(config.n_cancers, config.cancer_adapter_rank, h)
            )
            nn.init.zeros_(self.cancer_identity.weight)
            nn.init.normal_(self.cancer_adapter_down, std=0.02)
        else:
            self.cancer_identity = None
            self.register_parameter("cancer_adapter_down", None)
            self.register_parameter("cancer_adapter_up", None)

        nn.init.normal_(self.branch_coefficients.weight, std=0.02)
        nn.init.normal_(self.site_basis.weight, std=0.02)
        nn.init.normal_(self.phospho_latent_identity, std=0.02)
        nn.init.zeros_(self.site_bias.weight)
        if self.parent_direct_coefficients is not None:
            nn.init.zeros_(self.parent_direct_coefficients.weight)
        if config.esm2_dimension and config.esm2_fusion == "residue_residual":
            # Preserve the checkpoint's predictions exactly at sequence-prior
            # insertion.  The final ESM projection receives gradients on the
            # first update and then opens the upstream residue projection.
            nn.init.zeros_(self.esm2_value.weight)
        if config.esm3_dimension:
            assert self.esm3_value is not None
            nn.init.zeros_(self.esm3_value.weight)

    def encode_context(
        self,
        rna_rank: torch.Tensor,
        protein_value: torch.Tensor,
        protein_reliability: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if rna_rank.ndim != 2 or rna_rank.shape[1] != self.config.n_rna:
            raise ValueError("RNA input differs from the configured vocabulary")
        if protein_value.shape != protein_reliability.shape:
            raise ValueError("protein values and reliability shapes differ")
        if protein_value.ndim != 2 or protein_value.shape[1] != self.config.n_proteins:
            raise ValueError("protein input differs from the configured vocabulary")

        scalar_tokens = torch.stack(
            [
                self.rna_token_projection(rna_rank),
                self.protein_token_projection(protein_value),
                self.reliability_token_projection(protein_reliability),
            ],
            dim=1,
        )
        batch = int(scalar_tokens.shape[0])
        modalities = int(scalar_tokens.shape[1])
        slots = int(scalar_tokens.shape[2])
        token_state = self.token_value_projection(scalar_tokens.unsqueeze(-1))
        modality_index = torch.arange(modalities, device=rna_rank.device)
        slot_index = torch.arange(slots, device=rna_rank.device)
        token_state = (
            token_state
            + self.modality_identity(modality_index)[None, :, None, :]
            + self.modality_slot_identity(slot_index)[None, None, :, :]
        )
        token_state = token_state.reshape(batch, modalities * slots, -1)
        token_state = self.modality_transformer(token_state)

        latent_query = self.phospho_latent_identity.unsqueeze(0).expand(batch, -1, -1)
        latent_delta, _ = self.phospho_cross_attention(
            latent_query,
            token_state,
            token_state,
            need_weights=False,
        )
        phospho_latents = self.phospho_latent_transformer(
            latent_query + latent_delta
        )
        patient = self.patient_pool_norm(phospho_latents.mean(dim=1))

        kinase_rna = rna_rank.index_select(1, self.kinase_rna_index.clamp_min(0))
        kinase_protein = protein_value.index_select(
            1, self.kinase_protein_index.clamp_min(0)
        )
        kinase_reliability = protein_reliability.index_select(
            1, self.kinase_protein_index.clamp_min(0)
        )
        kinase_features = torch.stack(
            [
                kinase_rna * self.kinase_rna_mask,
                kinase_protein * self.kinase_protein_mask,
                kinase_reliability * self.kinase_protein_mask,
            ],
            dim=-1,
        )
        kinase_state = (
            self.kinase_identity.weight.unsqueeze(0)
            + self.kinase_value_projection(kinase_features)
        )
        return {
            "patient": patient,
            "phospho_latents": phospho_latents,
            "rna": rna_rank,
            "protein": protein_value,
            "reliability": protein_reliability,
            "kinase": kinase_state,
        }

    def _query_phospho_latents(
        self,
        site_coordinate: torch.Tensor,
        phospho_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Let every site read one patient's phosphoproteome latent tokens."""
        batch, latent_count, hidden = phospho_latents.shape
        heads = self.config.heads
        head_width = hidden // heads
        sites = int(site_coordinate.shape[0])
        query = self.site_latent_query(site_coordinate).view(sites, heads, head_width)
        key = self.site_latent_key(phospho_latents).view(
            batch, latent_count, heads, head_width
        )
        value = self.site_latent_value(phospho_latents).view(
            batch, latent_count, heads, head_width
        )
        logits = torch.einsum("shd,blhd->bhsl", query, key)
        logits = logits / math.sqrt(head_width)
        attention = torch.softmax(logits, dim=-1)
        context = torch.einsum("bhsl,blhd->bshd", attention, value)
        context = context.reshape(batch, sites, hidden)
        return self.site_latent_output(context)

    def _adapt_patient(
        self,
        patient: torch.Tensor,
        cancer_index: torch.Tensor | None,
        enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta = torch.zeros_like(patient)
        if not enabled:
            return patient, delta
        if self.config.n_cancers < 1 or cancer_index is None:
            raise ValueError("cancer adaptation requires one cancer index per sample")
        cancer_index = torch.as_tensor(cancer_index, dtype=torch.long, device=patient.device)
        if cancer_index.shape != (patient.shape[0],):
            raise ValueError("cancer index must contain one value per sample")
        if bool(((cancer_index < 0) | (cancer_index >= self.config.n_cancers)).any()):
            raise ValueError("cancer index lies outside the adapter vocabulary")
        down = self.cancer_adapter_down.index_select(0, cancer_index)
        up = self.cancer_adapter_up.index_select(0, cancer_index)
        reduced = torch.einsum("bh,bhr->br", patient, down)
        delta = torch.einsum("br,brh->bh", torch.nn.functional.gelu(reduced), up)
        delta = delta + self.cancer_identity(cancer_index)
        return patient + delta, delta

    def _site_coordinates(
        self,
        site_index: torch.Tensor,
        kinase_index: torch.Tensor,
        kinase_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        parent_index = self.parent_protein_index.index_select(0, site_index).clamp_min(0)
        parent_mask = self.parent_protein_mask.index_select(0, site_index)
        residue = self.site_residue_index.index_select(0, site_index).clamp(0, 3)
        position = self.site_position.index_select(0, site_index).clamp_min(0)
        position = torch.log1p(position).unsqueeze(-1) / 10.0
        base_coordinate = (
            self.site_identity(site_index)
            + self.parent_identity_projection(self.protein_identity(parent_index))
            * parent_mask.unsqueeze(-1)
            + self.residue_identity(residue)
            + self.position_projection(position)
        )

        if self.config.esm2_dimension:
            assert self.esm2_input_norm is not None
            assert self.esm2_residue_projection is not None
            assert self.esm2_value is not None
            if self.config.esm2_fusion in {
                "local_window_attention",
                "local_window_pair",
            }:
                assert self.esm2_query is not None
                assert self.esm2_key is not None
                assert self.esm2_local_position is not None
                assert self.esm2_local_slot is not None
                local_mask = self.site_esm2_local_mask.index_select(0, site_index)
                local_raw = self.site_esm2_local_embedding.index_select(0, site_index)
                local_token = self.esm2_residue_projection(
                    self.esm2_input_norm(local_raw.to(base_coordinate.dtype))
                )
                slots = int(local_token.shape[1])
                width = int(local_token.shape[2])
                position_index = torch.arange(width, device=base_coordinate.device)
                slot_index = torch.arange(slots, device=base_coordinate.device)
                local_token = (
                    local_token
                    + self.esm2_local_position(position_index)[None, None, :, :]
                    + self.esm2_local_slot(slot_index)[None, :, None, :]
                )
                local_token = local_token * local_mask.unsqueeze(-1).to(
                    local_token.dtype
                )
                flat_token = local_token.flatten(1, 2)
                esm2_mask = local_mask.flatten(1, 2)
                has_esm2 = esm2_mask.any(-1)
                esm2_logits = (
                    self.esm2_query(base_coordinate).unsqueeze(1)
                    * self.esm2_key(flat_token)
                ).sum(-1) / math.sqrt(self.config.hidden)
                esm2_logits = esm2_logits.masked_fill(~esm2_mask, -torch.inf)
                safe_esm2_logits = torch.where(
                    has_esm2.unsqueeze(-1), esm2_logits, torch.zeros_like(esm2_logits)
                )
                esm2_attention = torch.softmax(safe_esm2_logits, dim=-1)
                esm2_attention = esm2_attention * esm2_mask.to(esm2_logits.dtype)
                esm2_attention = esm2_attention / esm2_attention.sum(
                    -1, keepdim=True
                ).clamp_min(1.0e-8)
                esm2_coordinate = (
                    esm2_attention.unsqueeze(-1) * self.esm2_value(flat_token)
                ).sum(1)
                if self.config.esm2_fusion == "local_window_pair":
                    assert self.esm2_pair is not None
                    center = local_token[:, :, width // 2, :]
                    center_mask = local_mask[:, :, width // 2]
                    center = center * center_mask.unsqueeze(-1).to(center.dtype)
                    center_sum = center.sum(1)
                    pair_sum = (
                        center_sum.square() - center.square().sum(1)
                    ) * 0.5
                    pair_count = (
                        center_mask.sum(1).to(center.dtype)
                        * (center_mask.sum(1).to(center.dtype) - 1.0)
                        * 0.5
                    )
                    pair_coordinate = self.esm2_pair(
                        pair_sum / pair_count.clamp_min(1.0).unsqueeze(-1)
                    )
                    esm2_coordinate = esm2_coordinate + pair_coordinate * (
                        pair_count > 0
                    ).unsqueeze(-1).to(pair_coordinate.dtype)
            else:
                esm2_mask = self.site_esm2_residue_mask.index_select(0, site_index)
                esm2_raw = self.site_esm2_residue_embedding.index_select(0, site_index)
                esm2_token = self.esm2_residue_projection(
                    self.esm2_input_norm(esm2_raw.to(base_coordinate.dtype))
                )
                esm2_token = esm2_token * esm2_mask.unsqueeze(-1).to(
                    esm2_token.dtype
                )
                has_esm2 = esm2_mask.any(-1)
                if self.config.esm2_fusion in {
                    "residue_attention",
                    "residue_residual",
                }:
                    assert self.esm2_query is not None
                    assert self.esm2_key is not None
                    esm2_logits = (
                        self.esm2_query(base_coordinate).unsqueeze(1)
                        * self.esm2_key(esm2_token)
                    ).sum(-1) / math.sqrt(self.config.hidden)
                    esm2_logits = esm2_logits.masked_fill(~esm2_mask, -torch.inf)
                    safe_esm2_logits = torch.where(
                        has_esm2.unsqueeze(-1),
                        esm2_logits,
                        torch.zeros_like(esm2_logits),
                    )
                    esm2_attention = torch.softmax(safe_esm2_logits, dim=-1)
                    esm2_attention = esm2_attention * esm2_mask.to(
                        esm2_logits.dtype
                    )
                    esm2_attention = esm2_attention / esm2_attention.sum(
                        -1, keepdim=True
                    ).clamp_min(1.0e-8)
                    esm2_coordinate = (
                        esm2_attention.unsqueeze(-1) * self.esm2_value(esm2_token)
                    ).sum(1)
                else:
                    assert self.esm2_gate is not None
                    assert self.esm2_center_norm is not None
                    count = esm2_mask.sum(-1, keepdim=True).clamp_min(1)
                    esm2_attention = esm2_mask.to(esm2_token.dtype) / count
                    center = esm2_token.sum(1) / count
                    center = self.esm2_center_norm(center)
                    gate = torch.sigmoid(
                        self.esm2_gate(
                            torch.cat(
                                [
                                    base_coordinate,
                                    center,
                                    has_esm2.to(center.dtype).unsqueeze(-1),
                                ],
                                dim=-1,
                            )
                        )
                    )
                    esm2_coordinate = gate * self.esm2_value(center)
            esm2_coordinate = esm2_coordinate * has_esm2.unsqueeze(-1).to(
                esm2_coordinate.dtype
            )
        else:
            esm2_attention = torch.zeros(
                (site_index.numel(), 1),
                dtype=base_coordinate.dtype,
                device=base_coordinate.device,
            )
            esm2_coordinate = torch.zeros_like(base_coordinate)

        if self.config.esm3_dimension:
            assert self.esm3_input_norm is not None
            assert self.esm3_residue_projection is not None
            assert self.esm3_value is not None
            esm3_mask = self.site_esm3_residue_mask.index_select(0, site_index)
            esm3_raw = self.site_esm3_residue_embedding.index_select(0, site_index)
            esm3_token = self.esm3_residue_projection(
                self.esm3_input_norm(esm3_raw.to(base_coordinate.dtype))
            )
            esm3_token = esm3_token * esm3_mask.unsqueeze(-1).to(esm3_token.dtype)
            esm3_count = esm3_mask.sum(-1, keepdim=True).clamp_min(1)
            esm3_coordinate = self.esm3_value(
                esm3_token.sum(1) / esm3_count.to(esm3_token.dtype)
            )
            esm3_coordinate = esm3_coordinate * esm3_mask.any(-1).unsqueeze(-1).to(
                esm3_coordinate.dtype
            )
        else:
            esm3_coordinate = torch.zeros_like(base_coordinate)

        kinase_identity = self.kinase_identity(kinase_index.clamp_min(0))
        logits = (
            self.kinase_query(base_coordinate).unsqueeze(1)
            * self.kinase_key(kinase_identity)
        ).sum(-1) / math.sqrt(self.config.hidden)
        logits = logits.masked_fill(~kinase_mask, -torch.inf)
        has_kinase = kinase_mask.any(-1)
        safe_logits = torch.where(has_kinase.unsqueeze(-1), logits, torch.zeros_like(logits))
        weights = torch.softmax(safe_logits, dim=-1) * kinase_mask.to(logits.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1.0e-8)
        kinase_coordinate = (weights.unsqueeze(-1) * kinase_identity).sum(1)
        if (
            self.config.esm2_dimension
            and self.config.esm2_fusion != "residue_residual"
        ):
            coordinate = self.site_trunk(
                base_coordinate + kinase_coordinate + esm2_coordinate
            )
        else:
            coordinate = self.site_trunk(base_coordinate + kinase_coordinate)
            if self.config.esm2_dimension:
                coordinate = coordinate + esm2_coordinate
        if self.config.esm3_dimension:
            coordinate = coordinate + esm3_coordinate
        return coordinate, weights, esm2_attention

    def decode_sites(
        self,
        context: dict[str, torch.Tensor],
        site_index: torch.Tensor,
        kinase_index: torch.Tensor,
        kinase_mask: torch.Tensor,
        *,
        cancer_index: torch.Tensor | None = None,
        enable_cancer_adapter: bool = False,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        device = context["patient"].device
        site_index = torch.as_tensor(site_index, dtype=torch.long, device=device)
        kinase_index = torch.as_tensor(kinase_index, dtype=torch.long, device=device)
        kinase_mask = torch.as_tensor(kinase_mask, dtype=torch.bool, device=device)
        sites = int(site_index.numel())
        batch = int(context["patient"].shape[0])
        if sites > 1000:
            raise ValueError("a neural-operator chunk may contain at most 1000 sites")
        if kinase_index.shape != kinase_mask.shape or kinase_index.shape[0] != sites:
            raise ValueError("padded kinase mapping differs from the site chunk")

        site_coordinate, kinase_attention, esm2_attention = self._site_coordinates(
            site_index, kinase_index, kinase_mask
        )
        parent_p = self.parent_protein_index.index_select(0, site_index).clamp_min(0)
        parent_pm = self.parent_protein_mask.index_select(0, site_index)
        parent_r = self.parent_rna_index.index_select(0, site_index).clamp_min(0)
        parent_rm = self.parent_rna_mask.index_select(0, site_index)
        parent_features = torch.stack(
            [
                context["rna"].index_select(1, parent_r) * parent_rm,
                context["protein"].index_select(1, parent_p) * parent_pm,
                context["reliability"].index_select(1, parent_p) * parent_pm,
            ],
            dim=-1,
        )
        parent_state = self.parent_value_projection(parent_features)

        kinase_state = context["kinase"][:, kinase_index.clamp_min(0), :]
        dynamic_kinase = (
            kinase_attention.unsqueeze(0).unsqueeze(-1) * kinase_state
        ).sum(2)

        pan_patient = context["patient"]
        patient, adapter_delta = self._adapt_patient(
            pan_patient, cancer_index, enable_cancer_adapter
        )

        def operator_output(patient_state: torch.Tensor) -> torch.Tensor:
            coefficients = self.branch_coefficients(patient_state)
            basis = self.site_basis(site_coordinate)
            global_value = torch.einsum("br,sr->bs", coefficients, basis)
            global_value = global_value / math.sqrt(self.config.site_output_rank)
            site_patient_context = self._query_phospho_latents(
                site_coordinate,
                context["phospho_latents"],
            )
            site_patient_context = site_patient_context + (
                patient_state - context["patient"]
            ).unsqueeze(1)
            if self.use_separated_readout:
                assert self.separated_local_output is not None
                interaction = torch.cat(
                    [
                        site_patient_context * site_coordinate.unsqueeze(0),
                        parent_state * site_coordinate.unsqueeze(0),
                        dynamic_kinase * site_coordinate.unsqueeze(0),
                    ],
                    dim=-1,
                )
                local_value = self.separated_local_output(interaction).squeeze(-1)
            else:
                interaction = site_patient_context * (
                    site_coordinate.unsqueeze(0) + parent_state + dynamic_kinase
                )
                local_value = self.local_output(interaction).squeeze(-1)
            direct_value = torch.zeros_like(local_value)
            if self.parent_direct_coefficients is not None:
                direct_features = torch.stack(
                    [
                        parent_features[..., 0],
                        parent_features[..., 1],
                        parent_features[..., 2],
                    ],
                    dim=-1,
                )
                direct_coefficients = self.parent_direct_coefficients(site_index)
                direct_value = (
                    direct_features * direct_coefficients.unsqueeze(0)
                ).sum(-1)
            bias = self.site_bias(site_index).squeeze(-1).unsqueeze(0)
            return bias + global_value + local_value + direct_value

        output_scale = self.site_output_scale.index_select(0, site_index).unsqueeze(0)
        output_shift = self.site_output_shift.index_select(0, site_index).unsqueeze(0)
        pan_prediction = operator_output(pan_patient) * output_scale + output_shift
        prediction = operator_output(patient) * output_scale + output_shift
        return {
            "normalized_profile": prediction,
            "pan_normalized_profile": pan_prediction,
            "cancer_adapter_delta_norm": adapter_delta.float().square().mean().sqrt(),
            "kinase_attention": kinase_attention,
            "esm2_residue_attention": esm2_attention,
            "esm2_site_coverage": self.site_esm2_residue_mask.index_select(
                0, site_index
            ).any(-1),
            "esm3_site_coverage": self.site_esm3_residue_mask.index_select(
                0, site_index
            ).any(-1),
            "patient_state": patient,
            "site_coordinate": site_coordinate,
        }

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "architecture": "latent_transformer_biology_operator",
            "patient_encoder": "modality_tokens_and_phosphoproteome_latents",
            "phosphosite_decoder": "site_query_cross_attention",
            "configuration": self.configuration,
            "prediction_domain": "configured_by_training_entrypoint",
            "ridge_additive_floor": False,
            "graph_message_passing": False,
            "training_label_retrieval": False,
            "site_chunk_self_attention": False,
            "chunk_invariant_site_decoding": True,
            "esm2_sequence_prior": self.config.esm2_dimension > 0,
            "esm2_sequence_prior_role": (
                "frozen_contextual_residue_residual"
                if self.config.esm2_fusion == "residue_residual"
                else "frozen_contextual_residue_query"
            ),
            "esm2_fusion": self.config.esm2_fusion,
            "esm3_sequence_prior": self.config.esm3_dimension > 0,
            "esm3_sequence_prior_role": "frozen_contextual_residue_residual",
            "parent_direct": self.use_parent_direct,
            "separated_biology_readout": self.use_separated_readout,
            "config": asdict(self.config),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
        }

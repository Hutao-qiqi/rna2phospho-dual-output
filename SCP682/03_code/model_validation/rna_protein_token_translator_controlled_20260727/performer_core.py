"""Self-contained FAVOR+ attention used by the RNA-protein translator.

Masks use ``True`` for valid tokens.  The implementation materializes random
feature maps for queries and keys, but never a query-by-key attention matrix.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


__all__ = ["PerformerCrossAttention", "PerformerEncoder"]


def _validate_attention_config(
    dim: int,
    heads: int,
    dim_head: int,
    dropout: float,
) -> None:
    if dim <= 0:
        raise ValueError("dim must be positive")
    if heads <= 0:
        raise ValueError("heads must be positive")
    if dim_head <= 0:
        raise ValueError("dim_head must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")


def _validate_mask(
    mask: Tensor | None,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    name: str,
) -> Tensor | None:
    if mask is None:
        return None
    if mask.ndim != 2 or tuple(mask.shape) != (batch_size, sequence_length):
        raise ValueError(
            f"{name} must have shape ({batch_size}, {sequence_length}), "
            f"received {tuple(mask.shape)}"
        )
    return mask.to(device=device, dtype=torch.bool)


def _orthogonal_matrix_chunk(columns: int, device: torch.device) -> Tensor:
    unstructured = torch.randn(columns, columns, device=device, dtype=torch.float32)
    q, _ = torch.linalg.qr(unstructured, mode="reduced")
    return q.transpose(0, 1)


def _gaussian_orthogonal_random_matrix(
    rows: int,
    columns: int,
    *,
    device: torch.device,
) -> Tensor:
    blocks = []
    full_blocks, remaining_rows = divmod(rows, columns)
    for _ in range(full_blocks):
        blocks.append(_orthogonal_matrix_chunk(columns, device))
    if remaining_rows:
        blocks.append(_orthogonal_matrix_chunk(columns, device)[:remaining_rows])

    matrix = torch.cat(blocks, dim=0)
    gaussian_row_norms = torch.randn(
        rows, columns, device=device, dtype=torch.float32
    ).norm(dim=1)
    return matrix * gaussian_row_norms.unsqueeze(1)


def _softmax_kernel(
    data: Tensor,
    projection_matrix: Tensor,
    *,
    is_query: bool,
    mask: Tensor | None = None,
    epsilon: float = 1e-4,
) -> Tensor:
    """Map query or key vectors to positive orthogonal random features."""

    data_normalizer = data.shape[-1] ** -0.25
    feature_normalizer = projection_matrix.shape[0] ** -0.5
    projection = projection_matrix.to(device=data.device, dtype=data.dtype)

    projected = torch.einsum(
        "bhnd,md->bhnm", data_normalizer * data, projection
    )
    diagonal = 0.5 * data.square().sum(dim=-1, keepdim=True)
    diagonal = diagonal * data_normalizer**2

    expanded_mask = None
    if mask is not None:
        expanded_mask = mask[:, None, :, None]

    if is_query:
        stabilizer = projected.amax(dim=-1, keepdim=True).detach()
    elif expanded_mask is None:
        stabilizer = projected.amax(dim=(-2, -1), keepdim=True).detach()
    else:
        valid_projected = projected.masked_fill(~expanded_mask, -torch.inf)
        stabilizer = valid_projected.amax(dim=(-2, -1), keepdim=True).detach()
        has_valid_key = mask.any(dim=-1)[:, None, None, None]
        stabilizer = torch.where(has_valid_key, stabilizer, torch.zeros_like(stabilizer))

    features = feature_normalizer * (
        torch.exp(projected - diagonal - stabilizer) + epsilon
    )
    if expanded_mask is not None:
        features = features * expanded_mask.to(dtype=features.dtype)
    return features


def _linear_attention(
    query_features: Tensor,
    key_features: Tensor,
    values: Tensor,
    *,
    epsilon: float = 1e-6,
) -> Tensor:
    """Compute non-causal attention without a query-by-context tensor."""

    key_sum = key_features.sum(dim=-2)
    denominator = torch.matmul(query_features, key_sum.unsqueeze(-1))
    denominator = denominator.clamp_min(epsilon)

    context = torch.matmul(key_features.transpose(-2, -1), values)
    output = torch.matmul(query_features, context)
    return output / denominator


class _FastAttention(nn.Module):
    def __init__(
        self,
        dim_head: int,
        nb_features: int | None,
        feature_redraw_interval: int | None,
    ) -> None:
        super().__init__()
        if nb_features is None:
            nb_features = max(1, math.ceil(dim_head * math.log(dim_head)))
        if nb_features <= 0:
            raise ValueError("nb_features must be positive")
        if feature_redraw_interval is not None and feature_redraw_interval <= 0:
            raise ValueError("feature_redraw_interval must be positive or None")

        self.dim_head = dim_head
        self.nb_features = nb_features
        self.feature_redraw_interval = feature_redraw_interval
        projection = _gaussian_orthogonal_random_matrix(
            nb_features,
            dim_head,
            device=torch.device("cpu"),
        )
        self.register_buffer("projection_matrix", projection)
        self.register_buffer(
            "calls_since_last_redraw", torch.tensor(0, dtype=torch.long)
        )

    @torch.no_grad()
    def redraw_projection_matrix_(self) -> _FastAttention:
        projection = _gaussian_orthogonal_random_matrix(
            self.nb_features,
            self.dim_head,
            device=self.projection_matrix.device,
        )
        self.projection_matrix.copy_(projection)
        self.calls_since_last_redraw.zero_()
        return self

    def fix_projection_matrix_(self) -> _FastAttention:
        self.feature_redraw_interval = None
        self.calls_since_last_redraw.zero_()
        return self

    def set_projection_redraw_interval_(
        self, interval: int | None
    ) -> _FastAttention:
        if interval is not None and interval <= 0:
            raise ValueError("projection redraw interval must be positive or None")
        self.feature_redraw_interval = interval
        self.calls_since_last_redraw.zero_()
        return self

    @torch.no_grad()
    def _maybe_redraw_projection(self) -> None:
        if not self.training or self.feature_redraw_interval is None:
            return
        if self.calls_since_last_redraw >= self.feature_redraw_interval:
            self.redraw_projection_matrix_()
        else:
            self.calls_since_last_redraw.add_(1)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        query_mask: Tensor | None = None,
        key_mask: Tensor | None = None,
    ) -> Tensor:
        self._maybe_redraw_projection()
        query_features = _softmax_kernel(
            query,
            self.projection_matrix,
            is_query=True,
            mask=query_mask,
        )
        key_features = _softmax_kernel(
            key,
            self.projection_matrix,
            is_query=False,
            mask=key_mask,
        )
        if key_mask is not None:
            value = value * key_mask[:, None, :, None].to(dtype=value.dtype)
        return _linear_attention(query_features, key_features, value)


class PerformerCrossAttention(nn.Module):
    """FAVOR+ cross-attention with linear query-plus-context scaling.

    Args:
        dim: Input and output feature dimension.
        heads: Number of attention heads.
        dim_head: Feature dimension of each head.
        dropout: Output dropout probability.
        nb_features: Number of FAVOR+ random features per head.
        feature_redraw_interval: Training forwards between projection redraws.
            Use ``None`` to keep the initialized projection fixed.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float,
        *,
        nb_features: int | None = None,
        feature_redraw_interval: int | None = 1000,
    ) -> None:
        super().__init__()
        _validate_attention_config(dim, heads, dim_head, dropout)
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head

        self.to_query = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_key = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_value = nn.Linear(dim, self.inner_dim, bias=False)
        self.fast_attention = _FastAttention(
            dim_head,
            nb_features=nb_features,
            feature_redraw_interval=feature_redraw_interval,
        )
        self.to_output = nn.Linear(self.inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, tensor: Tensor) -> Tensor:
        batch_size, sequence_length, _ = tensor.shape
        tensor = tensor.reshape(
            batch_size, sequence_length, self.heads, self.dim_head
        )
        return tensor.transpose(1, 2)

    def _merge_heads(self, tensor: Tensor) -> Tensor:
        batch_size, _, sequence_length, _ = tensor.shape
        tensor = tensor.transpose(1, 2).contiguous()
        return tensor.reshape(batch_size, sequence_length, self.inner_dim)

    def _forward_attention(
        self,
        query: Tensor,
        context: Tensor,
        *,
        query_mask: Tensor | None,
        context_mask: Tensor | None,
    ) -> Tensor:
        if query.ndim != 3 or context.ndim != 3:
            raise ValueError("query and context must have shape (batch, sequence, dim)")
        if query.shape[0] != context.shape[0]:
            raise ValueError("query and context batch sizes must match")
        if query.shape[-1] != self.dim or context.shape[-1] != self.dim:
            raise ValueError(f"query and context feature dimension must equal {self.dim}")
        if context.shape[1] == 0:
            raise ValueError("context must contain at least one token")

        context_mask = _validate_mask(
            context_mask,
            batch_size=context.shape[0],
            sequence_length=context.shape[1],
            device=context.device,
            name="context_mask",
        )
        query_mask = _validate_mask(
            query_mask,
            batch_size=query.shape[0],
            sequence_length=query.shape[1],
            device=query.device,
            name="query_mask",
        )

        projected_query = self._split_heads(self.to_query(query))
        projected_key = self._split_heads(self.to_key(context))
        projected_value = self._split_heads(self.to_value(context))
        attended = self.fast_attention(
            projected_query,
            projected_key,
            projected_value,
            query_mask=query_mask,
            key_mask=context_mask,
        )
        output = self.to_output(self._merge_heads(attended))
        output = self.dropout(output)
        if query_mask is not None:
            output = output * query_mask.unsqueeze(-1).to(dtype=output.dtype)
        return output

    def forward(
        self,
        query: Tensor,
        context: Tensor,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        return self._forward_attention(
            query,
            context,
            query_mask=None,
            context_mask=context_mask,
        )

    def fix_projection_matrices_(self) -> PerformerCrossAttention:
        self.fast_attention.fix_projection_matrix_()
        return self

    def redraw_projection_matrices_(self) -> PerformerCrossAttention:
        self.fast_attention.redraw_projection_matrix_()
        return self

    def set_projection_redraw_interval_(
        self, interval: int | None
    ) -> PerformerCrossAttention:
        self.fast_attention.set_projection_redraw_interval_(interval)
        return self


class _PerformerSelfAttention(PerformerCrossAttention):
    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        return self._forward_attention(
            x,
            x,
            query_mask=mask,
            context_mask=mask,
        )


class _PerformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float,
        *,
        ff_mult: int,
        nb_features: int | None,
        feature_redraw_interval: int | None,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = _PerformerSelfAttention(
            dim,
            heads,
            dim_head,
            dropout,
            nb_features=nb_features,
            feature_redraw_interval=feature_redraw_interval,
        )
        self.feed_forward_norm = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        x = x + self.attention(self.attention_norm(x), mask=mask)
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        x = x + self.feed_forward(self.feed_forward_norm(x))
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        return x


class PerformerEncoder(nn.Module):
    """Stacked pre-normalized FAVOR+ self-attention encoder."""

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        dropout: float,
        *,
        ff_mult: int = 4,
        nb_features: int | None = None,
        feature_redraw_interval: int | None = 1000,
    ) -> None:
        super().__init__()
        _validate_attention_config(dim, heads, dim_head, dropout)
        if depth <= 0:
            raise ValueError("depth must be positive")
        if ff_mult <= 0:
            raise ValueError("ff_mult must be positive")

        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList(
            [
                _PerformerEncoderLayer(
                    dim,
                    heads,
                    dim_head,
                    dropout,
                    ff_mult=ff_mult,
                    nb_features=nb_features,
                    feature_redraw_interval=feature_redraw_interval,
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, sequence, dim)")
        if x.shape[-1] != self.dim:
            raise ValueError(f"x feature dimension must equal {self.dim}")
        if x.shape[1] == 0:
            raise ValueError("x must contain at least one token")
        mask = _validate_mask(
            mask,
            batch_size=x.shape[0],
            sequence_length=x.shape[1],
            device=x.device,
            name="mask",
        )
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)

        for layer in self.layers:
            x = layer(x, mask)
        x = self.output_norm(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        return x

    def fix_projection_matrices_(self) -> PerformerEncoder:
        for layer in self.layers:
            layer.attention.fix_projection_matrices_()
        return self

    def redraw_projection_matrices_(self) -> PerformerEncoder:
        for layer in self.layers:
            layer.attention.redraw_projection_matrices_()
        return self

    def set_projection_redraw_interval_(
        self, interval: int | None
    ) -> PerformerEncoder:
        for layer in self.layers:
            layer.attention.set_projection_redraw_interval_(interval)
        return self

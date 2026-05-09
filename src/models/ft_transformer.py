"""Feature Tokenizer + Transformer for temporal EHR readmission prediction.

Architecture:
    1. FeatureTokenizer: project each tabular feature to a token embedding
    2. Learnable time encoding (sinusoidal base + linear projection)
    3. TemporalTransformer: multi-head self-attention over time steps
    4. Classification head: readmission probability
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError(
        "PyTorch is required. Install with: pip install torch"
    )

from src.models.layers import (
    FeatureTokenEmbedding,
    GatedResidualConnection,
    TemporalAttentionLayer,
)
from src.models.temporal_utils import (
    create_attention_mask,
    pad_sequence,
    sinusoidal_time_encoding,
)


class FeatureTokenizer(nn.Module):
    """Tokenizes tabular features at each time step into d_model embeddings.

    For a patient with T visits and F features per visit, produces (T, F, d_model)
    then pools across features to get (T, d_model) per-step representations.
    """

    def __init__(self, num_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.token_embed = FeatureTokenEmbedding(num_features, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pool_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, num_features)
        batch, seq_len, _ = x.shape

        # Tokenize features at each time step
        tokens = []
        for t in range(seq_len):
            step_tokens = self.token_embed(x[:, t, :])  # (batch, num_features, d_model)
            cls = self.cls_token.expand(batch, -1, -1)
            step_with_cls = torch.cat([cls, step_tokens], dim=1)
            # Mean-pool across feature tokens (including CLS)
            pooled = step_with_cls.mean(dim=1)  # (batch, d_model)
            tokens.append(pooled)

        out = torch.stack(tokens, dim=1)  # (batch, seq_len, d_model)
        return self.dropout(self.pool_norm(out))


class LearnableTimeEncoding(nn.Module):
    """Sinusoidal base + learned linear projection for continuous timestamps."""

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        # times: (batch, seq_len)
        base = sinusoidal_time_encoding(times, self.linear.in_features)
        return self.linear(base)


class TemporalTransformer(nn.Module):
    """Stacked self-attention layers over the temporal dimension."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attention": TemporalAttentionLayer(d_model, num_heads, dropout),
                        "grn": GatedResidualConnection(d_model, dropout=dropout),
                    }
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer["attention"](x, key_padding_mask=key_padding_mask)
            x = layer["grn"](x)
        return x


class FTTransformerEHR(nn.Module):
    """Full FT-Transformer pipeline for 30-day readmission prediction.

    Args:
        num_features: number of tabular features per visit.
        d_model: hidden dimension throughout.
        num_heads: attention heads per layer.
        num_layers: transformer depth.
        max_seq_len: maximum number of visits/time steps.
        dropout: dropout rate.
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        max_seq_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.tokenizer = FeatureTokenizer(num_features, d_model, dropout)
        self.time_enc = LearnableTimeEncoding(d_model)
        self.transformer = TemporalTransformer(d_model, num_heads, num_layers, dropout)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        times: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            features: (batch, seq_len, num_features) — padded visit features.
            times: (batch, seq_len) — timestamps (e.g. hours since admission).
            lengths: (batch,) — actual sequence lengths before padding.

        Returns:
            (batch,) readmission probabilities.
        """
        seq_len = features.size(1)

        # Pad/truncate to max_seq_len
        if seq_len > self.max_seq_len:
            features = features[:, : self.max_seq_len, :]
            times = times[:, : self.max_seq_len]
            seq_len = self.max_seq_len
            if lengths is not None:
                lengths = lengths.clamp(max=self.max_seq_len)

        mask = None
        if lengths is not None:
            mask = create_attention_mask(lengths, seq_len)

        tokens = self.tokenizer(features)
        tokens = tokens + self.time_enc(times)
        encoded = self.transformer(tokens, key_padding_mask=mask)

        # Pool: take the representation at the last valid time step
        if lengths is not None:
            idx = (lengths - 1).clamp(min=0)
            batch_idx = torch.arange(encoded.size(0), device=encoded.device)
            pooled = encoded[batch_idx, idx]
        else:
            pooled = encoded[:, -1]

        return self.head(pooled).squeeze(-1)

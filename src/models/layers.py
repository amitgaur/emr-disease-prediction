try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise ImportError(
        "PyTorch is required. Install with: pip install torch"
    )

import math


class FeatureTokenEmbedding(nn.Module):
    """Projects each tabular feature to a d_model-dimensional token embedding."""

    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(num_features)]
        )
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_features)
        tokens = [
            proj(x[:, i : i + 1]) for i, proj in enumerate(self.projections)
        ]
        return torch.stack(tokens, dim=1)  # (batch, num_features, d_model)


class TemporalAttentionLayer(nn.Module):
    """Multi-head self-attention with optional attention mask for padding."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        attn_out, _ = self.attention(
            x, x, x, key_padding_mask=key_padding_mask
        )
        return self.norm(x + self.dropout(attn_out))


class GatedResidualConnection(nn.Module):
    """Gated residual network: learns how much of the transformed input to add."""

    def __init__(self, d_model: int, d_hidden: int | None = None, dropout: float = 0.1):
        super().__init__()
        d_hidden = d_hidden or d_model
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        hidden = self.dropout(self.fc2(hidden))
        gate = torch.sigmoid(self.gate(x))
        return self.norm(x + gate * hidden)

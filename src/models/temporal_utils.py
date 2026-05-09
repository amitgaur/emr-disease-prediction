from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError(
        "PyTorch is required. Install with: pip install torch"
    )

import math


def pad_sequence(
    sequence: torch.Tensor, max_len: int, pad_value: float = 0.0
) -> torch.Tensor:
    """Pad a (seq_len, ...) tensor to (max_len, ...) along dim 0."""
    seq_len = sequence.size(0)
    if seq_len >= max_len:
        return sequence[:max_len]
    pad_shape = (max_len - seq_len, *sequence.shape[1:])
    padding = torch.full(pad_shape, pad_value, dtype=sequence.dtype, device=sequence.device)
    return torch.cat([sequence, padding], dim=0)


def truncate_sequence(sequence: torch.Tensor, max_len: int) -> torch.Tensor:
    """Truncate a (seq_len, ...) tensor to (max_len, ...) along dim 0."""
    return sequence[:max_len]


def create_attention_mask(
    lengths: torch.Tensor, max_len: int
) -> torch.Tensor:
    """Create a boolean key_padding_mask: True = masked (padded) position.

    Args:
        lengths: (batch,) actual sequence lengths.
        max_len: padded sequence length.

    Returns:
        (batch, max_len) boolean mask.
    """
    arange = torch.arange(max_len, device=lengths.device)
    return arange.unsqueeze(0) >= lengths.unsqueeze(1)


def sinusoidal_time_encoding(
    times: torch.Tensor, d_model: int
) -> torch.Tensor:
    """Sinusoidal positional encoding from continuous time values.

    Args:
        times: (batch, seq_len) — time values (e.g. hours since admission).
        d_model: embedding dimension (must be even).

    Returns:
        (batch, seq_len, d_model) time encodings.
    """
    half = d_model // 2
    freq = torch.exp(
        torch.arange(half, dtype=torch.float32, device=times.device)
        * -(math.log(10000.0) / half)
    )
    # (batch, seq_len, half)
    angles = times.unsqueeze(-1) * freq
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class CollaterNetwork(nn.Module):
    """Collates variable-length patient visit sequences into padded batches.

    Handles padding, mask creation, and optional time encoding in one pass.
    Use as the collate_fn target in a DataLoader.
    """

    def __init__(self, max_len: int, d_model: int, pad_value: float = 0.0):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.pad_value = pad_value

    def forward(
        self, sequences: list[torch.Tensor], times: list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Collate a list of variable-length sequences.

        Args:
            sequences: list of (seq_len_i, num_features) tensors.
            times: list of (seq_len_i,) time tensors.

        Returns:
            dict with keys: features, time_enc, mask, lengths.
        """
        lengths = torch.tensor(
            [min(s.size(0), self.max_len) for s in sequences],
            dtype=torch.long,
        )

        padded_features = torch.stack(
            [pad_sequence(s, self.max_len, self.pad_value) for s in sequences]
        )
        padded_times = torch.stack(
            [pad_sequence(t, self.max_len, self.pad_value) for t in times]
        )

        mask = create_attention_mask(lengths, self.max_len)
        time_enc = sinusoidal_time_encoding(padded_times, self.d_model)

        return {
            "features": padded_features,
            "time_enc": time_enc,
            "mask": mask,
            "lengths": lengths,
        }

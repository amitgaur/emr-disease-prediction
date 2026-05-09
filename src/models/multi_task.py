"""Multi-task learning wrapper for joint mortality + readmission + disease onset.

This module provides a multi-task learning architecture that jointly trains
on multiple prediction tasks with shared temporal representations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


@dataclass
class MultiTaskConfig:
    """Configuration for multi-task learning."""

    # Shared encoder
    d_model: int = 128
    nhead: int = 4
    num_encoder_layers: int = 2
    dropout: float = 0.1

    # Task-specific heads
    tasks: list[str] = None
    task_weights: Optional[dict[str, float]] = None

    # Curriculum learning
    curriculum_start: dict[str, int] = None  # epoch to start each task
    freeze_shared_until: int = 0

    def __post_init__(self):
        if self.tasks is None:
            self.tasks = ["readmission", "mortality", "disease_onset"]
        if self.task_weights is None:
            # Equal weights by default
            self.task_weights = {t: 1.0 for t in self.tasks}
        if self.curriculum_start is None:
            self.curriculum_start = {t: 0 for t in self.tasks}


class MultiTaskLoss(nn.Module):
    """Weighted multi-task loss with optional curriculum learning."""

    def __init__(self, task_weights: dict[str, float], curriculum_start: dict[str, int]):
        super().__init__()
        self.task_weights = task_weights
        self.curriculum_start = curriculum_start
        self.current_epoch = 0
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def forward(self, logits: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute weighted loss across tasks.

        Returns:
            total_loss: Combined weighted loss
            task_losses: Dict of individual task losses
        """
        total_loss = 0.0
        task_losses = {}

        for task_name, logit in logits.items():
            if task_name not in targets:
                continue

            target = targets[task_name]

            # Apply curriculum: gradually increase task weight
            curriculum_weight = self._get_curriculum_weight(task_name)

            # Task weight from config
            base_weight = self.task_weights.get(task_name, 1.0)

            # BCE loss per sample
            loss_per_sample = self.bce(logit.squeeze(), target)

            # Weighted mean
            task_loss = loss_per_sample.mean() * base_weight * curriculum_weight
            total_loss = total_loss + task_loss
            task_losses[task_name] = task_loss.item()

        return total_loss, task_losses

    def _get_curriculum_weight(self, task_name: str) -> float:
        """Get curriculum weight based on current epoch.

        Starts at 0, gradually increases to 1.0 after curriculum_start.
        """
        start_epoch = self.curriculum_start.get(task_name, 0)
        if self.current_epoch < start_epoch:
            return 0.0
        elif self.current_epoch >= start_epoch + 5:
            return 1.0
        else:
            # Linear warmup over 5 epochs
            return (self.current_epoch - start_epoch) / 5


class SharedEncoder(nn.Module):
    """Shared temporal encoder for multi-task learning.

    Processes patient time series and produces shared representations.
    """

    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int, dropout: float = 0.1):
        super().__init__()

        # Project input to d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, input_dim]
            mask: Attention mask [batch, seq_len]

        Returns:
            cls_repr: [batch, d_model] — CLS token representation
        """
        batch_size = x.size(0)

        # Project and add positional encoding
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Transformer encoding
        if mask is not None:
            # Adjust mask for CLS token
            cls_mask = torch.zeros(batch_size, 1, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)
            mask = mask.unsqueeze(1)  # For transformer: [batch, 1, seq_len]
            src_key_padding_mask = ~mask.bool()
        else:
            src_key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Return CLS token representation
        return x[:, 0, :]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TaskSpecificHead(nn.Module):
    """Task-specific prediction head."""

    def __init__(self, d_model: int, hidden_dim: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, shared_repr: torch.Tensor) -> torch.Tensor:
        return self.head(shared_repr)


class MultiTaskModel(nn.Module if TORCH_AVAILABLE else object):
    """Multi-task model for joint mortality + readmission + disease onset.

    Architecture:
        - Shared temporal encoder (Transformer)
        - Task-specific heads for each prediction task
        - Optional shared representations vs task-specific (via curriculum)
    """

    def __init__(
        self,
        input_dim: int,
        tasks: list[str],
        config: Optional[MultiTaskConfig] = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for MultiTaskModel")

        super().__init__()
        self.config = config or MultiTaskConfig(tasks=tasks)
        self.tasks = self.config.tasks

        # Shared encoder
        self.encoder = SharedEncoder(
            input_dim=input_dim,
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            num_layers=self.config.num_encoder_layers,
            dropout=self.config.dropout,
        )

        # Task-specific heads
        self.task_heads = nn.ModuleDict(
            {task: TaskSpecificHead(self.config.d_model) for task in self.tasks}
        )

        # Loss function
        self.loss_fn = MultiTaskLoss(
            task_weights=self.config.task_weights,
            curriculum_start=self.config.curriculum_start,
        )

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[dict[str, torch.Tensor]] = None,
        epoch: int = 0,
        mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward pass.

        Args:
            x: Input [batch, seq_len, input_dim]
            targets: Optional dict of target tensors for training
            epoch: Current epoch (for curriculum learning)
            mask: Optional attention mask [batch, seq_len]

        Returns:
            If targets provided: (loss, logits, targets) — for training
            Else: logits dict — for inference
        """
        # Shared representation
        shared_repr = self.encoder(x, mask=mask)

        # Task-specific logits
        logits = {task: self.task_heads[task](shared_repr) for task in self.tasks}

        if targets is not None:
            self.loss_fn.set_epoch(epoch)
            loss, task_losses = self.loss_fn(logits, targets)
            return loss, logits, task_losses

        return logits

    def predict(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> dict[str, np.ndarray]:
        """Inference mode — returns probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, mask=mask)
            probs = {task: torch.sigmoid(logit).cpu().numpy() for task, logit in logits.items()}
        return probs

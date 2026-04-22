from __future__ import annotations

import torch
import torch.nn as nn


class ChurnMLP(nn.Module):
    """
    MLP с BatchNorm для бинарной классификации оттока клиентов.

    Архитектура:
        Input → [Linear(128) → BN → ReLU → Dropout]
              → [Linear(64)  → BN → ReLU → Dropout]
              → [Linear(32)  → ReLU]
              → Linear(1)   ← логит (без Sigmoid: используем BCEWithLogitsLoss)
    """

    def __init__(self, input_dim: int, dropout: float = 0.3) -> None:
        super().__init__()

        self.net = nn.Sequential(
            # Блок 1
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Блок 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Блок 3 — без BN и Dropout, плавный переход к выходу
            nn.Linear(64, 32),
            nn.ReLU(),

            # Выходной логит
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # squeeze(1): (batch, 1) → (batch,) — удобнее для BCEWithLogitsLoss
        return self.net(x).squeeze(1)

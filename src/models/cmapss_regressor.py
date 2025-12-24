"""Baseline feed-forward model for C-MAPSS RUL prediction."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
from torch import nn


class CmapssRegressor(nn.Module):
    """A lightweight multilayer perceptron for RUL regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or (128, 64)

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.network(inputs)


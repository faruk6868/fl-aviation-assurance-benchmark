"""Helper factories for PyTorch optimizers."""

from __future__ import annotations

from typing import Iterable

import torch


def adam_optimizer_factory(
    params: Iterable[torch.nn.Parameter],
    learning_rate: float,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    return torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)


def sgd_optimizer_factory(
    params: Iterable[torch.nn.Parameter],
    learning_rate: float,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    return torch.optim.SGD(params, lr=learning_rate, weight_decay=weight_decay, momentum=0.0)

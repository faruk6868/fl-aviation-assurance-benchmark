"""Configuration schemas for federated training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FederatedAlgorithm(str, Enum):
    """Supported federated optimization algorithms."""

    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    SCAFFOLD = "scaffold"

    @classmethod
    def from_value(cls, value: "FederatedAlgorithm | str | None") -> "FederatedAlgorithm":
        if value is None:
            return cls.FEDAVG
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Unsupported federated algorithm '{value}'. Valid options: {[m.value for m in cls]}")


@dataclass(frozen=True)
class ClientConfig:
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0


@dataclass(frozen=True)
class FederatedConfig:
    num_rounds: int
    client_fraction: float = 1.0
    client_config: ClientConfig = ClientConfig()
    seed: int = 42
    evaluation_interval: int = 1
    max_rounds_without_improvement: Optional[int] = None


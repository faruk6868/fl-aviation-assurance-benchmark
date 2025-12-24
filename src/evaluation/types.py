"""Shared evaluation data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from src.metrics import MetricResult


@dataclass
class TestOutput:
    """Represents the outcome of a test bed execution."""

    test_id: str
    metrics: List[MetricResult]
    context: Dict[str, object] = field(default_factory=dict)


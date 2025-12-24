"""Utilities for reproducible experiment seeding."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch optional for type checkers
    torch = None  # type: ignore


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Set seeds for Python, NumPy, and (optionally) PyTorch.

    Parameters
    ----------
    seed:
        Global random seed value.
    deterministic_torch:
        If True, configure PyTorch for deterministic operations when possible.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


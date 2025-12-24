"""TB-14 attribution fidelity & stability evaluation."""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.data.cmapss import prepare_cmapss_dataset
from src.testbeds.tb04 import (
    DatasetConfig as TB04DatasetConfig,
    ModelConfig as TB04ModelConfig,
    SequenceRegressor,
    _feature_columns,
    _windowed_sequences,
)


@dataclass(slots=True)
class ExplainabilityConfig:
    method: str
    repeats: int
    noise_std: float


@dataclass(slots=True)
class EvalConfig:
    device: str
    seed: int


@dataclass(slots=True)
class TB14Config:
    dataset: TB04DatasetConfig
    model: TB04ModelConfig
    explainability: ExplainabilityConfig
    evaluation: EvalConfig


def load_tb14_config(config_path: Path) -> TB14Config:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    dataset_cfg = TB04DatasetConfig(
        dataset_id=data["dataset"].get("id", "FD002"),
        sequence_length=int(data["dataset"].get("sequence_length", 30)),
        sequence_stride=int(data["dataset"].get("sequence_stride", 10)),
        normalize=bool(data["dataset"].get("normalize", True)),
        max_sequences_per_client=int(data["dataset"].get("samples", 100)),
    )
    model_cfg = TB04ModelConfig(
        hidden_size=int(data["model"].get("hidden_size", 64)),
        num_layers=int(data["model"].get("num_layers", 1)),
        dropout=float(data["model"].get("dropout", 0.2)),
    )
    explain_cfg = ExplainabilityConfig(
        method=str(data["explainability"].get("method", "grad")),
        repeats=int(data["explainability"].get("repeats", 5)),
        noise_std=float(data["explainability"].get("noise_std", 0.01)),
    )
    eval_cfg = EvalConfig(
        device=str(data["evaluation"].get("device", "cpu")),
        seed=int(data["evaluation"].get("seed", 2037)),
    )

    return TB14Config(
        dataset=dataset_cfg,
        model=model_cfg,
        explainability=explain_cfg,
        evaluation=eval_cfg,
    )


def run_tb14_pipeline(
    project_root: Path,
    algo: str,
) -> Tuple[List[Dict[str, float]], Dict[str, object]]:
    """Execute TB-14 explainability stability evaluation."""

    config_path = project_root / "config" / "config_v2" / "tb14_xai_stability.yaml"
    tb14_config = load_tb14_config(config_path)

    torch.manual_seed(tb14_config.evaluation.seed)
    np.random.seed(tb14_config.evaluation.seed)
    random.seed(tb14_config.evaluation.seed)

    _, test_df, _ = prepare_cmapss_dataset(
        dataset_id=tb14_config.dataset.dataset_id,
        root=project_root / "data" / "c-mapss",
        normalize=tb14_config.dataset.normalize,
        rul_clip=125,
    )
    feature_cols = _feature_columns(test_df)
    sequences, labels = _windowed_sequences(
        test_df,
        feature_cols,
        tb14_config.dataset.sequence_length,
        tb14_config.dataset.sequence_stride,
        tb14_config.dataset.max_sequences_per_client,
    )
    if len(sequences) == 0:
        raise RuntimeError("TB-14 requires non-empty test sequences.")
    dataset = TensorDataset(torch.from_numpy(sequences), torch.from_numpy(labels))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    checkpoint_path = project_root / "artifacts" / "testbeds" / "TB-04" / "models" / f"{algo}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"TB-14 requires TB-04 model checkpoint for '{algo}'. Expected at {checkpoint_path}."
        )
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    device = torch.device(tb14_config.evaluation.device)
    model = SequenceRegressor(len(feature_cols), tb14_config.model)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    per_sample_scores: List[Tuple[int, float]] = []
    for idx, (batch_x, _) in enumerate(dataloader):
        if idx >= tb14_config.dataset.max_sequences_per_client:
            break
        sample = batch_x.to(device)
        stability = _compute_sample_stability(
            model=model,
            sample=sample,
            repeats=tb14_config.explainability.repeats,
            noise_std=tb14_config.explainability.noise_std,
        )
        per_sample_scores.append((idx, stability))

    stability_values = [score for _, score in per_sample_scores]
    mean_stability = float(np.mean(stability_values))
    median_stability = float(np.median(stability_values))
    min_stability = float(np.min(stability_values))

    measurement_entries = [
        {
            "metric_id": "M.XAI.FIDE_STAB",
            "value": mean_stability,
            "extras": {
                "median": median_stability,
                "min": min_stability,
                "samples": len(per_sample_scores),
            },
        },
        {
            "metric_id": "M.XAI.FIDELITY",
            "value": mean_stability,
            "extras": {
                "median": median_stability,
                "min": min_stability,
                "samples": len(per_sample_scores),
            },
        },
        {
            "metric_id": "M.XAI.STABILITY",
            "value": min_stability,
            "extras": {
                "median": median_stability,
                "min": min_stability,
                "samples": len(per_sample_scores),
            },
        },
    ]

    sample_details = [
        {"sample_index": idx, "stability_score": score} for idx, score in per_sample_scores
    ]
    details = {
        "stability_mean": mean_stability,
        "stability_median": median_stability,
        "stability_min": min_stability,
        "sample_scores": sample_details,
    }

    return measurement_entries, details


def _compute_sample_stability(
    model: SequenceRegressor,
    sample: torch.Tensor,
    repeats: int,
    noise_std: float,
) -> float:
    attrs: List[np.ndarray] = []
    for _ in range(repeats):
        noisy_sample = sample.clone()
        if noise_std > 0:
            noisy_sample = noisy_sample + noise_std * torch.randn_like(noisy_sample)
        noisy_sample.requires_grad_(True)
        output = model(noisy_sample)
        grads = torch.autograd.grad(outputs=output, inputs=noisy_sample)[0]
        attrs.append(grads.detach().cpu().numpy().flatten())

    if len(attrs) < 2:
        return 1.0

    correlations: List[float] = []
    for a, b in itertools.combinations(attrs, 2):
        corr = _cosine_similarity(a, b)
        correlations.append(corr)
    return float(np.mean(correlations)) if correlations else 1.0


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)



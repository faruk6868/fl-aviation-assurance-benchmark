"""TB-13 runtime footprint (inference latency) evaluation."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
class HardwareProfile:
    name: str
    backend: str
    num_threads: int
    use_gpu: bool
    warmup_runs: int
    measured_runs: int


@dataclass(slots=True)
class EvalConfig:
    device: str
    seed: int


@dataclass(slots=True)
class TB13Config:
    dataset: TB04DatasetConfig
    model: TB04ModelConfig
    evaluation: EvalConfig
    hardware_profiles: List[HardwareProfile]


def load_tb13_config(config_path: Path) -> TB13Config:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    dataset_cfg = TB04DatasetConfig(
        dataset_id=data["dataset"].get("id", "FD002"),
        sequence_length=int(data["dataset"].get("sequence_length", 30)),
        sequence_stride=int(data["dataset"].get("sequence_stride", 10)),
        normalize=bool(data["dataset"].get("normalize", True)),
        max_sequences_per_client=int(data["dataset"].get("max_sequences", 1000)),
    )
    model_cfg = TB04ModelConfig(
        hidden_size=int(data["model"].get("hidden_size", 64)),
        num_layers=int(data["model"].get("num_layers", 1)),
        dropout=float(data["model"].get("dropout", 0.2)),
    )
    eval_cfg = EvalConfig(
        device=str(data["evaluation"].get("device", "cpu")),
        seed=int(data["evaluation"].get("seed", 2036)),
    )

    profiles: List[HardwareProfile] = []
    for entry in data.get("hardware_profiles", []):
        profiles.append(
            HardwareProfile(
                name=str(entry.get("name", "profile")),
                backend=str(entry.get("backend", "pytorch")),
                num_threads=int(entry.get("num_threads", 1)),
                use_gpu=bool(entry.get("use_gpu", False)),
                warmup_runs=int(entry.get("warmup_runs", 50)),
                measured_runs=int(entry.get("measured_runs", 500)),
            )
        )
    if not profiles:
        profiles.append(
            HardwareProfile(
                name="cpu_default",
                backend="pytorch",
                num_threads=1,
                use_gpu=False,
                warmup_runs=50,
                measured_runs=500,
            )
        )

    return TB13Config(
        dataset=dataset_cfg,
        model=model_cfg,
        evaluation=eval_cfg,
        hardware_profiles=profiles,
    )


def run_tb13_pipeline(
    project_root: Path,
    algo: str,
) -> Tuple[List[Dict[str, float]], Dict[str, object]]:
    """Execute TB-13 runtime evaluation for a specific algorithm."""

    config_path = project_root / "config" / "config_v2" / "tb13_runtime.yaml"
    tb13_config = load_tb13_config(config_path)

    torch.manual_seed(tb13_config.evaluation.seed)
    np.random.seed(tb13_config.evaluation.seed)

    _, test_df, _ = prepare_cmapss_dataset(
        dataset_id=tb13_config.dataset.dataset_id,
        root=project_root / "data" / "c-mapss",
        normalize=tb13_config.dataset.normalize,
        rul_clip=125,
    )

    feature_cols = _feature_columns(test_df)
    sequences, labels = _windowed_sequences(
        test_df,
        feature_cols,
        tb13_config.dataset.sequence_length,
        tb13_config.dataset.sequence_stride,
        tb13_config.dataset.max_sequences_per_client,
    )
    dataset = TensorDataset(torch.from_numpy(sequences), torch.from_numpy(labels))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    checkpoint_path = project_root / "artifacts" / "testbeds" / "TB-04" / "models" / f"{algo}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"TB-13 requires TB-04 model checkpoint for '{algo}'. Expected at {checkpoint_path}."
        )
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    measurement_entries: List[Dict[str, float]] = []
    hw_details: List[Dict[str, object]] = []

    for profile in tb13_config.hardware_profiles:
        latencies_ms = _measure_latency(
            tb13_config=tb13_config,
            model_state=state_dict,
            feature_count=len(feature_cols),
            dataloader=dataloader,
            profile=profile,
        )
        median_ms = statistics.median(latencies_ms)
        mean_ms = statistics.fmean(latencies_ms)
        p95_ms = float(np.percentile(latencies_ms, 95))
        max_ms = max(latencies_ms)
        measurement_entries.append(
            {
                "metric_id": "M.RT.LAT",
                "value": median_ms,
                "extras": {
                    "hw_profile": profile.name,
                    "mean_ms": mean_ms,
                    "p95_ms": p95_ms,
                    "max_ms": max_ms,
                    "backend": profile.backend,
                    "num_threads": profile.num_threads,
                    "use_gpu": profile.use_gpu,
                },
            }
        )
        hw_details.append(
            {
                "hw_profile": profile.name,
                "backend": profile.backend,
                "num_threads": profile.num_threads,
                "use_gpu": profile.use_gpu,
                "median_ms": median_ms,
                "mean_ms": mean_ms,
                "p95_ms": p95_ms,
                "max_ms": max_ms,
                "warmup_runs": profile.warmup_runs,
                "measured_runs": profile.measured_runs,
            }
        )

    details = {
        "latency_profiles": hw_details,
    }
    return measurement_entries, details


def _measure_latency(
    tb13_config: TB13Config,
    model_state: Dict[str, torch.Tensor],
    feature_count: int,
    dataloader: DataLoader,
    profile: HardwareProfile,
) -> List[float]:
    if profile.backend != "pytorch":
        raise NotImplementedError(f"Only pytorch backend is supported currently (got: {profile.backend}).")

    torch.set_num_threads(profile.num_threads)
    device = torch.device("cuda" if profile.use_gpu and torch.cuda.is_available() else "cpu")

    model = SequenceRegressor(feature_count, tb13_config.model)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    samples: List[torch.Tensor] = []
    for batch_x, _ in dataloader:
        samples.append(batch_x)
        if len(samples) >= profile.measured_runs + profile.warmup_runs:
            break
    if not samples:
        raise RuntimeError("No test samples available for TB-13 latency measurement.")

    with torch.no_grad():
        for sample in samples[: profile.warmup_runs]:
            sample = sample.to(device)
            _ = model(sample)

        latencies_ms: List[float] = []
        for sample in samples[profile.warmup_runs : profile.warmup_runs + profile.measured_runs]:
            sample = sample.to(device)
            start = time.perf_counter()
            _ = model(sample)
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000.0)

    return latencies_ms



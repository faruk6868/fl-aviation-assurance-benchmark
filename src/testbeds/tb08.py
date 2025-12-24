"""TB-08 drift readiness & detection power orchestration."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import yaml

from src.data.cmapss import prepare_cmapss_dataset
from src.data.partitioning import dirichlet_quantity_skew
from src.testbeds.tb04 import (
    DatasetConfig as TB04DatasetConfig,
    ModelConfig as TB04ModelConfig,
    SequenceRegressor,
    _feature_columns,
    _windowed_sequences,
)


@dataclass(slots=True)
class ClientConfig:
    num_clients: int
    dirichlet_alpha: float
    seed: int


@dataclass(slots=True)
class TimelineConfig:
    total_batches: int
    batch_size: int
    pre_drift_batches: int
    drift_ramp_batches: int


@dataclass(slots=True)
class DriftConfig:
    noise_max: float
    bias_max: float
    dropout_max: float
    affected_client_fraction: float


@dataclass(slots=True)
class DetectionConfig:
    reference_window: int
    rolling_window: int
    rmse_threshold: float
    ks_threshold: float


@dataclass(slots=True)
class EvalConfig:
    device: str
    seed: int


@dataclass(slots=True)
class ClientStream:
    sequences: np.ndarray
    labels: np.ndarray
    cursor: int = 0


@dataclass(slots=True)
class TB08Config:
    dataset: TB04DatasetConfig
    clients: ClientConfig
    model: TB04ModelConfig
    timeline: TimelineConfig
    drift: DriftConfig
    detection: DetectionConfig
    evaluation: EvalConfig


def load_tb08_config(config_path: Path) -> TB08Config:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_data = data.get("dataset", {})
    clients_data = data.get("clients", {})
    model_data = data.get("model", {})
    timeline_data = data.get("timeline", {})
    drift_data = data.get("drift", {})
    detection_data = data.get("detection", {})
    eval_data = data.get("evaluation", {})

    dataset_cfg = TB04DatasetConfig(
        dataset_id=dataset_data.get("id", "FD002"),
        sequence_length=int(dataset_data.get("sequence_length", 30)),
        sequence_stride=int(dataset_data.get("sequence_stride", 10)),
        normalize=bool(dataset_data.get("normalize", True)),
        max_sequences_per_client=int(dataset_data.get("max_sequences_per_client", 600)),
    )
    clients_cfg = ClientConfig(
        num_clients=int(clients_data.get("num_clients", 20)),
        dirichlet_alpha=float(clients_data.get("dirichlet_alpha", 0.6)),
        seed=int(clients_data.get("seed", 2030)),
    )
    model_cfg = TB04ModelConfig(
        hidden_size=int(model_data.get("hidden_size", 64)),
        num_layers=int(model_data.get("num_layers", 1)),
        dropout=float(model_data.get("dropout", 0.2)),
    )
    timeline_cfg = TimelineConfig(
        total_batches=int(timeline_data.get("total_batches", 60)),
        batch_size=int(timeline_data.get("batch_size", 128)),
        pre_drift_batches=int(timeline_data.get("pre_drift_batches", 20)),
        drift_ramp_batches=int(timeline_data.get("drift_ramp_batches", 10)),
    )
    drift_cfg = DriftConfig(
        noise_max=float(drift_data.get("noise_max", 0.08)),
        bias_max=float(drift_data.get("bias_max", 0.05)),
        dropout_max=float(drift_data.get("dropout_max", 0.05)),
        affected_client_fraction=float(drift_data.get("affected_client_fraction", 0.5)),
    )
    detection_cfg = DetectionConfig(
        reference_window=int(detection_data.get("reference_window", 10)),
        rolling_window=int(detection_data.get("rolling_window", 5)),
        rmse_threshold=float(detection_data.get("rmse_threshold", 0.25)),
        ks_threshold=float(detection_data.get("ks_threshold", 0.3)),
    )
    eval_cfg = EvalConfig(
        device=str(eval_data.get("device", "cpu")),
        seed=int(eval_data.get("seed", 2030)),
    )

    return TB08Config(
        dataset=dataset_cfg,
        clients=clients_cfg,
        model=model_cfg,
        timeline=timeline_cfg,
        drift=drift_cfg,
        detection=detection_cfg,
        evaluation=eval_cfg,
    )


def run_tb08_pipeline(
    project_root: Path,
    algo: str,
    rounds_override: int | None = None,
) -> Tuple[List[Dict[str, float]], Dict[str, object]]:
    """Execute TB-08 drift readiness workflow."""

    config_path = project_root / "config" / "config_v2" / "tb08_drift_detection.yaml"
    tb08_config = load_tb08_config(config_path)
    if rounds_override is not None and rounds_override > 0:
        _adjust_timeline(tb08_config.timeline, rounds_override)

    random.seed(tb08_config.clients.seed)
    np.random.seed(tb08_config.clients.seed)
    torch.manual_seed(tb08_config.evaluation.seed)

    train_df, test_df, _ = prepare_cmapss_dataset(
        dataset_id=tb08_config.dataset.dataset_id,
        root=project_root / "data" / "c-mapss",
        normalize=tb08_config.dataset.normalize,
        rul_clip=125,
    )
    feature_cols = _feature_columns(train_df)

    partitions = dirichlet_quantity_skew(
        test_df,
        num_clients=tb08_config.clients.num_clients,
        alpha=tb08_config.clients.dirichlet_alpha,
        seed=tb08_config.clients.seed,
    )

    client_streams: Dict[int, ClientStream] = {}
    for client_id, df in partitions.items():
        sequences, labels = _windowed_sequences(
            df,
            feature_cols,
            tb08_config.dataset.sequence_length,
            tb08_config.dataset.sequence_stride,
            tb08_config.dataset.max_sequences_per_client,
        )
        if len(sequences) == 0:
            continue
        client_streams[client_id] = ClientStream(
            sequences=sequences,
            labels=labels,
            cursor=0,
        )

    client_ids = sorted(client_streams.keys())
    if not client_ids:
        raise RuntimeError("No client sequences available for TB-08 timeline generation.")

    affected_clients = _select_affected_clients(client_ids, tb08_config.drift.affected_client_fraction)

    model_state = _load_base_model(project_root, algo)
    model = SequenceRegressor(len(feature_cols), tb08_config.model)
    model.load_state_dict(model_state)
    model.to(torch.device(tb08_config.evaluation.device))
    model.eval()

    (
        metrics,
        details,
    ) = _evaluate_drift_detection(tb08_config, model, client_streams, client_ids, affected_clients)

    return metrics, details


def _evaluate_drift_detection(
    config: TB08Config,
    model: SequenceRegressor,
    client_streams: Dict[int, ClientStream],
    client_ids: Sequence[int],
    affected_clients: set[int],
) -> Tuple[List[Dict[str, float]], Dict[str, object]]:
    device = torch.device(config.evaluation.device)
    timeline_cfg = config.timeline
    detection_cfg = config.detection
    drift_cfg = config.drift

    ref_feature_buffer: List[np.ndarray] = []
    ref_rmse_buffer: List[float] = []
    baseline_rmse: float | None = None
    rolling_rmse: List[float] = []

    timeline: List[Dict[str, object]] = []
    drift_labels: List[int] = []
    alarms: List[bool] = []
    rmse_values: List[float] = []
    ks_values: List[float] = []

    total_batches = timeline_cfg.total_batches
    drift_start = timeline_cfg.pre_drift_batches
    ramp = max(0, timeline_cfg.drift_ramp_batches)
    num_clients = len(client_ids)

    for batch_idx in range(total_batches):
        client_id = client_ids[batch_idx % num_clients]
        batch_x_np, batch_y_np = _sample_stream_batch(client_streams[client_id], timeline_cfg.batch_size)

        batch_x = torch.from_numpy(batch_x_np).to(device)
        batch_y = torch.from_numpy(batch_y_np).to(device)

        progress = _drift_progress(batch_idx, drift_start, ramp)
        label = 1 if progress > 0 else 0
        apply_drift = progress > 0 and client_id in affected_clients
        noise_level = drift_cfg.noise_max * progress if apply_drift else 0.0
        bias_level = drift_cfg.bias_max * progress if apply_drift else 0.0
        dropout_level = drift_cfg.dropout_max * progress if apply_drift else 0.0

        if apply_drift:
            batch_x = _apply_drift(batch_x, noise_level, bias_level, dropout_level)

        with torch.no_grad():
            preds = model(batch_x)
            mse = torch.mean((preds - batch_y) ** 2)
            rmse = float(torch.sqrt(mse).item())

        feature_snapshot = batch_x[:, -1, :].detach().cpu().numpy()
        if label == 0:
            ref_feature_buffer.append(feature_snapshot)
            if len(ref_feature_buffer) > detection_cfg.reference_window:
                ref_feature_buffer.pop(0)
            ref_rmse_buffer.append(rmse)
            if len(ref_rmse_buffer) > detection_cfg.reference_window:
                ref_rmse_buffer.pop(0)
            baseline_rmse = float(np.mean(ref_rmse_buffer))

        rolling_rmse.append(rmse)
        if len(rolling_rmse) > detection_cfg.rolling_window:
            rolling_rmse.pop(0)

        reference_ready = len(ref_feature_buffer) >= detection_cfg.reference_window
        baseline_ready = baseline_rmse is not None
        ref_samples = np.concatenate(ref_feature_buffer, axis=0) if reference_ready else None
        ks_stat = _ks_statistic(ref_samples, feature_snapshot)
        observed_rmse = float(np.mean(rolling_rmse))
        rmse_delta = (
            abs(observed_rmse - baseline_rmse) / (baseline_rmse + 1e-8)
            if baseline_ready and baseline_rmse is not None
            else 0.0
        )

        alarm = bool(
            reference_ready
            and (
                ks_stat >= detection_cfg.ks_threshold
                or rmse_delta >= detection_cfg.rmse_threshold
            )
        )

        alarms.append(alarm)
        drift_labels.append(label)
        rmse_values.append(rmse)
        ks_values.append(ks_stat)

        timeline.append(
            {
                "t": batch_idx,
                "client_id": int(client_id),
                "drift_label": bool(label),
                "alarm": alarm,
                "rmse": rmse,
                "rmse_delta": rmse_delta,
                "ks_stat": ks_stat,
                "noise_std": noise_level,
                "bias_level": bias_level,
                "dropout": dropout_level,
                "progress": progress,
            }
        )

    metrics, ks_summary = _summarize_detection(drift_labels, alarms, ks_values)
    details = {
        "timeline": timeline,
        "baseline_rmse": baseline_rmse if baseline_rmse is not None else 0.0,
        "ks_mean_drift": ks_summary["drift_mean"],
        "ks_mean_nodrift": ks_summary["nodrift_mean"],
        "drift_start_batch": drift_start,
        "affected_clients": sorted(int(cid) for cid in affected_clients),
        "reference_window": detection_cfg.reference_window,
        "timeline_config": {
            "total_batches": total_batches,
            "pre_drift": drift_start,
            "ramp": ramp,
        },
    }
    return metrics, details


def _adjust_timeline(timeline: TimelineConfig, override_batches: int) -> None:
    original_total = timeline.total_batches
    if override_batches <= 0 or override_batches == original_total:
        return
    override_batches = max(override_batches, timeline.pre_drift_batches + 2)
    scale = override_batches / float(original_total)
    timeline.total_batches = override_batches
    timeline.pre_drift_batches = max(1, int(round(timeline.pre_drift_batches * scale)))
    timeline.drift_ramp_batches = max(1, int(round(timeline.drift_ramp_batches * scale)))
    if timeline.pre_drift_batches >= timeline.total_batches - 1:
        timeline.pre_drift_batches = max(1, timeline.total_batches // 2)


def _select_affected_clients(client_ids: Sequence[int], fraction: float) -> set[int]:
    if not client_ids:
        return set()
    clamped = min(max(fraction, 0.0), 1.0)
    count = max(1, int(math.ceil(len(client_ids) * clamped)))
    count = min(len(client_ids), count)
    selected = set(random.sample(list(client_ids), count))
    return selected


def _sample_stream_batch(stream: ClientStream, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(stream.sequences) == 0:
        raise RuntimeError("Cannot sample from empty client stream.")
    x_samples: List[np.ndarray] = []
    y_samples: List[float] = []
    total = len(stream.sequences)
    for _ in range(batch_size):
        x_samples.append(stream.sequences[stream.cursor])
        y_samples.append(stream.labels[stream.cursor])
        stream.cursor = (stream.cursor + 1) % total
    return np.stack(x_samples).astype(np.float32), np.array(y_samples, dtype=np.float32)


def _drift_progress(batch_idx: int, start: int, ramp: int) -> float:
    if batch_idx < start:
        return 0.0
    if ramp <= 0:
        return 1.0
    if batch_idx < start + ramp:
        return float(batch_idx - start + 1) / float(ramp)
    return 1.0


def _apply_drift(
    batch: torch.Tensor,
    noise_std: float,
    bias_level: float,
    dropout_rate: float,
) -> torch.Tensor:
    if noise_std > 0.0:
        batch = batch + noise_std * torch.randn_like(batch)
    if bias_level != 0.0:
        batch = batch * (1.0 + bias_level)
    if dropout_rate > 0.0:
        keep = torch.bernoulli(torch.ones_like(batch) * (1.0 - dropout_rate))
        batch = batch * keep
    return batch


def _load_base_model(project_root: Path, algo: str) -> Dict[str, torch.Tensor]:
    model_path = project_root / "artifacts" / "testbeds" / "TB-04" / "models" / f"{algo}.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"TB-08 requires a trained TB-04 model for '{algo}'. "
            f"Please run TB-04 first (measurement script) to generate {model_path.name}."
        )
    state_dict = torch.load(model_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Invalid checkpoint format at {model_path}.")
    return state_dict


def _ks_statistic(
    reference_samples: np.ndarray | None,
    batch_samples: np.ndarray,
) -> float:
    if reference_samples is None or len(reference_samples) == 0 or len(batch_samples) == 0:
        return 0.0
    stats: List[float] = []
    num_features = min(reference_samples.shape[1], batch_samples.shape[1])
    for feat_idx in range(num_features):
        stats.append(_ks_1d(reference_samples[:, feat_idx], batch_samples[:, feat_idx]))
    if not stats:
        return 0.0
    return float(max(stats))


def _ks_1d(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    if sample_a.size == 0 or sample_b.size == 0:
        return 0.0
    a_sorted = np.sort(sample_a)
    b_sorted = np.sort(sample_b)
    data = np.concatenate([a_sorted, b_sorted])
    cdf_a = np.searchsorted(a_sorted, data, side="right") / a_sorted.size
    cdf_b = np.searchsorted(b_sorted, data, side="right") / b_sorted.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _summarize_detection(
    labels: Sequence[int],
    alarms: Sequence[bool],
    ks_values: Sequence[float],
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    drift_starts: List[int] = []
    if labels and labels[0] == 1:
        drift_starts.append(0)
    for idx in range(1, len(labels)):
        if labels[idx - 1] == 0 and labels[idx] == 1:
            drift_starts.append(idx)

    delays: List[int] = []
    for start in drift_starts:
        # Find first alarm after drift starts (not before)
        detection_idx = next((i for i in range(start, len(labels)) if alarms[i]), None)
        if detection_idx is None:
            # Drift detected but no alarm triggered - count as full delay
            delays.append(len(labels) - start)
        else:
            # Delay = batches from drift start to detection
            delays.append(detection_idx - start)
    mttd = float(np.mean(delays)) if delays else 0.0

    tp = sum(1 for label, alarm in zip(labels, alarms) if label == 1 and alarm)
    fn = sum(1 for label, alarm in zip(labels, alarms) if label == 1 and not alarm)
    fp = sum(1 for label, alarm in zip(labels, alarms) if label == 0 and alarm)
    tn = sum(1 for label, alarm in zip(labels, alarms) if label == 0 and not alarm)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    ks_drift = [value for value, label in zip(ks_values, labels) if label == 1]
    ks_nodrift = [value for value, label in zip(ks_values, labels) if label == 0]
    ks_summary = {
        "drift_mean": float(np.mean(ks_drift)) if ks_drift else 0.0,
        "nodrift_mean": float(np.mean(ks_nodrift)) if ks_nodrift else 0.0,
    }

    metrics = [
        {"metric_id": "M.DRIFT.DETECT.MTTD", "value": float(mttd)},
        {"metric_id": "M.DRIFT.DETECT.RECALL", "value": float(recall)},
        {"metric_id": "M.DRIFT.DETECT.FAR", "value": float(far)},
        {"metric_id": "M.DRIFT.KS", "value": ks_summary["drift_mean"]},
    ]
    return metrics, ks_summary



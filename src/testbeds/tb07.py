"""TB-07 robustness to ODD segment shifts and perturbations."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.cmapss import prepare_cmapss_dataset
from src.data.partitioning import dirichlet_quantity_skew
from src.testbeds.tb04 import (
    DatasetConfig as TB04DatasetConfig,
    EvalConfig as TB04EvalConfig,
    FederatedConfig as TB04FederatedConfig,
    ModelConfig as TB04ModelConfig,
    SequenceRegressor,
    _apply_feature_shift,
)


@dataclass(slots=True)
class ClientConfig:
    num_clients: int
    dirichlet_alpha: float
    feature_shift_std: float
    seed: int


@dataclass(slots=True)
class PerturbationConfig:
    noise_levels: List[float]
    bias_levels: List[float]
    dropout_rate: float
    repeats: int


@dataclass(slots=True)
class TB07Config:
    dataset: TB04DatasetConfig
    clients: ClientConfig
    federated: TB04FederatedConfig
    model: TB04ModelConfig
    evaluation: TB04EvalConfig
    odd_segments: List[Dict[str, object]]
    perturbations: PerturbationConfig


def load_tb07_config(config_path: Path) -> TB07Config:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_cfg = TB04DatasetConfig(
        dataset_id=data["dataset"].get("id", "FD002"),
        sequence_length=int(data["dataset"].get("sequence_length", 30)),
        sequence_stride=int(data["dataset"].get("sequence_stride", 10)),
        normalize=bool(data["dataset"].get("normalize", True)),
        max_sequences_per_client=int(data["dataset"].get("max_sequences_per_client", 400)),
    )
    client_cfg = ClientConfig(
        num_clients=int(data["clients"].get("num_clients", 20)),
        dirichlet_alpha=float(data["clients"].get("dirichlet_alpha", 0.6)),
        feature_shift_std=float(data["clients"].get("feature_shift_std", 0.05)),
        seed=int(data["clients"].get("seed", 2029)),
    )
    federated_cfg = TB04FederatedConfig(
        rounds=int(data["federated"].get("rounds", 100)),
        clients_per_round=float(data["federated"].get("clients_per_round", 0.4)),
        local_epochs=int(data["federated"].get("local_epochs", 1)),
        batch_size=int(data["federated"].get("batch_size", 32)),
        test_batch_size=int(data["federated"].get("test_batch_size", 128)),
        lr=float(data["federated"].get("optimizer", {}).get("lr", 1e-3)),
        weight_decay=float(data["federated"].get("optimizer", {}).get("weight_decay", 0.0)),
        fedprox_mu=float(data["federated"].get("fedprox_mu", 0.01)),
        scaffold_eta=float(data["federated"].get("scaffold_eta", 1.0)),
    )
    model_cfg = TB04ModelConfig(
        hidden_size=int(data["model"].get("hidden_size", 64)),
        num_layers=int(data["model"].get("num_layers", 1)),
        dropout=float(data["model"].get("dropout", 0.2)),
    )
    eval_cfg = TB04EvalConfig(
        device=str(data["evaluation"].get("device", "cpu")),
        seed=int(data["evaluation"].get("seed", 2029)),
    )
    perturb_cfg = PerturbationConfig(
        noise_levels=[float(x) for x in data["perturbations"].get("noise_levels", [0.02, 0.05])],
        bias_levels=[float(x) for x in data["perturbations"].get("bias_levels", [0.0, 0.05])],
        dropout_rate=float(data["perturbations"].get("dropout_rate", 0.05)),
        repeats=int(data["perturbations"].get("repeats", 5)),
    )
    return TB07Config(
        dataset=dataset_cfg,
        clients=client_cfg,
        federated=federated_cfg,
        model=model_cfg,
        evaluation=eval_cfg,
        odd_segments=data.get("odd_segments", []),
        perturbations=perturb_cfg,
    )


def run_tb07_pipeline(
    project_root: Path,
    algo: str,
    rounds_override: int | None = None,
) -> Tuple[List[Dict[str, float]], Dict[str, object]]:
    config_path = project_root / "config" / "config_v2" / "tb07_robustness.yaml"
    tb07_config = load_tb07_config(config_path)
    if rounds_override is not None and rounds_override > 0:
        tb07_config.federated.rounds = int(rounds_override)

    random.seed(tb07_config.clients.seed)
    np.random.seed(tb07_config.clients.seed)
    torch.manual_seed(tb07_config.evaluation.seed)

    train_df, test_df, _ = prepare_cmapss_dataset(
        dataset_id=tb07_config.dataset.dataset_id,
        root=project_root / "data" / "c-mapss",
        normalize=tb07_config.dataset.normalize,
    )

    partitions = dirichlet_quantity_skew(
        train_df,
        num_clients=tb07_config.clients.num_clients,
        alpha=tb07_config.clients.dirichlet_alpha,
        seed=tb07_config.clients.seed,
    )

    feature_cols = [col for col in train_df.columns if col not in {"engine_id", "cycle", "RUL"}]

    client_partitions: Dict[int, TensorDataset] = {}
    for client_id, df in partitions.items():
        client_df = df.copy()
        if tb07_config.clients.feature_shift_std > 0:
            _apply_feature_shift(client_df, feature_cols, tb07_config.clients.feature_shift_std)
        sequences, labels, _ = _windowed_sequences_with_segment(
            client_df,
            feature_cols,
            tb07_config.dataset.sequence_length,
            tb07_config.dataset.sequence_stride,
            None,
            _segment_assigner(tb07_config, client_df),
        )
        dataset = TensorDataset(
            torch.from_numpy(sequences),
            torch.from_numpy(labels),
        )
        client_partitions[client_id] = dataset

    test_sequences, test_labels, segments = _windowed_sequences_with_segment(
        test_df,
        feature_cols,
        tb07_config.dataset.sequence_length,
        tb07_config.dataset.sequence_stride,
        None,
        _segment_assigner(tb07_config, test_df),
    )

    test_dataset = TensorDataset(
        torch.from_numpy(test_sequences),
        torch.from_numpy(test_labels),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=tb07_config.federated.test_batch_size,
        shuffle=False,
    )

    trainer = RobustnessTrainer(
        tb07_config,
        client_partitions,
        test_loader,
        torch.from_numpy(segments),
    )
    metrics, details = trainer.evaluate(algo)
    return metrics, details


class RobustnessTrainer:
    def __init__(
        self,
        config: TB07Config,
        clients: Dict[int, TensorDataset],
        test_loader: DataLoader,
        segment_ids: torch.Tensor,
    ) -> None:
        self.config = config
        self.clients = clients
        self.test_loader = test_loader
        self.segment_ids = segment_ids
        self.device = torch.device(config.evaluation.device)
        self.criterion = nn.MSELoss()
        sample_batch, _ = next(iter(test_loader))
        self.input_size = sample_batch.shape[-1]

    def evaluate(self, algo: str) -> Tuple[List[Dict[str, float]], Dict[str, object]]:
        fl_state = self._train_fl(algo)
        model = self._new_model()
        model.load_state_dict(fl_state)
        model.to(self.device)

        nominal_rmse = self._global_rmse(model, self.test_loader)
        segment_rmse = self._segment_rmse(model, self.test_loader, self.segment_ids)
        cross_stab = float(np.std(segment_rmse) / (np.mean(segment_rmse) + 1e-8))

        degrad_values = []
        perturb_results = []
        for noise in self.config.perturbations.noise_levels:
            for bias in self.config.perturbations.bias_levels:
                pert_loader = self._perturbed_loader(self.test_loader, noise, bias, self.config.perturbations.dropout_rate)
                pert_rmse = self._global_rmse(model, pert_loader)
                degrad = (pert_rmse - nominal_rmse) / (nominal_rmse + 1e-8)
                degrad_values.append(degrad)
                perturb_results.append(
                    {"noise": noise, "bias": bias, "rmse": pert_rmse, "degradation": degrad}
                )
        robustness_degrad = float(np.mean(degrad_values)) if degrad_values else 0.0

        stability_scores = []
        subset_loader = self._subset_loader(self.test_loader, max_samples=200)
        for _ in range(self.config.perturbations.repeats):
            pert_loader = self._perturbed_loader(subset_loader, self.config.perturbations.noise_levels[-1], 0.0, 0.0)
            preds = self._collect_predictions(model, pert_loader)
            stability_scores.append(preds)
        if stability_scores and len(stability_scores) > 1:
            stacked = torch.stack(stability_scores, dim=0)  # Shape: [repeats, n_samples]
            std = stacked.std(dim=0)  # Standard deviation across repeats for each sample
            mean = stacked.mean(dim=0).abs() + 1e-8  # Mean across repeats for each sample
            # Return CV directly (not 1.0 - CV) to match threshold definition
            # CV = std/mean (Coefficient of Variation)
            cv = std / mean
            stability = torch.mean(cv).item()
        elif stability_scores and len(stability_scores) == 1:
            # Only one repeat: cannot compute CV, return 0 (perfect stability)
            stability = 0.0
        else:
            stability = 0.0

        metrics = [
            {"metric_id": "M.ROB.STAB_ODD", "value": cross_stab},
            {"metric_id": "M.ROB.DEGRAD", "value": robustness_degrad},
            {"metric_id": "M.INFER.STAB", "value": stability},
        ]

        details = {
            "nominal_rmse": nominal_rmse,
            "segment_rmse": segment_rmse,
            "perturbations": perturb_results,
            "stability_score": stability,
        }
        return metrics, details

    def _train_fl(self, algo: str) -> Dict[str, torch.Tensor]:
        algo_key = algo.lower()
        if algo_key not in {"fedavg", "fedprox", "scaffold"}:
            raise ValueError(f"Unsupported algorithm '{algo}'.")

        model = self._new_model()
        state = model.state_dict()
        c_global = None
        c_clients: Dict[int, Dict[str, torch.Tensor]] = {}
        if algo_key == "scaffold":
            c_global = _zero_like_state(state, device=self.device)
            c_clients = {client_id: _zero_like_state(state, device=self.device) for client_id in self.clients}

        client_ids = list(self.clients.keys())
        rounds = self.config.federated.rounds
        clients_per_round = max(1, int(math.ceil(self.config.federated.clients_per_round * len(client_ids))))

        for _ in range(rounds):
            random.shuffle(client_ids)
            selected = client_ids[:clients_per_round]
            updates: List[Tuple[Dict[str, torch.Tensor], int]] = []
            delta_cs: List[Dict[str, torch.Tensor]] = []

            for client_id in selected:
                partition = self.clients[client_id]
                if len(partition) == 0:
                    continue
                loader = DataLoader(
                    partition,
                    batch_size=self.config.federated.batch_size,
                    shuffle=True,
                )
                local_model = self._new_model()
                local_model.load_state_dict(state)
                local_model.to(self.device)
                global_params = {key: value.to(self.device) for key, value in state.items()}

                update_state, sample_count, delta_c = self._train_client(
                    local_model,
                    loader,
                    algo_key,
                    global_params,
                    c_global,
                    c_clients.get(client_id),
                )
                updates.append((update_state, sample_count))
                if algo_key == "scaffold" and delta_c is not None and c_global is not None:
                    c_clients[client_id] = delta_c
                    delta_cs.append(delta_c)

            if updates:
                state = _aggregate_weighted(updates)
                if algo_key == "scaffold" and delta_cs and c_global is not None:
                    c_global = _update_control_variate(c_global, delta_cs, self.config.federated.scaffold_eta)

        return {name: tensor.clone().cpu() for name, tensor in state.items()}

    def _train_client(
        self,
        model: SequenceRegressor,
        loader: DataLoader,
        algo: str,
        global_state: Dict[str, torch.Tensor],
        c_global: Dict[str, torch.Tensor] | None,
        c_client: Dict[str, torch.Tensor] | None,
    ) -> Tuple[Dict[str, torch.Tensor], int, Dict[str, torch.Tensor] | None]:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.federated.lr,
            weight_decay=self.config.federated.weight_decay,
        )
        total_samples = 0
        total_steps = 0
        for _ in range(self.config.federated.local_epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                preds = model(batch_x)
                loss = self.criterion(preds, batch_y)
                if algo == "fedprox":
                    prox = 0.0
                    for name, param in model.named_parameters():
                        prox += torch.sum((param - global_state[name]) ** 2)
                    loss = loss + (self.config.federated.fedprox_mu / 2.0) * prox
                optimizer.zero_grad()
                loss.backward()
                if algo == "scaffold" and c_global is not None and c_client is not None:
                    for name, param in model.named_parameters():
                        param.grad = param.grad + (c_client[name] - c_global[name])
                optimizer.step()
                total_samples += batch_x.size(0)
                total_steps += 1
        state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
        delta_c: Dict[str, torch.Tensor] | None = None
        if algo == "scaffold" and c_client is not None and total_steps > 0:
            delta_c = {}
            lr = self.config.federated.lr
            for name, param in state_dict.items():
                correction = (global_state[name].cpu() - param) / (lr * total_steps)
                delta_c[name] = c_client[name].cpu() - c_global[name].cpu() + correction
        samples = total_samples if total_samples else loader.batch_size * max(1, len(loader))
        return state_dict, samples, delta_c

    # Evaluation helpers -----------------------------------------------------
    def _global_rmse(self, model: SequenceRegressor, loader: DataLoader) -> float:
        model.eval()
        preds: List[float] = []
        targets: List[float] = []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                outputs = model(batch_x).cpu().numpy()
                preds.extend(outputs.tolist())
                targets.extend(batch_y.numpy().tolist())
        return float(np.sqrt(np.mean(np.square(np.array(preds) - np.array(targets)))))

    def _segment_rmse(self, model: SequenceRegressor, loader: DataLoader, segment_ids: torch.Tensor) -> List[float]:
        model.eval()
        preds_all: List[float] = []
        targets_all: List[float] = []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                outputs = model(batch_x).cpu().numpy()
                preds_all.extend(outputs.tolist())
                targets_all.extend(batch_y.numpy().tolist())
        preds_arr = np.array(preds_all)
        targets_arr = np.array(targets_all)
        segments = segment_ids.numpy()
        rmse_values: List[float] = []
        for seg in np.unique(segments):
            mask = segments == seg
            if mask.any():
                rmse = np.sqrt(np.mean(np.square(preds_arr[mask] - targets_arr[mask])))
                rmse_values.append(float(rmse))
        return rmse_values

    def _perturbed_loader(self, loader: DataLoader, noise: float, bias: float, dropout: float) -> DataLoader:
        perturbed = []
        for batch_x, batch_y in loader:
            x = batch_x.clone()
            if noise > 0:
                x += noise * torch.randn_like(x)
            if bias != 0:
                x = x * (1.0 + bias)
            if dropout > 0:
                mask = torch.bernoulli(torch.ones_like(x) * (1 - dropout))
                x = x * mask
            perturbed.append((x, batch_y))
        tensors = [
            (torch.cat([pair[0] for pair in perturbed], dim=0), torch.cat([pair[1] for pair in perturbed], dim=0))
        ]
        return DataLoader(
            TensorDataset(tensors[0][0], tensors[0][1]),
            batch_size=self.config.federated.test_batch_size,
            shuffle=False,
        )

    def _subset_loader(self, loader: DataLoader, max_samples: int) -> DataLoader:
        samples_x: List[torch.Tensor] = []
        samples_y: List[torch.Tensor] = []
        collected = 0
        for batch_x, batch_y in loader:
            if collected >= max_samples:
                break
            take = min(batch_x.size(0), max_samples - collected)
            samples_x.append(batch_x[:take])
            samples_y.append(batch_y[:take])
            collected += take
        subset = TensorDataset(torch.cat(samples_x, dim=0), torch.cat(samples_y, dim=0))
        return DataLoader(subset, batch_size=self.config.federated.test_batch_size, shuffle=False)

    def _collect_predictions(self, model: SequenceRegressor, loader: DataLoader) -> torch.Tensor:
        preds: List[torch.Tensor] = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                preds.append(model(batch_x).cpu())
        return torch.cat(preds, dim=0)

    def _new_model(self) -> SequenceRegressor:
        return SequenceRegressor(self.input_size, self.config.model)


def _windowed_sequences_with_segment(
    df,
    feature_cols: Sequence[str],
    window: int,
    stride: int,
    max_sequences: int | None,
    segment_fn,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequences: List[np.ndarray] = []
    labels: List[float] = []
    segments: List[int] = []

    for _, engine_df in df.groupby("engine_id"):
        engine_df = engine_df.sort_values("cycle")
        values = engine_df[feature_cols].to_numpy(dtype=np.float32)
        targets = engine_df["RUL"].to_numpy(dtype=np.float32)
        for end_idx in range(window, len(engine_df), stride):
            start_idx = end_idx - window
            seq = values[start_idx:end_idx]
            label = targets[end_idx - 1]
            sequences.append(seq)
            labels.append(label)
            segments.append(segment_fn(engine_df.iloc[end_idx - 1]))

    if not sequences:
        return (
            np.zeros((0, window, len(feature_cols)), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    seq_array = np.stack(sequences)
    label_array = np.array(labels, dtype=np.float32)
    segment_array = np.array(segments, dtype=np.int64)

    if max_sequences is not None and len(seq_array) > max_sequences:
        indices = np.random.choice(len(seq_array), size=max_sequences, replace=False)
        seq_array = seq_array[indices]
        label_array = label_array[indices]
        segment_array = segment_array[indices]

    return seq_array, label_array, segment_array


def _segment_assigner(config: TB07Config, df) -> callable:
    names = [seg.get("name", str(idx)) for idx, seg in enumerate(config.odd_segments)]
    if not names:
        names = ["segment_%d" % i for i in range(3)]
    segment_count = len(names)
    setting_values = df["setting_1"].to_numpy()
    thresholds = np.quantile(setting_values, np.linspace(0, 1, segment_count + 1))

    def assign(row) -> int:
        value = row["setting_1"]
        idx = np.searchsorted(thresholds, value, side="right") - 1
        idx = max(0, min(idx, segment_count - 1))
        return idx

    return assign


def _aggregate_weighted(updates: List[Tuple[Dict[str, torch.Tensor], int]]) -> Dict[str, torch.Tensor]:
    total = sum(weight for _, weight in updates)
    if total == 0:
        raise ValueError("Cannot aggregate zero-weight updates.")
    keys = updates[0][0].keys()
    aggregated: Dict[str, torch.Tensor] = {}
    for key in keys:
        num = sum(state[key] * weight for state, weight in updates)
        aggregated[key] = num / total
    return aggregated


def _zero_like_state(state: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: torch.zeros_like(value, device=device) for key, value in state.items()}


def _update_control_variate(
    c_global: Dict[str, torch.Tensor],
    deltas: Iterable[Dict[str, torch.Tensor]],
    eta: float,
) -> Dict[str, torch.Tensor]:
    deltas = list(deltas)
    if not deltas:
        return c_global
    avg_delta: Dict[str, torch.Tensor] = {}
    for key in c_global.keys():
        stacked = torch.stack([delta[key] for delta in deltas], dim=0)
        avg_delta[key] = stacked.mean(dim=0)
    return {key: c_global[key] + eta * avg_delta[key] for key in c_global.keys()}



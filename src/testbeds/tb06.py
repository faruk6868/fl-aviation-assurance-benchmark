"""TB-06 benefit equity evaluation against centralized baseline."""

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
    _feature_columns as tb04_feature_columns,
    _windowed_sequences as tb04_windowed_sequences,
    ClientPartition as TB04ClientPartition,
)


@dataclass(slots=True)
class ClientConfig:
    num_clients: int
    dirichlet_alpha: float
    feature_shift_std: float
    seed: int


@dataclass(slots=True)
class BaselineConfig:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float


@dataclass(slots=True)
class TB06Config:
    dataset: TB04DatasetConfig
    clients: ClientConfig
    federated: TB04FederatedConfig
    baseline: BaselineConfig
    model: TB04ModelConfig
    evaluation: TB04EvalConfig


def load_tb06_config(config_path: Path) -> TB06Config:
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
        seed=int(data["clients"].get("seed", 2028)),
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
    baseline_cfg = BaselineConfig(
        epochs=int(data["baseline"].get("epochs", 5)),
        batch_size=int(data["baseline"].get("batch_size", 128)),
        lr=float(data["baseline"].get("optimizer", {}).get("lr", 1e-3)),
        weight_decay=float(data["baseline"].get("optimizer", {}).get("weight_decay", 0.0)),
    )
    model_cfg = TB04ModelConfig(
        hidden_size=int(data["model"].get("hidden_size", 64)),
        num_layers=int(data["model"].get("num_layers", 1)),
        dropout=float(data["model"].get("dropout", 0.2)),
    )
    eval_cfg = TB04EvalConfig(
        device=str(data["evaluation"].get("device", "cpu")),
        seed=int(data["evaluation"].get("seed", 2028)),
    )
    return TB06Config(
        dataset=dataset_cfg,
        clients=client_cfg,
        federated=federated_cfg,
        baseline=baseline_cfg,
        model=model_cfg,
        evaluation=eval_cfg,
    )


def run_tb06_pipeline(
    project_root: Path,
    algo: str,
    rounds_override: int | None = None,
) -> Tuple[List[Dict[str, float]], List[Dict[str, object]]]:
    """Evaluate benefit equity compared to centralized baseline."""

    config_path = project_root / "config" / "config_v2" / "tb06_benefit_equity.yaml"
    tb06_config = load_tb06_config(config_path)
    if rounds_override is not None and rounds_override > 0:
        tb06_config.federated.rounds = int(rounds_override)

    random.seed(tb06_config.clients.seed)
    np.random.seed(tb06_config.clients.seed)
    torch.manual_seed(tb06_config.evaluation.seed)

    train_df, test_df, _ = prepare_cmapss_dataset(
        dataset_id=tb06_config.dataset.dataset_id,
        root=project_root / "data" / "c-mapss",
        normalize=tb06_config.dataset.normalize,
    )

    partitions = dirichlet_quantity_skew(
        train_df,
        num_clients=tb06_config.clients.num_clients,
        alpha=tb06_config.clients.dirichlet_alpha,
        seed=tb06_config.clients.seed,
    )

    feature_cols = tb04_feature_columns(train_df)
    client_partitions: Dict[int, TB04ClientPartition] = {}
    centralized_sequences: List[np.ndarray] = []
    centralized_labels: List[np.ndarray] = []

    for client_id, df in partitions.items():
        client_df = df.copy()
        if tb06_config.clients.feature_shift_std > 0:
            _apply_feature_shift(client_df, feature_cols, tb06_config.clients.feature_shift_std)
        sequences, labels = tb04_windowed_sequences(
            client_df,
            feature_cols,
            tb06_config.dataset.sequence_length,
            tb06_config.dataset.sequence_stride,
            tb06_config.dataset.max_sequences_per_client,
        )
        centralized_sequences.append(sequences)
        centralized_labels.append(labels)
        dataset = TensorDataset(
            torch.from_numpy(sequences),
            torch.from_numpy(labels),
        )
        client_partitions[client_id] = TB04ClientPartition(dataset=dataset, num_samples=len(dataset))

    if centralized_sequences:
        all_sequences = np.concatenate(centralized_sequences, axis=0)
        all_labels = np.concatenate(centralized_labels, axis=0)
    else:
        all_sequences = np.zeros((0, tb06_config.dataset.sequence_length, len(feature_cols)), dtype=np.float32)
        all_labels = np.zeros((0,), dtype=np.float32)
    centralized_dataset = TensorDataset(
        torch.from_numpy(all_sequences),
        torch.from_numpy(all_labels),
    )

    test_sequences, test_labels = tb04_windowed_sequences(
        test_df,
        feature_cols,
        tb06_config.dataset.sequence_length,
        tb06_config.dataset.sequence_stride,
        max_sequences=None,
    )
    test_dataset = TensorDataset(
        torch.from_numpy(test_sequences),
        torch.from_numpy(test_labels),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=tb06_config.federated.test_batch_size,
        shuffle=False,
    )

    trainer = BenefitEquityTrainer(
        tb06_config,
        client_partitions,
        centralized_dataset,
        test_loader,
        feature_cols=feature_cols,
    )
    metrics, client_details = trainer.evaluate(algo)
    return metrics, client_details


class BenefitEquityTrainer:
    """Trains centralized + FL models and computes benefit equity per client."""

    def __init__(
        self,
        config: TB06Config,
        clients: Dict[int, TB04ClientPartition],
        centralized_dataset: TensorDataset,
        test_loader: DataLoader,
        feature_cols: List[str],
    ) -> None:
        self.config = config
        self.clients = clients
        self.centralized_dataset = centralized_dataset
        self.test_loader = test_loader
        self.input_size = len(feature_cols)
        self.device = torch.device(config.evaluation.device)
        self.criterion = nn.MSELoss()

    def evaluate(self, algo: str) -> Tuple[List[Dict[str, float]], List[Dict[str, object]]]:
        print(f"[TB-06] Training centralized baseline for {self.config.baseline.epochs} epochs.")
        central_state = self._train_centralized()
        central_model = self._new_model()
        central_model.load_state_dict(central_state)
        central_model.to(self.device)

        print(f"[TB-06] Training {algo} FL model for {self.config.federated.rounds} rounds.")
        fl_state = self._train_fl(algo)
        fl_model = self._new_model()
        fl_model.load_state_dict(fl_state)
        fl_model.to(self.device)

        central_scores = self._per_client_rmse(central_model)
        fl_scores = self._per_client_rmse(fl_model)

        benefits = []
        client_rows = []
        for client_id in sorted(self.clients.keys()):
            central_value = central_scores.get(client_id, float("nan"))
            fl_value = fl_scores.get(client_id, float("nan"))
            benefit = central_value - fl_value
            benefits.append(benefit)
            client_rows.append(
                {
                    "client_id": client_id,
                    "central_rmse": central_value,
                    "fl_rmse": fl_value,
                    "benefit": benefit,
                }
            )

        benefit_equity = 1.0 - _gini(benefits)
        metrics = [{"metric_id": "M.FL.BENEFIT_EQ", "value": benefit_equity}]

        return metrics, client_rows

    # -------------------- Training helpers --------------------
    def _train_centralized(self) -> Dict[str, torch.Tensor]:
        model = self._new_model()
        model.to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.baseline.lr,
            weight_decay=self.config.baseline.weight_decay,
        )
        loader = DataLoader(
            self.centralized_dataset,
            batch_size=self.config.baseline.batch_size,
            shuffle=True,
        )

        for _ in range(self.config.baseline.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                preds = model(batch_x)
                loss = self.criterion(preds, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}

    def _train_fl(self, algo: str) -> Dict[str, torch.Tensor]:
        algo_key = algo.lower()
        if algo_key not in {"fedavg", "fedprox", "scaffold"}:
            raise ValueError(f"Unsupported algorithm '{algo}'.")

        model = self._new_model()
        current_state = model.state_dict()
        c_global = None
        c_clients: Dict[int, Dict[str, torch.Tensor]] = {}
        if algo_key == "scaffold":
            c_global = _zero_like_state(current_state, device=self.device)
            c_clients = {client_id: _zero_like_state(current_state, device=self.device) for client_id in self.clients}

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
                if partition.num_samples == 0:
                    continue
                loader = DataLoader(
                    partition.dataset,
                    batch_size=self.config.federated.batch_size,
                    shuffle=True,
                )
                local_model = self._new_model()
                local_model.load_state_dict(current_state)
                local_model.to(self.device)
                global_params = {key: value.to(self.device) for key, value in current_state.items()}

                update_state, sample_count, delta_c = self._train_fl_client(
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
                current_state = _aggregate_weighted(updates)
                if algo_key == "scaffold" and delta_cs and c_global is not None:
                    c_global = _update_control_variate(c_global, delta_cs, self.config.federated.scaffold_eta)

        return {name: tensor.clone().cpu() for name, tensor in current_state.items()}

    def _train_fl_client(
        self,
        model: SequenceRegressor,
        loader: DataLoader,
        algo: str,
        global_state: Dict[str, torch.Tensor],
        c_global: Dict[str, torch.Tensor] | None,
        c_client: Dict[str, torch.Tensor] | None,
    ) -> Tuple[Dict[str, torch.Tensor], int, Dict[str, torch.Tensor] | None]:
        model.train()
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

    # -------------------- Evaluation helpers --------------------
    def _per_client_rmse(self, model: SequenceRegressor) -> Dict[int, float]:
        model.eval()
        rmse_per_client: Dict[int, float] = {}
        with torch.no_grad():
            for client_id, partition in self.clients.items():
                loader = DataLoader(
                    partition.dataset,
                    batch_size=self.config.federated.test_batch_size,
                    shuffle=False,
                )
                preds: List[float] = []
                targets: List[float] = []
                for batch_x, batch_y in loader:
                    batch_x = batch_x.to(self.device)
                    outputs = model(batch_x).cpu().numpy()
                    preds.extend(outputs.tolist())
                    targets.extend(batch_y.numpy().tolist())
                if preds:
                    rmse_per_client[client_id] = float(np.sqrt(np.mean(np.square(np.array(preds) - np.array(targets)))))
        return rmse_per_client

    def _new_model(self) -> SequenceRegressor:
        return SequenceRegressor(self.input_size, self.config.model)


# -------------------- Utility functions --------------------
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


def _gini(values: Sequence[float]) -> float:
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    shift = 0.0
    min_val = arr.min()
    if min_val < 0:
        shift = -min_val + 1e-9
    arr = arr + shift
    if np.allclose(arr, 0.0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    cumulative = np.cumsum(arr)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return float(gini)



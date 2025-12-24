"""TB-10 convergence performance across FL algorithms."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.cmapss import prepare_cmapss_dataset
from src.data.partitioning import dirichlet_quantity_skew
from src.testbeds.tb04 import (
    DatasetConfig as TB04DatasetConfig,
    ClientConfig as TB04ClientConfig,
    FederatedConfig as TB04FederatedConfig,
    ModelConfig as TB04ModelConfig,
    SequenceRegressor,
    ClientPartition as TB04ClientPartition,
    _apply_feature_shift,
    _feature_columns,
    _windowed_sequences,
)


@dataclass(slots=True)
class EvalConfig:
    device: str
    seed: int
    target_rmse: float


@dataclass(slots=True)
class TB10Config:
    dataset: TB04DatasetConfig
    clients: TB04ClientConfig
    federated: TB04FederatedConfig
    model: TB04ModelConfig
    evaluation: EvalConfig


def load_tb10_config(config_path: Path) -> TB10Config:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    dataset_cfg = TB04DatasetConfig(
        dataset_id=data["dataset"].get("id", "FD002"),
        sequence_length=int(data["dataset"].get("sequence_length", 30)),
        sequence_stride=int(data["dataset"].get("sequence_stride", 10)),
        normalize=bool(data["dataset"].get("normalize", True)),
        max_sequences_per_client=int(data["dataset"].get("max_sequences_per_client", 500)),
    )
    clients_cfg = TB04ClientConfig(
        num_clients=int(data["clients"].get("num_clients", 20)),
        dirichlet_alpha=float(data["clients"].get("dirichlet_alpha", 0.6)),
        feature_shift_std=float(data["clients"].get("feature_shift_std", 0.05)),
        seed=int(data["clients"].get("seed", 2033)),
    )
    federated_cfg = TB04FederatedConfig(
        rounds=int(data["federated"].get("rounds", 200)),
        clients_per_round=float(data["federated"].get("clients_per_round", 1.0)),
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
    eval_cfg = EvalConfig(
        device=str(data["evaluation"].get("device", "cpu")),
        seed=int(data["evaluation"].get("seed", 2033)),
        target_rmse=float(data["evaluation"].get("target_rmse", 110.0)),
    )

    return TB10Config(
        dataset=dataset_cfg,
        clients=clients_cfg,
        federated=federated_cfg,
        model=model_cfg,
        evaluation=eval_cfg,
    )


def run_tb10_pipeline(
    project_root: Path,
    algo: str,
    rounds_override: int | None = None,
) -> Tuple[List[Dict[str, float]], Dict[str, object]]:
    """Execute TB-10 workflow for a specific algorithm."""

    config_path = project_root / "config" / "config_v2" / "tb10_convergence.yaml"
    tb10_config = load_tb10_config(config_path)
    if rounds_override is not None and rounds_override > 0:
        tb10_config.federated.rounds = int(rounds_override)

    random.seed(tb10_config.clients.seed)
    np.random.seed(tb10_config.clients.seed)
    torch.manual_seed(tb10_config.evaluation.seed)

    train_df, val_df, _ = prepare_cmapss_dataset(
        dataset_id=tb10_config.dataset.dataset_id,
        root=project_root / "data" / "c-mapss",
        normalize=tb10_config.dataset.normalize,
        rul_clip=125,
    )
    feature_cols = _feature_columns(train_df)

    partitions = dirichlet_quantity_skew(
        train_df,
        num_clients=tb10_config.clients.num_clients,
        alpha=tb10_config.clients.dirichlet_alpha,
        seed=tb10_config.clients.seed,
    )

    client_partitions: Dict[int, TB04ClientPartition] = {}
    for client_id, df in partitions.items():
        client_df = df.copy()
        if tb10_config.clients.feature_shift_std > 0:
            _apply_feature_shift(client_df, feature_cols, tb10_config.clients.feature_shift_std)
        sequences, labels = _windowed_sequences(
            client_df,
            feature_cols,
            tb10_config.dataset.sequence_length,
            tb10_config.dataset.sequence_stride,
            tb10_config.dataset.max_sequences_per_client,
        )
        if len(sequences) == 0:
            continue
        dataset = TensorDataset(
            torch.from_numpy(sequences),
            torch.from_numpy(labels),
        )
        client_partitions[client_id] = TB04ClientPartition(dataset=dataset, num_samples=len(dataset))

    if not client_partitions:
        raise RuntimeError("No client partitions available for TB-10.")

    val_sequences, val_labels = _windowed_sequences(
        val_df,
        feature_cols,
        tb10_config.dataset.sequence_length,
        tb10_config.dataset.sequence_stride,
        max_sequences=None,
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_sequences),
        torch.from_numpy(val_labels),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=tb10_config.federated.test_batch_size,
        shuffle=False,
    )

    trainer = ConvergenceTrainer(
        config=tb10_config,
        clients=client_partitions,
        val_loader=val_loader,
        input_size=len(feature_cols),
    )
    stats = trainer.run(algo)

    measurement_entries = [
        {
            "metric_id": "M.FL.CONV_TIME",
            "value": stats["convergence_round"],
            "extras": {
                "target_rmse": stats["target_rmse"],
                "achieved_rmse": stats["best_rmse"],
                "converged": stats["converged"],
                "rounds": stats["convergence_round"],
                "time_seconds": stats["convergence_time_seconds"],
            },
        }
    ]

    details = {
        "target_rmse": stats["target_rmse"],
        "curve": stats["curve"],
        "convergence_round": stats["convergence_round"],
        "convergence_time_seconds": stats["convergence_time_seconds"],
        "converged": stats["converged"],
    }

    return measurement_entries, details


class ConvergenceTrainer:
    """Federated trainer that records convergence curves."""

    def __init__(
        self,
        config: TB10Config,
        clients: Dict[int, TB04ClientPartition],
        val_loader: DataLoader,
        input_size: int,
    ) -> None:
        self.config = config
        self.clients = clients
        self.val_loader = val_loader
        self.input_size = input_size
        self.device = torch.device(config.evaluation.device)
        self.criterion = nn.MSELoss()

    def run(self, algo: str) -> Dict[str, object]:
        algo_key = algo.lower()
        if algo_key not in {"fedavg", "fedprox", "scaffold"}:
            raise ValueError(f"Unsupported algorithm '{algo}'.")

        global_model = self._new_model()
        global_state = global_model.state_dict()

        c_global = None
        c_clients: Dict[int, Dict[str, torch.Tensor]] = {}
        if algo_key == "scaffold":
            c_global = _zero_like_state(global_state, device=self.device)
            c_clients = {
                client_id: _zero_like_state(global_state, device=self.device)
                for client_id in self.clients
            }

        client_ids = list(self.clients.keys())
        rounds = self.config.federated.rounds
        clients_per_round = max(
            1, int(math.ceil(self.config.federated.clients_per_round * len(client_ids)))
        )

        curve: List[Dict[str, float]] = []
        best_rmse = float("inf")
        convergence_round = rounds + 1
        convergence_time = 0.0
        converged = False

        start_time = time.perf_counter()
        for round_idx in range(1, rounds + 1):
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
                local_model.load_state_dict(global_state)
                local_model.to(self.device)
                global_params = {key: value.to(self.device) for key, value in global_state.items()}

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
                global_state = _aggregate_weighted(updates)
                if algo_key == "scaffold" and delta_cs and c_global is not None:
                    c_global = _update_control_variate(c_global, delta_cs, self.config.federated.scaffold_eta)

            global_model.load_state_dict(global_state)
            global_model.to(self.device)
            val_rmse = self._evaluate(global_model)
            elapsed = time.perf_counter() - start_time

            curve.append(
                {
                    "round": round_idx,
                    "val_rmse": val_rmse,
                    "elapsed_seconds": elapsed,
                }
            )

            if val_rmse < best_rmse:
                best_rmse = val_rmse

            if not converged and val_rmse <= self.config.evaluation.target_rmse:
                convergence_round = round_idx
                convergence_time = elapsed
                converged = True

        if not converged:
            convergence_time = time.perf_counter() - start_time

        return {
            "target_rmse": self.config.evaluation.target_rmse,
            "best_rmse": best_rmse,
            "convergence_round": convergence_round,
            "convergence_time_seconds": convergence_time,
            "converged": converged,
            "curve": curve,
        }

    def _train_client(
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

        return state_dict, total_samples, delta_c

    def _evaluate(self, model: SequenceRegressor) -> float:
        model.eval()
        preds: List[float] = []
        targets: List[float] = []

        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(self.device)
                outputs = model(batch_x).cpu().numpy()
                preds.extend(outputs.tolist())
                targets.extend(batch_y.numpy().tolist())

        preds_arr = np.array(preds, dtype=np.float64)
        targets_arr = np.array(targets, dtype=np.float64)
        rmse = float(np.sqrt(np.mean(np.square(preds_arr - targets_arr))))
        return rmse

    def _new_model(self) -> SequenceRegressor:
        return SequenceRegressor(self.input_size, self.config.model)


def _zero_like_state(state: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: torch.zeros_like(value, device=device) for key, value in state.items()}


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


def _update_control_variate(
    c_global: Dict[str, torch.Tensor],
    deltas: List[Dict[str, torch.Tensor]],
    eta: float,
) -> Dict[str, torch.Tensor]:
    if not deltas:
        return c_global
    avg_delta: Dict[str, torch.Tensor] = {}
    for key in c_global.keys():
        stacked = torch.stack([delta[key] for delta in deltas], dim=0)
        avg_delta[key] = stacked.mean(dim=0)
    return {key: c_global[key] + eta * avg_delta[key] for key in c_global.keys()}



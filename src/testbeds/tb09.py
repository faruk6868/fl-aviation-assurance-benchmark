"""TB-09 communication budget & compression efficiency orchestration."""

from __future__ import annotations

import math
import random
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
    _windowed_sequences,
    _apply_feature_shift,
    _feature_columns,
)


@dataclass(slots=True)
class CompressionVariant:
    name: str
    upload_ratio: float
    download_ratio: float
    utility_penalty: float = 0.0


@dataclass(slots=True)
class EvalConfig:
    device: str
    seed: int
    utility_guardrail: float


@dataclass(slots=True)
class TB09Config:
    dataset: TB04DatasetConfig
    clients: TB04ClientConfig
    federated: TB04FederatedConfig
    model: TB04ModelConfig
    evaluation: EvalConfig
    compression_variants: List[CompressionVariant]


def load_tb09_config(config_path: Path) -> TB09Config:
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
        seed=int(data["clients"].get("seed", 2032)),
    )
    federated_cfg = TB04FederatedConfig(
        rounds=int(data["federated"].get("rounds", 100)),
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
        seed=int(data["evaluation"].get("seed", 2032)),
        utility_guardrail=float(data["evaluation"].get("utility_guardrail", 0.05)),
    )

    variants = []
    for entry in data.get("compression_variants", []):
        variants.append(
            CompressionVariant(
                name=str(entry.get("name", "variant")),
                upload_ratio=float(entry.get("upload_ratio", 1.0)),
                download_ratio=float(entry.get("download_ratio", 1.0)),
                utility_penalty=float(entry.get("utility_penalty", 0.0)),
            )
        )
    if not variants:
        variants.append(CompressionVariant(name="none", upload_ratio=1.0, download_ratio=1.0, utility_penalty=0.0))

    return TB09Config(
        dataset=dataset_cfg,
        clients=clients_cfg,
        federated=federated_cfg,
        model=model_cfg,
        evaluation=eval_cfg,
        compression_variants=variants,
    )


def run_tb09_pipeline(
    project_root: Path,
    algo: str,
    rounds_override: int | None = None,
) -> Tuple[List[Dict[str, float]], Dict[str, object]]:
    """Execute TB-09 workflow for a specific algorithm."""

    config_path = project_root / "config" / "config_v2" / "tb09_communication.yaml"
    tb09_config = load_tb09_config(config_path)
    if rounds_override is not None and rounds_override > 0:
        tb09_config.federated.rounds = int(rounds_override)

    random.seed(tb09_config.clients.seed)
    np.random.seed(tb09_config.clients.seed)
    torch.manual_seed(tb09_config.evaluation.seed)

    train_df, test_df, _ = prepare_cmapss_dataset(
        dataset_id=tb09_config.dataset.dataset_id,
        root=project_root / "data" / "c-mapss",
        normalize=tb09_config.dataset.normalize,
        rul_clip=125,
    )
    feature_cols = _feature_columns(train_df)

    partitions = dirichlet_quantity_skew(
        train_df,
        num_clients=tb09_config.clients.num_clients,
        alpha=tb09_config.clients.dirichlet_alpha,
        seed=tb09_config.clients.seed,
    )

    client_partitions: Dict[int, TB04ClientPartition] = {}
    for client_id, df in partitions.items():
        client_df = df.copy()
        if tb09_config.clients.feature_shift_std > 0:
            _apply_feature_shift(client_df, feature_cols, tb09_config.clients.feature_shift_std)
        sequences, labels = _windowed_sequences(
            client_df,
            feature_cols,
            tb09_config.dataset.sequence_length,
            tb09_config.dataset.sequence_stride,
            tb09_config.dataset.max_sequences_per_client,
        )
        if len(sequences) == 0:
            continue
        dataset = TensorDataset(
            torch.from_numpy(sequences),
            torch.from_numpy(labels),
        )
        client_partitions[client_id] = TB04ClientPartition(dataset=dataset, num_samples=len(dataset))

    if not client_partitions:
        raise RuntimeError("No client partitions available for TB-09.")

    test_sequences, test_labels = _windowed_sequences(
        test_df,
        feature_cols,
        tb09_config.dataset.sequence_length,
        tb09_config.dataset.sequence_stride,
        max_sequences=None,
    )
    test_dataset = TensorDataset(
        torch.from_numpy(test_sequences),
        torch.from_numpy(test_labels),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=tb09_config.federated.test_batch_size,
        shuffle=False,
    )

    trainer = CommunicationAwareTrainer(
        config=tb09_config,
        clients=client_partitions,
        test_loader=test_loader,
        input_size=len(feature_cols),
    )

    metrics, stats = trainer.run(algo)

    base_client = np.array(stats["round_client_bytes"], dtype=np.float64)
    base_server = np.array(stats["round_server_bytes"], dtype=np.float64)
    base_total = base_client + base_server
    avg_uncompressed = float(np.mean(base_total))

    measurement_entries: List[Dict[str, object]] = []
    variant_summaries: List[Dict[str, object]] = []

    baseline_rmse = metrics["rmse"]

    for variant in tb09_config.compression_variants:
        compressed_client = base_client * variant.upload_ratio
        compressed_server = base_server * variant.download_ratio
        total_bytes = compressed_client + compressed_server
        avg_bytes = float(np.mean(total_bytes))
        if avg_bytes == 0.0:
            avg_bytes = 1e-12
        # Convert bytes to MB for M.FL.COMM_BYTES metric (threshold is in MB/round)
        avg_bytes_mb = avg_bytes / (1024.0 * 1024.0)
        # Communication Efficiency Ratio = uncompressed/compressed
        # Higher ratio = better compression (more efficient communication)
        # Example: uncompressed=32.952 MB, compressed=16.476 MB → ratio = 2.0 (2x compression)
        comm_efficiency_ratio = float(avg_uncompressed / avg_bytes)  # uncompressed/compressed (higher = better)
        utility_rmse = float(baseline_rmse * (1.0 + variant.utility_penalty))
        utility_delta = float(variant.utility_penalty)
        guardrail_pass = utility_delta <= tb09_config.evaluation.utility_guardrail

        extras = {
            "variant": variant.name,
            "unit": "MB/round",
            "avg_bytes": avg_bytes,
            "avg_bytes_mb": avg_bytes_mb,
            "avg_uncompressed": avg_uncompressed,
            "utility_rmse": utility_rmse,
            "utility_delta": utility_delta,
            "guardrail_pass": guardrail_pass,
            "upload_ratio": variant.upload_ratio,
            "download_ratio": variant.download_ratio,
        }

        measurement_entries.append(
            {
                "metric_id": "M.FL.COMM_BYTES",
                "value": avg_bytes_mb,  # Report in MB/round to match threshold unit
                "extras": extras,
            }
        )
        measurement_entries.append(
            {
                "metric_id": "M.FL.COMM_COMP",
                "value": comm_efficiency_ratio,  # uncompressed/compressed (higher = better compression)
                "extras": extras,
            }
        )

        variant_summaries.append(
            {
                "variant": variant.name,
                "avg_bytes_per_round": avg_bytes,
                "avg_bytes_per_round_mb": avg_bytes_mb,
                "avg_uncompressed_bytes": avg_uncompressed,
                "compression_efficiency_ratio": comm_efficiency_ratio,
                "compression_ratio": float(avg_bytes / avg_uncompressed),  # compressed/uncompressed (for reference)
                "utility_rmse": utility_rmse,
                "utility_delta": utility_delta,
                "guardrail_pass": guardrail_pass,
                "upload_ratio": variant.upload_ratio,
                "download_ratio": variant.download_ratio,
            }
        )

    details = {
        "baseline_rmse": baseline_rmse,
        "round_client_bytes": stats["round_client_bytes"],
        "round_server_bytes": stats["round_server_bytes"],
        "model_payload_bytes": stats["model_payload_bytes"],
        "variants": variant_summaries,
    }

    return measurement_entries, details


class CommunicationAwareTrainer:
    """Federated trainer that records per-round communication load."""

    def __init__(
        self,
        config: TB09Config,
        clients: Dict[int, TB04ClientPartition],
        test_loader: DataLoader,
        input_size: int,
    ) -> None:
        self.config = config
        self.clients = clients
        self.test_loader = test_loader
        self.input_size = input_size
        self.device = torch.device(config.evaluation.device)
        self.criterion = nn.MSELoss()
        self.round_client_bytes: List[float] = []
        self.round_server_bytes: List[float] = []

    def run(self, algo: str) -> Tuple[Dict[str, float], Dict[str, object]]:
        algo_key = algo.lower()
        if algo_key not in {"fedavg", "fedprox", "scaffold"}:
            raise ValueError(f"Unsupported algorithm '{algo}'.")

        global_model = self._new_model()
        global_state = global_model.state_dict()
        payload_bytes = self._payload_size_bytes(global_state)

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

            self.round_client_bytes.append(len(selected) * payload_bytes)
            self.round_server_bytes.append(len(selected) * payload_bytes)

        global_model.load_state_dict(global_state)
        global_model.to(self.device)
        metrics = self._evaluate(global_model)
        stats = {
            "round_client_bytes": self.round_client_bytes,
            "round_server_bytes": self.round_server_bytes,
            "model_payload_bytes": payload_bytes,
        }
        return metrics, stats

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

    def _evaluate(self, model: SequenceRegressor) -> Dict[str, float]:
        model.eval()
        preds: List[float] = []
        targets: List[float] = []

        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x = batch_x.to(self.device)
                outputs = model(batch_x).cpu().numpy()
                preds.extend(outputs.tolist())
                targets.extend(batch_y.numpy().tolist())

        preds_arr = np.array(preds, dtype=np.float64)
        targets_arr = np.array(targets, dtype=np.float64)

        rmse = float(np.sqrt(np.mean(np.square(preds_arr - targets_arr))))
        return {"rmse": rmse}

    def _new_model(self) -> SequenceRegressor:
        return SequenceRegressor(self.input_size, self.config.model)

    @staticmethod
    def _payload_size_bytes(state: Dict[str, torch.Tensor]) -> float:
        size = 0
        for tensor in state.values():
            size += tensor.numel() * tensor.element_size()
        return float(size)


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



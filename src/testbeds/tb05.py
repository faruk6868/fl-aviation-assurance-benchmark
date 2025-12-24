"""TB-05 cross-population stability and transformation impact evaluation."""

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
    groups: Dict[str, str]


@dataclass(slots=True)
class TransformConfig:
    type: str
    noise_std: float


@dataclass(slots=True)
class TB05Config:
    dataset: TB04DatasetConfig
    clients: ClientConfig
    federated: TB04FederatedConfig
    model: TB04ModelConfig
    transforms: TransformConfig
    evaluation: TB04EvalConfig


def load_tb05_config(config_path: Path) -> TB05Config:
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
        seed=int(data["clients"].get("seed", 2027)),
        groups={str(k): str(v) for k, v in (data["clients"].get("groups") or {}).items()},
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
    transform_cfg = TransformConfig(
        type=str(data["transforms"].get("type", "quantization")),
        noise_std=float(data["transforms"].get("noise_std", 0.01)),
    )
    eval_cfg = TB04EvalConfig(
        device=str(data["evaluation"].get("device", "cpu")),
        seed=int(data["evaluation"].get("seed", 2027)),
    )
    return TB05Config(
        dataset=dataset_cfg,
        clients=client_cfg,
        federated=federated_cfg,
        model=model_cfg,
        transforms=transform_cfg,
        evaluation=eval_cfg,
    )


def run_tb05_pipeline(
    project_root: Path,
    algo: str,
    rounds_override: int | None = None,
) -> Tuple[List[Dict[str, float]], List[Dict[str, object]]]:
    """Run cross-population stability assessment for the given algorithm."""

    config_path = project_root / "config" / "config_v2" / "tb05_cross_pop_stability.yaml"
    tb05_config = load_tb05_config(config_path)
    if rounds_override is not None and rounds_override > 0:
        tb05_config.federated.rounds = int(rounds_override)

    random.seed(tb05_config.clients.seed)
    np.random.seed(tb05_config.clients.seed)
    torch.manual_seed(tb05_config.evaluation.seed)

    train_df, test_df, _ = prepare_cmapss_dataset(
        dataset_id=tb05_config.dataset.dataset_id,
        root=project_root / "data" / "c-mapss",
        normalize=tb05_config.dataset.normalize,
    )

    partitions = dirichlet_quantity_skew(
        train_df,
        num_clients=tb05_config.clients.num_clients,
        alpha=tb05_config.clients.dirichlet_alpha,
        seed=tb05_config.clients.seed,
    )

    feature_cols = tb04_feature_columns(train_df)
    client_partitions: Dict[int, TB04ClientPartition] = {}
    for client_id, df in partitions.items():
        client_df = df.copy()
        if tb05_config.clients.feature_shift_std > 0:
            _apply_feature_shift(client_df, feature_cols, tb05_config.clients.feature_shift_std)
        sequences, labels = tb04_windowed_sequences(
            client_df,
            feature_cols,
            tb05_config.dataset.sequence_length,
            tb05_config.dataset.sequence_stride,
            tb05_config.dataset.max_sequences_per_client,
        )
        dataset = TensorDataset(
            torch.from_numpy(sequences),
            torch.from_numpy(labels),
        )
        client_partitions[client_id] = TB04ClientPartition(dataset=dataset, num_samples=len(dataset))

    test_sequences, test_labels = tb04_windowed_sequences(
        test_df,
        feature_cols,
        tb05_config.dataset.sequence_length,
        tb05_config.dataset.sequence_stride,
        max_sequences=None,
    )
    test_dataset = TensorDataset(
        torch.from_numpy(test_sequences),
        torch.from_numpy(test_labels),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=tb05_config.federated.test_batch_size,
        shuffle=False,
    )

    trainer = FederatedRegressorTrainer(
        tb05_config,
        client_partitions,
        test_loader,
        feature_cols=feature_cols,
    )
    metrics, group_details = trainer.evaluate(algo)

    return metrics, group_details


class FederatedRegressorTrainer:
    """Runs FL once and evaluates group stability/transform impact."""

    def __init__(
        self,
        config: TB05Config,
        clients: Dict[int, TB04ClientPartition],
        test_loader: DataLoader,
        feature_cols: List[str],
    ) -> None:
        self.config = config
        self.clients = clients
        self.test_loader = test_loader
        self.input_size = len(feature_cols)
        self.device = torch.device(config.evaluation.device)
        self.criterion = nn.MSELoss()
        self.feature_cols = feature_cols
        self._latest_state: Dict[str, torch.Tensor] | None = None

    def evaluate(self, algo: str) -> Tuple[List[Dict[str, float]], List[Dict[str, object]]]:
        model_state = self._train_model(algo)
        baseline_model = self._new_model()
        baseline_model.load_state_dict(model_state)
        baseline_model.to(self.device)

        group_perf = self._group_performance(baseline_model)
        rmse_values = group_perf["pre_metric"]
        # M.PERF.CROSS_STAB: Coefficient of Variation = std/mean
        mean_rmse = float(np.mean(rmse_values))
        std_rmse = float(np.std(rmse_values))
        cross_stab = std_rmse / (mean_rmse + 1e-8) if mean_rmse > 0 else 0.0
        
        # M.FL.WORST_CLIENT: Worst-group performance as ratio vs. average
        # For RMSE, worst = highest RMSE (max)
        # But threshold 0.75-0.9 suggests we need a "performance" metric (higher is better)
        # Convert RMSE to performance: use inverse or normalized score
        # Worst-group performance = min(performance) / mean(performance)
        # where performance = 1 / (1 + RMSE) or similar normalization
        # Actually, for fairness: worst_group_rmse / mean_rmse should be close to 1.0
        # But since RMSE is "lower is better", we need: best_rmse / worst_rmse
        # Or: mean_rmse / worst_rmse (how close worst is to mean)
        worst_rmse = float(max(rmse_values))
        # Ratio: mean/worst (how close worst is to average, higher is better)
        # Threshold 0.75-0.9 means worst should be within 75-90% of average
        worst_perf = mean_rmse / (worst_rmse + 1e-8) if worst_rmse > 0 else 0.0

        transformed_model = self._apply_transformation(baseline_model)
        group_perf_post = self._group_performance(transformed_model)
        deltas = [post - pre for post, pre in zip(group_perf_post["pre_metric"], group_perf["pre_metric"])]
        max_delta = float(max(abs(d) for d in deltas))

        metrics_list = [
            {"metric_id": "M.PERF.CROSS_STAB", "value": cross_stab},
            {"metric_id": "M.FL.WORST_CLIENT", "value": worst_perf},
            {"metric_id": "M.XFORM.DELTA", "value": max_delta},
        ]

        group_details = []
        for group_id, pre_value, post_value, delta in zip(
            group_perf["group_id"],
            group_perf["pre_metric"],
            group_perf_post["pre_metric"],
            deltas,
        ):
            group_details.append(
                {
                    "group_id": group_id,
                    "pre_metric": pre_value,
                    "post_metric": post_value,
                    "delta": delta,
                }
            )

        return metrics_list, group_details

    def _train_model(self, algo: str) -> Dict[str, torch.Tensor]:
        algo_key = algo.lower()
        if algo_key not in {"fedavg", "fedprox", "scaffold"}:
            raise ValueError(f"Unsupported algorithm '{algo}'.")

        model = self._new_model()
        model_state = model.state_dict()

        c_global = None
        c_clients: Dict[int, Dict[str, torch.Tensor]] = {}
        if algo_key == "scaffold":
            c_global = _zero_like_state(model_state, device=self.device)
            c_clients = {
                client_id: _zero_like_state(model_state, device=self.device)
                for client_id in self.clients
            }

        client_ids = list(self.clients.keys())
        rounds = self.config.federated.rounds
        clients_per_round = max(1, int(math.ceil(self.config.federated.clients_per_round * len(client_ids))))

        current_state = {name: tensor.clone() for name, tensor in model_state.items()}
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
                current_state = _aggregate_weighted(updates)
                if algo_key == "scaffold" and delta_cs and c_global is not None:
                    c_global = _update_control_variate(c_global, delta_cs, self.config.federated.scaffold_eta)

        self._latest_state = {name: tensor.clone().cpu() for name, tensor in current_state.items()}
        return self._latest_state

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

        # Store initial parameters for Scaffold control variate correction
        initial_params: Dict[str, torch.Tensor] | None = None
        if algo == "scaffold":
            initial_params = {
                name: param.detach().clone()
                for name, param in model.named_parameters()
            }

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
                        if param.grad is not None:
                            # Standard SCAFFOLD gradient correction: g - c_global + c_client
                            param.grad = param.grad - c_global[name] + c_client[name]

                optimizer.step()

                total_samples += batch_x.size(0)
                total_steps += 1

        state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}

        delta_c: Dict[str, torch.Tensor] | None = None
        if algo == "scaffold" and c_client is not None and c_global is not None and initial_params is not None and total_steps > 0:
            delta_c = {}
            lr = self.config.federated.lr
            # Standard SCAFFOLD formula: c_i_new = c_i_old - c + (1/(K*η)) * (w_old - w_new)
            # where K = total_steps (number of local steps), η = learning rate
            for name, param in model.named_parameters():
                initial_param = initial_params[name]
                final_param = param.detach()
                correction = (initial_param - final_param) / (total_steps * lr)
                new_control = c_client[name] - c_global[name] + correction
                delta_c[name] = (new_control - c_client[name]).cpu()

        samples = total_samples if total_samples else sum(loader.batch_size for _ in loader)
        if samples == 0:
            samples = loader.batch_size * max(1, len(loader))
        return state_dict, samples, delta_c

    def _group_performance(self, model: SequenceRegressor) -> Dict[str, List[float]]:
        model.eval()
        group_scores: Dict[str, List[float]] = {}
        group_map = self.config.clients.groups

        with torch.no_grad():
            for client_id, partition in self.clients.items():
                group_id = group_map.get(str(client_id), "unknown")
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
                    rmse = float(np.sqrt(np.mean(np.square(np.array(preds) - np.array(targets)))))
                    group_scores.setdefault(group_id, []).append(rmse)

        group_ids: List[str] = []
        rmse_values: List[float] = []
        for group_id, rmse_list in group_scores.items():
            group_ids.append(group_id)
            rmse_values.append(float(np.mean(rmse_list)))

        return {"group_id": group_ids, "pre_metric": rmse_values}

    def _apply_transformation(self, model: SequenceRegressor) -> SequenceRegressor:
        new_model = self._new_model()
        new_model.load_state_dict(model.state_dict())
        new_model.to(self.device)
        if self.config.transforms.type == "quantization":
            self._add_noise(new_model, self.config.transforms.noise_std)
        return new_model

    def _add_noise(self, model: nn.Module, noise_std: float) -> None:
        for param in model.parameters():
            noise = torch.normal(mean=0.0, std=noise_std, size=param.data.size(), device=param.data.device)
            param.data.add_(noise)

    def _new_model(self) -> SequenceRegressor:
        return SequenceRegressor(self.input_size, self.config.model)


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



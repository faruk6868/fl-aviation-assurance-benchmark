"""TB-04 safety prognostics (RUL regression) orchestration."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.cmapss import prepare_cmapss_dataset
from src.data.partitioning import dirichlet_quantity_skew


@dataclass(slots=True)
class DatasetConfig:
    dataset_id: str
    sequence_length: int
    sequence_stride: int
    normalize: bool
    max_sequences_per_client: int


@dataclass(slots=True)
class ClientConfig:
    num_clients: int
    dirichlet_alpha: float
    feature_shift_std: float
    seed: int


@dataclass(slots=True)
class FederatedConfig:
    rounds: int
    clients_per_round: float
    local_epochs: int
    batch_size: int
    test_batch_size: int
    lr: float
    weight_decay: float
    fedprox_mu: float
    scaffold_eta: float


@dataclass(slots=True)
class ModelConfig:
    hidden_size: int
    num_layers: int
    dropout: float


@dataclass(slots=True)
class EvalConfig:
    device: str
    seed: int


@dataclass(slots=True)
class TB04Config:
    dataset: DatasetConfig
    clients: ClientConfig
    federated: FederatedConfig
    model: ModelConfig
    evaluation: EvalConfig


def load_tb04_config(config_path: Path) -> TB04Config:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = data.get("dataset", {})
    clients = data.get("clients", {})
    federated = data.get("federated", {})
    model = data.get("model", {})
    evaluation = data.get("evaluation", {})

    return TB04Config(
        dataset=DatasetConfig(
            dataset_id=dataset.get("id", "FD002"),
            sequence_length=int(dataset.get("sequence_length", 30)),
            sequence_stride=int(dataset.get("sequence_stride", 10)),
            normalize=bool(dataset.get("normalize", True)),
            max_sequences_per_client=int(dataset.get("max_sequences_per_client", 500)),
        ),
        clients=ClientConfig(
            num_clients=int(clients.get("num_clients", 20)),
            dirichlet_alpha=float(clients.get("dirichlet_alpha", 0.6)),
            feature_shift_std=float(clients.get("feature_shift_std", 0.05)),
            seed=int(clients.get("seed", 2026)),
        ),
        federated=FederatedConfig(
            rounds=int(federated.get("rounds", 100)),
            clients_per_round=float(federated.get("clients_per_round", 0.4)),
            local_epochs=int(federated.get("local_epochs", 1)),
            batch_size=int(federated.get("batch_size", 32)),
            test_batch_size=int(federated.get("test_batch_size", 128)),
            lr=float(federated.get("optimizer", {}).get("lr", 1e-3)),
            weight_decay=float(federated.get("optimizer", {}).get("weight_decay", 0.0)),
            fedprox_mu=float(federated.get("fedprox_mu", 0.01)),
            scaffold_eta=float(federated.get("scaffold_eta", 1.0)),
        ),
        model=ModelConfig(
            hidden_size=int(model.get("hidden_size", 64)),
            num_layers=int(model.get("num_layers", 1)),
            dropout=float(model.get("dropout", 0.2)),
        ),
        evaluation=EvalConfig(
            device=str(evaluation.get("device", "cpu")),
            seed=int(evaluation.get("seed", 2026)),
        ),
    )


class SequenceRegressor(nn.Module):
    """LSTM-based sequence regressor for RUL prediction."""

    def __init__(self, input_size: int, config: ModelConfig) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out, _ = self.lstm(x)
        logits = self.head(out[:, -1, :]).squeeze(-1)
        return logits


@dataclass(slots=True)
class ClientPartition:
    dataset: TensorDataset
    num_samples: int


def run_tb04_pipeline(project_root: Path, algo: str, rounds_override: int | None = None) -> List[Dict[str, float]]:
    """Execute TB-04 workflow for a specific algorithm."""

    config_path = project_root / "config" / "config_v2" / "tb04_rul_regression.yaml"
    tb04_config = load_tb04_config(config_path)
    if rounds_override is not None and rounds_override > 0:
        tb04_config.federated.rounds = int(rounds_override)
        print(f"[TB-04] Overriding rounds to {tb04_config.federated.rounds}.")

    random.seed(tb04_config.clients.seed)
    np.random.seed(tb04_config.clients.seed)
    torch.manual_seed(tb04_config.evaluation.seed)

    train_df, test_df, _ = prepare_cmapss_dataset(
        dataset_id=tb04_config.dataset.dataset_id,
        root=project_root / "data" / "c-mapss",
        rul_clip=125,
        normalize=tb04_config.dataset.normalize,
    )

    partitions = dirichlet_quantity_skew(
        train_df,
        num_clients=tb04_config.clients.num_clients,
        alpha=tb04_config.clients.dirichlet_alpha,
        seed=tb04_config.clients.seed,
    )

    feature_cols = _feature_columns(train_df)
    client_partitions: Dict[int, ClientPartition] = {}
    for client_id, df in partitions.items():
        client_df = df.copy()
        if tb04_config.clients.feature_shift_std > 0:
            _apply_feature_shift(client_df, feature_cols, tb04_config.clients.feature_shift_std)
        sequences, labels = _windowed_sequences(
            client_df,
            feature_cols,
            tb04_config.dataset.sequence_length,
            tb04_config.dataset.sequence_stride,
            tb04_config.dataset.max_sequences_per_client,
        )
        dataset = TensorDataset(
            torch.from_numpy(sequences),
            torch.from_numpy(labels),
        )
        client_partitions[client_id] = ClientPartition(dataset=dataset, num_samples=len(dataset))

    test_sequences, test_labels = _windowed_sequences(
        test_df,
        feature_cols,
        tb04_config.dataset.sequence_length,
        tb04_config.dataset.sequence_stride,
        max_sequences=None,
    )
    test_dataset = TensorDataset(
        torch.from_numpy(test_sequences),
        torch.from_numpy(test_labels),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=tb04_config.federated.test_batch_size,
        shuffle=False,
    )

    trainer = FederatedRegressorTrainer(
        tb04_config,
        client_partitions,
        test_loader,
        input_size=len(feature_cols),
    )
    print(
        f"[TB-04] Training {algo} for {tb04_config.federated.rounds} rounds "
        f"with {len(client_partitions)} clients."
    )
    metrics, model_state = trainer.run(algo)
    print(
        "[TB-04] Completed %s: RMSE=%.4f PHM08=%.4f"
        % (algo, metrics["rmse"], metrics["phm08"])
    )

    model_dir = project_root / "artifacts" / "testbeds" / "TB-04" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{algo}.pt"
    torch.save(model_state, model_path)
    print(f"[TB-04] Saved model checkpoint to {model_path}")

    return [
        {"metric_id": "M.SAF.RMSE", "value": metrics["rmse"]},
        {"metric_id": "M.PRG.PHM", "value": metrics["phm08"]},
    ]


class FederatedRegressorTrainer:
    """Synchronous FL trainer for regression."""

    def __init__(
        self,
        config: TB04Config,
        clients: Dict[int, ClientPartition],
        test_loader: DataLoader,
        input_size: int,
    ) -> None:
        self.config = config
        self.clients = clients
        self.test_loader = test_loader
        self.input_size = input_size
        self.device = torch.device(config.evaluation.device)
        self.criterion = nn.MSELoss()

    def run(self, algo: str) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
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
        clients_per_round = max(1, int(math.ceil(self.config.federated.clients_per_round * len(client_ids))))

        for round_idx in range(rounds):
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

                global_params = {
                    key: value.to(self.device)
                    for key, value in global_state.items()
                }

                update_state, sample_count, delta_c = self._train_client(
                    local_model,
                    loader,
                    algo_key,
                    global_params,
                    c_global,
                    c_clients.get(client_id),
                )
                updates.append((update_state, sample_count))
                if delta_c is not None and algo_key == "scaffold":
                    c_clients[client_id] = delta_c
                    delta_cs.append(delta_c)

            if not updates:
                continue

            global_state = _aggregate_weighted(updates)
            if algo_key == "scaffold" and delta_cs and c_global is not None:
                c_global = _update_control_variate(c_global, delta_cs, self.config.federated.scaffold_eta)

        global_model.load_state_dict(global_state)
        global_model.to(self.device)
        metrics = self._evaluate(global_model)
        final_state = {name: tensor.clone().cpu() for name, tensor in global_state.items()}
        return metrics, final_state

    def _new_model(self) -> SequenceRegressor:
        return SequenceRegressor(self.input_size, self.config.model)

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

        state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}

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
        phm08 = _phm08_score(preds_arr, targets_arr)

        return {"rmse": rmse, "phm08": phm08}


EXCLUDED_COLUMNS = {
    "engine_id",
    "cycle",
    "RUL",
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_01",
    "sensor_05",
    "sensor_06",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
}


def _feature_columns(df) -> List[str]:
    return [col for col in df.columns if col not in EXCLUDED_COLUMNS]


def _apply_feature_shift(df, feature_cols: Sequence[str], std: float) -> None:
    if std <= 0.0:
        return
    noise = np.random.normal(loc=0.0, scale=std, size=(len(df), len(feature_cols)))
    df.loc[:, feature_cols] = df[feature_cols] + noise.astype(np.float32)


def _windowed_sequences(
    df,
    feature_cols: Sequence[str],
    window: int,
    stride: int,
    max_sequences: int | None,
) -> Tuple[np.ndarray, np.ndarray]:
    sequences: List[np.ndarray] = []
    labels: List[float] = []

    for _, engine_df in df.groupby("engine_id"):
        engine_df = engine_df.sort_values("cycle")
        values = engine_df[feature_cols].to_numpy(dtype=np.float32)
        targets = engine_df["RUL"].to_numpy(dtype=np.float32)
        if len(values) <= window:
            continue
        for end_idx in range(window, len(engine_df), stride):
            start_idx = end_idx - window
            seq = values[start_idx:end_idx]
            rul = targets[end_idx - 1]
            sequences.append(seq)
            labels.append(rul)

    if not sequences:
        return (
            np.zeros((0, window, len(feature_cols)), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    seq_array = np.stack(sequences).astype(np.float32)
    label_array = np.array(labels, dtype=np.float32)

    if max_sequences is not None and len(seq_array) > max_sequences:
        indices = np.random.choice(len(seq_array), size=max_sequences, replace=False)
        seq_array = seq_array[indices]
        label_array = label_array[indices]

    return seq_array, label_array


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


def _zero_like_state(state: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
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


def _phm08_score(preds: np.ndarray, targets: np.ndarray) -> float:
    error = preds - targets
    score = np.where(
        error >= 0,
        np.exp(error / 13.0) - 1.0,
        np.exp((-error) / 10.0) - 1.0,
    )
    return float(np.sum(score))



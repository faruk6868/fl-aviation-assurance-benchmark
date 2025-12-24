"""TB-03 safety classification orchestration."""

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

from src.data.cmapss import add_failure_label, prepare_cmapss_dataset
from src.data.partitioning import dirichlet_quantity_skew


@dataclass(slots=True)
class DatasetConfig:
    dataset_id: str
    failure_horizon: int
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
    threshold: float
    seed: int


@dataclass(slots=True)
class TB03Config:
    dataset: DatasetConfig
    clients: ClientConfig
    federated: FederatedConfig
    model: ModelConfig
    evaluation: EvalConfig


def load_tb03_config(config_path: Path) -> TB03Config:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = data.get("dataset", {})
    clients = data.get("clients", {})
    federated = data.get("federated", {})
    model = data.get("model", {})
    evaluation = data.get("evaluation", {})

    return TB03Config(
        dataset=DatasetConfig(
            dataset_id=dataset.get("id", "FD002"),
            failure_horizon=int(dataset.get("failure_horizon", 30)),
            sequence_length=int(dataset.get("sequence_length", 30)),
            sequence_stride=int(dataset.get("sequence_stride", 5)),
            normalize=bool(dataset.get("normalize", True)),
            max_sequences_per_client=int(dataset.get("max_sequences_per_client", 600)),
        ),
        clients=ClientConfig(
            num_clients=int(clients.get("num_clients", 20)),
            dirichlet_alpha=float(clients.get("dirichlet_alpha", 0.6)),
            feature_shift_std=float(clients.get("feature_shift_std", 0.05)),
            seed=int(clients.get("seed", 1337)),
        ),
        federated=FederatedConfig(
            rounds=int(federated.get("rounds", 50)),
            clients_per_round=float(federated.get("clients_per_round", 0.5)),
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
            threshold=float(evaluation.get("threshold", 0.5)),
            seed=int(evaluation.get("seed", 2025)),
        ),
    )


class SequenceClassifier(nn.Module):
    """Single-layer LSTM sequence classifier."""

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
        out = out[:, -1, :]
        logits = self.head(out).squeeze(-1)
        return logits


@dataclass(slots=True)
class ClientPartition:
    dataset: TensorDataset
    num_samples: int


def run_tb03_pipeline(project_root: Path, algo: str, rounds_override: int | None = None) -> List[Dict[str, float]]:
    """Run the TB-03 workflow for a given algorithm."""

    config_path = project_root / "config" / "config_v2" / "tb03_safety_classification.yaml"
    tb03_config = load_tb03_config(config_path)
    random.seed(tb03_config.clients.seed)
    np.random.seed(tb03_config.clients.seed)
    torch.manual_seed(tb03_config.evaluation.seed)

    train_df, test_df, _ = prepare_cmapss_dataset(
        dataset_id=tb03_config.dataset.dataset_id,
        root=project_root / "data" / "c-mapss",
        normalize=tb03_config.dataset.normalize,
    )

    train_df = add_failure_label(train_df, tb03_config.dataset.failure_horizon)
    test_df = add_failure_label(test_df, tb03_config.dataset.failure_horizon)

    partitions = dirichlet_quantity_skew(
        train_df,
        num_clients=tb03_config.clients.num_clients,
        alpha=tb03_config.clients.dirichlet_alpha,
        seed=tb03_config.clients.seed,
    )

    feature_cols = _feature_columns(train_df)

    client_partitions: Dict[int, ClientPartition] = {}
    for client_id, df in partitions.items():
        client_df = df.copy()
        if tb03_config.clients.feature_shift_std > 0:
            _apply_feature_shift(client_df, feature_cols, tb03_config.clients.feature_shift_std)

        sequences, labels = _windowed_sequences(
            client_df,
            feature_cols,
            tb03_config.dataset.sequence_length,
            tb03_config.dataset.sequence_stride,
            tb03_config.dataset.max_sequences_per_client,
        )
        if len(sequences) == 0:
            dataset = TensorDataset(torch.zeros(0, tb03_config.dataset.sequence_length, len(feature_cols)), torch.zeros(0))
        else:
            dataset = TensorDataset(
                torch.from_numpy(sequences),
                torch.from_numpy(labels),
            )
        client_partitions[client_id] = ClientPartition(
            dataset=dataset,
            num_samples=len(dataset),
        )

    test_sequences, test_labels = _windowed_sequences(
        test_df,
        feature_cols,
        tb03_config.dataset.sequence_length,
        tb03_config.dataset.sequence_stride,
        max_sequences=None,
    )
    test_dataset = TensorDataset(
        torch.from_numpy(test_sequences),
        torch.from_numpy(test_labels),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=tb03_config.federated.test_batch_size,
        shuffle=False,
    )

    if rounds_override is not None and rounds_override > 0:
        tb03_config.federated.rounds = int(rounds_override)
        print(f"[TB-03] Overriding rounds to {tb03_config.federated.rounds}.")

    trainer = FederatedSequenceTrainer(
        tb03_config,
        client_partitions,
        test_loader,
        input_size=len(feature_cols),
    )
    print(
        f"[TB-03] Training {algo} for {tb03_config.federated.rounds} rounds "
        f"with {len(client_partitions)} clients."
    )
    metrics = trainer.run(algo)
    print(
        "[TB-03] Completed %s: acc=%.4f recall=%.4f precision=%.4f F1=%.4f worst_rate=%.4f auc=%.4f"
        % (
            algo,
            metrics["accuracy"],
            metrics["recall"],
            metrics["precision"],
            metrics["f1"],
            metrics["worst_rate"],
            metrics["auc"],
        )
    )

    return [
        {"metric_id": "M.SAF.ACC", "value": metrics["accuracy"]},
        {"metric_id": "M.SAF.REC", "value": metrics["recall"]},
        {"metric_id": "M.SAF.FPFN", "value": metrics["worst_rate"]},
        {"metric_id": "M.SAF.FPR_MAX", "value": metrics["fpr"]},
        {"metric_id": "M.SAF.FNR_MAX", "value": metrics["fnr"]},
        {"metric_id": "M.CLS.Pre", "value": metrics["precision"]},
        {"metric_id": "M.CLS.F1_Sc", "value": metrics["f1"]},
        {"metric_id": "M.CLS.AUC", "value": metrics["auc"]},
    ]


class FederatedSequenceTrainer:
    """Minimal synchronous FL trainer for TB-03."""

    def __init__(
        self,
        config: TB03Config,
        clients: Dict[int, ClientPartition],
        test_loader: DataLoader,
        input_size: int,
    ) -> None:
        self.config = config
        self.clients = clients
        self.test_loader = test_loader
        self.input_size = input_size
        self.device = torch.device(config.evaluation.device)
        self.criterion = nn.BCEWithLogitsLoss()

    def run(self, algo: str) -> Dict[str, float]:
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
            control_updates: Dict[str, torch.Tensor] | None = None

            if algo_key == "scaffold" and c_global is not None:
                control_updates = {
                    name: torch.zeros_like(control, device=self.device)
                    for name, control in c_global.items()
                }

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

                old_client_control = c_clients.get(client_id)
                update_state, sample_count, delta_c = self._train_client(
                    local_model,
                    loader,
                    algo_key,
                    global_params,
                    c_global,
                    old_client_control,
                )
                updates.append((update_state, sample_count))
                if delta_c is not None and algo_key == "scaffold" and control_updates is not None:
                    # Update client control variate: new = old + delta
                    # old_client_control is guaranteed to exist (initialized at start)
                    if old_client_control is not None:
                        c_clients[client_id] = {
                            name: (old_client_control[name].cpu() + delta_c[name]).to(self.device)
                            for name in delta_c.keys()
                        }
                    else:
                        # First time this client is selected: initialize from delta_c
                        c_clients[client_id] = {
                            name: delta_c[name].to(self.device)
                            for name in delta_c.keys()
                        }
                    # Accumulate control deltas for global update
                    for name in delta_c.keys():
                        control_updates[name] += delta_c[name].to(self.device)

            if not updates:
                continue

            global_state = _aggregate_weighted(updates)
            if algo_key == "scaffold" and control_updates is not None and c_global is not None and len(selected) > 0:
                # Use scaffold_eta to control the update strength of global control variate
                # Standard SCAFFOLD: c = c + (eta/N) * sum(delta_c_i)
                scaling = self.config.federated.scaffold_eta / len(selected)
                for name in c_global.keys():
                    c_global[name] = c_global[name] + control_updates[name] * scaling

        global_model.load_state_dict(global_state)
        global_model.to(self.device)
        metrics = self._evaluate(global_model)
        return metrics

    def _new_model(self) -> SequenceClassifier:
        return SequenceClassifier(self.input_size, self.config.model)

    def _train_client(
        self,
        model: SequenceClassifier,
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

                logits = model(batch_x)
                loss = self.criterion(logits, batch_y)
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
            # correction = (1/(K*η)) * (w_old - w_new) = (initial_param - final_param) / (total_steps * lr)
            # Note: Ensure total_steps * lr is not zero to avoid division by zero
            for name, param in model.named_parameters():
                initial_param = initial_params[name]
                final_param = param.detach()
                if total_steps > 0 and lr > 0:
                    correction = (initial_param - final_param) / (total_steps * lr)
                    new_control = c_client[name] - c_global[name] + correction
                    delta_c[name] = (new_control - c_client[name]).cpu()
                else:
                    # Fallback: use zero delta if calculation is invalid
                    delta_c[name] = torch.zeros_like(c_client[name]).cpu()

        return state_dict, total_samples, delta_c

    def _evaluate(self, model: SequenceClassifier) -> Dict[str, float]:
        model.eval()
        sigmoid = nn.Sigmoid()
        all_scores: List[float] = []
        all_targets: List[int] = []

        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x = batch_x.to(self.device)
                logits = model(batch_x)
                scores = sigmoid(logits).cpu().numpy()
                all_scores.extend(scores.tolist())
                all_targets.extend(batch_y.numpy().tolist())

        scores_arr = np.array(all_scores)
        targets_arr = np.array(all_targets)
        threshold = self.config.evaluation.threshold
        preds = (scores_arr >= threshold).astype(int)

        tp = float(((preds == 1) & (targets_arr == 1)).sum())
        tn = float(((preds == 0) & (targets_arr == 0)).sum())
        fp = float(((preds == 1) & (targets_arr == 0)).sum())
        fn = float(((preds == 0) & (targets_arr == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        fnr = fn / (fn + tp) if (fn + tp) else 0.0
        worst_rate = max(fpr, fnr)
        auc = _roc_auc(scores_arr, targets_arr)

        return {
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "f1": f1,
            "worst_rate": worst_rate,
            "fpr": fpr,
            "fnr": fnr,
            "auc": auc,
        }


def _feature_columns(df) -> List[str]:
    return [col for col in df.columns if col not in {"engine_id", "cycle", "RUL", "is_event"}]


def _apply_feature_shift(df, feature_cols: Sequence[str], std: float) -> None:
    if std <= 0.0:
        return
    noise = np.random.normal(loc=0.0, scale=std, size=(len(df), len(feature_cols)))
    df.loc[:, feature_cols] = df[feature_cols] + noise


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
        if len(engine_df) <= window:
            continue
        values = engine_df[feature_cols].to_numpy(dtype=np.float32)
        targets = engine_df["is_event"].to_numpy(dtype=np.float32)
        for end_idx in range(window, len(engine_df), stride):
            start_idx = end_idx - window
            seq = values[start_idx:end_idx]
            label = targets[end_idx - 1]
            sequences.append(seq)
            labels.append(label)

    if not sequences:
        return np.zeros((0, window, len(feature_cols)), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    seq_array = np.stack(sequences)
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


def _roc_auc(scores: np.ndarray, targets: np.ndarray) -> float:
    """Fallback ROC-AUC implementation without sklearn."""

    order = np.argsort(scores)
    sorted_targets = targets[order]
    cum_pos = np.cumsum(sorted_targets[::-1])[::-1]
    cum_neg = np.cumsum(1 - sorted_targets[::-1])[::-1]
    total_pos = sorted_targets.sum()
    total_neg = len(sorted_targets) - total_pos
    if total_pos == 0 or total_neg == 0:
        return 0.5

    tpr = cum_pos / total_pos
    fpr = cum_neg / total_neg
    auc = np.trapz(tpr, fpr)
    return float(abs(auc))


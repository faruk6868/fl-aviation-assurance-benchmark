"""Training utilities for centralized and federated experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.federated import (
    ClientConfig,
    FederatedAlgorithm,
    FederatedConfig,
    FedAvgRoundResult,
    FedAvgTrainer,
    FedProxTrainer,
    ScaffoldTrainer,
)
from src.utils import adam_optimizer_factory, sgd_optimizer_factory, set_global_seed


@dataclass
class CentralizedConfig:
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42


def _create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_centralized(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    config: CentralizedConfig,
    loss_fn: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    progress_prefix: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[nn.Module, Dict[str, float]]:
    """Train a model centrally and return best-performing weights."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(config.seed)

    model = model.to(device)
    loss_fn = loss_fn or nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    train_loader = _create_dataloader(train_dataset, config.batch_size, shuffle=True)
    val_loader = _create_dataloader(val_dataset, config.batch_size, shuffle=False) if val_dataset else None

    best_state = None
    best_val_loss = float("inf")
    history = []

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        total = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(features).view(-1)
            targets = targets.view(-1)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            epoch_loss += loss.item() * batch_size
            total += batch_size

        train_loss = epoch_loss / max(total, 1)

        val_loss = float("nan")
        if val_loader:
            val_loss = _evaluate_loss(model, val_loader, loss_fn, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose and progress_prefix:
            print(
                f"[{progress_prefix}] Epoch {epoch + 1}/{config.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}",
                flush=True,
            )

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "train_loss": history[-1]["train_loss"] if history else float("nan"),
        "val_loss": best_val_loss,
        "training_history": history,
    }
    return model, metrics


def _evaluate_loss(model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(features).view(-1)
            targets = targets.view(-1)
            loss = loss_fn(outputs, targets)
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    return total_loss / max(total_samples, 1)


def evaluate_model(model: nn.Module, dataset: Dataset, device: Optional[torch.device] = None) -> Dict[str, np.ndarray]:
    model.eval()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = _create_dataloader(dataset, batch_size=256, shuffle=False)
    predictions = []
    targets = []
    with torch.no_grad():
        for features, target in loader:
            features = features.to(device)
            outputs = model(features).view(-1).cpu().numpy()
            predictions.append(outputs)
            targets.append(target.view(-1).numpy())
    return {
        "predictions": np.concatenate(predictions),
        "targets": np.concatenate(targets),
    }


def train_federated(
    model_factory: Callable[[], nn.Module],
    client_datasets: Mapping[int, Dataset],
    validation_dataset: Optional[Dataset],
    config: FederatedConfig,
    device: Optional[torch.device] = None,
    loss_factory: Callable[[], nn.Module] | None = None,
    optimizer_factory: Callable[[Iterable[torch.nn.Parameter], float, float], torch.optim.Optimizer] | None = None,
    client_postprocess: Callable[[int, Mapping[str, torch.Tensor]], Mapping[str, torch.Tensor]] | None = None,
    progress_prefix: Optional[str] = None,
    verbose: bool = False,
    algorithm: FederatedAlgorithm | str = FederatedAlgorithm.FEDAVG,
    algorithm_kwargs: Optional[Dict[str, object]] = None,
) -> Tuple[nn.Module, Dict[str, object]]:
    """Train a model using FedAvg and return final model and telemetry."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_factory = loss_factory or (lambda: nn.MSELoss())
    algorithm_kwargs = algorithm_kwargs or {}

    algorithm_enum = FederatedAlgorithm.from_value(algorithm)
    if optimizer_factory is None:
        optimizer_factory = (
            sgd_optimizer_factory if algorithm_enum == FederatedAlgorithm.SCAFFOLD else adam_optimizer_factory
        )
    trainer_cls_map = {
        FederatedAlgorithm.FEDAVG: FedAvgTrainer,
        FederatedAlgorithm.FEDPROX: FedProxTrainer,
        FederatedAlgorithm.SCAFFOLD: ScaffoldTrainer,
    }
    trainer_cls = trainer_cls_map[algorithm_enum]

    trainer = trainer_cls(
        model_factory=model_factory,
        loss_factory=loss_factory,
        optimizer_factory=optimizer_factory,
        config=config,
        client_datasets=client_datasets,
        device=device,
        client_postprocess=client_postprocess,
        **algorithm_kwargs,
    )

    round_history = []

    def evaluation_hook(round_idx: int, model: nn.Module) -> None:
        entry = {"round": round_idx}
        if validation_dataset is not None:
            eval_loader = _create_dataloader(validation_dataset, batch_size=config.client_config.batch_size, shuffle=False)
            loss_fn = loss_factory().to(device)
            loss = _evaluate_loss(model, eval_loader, loss_fn, device)
            entry["val_loss"] = loss
        round_history.append(entry)

    progress_interval = max(1, config.evaluation_interval)

    def progress_update(result: FedAvgRoundResult) -> None:
        if not verbose or progress_prefix is None:
            return
        if result.round_idx in {1, config.num_rounds} or result.round_idx % progress_interval == 0:
            print(
                f"[{progress_prefix}::{trainer.algorithm.upper()}] Round {result.round_idx}/{config.num_rounds} "
                f"loss={result.aggregated_loss:.4f} duration={result.duration_s:.2f}s",
                flush=True,
            )

    round_results = trainer.train(evaluation_fn=evaluation_hook, progress_fn=progress_update)

    telemetry = {
        "round_results": [result.__dict__ for result in round_results],
        "evaluation_history": round_history,
        "final_state": {k: v.detach().cpu() for k, v in trainer.state_dict().items()},
    }

    federated_model = trainer.global_model
    return federated_model, telemetry


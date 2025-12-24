"""Federated Averaging trainer implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import ClientConfig, FederatedAlgorithm, FederatedConfig


StateDict = MutableMapping[str, torch.Tensor]
ModelFactory = Callable[[], nn.Module]
LossFactory = Callable[[], nn.Module]
OptimizerFactory = Callable[[Iterable[torch.nn.Parameter], float, float], torch.optim.Optimizer]


@dataclass
class FedAvgRoundResult:
    round_idx: int
    selected_clients: List[int]
    client_losses: Dict[int, float]
    total_samples: int
    aggregated_loss: float
    client_update_norms: Dict[int, float]
    client_update_cosines: Dict[int, float]
    duration_s: float
    algorithm: str = "fedavg"
    extras: Optional[Dict[str, object]] = None


class FedAvgTrainer:
    """Orchestrates the FedAvg training process."""

    def __init__(
        self,
        model_factory: ModelFactory,
        loss_factory: LossFactory,
        optimizer_factory: OptimizerFactory,
        config: FederatedConfig,
        client_datasets: Mapping[int, Dataset],
        device: Optional[torch.device] = None,
        aggregation_fn: Optional[Callable[[Sequence[Tuple[StateDict, int]]], StateDict]] = None,
        client_postprocess: Optional[Callable[[int, StateDict], StateDict]] = None,
        algorithm: FederatedAlgorithm | str = FederatedAlgorithm.FEDAVG,
    ) -> None:
        self.model_factory = model_factory
        self.loss_factory = loss_factory
        self.optimizer_factory = optimizer_factory
        self.config = config
        self.client_datasets = client_datasets
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.aggregation_fn = aggregation_fn or weighted_average_state_dicts
        self.client_postprocess = client_postprocess
        self.algorithm = FederatedAlgorithm.from_value(algorithm).value

        self.global_model = self.model_factory().to(self.device)

        self.client_loaders: Dict[int, DataLoader] = {
            client_id: DataLoader(
                dataset,
                batch_size=config.client_config.batch_size,
                shuffle=True,
                drop_last=False,
            )
            for client_id, dataset in client_datasets.items()
        }

        self.sample_counts = {
            client_id: len(dataset) for client_id, dataset in client_datasets.items()
        }

        if not 0 < config.client_fraction <= 1:
            raise ValueError("client_fraction must be in (0, 1]")

    def state_dict(self) -> StateDict:
        return self.global_model.state_dict()

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        self.global_model.load_state_dict(state_dict)

    def train(
        self,
        evaluation_fn: Optional[Callable[[int, nn.Module], None]] = None,
        progress_fn: Optional[Callable[[FedAvgRoundResult], None]] = None,
    ) -> List[FedAvgRoundResult]:
        """Execute federated training for ``config.num_rounds`` rounds."""

        results: List[FedAvgRoundResult] = []
        rng = np.random.default_rng(self.config.seed)

        for round_idx in range(1, self.config.num_rounds + 1):
            selected_clients = self._sample_clients(rng)
            round_result = self._run_round(round_idx, selected_clients)
            results.append(round_result)

            if progress_fn is not None:
                progress_fn(round_result)

            if evaluation_fn and (round_idx % self.config.evaluation_interval == 0):
                evaluation_fn(round_idx, self.global_model)

        return results

    def _sample_clients(self, rng: np.random.Generator) -> List[int]:
        num_clients = len(self.client_datasets)
        num_selected = max(1, int(np.ceil(self.config.client_fraction * num_clients)))
        all_clients = list(self.client_datasets.keys())
        if num_selected == num_clients:
            return all_clients
        return sorted(rng.choice(all_clients, size=num_selected, replace=False).tolist())

    def _run_round(self, round_idx: int, selected_clients: List[int]) -> FedAvgRoundResult:
        client_states: List[Tuple[StateDict, int]] = []
        client_losses: Dict[int, float] = {}
        client_update_norms: Dict[int, float] = {}
        client_update_cosines: Dict[int, float] = {}
        total_samples = 0

        start_time = time.perf_counter()
        global_state = {k: v.detach().cpu().clone() for k, v in self.global_model.state_dict().items()}
        global_vector = _state_dict_to_vector(global_state)

        for client_id in selected_clients:
            updated_state, metrics = self._client_update(client_id, global_state)
            update_vector = _state_dict_difference(updated_state, global_state)
            update_norm = update_vector.norm().item()
            cosine = _cosine_similarity(update_vector, global_vector)
            client_states.append((updated_state, metrics["samples"]))
            client_losses[client_id] = metrics["loss"]
            total_samples += metrics["samples"]
            client_update_norms[client_id] = update_norm
            client_update_cosines[client_id] = cosine

        aggregated_state = self.aggregation_fn(client_states)
        self.global_model.load_state_dict(aggregated_state)

        aggregated_loss = (
            sum(client_losses[cid] * self.sample_counts[cid] for cid in client_losses)
            / max(total_samples, 1)
        )

        duration = time.perf_counter() - start_time

        return FedAvgRoundResult(
            round_idx=round_idx,
            selected_clients=selected_clients,
            client_losses=client_losses,
            total_samples=total_samples,
            aggregated_loss=aggregated_loss,
            client_update_norms=client_update_norms,
            client_update_cosines=client_update_cosines,
            duration_s=duration,
            algorithm=self.algorithm,
        )

    def _client_update(self, client_id: int, initial_state: Mapping[str, torch.Tensor]) -> Tuple[StateDict, Dict[str, float]]:
        client_model = self.model_factory().to(self.device)
        client_model.load_state_dict(initial_state)
        criterion = self.loss_factory().to(self.device)

        optimizer = self.optimizer_factory(
            client_model.parameters(),
            self.config.client_config.learning_rate,
            self.config.client_config.weight_decay,
        )

        client_model.train()
        data_loader = self.client_loaders[client_id]
        samples = 0
        running_loss = 0.0

        for _ in range(self.config.client_config.local_epochs):
            for batch_features, batch_targets in data_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()
                outputs = client_model(batch_features).view(-1)
                targets = batch_targets.view(-1)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                batch_size = batch_targets.size(0)
                samples += batch_size
                running_loss += loss.item() * batch_size

        updated_state = {k: v.detach().cpu().clone() for k, v in client_model.state_dict().items()}
        if self.client_postprocess is not None:
            updated_state = self.client_postprocess(client_id, updated_state)
        metrics = {
            "loss": running_loss / max(samples, 1),
            "samples": float(samples),
        }
        return updated_state, metrics


def weighted_average_state_dicts(states: Sequence[Tuple[StateDict, int]]) -> StateDict:
    """Compute sample-size weighted average of model parameters."""

    total_samples = sum(samples for _, samples in states)
    if total_samples == 0:
        raise ValueError("Cannot aggregate zero samples")

    first_state = states[0][0]
    averaged_state: Dict[str, torch.Tensor] = {
        key: torch.zeros_like(param) for key, param in first_state.items()
    }

    for state_dict, num_samples in states:
        weight = num_samples / total_samples
        for key, param in state_dict.items():
            averaged_state[key] += param * weight

    return averaged_state


def _state_dict_difference(state_a: Mapping[str, torch.Tensor], state_b: Mapping[str, torch.Tensor]) -> torch.Tensor:
    diffs = []
    for key in state_a.keys():
        diffs.append((state_a[key] - state_b[key]).flatten())
    return torch.cat(diffs)


def _state_dict_to_vector(state: Mapping[str, torch.Tensor]) -> torch.Tensor:
    vectors = [tensor.flatten() for tensor in state.values()]
    return torch.cat(vectors)


def _cosine_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    if vec_a.numel() == 0 or vec_b.numel() == 0:
        return float("nan")
    return float(F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item())


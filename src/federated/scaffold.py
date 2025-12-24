"""SCAFFOLD trainer leveraging control variates to mitigate client drift."""

from __future__ import annotations

import time
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset

from .config import FederatedAlgorithm, FederatedConfig
from .fedavg import (
    FedAvgRoundResult,
    FedAvgTrainer,
    StateDict,
    _cosine_similarity,
    _state_dict_difference,
    _state_dict_to_vector,
    weighted_average_state_dicts,
)


class ScaffoldTrainer(FedAvgTrainer):
    """Implements the SCAFFOLD algorithm with control variate corrections."""

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        loss_factory: Callable[[], nn.Module],
        optimizer_factory: Callable[[Iterable[torch.nn.Parameter], float, float], torch.optim.Optimizer],
        config: FederatedConfig,
        client_datasets: Mapping[int, Dataset],
        device: Optional[torch.device] = None,
        aggregation_fn: Optional[Callable[[Sequence[Tuple[StateDict, int]]], StateDict]] = None,
        client_postprocess: Optional[Callable[[int, StateDict], StateDict]] = None,
        max_grad_norm: float | None = 5.0,
    ) -> None:
        super().__init__(
            model_factory=model_factory,
            loss_factory=loss_factory,
            optimizer_factory=optimizer_factory,
            config=config,
            client_datasets=client_datasets,
            device=device,
            aggregation_fn=aggregation_fn or weighted_average_state_dicts,
            client_postprocess=client_postprocess,
            algorithm=FederatedAlgorithm.SCAFFOLD,
        )
        self.parameter_names = [name for name, _ in self.global_model.named_parameters()]
        self.global_control = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.global_model.named_parameters()
        }
        self.client_controls: Dict[int, Dict[str, torch.Tensor]] = {
            client_id: {
                name: torch.zeros_like(param, device=self.device)
                for name, param in self.global_model.named_parameters()
            }
            for client_id in self.client_datasets.keys()
        }
        self.max_grad_norm = max_grad_norm

    def _run_round(self, round_idx: int, selected_clients: Sequence[int]) -> FedAvgRoundResult:
        client_states: list[Tuple[StateDict, int]] = []
        client_losses: Dict[int, float] = {}
        client_update_norms: Dict[int, float] = {}
        client_update_cosines: Dict[int, float] = {}
        client_control_deltas: Dict[int, float] = {}
        total_samples = 0

        global_state = {k: v.detach().cpu().clone() for k, v in self.global_model.state_dict().items()}
        global_vector = _state_dict_to_vector(global_state)
        control_updates = {
            name: torch.zeros_like(control, device=self.device) for name, control in self.global_control.items()
        }
        start_time = time.perf_counter()

        for client_id in selected_clients:
            updated_state, metrics, control_delta = self._client_update_with_control(client_id, global_state)
            update_vector = _state_dict_difference(updated_state, global_state)
            update_norm = update_vector.norm().item()
            cosine = _cosine_similarity(update_vector, global_vector)

            client_states.append((updated_state, metrics["samples"]))
            client_losses[client_id] = metrics["loss"]
            total_samples += metrics["samples"]
            client_update_norms[client_id] = update_norm
            client_update_cosines[client_id] = cosine
            control_delta_vector = torch.cat([tensor.flatten() for tensor in control_delta.values()])
            client_control_deltas[client_id] = float(control_delta_vector.norm().item())

            for name in self.parameter_names:
                control_updates[name] += control_delta[name]

        if client_states:
            aggregated_state = self.aggregation_fn(client_states)
            self.global_model.load_state_dict(aggregated_state)

            scaling = 1.0 / len(selected_clients)
            for name in self.parameter_names:
                self.global_control[name] = self.global_control[name] + control_updates[name] * scaling

        aggregated_loss = (
            sum(client_losses[cid] * self.sample_counts[cid] for cid in client_losses)
            / max(total_samples, 1)
        )

        duration = time.perf_counter() - start_time

        return FedAvgRoundResult(
            round_idx=round_idx,
            selected_clients=list(selected_clients),
            client_losses=client_losses,
            total_samples=total_samples,
            aggregated_loss=aggregated_loss,
            client_update_norms=client_update_norms,
            client_update_cosines=client_update_cosines,
            duration_s=duration,
            algorithm=self.algorithm,
            extras={"client_control_delta_norms": client_control_deltas},
        )

    def _client_update_with_control(
        self,
        client_id: int,
        global_state: Mapping[str, torch.Tensor],
    ) -> Tuple[StateDict, dict[str, float], Dict[str, torch.Tensor]]:
        client_model = self.model_factory().to(self.device)
        client_model.load_state_dict(global_state)
        criterion = self.loss_factory().to(self.device)
        optimizer = self.optimizer_factory(
            client_model.parameters(),
            self.config.client_config.learning_rate,
            self.config.client_config.weight_decay,
        )

        local_control = self.client_controls[client_id]
        samples = 0
        running_loss = 0.0
        total_steps = 0

        initial_params = {
            name: param.detach().clone()
            for name, param in client_model.named_parameters()
        }

        data_loader = self.client_loaders[client_id]
        client_model.train()
        for _ in range(self.config.client_config.local_epochs):
            for batch_features, batch_targets in data_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()
                outputs = client_model(batch_features).view(-1)
                targets = batch_targets.view(-1)
                loss = criterion(outputs, targets)
                loss.backward()

                for name, param in client_model.named_parameters():
                    param.grad = param.grad - self.global_control[name] + local_control[name]

                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(client_model.parameters(), self.max_grad_norm)

                optimizer.step()

                batch_size = batch_targets.size(0)
                samples += batch_size
                running_loss += loss.detach().item() * batch_size
                total_steps += 1

        total_steps = max(total_steps, 1)
        step_size = self.config.client_config.learning_rate

        updated_state = {k: v.detach().cpu().clone() for k, v in client_model.state_dict().items()}
        control_delta: Dict[str, torch.Tensor] = {}
        for name, param in client_model.named_parameters():
            initial_param = initial_params[name]
            correction = (initial_param - param.detach()) / (total_steps * step_size)
            new_control = local_control[name] - self.global_control[name] + correction
            control_delta[name] = new_control - local_control[name]
            local_control[name] = new_control.detach()

        if self.client_postprocess is not None:
            updated_state = self.client_postprocess(client_id, updated_state)

        metrics = {
            "loss": running_loss / max(samples, 1),
            "samples": float(samples),
        }
        return updated_state, metrics, control_delta



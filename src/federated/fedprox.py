"""FedProx trainer built on top of the FedAvg base implementation."""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset

from .config import FederatedAlgorithm, FederatedConfig
from .fedavg import FedAvgTrainer, StateDict, weighted_average_state_dicts


class FedProxTrainer(FedAvgTrainer):
    """Implements the FedProx client objective with a proximal term."""

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
        proximal_mu: float = 0.01,
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
            algorithm=FederatedAlgorithm.FEDPROX,
        )
        if proximal_mu < 0:
            raise ValueError("proximal_mu must be non-negative")
        self.proximal_mu = proximal_mu

    def _client_update(
        self, client_id: int, initial_state: Mapping[str, torch.Tensor]
    ) -> Tuple[StateDict, dict[str, float]]:
        client_model = self.model_factory().to(self.device)
        client_model.load_state_dict(initial_state)
        criterion = self.loss_factory().to(self.device)
        optimizer = self.optimizer_factory(
            client_model.parameters(),
            self.config.client_config.learning_rate,
            self.config.client_config.weight_decay,
        )

        reference_params = {
            name: initial_state[name].to(self.device)
            for name, _ in client_model.named_parameters()
        }

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

                if self.proximal_mu > 0:
                    proximal_term = torch.zeros(1, device=self.device)
                    for name, param in client_model.named_parameters():
                        proximal_term += torch.sum((param - reference_params[name]) ** 2)
                    loss = loss + 0.5 * self.proximal_mu * proximal_term

                loss.backward()
                optimizer.step()

                batch_size = batch_targets.size(0)
                samples += batch_size
                running_loss += loss.detach().item() * batch_size

        updated_state = {k: v.detach().cpu().clone() for k, v in client_model.state_dict().items()}
        if self.client_postprocess is not None:
            updated_state = self.client_postprocess(client_id, updated_state)

        metrics = {
            "loss": running_loss / max(samples, 1),
            "samples": float(samples),
        }
        return updated_state, metrics



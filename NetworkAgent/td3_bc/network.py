import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from typing import Callable


class Network(nn.Module):
    def __init__(
        self,
        shape: list,
        output_activation: Callable,
        learning_rate: float,
        device: T.device,
    ):
        super().__init__()
        # initialize the network
        layers = []
        for i in range(1, len(shape)):
            dim1 = shape[i - 1]
            dim2 = shape[i]
            layers.append(nn.Linear(dim1, dim2))
            if i < len(shape) - 1:
                layers.append(nn.ReLU())
        layers.append(output_activation())
        self.network = nn.Sequential(*layers)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = device
        self.to(self.device)

    def forward(self, state: T.Tensor) -> T.Tensor:
        return self.network(state)

    def predict(
        self, state: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        if not deterministic:
            raise Exception("Non-deterministic Network.predict() not yet supported...")
        flattened_state = state.flatten()
        with T.no_grad():
            tensor_state = T.tensor(flattened_state, device=self.device)
            tensor_action = self.forward(tensor_state)
            return (tensor_action.cpu().detach().numpy(), None)

    def gradient_descent_step(self, loss: T.Tensor, retain_graph: bool = False):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self.optimizer.step()

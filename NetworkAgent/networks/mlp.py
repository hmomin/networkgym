import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim


class MLP(nn.Module):
    def __init__(
        self,
        shape: list[int],
        hidden_activation: nn.Module,
        output_activation: nn.Module,
        learning_rate: float,
        device: T.device,
        discrete_action_space: bool = False,
    ):
        super().__init__()
        # initialize the network
        self.layers: list[nn.Module] = []
        for i in range(1, len(shape)):
            dim1 = shape[i - 1]
            dim2 = shape[i]
            self.layers.append(nn.Linear(dim1, dim2))
            if i < len(shape) - 1:
                self.layers.append(hidden_activation)
        self.layers.append(output_activation)
        self.network = nn.Sequential(*self.layers)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = device
        self.discrete_action_space = discrete_action_space
        self.to(self.device)

    def forward(self, state: T.Tensor) -> T.Tensor:
        return self.network(state)

    def predict(
        self, state: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        flattened_state = state.flatten()
        with T.no_grad():
            tensor_state = T.tensor(flattened_state, device=self.device)
            if self.discrete_action_space:
                logits = self.forward(tensor_state.unsqueeze(0))
                probabilities = F.softmax(logits, dim=1)
                if deterministic:
                    tensor_action = T.argmax(probabilities, dim=1)
                else:
                    tensor_action = T.multinomial(
                        probabilities, num_samples=1, replacement=True
                    )
                return (tensor_action.item(), None)
            else:
                tensor_action = self.forward(tensor_state)
                return (tensor_action.cpu().detach().numpy(), None)

    def gradient_descent_step(self, loss: T.Tensor, retain_graph: bool = False) -> None:
        self.compute_gradients(loss, retain_graph)
        self.update_parameters()

    def compute_gradients(self, loss: T.Tensor, retain_graph: bool = False) -> None:
        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)

    def update_parameters(self) -> None:
        self.optimizer.step()

    def get_parameter_vector(self, gradient: bool = False) -> T.Tensor:
        running_vector = T.tensor([], device=self.device)
        for _, param in self.named_parameters():
            matrix_param = param.grad if gradient else param
            # FIXME HIGH: figure out how to reshape parameter into a vector
            # then, concatenate it with the running parameter vector
            matrix_param.reshape()
            raise
import numpy as np
import os
import pickle
import sys
import torch
import torch.nn as nn
from copy import deepcopy
from time import time
from torch.func import functional_call, vmap, grad
from typing import Callable

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from buffer import CombinedBuffer
from networks.mlp import MLP
from offline_env import OfflineEnv
from tqdm import tqdm


class PessimisticTD3:
    def __init__(
        self,
        env: OfflineEnv,
        learning_rate: float,
        gamma: float,
        tau: float,
        should_load: bool = True,
        save_folder: str = "saved",
    ):
        self.buffer: CombinedBuffer = env.buffer
        self.observation_dim = self.buffer.states.shape[1]
        self.action_dim = self.buffer.actions.shape[1]
        # NOTE: low_action_bound can be -1 depending on the environment
        self.low_action_bound = 0
        self.high_action_bound = +1
        self.gamma = gamma
        self.tau = tau
        # check if the save_folder path exists
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(script_dir, save_folder)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        self.env_name = os.path.join(save_dir, f"{env.algo_name}_PTD3.")
        name = self.env_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device} device...")
        self.training_stats: list[list[float]] = []
        # initialize the actor and critics
        # NOTE: actor activation is sigmoid instead of tanh (from the paper)
        # to satisfy the action bounds requirement
        self.actor = (
            pickle.load(open(name + "Actor", "rb"))
            if should_load and os.path.exists(name + "Actor")
            else MLP(
                [self.observation_dim, 64, 64, self.action_dim],
                nn.ReLU(),
                nn.Sigmoid(),
                learning_rate,
                self.device,
            )
        )
        self.critic1 = (
            pickle.load(open(name + "Critic1", "rb"))
            if should_load and os.path.exists(name + "Critic1")
            else MLP(
                [self.observation_dim + self.action_dim, 64, 64, 1],
                nn.ReLU(),
                nn.Identity(),
                learning_rate,
                self.device,
            )
        )
        self.critic2 = (
            pickle.load(open(name + "Critic2", "rb"))
            if should_load and os.path.exists(name + "Critic2")
            else MLP(
                [self.observation_dim + self.action_dim, 64, 64, 1],
                nn.ReLU(),
                nn.Identity(),
                learning_rate,
                self.device,
            )
        )
        # create target networks
        self.target_actor = (
            pickle.load(open(name + "TargetActor", "rb"))
            if should_load and os.path.exists(name + "TargetActor")
            else deepcopy(self.actor)
        )
        self.target_critic1 = (
            pickle.load(open(name + "TargetCritic1", "rb"))
            if should_load and os.path.exists(name + "TargetCritic1")
            else deepcopy(self.critic1)
        )
        self.target_critic2 = (
            pickle.load(open(name + "TargetCritic2", "rb"))
            if should_load and os.path.exists(name + "TargetCritic2")
            else deepcopy(self.critic2)
        )

    def get_noisy_action(self, state: np.ndarray, sigma: float) -> np.ndarray:
        deterministic_action = self.get_deterministic_action(state)
        noise = np.random.normal(0, sigma, deterministic_action.shape)
        return np.clip(
            deterministic_action + noise, self.low_action_bound, self.high_action_bound
        )

    def get_deterministic_action(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.tensor(state, device=self.device)
        actions: torch.Tensor = self.actor.forward(state_tensor)
        return actions.cpu().detach().numpy()

    def update(
        self,
        mini_batch_size: int,
        training_sigma: float,
        training_clip: float,
        update_policy: bool,
        beta: float = 0.0,
        ridge_lambda: float = 1.0e-3,
    ):
        self.training_stats.append([])
        self.mini_batch_size = mini_batch_size
        # randomly sample a mini-batch from the replay buffer
        mini_batch = self.buffer.get_mini_batch(mini_batch_size)
        # create tensors to start generating computational graph
        states = mini_batch["states"]
        actions = mini_batch["actions"]
        rewards = mini_batch["rewards"]
        next_states = mini_batch["next_states"]
        dones = mini_batch["dones"]
        # compute the targets
        targets = self.compute_targets(
            rewards, next_states, dones, training_sigma, training_clip
        )
        # do a single step on each critic network
        Q1_loss = self.compute_Q_loss(self.critic1, states, actions, targets)
        self.training_stats[-1].append(Q1_loss.item())
        # print(f"\tQ1_loss: {Q1_loss.item()}")
        self.critic1.gradient_descent_step(Q1_loss, True)
        Q2_loss = self.compute_Q_loss(self.critic2, states, actions, targets)
        self.training_stats[-1].append(Q2_loss.item())
        # print(f"\tQ2_loss: {Q2_loss.item()}")
        self.critic2.gradient_descent_step(Q2_loss)
        if update_policy:
            # do a single step on the actor network
            policy_loss = self.compute_policy_loss(states, beta, ridge_lambda)
            self.training_stats[-1].append(policy_loss.item())
            # print(f"\tpi_loss: {policy_loss.item()}")
            self.actor.gradient_descent_step(policy_loss)
            # update target networks
            self.update_target_network(self.target_actor, self.actor)
            self.update_target_network(self.target_critic1, self.critic1)
            self.update_target_network(self.target_critic2, self.critic2)

    def compute_targets(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        training_sigma: float,
        training_clip: float,
    ) -> torch.Tensor:
        target_actions = self.target_actor.forward(next_states.float())
        # create additive noise for target actions
        noise = torch.normal(
            0, training_sigma, target_actions.shape, device=self.device
        )
        clipped_noise = torch.clip(noise, -training_clip, +training_clip)
        target_actions = torch.clip(
            target_actions + clipped_noise,
            self.low_action_bound,
            self.high_action_bound,
        )
        # compute targets
        target_Q1_values = torch.squeeze(
            self.target_critic1.forward(
                torch.hstack([next_states, target_actions]).float()
            )
        )
        target_Q2_values = torch.squeeze(
            self.target_critic2.forward(
                torch.hstack([next_states, target_actions]).float()
            )
        )
        target_Q_values = torch.minimum(target_Q1_values, target_Q2_values)
        return rewards + self.gamma * (1 - dones) * target_Q_values

    def compute_Q_loss(
        self,
        network: MLP,
        states: torch.Tensor,
        actions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # compute the MSE of the Q function with respect to the targets
        Q_values = torch.squeeze(
            network.forward(torch.hstack([states, actions]).float())
        )
        return torch.square(Q_values - targets).mean()

    def compute_policy_loss(
        self, states: torch.Tensor, beta: float, ridge_lambda: float
    ):
        policy_actions = self.actor.forward(states.float())
        Q_values = torch.squeeze(
            self.critic1.forward(torch.hstack([states, policy_actions]).float())
        )
        mean_Q_value = Q_values.mean()

        if beta > 0.0:
            self.batch_gradient_function = self.get_per_sample_gradient_function()
            self.critic_params, self.critic_buffers = self.detach_critic_parameters()
            Sigma_matrix = self.compute_Sigma_matrix(ridge_lambda)
            Gamma = self.compute_uncertainty_estimate(
                beta, Sigma_matrix, states, policy_actions
            )
            policy_loss = -(mean_Q_value - Gamma)
            print(f"mean_Q_value: {mean_Q_value} | mean_Gamma: {Gamma}")
        else:
            policy_loss = -mean_Q_value
        return policy_loss

    def detach_critic_parameters(
        self,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        critic_params = {k: v.detach() for k, v in self.critic1.named_parameters()}
        critic_buffers = {k: v.detach() for k, v in self.critic1.named_buffers()}
        return critic_params, critic_buffers

    def compute_detached_Q_loss(
        self,
        critic_params: dict[str, torch.Tensor],
        critic_buffers: dict[str, torch.Tensor],
        sample: torch.Tensor,
    ) -> torch.Tensor:
        batch = sample.unsqueeze(0)
        model = self.critic1
        Q_value: torch.Tensor = functional_call(
            model, (critic_params, critic_buffers), (batch,)
        )
        loss = -Q_value.squeeze()
        return loss

    def get_per_sample_gradient_function(
        self,
    ) -> Callable[
        [dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        function_transform_compute_grad = grad(self.compute_detached_Q_loss)
        function_transform_compute_sample_grad = vmap(
            function_transform_compute_grad, in_dims=(None, None, 0)
        )
        return function_transform_compute_sample_grad

    def get_gradient_matrix(self, gradients: dict[str, torch.Tensor]) -> torch.Tensor:
        gradient_matrix = torch.tensor([], dtype=torch.float64, device=self.device)
        for _, param in gradients.items():
            vectorized_param_gradient = param.reshape((param.shape[0], -1))
            gradient_matrix = torch.hstack([gradient_matrix, vectorized_param_gradient])
        if not hasattr(self, "num_parameters"):
            self.num_parameters = gradient_matrix.shape[1]
        return gradient_matrix

    def compute_Sigma_matrix(self, ridge_lambda: float) -> torch.Tensor:
        # FIXME HIGH: maybe, this should be a hyperparameter
        large_batch_size = 10_000
        large_batch = self.buffer.get_mini_batch(large_batch_size)
        states = large_batch["states"]
        actions = large_batch["actions"]
        critic_inputs = torch.hstack([states, actions]).float()
        gradients = self.batch_gradient_function(
            self.critic_params, self.critic_buffers, critic_inputs
        )
        gradient_matrix = self.get_gradient_matrix(gradients)
        Sigma = gradient_matrix.T @ gradient_matrix + ridge_lambda * torch.eye(
            self.num_parameters, dtype=torch.float64, device=self.device
        )
        return Sigma

    def compute_uncertainty_estimate(
        self,
        beta: float,
        Sigma: torch.Tensor,
        states: torch.Tensor,
        policy_actions: torch.Tensor,
    ) -> torch.Tensor:
        Sigma_inverse: torch.Tensor = torch.linalg.inv(Sigma)
        critic_inputs = torch.hstack([states, policy_actions]).float()
        gradients = self.batch_gradient_function(
            self.critic_params, self.critic_buffers, critic_inputs
        )
        gradient_matrix = self.get_gradient_matrix(gradients)
        matrix_inside_sqrt = gradient_matrix @ Sigma_inverse @ gradient_matrix.T
        vector_inside_sqrt = torch.diag(matrix_inside_sqrt)
        Gamma = beta * torch.mean(torch.sqrt(vector_inside_sqrt))
        return Gamma

    def update_target_network(self, target_network: MLP, network: MLP):
        with torch.no_grad():
            for target_parameter, parameter in zip(
                target_network.parameters(), network.parameters()
            ):
                target_parameter.mul_(1 - self.tau)
                target_parameter.add_(self.tau * parameter)

    def save(self, step: int = 0, max_steps: int = 1_000_000):
        step_str = str(step).zfill(len(str(max_steps)))
        name = f"{self.env_name}{step_str}.{self.buffer.num_buffers}."
        pickle.dump(self.training_stats, open(name + "training_stats", "wb"))
        pickle.dump(self.actor, open(name + "Actor", "wb"))
        pickle.dump(self.critic1, open(name + "Critic1", "wb"))
        pickle.dump(self.critic2, open(name + "Critic2", "wb"))
        pickle.dump(self.target_actor, open(name + "TargetActor", "wb"))
        pickle.dump(self.target_critic1, open(name + "TargetCritic1", "wb"))
        pickle.dump(self.target_critic2, open(name + "TargetCritic2", "wb"))

import numpy as np
import os
import pickle
import sys
import torch
import torch.nn as nn
from copy import deepcopy
from torch.func import functional_call, vmap, grad
from tqdm import tqdm
from typing import Callable

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from buffer import CombinedBuffer
from networks.mlp import MLP
from offline_env import OfflineEnv


def Sherman_Morrison_inverse(
    A_inverse: torch.Tensor, u: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    denominator = 1 + v.T @ A_inverse @ u
    numerator = A_inverse @ u @ v.T @ A_inverse
    total_inverse = A_inverse - numerator / denominator
    return total_inverse


class PessimisticTD3:
    def __init__(
        self,
        env: OfflineEnv,
        beta_pessimism: float,
        alpha_fisher: float,
        learning_rate: float,
        gamma: float,
        tau: float,
        policy_delay: int,
        should_load: bool = True,
        device: str | None = None,
        save_folder: str = "saved",
    ):
        self.buffer: CombinedBuffer = env.buffer
        self.observation_dim = self.buffer.states.shape[1]
        self.action_dim = self.buffer.actions.shape[1]
        # NOTE: low_action_bound can be -1 depending on the environment
        self.low_action_bound = 0
        self.high_action_bound = +1
        self.beta = beta_pessimism
        if alpha_fisher < 0.0:
            raise Exception(f"alpha_fisher ({alpha_fisher} passed in) must be >= 0.0")
        self.alpha = alpha_fisher
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.current_update_step = 0
        # check if the save_folder path exists
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(script_dir, save_folder)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        self.env_name = os.path.join(
            save_dir, f"{env.algo_name}_PTD3_beta_{self.beta}_alpha_{self.alpha}"
        )
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device} device...")
        self.training_stats: list[list[float]] = []
        self.initialize_actor_critic_params(learning_rate, should_load)
        self.initialize_Fisher_information_matrix()

    def initialize_actor_critic_params(
        self, learning_rate: float, should_load: bool
    ) -> None:
        # NOTE: actor activation is sigmoid instead of tanh (from the paper)
        # to satisfy the action bounds requirement
        self.actor = (
            pickle.load(open(self.env_name + ".Actor", "rb"))
            if should_load and os.path.exists(self.env_name + ".Actor")
            else MLP(
                [self.observation_dim, 400, 300, self.action_dim],
                nn.ReLU(),
                nn.Sigmoid(),
                learning_rate,
                self.device,
            )
        )
        self.critic1 = (
            pickle.load(open(self.env_name + ".Critic1", "rb"))
            if should_load and os.path.exists(self.env_name + ".Critic1")
            else MLP(
                [self.observation_dim + self.action_dim, 64, 64, 1],
                nn.ReLU(),
                nn.Identity(),
                learning_rate,
                self.device,
            )
        )
        self.critic2 = (
            pickle.load(open(self.env_name + ".Critic2", "rb"))
            if should_load and os.path.exists(self.env_name + ".Critic2")
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
            pickle.load(open(self.env_name + ".TargetActor", "rb"))
            if should_load and os.path.exists(self.env_name + ".TargetActor")
            else deepcopy(self.actor)
        )
        self.target_critic1 = (
            pickle.load(open(self.env_name + ".TargetCritic1", "rb"))
            if should_load and os.path.exists(self.env_name + ".TargetCritic1")
            else deepcopy(self.critic1)
        )
        self.target_critic2 = (
            pickle.load(open(self.env_name + ".TargetCritic2", "rb"))
            if should_load and os.path.exists(self.env_name + ".TargetCritic2")
            else deepcopy(self.critic2)
        )

    def initialize_Fisher_information_matrix(self) -> None:
        self.d = self.critic1.get_num_parameters()
        print(f"Each critic network has {self.d} parameters...")
        if self.alpha > 0.0:
            self.Sigma = torch.eye(self.d, dtype=torch.float64, device=self.device)
            self.Sigma_inverse = torch.eye(
                self.d, dtype=torch.float64, device=self.device
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
        if self.current_update_step % self.policy_delay == 0:
            # do a single step on the actor network
            policy_loss = self.compute_policy_loss(states)
            self.training_stats[-1].append(policy_loss.item())
            # print(f"\tpi_loss: {policy_loss.item()}")
            self.actor.gradient_descent_step(policy_loss)
            # update target networks
            self.update_target_network(self.target_actor, self.actor)
            self.update_target_network(self.target_critic1, self.critic1)
            self.update_target_network(self.target_critic2, self.critic2)
        self.current_update_step += 1

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

    def compute_policy_loss(self, states: torch.Tensor):
        policy_actions = self.actor.forward(states.float())
        Q_values = torch.squeeze(
            self.critic1.forward(torch.hstack([states, policy_actions]).float())
        )
        mean_Q_value = Q_values.mean()

        if self.beta > 0.0:
            self.batch_gradient_function = self.get_per_sample_gradient_function()
            self.critic_params, self.critic_buffers = self.detach_critic_parameters()
            self.update_Fisher_information_matrix()
            Gamma = self.compute_uncertainty_estimate(states, policy_actions)
            # FIXME HIGH: change this back! testing BC
            # policy_loss = -(mean_Q_value - Gamma)
            policy_loss = Gamma
            print(
                f"iteration: {self.current_update_step:07d} | mean_Q_value: {mean_Q_value} | mean_Gamma: {Gamma}"
            )
        else:
            policy_loss = -mean_Q_value
        if torch.isnan(policy_loss).any():
            raise Exception("NaNs detected! Crashing...")
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
        return gradient_matrix

    def update_Fisher_information_matrix(self) -> None:
        # NOTE: if alpha is 0, then calculate Sigma using a large-batch estimate.
        # if alpha > 0.0, then estimate Sigma and Sigma_inverse with incremental
        # "SGD"-like updates to the Sigma matrix
        # FIXME LOW: maybe, the large-batch size should be a hyperparameter?
        sample_size = 1 if self.alpha > 0.0 else 2 ** 14
        buffer_batch = self.buffer.get_mini_batch(sample_size)
        states = buffer_batch["states"]
        actions = buffer_batch["actions"]
        critic_input = torch.hstack([states, actions]).float()
        gradient_dict = self.batch_gradient_function(
            self.critic_params, self.critic_buffers, critic_input
        )
        with torch.no_grad():
            gradients_transpose = self.get_gradient_matrix(gradient_dict)
            gradients = gradients_transpose.T
            if self.alpha > 0.0:
                # NOTE: have to add a little noise to maintain full rank
                noisy_gradient = gradients + (1.0e-9) * torch.randn_like(
                    gradients, dtype=torch.float64, device=self.device
                )
                new_Sigma = self.alpha * self.Sigma + noisy_gradient @ noisy_gradient.T
                new_Sigma_inverse = Sherman_Morrison_inverse(
                    self.Sigma_inverse / self.alpha, noisy_gradient, noisy_gradient
                )
                self.Sigma = new_Sigma
                self.Sigma_inverse = new_Sigma_inverse
                if self.current_update_step > 0 and self.current_update_step % 100 == 0:
                    self.ground_inverse_computation()
            else:
                # FIXME LOW: ridge regression lambda could be a hyperparameter?
                self.Sigma = gradients @ gradients_transpose + (1.0e-6) * torch.eye(
                    self.d, dtype=torch.float64, device=self.device
                )
                self.Sigma_inverse = torch.linalg.inv(self.Sigma)

    def compute_uncertainty_estimate(
        self,
        states: torch.Tensor,
        policy_actions: torch.Tensor,
    ) -> torch.Tensor:
        critic_inputs = torch.hstack([states, policy_actions]).float()
        gradients = self.batch_gradient_function(
            self.critic_params, self.critic_buffers, critic_inputs
        )
        gradients_transpose = self.get_gradient_matrix(gradients)
        gradients = gradients_transpose.T
        inside_sqrt = torch.sum(gradients * (self.Sigma_inverse @ gradients), dim=0)
        Gamma = self.beta * torch.mean(torch.sqrt(inside_sqrt))
        return Gamma

    def ground_inverse_computation(self) -> None:
        true_Sigma_inverse = torch.linalg.inv(self.Sigma)
        inverse_deviation = true_Sigma_inverse - self.Sigma_inverse
        frobenius_norm = torch.linalg.matrix_norm(inverse_deviation)
        print(f"Frobenius norm of Sigma_inverse deviation: {frobenius_norm}")
        self.Sigma_inverse = true_Sigma_inverse

    def update_target_network(self, target_network: MLP, network: MLP):
        with torch.no_grad():
            for target_parameter, parameter in zip(
                target_network.parameters(), network.parameters()
            ):
                target_parameter.mul_(1 - self.tau)
                target_parameter.add_(self.tau * parameter)

    def save(self, step: int = 0, max_steps: int = 1_000_000):
        step_str = str(step).zfill(len(str(max_steps)))
        name = f"{self.env_name}_step_{step_str}"
        pickle.dump(self.training_stats, open(name + ".training_stats", "wb"))
        pickle.dump(self.actor, open(name + ".Actor", "wb"))
        pickle.dump(self.critic1, open(name + ".Critic1", "wb"))
        pickle.dump(self.critic2, open(name + ".Critic2", "wb"))
        pickle.dump(self.target_actor, open(name + ".TargetActor", "wb"))
        pickle.dump(self.target_critic1, open(name + ".TargetCritic1", "wb"))
        pickle.dump(self.target_critic2, open(name + ".TargetCritic2", "wb"))

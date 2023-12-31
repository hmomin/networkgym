import numpy as np
import os
import pickle
import sys
import torch as T
import torch.nn as nn
import torch.nn.functional as F

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from buffer import CombinedBuffer
from discrete_action_util import convert_continuous_to_discrete_ratio_action
from networks.mlp import MLP
from offline_env import OfflineEnv


class LSPI_BC:
    def __init__(
        self,
        env: OfflineEnv,
        num_actions: int,
        learning_rate: float,
        should_load: bool = True,
        save_folder: str = "saved",
    ):
        self.gamma = 0.99
        self.buffer: CombinedBuffer = env.buffer
        self.observation_dim = self.buffer.states.shape[1]
        self.num_actions = num_actions
        # check if the save_folder path exists
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(script_dir, save_folder)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        self.env_name = os.path.join(save_dir, f"{env.algo_name}.")
        name = self.env_name
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        print(f"Using {self.device} device...")
        # initialize the actor
        self.actor = (
            pickle.load(open(name + "Actor", "rb"))
            if should_load and os.path.exists(name + "Actor")
            else MLP(
                [self.observation_dim, 256, 256, self.num_actions],
                nn.ReLU,
                nn.Identity,
                learning_rate,
                self.device,
                discrete_action_space=True,
            )
        )
        # FIXME HIGH: perhaps, this should be a specifiable hyperparameter...
        self.large_batch_size: int = 500
        self.initialize_action_feature_map()
        self.initialize_Q_weights()

    def initialize_action_feature_map(self) -> None:
        action_indices = list(range(self.num_actions))
        self.action_one_hots: T.Tensor = F.one_hot(
            T.tensor(action_indices, device="cuda:0"),
            num_classes=self.num_actions,
        ).unsqueeze(0)

    def initialize_Q_weights(self):
        phi_s_layer = self.actor.layers[-4]
        phi_s_shape: int = phi_s_layer.bias.shape[0]
        self.k = phi_s_shape + self.num_actions
        self.w_tilde = T.randn((self.k, 1), device="cuda:0")

    def update(self, mini_batch_size: int, alpha_bc: float = 0.0):
        small_batch = self.buffer.get_mini_batch(mini_batch_size)
        states = small_batch["states"]
        actions = small_batch["actions"]
        kl_div_loss = self.compute_kl_divergence_loss(states, actions)
        # print(f"\tKL_div_loss: {kl_div_loss.item()}")
        large_batch = self.buffer.get_mini_batch(self.large_batch_size)
        states = large_batch["states"]
        actions = large_batch["actions"]
        rewards = large_batch["rewards"]
        next_states = large_batch["next_states"]
        Q_loss = self.compute_LSPI_Q_loss(
            alpha_bc, states, actions, rewards, next_states
        )
        total_loss = Q_loss + kl_div_loss
        self.actor.gradient_descent_step(total_loss)

    def state_featurizer_forward(self, x: T.Tensor) -> T.Tensor:
        num_actor_layers = len(self.actor.layers)
        # NOTE: assuming the last three layers are not used to generate state features
        for idx in range(num_actor_layers - 3):
            module = self.actor.layers[idx]
            x = module(x)
        return x

    def compute_kl_divergence_loss(self, states: T.Tensor, actions: T.Tensor):
        actor_logits = self.actor.forward(states.float())
        actor_probs = F.softmax(actor_logits, dim=1)
        log_probs = T.log(actor_probs)
        behavior_actions = convert_continuous_to_discrete_ratio_action(actions)
        behavior_one_hots = F.one_hot(behavior_actions, num_classes=5**4)
        neg_log_p = T.sum(behavior_one_hots * -log_probs, dim=1)
        policy_loss = T.mean(neg_log_p)
        return policy_loss

    def compute_LSPI_Q_loss(
        self,
        alpha_bc: float,
        states: T.Tensor,
        actions: T.Tensor,
        rewards: T.Tensor,
        next_states: T.Tensor,
    ) -> T.Tensor:
        actual_actions = convert_continuous_to_discrete_ratio_action(actions)
        action_one_hots = F.one_hot(actual_actions, num_classes=5**4)
        rewards = rewards.unsqueeze(1)
        w_tilde = self.compute_LSPI_weights(
            states, action_one_hots, rewards, next_states
        )
        raise Exception("FIXME HIGH: running into CUDA overflow errors. Might work on desktop instead of laptop...")

    def compute_LSPI_weights(
        self,
        states: T.Tensor,
        actions: T.Tensor,
        rewards: T.Tensor,
        next_states: T.Tensor,
    ) -> None:
        norm_difference = float("inf")
        iteration = 0
        while norm_difference >= 1.0e-6 and iteration < 6:
            phi_tilde = self.construct_phi_matrix(states, actions)
            phi_prime_tilde = self.construct_phi_prime_matrix(next_states)
            A_tilde = phi_tilde.T @ (phi_tilde - self.gamma * phi_prime_tilde)
            b_tilde = phi_tilde.T @ rewards.to(T.float32)
            w_tilde = T.linalg.lstsq(A_tilde, b_tilde).solution
            # NOTE: if the matrix is rank-deficient, w_tilde will be all NaNs
            # in this case, just refuse to do an update
            if T.isnan(w_tilde).any():
                raise Exception("A_tilde of low-rank in LSTDQ computation...")
            else:
                norm_difference = T.norm(w_tilde - self.w_tilde, float("inf"))
                # print(f"||w - w'||_inf: {norm_difference}")
                self.w_tilde = w_tilde
            iteration += 1

    def construct_phi_matrix(
        self, batch_states: T.Tensor, batch_actions: T.Tensor
    ) -> T.Tensor:
        with T.no_grad():
            phi_s = self.state_featurizer_forward(batch_states)
            phi_matrix = T.cat([phi_s, batch_actions], dim=1)
            return phi_matrix

    def construct_phi_prime_matrix(self, batch_next_states: T.Tensor) -> T.Tensor:
        with T.no_grad():
            phi_s_prime = self.state_featurizer_forward(batch_next_states)
            pi_s_prime = self.Q_policy(batch_next_states, one_hot=True)
            phi_prime_matrix = T.cat([phi_s_prime, pi_s_prime], dim=1)
            return phi_prime_matrix

    def Q_policy(self, observations: T.Tensor, one_hot: bool = True) -> T.Tensor:
        with T.no_grad():
            batch_size = observations.shape[0]
            phi_s = self.state_featurizer_forward(observations)
            phi_s_repeated = phi_s.unsqueeze(1).repeat(1, self.num_actions, 1)
            action_features_repeated = self.action_one_hots.repeat(batch_size, 1, 1)
            phi_matrix = T.cat([phi_s_repeated, action_features_repeated], dim=2)
            batch_w_tilde = self.w_tilde.unsqueeze(0).repeat(batch_size, 1, 1)
            Q_values = T.bmm(phi_matrix, batch_w_tilde)
            optimal_actions = T.argmax(Q_values, dim=1)
            if not one_hot:
                return optimal_actions
            argmax_indices = T.squeeze(optimal_actions, 1)
            optimal_action_one_hots = T.index_select(
                self.action_one_hots, dim=1, index=argmax_indices
            ).squeeze(0)
            return optimal_action_one_hots

    def save(self, step: int = 0, max_steps: int = 1_000_000):
        step_str = str(step).zfill(len(str(max_steps)))
        name = f"{self.env_name}{step_str}.{self.buffer.num_buffers}."
        pickle.dump(self.actor, open(name + "Actor", "wb"))
import numpy as np
import os
import pickle
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from time import sleep
from typing import Callable

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from networks.mlp import MLP


class FastLSPI:
    def __init__(
        self,
        observation_dim: int,
        num_actions: int,
        hidden_dims: list[int] = [256, 256],
        activation_function: Callable[
            [], Callable[[torch.Tensor], torch.Tensor]
        ] = nn.ELU,
        epsilon: float = 1.0e-3,
        learning_rate: float = 3.0e-4,
        gamma: float = 0.99,
        should_load: bool = True,
        save_folder: str = "saved",
    ):
        self.gamma = 0.99
        self.epsilon = epsilon
        self.observation_dim = observation_dim
        self.hidden_dims = hidden_dims
        self.activation_function = activation_function
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device} device...")
        # check if the save_folder path exists
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(script_dir, save_folder)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        self.actor_name = os.path.join(save_dir, "FastLSPI.actor")
        self.initialize_action_feature_map()
        self.initialize_parameters(should_load)
        self.initialize_Q_weights()

    def initialize_action_feature_map(self) -> None:
        action_indices = list(range(self.num_actions))
        self.action_one_hots: torch.Tensor = F.one_hot(
            torch.tensor(action_indices, device="cuda:0"),
            num_classes=self.num_actions,
        ).unsqueeze(0)

    def initialize_parameters(self, should_load: bool = True) -> None:
        self.actor = (
            pickle.load(open(self.actor_name, "rb"))
            if should_load and os.path.exists(self.actor_name)
            else MLP(
                [self.observation_dim, *self.hidden_dims, self.num_actions],
                self.activation_function,
                nn.Identity,
                self.learning_rate,
                self.device,
            )
        )

    def initialize_Q_weights(self) -> None:
        self.k = 2 * self.num_actions
        self.w_tilde = torch.randn((self.k, 1), device="cuda:0")
        self.full_rank_reached = False

    def store_to_buffer(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray
    ) -> None:
        tensor_state = torch.tensor(
            state, dtype=torch.float32, device="cuda:0"
        ).unsqueeze(1)
        tensor_action: torch.Tensor = F.one_hot(
            torch.tensor(action, device="cuda:0"),
            num_classes=self.num_actions,
        ).unsqueeze(1)
        tensor_reward = torch.tensor(
            [reward], dtype=torch.float32, device="cuda:0"
        ).unsqueeze(1)
        tensor_next_state = torch.tensor(
            next_state, dtype=torch.float32, device="cuda:0"
        ).unsqueeze(1)
        if hasattr(self, "states"):
            self.states = torch.cat([self.states, tensor_state], dim=1)
            self.actions = torch.cat([self.actions, tensor_action], dim=1)
            self.rewards = torch.cat([self.rewards, tensor_reward], dim=1)
            self.next_states = torch.cat([self.next_states, tensor_next_state], dim=1)
        else:
            # create new tensors for (s,a,r,s') tuples
            self.states = tensor_state
            self.actions = tensor_action
            self.rewards = tensor_reward
            self.next_states = tensor_next_state
        self.L = self.rewards.shape[1]
        print("----- BUFFER SIZES -----")
        print(self.states.shape)
        print(self.actions.shape)
        print(self.rewards.shape)
        print(self.next_states.shape)
        print("------------------------")

    def actor_policy(
        self, observations: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        with torch.no_grad():
            logits = self.actor.forward(observations)
            probabilities = F.softmax(logits, dim=1)
            if deterministic:
                actor_actions = torch.argmax(probabilities, dim=1)
            else:
                actor_actions = torch.multinomial(
                    probabilities, num_samples=1, replacement=True
                )
            return actor_actions

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[int, dict]:
        flat_observation = observation.flatten()
        tensor_observation = torch.tensor(
            flat_observation, dtype=torch.float32, device="cuda:0"
        ).unsqueeze(0)
        actor_action_tensor = self.actor_policy(
            tensor_observation, deterministic=deterministic
        )
        actor_action = int(actor_action_tensor.item())
        return actor_action, {}

    def Q_policy(
        self, observations: torch.Tensor, one_hot: bool = True
    ) -> torch.Tensor:
        with torch.no_grad():
            phi_s = self.actor.forward(observations.T)
            # print("phi_s.shape")
            # print(phi_s.shape)
            phi_s_repeated = phi_s.unsqueeze(1).repeat(1, self.num_actions, 1)
            # print("phi_s_repeated.shape")
            # print(phi_s_repeated.shape)
            action_features_repeated = self.action_one_hots.repeat(self.L, 1, 1)
            # print("action_features_repeated.shape")
            # print(action_features_repeated.shape)
            phi_matrix = torch.cat([phi_s_repeated, action_features_repeated], dim=2)
            # print("phi_matrix.shape")
            # print(phi_matrix.shape)
            batch_w_tilde = self.w_tilde.unsqueeze(0).repeat(self.L, 1, 1)
            # print("batch_w_tilde.shape")
            # print(batch_w_tilde.shape)
            Q_values = torch.bmm(phi_matrix, batch_w_tilde)
            # print("Q_values.shape")
            # print(Q_values.shape)
            optimal_actions = torch.argmax(Q_values, dim=1)
            # print("optimal_actions.shape")
            # print(optimal_actions.shape)
            if not one_hot:
                return optimal_actions
            argmax_indices = torch.squeeze(optimal_actions, 1)
            # print("argmax_indices.shape")
            # print(argmax_indices.shape)
            optimal_action_one_hots = torch.index_select(
                self.action_one_hots, dim=1, index=argmax_indices
            ).squeeze(0)
            # print("optimal_action_one_hots.shape")
            # print(optimal_action_one_hots.shape)
            return optimal_action_one_hots

    def construct_phi_matrix(self) -> torch.Tensor:
        with torch.no_grad():
            phi_s = self.actor.forward(self.states.T)
            phi_matrix = torch.cat([phi_s, self.actions.T], dim=1)
            return phi_matrix

    def construct_phi_prime_matrix(self) -> torch.Tensor:
        with torch.no_grad():
            phi_s_prime = self.actor.forward(self.next_states.T)
            # print("phi_s_prime.shape")
            # print(phi_s_prime.shape)
            pi_s_prime = self.Q_policy(self.next_states, one_hot=True)
            # print("pi_s_prime.shape")
            # print(pi_s_prime.shape)
            phi_prime_matrix = torch.cat([phi_s_prime, pi_s_prime], dim=1)
            return phi_prime_matrix

    def update(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray
    ) -> None:
        self.LSTDQ_update(state, action, reward, next_state)
        if self.full_rank_reached:
            self.actor_update()

    # NOTE: incremental update of weight vector for Q-hat
    def LSTDQ_update(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray
    ) -> None:
        state = state.flatten()
        next_state = next_state.flatten()
        self.store_to_buffer(state, action, reward, next_state)
        # NOTE: no point in doing LSTDQ update if the rank is too small
        # FIXME HIGH - sometimes, full rank can be reached pretty quickly?
        if self.L < self.k:
            return
        norm_difference = float("inf")
        iteration = 0
        while norm_difference >= self.epsilon and iteration < 6:
            phi_tilde = self.construct_phi_matrix()
            phi_prime_tilde = self.construct_phi_prime_matrix()
            # print("phi_tilde.shape")
            # print(phi_tilde.shape)
            A_tilde = phi_tilde.T @ (phi_tilde - self.gamma * phi_prime_tilde)
            # print("self.rewards.shape")
            # print(self.rewards.shape)
            b_tilde = phi_tilde.T @ self.rewards.T
            w_tilde = torch.linalg.lstsq(A_tilde, b_tilde).solution
            # NOTE: if the matrix is rank-deficient, w_tilde will be all NaNs
            # in this case, just don't do an update
            if torch.isnan(w_tilde).any():
                self.full_rank_reached = False
                rank_A_tilde = torch.linalg.matrix_rank(A_tilde)
                print(f"A_tilde defective - rank {rank_A_tilde} < {A_tilde.shape[0]}")
                break
            else:
                self.full_rank_reached = True
                norm_difference = torch.norm(w_tilde - self.w_tilde, float("inf"))
                print(f"||w - w'||_inf: {norm_difference}")
                self.w_tilde = w_tilde
            iteration += 1

    def actor_update(self) -> None:
        print("FIXME HIGH: implement actor update!")
        # determine the "optimal actions" using self.w_tilde
        # (with gradient calculation off)
        
        # determine the actions that the actor network would actually recommend
        # (with gradient calculation on)
        
        # do N steps of gradient descent on the cross-entropy loss between the two
        # actions

    def get_Q_value(self, state: np.ndarray, action: int) -> float:
        with torch.no_grad():
            state = state.flatten()
            tensor_state = torch.tensor(
                state, dtype=torch.float32, device="cuda:0"
            ).unsqueeze(0)
            phi_s = self.actor.forward(tensor_state)
            phi_a = self.action_one_hots[:, action, :]
            phi_s_a = torch.cat([phi_s, phi_a], dim=1)
            Q_value = (phi_s_a @ self.w_tilde).item()
            return Q_value


def main() -> None:
    agent_thingy = FastLSPI(
        observation_dim=14 * 4,
        num_actions=3**4,
        hidden_dims=[400, 300],
        activation_function=nn.ELU,
    )
    random_state = np.random.random((14, 4))
    # testing generation of weights for Q-hat
    for iter in range(1000):
        action, _ = agent_thingy.predict(random_state)
        print(f"iteration: {iter}, action: {action}")
        # NOTE: expect the action to eventually converge to 42, independent of state
        if action == 42:
            random_reward = np.random.uniform(0.0, 2.0)
        elif action == 24:
            random_reward = np.random.uniform(0.0, 1.0)
        else:
            random_reward = np.random.uniform(-11.0, -9.0)
        random_next_state = np.random.random((14, 4))
        agent_thingy.LSTDQ_update(
            random_state, action, random_reward, random_next_state
        )
        random_state = random_next_state
    # evaluation
    for action in range(agent_thingy.num_actions):
        Q_value = agent_thingy.get_Q_value(random_state, action)
        print(f"action: {action} -> Q_value: {Q_value}")


if __name__ == "__main__":
    main()

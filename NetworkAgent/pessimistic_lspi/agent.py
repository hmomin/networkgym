import numpy as np
import os
import pickle
import sys
import torch
import torch.nn.functional as F
from time import sleep
from tqdm import tqdm

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from buffer import CombinedBuffer
from offline_env import OfflineEnv


class PessimisticLSPI:
    def __init__(
        self,
        env: OfflineEnv,
        observation_power: int,
        num_actions: int,
        beta: float,
        large_batch_size: int = 2 ** 14,
        save_folder: str = "saved",
    ):
        self.gamma = 0.99
        self.observation_power = observation_power
        self.beta = beta
        self.buffer: CombinedBuffer = env.buffer
        self.observation_dim = self.buffer.states.shape[1]
        self.num_actions = num_actions
        self.large_batch_size = large_batch_size
        # check if the save_folder path exists
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(script_dir, save_folder)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        self.env_name = os.path.join(
            save_dir,
            f"PessimisticLSPI_{env.algo_name}_beta_{self.beta}_obs_power_{self.observation_power}",
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device} device...")
        self.initialize_action_feature_map()
        self.initialize_Q_weights()
        self.compute_Sigma_inverse()

    def initialize_action_feature_map(self) -> None:
        action_indices = list(range(self.num_actions))
        self.action_one_hots: torch.Tensor = F.one_hot(
            torch.tensor(action_indices, device="cuda:0"),
            num_classes=self.num_actions,
        ).unsqueeze(0)

    def initialize_Q_weights(self) -> None:
        # NOTE: removing the last four elements of observation, because the y-dimension
        # of the UE's stays zero for the whole simulation(s)
        # NOTE: also removing first four elements (LTE max rate never changes)
        self.k = (self.observation_dim - 8) * self.observation_power + self.num_actions
        print(f"k = {self.k}")
        self.w_tilde = torch.randn((self.k, 1), dtype=torch.float64, device="cuda:0")

    def compute_Sigma_inverse(self) -> None:
        L = self.buffer.buffer_size
        print("Computing Sigma_inverse...")
        Sigma = torch.zeros((self.k, self.k), dtype=torch.float64, device=self.device)
        for start_index in tqdm(range(0, L, self.large_batch_size)):
            end_index = start_index + self.large_batch_size
            next_batch = self.buffer.get_batch_from_indices(start_index, end_index)
            states = next_batch["states"].to(torch.float64)
            discrete_actions = next_batch["actions"]
            flattened_actions = discrete_actions.flatten().to(torch.int64)
            actions = F.one_hot(flattened_actions, num_classes=self.num_actions)
            phi_tilde = self.construct_phi_matrix(states, actions)
            Sigma += phi_tilde.T @ phi_tilde
        try:
            self.Sigma_inverse = torch.linalg.inv(Sigma)
        except:
            # try ridge regression if Sigma is non-invertible
            Sigma += (1.0e-6) * torch.eye(
                self.k, dtype=torch.float64, device=self.device
            )
            self.Sigma_inverse = torch.linalg.inv(Sigma)

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[int, dict]:
        if deterministic:
            flat_observation = observation.flatten()
            tensor_observation = torch.tensor(
                flat_observation, dtype=torch.float64, device="cuda:0"
            ).unsqueeze(0)
            actor_action_tensor = self.Q_policy(
                tensor_observation, one_hot=False, pessimistic=True
            )
            actor_action = int(actor_action_tensor.item())
        else:
            actor_action = np.random.randint(0, self.num_actions)
        return actor_action, {}

    def LSTDQ_update(self) -> None:
        w_difference_norm = 1.0
        iteration = 0
        while w_difference_norm >= 1.0e-6:
            A_tilde, b_tilde = self.compute_least_squares_system()
            w_tilde = torch.linalg.lstsq(A_tilde, b_tilde).solution
            while torch.isnan(w_tilde).any():
                print(f"WARNING: A_tilde rank-deficient...")
                A_tilde += 1.0e-6 * torch.eye(
                    self.k, dtype=torch.float64, device=self.device
                )
                w_tilde = torch.linalg.lstsq(A_tilde, b_tilde).solution
            w_difference_norm = torch.linalg.vector_norm(
                w_tilde - self.w_tilde, ord=torch.inf
            )
            print(f"||w - w'||_inf = {w_difference_norm}")
            self.w_tilde = w_tilde
            iteration += 1

    def compute_least_squares_system(self) -> tuple[torch.Tensor, torch.Tensor]:
        A_tilde = torch.zeros((self.k, self.k), dtype=torch.float64, device=self.device)
        b_tilde = torch.zeros((self.k, 1), dtype=torch.float64, device=self.device)
        L = self.buffer.buffer_size
        print("Computing least squares system...")
        for start_index in tqdm(range(0, L, self.large_batch_size)):
            end_index = start_index + self.large_batch_size
            next_batch = self.buffer.get_batch_from_indices(start_index, end_index)
            states = next_batch["states"].to(torch.float64)
            discrete_actions = next_batch["actions"]
            rewards = next_batch["rewards"].to(torch.float64)
            next_states = next_batch["next_states"].to(torch.float64)
            flattened_actions = discrete_actions.flatten().to(torch.int64)
            actions = F.one_hot(flattened_actions, num_classes=self.num_actions)
            rewards = rewards.unsqueeze(1)
            phi_tilde = self.construct_phi_matrix(states, actions)
            phi_prime_tilde = self.construct_phi_prime_matrix(next_states)
            A_tilde += 1 / L * phi_tilde.T @ (phi_tilde - self.gamma * phi_prime_tilde)
            b_tilde += 1 / L * phi_tilde.T @ rewards
        return A_tilde, b_tilde

    def state_featurizer(self, states: torch.Tensor) -> torch.Tensor:
        augmented_states = states[:, 4:-4]
        # FIXME MED: could run below at beginning to pick out degenerate columns
        # running_identical_tensor = torch.ones_like(augmented_states[0, :], dtype=torch.int64)
        # print(f"running: {torch.sum(running_identical_tensor).item()}")
        # for row_idx in range(1, augmented_states.shape[0]):
        #     current_identical_tensor = augmented_states[row_idx - 1, :] == augmented_states[row_idx, :]
        #     print(f"current: {torch.sum(current_identical_tensor).item()}")
        #     running_identical_tensor *= current_identical_tensor.to(torch.int64)
        #     print(f"running: {torch.sum(running_identical_tensor).item()}")
        # print(running_identical_tensor)
        # for idx in range(states.shape[0]):
        #     print(states[idx, :])
        #     sleep(1)
        new_states = augmented_states
        for power in range(2, self.observation_power + 1):
            polynomial_states = augmented_states ** power
            new_states = torch.cat([new_states, polynomial_states], dim=1)
        return new_states

    def construct_phi_matrix(
        self, batch_states: torch.Tensor, batch_actions: torch.Tensor
    ) -> torch.Tensor:
        phi_s = self.state_featurizer(batch_states)
        phi_matrix = torch.cat([phi_s, batch_actions], dim=1)
        return phi_matrix

    def construct_phi_prime_matrix(
        self, batch_next_states: torch.Tensor
    ) -> torch.Tensor:
        phi_s_prime = self.state_featurizer(batch_next_states)
        # NOTE: should the policy be pessimistic here? probably not...
        pi_s_prime = self.Q_policy(batch_next_states, one_hot=True, pessimistic=False)
        phi_prime_matrix = torch.cat([phi_s_prime, pi_s_prime], dim=1)
        return phi_prime_matrix

    def Q_policy(
        self, observations: torch.Tensor, one_hot: bool = True, pessimistic: bool = True
    ) -> torch.Tensor:
        batch_size = observations.shape[0]
        phi_s = self.state_featurizer(observations)
        phi_s_repeated = phi_s.unsqueeze(1).repeat(1, self.num_actions, 1)
        action_features_repeated = self.action_one_hots.repeat(batch_size, 1, 1)
        phi_matrix = torch.cat([phi_s_repeated, action_features_repeated], dim=2)
        phi_columns = phi_matrix.permute((0, 2, 1))
        batch_w_tilde = self.w_tilde.unsqueeze(0).repeat(batch_size, 1, 1)
        Q_values = torch.bmm(phi_matrix, batch_w_tilde)
        if pessimistic:
            Gamma = self.compute_uncertainty_estimate(batch_size, phi_columns)
            Q_values -= Gamma
        optimal_actions = torch.argmax(Q_values, dim=1)
        if not one_hot:
            return optimal_actions
        argmax_indices = torch.squeeze(optimal_actions, 1)
        optimal_action_one_hots = torch.index_select(
            self.action_one_hots, dim=1, index=argmax_indices
        ).squeeze(0)
        return optimal_action_one_hots

    def compute_uncertainty_estimate(
        self, batch_size: int, phi_columns: torch.Tensor
    ) -> torch.Tensor:
        batch_Sigma_inverse = self.Sigma_inverse.unsqueeze(0).repeat(batch_size, 1, 1)
        Sigma_inverse_phi = torch.bmm(batch_Sigma_inverse, phi_columns)
        inside_sqrt = torch.sum(phi_columns * Sigma_inverse_phi, dim=1)
        inside_sqrt = inside_sqrt.unsqueeze(-1)
        Gamma = self.beta * torch.sqrt(inside_sqrt)
        return Gamma

    def save(self) -> None:
        # NOTE: removing buffer to save space
        del self.buffer
        pickle.dump(self, open(self.env_name + ".Actor", "wb"))

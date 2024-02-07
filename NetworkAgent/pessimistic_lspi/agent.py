import numpy as np
import os
import pickle
import sys
import torch
from tqdm import tqdm

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from buffer import CombinedBuffer
from offline_env import OfflineEnv


class PessimisticLSPI:
    def __init__(
        self,
        env: OfflineEnv,
        num_users: int,
        beta: float,
        large_batch_size: int = 2 ** 9,
        save_folder: str = "saved",
    ):
        self.gamma = 0.99
        self.beta = beta
        self.buffer: CombinedBuffer = env.buffer
        self.observation_dim = self.buffer.states.shape[1]
        self.num_users = num_users
        self.large_batch_size = large_batch_size
        # check if the save_folder path exists
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(script_dir, save_folder)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        self.env_name = os.path.join(
            save_dir,
            f"PessimisticLSPI_{env.algo_name}_beta_{self.beta}",
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device} device...")
        self.initialize_Q_weights()
        self.initialize_full_action_map()
        self.compute_Sigma_inverse()

    def initialize_Q_weights(self) -> None:
        # NOTE: removing the last four elements of observation, because the y-dimension
        # of the UE's stays zero for the whole simulation(s)
        # NOTE: also removing first four elements (LTE max rate never changes)
        self.pruned_observation_dim = self.observation_dim - 8
        self.k = self.pruned_observation_dim * (3 * self.num_users)
        print(f"k = {self.k}")
        self.w_tilde = torch.randn((self.k, 1), dtype=torch.float64, device="cuda:0")
        self.previous_w_difference_norm = torch.inf

    def initialize_full_action_map(self) -> None:
        self.num_actions_per_user = 3
        self.num_actions = self.num_actions_per_user ** self.num_users
        self.all_actions = torch.arange(
            0, self.num_actions, 1, dtype=torch.int64, device=self.device
        ).unsqueeze(-1)
        all_user_discretized_actions = self.get_user_discretized_actions_tensor(
            self.all_actions, self.num_users, self.num_actions_per_user
        )
        self.all_action_mask = self.remap_actions_for_phi(all_user_discretized_actions)

    def compute_Sigma_inverse(self) -> None:
        L = self.buffer.buffer_size
        print("Computing Sigma_inverse...")
        Sigma = torch.zeros((self.k, self.k), dtype=torch.float64, device=self.device)
        for start_index in tqdm(range(0, L, self.large_batch_size)):
            end_index = start_index + self.large_batch_size
            next_batch = self.buffer.get_batch_from_indices(start_index, end_index)
            states = next_batch["states"].to(torch.float64)
            discrete_actions = next_batch["actions"].to(torch.int64)
            phi_tilde = self.state_action_featurizer(states, discrete_actions)
            Sigma += phi_tilde.T @ phi_tilde
        try:
            self.Sigma_inverse: torch.Tensor = torch.linalg.inv(Sigma)
        except:
            print("WARNING: Sigma_inverse non-invertible. Trying ridge regression...")
            Sigma += (1.0e-6) * torch.eye(
                self.k, dtype=torch.float64, device=self.device
            )
            self.Sigma_inverse: torch.Tensor = torch.linalg.inv(Sigma)
        finally:
            self.batch_Sigma_inverse = self.Sigma_inverse.unsqueeze(0).repeat(self.large_batch_size, 1, 1)

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[int, dict]:
        if deterministic:
            flat_observation = observation.flatten()
            tensor_observation = torch.tensor(
                flat_observation, dtype=torch.float64, device="cuda:0"
            ).unsqueeze(0)
            actor_action_tensor = self.Q_policy(tensor_observation, pessimistic=True)
            actor_action = int(actor_action_tensor.item())
        else:
            actor_action = np.random.randint(0, self.num_actions)
        return actor_action, {}

    def LSTDQ_update(self) -> None:
        w_difference_norm = 1.0
        iteration = 0
        while (
            w_difference_norm >= 1.0e-6
            and w_difference_norm != self.previous_w_difference_norm
        ):
            self.previous_w_difference_norm = w_difference_norm
            A_tilde, b_tilde = self.compute_least_squares_system()
            print("Computing w_tilde...")
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
            update_A_tilde, update_b_tilde = self.get_update_to_least_squares_system(
                L, start_index, end_index
            )
            A_tilde += update_A_tilde
            b_tilde += update_b_tilde
        return A_tilde, b_tilde

    def get_update_to_least_squares_system(
        self, L: int, start_index: int, end_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_batch = self.buffer.get_batch_from_indices(start_index, end_index)
        states = next_batch["states"].to(torch.float64)
        discrete_actions = next_batch["actions"].to(torch.int64)
        rewards = next_batch["rewards"].to(torch.float64)
        next_states = next_batch["next_states"].to(torch.float64)
        rewards = rewards.unsqueeze(1)
        phi_tilde = self.state_action_featurizer(states, discrete_actions)
        # NOTE: should the policy be pessimistic here?
        pi_s_prime = self.Q_policy(next_states, pessimistic=True)
        phi_prime_tilde = self.state_action_featurizer(next_states, pi_s_prime)
        update_A_tilde = (
            1 / L * phi_tilde.T @ (phi_tilde - self.gamma * phi_prime_tilde)
        )
        update_b_tilde = 1 / L * phi_tilde.T @ rewards
        return update_A_tilde, update_b_tilde

    def get_augmented_states(self, states: torch.Tensor) -> torch.Tensor:
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
        return augmented_states

    def get_user_discretized_actions_tensor(
        self, discrete_actions: torch.Tensor, num_users: int, num_actions_per_user: int
    ) -> torch.Tensor:
        discrete_actions = discrete_actions.squeeze(-1)
        user_specific_actions = torch.zeros(
            (discrete_actions.shape[0], num_users),
            dtype=torch.int64,
            device=self.device,
        )
        num_possible_actions: int = num_actions_per_user ** num_users
        running_divisors = discrete_actions
        running_dividends = num_possible_actions * torch.ones_like(
            discrete_actions, device=self.device
        )
        for user in range(num_users):
            running_dividends //= num_actions_per_user
            user_actions = running_divisors // running_dividends
            running_divisors = running_divisors % running_dividends
            user_specific_actions[:, user] = user_actions
        return user_specific_actions

    def state_action_featurizer(
        self, batch_states: torch.Tensor, batch_actions: torch.Tensor
    ) -> torch.Tensor:
        # featurize with different weights for each individual user action
        user_discretized_actions = self.get_user_discretized_actions_tensor(
            batch_actions, self.num_users, self.num_actions_per_user
        )
        action_mask = self.remap_actions_for_phi(user_discretized_actions)
        augmented_states = self.get_augmented_states(batch_states)
        repeated_states = augmented_states.repeat(
            [1, self.num_actions_per_user * self.num_users]
        )
        phi_matrix = repeated_states * action_mask
        return phi_matrix

    def remap_actions_for_phi(self, user_actions: torch.Tensor) -> torch.Tensor:
        batch_size = user_actions.shape[0]
        action_mask = torch.empty((batch_size, 0), device=self.device)
        # NOTE: double loop for 12 iterations total (4*3) - any way to speed it up?
        # does it matter / make a difference?
        for k in range(self.num_users):
            individual_actions = user_actions[:, k]
            for n in range(self.num_actions_per_user):
                small_action_mask = (
                    (individual_actions == n).unsqueeze(-1).to(torch.int64)
                )
                repeated_mask = small_action_mask.repeat(
                    (1, self.pruned_observation_dim)
                )
                action_mask = torch.cat([action_mask, repeated_mask], dim=1)
        assert action_mask.shape[1] == self.k
        return action_mask

    def Q_policy(
        self, observations: torch.Tensor, pessimistic: bool = True
    ) -> torch.Tensor:
        batch_size = observations.shape[0]
        augmented_states = self.get_augmented_states(observations)
        repeated_states = augmented_states.unsqueeze(1).repeat(
            1, self.num_actions, self.num_actions_per_user * self.num_users
        )
        # repeated_states shape: (batch_size, num_actions, self.k)
        action_mask_repeated = self.all_action_mask.repeat(batch_size, 1, 1)
        phi_matrix = repeated_states * action_mask_repeated
        batch_w_tilde = self.w_tilde.unsqueeze(0).repeat(batch_size, 1, 1)
        Q_values = torch.bmm(phi_matrix, batch_w_tilde)
        if pessimistic and self.beta > 0.0:
            phi_columns = phi_matrix.permute((0, 2, 1))
            Gamma = self.compute_uncertainty_estimate(batch_size, phi_columns)
            Q_values -= Gamma
        optimal_actions = torch.argmax(Q_values, dim=1)
        return optimal_actions

    def compute_uncertainty_estimate(
        self, batch_size: int, phi_columns: torch.Tensor
    ) -> torch.Tensor:
        # print(
        #     "torch.cuda.memory_allocated: %fGB"
        #     % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
        # )
        # print(
        #     "torch.cuda.memory_reserved: %fGB"
        #     % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024)
        # )
        # print(
        #     "torch.cuda.max_memory_reserved: %fGB"
        #     % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024)
        # )
        batch_Sigma_inverse = self.batch_Sigma_inverse[:batch_size, :, :]
        Sigma_inverse_phi = torch.bmm(batch_Sigma_inverse, phi_columns)
        inside_sqrt = torch.sum(phi_columns * Sigma_inverse_phi, dim=1)
        inside_sqrt = inside_sqrt.unsqueeze(-1)
        Gamma = self.beta * torch.sqrt(inside_sqrt)
        return Gamma

    def save(self) -> None:
        # NOTE: saving space before writing to disk
        del self.buffer
        # FIXME: this doesn't save memory...
        self.batch_Sigma_inverse = self.batch_Sigma_inverse[self.num_actions, :, :]
        pickle.dump(self, open(self.env_name + ".Actor", "wb"))

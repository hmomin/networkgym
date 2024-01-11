import numpy as np
import torch
import torch.nn.functional as F


class FastLSPI:
    def __init__(
        self,
        observation_dim: int,
        num_actions: int,
    ):
        self.gamma = 0.99
        self.observation_dim = observation_dim
        self.num_actions = num_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device} device...")
        self.initialize_action_feature_map()
        self.initialize_Q_weights()

    def initialize_action_feature_map(self) -> None:
        action_indices = list(range(self.num_actions))
        self.action_one_hots: torch.Tensor = F.one_hot(
            torch.tensor(action_indices, device="cuda:0"),
            num_classes=self.num_actions,
        ).unsqueeze(0)

    def initialize_Q_weights(self) -> None:
        self.k = self.observation_dim + self.num_actions
        self.batch_size = 1
        while self.batch_size < self.k:
            self.batch_size *= 2
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
        # print("----- BUFFER SIZES -----")
        # print(self.states.shape)
        # print(self.actions.shape)
        # print(self.rewards.shape)
        # print(self.next_states.shape)
        # print("------------------------")

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[int, dict]:
        if self.full_rank_reached and deterministic:
            flat_observation = observation.flatten()
            tensor_observation = torch.tensor(
                flat_observation, dtype=torch.float32, device="cuda:0"
            ).unsqueeze(0)
            actor_action_tensor = self.Q_policy(tensor_observation.T, one_hot=False)
            actor_action = int(actor_action_tensor.item())
        else:
            actor_action = np.random.randint(0, self.num_actions)
        return actor_action, {}

    def Q_policy(
        self, observations: torch.Tensor, one_hot: bool = True
    ) -> torch.Tensor:
        with torch.no_grad():
            phi_s = observations.T
            phi_s_repeated = phi_s.unsqueeze(1).repeat(1, self.num_actions, 1)
            action_features_repeated = self.action_one_hots.repeat(
                phi_s_repeated.shape[0], 1, 1
            )
            phi_matrix = torch.cat([phi_s_repeated, action_features_repeated], dim=2)
            batch_w_tilde = self.w_tilde.unsqueeze(0).repeat(
                phi_s_repeated.shape[0], 1, 1
            )
            Q_values = torch.bmm(phi_matrix, batch_w_tilde)
            optimal_actions = torch.argmax(Q_values, dim=1)
            if not one_hot:
                return optimal_actions
            argmax_indices = torch.squeeze(optimal_actions, 1)
            optimal_action_one_hots = torch.index_select(
                self.action_one_hots, dim=1, index=argmax_indices
            ).squeeze(0)
            return optimal_action_one_hots

    def construct_phi_matrix(self, indices: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            batch_states = self.states[:, indices]
            batch_actions = self.actions[:, indices]
            phi_s = batch_states.T
            phi_matrix = torch.cat([phi_s, batch_actions.T], dim=1)
            return phi_matrix

    def construct_phi_prime_matrix(self, indices: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            batch_next_states = self.next_states[:, indices]
            phi_s_prime = batch_next_states.T
            pi_s_prime = self.Q_policy(batch_next_states, one_hot=True)
            phi_prime_matrix = torch.cat([phi_s_prime, pi_s_prime], dim=1)
            return phi_prime_matrix

    def update(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray
    ) -> None:
        self.LSTDQ_update(state, action, reward, next_state)

    # NOTE: incremental update of weight vector for Q-hat
    def LSTDQ_update(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray
    ) -> None:
        state = state.flatten()
        next_state = next_state.flatten()
        self.store_to_buffer(state, action, reward, next_state)
        # NOTE: no point in doing LSTDQ update if the rank is too small
        if self.L < max(self.k, self.batch_size):
            return
        norm_difference = float("inf")
        iteration = 0
        indices = torch.randint(0, self.L, (self.batch_size,), device=self.device)
        while norm_difference >= 1.0e-6 and iteration < 6:
            phi_tilde = self.construct_phi_matrix(indices)
            phi_prime_tilde = self.construct_phi_prime_matrix(indices)
            batch_rewards = self.rewards[:, indices]
            A_tilde = phi_tilde.T @ (phi_tilde - self.gamma * phi_prime_tilde)
            b_tilde = phi_tilde.T @ batch_rewards.T
            w_tilde = torch.linalg.lstsq(A_tilde, b_tilde).solution
            # NOTE: if the matrix is rank-deficient, w_tilde will be all NaNs
            # in this case, just don't do an update
            if torch.isnan(w_tilde).any():
                self.full_rank_reached = False
                rank_A_tilde = torch.linalg.matrix_rank(A_tilde)
                print(f"A_tilde defective - rank {rank_A_tilde} < {A_tilde.shape[0]}")
                print(
                    f"Updating batch size from {self.batch_size} to {2 * self.batch_size}..."
                )
                self.batch_size *= 2
                break
            else:
                self.full_rank_reached = True
                norm_difference = torch.norm(w_tilde - self.w_tilde, float("inf"))
                # print(f"||w - w'||_inf: {norm_difference}")
                self.w_tilde = w_tilde
            iteration += 1

    def get_Q_value(self, state: np.ndarray, action: int) -> float:
        with torch.no_grad():
            state = state.flatten()
            tensor_state = torch.tensor(
                state, dtype=torch.float32, device="cuda:0"
            ).unsqueeze(0)
            phi_s = tensor_state
            phi_a = self.action_one_hots[:, action, :]
            phi_s_a = torch.cat([phi_s, phi_a], dim=1)
            Q_value = (phi_s_a @ self.w_tilde).item()
            return Q_value


def main() -> None:
    agent_thingy = FastLSPI(
        observation_dim=14 * 4,
        num_actions=3 ** 4,
    )
    random_state = np.random.random((14, 4))
    # testing generation of weights for Q-hat
    for iter in range(2500):
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
        agent_thingy.update(random_state, action, random_reward, random_next_state)
        random_state = random_next_state
    # evaluation
    for action in range(agent_thingy.num_actions):
        Q_value = agent_thingy.get_Q_value(random_state, action)
        print(f"action: {action} -> Q_value: {Q_value}")


if __name__ == "__main__":
    main()

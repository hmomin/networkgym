import numpy as np
import os
import torch
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO


class PPO_LSPI:
    def __init__(self, model_name: str = "PPO", epsilon: float = 1.0e-3):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(this_dir, "models", model_name)
        agent = PPO.load(model_path)
        if type(agent.action_space) != spaces.Discrete:
            raise Exception("ERROR: discrete action space expected for LSPI usage.")
        self.gamma = 0.99
        self.epsilon = epsilon
        self.num_actions: int = int(agent.action_space.n)
        self.store_parameters(agent.get_parameters()["policy"])
        self.initialize_Q_weights()

    # NOTE: store the policy and value function parameters before the last layer.
    # need them for calculating featurization
    def store_parameters(self, param_dict: dict[str, torch.Tensor]) -> None:
        self.pi_W1 = param_dict["mlp_extractor.policy_net.0.weight"]
        self.pi_b1 = param_dict["mlp_extractor.policy_net.0.bias"].unsqueeze(1)
        self.pi_W2 = param_dict["mlp_extractor.policy_net.2.weight"]
        self.pi_b2 = param_dict["mlp_extractor.policy_net.2.bias"].unsqueeze(1)

        self.vf_W1 = param_dict["mlp_extractor.value_net.0.weight"]
        self.vf_b1 = param_dict["mlp_extractor.value_net.0.bias"].unsqueeze(1)
        self.vf_W2 = param_dict["mlp_extractor.value_net.2.weight"]
        self.vf_b2 = param_dict["mlp_extractor.value_net.2.bias"].unsqueeze(1)

    def initialize_Q_weights(self) -> None:
        self.k = self.pi_b2.shape[0] + self.vf_b2.shape[0] + self.num_actions
        self.w_tilde = torch.randn((self.k, 1), device="cuda:0")

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
        tensor_reward = torch.tensor([reward], dtype=torch.float32, device="cuda:0").unsqueeze(1)
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
        print("----- BUFFER SIZES -----")
        print(self.states.shape)
        print(self.actions.shape)
        print(self.rewards.shape)
        print(self.next_states.shape)
        print("------------------------")

    def policy(self, observation: torch.Tensor, one_hot: bool = True) -> torch.Tensor:
        phi_s = self.compute_state_features(observation)
        phi_s_T = phi_s.T
        phi_s_repeated = phi_s_T.unsqueeze(1).repeat(1, self.num_actions, 1)
        action_indices = list(range(self.num_actions))
        all_actions: torch.Tensor = F.one_hot(
            torch.tensor(action_indices, device="cuda:0"),
            num_classes=self.num_actions,
        ).unsqueeze(0)
        L = phi_s.shape[1]
        repeated_actions = all_actions.repeat(L, 1, 1)
        phi_matrix = torch.cat([phi_s_repeated, repeated_actions], dim=2)
        batch_w_tilde = self.w_tilde.unsqueeze(0).repeat(L, 1, 1)
        Q_values = torch.bmm(phi_matrix, batch_w_tilde)
        optimal_actions = torch.argmax(Q_values, dim=1)
        if not one_hot:
            return optimal_actions
        argmax_indices = torch.squeeze(optimal_actions, 1)
        optimal_action_one_hots = torch.index_select(
            all_actions, dim=1, index=argmax_indices
        ).squeeze(0)
        return optimal_action_one_hots

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[int, dict]:
        flat_observation = observation.flatten()
        tensor_observation = torch.tensor(
            flat_observation, dtype=torch.float32, device="cuda:0"
        ).unsqueeze(1)
        if not deterministic:
            raise Exception("ERROR: stochastic action not supported by LSPI!")
        optimal_action_tensor = self.policy(tensor_observation, one_hot=False)
        optimal_action = int(optimal_action_tensor.item())
        return optimal_action, {}

    def compute_state_features(self, states: torch.Tensor) -> torch.Tensor:
        pi_layer_1 = torch.tanh(self.pi_W1 @ states + self.pi_b1)
        pi_layer_2 = torch.tanh(self.pi_W2 @ pi_layer_1 + self.pi_b2)

        vf_layer_1 = torch.tanh(self.vf_W1 @ states + self.vf_b1)
        vf_layer_2 = torch.tanh(self.vf_W2 @ vf_layer_1 + self.vf_b2)

        phi_s = torch.cat([pi_layer_2, vf_layer_2], dim=0)
        return phi_s

    def construct_phi_matrix(self) -> torch.Tensor:
        phi_s = self.compute_state_features(self.states)
        phi_matrix_transpose = torch.cat([phi_s, self.actions], dim=0)
        phi_matrix = phi_matrix_transpose.T
        return phi_matrix

    def construct_phi_prime_matrix(self) -> torch.Tensor:
        phi_s_prime = self.compute_state_features(self.next_states)
        pi_s_prime = self.policy(self.next_states, one_hot=True)
        phi_prime_matrix = torch.cat([phi_s_prime.T, pi_s_prime], dim=1)
        return phi_prime_matrix

    # NOTE: incremental update of weight vector for Q-hat
    def LSTDQ_update(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray
    ) -> None:
        state = state.flatten()
        next_state = next_state.flatten()
        self.store_to_buffer(state, action, reward, next_state)
        norm_difference = float('inf')
        iteration = 0
        while norm_difference >= self.epsilon and iteration < 6:
            phi_tilde = self.construct_phi_matrix()
            phi_prime_tilde = self.construct_phi_prime_matrix()

            A_tilde = phi_tilde.T @ (phi_tilde - self.gamma * phi_prime_tilde)
            b_tilde = phi_tilde.T @ self.rewards.T
            w_tilde = torch.linalg.lstsq(A_tilde, b_tilde).solution
            # NOTE: if the matrix is rank-deficient, w_tilde will be all NaNs
            # in this case, do ridge regression
            ridge_lambda = 1.0
            if torch.isnan(w_tilde).any():
                print(f"RIDGE_LAMBDA: {ridge_lambda}")
                A_ridge = A_tilde.T @ A_tilde + ridge_lambda * torch.eye(self.k, device="cuda:0")
                b_ridge = A_tilde.T @ b_tilde
                w_tilde = torch.linalg.lstsq(A_ridge, b_ridge).solution
            norm_difference = torch.norm(w_tilde - self.w_tilde, float("inf"))
            self.w_tilde = w_tilde
            print(f"||w - w'||_inf: {norm_difference}")
            iteration += 1

    def get_Q_value(self, state: np.ndarray, action: int) -> float:
        state = state.flatten()
        tensor_state = torch.tensor(
            state, dtype=torch.float32, device="cuda:0"
        ).unsqueeze(1)
        phi_s = self.compute_state_features(tensor_state)
        action_one_hot_vector = F.one_hot(
            torch.tensor(action, device="cuda:0"),
            num_classes=self.num_actions,
        ).unsqueeze(1)
        phi_s_a = torch.cat([phi_s, action_one_hot_vector], dim=0)
        Q_value = (phi_s_a.T @ self.w_tilde).item()
        return Q_value


def main() -> None:
    agent_thingy = PPO_LSPI()
    random_state = np.random.random((14, 4))
    # testing generation of weights for Q-hat
    for iter in range(3000):
        action, _ = agent_thingy.predict(random_state)
        print(f"iteration: {iter}, action: {action}")
        # NOTE: expect the action to eventually converge to 42, independent of state
        if action == 42:
            random_reward = 1.0
        elif action == 24:
            random_reward = 0.5
        else:
            random_reward = -10.0
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

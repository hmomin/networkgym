import numpy as np
import os
import torch
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from tqdm import tqdm


class PPO_LSPI:
    def __init__(self, model_name: str = "PPO"):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(this_dir, "models", model_name)
        agent = PPO.load(model_path)
        if type(agent.action_space) != spaces.Discrete:
            raise Exception("ERROR: discrete action space expected for LSPI usage.")
        self.gamma = 0.99
        self.num_actions: np.int64 = agent.action_space.n
        self.store_parameters(agent.get_parameters()["policy"])
        self.initialize_Q_weights()
        self.initialize_buffer()

    # NOTE: store the policy and value function parameters before the last layer
    # need them for calculating featurization
    def store_parameters(self, param_dict: dict[str, torch.Tensor]) -> None:
        self.pi_W1 = param_dict["mlp_extractor.policy_net.0.weight"]
        self.pi_b1 = param_dict["mlp_extractor.policy_net.0.bias"]
        self.pi_W2 = param_dict["mlp_extractor.policy_net.2.weight"]
        self.pi_b2 = param_dict["mlp_extractor.policy_net.2.bias"]

        self.vf_W1 = param_dict["mlp_extractor.value_net.0.weight"]
        self.vf_b1 = param_dict["mlp_extractor.value_net.0.bias"]
        self.vf_W2 = param_dict["mlp_extractor.value_net.2.weight"]
        self.vf_b2 = param_dict["mlp_extractor.value_net.2.bias"]

    def initialize_Q_weights(self) -> None:
        self.k = int(self.pi_b2.shape[0] + self.vf_b2.shape[0] + self.num_actions)
        self.w_tilde = torch.randn((self.k), device="cuda:0")

    def initialize_buffer(self) -> None:
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.next_states: list[np.ndarray] = []

    def store_to_buffer(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        if len(self.states) > 100:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.states.pop(0)

    def policy(self, observation: np.ndarray) -> int:
        phi_s = self.compute_state_features(observation)
        max_Q_value = -torch.inf
        best_action = -1
        for discrete_action in range(self.num_actions):
            phi_s_a = self.compute_state_action_features(phi_s, discrete_action)
            Q_value = torch.dot(phi_s_a, self.w_tilde)
            if Q_value > max_Q_value:
                best_action = discrete_action
                max_Q_value = Q_value
        if best_action == -1:
            raise Exception("ERROR: couldn't find Q-value higher than -infinity?...")
        return best_action

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[int, dict]:
        if not deterministic:
            raise Exception("ERROR: stochastic action not supported by LSPI!")
        return self.policy(observation), {}

    def compute_state_features(self, observation: np.ndarray) -> torch.Tensor:
        flat_observation = observation.flatten()
        x = torch.tensor(flat_observation, dtype=torch.float32, device="cuda:0")

        pi_layer_1 = torch.tanh(self.pi_W1 @ x + self.pi_b1)
        pi_layer_2 = torch.tanh(self.pi_W2 @ pi_layer_1 + self.pi_b2)

        vf_layer_1 = torch.tanh(self.vf_W1 @ x + self.vf_b1)
        vf_layer_2 = torch.tanh(self.vf_W2 @ vf_layer_1 + self.vf_b2)

        phi_s = torch.cat([pi_layer_2, vf_layer_2])
        return phi_s

    def compute_state_action_features(
        self, phi_s: torch.Tensor, discrete_action: int
    ) -> torch.Tensor:
        action_one_hot_vector = F.one_hot(
            torch.tensor(discrete_action, device="cuda:0"),
            num_classes=self.num_actions,
        )
        phi_s_a = torch.cat([phi_s, action_one_hot_vector])
        return phi_s_a

    # NOTE: incremental update of weight vector for Q-hat
    def LSTDQ_update(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray
    ) -> None:
        self.store_to_buffer(state, action, reward, next_state)

        delta = 1.0e-6
        # B_tilde = 1 / delta * torch.eye(self.k, device="cuda:0")
        A_tilde = delta * torch.eye(self.k, device="cuda:0")
        b_tilde = torch.zeros((self.k), device="cuda:0")
        for s, a, r, s_prime in tqdm(zip(
            self.states, self.actions, self.rewards, self.next_states
        )):
            phi_s = self.compute_state_features(s)
            phi_s_a = self.compute_state_action_features(phi_s, a)
            phi_s_prime = self.compute_state_features(s_prime)
            phi_s_prime_policy_s_prime = self.compute_state_action_features(
                phi_s_prime, self.policy(s_prime)
            )
            # x = φ(s,a)
            # y_transpose = (φ(s,a) - γφ(s',π(s')))^T
            # A += x * y_transpose
            x = torch.unsqueeze(phi_s_a, 1)
            y_transpose = torch.unsqueeze(
                phi_s_a - self.gamma * phi_s_prime_policy_s_prime, 1
            ).T
            A_tilde += x @ y_transpose
            # B_tilde -= (B_tilde @ x @ y_transpose @ B_tilde) / (
            #     1 + y_transpose @ B_tilde @ x
            # )
            # b += x*reward
            b_tilde += phi_s_a * r
        # matrix_difference = torch.inverse(A_tilde) - B_tilde
        # print(matrix_difference)
        # print(f"MATRIX DIFFERENCE: {torch.norm(matrix_difference, 'fro').item()}")

        self.w_tilde = torch.inverse(A_tilde) @ b_tilde
        # self.w_tilde = B_tilde @ b_tilde

    def get_Q_value(self, state: np.ndarray, action: int) -> float:
        phi_s = self.compute_state_features(state)
        phi_s_a = self.compute_state_action_features(phi_s, action)
        Q_value = torch.dot(phi_s_a, self.w_tilde).item()
        return Q_value


def main() -> None:
    agent_thingy = PPO_LSPI()
    random_state = np.random.random((14, 4))
    # testing generation of weights for Q-hat
    for iter in range(100):
        action, _ = agent_thingy.predict(random_state)
        print(f"iteration: {iter}, action: {action}")
        random_reward = 0.2 * np.random.random() - 0.1
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

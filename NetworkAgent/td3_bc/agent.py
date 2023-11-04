import numpy as np
import os
import pickle
import torch as T
import torch.nn as nn
from copy import deepcopy
from buffer import CombinedBuffer
from td3_bc.network import Network
from offline_env import OfflineEnv


class Agent:
    def __init__(
        self,
        env: OfflineEnv,
        learning_rate: float,
        gamma: float,
        tau: float,
        # NOTE: here, we divide the standard literature value for alpha (2.5) by 4.0 to
        # preserve the shift in action disparity range (1^2 vs. 2^2)
        alpha_bc: float = 2.5 / 4.0,
        enable_behavioral_cloning: bool = False,
        normalize: bool = True,
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
        self.alpha_bc = alpha_bc
        self.behavioral_cloning = enable_behavioral_cloning
        self.normalize = normalize
        # check if the save_folder path exists
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(script_dir, save_folder)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        bc_string = "bc" if self.behavioral_cloning else "normal"
        self.env_name = os.path.join(save_dir, f"{env.algo_name}_{bc_string}.")
        name = self.env_name
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        print(f"Using {self.device} device...")
        self.training_stats: list[list[float]] = []
        # initialize the actor and critics
        # NOTE: actor activation is sigmoid instead of tanh (from the paper)
        # to satisfy the action bounds requirement
        self.actor = (
            pickle.load(open(name + "Actor", "rb"))
            if should_load and os.path.exists(name + "Actor")
            else Network(
                [self.observation_dim, 400, 300, self.action_dim],
                nn.Sigmoid,
                learning_rate,
                self.device,
            )
        )
        self.critic1 = (
            pickle.load(open(name + "Critic1", "rb"))
            if should_load and os.path.exists(name + "Critic1")
            else Network(
                [self.observation_dim + self.action_dim, 400, 300, 1],
                nn.Identity,
                learning_rate,
                self.device,
            )
        )
        self.critic2 = (
            pickle.load(open(name + "Critic2", "rb"))
            if should_load and os.path.exists(name + "Critic2")
            else Network(
                [self.observation_dim + self.action_dim, 400, 300, 1],
                nn.Identity,
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
        state_tensor = T.tensor(state, device=self.device)
        actions: T.Tensor = self.actor.forward(state_tensor)
        return actions.cpu().detach().numpy()

    def update(
        self,
        mini_batch_size: int,
        training_sigma: float,
        training_clip: float,
        update_policy: bool,
    ):
        self.training_stats.append([])
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
            policy_loss = self.compute_policy_loss(states, actions)
            self.training_stats[-1].append(policy_loss.item())
            # print(f"\tpi_loss: {policy_loss.item()}")
            self.actor.gradient_descent_step(policy_loss)
            # update target networks
            self.update_target_network(self.target_actor, self.actor)
            self.update_target_network(self.target_critic1, self.critic1)
            self.update_target_network(self.target_critic2, self.critic2)

    def compute_targets(
        self,
        rewards: T.Tensor,
        next_states: T.Tensor,
        dones: T.Tensor,
        training_sigma: float,
        training_clip: float,
    ) -> T.Tensor:
        target_actions = self.target_actor.forward(next_states.float())
        # create additive noise for target actions
        noise = T.normal(0, training_sigma, target_actions.shape, device=self.device)
        clipped_noise = T.clip(noise, -training_clip, +training_clip)
        target_actions = T.clip(
            target_actions + clipped_noise,
            self.low_action_bound,
            self.high_action_bound,
        )
        # compute targets
        target_Q1_values = T.squeeze(
            self.target_critic1.forward(T.hstack([next_states, target_actions]).float())
        )
        target_Q2_values = T.squeeze(
            self.target_critic2.forward(T.hstack([next_states, target_actions]).float())
        )
        target_Q_values = T.minimum(target_Q1_values, target_Q2_values)
        return rewards + self.gamma * (1 - dones) * target_Q_values

    def compute_Q_loss(
        self, network: Network, states: T.Tensor, actions: T.Tensor, targets: T.Tensor
    ) -> T.Tensor:
        # compute the MSE of the Q function with respect to the targets
        Q_values = T.squeeze(network.forward(T.hstack([states, actions]).float()))
        return T.square(Q_values - targets).mean()

    def compute_policy_loss(self, states: T.Tensor, actions: T.Tensor):
        policy_actions = self.actor.forward(states.float())
        Q_values = T.squeeze(
            self.critic1.forward(T.hstack([states, policy_actions]).float())
        )
        Q_term = Q_values.mean()
        if self.behavioral_cloning:
            mean_absolute_Q_values = T.abs(Q_values).mean().detach()
            lambda_bc = self.alpha_bc / mean_absolute_Q_values
            behavioral_cloning_term = (policy_actions - actions).square().mean()
            policy_loss = -(lambda_bc * Q_term - behavioral_cloning_term)
        else:
            policy_loss = -Q_term
        return policy_loss

    def update_target_network(self, target_network: Network, network: Network):
        with T.no_grad():
            for target_parameter, parameter in zip(
                target_network.parameters(), network.parameters()
            ):
                target_parameter.mul_(1 - self.tau)
                target_parameter.add_(self.tau * parameter)

    def save(self, step: int = 0, max_steps: int = 1_000_000):
        step_str = str(step).zfill(len(str(max_steps)))
        alpha_str = f"{self.alpha_bc:.3f}"
        norm_str = "normalized" if self.normalize else "not_normalized"
        name = f"{self.env_name}{step_str}.{self.buffer.num_buffers}.{alpha_str}.{norm_str}."
        pickle.dump(self.training_stats, open(name + "training_stats", "wb"))
        pickle.dump(self.actor, open(name + "Actor", "wb"))
        if self.normalize:
            pickle.dump(
                (self.buffer.mean_state, self.buffer.stdev_state),
                open(name + "Normalizers", "wb"),
            )
        pickle.dump(self.critic1, open(name + "Critic1", "wb"))
        pickle.dump(self.critic2, open(name + "Critic2", "wb"))
        pickle.dump(self.target_actor, open(name + "TargetActor", "wb"))
        pickle.dump(self.target_critic1, open(name + "TargetCritic1", "wb"))
        pickle.dump(self.target_critic2, open(name + "TargetCritic2", "wb"))

import numpy as np
import os
import pickle
import torch as T
import torch.nn as nn
from copy import deepcopy
from gym.core import Env
from buffer import Buffer
from td3.network import Network
from NetworkAgent.offline_env import OfflineEnv


class Agent:
    # FIXME: a lot of the saving/loading needs to be reworked!
    def __init__(
        self,
        env: OfflineEnv,
        learning_rate: float,
        gamma: float,
        tau: float,
        should_load: bool = True,
        save_folder: str = "saved",
    ):
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        # check if the save_folder path exists
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        self.env_name = os.path.join(save_folder, env.name + ".")
        name = self.env_name
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.buffer = (
            pickle.load(open(name + "Replay", "rb"))
            if should_load and os.path.exists(name + "Replay")
            else Buffer(self.observation_dim, self.action_dim)
        )
        # initialize the actor and critics
        self.actor = (
            pickle.load(open(name + "Actor", "rb"))
            if should_load and os.path.exists(name + "Actor")
            else Network(
                [self.observation_dim, 256, 256, self.action_dim],
                nn.Tanh,
                learning_rate,
                self.device,
            )
        )
        self.critic1 = (
            pickle.load(open(name + "Critic1", "rb"))
            if should_load and os.path.exists(name + "Critic1")
            else Network(
                [self.observation_dim + self.action_dim, 256, 256, 1],
                nn.Identity,
                learning_rate,
                self.device,
            )
        )
        self.critic2 = (
            pickle.load(open(name + "Critic2", "rb"))
            if should_load and os.path.exists(name + "Critic2")
            else Network(
                [self.observation_dim + self.action_dim, 256, 256, 1],
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
        return np.clip(deterministic_action + noise, -1, +1)

    def get_deterministic_action(self, state: np.ndarray) -> np.ndarray:
        actions: T.Tensor = self.actor.forward(T.tensor(state, device=self.device))
        return actions.cpu().detach().numpy()

    def update(
        self,
        mini_batch_size: int,
        training_sigma: float,
        training_clip: float,
        update_policy: bool,
    ):
        # randomly sample a mini-batch from the replay buffer
        mini_batch = self.buffer.get_mini_batch(mini_batch_size)
        # create tensors to start generating computational graph
        states = T.tensor(mini_batch["states"], requires_grad=True, device=self.device)
        actions = T.tensor(
            mini_batch["actions"], requires_grad=True, device=self.device
        )
        rewards = T.tensor(
            mini_batch["rewards"], requires_grad=True, device=self.device
        )
        next_states = T.tensor(
            mini_batch["next_states"], requires_grad=True, device=self.device
        )
        dones = T.tensor(
            mini_batch["done_flags"], requires_grad=True, device=self.device
        )
        # compute the targets
        targets = self.compute_targets(
            rewards, next_states, dones, training_sigma, training_clip
        )
        # do a single step on each critic network
        Q1_loss = self.compute_Q_loss(self.critic1, states, actions, targets)
        self.critic1.gradient_descent_step(Q1_loss, True)
        Q2_loss = self.compute_Q_loss(self.critic2, states, actions, targets)
        self.critic2.gradient_descent_step(Q2_loss)
        if update_policy:
            # do a single step on the actor network
            policy_loss = self.compute_policy_loss(states)
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
        noise = np.random.normal(0, training_sigma, target_actions.shape)
        clipped_noise = T.tensor(
            np.clip(noise, -training_clip, +training_clip), device=self.device
        )
        target_actions = T.clip(target_actions + clipped_noise, -1, +1)
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

    def compute_policy_loss(self, states: T.Tensor):
        actions = self.actor.forward(states.float())
        Q_values = T.squeeze(self.critic1.forward(T.hstack([states, actions]).float()))
        return -Q_values.mean()

    def update_target_network(self, target_network: Network, network: Network):
        with T.no_grad():
            for target_parameter, parameter in zip(
                target_network.parameters(), network.parameters()
            ):
                target_parameter.mul_(1 - self.tau)
                target_parameter.add_(self.tau * parameter)

    def save(self):
        name = self.env_name
        pickle.dump(self.buffer, open(name + "Replay", "wb"))
        pickle.dump(self.actor, open(name + "Actor", "wb"))
        pickle.dump(self.critic1, open(name + "Critic1", "wb"))
        pickle.dump(self.critic2, open(name + "Critic2", "wb"))
        pickle.dump(self.target_actor, open(name + "TargetActor", "wb"))
        pickle.dump(self.target_critic1, open(name + "TargetCritic1", "wb"))
        pickle.dump(self.target_critic2, open(name + "TargetCritic2", "wb"))
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


class KL_BC:
    def __init__(
        self,
        env: OfflineEnv,
        num_actions: int,
        learning_rate: float,
        should_load: bool = True,
        save_folder: str = "saved",
    ):
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
        self.training_stats: list[list[float]] = []
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

    def update(self, mini_batch_size: int):
        self.training_stats.append([])
        # randomly sample a mini-batch from the replay buffer
        mini_batch = self.buffer.get_mini_batch(mini_batch_size)
        # create tensors to start generating computational graph
        states = mini_batch["states"]
        actions = mini_batch["actions"]
        # do a single step on the actor network
        policy_loss = self.compute_policy_loss(states, actions)
        self.training_stats[-1].append(policy_loss.item())
        # print(f"\tpi_loss: {policy_loss.item()}")
        self.actor.gradient_descent_step(policy_loss)

    def compute_policy_loss(self, states: T.Tensor, actions: T.Tensor):
        actor_logits = self.actor.forward(states.float())
        actor_probs = F.softmax(actor_logits, dim=1)
        log_probs = T.log(actor_probs)
        behavior_actions = convert_continuous_to_discrete_ratio_action(actions)
        behavior_one_hots = F.one_hot(behavior_actions, num_classes=5**4)
        neg_log_p = T.sum(behavior_one_hots * -log_probs, dim=1)
        policy_loss = T.mean(neg_log_p)
        return policy_loss

    def save(self, step: int = 0, max_steps: int = 1_000_000):
        step_str = str(step).zfill(len(str(max_steps)))
        name = f"{self.env_name}{step_str}.{self.buffer.num_buffers}."
        pickle.dump(self.training_stats, open(name + "training_stats", "wb"))
        pickle.dump(self.actor, open(name + "Actor", "wb"))

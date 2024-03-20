import argparse
import numpy as np
import os
import pickle
import sys
import time

from typing import Callable, Optional, Tuple


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

import torch
from torch import nn
from torch.distributions import Normal, TanhTransform, TransformedDistribution

import sys

sys.path.append(".")
sys.path.append("../")
sys.path.append("../pessimistic_lspi")
sys.path.append("../../")

from network_gym_client import load_config_file
from network_gym_client import Env as NetworkGymEnv
from network_gym_client import ParallelEnv
from ppo_lspi import PPO_LSPI
from fast_lspi.agent_linear import FastLSPI
from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from gymnasium.wrappers.normalize import NormalizeObservation

from NetworkAgent.heuristic_policies import (
    argmax_policy,
    argmin_policy,
    delay_increment_policy,
    utility_increment_policy,
    random_action,
    random_policy,
    random_discrete_policy,
    utility_logistic_policy,
)
from NetworkAgent.config_lock.client_utils import release_lock

# NOTE: seed torch and numpy for reproducibility
torch.manual_seed(0)
np.random.seed(1)


# NOTE: classes for loading other algorithm models...
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()

    def predict(
        self, state: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        return (self.act(state) + 1.0) / 2.0, None


class LB_Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        edac_init: bool,
        max_action: float = 1.0,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # with separate layers works better than with Linear(hidden_dim, 2 * action_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_sigma = nn.Linear(hidden_dim, action_dim)

        if edac_init:
            # init as in the EDAC paper
            for layer in self.trunk[::2]:
                torch.nn.init.constant_(layer.bias, 0.1)

            torch.nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
            torch.nn.init.uniform_(self.mu.bias, -1e-3, 1e-3)
            torch.nn.init.uniform_(self.log_sigma.weight, -1e-3, 1e-3)
            torch.nn.init.uniform_(self.log_sigma.bias, -1e-3, 1e-3)

        self.action_dim = action_dim
        self.max_action = max_action

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        need_log_prob: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden = self.trunk(state)
        mu, log_sigma = self.mu(hidden), self.log_sigma(hidden)

        # clipping params from EDAC paper, not as in SAC paper (-20, 2)
        log_sigma = torch.clip(log_sigma, -5, 2)
        policy_dist = Normal(mu, torch.exp(log_sigma))

        if deterministic:
            action = mu
        else:
            action = policy_dist.rsample()

        tanh_action, log_prob = torch.tanh(action), None
        if need_log_prob:
            # change of variables formula (SAC paper, appendix C, eq 21)
            log_prob = policy_dist.log_prob(action).sum(axis=-1)
            log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(axis=-1)

        return tanh_action * self.max_action, log_prob

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        state = state.flatten()
        deterministic = True
        state = torch.tensor(state, device=device, dtype=torch.float32)
        action = self(state, deterministic=deterministic)[0].cpu().numpy()
        return action

    def predict(
        self, state: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        return (self.act(state, "cpu") + 1.0) / 2.0, None


class SAC_N_Actor(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, max_action: float = 1.0
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # with separate layers works better than with Linear(hidden_dim, 2 * action_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_sigma = nn.Linear(hidden_dim, action_dim)

        # init as in the EDAC paper
        for layer in self.trunk[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.mu.bias, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.bias, -1e-3, 1e-3)

        self.action_dim = action_dim
        self.max_action = max_action

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        need_log_prob: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden = self.trunk(state)
        mu, log_sigma = self.mu(hidden), self.log_sigma(hidden)

        # clipping params from EDAC paper, not as in SAC paper (-20, 2)
        log_sigma = torch.clip(log_sigma, -5, 2)
        policy_dist = Normal(mu, torch.exp(log_sigma))

        if deterministic:
            action = mu
        else:
            action = policy_dist.rsample()

        tanh_action, log_prob = torch.tanh(action), None
        if need_log_prob:
            # change of variables formula (SAC paper, appendix C, eq 21)
            log_prob = policy_dist.log_prob(action).sum(axis=-1)
            log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(axis=-1)

        return tanh_action * self.max_action, log_prob

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        deterministic = True
        state = torch.tensor(state, device=device, dtype=torch.float32)
        action = self(state, deterministic=deterministic)[0].cpu().numpy()
        return action

    def predict(
        self, state: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        state = state.flatten()
        return (self.act(state, "cpu") + 1.0) / 2.0, None


def init_module_weights(module: torch.nn.Sequential, orthogonal_init: bool = False):
    # Specific orthgonal initialization for inner layers
    # If orthogonal init is off, we do not change default initialization
    if orthogonal_init:
        for submodule in module[:-1]:
            if isinstance(submodule, nn.Linear):
                nn.init.orthogonal_(submodule.weight, gain=np.sqrt(2))
                nn.init.constant_(submodule.bias, 0.0)

    # Lasy layers should be initialzied differently as well
    if orthogonal_init:
        nn.init.orthogonal_(module[-1].weight, gain=1e-2)
    else:
        nn.init.xavier_uniform_(module[-1].weight, gain=1e-2)

    nn.init.constant_(module[-1].bias, 0.0)


class ReparameterizedTanhGaussian(nn.Module):
    def __init__(
        self,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        no_tanh: bool = False,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    def log_prob(
        self, mean: torch.Tensor, log_std: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(
        self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(action_distribution.log_prob(action_sample), dim=-1)

        return action_sample, log_prob


class TanhGaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        log_std_multiplier: float = 1.0,
        log_std_offset: float = -1.0,
        orthogonal_init: bool = False,
        no_tanh: bool = False,
    ):
        super().__init__()
        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * action_dim),
        )

        init_module_weights(self.base_network)

        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        _, log_probs = self.tanh_gaussian(mean, log_std, False)
        return log_probs

    def forward(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
        repeat: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        return self.max_action * actions, log_probs

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        with torch.no_grad():
            actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()

    def predict(
        self, state: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        return (self.act(state, "cpu") + 1.0) / 2.0, None


class EDAC_Actor(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, max_action: float = 1.0
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # with separate layers works better than with Linear(hidden_dim, 2 * action_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_sigma = nn.Linear(hidden_dim, action_dim)

        # init as in the EDAC paper
        for layer in self.trunk[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.mu.bias, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.bias, -1e-3, 1e-3)

        self.action_dim = action_dim
        self.max_action = max_action

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        need_log_prob: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden = self.trunk(state)
        mu, log_sigma = self.mu(hidden), self.log_sigma(hidden)

        # clipping params from EDAC paper, not as in SAC paper (-20, 2)
        log_sigma = torch.clip(log_sigma, -5, 2)
        policy_dist = Normal(mu, torch.exp(log_sigma))

        if deterministic:
            action = mu
        else:
            action = policy_dist.rsample()

        tanh_action, log_prob = torch.tanh(action), None
        if need_log_prob:
            # change of variables formula (SAC paper, appendix C, eq 21)
            log_prob = policy_dist.log_prob(action).sum(axis=-1)
            log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(axis=-1)

        return tanh_action * self.max_action, log_prob

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        state = state.flatten()
        deterministic = True
        state = torch.tensor(state, device=device, dtype=torch.float32)
        action = self(state, deterministic=deterministic)[0].cpu().numpy()
        return action

    def predict(
        self, state: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        return (self.act(state, "cpu") + 1.0) / 2.0, None


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(
            self.max_action * action, -self.max_action, self.max_action
        )
        return action.cpu().data.numpy().flatten()

    def predict(
        self, state: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        return (self.act(state) + 1.0) / 2.0, None


def train(agent, config_json):
    steps_per_episode = int(config_json["env_config"]["steps_per_episode"])
    episodes_per_session = int(config_json["env_config"]["episodes_per_session"])
    num_steps = steps_per_episode * episodes_per_session
    parallel_env: bool = config_json["rl_config"]["parallel_env"]
    if parallel_env:
        num_steps *= 8
    agent_name: str = config_json["rl_config"]["agent"]
    try:
        if agent_name == "SAC":
            try:
                agent.load_replay_buffer("models/" + agent_name + ".ReplayBuffer")
                print(
                    f"The loaded_model has {agent.replay_buffer.size()} transitions in its buffer."
                )
                model = agent.learn(total_timesteps=num_steps, reset_num_timesteps=True)
            except:
                model = agent.learn(total_timesteps=num_steps)
        else:
            model = agent.learn(total_timesteps=num_steps)
    finally:
        if agent_name == "SAC":
            agent.save_replay_buffer(
                "models/"
                + agent_name
                + time.strftime("_%Y_%m_%d_%H_%M_%S", time.localtime())
                + ".ReplayBuffer.pkl"
            )
        # NOTE: adding timestamp to tell different models apart!
        agent.save(
            "models/"
            + agent_name
            + time.strftime("_%Y_%m_%d_%H_%M_%S", time.localtime())
        )

    # TODD Terminate the RL agent when the simulation ends.


def system_default_policy(env, config_json):
    steps_per_episode = int(config_json["env_config"]["steps_per_episode"])
    episodes_per_session = int(config_json["env_config"]["episodes_per_session"])
    num_steps = steps_per_episode * episodes_per_session

    truncated = True  # episode end
    terminated = False  # episode end and simulation end
    for step in range(num_steps + 10):
        # if simulation end, exit
        if terminated:
            print("simulation end")
            break

        # If the epsiode is up, then start another one
        if truncated:
            print("new episode")
            obs = env.reset()

        action = np.array([])  # no action from the rl agent

        # apply the action
        obs, reward, terminated, truncated, info = env.step(action)
        # print(obs)


def evaluate(
    model,
    env,
    num_steps,
    random_action_prob: float = 0.0,
    mean_state=None,
    stdev_state=None,
):
    done = True
    for _ in range(num_steps):
        if done:
            obs, info = env.reset()
        # at this point, obs should be a numpy array
        if type(mean_state) == torch.Tensor and type(stdev_state) == torch.Tensor:
            mean_state = mean_state.cpu().numpy()
            stdev_state = stdev_state.cpu().numpy()
        if type(mean_state) == np.ndarray and type(stdev_state) == np.ndarray:
            obs = obs.flatten()
            mean_state = mean_state.flatten()
            stdev_state = stdev_state.flatten()
            obs = (obs - mean_state) / stdev_state
        elif mean_state is not None or stdev_state is not None:
            raise Exception(
                f"mean_state type ({type(mean_state)}) and/or stdev_state type ({type(stdev_state)}) incompatible."
            )
        if random_action_prob > np.random.uniform(0, 1):
            print("TAKING RANDOM ACTION")
            action = random_action(obs)
        else:
            print("TAKING DETERMINISTIC ACTION")
            action, _ = model.predict(obs, deterministic=True)
            # FIXME: taking non-deterministic action to see if it helps
            # print("TAKING STOCHASTIC ACTION")
            # action, _ = model.predict(obs, deterministic=False)
        new_obs, reward, done, truncated, info = env.step(action)
        # FIXME: add in FastLSPI for training (clean this up!)
        if type(model) == PPO_LSPI:
            model.LSTDQ_update(obs, action, reward, new_obs)
        obs = new_obs


def main():
    args = arg_parser()

    # load config files
    config_json = load_config_file(args.env)
    config_json["env_config"]["env"] = args.env
    rl_config = config_json["rl_config"]

    seed: int = config_json["env_config"]["random_seed"]
    try:
        release_lock(seed)
    except Exception as e:
        print("WARNING: Exception occurred while trying to release config lock!")
        print(e)
        print("Continuing anyway...")

    if args.lte_rb != -1:
        config_json["env_config"]["LTE"]["resource_block_num"] = args.lte_rb

    if rl_config["agent"] == "":
        rl_config["agent"] = "system_default"

    rl_alg = rl_config["agent"]
    parallel_env: bool = rl_config["parallel_env"]

    alg_map = {
        "PPO": PPO,
        "PPO_LSPI": PPO_LSPI,
        "DDPG": DDPG,
        "SAC": SAC,
        "TD3": TD3,
        "A2C": A2C,
        "FastLSPI": FastLSPI,
        "throughput_argmax": argmax_policy,
        "throughput_argmin": argmin_policy,
        "delay_increment": delay_increment_policy,
        "random": random_policy,
        "random_discrete_increment": random_discrete_policy,
        "utility_discrete_increment": utility_increment_policy,
        "system_default": system_default_policy,
        "utility_logistic": utility_logistic_policy,
    }

    # Choose the agent
    agent_class = alg_map.get(rl_alg, None)
    client_id = args.client_id
    # Create the environment
    print("[" + args.env + "] environment selected.")
    # NOTE: can choose parallel env for training
    env = ParallelEnv() if parallel_env else NetworkGymEnv(client_id, config_json)
    # NOTE: disabling normalization
    normal_obs_env = env
    # normal_obs_env = NormalizeObservation(env)

    # It will check your custom environment and output additional warnings if needed
    # only use this function for debug,
    # check_env(env)

    heuristic_algorithms = [
        "system_default",
        "delay_increment",
        "random_discrete_increment",
        "utility_discrete_increment",
        "throughput_argmax",
        "throughput_argmin",
        "random",
        "utility_logistic",
    ]

    if rl_alg in heuristic_algorithms:
        agent_class(normal_obs_env, config_json)
        return

    train_flag = rl_config["train"]

    # Load the model if eval is True
    this_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(this_dir, "models", rl_alg)

    # Testing/Evaluation
    if not train_flag:
        print("Testing model...")
        mean_state, stdev_state = None, None
        if "checkpoint" in rl_alg:
            state_dict = torch.load(open(model_path + ".pt", "rb"), "cuda:0")
            if "checkpoint_BC" in rl_alg or "checkpoint_TD3_BC" in rl_alg:
                agent = Actor(56, 4, 1.0)
            elif "checkpoint_CQL" in rl_alg:
                agent = TanhGaussianPolicy(
                    56,
                    4,
                    1.0,
                    log_std_multiplier=1.0,
                    orthogonal_init=True,
                )
            elif "checkpoint_IQL" in rl_alg:
                agent = GaussianPolicy(56, 4, 1.0)
            elif "checkpoint_LB-SAC" in rl_alg or "checkpoint_LB_SAC" in rl_alg:
                agent = LB_Actor(56, 4, 256, False, 1.0)
            elif "checkpoint_EDAC" in rl_alg:
                agent = EDAC_Actor(56, 4, 256, 1.0)
            elif "checkpoint_SAC-N" in rl_alg or "checkpoint_SAC_N" in rl_alg:
                agent = SAC_N_Actor(56, 4, 256, 1.0)
            else:
                raise Exception("haven't implemented this actor type yet...")
            try:
                mean_state = state_dict["state_mean"]
                stdev_state = state_dict["state_std"]
            except:
                print("WARNING: no normalizers found for this agent...")
            agent.load_state_dict(state_dict["actor"])
        elif agent_class is None:
            print(
                f"WARNING: rl_alg ({rl_alg}) not found in alg_map. Trying personal mode..."
            )
            # FIXME MED: control for pickle.load() vs torch.load() -> move everything
            # to torch.load() / torch.save()
            try:
                agent = pickle.load(open(model_path + ".Actor", "rb"))
            except Exception as e:
                print("WARNING: exception occurred using pickle to load model.")
                print(e)
                print("Trying torch...")
                agent = torch.load(open(model_path + ".Actor", "rb"), "cuda:0")

            try:
                normalizers: tuple[torch.Tensor, torch.Tensor] = pickle.load(
                    open(model_path + ".Normalizers", "rb")
                )
                mean_state, stdev_state = normalizers
            except:
                print("No normalizers found for this agent...")
        elif rl_alg == "PPO_LSPI":
            # FIXME: num users hardcoded here!
            agent = agent_class(num_network_users=4)
        elif rl_alg == "FastLSPI":
            observation_dim = (
                normal_obs_env.observation_space.shape[0]
                * normal_obs_env.observation_space.shape[1]
            )
            num_actions = normal_obs_env.action_space.n
            agent = agent_class(observation_dim, num_actions, capped_buffer=False)
        else:
            agent = agent_class.load(model_path)

        steps_per_episode = int(config_json["env_config"]["steps_per_episode"])
        episodes_per_session = int(config_json["env_config"]["episodes_per_session"])
        num_steps = steps_per_episode * episodes_per_session
        # n_episodes = config_json['rl_config']['timesteps'] / 100
        random_action_prob: float = (
            rl_config["random_action_prob"]
            if "random_action_prob" in rl_config
            else 0.0
        )
        evaluate(
            agent,
            normal_obs_env,
            num_steps,
            random_action_prob,
            mean_state,
            stdev_state,
        )
    else:
        print("Training model...")
        if agent_class is None:
            raise Exception(f"ERROR: rl_alg ({rl_alg}) not found in alg_map!")
        elif (
            "load_model_for_training" in rl_config
            and rl_config["load_model_for_training"]
        ):
            print("LOADING MODEL FROM MODEL_PATH")
            agent = agent_class.load(
                model_path,
                normal_obs_env,
                verbose=1,
            )
        else:
            # init_std = (
            #     rl_config["starting_action_std"]
            #     if "starting_action_std" in rl_config
            #     else 1.0
            # )
            # policy_kwargs = dict(log_std_init=float(np.log(init_std)))
            if rl_alg == "PPO":
                # print("TRAINING PPO WITH MODIFIED STARTING STDEV")
                # print(policy_kwargs)
                n_steps = 2048
                if type(env) == ParallelEnv:
                    n_steps //= 8
                agent = agent_class(
                    rl_config["policy"],
                    normal_obs_env,
                    verbose=1,
                    # policy_kwargs=policy_kwargs,
                    n_steps=n_steps,
                )
            else:
                agent = agent_class(rl_config["policy"], normal_obs_env, verbose=1)
        print(agent.policy)

        train(agent, config_json)


def arg_parser():
    parser = argparse.ArgumentParser(description="Network Gym Client")
    parser.add_argument(
        "--env",
        type=str,
        choices=["nqos_split", "qos_steer", "network_slicing"],
        default="nqos_split",
        help="Select a environment to start Network Gym Client (nqos_split, qos_steer, network_slicing)",
    )
    parser.add_argument(
        "--client_id",
        type=int,
        required=False,
        default=0,
        help="Select client id to start simulation",
    )
    parser.add_argument(
        "--lte_rb",
        type=int,
        required=False,
        default=-1,
        help="Select number of LTE Resource Blocks",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
    time.sleep(5)

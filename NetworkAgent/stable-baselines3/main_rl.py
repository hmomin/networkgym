import argparse
import numpy as np
import os
import pickle
import random
import sys
import time

import torch

import sys

sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")

from network_gym_client import load_config_file
from network_gym_client import Env as NetworkGymEnv
from network_gym_client import PseudoParallelEnv
from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from gymnasium.wrappers import NormalizeObservation

from NetworkAgent.heuristic_policies import (
    argmax_policy,
    argmin_policy,
    random_action,
    random_policy,
    utility_argmax_policy,
    utility_logistic_policy,
)
from NetworkAgent.config_lock.client_utils import release_lock


def train(agent, config_json):

    steps_per_episode = int(config_json["env_config"]["steps_per_episode"])
    episodes_per_session = int(config_json["env_config"]["episodes_per_session"])
    num_steps = steps_per_episode * episodes_per_session

    try:
        model = agent.learn(total_timesteps=num_steps)
    finally:
        # NOTE: adding timestamp to tell different models apart!
        agent.save(
            "models/"
            + config_json["rl_config"]["agent"]
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


def evaluate(model, env, num_steps, random_action_prob: float = 0.0, mean_state=None, stdev_state=None):
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
        if random_action_prob > random.uniform(0, 1):
            print("TAKING RANDOM ACTION")
            action = random_action(obs)
        else:
            print("TAKING DETERMINISTIC ACTION")
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)


def main():

    args = arg_parser()

    # load config files
    config_json = load_config_file(args.env)
    config_json["env_config"]["env"] = args.env
    rl_config = config_json['rl_config']

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
        "DDPG": DDPG,
        "SAC": SAC,
        "TD3": TD3,
        "A2C": A2C,
        "ArgMax": argmax_policy,
        "ArgMin": argmin_policy,
        "Random": random_policy,
        "system_default": system_default_policy,
        "UtilityFull": utility_argmax_policy,
        "UtilityLogistic": utility_logistic_policy,
    }

    # Choose the agent
    agent_class = alg_map.get(rl_alg, None)
    client_id = args.client_id
    # Create the environment
    print("[" + args.env + "] environment selected.")
    # NOTE: can choose parallel env for training
    env = PseudoParallelEnv() if parallel_env else NetworkGymEnv(
    client_id, config_json
    )
    # NOTE: disabling normalization
    normal_obs_env = env
    # normal_obs_env = NormalizeObservation(env)

    # It will check your custom environment and output additional warnings if needed
    # only use this function for debug,
    # check_env(env)

    heuristic_algorithms = [
        "system_default",
        "ArgMax",
        "ArgMin",
        "Random",
        "UtilityFull",
        "UtilityLogistic",
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
        if agent_class is None:
            print(
                f"WARNING: rl_alg ({rl_alg}) not found in alg_map. Trying personal mode..."
            )
            agent = pickle.load(open(model_path + ".Actor", "rb"))
            try:
                normalizers: tuple[torch.Tensor, torch.Tensor] = pickle.load(
                    open(model_path + ".Normalizers", "rb")
                )
                mean_state, stdev_state = normalizers
            except:
                print("No normalizers found for this agent...")
        else:
            agent = agent_class.load(model_path)

        steps_per_episode = int(config_json["env_config"]["steps_per_episode"])
        episodes_per_session = int(config_json["env_config"]["episodes_per_session"])
        num_steps = steps_per_episode * episodes_per_session
        # n_episodes = config_json['rl_config']['timesteps'] / 100
        random_action_prob: float = rl_config["random_action_prob"] if "random_action_prob" in rl_config else 0.0
        evaluate(agent, normal_obs_env, num_steps, random_action_prob, mean_state, stdev_state)
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
            agent = agent_class(
                rl_config["policy"], normal_obs_env, verbose=1
            )
        print(agent.policy)

        train(agent, config_json)


def arg_parser():
    parser = argparse.ArgumentParser(description="Network Gym Client")
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=["nqos_split", "qos_steer", "network_slicing"],
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
    time.sleep(1)

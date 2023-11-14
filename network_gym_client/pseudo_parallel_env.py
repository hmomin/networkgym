import gymnasium as gym
from .env import Env, load_config_file
from stable_baselines3.common.vec_env import DummyVecEnv


def get_env_0() -> gym.Env:
    config_json = load_config_file("nqos_split")
    config_json["env_config"]["random_seed"] += 0
    return Env(0, config_json)


def get_env_1() -> gym.Env:
    config_json = load_config_file("nqos_split")
    config_json["env_config"]["random_seed"] += 1
    return Env(1, config_json)


def get_env_2() -> gym.Env:
    config_json = load_config_file("nqos_split")
    config_json["env_config"]["random_seed"] += 2
    return Env(2, config_json)


def get_env_3() -> gym.Env:
    config_json = load_config_file("nqos_split")
    config_json["env_config"]["random_seed"] += 3
    return Env(3, config_json)


def get_env_4() -> gym.Env:
    config_json = load_config_file("nqos_split")
    config_json["env_config"]["random_seed"] += 4
    return Env(4, config_json)


def get_env_5() -> gym.Env:
    config_json = load_config_file("nqos_split")
    config_json["env_config"]["random_seed"] += 5
    return Env(5, config_json)


def get_env_6() -> gym.Env:
    config_json = load_config_file("nqos_split")
    config_json["env_config"]["random_seed"] += 6
    return Env(6, config_json)


def get_env_7() -> gym.Env:
    config_json = load_config_file("nqos_split")
    config_json["env_config"]["random_seed"] += 7
    return Env(7, config_json)


class PseudoParallelEnv(DummyVecEnv):
    def __init__(self):
        env_functions = [
            get_env_0,
            get_env_1,
            get_env_2,
            get_env_3,
            get_env_4,
            get_env_5,
            get_env_6,
            get_env_7,
        ]
        super().__init__(env_functions)

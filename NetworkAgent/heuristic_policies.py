import numpy as np
import sys
from typing import Callable, Dict, List, Tuple
from network_gym_client.env import Env


def generic_policy(
    env: Env,
    config_json: Dict,
    action_chooser: Callable[[np.ndarray], List[float]],
) -> None:
    num_steps = get_num_steps_from_config(config_json)

    done = True
    for _ in range(num_steps):
        if done:
            obs = env.reset()
        if type(obs) == tuple:
            obs = obs[0]
        # at this point, obs should be a numpy array
        actions = action_chooser(obs)
        obs, reward, done, truncated, info = env.step(actions)


def get_num_steps_from_config(config_json: Dict) -> int:
    num_steps = 0

    # configure the num_steps based on the JSON file
    if (
        config_json["env_config"]["GMA"]["measurement_interval_ms"]
        + config_json["env_config"]["GMA"]["measurement_guard_interval_ms"]
        == config_json["env_config"]["Wi-Fi"]["measurement_interval_ms"]
        + config_json["env_config"]["Wi-Fi"]["measurement_guard_interval_ms"]
        == config_json["env_config"]["LTE"]["measurement_interval_ms"]
        + config_json["env_config"]["LTE"]["measurement_guard_interval_ms"]
    ):
        num_steps = int(
            config_json["env_config"]["steps_per_episode"]
            * config_json["env_config"]["episodes_per_session"]
        )
        return num_steps
    else:
        print(config_json["env_config"]["GMA"]["measurement_interval_ms"])
        print(config_json["env_config"]["GMA"]["measurement_guard_interval_ms"])
        print(config_json["env_config"]["Wi-Fi"]["measurement_interval_ms"])
        print(config_json["env_config"]["Wi-Fi"]["measurement_guard_interval_ms"])
        print(config_json["env_config"]["LTE"]["measurement_interval_ms"])
        print(config_json["env_config"]["LTE"]["measurement_guard_interval_ms"])
        sys.exit(
            "[Error!] The value of GMA, Wi-Fi, and LTE measurement_interval_ms + measurement_guard_interval_ms should be the same!"
        )


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def get_utilities(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lte_rate = obs[3, :]
    wifi_rate = obs[4, :]
    lte_owd = obs[5, :]
    wifi_owd = obs[6, :]
    lte_utilities = np.log(lte_rate) - np.log(lte_owd)
    wifi_utilities = np.log(wifi_rate) - np.log(wifi_owd)
    return (lte_utilities, wifi_utilities)


def arg_action(obs: np.ndarray, max_link: bool) -> List[float]:
    first_two_rows = obs[:2, :]
    argmax_values = np.argmax(first_two_rows, axis=0)
    ratio_vals = list(argmax_values)
    if not max_link:
        ratio_vals = [1 - val for val in ratio_vals]
    return ratio_vals


def utility_argmax_action(obs: np.ndarray) -> List[float]:
    lte_utilities, wifi_utilities = get_utilities(obs)
    actions = []
    for lte_utility, wifi_utility in zip(lte_utilities, wifi_utilities):
        if np.isnan(lte_utility):
            actions.append(0)
        elif np.isnan(wifi_utility):
            actions.append(1)
        else:
            actions.append(int(wifi_utility > lte_utility))
    return actions


def utility_logistic_action(obs: np.ndarray) -> List[float]:
    lte_utilities, wifi_utilities = get_utilities(obs)
    actions = []
    for lte_utility, wifi_utility in zip(lte_utilities, wifi_utilities):
        if np.isnan(lte_utility):
            actions.append(0)
        elif np.isnan(wifi_utility):
            actions.append(1)
        else:
            actions.append(sigmoid(wifi_utility - lte_utility))
    return actions


def argmax_action(obs: np.ndarray) -> List[float]:
    return arg_action(obs, max_link=True)


def argmin_action(obs: np.ndarray) -> List[float]:
    return arg_action(obs, max_link=False)


def random_action(obs: np.ndarray) -> List[float]:
    num_users = obs.shape[1]
    return list(np.random.random((num_users)))


def argmax_policy(env: Env, config_json: Dict) -> None:
    generic_policy(env, config_json, argmax_action)


def argmin_policy(env: Env, config_json: Dict) -> None:
    generic_policy(env, config_json, argmin_action)


def random_policy(env: Env, config_json: Dict) -> None:
    generic_policy(env, config_json, random_action)


def utility_argmax_policy(env: Env, config_json: Dict) -> None:
    generic_policy(env, config_json, utility_argmax_action)


def utility_logistic_policy(env: Env, config_json: Dict) -> None:
    generic_policy(env, config_json, utility_logistic_action)
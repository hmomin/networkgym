import numpy as np
import sys
from time import sleep
from typing import Callable
from random import randint
from discrete_action_util import convert_user_increment_to_discrete_increment_action
from network_gym_client.env import Env


def generic_policy(
    env: Env,
    config_json: dict,
    action_chooser: Callable[[np.ndarray], list[float]],
) -> None:
    num_steps = get_num_steps_from_config(config_json)

    done = True
    for _ in range(num_steps):
        if done:
            obs, info = env.reset()
        # at this point, obs should be a numpy array
        actions = action_chooser(obs)
        obs, reward, done, truncated, info = env.step(actions)


def get_num_steps_from_config(config_json: dict) -> int:
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


def arg_action(obs: np.ndarray, max_link: bool) -> list[float]:
    first_two_rows = obs[:2, :]
    argmax_values = np.argmax(first_two_rows, axis=0)
    ratio_vals = list(float(val) for val in argmax_values)
    if not max_link:
        ratio_vals = [1.0 - val for val in ratio_vals]
    return ratio_vals


def utility_argmax_action(obs: np.ndarray) -> list[float]:
    lte_utilities, wifi_utilities = get_utilities(obs)
    actions = []
    for lte_utility, wifi_utility in zip(lte_utilities, wifi_utilities):
        if np.isnan(lte_utility):
            actions.append(0.0)
        elif np.isnan(wifi_utility):
            actions.append(1.0)
        else:
            actions.append(float(int(wifi_utility > lte_utility)))
    return actions


def utility_logistic_action(obs: np.ndarray) -> list[float]:
    lte_utilities, wifi_utilities = get_utilities(obs)
    actions = []
    for lte_utility, wifi_utility in zip(lte_utilities, wifi_utilities):
        if np.isnan(lte_utility):
            actions.append(0.0)
        elif np.isnan(wifi_utility):
            actions.append(1.0)
        else:
            utility_difference = wifi_utility - lte_utility
            # NOTE: (+inf) - (+inf) = nan, and same for (-inf)
            new_action = (
                0.5
                if np.isnan(utility_difference)
                else float(sigmoid(utility_difference))
            )
            actions.append(new_action)
    return actions


def argmax_action(obs: np.ndarray) -> list[float]:
    return arg_action(obs, max_link=True)


def argmin_action(obs: np.ndarray) -> list[float]:
    return arg_action(obs, max_link=False)


def random_action(obs: np.ndarray) -> list[float]:
    num_users = obs.shape[1]
    return list(np.random.random((num_users)))


def utility_increment_policy(env: Env, config_json: dict) -> None:
    generic_policy(env, config_json, utility_increment_action)


def utility_increment_action(obs: np.ndarray) -> int:
    lte_utilities, wifi_utilities = get_utilities(obs)
    actions: list[int] = []
    for lte_utility, wifi_utility in zip(lte_utilities, wifi_utilities):
        update_to_split_ratio = utility_increment(wifi_utility, lte_utility)
        actions.append(update_to_split_ratio)
    discrete_action = convert_user_increment_to_discrete_increment_action(actions)
    return discrete_action   


def utility_increment(wifi_utility: float, lte_utility: float) -> int:
    if np.isnan(wifi_utility) and np.isnan(lte_utility):
        return 0
    elif np.isnan(wifi_utility):
        return +1
    elif np.isnan(lte_utility):
        return -1
    elif wifi_utility > lte_utility:
        return +1
    elif lte_utility < wifi_utility:
        return -1
    else:
        return 0


def delay_increment_policy(env: Env, config_json: dict) -> None:
    generic_policy(env, config_json, delay_increment_action)


def delay_increment_action(obs: np.ndarray) -> int:
    lte_owds = obs[5, :]
    wifi_owds = obs[6, :]
    actions: list[int] = []
    for lte_owd, wifi_owd in zip(lte_owds, wifi_owds):
        update_to_split_ratio = delay_increment(wifi_owd, lte_owd)
        actions.append(update_to_split_ratio)
    discrete_action = convert_user_increment_to_discrete_increment_action(actions)
    return discrete_action


def delay_increment(wifi_owd: float, lte_owd: float) -> int:
    if wifi_owd > lte_owd:
        return -1
    elif wifi_owd < lte_owd:
        return +1
    else:
        return 0


def random_discrete_increment_action(obs: np.ndarray) -> int:
    num_users = obs.shape[1]
    random_action = randint(0, 3 ** num_users - 1)
    return random_action


def random_discrete_policy(env: Env, config_json: dict) -> None:
    generic_policy(env, config_json, random_discrete_increment_action)


def argmax_policy(env: Env, config_json: dict) -> None:
    generic_policy(env, config_json, argmax_action)


def argmin_policy(env: Env, config_json: dict) -> None:
    generic_policy(env, config_json, argmin_action)


def random_policy(env: Env, config_json: dict) -> None:
    generic_policy(env, config_json, random_action)


def utility_argmax_policy(env: Env, config_json: dict) -> None:
    generic_policy(env, config_json, utility_argmax_action)


def utility_logistic_policy(env: Env, config_json: dict) -> None:
    generic_policy(env, config_json, utility_logistic_action)

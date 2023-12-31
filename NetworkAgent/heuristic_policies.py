import numpy as np
import sys
from time import sleep
from typing import Callable
from network_gym_client.env import Env

# NOTE: num users is hardcoded here - there's a check for it later...
CURRENT_SPLIT_RATIO = [32] * 4
LAST_WIFI_OWDS = [0] * 4
LAST_LTE_OWDS = [0] * 4
WIFI_INDEX_CHANGE_ALPHAS = [0] * 4
STEP_ALPHA_THRESHOLD = 4
TOLERANCE_DELAY_BOUND = 5


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
            actions.append(float(sigmoid(wifi_utility - lte_utility)))
    return actions


def system_default_action(obs: np.ndarray) -> list[float]:
    # use obs and whatever the current split ratio is to update new split ratio
    num_users = obs.shape[1]
    lte_owds = obs[5, :] * 100
    wifi_owds = obs[6, :] * 100
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("UE\tWIFI\tLTE\tUPDATE")
    for user, lte_owd, wifi_owd in zip(range(num_users), lte_owds, wifi_owds):
        update_to_split_ratio = delay_based_algorithm(user, wifi_owd, lte_owd)
        print(
            f"{user}\t{int(round(wifi_owd))}\t{int(round(lte_owd))}\t{update_to_split_ratio}"
        )
        CURRENT_SPLIT_RATIO[user] += update_to_split_ratio
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    return [x / 32.0 for x in CURRENT_SPLIT_RATIO]


def delay_based_algorithm(user: int, wifi_owd: int, lte_owd: int) -> int:
    last_wifi_index = CURRENT_SPLIT_RATIO[user]
    wifi_index_change_alpha = WIFI_INDEX_CHANGE_ALPHAS[user]
    last_decision_wifi_owd = LAST_WIFI_OWDS[user]
    last_decision_lte_owd = LAST_LTE_OWDS[user]
    if TOLERANCE_DELAY_BOUND < wifi_owd - lte_owd:
        if wifi_index_change_alpha >= 0:
            wifi_index_change_alpha = -1
        if wifi_owd >= last_decision_wifi_owd and last_wifi_index > 0:
            last_wifi_index += min(-1, wifi_index_change_alpha + STEP_ALPHA_THRESHOLD)
            WIFI_INDEX_CHANGE_ALPHAS[user] -= 1
            last_wifi_index = max(0, last_wifi_index)
    elif TOLERANCE_DELAY_BOUND < lte_owd - wifi_owd:
        if wifi_index_change_alpha <= 0:
            wifi_index_change_alpha = 1
        if lte_owd >= last_decision_lte_owd and last_wifi_index < 32:
            last_wifi_index += max(1, wifi_index_change_alpha - STEP_ALPHA_THRESHOLD)
            WIFI_INDEX_CHANGE_ALPHAS[user] += 1
            last_wifi_index = min(32, last_wifi_index)
    else:
        WIFI_INDEX_CHANGE_ALPHAS[user] = 0
    LAST_WIFI_OWDS[user] = wifi_owd
    LAST_LTE_OWDS[user] = lte_owd

    return last_wifi_index - CURRENT_SPLIT_RATIO[user]


def argmax_action(obs: np.ndarray) -> list[float]:
    return arg_action(obs, max_link=True)


def argmin_action(obs: np.ndarray) -> list[float]:
    return arg_action(obs, max_link=False)


def random_action(obs: np.ndarray) -> list[float]:
    num_users = obs.shape[1]
    return list(np.random.random((num_users)))


def system_default_proxy_policy(env: Env, config_json: dict) -> None:
    generic_policy(env, config_json, system_default_action)


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

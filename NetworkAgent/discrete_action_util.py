import numpy as np


def convert_discrete_action_to_continuous(
    agent_action: np.int64, previous_split_ratio: list[float]
) -> list[float]:
    num_users: int = len(previous_split_ratio)
    ratio_increment = get_ratio_increment_from_discrete_action(agent_action, num_users)
    split_ratio_change = [x / 32.0 for x in ratio_increment]
    new_split_ratio = [x + y for x, y in zip(previous_split_ratio, split_ratio_change)]
    new_split_ratio = [np.clip(x, 0, 1) for x in new_split_ratio]
    print("NEW SPLIT RATIO")
    print(new_split_ratio)
    return new_split_ratio


def get_ratio_increment_from_discrete_action(
    discrete_action: np.int64, num_users: int
) -> list[int]:
    user_specific_increments: list[int] = []
    num_possible_actions = 3 ** num_users
    running_divisor = discrete_action
    running_dividend = num_possible_actions
    for _ in range(num_users):
        running_dividend //= 3
        threeway_action = (running_divisor // running_dividend) - 1
        running_divisor = running_divisor % running_dividend
        user_specific_increments.append(threeway_action)
    return user_specific_increments

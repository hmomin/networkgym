import numpy as np
from time import sleep


def convert_discrete_increment_action_to_continuous(
    agent_action: np.int64 | int, previous_split_ratio: list[float]
) -> list[float]:
    num_users = len(previous_split_ratio)
    user_specific_actions = get_user_discretized_actions(agent_action, num_users, 3)
    split_ratio_change = [(x - 1) / 32.0 for x in user_specific_actions]
    new_split_ratio = [x + y for x, y in zip(previous_split_ratio, split_ratio_change)]
    new_split_ratio = [np.clip(x, 0, 1) for x in new_split_ratio]
    print("NEW SPLIT RATIO")
    print(new_split_ratio)
    return new_split_ratio


def convert_discrete_ratio_action_to_continuous(
    agent_action: np.int64 | int, num_users: int
) -> list[float]:
    user_specific_actions = get_user_discretized_actions(agent_action, num_users, 9)
    new_split_ratio = [x / 8.0 for x in user_specific_actions]
    print("NEW SPLIT RATIO")
    print(new_split_ratio)
    return new_split_ratio


def get_user_discretized_actions(
    discrete_action: np.int64 | int, num_users: int, num_actions_per_user: int
) -> list[int]:
    user_specific_actions: list[int] = []
    num_possible_actions = num_actions_per_user**num_users
    running_divisor = discrete_action
    running_dividend = num_possible_actions
    for _ in range(num_users):
        running_dividend //= num_actions_per_user
        user_action = running_divisor // running_dividend
        running_divisor = running_divisor % running_dividend
        user_specific_actions.append(user_action)
    return user_specific_actions


# NOTE: just some tests below...
def main() -> None:
    num_users = 4
    # sleep(1)
    # for discrete_action in range(3 ** num_users):
    #     user_specific_action = get_user_discretized_actions(
    #         discrete_action,
    #         num_users,
    #         3
    #     )
    #     print(f"{discrete_action}: {user_specific_action}")
    print("TESTING user discretized actions...")
    discrete_action = 5537
    user_specific_actions = get_user_discretized_actions(discrete_action, num_users, 9)
    assert user_specific_actions == [7, 5, 3, 2]
    print("TESTING discrete ratio action")
    discrete_ratio_actions = convert_discrete_ratio_action_to_continuous(
        discrete_action, num_users
    )
    assert discrete_ratio_actions == [0.875, 0.625, 0.375, 0.250]
    print("TESTING discrete increment actions")
    discrete_action = 38
    user_specific_actions = get_user_discretized_actions(discrete_action, num_users, 3)
    assert user_specific_actions == [1, 1, 0, 2]
    previous_split_ratio = [0.5 for _ in range(num_users)]
    discrete_increment_action = convert_discrete_increment_action_to_continuous(
        discrete_action, previous_split_ratio
    )
    assert discrete_increment_action == [0.50000, 0.50000, 0.46875, 0.53125]
    previous_split_ratio = [1.0 for _ in range(num_users)]
    discrete_increment_action = convert_discrete_increment_action_to_continuous(
        discrete_action, previous_split_ratio
    )
    assert discrete_increment_action == [1.00000, 1.00000, 0.96875, 1.00000]


if __name__ == "__main__":
    main()

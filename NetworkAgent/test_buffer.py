from random import randint
from buffer import Buffer

buffer_name = "system_default_utility_seed_08"


def main() -> None:
    buffer = Buffer(buffer_name)
    idx = randint(0, len(buffer.rewards) - 1)
    state = buffer.states[idx]
    action = buffer.actions[idx]
    reward = buffer.rewards[idx]
    next_state = buffer.next_states[idx]
    print("STATE")
    print(type(state))
    print(state)
    print("ACTION")
    print(type(action))
    print(action)
    print("REWARD")
    print(type(reward))
    print(reward)
    print("NEXT STATE")
    print(type(next_state))
    print(next_state)


if __name__ == "__main__":
    main()

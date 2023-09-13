from random import randint
from buffer import Buffer

buffer_name = "system_default_utility_seed_00"


def main() -> None:
    buffer = Buffer(buffer_name)
    idx = randint(0, len(buffer.container) - 1)
    state = buffer.container[0][idx]
    action = buffer.container[1][idx]
    reward = buffer.container[2][idx]
    next_state = buffer.container[3][idx]
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

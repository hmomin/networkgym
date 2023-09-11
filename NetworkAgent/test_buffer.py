from random import randint
from buffer import Buffer

buffer_name = "system_default_2023_09_08_23_13_27"


def main() -> None:
    buffer = Buffer(buffer_name)
    idx = randint(0, len(buffer.container) - 1)
    state, action, reward, next_state = buffer.container[idx]
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

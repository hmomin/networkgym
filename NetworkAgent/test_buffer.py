from random import randint
from buffer import Buffer

buffer_name = "random_discrete_increment_utility_seed_51"


def main() -> None:
    buffer = Buffer(buffer_name)
    buffer_1 = Buffer("b1.pickle")
    buffer_2 = Buffer("b2.pickle")
    num_states = buffer.states.shape[0]
    b1_num_states = num_states // 2
    buffer_1.states = buffer.states[:b1_num_states, :]
    buffer_2.states = buffer.states[b1_num_states:, :]
    assert buffer_1.states.shape == buffer_2.states.shape
    for idx in range(48, 53):
        assert buffer_1.states[756, idx] == buffer_2.states[756, idx]
    buffer_1.actions = buffer.actions[:b1_num_states, :]
    buffer_2.actions = buffer.actions[b1_num_states:, :]
    assert buffer_1.actions.shape == buffer_2.actions.shape
    buffer_1.rewards = buffer.rewards[:b1_num_states]
    buffer_2.rewards = buffer.rewards[b1_num_states:]
    assert buffer_1.rewards.shape == buffer_2.rewards.shape
    buffer_1.next_states = buffer.next_states[:b1_num_states, :]
    buffer_2.next_states = buffer.next_states[b1_num_states:, :]
    assert buffer_1.next_states.shape == buffer_2.next_states.shape
    for idx in range(48, 53):
        assert buffer_1.next_states[1064, idx] == buffer_2.next_states[1064, idx]

    buffer_1.write_to_disk()
    buffer_2.write_to_disk()

    # idx = randint(0, len(buffer.rewards) - 1)
    # state = buffer.states[idx]
    # action = buffer.actions[idx]
    # reward = buffer.rewards[idx]
    # next_state = buffer.next_states[idx]
    # print("STATE")
    # print(type(state))
    # print(state)
    # print("ACTION")
    # print(type(action))
    # print(action)
    # print("REWARD")
    # print(type(reward))
    # print(reward)
    # print("NEXT STATE")
    # print(type(next_state))
    # print(next_state)


if __name__ == "__main__":
    main()

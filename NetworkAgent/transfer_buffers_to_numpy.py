import os
from buffer import Buffer


algo_name = "system_default"


def get_buffers(algo_name: str) -> list[Buffer]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    buffers_dir = os.path.join(script_dir, "buffers", algo_name)
    buffer_locs = os.listdir(buffers_dir)
    buffers: list[Buffer] = []
    for idx, buffer_loc in enumerate(buffer_locs):
        buffer_name = os.path.join(algo_name, buffer_loc)
        new_buffer = Buffer(buffer_name)
        buffers.append(new_buffer)
        print(f"Buffer {idx} loaded.")
    return buffers


def main() -> None:
    buffers = get_buffers(algo_name)
    for buffer in buffers:
        buffer.states = buffer.numpy_states
        buffer.actions = buffer.numpy_actions
        buffer.rewards = buffer.numpy_rewards
        buffer.next_states = buffer.numpy_next_states
        buffer.write_to_disk()
        # raise Exception("STOP")


if __name__ == "__main__":
    main()

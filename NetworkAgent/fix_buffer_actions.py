import numpy as np
import os
from time import sleep
from buffer import Buffer, safe_stack
from discrete_action_util import convert_user_increment_to_discrete_increment_action


def get_buffer_names() -> list[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    buffer_dir = os.path.join(script_dir, "buffers")
    all_contents = os.listdir(buffer_dir)
    buffer_names = [filename for filename in all_contents if ".pickle" in filename]
    return buffer_names


def get_discrete_increment_actions(buffer: Buffer) -> np.ndarray:
    num_actions = buffer.actions.shape[0]
    previous_split_ratio = np.ones((4,), dtype=np.float64)
    new_actions = np.array([])
    for idx in range(num_actions):
        current_split_ratio = buffer.actions[idx, :]
        current_increment = 32 * (current_split_ratio - previous_split_ratio)
        current_increment = list(np.array(current_increment, dtype=np.int64))
        discrete_increment_action = convert_user_increment_to_discrete_increment_action(
            current_increment
        )
        new_actions = safe_stack(new_actions, np.array([discrete_increment_action]))
        previous_split_ratio = current_split_ratio
    return new_actions


def main() -> None:
    buffer_names = get_buffer_names()
    for buffer_name in buffer_names:
        buffer = Buffer(buffer_name)
        new_actions = get_discrete_increment_actions(buffer)
        # hist = np.histogram(new_actions, 81, (-0.5, 80.5))
        # print(hist)
        # sleep(1)
        buffer.actions = new_actions
        buffer.write_to_disk()


if __name__ == "__main__":
    main()

import numpy as np
import os
import pickle


class Buffer:
    def __init__(self, name: str) -> None:
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        self.filename = os.path.join(this_file_dir, "buffers", name + ".pickle")
        if os.path.exists(self.filename):
            with open(self.filename, "rb") as f:
                self.container: tuple[
                    list[np.ndarray],
                    list[list[float]],
                    list[np.float64],
                    list[np.ndarray],
                ] = pickle.load(f)
                (
                    self.states,
                    self.actions,
                    self.rewards,
                    self.next_states,
                ) = self.container
                self.construct_finalized_buffer()
        else:
            self.container: tuple[
                list[np.ndarray], list[list[float]], list[np.float64], list[np.ndarray]
            ] = ([], [], [], [])
            self.states, self.actions, self.rewards, self.next_states = self.container

    def flatten_states(self) -> None:
        for state_list in (self.states, self.next_states):
            for idx, element in enumerate(state_list):
                state_list[idx] = element.flatten()

    def construct_finalized_buffer(self) -> None:
        self.flatten_states()
        self.numpy_states = np.vstack(self.states)
        self.numpy_actions = np.array(self.actions)
        self.numpy_rewards = np.array(self.rewards)
        self.numpy_next_states = np.vstack(self.next_states)
        self.numpy_size = self.numpy_states.shape[0]

    def store(
        self,
        state: np.ndarray,
        action: list[float],
        reward: np.float64,
        next_state: np.ndarray,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        if len(self.states) % 100 == 0:
            self.write_to_disk()

    def write_to_disk(self) -> None:
        with open(self.filename, "wb") as f:
            pickle.dump(self.container, f)

    def get_mini_batch(
        self,
        size: int,
    ) -> dict[str, np.ndarray]:
        buffer_size = self.numpy_size
        indices = np.random.choice(buffer_size, size, replace=False)
        return {
            "states": self.numpy_states[indices],
            "actions": self.numpy_actions[indices],
            "rewards": self.numpy_rewards[indices],
            "next_states": self.numpy_next_states[indices],
        }

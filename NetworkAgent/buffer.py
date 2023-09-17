import numpy as np
import os
import pickle
from pprint import pprint
from typing import Any


class Buffer:
    def __init__(self, name: str) -> None:
        if ".pickle" not in name:
            name += ".pickle"
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        buffer_dir = os.path.join(this_file_dir, "buffers")
        self.filename = os.path.join(buffer_dir, name)
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
        # FIXME: can we do this on store? it would make things a lot simpler...
        self.numpy_states = np.vstack(self.states)
        self.numpy_actions = np.array(self.actions)
        self.numpy_rewards = np.array(self.rewards)
        self.numpy_next_states = np.vstack(self.next_states)

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
        buffer_size = self.numpy_states.shape[0]
        indices = np.random.choice(buffer_size, size, replace=False)
        # NOTE: an environment never actually terminates in the way that the MDP
        # framework expects...
        dones = np.zeros_like(self.numpy_rewards[indices])
        return {
            "states": self.numpy_states[indices],
            "actions": self.numpy_actions[indices],
            "rewards": self.numpy_rewards[indices],
            "next_states": self.numpy_next_states[indices],
            "dones": dones,
        }


class CombinedBuffer(Buffer):
    def __init__(self, buffers: list[Buffer]) -> None:
        for buffer in buffers:
            self.fill_from_buffer(buffer)

    def fill_from_buffer(self, buffer: Buffer) -> None:
        # NOTE: buffer is expected to contain numpy variables
        # FIXME: this can be made more efficient using eval()?
        if hasattr(self, "numpy_states"):
            self.numpy_states = np.concatenate(
                (self.numpy_states, buffer.numpy_states), axis=0
            )
        else:
            self.numpy_states = buffer.numpy_states
        if hasattr(self, "numpy_actions"):
            self.numpy_actions = np.concatenate(
                (self.numpy_actions, buffer.numpy_actions), axis=0
            )
        else:
            self.numpy_actions = buffer.numpy_actions
        if hasattr(self, "numpy_rewards"):
            self.numpy_rewards = np.concatenate(
                (self.numpy_rewards, buffer.numpy_rewards), axis=0
            )
        else:
            self.numpy_rewards = buffer.numpy_rewards
        if hasattr(self, "numpy_next_states"):
            self.numpy_next_states = np.concatenate(
                (self.numpy_next_states, buffer.numpy_next_states), axis=0
            )
        else:
            self.numpy_next_states = buffer.numpy_next_states

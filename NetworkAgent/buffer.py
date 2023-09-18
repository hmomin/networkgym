import numpy as np
import os
import pickle
import torch as T
from tqdm import tqdm


def safe_stack(array_1: np.ndarray, array_2: np.ndarray) -> np.ndarray:
    return np.vstack((array_1, array_2)) if array_1.size else array_2


def safe_concat(array_1: np.ndarray, array_2: np.ndarray) -> np.ndarray:
    concatenated_array = (
        np.concatenate((array_1, array_2), axis=0) if array_1.size else array_2
    )
    return concatenated_array


class Buffer:
    def __init__(self, name: str) -> None:
        if ".pickle" not in name:
            name += ".pickle"
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        buffer_dir = os.path.join(this_file_dir, "buffers")
        self.filename = os.path.join(buffer_dir, name)
        if os.path.exists(self.filename):
            with open(self.filename, "rb") as f:
                # NOTE: pickle will save each element as a list of 1-D arrays
                container: tuple[
                    list[np.ndarray],
                    list[np.ndarray],
                    list[np.ndarray],
                    list[np.ndarray],
                ] = pickle.load(f)
                self.states = np.array(container[0])
                self.actions = np.array(container[1])
                self.rewards = np.array(container[2])
                self.next_states = np.array(container[3])
        else:
            self.states = np.array([])
            self.actions = np.array([])
            self.rewards = np.array([])
            self.next_states = np.array([])

    def store(
        self,
        state: np.ndarray,
        action: list[float],
        reward: np.float64,
        next_state: np.ndarray,
    ) -> None:
        state = state.flatten()
        next_state = next_state.flatten()
        np_action = np.array(action)
        self.states = safe_stack(self.states, state)
        self.actions = safe_stack(self.actions, np_action)
        self.rewards = np.concatenate((self.rewards, np.array([reward])))
        self.next_states = safe_stack(self.next_states, next_state)
        self.write_to_disk()

    def write_to_disk(self) -> None:
        print(f"states:      {self.states.shape}")
        print(f"actions:     {self.actions.shape}")
        print(f"rewards:     {self.rewards.shape}")
        print(f"next_states: {self.next_states.shape}")
        with open(self.filename, "wb") as f:
            pickle.dump((self.states, self.actions, self.rewards, self.next_states), f)

    def get_mini_batch(
        self,
        size: int,
    ) -> dict[str, np.ndarray]:
        buffer_size = self.states.shape[0]
        indices = np.random.choice(buffer_size, size, replace=False)
        # NOTE: an environment never actually terminates in the way that the MDP
        # framework expects...
        dones = np.zeros_like(self.rewards[indices])
        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": dones,
        }


class CombinedBuffer(Buffer):
    def __init__(self, buffers: list[Buffer]) -> None:
        super().__init__("this_buffer_doesn't_exist")
        print("Combining buffers...")
        for buffer in tqdm(buffers):
            self.fill_from_buffer(buffer)
        # NOTE: it's possible that calling requires_grad on the full tensor might
        # be too computationally expensive and unnecessary. Verify this...
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.tensor_states = T.tensor(self.states, device=self.device)
        self.tensor_actions = T.tensor(self.actions, device=self.device)
        self.tensor_rewards = T.tensor(self.rewards, device=self.device)
        self.tensor_next_states = T.tensor(self.next_states, device=self.device)

    def fill_from_buffer(self, buffer: Buffer) -> None:
        self.states = safe_concat(self.states, buffer.states)
        self.actions = safe_concat(self.actions, buffer.actions)
        self.rewards = safe_concat(self.rewards, buffer.rewards)
        self.next_states = safe_concat(self.next_states, buffer.next_states)

    def get_mini_batch(
        self,
        size: int,
    ) -> dict[str, T.Tensor]:
        buffer_size = self.states.shape[0]
        indices = T.randint(0, buffer_size, (size,), device=self.device)
        # NOTE: an environment never actually terminates in the way that the MDP
        # framework expects...
        dones = T.zeros_like(self.tensor_rewards[indices], device=self.device)
        return {
            "states": self.tensor_states[indices, :],
            "actions": self.tensor_actions[indices, :],
            "rewards": self.tensor_rewards[indices],
            "next_states": self.tensor_next_states[indices, :],
            "dones": dones,
        }

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
                    list[np.ndarray],
                    list[np.float64],
                    list[np.ndarray],
                ] = pickle.load(f)
                self.construct_finalized_buffer()
        else:
            self.container: tuple[
                list[np.ndarray], list[np.ndarray], list[np.float64], list[np.ndarray]
            ] = ([], [], [], [])
    
    def construct_finalized_buffer(self) -> None:
        # FIXME: fill this out!
        pass

    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.float64,
        next_state: np.ndarray,
    ) -> None:
        for idx, item in enumerate((state, action, reward, next_state)):
            self.container[idx].append(item)
        if len(self.container[0]) % 100 == 0:
            self.write_to_disk()

    def write_to_disk(self) -> None:
        with open(self.filename, "wb") as f:
            pickle.dump(self.container, f)

    def get_mini_batch(
        self,
        size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.float64, np.ndarray]:
        current_size = len(self.container[0])
        size = min(size, current_size)
        indices = np.random.choice(current_size, size, replace=False)
        # FIXME: need to convert the lists to numpy arrays to do efficient
        # mini-batch index selection
        # return {
        #     "states": self.states[indices],
        #     "actions": self.actions[indices],
        #     "rewards": self.rewards[indices],
        #     "nextStates": self.nextStates[indices],
        # }

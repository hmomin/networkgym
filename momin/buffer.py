import numpy as np
import os
import pickle
from typing import List, Tuple


class Buffer:
    def __init__(self, name: str) -> None:
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        self.filename = os.path.join(this_file_dir, "buffers", name + ".pickle")
        if os.path.exists(self.filename):
            with open(self.filename, "rb") as f:
                self.container: List[
                    Tuple[np.ndarray, np.ndarray, np.float64, np.ndarray]
                ] = pickle.load(f)
        else:
            self.container: List[
                Tuple[np.ndarray, np.ndarray, np.float64, np.ndarray]
            ] = []

    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.float64,
        next_state: np.ndarray,
    ) -> None:
        self.container.append((state, action, reward, next_state))
        if len(self.container) % 100 == 0:
            self.write_to_disk()

    def write_to_disk(self) -> None:
        with open(self.filename, "wb") as f:
            pickle.dump(self.container, f)

    def get_mini_batch(
        self,
        mini_batch_size: int,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.float64, np.ndarray]]:
        # FIXME: fill this out!
        pass

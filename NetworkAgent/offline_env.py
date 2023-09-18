import numpy as np
import os
from buffer import Buffer, CombinedBuffer

# FIXME: use tqdm if possible
# from tqdm import tqdm

# NOTE: this script facilitates an offline learning environment for offline RL


class OfflineEnv:
    def __init__(self, algo_name: str = "system_default") -> None:
        self.algo_name = algo_name
        buffers = self.get_buffers()
        self.buffer = CombinedBuffer(buffers)

    def get_buffers(self) -> list[Buffer]:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        buffers_dir = os.path.join(script_dir, "buffers", self.algo_name)
        buffer_locs = os.listdir(buffers_dir)
        buffers: list[Buffer] = []
        for idx, buffer_loc in enumerate(buffer_locs):
            buffer_name = os.path.join(self.algo_name, buffer_loc)
            new_buffer = Buffer(buffer_name)
            buffers.append(new_buffer)
            print(f"Buffer {idx} loaded.")
        return buffers

    def get_mini_batch(self, mini_batch_size: int = 100) -> dict[str, np.ndarray]:
        mini_batch = self.buffer.get_mini_batch(mini_batch_size)
        return mini_batch
import numpy as np
import os
import pickle
from time import sleep

# reconstruct each buffer
# store the new buffer in a reconstructed zone


def get_buffer_filenames() -> list[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    buffer_dir = os.path.join(script_dir, "buffers")
    buffer_dir_contents = os.listdir(buffer_dir)
    buffer_filenames: list[str] = []
    for buffer_dir_content in buffer_dir_contents:
        content_path = os.path.join(buffer_dir, buffer_dir_content)
        # NOTE: assuming it's a buffer pickle file if it's not a folder
        if os.path.isfile(content_path):
            buffer_filenames.append(content_path)
    return buffer_filenames


def open_buffer(
    file_path: str,
) -> list[tuple[np.ndarray, np.ndarray, np.float64, np.ndarray]]:
    with open(file_path, "rb") as f:
        buffer: list[
            tuple[np.ndarray, np.ndarray, np.float64, np.ndarray]
        ] = pickle.load(f)
    return buffer


def reconstruct_buffer(
    buffer: list[tuple[np.ndarray, np.ndarray, np.float64, np.ndarray]]
) -> None:
    for tupe in buffer:
        state, action, reward, next_state = tupe
        print(state)
        print(action)
        sleep(1)
        # FIXME: fill out the rest of this!


def main() -> None:
    buffer_filenames = get_buffer_filenames()
    for buffer_filename in buffer_filenames:
        buffer = open_buffer(buffer_filename)
        reconstructed_buffer = reconstruct_buffer(buffer)


if __name__ == "__main__":
    main()

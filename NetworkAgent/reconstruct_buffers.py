# NOTE: this script might not actually be necessary - think about it...

import numpy as np
import os
import pickle
from time import sleep


def get_buffer_filenames() -> list[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # NOTE: adding an extra directory for testing
    buffer_dir = os.path.join(script_dir, "buffers", "test")
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
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.float64], list[np.ndarray],]:
    with open(file_path, "rb") as f:
        buffer: tuple[
            list[np.ndarray],
            list[np.ndarray],
            list[np.float64],
            list[np.ndarray],
        ] = pickle.load(f)
    return buffer


def reconstruct_buffer(
    buffer: tuple[
        list[np.ndarray],
        list[np.ndarray],
        list[np.float64],
        list[np.ndarray],
    ]
) -> None:
    print(buffer[1])


def main() -> None:
    buffer_filenames = get_buffer_filenames()
    for buffer_filename in buffer_filenames:
        buffer = open_buffer(buffer_filename)
        reconstructed_buffer = reconstruct_buffer(buffer)


if __name__ == "__main__":
    main()

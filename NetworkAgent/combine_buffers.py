# NOTE: maybe don't need this script...

import os
from buffer import Buffer

subfolder_name = "system_default"


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_buffer_dir = os.path.join(script_dir, "buffers", subfolder_name)
    buffer_locs = os.listdir(raw_buffer_dir)
    buffer_locs = [buffer_loc.split(".")[0] for buffer_loc in buffer_locs]
    buffer_names = [
        os.path.join(subfolder_name, buffer_loc) for buffer_loc in buffer_locs
    ]
    for _ in range(4):
        buffer_names.pop(0)
    for buffer_name in buffer_names:
        buffer = Buffer(buffer_name)
        raise Exception("STOP")


if __name__ == "__main__":
    main()

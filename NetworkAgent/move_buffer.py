import os


env_name = "PPO_50000_step"


def get_buffer_dir(env_name: str) -> str:
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    buffer_dir = os.path.join(this_file_dir, "buffers", env_name)
    return buffer_dir


def get_temp_dir() -> str:
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(this_file_dir, "buffers", "temp")
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    return temp_dir


def move_buffer_to_temp_dir(buffer_dir: str, temp_dir: str) -> None:
    base_buffer_locs = os.listdir(buffer_dir)
    last_base_buffer = base_buffer_locs[-1]
    old_buffer_loc = os.path.join(buffer_dir, last_base_buffer)
    new_buffer_loc = os.path.join(temp_dir, last_base_buffer)
    os.rename(old_buffer_loc, new_buffer_loc)


def move_buffer(env_name: str) -> None:
    buffer_dir = get_buffer_dir(env_name)
    temp_dir = get_temp_dir()
    move_buffer_to_temp_dir(buffer_dir, temp_dir)


def main() -> None:
    move_buffer(env_name)


if __name__ == "__main__":
    main()

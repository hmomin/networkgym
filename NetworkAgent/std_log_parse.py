import matplotlib.pyplot as plt
import os

log_file_paths = [
    "data/2023_11_26_std_logs/seeds_0-7.log",
    "data/2023_11_26_std_logs/seeds_8-15.log",
    "data/2023_11_26_std_logs/seeds_16-23.log",
]


def get_real_path(log_file_path: str) -> str:
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(this_file_dir, log_file_path)


def get_data_dir() -> str:
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(this_file_dir, "data")
    return data_dir


def parse_log_file(
    log_file_path, total_timesteps: list[float], std_values: list[float]
) -> None:
    if len(std_values) == 0:
        std_values.append(1.0)
    else:
        std_values.append(std_values[-1])
    true_file_path = get_real_path(log_file_path)
    with open(true_file_path, "r") as file:
        for line in file:
            line = line.strip()
            if "total_timesteps" in line or "std" in line:
                split_line = line.split("|")
                relevant_value = split_line[2].strip()
                if "total_timesteps" in line:
                    total_timestep = int(relevant_value)
                    if (
                        len(total_timesteps) > 0
                        and total_timestep <= total_timesteps[-1]
                    ):
                        total_timesteps.append(total_timesteps[-1] + 2048)
                    else:
                        total_timesteps.append(total_timestep)
                else:
                    std_values.append(float(relevant_value))


def plot_std_values(total_timesteps, std_values) -> None:
    plt.figure(figsize=(19, 9))
    plt.rc("font", weight="normal", size=20)
    plt.grid(
        visible=True,
        which="both",
        axis="both",
        color="k",
        linestyle="-",
        linewidth=0.1,
    )
    plt.plot(total_timesteps, std_values, marker="o")
    plt.tight_layout(rect=(0.02, 0.02, 0.99, 0.98))
    plt.xlabel("Step")
    plt.ylabel("Average $\sigma$ per user")
    plt.savefig(os.path.join(get_data_dir(), f"std.png"))
    plt.show()


if __name__ == "__main__":
    total_timesteps = []
    std_values = []
    for log_file_path in log_file_paths:
        parse_log_file(log_file_path, total_timesteps, std_values)
        if len(total_timesteps) != len(std_values):
            print(f"{len(total_timesteps)} != {len(std_values)}")
    plot_std_values(total_timesteps, std_values)

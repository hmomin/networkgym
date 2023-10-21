import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pprint import pprint

# DATA_DIR = "2023_10_16_10_20_20000_steps_bc"
# DATA_DIR = "2023_10_16_10_21_20000_steps_normal"
# DATA_DIR = "2023_10_16_10_22_50000_steps_bc"
DATA_DIR = "2023_10_16_10_23_50000_steps_normal"

COLOR_MAP = {
    "PPO": "#e41a1c",
    "PPO_20000_bc": "#3700b8",
    "PPO_20000_normal": "#3700b8",
    "PPO_50000_bc": "#3700b8",
    "PPO_50000_normal": "#3700b8",
    "system_default": "#000000",
    "system_default_20000_bc": "#4daf4a",
    "system_default_20000_normal": "#4daf4a",
    "system_default_50000_bc": "#4daf4a",
    "system_default_50000_normal": "#4daf4a",
}

Y_MIN = -4423
Y_MAX = -764


def get_data_dir() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data", DATA_DIR)
    return data_dir


def get_data_filenames() -> list[str]:
    data_dir = get_data_dir()
    base_names = os.listdir(data_dir)
    full_file_paths = [os.path.join(data_dir, base_name) for base_name in base_names]
    return full_file_paths


def parse_data(df: pd.DataFrame) -> dict[str | int, tuple[float, float]]:
    data_dict: dict[str | int, tuple[float, float]] = {}
    # NOTE: each column is a key
    for column_key in df:
        num_buffers = parse_key(column_key)
        values: list[float] = df[column_key].to_list()
        data_dict[num_buffers] = get_mean_std_from_section(values)
    return data_dict


def get_algorithm_name(filename: str) -> str:
    base_csv_filename = os.path.basename(filename)
    algorithm_pieces = base_csv_filename.split(".")[0:-1]
    algorithm_name = "".join(algorithm_pieces)
    return algorithm_name


def parse_key(column_key: float | str) -> str | int:
    num_buffers_str = column_key.split(".")[-1]
    try:
        int(num_buffers_str)
    except ValueError:
        return num_buffers_str
    return int(num_buffers_str)


def get_mean_std_from_section(values: list[float]) -> tuple[float, float]:
    mean_value = np.mean(values)
    stdev = np.std(values)
    return (mean_value, stdev)


def process_data_for_plotting(
    data_dict: dict[str | int, tuple[float, float]]
) -> tuple[list[int], list[tuple[float, float]]]:
    # NOTE: treat singular case separately
    if len(data_dict.keys()) == 1:
        x_vals = get_x_vals()
        mean_std_vals = list(data_dict.values())[0]
        y_vals = [mean_std_vals] * len(x_vals)
    else:
        x_vals: list[int] = []
        y_vals: list[tuple[float, float]] = []
        for key, value in data_dict.items():
            x_val = int(key)
            x_vals.append(x_val)
            y_vals.append(value)
    sorted_pairs = sorted(zip(x_vals, y_vals), key=lambda pair: pair[0])
    sorted_x, sorted_y = zip(*sorted_pairs)
    return sorted_x, sorted_y


def get_x_vals() -> list[int]:
    return [1, 2, 4, 8, 16, 32, 64]


def plot_data(
    plotter_dict: dict[str, tuple[list[int], list[tuple[float, float]]]]
) -> None:
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
    for name, (x_vals, mean_std_list) in plotter_dict.items():
        mean_rewards = [mean_val for mean_val, stdev in mean_std_list]
        lower_bounds = [mean_val - stdev for mean_val, stdev in mean_std_list]
        upper_bounds = [mean_val + stdev for mean_val, stdev in mean_std_list]
        line_color = COLOR_MAP[name] if name in COLOR_MAP else (0, 0, 0)
        plt.plot(x_vals, mean_rewards, color=line_color, label=name)
        plt.xscale("log", base=2)
        x_ticks = get_x_vals()
        x_labels = [str(x_val) for x_val in x_ticks]
        plt.xticks(x_ticks, x_labels)
        plt.fill_between(
            x_vals,
            lower_bounds,
            upper_bounds,
            color=line_color,
            alpha=0.1,
            label="Filled Area",
        )
    handles, labels = plt.gca().get_legend_handles_labels()
    good_handles = [
        handle for idx, handle in enumerate(handles) if labels[idx] != "Filled Area"
    ]
    good_labels = [
        label for idx, label in enumerate(labels) if labels[idx] != "Filled Area"
    ]
    # plt.legend(good_handles, good_labels)
    plt.legend(good_handles, good_labels, loc="upper left", bbox_to_anchor=(1, 0.75))
    plt.legend(good_handles, good_labels, loc="upper left", bbox_to_anchor=(1, 0.75))
    plt.tight_layout(rect=(0.02, 0.02, 0.99, 0.98))
    plt.ylim(Y_MIN, Y_MAX)
    # plt.tight_layout(rect=(0, 0, 0.99, 1))
    # plt.title(TITLE)
    plt.xlabel("Number of Training Buffers")
    plt.ylabel("Cumulative Rollout Return")
    plt.savefig(os.path.join(get_data_dir(), f"../{DATA_DIR}.png"))
    plt.show()


def main() -> None:
    data_filenames = get_data_filenames()
    # NOTE: key is the algorithm name
    plotter_dict: dict[str, tuple[list[int], list[tuple[float, float]]]] = {}
    for filename in data_filenames:
        dataframe = pd.read_csv(filename)
        data_dict = parse_data(dataframe)
        pprint(data_dict)
        algorithm_name = get_algorithm_name(filename)
        x_vals, mean_std_vals = process_data_for_plotting(data_dict)
        plotter_dict[algorithm_name] = (x_vals, mean_std_vals)
    plot_data(plotter_dict)


if __name__ == "__main__":
    main()

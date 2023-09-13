import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from typing import Dict, List, Tuple

title = "Seed 6 Utility Reward Function - Off-Policy Training"
# title = "Seed 6 Utility Reward Function - On-Policy Training"
# title = "Seed 6 Utility Reward Function - Heuristic Algorithms"

# period_value = 1000 if "Training" in title else 100
period_value = 1000
# period_value = 100

ignore_losers = True
losers = ["A2C", "ArgMax", "PPO", "Random"]
# losers = ["ArgMax", "DDPG", "Random", "SAC"]
# losers = ["A2C", "DDPG", "PPO", "SAC"]

color_map = {
    "A2C": "#e41a1c",
    "SAC": "#3700b8",
    "Random": "#999999",
    "TD3": "#984ea3",
    "ArgMin": "#aaaa33",
    "ArgMax": "#ff7f00",
    "DDPG": "#a65628",
    "PPO": "#f781bf",
    "system_default": "#4daf4a",
}


def get_data_dir() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data", "separated_by_algorithms")
    return data_dir


def get_data_filenames() -> list[str]:
    data_dir = get_data_dir()
    base_names = os.listdir(data_dir)
    full_file_paths = [os.path.join(data_dir, base_name) for base_name in base_names]
    return full_file_paths


def parse_data(df: pd.DataFrame) -> tuple[str, list[int], list[float]]:
    # NOTE: only two columns expected in the dataframe: step and algo reward
    for column_key in df:
        if column_key == "Step":
            step_data = df[column_key].to_list()
        else:
            reward_data = df[column_key].to_list()
            algorithm_name = column_key.split()[0]
    return algorithm_name, step_data, reward_data


def construct_means_and_stds(
    data_dict: dict[str, dict[str, list]]
) -> dict[str, list[tuple[float, float]]]:
    parsed_dict: dict[str, list[tuple[float, float]]] = {}
    for algorithm_name, algorithm_data in data_dict.items():
        x_values: list[int] = algorithm_data["x"]
        y_values: list[float] = algorithm_data["y"]
        mean_std_values = split_by_section(x_values, y_values, period_value)
        parsed_dict[algorithm_name] = mean_std_values
    return parsed_dict


def split_by_section(
    x_values: list[int], y_values: list[float], period: int
) -> list[tuple[float, float]]:
    end_idx = period
    mean_stdev_values = []
    sectioned_values: list[float] = []
    for x_val, y_val in zip(x_values, y_values):
        if x_val < end_idx:
            sectioned_values.append(y_val)
        else:
            mean_val, stdev = get_mean_std_from_section(sectioned_values)
            mean_stdev_values.append((mean_val, stdev))
            sectioned_values = []
            end_idx += period
    if len(sectioned_values) > 0:
        mean_val, stdev = get_mean_std_from_section(sectioned_values)
        mean_stdev_values.append((mean_val, stdev))
    return mean_stdev_values


def get_mean_std_from_section(values: list[float]) -> tuple[float, float]:
    mean_value = np.mean(values)
    stdev = np.std(values)
    return mean_value, stdev


def plot_data(mean_std_dict: Dict[str, List[Tuple[float, float]]]) -> None:
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
    for name, mean_std_list in mean_std_dict.items():
        parsed_name = name.split()[0]
        if parsed_name in losers and ignore_losers:
            continue
        counts = list(range(len(mean_std_list)))
        mean_rewards = [mean_val for mean_val, stdev in mean_std_list]
        lower_bounds = [mean_val - stdev for mean_val, stdev in mean_std_list]
        upper_bounds = [mean_val + stdev for mean_val, stdev in mean_std_list]
        line_color = color_map[parsed_name] if parsed_name in color_map else (0, 0, 0)
        label = parsed_name
        plt.plot(counts, mean_rewards, color=line_color, label=label)
        plt.fill_between(
            counts,
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
    # plt.tight_layout(rect=(0, 0, 0.99, 1))
    # plt.xlim(xmin=0.0)
    plt.ylim(ymin=-2.5, ymax=1.5)
    plt.title(title)
    plt.xlabel(f"Step (x {period_value})")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(get_data_dir(), f"../{title}.png"))
    plt.show()


def main() -> None:
    data_filenames = get_data_filenames()
    data_dict: dict[str, dict[str, list]] = {}
    for filename in data_filenames:
        dataframe = pd.read_csv(filename)
        algorithm, steps, rewards = parse_data(dataframe)
        data_dict[algorithm] = {"x": steps, "y": rewards}
    mean_std_dict = construct_means_and_stds(data_dict)
    plot_data(mean_std_dict)


if __name__ == "__main__":
    main()

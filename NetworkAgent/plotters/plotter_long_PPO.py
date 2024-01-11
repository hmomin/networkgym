import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from typing import Any

filenames = [
    "2023_12_04_lspi_testing/lspi.csv",
    "2023_12_04_lspi_testing/ppo.csv",
    "2023_12_04_lspi_testing/sys_default.csv",
]

# filenames = [
#     "2023_12_04_ppo_discrete_training/PPO_discrete_0-7.csv",
#     "2023_12_04_ppo_discrete_training/sys_default.csv"
# ]

# filenames = [
#     "2023_11_20_ppo_training/ppo_seeds_0-7.csv",
#     "2023_11_20_ppo_training/ppo_seeds_8-15.csv",
#     "2023_11_20_ppo_training/ppo_seeds_16-23.csv",
# ]

# filenames = [
#     "2023_11_26_sac_training/sac_seeds_0-7.csv",
#     "2023_11_26_sac_training/sac_seeds_8-15.csv",
#     "2023_11_26_sac_training/sac_seeds_16-23.csv",
# ]

title = "Training Utility Reward Curves (deterministic walk)"

period_value = 1000 if "Training" in title else 100

color_map = {
    "PPO_seeds_0-7": "#7D54B2",
    "PPO_seeds_8-15": "#E87B9F",
    "PPO_seeds_16-23": "#87CEBF",
    "SAC_seeds_0-7": "#7D54B2",
    "SAC_seeds_8-15": "#E87B9F",
    "SAC_seeds_16-23": "#87CEBF",
    "sys_default_seed_0": "#4daf4a",
    "sys_default_seed_8": "#4daf4a",
    "PPO_seed_8": "#a751bf",
    "LSPI_seed_8": "#DC143C",
}


def get_data_dir() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    return data_dir


def parse_data(df: pd.DataFrame) -> tuple[str, list[float], list[float]]:
    for column_key in df:
        if column_key == "Step":
            x_vals = df[column_key].to_list()
        else:
            y_vals = df[column_key].to_list()
            algorithm = column_key
    return algorithm, x_vals, y_vals


def construct_means_and_stds(
    x_vals: list[float], y_vals: list[float], period: int
) -> tuple[list[float], list[tuple[float, float]]]:
    start_idx = 0
    mean_stdev_values = []
    new_x_vals = []
    while start_idx < len(y_vals):
        sectioned_values = y_vals[start_idx : start_idx + period]
        mean_value = np.mean(sectioned_values)
        stdev = np.std(sectioned_values)
        new_x_vals.append(x_vals[start_idx])
        mean_stdev_values.append((mean_value, stdev))
        start_idx += period
    return new_x_vals, mean_stdev_values


def plot_data(full_dict: dict[str, dict[str, list[Any]]]) -> None:
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
    for name, mean_std_dict in full_dict.items():
        parsed_name = name.split()[0]
        counts = mean_std_dict["x"]
        mean_std_list = mean_std_dict["y"]
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
    # plt.ylim(ymin=0.0)
    # plt.ylim(ymin=-2.5, ymax=1.5)
    # plt.title(title)
    plt.xlabel(f"Step")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(get_data_dir(), f"{title}.png"))
    plt.show()


def main() -> None:
    full_dict: dict[str, dict[str, list[Any]]] = {}
    for filename in filenames:
        data_filepath = os.path.join(get_data_dir(), filename)
        dataframe = pd.read_csv(data_filepath)
        algorithm, x_vals, y_vals = parse_data(dataframe)
        new_x_vals, mean_std_values = construct_means_and_stds(
            x_vals, y_vals, period_value
        )
        full_dict[algorithm] = {"x": new_x_vals, "y": mean_std_values}
    plot_data(full_dict)


if __name__ == "__main__":
    main()

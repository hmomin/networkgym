import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from typing import Any, Dict, List, Tuple

filename = "seed_3_throughput_good_state_minus_apid_test.csv"
# filename = "seed_2_throughput_good_state_minus_apid_no_normalize_long.csv"
# filename = "seed_3_delay_good_state_test.csv"
# filename = "seed_2_delay_good_state_no_normalize_long.csv"
# filename = "seed_3_utility_good_state_test.csv"
# filename = "seed_2_utility_good_state_no_normalize_long.csv"
# filename = "seed_2_utility_good_state_PPO_normalize_net_size_testing.csv"

title = "Throughput Reward Function - Testing"
# title = "Throughput Reward Function - Training"
# title = "Delay Reward Function - Testing"
# title = "Delay Reward Function - Training"
# title = "Utility Reward Function - Testing"
# title = "Utility Reward Function - Training"
# title = "PPO Training - Utility Reward Function"

period_value = 1000 if "Training" in title else 100

ignore_losers = True
losers = ["Random", "TD3", "ArgMin", "DDPG"]

use_label_map = False
label_map = {
    "PPO_small_net": "no normalize - [64, 64]",
    "PPO_raw": "normalize - [64, 64]",
    "PPO_normalize": "normalize - [256, 256]",
    "PPO": "no normalize - [256, 256]",
}

color_map = {
    "A2C": "#e41a1c",
    "SAC": "#377eb8",
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
    data_dir = os.path.join(script_dir, "data")
    return data_dir


def parse_data(df: pd.DataFrame) -> Dict[str, List[float]]:
    dict_data: Dict = {}
    for column_key in df:
        if column_key == "Step":
            continue
        dict_data[column_key] = df[column_key].to_list()
    return dict_data


def construct_means_and_stds(
    data_dict: Dict[str, List[float]]
) -> Dict[str, List[Tuple[float, float]]]:
    parsed_dict = {}
    for key, values in data_dict.items():
        mean_std_values = split_by_section(values, period_value)
        parsed_dict[key] = mean_std_values
    return parsed_dict


def split_by_section(values: List[float], period: int) -> List[Tuple[float, float]]:
    start_idx = 0
    mean_stdev_values = []
    while start_idx < len(values):
        sectioned_values = values[start_idx : start_idx + period]
        mean_value = np.mean(sectioned_values)
        stdev = np.std(sectioned_values)
        mean_stdev_values.append((mean_value, stdev))
        start_idx += period
    return mean_stdev_values


def get_indices(lst: List[Any], value: Any) -> List[int]:
    return [index for index, element in enumerate(lst) if element == value]


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
        label = (
            label_map[parsed_name]
            if parsed_name in label_map and use_label_map
            else parsed_name
        )
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
    plt.title(title)
    plt.xlabel(f"Step (x {period_value})")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(get_data_dir(), f"{title}.png"))
    plt.show()


def main() -> None:
    data_filepath = os.path.join(get_data_dir(), filename)
    dataframe = pd.read_csv(data_filepath)
    dict_data = parse_data(dataframe)
    mean_std_dict = construct_means_and_stds(dict_data)
    plot_data(mean_std_dict)


if __name__ == "__main__":
    main()
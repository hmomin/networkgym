import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

data_dir = "2023_09_23_preliminary_offline_RL_testing_48_to_55"

title = ""

color_map = {
    "system_default": "#ff0000",
    "system_default_td3_bc": "#00aa00",
    "system_default_td3_normal": "#0000ff",
}


def get_data_dir() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_data_dir = os.path.join(script_dir, "..", "data", data_dir)
    return full_data_dir


def get_csv_filenames(full_data_dir: str) -> list[str]:
    all_filenames = os.listdir(full_data_dir)
    csv_filenames: list[str] = []
    for filename in all_filenames:
        full_filepath = os.path.join(full_data_dir, filename)
        if os.path.isfile(full_filepath) and full_filepath.endswith(".csv"):
            csv_filenames.append(full_filepath)
    return csv_filenames


def get_data_from_csvs(csv_filenames: list[str]) -> dict[str, list[list[float]]]:
    data_dict: dict[str, list[list[float]]] = {}
    for filepath in csv_filenames:
        nested_reward_list: list[list[float]] = []
        df = pd.read_csv(filepath)
        for column_key in df:
            if column_key == "Step":
                continue
            reward_list = df[column_key].to_list()
            nested_reward_list.append(reward_list[:9_999])
        basename = os.path.basename(filepath)
        data_dict[basename] = nested_reward_list
    return data_dict


def construct_means_and_stds(
    data_dict: dict[str, list[list[float]]]
) -> dict[str, list[tuple[float, float]]]:
    parsed_dict = {}
    for key, nested_values in data_dict.items():
        mean_std_values = get_row_wise_mean_std(nested_values)
        parsed_dict[key] = mean_std_values
    return parsed_dict


def get_row_wise_mean_std(
    nested_values: list[list[float]],
) -> list[tuple[float, float]]:
    num_values = len(nested_values[0])
    mean_stdev_values = []
    for idx in range(num_values):
        row_values: list[float] = []
        for reward_list in nested_values:
            row_values.append(reward_list[idx])
        mean_value = np.mean(row_values)
        stdev = np.std(row_values)
        mean_stdev_values.append((mean_value, stdev))
    return mean_stdev_values


def plot_data(mean_std_dict: dict[str, list[tuple[float, float]]]) -> None:
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
        parsed_name = name.split("_utility_")[0]
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
    # plt.ylim(ymin=0.0)
    # plt.ylim(ymin=-1.5, ymax=0.5)
    if title:
        plt.title(title)
    plt.xlabel(f"Step")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(get_data_dir(), f"{title}.png"))
    plt.show()


def main() -> None:
    full_data_dir = get_data_dir()
    csv_filenames = get_csv_filenames(full_data_dir)
    data_dict = get_data_from_csvs(csv_filenames)
    mean_std_dict = construct_means_and_stds(data_dict)
    # FIXME LOW: figure out an easy way to smooth the curves
    # otherwise, looks good/expected
    plot_data(mean_std_dict)


if __name__ == "__main__":
    main()

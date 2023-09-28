import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

title = "Rollout Cumulative Return during Training Process"

color_map = {
    "system_default": "#e41a1c",
    "system_default_td3": "#3700b8",
    "system_default_td3_bc": "#4daf4a",
}

name_map = {
    "system_default_normal": "system_default_td3",
    "system_default_bc": "system_default_td3_bc",
}


def get_data_dir() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data", "2023_09_27_td3_with(out)_bc_testing")
    return data_dir


def get_data_filenames() -> list[str]:
    data_dir = get_data_dir()
    base_names = os.listdir(data_dir)
    full_file_paths = [os.path.join(data_dir, base_name) for base_name in base_names]
    return full_file_paths


def parse_data(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    data_dict: dict[str, tuple[float, float]] = {}
    # NOTE: each column is a key
    for column_key in df:
        values: list[float] = df[column_key].to_list()
        data_dict[column_key] = get_mean_std_from_section(values)
    return data_dict


def get_mean_std_from_section(values: list[float]) -> tuple[float, float]:
    mean_value = np.mean(values)
    stdev = np.std(values)
    return (mean_value, stdev)


def process_data_for_plotting(
    data_dict: dict[str, tuple[float, float]]
) -> tuple[str, list[int], list[tuple[float, float]]]:
    algorithm_name = list(data_dict.keys())[0]
    algorithm_name = algorithm_name.split(".")[0]
    # NOTE: treat system_default case separately
    if len(data_dict.keys()) == 1:
        x_vals = list(range(0, 1_000_001, 10_000))
        mean_std_vals = data_dict[algorithm_name]
        y_vals = [mean_std_vals] * len(x_vals)
    else:
        x_vals: list[int] = []
        y_vals: list[tuple[float, float]] = []
        for key, value in data_dict.items():
            x_val = int(key.split(".")[-1])
            x_vals.append(x_val)
            y_vals.append(value)
    return algorithm_name, x_vals, y_vals


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
        parsed_name = name_map[name] if name in name_map else name
        mean_rewards = [mean_val for mean_val, stdev in mean_std_list]
        lower_bounds = [mean_val - stdev for mean_val, stdev in mean_std_list]
        upper_bounds = [mean_val + stdev for mean_val, stdev in mean_std_list]
        line_color = color_map[parsed_name] if parsed_name in color_map else (0, 0, 0)
        plt.plot(x_vals, mean_rewards, color=line_color, label=parsed_name)
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
    # plt.tight_layout(rect=(0, 0, 0.99, 1))
    # plt.xlim(xmin=0.0)
    # plt.ylim(ymin=-2.5, ymax=1.5)
    plt.title(title)
    plt.xlabel("Training Iteration")
    plt.ylabel("Cumulative Rollout Return")
    plt.savefig(os.path.join(get_data_dir(), f"../{title}.png"))
    plt.show()


def main() -> None:
    data_filenames = get_data_filenames()
    # NOTE: key is the algorithm name
    plotter_dict: dict[str, tuple[list[int], list[tuple[float, float]]]] = {}
    for filename in data_filenames:
        dataframe = pd.read_csv(filename)
        data_dict = parse_data(dataframe)
        algorithm_name, x_vals, mean_std_vals = process_data_for_plotting(data_dict)
        plotter_dict[algorithm_name] = (x_vals, mean_std_vals)
    plot_data(plotter_dict)


if __name__ == "__main__":
    main()

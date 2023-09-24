import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

data_dir = "2023_09_20_training_stats"

color_map = {
    "system_default_td3_Q1_loss": "#ff0000",
    "system_default_td3_Q2_loss": "#0000ff",
    "system_default_td3_policy_loss": "#000000",
    "system_default_td3_bc_Q1_loss": "#0000ff",
    "system_default_td3_bc_Q2_loss": "#ff0000",
    "system_default_td3_bc_policy_loss": "#000000",
}

period_value = 1000

loss_index_map = ["Q1_loss", "Q2_loss", "policy_loss"]

COUNTER = 0


def get_data_dir() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_data_dir = os.path.join(script_dir, "data", data_dir)
    return full_data_dir


def get_training_stats_filenames(full_data_dir: str) -> list[str]:
    all_filenames = os.listdir(full_data_dir)
    training_stats_filenames: list[str] = []
    for filename in all_filenames:
        full_filepath = os.path.join(full_data_dir, filename)
        if os.path.isfile(full_filepath) and full_filepath.endswith(".training_stats"):
            training_stats_filenames.append(full_filepath)
    return training_stats_filenames


def get_data_from_files(
    training_stats_filenames: list[str],
) -> dict[str, list[list[float]]]:
    data_dict: dict[str, list[list[float]]] = {}
    for filepath in training_stats_filenames:
        training_stats: list[list[float]] = pickle.load(open(filepath, "rb"))
        basename = os.path.basename(filepath)
        data_dict[basename] = training_stats
    return data_dict


def parse_data(
    data_dict: dict[str, list[list[float]]]
) -> dict[str, dict[str, tuple[list[float], list[float]]]]:
    parsed_data: dict[str, dict[str, tuple[list[float], list[float]]]] = {}
    for algorithm_name, nested_list in data_dict.items():
        algorithm_name = algorithm_name.split(".")[0]
        algorithm_dict: dict[str, tuple[list[float], list[float]]] = {}
        loss_tracker: tuple[
            tuple[list[float], list[float]],
            tuple[list[float], list[float]],
            tuple[list[float], list[float]],
        ] = (([], []), ([], []), ([], []))
        for idx, loss_list in enumerate(nested_list):
            for idx2, loss_value in enumerate(loss_list):
                loss_tracker[idx2][0].append(idx)
                loss_tracker[idx2][1].append(loss_value)
        for idx, loss_name in enumerate(loss_index_map):
            algorithm_dict[loss_name] = loss_tracker[idx]
        parsed_data[algorithm_name] = algorithm_dict
    return parsed_data


def get_xy_data(
    labels: list[str], algorithm_dict: dict[str, tuple[list[float], list[float]]]
) -> list[tuple[list[float], list[float]]]:
    xy_data: list[tuple[list[float], list[float]]] = []
    for label in labels:
        xy_data.append(algorithm_dict[label])
    return xy_data


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


def plot_data(
    algorithm_name: str,
    labels: list[str],
    algorithm_dict: dict[str, tuple[list[float], list[float]]],
) -> None:
    global COUNTER
    xy_data = get_xy_data(labels, algorithm_dict)
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
    for idx, (x, y) in enumerate(xy_data):
        parsed_name = algorithm_name.split(".")[0] + "_" + labels[idx]
        line_color = color_map[parsed_name] if parsed_name in color_map else (0, 0, 0)
        plt.plot(x, y, color=line_color, label=parsed_name)
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
    plt.ylim(ymin=-10, ymax=110)
    # plt.ylim(ymin=-10, ymax=260)
    title = algorithm_name
    plt.title(title)
    plt.xlabel(f"Step")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(get_data_dir(), f"{title}_{COUNTER}.png"))
    COUNTER += 1
    plt.show()


def main() -> None:
    full_data_dir = get_data_dir()
    training_stats_filenames = get_training_stats_filenames(full_data_dir)
    data_dict = get_data_from_files(training_stats_filenames)
    parsed_data = parse_data(data_dict)
    for key1, value1 in parsed_data.items():
        print(key1)
        for key2, value2 in value1.items():
            print("\t" + key2)
            for idx, value3 in enumerate(value2):
                print(f"\t\t{idx}: {str(len(value3))}")
    for algorithm_name, algorithm_dict in parsed_data.items():
        plot_data(algorithm_name, loss_index_map[:2], algorithm_dict)
        plot_data(algorithm_name, loss_index_map[2:], algorithm_dict)


if __name__ == "__main__":
    main()

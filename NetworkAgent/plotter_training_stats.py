import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from pprint import pprint

data_dir = "2023_09_20_training_stats"

title = "Training Stats"

color_map = {
    "system_default_bc_Q1_loss": "#ff0000",
    "system_default_bc_Q2_loss": "#00ff00",
    "system_default_td3_normal": "#0000ff",
}

loss_index_map = ["Q1_loss", "Q2_loss", "policy_loss"]


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


def plot_data(
    algorithm_name: str,
    labels: list[str],
    algorithm_dict: dict[str, tuple[list[float], list[float]]],
) -> None:
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
    # plt.ylim(ymin=-2.5, ymax=1.5)
    plt.title(title)
    plt.xlabel(f"Step")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(get_data_dir(), f"{title}.png"))
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

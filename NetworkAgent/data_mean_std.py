import numpy as np
import os
from csv import reader


# data_folder = "2024_01_27_GOOD_heuristic_and_td3_bc"
# data_folder = "2024_01_27_GOOD_PTD3"
# data_folder = "2024_01_27_GOOD_PTD3_beta_10.0"
data_folder = "2024_01_27_GOOD_PTD3_alpha_0.0"
# data_folder = "2024_01_27_GOOD_online_RL_deterministic"
# data_folder = "2024_01_27_GOOD_online_RL_stochastic"


def get_filepaths_from_folder(folder: str) -> list[str]:
    true_folder_path = get_full_path(folder)
    folder_filenames = os.listdir(true_folder_path)
    true_filepaths = []
    for filename in folder_filenames:
        if ".csv" in filename:
            true_filepaths.append(os.path.join(true_folder_path, filename))
    return true_filepaths


def get_full_path(folder_name: str) -> str:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    data_path = os.path.join(script_dir, "data", folder_name)
    return data_path


def read_csv(filepath: str) -> tuple[list[str], list[list[str]]]:
    data = []
    categories = []
    with open(filepath, "r") as csvfile:
        csv_reader = reader(csvfile)
        for idx, row in enumerate(csv_reader):
            if idx == 0:
                categories = row
            else:
                data.append(row)
    return categories, data


def construct_data_dict(
    categories: list[str], data: list[list[str]]
) -> dict[str, list[float]]:
    data_dict: dict[str, list[float]] = {}
    for row in data:
        for idx, string_value in enumerate(row):
            category = categories[idx]
            if category not in data_dict:
                data_dict[category] = []
            float_value = float(string_value)
            data_dict[category].append(float_value)
    return data_dict


def main() -> None:
    filepaths = get_filepaths_from_folder(data_folder)
    full_dict: dict[str, list[float]] = {}
    for filepath in filepaths:
        categories, data = read_csv(filepath)
        data_dict = construct_data_dict(categories, data)
        full_dict.update(data_dict)
    categories = sorted(list(full_dict.keys()))
    for category in categories:
        run_returns = full_dict[category]
        run_returns = [x / 3200.0 for x in run_returns]
        mean_run_return = np.mean(run_returns)
        stdev_run_return = np.std(run_returns)
        print(f"{category}:\n\t{mean_run_return} | {stdev_run_return}\n")


if __name__ == "__main__":
    main()

import numpy as np
import os
from csv import reader


# data_folder = "2024_02_12_heuristic_algorithms"
# data_folder = "2024_02_12_online_DRL_algorithms"
# data_folder = "2024_02_12_sys_default_TD3_BC"
# data_folder = "2024_02_12_sys_default_PTD3"
# data_folder = "2024_02_12_utility_logistic_TD3_BC"
# data_folder = "2024_02_16_sys_default_reproducible"
# data_folder = "2024_02_19_utility_logistic_reproducible"
# data_folder = "2024_02_19_td3_bc_reproducible"
# data_folder = "2024_02_26_heuristic_algorithms"
# data_folder = "2024_02_26_online_DRL_deterministic_reproducible"
# data_folder = "2024_02_26_online_DRL_stochastic_reproducible"
# data_folder = "2024_02_26_throughput_argmax_norm_utility_PTD3_reproducible"
# data_folder = "2024_02_29_sys_default_alpha_1.0_reproducible"
# data_folder = "2024_02_29_utility_logistic_alpha_1.0_reproducible"
# data_folder = "2024_02_29_throughput_argmax_alpha_1.0_reproducible"
# data_folder = "2024_03_02_online_DRL_deterministic_more_seeds"
data_folder = "2024_03_06_online_DRL_stochastic_more_seeds"
# data_folder = "2024_03_06_heuristic_algorithms_more_seeds"
# data_folder = "2024_03_06_td3_bc_more_seeds"


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


def update_dict(
    source_dict: dict[str, list[float]], new_dict: dict[str, list[float]]
) -> None:
    for key, value in new_dict.items():
        if key in source_dict:
            source_dict[key] += value
        else:
            source_dict[key] = value


def main() -> None:
    filepaths = get_filepaths_from_folder(data_folder)
    full_dict: dict[str, list[float]] = {}
    for filepath in filepaths:
        categories, data = read_csv(filepath)
        data_dict = construct_data_dict(categories, data)
        update_dict(full_dict, data_dict)
    categories = sorted(list(full_dict.keys()))
    for category in categories:
        run_returns = full_dict[category]
        run_returns = [x / 3200.0 for x in run_returns]
        mean_run_return = np.mean(run_returns)
        stdev_run_return = np.std(run_returns)
        standard_error = 1.96 * stdev_run_return / np.sqrt(len(run_returns))
        print(
            f"{category} ({len(run_returns)} runs):\n\t{mean_run_return:.3f} | {standard_error:.3f}\n"
        )


if __name__ == "__main__":
    main()

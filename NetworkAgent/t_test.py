import numpy as np
import os.path as path
from csv import reader
from scipy.stats import ttest_ind
from typing import Dict, List, Tuple


data_filename = "data/seed_3_delay_good_state_test.csv"
benchmark = "ArgMax"
# benchmark = "system_default"


def read_csv(filepath: str) -> Tuple[List[str], List[List[str]]]:
    true_filepath = get_full_path(filepath)
    data = []
    categories = []
    with open(true_filepath, "r") as csvfile:
        csv_reader = reader(csvfile)
        for idx, row in enumerate(csv_reader):
            # NOTE: remove step value (monotonic counter)
            row.pop(0)
            if idx == 0:
                categories = row
            else:
                data.append(row)
    parse_categories(categories)
    return categories, data


def get_full_path(filename: str) -> str:
    script_path = path.abspath(__file__)
    script_dir = path.dirname(script_path)
    data_path = path.join(script_dir, filename)
    return data_path


def parse_categories(categories: List[str]) -> None:
    for idx, category in enumerate(categories):
        categories[idx] = category.split()[0]


def construct_data_dict(
    categories: List[str], data: List[List[str]]
) -> Dict[str, List[float]]:
    data_dict: Dict[str, List[float]] = {}
    for row in data:
        for idx, string_value in enumerate(row):
            category = categories[idx]
            if category not in data_dict:
                data_dict[category] = []
            float_value = float(string_value)
            data_dict[category].append(float_value)
    return data_dict


def main() -> None:
    categories, data = read_csv(data_filename)
    data_dict = construct_data_dict(categories, data)
    for category, values in data_dict.items():
        if category == benchmark:
            continue
        benchmark_values = data_dict[benchmark]
        t_test_result = ttest_ind(
            values,
            benchmark_values,
            equal_var=False,
            nan_policy="raise",
            alternative="greater",
        )
        print(f"{category}:")

        print(
            f"    t_statistic = {t_test_result.statistic:.2f} | p-value = {t_test_result.pvalue:.2f} | df = {int(t_test_result.df)}"
        )


if __name__ == "__main__":
    main()

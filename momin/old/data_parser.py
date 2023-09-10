# NOTE: this is a poor statistic for comparing the difference between performance
# of algorithms. Better is using a t-test.

import os.path as path
from csv import reader
from typing import Dict, List, Tuple


data_filename = "data/all_compare_seed_3.csv"


def get_full_path(filename: str) -> str:
    script_path = path.abspath(__file__)
    script_dir = path.dirname(script_path)
    data_path = path.join(script_dir, filename)
    return data_path


def parse_categories(categories: List[str]) -> None:
    for idx, category in enumerate(categories):
        categories[idx] = category.split()[0]


def read_csv(filepath: str) -> Tuple[List[str], List[List[str]]]:
    true_filepath = get_full_path(filepath)
    data = []
    categories = []
    with open(true_filepath, "r") as csvfile:
        csv_reader = reader(csvfile)
        for idx, row in enumerate(csv_reader):
            row.pop(0)
            if idx == 0:
                categories = row
            else:
                data.append(row)
    parse_categories(categories)
    return categories, data


def fill_missing_with_previous(data: List[List[float]]) -> List[List[float]]:
    # NOTE: assumes the first row is fully populated
    data[0] = [float(cell) for cell in data[0]]
    previous_row = data[0]
    for row in data[1:]:
        for idx, cell in enumerate(row):
            if cell == "":
                row[idx] = float(previous_row[idx])
            else:
                row[idx] = float(row[idx])
        previous_row = row
    return data


def compute_percentage_max_values(
    categories: List[str], data: List[List[float]]
) -> Dict[str, float]:
    percentages = {}
    for row in data:
        max_index = row.index(max(row))
        max_category = categories[max_index]
        percentages[max_category] = percentages.get(max_category, 0) + 1
    total_count = len(data)
    for key in percentages:
        percentages[key] *= 100 / total_count
    return percentages


if __name__ == "__main__":
    categories, data = read_csv(data_filename)
    data = fill_missing_with_previous(data)
    percentages = compute_percentage_max_values(categories, data)

    print("\nPercentage of time each column takes on the maximum value:")
    for key, value in percentages.items():
        print(f"{key}: {value:.2f}%")

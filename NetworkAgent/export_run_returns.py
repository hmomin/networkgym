import csv
import os
import time
import wandb
from datetime import datetime, timedelta
from pprint import pprint
from tqdm import tqdm


"""
NOTE (times should be GMT!):
- PPO_deterministic_walk_bc\.10000\.64\.\d\.\d\d\d\.normalized
    CREATED_AFTER = datetime(2023, 10, 30, 4, 21, 0)
    CREATED_BEFORE = datetime(2023, 10, 30, 4, 23, 0)
"""


PROJECT_NAME = "hmomin/network_gym_client"
RUN_NAME = "PPO"
CREATED_AFTER = datetime(2023, 10, 30, 4, 21, 0)
CREATED_BEFORE = datetime(2023, 10, 30, 4, 23, 0)
MIN_RUNTIME = timedelta(minutes=1)


def get_runs() -> list:
    api = wandb.Api()
    filters = {
        "config.RL_algo": {"$regex": RUN_NAME},
        "createdAt": {
            "$gte": CREATED_AFTER.isoformat(),
            "$lte": CREATED_BEFORE.isoformat(),
        },
        "duration": {"$gte": MIN_RUNTIME.total_seconds()},
    }
    runs: list = api.runs(path=PROJECT_NAME, filters=filters)
    print(f"Found {len(runs)} runs with these filters...")
    return runs


def get_returns_from_runs(runs: list, num_runs: int = 0) -> dict[str, list[float]]:
    returns_dict: dict[str, list[float]] = {}
    print("Processing runs...")
    for idx, run in enumerate(tqdm(runs)):
        history = run.scan_history()
        run_rewards: list[float] = [row["reward"] for row in history]
        run_return: float = sum(run_rewards)
        run_name: str = run.name
        base_run_name = run_name.split("_seed_")[0]
        if base_run_name not in returns_dict:
            returns_dict[base_run_name] = []
        returns_dict[base_run_name].append(run_return)
        if num_runs > 0 and idx + 1 == num_runs:
            break
    return returns_dict


def write_to_csv(returns_dict: dict[str, list[float]]) -> None:
    headers = sorted(returns_dict.keys())
    columns = [returns_dict[key] for key in headers]
    # determine number of rows needed (length of the longest column)
    num_rows = max(len(column) for column in columns)
    # transpose data to have lists of values as rows
    rows = [
        [column[i] if i < len(column) else "" for column in columns]
        for i in range(num_rows)
    ]
    csv_filename = get_csv_filename()
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)


def get_csv_filename() -> str:
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(this_file_dir, "data")
    csv_filename = os.path.join(
        data_dir,
        f"{RUN_NAME}_{time.strftime('_%Y_%m_%d_%H_%M_%S', time.localtime())}.csv",
    )
    return csv_filename


def main() -> None:
    runs = get_runs()
    returns_dict = get_returns_from_runs(runs)
    pprint(returns_dict)
    write_to_csv(returns_dict)


if __name__ == "__main__":
    main()

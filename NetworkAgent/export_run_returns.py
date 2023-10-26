import csv
import os
import time
import wandb
from datetime import datetime, timedelta
from pprint import pprint
from tqdm import tqdm


"""
NOTE (times should be GMT!):
- 1_000_000 steps
    CREATED_AFTER = datetime(2023, 9, 26, 0, 35, 0)
    CREATED_BEFORE = datetime(2023, 9, 27, 10, 46, 0)
- 10_000 steps
    CREATED_AFTER = datetime(2023, 9, 28, 4, 51, 0)
    CREATED_BEFORE = datetime(2023, 9, 29, 11, 3, 0)
- 100_000 steps
    CREATED_AFTER = datetime(2023, 10, 3, 16, 11, 0)
    CREATED_BEFORE = datetime(2023, 10, 5, 22, 34, 0)
- PPO_20000_steps_training_normal
    CREATED_AFTER = datetime(2023, 10, 14, 4, 41, 0)
    CREATED_BEFORE = datetime(2023, 10, 14, 5, 43, 0)
- PPO_20000_steps_training_bc
    CREATED_AFTER = datetime(2023, 10, 14, 5, 43, 0)
    CREATED_BEFORE = datetime(2023, 10, 14, 6, 47, 0)
- PPO_50000_steps_training_normal
    CREATED_AFTER = datetime(2023, 10, 14, 6, 48, 0)
    CREATED_BEFORE = datetime(2023, 10, 14, 7, 49, 0)
- PPO_50000_steps_training_bc
    CREATED_AFTER = datetime(2023, 10, 14, 7, 44, 0)
    CREATED_BEFORE = datetime(2023, 10, 14, 8, 52, 0)
- system_default_20000_steps_training_bc
    CREATED_AFTER = datetime(2023, 10, 16, 5, 28, 0)
    CREATED_BEFORE = datetime(2023, 10, 16, 16, 7, 0)
- system_default_20000_steps_training_normal
    CREATED_AFTER = datetime(2023, 10, 16, 6, 24, 0)
    CREATED_BEFORE = datetime(2023, 10, 16, 7, 37, 0)
- system_default_50000_steps_training_bc
    CREATED_AFTER = datetime(2023, 10, 16, 7, 25, 0)
    CREATED_BEFORE = datetime(2023, 10, 16, 8, 42, 0)
- system_default_50000_steps_training_normal
    CREATED_AFTER = datetime(2023, 10, 16, 14, 44, 0)
    CREATED_BEFORE = datetime(2023, 10, 16, 15, 40, 0)
- PPO
    CREATED_AFTER = datetime(2023, 10, 16, 16, 53, 0)
    CREATED_BEFORE = datetime(2023, 10, 16, 16, 57, 0)
- system_default
    CREATED_AFTER = datetime(2023, 10, 16, 19, 36, 0)
    CREATED_BEFORE = datetime(2023, 10, 16, 19, 39, 0)
- PPO_95_test_20000_steps_buffers_bc
    CREATED_AFTER = datetime(2023, 10, 20, 4, 28, 0)
    CREATED_BEFORE = datetime(2023, 10, 20, 5, 31, 0)
- PPO_95_test_20000_steps_buffers_normal
    CREATED_AFTER = datetime(2023, 10, 20, 5, 32, 0)
    CREATED_BEFORE = datetime(2023, 10, 20, 6, 31, 0)
- PPO (95% convergence test (2048 batch, 64 mini, 80000 steps, 3e-4 lr))
    CREATED_AFTER = datetime(2023, 10, 20, 21, 50, 0)
    CREATED_BEFORE = datetime(2023, 10, 20, 21, 53, 0)
- PPO (higher convergence test (8192 batch, 256 mini, 100000 steps, 3e-4 lr))
    CREATED_AFTER = datetime(2023, 10, 20, 22, 5, 0)
    CREATED_BEFORE = datetime(2023, 10, 20, 22, 8, 0)
- PPO (even higher convergence test (8192 batch, 256 mini, 81920 steps, 1e-4 lr))
    CREATED_AFTER = datetime(2023, 10, 20, 22, 16, 0)
    CREATED_BEFORE = datetime(2023, 10, 20, 22, 18, 0)
"""


PROJECT_NAME = "hmomin/network_gym_client"
RUN_NAME = "PPO"
CREATED_AFTER = datetime(2023, 10, 20, 22, 16, 0)
CREATED_BEFORE = datetime(2023, 10, 20, 22, 18, 0)
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

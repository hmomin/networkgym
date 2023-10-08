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

PT
1_000_000 normal: 9/24 18:55 --> 9/26 09:18
1_000_000 bc:     9/26 11:27 --> 9/27 03:45
10_000 bc:        9/27 21:52 --> 9/28 13:16
10_000 normal:    9/28 13:50 --> 9/29 04:02
100_000 bc:       10/3 09:12 --> 10/4 18:22
100_000 normal:   10/4 21:15 --> 10/5 15:33

GMT
1_000_000 normal: 9/26 00:36 --> 9/26 16:18
1_000_000 bc:     9/26 18:27 --> 9/27 10:45
10_000 bc:        9/28 04:52 --> 9/28 20:16
10_000 normal:    9/28 20:50 --> 9/29 11:02
100_000 bc:       10/3 16:12 --> 10/5 01:22
100_000 normal:   10/5 04:15 --> 10/5 22:33
"""


PROJECT_NAME = "hmomin/network_gym_client"
RUN_NAME = "system_default_normal."
CREATED_AFTER = datetime(2023, 9, 26, 0, 35, 0)
CREATED_BEFORE = datetime(2023, 9, 27, 10, 46, 0)
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
    assert len(runs) == 808
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

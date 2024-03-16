import csv
import os
import time
import wandb
from datetime import datetime, timedelta
from pprint import pprint
from tqdm import tqdm


"""
NOTE (times should be GMT!):
- sys_default_norm_utility_bc (non-reproducible)
    CREATED_AFTER = datetime(2024, 2, 8, 21, 13, 0)
    CREATED_BEFORE = datetime(2024, 2, 9, 4, 28, 0)

- sys_default_norm_utility_PTD3 (non-reproducible)
    CREATED_AFTER = datetime(2024, 2, 9, 4, 54, 0)
    CREATED_BEFORE = datetime(2024, 2, 12, 19, 13, 0)

- PPO (deterministic) (non-reproducible)
    CREATED_AFTER = datetime(2024, 2, 11, 6, 21, 0)
    CREATED_BEFORE = datetime(2024, 2, 11, 6, 56, 0)

- SAC (deterministic) (non-reproducible)
    CREATED_AFTER = datetime(2024, 2, 11, 6, 21, 0)
    CREATED_BEFORE = datetime(2024, 2, 11, 6, 56, 0)

- utility_logistic_norm_utility_bc (non-reproducible)
    CREATED_AFTER = datetime(2024, 2, 12, 19, 46, 0)
    CREATED_BEFORE = datetime(2024, 2, 12, 19, 50, 0)

- utility_logistic_norm_utility_PTD3 (non-reproducible)
    CREATED_AFTER = datetime(2024, 2, 12, 16, 53, 0)
    CREATED_BEFORE = datetime(2024, 2, 15, 0, 0, 0)

- PPO (stochastic) (non-reproducible)
    CREATED_AFTER = datetime(2024, 2, 12, 23, 11, 0)
    CREATED_BEFORE = datetime(2024, 2, 12, 23, 45, 0)

- SAC (stochastic) (non-reproducible)
    CREATED_AFTER = datetime(2024, 2, 12, 23, 11, 0)
    CREATED_BEFORE = datetime(2024, 2, 12, 23, 45, 0)

REPRODUCIBLES start below vvvvv

- throughput_argmax (part 1)
    CREATED_AFTER = datetime(2024, 2, 8, 20, 12, 0)
    CREATED_BEFORE = datetime(2024, 2, 8, 20, 16, 0)

- throughput_argmax (part 2)
    CREATED_AFTER = datetime(2024, 2, 23, 23, 30, 0)
    CREATED_BEFORE = datetime(2024, 2, 24, 3, 18, 0)

- utility_logistic (part 1)
    CREATED_AFTER = datetime(2024, 2, 8, 20, 42, 0)
    CREATED_BEFORE = datetime(2024, 2, 8, 20, 47, 0)

- utility_logistic (part 2)
    CREATED_AFTER = datetime(2024, 2, 23, 23, 33, 0)
    CREATED_BEFORE = datetime(2024, 2, 24, 0, 4, 0)

- system_default (part 1)
    CREATED_AFTER = datetime(2024, 2, 8, 21, 13, 0)
    CREATED_BEFORE = datetime(2024, 2, 8, 22, 34, 0)

- system_default (part 2)
    CREATED_AFTER = datetime(2024, 2, 23, 22, 57, 0)
    CREATED_BEFORE = datetime(2024, 2, 23, 23, 2, 0)

- random (part 1)
    CREATED_AFTER = datetime(2024, 2, 8, 23, 3, 0)
    CREATED_BEFORE = datetime(2024, 2, 8, 23, 15, 0)

- random (part 2)
    CREATED_AFTER = datetime(2024, 2, 23, 22, 25, 0)
    CREATED_BEFORE = datetime(2024, 2, 23, 22, 29, 0)

- throughput_argmin (part 1)
    CREATED_AFTER = datetime(2024, 2, 8, 23, 54, 0)
    CREATED_BEFORE = datetime(2024, 2, 8, 23, 57, 0)

- throughput_argmin (part 2)
    CREATED_AFTER = datetime(2024, 2, 23, 21, 54, 0)
    CREATED_BEFORE = datetime(2024, 2, 23, 21, 58, 0)

- sys_default_norm_utility_PTD3 (reproducible)
    CREATED_AFTER = datetime(2024, 2, 14, 21, 6, 0)
    CREATED_BEFORE = datetime(2024, 2, 19, 22, 11, 0)

- utility_logistic_norm_utility_PTD3 (reproducible)
    CREATED_AFTER = datetime(2024, 2, 14, 21, 6, 0)
    CREATED_BEFORE = datetime(2024, 2, 19, 22, 11, 0)

- sys_default_norm_utility_td3_bc (reproducible)
    CREATED_AFTER = datetime(2024, 2, 19, 23, 30, 0)
    CREATED_BEFORE = datetime(2024, 2, 20, 1, 6, 0)

- utility_logistic_norm_utility_td3_bc (reproducible)
    CREATED_AFTER = datetime(2024, 2, 19, 23, 30, 0)
    CREATED_BEFORE = datetime(2024, 2, 20, 1, 6, 0)

- PPO (deterministic) (reproducible)
    CREATED_AFTER = datetime(2024, 2, 25, 16, 47, 0)
    CREATED_BEFORE = datetime(2024, 2, 25, 18, 30, 0)

- SAC (deterministic) (reproducible)
    CREATED_AFTER = datetime(2024, 2, 25, 16, 47, 0)
    CREATED_BEFORE = datetime(2024, 2, 25, 18, 30, 0)

- throughput_argmax_norm_utility_PTD3 (reproducible)
    CREATED_AFTER = datetime(2024, 2, 25, 23, 9, 0)
    CREATED_BEFORE = datetime(2024, 2, 26, 17, 15, 0)

- PPO (stochastic) (reproducible)
    CREATED_AFTER = datetime(2024, 2, 26, 19, 14, 0)
    CREATED_BEFORE = datetime(2024, 2, 26, 19, 48, 0)

- SAC (stochastic) (reproducible)
    CREATED_AFTER = datetime(2024, 2, 26, 20, 15, 0)
    CREATED_BEFORE = datetime(2024, 2, 26, 20, 50, 0)

- throughput_argmax_norm_utility_PTD3_beta_1.0_alpha_1.0 (reproducible)
    CREATED_AFTER = datetime(2024, 2, 26, 8, 17, 0)
    CREATED_BEFORE = datetime(2024, 2, 26, 23, 38, 0)

- throughput_argmax_norm_utility_PTD3 (reproducible)
    CREATED_AFTER = datetime(2024, 2, 27, 0, 7, 0)
    CREATED_BEFORE = datetime(2024, 2, 28, 0, 10, 0)

- sys_default_norm_utility_PTD3 (reproducible)
    CREATED_AFTER = datetime(2024, 2, 28, 6, 35, 0)
    CREATED_BEFORE = datetime(2024, 2, 28, 11, 15, 0)

- utility_logistic_norm_utility_PTD3 (reproducible)
    CREATED_AFTER = datetime(2024, 2, 28, 11, 36, 0)
    CREATED_BEFORE = datetime(2024, 2, 29, 2, 57, 0)

- throughput_argmax_norm_utility_PTD3 (reproducible)
    CREATED_AFTER = datetime(2024, 2, 28, 21, 34, 0)
    CREATED_BEFORE = datetime(2024, 2, 29, 2, 57, 0)

- sys_default_norm_utility_PTD3 (reproducible)
    CREATED_AFTER = datetime(2024, 2, 29, 6, 58, 0)
    CREATED_BEFORE = datetime(2024, 2, 29, 12, 46, 0)

- utility_logistic_norm_utility_PTD3 (reproducible)
    CREATED_AFTER = datetime(2024, 2, 29, 13, 2, 0)
    CREATED_BEFORE = datetime(2024, 2, 29, 19, 9, 0)

- throughput_argmax_norm_utility_PTD3 (reproducible)
    CREATED_AFTER = datetime(2024, 3, 1, 1, 2, 0)
    CREATED_BEFORE = datetime(2024, 3, 1, 5, 26, 0)

- throughput_argmin (more seeds)
    CREATED_AFTER = datetime(2024, 3, 1, 6, 30, 0)
    CREATED_BEFORE = datetime(2024, 3, 1, 7, 2, 0)

- random (more seeds)
    CREATED_AFTER = datetime(2024, 3, 1, 7, 30, 0)
    CREATED_BEFORE = datetime(2024, 3, 1, 8, 4, 0)

- system_default (more seeds)
    CREATED_AFTER = datetime(2024, 3, 1, 8, 32, 0)
    CREATED_BEFORE = datetime(2024, 3, 1, 9, 6, 0)

- throughput_argmax (more seeds)
    CREATED_AFTER = datetime(2024, 3, 1, 9, 33, 0)
    CREATED_BEFORE = datetime(2024, 3, 1, 10, 8, 0)

- utility_logistic (more seeds)
    CREATED_AFTER = datetime(2024, 3, 1, 10, 35, 0)
    CREATED_BEFORE = datetime(2024, 3, 1, 11, 10, 0)

- sys_default_norm_utility_td3_bc (more seeds)
    CREATED_AFTER = datetime(2024, 3, 1, 11, 35, 0)
    CREATED_BEFORE = datetime(2024, 3, 1, 12, 12, 0)

- utility_logistic_norm_utility_td3_bc (more seeds)
    CREATED_AFTER = datetime(2024, 3, 1, 12, 36, 0)
    CREATED_BEFORE = datetime(2024, 3, 1, 13, 14, 0)

- PPO (deterministic - more seeds)
    CREATED_AFTER = datetime(2024, 3, 1, 13, 36, 0)
    CREATED_BEFORE = datetime(2024, 3, 1, 14, 15, 0)

- throughput_argmax_norm_utility_PTD3_beta_300.0_alpha_1.0 (more seeds)
    CREATED_AFTER = datetime(2024, 3, 1, 17, 28, 0)
    CREATED_BEFORE = datetime(2024, 3, 1, 19, 4, 0)

- throughput_argmax_norm_utility_td3_bc (more seeds)
    CREATED_AFTER = datetime(2024, 3, 1, 19, 30, 0)
    CREATED_BEFORE = datetime(2024, 3, 1, 21, 8, 0)

- SAC (deterministic - more seeds)
    CREATED_AFTER = datetime(2024, 3, 1, 14, 36, 0)
    CREATED_BEFORE = datetime(2024, 3, 2, 3, 9, 0)

- PPO (stochastic - more seeds)
    CREATED_AFTER = datetime(2024, 3, 6, 22, 21, 0)
    CREATED_BEFORE = datetime(2024, 3, 6, 23, 58, 0)

- SAC (stochastic - more seeds)
    CREATED_AFTER = datetime(2024, 3, 6, 22, 21, 0)
    CREATED_BEFORE = datetime(2024, 3, 6, 23, 58, 0)

- checkpoint_BC
    CREATED_AFTER = datetime(2024, 3, 11, 7, 23, 0)
    CREATED_BEFORE = datetime(2024, 3, 11, 9, 0, 0)

- checkpoint_IQL
    CREATED_AFTER = datetime(2024, 3, 11, 9, 25, 0)
    CREATED_BEFORE = datetime(2024, 3, 11, 15, 17, 0)

- checkpoint_LB-SAC
    CREATED_AFTER = datetime(2024, 3, 11, 11, 0, 0)
    CREATED_BEFORE = datetime(2024, 3, 11, 15, 17, 0)

- checkpoint_TD3_BC
    CREATED_AFTER = datetime(2024, 3, 11, 11, 0, 0)
    CREATED_BEFORE = datetime(2024, 3, 11, 15, 17, 0)

- checkpoint_CQL
    CREATED_AFTER = datetime(2024, 3, 11, 15, 57, 0)
    CREATED_BEFORE = datetime(2024, 3, 11, 17, 33, 0)

- checkpoint_EDAC
    CREATED_AFTER = datetime(2024, 3, 11, 17, 59, 0)
    CREATED_BEFORE = datetime(2024, 3, 11, 19, 37, 0)

- checkpoint_BC_not_normalized
- checkpoint_CQL_not_normalized
- checkpoint_IQL_not_normalized
- checkpoint_TD3_BC_not_normalized
    CREATED_AFTER = datetime(2024, 3, 12, 18, 40, 0)
    CREATED_BEFORE = datetime(2024, 3, 13, 2, 35, 0)

- checkpoint_SAC-N
    CREATED_AFTER = datetime(2024, 3, 14, 16, 34, 0)
    CREATED_BEFORE = datetime(2024, 3, 14, 18, 29, 0)

- sys_default_norm_utility_PTD3_BC
- utility_logistic_norm_utility_PTD3_BC
- throughput_argmax_norm_utility_PTD3_BC
- BC_sys_default
- BC_throughput_argmax
- BC_utility_logistic
    CREATED_AFTER = datetime(2024, 3, 14, 21, 6, 0)
    CREATED_BEFORE = datetime(2024, 3, 14, 18, 29, 0)

- PPO (deterministic)
    CREATED_AFTER = datetime(2024, 3, 15, 3, 23, 0)
    CREATED_BEFORE = datetime(2024, 3, 15, 5, 0, 0)

- SAC (deterministic)
    CREATED_AFTER = datetime(2024, 3, 15, 14, 35, 0)
    CREATED_BEFORE = datetime(2024, 3, 15, 16, 12, 0)

- PPO (stochastic)
    CREATED_AFTER = datetime(2024, 3, 15, 16, 49, 0)
    CREATED_BEFORE = datetime(2024, 3, 15, 20, 30, 0)

- SAC (stochastic)
    CREATED_AFTER = datetime(2024, 3, 15, 14, 35, 0)
    CREATED_BEFORE = datetime(2024, 3, 15, 16, 12, 0)
"""


PROJECT_NAME = "hmomin/network_gym_client"
RUN_NAME = "checkpoint_SAC-N"
CREATED_AFTER = datetime(2024, 3, 14, 16, 34, 0)
CREATED_BEFORE = datetime(2024, 3, 14, 18, 29, 0)
MIN_RUNTIME = timedelta(minutes=1)
MAX_STEPS = -1
MIN_SEED = 100
TEST_EXPORT = False


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
        if MAX_STEPS > 0:
            run_rewards = run_rewards[0 : MAX_STEPS - 1]
        run_return: float = sum(run_rewards)
        run_name: str = run.name
        split_run_name = run_name.split("_seed_")
        base_run_name = split_run_name[0]
        seed = int(split_run_name[1])
        if seed < MIN_SEED:
            continue
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
    truncated_name = "".join(RUN_NAME.split("\\"))
    csv_filename = os.path.join(
        data_dir,
        f"{truncated_name}_{time.strftime('_%Y_%m_%d_%H_%M_%S', time.localtime())}.csv",
    )
    return csv_filename


def main() -> None:
    runs = get_runs()
    returns_dict = get_returns_from_runs(runs)
    pprint(returns_dict)
    if TEST_EXPORT:
        return
    write_to_csv(returns_dict)


if __name__ == "__main__":
    main()

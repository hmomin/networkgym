import csv
import os
import time
import wandb
from datetime import datetime, timedelta
from pprint import pprint
from tqdm import tqdm


"""
NOTE (times should be GMT!):
- system_default
- PPO
- SAC
    CREATED_AFTER = datetime(2023, 11, 9, 22, 44, 0)
    CREATED_BEFORE = datetime(2023, 11, 11, 22, 44, 0)

- system_default_deterministic_walk_bc\.10000\.64\.\d\.\d\d\d\.normalized
- system_default_deterministic_walk_bc\.10000\.64\.\d\.\d\d\d\.not_normalized
- PPO_deterministic_walk_bc\.10000\.64\.\d\.\d\d\d\.normalized
- PPO_deterministic_walk_bc\.10000\.64\.\d\.\d\d\d\.not_normalized
- SAC_deterministic_walk_bc\.10000\.64\.\d\.\d\d\d\.normalized
- SAC_deterministic_walk_bc\.10000\.64\.\d\.\d\d\d\.not_normalized
    CREATED_AFTER = datetime(2023, 10, 30, 4, 21, 0)
    CREATED_BEFORE = datetime(2023, 11, 9, 22, 44, 0)

<TD3+BC offline agents tested>
- PPO_eps_0.0
- PPO_eps_0.1
- PPO_eps_0.3
- PPO_eps_1.0
    CREATED_AFTER = datetime(2023, 11, 18, 23, 49, 0)
    CREATED_BEFORE = datetime(2023, 11, 19, 10, 17, 0)

<actual PPO agent tested with epsilon-greedy actions>
NOTE: all the run names saved as PPO - check for run_name below
- PPO
    CREATED_AFTER = datetime(2023, 11, 19, 19, 50, 0)
    CREATED_BEFORE = datetime(2023, 11, 19, 21, 40, 0)

<actual PPO agent with deterministic=False in evaluate()>
- PPO
    CREATED_AFTER = datetime(2023, 11, 20, 5, 50, 0)
    CREATED_BEFORE = datetime(2023, 11, 20, 6, 0, 0)

<actual PPO agent tested with epsilon-greedy actions>
- SAC (eps 0.0)
    CREATED_AFTER = datetime(2023, 11, 26, 17, 24, 0)
    CREATED_BEFORE = datetime(2023, 11, 26, 17, 27, 0)
- SAC (eps 0.1)
    CREATED_AFTER = datetime(2023, 11, 26, 17, 45, 0)
    CREATED_BEFORE = datetime(2023, 11, 26, 17, 47, 0)

<offline RL agents on SAC data>
- SAC_eps_0.0
    CREATED_AFTER = datetime(2023, 11, 27, 1, 41, 0)
    CREATED_BEFORE = datetime(2023, 11, 27, 15, 47, 0)
    
- SAC (eps 0.0 - stochastic policy)
    CREATED_AFTER = datetime(2023, 11, 27, 18, 23, 0)
    CREATED_BEFORE = datetime(2023, 11, 27, 18, 26, 0)

<offline RL agents on PPO data>
- PPO_eps_0.6
    CREATED_AFTER = datetime(2023, 11, 30, 22, 8, 0)
    CREATED_BEFORE = datetime(2023, 12, 1, 20, 41, 0)

- PPO (eps 0.6 actual behavior policy)
    CREATED_AFTER = datetime(2023, 12, 4, 18, 1, 0)
    CREATED_BEFORE = datetime(2023, 12, 4, 18, 5, 0)

- system_default_deterministic_walk_PTD3 (beta 1.0, ridge_lambda 1.0e-3)
    CREATED_AFTER = datetime(2023, 12, 4, 18, 1, 0)
    CREATED_BEFORE = datetime(2024, 1, 11, 0, 20, 0)

- system_default (seed 256)
    CREATED_AFTER = datetime(2024, 1, 11, 0, 20, 0)
    CREATED_BEFORE = datetime(2024, 1, 11, 2, 20, 0)

- system_default_deterministic_walk_PTD3 (beta 3.0, ridge_lambda 1.0e-3)
    CREATED_AFTER = datetime(2024, 1, 14, 6, 2, 0)
    CREATED_BEFORE = datetime(2024, 1, 14, 18, 56, 0)

- system_default_deterministic_walk_PTD3 (beta 10.0, ridge_lambda 1.0e-3)
    CREATED_AFTER = datetime(2024, 1, 14, 20, 53, 0)
    CREATED_BEFORE = datetime(2024, 1, 15, 7, 57, 0)

- system_default_deterministic_walk_PTD3 (beta 1.0, pessimism all the way through)
    CREATED_AFTER = datetime(2024, 1, 17, 20, 22, 0)
    CREATED_BEFORE = datetime(2024, 1, 18, 17, 25, 0)

- system_default_deterministic_walk_PTD3_beta_1.0_alpha_0.999 (new iterative "SGD" scheme for Sigma matrix)
    CREATED_AFTER = datetime(2024, 1, 18, 18, 30, 0)
    CREATED_BEFORE = datetime(2024, 1, 19, 1, 11, 0)

- system_default_deterministic_walk_PTD3_beta_1.0_alpha_0.995
    CREATED_AFTER = datetime(2024, 1, 19, 1, 54, 0)
    CREATED_BEFORE = datetime(2024, 1, 19, 20, 11, 0)

- system_default_deterministic_walk_PTD3_beta_1.0_alpha_0.9995 (seed 256)
    CREATED_AFTER = datetime(2024, 1, 21, 7, 34, 0)
    CREATED_BEFORE = datetime(2024, 1, 22, 1, 57, 0)

- system_default_deterministic_walk_PTD3_beta_1.0_alpha_0.9995 (seed 257)
    CREATED_AFTER = datetime(2024, 1, 22, 6, 43, 0)
    CREATED_BEFORE = datetime(2024, 1, 22, 16, 43, 0)
    
- system_default_seed_257 (seed 257)
    CREATED_AFTER = datetime(2024, 1, 22, 19, 42, 0)
    CREATED_BEFORE = datetime(2024, 1, 22, 19, 44, 0)
    
- system_default_deterministic_walk_PTD3_beta_1.0_alpha_1.0 (seed 256)
    CREATED_AFTER = datetime(2024, 1, 23, 5, 55, 0)
    CREATED_BEFORE = datetime(2024, 1, 23, 16, 36, 0)
    
- system_default_deterministic_walk_PTD3_beta_1.0 (pessimism all the way through)
    CREATED_AFTER = datetime(2024, 1, 24, 4, 29, 0)
    CREATED_BEFORE = datetime(2024, 1, 24, 19, 27, 0)

- system_default
    CREATED_AFTER = datetime(2024, 1, 26, 1, 58, 0)
    CREATED_BEFORE = datetime(2024, 1, 26, 2, 1, 0)

- ArgMax
    CREATED_AFTER = datetime(2024, 1, 26, 1, 58, 0)
    CREATED_BEFORE = datetime(2024, 1, 26, 5, 47, 0)

- ArgMin
    CREATED_AFTER = datetime(2024, 1, 26, 1, 58, 0)
    CREATED_BEFORE = datetime(2024, 1, 26, 5, 47, 0)

- Random
    CREATED_AFTER = datetime(2024, 1, 26, 1, 58, 0)
    CREATED_BEFORE = datetime(2024, 1, 26, 5, 47, 0)

- system_default_deterministic_td3_bc_10000_alpha_0.000
    CREATED_AFTER = datetime(2024, 1, 26, 1, 58, 0)
    CREATED_BEFORE = datetime(2024, 1, 26, 5, 47, 0)

- UtilityLogistic
    CREATED_AFTER = datetime(2024, 1, 26, 1, 58, 0)
    CREATED_BEFORE = datetime(2024, 1, 26, 5, 47, 0)

- system_default_deterministic_td3_bc_10000_alpha_0.625
    CREATED_AFTER = datetime(2024, 1, 26, 6, 16, 0)
    CREATED_BEFORE = datetime(2024, 1, 27, 17, 8, 0)

- system_default_deterministic_walk_PTD3 (some of them)
    CREATED_AFTER = datetime(2024, 1, 26, 6, 16, 0)
    CREATED_BEFORE = datetime(2024, 1, 27, 17, 8, 0)

- system_default_deterministic_walk_PTD3 (all the other runs)
    CREATED_AFTER = datetime(2024, 1, 29, 2, 47, 0)
    CREATED_BEFORE = datetime(2024, 1, 29, 17, 10, 0)

- PPO (deterministic)
    CREATED_AFTER = datetime(2024, 1, 29, 5, 50, 0)
    CREATED_BEFORE = datetime(2024, 1, 29, 6, 57, 0)

- SAC (deterministic)
    CREATED_AFTER = datetime(2024, 1, 29, 5, 50, 0)
    CREATED_BEFORE = datetime(2024, 1, 29, 6, 57, 0)

- PPO (stochastic)
    CREATED_AFTER = datetime(2024, 1, 29, 17, 35, 0)
    CREATED_BEFORE = datetime(2024, 1, 29, 18, 9, 0)

- SAC (stochastic)
    CREATED_AFTER = datetime(2024, 1, 29, 17, 35, 0)
    CREATED_BEFORE = datetime(2024, 1, 29, 18, 9, 0)

- system_default_deterministic_walk_PTD3_beta_10.0_alpha_0.9999_step_0010000
    CREATED_AFTER = datetime(2024, 1, 30, 5, 1, 0)
    CREATED_BEFORE = datetime(2024, 1, 30, 5, 5, 0)

- system_default_deterministic_walk_PTD3_beta_10.0_alpha_1.0_step_0010000
    CREATED_AFTER = datetime(2024, 1, 30, 5, 31, 0)
    CREATED_BEFORE = datetime(2024, 1, 30, 5, 34, 0)

- system_default_deterministic_walk_PTD3_beta_10.0_alpha_0.999_step_0010000
    CREATED_AFTER = datetime(2024, 1, 31, 3, 35, 0)
    CREATED_BEFORE = datetime(2024, 1, 31, 3, 38, 0)

- system_default_deterministic_walk_PTD3_beta_10.0_alpha_0.9995_step_0010000
    CREATED_AFTER = datetime(2024, 1, 31, 4, 6, 0)
    CREATED_BEFORE = datetime(2024, 1, 31, 4, 9, 0)

- system_default_deterministic_walk_PTD3_beta_1.0_alpha_0.0_step_0010000
    CREATED_AFTER = datetime(2024, 2, 1, 23, 30, 0)
    CREATED_BEFORE = datetime(2024, 2, 2, 0, 5, 0)

- system_default_deterministic_walk_PTD3_beta_3.0_alpha_0.0_step_0010000
    CREATED_AFTER = datetime(2024, 2, 1, 23, 30, 0)
    CREATED_BEFORE = datetime(2024, 2, 2, 0, 5, 0)

- system_default_deterministic_walk_PTD3_beta_..._alpha_0.0_step_0010000
    CREATED_AFTER = datetime(2024, 2, 2, 18, 19, 0)
    CREATED_BEFORE = datetime(2024, 2, 2, 18, 54, 0)

- system_default_deterministic_walk_PTD3_beta_10.0_alpha_0.0_step_0010000
    CREATED_AFTER = datetime(2024, 2, 2, 23, 36, 0)
    CREATED_BEFORE = datetime(2024, 2, 2, 23, 40, 0)

- PessimisticLSPI (no pessimism during training)
    CREATED_AFTER = datetime(2024, 2, 3, 6, 32, 0)
    CREATED_BEFORE = datetime(2024, 2, 4, 2, 38, 0)

- random_discrete
    CREATED_AFTER = datetime(2024, 2, 5, 19, 47, 0)
    CREATED_BEFORE = datetime(2024, 2, 5, 22, 38, 0)

- PessimisticLSPI_random
    CREATED_AFTER = datetime(2024, 2, 7, 5, 36, 0)
    CREATED_BEFORE = datetime(2024, 2, 7, 16, 59, 0)
"""


PROJECT_NAME = "hmomin/network_gym_client"
RUN_NAME = "PessimisticLSPI_random"
CREATED_AFTER = datetime(2024, 2, 7, 5, 36, 0)
CREATED_BEFORE = datetime(2024, 2, 7, 16, 59, 0)
MIN_RUNTIME = timedelta(minutes=1)
MAX_STEPS = -1
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
    truncated_name = "".join(RUN_NAME.split("\\"))
    csv_filename = os.path.join(
        data_dir,
        f"{truncated_name}_{time.strftime('_%Y_%m_%d_%H_%M_%S', time.localtime())}.csv",
    )
    return csv_filename


def main() -> None:
    runs = get_runs()
    if TEST_EXPORT:
        return
    returns_dict = get_returns_from_runs(runs)
    pprint(returns_dict)
    write_to_csv(returns_dict)


if __name__ == "__main__":
    main()

import argparse
import torch
from offline_env import OfflineEnv


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="For changing hyperparameters in coverage_dataset.py"
    )
    parser.add_argument(
        "--env_name",
        help="algorithm with offline buffers to train from",
        required=False,
        default="sys_default_norm_utility",
        type=str,
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="whether to do feature normalization or not",
        required=False,
        default=False,
    )
    args = parser.parse_args()
    return args


def get_augmented_states(states: torch.Tensor) -> torch.Tensor:
    valid_indices = []
    invalid_indices = []
    for column_idx in range(0, states.shape[1]):
        state_values = states[:, column_idx]
        unique_values: torch.Tensor = state_values.unique()
        if unique_values.shape[0] > 1:
            valid_indices.append(column_idx)
        else:
            invalid_indices.append(column_idx)
    print(f"invalid state indices: {invalid_indices}")
    augmented_states = states[:, valid_indices]
    print(f"augmented_states.shape: {augmented_states.shape}")
    return augmented_states


def compute_population_covariance_matrix(
    states: torch.Tensor, actions: torch.Tensor
) -> torch.Tensor:
    state_action_rows = torch.cat([states, actions], dim=1)
    state_action_columns = state_action_rows.T
    L = state_action_columns.shape[0]
    population_covariance_matrix = (1 / L) * state_action_columns @ state_action_rows
    return population_covariance_matrix


def main() -> None:
    args = get_args()
    env_name: str = args.env_name
    normalize: bool = args.normalize

    env = OfflineEnv(env_name, -1, normalize)

    # compute the coverage of the state-action space using the offline dataset
    buffer = env.buffer
    states = buffer.tensor_states
    actions = buffer.tensor_actions
    augmented_states = get_augmented_states(states)
    population_covariance_matrix = compute_population_covariance_matrix(
        augmented_states, actions
    )
    eigenvalues = torch.linalg.eigvals(population_covariance_matrix)
    if torch.sum(torch.isreal(eigenvalues)) != population_covariance_matrix.shape[0]:
        raise Exception("Complex eigenvalues in symmetric matrix detected...")
    eigenvalues = torch.real(eigenvalues)
    min_eigenvalue = torch.min(eigenvalues).item()
    max_eigenvalue = torch.max(eigenvalues).item()
    condition_number = torch.linalg.cond(population_covariance_matrix)
    # print(f"max_eigenvalue: {max_eigenvalue}")
    print(f"min_eigenvalue: {min_eigenvalue}")
    print(f"condition_number: {condition_number}")


if __name__ == "__main__":
    main()

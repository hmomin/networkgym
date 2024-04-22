import argparse
import torch
from agent import PessimisticLSPI
from offline_env import OfflineEnv


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="For changing hyperparameters in train_offline.py"
    )
    parser.add_argument(
        "--env_name",
        help="algorithm with offline buffers to train from",
        required=False,
        default="random_discrete_increment_utility",
        type=str,
    )
    parser.add_argument(
        "--beta",
        help="value of beta to use for pessimism weighting",
        required=False,
        default=1.0,
        type=float,
    )
    args = parser.parse_args()
    return args


def main() -> None:
    # ------------------------- HYPERPARAMETERS -------------------------
    buffer_max_size = 10_000
    args = get_args()
    env_name: str = args.env_name
    beta: int = args.beta

    env = OfflineEnv(env_name, buffer_max_size, normalize=False)

    agent = PessimisticLSPI(env, 4, beta)

    agent.LSTDQ_update()
    print(f"max value in w_tilde: {torch.max(agent.w_tilde)}")
    print(f"min value in w_tilde: {torch.min(agent.w_tilde)}")
    agent.save()


if __name__ == "__main__":
    main()

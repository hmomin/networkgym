import argparse
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
        default="utility_discrete_increment_utility",
        type=str,
    )
    parser.add_argument(
        "--observation_power",
        help="maximum power of observation values in state featurization",
        required=False,
        default=1,
        type=int,
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
    num_actions = 3 ** 4
    # -------------------------------------------------------------------
    args = get_args()
    env_name: str = args.env_name
    observation_power: int = args.observation_power
    beta: int = args.beta

    env = OfflineEnv(env_name, buffer_max_size, normalize=False)

    agent = PessimisticLSPI(env, observation_power, num_actions, beta)
    
    agent.LSTDQ_update()
    


if __name__ == "__main__":
    main()

import argparse
from agent import LSPI_BC
from tqdm import tqdm
from offline_env import OfflineEnv


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="For changing hyperparameters in train_offline.py"
    )
    parser.add_argument(
        "--env_name",
        help="algorithm with offline buffers to train from",
        required=False,
        default="system_default",
        type=str,
    )
    parser.add_argument(
        "--alpha",
        help="value of alpha to use for Q-loss weighting",
        required=False,
        default=0.0,
        type=float,
    )
    args = parser.parse_args()
    return args


def main() -> None:
    # ------------------------- HYPERPARAMETERS -------------------------
    buffer_max_size = 10_000
    num_actions = 625
    training_steps = 10_000  # number of mini-batch steps for training
    learning_rate = 0.001  # learning rate for actor and critic networks
    mini_batch_size = 100  # how large a mini-batch should be when updating
    resume = False  # resume from previous checkpoint if possible?
    # -------------------------------------------------------------------
    args = get_args()
    env_name: str = args.env_name
    alpha_bc: float = args.alpha

    env = OfflineEnv(env_name, buffer_max_size, normalize=False)

    agent = LSPI_BC(env, num_actions, learning_rate, resume)

    print("Training agent with offline data...")
    for _ in range(training_steps):
        agent.update(mini_batch_size, 0.0)
    agent.save(training_steps)
    for _ in range(training_steps):
        agent.update(mini_batch_size, alpha_bc)
    agent.save(2 * training_steps)


if __name__ == "__main__":
    main()

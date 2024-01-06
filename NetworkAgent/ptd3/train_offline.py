import argparse
from agent import PessimisticTD3
from tqdm import tqdm
from offline_env import OfflineEnv


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="For changing hyperparameters in train_offline.py"
    )
    parser.add_argument(
        "--beta",
        help="beta >= 0 (larger beta means more pessimism)",
        required=False,
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--env_name",
        help="algorithm with offline buffers to train from",
        required=False,
        default="system_default_deterministic_walk",
        type=str,
    )
    args = parser.parse_args()
    return args


def main() -> None:
    # ------------------------- HYPERPARAMETERS -------------------------
    buffer_max_size = 10_000
    training_steps = 10_000  # number of mini-batch steps for training
    save_step = int(training_steps / 100)  # how frequently models should update
    gamma = 0.99  # discount factor for rewards
    learning_rate = 0.001  # learning rate for actor and critic networks
    tau = 0.005  # tracking parameter used to update target networks slowly
    training_sigma = 0.2  # contributes noise to target actions
    training_clip = 0.5  # clips target actions to keep them close to true actions
    mini_batch_size = 100  # how large a mini-batch should be when updating
    policy_delay = 2  # how many steps to wait before updating the policy
    resume = False  # resume from previous checkpoint if possible?
    # -------------------------------------------------------------------
    args = get_args()
    beta_pessimism: float = args.beta
    env_name: str = args.env_name

    env = OfflineEnv(env_name, buffer_max_size, normalize=False)

    agent = PessimisticTD3(env, learning_rate, gamma, tau, resume)

    print("Training agent with offline data...")
    for step in range(training_steps):
        pessimism = step >= training_steps // 2
        should_update_policy = step % policy_delay == 0
        beta = beta_pessimism if pessimism else 0.0
        agent.update(
            mini_batch_size, training_sigma, training_clip, should_update_policy, beta
        )
        if step % save_step == 0:
            print(f"iteration: {step:5d} | beta_pessimism: {beta}")
            agent.save(step)
    agent.save(training_steps)


if __name__ == "__main__":
    main()

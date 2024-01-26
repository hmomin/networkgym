import argparse
from agent import PessimisticTD3
from offline_env import OfflineEnv


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="For changing hyperparameters in train_offline.py"
    )
    parser.add_argument(
        "--env_name",
        help="algorithm with offline buffers to train from",
        required=False,
        default="system_default_deterministic_walk",
        type=str,
    )
    parser.add_argument(
        "--alpha",
        help="alpha ∈ [0, 1] is the discount factor to use when calculating updates to the Fisher information matrix. Larger values of alpha would promote stability/smoothness in updates to the matrix, but make use of more obsolete information. Smaller values of alpha make the matrix change more reactively, but take advantage of less information.",
        required=False,
        default=0.999,
        type=float,
    )
    parser.add_argument(
        "--beta",
        help="beta >= 0 is the uncertainty estimate multiplier. Larger values of beta mean actor updates with more pessimism baked into them.",
        required=False,
        default=1.0,
        type=float,
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
    env_name: str = args.env_name
    alpha_fisher: float = args.alpha
    beta_pessimism: float = args.beta

    env = OfflineEnv(env_name, buffer_max_size, normalize=False)

    agent = PessimisticTD3(
        env,
        beta_pessimism,
        alpha_fisher,
        learning_rate,
        gamma,
        tau,
        policy_delay,
        resume,
    )

    print("Training agent with offline data...")
    for step in range(training_steps):
        agent.update(mini_batch_size, training_sigma, training_clip)
        if step % save_step == 0:
            agent.save(step)
    agent.save(training_steps)


if __name__ == "__main__":
    main()

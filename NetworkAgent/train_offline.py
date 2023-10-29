import argparse
from td3_bc.agent import Agent
from tqdm import tqdm
from offline_env import OfflineEnv


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="For changing hyperparameters in train_offline.py"
    )
    parser.add_argument(
        "--alpha",
        help="alpha >= 0 (larger alpha means less behavioral cloning influence)",
        required=False,
        default=0.625,
        type=float,
    )
    args = parser.parse_args()
    return args


def main() -> None:
    # ------------------------- HYPERPARAMETERS -------------------------
    env_name = "PPO_95_test_10000_step_buffers"
    buffer_max_size = 10_000
    behavioral_cloning = True  # whether or not to include behavioral cloning
    training_steps = 10_000  # number of mini-batch steps for training
    save_step = training_steps  # int(training_steps / 100)  # how frequently models should update
    gamma = 0.99  # discount factor for rewards
    learning_rate = 0.001  # learning rate for actor and critic networks
    tau = 0.005  # tracking parameter used to update target networks slowly
    training_sigma = 0.2  # contributes noise to target actions
    training_clip = 0.5  # clips target actions to keep them close to true actions
    mini_batch_size = 100  # how large a mini-batch should be when updating
    policy_delay = 2  # how many steps to wait before updating the policy
    resume = False  # resume from previous checkpoint if possible?
    # -------------------------------------------------------------------

    env = OfflineEnv(env_name, buffer_max_size)

    args = get_args()
    alpha_bc: float = args.alpha

    agent = Agent(env, learning_rate, gamma, tau, alpha_bc, behavioral_cloning, resume)

    print("Training agent with offline data...")
    for step in tqdm(range(training_steps)):
        should_update_policy = step % policy_delay == 0
        agent.update(
            mini_batch_size, training_sigma, training_clip, should_update_policy
        )
        if step % save_step == 0:
            agent.save(step, training_steps)
    agent.save(training_steps, training_steps)


if __name__ == "__main__":
    main()

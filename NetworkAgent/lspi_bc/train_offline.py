import argparse
from agent import KL_BC
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
    args = parser.parse_args()
    return args


def main() -> None:
    # ------------------------- HYPERPARAMETERS -------------------------
    buffer_max_size = 10_000
    num_actions = 625
    training_steps = 10_000  # number of mini-batch steps for training
    save_step = training_steps  # int(training_steps / 100)  # how frequently models should update
    learning_rate = 0.001  # learning rate for actor and critic networks
    mini_batch_size = 100  # how large a mini-batch should be when updating
    resume = False  # resume from previous checkpoint if possible?
    # -------------------------------------------------------------------
    args = get_args()
    env_name: str = args.env_name

    env = OfflineEnv(env_name, buffer_max_size, normalize=False)

    agent = KL_BC(env, num_actions, learning_rate, resume)

    print("Training agent with offline data...")
    for step in tqdm(range(training_steps)):
        agent.update(mini_batch_size)
        # if step % save_step == 0:
        #     agent.save(step, training_steps)
    agent.save(training_steps, training_steps)


if __name__ == "__main__":
    main()

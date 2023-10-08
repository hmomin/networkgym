from td3_bc.agent import Agent
from tqdm import tqdm
from offline_env import OfflineEnv


def main() -> None:
    # ------------------------- HYPERPARAMETERS -------------------------
    behavioral_cloning = False  # whether or not to include behavioral cloning
    training_steps = 100_000  # number of mini-batch steps for training
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

    env = OfflineEnv("system_default")

    agent = Agent(env, learning_rate, gamma, tau, behavioral_cloning, resume)

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

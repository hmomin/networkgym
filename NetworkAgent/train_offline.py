from td3_bc.agent import Agent
from offline_env import OfflineEnv


def main() -> None:
    # ------------------------- HYPERPARAMETERS -------------------------
    behavioral_cloning = True  # whether or not to include behavioral cloning
    training_steps = 1_000_000  # number of mini-batch steps for training
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

    for step in range(training_steps):
        print(f"Step {step}:")
        should_update_policy = step % policy_delay == 0
        agent.update(
            mini_batch_size, training_sigma, training_clip, should_update_policy
        )
        if step % 100 == 0:
            agent.save()
    agent.save()


if __name__ == "__main__":
    main()

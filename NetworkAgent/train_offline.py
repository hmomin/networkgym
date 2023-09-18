from td3.agent import Agent
from offline_env import OfflineEnv

# HYPERPARAMETERS BELOW
training_steps = 1_000_000  # number of mini-batch steps for training
gamma = 0.99  # discount factor for rewards
learning_rate = 0.001  # learning rate for actor and critic networks
tau = 0.005  # tracking parameter used to update target networks slowly
action_sigma = 0.1  # contributes noise to deterministic policy output
training_sigma = 0.2  # contributes noise to target actions
training_clip = 0.5  # clips target actions to keep them close to true actions
mini_batch_size = 100  # how large a mini-batch should be when updating
policy_delay = 2  # how many steps to wait before updating the policy
resume = False  # resume from previous checkpoint if possible?
# FIXME: implement behavioral cloning!
behavioral_cloning = False  # whether or not to include behavioral cloning


def main() -> None:
    env = OfflineEnv("system_default")
    # FIXME: how is agent gonna get the mini-batches?
    mini_batch = env.get_mini_batch(100)

    agent = Agent(env, learning_rate, gamma, tau, resume)
    agent.save()

    for step in range(training_steps):
        should_update_policy = step % policy_delay == 0
        agent.buffer = env.buffer
        agent.update(
            mini_batch_size, training_sigma, training_clip, should_update_policy
        )
        agent.save()


if __name__ == "__main__":
    main()

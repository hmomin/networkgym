import csv
import gym
from agent import Agent
from os import path

# FIXME LOW: a lot of this needs to be reworked if we want to do online training with it.
# For offline training, see train_offline.py one level up.

# HYPERPARAMETERS BELOW
gamma = 0.99  # discount factor for rewards
learning_rate = 0.001  # learning rate for actor and critic networks
tau = 0.005  # tracking parameter used to update target networks slowly
action_sigma = 0.1  # contributes noise to deterministic policy output
training_sigma = 0.2  # contributes noise to target actions
training_clip = 0.5  # clips target actions to keep them close to true actions
mini_batch_size = 100  # how large a mini-batch should be when updating
policy_delay = 2  # how many steps to wait before updating the policy
resume = False  # resume from previous checkpoint if possible?


def main() -> None:
    env = gym.make(env_name)
    env.name = env_name + "_" + str(trial)
    csv_name = env.name + "-data.csv"
    agent = Agent(env, learning_rate, gamma, tau, resume)
    state = env.reset()
    step = 0
    running_reward = None

    # determine the last episode if we have saved training in progress
    num_episode = 0
    if path.exists(csv_name):
        file_data = list(csv.reader(open(csv_name)))
        last_line = file_data[-1]
        num_episode = int(last_line[0])

    while num_episode <= 2000:
        # choose an action from the agent's policy
        action = agent.get_noisy_action(state, action_sigma)
        # take a step in the environment and collect information
        next_state, reward, done, info = env.step(action)
        # store data in buffer
        agent.buffer.store(state, action, reward, next_state, done)

        if done:
            num_episode += 1
            # evaluate the deterministic agent on a test episode
            sum_rewards = 0
            state = env.reset()
            done = False
            while not done:
                action = agent.get_deterministic_action(state)
                next_state, reward, done, info = env.step(action)
                state = next_state
                sum_rewards += reward
            state = env.reset()
            # keep a running average to see how well we're doing
            running_reward = (
                sum_rewards
                if running_reward is None
                else running_reward * 0.99 + sum_rewards * 0.01
            )
            # log progress in csv file
            fields = [num_episode, sum_rewards, running_reward]
            with open(env.name + "-data.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(fields)
            agent.save()
            # print episode tracking
            print(
                f"episode {num_episode:6d} --- "
                + f"total reward: {sum_rewards:7.2f} --- "
                + f"running average: {running_reward:7.2f}",
                flush=True,
            )
        else:
            state = next_state
        step += 1

        should_update_policy = step % policy_delay == 0
        agent.update(
            mini_batch_size, training_sigma, training_clip, should_update_policy
        )


if __name__ == "__main__":
    main()

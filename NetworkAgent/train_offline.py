import csv
import gym
from td3.agent import Agent
from offline_env import OfflineEnv
from os import path

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
    env = OfflineEnv("system_default")
    mini_batch = env.get_mini_batch(100)


if __name__ == "__main__":
    main()

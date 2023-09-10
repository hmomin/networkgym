from momin.buffer import Buffer
from momin.offline_q_learning.q_agent import OfflineQAgent

# HYPERPARAMETERS BELOW
buffer_name = "FIXME"  # name of buffer to perform offline learning on
gamma = 0.99  # discount factor for rewards
learning_rate = 3e-4  # learning rate for q-function gradient step
mini_batch_size = 100  # how large a mini-batch should be when updating
num_iterations = 100  # how many iterations of training to perform
num_gradient_steps = 100  # how many gradient descent steps to take per iteration
resume = True  # resume from previous checkpoint if possible?


def main() -> None:
    buffer = Buffer(buffer_name)
    agent = OfflineQAgent(learning_rate=learning_rate, gamma=gamma, should_load=resume)
    for _ in range(num_iterations):
        # FIXME: create a new agent with the same parameters as the current agent
        for _ in range(num_gradient_steps):
            mini_batch = buffer.get_mini_batch(mini_batch_size)
            # FIXME: estimate the MSE - be careful with which parameters to use!

            # FIXME: update parameters with gradient descent
            pass
        # FIXME: set current agent's parameters with new agent's parameters


if __name__ == "__main__":
    main()

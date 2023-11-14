from network_gym_client.env import load_config_file
from network_gym_client.parallel_env import ParallelEnv


def main() -> None:
    config_json = load_config_file("nqos_split")
    env = ParallelEnv(config_json)
    states = env.reset()
    print(states.shape)


if __name__ == "__main__":
    main()

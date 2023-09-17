from offline_env import OfflineEnv


def main() -> None:
    env = OfflineEnv("system_default")
    mini_batch = env.get_mini_batch(100)


if __name__ == "__main__":
    main()

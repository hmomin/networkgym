# NOTE: the goal of this script is to compare the performance of FastLSPI with
# PPO in a discrete action space setting

import gymnasium as gym
from agent import FastLSPI
from stable_baselines3 import PPO


def main() -> None:
    # env = gym.make("LunarLander-v2", render_mode=None)
    env = gym.make("CartPole-v1", render_mode=None)

    # ppo_agent = PPO("MlpPolicy", env)
    # for _ in range(20):
    #     state, info = env.reset()
    #     terminated = truncated = False
    #     episode_return = 0.0
    #     while not terminated and not truncated:
    #         action, _ = ppo_agent.predict(state, deterministic=True)
    #         state, reward, terminated, truncated, info = env.step(action)
    #         episode_return += reward
    #     print(f"EPISODE RETURN: {episode_return}")
    # ppo_agent = ppo_agent.learn(42_000, progress_bar=True)
    # for _ in range(20):
    #     state, info = env.reset()
    #     terminated = truncated = False
    #     episode_return = 0.0
    #     while not terminated and not truncated:
    #         action, _ = ppo_agent.predict(state, deterministic=True)
    #         state, reward, terminated, truncated, info = env.step(action)
    #         episode_return += reward
    #     print(f"EPISODE RETURN: {episode_return}")

    fastlspi_agent = FastLSPI(env.observation_space.shape[0], env.action_space.n)
    state, info = env.reset()
    episode_return = 0.0
    for _ in range(42_000):
        action, _ = fastlspi_agent.predict(state, deterministic=False)
        next_state, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        fastlspi_agent.update(state, action, reward, next_state)
        if terminated or truncated:
            state, info = env.reset()
            print(f"EPISODE RETURN: {episode_return} - BUFFER SIZE: {fastlspi_agent.L}")
            episode_return = 0.0
        else:
            state = next_state

    env.close()


if __name__ == "__main__":
    main()

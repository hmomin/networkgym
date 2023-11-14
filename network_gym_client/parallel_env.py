import numpy as np
from copy import deepcopy
from .env import Env
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from threading import Thread


# FIXME: this script doesn't work with Stable Baselines, because there are some
# step() dependencies missing. It turns out that DummyVecEnv can actually get the job
# done pretty easily, albeit at the cost of speed...


def threaded_reset(observation_dict: dict[int, np.ndarray], idx: int, env: Env) -> None:
    obs, info = env.reset()
    observation_dict[idx] = obs


def threaded_step(
    actions: np.ndarray,
    observation_dict: dict[int, np.ndarray],
    reward_dict: dict[int, float],
    idx: int,
    env: Env,
) -> None:
    observation, reward, terminated, truncated, info = env.step(action)


class ParallelEnv(DummyVecEnv):
    def __init__(self, config_json: dict, client_ids: list = list(range(8))) -> None:
        self.envs: list[Env] = []
        self.observations: list[np.ndarray] = []
        self.rewards: list[float] = [0.0] * len(client_ids)
        for client_id in client_ids:
            new_config = deepcopy(config_json)
            new_config["env_config"]["random_seed"] += client_id
            self.envs.append(Env(client_id, new_config))

    def reset(self) -> np.ndarray:
        # NOTE: assumption is that an environment will never reset after starting
        # (infinite horizon case)
        observation_dict: dict[int, np.ndarray] = {}
        threads: list[Thread] = []
        for idx, env in enumerate(self.envs):
            thread = Thread(target=threaded_reset, args=(observation_dict, idx, env))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        for idx in range(len(observation_dict)):
            obs = observation_dict[idx].flatten()
            self.observations.append(obs)
        return np.vstack(self.observations)

    # def step_wait(self):
    #     pass

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        print("ACTIONS SHAPE")
        print(actions.shape)
        raise Exception("STOP")
        observation_dict: dict[int, np.ndarray] = {}
        reward_dict: dict[int, float] = {}
        threads: list[Thread] = []
        for idx, env in enumerate(self.envs):
            thread = Thread(
                target=threaded_step,
                args=(actions, observation_dict, reward_dict, idx, env),
            )
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        for idx in range(len(observation_dict)):
            obs = observation_dict[idx].flatten()
            self.observations[idx] = obs
            reward = reward_dict[idx]
            self.rewards[idx] = reward
        observations = np.vstack(self.observations)
        rewards = np.array(self.rewards)
        dones = np.array([False for _ in range(len(self.rewards))])
        return observations, rewards, dones, {}

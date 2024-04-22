#Copyright(C) 2023 Intel Corporation
#SPDX-License-Identifier: Apache-2.0
#File : env.py

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Box
import pandas as pd

# NOTE: importing for buffer and previous action recording
from NetworkAgent.buffer import Buffer
from NetworkAgent.full_observation import get_previous_action
from NetworkAgent.discrete_action_util import (
    convert_discrete_increment_action_to_continuous,
    convert_discrete_ratio_action_to_continuous
)

import time
import importlib
import pathlib
import json
import sys
import os
from network_gym_client.northbound_interface import NorthBoundClient
np.set_printoptions(precision=3)

FILE_PATH = pathlib.Path(__file__).parent

def load_config_file(env_name):
    """Load the environment configuration file

    Args:
        env_name (str): the environment name, e.g., `nqos_split`, `qos_steer`, or `network_slicing`

    Returns:
        json: the configuration paramters
    """
    #load config files
    FILE_PATH = pathlib.Path(__file__).parent

    # check available env folders          
    env_list = os.listdir(FILE_PATH / 'envs/')
    if '__init__.py' in env_list: env_list.remove('__init__.py')
    if '__pycache__' in env_list: env_list.remove('__pycache__')
    # prints all files

    #common_config.json is shared by all environments
    f = open(FILE_PATH / 'common_config.json')
    common_config_json = json.load(f)
    
    if env_name in env_list:

        #load the environment dependent config file
        file_name = 'envs/' +env_name + '/config.json'
        f = open(FILE_PATH / file_name)
    else:
        sys.exit("[Error] Cannot find env: '" + env_name + "'. Available Env: " + str(env_list))


    env_dependent_config_json = json.load(f)
    config_json = {**common_config_json, **env_dependent_config_json}
    config_json['env_config']['env'] = env_name
    return config_json

class Env(gym.Env):
    """Custom NetworkGym Environment that follows gym interface."""

    def __init__(self, id, config_json):
        """Initilize Env.

        Args:
            id (int): the client ID number
            config_json (json): configuration file
        """
        super().__init__()
        
        self.use_discrete_increment_actions: bool =\
            config_json["rl_config"]["use_discrete_increment_actions"]
        self.use_discrete_ratio_actions: bool =\
            config_json["rl_config"]["use_discrete_ratio_actions"]
        if self.use_discrete_increment_actions and self.use_discrete_ratio_actions:
            raise Exception(
                "ERROR: can't use discrete increment and ratio actions together!"
            )

        if config_json['session_name'] == 'test':
            print('***[WARNING]*** You are using the default "test" to connect to the server, which may conflict with the simulations launched by other users.')
            print('***[WARNING]*** Please change the "session_name" attribute in the common_config.json file to your assigned session name.')
        
        if 'GMA' in config_json['env_config']:
            #check if the measurement interval for all measurements are the same.
            if (config_json['env_config']['GMA']['measurement_interval_ms'] + config_json['env_config']['GMA']['measurement_guard_interval_ms']
                == config_json['env_config']['Wi-Fi']['measurement_interval_ms'] + config_json['env_config']['Wi-Fi']['measurement_guard_interval_ms']
                == config_json['env_config']['LTE']['measurement_interval_ms'] + config_json['env_config']['LTE']['measurement_guard_interval_ms']):
                config_json['env_config']['measurement_interval_ms'] = config_json['env_config']['GMA']['measurement_interval_ms']
                config_json['env_config']['measurement_guard_interval_ms'] = config_json['env_config']['GMA']['measurement_guard_interval_ms']
            else:
                print(config_json['env_config']['GMA']['measurement_interval_ms'])
                print(config_json['env_config']['GMA']['measurement_guard_interval_ms'])
                print(config_json['env_config']['Wi-Fi']['measurement_interval_ms'])
                print(config_json['env_config']['Wi-Fi']['measurement_guard_interval_ms'])
                print(config_json['env_config']['LTE']['measurement_interval_ms'])
                print(config_json['env_config']['LTE']['measurement_guard_interval_ms'])
                sys.exit('[Error!] The value of GMA, Wi-Fi, and LTE measurement_interval_ms + measurement_guard_interval_ms should be the same!')


        self.steps_per_episode = int(config_json['env_config']['steps_per_episode'])
        if self.steps_per_episode < 2:
            sys.exit('In crease the "steps_per_episode", the min value is 2!')
        self.episodes_per_session = int(config_json['env_config']['episodes_per_session'])
        # NOTE: need to include num_users
        self.num_users = int(config_json['env_config']['num_users'])

        step_length = config_json['env_config']['measurement_interval_ms']
        if 'measurement_guard_interval_ms' in config_json['env_config']:
            step_length = step_length + config_json['env_config']['measurement_guard_interval_ms']


        # compute the simulation time based on setting
        config_json['env_config']['env_end_time_ms'] = int(config_json['env_config']['measurement_start_time_ms'] + step_length * (self.steps_per_episode * self.episodes_per_session + 1))
        print("Environment duration: " + str(config_json['env_config']['env_end_time_ms']) + " ms")
        #Define config params
        module_path = 'network_gym_client.envs.'+config_json['env_config']['env']+'.adapter'
        module = importlib.import_module(module_path, package=None)
        self.adapter = module.Adapter(config_json)

        self.enable_rl_agent = True
        if config_json['rl_config']['agent']=="system_default":
            self.enable_rl_agent = False

        self.action_space = self.adapter.get_action_space()
        self.observation_space = self.adapter.get_observation_space()

        self.northbound_interface_client = NorthBoundClient(id, config_json) #initial northbound_interface_client

        #self.link_type = config_json['rl_config']['link_type'] 
        self.current_step = 0
        self.current_ep = 0
        self.first_episode = True
        self.last_action = np.array([])
        
        # NOTE: recording rl_alg and setting up buffer
        self.rl_alg: str = config_json["rl_config"]["agent"]
        random_seed: int = config_json["env_config"]["random_seed"]
        seed_str = str(random_seed).zfill(2)
        reward_type: str = config_json["rl_config"]["reward_type"]
        self.store_offline: bool = config_json["rl_config"]["store_offline"]\
            if "store_offline" in config_json["rl_config"]\
                else False
        if self.store_offline:
            filename: str = f"{self.rl_alg}_{reward_type}_seed_{seed_str}"
            self.buffer = Buffer(filename)

        
    def reset(self, seed=None, options=None):
        """Resets the environment to an initial internal state, returning an initial observation and info.

        Args:
            seed (optional int): The seed that is used to initialize the environment's PRNG (`np_random`).
                If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
                a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
                However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
                If you pass an integer, the PRNG will be reset even if it already exists.
                Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
                Please refer to the minimal example above to see this paradigm in action.
            options (optional dict): Additional information to specify how the environment is reset (optional,
                depending on the specific environment)

        Returns:
            observation (ObsType): Observation of the initial state.
            info (dictionary):  This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
        """
        self.current_step = 1
        self.current_ep += 1

        print("reset() at episode:" + str(self.current_ep) + ", step:" + str(self.current_step))

        # if a new simulation starts (first episode) in the reset function, we need to connect to server
        # else a new episode of the same simulation.
            # do not need to connect, send action directly
        if self.first_episode:
            self.northbound_interface_client.connect()
        else:
            policy = self.adapter.get_policy(self.last_action) # calling this function updates the timestamp
            self.northbound_interface_client.send(policy) #send network policy to network gym server

        network_stats = self.northbound_interface_client.recv()#first measurement

        observation = self.adapter.get_observation(network_stats)
        if (observation.shape != self.adapter.get_observation_space().shape):
            sys.exit("The shape of the observation and self.get_observation is not the same!")
        # print(observation.shape)
        # NOTE: need to record previous state
        if self.store_offline:
            self.previous_state = observation.astype(np.float32)
        # NOTE: need to record previous split ratio
        self.previous_split_ratio = get_previous_action(network_stats)
        return observation.astype(np.float32), {"network_stats": network_stats}

    def step(self, agent_action):
        """Run one timestep of the environment's dynamics using the agent actions.

        Get action lists from RL agent and send to network gym server
        Get measurements from gamsim and obs and reward
        Check if it is the last step in the episode
        Return obs,reward,done,info

        Args:
            agent_action (ActType): an action provided by the agent to update the environment state.

        Returns:
            observation (ObsType): An element of the environment's `observation_space` as the next observation due to the agent actions.
            reward (SupportsFloat): The reward as a result of taking the action.
            terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
                which can be positive or negative. An example is reaching the goal state or moving into the lava from
                the Sutton and Barton, Gridworld. If true, the user needs to call :meth:`reset`.
            truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
                Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
                Can be used to end the episode prematurely before a terminal state is reached.
                If true, the user needs to call :meth:`reset`.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging). {one-way delay, raw observation, and termination flag} 
            done (bool): (Deprecated) A boolean value for if the episode has ended, in which case further :meth:`step` calls will
                return undefined results. This was removed in OpenAI Gym v26 in favor of terminated and truncated attributes.
                A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully,
                a certain timelimit was exceeded, or the physics simulation has entered an invalid state.
        """
        # NOTE: this is to account for the discrete action space
        if self.use_discrete_increment_actions:
            action = convert_discrete_increment_action_to_continuous(
                agent_action, self.previous_split_ratio
            )
            self.previous_split_ratio = action
        elif self.use_discrete_ratio_actions:
            action = convert_discrete_ratio_action_to_continuous(
                agent_action, len(self.previous_split_ratio)
            )
            self.previous_split_ratio = action
        else:
            action = agent_action
        self.current_step += 1
        
        print("----------| step() at episode:" + str(self.current_ep) + ", step:" + str(self.current_step) + " |----------")

        #1.) Get action from RL agent and send to network gym server
        if not self.enable_rl_agent or (hasattr(action, "size") and action.size == 0):
            #empty action
            policy = self.adapter.get_policy(np.array([]))#send empty action to network gym server
            self.northbound_interface_client.send(policy) #send network policy to network gym server
        else:
            # TODO: we need to have the same action format... e.g., [0, 1]
            if (hasattr(action, "shape") and action.shape != self.adapter.get_action_space().shape):
                sys.exit("The shape of the observation and self.get_observation is not the same!")
            self.last_action = action
            policy = self.adapter.get_policy(action)
            self.northbound_interface_client.send(policy) #send network policy to network gym server

        #2.) Get measurements from gamsim and obs and reward
        network_stats = self.northbound_interface_client.recv()
        
        observation = self.adapter.get_observation(network_stats)

        # NOTE: system_default doesn't always show split_ratio
        if self.store_offline and self.rl_alg == "system_default":
            # NOTE: this action should be the previous action based off of the current
            # state
            action = get_previous_action(network_stats)

        # with np.printoptions(precision=4, suppress=True):
        #     print(observation)

        #Get reward
        reward = self.adapter.get_reward(network_stats)


        self.adapter.wandb_log()

        #3.) Check end of Episode
        truncated = self.current_step >= self.steps_per_episode

        # print("Episdoe", self.current_ep ,"step", self.current_step, "reward", reward, "Done", done)

        terminated = False
        if truncated:
            if self.first_episode:
                self.first_episode = False

            if self.current_ep == self.episodes_per_session:
                terminated = True
                self.northbound_interface_client.close()
                time.sleep(1) # sleep 1 second to let the server disconnect client and env worker. In case the client restart connection right after a env termination.
        #4.) return observation, reward, done, info
        if terminated or truncated:
            print("Episode End ---> terminated:" + str(terminated) + " truncated:" + str(truncated))
        #print("Episode End ---> terminated:" + str(terminated) + " truncated:" + str(truncated))
        
        # NOTE: storing to buffer
        if self.store_offline:
            self.buffer.store(
                self.previous_state,
                action,
                reward,
                observation.astype(np.float32)
            )
            if terminated or truncated:
                self.buffer.write_to_disk()
            self.previous_state = observation.astype(np.float32)
        
        return observation.astype(np.float32), reward, terminated, truncated, {"network_stats": network_stats}

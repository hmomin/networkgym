import argparse
import numpy as np
import os
import sys
import time

import sys
sys.path.append('.')
sys.path.append('../')
sys.path.append('../../')

from network_gym_client import load_config_file
from network_gym_client import Env as NetworkGymEnv
from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from gymnasium.wrappers import NormalizeObservation

from momin.heuristic_policies import argmax_policy, argmin_policy, random_policy

def train(agent, config_json):

    steps_per_episode = int(config_json['env_config']['steps_per_episode'])
    episodes_per_session = int(config_json['env_config']['episodes_per_session'])
    num_steps = steps_per_episode*episodes_per_session
    
    try:
        model = agent.learn(total_timesteps=num_steps)
    finally:
        # NOTE: adding timestamp to tell different models apart!
        agent.save(
            "models/"
            + config_json['rl_config']['agent']
            + time.strftime("_%Y_%m_%d_%H_%M_%S", time.localtime())
        )

    #TODD Terminate the RL agent when the simulation ends.


def system_default_policy(env, config_json):

    steps_per_episode = int(config_json['env_config']['steps_per_episode'])
    episodes_per_session = int(config_json['env_config']['episodes_per_session'])
    num_steps = steps_per_episode*episodes_per_session

    truncated = True # episode end
    terminated = False # episode end and simulation end
    for step in range(num_steps+10):

        # if simulation end, exit
        if terminated:
            print("simulation end")
            break

        # If the epsiode is up, then start another one
        if truncated:
            print("new episode")
            obs = env.reset()


        
        action = np.array([])#no action from the rl agent

        # apply the action
        obs, reward, terminated, truncated, info = env.step(action)
        #print(obs)


def evaluate(model, env, num_steps):
    done = True
    for _ in range(num_steps):
        if done:
            obs = env.reset()
        if type(obs) == tuple:
            obs = obs[0]
        # at this point, obs should be a numpy array
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)


def main():

    args = arg_parser()

    #load config files
    config_json = load_config_file(args.env)
    config_json['env_config']['env'] = args.env

    if args.lte_rb !=-1:
        config_json['env_config']['LTE']['resource_block_num'] = args.lte_rb

    if config_json['rl_config']['agent'] == "":
        config_json['rl_config']['agent']  = 'system_default'

    if not config_json['env_config']['respond_action_after_measurement']:
        sys.exit('[Error!] RL agent must set "respond_action_after_measurement" to true !')
    
    rl_alg = config_json['rl_config']['agent'] 

    config = {
        "policy_type": "MlpPolicy",
        "env_id": "network_gym_client",
        "RL_algo" : rl_alg
    }

    alg_map = {
        'PPO': PPO,
        'DDPG': DDPG,
        'SAC': SAC,
        'TD3': TD3,
        'A2C': A2C,
        'ArgMax': argmax_policy,	
        'ArgMin': argmin_policy,	
        'Random': random_policy,
        'system_default': system_default_policy,
    }

    # Choose the agent
    agent_class = alg_map.get(rl_alg, None)
    if agent_class is None:
        raise ValueError(f"Invalid RL algorithm name: {rl_alg}")
    client_id = args.client_id
    # Create the environment
    print("[" + args.env + "] environment selected.")
    env = NetworkGymEnv(client_id, config_json) # make a network env using pass client id, adatper and configure file arguements.
    # NOTE: disabling normalization
    normal_obs_env = env
    # normal_obs_env = NormalizeObservation(env)

    # It will check your custom environment and output additional warnings if needed
    # only use this function for debug, 
    # check_env(env)

    if rl_alg != "system_default":
        
        if rl_alg == "ArgMax":	
            argmax_policy(env, config_json)	
            return	
        if rl_alg == "ArgMin":	
            argmin_policy(env, config_json)	
            return	
        if rl_alg == "Random":	
            random_policy(env, config_json)	
            return

        train_flag = config_json['rl_config']['train']

        # Load the model if eval is True
        if not train_flag:
            # Testing/Evaluation
            this_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(this_dir, "models", rl_alg)
            agent = agent_class.load(model_path)
            
            steps_per_episode = int(config_json['env_config']['steps_per_episode'])
            episodes_per_session = int(config_json['env_config']['episodes_per_session'])
            num_steps = steps_per_episode*episodes_per_session
            # n_episodes = config_json['rl_config']['timesteps'] / 100

            evaluate(agent, normal_obs_env, num_steps)
        else:
            # NOTE: action noise for off-policy networks - num users hardcoded!
            if rl_alg in ["DDPG", "TD3"]:
                action_noise = NormalActionNoise(
                    mean=np.zeros(4), sigma=0.3 * np.ones(4)
                )
                agent = agent_class(config_json['rl_config']['policy'], normal_obs_env, action_noise=action_noise, verbose=1)
            else:
                agent = agent_class(config_json['rl_config']['policy'], normal_obs_env, verbose=1)
            print(agent.policy)
            
            train(agent, config_json)
    else:
        #use the system_default algorithm...
        agent_class(normal_obs_env, config_json)
        
def arg_parser():
    parser = argparse.ArgumentParser(description='Network Gym Client')
    parser.add_argument('--env', type=str, required=True, choices=['nqos_split', 'qos_steer', 'network_slicing'],
                        help='Select a environment to start Network Gym Client (nqos_split, qos_steer, network_slicing)')
    parser.add_argument('--client_id', type=int, required=False, default=0,
                        help='Select client id to start simulation')
    parser.add_argument('--lte_rb', type=int, required=False, default=-1,
                        help='Select number of LTE Resource Blocks')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
    time.sleep(1)
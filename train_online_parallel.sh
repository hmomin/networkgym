#!/bin/bash

# NOTE: trains PPO and SAC on parallel envs

# echo "Cleaning up broken environments"
# python -u NetworkAgent/config_changer.py --test --agent PPO --seed 0 --steps 2
# cd NetworkAgent/stable-baselines3
# python -u main_rl.py --env nqos_split --client_id 0
# python -u main_rl.py --env nqos_split --client_id 1
# python -u main_rl.py --env nqos_split --client_id 2
# python -u main_rl.py --env nqos_split --client_id 3
# python -u main_rl.py --env nqos_split --client_id 4
# python -u main_rl.py --env nqos_split --client_id 5
# python -u main_rl.py --env nqos_split --client_id 6
# python -u main_rl.py --env nqos_split --client_id 7
# cd ../..

# echo "Training PPO"
# python -u NetworkAgent/config_changer.py --train --parallel_env --agent PPO --seed 16 --steps 80000
# cd NetworkAgent/stable-baselines3
# python -u main_rl.py --env nqos_split --client_id 0
# cd ../..

# echo "Cleaning up broken environments"
# python -u NetworkAgent/config_changer.py --test --agent PPO --seed 0 --steps 2
# cd NetworkAgent/stable-baselines3
# python -u main_rl.py --env nqos_split --client_id 0
# python -u main_rl.py --env nqos_split --client_id 1
# python -u main_rl.py --env nqos_split --client_id 2
# python -u main_rl.py --env nqos_split --client_id 3
# python -u main_rl.py --env nqos_split --client_id 4
# python -u main_rl.py --env nqos_split --client_id 5
# python -u main_rl.py --env nqos_split --client_id 6
# python -u main_rl.py --env nqos_split --client_id 7
# cd ../..

echo "Training SAC"
python -u NetworkAgent/config_changer.py --train --parallel_env --agent SAC --seed 16 --steps 80000
cd NetworkAgent/stable-baselines3
python -u main_rl.py --env nqos_split --client_id 0
cd ../..
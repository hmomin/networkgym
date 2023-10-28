#!/bin/bash

# agents=("system_default" "ArgMax" "ArgMin" "Random" "A2C" "PPO" "DDPG" "TD3" "SAC")
agents=("system_default" "ArgMax" "UtilityFull" "A2C" "PPO" "UtilityLogistic" "TD3" "SAC")

# exactly one argument must be provided
if [ $# -ne 1 ]; then
    echo "ERROR: expected usage -> bash $0 <client_id>"
    exit 1
fi
client_id=$1

# check if client_id is within valid range
if [ $client_id -ge 0 ] && [ $client_id -le 7 ]; then
    agent=${agents[$client_id]}
    echo "Selected Agent: $agent"
    python -u NetworkAgent/config_changer.py --agent $agent --train
    cd NetworkAgent/stable-baselines3
    python -u main_rl.py --env nqos_split --client_id $client_id
    cd ../..
else
    echo "ERROR: client_id must be an int between 0 and 7, inclusive. ($client_id provided)"
    exit 1
fi
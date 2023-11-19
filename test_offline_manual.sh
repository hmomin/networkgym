#!/bin/bash

set -e

# exactly one argument must be provided
if [ $# -ne 1 ]; then
    echo "ERROR: expected usage -> bash $0 <client_id>"
    exit 1
fi
client_id=$1

seed=135
agents=(
    PPO_eps_0.3_bc.10000.64.10.000.not_normalized
)

# check if client_id is within valid range
if [ $client_id -ge 0 ] && [ $client_id -le 7 ]; then
    for agent in "${agents[@]}"; do
        echo "SEED: $seed --- AGENT: $agent"
        python -u NetworkAgent/config_changer.py --test --agent $agent --seed $seed --steps 2000
        cd NetworkAgent/stable-baselines3
        python -u main_rl.py --env nqos_split --client_id $client_id
        cd ../..
    done
else
    echo "ERROR: client_id must be an int between 0 and 7, inclusive. ($client_id provided)"
    exit 1
fi
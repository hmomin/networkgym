#!/bin/bash

set -e

# exactly one argument must be provided
if [ $# -ne 1 ]; then
    echo "ERROR: expected usage -> bash $0 <client_id>"
    exit 1
fi
client_id=$1

let "seed = $client_id + 128"

eps_values=(0.1 1.0)
alphas=(0.000 0.100 0.300 1.000 3.000 10.000)
normalize_status=("not_normalized")

# check if client_id is within valid range
if [ $client_id -ge 0 ] && [ $client_id -le 7 ]; then
    for eps_value in "${eps_values[@]}"; do
        for normalize in "${normalize_status[@]}"; do
            for alpha in "${alphas[@]}"; do
                agent="PPO_eps_${eps_value}_bc.10000.64.${alpha}.${normalize}"
                echo "SEED: $seed --- AGENT: $agent"
                python -u NetworkAgent/config_changer.py --test --agent $agent --seed $seed --steps 2000
                cd NetworkAgent/stable-baselines3
                python -u main_rl.py --env nqos_split --client_id $client_id
                cd ../..
            done
        done
    done
else
    echo "ERROR: client_id must be an int between 0 and 7, inclusive. ($client_id provided)"
    exit 1
fi
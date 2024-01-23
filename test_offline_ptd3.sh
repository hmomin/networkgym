#!/bin/bash

# exactly one argument must be provided
if [ $# -ne 1 ]; then
    echo "ERROR: expected usage -> bash $0 <client_id>"
    exit 1
fi
client_id=$1
seed=256
start_value=$((client_id * 100))

# check if client_id is within valid range
if [ $client_id -ge 0 ] && [ $client_id -le 7 ]; then
    while [ $start_value -le 10000 ]
    do
        padded_value=$(printf "%07d" $start_value)
        agent="system_default_deterministic_walk_PTD3_beta_1.0_alpha_1.0_step_${padded_value}"
        echo "SEED: $seed --- AGENT: $agent"

        python -u NetworkAgent/config_changer.py --test --agent $agent --seed $seed --steps 3200
        cd NetworkAgent/stable-baselines3
        python -u main_rl.py --env nqos_split --client_id $client_id
        cd ../..

        start_value=$((start_value + 800))
    done
else
    echo "ERROR: client_id must be an int between 0 and 7, inclusive. ($client_id provided)"
    exit 1
fi
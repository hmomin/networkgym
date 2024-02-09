#!/bin/bash

agents=(
    sys_default_norm_utility_bc.10000.64.0.625.not_normalized
    sys_default_norm_utility_PTD3_beta_0.1_alpha_0.999_step_0010000
    sys_default_norm_utility_PTD3_beta_0.1_alpha_0.9995_step_0010000
)

# exactly one argument must be provided
if [ $# -ne 1 ]; then
    echo "ERROR: expected usage -> bash $0 <client_id>"
    exit 1
fi
client_id=$1

let "seed = $client_id + 128"

# check if client_id is within valid range
if [ $client_id -ge 0 ] && [ $client_id -le 7 ]; then
    for agent in "${agents[@]}"; do
        echo "SEED: $seed --- AGENT: $agent"
        python -u NetworkAgent/config_changer.py --test --agent $agent --seed $seed --steps 3200
        cd NetworkAgent/stable-baselines3
        python -u main_rl.py --env nqos_split --client_id $client_id
        cd ../..
    done
else
    echo "ERROR: client_id must be an int between 0 and 7, inclusive. ($client_id provided)"
    exit 1
fi
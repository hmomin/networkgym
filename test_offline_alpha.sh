#!/bin/bash

set -e

# exactly one argument must be provided
if [ $# -ne 1 ]; then
    echo "ERROR: expected usage -> bash $0 <client_id>"
    exit 1
fi
client_id=$1

let "seed = $client_id + 128"

algorithms=("PPO_95_test_10000_step_buffers" "system_default")
alphas=($(seq 0.000 0.125 2.000))

# check if client_id is within valid range
if [ $client_id -ge 0 ] && [ $client_id -le 7 ]; then
    for algorithm in "${algorithms[@]}"; do
        for alpha in "${alphas[@]}"; do
            agent="${algorithm}_bc.10000.64.$alpha"
            echo "SEED: $seed --- AGENT: $agent"
            python NetworkAgent/config_changer.py --test --agent $agent --seed $seed --steps 2000
            cd NetworkAgent/stable-baselines3
            python main_rl.py --env nqos_split --client_id $client_id
            cd ../..
        done
    done
else
    echo "ERROR: client_id must be an int between 0 and 7, inclusive. ($client_id provided)"
    exit 1
fi
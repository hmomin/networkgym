#!/bin/bash

set -e

# exactly one argument must be provided
if [ $# -ne 1 ]; then
    echo "ERROR: expected usage -> bash $0 <client_id>"
    exit 1
fi
client_id=$1

let "seed = $client_id + 128"

algorithm="PPO_95_test_20000_steps_buffers"
training=("bc" "normal")

# check if client_id is within valid range
if [ $client_id -ge 0 ] && [ $client_id -le 7 ]; then
    for train in "${training[@]}"; do
        for ((b=1; b<=64; b*=2)); do
            agent="${algorithm}_${train}.10000.$b"
            echo "SEED: $seed --- AGENT: $agent"
            python -u NetworkAgent/config_changer.py --test --agent $agent --seed $seed --steps 2000
            cd NetworkAgent/stable-baselines3
            python -u main_rl.py --env nqos_split --client_id $client_id
            cd ../..
        done
    done
else
    echo "ERROR: client_id must be an int between 0 and 7, inclusive. ($client_id provided)"
    exit 1
fi
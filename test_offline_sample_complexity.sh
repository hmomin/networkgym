#!/bin/bash

set -e

# exactly one argument must be provided
if [ $# -ne 1 ]; then
    echo "ERROR: expected usage -> bash $0 <client_id>"
    exit 1
fi
client_id=$1

let "seed = $client_id + 128"

steps=("20000" "50000")
training=("normal" "bc")

# check if client_id is within valid range
if [ $client_id -ge 0 ] && [ $client_id -le 7 ]; then
    for step in "${steps[@]}"; do
        for train in "${training[@]}"; do
            for ((b=1; b<=64; b*=2)); do
                agent="PPO_${step}_steps_training_${train}.10000.$b"
                echo "SEED: $seed --- AGENT: $agent"
                python NetworkAgent/config_changer.py --test --agent $agent --seed $seed --steps 2000
                cd NetworkAgent/stable-baselines3
                python main_rl.py --env nqos_split --client_id $client_id
                cd ../..
            done
        done
    done
else
    echo "ERROR: client_id must be an int between 0 and 7, inclusive. ($client_id provided)"
    exit 1
fi
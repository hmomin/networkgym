#!/bin/bash

# exactly one argument must be provided
if [ $# -ne 1 ]; then
    echo "ERROR: expected usage -> bash $0 <client_id>"
    exit 1
fi
client_id=$1

seed=256
agents=(
    system_default_deterministic_walk_PTD3.0005000.64
    system_default_deterministic_walk_PTD3.0005100.64
    system_default_deterministic_walk_PTD3.0005200.64
    system_default_deterministic_walk_PTD3.0005300.64
    system_default_deterministic_walk_PTD3.0005400.64
    system_default_deterministic_walk_PTD3.0005500.64
)

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
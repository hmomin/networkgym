#!/bin/bash

set -e

# exactly one argument must be provided
if [ $# -ne 1 ]; then
    echo "ERROR: expected usage -> bash $0 <client_id>"
    exit 1
fi
client_id=$1

seed=129
agents=(
    system_default_deterministic_walk_bc.10000.64.1.250.normalized
    system_default_deterministic_walk_bc.10000.64.1.375.normalized
    system_default_deterministic_walk_bc.10000.64.1.500.normalized
    system_default_deterministic_walk_bc.10000.64.1.625.normalized
    system_default_deterministic_walk_bc.10000.64.1.750.normalized
    system_default_deterministic_walk_bc.10000.64.1.875.normalized
    system_default_deterministic_walk_bc.10000.64.2.000.normalized
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
#!/bin/bash

# exactly one argument must be provided
if [ $# -ne 1 ]; then
    echo "ERROR: expected usage -> bash $0 <client_id>"
    exit 1
fi
client_id=$1
seed=256
start_value=$((client_id * 100))

while [ $start_value -le 10000 ]
do
    start_value=$((start_value + 800))
done

start_value=$((start_value - 800))

# check if client_id is within valid range
if [ $client_id -ge 0 ] && [ $client_id -le 7 ]; then
    while [ $start_value -ge 0 ]
    do
        padded_value=$(printf "%07d" $start_value)
        agent="system_default_deterministic_walk_PTD3.${padded_value}.64"
        echo "SEED: $seed --- AGENT: $agent"

        python -u NetworkAgent/config_changer.py --test --agent $agent --seed $seed --steps 5000
        cd NetworkAgent/stable-baselines3
        python -u main_rl.py --env nqos_split --client_id $client_id
        cd ../..

        start_value=$((start_value - 800))
    done
else
    echo "ERROR: client_id must be an int between 0 and 7, inclusive. ($client_id provided)"
    exit 1
fi
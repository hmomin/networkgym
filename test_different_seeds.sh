#!/bin/bash

agents=(
    system_default
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
        # time python script time, to determine if it should be reran
        start_time=$(date +%s)
        python -u main_rl.py --env nqos_split --client_id $client_id
        end_time=$(date +%s)
        elapsed_time=$((end_time - start_time))
        cd ../..
        if [ $elapsed_time -lt 60 ]; then
            echo "RERUNNING SCRIPT..."
            python -u NetworkAgent/config_changer.py --test --agent $agent --seed $seed --steps 3200
            cd NetworkAgent/stable-baselines3
            python -u main_rl.py --env nqos_split --client_id $client_id
            cd ../..
        fi
    done
else
    echo "ERROR: client_id must be an int between 0 and 7, inclusive. ($client_id provided)"
    exit 1
fi
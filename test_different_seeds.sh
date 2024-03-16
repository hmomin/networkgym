#!/bin/bash

agents=(
    # checkpoint_CQL_utility_logistic
    checkpoint_EDAC_utility_logistic
    checkpoint_IQL_utility_logistic
    checkpoint_LB_SAC_utility_logistic
    checkpoint_SAC_N_utility_logistic
    checkpoint_TD3_BC_utility_logistic
)

# exactly one argument must be provided
if [ $# -ne 1 ]; then
    echo "ERROR: expected usage -> bash $0 <client_id>"
    exit 1
fi
client_id=$1

let "seed1 = $client_id + 128"
let "seed2 = $client_id + 128 + 8"
let "seed3 = $client_id + 128 + 16"
let "seed4 = $client_id + 128 + 24"
seeds=(
    $seed1
    $seed2
    $seed3
    $seed4
)
echo "${seeds[@]}"

# check if client_id is within valid range
if [ $client_id -ge 0 ] && [ $client_id -le 7 ]; then
    for agent in "${agents[@]}"; do
        for seed in "${seeds[@]}"; do
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
    done
else
    echo "ERROR: client_id must be an int between 0 and 7, inclusive. ($client_id provided)"
    exit 1
fi
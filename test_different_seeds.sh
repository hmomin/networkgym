#!/bin/bash

agents=(
    # throughput_argmin
    # random
    # system_default
    # throughput_argmax
    # utility_logistic
    # sys_default_norm_utility_td3_bc.10000.64.0.625.not_normalized
    # utility_logistic_norm_utility_td3_bc.10000.64.0.625.not_normalized
    # PPO
    # SAC

    throughput_argmax_norm_utility_PTD3_beta_300.0_alpha_1.0_step_0010000
    throughput_argmax_norm_utility_td3_bc.10000.64.0.625.not_normalized
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
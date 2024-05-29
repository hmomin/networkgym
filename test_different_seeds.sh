#!/bin/bash

agents=(
    # DONE
    # system_default
    # utility_logistic
    # throughput_argmax
    
    # sys_default_norm_utility_PTD3_beta_10.0_alpha_1.0_step_0010000
    # throughput_argmax_norm_utility_PTD3_beta_10.0_alpha_1.0_step_0010000
    # utility_logistic_norm_utility_PTD3_beta_10.0_alpha_1.0_step_0010000

    # checkpoint_BC_thrpt_argmax_no_norm

    # checkpoint_IQL_sys_default_no_norm
    # checkpoint_IQL_utility_logistic_no_norm
    # checkpoint_IQL_thrpt_argmax_no_norm

    # checkpoint_TD3_BC_sys_default_no_norm
    # checkpoint_TD3_BC_utility_logistic_no_norm
    # checkpoint_TD3_BC_thrpt_argmax_no_norm

    # checkpoint_BC_sys_default_no_norm
    # checkpoint_BC_sys_default_norm
    # checkpoint_BC_utility_logistic_no_norm
    # checkpoint_BC_utility_logistic_norm
    # checkpoint_BC_thrpt_argmax_norm
    # checkpoint_CQL_sys_default_no_norm
    # checkpoint_CQL_sys_default_norm
    # checkpoint_CQL_utility_logistic_no_norm
    # checkpoint_CQL_utility_logistic_norm
    # checkpoint_CQL_thrpt_argmax_no_norm
    # checkpoint_CQL_thrpt_argmax_norm
    # checkpoint_EDAC_sys_default
    # checkpoint_EDAC_utility_logistic
    # checkpoint_EDAC_thrpt_argmax
    # checkpoint_IQL_sys_default_norm
    # checkpoint_IQL_utility_logistic_norm

    # IN PROGRESS
    
    checkpoint_IQL_thrpt_argmax_norm
    checkpoint_LB-SAC_sys_default
    checkpoint_LB-SAC_thrpt_argmax
    checkpoint_LB-SAC_utility_logistic
    checkpoint_SAC-N_sys_default
    checkpoint_SAC-N_thrpt_argmax
    checkpoint_SAC-N_utility_logistic
    checkpoint_TD3_BC_sys_default_norm
    checkpoint_TD3_BC_thrpt_argmax_norm
    checkpoint_TD3_BC_utility_logistic_norm
)

# exactly one argument must be provided
if [ $# -ne 1 ]; then
    echo "ERROR: expected usage -> bash $0 <client_id>"
    exit 1
fi
client_id=$1

# let "seed1 = $client_id + 128"
# let "seed2 = $client_id + 128 + 8"
# let "seed3 = $client_id + 128 + 16"
# let "seed4 = $client_id + 128 + 24"
let "seed5 = $client_id + 128 + 32"
seeds=(
    # $seed1
    # $seed2
    # $seed3
    # $seed4
    $seed5
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
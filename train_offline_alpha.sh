#!/bin/bash

set -e

env_names=("system_default_dfp" "PPO_dfp_95_test_10000_step_buffers")
alphas=($(seq 0.000 0.125 2.000))

for env_name in "${env_names[@]}"; do
    for alpha in "${alphas[@]}"; do
        echo "ENV_NAME: $env_name - ALPHA: $alpha"
        python -u NetworkAgent/train_offline.py --env_name $env_name --alpha $alpha
    done
done
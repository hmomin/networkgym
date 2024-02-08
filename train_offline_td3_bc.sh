#!/bin/bash

set -e

env_names=("sys_default_norm_utility")
alphas=(0.625)

for env_name in "${env_names[@]}"; do
    for alpha in "${alphas[@]}"; do
        echo "python -u NetworkAgent/td3_bc/train_offline.py --env_name $env_name --alpha $alpha"
        python -u NetworkAgent/td3_bc/train_offline.py --env_name $env_name --alpha $alpha
    done
done
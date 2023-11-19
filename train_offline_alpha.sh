#!/bin/bash

set -e

eps_values=(0.1 1.0)
alphas=(0.000 0.100 0.300 1.000 3.000 10.000)

for eps in "${eps_values[@]}"; do
    for alpha in "${alphas[@]}"; do
        env_name="PPO_eps_${eps}"
        echo "ENV_NAME: $env_name - ALPHA: $alpha"
        python -u NetworkAgent/train_offline.py --env_name $env_name --alpha $alpha
    done
done
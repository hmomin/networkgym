#!/bin/bash

set -e

# alphas=(0.000 0.100 0.300 1.000 3.000 10.000)
alphas=(0.1)

env_name="PPO_discrete_ratio_action_space_625"
echo "ENV_NAME: $env_name"

for alpha in "${alphas[@]}"; do
    echo "ALPHA: $alpha"
    python -u NetworkAgent/lspi_bc/train_offline.py --env_name $env_name --alpha $alpha
done
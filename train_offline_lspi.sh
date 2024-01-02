#!/bin/bash

set -e

alphas=(0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00)

env_name="random_discrete_increment_utility"
echo "ENV_NAME: $env_name"

for alpha in "${alphas[@]}"; do
    echo "ALPHA: $alpha"
    python -u NetworkAgent/lspi_bc/train_offline.py --env_name $env_name --alpha $alpha
done
#!/bin/bash

set -e

env_names=(
    "random_discrete_increment_utility"
    "utility_discrete_increment_utility"
)
betas=(0.0 0.1 0.3 1.0 3.0 10.0)

for env_name in "${env_names[@]}"; do
    for beta in "${betas[@]}"; do
        echo "python -u NetworkAgent/pessimistic_lspi/train_offline.py --env_name $env_name --beta $beta"
        python -u NetworkAgent/pessimistic_lspi/train_offline.py --env_name $env_name --beta $beta
    done
done
#!/bin/bash

set -e

env_names=(
    "random_discrete_increment_utility"
    "utility_discrete_increment_utility"
)
betas=(0.0 0.1 0.3 1.0 3.0 10.0)
obs_powers=(1 2 3)

for env_name in "${env_names[@]}"; do
    for beta in "${betas[@]}"; do
        for obs_power in "${obs_powers[@]}"; do
            echo "python -u NetworkAgent/pessimistic_lspi/train_offline.py --env_name $env_name --beta $beta --obs_power $obs_power"
            python -u NetworkAgent/pessimistic_lspi/train_offline.py --env_name $env_name --beta $beta --obs_power $obs_power
        done
    done
done
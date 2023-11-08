#!/bin/bash

set -e

env_names=("SAC_deterministic_walk")
alphas=($(seq 0.000 0.125 2.000))

for env_name in "${env_names[@]}"; do
    for alpha in "${alphas[@]}"; do
        echo "ENV_NAME: $env_name - ALPHA: $alpha"
        python -u NetworkAgent/train_offline.py --env_name $env_name --alpha $alpha --normalize
    done
done
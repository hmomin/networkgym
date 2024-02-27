#!/bin/bash

set -e

betas=(0.1)
alphas=(0.0)

for beta in "${betas[@]}"; do
    for alpha in "${alphas[@]}"; do
        echo "python -u NetworkAgent/ptd3/train_offline.py --env_name throughput_argmax_norm_utility --beta $beta --alpha $alpha"
        python -u NetworkAgent/ptd3/train_offline.py --env_name throughput_argmax_norm_utility --beta $beta --alpha $alpha
    done
done
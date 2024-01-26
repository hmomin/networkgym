#!/bin/bash

set -e

betas=(1.0 0.3 0.1)
alphas=(0.999 0.9995 0.9999)

for beta in "${betas[@]}"; do
    for alpha in "${alphas[@]}"; do
        echo "python -u NetworkAgent/ptd3/train_offline.py --beta $beta --alpha $alpha"
        python -u NetworkAgent/ptd3/train_offline.py --beta $beta --alpha $alpha
    done
done
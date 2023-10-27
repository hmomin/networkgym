#!/bin/bash

set -e

alphas=($(seq 0.000 0.125 2.000))

for alpha in "${alphas[@]}"; do
    python -u NetworkAgent/train_offline.py --alpha $alpha
done
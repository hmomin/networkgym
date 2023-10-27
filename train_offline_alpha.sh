#!/bin/bash

set -e

alphas=(0.0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1.0 1.125 1.25 1.375 1.5 1.625 1.75 1.875 2.0)

for alpha in "${alphas[@]}"; do
    python -u NetworkAgent/train_offline.py --alpha $alpha
done
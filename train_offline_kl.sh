#!/bin/bash

set -e

env_name="PPO_discrete_ratio_action_space_625"
echo "ENV_NAME: $env_name"
python -u NetworkAgent/kl_bc/train_offline.py --env_name $env_name
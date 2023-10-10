#!/bin/bash

set -e

# exactly one argument must be provided
if [ $# -ne 1 ]; then
    echo "ERROR: expected usage -> bash $0 <client_id>"
    exit 1
fi
client_id=$1

# check if client_id is within valid range
if [ $client_id -ge 0 ] && [ $client_id -le 7 ]; then
    for ((seed = client_id + 48; seed <= client_id + 56; seed += 8)); do
        python NetworkAgent/config_changer.py --train --seed $seed --agent PPO --steps 20000
        cd NetworkAgent/stable-baselines3
        python main_rl.py --env nqos_split --client_id $client_id
        cd ../..
    done
else
    echo "ERROR: client_id must be an int between 0 and 7, inclusive. ($client_id provided)"
    exit 1
fi

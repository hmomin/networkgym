#!/bin/bash

set -e

# exactly one argument must be provided
if [ $# -ne 1 ]; then
    echo "ERROR: expected usage -> bash $0 <client_id>"
    exit 1
fi
client_id=$1

let "seed_1 = $client_id + 16"
let "seed_2 = $client_id + 24"

# check if client_id is within valid range
if [ $client_id -ge 0 ] && [ $client_id -le 7 ]; then
    echo "Selected seed_1: $seed_1"
    echo "Selected seed_2: $seed_2"
    python NetworkAgent/config_changer.py --train --seed $seed_1 --agent system_default --steps 100000
    cd NetworkAgent/stable-baselines3
    python main_rl.py --env nqos_split --client_id $client_id
    cd ../..
    python NetworkAgent/config_changer.py --train --seed $seed_2 --agent system_default --steps 100000
    cd NetworkAgent/stable-baselines3
    python main_rl.py --env nqos_split --client_id $client_id
    cd ../..
else
    echo "ERROR: client_id must be an int between 0 and 7, inclusive. ($client_id provided)"
    exit 1
fi
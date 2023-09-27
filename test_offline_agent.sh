#!/bin/bash

set -e

agent_basename="system_default_bc"
# define the range of starting values for the agent suffix (depending on client id)
start_vals=(690000 650000 600000 610000 630000 650000 620000 650000)

# exactly one argument must be provided
if [ $# -ne 1 ]; then
    echo "ERROR: expected usage -> bash $0 <client_id>"
    exit 1
fi
client_id=$1
start_val=${start_vals[$client_id]}
end_val=1000000
step=10000

let "seed = $client_id + 56"

# check if client_id is within valid range
if [ $client_id -ge 0 ] && [ $client_id -le 7 ]; then
    for ((i=start_val; i<=end_val; i+=step)); do
        agent="$agent_basename.$(printf "%07d" $i)"
        echo "SEED: $seed --- AGENT: $agent"
        python NetworkAgent/config_changer.py --test --agent $agent --seed $seed --steps 2000
        cd NetworkAgent/stable-baselines3
        python main_rl.py --env nqos_split --client_id $client_id
        cd ../..
    done
else
    echo "ERROR: client_id must be an int between 0 and 7, inclusive. ($client_id provided)"
    exit 1
fi
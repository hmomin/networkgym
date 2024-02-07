#!/bin/bash

agents=(
    PessimisticLSPI_random_discrete_increment_utility_beta_0.0
    PessimisticLSPI_random_discrete_increment_utility_beta_0.1
    PessimisticLSPI_random_discrete_increment_utility_beta_0.3
    PessimisticLSPI_random_discrete_increment_utility_beta_1.0
    PessimisticLSPI_random_discrete_increment_utility_beta_3.0
    PessimisticLSPI_random_discrete_increment_utility_beta_10.0
    PessimisticLSPI_random_discrete_increment_utility_beta_30.0
    PessimisticLSPI_utility_discrete_increment_utility_beta_0.0
    PessimisticLSPI_utility_discrete_increment_utility_beta_0.1
    PessimisticLSPI_utility_discrete_increment_utility_beta_0.3
    PessimisticLSPI_utility_discrete_increment_utility_beta_1.0
    PessimisticLSPI_utility_discrete_increment_utility_beta_3.0
    PessimisticLSPI_utility_discrete_increment_utility_beta_10.0
    PessimisticLSPI_utility_discrete_increment_utility_beta_30.0
)

# exactly one argument must be provided
if [ $# -ne 1 ]; then
    echo "ERROR: expected usage -> bash $0 <client_id>"
    exit 1
fi
client_id=$1

let "seed = $client_id + 128"

# check if client_id is within valid range
if [ $client_id -ge 0 ] && [ $client_id -le 7 ]; then
    for agent in "${agents[@]}"; do
        echo "SEED: $seed --- AGENT: $agent"
        python -u NetworkAgent/config_changer.py --test --agent $agent --seed $seed --steps 3200
        cd NetworkAgent/stable-baselines3
        python -u main_rl.py --env nqos_split --client_id $client_id
        cd ../..
    done
else
    echo "ERROR: client_id must be an int between 0 and 7, inclusive. ($client_id provided)"
    exit 1
fi
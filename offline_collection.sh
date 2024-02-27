#!/bin/bash

algorithm="throughput_argmax"

done_seeds=()

# exactly one argument must be provided
if [ $# -ne 1 ]; then
    echo "ERROR: expected usage -> bash $0 <client_id>"
    exit 1
fi
client_id=$1

# check if client_id is within valid range
if [ $client_id -ge 0 ] && [ $client_id -le 7 ]; then
    for ((seed = client_id + 0; seed <= client_id + 56; seed += 8)); do
        # don't run it if we've done the seed already
        seed_done=0
        for done_seed in "${done_seeds[@]}"; do
            if [[ "$seed" == "$done_seed" ]]; then
                seed_done=1
                break
            fi
        done

        if [[ $seed_done -eq 1 ]]; then
            echo "seed $seed done already."
        else
            echo "SEED: $seed --- AGENT: $algorithm"
            python -u NetworkAgent/config_changer.py --test --store_offline --seed $seed --agent $algorithm --steps 10000
            cd NetworkAgent/stable-baselines3
            # time python script time, to determine if it should be reran
            start_time=$(date +%s)
            python -u main_rl.py --env nqos_split --client_id $client_id
            end_time=$(date +%s)
            elapsed_time=$((end_time - start_time))
            cd ../..
            if [ $elapsed_time -lt 60 ]; then
                echo "RERUNNING SCRIPT..."
                python -u NetworkAgent/config_changer.py --test --store_offline --seed $seed --agent $algorithm --steps 10000
                cd NetworkAgent/stable-baselines3
                python -u main_rl.py --env nqos_split --client_id $client_id
                cd ../..
            fi

        fi
    done
else
    echo "ERROR: client_id must be an int between 0 and 7, inclusive. ($client_id provided)"
    exit 1
fi

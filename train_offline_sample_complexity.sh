#!/bin/bash

set -e

num_buffers=64

while [ $num_buffers -gt 0 ]; do
    python NetworkAgent/train_offline.py
    num_buffers=$((num_buffers / 2))
    
    for ((i = 0; i < num_buffers; i++)); do
        python NetworkAgent/move_buffer.py
    done
done
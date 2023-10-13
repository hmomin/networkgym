#!/bin/bash

set -e

for ((num_buffers = 64; num_buffers > 0; num_buffers -= 1)); do
    python NetworkAgent/train_offline.py
    python NetworkAgent/move_buffer.py
done
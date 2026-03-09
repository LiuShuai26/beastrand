#!/bin/bash
# Atari benchmark: 3 classic games × 1 seed × 10M steps
# Usage: bash run_atari_benchmark.sh

set -e

SEED=1
GAMES=("BreakoutNoFrameskip-v4" "PongNoFrameskip-v4" "SpaceInvadersNoFrameskip-v4")

for game in "${GAMES[@]}"; do
    echo "=== Starting $game seed=$SEED ==="
    python -m projects.atari.train \
        --env-id "$game" \
        --seed "$SEED" \
        --total-env-steps 10000000 \
        2>&1 | tee "train_logs/atari_${game}_s${SEED}.log"
    echo "=== Finished $game seed=$SEED ==="
done

echo "All Atari benchmarks complete."

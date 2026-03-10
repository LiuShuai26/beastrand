#!/bin/bash
# Humanoid-v4 benchmark: PPO and PPO-LSTM, seeds 42 / 123 / 7, 10M steps each
#
# Usage:
#   bash run_benchmark.sh           # run all 6 jobs sequentially
#   bash run_benchmark.sh ppo       # PPO only
#   bash run_benchmark.sh lstm      # PPO-LSTM only
#
# Logs written to train_logs/bench_<algo>_<seed>.log
# FPS and final reward are printed at the end of each run by the training loop.

set -e

SEEDS=(42 123 7)
STEPS=10000000
LOGDIR="train_logs"
mkdir -p "$LOGDIR"

MODE="${1:-all}"

run_ppo() {
    local seed=$1
    local tag="ppo_s${seed}"
    echo "=== PPO seed=${seed} ==="
    python -m projects.mujoco.train \
        --env-id Humanoid-v4 \
        --seed "$seed" \
        --total-env-steps "$STEPS" \
        --run-name "bench_${tag}" \
        2>&1 | tee "${LOGDIR}/bench_${tag}.log"
    echo "=== Done PPO seed=${seed} ==="
}

run_lstm() {
    local seed=$1
    local tag="lstm_s${seed}"
    echo "=== PPO-LSTM seed=${seed} ==="
    python -m projects.ppo_lstm.train \
        --env-id Humanoid-v4 \
        --seed "$seed" \
        --total-env-steps "$STEPS" \
        --run-name "bench_${tag}" \
        2>&1 | tee "${LOGDIR}/bench_${tag}.log"
    echo "=== Done PPO-LSTM seed=${seed} ==="
}

for seed in "${SEEDS[@]}"; do
    if [[ "$MODE" == "all" || "$MODE" == "ppo" ]]; then
        run_ppo "$seed"
    fi
    if [[ "$MODE" == "all" || "$MODE" == "lstm" ]]; then
        run_lstm "$seed"
    fi
done

echo ""
echo "=== All benchmark runs complete ==="
echo "Logs in ${LOGDIR}/bench_*.log"
echo "To view results: grep -E 'fps|reward|FPS' ${LOGDIR}/bench_*.log"

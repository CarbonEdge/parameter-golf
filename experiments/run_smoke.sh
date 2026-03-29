#!/bin/bash
# Smoke test all 4 experiments with 200 steps on a single GPU.
# Purpose: verify each script runs without errors before committing H100 time.
#
# Usage (from repo root on a GPU machine):
#   bash experiments/run_smoke.sh
#
# Requires: 1xH100 (or any CUDA GPU), sp1024 + sp4096 data downloaded.
# Data setup:
#   python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
#   python3 data/cached_challenge_fineweb.py --variant sp4096 --train-shards 1

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXPERIMENTS="$REPO_ROOT/experiments"
LOGS_DIR="$EXPERIMENTS/smoke_logs"
mkdir -p "$LOGS_DIR"

# Use the conda env that has PyTorch + CUDA
PYTHON="${PYTHON:-/c/Users/HomePc/.conda/envs/ltxvideo/python.exe}"

# On H100 with flash_attn3 installed, use torchrun for distributed.
# On local consumer GPU (no Triton/flash_attn), run directly:
#   - TORCHDYNAMO_DISABLE=1 skips torch.compile (requires Triton)
#   - TRAIN_SEQ_LEN/TRAIN_BATCH_TOKENS reduced to fit within 16GB VRAM using math attention
LOCAL_GPU="${LOCAL_GPU:-1}"  # set to 0 on H100 to use torchrun
if [ "$LOCAL_GPU" = "1" ]; then
    RUNNER="$PYTHON"
    LOCAL_ARGS="TORCHDYNAMO_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRAIN_SEQ_LEN=512 TRAIN_BATCH_TOKENS=65536 VAL_BATCH_SIZE=65536"
else
    RUNNER="$PYTHON -m torch.distributed.run --standalone --nproc_per_node=8"
    LOCAL_ARGS=""
fi

SMOKE_ARGS="ITERATIONS=200 MAX_WALLCLOCK_SECONDS=3600 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=50"

run_smoke() {
    local name="$1"
    local script="$2"
    local extra_env="$3"
    local log="$LOGS_DIR/${name}_smoke.log"

    echo "=== Smoke test: $name ==="
    env $LOCAL_ARGS $SMOKE_ARGS $extra_env \
        RUN_ID="smoke_${name}" \
        $RUNNER "$script" 2>&1 | tee "$log"

    if grep -q "val_bpb" "$log"; then
        echo "  PASS: val_bpb found in output"
    else
        echo "  WARN: val_bpb not found (may be normal for short smoke run)"
    fi
    echo ""
}

cd "$REPO_ROOT"

run_smoke "exp1_silu2"     "$EXPERIMENTS/exp1_silu2/train_gpt.py"     ""
run_smoke "exp2_prelu2"    "$EXPERIMENTS/exp2_prelu2/train_gpt.py"    ""
run_smoke "exp3_adam_ttt"  "$EXPERIMENTS/exp3_adam_ttt/train_gpt.py"  "TTT_ENABLED=0"  # disable TTT for smoke
run_smoke "exp4_4kseqlen"  "$EXPERIMENTS/exp4_4096bpe/train_gpt.py"   ""  # 4096 seq_len, uses sp1024 data

echo "=== All smoke tests complete. Logs in $LOGS_DIR ==="
echo "Check for errors with: grep -l 'Error\|Traceback' $LOGS_DIR/*.log"

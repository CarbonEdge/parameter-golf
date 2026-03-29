#!/bin/bash
# Full 8xH100 runs for all 4 experiments, sequential.
# Each run uses the full 10-minute wall clock.
#
# Usage (from repo root on 8xH100 RunPod):
#   bash experiments/run_h100_all.sh
#
# Data prerequisites:
#   python3 data/cached_challenge_fineweb.py --variant sp1024
#   python3 data/cached_challenge_fineweb.py --variant sp4096
#
# Results are logged to experiments/h100_logs/.
# After all runs: python3 experiments/compare_results.py

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXPERIMENTS="$REPO_ROOT/experiments"
LOGS_DIR="$EXPERIMENTS/h100_logs"
mkdir -p "$LOGS_DIR"

SEEDS="1337 42 2025"
NPROC=8

run_experiment() {
    local name="$1"
    local script="$2"
    local extra_env="$3"

    echo ""
    echo "=========================================="
    echo "  Experiment: $name"
    echo "  Script:     $script"
    echo "  Extra env:  $extra_env"
    echo "=========================================="

    for SEED in $SEEDS; do
        local log="$LOGS_DIR/${name}_seed${SEED}.log"
        echo "  Running seed $SEED -> $log"
        env $extra_env \
            SEED="$SEED" \
            RUN_ID="${name}_seed${SEED}" \
            MAX_WALLCLOCK_SECONDS=600 \
            torchrun --standalone --nproc_per_node=$NPROC "$script" 2>&1 | tee "$log"
        echo "  Done seed $SEED"
    done

    # Quick BPB summary
    echo ""
    echo "  Results for $name:"
    for SEED in $SEEDS; do
        local log="$LOGS_DIR/${name}_seed${SEED}.log"
        local bpb
        bpb=$(grep "val_bpb" "$log" | tail -1 | grep -oP '[0-9]+\.[0-9]+' | head -1 || echo "N/A")
        echo "    seed=$SEED  val_bpb=$bpb"
    done
}

cd "$REPO_ROOT"

# --- Experiment 1: SiLU² activation ---
run_experiment "exp1_silu2" \
    "$EXPERIMENTS/exp1_silu2/train_gpt.py" \
    "NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 ITERATIONS=9000 EVAL_STRIDE=64"

# --- Experiment 2: PReLU² (learnable negative slope) ---
run_experiment "exp2_prelu2" \
    "$EXPERIMENTS/exp2_prelu2/train_gpt.py" \
    "NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 ITERATIONS=9000 EVAL_STRIDE=64"

# --- Experiment 3: Adam TTT ---
run_experiment "exp3_adam_ttt" \
    "$EXPERIMENTS/exp3_adam_ttt/train_gpt.py" \
    "NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 TTT_ENABLED=1 TTT_ADAM_LR=0.0002 TTT_ADAM_BETA1=0.9 TTT_ADAM_BETA2=0.999 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 TTT_FREEZE_BLOCKS=0 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 ITERATIONS=9000 EVAL_STRIDE=64"

# --- Experiment 4: 4096 sequence length (sp4096 vocab not yet in challenge manifest) ---
run_experiment "exp4_4kseqlen" \
    "$EXPERIMENTS/exp4_4096bpe/train_gpt.py" \
    "XSA_LAST_N=4 EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 ITERATIONS=9000 EVAL_STRIDE=64"

echo ""
echo "=========================================="
echo "  All experiments complete."
echo "  Compare results:"
echo "    python3 experiments/compare_results.py"
echo "=========================================="

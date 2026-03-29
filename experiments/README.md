# Parameter Golf Experiments

Four experiments derived from the current SOTA (1.1194 bpb, LeakyReLU² + Legal TTT + Parallel Muon).

## Sprint Contract

Each experiment must:
1. Pass `run_smoke.sh` (200 steps, no errors)
2. Log a `val_bpb` value
3. Produce an artifact ≤ 16,000,000 bytes compressed

## Experiments

| Dir | Change | Hypothesis |
|-----|--------|-----------|
| `exp1_silu2/` | SiLU² instead of LeakyReLU² | Smoother gradient profile, different inductive bias |
| `exp2_prelu2/` | PReLU² (learnable slope, init=0.5) | Model learns optimal negative slope rather than fixing it |
| `exp3_adam_ttt/` | Adam (lr=2e-4) instead of SGD for TTT | Adaptive per-param LR converges faster in 3 TTT epochs |
| `exp4_4096bpe/` | 4096 BPE vocab, 10L (from 11L) | More information per token; compensate size with 1 fewer layer |

## SOTA baseline (for comparison)

`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- Score: **1.1194 bpb** (3-seed mean)
- Seeds: 1337 (1.1192), 42 (1.1200), 2025 (1.1189)

## Quickstart

```bash
# 1. Smoke test (200 steps, 1 GPU)
bash experiments/run_smoke.sh

# 2. Full 8xH100 runs (all 4 experiments, 3 seeds each)
bash experiments/run_h100_all.sh

# 3. Compare results
python3 experiments/compare_results.py
```

## Data prerequisites

```bash
# 1024-vocab (experiments 1-3)
python3 data/cached_challenge_fineweb.py --variant sp1024

# 4096-vocab (experiment 4 only)
python3 data/cached_challenge_fineweb.py --variant sp4096
```

## Interpreting results

- **Best BPB wins** (lower is better)
- Beat SOTA by > 0.001 to be statistically significant (std ≈ 0.0006)
- Check `artifact_mb` — must stay ≤ 16.0 MB
- If an experiment beats SOTA pre-TTT, it's a strong architecture win worth submitting

## If an experiment wins

1. Run with seeds 1337, 42, 2025 (required by submission rules)
2. Copy the script to `records/track_10min_16mb/<date>_<name>/train_gpt.py`
3. Write a README.md in that dir with ablation table
4. Submit as PR to the challenge repo

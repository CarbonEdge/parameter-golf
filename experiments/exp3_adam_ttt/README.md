# exp3_adam_ttt — Adam optimizer for TTT

## What changed

Single change vs. SOTA (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`):

- **TTT inner optimizer**: replaced `torch.optim.SGD` with `torch.optim.Adam`
- Cosine LR decay formula updated to scale `ttt_adam_lr` instead of `ttt_lr`
- Three new hyperparameters added to `Hyperparameters` (`ttt_adam_lr`, `ttt_adam_beta1`, `ttt_adam_beta2`)
- Original `ttt_lr` and `ttt_momentum` kept in the class (unused) to avoid breaking any external references

## Why Adam for TTT

SGD with momentum has two structural weaknesses in the TTT setting:

1. **Momentum warmup**: with only ~3 epochs over a chunk, the momentum buffer never reaches steady state — the first epoch effectively runs at a lower effective LR. Adam's per-parameter adaptive scaling removes this dependency.
2. **Uniform LR for all parameters**: the model has a mix of large bank matrices and small scalar controls with very different gradient scales. Adam's per-parameter second-moment normalization handles this without hand-tuning per-group LRs.
3. **Convergence speed**: in a fixed-budget fine-tune (few epochs), Adam typically makes larger progress per step early on vs. SGD with momentum, which matters here since TTT is budget-constrained.

The cosine LR schedule (decays `ttt_adam_lr` from its initial value to 0 across chunks) is preserved — this pattern of scaling `optimizer.param_groups[0]['lr']` works identically for Adam.

## Key hyperparameters

| Variable | Env var | Default | Notes |
|---|---|---|---|
| `ttt_adam_lr` | `TTT_ADAM_LR` | `0.0002` | Base LR, ~10x lower than SGD default (0.002) since Adam adapts per-parameter |
| `ttt_adam_beta1` | `TTT_ADAM_BETA1` | `0.9` | First moment decay — standard value |
| `ttt_adam_beta2` | `TTT_ADAM_BETA2` | `0.999` | Second moment decay — standard value |
| `ttt_epochs` | `TTT_EPOCHS` | `3` | Number of epochs per chunk (unchanged) |
| `ttt_freeze_blocks` | `TTT_FREEZE_BLOCKS` | `2` | Freeze first N blocks (unchanged) |
| `ttt_chunk_tokens` | `TTT_CHUNK_TOKENS` | `32768` | Tokens per TTT chunk (unchanged) |

## Suggested ablation range for TTT_ADAM_LR

| LR | Rationale |
|---|---|
| `1e-4` | Conservative; safe floor |
| `2e-4` | Default; analogous to 10x reduction from SGD 2e-3 |
| `3e-4` | Slightly more aggressive |
| `5e-4` | Upper bound; risk of overshooting in short 3-epoch runs |

Recommended sweep order: `2e-4` (baseline) → `3e-4` → `1e-4` → `5e-4`.

## Run command

Same environment as the SOTA run, with `RUN_ID` and `TTT_ADAM_LR` overrides:

```bash
TTT_ENABLED=1 RUN_ID=exp3_adam_ttt TTT_ADAM_LR=0.0002 \
  torchrun --nproc_per_node=8 train_gpt.py
```

To sweep LR, e.g. the `3e-4` point:

```bash
TTT_ENABLED=1 RUN_ID=exp3_adam_ttt_lr3e4 TTT_ADAM_LR=0.0003 \
  torchrun --nproc_per_node=8 train_gpt.py
```

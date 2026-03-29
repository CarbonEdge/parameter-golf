# exp1_silu2 — SiLU² MLP activation

## What changed

The MLP activation in the `MLP.forward` method is changed from **LeakyReLU²** (negative slope 0.5) to **SiLU²**.

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

## Why

- **SiLU (Swish)** has a smooth, continuously differentiable gradient everywhere. LeakyReLU has a kink at zero that can cause gradient noise near the boundary.
- **No dead-neuron risk.** LeakyReLU with negative_slope=0.5 already avoids dead neurons, but SiLU's output is strictly positive only in the mean (it dips slightly negative), giving a different inductive bias once squared.
- **SiLU²** = SiLU followed by element-wise squaring keeps the same structural pattern as the SOTA (activation then square). The squaring enforces non-negativity and concentrates gradient flow on large activations; combining it with SiLU's smooth sigmoid gating may give better-calibrated hidden units than the piecewise-linear LeakyReLU gating.
- This is a cheap, zero-hyperparameter change — same parameter count, same compute budget.

## Exact diff

```diff
# MLP.forward  (line 729 of train_gpt.py)

-        x = F.leaky_relu(F.linear(x, up_w.to(x.dtype)), negative_slope=0.5)
+        x = F.silu(F.linear(x, up_w.to(x.dtype)))
         return F.linear(x.square(), down_w.to(x.dtype))
```

No other lines were modified.

## Run command

Same environment as the SOTA run, with `RUN_ID` set to this experiment name:

```bash
RUN_ID=exp1_silu2 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All other hyperparameters default to the SOTA values (see `Hyperparameters` class at the top of the script).

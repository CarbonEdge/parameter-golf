# exp2_prelu2 — PReLU² (Learnable Negative Slope)

## What changed

The `MLP` class (renamed to `BankedMLP`) now has a **per-layer learnable negative slope** for the LeakyReLU activation, initialized to 0.5. This makes the activation function PReLU² — a parametric version of LeakyReLU combined with the squaring in the down-projection.

## Why

In the SOTA script the negative slope of `F.leaky_relu` is hard-coded to `0.5`. There is no principled reason that 0.5 is optimal — it was chosen by hand. Making it a learned scalar (`nn.Parameter`) costs 11 parameters (one per layer, 11 layers) and allows gradient descent to find the optimal slope during training. The slope is still initialized to 0.5 so the model starts from the same point as the SOTA baseline.

The `neg_slope` parameter is a scalar (`ndim=0`), so the existing optimizer-split logic automatically assigns it to `optimizer_scalar` (AdamW) via the `p.ndim < 2` branch in `scalar_params` collection — no optimizer changes required.

## Exact diff

```diff
--- a/train_gpt.py  (SOTA: 2026-03-23_LeakyReLU_LegalTTT_ParallelMuon)
+++ b/train_gpt.py  (exp2_prelu2)
@@ -724,9 +724,11 @@
-class MLP(nn.Module):
+class BankedMLP(nn.Module):
     def __init__(self, dim: int, mlp_mult: int):
         super().__init__()
         # No CastedLinear -- weights come from banks
+        # PReLU²: learnable negative slope, initialized to 0.5 (same as SOTA fixed value)
+        self.neg_slope = nn.Parameter(torch.tensor(0.5))
     def forward(self, x: Tensor, up_w: Tensor, down_w: Tensor) -> Tensor:
-        x = F.leaky_relu(F.linear(x, up_w.to(x.dtype)), negative_slope=0.5)
+        x = F.leaky_relu(F.linear(x, up_w.to(x.dtype)), negative_slope=self.neg_slope.item())
         return F.linear(x.square(), down_w.to(x.dtype))

@@ -754,1 +754,1 @@
-        self.mlp = MLP(dim, mlp_mult)
+        self.mlp = BankedMLP(dim, mlp_mult)
```

**Total parameter delta:** +11 scalars (one `neg_slope` per layer, 11 layers). These are fp32 scalars optimized by AdamW at `scalar_lr`.

## Run command

```bash
RUN_ID=exp2_prelu2 torchrun --nproc_per_node=8 train_gpt.py
```

Same hyperparameters as SOTA. Set additional env vars as needed (e.g. `DATA_PATH`, `TOKENIZER_PATH`).

# Smoke Test Results (RTX 4080 SUPER, 200 steps each)

All tests run locally with: `TORCHDYNAMO_DISABLE=1 TRAIN_SEQ_LEN=512 TRAIN_BATCH_TOKENS=65536 VAL_BATCH_SIZE=8192`

| Experiment | Status | train_loss @200 | val_bpb | Artifact size | Notes |
|-----------|--------|-----------------|---------|---------------|-------|
| exp1_silu2 | ✅ PASS | 3.32 | 1.9379 | **6.1 MB** | Full pipeline complete incl. int6+lzma |
| exp2_prelu2 | ⚠️ PARTIAL | 3.48 | 2.0209 | — | Training+step eval complete; quant pipeline interrupted (concurrent GPU) |
| exp3_adam_ttt | ✅ PASS | 3.38 | 1.9995 | **5.1 MB** | Full pipeline complete incl. int6+lzma |
| exp4_4kseqlen | ⚠️ PARTIAL | 4.07 | — | — | Training complete; val eval not reached (slow at 4k seq_len on single GPU) |

**Key finding**: exp1 compressed artifact = 6.1 MB — well within the 16 MB challenge limit.

Note: val_bpb values are from undertrained models (200 steps vs 7000+ for leaderboard) and are not comparable to leaderboard scores. The purpose is to verify correctness, not quality.

## Re-run needed

exp2 and exp4 should be re-run individually (not concurrent) to confirm full pipeline:
- `bash experiments/run_smoke.sh` runs all 4 sequentially
- Or run each script directly with the env vars in run_smoke.sh

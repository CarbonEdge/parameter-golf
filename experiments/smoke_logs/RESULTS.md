# Smoke Test Results (RTX 4080 SUPER, 200 steps each)

All tests run locally with: `TORCHDYNAMO_DISABLE=1 TRAIN_SEQ_LEN=512 TRAIN_BATCH_TOKENS=65536 VAL_BATCH_SIZE=65536`

| Experiment | Status | train_loss @200 | val_bpb | Artifact size | Notes |
|-----------|--------|-----------------|---------|---------------|-------|
| exp1_silu2 | ✅ PASS | 3.32 | 1.9379 | **6.1 MB** | Full pipeline complete incl. int6+lzma |
| exp2_prelu2 | ✅ PASS | 3.48 | 2.0209 | **5.0 MB** | Full pipeline complete incl. int6+lzma |
| exp3_adam_ttt | ✅ PASS | 3.38 | 1.9995 | **5.1 MB** | Full pipeline complete incl. int6+lzma |
| exp4_4kseqlen | ✅ PASS | 4.07 | 2.3821 | **4.8 MB** | Full pipeline complete; higher bpb expected (4k seqlen needs more steps to converge) |

**Key finding**: All 4 compressed artifacts are well within the 16 MB challenge limit (largest: 6.1 MB).

Note: val_bpb values are from undertrained models (200 steps vs 7000+ for leaderboard) and are not comparable to leaderboard scores. The purpose is to verify correctness, not quality. exp4's higher bpb is expected — 4096-token sequences need proportionally more training steps to reach equivalent loss.

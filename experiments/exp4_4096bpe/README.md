# exp4_4096bpe — 4096-token BPE vocabulary

## What changed

| Hyperparameter | SOTA (sp1024) | This experiment (sp4096) |
|---|---|---|
| `vocab_size` | 1024 | **4096** |
| `data_path` | `fineweb10B_sp1024` | **`fineweb10B_sp4096`** |
| `tokenizer_path` | `fineweb_1024_bpe.model` | **`fineweb_4096_bpe.model`** |
| `num_layers` | 11 | **10** |
| `bigram_vocab_size` | 2048 | 2048 (unchanged) |

Everything else — XSA, EMA, legal score-first TTT, Parallel Muon, GQA, LeakyReLU MLP, parameter banks, bigram hash embedding, value embedding, skip connections — is identical to the SOTA script.

## Why a larger vocabulary?

With 1024 tokens, the tokenizer must use ~1.1 tokens/byte on English text. With 4096 tokens that improves to roughly ~0.85 tokens/byte. Fewer tokens per byte means:

- Each token carries more information about the underlying text
- The model sees more context (in tokens) per fixed-length sequence
- The cross-entropy loss in bits-per-byte has a lower floor given the same model capacity

Put differently: a 4096-vocab model that achieves the same token-level loss as a 1024-vocab model will have a *better* bits-per-byte score — which is the actual competition metric.

## 16 MB budget tradeoff

The tied embedding table grows with vocab size. At FP16:

- 1024 vocab: 1024 × 512 × 2 B = **1.05 MB**
- 4096 vocab: 4096 × 512 × 2 B = **4.19 MB**
- Extra cost: **+3.14 MB raw** (or ~+1.65 MB after int6 + zstd compression)

To compensate, we drop from 11 to 10 transformer layers. Each layer is approximately:

- QO bank slice: 2 × 512 × 512 = 524 K params
- KV bank slice: 2 × 256 × 512 = 262 K params
- MLP up/down: 2 × 1536 × 512 = 1 572 K params
- Small scalars: ~5 K params

Total per layer ≈ 2 363 K params ≈ **2.36 M params**.

At int6 (0.75 bytes/param) and zstd compression (~0.7×): 2.36 M × 0.525 ≈ **1.24 MB saved per layer**.

Dropping 1 layer saves ~1.24 MB compressed; the vocab increase costs ~1.65 MB compressed. The net is roughly +0.4 MB, which should still fit well within the 16 MB limit given the SOTA model currently comes in around 14–15 MB compressed.

> Note: `bigram_vocab_size` is intentionally left at 2048. The bigram hash embedding maps token pairs to a fixed-size table via a hash function; this table size is a capacity knob independent of the true vocabulary size, and 2048 is already larger than the 1024-vocab embedding table was.

## Data download

```bash
python3 data/cached_challenge_fineweb.py --variant sp4096 --train-shards 10
```

> **Important**: The actual tokenizer filename depends on the HuggingFace manifest.
> After running the download, check what was downloaded:
> ```bash
> ls data/tokenizers/
> ```
> Then set `TOKENIZER_PATH` to the correct `.model` file.

## Run command

```bash
VOCAB_SIZE=4096 \
DATA_PATH=./data/datasets/fineweb10B_sp4096 \
TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model \
NUM_LAYERS=10 \
RUN_ID=exp4_4096bpe \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All other hyperparameters default to their SOTA values and can be overridden via environment variables as usual.

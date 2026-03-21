# Lab Notebook — Serialization Optimization

## Baseline analysis

Source: MLX artifact re-quantized to SOTA format (mixed int6/int8 + zstd-22).
This simulates the `train_gpt_sota.py` pipeline: `mixed_quantize_int6` → `torch.save` → `zstd-22`.

### SOTA pipeline (train_gpt_sota.py lines 991-1022, 1459-1464)

1. `mixed_quantize_int6(state_dict, {"mlp", "attn"})` — int6 for MLP+attn, int8 for embedding
2. Result format: `{"w": {name.q, name.scale, passthrough}, "m": {name: metadata}}`
3. `torch.save({"w": result, "m": meta}, buf)` — pickle-based serialization
4. `zstandard.ZstdCompressor(level=22).compress(raw)` — zstd compression

### Weight value ranges after int6 re-quantization

Block weights (int6): actual range [-32, 31] but actual values in [-7, 7] for this artifact.
- `attn.c_q/c_k`: [-3, 4] — only 3 bits needed
- `attn.c_v/proj`: [-6, 5] — 4 bits
- `mlp.fc`: [-7, 6] — 4 bits
- `mlp.proj`: [-3, 4] — 3 bits
- `tok_emb.weight` (int8): [-114, 122] — full 8 bits

### Key insight from experiments

zstd-22 is VERY good at compressing these int8 arrays with small values. It already handles byte-level redundancy efficiently. Approaches that help:
- **Separate data streams** (homogeneous data per stream)
- **Transpose** (column patterns compress better)

Approaches that HURT:
- Bit-packing (destroys byte alignment zstd exploits)
- Delta encoding (adjacent values not correlated in weight matrices)
- Custom binary format (torch.save pickle overhead is negligible after compression)

## Ideas queue

- Split int6 .q and int8 .q into separate sub-streams (different value distributions)
- Byte shuffling for fp16 scales (split high/low bytes)
- Per-tensor zstd frames (tailored entropy models)
- Value zigzag encoding (concentrate values near 0)
- Interleave tensor bytes differently (e.g., row-interleaved across tensors)
- Try zstd with different windowLog/chainLog parameters
- Try combining transpose with nibble-level reorganization
- XOR with predicted values (e.g., XOR each row with previous row)

## Experiment log

### Exp 1: Custom binary format (no torch.save) — REVERTED
- **Result**: 3,385,337 bytes (+2.3% worse)

### Exp 2: True int6 bit-packing (4 values → 3 bytes) — REVERTED
- **Result**: 3,638,084 bytes (+9.9% worse)

### Exp 3: Transpose weight matrices — KEPT (superseded by exp 7)
- **Result**: 3,288,084 bytes (-0.7%)

### Exp 4: Sort rows by scale + transpose — REVERTED
- **Result**: 3,298,115 bytes (worse than transpose alone)

### Exp 5: Delta encoding + transpose — REVERTED
- **Result**: 4,416,026 bytes (+33.4% worse)

### Exp 6: Concatenate homogeneous tensors — REVERTED
- **Result**: 3,293,403 bytes (worse than transpose alone)

### Exp 7: Separate zstd-22 streams per data type + transpose — KEPT (current best)
- **Result**: 3,275,790 bytes (-1.0% vs baseline)
- **Insight**: Each data type gets its own entropy model → more efficient compression.

### Exp 8: zstd dictionary for weight stream — REVERTED
- **Result**: 3,379,193 bytes (+2.1% worse)
- **Insight**: 32KB dictionary overhead not recovered.

### Exp 9: 4-stream split int6/int8 weights — REVERTED
- **Result**: 3,278,435 bytes. Barely different from 3-stream, extra header overhead.

### Exp 10: LZMA for weight stream — KEPT (superseded by exp 11)
- **Result**: 3,272,164 bytes (-1.2% vs baseline)

### Exp 11: LZMA for ALL streams — KEPT (superseded by exp 12)
- **Result**: 3,267,661 bytes (-1.3% vs baseline)

### Exp 12: LZMA EXTREME preset for all streams — KEPT (current best)
- **Result**: 3,257,949 bytes (-1.6% vs baseline)
- **Insight**: PRESET_EXTREME squeezes ~10KB more out of LZMA.

### Exp 13: LZMA extreme single combined stream — REVERTED
- **Result**: 3,269,124 bytes. Worse than separate streams. LZMA benefits from per-type homogeneity.

### Exp 14: LZMA with FILTER_DELTA for weight stream — REVERTED
- **Result**: 3,449,705 bytes (+4.2% worse). Delta filter hurts even inside LZMA. Adjacent bytes not correlated.

### Exp 15: LZMA extreme without transpose — REVERTED
- **Result**: 3,283,697 bytes. Transpose helps LZMA by ~26KB. Column patterns still compress better.

### Exp 16: Byte shuffle fp16 scales — REVERTED
- **Result**: 3,257,957 bytes. Only 8 bytes diff from lzma_extreme. Scales too small (57KB) to matter.

### Exp 17: Zigzag encoding for weights — REVERTED
- **Result**: 3,264,981 bytes. Worse than signed representation. LZMA handles signed int8 well.

### Exp 18: Group same tensor types across layers — KEPT (current best)
- **Result**: 3,250,029 bytes (-1.8% vs baseline, -7,920 vs lzma_extreme)
- **Insight**: Ordering tensors so all c_q.weight.q across layers are adjacent (instead of sorted by name) gives LZMA cross-layer patterns. Different layers' same-type weights share structural similarity.

### Exp 19: Row-interleave same-type tensors across layers — KEPT (current best)
- **Result**: 3,248,601 bytes (-1.9% vs baseline, -1,428 vs typegroup)
- **Insight**: Row-level interleaving gives LZMA even more cross-layer correlation. Corresponding rows from different layers' same-type weights are adjacent → more pattern matches.

### Exp 20: Sparse representation (bitmask + nonzero values) — KEPT (superseded by exp 21)
- **Result**: 3,034,601 bytes (-8.3% vs baseline, -214K vs interleave!)
- **Insight**: MASSIVE win. 72.5% of values are zero → store a bitmask (zero/nonzero) + only non-zero values. Each component compresses extremely well separately: bitmask is highly regular, non-zero values are 93% +-1. This is the single biggest improvement in the entire experiment log. Encode is also much faster (3.3s vs 12.8s) because LZMA operates on smaller data.

### Exp 21: Sign+abs split for non-zero values — KEPT (superseded by exp 22)
- **Result**: 2,977,925 bytes (-10.0% vs baseline, -56.7K vs plain sparse!)
- **Insight**: Splitting non-zero values into packed sign bits + absolute values compresses much better. Signs are ~50/50 → pack and compress efficiently. Absolute values are 93% = 1 → extremely compressible as a separate stream.

### Exp 22: Decompose abs into abs==1 mask + abs>1 values — KEPT (current best)
(Also tested but not committed: ternary encoding, column-interleaving, delta filter on mask, LZMA param tuning, bz2 — all worse)
- **Result**: 2,957,897 bytes (-10.7% vs baseline, -20K vs sign+abs)
- **Insight**: 83.5% of non-zero abs values are exactly 1. Storing a bitmask for "is it 1?" and only the remaining values (16.5% of non-zero) saves 20KB. Total: 5 compressed streams for weights (zero-mask, signs, abs==1-mask, abs>1-values) + scales + passthrough + meta = 8 streams.

### Value distribution insight (from analysis)
- 72.5% of int6 values are 0, 12.8% are +1, 12.8% are -1
- Shannon entropy: 1.235 bits/value
- Theoretical minimum: 2,548,860 bytes for weight data
- ~700KB gap between LZMA output and theoretical minimum

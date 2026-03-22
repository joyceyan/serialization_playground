# Lab Notebook — Serialization Optimization (Phase 2: H100 artifact)

## Setup

Input: `final_model.pt` — unquantized state dict from 11L 512dim MLP3x model trained on 8xH100.
Pipeline: `mixed_quantize_int6(sd, {"mlp", "attn"})` → `torch.save` + `zstd-22`.
Baseline size: TBD (waiting for final_model.pt from RunPod).

## Key lessons from Phase 1 (MLX artifact, experiments 1-30)

Phase 1 optimized against an MLX smoke-test artifact with 72.5% zero values and tiny ranges [-7, 7]. The sparse decomposition (bitmask + signs + abs masks) saved 11.2% locally but **produced a larger artifact than torch.save + zstd-22 on the real H100 model** (16.0MB vs ~15.5MB). The H100 model has dense weights with far fewer zeros.

**Techniques that transfer regardless of sparsity:**
- Transpose 2D tensors before compression (-0.7%)
- Separate streams per data type (-1.0%)
- LZMA extreme over zstd-22 (-1.6%)
- Group same tensor types across layers (-0.2%)
- Row-interleave same-type tensors (-0.1%)

**Techniques that DON'T transfer to dense weights:**
- Sparse decomposition (bitmask + signs + abs masks) — overhead exceeds savings when <30% zeros
- Bit-packing — destroys byte alignment that zstd/LZMA exploit
- Delta encoding — adjacent weight values are not correlated

**Techniques that had no effect:**
- Custom binary format (pickle overhead is negligible after compression)
- Byte-shuffle fp16 scales (too small to matter)
- Zigzag encoding (LZMA handles signed int8 well)
- zstd dictionary (overhead not recovered)

## Ideas queue

**Try first (proven on MLX, likely transfer):**
- Transpose + separate LZMA extreme streams (the ~3% combined win from Phase 1)
- Type-grouping + row-interleaving across layers

**Try if above works:**
- LZMA filter chains tuned for dense int6 data
- Different LZMA dict_size per stream
- Byte-shuffle int8 data (may help for denser values)

**New ideas for dense weights:**
- Analyze the actual zero rate and value distribution of H100 weights
- If zero rate is >30%, sparse decomposition may still help partially
- Per-tensor adaptive strategy: sparse for sparse tensors, direct for dense ones

## Experiment log

### Exp 1: LZMA extreme instead of zstd-22 — REVERTED

**Hypothesis**: LZMA extreme was -1.6% in Phase 1, should transfer to H100 artifact.
**Result**: 15,824,088 bytes (+311,057, +2.0% worse). LZMA is worse on this data.
**Insight**: The torch.save pickle format with zstd-22 is already very efficient on this dense data. LZMA's larger block size doesn't help here — the overhead of the .xz container and different compression model hurts. zstd-22 is the compressor to beat, not replace. Future experiments should focus on **preprocessing the data** to be more compressible, not changing the compressor.

### Exp 2: Transpose 2D tensors before compression — KEPT

**Hypothesis**: Transposing makes column values (same output neuron) adjacent, improving compressibility. Was -0.7% in Phase 1.
**Result**: 15,500,303 bytes (-12,728, -0.08%). Small win.
**Insight**: Only 0.08% vs 0.7% in Phase 1. The dense H100 weights have less column correlation than the MLX artifact. Still a free win with no downside. Need bigger ideas.

### Exp 3: Separate dtype streams + transpose + custom binary format — KEPT

**Hypothesis**: Grouping all int8 data together (removing pickle interleaving) gives zstd a contiguous block of similar data to compress. Combined with transpose and no pickle overhead.
**Result**: 15,350,003 bytes (-163,028, -1.05%). Significant win!
**Insight**: Stripping pickle overhead and grouping by dtype is the real win here, not just transpose. The int8 stream is 25.9MB of int6-range values — compressing it as one contiguous block lets zstd find much longer matches. Encode is actually faster too (15.9s vs 17.1s) since we avoid pickle serialization overhead.

### Exp 4: Group tensors by type across layers — REVERTED

**Hypothesis**: Grouping all c_q together, all c_k together, etc. would let zstd find more repeated patterns across layers.
**Result**: 15,370,267 bytes (+20,264 vs exp3). Worse.
**Insight**: Alphabetical ordering already groups similarly-named tensors. Reordering by type breaks the natural block-0, block-1, ... sequence that zstd was exploiting for cross-layer delta patterns.

### Exp 5: Row-interleave same-shape int8 tensors — REVERTED

**Hypothesis**: Interleaving rows from tensors of the same shape would create repeating patterns the compressor could exploit.
**Result**: 15,407,265 bytes (worse + roundtrip error). Broken implementation, and even ignoring the bug, the size was worse.
**Insight**: Row-interleaving breaks the within-tensor locality that zstd relies on. Abandon this direction.

**Status**: Current best is exp3 at 15,350,003 (-1.05%). Need fundamentally different approaches.

### Exp 6: Fine-grained streams (int6_q, int8_q, int6_scale, etc.) — REVERTED

**Hypothesis**: Separating int6 values ([-32,31]) from int8 values ([-127,127]) into separate streams would let zstd build better per-stream frequency tables.
**Result**: 15,446,020 bytes (worse than exp3's 15,350,003). More streams = more zstd frame headers = more overhead.
**Insight**: 3 streams (int8/fp16/fp32) is the sweet spot. More streams adds overhead that exceeds any distribution-matching benefit.

### Exp 7: Enable long distance matching — REVERTED

**Result**: 15,350,001 bytes (-2 bytes vs exp3). Not worth it.

### Exp 8: Byte-shuffle fp16 and fp32 streams — KEPT

**Hypothesis**: Separating high/low bytes of fp16 and the 4 byte planes of fp32 should create more compressible streams since exponent bytes have low entropy.
**Result**: 15,335,927 bytes (-14,076 vs exp3, -177,104 vs baseline = -1.14%). New best!
**Insight**: Despite fp16+fp32 being only 267KB raw, byte-shuffling saves 14KB compressed. The high bytes of fp16 scales are very repetitive (similar exponents across all scales). This is a free win.

### Exp 9: LZMA extreme for byte-shuffled fp16 stream — KEPT

**Result**: 15,334,299 bytes (-1,628 from exp8). LZMA is better than zstd for the small, repetitive fp16 data.

### Exp 10: LZMA extreme for int8 stream — REVERTED

**Result**: 15,631,066 bytes (+296,767 vs exp9). LZMA is decisively worse for the large int8 stream. zstd-22's FSE entropy coding and match-finding are better suited for this data.

**Current state**: 15,334,299 (-1.15%). The int8 stream is ~15.1MB and dominates. Need to find ways to make it more compressible.

### Exp 11: Unsigned offset for int8 values — REVERTED
**Result**: +577 bytes. No benefit.

### Exp 11b: Adaptive zstd/zlib for int8 — REVERTED
**Result**: zstd always wins. zlib never smaller on this data.

### Exp 12: Ablation — remove transpose — REVERTED
**Result**: +576 bytes. Transpose helps marginally.

**Data analysis** (critical for future experiments):
- 26.7M int8 values, per-value entropy = 4.67 bits
- Theoretical min (entropy only) = 15,624,070 bytes
- Current compressed = 15,334,299 bytes — **already 1.85% BELOW entropy floor**
- This means zstd is exploiting positional correlations via LZ77
- Adjacent value correlation: only 5% of adjacent pairs match
- Distribution: peaked at 0 (8%), symmetric, Laplace-like decay

**Key insight**: We're squeezing blood from a stone on the int8 stream. The remaining gains must come from:
1. More compact metadata encoding (header is only 1.4KB though)
2. Fundamentally different data representation (bit-packing, prediction + residual)
3. Per-tensor custom encoding (avoid the one-size-fits-all approach)

### Exp 13: zstd custom params (higher search_log) — REVERTED
**Result**: -1 byte. Not meaningful.

### Exp 13b: Nibble-split int8 stream — REVERTED
**Result**: 17,443,868 bytes (+12.4% worse). Catastrophic. Packing nibbles destroys positional correlations.
**Insight**: Any scheme that reorders bytes at sub-byte level destroys the LZ77 match patterns that zstd exploits. This definitively rules out bit-plane decomposition and nibble splitting.

**Strategy pivot**: We're at 15,334,299 (-1.15%) and running out of incremental ideas for the int8 stream. Need to think about this differently:

### Exp 14: zstd target_length=4096 — REVERTED
**Result**: Identical. Default is already optimal.

### Exp 14b: Brotli-11 for int8 stream — REVERTED
**Result**: zstd-22 is still smaller. Brotli can't beat zstd on this data.

**Assessment at exp 14**: We've exhausted compression algorithm improvements. zstd-22 is optimal for the int8 stream. We're below per-symbol entropy. The int8 stream accounts for 99% of the output. Any further gains must come from:

1. **Reducing the raw data size** before compression (lossy or bit-packing that preserves compressibility)
2. **Two-pass compression**: compress, then compress again
3. **Prediction + residual**: use row/column neighbors to predict values, compress the residuals

**Next ideas**:
- Two-pass: zstd-22 → zstd (or zlib) on the compressed output
- XOR each row with the mean-quantized row (prediction-based)
- Try compressing int8 data as 2D images using PNG-style prediction filters (sub, up, average, Paeth)

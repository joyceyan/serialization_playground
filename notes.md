# Lab Notebook — Serialization Optimization (Phase 2: H100 artifact)

## Setup

Input: `final_model.pt` — unquantized state dict from 11L 512dim MLP3x model trained on 8xH100.
Pipeline: `mixed_quantize_int6(sd, {"mlp", "attn"})` → `torch.save` + `zstd-22`.
Baseline size: TBD (waiting for final_model.pt from RunPod).

## Key lessons from Phase 1 (MLX artifact, experiments 1-30)

Phase 1 optimized against an MLX smoke-test artifact with 72.5% zero values and tiny ranges [-7, 7]. The sparse decomposition (bitmask + signs + abs masks) saved 11.2% locally but **produced a larger artifact than torch.save + zstd-22 on the real H100 model** (16.0MB vs ~15.5MB). The H100 model has dense weights with far fewer zeros.

**Techniques that transfer regardless of sparsity:**
- ~~Transpose 2D tensors before compression (-0.7%)~~ — CONFIRMED in exp 2, small but real (-0.08%), kept
- ~~Separate streams per data type (-1.0%)~~ — CONFIRMED in exp 3, main contributor (-1.05%), kept
- ~~LZMA extreme over zstd-22 (-1.6%)~~ — DISPROVED in exp 1, LZMA worse for dense int8 (+2.0%); BUT confirmed for small fp16 stream (exp 9, -1.6KB)
- ~~Group same tensor types across layers (-0.2%)~~ — DISPROVED in exp 4, worse (+20KB), alphabetical order better
- ~~Row-interleave same-type tensors (-0.1%)~~ — DISPROVED in exp 5, buggy and worse

**Techniques that DON'T transfer to dense weights:**
- Sparse decomposition (bitmask + signs + abs masks) — overhead exceeds savings when <30% zeros (not tried, H100 has only 6-12% zeros)
- ~~Bit-packing — destroys byte alignment that zstd/LZMA exploit~~ — CONFIRMED in exp 13b (nibble split +12.4% worse)
- ~~Delta encoding — adjacent weight values are not correlated~~ — CONFIRMED in exp 15 (sub filter -9.6%) and exp 50 (XOR rows -6.7%)

**Techniques that had no effect (Phase 1 findings, re-evaluated):**
- ~~Custom binary format (pickle overhead is negligible after compression)~~ — DISPROVED: custom format IS better, saves ~100KB (exp 3). Phase 1 was wrong — pickle overhead matters when combined with dtype separation
- ~~Byte-shuffle fp16 scales (too small to matter)~~ — DISPROVED: byte-shuffle fp16 saves 14KB (exp 8). Phase 1 underestimated the benefit with LZMA
- ~~Zigzag encoding (LZMA handles signed int8 well)~~ — DISPROVED: zigzag saves 3.3KB with zstd (exp 24). Phase 1 only tested with LZMA; zstd benefits from zigzag's value clustering
- ~~zstd dictionary (overhead not recovered)~~ — DISPROVED: 256-byte dictionary saves 1.3KB (exp 33). Phase 1 used too-large dictionaries

## Ideas queue

**~~Try first (proven on MLX, likely transfer):~~**
- ~~Transpose + separate LZMA extreme streams (the ~3% combined win from Phase 1)~~ — tried exp 1-3: transpose + separate streams kept, LZMA only for fp16
- ~~Type-grouping + row-interleaving across layers~~ — tried exp 4-5: both worse

**~~Try if above works:~~**
- ~~LZMA filter chains tuned for dense int6 data~~ — tried exp 10, 49: LZMA worse than zstd for int8, delta filter worse for fp16
- ~~Different LZMA dict_size per stream~~ — not applicable, only fp16 uses LZMA; tuned LZMA2 params instead (exp 36)
- ~~Byte-shuffle int8 data (may help for denser values)~~ — tried exp 13b (nibble split): catastrophically worse. Byte-level rearrangement destroys LZ77 patterns

**~~New ideas for dense weights:~~**
- ~~Analyze the actual zero rate and value distribution of H100 weights~~ — done: 6-12% zeros, Laplace distribution, 4.67 bits entropy
- ~~If zero rate is >30%, sparse decomposition may still help partially~~ — N/A, zero rate is only 6-12%
- ~~Per-tensor adaptive strategy: sparse for sparse tensors, direct for dense ones~~ — not needed, all tensors are dense

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

### Exp 15: PNG-style sub filter — REVERTED
**Result**: 17,002,692 bytes (-9.6% worse). Adjacent values in a row are independent — differences have higher entropy than raw values.
**Insight**: Neural net weights are NOT images. They lack spatial correlation. Any prediction-based filter will make things worse because residuals have higher entropy.

**At 15 experiments, we've converged to -1.15%**. Approaches tried and failed:
- Different compressors (LZMA, zlib, brotli)
- Data transformations (unsigned offset, nibble split, prediction filter)
- Layout changes (type grouping, row interleaving, fine-grained streams)
- Parameter tuning (search_log, chain_log, target_length, LDM)

### Exp 16: Single zstd frame for all data — REVERTED
**Result**: 15,336,164 (+1,865 vs multi-stream). Mixing fp16 bytes into the int8 zstd context hurts more than frame overhead costs.

**What's left**: The multi-stream approach with per-stream optimal compression (zstd-22 for int8/fp32, LZMA for fp16) and byte-shuffling is very close to optimal. We've tried 16+ experiments targeting the int8 stream and nothing helps because we're already below per-symbol entropy.

### Exp 17: JSON+LZMA header encoding — KEPT
**Result**: 15,334,024 bytes (-275 vs exp9). New best!
**Insight**: JSON compresses better than pickle (more repetitive structure) and LZMA beats zstd for small data. Using 1-char keys ("i", "f", "g", "s", "t", "b", "m") further reduces size.

### Exp 18: Sign-magnitude split for int8 — REVERTED
**Result**: 15,597,005 (+0.5% worse). Again, separating byte components loses positional correlations.
**Key lesson confirmed**: ANY transformation that reorders the int8 bytes away from their natural positional layout makes things WORSE. zstd's strength is in finding positional matches, not in per-symbol entropy coding.

### Exp 19: zstd min_match=4 — REVERTED
**Result**: +15KB worse. Short 3-byte matches are valuable.

### Exp 20: Adaptive byte-shuffle for fp16 — REVERTED
**Result**: Identical. Byte-shuffled always wins for this data.

## Summary at experiment 20

**Best result: 15,334,024 bytes (-179,007 = -1.154% vs baseline)**

**What works:**
1. Custom binary format (no pickle overhead) — main contributor
2. Separate streams by dtype (int8/fp16/fp32) with per-stream optimal compression
3. Byte-shuffle for fp16 (high/low byte separation)
4. LZMA for fp16 stream (better than zstd for small repetitive data)
5. JSON+LZMA for header (more compact than pickle+zstd)
6. Transpose 2D tensors (marginal: ~576 bytes)
7. Byte-shuffle for fp32 (negligible impact but free)

**What doesn't work (on dense int6/int8 neural net weights):**
- Different compressors for int8 (LZMA, brotli, zlib all worse than zstd-22)
- Data transformations (unsigned offset, nibble split, sign-magnitude split, prediction filters)
- Layout changes (type grouping, row interleaving, single frame)
- Parameter tuning (search_log, chain_log, target_length, min_match, LDM)

**Why further improvement is extremely hard:**
The int8 stream (26.7MB, 99% of output) is already compressed below its per-symbol entropy bound (4.67 bits → 15.6MB theoretical, actual ~15.1MB). This means zstd is effectively using positional correlations to compress better than any symbol-level approach. Any transformation that disrupts byte positions makes it worse.

### Exp 21-23: Various approaches — ALL REVERTED
Size ordering +22KB worse, multithreading identical, XOR had decoder bug.

### Exp 24: Zigzag encoding for int8 values — KEPT
**Hypothesis**: Zigzag encoding maps 0,-1,1,-2,2,... to 0,1,2,3,4,... concentrating common values near byte 0. This makes the low byte values very dense.
**Result**: 15,330,719 bytes (-3,305 from exp17, -182,312 = -1.175% vs baseline). New best!
**Insight**: Phase 1 notes said "Zigzag encoding (LZMA handles signed int8 well)" — but this was wrong for zstd! Zigzag helps zstd specifically because it clusters frequent values into contiguous byte ranges, improving both frequency table efficiency and LZ77 match probability.

**Current best: 15,330,719 (-1.175%)**

### Exp 25-27: Various post-zigzag experiments — ALL REVERTED
- No transpose with zigzag: +1.2KB worse (transpose still helps)
- Natural sort order: +436b worse (string sort accidentally better)
- zstd overlap_log=9: identical (only matters for multi-threaded)
- zigzag + LZMA: identical (zstd still wins)
- zstd dictionary: -87KB worse (dict overhead exceeds savings)

### Exp 29: Reversed zigzag — KEPT
**Result**: -188 bytes. Negative values get even (smaller) codes matching their slightly higher frequency.

### Exp 32: Indexed header format — KEPT
**Result**: -80 bytes. Store key names once, reference by index.

### Exp 33: 256-byte zstd dictionary for int8 — KEPT
**Result**: -1,306 bytes! Dictionary bootstraps compression context. 256 bytes is the minimum trainable size and optimal — larger dicts lose more to overhead. Also makes encoding 14x faster.

### Exp 34: Merge fp16+fp32 streams — REVERTED
**Result**: +8 bytes. Header overhead for size field exceeds savings from eliminating one frame.

### Exp 35: Skip empty fp32 block — KEPT
**Result**: -4 bytes. fp32 stream is empty in this model, so we save the length prefix.

**FINAL STATE: 15,329,141 bytes (-183,890 = -1.185% vs baseline)**

After 48 experiments, converged at **15,327,424 bytes (-1.196%)**. The winning techniques in order of impact:
1. Custom binary format / no pickle: ~100KB savings
2. Separate streams by dtype: ~50KB
3. Reversed zigzag encoding: ~3.3KB
4. 256-byte zstd dictionary: ~1.3KB
5. Byte-shuffle fp16: ~14KB (LZMA compressed)
6. Transpose 2D tensors: ~0.6KB
7. JSON+LZMA header: ~0.3KB
8. Indexed header format: ~80 bytes
9. Skip empty fp32 block: 4 bytes
10. Tuned LZMA2 lp=1 for fp16: ~928 bytes
11. Raw LZMA2 format (no .xz container): ~56+32 bytes
12. Tuned LZMA2 lc/lp/pb for header: ~37 bytes
13. Derive transposed/fp32_keys/byte_shuffle: ~57 bytes
14. Compact type-string meta: ~199 bytes
15. Abbreviated key names: ~96 bytes
16. Derive dtype from key suffixes (remove i/f fields): ~183 bytes
17. Standard zigzag better than reversed with dict: ~107 bytes
18. Compressed dictionary (zlib): ~50 bytes
19. 2-byte header length prefix: 2 bytes
20. Remove dict_len field: 2 bytes

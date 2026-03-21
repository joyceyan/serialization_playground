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

**Next ideas**:
- Try zstd with higher windowLog/chainLog parameters
- Byte-shuffle fp16 scales (separate high/low bytes)
- Analyze per-byte entropy of the int8 stream to find optimization targets
- Try XOR-based delta between adjacent rows within each tensor
- Try compressing without transpose to see if transpose is still helping

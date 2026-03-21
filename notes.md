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

(Experiments will be appended below once final_model.pt is available)

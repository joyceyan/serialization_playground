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
- **Insight**: 32KB dictionary overhead not recovered. zstd already learns patterns well within a single stream.

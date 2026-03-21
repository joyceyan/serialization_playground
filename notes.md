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

Block weights (int6): actual range [-32, 31] but most values within [-20, 20].
- `attn.c_q/c_k` weights: typically [-12, 12] → could fit int5 [-16, 15]
- `attn.c_v/proj` weights: typically [-20, 20] → needs int6
- `mlp.fc` weights: typically [-25, 25] → needs int6
- `mlp.proj` weights: typically [-12, 12] → could fit int5 [-16, 15]
- `tok_emb.weight` (int8): [-114, 122] → needs int8

### Bit budget analysis

For an 11L SOTA model (~26.8M params, ~15.5MB compressed):
- Int6 values stored in int8 containers waste 2 bits per value
- True int6 bit-packing (4 values → 3 bytes): 25% reduction in raw weight bytes
- Mixed int5/int6: additional savings for small-range tensors
- zstd-22 already exploits some of this redundancy, but true bit-packing gives it denser input

## Ideas queue

See program.md for full list.

## Experiment log

### Experiment 1: Custom binary format (no torch.save)
- **Hypothesis**: Removing pickle/torch.save overhead and writing raw numpy bytes will reduce compressed size.
- **Result**: 3,385,337 bytes (+74,807 / +2.3% WORSE). Reverted.
- **Insight**: torch.save's pickle format actually compresses well under zstd-22. The overhead is negligible in compressed form. Focus on data representation, not container format.

### Experiment 2: True int6 bit-packing (4 values → 3 bytes)
- **Hypothesis**: Packing int6 values into 6 bits each (25% raw savings) will reduce compressed size.
- **Result**: 3,638,084 bytes (+327,554 / +9.9% WORSE). Reverted.
- **Insight**: Bit-packing destroys byte-aligned patterns that zstd-22 exploits very efficiently. With values mostly in [-7,7], each int8 byte has lots of redundant high bits that zstd compresses away. Bit-packing scrambles these patterns. **Key lesson: work WITH the compressor, not against it.**

# Lab Notebook — Serialization Optimization

## Baseline analysis

Artifact: `f3ffa88d-62a7-4575-8c89-0ef6a07eb11a_mlx_model.int8.ptz` (9L, 512dim, MLP2x, MLX smoke test)

| Metric | Value |
|--------|-------|
| Compressed (zlib-9) | 8,446,420 bytes |
| Decompressed (pickle) | 17,188,361 bytes |
| Pickle overhead | 9,449 bytes (0.1%) |
| Quantized int8 | 17,039,360 bytes (54 tensors) |
| Scales (fp16) | 57,344 bytes |
| Passthrough (fp32) | 82,208 bytes |
| Compression ratio | 2.03x |

### Weight value ranges (int8 quantized)

Block weights (attn + MLP) typically use [-28, 28] range — fits in int6 [-32, 31].
- `attn.c_q/c_k` weights: [-14, 15] range — fits in int5 [-16, 15]
- `attn.c_v/proj` weights: [-25, 22] range — fits in int6
- `mlp.fc` weights: [-28, 25] range — fits in int6
- `mlp.proj` weights: [-13, 16] range — fits in int5 [-16, 15]
- `tok_emb.weight`: [-114, 122] range — needs int8

### Bit budget analysis

With 17,039,360 int8 bytes across 54 tensors:
- If all block weights re-quantized to int6 in int8 containers: same raw size, but zlib/zstd compresses better (zero high bits)
- True int6 bit-packing (4 values → 3 bytes): raw size drops to ~12.8MB
- Mixed int5/int6: MLP proj + attn q/k could use int5 (5 bits), saving more
- Embedding stays int8 (needs full range)

## Ideas queue

See program.md for full list.

## Experiment log

(Experiments will be appended below)

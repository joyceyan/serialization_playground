# Serialization Optimization for Parameter Golf

## Goal

Beat the baseline serialization (`torch.save` + `zstd-22`) by producing a smaller compressed artifact while preserving dequantization accuracy.

## Context

The Parameter Golf competition stores models as quantized tensors + metadata, compressed with zlib or zstd. The current pipeline:

1. **Quantize**: float32/bf16 weights → int8 per-row (large 2D tensors) or per-tensor (vectors)
2. **Pack**: Python dict with `quantized`, `scales`, `passthrough`, `dtypes`, `qmeta` keys
3. **Serialize**: `pickle.dumps(dict)` or `torch.save(dict, buf)`
4. **Compress**: `zlib.compress(raw, 9)` or `zstandard.ZstdCompressor(level=22).compress(raw)`
5. **Output**: Single `.ptz` file

### Baseline artifact structure (9L, 512dim, MLX smoke test)

From analysis of a real artifact:
- **Compressed**: ~8.4MB (zlib-9)
- **Decompressed**: ~17.2MB
- **Pickle overhead**: ~9KB (0.1%) — negligible
- **Quantized int8 weights**: ~17.0MB (54 tensors, all int8)
- **Scales (fp16)**: ~57KB
- **Passthrough (fp32)**: ~82KB (control tensors: q_gain, attn_scale, mlp_scale, resid_mix, skip_weights)
- **Compression ratio**: 2.03x

### Key observation: weights don't use full int8 range

Most block weights have actual ranges of [-28, 28] or smaller — well within int6 [-32, 31]. Only `tok_emb.weight` uses the full int8 range [-114, 122]. This means most weights waste 2+ bits per value in int8 containers.

## Test artifacts

Use real artifacts from the parameter-golf-fork repo:
- **MLX artifacts**: `/Users/jyan/src/parameter-golf-fork/logs/*.int8.ptz` (pickle + zlib format, converted to SOTA format via `load_and_convert`)

## Experiment methodology

### What to measure

For every serialization scheme, report:
1. **compressed_bytes**: Final compressed artifact size
2. **raw_bytes**: Uncompressed payload size (before compression)
3. **ratio**: Compression ratio (raw/compressed)
4. **max_abs_error**: Maximum absolute error after dequantize roundtrip (0.0 for lossless schemes)
5. **mean_abs_error**: Mean absolute error (for lossy schemes)
6. **encode_time_ms**: Time to serialize + compress
7. **decode_time_ms**: Time to decompress + deserialize + dequantize

### Ideas to explore (ordered by expected impact)

**High priority — bit packing:**
- Re-quantize int8 weights to int6 [-32,31] for block weights (not embedding). Store in int8 containers — zlib/zstd will compress the zero high bits efficiently.
- True int6 bit-packing: pack 4 int6 values into 3 bytes (24 bits). Saves 25% vs int8 containers before compression.
- Int5 for weights with small ranges (e.g., proj weights [-12,16]). Pack 8 int5 values into 5 bytes.
- Mixed precision: int5 for MLP proj, int6 for attn/MLP fc, int8 for embedding.

**Medium priority — compression:**
- Compare zlib-9 vs zstd at various levels (1-22)
- Delta encoding between rows before compression (consecutive rows may be similar)
- Transpose weight matrices before compression (column-major may compress better if adjacent rows share patterns)
- Sort rows by scale value before compression

**Lower priority — format:**
- Custom binary header (tensor names, shapes, dtypes as fixed-size records) instead of pickle
- Separate compression streams per tensor type (int5/int6/int8/fp16/fp32) — different compressors may suit different data
- Pack scales into fewer bits (fp8 or even int8 scales)

**Speculative:**
- Arithmetic coding tuned to the empirical weight distribution
- Huffman coding on quantized values
- LZ4 or brotli as alternative compressors
- Weight matrix SVD: store low-rank approximation for near-zero singular values

## File structure

- `serialize.py` — All serialization/deserialization implementations
- `benchmark.py` — Loads real artifacts, runs all schemes, prints comparison table
- `test_serialize.py` — Correctness tests (roundtrip accuracy, edge cases)
- `notes.md` — Lab notebook with experiment results
- `results.tsv` — Machine-readable results log

## Important notes

- The baseline uses `torch.save` + `zstd-22`. Our schemes must produce smaller output.
- Any custom format needs a corresponding decoder. Keep decoders simple and fast.
- Decompression speed is secondary to compressed size, but should remain reasonable (under a few seconds).

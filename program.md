# Serialization Optimization for Parameter Golf

## Goal

Beat the baseline serialization (`torch.save` + `zstd-22`) by producing a smaller compressed artifact while preserving dequantization accuracy. Every byte saved means more room for model parameters in the competition.

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

### H100 SOTA structure (11L, 512dim, MLP3x)

The actual competition artifacts are larger:
- ~26.8M parameters
- Mixed int6 (attn+MLP) + int8 (embedding) quantization
- zstd-22 compression
- Artifacts ~15.5-15.9MB (tight against 16MB limit)

## Test artifacts

Use real artifacts from the parameter-golf-fork repo:
- **MLX artifacts**: `/Users/jyan/src/parameter-golf-fork/logs/*.int8.ptz` (pickle + zlib format)
- **H100 artifacts**: Would use torch.save + zstd format (not available locally, but we can simulate)

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

- The H100 pipeline uses `torch.save` not `pickle.dumps`. Both use pickle under the hood, but torch.save adds its own header. For local experiments, work with the MLX pickle format since we have real artifacts. The techniques (bit packing, compression, format) transfer directly.
- The eval harness must be able to decompress and dequantize the artifact. Any custom format needs a decoder that fits in the code budget (~50-60KB of Python).
- Decompression + dequantization speed matters (10-min eval budget), but is secondary to size.

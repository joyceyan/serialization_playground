# Serialization Optimization

## Goal

Beat `torch.save` + `zstd-22` on compressed size while preserving lossless roundtrip accuracy.

## Baseline

The baseline serializes a dict of torch tensors:
```python
buf = io.BytesIO()
torch.save({"w": tensor_dict, "m": metadata_dict}, buf)
compressed = zstandard.ZstdCompressor(level=22).compress(buf.getvalue())
```

The input data is a dict `{"w": ..., "m": ...}` where:
- `w` contains **int8 tensors** (quantized weights with values typically in [-32, 31] or [-127, 127]), **fp16 tensors** (per-row scales), and **fp32 tensors** (small control parameters)
- `m` contains string/dict metadata describing each tensor's quantization type

## What to measure

For every scheme, report:
1. **compressed_bytes** — final output size (the metric we're optimizing)
2. **max_abs_error** — must be 0.0 for lossless schemes
3. **encode_ms** / **decode_ms** — should remain reasonable (under a few seconds)

## Ideas to explore

**Bit packing:**
- True int6 bit-packing: pack 4 int6 values into 3 bytes instead of 4 int8 containers (25% raw savings)
- Int5 packing for tensors with small value ranges (8 values → 5 bytes)
- Mixed precision packing based on per-tensor value range analysis

**Compression:**
- Alternative compressors (brotli, lz4, etc.)
- Delta encoding between rows before compression
- Transpose matrices before compression (column-major may compress better)
- Sort rows by scale value before compression
- Separate compression streams per data type

**Format:**
- Custom binary format instead of pickle/torch.save (skip serialization overhead)
- Pack fp16 scales into fewer bits (fp8 or int8 scales)
- Deduplicate metadata strings

**Speculative:**
- Arithmetic/Huffman coding tuned to empirical weight distributions
- Weight matrix SVD for near-zero singular values

## Test artifacts

Use real quantized model artifacts from:
`/Users/jyan/src/parameter-golf-fork/logs/*.int8.ptz`

These are loaded and converted to SOTA format via `serialize.load_and_convert()`.

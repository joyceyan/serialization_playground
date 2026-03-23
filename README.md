# Serialization Playground

Experimental workspace for developing a custom serializer/deserializer that beats `torch.save` + `zstd-22` on compressed checkpoint size for quantized GPT models.

## Results

| Scheme | Compressed Size | vs Baseline |
|---|---|---|
| `torch.save` + `zstd-22` (baseline) | 15,513,031 bytes | — |
| Custom format (current best) | 15,156,254 bytes | **-2.30%** |

## How it works

The custom format (`encode_experiment` / `decode_experiment` in `serialize.py`) replaces `torch.save` with:

- **Separate streams by dtype** — int8 weights, fp16 scales, and fp32 control params are compressed independently
- **Per-tensor ANS entropy coding** — int8 data uses zigzag encoding + asymmetric numeral systems with per-tensor frequency tables
- **Byte-shuffle + LZMA** — fp16 scales are split into high/low byte planes and compressed with tuned LZMA2
- **Compact header** — JSON with abbreviated key names, compressed with LZMA2

## Usage

```bash
# Run correctness tests + benchmarks (requires final_model.pt)
python test_serialize.py

# Encode
from serialize import encode_experiment, decode_experiment, load_and_quantize
quant_result, quant_meta = load_and_quantize("final_model.pt")
blob = encode_experiment(quant_result, quant_meta)

# Decode
w, meta = decode_experiment(blob)
```

## Dependencies

```
torch, numpy, zstandard, constriction
```

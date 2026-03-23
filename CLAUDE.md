# Serialization Playground

Autonomous experiment loop for optimizing model serialization/compression.

Read `program.md` for full setup and experiment methodology.

## Critical constraints

- **Goal**: Minimize the compressed size of `final_model.pt` by developing a custom serializer/deserializer that beats `torch.save` + `zstd-22`.
- **Roundtrip correctness**: Every serialization scheme MUST produce identical dequantized tensors (max absolute error = 0.0 for lossless).
- **Packages**: torch, numpy, zstandard, constriction, and stdlib. See requirements.txt.
- **Test artifact**: `final_model.pt` (H100 unquantized state dict). Loaded and quantized via `serialize.load_and_quantize()`.

## Experiment loop checklist

Every iteration, follow these steps in order:

1. **Read** `notes.md` and `results.tsv` to understand what's been tried.
2. **Design** one experiment. Edit `serialize.py` to implement the change.
3. **Run tests + benchmark**: `python test_serialize.py`
4. **Log to `results.tsv`** (tab-separated).
5. **Update `notes.md`**: hypothesis, result, insights.
6. **Keep/discard**: smaller compressed size with acceptable error → keep. Otherwise → `git checkout -- serialize.py`.
7. **Go to step 1. NEVER STOP.**

## Key principles

- **Measure everything**: Always compare against the baseline (`torch.save` + `zstd-22` = 15,513,031 bytes).
- **Bit-level thinking**: Every wasted bit × 26M values = significant bytes.
- **Compression-friendly**: Patterns that compress well matter more than raw size.
- **Roundtrip accuracy**: Any lossy scheme must measure and report max absolute error.

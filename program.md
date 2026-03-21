# Serialization Optimization

## Goal

Beat `torch.save` + `zstd-22` on compressed size while preserving lossless roundtrip accuracy.

## Setup

Do all of these steps immediately without asking for confirmation. You are fully autonomous — never pause to ask the human anything.

This setup is **idempotent** — it can be re-run safely.

1. **Read the codebase**: Read these files for full context:
   - `serialize.py` — all serialization/deserialization implementations. This is the primary file you edit.
   - `benchmark.py` — loads real artifacts, runs all schemes, prints comparison table.
   - `test_serialize.py` — correctness tests (roundtrip accuracy).

3. **Run tests** to confirm everything is working:
   ```bash
   cd /Users/jyan/src/serialization_playground && python test_serialize.py
   ```

4. **Run the benchmark** to see the current baseline:
   ```bash
   cd /Users/jyan/src/serialization_playground && python benchmark.py
   ```

5. **Check for existing progress**: Read `results.tsv` and `notes.md` (if they have entries beyond the baseline). Review what's been tried and what to try next.

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

## Research methodology

**Isolate variables**: Make exactly one change per experiment. If you change the bit packing and the compressor at the same time, you won't know which one mattered.

**Ablations**: When a complex change improves results, strip parts away to find the essential ingredient.

**Diminishing returns**: If the last 5 experiments were all minor tweaks with tiny deltas, change strategy entirely.

## The experiment loop

The experiment runs on `main`. All kept experiments are committed directly to `main`.

LOOP FOREVER:

1. **Review context**: Read `notes.md` and `results.tsv` to review what's been tried, what worked, and what to try next.

2. **Design the next experiment** based on insights from the notebook. Check the "Ideas queue" section of `notes.md` first. Make exactly one change to `serialize.py`.

3. `git commit -am "description of experiment"`

4. **Run tests**:
   ```bash
   cd /Users/jyan/src/serialization_playground && python test_serialize.py
   ```
   If tests fail, fix the issue before proceeding. If the approach is fundamentally broken, revert and try something else.

5. **Run benchmark**:
   ```bash
   cd /Users/jyan/src/serialization_playground && python benchmark.py
   ```

6. **Record results** in `results.tsv` (tab-separated):
   ```
   experiment	compressed_bytes	raw_bytes	ratio	max_abs_error	description
   ```

7. **Update `notes.md`**: Append an entry to the "Experiment log" with the hypothesis, result, and insights. Do this for every experiment — successes and failures both contain useful information.
   - **Update the "Ideas queue"**: Add new ideas sparked by this experiment. Remove ideas you just tried.

8. **Keep/discard decision**:
   - If compressed_bytes **decreased** with acceptable roundtrip error (0.0 for lossless): keep the commit.
   - If compressed_bytes is **equal or worse**: `git reset --hard HEAD~1` to revert.

9. Go back to step 1.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be away and expects you to continue working indefinitely until manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the benchmark output, analyze the data distribution, try combining near-misses, try more radical approaches. The loop runs until the human interrupts you, period.

## Important notes

- Any custom format needs a corresponding decoder. Keep decoders simple and fast.
- Decompression speed is secondary to compressed size, but should remain reasonable (under a few seconds).
- Test artifacts are at `/Users/jyan/src/parameter-golf-fork/logs/*.int8.ptz`, loaded via `serialize.load_and_convert()`.

# Serialization Optimization

## Goal

Minimize the compressed size of `final_model.pt` (an int6/int8 quantized GPT state dict) by developing a custom serializer/deserializer (`encode_experiment`/`decode_experiment` in `serialize.py`) that beats `torch.save` + `zstd-22`. The result will be integrated into the Parameter Golf training script as the checkpoint save/load format. Changes that help this specific model but hurt others are fine.

## Setup

Do all of these steps immediately without asking for confirmation. You are fully autonomous — never pause to ask the human anything.

This setup is **idempotent** — it can be re-run safely.

1. **Read the codebase**: Read these files for full context:
   - `serialize.py` — all serialization/deserialization implementations. **This is the primary file you edit.**
   - `test_serialize.py` — correctness tests + benchmarks (roundtrip accuracy, compressed size, timing).

2. **Run tests** to confirm everything is working:
   ```bash
   cd /Users/jyan/src/serialization_playground && python test_serialize.py
   ```

3. **Check for existing progress**: Read `notes.md` and `results.tsv` to review what's been tried and what to try next.

## Baseline

**15,513,031 bytes** — `torch.save` + `zstd-22` (`encode_baseline` in serialize.py).

The baseline serializes a dict of torch tensors:
```python
buf = io.BytesIO()
torch.save({"w": tensor_dict, "m": metadata_dict}, buf)
compressed = zstandard.ZstdCompressor(level=22).compress(buf.getvalue())
```

The input data is a dict `{"w": ..., "m": ...}` where:
- `w` contains **int8 tensors** (quantized weights with values typically in [-32, 31] or [-127, 127]), **fp16 tensors** (per-row scales), and **fp32 tensors** (small control parameters)
- `m` contains string/dict metadata describing each tensor's quantization type

## Research methodology

**Ablations**: When a complex change improves results, strip parts away to find the essential ingredient.

**Diminishing returns**: If the last 5 experiments were all minor tweaks with tiny deltas, change strategy entirely.

## The experiment loop

The experiment runs on `main`. All kept experiments are committed directly to `main`.

LOOP FOREVER:

1. **Review context**: Read `notes.md` and `results.tsv` to review what's been tried, what worked, and what to try next.

2. **Design the next experiment** based on insights from the notebook. Check the "Ideas queue" section of `notes.md` first. Make a change to `serialize.py`.

3. `git commit -am "description of experiment"`

4. **Run tests + benchmark**:
   ```bash
   cd /Users/jyan/src/serialization_playground && python test_serialize.py
   ```
   If tests fail, fix the issue before proceeding. If the approach is fundamentally broken, revert and try something else.

5. **Record results** in `results.tsv` (tab-separated, with status column):
   ```
   experiment	compressed_bytes	raw_bytes	ratio	max_abs_error	status	description
   ```
   The `status` column must be one of: `baseline`, `kept`, or `reverted`.

6. **Update `notes.md`**: Append an entry to the "Experiment log" with the hypothesis, result, and insights. Do this for every experiment — successes and failures both contain useful information.
   - **Mark kept/reverted clearly** in each entry heading (e.g., "### Exp 8: ... — KEPT" or "— REVERTED").
   - **Update the "Ideas queue"**: Add new ideas sparked by this experiment. Remove ideas you just tried.

7. **Keep/discard decision**:
   - If compressed_bytes **decreased** with acceptable roundtrip error (0.0 for lossless): keep the commit.
   - If compressed_bytes is **equal or worse**: `git reset --hard HEAD~1` to revert.

8. Go back to step 1.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be away and expects you to continue working indefinitely until manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the benchmark output, analyze the data distribution, try combining near-misses, try more radical approaches. The loop runs until the human interrupts you, period.

## Important notes

- Any custom format needs a corresponding decoder. Keep decoders simple and fast.
- Decompression speed is secondary to compressed size, but should remain reasonable (under a few seconds).
- Test artifact: `final_model.pt` in this repo, loaded via `serialize.load_and_quantize()`.

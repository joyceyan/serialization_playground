# Serialization Playground

Autonomous experiment loop for optimizing model serialization/compression for the OpenAI Parameter Golf challenge.

Read `program.md` for full setup and experiment methodology.

## Critical constraints

- **Goal**: Minimize compressed artifact size vs the baseline (`torch.save` + `zstd-22`). Every byte saved here means more room for model parameters in the competition.
- **Roundtrip correctness**: Every serialization scheme MUST produce identical dequantized tensors (or acceptably close — measure max absolute error).
- **One change at a time**: Exactly one modification per experiment. Never combine changes.
- **Venv**: Always use `source /Users/jyan/src/parameter-golf-fork/.venv/bin/activate &&` before python commands.
- **No new packages**: Only use numpy, zlib, zstandard, pickle, struct, io, and stdlib.
- **Test artifacts**: Use real artifacts from `/Users/jyan/src/parameter-golf-fork/logs/*.int8.ptz`.

## Experiment loop checklist

Every iteration, follow these steps in order:

1. **Read** `notes.md` and `results.tsv` to understand what's been tried.
2. **Design** one experiment. Edit `serialize.py` to implement the new scheme.
3. **Run tests**: `source /Users/jyan/src/parameter-golf-fork/.venv/bin/activate && python test_serialize.py`
4. **Run benchmark**: `source /Users/jyan/src/parameter-golf-fork/.venv/bin/activate && python benchmark.py`
5. **Log to `results.tsv`** (tab-separated):
   ```
   experiment	compressed_bytes	raw_bytes	ratio	max_abs_error	description
   ```
6. **Update `notes.md`**: hypothesis, result, insights.
7. **Keep/discard**: smaller compressed size with acceptable error → keep. Otherwise → `git checkout -- serialize.py`.
8. **Go to step 1. NEVER STOP.**

## Key principles

- **Measure everything**: Always compare against the baseline (pickle + zlib-9).
- **Bit-level thinking**: Every wasted bit × 17M values = significant bytes.
- **Compression-friendly**: Patterns that compress well matter more than raw size.
- **Roundtrip accuracy**: Any lossy scheme must measure and report max absolute error.

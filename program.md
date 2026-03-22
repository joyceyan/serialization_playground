# Serialization Optimization

## Goal

Beat `torch.save` + `zstd-22` on compressed size while preserving lossless roundtrip accuracy.

## Setup

Do all of these steps immediately without asking for confirmation. You are fully autonomous — never pause to ask the human anything.

This setup is **idempotent** — it can be re-run safely.

1. **Read the codebase**: Read these files for full context:
   - `serialize.py` — all serialization/deserialization implementations.
   - `torch_fork_repo/torch/serialization.py` — **forked torch serialization (Python). Primary file to edit in Phase 3.**
   - `torch_fork_repo/caffe2/serialize/inline_container.cc` — forked C++ ZIP writer (requires rebuild after changes).
   - `test_serialize.py` — correctness tests + benchmarks (roundtrip accuracy, compressed size, timing).

2. **Run tests** to confirm everything is working:
   ```bash
   cd /Users/jyan/src/serialization_playground && ./run_fork.sh python test_serialize.py
   ```

3. **Check for existing progress**: Read `notes_phase3.md` and `results_phase3.tsv` to review what's been tried and what to try next.

## Baselines

### Original baseline: torch.save + zstd-22
**15,513,031 bytes** — this is the number to beat.

### Fork baseline: torch_fork_repo + zstd-22
**15,513,031 bytes** (byte-identical to original — true fork, starting point for Phase 3).

The fork is a full build of PyTorch from source at commit `449b1768` (v2.10.0). Run via `./run_fork.sh` which sets PYTHONPATH to use the fork. Edit `torch_fork_repo/torch/serialization.py` for Python changes (immediate effect) or `torch_fork_repo/caffe2/serialize/inline_container.cc` for C++ changes (requires `cd torch_fork_repo && python setup.py build_ext --inplace`).

### Phase 2 custom format (reference only)
**15,327,351 bytes** (-186KB, -1.20%) — achieved by the fully custom `encode_experiment` format. This is tracked separately in `notes.md` and `results.tsv`. Phase 3 does not need to beat this number.

## Input data

The input data is a dict `{"w": ..., "m": ...}` where:
- `w` contains **int8 tensors** (quantized weights with values typically in [-32, 31] or [-127, 127]), **fp16 tensors** (per-row scales), and **fp32 tensors** (small control parameters)
- `m` contains string/dict metadata describing each tensor's quantization type

## Phase 3: Forked torch.save experiments

**Approach**: Modify the actual PyTorch serialization pipeline by editing a full fork of PyTorch built from source at `torch_fork_repo/`. The fork is byte-identical to stock torch at the start — every experiment modifies the fork to reduce compressed size.

**Key files to edit**:
- `torch_fork_repo/torch/serialization.py` — Python serialization (`_save`, `_load`). Changes take effect immediately.
- `torch_fork_repo/caffe2/serialize/inline_container.cc` — C++ ZIP writer (alignment, CRC32, format). Requires rebuild.
- `torch_fork_repo/third_party/miniz-3.0.2/miniz.c` — ZIP/deflate library. Requires rebuild.
- `serialize.py` — `encode_fork_baseline()` / `decode_fork_baseline()` wrappers, and any new fork-based schemes.

**How to run**:
```bash
./run_fork.sh python test_serialize.py    # Run tests with fork
./run_fork.sh python -c "..."             # Any command with fork
```

**How to rebuild C++ after changes**:
```bash
cd torch_fork_repo && MAX_JOBS=12 CXXFLAGS="-I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1 -I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include" CFLAGS="-I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include" python setup.py build_ext --inplace
```

**Tracking files**:
- `notes_phase3.md` — experiment log, ideas queue, known techniques from prior work
- `results_phase3.tsv` — results (baseline: torch.save at 15,513,031)

**Phase 2 files** (do not modify, reference only):
- `notes.md` — Phase 2 experiment log
- `results.tsv` — Phase 2 results

## Research methodology

**Ablations**: When a complex change improves results, strip parts away to find the essential ingredient.

**Diminishing returns**: If the last 5 experiments were all minor tweaks with tiny deltas, change strategy entirely.

## The experiment loop

The experiment runs on `main`. All kept experiments are committed directly to `main`.

LOOP FOREVER:

1. **Review context**: Read `notes_phase3.md` and `results_phase3.tsv` to review what's been tried, what worked, and what to try next.

2. **Design the next experiment** based on insights from the notebook. Check the "Ideas queue" section of `notes_phase3.md` first. Make exactly one change to the fork (`torch_fork_repo/torch/serialization.py` or C++ files) or `serialize.py`.

3. `git commit -am "Phase 3: description of experiment"`

4. **Run tests + benchmark**:
   ```bash
   cd /Users/jyan/src/serialization_playground && ./run_fork.sh python test_serialize.py
   ```
   If tests fail, fix the issue before proceeding. If the approach is fundamentally broken, revert and try something else.

5. **Record results** in `results_phase3.tsv` (tab-separated, with status column):
   ```
   experiment	compressed_bytes	raw_bytes	ratio	max_abs_error	status	description
   ```
   The `status` column must be one of: `baseline`, `kept`, or `reverted`.

6. **Update `notes_phase3.md`**: Append an entry to the "Experiment log" with the hypothesis, result, and insights. Do this for every experiment — successes and failures both contain useful information.
   - **Mark kept/reverted clearly** in each entry heading (e.g., "### Exp P3-1: ... — KEPT" or "— REVERTED").
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

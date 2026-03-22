# Serialization Playground

Autonomous experiment loop for optimizing model serialization/compression.

Read `program.md` for full setup and experiment methodology.

## Critical constraints

- **Goal**: Minimize compressed artifact size vs the baseline (`torch.save` + `zstd-22`).
- **Roundtrip correctness**: Every serialization scheme MUST produce identical dequantized tensors (or acceptably close — measure max absolute error).
- **Packages**: torch, numpy, zstandard, and stdlib. See requirements.txt.
- **Test artifact**: `final_model.pt` (H100 unquantized state dict). Loaded and quantized via `serialize.load_and_quantize()`.

## Phase 3 — Forked torch.save experiments (ACTIVE)

The current experiment phase modifies a **full fork of PyTorch** built from source at `torch_fork_repo/`.

**All commands must use `./run_fork.sh`** to ensure the forked torch is loaded:
```bash
cd /Users/jyan/src/serialization_playground && ./run_fork.sh python test_serialize.py
```

**Files to edit**:
- `torch_fork_repo/torch/serialization.py` — Python serialization. Changes take effect immediately.
- `torch_fork_repo/caffe2/serialize/inline_container.cc` — C++ ZIP writer. Requires rebuild after changes.
- `serialize.py` — compression wrappers around torch.save/load.

**Rebuild C++ after changes**:
```bash
cd torch_fork_repo && MAX_JOBS=12 CXXFLAGS="-I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1 -I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include" CFLAGS="-I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include" python setup.py build_ext --inplace
```

**Tracking**: `notes_phase3.md` and `results_phase3.tsv`.

## Key principles

- **Measure everything**: Always compare against the baseline (`torch.save` + `zstd-22` = 15,513,031 bytes).
- **Bit-level thinking**: Every wasted bit × 26M values = significant bytes.
- **Compression-friendly**: Patterns that compress well matter more than raw size.
- **Roundtrip accuracy**: Any lossy scheme must measure and report max absolute error.

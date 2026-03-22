# Lab Notebook — Phase 3: Forked torch.save Experiments

## Goal

Beat the original `torch.save` + `zstd-22` baseline (**15,513,031 bytes**) by modifying a pure-Python fork of torch's serialization pipeline (`torch_save_fork.py`), rather than writing a fully custom format from scratch.

## Setup

Input: `final_model.pt` — unquantized state dict from 11L 512dim MLP3x model trained on 8xH100.
Pipeline: `mixed_quantize_int6(sd, {"mlp", "attn"})` → `torch_save_fork.save` + `zstd-22`.

**Original torch.save baseline**: 15,513,031 bytes
**Fork baseline**: 15,513,031 bytes (byte-identical — true fork, not a reimplementation)

## Data characteristics

- 68 tensors total: 66 int6-quantized (stored as int8), 2 int8-quantized, plus fp16 scales and fp32 control params
- Raw bytes: 27,006,130
- Int8 values: ~26.7M, Laplace distribution, 6-12% zeros, 4.67 bits per-symbol entropy
- fp16 scales: ~85KB raw, high-byte very repetitive (similar exponents)
- fp32 control: ~182KB raw

## Known techniques from prior experiments (Phase 2 custom format)

These strategies were proven effective in 55+ experiments on the same data. They're listed here as a reference — the fork should discover which ones are applicable within the pickle+storage framework.

**Confirmed winners (in order of impact):**
1. **Separate streams by dtype** (-50KB): Grouping all int8, fp16, fp32 data separately before compression lets the compressor build better models for each data type
2. **Byte-shuffle fp16** (-14KB): Separating high/low bytes of fp16 values exposes repetitive exponent bytes
3. **Zigzag encoding for int8** (-3.3KB): Maps signed values to unsigned (0,-1,1,-2,... → 0,1,2,3,...), clustering frequent values near zero
4. **zstd dictionary** (-1.3KB): 256-byte trained dictionary bootstraps compression context
5. **LZMA for fp16** (-1.6KB): LZMA beats zstd for the small, repetitive fp16 stream
6. **Transpose 2D tensors** (-0.6KB): Column-major layout has slightly better compressibility
7. **Compact metadata encoding**: JSON+LZMA header with abbreviated keys, type strings

**Confirmed failures (avoid these):**
- Bit-packing / nibble split: destroys byte alignment zstd needs (+12.4%)
- Delta/prediction filters: adjacent weights are uncorrelated (+9.6%)
- Row interleaving across tensors: breaks within-tensor locality
- Different compressors for int8 (LZMA, brotli, zlib): all worse than zstd-22
- Fine-grained stream splitting (>3 streams): frame overhead exceeds benefit

## Ideas queue

### Priority 1: Low-hanging fruit within the fork
- Pickle protocol 5 (Python 3.8+): more compact framing, out-of-band data buffers
- Reorder storages by dtype before writing (all int8 storages first, then fp16, then fp32)
- Strip storage length prefixes (derive sizes from pickle metadata on load)
- Remove the per-storage `nbytes` field entirely — encode total sizes in header

### Priority 2: Apply proven transforms inside the fork
- Apply zigzag encoding to int8 storages before writing
- Apply transpose to 2D storages before writing
- Byte-shuffle fp16 storages before writing
- Use separate compression for int8 vs fp16 vs fp32 storage regions

### Priority 3: Fork-specific opportunities
- Custom pickle reducer for tensors (reduce metadata per tensor)
- Merge adjacent storages of the same dtype into one contiguous block
- Write a single "storage region" per dtype instead of per-tensor
- Hybrid: use fork's pickle for structure + custom binary for storage data

## Experiment log

### Exp P3-1: Reorder storages by dtype — KEPT

**Hypothesis**: Sorting storage writes so all int8 data comes first, then fp16, then fp32 groups same-dtype data contiguously. After zstd-22 compression, this should improve compression because the compressor sees long runs of same-type data.
**Change**: Added dtype-based sorting of `serialized_storages` keys before the write loop in `torch_fork_repo/torch/serialization.py:1253`.
**Result**: 15,384,952 bytes (-128,079 = **-0.83%** vs 15,513,031 baseline). Major win!
**Insight**: This is the torch.save equivalent of Phase 2's "separate streams by dtype" (-1.05%). The pickle+ZIP format adds overhead that limits the benefit compared to Phase 2's custom format, but grouping by dtype is still very effective even within the standard format.

### Exp P3-2: Sort by size within dtype — REVERTED

**Hypothesis**: Putting largest tensors first within each dtype group would help zstd build better context.
**Result**: 15,579,437 bytes (+194,485 vs P3-1, worse than original baseline!). Sorting by size completely breaks the natural tensor ordering that zstd exploits. The original insertion order preserves layer-by-layer locality which is critical.
**Insight**: Within each dtype group, the natural pickle traversal order (which follows dict insertion order = layer-by-layer) is optimal. Only the inter-dtype ordering should be changed.

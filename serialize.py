"""
Serialization schemes for Parameter Golf model artifacts.

Operates on the SOTA format:
  - Input: final_model.pt (unquantized state dict from H100 training)
  - Quantization: mixed_quantize_int6 (int6 for attn+MLP, int8 for embedding)
  - Baseline serialization: torch.save + zstd-22
  - Goal: beat the baseline on compressed size with lossless roundtrip
"""

from __future__ import annotations

import io
import json
import lzma
import pickle
import re
import struct
import time
import zlib
from typing import Any

import numpy as np
import torch
from torch import Tensor

try:
    import zstandard
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


# ==============================================================================
# QUANTIZATION (from train_gpt_submit.py)
# ==============================================================================

CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
    "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights", "smear",
)

INT8_CLIP_Q = 99.99984 / 100.0
INT8_PER_ROW_SCALE_DTYPE = torch.float16


def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_int6_per_row(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        row_max = t32.abs().amax(dim=1)
        scale = (row_max / 31.0).clamp_min(1.0 / 31.0).to(torch.float16)
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -32, 31).to(torch.int8)
        return q, scale
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / 31.0 if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -32, 31).to(torch.int8)
    return q, scale


def mixed_quantize_int6(state_dict: dict[str, Tensor], int6_cats: set[str]):
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if cat in int6_cats and t.ndim >= 1:
            q, s = quantize_int6_per_row(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
        else:
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta


# ==============================================================================
# LOAD + QUANTIZE
# ==============================================================================

def load_and_quantize(path: str = "final_model.pt") -> tuple[dict[str, Tensor], dict[str, object]]:
    """Load an unquantized model and quantize to mixed int6/int8."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    return mixed_quantize_int6(sd, {"mlp", "attn"})


# ==============================================================================
# BASELINE: torch.save + zstd-22
# ==============================================================================

def encode_baseline(quant_result: dict[str, Tensor], quant_meta: dict[str, object]) -> bytes:
    """Baseline: torch.save({"w": ..., "m": ...}) + zstd-22."""
    if not HAS_ZSTD:
        raise ImportError("zstandard not installed")
    buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, buf)
    raw = buf.getvalue()
    return zstandard.ZstdCompressor(level=22).compress(raw)


def decode_baseline(blob: bytes) -> tuple[dict[str, Tensor], dict[str, object]]:
    """Baseline decoder."""
    if not HAS_ZSTD:
        raise ImportError("zstandard not installed")
    raw = zstandard.ZstdDecompressor().decompress(blob)
    obj = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    return obj["w"], obj["m"]


# ==============================================================================
# EXPERIMENT: separate streams by dtype, transpose, zstd-22
# ==============================================================================

def _sorted_keys(d):
    return sorted(d.keys())


def encode_experiment(quant_result: dict[str, Tensor], quant_meta: dict[str, object]) -> bytes:
    """Separate int8/fp16/fp32 into distinct streams, transpose 2D, zstd-22 each."""
    if not HAS_ZSTD:
        raise ImportError("zstandard not installed")
    comp = zstandard.ZstdCompressor(level=22)

    # Classify tensors into streams by dtype
    int8_keys = []
    fp16_keys = []
    fp32_keys = []
    for k in _sorted_keys(quant_result):
        t = quant_result[k]
        if t.dtype == torch.int8:
            int8_keys.append(k)
        elif t.dtype == torch.float16:
            fp16_keys.append(k)
        else:
            fp32_keys.append(k)

    # Build raw byte streams — transpose 2D tensors
    streams = {}
    # int8 stream: zigzag encode (maps 0,-1,1,-2,2,... to 0,1,2,3,4,...)
    int8_parts = []
    for k in int8_keys:
        t = quant_result[k]
        if t.ndim == 2:
            t = t.t().contiguous()
        arr = t.numpy().astype(np.int16)
        # Reversed zigzag: 0→0, 1→1, -1→2, 2→3, -2→4, ...
        # This is: positive v → 2v-1, negative v → -2v, zero → 0
        zigzag = np.where(arr > 0, 2*arr - 1, -2*arr).astype(np.uint8)
        int8_parts.append(zigzag.tobytes())
    if int8_parts:
        streams["int8"] = comp.compress(b"".join(int8_parts))

    # fp16 stream: byte-shuffle (separate high and low bytes)
    fp16_parts = []
    for k in fp16_keys:
        t = quant_result[k]
        if t.ndim == 2:
            t = t.t().contiguous()
        fp16_parts.append(t.numpy().tobytes())
    if fp16_parts:
        raw = b"".join(fp16_parts)
        arr = np.frombuffer(raw, dtype=np.uint8)
        high = arr[0::2].tobytes()  # high bytes
        low = arr[1::2].tobytes()   # low bytes
        streams["fp16"] = lzma.compress(high + low, preset=9 | lzma.PRESET_EXTREME)

    # fp32 stream: byte-shuffle (4 sub-streams)
    fp32_parts = []
    for k in fp32_keys:
        t = quant_result[k]
        if t.ndim == 2:
            t = t.t().contiguous()
        fp32_parts.append(t.numpy().tobytes())
    if fp32_parts:
        raw = b"".join(fp32_parts)
        arr = np.frombuffer(raw, dtype=np.uint8)
        shuffled = b"".join(arr[i::4].tobytes() for i in range(4))
        streams["fp32"] = comp.compress(shuffled)

    # Encode metadata as JSON + LZMA (smaller than pickle + zstd)
    header = json.dumps({
        "i": int8_keys,
        "f": fp16_keys,
        "g": fp32_keys,
        "s": {k: list(quant_result[k].shape) for k in _sorted_keys(quant_result)},
        "t": sorted(k for k in _sorted_keys(quant_result) if quant_result[k].ndim == 2),
        "b": 1,
        "m": quant_meta,
    }, separators=(",", ":")).encode()
    header_c = lzma.compress(header, preset=9 | lzma.PRESET_EXTREME)

    # Pack: [header_len(4)] [header_compressed] [int8_len(4)] [int8_compressed] [fp16_len(4)] [fp16_compressed] [fp32_compressed]
    out = struct.pack("<I", len(header_c)) + header_c
    for label in ["int8", "fp16", "fp32"]:
        blob = streams.get(label, b"")
        out += struct.pack("<I", len(blob)) + blob
    return out


def decode_experiment(blob: bytes) -> tuple[dict[str, Tensor], dict[str, object]]:
    """Decode separate-stream format."""
    if not HAS_ZSTD:
        raise ImportError("zstandard not installed")
    decomp = zstandard.ZstdDecompressor()

    off = 0
    def read_block():
        nonlocal off
        sz = struct.unpack_from("<I", blob, off)[0]
        off += 4
        data = blob[off:off + sz]
        off += sz
        return data

    raw_header = read_block()
    header_json = json.loads(lzma.decompress(raw_header))
    # Remap short keys to original names for compatibility
    header = {
        "int8_keys": header_json["i"],
        "fp16_keys": header_json["f"],
        "fp32_keys": header_json["g"],
        "shapes": header_json["s"],
        "transposed": set(header_json["t"]),
        "byte_shuffle": bool(header_json.get("b")),
        "meta": header_json["m"],
    }
    int8_raw_zigzag = decomp.decompress(read_block()) if header["int8_keys"] else b""
    # Reverse zigzag encoding
    if int8_raw_zigzag:
        arr = np.frombuffer(int8_raw_zigzag, dtype=np.uint8).astype(np.int16)
        # Reversed zigzag decode: odd → positive, even → negative
        # v=0 → 0, v=1 → 1, v=2 → -1, v=3 → 2, v=4 → -2, ...
        decoded = np.where(arr % 2 == 1, (arr + 1) // 2, -(arr // 2)).astype(np.int8)
        int8_raw = decoded.tobytes()
    else:
        int8_raw = b""
    fp16_block = read_block()
    fp16_raw_shuffled = lzma.decompress(fp16_block) if header["fp16_keys"] else b""
    fp32_raw_shuffled = decomp.decompress(read_block()) if header["fp32_keys"] else b""

    # Unshuffle fp16: high bytes then low bytes → interleave
    if fp16_raw_shuffled and header.get("byte_shuffle"):
        half = len(fp16_raw_shuffled) // 2
        high = np.frombuffer(fp16_raw_shuffled[:half], dtype=np.uint8)
        low = np.frombuffer(fp16_raw_shuffled[half:], dtype=np.uint8)
        interleaved = np.empty(len(fp16_raw_shuffled), dtype=np.uint8)
        interleaved[0::2] = high
        interleaved[1::2] = low
        fp16_raw = interleaved.tobytes()
    else:
        fp16_raw = fp16_raw_shuffled

    # Unshuffle fp32: 4 sub-streams → interleave
    if fp32_raw_shuffled and header.get("byte_shuffle"):
        quarter = len(fp32_raw_shuffled) // 4
        subs = [np.frombuffer(fp32_raw_shuffled[i*quarter:(i+1)*quarter], dtype=np.uint8) for i in range(4)]
        interleaved = np.empty(len(fp32_raw_shuffled), dtype=np.uint8)
        for i in range(4):
            interleaved[i::4] = subs[i]
        fp32_raw = interleaved.tobytes()
    else:
        fp32_raw = fp32_raw_shuffled

    dtype_map = {"int8": (torch.int8, int8_raw, np.int8),
                 "fp16": (torch.float16, fp16_raw, np.float16),
                 "fp32": (torch.float32, fp32_raw, np.float32)}

    w = {}
    transposed = header.get("transposed", set())
    offsets = {"int8": 0, "fp16": 0, "fp32": 0}
    for label, keys in [("int8", header["int8_keys"]), ("fp16", header["fp16_keys"]), ("fp32", header["fp32_keys"])]:
        dt, raw, npdt = dtype_map[label]
        for k in keys:
            shape = header["shapes"][k]
            numel = 1
            for s in shape:
                numel *= s
            nbytes = numel * np.dtype(npdt).itemsize
            arr = np.frombuffer(raw, dtype=npdt, count=numel, offset=offsets[label])
            offsets[label] += nbytes
            if k in transposed:
                t = torch.from_numpy(arr.copy()).reshape(shape[1], shape[0]).t().contiguous()
            else:
                t = torch.from_numpy(arr.copy()).reshape(shape)
            w[k] = t
    return w, header["meta"]


# ==============================================================================
# HELPERS
# ==============================================================================

def measure_scheme(
    name: str,
    encode_fn,
    decode_fn,
    quant_result: dict[str, Tensor],
    quant_meta: dict[str, object],
    n_trials: int = 3,
) -> dict:
    """Benchmark a serialization scheme. Returns metrics dict."""
    encode_times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        blob = encode_fn(quant_result, quant_meta)
        encode_times.append(1000.0 * (time.perf_counter() - t0))

    compressed_bytes = len(blob)

    decode_times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        decoded_w, decoded_m = decode_fn(blob)
        decode_times.append(1000.0 * (time.perf_counter() - t0))

    # Roundtrip accuracy
    max_abs_error = 0.0
    for key in quant_result:
        orig = quant_result[key]
        rt = decoded_w.get(key)
        if rt is None:
            max_abs_error = float("inf")
            continue
        orig_np = orig.numpy() if isinstance(orig, Tensor) else np.asarray(orig)
        rt_np = rt.numpy() if isinstance(rt, Tensor) else np.asarray(rt)
        if orig_np.shape != rt_np.shape:
            max_abs_error = float("inf")
            continue
        diff = np.abs(orig_np.astype(np.float32) - rt_np.astype(np.float32))
        max_abs_error = max(max_abs_error, float(diff.max()))

    return {
        "name": name,
        "compressed_bytes": compressed_bytes,
        "encode_ms": min(encode_times),
        "decode_ms": min(decode_times),
        "max_abs_error": max_abs_error,
    }

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
        # Standard zigzag: 0→0, -1→1, 1→2, -2→3, 2→4, ...
        zigzag = ((arr << 1) ^ (arr >> 15)).astype(np.uint8)
        int8_parts.append(zigzag.tobytes())
    if int8_parts:
        int8_blob = b"".join(int8_parts)
        # Train a small dictionary for better compression
        chunk_size = 65536
        samples = [int8_blob[i:i+chunk_size] for i in range(0, min(len(int8_blob), 10_000_000), chunk_size)]
        dict_data = zstandard.train_dictionary(256, samples[:100])
        dict_bytes = dict_data.as_bytes()
        comp_dict = zstandard.ZstdCompressor(level=22, dict_data=dict_data, write_content_size=False)
        compressed = comp_dict.compress(int8_blob)
        dict_c = zlib.compress(dict_bytes, 9)
        streams["int8"] = struct.pack("B", len(dict_c)) + dict_c + compressed

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
        # Use raw LZMA2 with lp=1 (byte-shuffled data has 2-byte period)
        filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 1, "pb": 0}]
        streams["fp16"] = lzma.compress(high + low, format=lzma.FORMAT_RAW, filters=filters)

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

    # Encode metadata as JSON + LZMA — indexed format for compactness
    all_keys = _sorted_keys(quant_result)
    # Compact meta: derive meta_keys from tensor keys, encode types as string
    meta_keys = []
    for k in all_keys:
        if k.endswith(".q") or k.endswith(".scale"):
            base = k.rsplit(".", 1)[0]
            if base not in meta_keys:
                meta_keys.append(base)
        elif k not in meta_keys:
            meta_keys.append(k)
    type_str = ""
    for mk in meta_keys:
        info = quant_meta.get(mk, "passthrough")
        if info == "passthrough": type_str += "p"
        elif info == "passthrough_ctrl": type_str += "c"
        elif isinstance(info, dict) and info.get("type") == "int6": type_str += "6"
        elif isinstance(info, dict) and info.get("type") == "int8": type_str += "8"
        else: type_str += "p"

    # Abbreviate key names for compactness
    def _shorten(k):
        return k.replace("blocks.", "B").replace(".attn.", ".a.").replace(".mlp.", ".m.").replace(".weight", ".w").replace(".scale", ".s").replace(".proj.", ".p.")

    short_keys = [_shorten(k) for k in all_keys]

    header = json.dumps({
        "k": short_keys,
        "s": [list(quant_result[k].shape) for k in all_keys],
        "q": type_str,
    }, separators=(",", ":")).encode()
    # Raw LZMA2 for header too (saves ~32 bytes of .xz container overhead)
    header_filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]
    header_c = lzma.compress(header, format=lzma.FORMAT_RAW, filters=header_filters)

    # Pack: [header_len(2)] [header_compressed] [int8_len(4)] [int8_compressed] [fp16_len(4)] [fp16_compressed] [fp32_len(4)?] [fp32_compressed?]
    out = struct.pack("<H", len(header_c)) + header_c
    for label in ["int8", "fp16"]:
        blob = streams.get(label, b"")
        out += struct.pack("<I", len(blob)) + blob
    # Only include fp32 block if non-empty
    fp32_blob = streams.get("fp32", b"")
    if fp32_blob:
        out += struct.pack("<I", len(fp32_blob)) + fp32_blob
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

    # Header uses 2-byte length prefix
    header_sz = struct.unpack_from("<H", blob, off)[0]
    off += 2
    raw_header = blob[off:off + header_sz]
    off += header_sz
    header_filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]
    header_json = json.loads(lzma.decompress(raw_header, format=lzma.FORMAT_RAW, filters=header_filters))
    # Decode indexed format
    def _expand(k):
        return k.replace("B", "blocks.").replace(".a.", ".attn.").replace(".m.", ".mlp.").replace(".w", ".weight").replace(".s", ".scale").replace(".p.", ".proj.")

    all_keys = [_expand(k) for k in header_json["k"]]
    shapes_list = header_json["s"]
    # Derive dtype classification from key suffixes (.q → int8, else → fp16)
    int8_indices = [i for i, k in enumerate(all_keys) if k.endswith(".q")]
    fp16_indices = [i for i, k in enumerate(all_keys) if not k.endswith(".q")]
    fp32_indices = []  # No fp32 in this format

    # Reconstruct quant_meta from type_str
    type_str = header_json.get("q", "")
    meta_keys = []
    for k in all_keys:
        if k.endswith(".q") or k.endswith(".scale"):
            base = k.rsplit(".", 1)[0]
            if base not in meta_keys:
                meta_keys.append(base)
        elif k not in meta_keys:
            meta_keys.append(k)
    type_map = {"p": "passthrough", "c": "passthrough_ctrl", "6": {"type": "int6"}, "8": {"type": "int8"}}
    quant_meta = {mk: type_map.get(type_str[i], "passthrough") for i, mk in enumerate(meta_keys)}

    header = {
        "int8_keys": [all_keys[i] for i in int8_indices],
        "fp16_keys": [all_keys[i] for i in fp16_indices],
        "fp32_keys": [all_keys[i] for i in fp32_indices],
        "shapes": {all_keys[i]: shapes_list[i] for i in range(len(all_keys))},
        "transposed": set(all_keys[i] for i in range(len(all_keys)) if len(shapes_list[i]) == 2),
        "byte_shuffle": True,
        "meta": quant_meta,
    }
    int8_block = read_block()
    if header["int8_keys"] and int8_block:
        dict_c_len = int8_block[0]
        dict_bytes = zlib.decompress(int8_block[1:1 + dict_c_len])
        dict_data = zstandard.ZstdCompressionDict(dict_bytes)
        decomp_dict = zstandard.ZstdDecompressor(dict_data=dict_data)
        int8_raw_zigzag = decomp_dict.decompress(int8_block[1 + dict_c_len:])
    else:
        int8_raw_zigzag = b""
    # Reverse zigzag encoding
    if int8_raw_zigzag:
        arr = np.frombuffer(int8_raw_zigzag, dtype=np.uint8).astype(np.int16)
        # Standard zigzag decode: (v >>> 1) ^ -(v & 1)
        decoded = ((arr >> 1) ^ -(arr & 1)).astype(np.int8)
        int8_raw = decoded.tobytes()
    else:
        int8_raw = b""
    fp16_block = read_block()
    if header["fp16_keys"]:
        filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 1, "pb": 0}]
        fp16_raw_shuffled = lzma.decompress(fp16_block, format=lzma.FORMAT_RAW, filters=filters)
    else:
        fp16_raw_shuffled = b""
    # fp32 block is optional (omitted when empty)
    if header["fp32_keys"] and off < len(blob):
        fp32_raw_shuffled = decomp.decompress(read_block())
    else:
        fp32_raw_shuffled = b""

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

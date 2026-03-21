"""
Serialization schemes for Parameter Golf model artifacts.

Matches the train_gpt_sota.py pipeline exactly:
  - Quantization: mixed_quantize_int6 (int6 for attn+MLP, int8 for embedding)
  - Serialization: torch.save({"w": result, "m": meta}, buf)
  - Compression: zstd-22

The sota_obj format (what gets serialized) has keys:
  "w": dict of torch tensors — includes "name.q" (int8), "name.scale" (fp16),
       and passthrough tensors (fp16/fp32)
  "m": dict of metadata — per-tensor: "passthrough", "passthrough_ctrl",
       {"type": "int6"}, or {"type": "int8"}
"""

from __future__ import annotations

import bz2
import io
import lzma
import pickle
import struct
import time
import zlib
from typing import Any

import numpy as np
import torch

try:
    import zstandard
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


# ==============================================================================
# CONTROL TENSOR PATTERNS (from train_gpt_sota.py)
# ==============================================================================

CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
    "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights", "smear",
)


# ==============================================================================
# CONVERT MLX ARTIFACT → SOTA FORMAT
# ==============================================================================

def _classify_param(name: str) -> str:
    """Classify a parameter name into embed/mlp/attn/other (from train_gpt_sota.py)."""
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"


def quantize_int6_per_row(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize to int6 [-32, 31] stored in int8 containers."""
    f32 = arr.astype(np.float32)
    if f32.ndim == 2:
        row_max = np.abs(f32).max(axis=1)
        scale = np.maximum(row_max / 31.0, 1.0 / 31.0).astype(np.float16)
        q = np.clip(np.round(f32 / scale.astype(np.float32)[:, None]), -32, 31).astype(np.int8)
        return q, scale
    amax = float(np.abs(f32).max())
    scale = np.array(amax / 31.0 if amax > 0 else 1.0, dtype=np.float16)
    q = np.clip(np.round(f32 / float(scale)), -32, 31).astype(np.int8)
    return q, scale


def quantize_int8_per_row(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize to int8 [-127, 127] with per-row scale."""
    f32 = arr.astype(np.float32)
    clip_q = 99.99984 / 100.0
    if f32.ndim == 2:
        clip_abs = np.quantile(np.abs(f32), clip_q, axis=1) if f32.size else np.empty((f32.shape[0],), dtype=np.float32)
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / 127.0, 1.0 / 127.0).astype(np.float16)
        q = np.clip(np.round(clipped / scale.astype(np.float32)[:, None]), -127, 127).astype(np.int8)
        return q, scale
    clip_abs = float(np.quantile(np.abs(f32).flatten(), clip_q)) if f32.size else 0.0
    scale = np.array(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=np.float16)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / float(scale)), -127, 127).astype(np.int8)
    return q, scale


def mlx_to_sota_format(mlx_obj: dict, int6_cats: set[str] = {"mlp", "attn"}) -> dict:
    """
    Convert an MLX-format quant_obj to the SOTA format used by train_gpt_sota.py.

    MLX format: separate "quantized", "scales", "passthrough" dicts with original tensor names.
    SOTA format: {"w": {name.q, name.scale, passthrough_names}, "m": {name: metadata}}.

    For block weights: re-quantize from int8 to int6 (lossy — clamps values outside [-32,31]).
    For embedding: keep as int8.
    For passthrough: keep as-is.
    """
    result = {}
    meta = {}

    # Re-quantize the int8 weights into mixed int6/int8
    for name in mlx_obj.get("quantized", {}):
        arr_int8 = np.asarray(mlx_obj["quantized"][name])
        scale_orig = np.asarray(mlx_obj["scales"][name])
        cat = _classify_param(name)

        # Dequantize back to float first
        if scale_orig.ndim > 0:
            f32 = arr_int8.astype(np.float32) * scale_orig.astype(np.float32).reshape(arr_int8.shape[0], *([1] * (arr_int8.ndim - 1)))
        else:
            f32 = arr_int8.astype(np.float32) * float(scale_orig)

        if cat in int6_cats:
            q, s = quantize_int6_per_row(f32)
            result[name + ".q"] = torch.from_numpy(q)
            result[name + ".scale"] = torch.from_numpy(s)
            meta[name] = {"type": "int6"}
        else:
            q, s = quantize_int8_per_row(f32)
            result[name + ".q"] = torch.from_numpy(q)
            result[name + ".scale"] = torch.from_numpy(s)
            meta[name] = {"type": "int8"}

    # Passthrough tensors
    for name in mlx_obj.get("passthrough", {}):
        arr = np.asarray(mlx_obj["passthrough"][name])
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = torch.from_numpy(arr.astype(np.float32))
            meta[name] = "passthrough_ctrl"
        else:
            t = torch.from_numpy(arr.astype(np.float16) if arr.dtype in (np.float32, np.float64) else arr.copy())
            result[name] = t
            meta[name] = "passthrough"

    return {"w": result, "m": meta}


# ==============================================================================
# BASELINE: torch.save + zstd-22 (matches train_gpt_sota.py pipeline)
# ==============================================================================

def encode_baseline(sota_obj: dict) -> bytes:
    """Baseline: torch.save + zstd-22. Matches train_gpt_sota.py exactly."""
    if not HAS_ZSTD:
        raise ImportError("zstandard not installed")
    buf = io.BytesIO()
    torch.save(sota_obj, buf)
    raw = buf.getvalue()
    return zstandard.ZstdCompressor(level=22).compress(raw)


def decode_baseline(blob: bytes) -> dict:
    """Baseline decoder: zstd decompress + torch.load."""
    if not HAS_ZSTD:
        raise ImportError("zstandard not installed")
    raw = zstandard.ZstdDecompressor().decompress(blob)
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)


# ==============================================================================
# EXPERIMENT: Transpose weight matrices + torch.save + zstd-22
# Column-major may compress better if columns have more local correlation.
# ==============================================================================

def encode_transpose_v1(sota_obj: dict) -> bytes:
    """Transpose 2D weight tensors before torch.save + zstd-22."""
    if not HAS_ZSTD:
        raise ImportError("zstandard not installed")

    w = sota_obj["w"]
    m = sota_obj["m"]

    new_w = {}
    for key, tensor in w.items():
        if tensor.ndim == 2:
            new_w[key] = tensor.t().contiguous()
        else:
            new_w[key] = tensor

    buf = io.BytesIO()
    torch.save({"w": new_w, "m": m}, buf)
    return zstandard.ZstdCompressor(level=22).compress(buf.getvalue())


def decode_transpose_v1(blob: bytes) -> dict:
    """Decode transposed format."""
    if not HAS_ZSTD:
        raise ImportError("zstandard not installed")

    raw = zstandard.ZstdDecompressor().decompress(blob)
    obj = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)

    w = obj["w"]
    new_w = {}
    for key, tensor in w.items():
        if tensor.ndim == 2:
            new_w[key] = tensor.t().contiguous()
        else:
            new_w[key] = tensor

    return {"w": new_w, "m": obj["m"]}


# ==============================================================================
# EXPERIMENT: Separate zstd-22 streams per data type
# Compress int8 weights, fp16 scales, fp32 passthrough, and metadata
# each independently, then concatenate. Each gets its own entropy model.
# ==============================================================================

def encode_separate_streams(sota_obj: dict) -> bytes:
    """Separate compression streams per data type + transpose."""
    if not HAS_ZSTD:
        raise ImportError("zstandard not installed")

    w = sota_obj["w"]
    m = sota_obj["m"]
    cctx = zstandard.ZstdCompressor(level=22)

    # Collect bytes by type, transposing 2D tensors
    q_parts = []   # int8 quantized weights
    s_parts = []   # fp16 scales
    p_parts = []   # passthrough (mixed types)
    manifest = []  # (name, dtype_str, shape) for reconstruction

    for name in sorted(w.keys()):
        t = w[name]
        arr = t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
        if arr.ndim == 2:
            arr = arr.T.copy()  # transpose for better compression
        manifest.append((name, str(arr.dtype), list(t.shape), arr.nbytes))

        if name.endswith(".q"):
            q_parts.append(arr.tobytes())
        elif name.endswith(".scale"):
            s_parts.append(arr.tobytes())
        else:
            p_parts.append(arr.tobytes())

    # Compress each stream independently
    q_blob = cctx.compress(b"".join(q_parts)) if q_parts else b""
    s_blob = cctx.compress(b"".join(s_parts)) if s_parts else b""
    p_blob = cctx.compress(b"".join(p_parts)) if p_parts else b""
    meta_blob = cctx.compress(pickle.dumps({"m": m, "manifest": manifest},
                                            protocol=pickle.HIGHEST_PROTOCOL))

    # Pack: [4B q_len][q_blob][4B s_len][s_blob][4B p_len][p_blob][meta_blob]
    out = io.BytesIO()
    for blob in (q_blob, s_blob, p_blob):
        out.write(struct.pack("<I", len(blob)))
        out.write(blob)
    out.write(meta_blob)
    return out.getvalue()


def decode_separate_streams(blob: bytes) -> dict:
    """Decode separate-streams format."""
    if not HAS_ZSTD:
        raise ImportError("zstandard not installed")

    dctx = zstandard.ZstdDecompressor()
    buf = io.BytesIO(blob)

    # Read 3 length-prefixed compressed streams
    streams = []
    for _ in range(3):
        slen = struct.unpack("<I", buf.read(4))[0]
        streams.append(dctx.decompress(buf.read(slen)))
    q_raw, s_raw, p_raw = streams

    # Remaining bytes = compressed metadata
    meta_obj = pickle.loads(dctx.decompress(buf.read()))
    m = meta_obj["m"]
    manifest = meta_obj["manifest"]

    # Reconstruct tensors
    w = {}
    offsets = {"q": 0, "s": 0, "p": 0}
    raw_map = {"q": q_raw, "s": s_raw, "p": p_raw}

    for name, dtype_str, shape, nbytes in manifest:
        if name.endswith(".q"):
            stream_key = "q"
        elif name.endswith(".scale"):
            stream_key = "s"
        else:
            stream_key = "p"

        off = offsets[stream_key]
        raw_bytes = raw_map[stream_key][off:off + nbytes]
        offsets[stream_key] = off + nbytes

        dt = np.dtype(dtype_str)
        # Data was stored transposed for 2D
        if len(shape) == 2:
            arr = np.frombuffer(raw_bytes, dtype=dt).reshape(shape[1], shape[0]).T.copy()
        else:
            arr = np.frombuffer(raw_bytes, dtype=dt).reshape(shape).copy()
        w[name] = torch.from_numpy(arr)

    return {"w": w, "m": m}


# ==============================================================================
# EXPERIMENT: LZMA for weight stream, zstd for rest
# LZMA (LZMA2) may achieve better ratio than zstd for weight data.
# ==============================================================================

def encode_lzma_streams(sota_obj: dict) -> bytes:
    """LZMA for weight stream, zstd-22 for scales/passthrough."""
    if not HAS_ZSTD:
        raise ImportError("zstandard not installed")

    w = sota_obj["w"]
    m = sota_obj["m"]
    cctx = zstandard.ZstdCompressor(level=22)

    q_parts = []
    s_parts = []
    p_parts = []
    manifest = []

    for name in sorted(w.keys()):
        t = w[name]
        arr = t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
        if arr.ndim == 2:
            arr = arr.T.copy()
        manifest.append((name, str(arr.dtype), list(t.shape), arr.nbytes))

        if name.endswith(".q"):
            q_parts.append(arr.tobytes())
        elif name.endswith(".scale"):
            s_parts.append(arr.tobytes())
        else:
            p_parts.append(arr.tobytes())

    # LZMA for weight stream (best ratio), zstd for small streams
    q_blob = lzma.compress(b"".join(q_parts), preset=9)
    s_blob = cctx.compress(b"".join(s_parts)) if s_parts else b""
    p_blob = cctx.compress(b"".join(p_parts)) if p_parts else b""
    meta_blob = cctx.compress(pickle.dumps({"m": m, "manifest": manifest},
                                            protocol=pickle.HIGHEST_PROTOCOL))

    # Mark format with magic byte 'L' for LZMA
    out = io.BytesIO()
    out.write(b"L")
    for blob in (q_blob, s_blob, p_blob):
        out.write(struct.pack("<I", len(blob)))
        out.write(blob)
    out.write(meta_blob)
    return out.getvalue()


def decode_lzma_streams(blob: bytes) -> dict:
    """Decode LZMA-streams format."""
    if not HAS_ZSTD:
        raise ImportError("zstandard not installed")

    dctx = zstandard.ZstdDecompressor()
    buf = io.BytesIO(blob)

    magic = buf.read(1)
    assert magic == b"L"

    # Weight stream: LZMA
    q_len = struct.unpack("<I", buf.read(4))[0]
    q_raw = lzma.decompress(buf.read(q_len))

    # Scale and passthrough: zstd
    s_len = struct.unpack("<I", buf.read(4))[0]
    s_raw = dctx.decompress(buf.read(s_len)) if s_len > 0 else b""
    p_len = struct.unpack("<I", buf.read(4))[0]
    p_raw = dctx.decompress(buf.read(p_len)) if p_len > 0 else b""
    meta_obj = pickle.loads(dctx.decompress(buf.read()))

    m = meta_obj["m"]
    manifest = meta_obj["manifest"]

    w = {}
    offsets = {"q": 0, "s": 0, "p": 0}
    raw_map = {"q": q_raw, "s": s_raw, "p": p_raw}

    for name, dtype_str, shape, nbytes in manifest:
        if name.endswith(".q"):
            stream_key = "q"
        elif name.endswith(".scale"):
            stream_key = "s"
        else:
            stream_key = "p"

        off = offsets[stream_key]
        raw_bytes = raw_map[stream_key][off:off + nbytes]
        offsets[stream_key] = off + nbytes

        dt = np.dtype(dtype_str)
        if len(shape) == 2:
            arr = np.frombuffer(raw_bytes, dtype=dt).reshape(shape[1], shape[0]).T.copy()
        else:
            arr = np.frombuffer(raw_bytes, dtype=dt).reshape(shape).copy()
        w[name] = torch.from_numpy(arr)

    return {"w": w, "m": m}


# ==============================================================================
# EXPERIMENT: LZMA for ALL streams
# ==============================================================================

def encode_lzma_all(sota_obj: dict) -> bytes:
    """LZMA preset-9 for all streams + transpose."""
    w = sota_obj["w"]
    m = sota_obj["m"]

    q_parts = []
    s_parts = []
    p_parts = []
    manifest = []

    for name in sorted(w.keys()):
        t = w[name]
        arr = t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
        if arr.ndim == 2:
            arr = arr.T.copy()
        manifest.append((name, str(arr.dtype), list(t.shape), arr.nbytes))

        if name.endswith(".q"):
            q_parts.append(arr.tobytes())
        elif name.endswith(".scale"):
            s_parts.append(arr.tobytes())
        else:
            p_parts.append(arr.tobytes())

    q_blob = lzma.compress(b"".join(q_parts), preset=9) if q_parts else b""
    s_blob = lzma.compress(b"".join(s_parts), preset=9) if s_parts else b""
    p_blob = lzma.compress(b"".join(p_parts), preset=9) if p_parts else b""
    meta_blob = lzma.compress(pickle.dumps({"m": m, "manifest": manifest},
                                            protocol=pickle.HIGHEST_PROTOCOL), preset=9)

    out = io.BytesIO()
    out.write(b"A")  # magic for all-LZMA
    for blob in (q_blob, s_blob, p_blob):
        out.write(struct.pack("<I", len(blob)))
        out.write(blob)
    out.write(meta_blob)
    return out.getvalue()


def decode_lzma_all(blob: bytes) -> dict:
    """Decode all-LZMA format."""
    buf = io.BytesIO(blob)
    assert buf.read(1) == b"A"

    q_len = struct.unpack("<I", buf.read(4))[0]
    q_raw = lzma.decompress(buf.read(q_len)) if q_len > 0 else b""
    s_len = struct.unpack("<I", buf.read(4))[0]
    s_raw = lzma.decompress(buf.read(s_len)) if s_len > 0 else b""
    p_len = struct.unpack("<I", buf.read(4))[0]
    p_raw = lzma.decompress(buf.read(p_len)) if p_len > 0 else b""
    meta_obj = pickle.loads(lzma.decompress(buf.read()))

    m = meta_obj["m"]
    manifest = meta_obj["manifest"]
    w = {}
    offsets = {"q": 0, "s": 0, "p": 0}
    raw_map = {"q": q_raw, "s": s_raw, "p": p_raw}

    for name, dtype_str, shape, nbytes in manifest:
        if name.endswith(".q"):
            stream_key = "q"
        elif name.endswith(".scale"):
            stream_key = "s"
        else:
            stream_key = "p"

        off = offsets[stream_key]
        raw_bytes = raw_map[stream_key][off:off + nbytes]
        offsets[stream_key] = off + nbytes

        dt = np.dtype(dtype_str)
        if len(shape) == 2:
            arr = np.frombuffer(raw_bytes, dtype=dt).reshape(shape[1], shape[0]).T.copy()
        else:
            arr = np.frombuffer(raw_bytes, dtype=dt).reshape(shape).copy()
        w[name] = torch.from_numpy(arr)

    return {"w": w, "m": m}


# ==============================================================================
# EXPERIMENT: LZMA extreme preset for all streams
# ==============================================================================

def encode_lzma_extreme(sota_obj: dict) -> bytes:
    """LZMA preset-9|EXTREME for all streams + transpose."""
    w = sota_obj["w"]
    m = sota_obj["m"]
    preset = 9 | lzma.PRESET_EXTREME

    q_parts = []
    s_parts = []
    p_parts = []
    manifest = []

    for name in sorted(w.keys()):
        t = w[name]
        arr = t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
        if arr.ndim == 2:
            arr = arr.T.copy()
        manifest.append((name, str(arr.dtype), list(t.shape), arr.nbytes))

        if name.endswith(".q"):
            q_parts.append(arr.tobytes())
        elif name.endswith(".scale"):
            s_parts.append(arr.tobytes())
        else:
            p_parts.append(arr.tobytes())

    q_blob = lzma.compress(b"".join(q_parts), preset=preset) if q_parts else b""
    s_blob = lzma.compress(b"".join(s_parts), preset=preset) if s_parts else b""
    p_blob = lzma.compress(b"".join(p_parts), preset=preset) if p_parts else b""
    meta_blob = lzma.compress(pickle.dumps({"m": m, "manifest": manifest},
                                            protocol=pickle.HIGHEST_PROTOCOL), preset=preset)

    out = io.BytesIO()
    out.write(b"E")  # magic for extreme LZMA
    for blob in (q_blob, s_blob, p_blob):
        out.write(struct.pack("<I", len(blob)))
        out.write(blob)
    out.write(meta_blob)
    return out.getvalue()


def decode_lzma_extreme(blob: bytes) -> dict:
    """Decode extreme-LZMA format."""
    buf = io.BytesIO(blob)
    assert buf.read(1) == b"E"

    q_len = struct.unpack("<I", buf.read(4))[0]
    q_raw = lzma.decompress(buf.read(q_len)) if q_len > 0 else b""
    s_len = struct.unpack("<I", buf.read(4))[0]
    s_raw = lzma.decompress(buf.read(s_len)) if s_len > 0 else b""
    p_len = struct.unpack("<I", buf.read(4))[0]
    p_raw = lzma.decompress(buf.read(p_len)) if p_len > 0 else b""
    meta_obj = pickle.loads(lzma.decompress(buf.read()))

    m = meta_obj["m"]
    manifest = meta_obj["manifest"]
    w = {}
    offsets = {"q": 0, "s": 0, "p": 0}
    raw_map = {"q": q_raw, "s": s_raw, "p": p_raw}

    for name, dtype_str, shape, nbytes in manifest:
        if name.endswith(".q"):
            stream_key = "q"
        elif name.endswith(".scale"):
            stream_key = "s"
        else:
            stream_key = "p"

        off = offsets[stream_key]
        raw_bytes = raw_map[stream_key][off:off + nbytes]
        offsets[stream_key] = off + nbytes

        dt = np.dtype(dtype_str)
        if len(shape) == 2:
            arr = np.frombuffer(raw_bytes, dtype=dt).reshape(shape[1], shape[0]).T.copy()
        else:
            arr = np.frombuffer(raw_bytes, dtype=dt).reshape(shape).copy()
        w[name] = torch.from_numpy(arr)

    return {"w": w, "m": m}


# ==============================================================================
# EXPERIMENT: Group same tensor types across layers
# Sort .q tensors by type (c_q, c_k, etc.) instead of by name.
# ==============================================================================

import re

def _tensor_type_key(name: str) -> tuple:
    """Extract tensor type for grouping: (suffix_type, layer_num)."""
    # E.g., "blocks.3.attn.c_q.weight.q" → ("attn.c_q.weight.q", 3)
    m_match = re.match(r"blocks\.(\d+)\.(.*)", name)
    if m_match:
        layer = int(m_match.group(1))
        suffix = m_match.group(2)
        return (suffix, layer)
    return (name, 0)


def encode_lzma_typegroup(sota_obj: dict) -> bytes:
    """LZMA extreme with tensors grouped by type across layers."""
    w = sota_obj["w"]
    m = sota_obj["m"]
    preset = 9 | lzma.PRESET_EXTREME

    q_parts = []
    s_parts = []
    p_parts = []
    manifest = []

    # Sort by tensor type, then layer number
    q_names = sorted([n for n in w if n.endswith(".q")], key=_tensor_type_key)
    s_names = sorted([n for n in w if n.endswith(".scale")], key=_tensor_type_key)
    p_names = sorted([n for n in w if not n.endswith(".q") and not n.endswith(".scale")],
                     key=_tensor_type_key)

    for name in q_names + s_names + p_names:
        t = w[name]
        arr = t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
        if arr.ndim == 2:
            arr = arr.T.copy()

        if name.endswith(".q"):
            stream = "q"
        elif name.endswith(".scale"):
            stream = "s"
        else:
            stream = "p"

        manifest.append((name, str(arr.dtype), list(t.shape), arr.nbytes, stream))

        if stream == "q":
            q_parts.append(arr.tobytes())
        elif stream == "s":
            s_parts.append(arr.tobytes())
        else:
            p_parts.append(arr.tobytes())

    q_blob = lzma.compress(b"".join(q_parts), preset=preset) if q_parts else b""
    s_blob = lzma.compress(b"".join(s_parts), preset=preset) if s_parts else b""
    p_blob = lzma.compress(b"".join(p_parts), preset=preset) if p_parts else b""
    meta_blob = lzma.compress(pickle.dumps({"m": m, "manifest": manifest},
                                            protocol=pickle.HIGHEST_PROTOCOL), preset=preset)

    out = io.BytesIO()
    out.write(b"G")  # magic for grouped
    for blob in (q_blob, s_blob, p_blob):
        out.write(struct.pack("<I", len(blob)))
        out.write(blob)
    out.write(meta_blob)
    return out.getvalue()


def decode_lzma_typegroup(blob: bytes) -> dict:
    """Decode type-grouped LZMA format."""
    buf = io.BytesIO(blob)
    assert buf.read(1) == b"G"

    q_len = struct.unpack("<I", buf.read(4))[0]
    q_raw = lzma.decompress(buf.read(q_len)) if q_len > 0 else b""
    s_len = struct.unpack("<I", buf.read(4))[0]
    s_raw = lzma.decompress(buf.read(s_len)) if s_len > 0 else b""
    p_len = struct.unpack("<I", buf.read(4))[0]
    p_raw = lzma.decompress(buf.read(p_len)) if p_len > 0 else b""
    meta_obj = pickle.loads(lzma.decompress(buf.read()))

    m_meta = meta_obj["m"]
    manifest = meta_obj["manifest"]
    w = {}
    offsets = {"q": 0, "s": 0, "p": 0}
    raw_map = {"q": q_raw, "s": s_raw, "p": p_raw}

    for name, dtype_str, shape, nbytes, stream in manifest:
        off = offsets[stream]
        raw_bytes = raw_map[stream][off:off + nbytes]
        offsets[stream] = off + nbytes

        dt = np.dtype(dtype_str)
        if len(shape) == 2:
            arr = np.frombuffer(raw_bytes, dtype=dt).reshape(shape[1], shape[0]).T.copy()
        else:
            arr = np.frombuffer(raw_bytes, dtype=dt).reshape(shape).copy()
        w[name] = torch.from_numpy(arr)

    return {"w": w, "m": m_meta}


# ==============================================================================
# EXPERIMENT: Row-interleave same-type tensors across layers
# For each tensor type (e.g., attn.c_q), write row 0 from all layers,
# then row 1 from all layers, etc. Maximizes cross-layer pattern exposure.
# ==============================================================================

def encode_lzma_interleave(sota_obj: dict) -> bytes:
    """LZMA extreme with row-interleaved same-type tensors."""
    w = sota_obj["w"]
    m = sota_obj["m"]
    preset = 9 | lzma.PRESET_EXTREME

    # Group .q tensors by type suffix
    type_groups = {}  # suffix → [(name, arr), ...]
    s_parts = []
    p_parts = []
    manifest = []
    manifest_order = []  # track order for decoding

    for name in sorted(w.keys()):
        t = w[name]
        arr = t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t)

        if name.endswith(".q"):
            suffix_match = re.match(r"blocks\.\d+\.(.*)", name)
            suffix = suffix_match.group(1) if suffix_match else name
            if suffix not in type_groups:
                type_groups[suffix] = []
            type_groups[suffix].append((name, arr))
        elif name.endswith(".scale"):
            manifest.append((name, str(arr.dtype), list(t.shape), arr.nbytes, "s"))
            s_parts.append(arr.tobytes())
        else:
            if arr.ndim == 2:
                arr_t = arr.T.copy()
            else:
                arr_t = arr
            manifest.append((name, str(arr.dtype), list(t.shape), arr_t.nbytes, "p"))
            p_parts.append(arr_t.tobytes())

    # Build interleaved q stream
    q_bytes = io.BytesIO()
    for suffix in sorted(type_groups.keys()):
        group = type_groups[suffix]
        # All tensors in a group should have same shape
        shapes = set(arr.shape for _, arr in group)
        if len(shapes) == 1 and group[0][1].ndim == 2:
            # Interleave rows: transpose first, then interleave
            arrays = [arr.T for _, arr in group]
            n_rows = arrays[0].shape[0]
            for row in range(n_rows):
                for arr in arrays:
                    q_bytes.write(arr[row].tobytes())
            for name, arr in group:
                manifest.append((name, str(arr.dtype), list(arr.shape),
                                arr.T.shape[0] * arr.T.shape[1], "q_interleave"))
                manifest_order.append(("q_interleave", suffix, name))
        else:
            # Different shapes or 1D — just concatenate with transpose
            for name, arr in group:
                if arr.ndim == 2:
                    arr_t = arr.T.copy()
                else:
                    arr_t = arr
                q_bytes.write(arr_t.tobytes())
                manifest.append((name, str(arr.dtype), list(arr.shape), arr_t.nbytes, "q_plain"))

    q_blob = lzma.compress(q_bytes.getvalue(), preset=preset)
    s_blob = lzma.compress(b"".join(s_parts), preset=preset) if s_parts else b""
    p_blob = lzma.compress(b"".join(p_parts), preset=preset) if p_parts else b""
    meta_blob = lzma.compress(pickle.dumps({"m": m, "manifest": manifest,
                                            "type_groups": {k: [(n, list(a.shape)) for n, a in v]
                                                           for k, v in type_groups.items()}},
                                            protocol=pickle.HIGHEST_PROTOCOL), preset=preset)

    out = io.BytesIO()
    out.write(b"I")  # magic for interleaved
    for blob in (q_blob, s_blob, p_blob):
        out.write(struct.pack("<I", len(blob)))
        out.write(blob)
    out.write(meta_blob)
    return out.getvalue()


def decode_lzma_interleave(blob: bytes) -> dict:
    """Decode row-interleaved LZMA format."""
    buf = io.BytesIO(blob)
    assert buf.read(1) == b"I"

    q_len = struct.unpack("<I", buf.read(4))[0]
    q_raw = lzma.decompress(buf.read(q_len)) if q_len > 0 else b""
    s_len = struct.unpack("<I", buf.read(4))[0]
    s_raw = lzma.decompress(buf.read(s_len)) if s_len > 0 else b""
    p_len = struct.unpack("<I", buf.read(4))[0]
    p_raw = lzma.decompress(buf.read(p_len)) if p_len > 0 else b""
    meta_obj = pickle.loads(lzma.decompress(buf.read()))

    m_meta = meta_obj["m"]
    manifest = meta_obj["manifest"]
    type_groups = meta_obj["type_groups"]
    w = {}

    # Decode scale and passthrough streams
    s_offset = 0
    p_offset = 0
    for name, dtype_str, shape, nbytes, stream in manifest:
        if stream == "s":
            dt = np.dtype(dtype_str)
            arr = np.frombuffer(s_raw[s_offset:s_offset + nbytes], dtype=dt).reshape(shape).copy()
            w[name] = torch.from_numpy(arr)
            s_offset += nbytes
        elif stream == "p":
            dt = np.dtype(dtype_str)
            if len(shape) == 2:
                arr = np.frombuffer(p_raw[p_offset:p_offset + nbytes], dtype=dt).reshape(shape[1], shape[0]).T.copy()
            else:
                arr = np.frombuffer(p_raw[p_offset:p_offset + nbytes], dtype=dt).reshape(shape).copy()
            w[name] = torch.from_numpy(arr)
            p_offset += nbytes

    # Decode q stream: de-interleave
    q_buf = io.BytesIO(q_raw)
    for suffix in sorted(type_groups.keys()):
        group = type_groups[suffix]  # [(name, shape), ...]
        shapes = set(tuple(s) for _, s in group)
        if len(shapes) == 1 and len(group[0][1]) == 2:
            # De-interleave: read row by row across all tensors
            shape = group[0][1]
            n_rows = shape[1]  # transposed rows
            n_cols = shape[0]  # transposed cols
            n_tensors = len(group)
            # Read all interleaved data
            total_bytes = n_rows * n_tensors * n_cols
            interleaved = np.frombuffer(q_buf.read(total_bytes), dtype=np.int8)
            interleaved = interleaved.reshape(n_rows, n_tensors, n_cols)
            for i, (name, _) in enumerate(group):
                arr = interleaved[:, i, :].T.copy()  # un-transpose
                w[name] = torch.from_numpy(arr)
        else:
            for name, shape in group:
                nbytes = 1
                for s in shape:
                    nbytes *= s
                raw_bytes = q_buf.read(nbytes)
                if len(shape) == 2:
                    arr = np.frombuffer(raw_bytes, dtype=np.int8).reshape(shape[1], shape[0]).T.copy()
                else:
                    arr = np.frombuffer(raw_bytes, dtype=np.int8).reshape(shape).copy()
                w[name] = torch.from_numpy(arr)

    return {"w": w, "m": m_meta}


# ==============================================================================
# EXPERIMENT: Sparse representation + LZMA extreme + interleave
# Store bitmask (zero/nonzero) + only non-zero values separately.
# ==============================================================================

def encode_lzma_sparse(sota_obj: dict) -> bytes:
    """LZMA extreme with sparse representation for weight stream."""
    w = sota_obj["w"]
    m = sota_obj["m"]
    preset = 9 | lzma.PRESET_EXTREME

    # Build interleaved q stream, separating int8 (dense) from int6 (sparse)
    type_groups = {}  # int6 tensors grouped by type
    int8_parts = []   # int8 tensors compressed directly
    s_parts = []
    p_parts = []
    manifest = []

    for name in sorted(w.keys()):
        t = w[name]
        arr = t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t)

        if name.endswith(".q"):
            base_name = name[:-2]
            info = m.get(base_name, {})
            is_int8 = isinstance(info, dict) and info.get("type") == "int8"
            if is_int8:
                arr_t = arr.T.copy() if arr.ndim == 2 else arr
                manifest.append((name, str(arr.dtype), list(t.shape), arr_t.nbytes, "q8"))
                int8_parts.append(arr_t.tobytes())
            else:
                suffix_match = re.match(r"blocks\.\d+\.(.*)", name)
                suffix = suffix_match.group(1) if suffix_match else name
                if suffix not in type_groups:
                    type_groups[suffix] = []
                type_groups[suffix].append((name, arr))
        elif name.endswith(".scale"):
            manifest.append((name, str(arr.dtype), list(t.shape), arr.nbytes, "s"))
            s_parts.append(arr.tobytes())
        else:
            arr_t = arr.T.copy() if arr.ndim == 2 else arr
            manifest.append((name, str(arr.dtype), list(t.shape), arr_t.nbytes, "p"))
            p_parts.append(arr_t.tobytes())

    # Build interleaved int6 weight bytes
    q_buf = io.BytesIO()
    for suffix in sorted(type_groups.keys()):
        group = type_groups[suffix]
        shapes = set(arr.shape for _, arr in group)
        if len(shapes) == 1 and group[0][1].ndim == 2:
            arrays = [arr.T for _, arr in group]
            for row in range(arrays[0].shape[0]):
                for arr in arrays:
                    q_buf.write(arr[row].tobytes())
            for name, arr in group:
                manifest.append((name, str(arr.dtype), list(arr.shape),
                                arr.T.shape[0] * arr.T.shape[1], "q6"))
        else:
            for name, arr in group:
                arr_t = arr.T.copy() if arr.ndim == 2 else arr
                q_buf.write(arr_t.tobytes())
                manifest.append((name, str(arr.dtype), list(arr.shape), arr_t.nbytes, "q6"))

    q_raw = q_buf.getvalue()
    q_arr = np.frombuffer(q_raw, dtype=np.int8)

    # Sparse encoding: bitmask + sign bits + abs==1 bitmask + abs>1 values
    nonzero_mask = (q_arr != 0)
    bitmask = np.packbits(nonzero_mask)
    nonzero_vals = q_arr[nonzero_mask]
    signs = np.packbits((nonzero_vals < 0).astype(np.uint8))
    absvals = np.abs(nonzero_vals).astype(np.uint8)
    not_one = (absvals != 1)
    abs_not_one_mask = np.packbits(not_one.astype(np.uint8))
    abs_gt1_vals = absvals[not_one]

    # Compress streams
    mask_blob = lzma.compress(bitmask.tobytes(), preset=preset)
    signs_blob = lzma.compress(signs.tobytes(), preset=preset)
    abs_mask_blob = lzma.compress(abs_not_one_mask.tobytes(), preset=preset)
    abs_vals_blob = lzma.compress(abs_gt1_vals.tobytes(), preset=preset)
    q8_blob = lzma.compress(b"".join(int8_parts), preset=preset) if int8_parts else b""
    s_blob = lzma.compress(b"".join(s_parts), preset=preset) if s_parts else b""
    # Byte-shuffle fp32 passthrough for better compression
    p_raw = b"".join(p_parts)
    if p_raw:
        p_arr = np.frombuffer(p_raw, dtype=np.uint8)
        p_shuffled = b"".join(p_arr[i::4].tobytes() for i in range(4))
        p_blob = lzma.compress(p_shuffled, preset=preset)
    else:
        p_blob = b""
    meta_blob = lzma.compress(pickle.dumps({
        "m": m, "manifest": manifest,
        "type_groups": {k: [(n, list(a.shape)) for n, a in v]
                       for k, v in type_groups.items()},
        "q6_total": len(q_arr),
    }, protocol=pickle.HIGHEST_PROTOCOL), preset=preset)

    # Pack: magic + [mask][signs][abs_mask][abs_vals][q8][s][p][meta]
    out = io.BytesIO()
    out.write(b"S")  # magic for sparse
    for blob in (mask_blob, signs_blob, abs_mask_blob, abs_vals_blob, q8_blob, s_blob, p_blob):
        out.write(struct.pack("<I", len(blob)))
        out.write(blob)
    out.write(meta_blob)
    return out.getvalue()


def decode_lzma_sparse(blob: bytes) -> dict:
    """Decode sparse LZMA format."""
    buf = io.BytesIO(blob)
    assert buf.read(1) == b"S"

    # Read compressed streams
    blobs = []
    for _ in range(7):
        slen = struct.unpack("<I", buf.read(4))[0]
        blobs.append(lzma.decompress(buf.read(slen)) if slen > 0 else b"")
    mask_raw, signs_raw, abs_mask_raw, abs_vals_raw, q8_raw, s_raw, p_shuffled = blobs
    # Un-shuffle fp32 passthrough
    if p_shuffled:
        p_arr = np.frombuffer(p_shuffled, dtype=np.uint8)
        plane_size = len(p_arr) // 4
        p_restored = np.empty(len(p_arr), dtype=np.uint8)
        for i in range(4):
            p_restored[i::4] = p_arr[i * plane_size:(i + 1) * plane_size]
        p_raw = p_restored.tobytes()
    else:
        p_raw = b""
    meta_obj = pickle.loads(lzma.decompress(buf.read()))

    m_meta = meta_obj["m"]
    manifest = meta_obj["manifest"]
    type_groups = meta_obj["type_groups"]
    q6_total = meta_obj["q6_total"]

    # Reconstruct dense int6 weight array from sparse sign+abs decomposition
    bitmask = np.unpackbits(np.frombuffer(mask_raw, dtype=np.uint8))[:q6_total]
    n_nonzero = int(bitmask.sum())
    signs = np.unpackbits(np.frombuffer(signs_raw, dtype=np.uint8))[:n_nonzero]
    not_one = np.unpackbits(np.frombuffer(abs_mask_raw, dtype=np.uint8))[:n_nonzero]
    abs_gt1 = np.frombuffer(abs_vals_raw, dtype=np.uint8)

    # Reconstruct abs values: default 1, override where not_one
    absvals = np.ones(n_nonzero, dtype=np.uint8)
    absvals[not_one.astype(bool)] = abs_gt1

    nonzero_vals = absvals.astype(np.int8)
    nonzero_vals[signs.astype(bool)] = -nonzero_vals[signs.astype(bool)]
    q6_dense = np.zeros(q6_total, dtype=np.int8)
    q6_dense[bitmask.astype(bool)] = nonzero_vals

    # Decode scale, passthrough, and int8 streams
    w = {}
    s_offset = 0
    p_offset = 0
    q8_offset = 0
    for name, dtype_str, shape, nbytes, stream in manifest:
        if stream == "q8":
            dt = np.dtype(dtype_str)
            if len(shape) == 2:
                arr = np.frombuffer(q8_raw[q8_offset:q8_offset + nbytes], dtype=dt).reshape(shape[1], shape[0]).T.copy()
            else:
                arr = np.frombuffer(q8_raw[q8_offset:q8_offset + nbytes], dtype=dt).reshape(shape).copy()
            w[name] = torch.from_numpy(arr)
            q8_offset += nbytes
        elif stream == "s":
            dt = np.dtype(dtype_str)
            arr = np.frombuffer(s_raw[s_offset:s_offset + nbytes], dtype=dt).reshape(shape).copy()
            w[name] = torch.from_numpy(arr)
            s_offset += nbytes
        elif stream == "p":
            dt = np.dtype(dtype_str)
            if len(shape) == 2:
                arr = np.frombuffer(p_raw[p_offset:p_offset + nbytes], dtype=dt).reshape(shape[1], shape[0]).T.copy()
            else:
                arr = np.frombuffer(p_raw[p_offset:p_offset + nbytes], dtype=dt).reshape(shape).copy()
            w[name] = torch.from_numpy(arr)
            p_offset += nbytes

    # De-interleave int6 q stream
    q_buf = io.BytesIO(q6_dense.tobytes())
    for suffix in sorted(type_groups.keys()):
        group = type_groups[suffix]
        shapes = set(tuple(s) for _, s in group)
        if len(shapes) == 1 and len(group[0][1]) == 2:
            shape = group[0][1]
            n_rows = shape[1]  # transposed
            n_cols = shape[0]
            n_tensors = len(group)
            total_bytes = n_rows * n_tensors * n_cols
            interleaved = np.frombuffer(q_buf.read(total_bytes), dtype=np.int8)
            interleaved = interleaved.reshape(n_rows, n_tensors, n_cols)
            for i, (name, _) in enumerate(group):
                arr = interleaved[:, i, :].T.copy()
                w[name] = torch.from_numpy(arr)
        else:
            for name, shape in group:
                nbytes = 1
                for s in shape:
                    nbytes *= s
                raw_bytes = q_buf.read(nbytes)
                if len(shape) == 2:
                    arr = np.frombuffer(raw_bytes, dtype=np.int8).reshape(shape[1], shape[0]).T.copy()
                else:
                    arr = np.frombuffer(raw_bytes, dtype=np.int8).reshape(shape).copy()
                w[name] = torch.from_numpy(arr)

    return {"w": w, "m": m_meta}


# ==============================================================================
# HELPERS
# ==============================================================================

def load_mlx_artifact(path: str) -> dict:
    """Load an MLX .int8.ptz artifact (pickle + zlib format)."""
    with open(path, "rb") as f:
        blob = f.read()
    return pickle.loads(zlib.decompress(blob))


def load_and_convert(path: str) -> dict:
    """Load MLX artifact and convert to SOTA format."""
    mlx_obj = load_mlx_artifact(path)
    return mlx_to_sota_format(mlx_obj)


def measure_scheme(
    name: str,
    encode_fn,
    decode_fn,
    sota_obj: dict,
    n_trials: int = 3,
) -> dict:
    """Benchmark a serialization scheme. Returns metrics dict."""
    # Encode
    encode_times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        blob = encode_fn(sota_obj)
        encode_times.append(1000.0 * (time.perf_counter() - t0))

    compressed_bytes = len(blob)

    # Decode
    decode_times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        decoded = decode_fn(blob)
        decode_times.append(1000.0 * (time.perf_counter() - t0))

    # Roundtrip accuracy on the "w" dict
    max_abs_error = 0.0
    mean_abs_error = 0.0
    total_values = 0

    orig_w = sota_obj.get("w", {})
    decoded_w = decoded.get("w", {})
    for key in orig_w:
        orig = orig_w[key]
        rt = decoded_w.get(key)
        if rt is None:
            max_abs_error = float("inf")
            continue
        # Convert to numpy for comparison (handles both torch.Tensor and np.ndarray)
        orig_np = orig.numpy() if isinstance(orig, torch.Tensor) else np.asarray(orig)
        rt_np = rt.numpy() if isinstance(rt, torch.Tensor) else np.asarray(rt)
        if orig_np.shape != rt_np.shape:
            max_abs_error = float("inf")
            continue
        diff = np.abs(orig_np.astype(np.float32) - rt_np.astype(np.float32))
        max_abs_error = max(max_abs_error, float(diff.max()))
        mean_abs_error += float(diff.sum())
        total_values += orig_np.size

    if total_values > 0:
        mean_abs_error /= total_values

    return {
        "name": name,
        "compressed_bytes": compressed_bytes,
        "encode_ms": min(encode_times),
        "decode_ms": min(decode_times),
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
    }

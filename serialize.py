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

"""
Serialization schemes for Parameter Golf model artifacts.

Each scheme implements:
  - encode(quant_obj) -> bytes  (serialize + compress)
  - decode(blob) -> quant_obj   (decompress + deserialize)

The quant_obj is the standard dict with keys:
  quantized: dict[str, np.ndarray]   # int8 weight tensors
  scales: dict[str, np.ndarray]      # fp16 per-row scales
  dtypes: dict[str, str]             # original dtype names
  passthrough: dict[str, np.ndarray] # small fp32/fp16 tensors
  qmeta: dict[str, dict]            # quantization metadata
  passthrough_orig_dtypes: dict[str, str]  # original dtypes for passthrough
"""

from __future__ import annotations

import io
import pickle
import struct
import time
import zlib
from typing import Any

import numpy as np

try:
    import zstandard
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


# ==============================================================================
# BASELINE: pickle + zlib (matches current MLX pipeline)
# ==============================================================================

def encode_baseline(quant_obj: dict) -> bytes:
    """Baseline: pickle + zlib-9. Matches the current MLX training script."""
    raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    return zlib.compress(raw, 9)


def decode_baseline(blob: bytes) -> dict:
    """Baseline decoder."""
    return pickle.loads(zlib.decompress(blob))


# ==============================================================================
# SCHEME 1: pickle + zstd-22 (matches current H100 pipeline)
# ==============================================================================

def encode_zstd22(quant_obj: dict) -> bytes:
    """pickle + zstd-22. Used by top H100 submissions."""
    if not HAS_ZSTD:
        raise ImportError("zstandard not installed")
    raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    return zstandard.ZstdCompressor(level=22).compress(raw)


def decode_zstd22(blob: bytes) -> dict:
    """zstd-22 decoder."""
    if not HAS_ZSTD:
        raise ImportError("zstandard not installed")
    return pickle.loads(zstandard.ZstdDecompressor().decompress(blob))


# ==============================================================================
# HELPERS
# ==============================================================================

def load_artifact(path: str) -> dict:
    """Load a .int8.ptz artifact (pickle + zlib format)."""
    with open(path, "rb") as f:
        blob = f.read()
    return pickle.loads(zlib.decompress(blob))


def measure_scheme(
    name: str,
    encode_fn,
    decode_fn,
    quant_obj: dict,
    n_trials: int = 3,
) -> dict:
    """Benchmark a serialization scheme. Returns metrics dict."""
    # Encode
    encode_times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        blob = encode_fn(quant_obj)
        encode_times.append(1000.0 * (time.perf_counter() - t0))

    compressed_bytes = len(blob)

    # Decode
    decode_times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        decoded = decode_fn(blob)
        decode_times.append(1000.0 * (time.perf_counter() - t0))

    # Roundtrip accuracy
    max_abs_error = 0.0
    mean_abs_error = 0.0
    total_values = 0
    for key in quant_obj.get("quantized", {}):
        orig = np.asarray(quant_obj["quantized"][key])
        rt = np.asarray(decoded["quantized"][key])
        diff = np.abs(orig.astype(np.float32) - rt.astype(np.float32))
        max_abs_error = max(max_abs_error, float(diff.max()))
        mean_abs_error += float(diff.sum())
        total_values += orig.size
    for key in quant_obj.get("passthrough", {}):
        orig = np.asarray(quant_obj["passthrough"][key]).astype(np.float32)
        rt = np.asarray(decoded["passthrough"][key]).astype(np.float32)
        diff = np.abs(orig - rt)
        max_abs_error = max(max_abs_error, float(diff.max()))
        mean_abs_error += float(diff.sum())
        total_values += orig.size
    for key in quant_obj.get("scales", {}):
        orig = np.asarray(quant_obj["scales"][key]).astype(np.float32)
        rt = np.asarray(decoded["scales"][key]).astype(np.float32)
        diff = np.abs(orig - rt)
        max_abs_error = max(max_abs_error, float(diff.max()))
        mean_abs_error += float(diff.sum())
        total_values += orig.size

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

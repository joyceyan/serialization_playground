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
# EXPERIMENT: transpose 2D tensors + zstd-22
# ==============================================================================

def encode_experiment(quant_result: dict[str, Tensor], quant_meta: dict[str, object]) -> bytes:
    """Transpose 2D tensors before torch.save + zstd-22."""
    if not HAS_ZSTD:
        raise ImportError("zstandard not installed")
    transposed = {}
    for k, v in quant_result.items():
        if v.ndim == 2:
            transposed[k] = v.t().contiguous()
        else:
            transposed[k] = v
    buf = io.BytesIO()
    torch.save({"w": transposed, "m": quant_meta}, buf)
    raw = buf.getvalue()
    return zstandard.ZstdCompressor(level=22).compress(raw)


def decode_experiment(blob: bytes) -> tuple[dict[str, Tensor], dict[str, object]]:
    """Decode transposed tensors."""
    if not HAS_ZSTD:
        raise ImportError("zstandard not installed")
    raw = zstandard.ZstdDecompressor().decompress(blob)
    obj = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    w = {}
    for k, v in obj["w"].items():
        if v.ndim == 2:
            w[k] = v.t().contiguous()
        else:
            w[k] = v
    return w, obj["m"]


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
